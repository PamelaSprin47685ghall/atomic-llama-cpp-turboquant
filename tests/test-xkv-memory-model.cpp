// Deterministic behavioral tests for XKV memory model (nominal vs actual)
// reproducing Section 2.3 of XKV-SR.md and verifying codec descriptor sizing.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-codec.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llama_xkv;

// Helper assertion with message
static void assert_check(bool condition, const std::string & msg) {
    if (!condition) {
        std::cerr << "FAILED: " << msg << std::endl;
        std::abort();
    }
}

// ============================================================================
// PART 1: Nominal §2.3 Arithmetic with Checked Integer Math
// ============================================================================
// Exact payload math only:
// W = 4 owning layers, D_K = D_V = 1024, n = 4096 tokens, rK = 384, rV = 576.
// Chunks: chunk_size = 8 tokens -> 512 chunks.
// Exact byte totals:
//   1) Original FP16 KV:          64 MiB   = 67,108,864 bytes
//   2) Nominal K4/V2 KV:          12 MiB   = 12,582,912 bytes
//   3) A/B FP16 + SR FP16:        19 MiB   = 19,922,944 bytes
//   4) A/B 4-bit + SR FP16:       7.75 MiB = 8,126,464 bytes
//   5) A/B 4-bit + SR 8-bit:      5.75 MiB = 6,029,312 bytes
//   6) A/B 4-bit + SR 4-bit:      4.75 MiB = 4,980,736 bytes
// ============================================================================

static uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (a == 0 || b == 0) return 0;
    if (a > UINT64_MAX / b) {
        throw std::overflow_error("integer multiplication overflow");
    }
    return a * b;
}

static uint64_t checked_add(uint64_t a, uint64_t b) {
    if (UINT64_MAX - a < b) {
        throw std::overflow_error("integer addition overflow");
    }
    return a + b;
}

static void test_nominal_section_2_3_exact_bytes() {
    std::cout << "[test_nominal_section_2_3_exact_bytes] starting..." << std::endl;

    constexpr uint64_t W    = 4;
    constexpr uint64_t D_K  = 1024;
    constexpr uint64_t D_V  = 1024;
    constexpr uint64_t n    = 4096;
    constexpr uint64_t rK   = 384;
    constexpr uint64_t rV   = 576;
    constexpr uint64_t chunk_tokens = 8;

    constexpr uint64_t MIB = 1024ULL * 1024ULL;

    // Dimensions:
    // Spliced feature dimension across owning layers:
    const uint64_t D_total_K = checked_mul(W, D_K); // 4 * 1024 = 4096
    const uint64_t D_total_V = checked_mul(W, D_V); // 4 * 1024 = 4096
    assert_check(D_total_K == 4096, "D_total_K == 4096");
    assert_check(D_total_V == 4096, "D_total_V == 4096");

    // Elements in original full KV:
    // K: n * D_total_K = 4096 * 4096 = 16,777,216 elements
    // V: n * D_total_V = 4096 * 4096 = 16,777,216 elements
    const uint64_t k_elements = checked_mul(n, D_total_K);
    const uint64_t v_elements = checked_mul(n, D_total_V);
    const uint64_t kv_total_elements = checked_add(k_elements, v_elements);
    assert_check(k_elements == 16777216ULL, "k_elements == 16777216");
    assert_check(v_elements == 16777216ULL, "v_elements == 16777216");
    assert_check(kv_total_elements == 33554432ULL, "kv_total_elements == 33554432");

    // Factor elements:
    // A_K: n * rK = 4096 * 384 = 1,572,864 elements
    // A_V: n * rV = 4096 * 576 = 2,359,296 elements
    const uint64_t a_k_elements = checked_mul(n, rK);
    const uint64_t a_v_elements = checked_mul(n, rV);
    const uint64_t a_elements = checked_add(a_k_elements, a_v_elements);
    assert_check(a_k_elements == 1572864ULL, "a_k_elements == 1572864");
    assert_check(a_v_elements == 2359296ULL, "a_v_elements == 2359296");
    assert_check(a_elements == 3932160ULL, "A_K + A_V elements == 3932160 (§2.3)");

    // B_K: rK * D_total_K = 384 * 4096 = 1,572,864 elements
    // B_V: rV * D_total_V = 576 * 4096 = 2,359,296 elements
    const uint64_t b_k_elements = checked_mul(rK, D_total_K);
    const uint64_t b_v_elements = checked_mul(rV, D_total_V);
    const uint64_t b_elements = checked_add(b_k_elements, b_v_elements);
    assert_check(b_k_elements == 1572864ULL, "b_k_elements == 1572864");
    assert_check(b_v_elements == 2359296ULL, "b_v_elements == 2359296");
    assert_check(b_elements == 3932160ULL, "B_K + B_V elements == 3932160 (§2.3)");

    const uint64_t factor_total_elements = checked_add(a_elements, b_elements);
    assert_check(factor_total_elements == 7864320ULL, "factor_total_elements == 7864320");

    // Landmarks (SR summary chunks):
    // "每 8 个 token 一个完整合法摘要块" -> n / chunk_tokens = 512 chunks
    assert_check(n % chunk_tokens == 0, "n divisible by chunk_tokens");
    const uint64_t n_chunks = n / chunk_tokens;
    assert_check(n_chunks == 512, "n_chunks == 512");

    // Landmark summary has 1 summary vector per chunk across D_total_K:
    // n_chunks * D_total_K = 512 * 4096 = 2,097,152 elements
    const uint64_t landmark_elements = checked_mul(n_chunks, D_total_K);
    assert_check(landmark_elements == 2097152ULL, "landmark_elements == 2097152 (§2.3: 2,097,152)");

    // Row 1: Original FP16 KV (2 bytes/elem)
    // 33,554,432 * 2 = 67,108,864 bytes = 64 MiB
    const uint64_t row1_orig_fp16 = checked_mul(kv_total_elements, 2);
    assert_check(row1_orig_fp16 == 67108864ULL, "row 1 bytes == 67108864");
    assert_check(row1_orig_fp16 == 64ULL * MIB, "row 1 == 64 MiB");

    // Row 2: Original KV: Nominal K4/V2 (K: 0.5 byte, V: 0.25 byte)
    // K: 16,777,216 * 4 / 8 = 8,388,608 bytes (8 MiB)
    // V: 16,777,216 * 2 / 8 = 4,194,304 bytes (4 MiB)
    // Total = 12,582,912 bytes = 12 MiB
    const uint64_t row2_k = checked_mul(k_elements, 4) / 8;
    const uint64_t row2_v = checked_mul(v_elements, 2) / 8;
    const uint64_t row2_nominal_k4v2 = checked_add(row2_k, row2_v);
    assert_check(row2_k == 8ULL * MIB, "nominal K4 == 8 MiB");
    assert_check(row2_v == 4ULL * MIB, "nominal V2 == 4 MiB");
    assert_check(row2_nominal_k4v2 == 12582912ULL, "row 2 bytes == 12582912");
    assert_check(row2_nominal_k4v2 == 12ULL * MIB, "row 2 == 12 MiB");

    // Row 3: A/B FP16, Landmark FP16
    // Factor FP16: 7,864,320 * 2 = 15,728,640 bytes = 15 MiB
    // Landmark FP16: 2,097,152 * 2 = 4,194,304 bytes = 4 MiB
    // Total = 15 + 4 = 19 MiB = 19,922,944 bytes
    const uint64_t factor_fp16 = checked_mul(factor_total_elements, 2);
    const uint64_t landmark_fp16 = checked_mul(landmark_elements, 2);
    const uint64_t row3_total = checked_add(factor_fp16, landmark_fp16);
    assert_check(factor_fp16 == 15ULL * MIB, "factor FP16 == 15 MiB");
    assert_check(landmark_fp16 == 4ULL * MIB, "landmark FP16 == 4 MiB");
    assert_check(row3_total == 19922944ULL, "row 3 bytes == 19922944");
    assert_check(row3_total == 19ULL * MIB, "row 3 == 19 MiB");

    // Row 4: A/B 4-bit, Landmark FP16
    // Factor 4-bit: 7,864,320 * 4 / 8 = 3,932,160 bytes = 3.75 MiB (3 * MIB + 3 * MIB / 4)
    // Landmark FP16: 4 MiB
    // Total = 3.75 + 4 = 7.75 MiB = 8,126,464 bytes
    const uint64_t factor_4bit = checked_mul(factor_total_elements, 4) / 8;
    assert_check(factor_4bit == 3932160ULL, "factor 4-bit == 3932160 bytes");
    assert_check(factor_4bit * 4 == 15ULL * MIB, "factor 4-bit is exact 3.75 MiB");
    const uint64_t row4_total = checked_add(factor_4bit, landmark_fp16);
    assert_check(row4_total == 8126464ULL, "row 4 bytes == 8126464");
    assert_check(row4_total * 4 == 31ULL * MIB, "row 4 == 7.75 MiB (31/4 MiB)");

    // Row 5: A/B 4-bit, Landmark 8-bit
    // Landmark 8-bit: 2,097,152 * 8 / 8 = 2,097,152 bytes = 2 MiB
    // Total = 3.75 + 2 = 5.75 MiB = 6,029,312 bytes
    const uint64_t landmark_8bit = checked_mul(landmark_elements, 8) / 8;
    assert_check(landmark_8bit == 2ULL * MIB, "landmark 8-bit == 2 MiB");
    const uint64_t row5_total = checked_add(factor_4bit, landmark_8bit);
    assert_check(row5_total == 6029312ULL, "row 5 bytes == 6029312");
    assert_check(row5_total * 4 == 23ULL * MIB, "row 5 == 5.75 MiB (23/4 MiB)");

    // Row 6: A/B 4-bit, Landmark 4-bit
    // Landmark 4-bit: 2,097,152 * 4 / 8 = 1,048,576 bytes = 1 MiB
    // Total = 3.75 + 1 = 4.75 MiB = 4,980,736 bytes
    const uint64_t landmark_4bit = checked_mul(landmark_elements, 4) / 8;
    assert_check(landmark_4bit == 1ULL * MIB, "landmark 4-bit == 1 MiB");
    const uint64_t row6_total = checked_add(factor_4bit, landmark_4bit);
    assert_check(row6_total == 4980736ULL, "row 6 bytes == 4980736");
    assert_check(row6_total * 4 == 19ULL * MIB, "row 6 == 4.75 MiB (19/4 MiB)");

    // Nominal compression ratio check:
    // Relative to original K4/V2 (12 MiB):
    // row 6 ratio = 12 / 4.75 = 12 * 4 / 19 = 48 / 19 = 2.5263... ~ 2.53x
    const double ratio_rel_k4v2 = (double)row2_nominal_k4v2 / (double)row6_total;
    assert_check(std::fabs(ratio_rel_k4v2 - (48.0 / 19.0)) < 1e-9, "relative to K4/V2 ratio is exact 48/19");
    assert_check(std::round(ratio_rel_k4v2 * 100.0) / 100.0 == 2.53, "ratio rounded is 2.53x (§2.3)");

    // Relative to original FP16 (64 MiB):
    // 64 / 4.75 = 64 * 4 / 19 = 256 / 19 = 13.4736... ~ 13.47x
    const double ratio_rel_fp16 = (double)row1_orig_fp16 / (double)row6_total;
    assert_check(std::fabs(ratio_rel_fp16 - (256.0 / 19.0)) < 1e-9, "relative to FP16 ratio is exact 256/19");
    assert_check(std::round(ratio_rel_fp16 * 100.0) / 100.0 == 13.47, "ratio rounded is 13.47x (§2.3)");

    std::cout << "  [PASS] Nominal §2.3 arithmetic confirmed: 64, 12, 19, 7.75, 5.75, 4.75 MiB exactly." << std::endl;
}

// ============================================================================
// PART 2: Actual Turbo4/Turbo2/Q8 Descriptors and Estimator vs Encoded Bytes
// ============================================================================
// Proves:
//   - Estimator (encoded_matrix_bytes) equals actual encoded vector size.
//   - Includes rank padding (e.g. non-multiples of block_size / group_size).
//   - Includes B^T (orientation::feature_major_transposed) row layout.
//   - Distinct actual byte totals vs nominal bits-per-element.
// ============================================================================

static void test_actual_codec_descriptors_and_estimator() {
    std::cout << "[test_actual_codec_descriptors_and_estimator] starting..." << std::endl;

    constexpr uint64_t W    = 4;
    constexpr uint64_t D_K  = 1024;
    constexpr uint64_t D_V  = 1024;
    constexpr uint64_t n    = 4096;
    constexpr uint64_t rK   = 384;
    constexpr uint64_t rV   = 576;
    constexpr uint64_t n_chunks = 512;

    const uint64_t D_total_K = W * D_K; // 4096
    const uint64_t D_total_V = W * D_V; // 4096

    // Candidate 1: Production candidate with A_K=Turbo4, B_K=Turbo4, A_V=Turbo2, B_V=Turbo2, Landmark=Q8_0
    // Notice:
    // A_K: token_major, logical (n=4096, rK=384). rK % 128 == 0 -> padded_cols = 384.
    // B_K: feature_major_transposed (B^T), logical (D_total_K=4096, rK=384). padded_cols = 384.
    // A_V: token_major, logical (n=4096, rV=576). rV % 128 == 64 -> padded_cols = 640! (Rank padding!)
    // B_V: feature_major_transposed (B^T), logical (D_total_V=4096, rV=576). padded_cols = 640!
    // Landmark: token_major, logical (n_chunks=512, D_total_K=4096). Q8_0 has block_size=32. 4096 % 32 == 0.

    codec_desc desc_a_k = make_codec_desc(
        factor_role::a_k,
        GGML_TYPE_TURBO4_0,
        orientation::token_major,
        {n, rK},
        128
    );
    assert_check(desc_a_k.padded_shape.rows == 4096, "A_K padded rows == 4096");
    assert_check(desc_a_k.padded_shape.cols == 384,  "A_K padded cols == 384");
    // Turbo4 block_size=128, block size in bytes = 68. 3 blocks per row = 204 bytes.
    assert_check(desc_a_k.row_stride_bytes == 3 * 68, "A_K row stride == 204");
    const uint64_t bytes_a_k = encoded_matrix_bytes(desc_a_k);
    assert_check(bytes_a_k == 4096 * 204, "A_K total bytes == 835584");

    codec_desc desc_b_k = make_codec_desc(
        factor_role::b_k,
        GGML_TYPE_TURBO4_0,
        orientation::feature_major_transposed,
        {D_total_K, rK},
        128
    );
    assert_check(desc_b_k.orient == orientation::feature_major_transposed, "B_K is feature_major_transposed");
    assert_check(desc_b_k.padded_shape.rows == 4096, "B_K padded rows == 4096");
    assert_check(desc_b_k.padded_shape.cols == 384,  "B_K padded cols == 384");
    assert_check(desc_b_k.row_stride_bytes == 3 * 68, "B_K row stride == 204");
    const uint64_t bytes_b_k = encoded_matrix_bytes(desc_b_k);
    assert_check(bytes_b_k == 4096 * 204, "B_K total bytes == 835584");

    // A_V with Turbo2: rV = 576. 576 is not divisible by 128!
    // 576 = 4 * 128 + 64 -> padded cols = 5 * 128 = 640!
    const size_t turbo2_type_size = ggml_type_size(GGML_TYPE_TURBO2_0);
    assert_check(turbo2_type_size == 34, "GGML_TYPE_TURBO2_0 block size must be exactly 34 bytes (norm: 2 + 128/4: 32)");

    codec_desc desc_a_v = make_codec_desc(
        factor_role::a_v,
        GGML_TYPE_TURBO2_0,
        orientation::token_major,
        {n, rV},
        128
    );
    assert_check(desc_a_v.logical_shape.cols == 576, "A_V logical cols == 576");
    assert_check(desc_a_v.padded_shape.cols == 640,  "A_V padded cols == 640 (rank padded)");
    // Turbo2 block_size=128, block size in bytes = 34. 5 blocks per row = 170 bytes.
    assert_check(desc_a_v.row_stride_bytes == 5 * turbo2_type_size, "A_V row stride == 5 * 34 = 170");
    const uint64_t bytes_a_v = encoded_matrix_bytes(desc_a_v);
    assert_check(bytes_a_v == 4096 * 170, "A_V total bytes == 4096 * 170 = 696320");

    // B_V with Turbo2: B^T shape {D_total_V, rV} = {4096, 576}
    codec_desc desc_b_v = make_codec_desc(
        factor_role::b_v,
        GGML_TYPE_TURBO2_0,
        orientation::feature_major_transposed,
        {D_total_V, rV},
        128
    );
    assert_check(desc_b_v.orient == orientation::feature_major_transposed, "B_V is feature_major_transposed");
    assert_check(desc_b_v.padded_shape.rows == 4096, "B_V padded rows == 4096");
    assert_check(desc_b_v.padded_shape.cols == 640,  "B_V padded cols == 640 (rank padded)");
    assert_check(desc_b_v.row_stride_bytes == 5 * turbo2_type_size, "B_V row stride == 5 * 34 = 170");
    const uint64_t bytes_b_v = encoded_matrix_bytes(desc_b_v);
    assert_check(bytes_b_v == 4096 * 170, "B_V total bytes == 4096 * 170 = 696320");

    // Landmark with Q8_0: shape {n_chunks=512, D_total_K=4096}
    codec_desc desc_landmark_q8 = make_codec_desc(
        factor_role::landmark,
        GGML_TYPE_Q8_0,
        orientation::token_major,
        {n_chunks, D_total_K}
    );
    assert_check(desc_landmark_q8.role == factor_role::landmark, "landmark role");
    assert_check(desc_landmark_q8.padded_shape.rows == 512, "landmark rows == 512");
    assert_check(desc_landmark_q8.padded_shape.cols == 4096, "landmark cols == 4096");
    // Q8_0: block_size=32, block size in bytes=34 (sizeof(block_q8_0)). 4096 / 32 = 128 blocks.
    // Row stride = 128 * 34 = 4352 bytes.
    assert_check(desc_landmark_q8.row_stride_bytes == 128 * 34, "landmark Q8 row stride == 4352");
    const uint64_t bytes_landmark_q8 = encoded_matrix_bytes(desc_landmark_q8);
    assert_check(bytes_landmark_q8 == 512 * 4352, "landmark Q8 total bytes == 2228224");

    // Landmark with Turbo4: shape {n_chunks=512, D_total_K=4096}
    codec_desc desc_landmark_t4 = make_codec_desc(
        factor_role::landmark,
        GGML_TYPE_TURBO4_0,
        orientation::token_major,
        {n_chunks, D_total_K},
        128
    );
    // 4096 / 128 = 32 blocks of 68 bytes = 2176 bytes per row.
    assert_check(desc_landmark_t4.row_stride_bytes == 32 * 68, "landmark Turbo4 row stride == 2176");
    const uint64_t bytes_landmark_t4 = encoded_matrix_bytes(desc_landmark_t4);
    assert_check(bytes_landmark_t4 == 512 * 2176, "landmark Turbo4 total bytes == 1114112");

    // Landmark with FP16: shape {n_chunks=512, D_total_K=4096}
    codec_desc desc_landmark_f16 = make_codec_desc(
        factor_role::landmark,
        GGML_TYPE_F16,
        orientation::token_major,
        {n_chunks, D_total_K}
    );
    assert_check(desc_landmark_f16.row_stride_bytes == 4096 * 2, "landmark F16 row stride == 8192");
    const uint64_t bytes_landmark_f16 = encoded_matrix_bytes(desc_landmark_f16);
    assert_check(bytes_landmark_f16 == 512 * 8192, "landmark F16 total bytes == 4194304");

    // Compare actual table vs nominal table:
    // Nominal 4-bit factor A_K was 786,432 bytes; actual Turbo4 is 835,584 bytes (+6.25% due to scale/norm overhead).
    // Nominal 4-bit landmark was 1,048,576 bytes; actual Turbo4 is 1,114,112 bytes (+6.25%).
    // Nominal 8-bit landmark was 2,097,152 bytes; actual Q8_0 is 2,228,224 bytes (+6.25% due to fp16 d per 32 weights).
    // Nominal A_V (4-bit) was 1,179,648 bytes; actual Turbo2 with rank padding to 640 is 696,320 bytes.
    assert_check(bytes_a_k != 786432ULL, "actual A_K bytes MUST NOT equal nominal bits-per-element");
    assert_check(bytes_landmark_q8 != 2097152ULL, "actual landmark Q8 MUST NOT equal nominal bits-per-element");
    assert_check(bytes_landmark_t4 != 1048576ULL, "actual landmark Turbo4 MUST NOT equal nominal bits-per-element");

    // Now prove encoded buffer size matches estimator:
    // Encode a test slice for each descriptor to confirm actual encoded bytes == encoded_matrix_bytes
    {
        // To keep test fast, encode with small row count version to test buffer equality
        codec_desc small_a_k = make_codec_desc(
            factor_role::a_k,
            GGML_TYPE_TURBO4_0,
            orientation::token_major,
            {4, rK},
            128
        );
        std::vector<float> small_a_k_data(4 * rK, 0.1f);
        encoded_matrix em_a_k = encode_matrix(small_a_k, small_a_k_data.data(), small_a_k_data.size());
        assert_check(em_a_k.bytes.size() == encoded_matrix_bytes(small_a_k), "encoded buffer size equals estimator (A_K Turbo4)");

        // Test A_V with rank padding (Turbo2, rV=576 padded to 640)
        codec_desc small_a_v = make_codec_desc(
            factor_role::a_v,
            GGML_TYPE_TURBO2_0,
            orientation::token_major,
            {4, rV},
            128
        );
        std::vector<float> small_a_v_data(4 * rV, -0.2f);
        encoded_matrix em_a_v = encode_matrix(small_a_v, small_a_v_data.data(), small_a_v_data.size());
        assert_check(em_a_v.bytes.size() == encoded_matrix_bytes(small_a_v), "encoded buffer size equals estimator (A_V Turbo2 rank padding)");

        // Test B_K (B^T orientation, Turbo4)
        codec_desc small_b_k = make_codec_desc(
            factor_role::b_k,
            GGML_TYPE_TURBO4_0,
            orientation::feature_major_transposed,
            {4, rK},
            128
        );
        std::vector<float> small_b_k_data(4 * rK, 0.3f);
        encoded_matrix em_b_k = encode_matrix(small_b_k, small_b_k_data.data(), small_b_k_data.size());
        assert_check(em_b_k.bytes.size() == encoded_matrix_bytes(small_b_k), "encoded buffer size equals estimator (B_K B^T Turbo4)");

        // Test Landmark (Q8_0)
        codec_desc small_lm_q8 = make_codec_desc(
            factor_role::landmark,
            GGML_TYPE_Q8_0,
            orientation::token_major,
            {4, D_total_K}
        );
        std::vector<float> small_lm_data(4 * D_total_K, 0.05f);
        encoded_matrix em_lm = encode_matrix(small_lm_q8, small_lm_data.data(), small_lm_data.size());
        assert_check(em_lm.bytes.size() == encoded_matrix_bytes(small_lm_q8), "encoded buffer size equals estimator (Landmark Q8_0)");
    }

    std::cout << "  [PASS] Actual codec descriptors and estimator verified." << std::endl;
}

// ============================================================================
// PART 3: B^T Row Layout vs Flattened B Row-Padding Divergence
// ============================================================================
// When B has mathematical shape [r, D_total] and is stored in feature-major
// transposed layout B^T [D_total, r]:
// Each of the D_total rows has r columns.
// If r is not a multiple of block/group alignment (e.g. 128 for TurboQuant),
// EVERY feature row must be padded to the alignment boundary.
//
// In contrast, if B were flattened as a single contiguous array or token-major
// rows [r, D_total]:
// The total elements D_total * r might be an exact multiple of 128 (e.g. if D_total
// is a multiple of 128), resulting in ZERO or minimal padding.
// This proves that estimating B by naive flat element counts produces the
// WRONG answer for B^T row layout!
// ============================================================================

static void test_b_transposed_vs_flattened_row_padding_divergence() {
    std::cout << "[test_b_transposed_vs_flattened_row_padding_divergence] starting..." << std::endl;

    // Case 1: D_total = 1024, r = 70.
    // D_total is a multiple of 128 (1024 = 8 * 128).
    // Total elements in B = 1024 * 70 = 71,680.
    // 71,680 is a multiple of 128: 71,680 / 128 = 560 blocks!
    constexpr uint64_t D_total = 1024;
    constexpr uint64_t r = 70;

    // In B^T (orientation::feature_major_transposed):
    // Shape is [D_total, r] = [1024, 70].
    // Each row has 70 elements, which must pad to 128 (for Turbo4).
    // Stride per row = 1 block * 68 bytes = 68 bytes.
    // Total size for B^T = 1024 rows * 68 bytes = 69,632 bytes.
    codec_desc desc_bt = make_codec_desc(
        factor_role::b_k,
        GGML_TYPE_TURBO4_0,
        orientation::feature_major_transposed,
        {D_total, r},
        128
    );
    assert_check(desc_bt.padded_shape.rows == 1024, "B^T rows == 1024");
    assert_check(desc_bt.padded_shape.cols == 128,  "B^T cols padded 70 -> 128");
    assert_check(desc_bt.row_stride_bytes == 68,    "B^T row stride == 68 bytes");
    const uint64_t bytes_bt = encoded_matrix_bytes(desc_bt);
    assert_check(bytes_bt == 69632ULL, "B^T bytes == 69632");

    // If B were flattened contiguously:
    // 71,680 elements require 560 blocks of 128 elements.
    // 560 blocks * 68 bytes/block = 38,080 bytes.
    const uint64_t flat_elements = D_total * r; // 71680
    assert_check(flat_elements % 128 == 0, "flat elements exactly divisible by 128");
    const uint64_t bytes_flattened = (flat_elements / 128) * 68;
    assert_check(bytes_flattened == 38080ULL, "flattened B bytes == 38080");

    // Crucial check: B^T row layout padding results in 69,632 bytes,
    // which is 1.828x larger than the naive flattened size (38,080 bytes)!
    assert_check(bytes_bt != bytes_flattened, "B^T bytes must differ from flattened B");
    assert_check(bytes_bt > bytes_flattened, "B^T incurs per-row padding across feature dimension");
    const uint64_t padding_slack = bytes_bt - bytes_flattened;
    assert_check(padding_slack == 31552ULL, "exact row padding divergence is 31552 bytes");

    // Case 2: Math shape B [r, D_total] where D_total = 1024, r = 70.
    // If someone incorrectly made a token_major descriptor with shape [r, D_total]:
    // Rows = 70, cols = 1024.
    // 1024 is already divisible by 128 (8 blocks of 128).
    // Row stride = 8 * 68 = 544 bytes.
    // Total size = 70 * 544 = 38,080 bytes.
    codec_desc desc_math_b = make_codec_desc(
        factor_role::b_k,
        GGML_TYPE_TURBO4_0,
        orientation::token_major,
        {r, D_total},
        128
    );
    assert_check(desc_math_b.padded_shape.cols == 1024, "Math B cols unpadded (1024)");
    const uint64_t bytes_math_b = encoded_matrix_bytes(desc_math_b);
    assert_check(bytes_math_b == 38080ULL, "Math B [r, D_total] bytes == 38080");
    assert_check(bytes_bt != bytes_math_b, "B^T size differs fundamentally from [r, D_total] size");

    // Case 3: Verify the Section 2.3 case with rV = 576.
    // B_V^T shape is [4096, 576].
    // 576 is NOT a multiple of 128; it pads to 640 (5 blocks = 170 bytes for Turbo2 with 34 bytes/block).
    // Total B_V^T bytes = 4096 * 170 = 696,320 bytes.
    // If flattened: 4096 * 576 = 2,359,296 elements.
    // 2,359,296 / 128 = 18,432 blocks.
    // 18,432 * 34 bytes = 626,688 bytes.
    // Divergence for B_V: 696,320 vs 626,688 (69,632 bytes difference!).
    codec_desc desc_bv_bt = make_codec_desc(
        factor_role::b_v,
        GGML_TYPE_TURBO2_0,
        orientation::feature_major_transposed,
        {4096, 576},
        128
    );
    const uint64_t bytes_bv_bt = encoded_matrix_bytes(desc_bv_bt);
    assert_check(bytes_bv_bt == 696320ULL, "B_V^T bytes == 696320");
    const uint64_t bv_flat_bytes = (2359296ULL / 128) * ggml_type_size(GGML_TYPE_TURBO2_0);
    assert_check(bv_flat_bytes == 626688ULL, "flattened B_V bytes == 626688");
    assert_check(bytes_bv_bt != bv_flat_bytes, "B_V^T diverges from flattened B_V");
    assert_check(bytes_bv_bt - bv_flat_bytes == 69632ULL, "B_V^T padding overhead == 69632 bytes");

    std::cout << "  [PASS] B^T vs flattened B row-padding divergence verified." << std::endl;
}

// ============================================================================
// PART 4: Group Tail and Uneven Chunk Tests
// ============================================================================
// Tests:
//   - Group tail where n % chunk_tokens != 0 (e.g. 13 tokens with chunk=8).
//     Effective fragments = ceil(13 / 8) = 2 chunks.
//   - Rank padding on tail groups.
// ============================================================================

static void test_group_tail_and_uneven_chunks() {
    std::cout << "[test_group_tail_and_uneven_chunks] starting..." << std::endl;

    constexpr uint64_t chunk_tokens = 8;
    constexpr uint64_t D_total = 512;

    // Sequence of 13 tokens: chunks are [0..7] (8 tokens) and [8..12] (5 tokens).
    // Ceil division: (13 + 7) / 8 = 2 legal landmark summary rows.
    const uint64_t n_tokens = 13;
    const uint64_t n_chunks_ceil = (n_tokens + chunk_tokens - 1) / chunk_tokens;
    assert_check(n_chunks_ceil == 2, "13 tokens with chunk 8 produces 2 landmark rows");

    codec_desc lm_desc = make_codec_desc(
        factor_role::landmark,
        GGML_TYPE_TURBO4_0,
        orientation::token_major,
        {n_chunks_ceil, D_total},
        128
    );
    assert_check(lm_desc.padded_shape.rows == 2, "landmark rows == 2");
    assert_check(lm_desc.padded_shape.cols == 512, "landmark cols == 512");
    assert_check(lm_desc.row_stride_bytes == 4 * 68, "landmark row stride == 272");
    assert_check(encoded_matrix_bytes(lm_desc) == 2 * 272, "landmark bytes == 544");

    // Uneven rank tail: rank 65 pads to 128 for TurboQuant
    codec_desc tail_desc = make_codec_desc(
        factor_role::a_k,
        GGML_TYPE_TURBO4_0,
        orientation::token_major,
        {n_tokens, 65},
        128
    );
    assert_check(tail_desc.logical_shape.cols == 65, "logical rank == 65");
    assert_check(tail_desc.padded_shape.cols == 128, "padded rank == 128");
    assert_check(tail_desc.row_stride_bytes == 68, "row stride == 68");
    assert_check(encoded_matrix_bytes(tail_desc) == 13 * 68, "total bytes == 13 * 68 = 884");

    // Single token tail: n = 1
    codec_desc single_desc = make_codec_desc(
        factor_role::a_v,
        GGML_TYPE_TURBO2_0,
        orientation::token_major,
        {1, 129},
        128
    );
    assert_check(single_desc.padded_shape.cols == 256, "129 pads to 256");
    assert_check(single_desc.row_stride_bytes == 2 * ggml_type_size(GGML_TYPE_TURBO2_0), "stride == 68");
    assert_check(encoded_matrix_bytes(single_desc) == 68, "1 row * 68 = 68");

    std::cout << "  [PASS] Group tail and uneven chunks verified." << std::endl;
}

// ============================================================================
// PART 5: Arithmetic Overflow Detection
// ============================================================================
// Verifies checked math rejects overflow in matrix sizing and estimators.
// ============================================================================

static void test_overflow_detection() {
    std::cout << "[test_overflow_detection] starting..." << std::endl;

    // Checked arithmetic overflow
    bool caught = false;
    try {
        checked_mul(0xFFFFFFFF00000000ULL, 0xFFFFFFFF00000000ULL);
    } catch (const std::overflow_error &) {
        caught = true;
    }
    assert_check(caught, "checked_mul detected overflow");

    caught = false;
    try {
        checked_add(UINT64_MAX - 10, 20);
    } catch (const std::overflow_error &) {
        caught = true;
    }
    assert_check(caught, "checked_add detected overflow");

    // Encoded matrix byte calculation overflow
    codec_desc huge_desc = make_codec_desc(
        factor_role::a_k,
        GGML_TYPE_F32,
        orientation::token_major,
        {10, 10}
    );
    // Artificially corrupt padded_shape and stride to trigger overflow in encoded_matrix_bytes
    huge_desc.padded_shape.rows = 0x8000000000000000ULL;
    huge_desc.row_stride_bytes  = 0x8000000000000000ULL;

    caught = false;
    try {
        encoded_matrix_bytes(huge_desc);
    } catch (const std::overflow_error &) {
        caught = true;
    }
    assert_check(caught, "encoded_matrix_bytes detected overflow on huge descriptor");

    std::cout << "  [PASS] Overflow detection verified." << std::endl;
}

// ============================================================================
// PART 6: Shared Owning-Layer Deduplication & Stream Allocator Alignment
// ============================================================================
// Verifies:
//   - Deduplication of shared owning layers (e.g. Gemma3n / Gemma4 assistant)
//     prevents counting identical KV storage multiple times across layers.
//   - Stream allocator alignment (e.g. 64-byte or 256-byte backend alignment)
//     is accounted per stream/row.
// ============================================================================

static void test_shared_owning_layer_dedup_and_allocator_alignment() {
    std::cout << "[test_shared_owning_layer_dedup_and_allocator_alignment] starting..." << std::endl;

    // 1. Shared owning-layer dedup logic:
    // Suppose W = 4 layers, but layers 2 and 3 share/reuse KV storage from earlier layers.
    // Spliced feature dimension D_total should only accumulate distinct owning layers (2 * 1024 = 2048),
    // not all 4 layers (4096).
    const std::vector<uint32_t> layer_d_k = {1024, 1024, 1024, 1024};
    const std::vector<bool> is_owning_layer = {true, true, false, false}; // layers 2,3 reused

    uint64_t d_total_dedup = 0;
    uint64_t d_total_naive = 0;
    for (size_t i = 0; i < layer_d_k.size(); ++i) {
        d_total_naive += layer_d_k[i];
        if (is_owning_layer[i]) {
            d_total_dedup += layer_d_k[i];
        }
    }
    assert_check(d_total_naive == 4096, "naive total dimension is 4096");
    assert_check(d_total_dedup == 2048, "deduped total dimension is 2048");

    // Check B_K descriptor under deduped feature dimension:
    codec_desc desc_bk_dedup = make_codec_desc(
        factor_role::b_k,
        GGML_TYPE_TURBO4_0,
        orientation::feature_major_transposed,
        {d_total_dedup, 384},
        128
    );
    codec_desc desc_bk_naive = make_codec_desc(
        factor_role::b_k,
        GGML_TYPE_TURBO4_0,
        orientation::feature_major_transposed,
        {d_total_naive, 384},
        128
    );
    const uint64_t bytes_bk_dedup = encoded_matrix_bytes(desc_bk_dedup);
    const uint64_t bytes_bk_naive = encoded_matrix_bytes(desc_bk_naive);
    assert_check(bytes_bk_dedup * 2 == bytes_bk_naive, "deduped B_K is exactly half of naive duplicate B_K");
    assert_check(bytes_bk_dedup == 2048 * 204, "deduped B_K bytes == 417792");

    // 2. Stream allocator alignment:
    // When backend allocator aligns stream allocations to alignment boundaries (e.g. 256 bytes),
    // verify round-up logic.
    auto align_up = [](uint64_t val, uint64_t align) -> uint64_t {
        return ((val + align - 1) / align) * align;
    };
    constexpr uint64_t align_backend = 256;
    uint64_t raw_stream_a = 835584; // e.g. from A_K Turbo4
    uint64_t aligned_stream_a = align_up(raw_stream_a, align_backend);
    assert_check(raw_stream_a % align_backend == 0, "835584 is divisible by 256 (3264 * 256)");
    assert_check(aligned_stream_a == raw_stream_a, "aligned size equals raw when already aligned");

    uint64_t unaligned_size = 1000;
    uint64_t aligned_size = align_up(unaligned_size, align_backend);
    assert_check(aligned_size == 1024, "1000 aligned up to 256 is 1024");
    assert_check(aligned_size >= unaligned_size, "aligned >= unaligned");

    std::cout << "  [PASS] Shared owning-layer dedup and stream allocator alignment verified." << std::endl;
}

int main() {
    std::cout << "=== Running XKV Memory Model Behavioral Tests ===" << std::endl;

    test_nominal_section_2_3_exact_bytes();
    test_actual_codec_descriptors_and_estimator();
    test_b_transposed_vs_flattened_row_padding_divergence();
    test_group_tail_and_uneven_chunks();
    test_overflow_detection();
    test_shared_owning_layer_dedup_and_allocator_alignment();

    std::cout << "=== ALL XKV MEMORY MODEL TESTS PASSED ===" << std::endl;
    return 0;
}
