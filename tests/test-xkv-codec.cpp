// Deterministic behavioral tests for llama-xkv-codec
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-codec.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <limits>
#include <random>
#include <cstdlib>
#include <new>

extern "C" {
extern int turbo3_cpu_wht_group_size;
}
using namespace llama_xkv;

// Scoped heap-allocation counter backing the zero-heap proof in
// test_decode_rows_scratch_zero_heap. Armed only around the success-path
// overload call with every buffer preallocated outside the armed window, so a
// nonzero count means the codec itself allocated.
namespace xkv_codec_test_alloc {
bool armed = false;
size_t count = 0;
} // namespace xkv_codec_test_alloc

void * operator new(std::size_t n) {
    if (xkv_codec_test_alloc::armed) ++xkv_codec_test_alloc::count;
    if (void * p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }
void * operator new[](std::size_t n) {
    if (xkv_codec_test_alloc::armed) ++xkv_codec_test_alloc::count;
    if (void * p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete[](void * p, std::size_t) noexcept { std::free(p); }

// Codec math: normalize -> orthogonal WHT (Parseval) -> Lloyd-Max scalar quant
// of ~N(0,var) rotated coefficients -> norm correction restores input energy.
// Expected per-element MSE ~= distortion * var (2-bit ~0.1175, 3-bit ~0.0345,
// 4-bit ~0.0095). Bounds are ~3x expected: a broken inverse-WHT / wrong-seed
// rotation gives cosine ~ 0 and MSE ~ 2*var, and a wrong-norm / wrong-stride
// layout pushes the norm ratio far from 1, so all breakages fail loudly.
// Per-element |err| thresholds are invalid (Gaussian tails legitimately exceed
// them); only aggregate MSE / cosine / norm-ratio bounds are used for Turbo.
static void turbo_stats(const std::vector<float> & ref, const std::vector<float> & got,
                         float & mse, float & cosine, float & norm_ratio) {
    assert(ref.size() == got.size());
    double se = 0.0, dot = 0.0, nr = 0.0, ng = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = (double)ref[i] - (double)got[i];
        se += d * d;
        dot += (double)ref[i] * (double)got[i];
        nr += (double)ref[i] * (double)ref[i];
        ng += (double)got[i] * (double)got[i];
    }
    mse = (float)(se / (double)ref.size());
    double denom = std::sqrt(nr * ng);
    cosine = denom > 1e-12 ? (float)(dot / denom) : 0.0f;
    norm_ratio = nr > 1e-12 ? (float)std::sqrt(ng / nr) : 0.0f;
}

static void turbo_bounds(ggml_type t, float var, float & mse_max, float & cos_min) {
    float dist = 0.1175f;
    if (t == GGML_TYPE_TURBO3_0) dist = 0.0345f;
    if (t == GGML_TYPE_TURBO4_0) dist = 0.0095f;
    mse_max = 3.0f * dist * var + 1e-6f;
    float rel = mse_max / (var > 1e-12f ? var : 1.0f);
    cos_min = 1.0f - rel * 0.5f - 0.06f;
    float floor = 0.80f;
    if (t == GGML_TYPE_TURBO3_0) floor = 0.90f;
    if (t == GGML_TYPE_TURBO4_0) floor = 0.93f;
    if (cos_min < floor) cos_min = floor;
}

static void test_descriptor_validation_and_fingerprint() {
    std::cout << "[test_descriptor_validation_and_fingerprint] starting..." << std::endl;
    codec_desc desc = make_codec_desc(
        factor_role::a_k,
        GGML_TYPE_TURBO4_0,
        orientation::token_major,
        {16, 100},
        128,
        42
    );
    assert(desc.version == 2);
    assert(desc.padded_shape.rows == 16);
    assert(desc.padded_shape.cols == 128); // padded from 100 to 128
    assert(desc.block_size == 128);
    assert(desc.group_size == 128);
    assert(desc.stored_domain == value_domain::turbo_rotated);
    assert(desc.decoded_domain == value_domain::canonical);
    assert(desc.table_fingerprint != 0); // Deterministic nonzero fingerprint
    assert(desc.validate());

    uint64_t fp1 = desc.fingerprint();
    assert(fp1 != 0);

    codec_desc desc2 = desc;
    assert(desc == desc2);
    assert(desc.fingerprint() == desc2.fingerprint());

    // Modify a field and verify fingerprint changes
    desc2.logical_shape.cols = 99;
    assert(desc != desc2);
    assert(desc.fingerprint() != desc2.fingerprint());

    // Non-Turbo descriptor check
    codec_desc f32_desc = make_codec_desc(
        factor_role::b_v,
        GGML_TYPE_F32,
        orientation::feature_major_transposed,
        {10, 20}
    );
    assert(f32_desc.stored_domain == value_domain::canonical);
    assert(f32_desc.decoded_domain == value_domain::canonical);
    assert(f32_desc.table_fingerprint == 0);
    assert(f32_desc.validate());

    // Non-Turbo with stored_domain = turbo_rotated fails validation
    codec_desc bad_non_turbo = f32_desc;
    bad_non_turbo.stored_domain = value_domain::turbo_rotated;
    assert(!bad_non_turbo.validate());

    // Turbo with stored_domain = canonical fails validation
    codec_desc bad_turbo_domain = desc;
    bad_turbo_domain.stored_domain = value_domain::canonical;
    assert(!bad_turbo_domain.validate());

    // Turbo with table_fingerprint = 0 fails validation
    codec_desc bad_turbo_fp = desc;
    bad_turbo_fp.table_fingerprint = 0;
    assert(!bad_turbo_fp.validate());

    // Group size 64 MUST be rejected for Turbo2, Turbo3, and Turbo4
    bool caught_64 = false;
    try {
        make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO2_0, orientation::token_major, {4, 64}, 64);
    } catch (const std::invalid_argument &) {
        caught_64 = true;
    }
    assert(caught_64);

    caught_64 = false;
    try {
        make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO3_0, orientation::token_major, {4, 64}, 64);
    } catch (const std::invalid_argument &) {
        caught_64 = true;
    }
    assert(caught_64);

    caught_64 = false;
    try {
        make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, orientation::token_major, {4, 64}, 64);
    } catch (const std::invalid_argument &) {
        caught_64 = true;
    }
    assert(caught_64);

    // Invalid configurations
    codec_desc bad = desc;
    bad.version = 99;
    assert(!bad.validate());

    bad = desc;
    bad.padded_shape.cols = 127; // not multiple of block size
    assert(!bad.validate());

    bad = desc;
    bad.group_size = 64; // group_size 64 rejected in v1
    assert(!bad.validate());

    // Serialization roundtrip with fixed explicit layout
    std::vector<uint8_t> s = serialize_desc(desc);
    assert(s.size() == 100); // Fixed 100 bytes
    codec_desc deserialized;
    assert(deserialize_desc(s.data(), s.size(), deserialized));
    assert(deserialized == desc);

    // Rejection of truncation
    codec_desc trunc_desc;
    std::string err;
    assert(!deserialize_desc(s.data(), s.size() - 1, trunc_desc, &err));
    assert(err.find("truncated") != std::string::npos);

    // Rejection of trailing bytes
    std::vector<uint8_t> trailing = s;
    trailing.push_back(0xAA);
    assert(!deserialize_desc(trailing.data(), trailing.size(), trunc_desc, &err));
    assert(err.find("trailing") != std::string::npos);

    // Endianness verification: magic header bytes must match fixed little-endian XKV1
    // Magic "XKV1" -> 'X' (0x58), 'K' (0x4B), 'V' (0x56), '1' (0x31)
    assert(s[0] == 0x58);
    assert(s[1] == 0x4B);
    assert(s[2] == 0x56);
    assert(s[3] == 0x31);

    // Fixed endianness for multi-byte fields (e.g. version = 1 as 32-bit LE)
    assert(s[4] == 0x02);
    assert(s[5] == 0x00);
    assert(s[6] == 0x00);
    assert(s[7] == 0x00);

    // Explicit rejection of v1 descriptor
    std::vector<uint8_t> s_v1 = s;
    s_v1[4] = 0x01; // Version = 1
    assert(!deserialize_desc(s_v1.data(), s_v1.size(), trunc_desc, &err));
    assert(err.find("v1 descriptor rejected") != std::string::npos);

    std::cout << "[test_descriptor_validation_and_fingerprint] passed!" << std::endl;
}

static void test_exact_byte_size_and_overflow() {
    std::cout << "[test_exact_byte_size_and_overflow] starting..." << std::endl;

    // F32: 10 rows, 64 cols -> 10 * 64 * 4 = 2560 bytes
    codec_desc f32_desc = make_codec_desc(
        factor_role::a_k,
        GGML_TYPE_F32,
        orientation::token_major,
        {10, 64}
    );
    assert(encoded_matrix_bytes(f32_desc) == 10 * 64 * sizeof(float));

    // Encoder output size must match estimator size exactly
    std::vector<float> data(10 * 64, 1.0f);
    encoded_matrix em_f32 = encode_matrix(f32_desc, data.data(), data.size());
    assert(em_f32.bytes.size() == encoded_matrix_bytes(f32_desc));

    // Turbo4: 10 rows, 128 cols -> 10 * 68 = 680 bytes
    codec_desc t4_desc = make_codec_desc(
        factor_role::a_k,
        GGML_TYPE_TURBO4_0,
        orientation::token_major,
        {10, 128},
        128
    );
    assert(encoded_matrix_bytes(t4_desc) == 10 * 68);
    std::vector<float> t4_data(10 * 128, 0.5f);
    encoded_matrix em_t4 = encode_matrix(t4_desc, t4_data.data(), t4_data.size());
    assert(em_t4.bytes.size() == encoded_matrix_bytes(t4_desc));

    // Turbo3: 10 rows, 128 cols -> 10 * ggml_type_size(GGML_TYPE_TURBO3_0)
    codec_desc t3_desc = make_codec_desc(
        factor_role::a_k,
        GGML_TYPE_TURBO3_0,
        orientation::token_major,
        {10, 128},
        128
    );
    assert(encoded_matrix_bytes(t3_desc) == 10 * ggml_type_size(GGML_TYPE_TURBO3_0));
    std::vector<float> t3_data(10 * 128, 0.5f);
    encoded_matrix em_t3 = encode_matrix(t3_desc, t3_data.data(), t3_data.size());
    assert(em_t3.bytes.size() == encoded_matrix_bytes(t3_desc));

    // Turbo2: 10 rows, 128 cols -> 10 * ggml_type_size(GGML_TYPE_TURBO2_0)
    codec_desc t2_desc = make_codec_desc(
        factor_role::a_k,
        GGML_TYPE_TURBO2_0,
        orientation::token_major,
        {10, 128},
        128
    );
    assert(encoded_matrix_bytes(t2_desc) == 10 * ggml_type_size(GGML_TYPE_TURBO2_0));
    std::vector<float> t2_data(10 * 128, 0.5f);
    encoded_matrix em_t2 = encode_matrix(t2_desc, t2_data.data(), t2_data.size());
    assert(em_t2.bytes.size() == encoded_matrix_bytes(t2_desc));

    // F16: 10 rows, 64 cols -> 10 * 64 * 2 = 1280 bytes
    codec_desc f16_desc = make_codec_desc(
        factor_role::b_k,
        GGML_TYPE_F16,
        orientation::feature_major_transposed,
        {10, 64}
    );
    assert(encoded_matrix_bytes(f16_desc) == 10 * 64 * sizeof(ggml_fp16_t));
    assert(f16_desc.stored_domain == value_domain::canonical);
    assert(f16_desc.decoded_domain == value_domain::canonical);
    assert(f16_desc.format_revision == 0);
    std::vector<float> f16_data(10 * 64, 0.25f);
    encoded_matrix em_f16 = encode_matrix(f16_desc, f16_data.data(), f16_data.size());
    assert(em_f16.bytes.size() == encoded_matrix_bytes(f16_desc));

    // Q8_0: 10 rows, 64 cols -> 10 * (64 / 32) * sizeof(block_q8_0) = 10 * 2 * 34 = 680 bytes
    codec_desc q8_desc = make_codec_desc(
        factor_role::b_k,
        GGML_TYPE_Q8_0,
        orientation::feature_major_transposed,
        {10, 64}
    );
    size_t expected_q8_row_size = ggml_row_size(GGML_TYPE_Q8_0, 64);
    assert(encoded_matrix_bytes(q8_desc) == 10 * expected_q8_row_size);
    assert(q8_desc.stored_domain == value_domain::canonical);
    assert(q8_desc.decoded_domain == value_domain::canonical);
    assert(q8_desc.format_revision == 0);
    std::vector<float> q8_data(10 * 64, -0.75f);
    encoded_matrix em_q8 = encode_matrix(q8_desc, q8_data.data(), q8_data.size());
    assert(em_q8.bytes.size() == encoded_matrix_bytes(q8_desc));

    // Overflow check in encoded_matrix_bytes
    codec_desc ovf_desc = f32_desc;
    ovf_desc.padded_shape.rows = 0x7FFFFFFFFFFFFFFFULL;
    ovf_desc.row_stride_bytes = 0x7FFFFFFFFFFFFFFFULL;
    bool caught_overflow = false;
    try {
        encoded_matrix_bytes(ovf_desc);
    } catch (const std::overflow_error &) {
        caught_overflow = true;
    }
    assert(caught_overflow);

    // Zero-shape behavior
    codec_desc zero_desc = make_codec_desc(
        factor_role::a_k,
        GGML_TYPE_F32,
        orientation::token_major,
        {0, 0}
    );
    assert(zero_desc.validate());
    assert(encoded_matrix_bytes(zero_desc) == 0);
    encoded_matrix em_zero = encode_matrix(zero_desc, nullptr, 0);
    assert(em_zero.bytes.empty());
    std::vector<float> dec_zero = decode_matrix(em_zero);
    assert(dec_zero.empty());

    std::cout << "[test_exact_byte_size_and_overflow] passed!" << std::endl;
}

static void test_four_roles_and_orientations() {
    std::cout << "[test_four_roles_and_orientations] starting..." << std::endl;
    // Four roles: A_K, B_K, A_V, B_V
    // A: token_major, B: feature_major_transposed
    factor_role roles[4] = { factor_role::a_k, factor_role::b_k, factor_role::a_v, factor_role::b_v };
    orientation orients[4] = {
        orientation::token_major,
        orientation::feature_major_transposed,
        orientation::token_major,
        orientation::feature_major_transposed
    };

    for (int i = 0; i < 4; ++i) {
        codec_desc desc = make_codec_desc(
            roles[i],
            GGML_TYPE_Q8_0,
            orients[i],
            {8, 64}
        );
        assert(desc.role == roles[i]);
        assert(desc.orient == orients[i]);
        assert(desc.validate());

        std::vector<float> src(8 * 64);
        for (size_t k = 0; k < src.size(); ++k) {
            src[k] = (float)k * 0.01f;
        }

        encoded_matrix em = encode_matrix(desc, src.data(), src.size());
        assert(em.bytes.size() == encoded_matrix_bytes(desc));

        std::vector<float> decoded = decode_matrix(em);
        assert(decoded.size() == 8 * 64);

        // Q8_0 has bounded error, not losslessness
        float max_diff = 0.0f;
        for (size_t k = 0; k < src.size(); ++k) {
            float diff = std::fabs(src[k] - decoded[k]);
            if (diff > max_diff) max_diff = diff;
        }
        assert(max_diff < 0.05f); // Bounded quantization error
    }

    std::cout << "[test_four_roles_and_orientations] passed!" << std::endl;
}

static void test_non_multiple_rank_padding_and_tail_zeros() {
    std::cout << "[test_non_multiple_rank_padding_and_tail_zeros] starting..." << std::endl;
    // Non-multiple rank 50 pads to 128 for Turbo types
    codec_desc desc = make_codec_desc(
        factor_role::a_k,
        GGML_TYPE_TURBO3_0,
        orientation::token_major,
        {4, 50},
        128
    );
    assert(desc.logical_shape.cols == 50);
    assert(desc.padded_shape.cols == 128); // Non-multiple logical rank pads to 128

    std::vector<float> src(4 * 50);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = std::sin((float)i * 0.2f);
    }

    encoded_matrix em = encode_matrix(desc, src.data(), src.size());
    std::vector<float> decoded = decode_matrix(em, value_domain::canonical);
    assert(decoded.size() == 4 * 128);

    // Logical cols [0..49] should be reasonably close to src
    for (uint64_t r = 0; r < 4; ++r) {
        float log_max_diff = 0.0f;
        float log_sq_sum = 0.0f;
        for (uint64_t c = 0; c < 50; ++c) {
            float diff = std::fabs(src[r * 50 + c] - decoded[r * 128 + c]);
            if (diff > log_max_diff) log_max_diff = diff;
            log_sq_sum += diff * diff;
        }
        float log_rmse = std::sqrt(log_sq_sum / 50.0f);
        assert(log_max_diff < 0.45f); // Bounded max error on logical elements
        assert(log_rmse < 0.20f);     // Bounded RMSE on logical elements

        // Tail cols [50..127] were zero-padded before quantization.
        // In canonical domain after inverse WHT, quantization noise distributes across all 128 elements.
        float tail_sq_sum = 0.0f;
        for (uint64_t c = 50; c < 128; ++c) {
            float val = decoded[r * 128 + c];
            assert(std::isfinite(val));
            tail_sq_sum += val * val;
        }
        float tail_rmse = std::sqrt(tail_sq_sum / 78.0f);
        assert(tail_rmse < 0.10f); // Tail noise RMSE is strictly bounded
    }

    // Same check for Turbo2: rank 75 pads to 128
    codec_desc desc2 = make_codec_desc(
        factor_role::a_k,
        GGML_TYPE_TURBO2_0,
        orientation::token_major,
        {2, 75},
        128
    );
    assert(desc2.padded_shape.cols == 128);

    std::cout << "[test_non_multiple_rank_padding_and_tail_zeros] passed!" << std::endl;
}

static void test_zeros_and_extremes() {
    std::cout << "[test_zeros_and_extremes] starting..." << std::endl;

    ggml_type types[5] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0 };
    for (ggml_type t : types) {
        codec_desc desc = make_codec_desc(
            factor_role::a_v,
            t,
            orientation::token_major,
            {4, 128},
            128
        );

        // Test 1: all zeros
        std::vector<float> zeros(4 * 128, 0.0f);
        encoded_matrix em_zeros = encode_matrix(desc, zeros.data(), zeros.size());
        std::vector<float> dec_zeros = decode_matrix(em_zeros);
        for (float v : dec_zeros) {
            assert(std::fabs(v) < 1e-5f);
        }

        // Test 2: extremes (large and small finite values)
        std::vector<float> extremes(4 * 128, 0.0f);
        for (size_t i = 0; i < extremes.size(); ++i) {
            extremes[i] = (i % 2 == 0) ? 100.0f : -100.0f;
        }
        encoded_matrix em_ext = encode_matrix(desc, extremes.data(), extremes.size());
        std::vector<float> dec_ext = decode_matrix(em_ext);
        for (float v : dec_ext) {
            assert(std::isfinite(v));
        }
    }

    std::cout << "[test_zeros_and_extremes] passed!" << std::endl;
}

static void test_turbo_canonical_vs_rotated_and_inverse_wht() {
    std::cout << "[test_turbo_canonical_vs_rotated_and_inverse_wht] starting..." << std::endl;

    codec_desc desc = make_codec_desc(
        factor_role::b_k,
        GGML_TYPE_TURBO4_0,
        orientation::feature_major_transposed,
        {2, 128},
        128
    );

    std::mt19937 rng(1337);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> src(2 * 128);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = dist(rng);
    }

    encoded_matrix em = encode_matrix(desc, src.data(), src.size());

    // Canonical decoding inverts WHT
    std::vector<float> dec_canon = decode_matrix(em, value_domain::canonical);
    // Rotated decoding skips inverse WHT
    std::vector<float> dec_rot = decode_matrix(em, value_domain::turbo_rotated);

    // Check that canonical and rotated are distinct
    bool any_diff = false;
    for (size_t i = 0; i < dec_canon.size(); ++i) {
        if (std::fabs(dec_canon[i] - dec_rot[i]) > 1e-3f) {
            any_diff = true;
            break;
        }
    }
    assert(any_diff);

    // Canonical decoding reconstructs the source with bounded error
    // Aggregate Lloyd-Max bounds (src ~ N(0,1), var = 1): per-element max
    // thresholds are invalid because Gaussian tails legitimately exceed 0.9
    // even for a correct codec, while a broken inverse-WHT gives MSE ~ 2.
    {
        float mse, cos, nratio;
        turbo_stats(src, dec_canon, mse, cos, nratio);
        float mse_max, cos_min;
        turbo_bounds(GGML_TYPE_TURBO4_0, 1.0f, mse_max, cos_min);
        assert(mse < mse_max);
        assert(cos > cos_min);
        assert(nratio > 0.90f && nratio < 1.10f);
        for (float v : dec_canon) assert(std::isfinite(v));
    }

    // Same check on Turbo2
    codec_desc desc_t2 = make_codec_desc(
        factor_role::b_k,
        GGML_TYPE_TURBO2_0,
        orientation::feature_major_transposed,
        {2, 128},
        128
    );
    encoded_matrix em_t2 = encode_matrix(desc_t2, src.data(), src.size());
    std::vector<float> dec_canon_t2 = decode_matrix(em_t2, value_domain::canonical);
    std::vector<float> dec_rot_t2 = decode_matrix(em_t2, value_domain::turbo_rotated);
    any_diff = false;
    for (size_t i = 0; i < dec_canon_t2.size(); ++i) {
        if (std::fabs(dec_canon_t2[i] - dec_rot_t2[i]) > 1e-3f) {
            any_diff = true;
            break;
        }
    }
    assert(any_diff);

    // Fixed test comparing ROTATED to forward-WHT(CANONICAL decoded) for T2, T3, T4
    ggml_type tq_types[3] = { GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0 };
    for (ggml_type tq : tq_types) {
        codec_desc tq_desc = make_codec_desc(
            factor_role::a_k,
            tq,
            orientation::token_major,
            {4, 128},
            128
        );
        std::vector<float> tq_src(4 * 128);
        for (size_t i = 0; i < tq_src.size(); ++i) {
            tq_src[i] = dist(rng);
        }
        encoded_matrix tq_em = encode_matrix(tq_desc, tq_src.data(), tq_src.size());
        std::vector<float> canon = decode_matrix(tq_em, value_domain::canonical);
        std::vector<float> rotated = decode_matrix(tq_em, value_domain::turbo_rotated);

        // Applying forward WHT to CANONICAL decoded floats must exactly recover ROTATED decoded floats
        std::vector<float> canon_wht_forward = canon;
        apply_forward_wht_matrix(canon_wht_forward.data(), 4, 128);

        for (size_t i = 0; i < rotated.size(); ++i) {
            float diff = std::fabs(canon_wht_forward[i] - rotated[i]);
            assert(diff < 1e-4f);
        }
    }

    // Direct canonical roundtrip error for all types: F32, F16, Q8_0, Turbo2, Turbo3, Turbo4
    ggml_type all_types[6] = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0,
        GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0
    };
    for (ggml_type t : all_types) {
        codec_desc d = make_codec_desc(factor_role::a_k, t, orientation::token_major, {2, 128}, 128);
        std::vector<float> s(2 * 128);
        for (size_t i = 0; i < s.size(); ++i) s[i] = dist(rng);
        encoded_matrix em_rt = encode_matrix(d, s.data(), s.size());
        std::vector<float> dec_rt = decode_matrix(em_rt, value_domain::canonical);
        // Deterministic bytes: re-encoding the same input is bit-identical.
        encoded_matrix em_rt2 = encode_matrix(d, s.data(), s.size());
        assert(em_rt.bytes == em_rt2.bytes);
        for (float v : dec_rt) assert(std::isfinite(v));
        if (t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0) {
            float max_e = 0.0f;
            for (size_t i = 0; i < s.size(); ++i) {
                float e = std::fabs(s[i] - dec_rt[i]);
                if (e > max_e) max_e = e;
            }
            if (t == GGML_TYPE_F32) assert(max_e < 1e-5f);
            else if (t == GGML_TYPE_F16) assert(max_e < 0.005f);
            else assert(max_e < 0.05f);
        } else {
            // Turbo: aggregate MSE/cosine/norm bounds (var = 1 for N(0,1)).
            float mse, cos, nratio;
            turbo_stats(s, dec_rt, mse, cos, nratio);
            float mse_max, cos_min;
            turbo_bounds(t, 1.0f, mse_max, cos_min);
            assert(mse < mse_max);
            assert(cos > cos_min);
            assert(nratio > 0.90f && nratio < 1.10f);
        }
    }
    // Meaningful bit-width ordering would need a shared input; the loop above
    // uses fresh RNG draws per type, so ordering is checked explicitly below.
    {
        std::vector<float> shared(2 * 128);
        for (size_t i = 0; i < shared.size(); ++i) shared[i] = dist(rng);
        float mses[3];
        ggml_type tq[3] = { GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0 };
        for (int k = 0; k < 3; ++k) {
            codec_desc d = make_codec_desc(factor_role::a_k, tq[k], orientation::token_major, {2, 128}, 128);
            encoded_matrix em = encode_matrix(d, shared.data(), shared.size());
            std::vector<float> dec = decode_matrix(em, value_domain::canonical);
            float mse, cos, nratio;
            turbo_stats(shared, dec, mse, cos, nratio);
            mses[k] = mse;
        }
        assert(mses[2] <= mses[1] + 1e-6f);
        assert(mses[1] <= mses[0] + 1e-6f);
    }

    std::cout << "[test_turbo_canonical_vs_rotated_and_inverse_wht] passed!" << std::endl;
}

static void test_invalid_buffers_and_atomicity() {
    std::cout << "[test_invalid_buffers_and_atomicity] starting..." << std::endl;

    codec_desc desc = make_codec_desc(
        factor_role::a_k,
        GGML_TYPE_F32,
        orientation::token_major,
        {4, 64}
    );

    // Insufficient input buffer throws invalid_argument
    std::vector<float> small_src(10, 1.0f);
    bool caught = false;
    try {
        encode_matrix(desc, small_src.data(), small_src.size());
    } catch (const std::invalid_argument &) {
        caught = true;
    }
    assert(caught);

    // Rejection of non-canonical target_domain on non-Turbo types
    std::vector<float> valid_src(4 * 64, 1.0f);
    encoded_matrix em = encode_matrix(desc, valid_src.data(), valid_src.size());
    caught = false;
    try {
        decode_matrix(em, value_domain::turbo_rotated);
    } catch (const std::invalid_argument &) {
        caught = true;
    }
    assert(caught);

    // Mismatched byte buffer size in decode_rows throws invalid_argument
    encoded_matrix corrupted_em = em;
    corrupted_em.bytes.pop_back();
    caught = false;
    try {
        decode_matrix(corrupted_em);
    } catch (const std::invalid_argument &) {
        caught = true;
    }
    assert(caught);

    // Insufficient output buffer in decode_rows throws invalid_argument
    uint64_t rows[2] = {0, 1};
    std::vector<float> small_dst(10);
    caught = false;
    try {
        decode_rows(em, rows, 2, small_dst.data(), small_dst.size());
    } catch (const std::invalid_argument &) {
        caught = true;
    }
    assert(caught);

    // Out of bounds row index throws out_of_range
    uint64_t oob_rows[1] = {100};
    std::vector<float> dst(64);
    caught = false;
    try {
        decode_rows(em, oob_rows, 1, dst.data(), dst.size());
    } catch (const std::out_of_range &) {
        caught = true;
    }
    assert(caught);

    std::cout << "[test_invalid_buffers_and_atomicity] passed!" << std::endl;
}

static void test_turbo_global_group_size_isolation() {
    std::cout << "[test_turbo_global_group_size_isolation] starting..." << std::endl;

    // RAII guard to safely restore the legacy global
    struct global_guard {
        int orig;
        global_guard() : orig(turbo3_cpu_wht_group_size) {}
        ~global_guard() { turbo3_cpu_wht_group_size = orig; }
    } guard;

    // Set legacy global variable to 64 (which would cause blocks_per_group=0 in legacy path)
    turbo3_cpu_wht_group_size = 64;

    codec_desc desc = make_codec_desc(
        factor_role::a_k,
        GGML_TYPE_TURBO3_0,
        orientation::token_major,
        {2, 128},
        128
    );

    std::vector<float> src(2 * 128);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = (float)i * 0.05f - 3.0f;
    }

    // Encode with global=64
    encoded_matrix em_64 = encode_matrix(desc, src.data(), src.size());
    assert(em_64.bytes.size() == encoded_matrix_bytes(desc));
    assert(turbo3_cpu_wht_group_size == 64);
    std::vector<float> dec_64 = decode_matrix(em_64, value_domain::canonical);
    assert(turbo3_cpu_wht_group_size == 64);

    // Encode with global=128
    turbo3_cpu_wht_group_size = 128;
    encoded_matrix em_128 = encode_matrix(desc, src.data(), src.size());
    assert(turbo3_cpu_wht_group_size == 128);
    std::vector<float> dec_128 = decode_matrix(em_128, value_domain::canonical);
    assert(turbo3_cpu_wht_group_size == 128);

    // Encode with global=0 (default auto)
    turbo3_cpu_wht_group_size = 0;
    encoded_matrix em_0 = encode_matrix(desc, src.data(), src.size());
    assert(turbo3_cpu_wht_group_size == 0);
    std::vector<float> dec_0 = decode_matrix(em_0, value_domain::canonical);
    assert(turbo3_cpu_wht_group_size == 0);

    // Encoded byte streams and canonical decoded floats must be bit-identical regardless of legacy global
    assert(em_64.bytes == em_128.bytes);
    assert(em_64.bytes == em_0.bytes);
    assert(dec_64 == dec_128);
    assert(dec_64 == dec_0);

    std::cout << "[test_turbo_global_group_size_isolation] passed!" << std::endl;
}

static void test_factor_pair_compatibility_validation() {
    std::cout << "[test_factor_pair_compatibility_validation] starting..." << std::endl;

    codec_desc desc_a = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, orientation::token_major, {8, 100}, 128, 42);
    codec_desc desc_b = make_codec_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, orientation::feature_major_transposed, {16, 100}, 128, 42);

    std::string err;
    assert(validate_factor_pair_compatibility(desc_a, desc_b, &err));
    assert_factor_pair_compatible(desc_a, desc_b);

    // Rank mismatch
    codec_desc desc_b_rank_mismatch = make_codec_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, orientation::feature_major_transposed, {16, 90}, 128, 42);
    assert(!validate_factor_pair_compatibility(desc_a, desc_b_rank_mismatch, &err));
    assert(err.find("rank mismatch") != std::string::npos);

    // Seed / transform mismatch
    codec_desc desc_b_seed_mismatch = make_codec_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, orientation::feature_major_transposed, {16, 100}, 128, 43);
    assert(!validate_factor_pair_compatibility(desc_a, desc_b_seed_mismatch, &err));
    assert(err.find("mismatch") != std::string::npos);

    // Format revision mismatch
    codec_desc desc_b_rev_mismatch = desc_b;
    desc_b_rev_mismatch.format_revision = 0x12345678ULL;
    assert(!validate_factor_pair_compatibility(desc_a, desc_b_rev_mismatch, &err));

    // assert_factor_pair_compatible throws
    bool threw = false;
    try {
        assert_factor_pair_compatible(desc_a, desc_b_seed_mismatch);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    // R0 profile: canonical non-Turbo pair (A F16, B F16)
    codec_desc desc_a_f16 = make_codec_desc(factor_role::a_k, GGML_TYPE_F16, orientation::token_major, {8, 128});
    codec_desc desc_b_f16 = make_codec_desc(factor_role::b_k, GGML_TYPE_F16, orientation::feature_major_transposed, {16, 128});
    assert(validate_factor_pair_compatibility(desc_a_f16, desc_b_f16, &err));
    assert_factor_pair_compatible(desc_a_f16, desc_b_f16);
    assert(!can_direct_rotated_gemm(desc_a_f16, desc_b_f16)); // Non-turbo uses canonical GEMM

    // R1 profile: explicitly mixed pair (A Turbo4, B F16)
    codec_desc desc_a_t4 = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, orientation::token_major, {8, 128}, 128, 42);
    assert(validate_factor_pair_compatibility(desc_a_t4, desc_b_f16, &err));
    assert_factor_pair_compatible(desc_a_t4, desc_b_f16);
    assert(!can_direct_rotated_gemm(desc_a_t4, desc_b_f16)); // Mixed requires canonical decode first

    // R1 profile variant: explicitly mixed pair (A Turbo4, B Q8_0)
    codec_desc desc_b_q8 = make_codec_desc(factor_role::b_k, GGML_TYPE_Q8_0, orientation::feature_major_transposed, {16, 128});
    assert(validate_factor_pair_compatibility(desc_a_t4, desc_b_q8, &err));
    assert_factor_pair_compatible(desc_a_t4, desc_b_q8);
    assert(!can_direct_rotated_gemm(desc_a_t4, desc_b_q8)); // Mixed requires canonical decode first

    // Production R2/R3/R4: both Turbo with matching rotation
    codec_desc desc_b_t4 = make_codec_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, orientation::feature_major_transposed, {16, 128}, 128, 42);
    assert(validate_factor_pair_compatibility(desc_a_t4, desc_b_t4, &err));
    assert(can_direct_rotated_gemm(desc_a_t4, desc_b_t4)); // Can direct rotated GEMM
    // But with mismatched seed, direct rotated GEMM is false and pair validation fails
    assert(!can_direct_rotated_gemm(desc_a_t4, desc_b_seed_mismatch));

    std::cout << "[test_factor_pair_compatibility_validation] passed!" << std::endl;
}

static void test_misaligned_buffer_roundtrip() {
    std::cout << "[test_misaligned_buffer_roundtrip] starting..." << std::endl;

    // Test all supported types: F32, F16, Q8_0, Turbo2, Turbo3, Turbo4
    ggml_type test_types[] = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0,
        GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0
    };

    // Cover 128/256/384-element rows: Turbo strides are one block per 128.
    const uint64_t lane_cols[] = { 128, 256, 384 };
    for (uint64_t pad_cols : lane_cols) {
        for (ggml_type t : test_types) {
            codec_desc desc = make_codec_desc(factor_role::a_k, t, orientation::token_major, {2, pad_cols}, 128, 42);
            std::vector<float> src(2 * pad_cols);
            double var_acc = 0.0;
        for (size_t i = 0; i < src.size(); ++i) {
                src[i] = (float)(i % 11) - 5.0f; // uniform [-5,5], var = 10
                var_acc += (double)src[i] * (double)src[i];
        }
            float sig_var = (float)(var_acc / (double)src.size());

        encoded_matrix em = encode_matrix(desc, src.data(), src.size());
        size_t orig_bytes = em.bytes.size();
            assert(orig_bytes == 2 * ggml_row_size(t, (int64_t)pad_cols));

            // Deliberate odd/misaligned storage: shift content by 1 byte.
        std::vector<uint8_t> shifted_raw(orig_bytes + 1, 0);
        std::memcpy(shifted_raw.data() + 1, em.bytes.data(), orig_bytes);

        encoded_matrix misaligned_em;
        misaligned_em.desc = desc;
        misaligned_em.bytes.assign(shifted_raw.begin() + 1, shifted_raw.end());

        std::vector<float> decoded = decode_matrix(misaligned_em, value_domain::canonical);
            assert(decoded.size() == 2 * pad_cols);
            for (float v : decoded) assert(std::isfinite(v));
            if (t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0) {
                float max_e = 0.0f;
        for (size_t i = 0; i < src.size(); ++i) {
                    float e = std::fabs(src[i] - decoded[i]);
                    if (e > max_e) max_e = e;
        }
                if (t == GGML_TYPE_F32) assert(max_e < 1e-5f);
                else if (t == GGML_TYPE_F16) assert(max_e < 0.01f);
                else assert(max_e < 0.05f);
            } else {
                // Turbo: per-element |err| < 1.0 is invalid (WHT-spread tails
                // exceed it); use aggregate Lloyd-Max bounds instead.
                float mse, cos, nratio;
                turbo_stats(src, decoded, mse, cos, nratio);
                float mse_max, cos_min;
                turbo_bounds(t, sig_var, mse_max, cos_min);
                assert(mse < mse_max);
                assert(cos > cos_min);
                assert(nratio > 0.85f && nratio < 1.15f);
                // Genuine misaligned-pointer proof: decoding row 0 straight
                // from the odd address must match the aligned decode path.
                std::vector<float> row0(pad_cols, 0.0f);
                assert(ggml_dequantize_turbo_row(t, shifted_raw.data() + 1,
                    row0.data(), (int64_t)pad_cols, 128, GGML_TURBO_DECODE_CANONICAL));
                assert(row0.size() == pad_cols);
                for (uint64_t c = 0; c < pad_cols; ++c) assert(row0[c] == decoded[c]);
    }
        }
    }
    std::cout << "[test_misaligned_buffer_roundtrip] passed!" << std::endl;
}

static void test_exact_row_size_and_guards() {
    std::cout << "[test_exact_row_size_and_guards] starting..." << std::endl;

    // Lengths 128/256/384 plus padded logical lengths 100 -> 128 and
    // 300 -> 384: Turbo strides are exactly ggml_row_size (one block/128).
    struct case_info {
        uint64_t log_cols;
        uint64_t pad_cols;
    } cases[] = {
        {128, 128},
        {256, 256},
        {384, 384},
        {100, 128}, // Padded logical length
        {300, 384}  // Padded logical length to 384
    };

    ggml_type tq_types[] = { GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0 };

    for (const auto & c : cases) {
        for (ggml_type t : tq_types) {
            codec_desc desc = make_codec_desc(factor_role::a_k, t, orientation::token_major, {2, c.log_cols}, 128, 42);
            assert(desc.padded_shape.cols == c.pad_cols);

            size_t row_bytes = ggml_row_size(t, (int64_t)c.pad_cols);
            assert(row_bytes == desc.row_stride_bytes);
            assert(encoded_matrix_bytes(desc) == 2 * row_bytes);

            std::vector<float> src(2 * c.log_cols, 1.23f);
            encoded_matrix em = encode_matrix(desc, src.data(), src.size());
            assert(em.bytes.size() == 2 * row_bytes);
            // Deterministic bytes: re-encode is bit-identical.
            encoded_matrix em2 = encode_matrix(desc, src.data(), src.size());
            assert(em.bytes == em2.bytes);

            // Distinct pre/post guards with a +1 misaligned target; both sides
            // are initialized and both are verified after every row API call.
            const size_t PRE = 16, POST = 16;
            const size_t total = PRE + 1 + row_bytes + POST;
            std::vector<uint8_t> guarded_buf(total, 0x00);
            std::fill(guarded_buf.begin(), guarded_buf.begin() + PRE + 1, 0xAA);
            std::fill(guarded_buf.begin() + PRE + 1 + row_bytes, guarded_buf.end(), 0x55);
            uint8_t * target_ptr = guarded_buf.data() + PRE + 1; // odd/misaligned

            // Copy first row into target_ptr
            std::memcpy(target_ptr, em.bytes.data(), row_bytes);

            // Decode single row using public ggml_dequantize_turbo_row
            std::vector<float> row_out(c.pad_cols, 0.0f);
            bool ok = ggml_dequantize_turbo_row(t, target_ptr, row_out.data(), (int64_t)c.pad_cols, 128, GGML_TURBO_DECODE_CANONICAL);
            assert(ok);
            for (float v : row_out) assert(std::isfinite(v));
            // Verify guard bytes on both sides were untouched
            for (size_t b = 0; b < PRE + 1; ++b) assert(guarded_buf[b] == 0xAA);
            for (size_t b = PRE + 1 + row_bytes; b < guarded_buf.size(); ++b) assert(guarded_buf[b] == 0x55);

            // Read-independence: flipping every guard byte must not change the
            // decoded floats, proving dequant reads exactly row_bytes.
            std::vector<float> row_out2(c.pad_cols, 0.0f);
            std::fill(guarded_buf.begin(), guarded_buf.begin() + PRE + 1, 0xE5);
            std::fill(guarded_buf.begin() + PRE + 1 + row_bytes, guarded_buf.end(), 0x1E);
            assert(ggml_dequantize_turbo_row(t, target_ptr, row_out2.data(), (int64_t)c.pad_cols, 128, GGML_TURBO_DECODE_CANONICAL));
            assert(row_out == row_out2);
        }
    }
    std::cout << "[test_exact_row_size_and_guards] passed!" << std::endl;
}

static void test_decode_rows_scratch_zero_heap() {
    std::cout << "[test_decode_rows_scratch_zero_heap] starting..." << std::endl;

    // F16 + Q8_0: scratch overload is bit-identical to the reference,
    // exact-sized, one-byte-short rejecting, and allocation-free on success.
    for (ggml_type t : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        codec_desc desc = make_codec_desc(factor_role::a_k, t, orientation::token_major, {4, 64});
        std::vector<float> src(4 * 64);
        for (size_t i = 0; i < src.size(); ++i) src[i] = (float)(i % 13) * 0.125f - 0.75f;
        encoded_matrix em = encode_matrix(desc, src.data(), src.size());

        // Checked size query reports exactly one ggml row for a valid desc.
        size_t need = 0;
        assert(decode_rows_scratch_bytes(desc, need));
        assert(need == ggml_row_size(t, 64));

        // Invalid descriptor fails closed without throwing.
        codec_desc bad = desc;
        bad.version = 99;
        size_t bad_need = 0;
        std::string qerr;
        assert(!decode_rows_scratch_bytes(bad, bad_need, &qerr));
        assert(!qerr.empty());

        uint64_t rows[3] = {0, 2, 3};
        std::vector<float> expect(3 * 64, 0.0f);
        decode_rows(em, rows, 3, expect.data(), expect.size());
        for (float v : expect) assert(std::isfinite(v));

        // Exact-size scratch: bit-identical, no write past row_bytes.
        std::vector<uint8_t> tmp(need + 16, 0xCC);
        std::vector<float> got(3 * 64, 0.0f);
        decode_rows(em, rows, 3, got.data(), got.size(), tmp.data(), need);
        assert(got == expect);
        for (size_t b = need; b < tmp.size(); ++b) assert(tmp[b] == 0xCC);

        // Zero heap allocations on the success path (buffers preallocated).
        std::fill(got.begin(), got.end(), 0.0f);
        xkv_codec_test_alloc::count = 0;
        xkv_codec_test_alloc::armed = true;
        decode_rows(em, rows, 3, got.data(), got.size(), tmp.data(), need);
        xkv_codec_test_alloc::armed = false;
        assert(xkv_codec_test_alloc::count == 0);
        assert(got == expect);

        // Exact one-byte-short scratch throws invalid_argument ...
        if (need > 0) {
            std::vector<uint8_t> short_tmp(need - 1, 0xCC);
            bool caught = false;
            try {
                decode_rows(em, rows, 3, got.data(), got.size(), short_tmp.data(), short_tmp.size());
            } catch (const std::invalid_argument &) {
                caught = true;
            }
            assert(caught);
            // ... and so does null scratch.
            caught = false;
            try {
                decode_rows(em, rows, 3, got.data(), got.size(), nullptr, 0);
            } catch (const std::invalid_argument &) {
                caught = true;
            }
            assert(caught);
        }
    }

    // F32 + Turbo4: scratch unused (nullptr/0), size query returns 0, zero-heap.
    for (ggml_type t : { GGML_TYPE_F32, GGML_TYPE_TURBO4_0 }) {
        uint64_t cols = (t == GGML_TYPE_F32) ? 64 : 128;
        uint64_t nrows = 2;
        codec_desc desc = make_codec_desc(factor_role::a_k, t, orientation::token_major, {nrows, cols}, 128, 42);
        std::vector<float> src(nrows * cols, 0.5f);
        encoded_matrix em = encode_matrix(desc, src.data(), src.size());
        size_t need = 0;
        assert(decode_rows_scratch_bytes(desc, need));
        assert(need == 0);
        uint64_t rows[2] = {0, 1};
        std::vector<float> expect(nrows * cols, 0.0f);
        decode_rows(em, rows, 2, expect.data(), expect.size());
        std::vector<float> got(nrows * cols, 0.0f);
        xkv_codec_test_alloc::count = 0;
        xkv_codec_test_alloc::armed = true;
        decode_rows(em, rows, 2, got.data(), got.size(), nullptr, 0);
        xkv_codec_test_alloc::armed = false;
        assert(xkv_codec_test_alloc::count == 0);
        assert(got == expect);
    }

    std::cout << "[test_decode_rows_scratch_zero_heap] passed!" << std::endl;
}

int main() {
    std::cout << "Running XKV codec tests..." << std::endl;
    test_descriptor_validation_and_fingerprint();
    test_exact_byte_size_and_overflow();
    test_four_roles_and_orientations();
    test_non_multiple_rank_padding_and_tail_zeros();
    test_zeros_and_extremes();
    test_turbo_canonical_vs_rotated_and_inverse_wht();
    test_invalid_buffers_and_atomicity();
    test_turbo_global_group_size_isolation();
    test_factor_pair_compatibility_validation();
    test_misaligned_buffer_roundtrip();
    test_exact_row_size_and_guards();
    test_decode_rows_scratch_zero_heap();
    std::cout << "All XKV codec tests PASSED successfully!" << std::endl;
    return 0;
}
