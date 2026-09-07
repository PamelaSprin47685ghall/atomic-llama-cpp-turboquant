// Focused regression: four factor streams + landmark estimate/encode/state equality.
// For each of A_K/B_K/A_V/B_V/landmark: encoded_matrix_bytes(desc) must equal
// encode_matrix(...).bytes.size(), serialize/deserialize round-trips byte-identically,
// and actual bytes must NOT equal a naive bits-per-element estimate (padding and
// per-block scales are observable). Fails a plausible bug where an estimator drops
// row padding or scale overhead.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-codec.h"
#include "ggml.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace llama_xkv;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

static std::vector<float> deterministic_data(uint64_t n, uint64_t seed) {
    std::vector<float> v((size_t) n);
    uint64_t s = seed ? seed : 0x243F6A8885A308D3ULL;
    for (size_t i = 0; i < v.size(); ++i) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v[i] = ((float) (s & 0x7FFFFFFF) / (float) 0x7FFFFFFF) * 2.0f - 1.0f;
    }
    return v;
}

struct stream_case {
    const char * name;
    factor_role role;
    ggml_type type;
    orientation orient;
    uint64_t rows;
    uint64_t cols;
    uint32_t group_size;
};

static void check_stream(const stream_case & c) {
    codec_desc desc = make_codec_desc(c.role, c.type, c.orient, {c.rows, c.cols}, c.group_size, 777);
    CHECK(desc.validate());
    CHECK(desc.role == c.role);
    CHECK(desc.type == c.type);
    CHECK(desc.orient == c.orient);

    // Estimator agrees with ggml row-size ground truth (same layout implementation).
    const int64_t expect_stride = ggml_row_size(c.type, (int64_t) desc.padded_shape.cols);
    CHECK((int64_t) desc.row_stride_bytes == expect_stride);

    const uint64_t estimate = encoded_matrix_bytes(desc);
    std::vector<float> src = deterministic_data(c.rows * c.cols, 1000 + (uint64_t) c.role);
    encoded_matrix em = encode_matrix(desc, src.data(), src.size());
    CHECK(em.bytes.size() == (size_t) estimate);

    // Deterministic encode: same input reproduces byte-identical state.
    encoded_matrix em2 = encode_matrix(desc, src.data(), src.size());
    CHECK(em2.bytes == em.bytes);

    // State form: exact versioned descriptor bytes round-trip.
    std::vector<uint8_t> desc_bytes = serialize_desc(desc);
    codec_desc back;
    std::string err;
    CHECK(deserialize_desc(desc_bytes.data(), desc_bytes.size(), back, &err));
    CHECK(back == desc);

    // Naive bits-per-element estimate must NOT match actual bytes whenever
    // padding or scale overhead exists (rank 70 pads 70->128; Q8 carries fp16
    // scale per 32 elems; Turbo blocks carry norms). This is the assertion
    // that fails a "flat element count * bits/8" estimator bug.
    const uint64_t flat_elems = c.rows * c.cols;
    double nominal_bpe = 0.0;
    if (c.type == GGML_TYPE_TURBO4_0) nominal_bpe = 0.5;
    else if (c.type == GGML_TYPE_TURBO2_0) nominal_bpe = 0.25;
    else if (c.type == GGML_TYPE_Q8_0) nominal_bpe = 1.0;
    if (nominal_bpe > 0.0) {
        const uint64_t naive = (uint64_t) ((double) flat_elems * nominal_bpe);
        CHECK(estimate != naive);
    }
}

int main() {
    std::cout << "=== test-xkv-factor-encoding-size ===" << std::endl;

    // Rank-padding case: rV=70 is not a multiple of the 128-wide Turbo group,
    // Realistic low-rank fixtures: rK=96 and rV=70 are neither multiples of the
    // 128-wide Turbo group, so every A row (N x r) and every B^T feature row
    // (D x r) pads to 128. A_K/B_K share rK; A_V/B_V share rV.
    const stream_case cases[] = {
        {"A_K Turbo4 token-major",   factor_role::a_k,      GGML_TYPE_TURBO4_0, orientation::token_major,                16, 96, 128},
        {"B_K Turbo4 B^T",           factor_role::b_k,      GGML_TYPE_TURBO4_0, orientation::feature_major_transposed,  256, 96, 128},
        {"A_V Turbo2 token-major",   factor_role::a_v,      GGML_TYPE_TURBO2_0, orientation::token_major,                16, 70,  128},
        {"B_V Turbo2 B^T",           factor_role::b_v,      GGML_TYPE_TURBO2_0, orientation::feature_major_transposed,  256, 70,  128},
        {"landmark Q8_0",            factor_role::landmark, GGML_TYPE_Q8_0,    orientation::token_major,                 4, 256, 0},
    };
    for (const auto & c : cases) {
        std::cout << "[case] " << c.name << std::endl;
        check_stream(c);
    }

    // Explicit layout-drift guards on the exact block sizes in play.
    CHECK(ggml_type_size(GGML_TYPE_TURBO4_0) == 68);
    CHECK(ggml_type_size(GGML_TYPE_TURBO2_0) == 34);
    CHECK(ggml_blck_size(GGML_TYPE_TURBO4_0) == 128);
    CHECK(ggml_blck_size(GGML_TYPE_TURBO2_0) == 128);
    CHECK(ggml_blck_size(GGML_TYPE_Q8_0) == 32);

    // Padded-shape spot checks: 96- and 70-wide Turbo rows pad to exactly one 128-group.
    codec_desc ak = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0,
                                    orientation::token_major, {16, 96}, 128, 777);
    CHECK(ak.padded_shape.cols == 128);
    CHECK(ak.row_stride_bytes == (uint64_t) ggml_type_size(GGML_TYPE_TURBO4_0));
    CHECK(encoded_matrix_bytes(ak) == 16u * (uint64_t) ggml_type_size(GGML_TYPE_TURBO4_0));
    codec_desc bk = make_codec_desc(factor_role::b_k, GGML_TYPE_TURBO4_0,
                                    orientation::feature_major_transposed, {256, 96}, 128, 777);
    CHECK(bk.padded_shape.cols == 128);
    CHECK(encoded_matrix_bytes(bk) == 256u * (uint64_t) ggml_type_size(GGML_TYPE_TURBO4_0));
    codec_desc av = make_codec_desc(factor_role::a_v, GGML_TYPE_TURBO2_0,
                                    orientation::token_major, {16, 70}, 128, 777);
    CHECK(av.padded_shape.cols == 128);
    CHECK(av.row_stride_bytes == (uint64_t) ggml_type_size(GGML_TYPE_TURBO2_0));
    codec_desc bv = make_codec_desc(factor_role::b_v, GGML_TYPE_TURBO2_0,
                                    orientation::feature_major_transposed, {256, 70}, 128, 777);
    CHECK(bv.padded_shape.cols == 128);
    CHECK(encoded_matrix_bytes(bv) == 256u * (uint64_t) ggml_type_size(GGML_TYPE_TURBO2_0));

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED: %d checks\n", g_failures);
        return 1;
    }
    std::cout << "=== ALL FACTOR-ENCODING-SIZE CHECKS PASSED ===" << std::endl;
    return 0;
}
