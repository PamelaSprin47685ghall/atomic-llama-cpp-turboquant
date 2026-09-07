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
#include "llama-xkv-factor.h"
#include "ggml.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
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

    // ---- One-row / empty boundaries (observable, not wiring) ----
    {
        // One-row A_K: exactly one stride of payload, estimate == encode.
        codec_desc one = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0,
                                           orientation::token_major, {1, 96}, 128, 777);
        CHECK(one.padded_shape.cols == 128);
        CHECK(encoded_matrix_bytes(one) == (uint64_t) ggml_type_size(GGML_TYPE_TURBO4_0));
        std::vector<float> src1(1u * 96u, 0.25f);
        encoded_matrix em1 = encode_matrix(one, src1.data(), src1.size());
        CHECK(em1.bytes.size() == (size_t) encoded_matrix_bytes(one));
        // One-row landmark: single Q8 row still carries per-block scales.
        codec_desc lm1 = make_codec_desc(factor_role::landmark, GGML_TYPE_Q8_0,
                                           orientation::token_major, {1, 256}, 0, 777);
        CHECK(encoded_matrix_bytes(lm1) == (uint64_t) ggml_row_size(GGML_TYPE_Q8_0, 256));
        std::vector<float> lsrc1(256, 0.1f);
        encoded_matrix lme1 = encode_matrix(lm1, lsrc1.data(), lsrc1.size());
        CHECK(lme1.bytes.size() == (size_t) encoded_matrix_bytes(lm1));
        // Empty shape: zero payload bytes, empty code stream.
        codec_desc empty = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0,
                                             orientation::token_major, {0, 96}, 128, 777);
        CHECK(encoded_matrix_bytes(empty) == 0u);
        encoded_matrix eme = encode_matrix(empty, nullptr, 0);
        CHECK(eme.bytes.empty());
    }

    // ---- Segment/chunk/page non-divisibility: row counts 19/33/65 ----
    {
        const uint64_t odd_rows[] = {19, 33, 65};
        for (uint64_t nr : odd_rows) {
            codec_desc d = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0,
                                             orientation::token_major, {nr, 96}, 128, 777);
            CHECK(encoded_matrix_bytes(d) == nr * (uint64_t) ggml_type_size(GGML_TYPE_TURBO4_0));
            std::vector<float> src((size_t)(nr * 96), 0.3f);
            encoded_matrix em = encode_matrix(d, src.data(), src.size());
            CHECK(em.bytes.size() == (size_t) encoded_matrix_bytes(d));
            // Landmark rows need not divide chunk (8) or Q8 block (32).
            codec_desc lm = make_codec_desc(factor_role::landmark, GGML_TYPE_Q8_0,
                                              orientation::token_major, {nr, 64}, 0, 777);
            CHECK(encoded_matrix_bytes(lm) == nr * (uint64_t) ggml_row_size(GGML_TYPE_Q8_0, 64));
            std::vector<float> lsrc((size_t)(nr * 64), 0.1f);
            encoded_matrix lme = encode_matrix(lm, lsrc.data(), lsrc.size());
            CHECK(lme.bytes.size() == (size_t) encoded_matrix_bytes(lm));
        }
        // Feature-major B with non-divisible feature rows (70 features x rank 96).
        codec_desc b70 = make_codec_desc(factor_role::b_k, GGML_TYPE_TURBO4_0,
                                           orientation::feature_major_transposed, {70, 96}, 128, 777);
        CHECK(b70.padded_shape.cols == 128);
        CHECK(encoded_matrix_bytes(b70) == 70u * (uint64_t) ggml_type_size(GGML_TYPE_TURBO4_0));
    }

    // ---- Head 128 vs 256/64 partial-IMRoPE widths (B feature rows = head_dim) ----
    {
        // head_dim=128 full: B^T has 128 feature rows.
        codec_desc b128 = make_codec_desc(factor_role::b_k, GGML_TYPE_TURBO4_0,
                                            orientation::feature_major_transposed, {128, 96}, 128, 777);
        CHECK(encoded_matrix_bytes(b128) == 128u * (uint64_t) ggml_type_size(GGML_TYPE_TURBO4_0));
        // head_dim=256 partial (rotary 64 passes through tail): B^T has 256 feature rows.
        codec_desc b256 = make_codec_desc(factor_role::b_k, GGML_TYPE_TURBO4_0,
                                            orientation::feature_major_transposed, {256, 96}, 128, 777);
        CHECK(encoded_matrix_bytes(b256) == 256u * (uint64_t) ggml_type_size(GGML_TYPE_TURBO4_0));
        CHECK(encoded_matrix_bytes(b256) == 2u * encoded_matrix_bytes(b128));
        std::vector<float> src256((size_t)(256 * 96), 0.2f);
        encoded_matrix em256 = encode_matrix(b256, src256.data(), src256.size());
        CHECK(em256.bytes.size() == (size_t) encoded_matrix_bytes(b256));
    }

    // ---- Remaining codec types: Turbo3 / F16 / F32 ----
    {
        stream_case extra[] = {
            {"A_K Turbo3 token-major", factor_role::a_k, GGML_TYPE_TURBO3_0, orientation::token_major, 16, 96, 128},
            {"B_K F16 B^T",            factor_role::b_k, GGML_TYPE_F16,     orientation::feature_major_transposed, 64, 32, 0},
            {"A_V F32 token-major",    factor_role::a_v, GGML_TYPE_F32,     orientation::token_major, 5, 7, 0},
        };
        for (const auto & c : extra) {
            std::cout << "[case] " << c.name << std::endl;
            check_stream(c);
        }
        // F16 exact: no block padding, bytes == rows*cols*2.
        codec_desc f16 = make_codec_desc(factor_role::b_k, GGML_TYPE_F16,
                                           orientation::feature_major_transposed, {64, 32}, 0, 777);
        CHECK(f16.padded_shape.cols == 32);
        CHECK(encoded_matrix_bytes(f16) == 64u * 32u * 2u);
    }

    // ---- State size: fixed 100-byte descriptor + payload; truncation/trailing fail ----
    {
        codec_desc d = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0,
                                         orientation::token_major, {16, 96}, 128, 777);
        std::vector<uint8_t> db = serialize_desc(d);
        CHECK(db.size() == 100u); // versioned descriptor state is fixed-size
        std::vector<float> src((size_t)(16 * 96), 0.5f);
        encoded_matrix em = encode_matrix(d, src.data(), src.size());
        const uint64_t full_state = (uint64_t) db.size() + encoded_matrix_bytes(d);
        CHECK(full_state == (uint64_t) db.size() + em.bytes.size()); // payload/metadata separation
        // Truncated and trailing descriptor bytes must be rejected, not tolerated.
        codec_desc back;
        std::string err;
        CHECK(!deserialize_desc(db.data(), db.size() - 1, back, &err));
        CHECK(!err.empty());
        err.clear();
        std::vector<uint8_t> trailing = db;
        trailing.push_back(0);
        CHECK(!deserialize_desc(trailing.data(), trailing.size(), back, &err));
        CHECK(!err.empty());
    }

    // ---- Four-factor + landmark bundle total with padding accounted ----
    {
        const uint64_t n = 19; // non-divisible row count exercises every stream at once
        codec_desc dak = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, orientation::token_major, {n, 96}, 128, 777);
        codec_desc dbk = make_codec_desc(factor_role::b_k, GGML_TYPE_TURBO4_0, orientation::feature_major_transposed, {256, 96}, 128, 777);
        codec_desc dav = make_codec_desc(factor_role::a_v, GGML_TYPE_TURBO2_0, orientation::token_major, {n, 70}, 128, 777);
        codec_desc dbv = make_codec_desc(factor_role::b_v, GGML_TYPE_TURBO2_0, orientation::feature_major_transposed, {256, 70}, 128, 777);
        codec_desc dlm = make_codec_desc(factor_role::landmark, GGML_TYPE_Q8_0, orientation::token_major, {4, 256}, 0, 777);
        const uint64_t bundle = encoded_matrix_bytes(dak) + encoded_matrix_bytes(dbk) +
                                encoded_matrix_bytes(dav) + encoded_matrix_bytes(dbv) +
                                encoded_matrix_bytes(dlm);
        const uint64_t expect = n * (uint64_t) ggml_type_size(GGML_TYPE_TURBO4_0) +
                                256u * (uint64_t) ggml_type_size(GGML_TYPE_TURBO4_0) +
                                n * (uint64_t) ggml_type_size(GGML_TYPE_TURBO2_0) +
                                256u * (uint64_t) ggml_type_size(GGML_TYPE_TURBO2_0) +
                                4u * (uint64_t) ggml_row_size(GGML_TYPE_Q8_0, 256);
        CHECK(bundle == expect);
        // Each stream encodes to exactly its estimate: no hidden per-stream overhead.
        std::vector<float> sak((size_t)(n * 96), 0.1f), sbk((size_t)(256 * 96), 0.2f);
        std::vector<float> sav((size_t)(n * 70), 0.3f), sbv((size_t)(256 * 70), 0.4f);
        std::vector<float> slm((size_t)(4 * 256), 0.05f);
        CHECK(encode_matrix(dak, sak.data(), sak.size()).bytes.size() == (size_t) encoded_matrix_bytes(dak));
        CHECK(encode_matrix(dbk, sbk.data(), sbk.size()).bytes.size() == (size_t) encoded_matrix_bytes(dbk));
        CHECK(encode_matrix(dav, sav.data(), sav.size()).bytes.size() == (size_t) encoded_matrix_bytes(dav));
        CHECK(encode_matrix(dbv, sbv.data(), sbv.size()).bytes.size() == (size_t) encoded_matrix_bytes(dbv));
        CHECK(encode_matrix(dlm, slm.data(), slm.size()).bytes.size() == (size_t) encoded_matrix_bytes(dlm));
    }

    // ---- Group 1/2/4 tails + alias dedup at the layer-group map ----
    {
        const std::vector<uint32_t> owning7 = {0, 1, 2, 3, 4, 5, 6};
        layer_group_map m4 = build_layer_group_map(owning7, 4);
        CHECK(m4.groups.size() == 2); // [0..3] + tail [4..6]
        CHECK(m4.groups[0].owning_layers.size() == 4);
        CHECK(m4.groups[1].owning_layers.size() == 3);
        layer_group_map m2 = build_layer_group_map(owning7, 2);
        CHECK(m2.groups.size() == 4); // 2+2+2+tail 1
        CHECK(m2.groups.back().owning_layers.size() == 1);
        layer_group_map m1 = build_layer_group_map(owning7, 1);
        CHECK(m1.groups.size() == 7);
        // Aliased owning layers collapse: duplicate layer counted once.
        layer_group_map md = build_layer_group_map({0, 1, 1, 2}, 4);
        CHECK(md.unique_owning_layers.size() == 3);
        CHECK(md.groups.size() == 1);
        CHECK(md.groups[0].owning_layers.size() == 3);
    }

    // ---- Overflow and invalid-descriptor fail-closed ----
    {
        codec_desc d = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0,
                                         orientation::token_major, {16, 96}, 128, 777);
        codec_desc huge = d;
        huge.padded_shape.rows = UINT64_MAX; // stride 68 cannot fit: must throw, not wrap
        bool threw_overflow = false;
        try {
            (void) encoded_matrix_bytes(huge);
        } catch (const std::overflow_error &) {
            threw_overflow = true;
        }
        CHECK(threw_overflow);
        codec_desc bad = d;
        bad.version = 1; // unsupported descriptor version
        bool threw_invalid = false;
        try {
            (void) encoded_matrix_bytes(bad);
        } catch (const std::invalid_argument &) {
            threw_invalid = true;
        }
        CHECK(threw_invalid);
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED: %d checks\n", g_failures);
        return 1;
    }
    std::cout << "=== ALL FACTOR-ENCODING-SIZE CHECKS PASSED ===" << std::endl;
    return 0;
}
