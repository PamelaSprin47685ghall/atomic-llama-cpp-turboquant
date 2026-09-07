// Focused regression: quantized pack copies survivor A code bytes verbatim —
// no re-encode, B handles pointer-identical, descriptors unchanged.
// Observable behavior: per-row A_K/A_V bytes of survivors are memcmp-identical
// after remove + pack_segment, and B shared_ptrs compare equal. A plausible
// "re-quantize survivors on pack" bug would change bytes (re-quantization is
// not bit-stable across layouts) and fail the identity checks.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-cache.h"
#include "llama-xkv-codec.h"
#include "llama-cparams.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
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

static llama_cparams test_cparams() {
    llama_cparams c = {};
    c.xkv_mode = LLAMA_XKV_MODE_SHADOW;
    c.xkv_storage_profile = LLAMA_XKV_STORAGE_PROFILE_REFERENCE;
    c.xkv_group_size = 4;
    c.xkv_rank_k = 16;
    c.xkv_rank_v = 16;
    c.xkv_segment_tokens = 64;
    c.xkv_chunk_tokens = 8;
    c.xkv_workspace_mib = 16;
    c.xkv_decode_cache_mib = 8;
    c.xkv_min_saving = 0.10;
    return c;
}

int main() {
    std::cout << "=== test-xkv-quantized-pack ===" << std::endl;

    llama_xkv_cache_store store(test_cparams());

    const uint32_t n_rows = 6;
    const uint32_t rank = 128; // Turbo-aligned rows: stride is one 68-byte block
    const uint32_t dim = 128;

    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = {0, 1, 2, 3};
    g.rank_k = rank;
    g.rank_v = rank;
    g.total_dim_k = dim;
    g.total_dim_v = dim;
    g.layer_feature_offsets_k = {0, 32, 64, 96};
    g.layer_feature_dims_k = {32, 32, 32, 32};
    g.layer_feature_offsets_v = {0, 32, 64, 96};
    g.layer_feature_dims_v = {32, 32, 32, 32};

    // Quantized streams: A_K Turbo4, A_V Turbo2, B pair Turbo4/Q8 mix.
    // Every A row holds distinct valid floats (row-scaled smooth pattern), so
    // survivor rows are distinguishable through valid codes only — no byte
    // tampering that could bypass integrity checks.
    {
        std::vector<float> srck((size_t) n_rows * rank);
        std::vector<float> srcv((size_t) n_rows * rank);
        for (uint32_t r = 0; r < n_rows; ++r) {
            for (uint32_t c = 0; c < rank; ++c) {
                const float wobble = (float) (((c * 13u + r * 7u) % 29u)) * 0.002f;
                srck[(size_t) r * rank + c] = 0.05f * (float) (r + 1) + wobble;
                srcv[(size_t) r * rank + c] = -0.04f * (float) (r + 1) - wobble;
            }
        }
        codec_desc dk = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0,
                                        orientation::token_major, {n_rows, rank}, 128, 501);
        g.a_k = encode_matrix(dk, srck.data(), srck.size());

        codec_desc dv = make_codec_desc(factor_role::a_v, GGML_TYPE_TURBO2_0,
                                        orientation::token_major, {n_rows, rank}, 128, 502);
        g.a_v = encode_matrix(dv, srcv.data(), srcv.size());

        codec_desc bk = make_codec_desc(factor_role::b_k, GGML_TYPE_TURBO4_0,
                                        orientation::feature_major_transposed, {dim, rank}, 128, 503);
        std::vector<float> srcbk((size_t) dim * rank, 0.3f);
        g.set_b_k(encode_matrix(bk, srcbk.data(), srcbk.size()));

        codec_desc bv = make_codec_desc(factor_role::b_v, GGML_TYPE_Q8_0,
                                        orientation::feature_major_transposed, {dim, rank}, 0, 504);
        std::vector<float> srcbv((size_t) dim * rank, 0.05f);
        g.set_b_v(encode_matrix(bv, srcbv.data(), srcbv.size()));
    }
    g.refresh_descriptor_fingerprint();
    g.update_byte_counters();

    const std::vector<uint64_t> pids = {10, 20, 30, 40, 50, 60};
    const std::vector<uint64_t> gens = {1, 2, 3, 4, 5, 6};
    for (size_t i = 0; i < pids.size(); ++i) {
        CHECK(store.register_hot_payload(pids[i], (uint32_t) i, gens[i], xkv_state::hot_committed));
    }
    CHECK(store.mark_seal_candidates(pids, gens, nullptr));

    auto seg = store.create_candidate_segment(LLAMA_XKV_STORAGE_PROFILE_REFERENCE,
                                              LLAMA_XKV_SOURCE_DECODED_HOT, {g});
    const uint64_t seg_id = seg->segment_id;

    // Snapshot the valid encoded survivor rows. Rows are byte-distinct by
    // construction (row-scaled inputs quantize to distinct norms/codes), so a
    // re-encode (which would normalize/rounding-differ) cannot accidentally match.
    const size_t stride_k = seg->groups[0].a_k.desc.row_stride_bytes;
    const size_t stride_v = seg->groups[0].a_v.desc.row_stride_bytes;
    CHECK(seg->groups[0].a_k.bytes.size() == (size_t) n_rows * stride_k);
    CHECK(seg->groups[0].a_v.bytes.size() == (size_t) n_rows * stride_v);
    CHECK(std::memcmp(seg->groups[0].a_k.bytes.data() + 0 * stride_k,
                        seg->groups[0].a_k.bytes.data() + 1 * stride_k, stride_k) != 0);
    CHECK(std::memcmp(seg->groups[0].a_v.bytes.data() + 0 * stride_v,
                        seg->groups[0].a_v.bytes.data() + 1 * stride_v, stride_v) != 0);
    std::vector<uint8_t> row0_k(seg->groups[0].a_k.bytes.begin(),
                                seg->groups[0].a_k.bytes.begin() + stride_k);
    std::vector<uint8_t> row2_k(seg->groups[0].a_k.bytes.begin() + 2 * stride_k,
                                seg->groups[0].a_k.bytes.begin() + 3 * stride_k);
    std::vector<uint8_t> row5_k(seg->groups[0].a_k.bytes.begin() + 5 * stride_k,
                                seg->groups[0].a_k.bytes.begin() + 6 * stride_k);
    std::vector<uint8_t> row0_v(seg->groups[0].a_v.bytes.begin(),
                                seg->groups[0].a_v.bytes.begin() + stride_v);
    std::vector<uint8_t> row5_v(seg->groups[0].a_v.bytes.begin() + 5 * stride_v,
                                seg->groups[0].a_v.bytes.begin() + 6 * stride_v);

    std::shared_ptr<const encoded_matrix> orig_bk = seg->groups[0].b_k;
    std::shared_ptr<const encoded_matrix> orig_bv = seg->groups[0].b_v;
    const codec_desc orig_desc_ak = seg->groups[0].a_k.desc;
    const codec_desc orig_desc_av = seg->groups[0].a_v.desc;

    std::string err;
    CHECK(store.publish_candidate(seg, pids, gens, &err));

    // Survivors: rows 0, 2, 5 (payloads 10, 30, 60). Remove the rest.
    CHECK(store.remove_payload(20, &err));
    CHECK(store.remove_payload(40, &err));
    CHECK(store.remove_payload(50, &err));
    CHECK(store.pack_segment(seg_id, &err));

    auto packed = store.get_segment(seg_id);
    CHECK(packed != nullptr);
    CHECK(packed->n_rows == 3);
    CHECK(packed->n_live_rows == 3);
    CHECK((packed->row_payload_ids == std::vector<uint64_t>({10, 30, 60})));

    // B handles are pointer-identical across the pack (immutable shared).
    CHECK(packed->groups[0].b_k == orig_bk);
    CHECK(packed->groups[0].b_v == orig_bv);
    CHECK(packed->groups[0].b_k->bytes == orig_bk->bytes);

    // Survivor A bytes are verbatim copies — no re-encode.
    CHECK(packed->groups[0].a_k.desc == orig_desc_ak);
    CHECK(packed->groups[0].a_v.desc == orig_desc_av);
    CHECK(std::memcmp(packed->groups[0].a_k.bytes.data() + 0 * stride_k, row0_k.data(), stride_k) == 0);
    CHECK(std::memcmp(packed->groups[0].a_k.bytes.data() + 1 * stride_k, row2_k.data(), stride_k) == 0);
    CHECK(std::memcmp(packed->groups[0].a_k.bytes.data() + 2 * stride_k, row5_k.data(), stride_k) == 0);
    CHECK(std::memcmp(packed->groups[0].a_v.bytes.data() + 0 * stride_v, row0_v.data(), stride_v) == 0);
    CHECK(std::memcmp(packed->groups[0].a_v.bytes.data() + 2 * stride_v, row5_v.data(), stride_v) == 0);
    CHECK(packed->groups[0].a_k.bytes.size() == 3 * stride_k);
    CHECK(packed->groups[0].a_v.bytes.size() == 3 * stride_v);

    // Survivor codes remain valid decodable streams (finite canonical output).
    {
        std::vector<float> dec_k = decode_matrix(packed->groups[0].a_k);
        std::vector<float> dec_v = decode_matrix(packed->groups[0].a_v);
        CHECK(dec_k.size() == 3 * (size_t) rank);
        CHECK(dec_v.size() == 3 * (size_t) rank);
        for (float v : dec_k) {
            CHECK(std::isfinite(v));
        }
        for (float v : dec_v) {
            CHECK(std::isfinite(v));
        }
    }


    // Landmark survivor rule: a partial-chunk removal on a LANDMARKS segment
    // without a semantic rebuild callback must refuse with zero mutation.
    // Chunk summaries are position-sensitive, so the store never copies the
    // first-N chunks after arbitrary deletion; A/B/fingerprints stay intact.
    {
        const uint32_t lm_rows = 8; // == chunk_tokens: exactly one chunk
        xkv_factor_group_payload gl;
        gl.group_index = 1;
        gl.owning_layers = {0, 1, 2, 3};
        gl.rank_k = rank;
        gl.rank_v = rank;
        gl.total_dim_k = dim;
        gl.total_dim_v = dim;
        gl.layer_feature_offsets_k = {0, 32, 64, 96};
        gl.layer_feature_dims_k = {32, 32, 32, 32};
        gl.layer_feature_offsets_v = {0, 32, 64, 96};
        gl.layer_feature_dims_v = {32, 32, 32, 32};
        {
            std::vector<float> srck((size_t) lm_rows * rank, 0.11f);
            std::vector<float> srcv((size_t) lm_rows * rank, -0.07f);
            codec_desc dk = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0,
                                            orientation::token_major, {lm_rows, rank}, 128, 601);
            gl.a_k = encode_matrix(dk, srck.data(), srck.size());
            codec_desc dv = make_codec_desc(factor_role::a_v, GGML_TYPE_TURBO2_0,
                                            orientation::token_major, {lm_rows, rank}, 128, 602);
            gl.a_v = encode_matrix(dv, srcv.data(), srcv.size());
            codec_desc bk = make_codec_desc(factor_role::b_k, GGML_TYPE_TURBO4_0,
                                            orientation::feature_major_transposed, {dim, rank}, 128, 603);
            std::vector<float> srcbk((size_t) dim * rank, 0.3f);
            gl.set_b_k(encode_matrix(bk, srcbk.data(), srcbk.size()));
            codec_desc bv = make_codec_desc(factor_role::b_v, GGML_TYPE_Q8_0,
                                            orientation::feature_major_transposed, {dim, rank}, 0, 604);
            std::vector<float> srcbv((size_t) dim * rank, 0.05f);
            gl.set_b_v(encode_matrix(bv, srcbv.data(), srcbv.size()));
            codec_desc dlm = make_codec_desc(factor_role::landmark, GGML_TYPE_TURBO4_0,
                                             orientation::token_major, {1, dim}, 128, 605);
            std::vector<float> srclm((size_t) dim, 0.2f);
            gl.landmark = encode_matrix(dlm, srclm.data(), srclm.size());
        }
        xkv_landmark_chunk ch;
        ch.row_begin = 0;
        ch.row_count = lm_rows;
        ch.error_bound = 0.1f;
        ch.source_fingerprint = 0x00ABCDEF12345678ULL;
        gl.landmark_chunks = {ch};
        gl.landmark_table_fingerprint = compute_landmark_table_fingerprint(gl.landmark_chunks);
        gl.refresh_descriptor_fingerprint();
        gl.update_byte_counters();

        std::vector<uint64_t> lpids;
        std::vector<uint64_t> lgens;
        for (uint64_t i = 0; i < lm_rows; ++i) {
            lpids.push_back(101 + i);
            lgens.push_back(11 + i);
        }
        for (size_t i = 0; i < lpids.size(); ++i) {
            CHECK(store.register_hot_payload(lpids[i], (uint32_t) i, lgens[i], xkv_state::hot_committed));
        }
        CHECK(store.mark_seal_candidates(lpids, lgens, nullptr));
        auto seg2 = store.create_candidate_segment(LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS,
                                                   LLAMA_XKV_SOURCE_DECODED_HOT, {gl});
        const uint64_t seg2_id = seg2->segment_id;
        std::shared_ptr<const encoded_matrix> lm_bk = seg2->groups[0].b_k;
        const std::vector<uint8_t> lm_ak = seg2->groups[0].a_k.bytes;
        const std::vector<uint8_t> lm_bytes = seg2->groups[0].landmark.bytes;
        const auto lm_chunks = seg2->groups[0].landmark_chunks;
        const uint64_t lm_table = seg2->groups[0].landmark_table_fingerprint;
        const uint64_t ver2 = seg2->segment_version;
        CHECK(store.publish_candidate(seg2, lpids, lgens, &err));
        // Partial-chunk removal (row 1 of the single 8-row chunk) without a
        // rebuild callback refuses; pack without a callback refuses too.
        CHECK(!store.remove_payload(102, &err));
        CHECK(!store.pack_segment(seg2_id, &err));
        auto kept = store.get_segment(seg2_id);
        CHECK(kept != nullptr);
        CHECK(kept->segment_version == ver2);
        CHECK(kept->n_rows == lm_rows);
        CHECK((kept->row_payload_ids == lpids));
        CHECK(kept->groups[0].b_k == lm_bk);
        CHECK(kept->groups[0].a_k.bytes == lm_ak);
        CHECK(kept->groups[0].landmark.bytes == lm_bytes);
        CHECK(kept->groups[0].landmark_chunks == lm_chunks);
        CHECK(kept->groups[0].landmark_table_fingerprint == lm_table);
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED: %d checks\n", g_failures);
        return 1;
    }
    std::cout << "=== ALL QUANTIZED-PACK CHECKS PASSED ===" << std::endl;
    return 0;
}
