// Deterministic tests for the XKV code-stream state image (v2 envelope):
// deterministic encode, byte-for-byte roundtrip (incl. shared-B identity),
// multi-group / multi-version / dead-row images, fingerprint+provenance+epoch
// roundtrip, hot-only images, zero-length invalid fields, integer overflow,
// truncation, duplicate ids, bad descriptors, checksum failures, group-closure
// violations, pair incompatibility, budget overflow, fingerprint/provenance
// mismatch refusal, trailing bytes, version/magic rejection, recomputed
// accounting agreement, config/enum/high-water rules, landmark-profile rules,
// tombstone-record refusal, readback overrides, and destination-unchanged on
// failure (including allocation failure paths where expressible).
//
// Decode-path duplicate/shape checks delegate to validate_image (the same
// function decode_image runs after framing); those cases are exercised through
// validate_image/encode_image directly.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-state.h"
#include "llama-xkv-codec.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// PID high-water tests need the cells metadata owner (counter lives there).
#include "llama-kv-cells.h"

using namespace llama_xkv;

static void fix_stream_checksum(xkv_state_image & img, size_t idx) {
    uint64_t h = xkv_state_checksum(img.allocations[idx].desc_bytes.data(),
                                    img.allocations[idx].desc_bytes.size());
    const auto & by = img.allocations[idx].bytes;
    for (size_t i = 0; i < by.size(); ++i) {
        h ^= by[i];
        h *= 0x100000001b3ULL;
    }
    img.allocations[idx].checksum = h;
}

static xkv_state_fingerprints test_fps() {
    return {0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666, 0x7777, 0x8888};
}

static xkv_state_provenance test_prov() {
    xkv_state_provenance p;
    std::memset(p.model_sha256, 0xA1, 32);
    std::memset(p.tri_calibration_sha256, 0xB2, 32);
    std::memset(p.source_sha256, 0xC3, 32);
    return p;
}

static xkv_state_config test_config() {
    xkv_state_config c;
    c.profile                   = (uint32_t) LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
    c.source                    = (uint32_t) LLAMA_XKV_SOURCE_DECODED_HOT;
    c.group_size                = 4;
    c.rank_k                    = 4;
    c.rank_v                    = 4;
    c.segment_tokens            = 64;
    c.chunk_tokens              = 8;
    c.sr_budget                 = 0;
    c.mode                      = 2; // DENSE
    c.factor_a_k                = (uint32_t) GGML_TYPE_TURBO4_0;
    c.factor_b_k                = (uint32_t) GGML_TYPE_TURBO4_0;
    c.factor_a_v                = (uint32_t) GGML_TYPE_TURBO4_0;
    c.factor_b_v                = (uint32_t) GGML_TYPE_TURBO4_0;
    c.factor_balance            = (uint32_t) LLAMA_XKV_FACTOR_BALANCE_UPSTREAM;
    c.landmark_type             = (uint32_t) GGML_TYPE_F32;
    c.landmark_refine           = (uint32_t) LLAMA_XKV_LANDMARK_REFINE_NONE;
    c.landmark_refine_max_rows  = 0;
    c.workspace_mib             = 16;
    c.decode_cache_mib          = 8;
    c.store_mib                 = 1; // resolved persistent factor-store budget in MiB
    c.seed                      = 42;
    c.min_saving_ppm            = 100000;
    c.min_coverage_ppm          = 500000;
    c.factorizer                = (uint32_t) LLAMA_XKV_FACTORIZER_CPU_REFERENCE;
    c.min_saving_bytes          = 1;
    c.next_segment_id           = 8;
    c.next_alloc_id             = 12;
    c.next_seal_tx_nonce        = 1;
    return c;
}

static xkv_state_limits test_limits() {
    // Factor-store budget comfortably above the ~1KiB world below; counts exact.
    return xkv_state_limits::from_store_budgets(1 << 20, 1 << 20,
                                                /*max_segments=*/8, /*max_groups=*/8,
                                                /*max_payloads=*/64, /*max_allocations=*/64,
                                                /*max_owning=*/8, /*max_rows=*/64);
}

static xkv_snapshot_stamp test_stamp() {
    xkv_snapshot_stamp s;
    s.view.topology_epoch = 11;
    s.view.publish_epoch   = 22;
    s.view.layout_epoch    = 33;
    s.live_epoch           = 44;
    s.content_epoch        = 55;
    s.codec_epoch          = 66;
    s.binding_epoch        = 77;
    return s;
}

static xkv_accounting test_accounting() {
    xkv_accounting a;
    a.live_payload_bytes = 0; // recomputed by capture; placeholder
    a.allocated_bytes    = 2000;
    a.reserved_bytes     = 3000;
    a.workspace_budget_bytes = 0; // derived from workspace_mib by capture
    a.hot_bytes          = 400;
    a.factored_bytes     = 0; // recomputed by capture
    a.active_segments    = 0; // recomputed
    a.total_payloads     = 0; // recomputed
    a.unique_b_matrices  = 0; // recomputed
    a.shared_b_bytes     = 0; // recomputed
    return a;
}

static encoded_matrix make_stream(factor_role role, orientation orient, uint64_t rows, uint64_t cols,
                                  uint64_t seed) {
    codec_desc d = make_codec_desc(role, GGML_TYPE_F32, orient, {rows, cols}, 0, seed);
    std::vector<float> src((size_t) (rows * cols));
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = (float) ((i * 3 + seed) % 17) * 0.01f;
    }
    return encode_matrix(d, src.data(), src.size());
}

// One group: owning {0,1}, rank 4/4, dims 8/8, slices {0,4}/{4,4}.
static xkv_factor_group_payload make_group(uint32_t index, uint64_t seed_base, bool with_landmark,
                                           std::shared_ptr<const encoded_matrix> share_bk = nullptr,
                                           std::shared_ptr<const encoded_matrix> share_bv = nullptr) {
    xkv_factor_group_payload g;
    g.group_index      = index;
    g.owning_layers    = {0, 1};
    g.rank_k           = 4;
    g.rank_v           = 4;
    g.layer_feature_offsets_k = {0, 4};
    g.layer_feature_dims_k    = {4, 4};
    g.layer_feature_offsets_v = {0, 4};
    g.layer_feature_dims_v    = {4, 4};
    g.total_dim_k = 8;
    g.total_dim_v = 8;
    g.a_k = make_stream(factor_role::a_k, orientation::token_major, 4, 4, seed_base + 1);
    if (share_bk != nullptr) {
        g.b_k = share_bk;
    } else {
        g.set_b_k(make_stream(factor_role::b_k, orientation::feature_major_transposed, 8, 4, seed_base + 2));
    }
    g.a_v = make_stream(factor_role::a_v, orientation::token_major, 4, 4, seed_base + 3);
    if (share_bv != nullptr) {
        g.b_v = share_bv;
    } else {
        g.set_b_v(make_stream(factor_role::b_v, orientation::feature_major_transposed, 8, 4, seed_base + 4));
    }
    if (with_landmark) {
        g.landmark = make_stream(factor_role::landmark, orientation::token_major, 4, 8, seed_base + 5);
    }
    g.config_fingerprint = 0xC0DE + index;
    g.refresh_descriptor_fingerprint();
    g.update_byte_counters();
    return g;
}

static std::shared_ptr<xkv_segment> make_segment(uint64_t id, uint64_t ver, uint64_t payload_base,
                                                 int n_groups, bool with_landmark,
                                                 std::shared_ptr<const encoded_matrix> share_bk = nullptr,
                                                 std::shared_ptr<const encoded_matrix> share_bv = nullptr) {
    auto seg = std::make_shared<xkv_segment>();
    seg->segment_id      = id;
    seg->segment_version = ver;
    seg->profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
    seg->source  = LLAMA_XKV_SOURCE_DECODED_HOT;
    seg->profile_fingerprint = 0xA1;
    seg->source_fingerprint  = 0xA2;
    seg->layer_group_map_fingerprint = 0xA3 + id;
    seg->descriptor_fingerprint      = 0xA4 + ver;
    seg->n_rows = 4;
    // rows 0,1,3 live; row 2 dead (retained id, no payload record)
    seg->row_payload_ids = {payload_base + 0, payload_base + 1, payload_base + 2, payload_base + 3};
    seg->live_rows.resize(4);
    seg->live_rows[0] = true;
    seg->live_rows[1] = true;
    seg->live_rows[2] = false;
    seg->live_rows[3] = true;
    seg->n_live_rows = 3;
    for (int gi = 0; gi < n_groups; ++gi) {
        // Shared B handles apply to group 0 only; higher groups own their B
        // (exercises both shared and unique B allocations in one bundle).
        const bool share = (gi == 0);
        seg->groups.push_back(make_group((uint32_t) gi, payload_base + 100 * (uint64_t) gi,
                                         with_landmark && gi == 0, share ? share_bk : nullptr,
                                         share ? share_bv : nullptr));
    }
    seg->update_byte_counters();
    return seg;
}

struct test_world {
    std::vector<std::shared_ptr<const xkv_segment>> segments;
    std::vector<xkv_hot_payload_binding> hot;
    std::vector<xkv_state_payload> factored; // live rows only
    xkv_accounting accounting = test_accounting();
};

static xkv_state_payload live_payload(uint64_t id, uint64_t seg, uint64_t ver, uint32_t row, uint64_t gen) {
    xkv_state_payload p;
    p.payload_id = id;
    p.generation = gen;
    p.live       = 1;
    p.locator.kind = xkv_location_kind::factored;
    p.locator.segment_id      = seg;
    p.locator.segment_version = ver;
    p.locator.row             = row;
    p.locator.storage_generation = gen;
    p.locator.state           = xkv_state::factored;
    p.locator.seal_tx_nonce   = 0;
    return p;
}

static test_world make_world() {
    test_world w;
    // Two segments share one B pair (dedup identity); versions of segment 7.
    auto s7v1 = make_segment(7, 1, 100, 1, true);
    auto s7v2 = make_segment(7, 2, 200, 2, false, s7v1->groups[0].b_k, s7v1->groups[0].b_v);
    w.segments.push_back(std::move(s7v1));
    w.segments.push_back(std::move(s7v2));
    w.hot.push_back({901, 5, 3, xkv_state::hot_committed});
    // Payload records cover live rows only (0,1,3); dead row 2 has no record.
    for (uint64_t seg_i = 0; seg_i < 2; ++seg_i) {
        const auto & seg  = w.segments[(size_t) seg_i];
        const uint64_t base = seg_i == 0 ? 100 : 200;
        const uint64_t ver  = seg_i == 0 ? 1 : 2;
        for (uint32_t r = 0; r < 4; ++r) {
            if (!seg->live_rows[r]) {
                continue;
            }
            w.factored.push_back(live_payload(base + r, 7, ver, r, 9));
        }
    }
    std::sort(w.factored.begin(), w.factored.end(),
              [](const auto & a, const auto & b) { return a.payload_id < b.payload_id; });
    return w;
}

static bool capture_world(const test_world & w, xkv_state_image & img, std::string * err = nullptr,
                          const xkv_state_readback * rb = nullptr) {
    return capture_image(w.segments, w.hot, w.factored, w.accounting, 16, 8, test_config(), test_fps(),
                         test_prov(), test_stamp(), test_limits(), img, rb, err);
}

static std::vector<uint8_t> encode_world(std::string * err = nullptr) {
    xkv_state_image img;
    assert(capture_world(make_world(), img, err));
    std::vector<uint8_t> bytes;
    assert(encode_image(img, bytes, test_limits(), err));
    return bytes;
}

// ---------------------------------------------------------------------------
// 1. Determinism / roundtrip / shared-B identity / accounting recompute
// ---------------------------------------------------------------------------

static void test_deterministic_encode() {
    std::cout << "[Test] deterministic encode..." << std::endl;
    test_world w = make_world();
    xkv_state_image a, b;
    std::string err;
    assert(capture_world(w, a, &err));
    assert(capture_world(w, b, &err));
    std::vector<uint8_t> ba, bb;
    assert(encode_image(a, ba, test_limits(), &err));
    assert(encode_image(b, bb, test_limits(), &err));
    assert(ba == bb);
    assert(image_matches_bytes(a, ba.data(), ba.size(), test_limits()));
    assert(!image_matches_bytes(b, ba.data(), ba.size() - 1, test_limits()));
    // Precomputed size matches the wire size exactly.
    uint64_t pre = 0;
    assert(encoded_size(a, pre, &err));
    assert(pre == ba.size());
}

static void test_roundtrip_byte_for_byte() {
    std::cout << "[Test] byte-for-byte roundtrip..." << std::endl;
    const std::vector<uint8_t> bytes = encode_world();
    xkv_state_image img;
    std::string err;
    assert(decode_image(bytes.data(), bytes.size(), img, test_fps(), test_prov(), test_limits(), &err));
    std::vector<uint8_t> re;
    assert(encode_image(img, re, test_limits(), &err));
    assert(re == bytes);
    assert(image_matches_bytes(img, bytes.data(), bytes.size(), test_limits()));
    // Every fingerprint, digest, epoch, and config field survives.
    assert(img.fingerprints == test_fps());
    assert(img.provenance == test_prov());
    assert(img.config == test_config());
    assert(img.stamp.view.topology_epoch == 11 && img.stamp.view.publish_epoch == 22);
    assert(img.stamp.view.layout_epoch == 33 && img.stamp.live_epoch == 44);
    assert(img.stamp.content_epoch == 55 && img.stamp.codec_epoch == 66 && img.stamp.binding_epoch == 77);
    assert(img.workspace_mib == 16 && img.decode_cache_mib == 8);
    // Dead rows retained without payload records.
    size_t dead = 0;
    for (const auto & s : img.segments) {
        for (uint8_t b : s.live_bits) {
            dead += (b == 0);
        }
    }
    assert(dead == 2);
    for (const auto & p : img.payloads) {
        assert(p.live == 1);
    }
    assert(img.segments.size() == 2); // two versions of segment 7
    assert(img.segments[1].groups.size() == 2);
}

static void test_shared_b_identity() {
    std::cout << "[Test] shared B identity..." << std::endl;
    test_world w = make_world();
    xkv_state_image img;
    std::string err;
    assert(capture_world(w, img, &err));
    // v1g0: ak+bk+av+bv+lm; v2g0: ak+av (B shared); v2g1: ak+bk+av+bv.
    // B: shared K, shared V, group1 K, group1 V = 4 allocs; A_K x3.
    size_t n_bk = 0, n_bv = 0, n_ak = 0;
    for (const auto & a : img.allocations) {
        if (a.desc.role == factor_role::b_k) {
            ++n_bk;
        }
        if (a.desc.role == factor_role::b_v) {
            ++n_bv;
        }
        if (a.desc.role == factor_role::a_k) {
            ++n_ak;
        }
    }
    assert(n_bk == 2 && n_bv == 2 && n_ak == 3);
    const uint32_t v1bk = img.segments[0].groups[0].stream_bk;
    assert(img.segments[1].groups[0].stream_bk == v1bk);
    assert(img.segments[1].groups[1].stream_bk != v1bk);
    // Recomputed accounting: all streams 1024B; shared B 512B / 4 matrices.
    assert(img.counters.factored_bytes == 1024);
    assert(img.counters.live_payload_bytes == 1024);
    assert(img.counters.unique_b_matrices == 4);
    assert(img.counters.shared_b_bytes == 512);
    assert(img.counters.active_segments == 2);
    assert(img.counters.total_payloads == 7);
    assert(img.counters.workspace_budget_bytes == (uint64_t) 16 << 20);
    // Materialize preserves pointer identity for the shared allocation.
    std::vector<std::shared_ptr<const xkv_segment>> segs;
    assert(materialize_segments(img, segs, test_limits(), &err));
    assert(segs.size() == 2);
    assert(segs[0]->groups[0].b_k.get() == segs[1]->groups[0].b_k.get());
    assert(segs[0]->groups[0].b_v.get() == segs[1]->groups[0].b_v.get());
    assert(segs[1]->groups[1].b_k.get() != segs[0]->groups[0].b_k.get());
    // Materialize -> capture -> encode reproduces the exact bytes.
    const std::vector<uint8_t> orig = encode_world();
    test_world w2 = make_world();
    w2.segments = segs;
    xkv_state_image img2;
    assert(capture_world(w2, img2, &err));
    std::vector<uint8_t> re;
    assert(encode_image(img2, re, test_limits(), &err));
    assert(re == orig);
}

static void test_hot_only_image() {
    std::cout << "[Test] hot-only image roundtrip..." << std::endl;
    test_world w = make_world();
    w.segments.clear();
    w.factored.clear();
    xkv_state_image img;
    std::string err;
    assert(capture_world(w, img, &err));
    assert(img.allocations.empty() && img.segments.empty());
    assert(img.counters.factored_bytes == 0 && img.counters.live_payload_bytes == 0);
    assert(img.counters.total_payloads == 1 && img.counters.active_segments == 0);
    std::vector<uint8_t> bytes;
    assert(encode_image(img, bytes, test_limits(), &err));
    xkv_state_image back;
    assert(decode_image(bytes.data(), bytes.size(), back, test_fps(), test_prov(), test_limits(), &err));
    std::vector<uint8_t> re;
    assert(encode_image(back, re, test_limits(), &err));
    assert(re == bytes);
}

static void test_accounting_disagreement() {
    std::cout << "[Test] accounting disagreement rejected..." << std::endl;
    test_world w = make_world();
    std::string err;
    std::vector<uint8_t> sink;
    auto tweak = [&](auto fn) {
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        fn(img);
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    };
    tweak([](xkv_state_image & img) { img.counters.factored_bytes += 1; });
    tweak([](xkv_state_image & img) { img.counters.live_payload_bytes -= 1; });
    tweak([](xkv_state_image & img) { img.counters.shared_b_bytes += 64; });
    tweak([](xkv_state_image & img) { img.counters.unique_b_matrices = 3; });
    tweak([](xkv_state_image & img) { img.counters.active_segments = 1; });
    tweak([](xkv_state_image & img) { img.counters.total_payloads = 99; });
    tweak([](xkv_state_image & img) { img.counters.workspace_budget_bytes = 0; });
    tweak([](xkv_state_image & img) { img.counters.allocated_bytes = 10; }); // below factored+hot
    tweak([](xkv_state_image & img) { img.counters.reserved_bytes = 10; });  // below allocated
    // Persistent store budget: world holds 1024 factored bytes; a zero or
    // undersized store_mib budget refuses, as does a tight limits cap.
    tweak([](xkv_state_image & img) { img.config.store_mib = 0; });
    {
        test_world w = make_world();
        std::string err;
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.counters.factored_bytes = 2 * 1024 * 1024; // exceeds 1MiB store budget
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        xkv_state_limits tight = test_limits();
        tight.max_store_bytes  = 512; // below actual 1024 factored bytes
        test_world w = make_world();
        std::string err;
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        assert(!validate_image(img, test_fps(), test_prov(), tight, &err));
        std::vector<uint8_t> out;
        assert(!encode_image(img, out, tight, &err));
        assert(out.empty());
    }
}

// ---------------------------------------------------------------------------
// 2. Config / enum / high-water / provenance rules
// ---------------------------------------------------------------------------

static void test_config_enums() {
    std::cout << "[Test] config enum violations rejected..." << std::endl;
    std::string err;
    std::vector<uint8_t> sink;
    auto tweak_cfg = [&](auto fn) {
        test_world w = make_world();
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        fn(img.config);
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    };
    tweak_cfg([](xkv_state_config & c) { c.profile = 9; });
    tweak_cfg([](xkv_state_config & c) { c.source = 9; });
    tweak_cfg([](xkv_state_config & c) { c.group_size = 0; });
    tweak_cfg([](xkv_state_config & c) { c.rank_k = 0; });
    tweak_cfg([](xkv_state_config & c) { c.segment_tokens = 0; });
    tweak_cfg([](xkv_state_config & c) { c.store_mib = 0; });
    tweak_cfg([](xkv_state_config & c) { c.factor_balance = 9; });
    tweak_cfg([](xkv_state_config & c) { c.landmark_refine = 9; });
    tweak_cfg([](xkv_state_config & c) {
        c.landmark_refine = (uint32_t) LLAMA_XKV_LANDMARK_REFINE_BOUNDARY;
        c.landmark_refine_max_rows = 0;
    });
    tweak_cfg([](xkv_state_config & c) { c.factorizer = 9; });
    tweak_cfg([](xkv_state_config & c) { c.min_saving_ppm = 1000001; });
    tweak_cfg([](xkv_state_config & c) { c.min_coverage_ppm = 2000000; });
    tweak_cfg([](xkv_state_config & c) { c.next_segment_id = 0; });
    tweak_cfg([](xkv_state_config & c) { c.next_alloc_id = 0; });
    tweak_cfg([](xkv_state_config & c) { c.next_seal_tx_nonce = 0; });
    // Unknown segment profile/source enums.
    {
        test_world w = make_world();
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.segments[0].profile = 9;
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        test_world w = make_world();
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.segments[0].source = 9;
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
}

static void test_high_water_closure() {
    std::cout << "[Test] high-water closure..." << std::endl;
    std::string err;
    std::vector<uint8_t> sink;
    {
        test_world w = make_world();
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.config.next_segment_id = 7; // max segment id reaches high-water
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        test_world w = make_world();
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.config.next_alloc_id = 11; // max alloc id reaches high-water
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        test_world w = make_world();
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.payloads[0].locator.seal_tx_nonce = 5;
        img.config.next_seal_tx_nonce = 5; // nonce reaches high-water
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
}

static void test_landmark_profile_rules() {
    std::cout << "[Test] landmark/profile consistency..." << std::endl;
    std::string err;
    std::vector<uint8_t> sink;
    {
        // Landmarks profile with a landmark-less group is rejected.
        test_world w = make_world();
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.segments[1].profile = (uint32_t) LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS;
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        // Landmarks profile with all groups landmarked passes.
        test_world w = make_world();
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.segments[0].profile = (uint32_t) LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS;
        assert(validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(encode_image(img, sink, test_limits(), &err));
    }
    {
        // Landmark stream type must match the configured landmark type.
        test_world w = make_world();
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.config.landmark_type = (uint32_t) GGML_TYPE_Q8_0;
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        // Unspecified landmark type skips the check.
        test_world w = make_world();
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.config.landmark_type = (uint32_t) GGML_TYPE_COUNT;
        assert(validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(encode_image(img, sink, test_limits(), &err));
    }
    {
        // v5 landmark chunk table roundtrip and validation with exact uint64 source_fingerprint
        test_world w = make_world();
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        auto & g = img.segments[0].groups[0];
        assert(g.has_landmark);
        // Populate landmark chunks table: 2 chunks of 2 rows each
        g.landmark_chunks = {
            xkv_state_landmark_chunk::make(0, 2, 0.25f, 0xCAFEFEED0001ULL),
            xkv_state_landmark_chunk::make(2, 2, 0.75f, 0xCAFEFEED0002ULL)
        };
        std::vector<int64_t> test_pos(img.segments[0].n_rows, 0);
        const uint64_t lm_fp = img.allocations[g.stream_landmark].desc.fingerprint();
        g.landmark_chunks_fingerprint = xkv_state_landmark_chunks_fingerprint(
            g.landmark_chunks, img.segments[0].n_rows,
            img.segments[0].row_payload_ids.data(), nullptr, test_pos.data(),
            lm_fp, img.stamp, img.fingerprints.rope);
        assert(validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        std::vector<uint8_t> bytes;
        assert(encode_image(img, bytes, test_limits(), &err));
        xkv_state_image back;
        assert(decode_image(bytes.data(), bytes.size(), back, test_fps(), test_prov(), test_limits(), &err));
        const auto & back_g = back.segments[0].groups[0];
        assert(back_g.landmark_chunks.size() == 2);
        assert(back_g.landmark_chunks[0].source_fingerprint == 0xCAFEFEED0001ULL);
        assert(back_g.landmark_chunks[1].source_fingerprint == 0xCAFEFEED0002ULL);
        assert(back_g.landmark_chunks == g.landmark_chunks);
        assert(back_g.landmark_chunks_fingerprint == g.landmark_chunks_fingerprint);

        // Fault: zero source fingerprint fails closed
        {
            xkv_state_image bad = img;
            bad.segments[0].groups[0].landmark_chunks[0].source_fingerprint = 0;
            assert(!validate_image(bad, test_fps(), test_prov(), test_limits(), &err));
            assert(std::string(err).find("landmark source fingerprint missing") != std::string::npos);
        }
        // Fault: tampered source fingerprint fails closed (closure fingerprint mismatch)
        {
            xkv_state_image bad = img;
            bad.segments[0].groups[0].landmark_chunks[0].source_fingerprint ^= 0x1ULL;
            assert(!validate_image(bad, test_fps(), test_prov(), test_limits(), &err));
            assert(std::string(err).find("fingerprint mismatch") != std::string::npos);
        }
        // Fault: truncation in wire landmark chunks fails closed
        {
            // Bytes size without trailing envelope checksum:
            assert(bytes.size() > 20);
            xkv_state_image truncated_out;
            assert(!decode_image(bytes.data(), bytes.size() - 25, truncated_out, test_fps(), test_prov(), test_limits(), &err));
            assert(truncated_out.allocations.empty());
        }
    }
}

static void test_provenance_match() {
    std::cout << "[Test] provenance matching..." << std::endl;
    const std::vector<uint8_t> bytes = encode_world();
    std::string err;
    xkv_state_image out;
    // All-zero expected entries are unavailable and skipped.
    xkv_state_provenance none;
    assert(decode_image(bytes.data(), bytes.size(), out, test_fps(), none, test_limits(), &err));
    // Single-byte flips in any digest refuse, destination unchanged.
    const int offs[3] = {0, 17, 31};
    for (int h = 0; h < 3; ++h) {
        xkv_state_provenance wrong = test_prov();
        uint8_t * dig = h == 0 ? wrong.model_sha256 : (h == 1 ? wrong.tri_calibration_sha256 : wrong.source_sha256);
        dig[offs[h]] ^= 0x01;
        assert(!decode_image(bytes.data(), bytes.size(), out, test_fps(), wrong, test_limits(), &err));
        assert(image_matches_bytes(out, bytes.data(), bytes.size(), test_limits()));
    }
    // Zeroed image digest vs nonzero expectation refuses at encode.
    {
        test_world w = make_world();
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        std::memset(img.provenance.source_sha256, 0, 32);
        std::vector<uint8_t> sink;
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
}

// ---------------------------------------------------------------------------
// 3. Invalid fields / overflow / truncation / readback
// ---------------------------------------------------------------------------

static void test_zero_length_invalid() {
    std::cout << "[Test] zero-length invalid fields..." << std::endl;
    test_world w = make_world();
    std::string err;
    std::vector<uint8_t> sink;
    {
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.allocations[0].bytes.clear();
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.segments[0].groups[0].owning_layers.clear();
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.segments[0].groups.clear();
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.payloads[0].payload_id = 0;
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.segments[0].groups[0].rank_k = 0;
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.allocations[0].alloc_id = 0;
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.segments[0].n_rows = 0;
        assert(!encode_image(img, sink, test_limits(), &err));
    }
}

static void test_integer_overflow() {
    std::cout << "[Test] integer overflow rejected before allocation..." << std::endl;
    std::string err;
    xkv_state_image out;
    // Envelope claims UINT64_MAX allocations: rejected by count limit, no allocation.
    std::vector<uint8_t> evil;
    auto push_u32 = [&](uint32_t v) {
        evil.push_back((uint8_t) (v & 0xFF));
        evil.push_back((uint8_t) ((v >> 8) & 0xFF));
        evil.push_back((uint8_t) ((v >> 16) & 0xFF));
        evil.push_back((uint8_t) ((v >> 24) & 0xFF));
    };
    auto push_u64 = [&](uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            evil.push_back((uint8_t) ((v >> (i * 8)) & 0xFF));
        }
    };
    push_u32(XKV_STATE_MAGIC);
    push_u32(XKV_STATE_VERSION);
    for (int i = 0; i < 8; ++i) {
        push_u64(0);
    }
    for (int i = 0; i < 96 / 8; ++i) {
        push_u64(0); // provenance
    }
    // Minimal valid config body is long; instead stop right after provenance
    // with a truncated tail: the limit/truncation checks fire before any big
    // allocation regardless of config validity.
    const uint64_t sum = xkv_state_checksum(evil.data(), evil.size());
    push_u64(sum);
    assert(!decode_image(evil.data(), evil.size(), out, test_fps(), test_prov(), test_limits(), &err));
    assert(out.allocations.empty() && out.segments.empty() && out.payloads.empty());
    // Declared stream length huge with tiny body: truncated, not allocated.
    const std::vector<uint8_t> good = encode_world();
    assert(!decode_image(good.data(), 240, out, test_fps(), test_prov(), test_limits(), &err));
    // Claimed UINT64_MAX allocations (n_alloc field at envelope offset 424):
    // rejected by the count limit with no large allocation.
    {
        std::vector<uint8_t> huge = good;
        for (int i = 0; i < 8; ++i) {
            huge[452 + (size_t) i] = 0xFF; // n_alloc field at envelope offset 452 (v4)
        }
        const uint64_t hsum = xkv_state_checksum(huge.data(), huge.size() - 8);
        for (int i = 0; i < 8; ++i) {
            huge[huge.size() - 8 + (size_t) i] = (uint8_t) ((hsum >> (i * 8)) & 0xFF);
        }
        assert(!decode_image(huge.data(), huge.size(), out, test_fps(), test_prov(), test_limits(), &err));
        assert(std::string(err).find("exceeds limit") != std::string::npos);
        assert(out.allocations.empty());
    }
    // from_store_budgets saturates instead of wrapping.
    const xkv_state_limits lim =
        xkv_state_limits::from_store_budgets(0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 8, 8, 64, 64, 8, 64);
    assert(lim.max_envelope_bytes == 0xFFFFFFFFFFFFFFFFULL);
}

static void test_truncated() {
    std::cout << "[Test] truncated inputs rejected..." << std::endl;
    const std::vector<uint8_t> bytes = encode_world();
    std::string err;
    xkv_state_image out;
    const size_t cuts[] = {0, 1, 4, 7, 8, 100, 440, bytes.size() / 2, bytes.size() - 9, bytes.size() - 8,
                           bytes.size() - 1};
    for (size_t c : cuts) {
        assert(!decode_image(bytes.data(), c, out, test_fps(), test_prov(), test_limits(), &err));
        assert(out.allocations.empty());
    }
    assert(!decode_image(nullptr, 0, out, test_fps(), test_prov(), test_limits(), &err));
}

static void test_device_readback() {
    std::cout << "[Test] device-resident streams need explicit readback..." << std::endl;
    std::string err;
    // Segment with empty host bytes for one A stream (device-resident).
    auto seg = make_segment(7, 1, 100, 1, false);
    seg->groups[0].a_k.bytes.clear();
    std::vector<std::shared_ptr<const xkv_segment>> segs = {seg};
    std::vector<xkv_state_payload> pays;
    pays.push_back(live_payload(100, 7, 1, 0, 9));
    pays.push_back(live_payload(101, 7, 1, 1, 9));
    pays.push_back(live_payload(103, 7, 1, 3, 9));
    xkv_accounting acc   = test_accounting();
    acc.allocated_bytes  = 1 << 20;
    acc.reserved_bytes   = 1 << 20;
    acc.hot_bytes        = 0;
    auto do_capture = [&](const xkv_state_readback * rb, xkv_state_image & img) {
        return capture_image(segs, {}, pays, acc, 16, 8, test_config(), test_fps(), test_prov(),
                             test_stamp(), test_limits(), img, rb, &err);
    };
    {
        // No readback: fail-closed, output unchanged.
        xkv_state_image img;
        assert(!do_capture(nullptr, img));
        assert(std::string(err).find("readback") != std::string::npos);
        assert(img.allocations.empty());
    }
    {
        // Wrong-size readback bytes: rejected.
        xkv_state_readback rb;
        xkv_state_readback_entry e;
        e.segment_id      = 7;
        e.segment_version = 1;
        e.group_index     = 0;
        e.role            = (uint32_t) factor_role::a_k;
        e.bytes.assign(63, 0x5A); // A_K needs 64 bytes
        rb.entries.push_back(e);
        xkv_state_image img;
        assert(!do_capture(&rb, img));
        assert(img.allocations.empty());
    }
    {
        // Exact-size readback wins and roundtrips byte-for-byte.
        xkv_state_readback rb;
        xkv_state_readback_entry e;
        e.segment_id      = 7;
        e.segment_version = 1;
        e.group_index     = 0;
        e.role            = (uint32_t) factor_role::a_k;
        e.bytes.assign(64, 0x5A);
        rb.entries.push_back(e);
        xkv_state_image img;
        assert(do_capture(&rb, img));
        assert(img.allocations[0].bytes == e.bytes);
        std::vector<uint8_t> bytes;
        assert(encode_image(img, bytes, test_limits(), &err));
        xkv_state_image back;
        assert(decode_image(bytes.data(), bytes.size(), back, test_fps(), test_prov(), test_limits(), &err));
        std::vector<uint8_t> re;
        assert(encode_image(back, re, test_limits(), &err));
        assert(re == bytes);
    }
}

// ---------------------------------------------------------------------------
// 4. Duplicates / descriptors / checksums / closure / budgets
// ---------------------------------------------------------------------------

static void test_duplicates_rejected() {
    std::cout << "[Test] duplicate ids rejected..." << std::endl;
    test_world w = make_world();
    std::string err;
    std::vector<uint8_t> sink;
    {
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.payloads[1].payload_id = img.payloads[0].payload_id;
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.segments[1].segment_version = img.segments[0].segment_version;
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        // Two groups sharing one A stream: duplication, not sharing.
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.segments[1].groups[0].stream_ak = img.segments[0].groups[0].stream_ak;
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        // Allocation referenced as both A and B.
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.segments[1].groups[1].stream_bk = img.segments[0].groups[0].stream_ak;
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        // Duplicate allocation ids break canonical order.
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.allocations[1].alloc_id = img.allocations[0].alloc_id;
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
}

static void test_bad_descriptor() {
    std::cout << "[Test] bad descriptors rejected..." << std::endl;
    test_world w = make_world();
    std::string err;
    std::vector<uint8_t> sink;
    {
        // Corrupt descriptor bytes, checksum repaired: descriptor error (not checksum).
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.allocations[0].desc_bytes[10] ^= 0xFF;
        fix_stream_checksum(img, 0);
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        // v1 descriptor bytes (old format) rejected.
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        img.allocations[0].desc_bytes[4] = 1; // version field -> 1
        img.allocations[0].desc_bytes[5] = 0;
        img.allocations[0].desc_bytes[6] = 0;
        img.allocations[0].desc_bytes[7] = 0;
        fix_stream_checksum(img, 0);
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(std::string(err).find("v1") != std::string::npos);
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        // Decode path: patch envelope desc version to v1 and repair the
        // whole-envelope checksum so the descriptor refusal is the reason.
        // Layout: magic(4) ver(4) fps(64) prov(96) config(108) stamp(56)
        // mibs(12) counters(80) n_alloc(8) alloc_id(4) desc_len(8) ->
        // desc_bytes start at 472, version field at desc_bytes[4..8) = 476 (v4).
        std::vector<uint8_t> bytes = encode_world();
        assert(bytes.size() > 452);
        bytes[476] = 1;
        bytes[477] = 0;
        bytes[478] = 0;
        bytes[479] = 0;
        const uint64_t sum = xkv_state_checksum(bytes.data(), bytes.size() - 8);
        for (int i = 0; i < 8; ++i) {
            bytes[bytes.size() - 8 + (size_t) i] = (uint8_t) ((sum >> (i * 8)) & 0xFF);
        }
        xkv_state_image out;
        assert(!decode_image(bytes.data(), bytes.size(), out, test_fps(), test_prov(), test_limits(), &err));
        assert(std::string(err).find("v1") != std::string::npos);
        assert(out.allocations.empty());
    }
}

static void test_bad_checksum() {
    std::cout << "[Test] checksum failures rejected..." << std::endl;
    const std::vector<uint8_t> bytes = encode_world();
    std::string err;
    xkv_state_image out;
    {
        std::vector<uint8_t> bad = bytes;
        bad[492] ^= 0x01; // inside the first descriptor bytes (desc+20: row count, v4)
        assert(!decode_image(bad.data(), bad.size(), out, test_fps(), test_prov(), test_limits(), &err));
        assert(out.allocations.empty());
    }
    {
        std::vector<uint8_t> bad = bytes;
        bad.back() ^= 0x01; // envelope checksum
        assert(!decode_image(bad.data(), bad.size(), out, test_fps(), test_prov(), test_limits(), &err));
    }
}

static void test_group_closure() {
    std::cout << "[Test] group closure violations rejected..." << std::endl;
    test_world w = make_world();
    std::string err;
    std::vector<uint8_t> sink;
    auto tweak = [&](auto fn) {
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        fn(img);
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    };
    tweak([](xkv_state_image & img) { // overlapping K slices
        img.segments[0].groups[0].layer_feature_offsets_k = {0, 3};
    });
    tweak([](xkv_state_image & img) { // gapped K slices
        img.segments[0].groups[0].layer_feature_offsets_k = {0, 5};
        img.segments[0].groups[0].layer_feature_dims_k    = {3, 3};
    });
    tweak([](xkv_state_image & img) { // offset/dim size vs owning layers
        img.segments[0].groups[0].layer_feature_dims_v.push_back(1);
    });
    tweak([](xkv_state_image & img) { // duplicate owning layer
        img.segments[0].groups[0].owning_layers = {0, 0};
    });
    tweak([](xkv_state_image & img) { // dangling stream ref
        img.segments[0].groups[0].stream_av = 999999;
    });
    tweak([](xkv_state_image & img) { // landmark flag/ref disagreement
        img.segments[0].groups[0].has_landmark = false;
    });
}

static void test_pair_mismatch() {
    std::cout << "[Test] descriptor pair mismatch rejected..." << std::endl;
    // Turbo4 A/B with different seeds: individually valid, pair-incompatible.
    codec_desc dak = make_codec_desc(factor_role::a_k, GGML_TYPE_TURBO4_0, orientation::token_major,
                                     {4, 128}, 128, 42);
    codec_desc dbk = make_codec_desc(factor_role::b_k, GGML_TYPE_TURBO4_0,
                                     orientation::feature_major_transposed, {8, 128}, 128, 43);
    encoded_matrix ak, bk;
    ak.desc = dak;
    ak.bytes.assign((size_t) encoded_matrix_bytes(dak), 0);
    bk.desc = dbk;
    bk.bytes.assign((size_t) encoded_matrix_bytes(dbk), 0);
    encoded_matrix av = make_stream(factor_role::a_v, orientation::token_major, 4, 4, 70);
    encoded_matrix bv = make_stream(factor_role::b_v, orientation::feature_major_transposed, 8, 4, 71);

    auto seg = std::make_shared<xkv_segment>();
    seg->segment_id = 3;
    seg->segment_version = 1;
    seg->profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
    seg->source  = LLAMA_XKV_SOURCE_DECODED_HOT;
    seg->n_rows = 4;
    seg->n_live_rows = 4;
    seg->row_payload_ids = {31, 32, 33, 34};
    seg->live_rows = {true, true, true, true};
    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = {0, 1};
    g.rank_k = 128;
    g.rank_v = 4;
    g.layer_feature_offsets_k = {0, 4};
    g.layer_feature_dims_k    = {4, 4};
    g.layer_feature_offsets_v = {0, 4};
    g.layer_feature_dims_v    = {4, 4};
    g.total_dim_k = 8;
    g.total_dim_v = 8;
    g.a_k = std::move(ak);
    g.set_b_k(std::move(bk));
    g.a_v = std::move(av);
    g.set_b_v(std::move(bv));
    g.config_fingerprint = 1;
    g.refresh_descriptor_fingerprint();
    g.update_byte_counters();
    seg->groups.push_back(std::move(g));
    seg->update_byte_counters();

    std::vector<xkv_state_payload> pays;
    for (uint32_t r = 0; r < 4; ++r) {
        pays.push_back(live_payload(31 + r, 3, 1, r, 5));
    }
    xkv_state_config cfg = test_config();
    cfg.next_segment_id  = 4;
    xkv_state_image img;
    std::string err;
    assert(capture_image({seg}, {}, pays, test_accounting(), 16, 8, cfg, test_fps(), test_prov(),
                         test_stamp(), test_limits(), img, nullptr, &err) == false);
    assert(std::string(err).find("incompatible") != std::string::npos);
}

static void test_budget_overflow() {
    std::cout << "[Test] budget overflow rejected..." << std::endl;
    const std::vector<uint8_t> bytes = encode_world();
    std::string err;
    xkv_state_image out;
    auto tiny = [](auto fn) {
        xkv_state_limits lim = test_limits();
        fn(lim);
        return lim;
    };
    {
        xkv_state_limits lim = tiny([](xkv_state_limits & l) { l.max_envelope_bytes = 10; });
        assert(!decode_image(bytes.data(), bytes.size(), out, test_fps(), test_prov(), lim, &err));
        assert(out.allocations.empty());
    }
    {
        // Encode under a starved budget fails with output unchanged.
        xkv_state_limits lim = tiny([](xkv_state_limits & l) { l.max_envelope_bytes = 10; });
        test_world w = make_world();
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        std::vector<uint8_t> sink = {0xAA};
        assert(!encode_image(img, sink, lim, &err));
        assert(sink == std::vector<uint8_t>({0xAA}));
    }
    {
        xkv_state_limits lim = tiny([](xkv_state_limits & l) { l.max_stream_bytes = 0; });
        assert(!decode_image(bytes.data(), bytes.size(), out, test_fps(), test_prov(), lim, &err));
    }
    {
        xkv_state_limits lim = tiny([](xkv_state_limits & l) { l.max_segments = 1; });
        assert(!decode_image(bytes.data(), bytes.size(), out, test_fps(), test_prov(), lim, &err));
    }
    {
        xkv_state_limits lim = tiny([](xkv_state_limits & l) { l.max_payloads = 2; });
        assert(!decode_image(bytes.data(), bytes.size(), out, test_fps(), test_prov(), lim, &err));
    }
    {
        // Store-budget-derived caps: a 10-byte store budget cannot hold the world.
        xkv_state_limits lim =
            xkv_state_limits::from_store_budgets(10, 0, 8, 8, 64, 64, 8, 64);
        assert(!decode_image(bytes.data(), bytes.size(), out, test_fps(), test_prov(), lim, &err));
    }
}

static void test_from_store_budgets_formula() {
    std::cout << "[Test] store-budget limit formula..." << std::endl;
    const xkv_state_limits lim =
        xkv_state_limits::from_store_budgets(1000, 500, 8, 8, 64, 64, 8, 64);
    const uint64_t slack = 1048576ULL + 128ULL * 64 + 256ULL * 8 + 64ULL * 64;
    assert(lim.max_envelope_bytes == 1000 + 500 + slack);
    assert(lim.max_stream_bytes == lim.max_envelope_bytes);
    assert(lim.max_segments == 8 && lim.max_payloads == 64 && lim.max_rows_per_segment == 64);
}

// ---------------------------------------------------------------------------
// 5. Mismatch / trailing / versions / locator rules / unchanged-on-failure
// ---------------------------------------------------------------------------

static void test_mismatch_refusal() {
    std::cout << "[Test] fingerprint mismatch refusal..." << std::endl;
    const std::vector<uint8_t> bytes = encode_world();
    std::string err;
    xkv_state_image out;
    // Prime `out` with a good image; failed decodes must leave it identical.
    assert(decode_image(bytes.data(), bytes.size(), out, test_fps(), test_prov(), test_limits(), &err));
    // Member-pointer walk over all eight fingerprint fields (no UB pointer arithmetic).
    static uint64_t xkv_state_fingerprints::* const members[8] = {
        &xkv_state_fingerprints::source, &xkv_state_fingerprints::model, &xkv_state_fingerprints::rope,
        &xkv_state_fingerprints::tri_calibration, &xkv_state_fingerprints::profile,
        &xkv_state_fingerprints::factorizer, &xkv_state_fingerprints::codec,
        &xkv_state_fingerprints::backend};
    for (int i = 0; i < 8; ++i) {
        xkv_state_fingerprints wrong = test_fps();
        wrong.*members[i] ^= 0xDEADBEEF;
        assert(!decode_image(bytes.data(), bytes.size(), out, wrong, test_prov(), test_limits(), &err));
        assert(image_matches_bytes(out, bytes.data(), bytes.size(), test_limits()));
    }
}

static void test_trailing_bytes() {
    std::cout << "[Test] trailing bytes rejected..." << std::endl;
    std::vector<uint8_t> bytes = encode_world();
    std::string err;
    xkv_state_image out;
    bytes.push_back(0x00);
    assert(!decode_image(bytes.data(), bytes.size(), out, test_fps(), test_prov(), test_limits(), &err));
    bytes.back() = 0xFF;
    bytes.push_back(0x01);
    bytes.push_back(0x02);
    assert(!decode_image(bytes.data(), bytes.size(), out, test_fps(), test_prov(), test_limits(), &err));
    assert(out.allocations.empty());
}

static void test_version_magic_rejects() {
    std::cout << "[Test] magic/version rejects..." << std::endl;
    std::vector<uint8_t> bytes = encode_world();
    std::string err;
    xkv_state_image out;
    {
        std::vector<uint8_t> bad = bytes; // bad magic
        bad[0] ^= 0xFF;
        assert(!decode_image(bad.data(), bad.size(), out, test_fps(), test_prov(), test_limits(), &err));
        assert(std::string(err).find("magic") != std::string::npos);
    }
    {
        std::vector<uint8_t> bad = bytes; // legacy version 0
        bad[4] = 0;
        bad[5] = 0;
        bad[6] = 0;
        bad[7] = 0;
        assert(!decode_image(bad.data(), bad.size(), out, test_fps(), test_prov(), test_limits(), &err));
        assert(std::string(err).find("legacy") != std::string::npos);
    }
    {
        std::vector<uint8_t> bad = bytes; // retired v1 layout
        bad[4] = 1;
        bad[5] = 0;
        bad[6] = 0;
        bad[7] = 0;
        assert(!decode_image(bad.data(), bad.size(), out, test_fps(), test_prov(), test_limits(), &err));
        assert(std::string(err).find("old") != std::string::npos);
    }
    {
        std::vector<uint8_t> bad = bytes; // retired v2 layout
        bad[4] = 2;
        bad[5] = 0;
        bad[6] = 0;
        bad[7] = 0;
        assert(!decode_image(bad.data(), bad.size(), out, test_fps(), test_prov(), test_limits(), &err));
        assert(std::string(err).find("old") != std::string::npos);
    }
    {
        std::vector<uint8_t> bad = bytes; // retired v3 layout (no exact codecs/budgets)
        bad[4] = 3;
        bad[5] = 0;
        bad[6] = 0;
        bad[7] = 0;
        assert(!decode_image(bad.data(), bad.size(), out, test_fps(), test_prov(), test_limits(), &err));
        assert(std::string(err).find("old") != std::string::npos);
    }
    {
        std::vector<uint8_t> bad = bytes; // retired v4 layout (landmark chunks without source fingerprint)
        bad[4] = 4;
        bad[5] = 0;
        bad[6] = 0;
        bad[7] = 0;
        assert(!decode_image(bad.data(), bad.size(), out, test_fps(), test_prov(), test_limits(), &err));
        assert(std::string(err).find("old") != std::string::npos);
    }
    {
        std::vector<uint8_t> bad = bytes; // unknown future version
        bad[4] = 99;
        bad[5] = 0;
        bad[6] = 0;
        bad[7] = 0;
        assert(!decode_image(bad.data(), bad.size(), out, test_fps(), test_prov(), test_limits(), &err));
        assert(std::string(err).find("unknown") != std::string::npos);
    }
}

static void test_locator_rules() {
    std::cout << "[Test] locator closure rules..." << std::endl;
    test_world w = make_world();
    std::string err;
    std::vector<uint8_t> sink;
    {
        // Flat payloads are non-bundle locators: live ones roundtrip, tombstones
        // and bundle references are rejected.
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        xkv_state_payload flat;
        flat.payload_id = 777;
        flat.generation = 4;
        flat.live       = 1;
        flat.locator.kind = xkv_location_kind::flat_quantized;
        flat.locator.segment_id      = 0;
        flat.locator.segment_version = 0;
        flat.locator.row             = 2;
        flat.locator.storage_generation = 4;
        flat.locator.state           = xkv_state::flat_tq;
        flat.locator.seal_tx_nonce   = 0;
        img.payloads.push_back(flat);
        std::sort(img.payloads.begin(), img.payloads.end(),
                  [](const auto & a, const auto & b) { return a.payload_id < b.payload_id; });
        // Counters must be refreshed for the added payload: total_payloads.
        img.counters.total_payloads += 1;
        assert(validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        std::vector<uint8_t> bytes;
        assert(encode_image(img, bytes, test_limits(), &err));
        xkv_state_image back;
        assert(decode_image(bytes.data(), bytes.size(), back, test_fps(), test_prov(), test_limits(), &err));
        std::vector<uint8_t> re;
        assert(encode_image(back, re, test_limits(), &err));
        assert(re == bytes);
        size_t flat_idx = 0;
        for (size_t i = 0; i < img.payloads.size(); ++i) {
            if (img.payloads[i].payload_id == 777) {
                flat_idx = i;
            }
        }
        assert(img.payloads[flat_idx].payload_id == 777);
        img.payloads[flat_idx].live = 0; // tombstone records rejected
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        img.payloads[flat_idx].live = 1;
        img.payloads[flat_idx].locator.segment_id = 7; // flat bundle ref
        img.payloads[flat_idx].locator.segment_version = 1;
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        // Fabricated tombstone records are rejected at capture and validate.
        test_world w2 = make_world();
        xkv_state_image img;
        assert(capture_world(w2, img, &err));
        xkv_state_payload tomb = live_payload(102, 7, 1, 2, 9);
        tomb.live              = 0;
        std::vector<xkv_state_payload> pays;
        pays.push_back(tomb);
        xkv_state_image img2;
        assert(!capture_image(w2.segments, w2.hot, pays, w2.accounting, 16, 8, test_config(), test_fps(),
                              test_prov(), test_stamp(), test_limits(), img2, nullptr, &err));
        assert(std::string(err).find("tombstone") != std::string::npos);
        // ... and a dead-row id smuggled into the live table is rejected.
        // id 102 is dead row 2's id; locator at live row 0 isolates the
        // row-mapping check (sorted order and counters kept consistent).
        img.payloads.push_back(live_payload(102, 7, 1, 0, 9));
        std::sort(img.payloads.begin(), img.payloads.end(),
                    [](const auto & a, const auto & b) { return a.payload_id < b.payload_id; });
        img.counters.total_payloads += 1;
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        xkv_state_image img; // unknown segment
        assert(capture_world(w, img, &err));
        img.payloads[0].locator.segment_id = 424242;
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        xkv_state_image img; // row out of range
        assert(capture_world(w, img, &err));
        for (auto & p : img.payloads) {
            if (p.live && p.locator.kind == xkv_location_kind::factored) {
                p.locator.row = 999;
                break;
            }
        }
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        xkv_state_image img; // generation mismatch
        assert(capture_world(w, img, &err));
        img.payloads[0].locator.storage_generation += 1;
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        xkv_state_image img; // live payload claiming a dead row
        assert(capture_world(w, img, &err));
        for (auto & p : img.payloads) {
            if (p.live && p.locator.kind == xkv_location_kind::factored) {
                p.locator.row = 2; // dead row in both bundles
                break;
            }
        }
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        xkv_state_image img; // orphan live row: drop a live payload
        assert(capture_world(w, img, &err));
        for (size_t i = 0; i < img.payloads.size(); ++i) {
            if (img.payloads[i].live && img.payloads[i].locator.kind == xkv_location_kind::factored) {
                img.payloads.erase(img.payloads.begin() + (ptrdiff_t) i);
                img.counters.total_payloads -= 1; // keep counters consistent to isolate the defect
                break;
            }
        }
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        xkv_state_image img; // unknown kind / state values
        assert(capture_world(w, img, &err));
        img.payloads[0].locator.kind = (xkv_location_kind) 9;
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        img.payloads[0].locator.kind = xkv_location_kind::factored;
        img.payloads[0].locator.state = (xkv_state) 9;
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    {
        // Dead-row id smuggled into the table via a hot record is rejected.
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        xkv_state_payload smuggled;
        smuggled.payload_id = 102; // dead row id in segment (7,1)
        smuggled.generation = 9;
        smuggled.live       = 1;
        smuggled.locator.kind = xkv_location_kind::hot;
        smuggled.locator.segment_id      = 0;
        smuggled.locator.segment_version = 0;
        smuggled.locator.row             = 6;
        smuggled.locator.storage_generation = 9;
        smuggled.locator.state           = xkv_state::hot_committed;
        smuggled.locator.seal_tx_nonce   = 0;
        img.payloads.push_back(smuggled);
        std::sort(img.payloads.begin(), img.payloads.end(),
                  [](const auto & a, const auto & b) { return a.payload_id < b.payload_id; });
        img.counters.total_payloads += 1;
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
}

static void test_failure_leaves_unchanged() {
    std::cout << "[Test] failure leaves destination unchanged..." << std::endl;
    const std::vector<uint8_t> bytes = encode_world();
    std::string err;
    xkv_state_image good;
    assert(decode_image(bytes.data(), bytes.size(), good, test_fps(), test_prov(), test_limits(), &err));
    // Garbage decodes.
    const uint8_t garbage[] = {0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01};
    for (size_t n = 1; n < sizeof(garbage); ++n) {
        assert(!decode_image(garbage, n, good, test_fps(), test_prov(), test_limits(), &err));
        assert(image_matches_bytes(good, bytes.data(), bytes.size(), test_limits()));
    }
    // Capture failure (null segment) leaves `out` unchanged.
    test_world w = make_world();
    std::vector<std::shared_ptr<const xkv_segment>> with_null = w.segments;
    with_null.push_back(nullptr);
    xkv_state_image keep = good;
    assert(!capture_image(with_null, w.hot, w.factored, w.accounting, 16, 8, test_config(), test_fps(),
                          test_prov(), test_stamp(), test_limits(), keep, nullptr, &err));
    assert(image_matches_bytes(keep, bytes.data(), bytes.size(), test_limits()));
    // Materialize failure leaves output vector unchanged.
    xkv_state_image bad_img = good;
    bad_img.payloads.clear(); // orphans live rows
    bad_img.counters.total_payloads = 0; // keep counters consistent to isolate the defect
    std::vector<std::shared_ptr<const xkv_segment>> mout;
    assert(!materialize_segments(bad_img, mout, test_limits(), &err));
    assert(mout.empty());
}

static void test_pid_reservation() {
    std::cout << "[Test] pid reservation (monotonic, concurrent, saturating)..." << std::endl;
    // Saturation input is refused with zero mutation.
    assert(!llama_kv_cells::reserve_payload_ids_through(UINT64_MAX));
    // A small reservation still works afterwards (nothing was mutated).
    assert(llama_kv_cells::reserve_payload_ids_through(1000));
    // Concurrent cell allocators (one cells instance per thread: the counter
    // is process-global) produce unique ids, all above the reservation.
    constexpr int kThreads = 8;
    constexpr int kPerThread = 64;
    std::mutex mu;
    std::vector<uint64_t> all;
    std::vector<std::thread> th;
    for (int t = 0; t < kThreads; ++t) {
        th.emplace_back([&, t] {
            llama_kv_cells cells;
            cells.resize(kPerThread);
            std::vector<uint64_t> local;
            for (int i = 0; i < kPerThread; ++i) {
                cells.pos_set((uint32_t) i, (llama_pos) (t * kPerThread + i));
                local.push_back(cells.payload_id_get((uint32_t) i));
            }
            // Racy reservation to a thread-dependent high-water.
            assert(llama_kv_cells::reserve_payload_ids_through(100000 + (uint64_t) t));
            std::lock_guard<std::mutex> lk(mu);
            all.insert(all.end(), local.begin(), local.end());
        });
    }
    for (auto & t : th) {
        t.join();
    }
    assert(all.size() == (size_t) kThreads * kPerThread);
    std::sort(all.begin(), all.end());
    for (size_t i = 1; i < all.size(); ++i) {
        assert(all[i] > all[i - 1]); // unique and strictly increasing
    }
    assert(all.front() > 1000); // everything issued past the reservation
    // Post-race reservation only advances: fresh cells keep unique fresh ids.
    assert(llama_kv_cells::reserve_payload_ids_through(200000));
    llama_kv_cells tail;
    tail.resize(2);
    tail.pos_set(0, 900000);
    tail.pos_set(1, 900001);
    const uint64_t a = tail.payload_id_get(0);
    const uint64_t b = tail.payload_id_get(1);
    assert(a > 200000 && b == a + 1);
    // Reserving below the high-water is a successful no-op (never retreats).
    assert(llama_kv_cells::reserve_payload_ids_through(10));
    llama_kv_cells tail2;
    tail2.resize(1);
    tail2.pos_set(0, 900002);
    assert(tail2.payload_id_get(0) > b);
}

static void test_config_compatible_matrix() {
    std::cout << "[Test] config compatibility matrix..." << std::endl;
    std::string err;
    const xkv_state_config live = test_config();
    assert(check_config_compatible(live, live, &err));
    // High-waters excluded: live advances past image values freely.
    {
        xkv_state_config img = live;
        img.next_segment_id += 100;
        img.next_alloc_id += 100;
        img.next_seal_tx_nonce += 100;
        assert(check_config_compatible(img, live, &err));
    }
    auto mismatch = [&](auto fn, const char * field) {
        xkv_state_config img = live;
        fn(img);
        assert(!check_config_compatible(img, live, &err));
        assert(std::string(err).find(field) != std::string::npos);
    };
    mismatch([](xkv_state_config & c) { c.mode = 1; }, "mode");
    mismatch([](xkv_state_config & c) { c.profile = 0; }, "profile");
    mismatch([](xkv_state_config & c) { c.source = 1; }, "source");
    mismatch([](xkv_state_config & c) { c.group_size = 8; }, "group_size");
    mismatch([](xkv_state_config & c) { c.rank_k = 8; }, "rank_k");
    mismatch([](xkv_state_config & c) { c.rank_v = 8; }, "rank_v");
    mismatch([](xkv_state_config & c) { c.segment_tokens = 128; }, "segment_tokens");
    mismatch([](xkv_state_config & c) { c.chunk_tokens = 16; }, "chunk_tokens");
    mismatch([](xkv_state_config & c) { c.sr_budget = 4; }, "sr_budget");
    // The reported bug: Turbo2-saved bundle under Turbo4-requested config.
    mismatch([](xkv_state_config & c) { c.factor_a_k = (uint32_t) GGML_TYPE_TURBO2_0; }, "factor_a_k");
    mismatch([](xkv_state_config & c) { c.factor_b_k = (uint32_t) GGML_TYPE_TURBO2_0; }, "factor_b_k");
    mismatch([](xkv_state_config & c) { c.factor_a_v = (uint32_t) GGML_TYPE_TURBO2_0; }, "factor_a_v");
    mismatch([](xkv_state_config & c) { c.factor_b_v = (uint32_t) GGML_TYPE_TURBO2_0; }, "factor_b_v");
    mismatch([](xkv_state_config & c) { c.factor_balance = 1; }, "factor_balance");
    mismatch([](xkv_state_config & c) { c.landmark_type = (uint32_t) GGML_TYPE_Q8_0; }, "landmark_type");
    mismatch([](xkv_state_config & c) {
        c.landmark_refine = (uint32_t) LLAMA_XKV_LANDMARK_REFINE_BOUNDARY;
    }, "landmark_refine");
    mismatch([](xkv_state_config & c) { c.landmark_refine_max_rows = 8; }, "landmark_refine_max");
    mismatch([](xkv_state_config & c) { c.workspace_mib = 32; }, "workspace_mib");
    mismatch([](xkv_state_config & c) { c.decode_cache_mib = 16; }, "decode_cache_mib");
    mismatch([](xkv_state_config & c) { c.store_mib = 2; }, "store_mib");
    mismatch([](xkv_state_config & c) { c.factorizer = 1; }, "factorizer");
    mismatch([](xkv_state_config & c) { c.min_saving_bytes = 8; }, "min_saving_bytes");
    mismatch([](xkv_state_config & c) { c.seed = 43; }, "seed");
    mismatch([](xkv_state_config & c) { c.min_saving_ppm = 200000; }, "min-saving");
    mismatch([](xkv_state_config & c) { c.min_coverage_ppm = 600000; }, "min-coverage");
}

static void test_codec_fingerprints() {
    std::cout << "[Test] codec/backend fingerprints..." << std::endl;
    const xkv_state_config cfg = test_config();
    const xkv_state_fingerprints fp = make_config_fingerprints(cfg);
    assert(fp.codec != 0 && fp.backend != 0);
    xkv_state_config other = cfg;
    other.factor_b_k = (uint32_t) GGML_TYPE_TURBO2_0;
    assert(make_config_fingerprints(other).codec != fp.codec);
    other = cfg;
    other.landmark_type = (uint32_t) GGML_TYPE_Q8_0;
    assert(make_config_fingerprints(other).codec != fp.codec);
}

static void test_zero_model_digest_refusal() {
    std::cout << "[Test] zero model digest refusal for factored state..." << std::endl;
    test_world w = make_world();
    std::string err;
    std::vector<uint8_t> sink;
    const std::vector<uint8_t> bytes = encode_world();
    // Zero expected digest with factored segments: refuse (not silently skip).
    {
        xkv_state_provenance none;
        xkv_state_image out;
        assert(!decode_image(bytes.data(), bytes.size(), out, test_fps(), none, test_limits(), &err));
        assert(std::string(err).find("provenance") != std::string::npos);
    }
    // Zero image digest with factored segments: refuse at encode/validate.
    {
        xkv_state_image img;
        assert(capture_world(w, img, &err));
        std::memset(img.provenance.model_sha256, 0, 32);
        assert(!validate_image(img, test_fps(), test_prov(), test_limits(), &err));
        assert(!encode_image(img, sink, test_limits(), &err));
    }
    // Segment-free image: zeros still skip (nothing factored to misattribute).
    {
        test_world w2 = make_world();
        w2.segments.clear();
        w2.factored.clear();
        xkv_state_image img;
        assert(capture_world(w2, img, &err));
        std::memset(img.provenance.model_sha256, 0, 32);
        xkv_state_provenance none;
        assert(validate_image(img, test_fps(), none, test_limits(), &err));
        assert(encode_image(img, sink, test_limits(), &err));
    }
}

int main() {
    std::cout << "=== Running XKV State Tests ===" << std::endl;
    test_deterministic_encode();
    test_roundtrip_byte_for_byte();
    test_shared_b_identity();
    test_hot_only_image();
    test_accounting_disagreement();
    test_config_enums();
    test_high_water_closure();
    test_landmark_profile_rules();
    test_provenance_match();
    test_zero_length_invalid();
    test_integer_overflow();
    test_truncated();
    test_device_readback();
    test_duplicates_rejected();
    test_bad_descriptor();
    test_bad_checksum();
    test_group_closure();
    test_pair_mismatch();
    test_budget_overflow();
    test_from_store_budgets_formula();
    test_mismatch_refusal();
    test_trailing_bytes();
    test_version_magic_rejects();
    test_locator_rules();
    test_failure_leaves_unchanged();
    test_config_compatible_matrix();
    test_codec_fingerprints();
    test_zero_model_digest_refusal();
    test_pid_reservation();
    std::cout << "All XKV state tests passed." << std::endl;
    return 0;
}
