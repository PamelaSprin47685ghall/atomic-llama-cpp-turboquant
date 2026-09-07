// Deterministic unit tests for PR07: Phase-aware landmarks and SR selection
// Covers:
// 1. Fragment partitioning: public/private/pending, run/phase boundaries, causal partial chunks, sparse gaps.
// 2. Multi-query isolation: future draft queries never affect prior selections (proven via future-high-score injection).
// 3. Exact 4-field segment_row_ref construction: inspect nonzero row, segment_version, storage_generation.
// 4. Same payload keeper dedup versus distinct logical occurrences at logical selection input.
// 5. GQA normalized max pooling with per-Q-head normalization.
// 6. Global span budget and top-k selection with segment_row_ref gather & CSR.
// 7. Landmark quantization recall (Q8_0 and Turbo4_0) with valid head_dim=128 / total_dim=256 bounds.
// 8. Normalized boundary refinement, refine_cap_hit tracking, and invalid bounds rejection.
// 9. Single-query causal enforcement & mandatory future protection rejection.
// 10. Landmark table deterministic LRU eviction with held external lease protection.
// 11. Heterogeneous generations per row in segment and fragment.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-landmark.h"
#include "llama-xkv-cache.h"
#include "llama-xkv-codec.h"
#include "llama-xkv-factor.h"
#include "llama-rerot.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <vector>

using namespace llama_xkv;
// ---------- production-bounded test instrumentation ----------
static std::shared_ptr<xkv_segment> create_test_segment(uint32_t n_rows, uint32_t rank_k,
    uint32_t total_dim_k, uint64_t seg_version);
#include <cstdlib>
#include <new>
namespace {
thread_local bool g_count_allocs = false;
thread_local size_t g_alloc_count = 0;
struct alloc_scope {
    bool prev;
    explicit alloc_scope(bool on) : prev(g_count_allocs) { g_count_allocs = on; }
    ~alloc_scope() { g_count_allocs = prev; }
};
}
void * operator new(std::size_t n) {
    if (g_count_allocs) ++g_alloc_count;
    void * p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void operator delete(void * p) noexcept { std::free(p); }
void * operator new[](std::size_t n) {
    if (g_count_allocs) ++g_alloc_count;
    void * p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }
void operator delete[](void * p, std::size_t) noexcept { std::free(p); }
static uint32_t test_padded_rank(const xkv_segment & seg) {
    return (uint32_t) seg.groups[0].b_k->desc.padded_shape.cols;
}
static std::vector<landmark_fragment_view> test_views(const std::vector<legal_fragment> & frags) {
    std::vector<landmark_fragment_view> v;
    for (const auto & f : frags) v.push_back(view_of_fragment(f));
    return v;
}
static uint32_t test_rmax(const std::vector<legal_fragment> & frags) {
    uint32_t m = 0;
    for (const auto & f : frags) m = std::max(m, f.row_count);
    return m;
}
struct bounded_single_out {
    std::vector<segment_row_ref> rows;
    std::vector<float> scores;
    uint32_t refined = 0;
    bool cap = false;
};
static bounded_single_out run_bounded_single(const sr_query & q,
    const std::vector<landmark_fragment_view> & views, const sr_selection_config & cfg,
    const xkv_segment * seg, uint32_t off, uint32_t dim, const phase_transform_fn & tx,
    uint64_t fp, std::vector<uint8_t> & scratch, const xkv_segment * const * segs = nullptr,
    size_t nsegs = 0, sr_bounded_attribution * attrib = nullptr) {
    bounded_single_out o;
    o.rows.resize(4096);
    o.scores.resize(views.size() + 8);
    size_t nr = 0, ns = 0;
    std::string err;
    bool ok = select_sr_query_bounded(q, views.data(), views.size(), cfg, seg, off, dim, tx,
        fp, scratch.data(), scratch.size(), o.rows.data(), o.rows.size(), &nr,
        o.scores.data(), o.scores.size(), &ns, &o.refined, &o.cap, &err, segs, nsegs,
        attrib);
    if (!ok) std::cerr << "bounded single failed: " << err << std::endl;
    assert(ok);
    o.rows.resize(nr);
    o.scores.resize(ns);
    return o;
}
static std::vector<uint8_t> test_make_scratch(uint32_t n, uint32_t h, uint32_t d, uint32_t r,
    uint32_t p) {
    size_t need = 0;
    std::string err;
    assert(landmark_select_scratch_bytes(n, h, d, r, p, need, &err));
    return std::vector<uint8_t>(need, 0);
}
static void test_base_landmark_reference() {
    std::cout << "[test_base_landmark_reference] starting..." << std::endl;
    auto seg = create_test_segment(8, 16, 64, 1);
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 8; ++i) {
        rows.push_back({i, 1000 + i, 1, 0, (int64_t) i, (int64_t) i, 0, true, true});
    }
    auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x1, -1);
    assert(frags.size() == 2);
    for (auto & f : frags) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
    auto lm0 = decode_matrix(frags[0].landmark_matrix, value_domain::canonical);
    auto lm1 = decode_matrix(frags[1].landmark_matrix, value_domain::canonical);
    std::vector<float> base_data = lm0;
    base_data.insert(base_data.end(), lm1.begin(), lm1.end());
    codec_desc bdesc = make_codec_desc(factor_role::landmark, GGML_TYPE_Q8_0,
        orientation::token_major, {2, 32}, 0, 0x5);
    auto base = std::make_shared<const encoded_matrix>(encode_matrix(bdesc, base_data.data(), base_data.size()));
    size_t pre_bytes = frags[0].allocated_bytes();
    assert(!attach_base_landmark(frags[0], base, 1, stamp, 0x1, (frags[0]).source_fingerprint, frags[0].payload_ids.data(), frags[0].generations.data(), frags[0].storage_positions.data(), frags[0].row_count, nullptr));
    assert(!frags[0].uses_base_landmark());
    assert(!frags[0].landmark_matrix.bytes.empty());
    assert(!attach_base_landmark(frags[0], base, 0, stamp, 0x1, (frags[0]).source_fingerprint, frags[0].payload_ids.data(), frags[0].generations.data(), frags[0].storage_positions.data(), frags[0].row_count - 1, nullptr));
    xkv_snapshot_stamp stale = stamp;
    stale.content_epoch = 99;
    assert(!attach_base_landmark(frags[0], base, 0, stale, 0x1, (frags[0]).source_fingerprint, frags[0].payload_ids.data(), frags[0].generations.data(), frags[0].storage_positions.data(), frags[0].row_count, nullptr));
    assert(!attach_base_landmark(frags[0], base, 0, stamp, 0x2, (frags[0]).source_fingerprint, frags[0].payload_ids.data(), frags[0].generations.data(), frags[0].storage_positions.data(), frags[0].row_count, nullptr));
    sr_query q;
    q.head_dim = 32;
    q.q_vec.assign(32, 0.6f);
    sr_selection_config cfg;
    cfg.sr_budget = 1;
    auto ref = select_sr_query(q, frags, cfg);
    assert(attach_base_landmark(frags[0], base, 0, stamp, 0x1, (frags[0]).source_fingerprint, frags[0].payload_ids.data(), frags[0].generations.data(), frags[0].storage_positions.data(), frags[0].row_count, nullptr));
    assert(attach_base_landmark(frags[1], base, 1, stamp, 0x1, (frags[1]).source_fingerprint, frags[1].payload_ids.data(), frags[1].generations.data(), frags[1].storage_positions.data(), frags[1].row_count, nullptr));
    assert(frags[0].uses_base_landmark() && frags[1].uses_base_landmark());
    assert(frags[0].landmark_matrix.bytes.empty());
    assert(frags[0].allocated_bytes() < pre_bytes);
    assert(frags[0].key.landmark_codec_fp == base->desc.fingerprint());
    auto ref2 = select_sr_query(q, frags, cfg);
    assert(ref2.selected_rows == ref.selected_rows);
    assert(ref2.scores == ref.scores);
    auto views = test_views(frags);
    std::vector<uint8_t> scratch =
        test_make_scratch(2, 1, 32, 4, test_padded_rank(*seg));
    auto out = run_bounded_single(q, views, cfg, nullptr, 0, 32, nullptr, 0, scratch);
    assert(out.rows == ref.selected_rows);
    bool threw = false;
    try {
        encode_fragment_landmark(frags[0], *seg, 0, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
    landmark_table table(1024 * 1024);
    auto fp0 = std::make_shared<const legal_fragment>(std::move(frags[0]));
    assert(table.insert(fp0));
    assert(table.fragment_count() == 1);
    table.invalidate_content_epoch(seg->segment_id, 12345);
    assert(table.fragment_count() == 0);
    fp0.reset();
    assert(base->bytes.size() > 0);
    std::cout << "[test_base_landmark_reference] passed!" << std::endl;
}
static void test_bounded_builder() {
    std::cout << "[test_bounded_builder] starting..." << std::endl;
    // NOTE: K landmarks take no V inputs by construction (no V params exist anywhere
    // in the builder API); the seal factory must not heap-decode V for landmark builds.
    auto seg = create_test_segment(10, 16, 64, 1); // F32 factors, rank 16
    const encoded_matrix * a_k = &seg->groups[0].a_k;
    const encoded_matrix * b_k = seg->groups[0].b_k.get();
    std::vector<uint32_t> srows(10);
    std::vector<int64_t> spos(10);
    for (uint32_t i = 0; i < 10; ++i) {
        srows[i] = i;
        spos[i] = (int64_t) i;
    }
    landmark_build_layer layers2[2] = {{0, 0, 32, 0}, {0, 32, 32, 0}};
    landmark_build_spec spec;
    spec.a_k = a_k;
    spec.b_k = b_k;
    spec.surviving_rows = srows.data();
    spec.n_rows = 10;
    spec.storage_positions = spos.data();
    spec.layers = layers2;
    spec.n_layers = 2;
    spec.chunk_tokens = 4; // 4 + 4 + 2 remainder coverage
    spec.landmark_type = GGML_TYPE_F32;
    spec.phase_tx_fingerprint = 0; // identity phase sentinel (null transform)
    spec.seed_extra = 0xBEEF;
    size_t ws = 0, ar = 0, nc = 0;
    assert(landmark_build_workspace_bytes(spec, landmark_encode_f32_row, ws, ar, nc, nullptr));
    assert(nc == 3);
    assert(ar == 3 * 64 * sizeof(float));
    std::vector<uint8_t> scratch(ws, 0);
    std::vector<uint8_t> arena(ar, 0);
    std::vector<landmark_build_output> outs(3);
    size_t no = 0;
    assert(build_landmarks_bounded(spec, nullptr, landmark_encode_f32_row, scratch.data(),
        scratch.size(), arena.data(), arena.size(), outs.data(), outs.size(), &no, nullptr));
    assert(no == 3);
    assert(outs[0].byte_offset == 0 && outs[1].byte_offset == 256 && outs[2].byte_offset == 512);
    assert(outs[0].byte_size == 256 && outs[2].byte_size == 256);
    assert(outs[0].error_bound == 0.0f);
    // Determinism: identical bytes/descs/fingerprints across runs.
    std::vector<uint8_t> arena2(ar, 0);
    std::vector<landmark_build_output> outs2(3);
    size_t no2 = 0;
    assert(build_landmarks_bounded(spec, nullptr, landmark_encode_f32_row, scratch.data(),
        scratch.size(), arena2.data(), arena2.size(), outs2.data(), outs2.size(), &no2, nullptr));
    assert(no2 == 3);
    assert(arena == arena2);
    for (int c = 0; c < 3; ++c) {
        assert(outs[c].source_fingerprint == outs2[c].source_fingerprint);
        assert(outs[c].desc == outs2[c].desc);
    }
    // One-byte-short refusals: arena is exact; scratch threshold is exact by search.
    assert(!build_landmarks_bounded(spec, nullptr, landmark_encode_f32_row, scratch.data(),
        scratch.size(), arena.data(), ar - 1, outs.data(), outs.size(), &no, nullptr));
    {
        auto attempt = [&](size_t bytes) {
            std::vector<uint8_t> sc(bytes, 0);
            size_t n = 0;
            std::vector<landmark_build_output> oo(3);
            return build_landmarks_bounded(spec, nullptr, landmark_encode_f32_row, sc.data(),
                sc.size(), arena.data(), arena.size(), oo.data(), oo.size(), &n, nullptr);
        };
        assert(attempt(ws));
        size_t lo = 0, hi = ws;
        while (lo + 1 < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (attempt(mid)) hi = mid;
            else lo = mid;
        }
        assert(!attempt(lo));
    }
    // Zero production heap after init (F32 factors + F32 encoder, identity phase).
    g_alloc_count = 0;
    {
        alloc_scope on(true);
        for (int rep = 0; rep < 3; ++rep) {
            size_t n = 0;
            bool ok = build_landmarks_bounded(spec, nullptr, landmark_encode_f32_row,
                scratch.data(), scratch.size(), arena.data(), arena.size(), outs.data(),
                outs.size(), &n, nullptr);
            assert(ok && n == 3);
        }
    }
    assert(g_alloc_count == 0);
    // Parity: single-slice builder means equal per-fragment F32 landmarks bit-exactly.
    {
        landmark_build_layer one[1] = {{0, 0, 32, 0}};
        landmark_build_spec s1 = spec;
        s1.layers = one;
        s1.n_layers = 1;
        s1.chunk_tokens = 4;
        std::vector<uint32_t> r8(8);
        std::vector<int64_t> p8(8);
        for (uint32_t i = 0; i < 8; ++i) {
            r8[i] = i;
            p8[i] = (int64_t) i;
        }
        s1.surviving_rows = r8.data();
        s1.n_rows = 8;
        s1.storage_positions = p8.data();
        size_t w1 = 0, a1 = 0, n1 = 0;
        assert(landmark_build_workspace_bytes(s1, landmark_encode_f32_row, w1, a1, n1, nullptr));
        assert(n1 == 2);
        std::vector<uint8_t> sc1(w1, 0), ar1(a1, 0);
        std::vector<landmark_build_output> o1(2);
        size_t nn = 0;
        assert(build_landmarks_bounded(s1, nullptr, landmark_encode_f32_row, sc1.data(),
            sc1.size(), ar1.data(), ar1.size(), o1.data(), o1.size(), &nn, nullptr));
        assert(nn == 2);
        xkv_snapshot_stamp stamp;
        stamp.live_epoch = 1;
        std::vector<row_meta> mrows;
        for (uint32_t i = 0; i < 8; ++i) {
            mrows.push_back({i, 1000 + i, 1, 0, (int64_t) i, (int64_t) i, 0, true, true});
        }
        auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, mrows, 4, 0, -1);
        assert(frags.size() == 2);
        for (auto & f : frags) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0, GGML_TYPE_F32);
        for (int c = 0; c < 2; ++c) {
            auto ref = decode_matrix(frags[c].landmark_matrix, value_domain::canonical);
            assert(ref.size() == 32);
            const float * got = reinterpret_cast<const float *>(ar1.data() + o1[c].byte_offset);
            for (int d = 0; d < 32; ++d) assert(got[d] == ref[d]);
            assert(o1[c].source_fingerprint == frags[c].source_fingerprint);
        }
    }
    // Closed failures: null encoder, empty rows, chunk 0, OOB row, layer gap/overrun.
    size_t nn = 0;
    std::vector<landmark_build_output> oo(4);
    assert(!build_landmarks_bounded(spec, nullptr, nullptr, scratch.data(), scratch.size(),
        arena.data(), arena.size(), oo.data(), oo.size(), &nn, nullptr));
    assert(!landmark_build_workspace_bytes(spec, nullptr, ws, ar, nn, nullptr));
    landmark_build_spec bad = spec;
    bad.n_rows = 0;
    assert(!build_landmarks_bounded(bad, nullptr, landmark_encode_f32_row, scratch.data(),
        scratch.size(), arena.data(), arena.size(), oo.data(), oo.size(), &nn, nullptr));
    bad = spec;
    bad.chunk_tokens = 0;
    assert(!build_landmarks_bounded(bad, nullptr, landmark_encode_f32_row, scratch.data(),
        scratch.size(), arena.data(), arena.size(), oo.data(), oo.size(), &nn, nullptr));
    std::vector<uint32_t> bad_rows = srows;
    bad_rows[3] = 999;
    bad = spec;
    bad.surviving_rows = bad_rows.data();
    assert(!build_landmarks_bounded(bad, nullptr, landmark_encode_f32_row, scratch.data(),
        scratch.size(), arena.data(), arena.size(), oo.data(), oo.size(), &nn, nullptr));
    landmark_build_layer gap[2] = {{0, 0, 32, 0}, {0, 40, 24, 0}};
    bad = spec;
    bad.layers = gap;
    assert(!build_landmarks_bounded(bad, nullptr, landmark_encode_f32_row, scratch.data(),
        scratch.size(), arena.data(), arena.size(), oo.data(), oo.size(), &nn, nullptr));
    landmark_build_layer over[1] = {{0, 48, 32, 0}};
    bad = spec;
    bad.layers = over;
    bad.n_layers = 1;
    assert(!build_landmarks_bounded(bad, nullptr, landmark_encode_f32_row, scratch.data(),
        scratch.size(), arena.data(), arena.size(), oo.data(), oo.size(), &nn, nullptr));
    std::cout << "[test_bounded_builder] passed!" << std::endl;
}

// Minimal scratch threshold by downward search: proves the capacity gate is exact.
static size_t test_min_scratch(const sr_query & q,
    const std::vector<landmark_fragment_view> & views, const sr_selection_config & cfg,
    const xkv_segment * seg, uint32_t off, uint32_t dim, size_t hi) {
    std::vector<segment_row_ref> rows(4096);
    std::vector<float> scores(views.size() + 8);
    auto attempt = [&](size_t bytes) {
        std::vector<uint8_t> sc(bytes, 0);
        size_t nr = 0, ns = 0;
        uint32_t rf = 0;
        bool ch = false;
        return select_sr_query_bounded(q, views.data(), views.size(), cfg, seg, off, dim,
            nullptr, 0, sc.data(), sc.size(), rows.data(), rows.size(), &nr, scores.data(),
            scores.size(), &ns, &rf, &ch, nullptr);
    };
    assert(attempt(hi));
    size_t lo = 0;
    while (lo + 1 < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (attempt(mid)) hi = mid;
        else lo = mid;
    }
    assert(!attempt(lo));
    return hi;
}
static void test_workspace_bytes_and_short_refusal() {
    std::cout << "[test_workspace_bytes_and_short_refusal] starting..." << std::endl;
    auto seg = create_test_segment(16, 16, 64, 1);
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 16; ++i) {
        rows.push_back({i, 1000 + i, 1, 0, (int64_t) i, (int64_t) i, 0, true, true});
    }
    auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x1, -1);
    assert(frags.size() == 4);
    for (auto & f : frags) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
    auto views = test_views(frags);
    sr_query q;
    q.head_dim = 32;
    q.q_vec.resize(32, 0.2f);
    q.scale = 1.0f;
    sr_selection_config cfg;
    cfg.sr_budget = 2;
    uint32_t n = (uint32_t) views.size(), rmax = test_rmax(frags);
    uint32_t p = test_padded_rank(*seg);
    std::vector<uint8_t> scratch = test_make_scratch(n, 1, 32, rmax, p);
    auto ref = select_sr_query(q, frags, cfg);
    auto out = run_bounded_single(q, views, cfg, nullptr, 0, 32, nullptr, 0, scratch);
    assert(out.rows == ref.selected_rows);
    assert(out.scores.size() == ref.scores.size());
    for (size_t i = 0; i < out.scores.size(); ++i) assert(out.scores[i] == ref.scores[i]);
    // Exact gate: minimal succeeding size works, one byte less refuses.
    size_t floor = test_min_scratch(q, views, cfg, nullptr, 0, 32, scratch.size());
    assert(floor <= scratch.size());
    assert(floor > 0);
    // Overflow-safe calculators reject degenerate input.
    size_t dummy = 0;
    assert(!landmark_select_scratch_bytes(4, 0, 32, 4, 16, dummy, nullptr));
    assert(!landmark_select_scratch_bytes(4, 1, 0, 4, 16, dummy, nullptr));
    assert(!landmark_encode_scratch_bytes(0, 16, 4, dummy, nullptr));
    landmark_workspace_config wcfg;
    assert(!landmark_workspace_required_bytes(wcfg, dummy, nullptr)); // all-zero maxima
    wcfg.max_fragments = 4;
    wcfg.max_rows_per_fragment = 4;
    wcfg.max_rows_total = 16;
    wcfg.max_queries = 2;
    wcfg.max_q_heads_per_query = 2;
    wcfg.max_head_dim = 32;
    wcfg.max_protected_rows = 8;
    wcfg.max_padded_rank = p;
    size_t wneed = 0;
    assert(landmark_workspace_required_bytes(wcfg, wneed, nullptr));
    landmark_workspace ws;
    assert(ws.init(wcfg, nullptr));
    assert(ws.valid());
    assert(ws.capacity_bytes() >= wneed);
    assert(reinterpret_cast<uintptr_t>(ws.data()) % 64 == 0);
    std::cout << "[test_workspace_bytes_and_short_refusal] passed!" << std::endl;
}
static void test_planner_parity() {
    std::cout << "[test_planner_parity] starting..." << std::endl;
    auto seg = create_test_segment(16, 16, 64, 7);
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 3;
    stamp.content_epoch = 5;
    stamp.binding_epoch = 9;
    stamp.view.topology_epoch = 2;
    std::vector<row_meta> rows = {
        {0, 1000, 1, 0, 10, 10, 0, 1, true, true},
        {1, 1001, 1, 0, 11, 11, 0, 1, true, true},
        {2, 1002, 2, 0, 12, 12, 0, 1, true, true},
        {3, 1003, 1, 0, 13, 13, 1, 1, true, true},
        {4, 1004, 1, 0, 14, 14, 1, 1, true, true},
        {5, 1005, 1, 0, 15, 20, 1, 1, true, true},
        {6, 1006, 1, 0, 16, 21, 1, 1, true, true},
        {9, 1009, 1, 0, 21, 26, 1, 1, true, false},
        {10, 1010, 1, 0, 22, 27, 1, 1, true, true},
    };
    // row 9 is reader-invisible but live in segment; rows vector holds 16 live rows
    auto ref = build_legal_fragments(*seg, stamp, 7, 1, 0, 0, rows, 4, 0xAB, -1);
    std::vector<landmark_fragment_plan> plans(rows.size());
    std::vector<uint32_t> o_idx(rows.size());
    std::vector<int64_t> o_pos(rows.size());
    std::vector<uint64_t> o_pid(rows.size()), o_gen(rows.size());
    size_t n_plans = 0, n_rows = 0;
    assert(build_legal_fragment_plans(*seg, stamp, 7, 1, 0, 0, rows.data(), rows.size(), 4,
        0xAB, -1, plans.data(), plans.size(), &n_plans, o_idx.data(), o_pos.data(),
        o_pid.data(), o_gen.data(), o_idx.size(), &n_rows, nullptr));
    assert(n_plans == ref.size());
    for (size_t k = 0; k < n_plans; ++k) {
        assert(plans[k].key == ref[k].key);
        assert(plans[k].row_begin == ref[k].row_begin);
        assert(plans[k].row_count == ref[k].row_count);
        for (uint32_t r = 0; r < plans[k].row_count; ++r) {
            assert(o_idx[plans[k].rows_offset + r] == ref[k].row_indices[r]);
            assert(o_pos[plans[k].rows_offset + r] == ref[k].storage_positions[r]);
            assert(o_pid[plans[k].rows_offset + r] == ref[k].payload_ids[r]);
            assert(o_gen[plans[k].rows_offset + r] == ref[k].generations[r]);
        }
    }
    // Capacity shortfall fails closed.
    size_t np = 0, nr = 0;
    assert(!build_legal_fragment_plans(*seg, stamp, 7, 1, 0, 0, rows.data(), rows.size(),
        4, 0xAB, -1, plans.data(), n_plans > 0 ? n_plans - 1 : 0, &np, o_idx.data(),
        o_pos.data(), o_pid.data(), o_gen.data(), o_idx.size(), &nr, nullptr));
    std::cout << "[test_planner_parity] passed!" << std::endl;
}

// Helper to create a dummy test segment with synthetic encoded A_K and B_K
static void test_seed_source_fidelity() {
    std::cout << "[test_seed_source_fidelity] starting..." << std::endl;
    auto segA = create_test_segment(8, 16, 64, 1);
    auto segB = create_test_segment(8, 16, 64, 1);
    segB->segment_id = 200; // different id, identical contentFingerprints
    segA->source_fingerprint = 0xBEEF;
    segB->source_fingerprint = 0xBEEF;
    segA->profile_fingerprint = 7;
    segB->profile_fingerprint = 7;
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 8; ++i) {
        rows.push_back({i, 1000 + i, 1, 0, (int64_t) i, (int64_t) i, 0, true, true});
    }
    // different storage positions must NOT change the code stream
    std::vector<row_meta> rowsB = rows;
    for (auto & r : rowsB) r.storage_pos += 500;
    for (auto & r : rowsB) r.virtual_pos += 500;
    for (ggml_type lt : {GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0}) {
        auto fA = build_legal_fragments(*segA, stamp, 1, 1, 0, 0, rows, 8, 0x5, -1);
        auto fB = build_legal_fragments(*segB, stamp, 1, 1, 0, 0, rowsB, 8, 0x5, -1);
        assert(fA.size() == 1 && fB.size() == 1);
        encode_fragment_landmark(fA[0], *segA, 0, 32, nullptr, 0x5, lt);
        encode_fragment_landmark(fB[0], *segB, 0, 32, nullptr, 0x5, lt);
        assert(fA[0].landmark_matrix.bytes == fB[0].landmark_matrix.bytes);
        assert(fA[0].source_fingerprint == fB[0].source_fingerprint);
        assert(fA[0].key.landmark_codec_fp == fB[0].key.landmark_codec_fp);
        assert(fA[0].key.source_fingerprint == 0xBEEF);
    }
    // Q8/Turbo4 landmark recall against the FP32 oracle stays exact on top-k
    auto seg = create_test_segment(16, 16, 64, 1);
    auto fp = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x5, -1);
    auto q8 = fp;
    auto t4 = fp;
    for (auto & f : fp) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x5, GGML_TYPE_F32);
    for (auto & f : q8) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x5, GGML_TYPE_Q8_0);
    for (auto & f : t4) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x5, GGML_TYPE_TURBO4_0);
    sr_query q;
    q.head_dim = 32;
    q.q_vec.assign(32, 0.2f);
    sr_selection_config cfg;
    cfg.sr_budget = 2;
    auto rfp = select_sr_query(q, fp, cfg);
    auto vq8 = test_views(q8);
    auto vt4 = test_views(t4);
    std::vector<uint8_t> sc =
        test_make_scratch(4, 1, 32, 4, test_padded_rank(*seg));
    auto rq8 = run_bounded_single(q, vq8, cfg, nullptr, 0, 32, nullptr, 0, sc);
    auto rt4 = run_bounded_single(q, vt4, cfg, nullptr, 0, 32, nullptr, 0, sc);
    assert(rq8.rows == rfp.selected_rows);
    assert(rt4.rows == rfp.selected_rows);
    std::cout << "[test_seed_source_fidelity] passed!" << std::endl;
}
static void test_refine_bounded_parity() {
    std::cout << "[test_refine_bounded_parity] starting..." << std::endl;
    auto seg = create_test_segment(16, 16, 64, 1);
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 16; ++i) {
        rows.push_back({i, 1000 + i, 1, 0, (int64_t) i, (int64_t) i, 0, true, true});
    }
    auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x1, -1);
    for (auto & f : frags) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
    auto views = test_views(frags);
    sr_query q;
    q.head_dim = 32;
    q.q_vec.assign(32, 0.2f);
    q.scale = 1.0f / std::sqrt(32.0f);
    std::vector<uint8_t> scratch =
        test_make_scratch(4, 1, 32, 4, test_padded_rank(*seg));
    for (uint32_t cap : {64u, 0u, 3u}) {
        sr_selection_config cfg;
        cfg.sr_budget = 2;
        cfg.refine_mode = LLAMA_XKV_LANDMARK_REFINE_BOUNDARY;
        cfg.refine_max_rows = cap;
        auto ref = select_sr_query(q, frags, cfg, seg.get(), 0, 32, nullptr, 0x1);
        auto out = run_bounded_single(q, views, cfg, seg.get(), 0, 32, nullptr, 0x1, scratch);
        assert(out.rows == ref.selected_rows);
        assert(out.scores.size() == ref.scores.size());
        for (size_t i = 0; i < out.scores.size(); ++i) assert(out.scores[i] == ref.scores[i]);
        assert(out.refined == ref.rows_refined);
        assert(out.cap == ref.refine_cap_hit);
    }
    std::cout << "[test_refine_bounded_parity] passed!" << std::endl;
}
static void test_budget_all_equals_dense() {
    std::cout << "[test_budget_all_equals_dense] starting..." << std::endl;
    auto seg = create_test_segment(12, 16, 64, 1);
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 12; ++i) {
        rows.push_back({i, 1000 + i, 5 + i, 0, (int64_t) i, (int64_t) i, 0, true, true});
    }
    auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x1, -1);
    for (auto & f : frags) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
    auto views = test_views(frags);
    sr_query q;
    q.head_dim = 32;
    q.q_vec.assign(32, -0.7f);
    sr_selection_config cfg;
    cfg.sr_budget = 99; // >= fragments: every legal row must be selected (dense)
    auto ref = select_sr_query(q, frags, cfg);
    assert(ref.selected_rows.size() == 12);
    for (uint32_t i = 0; i < 12; ++i) {
        segment_row_ref want{100, 1, (uint64_t) (5 + i), i};
        assert(std::binary_search(ref.selected_rows.begin(), ref.selected_rows.end(), want));
    }
    std::vector<uint8_t> scratch =
        test_make_scratch(3, 1, 32, 4, test_padded_rank(*seg));
    auto out = run_bounded_single(q, views, cfg, nullptr, 0, 32, nullptr, 0, scratch);
    assert(out.rows == ref.selected_rows);
    std::cout << "[test_budget_all_equals_dense] passed!" << std::endl;
}
static void test_multisegment_attribution() {
    std::cout << "[test_multisegment_attribution] starting..." << std::endl;
    auto segA = create_test_segment(8, 16, 64, 1);
    auto segB = create_test_segment(8, 16, 64, 1);
    segB->segment_id = 200;
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    std::vector<row_meta> rowsA, rowsB;
    for (uint32_t i = 0; i < 8; ++i) {
        rowsA.push_back({i, 1000 + i, 1, 0, (int64_t) i, (int64_t) i, 0, true, true});
        rowsB.push_back({i, 1000 + i, 1, 0, (int64_t) (100 + i), (int64_t) (100 + i), 0, true, true});
    }
    auto fA = build_legal_fragments(*segA, stamp, 1, 1, 0, 0, rowsA, 4, 0x1, -1);
    auto fB = build_legal_fragments(*segB, stamp, 1, 1, 0, 0, rowsB, 4, 0x1, -1);
    assert(fA.size() == 2 && fB.size() == 2);
    for (auto & f : fA) encode_fragment_landmark(f, *segA, 0, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
    for (auto & f : fB) encode_fragment_landmark(f, *segB, 0, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
    std::vector<legal_fragment> all = fA;
    all.insert(all.end(), fB.begin(), fB.end());
    auto views = test_views(all);
    const xkv_segment * table[2] = {segA.get(), segB.get()};
    uint32_t p = test_padded_rank(*segA);
    std::vector<uint8_t> scratch = test_make_scratch(4, 1, 32, 4, p);
    // global budget 1 with tied zero query: first fragment (segA frag0) wins
    sr_query qz;
    qz.head_dim = 32;
    qz.q_vec.assign(32, 0.0f);
    sr_selection_config one;
    one.sr_budget = 1;
    std::vector<uint32_t> fidx(64, 0);
    std::vector<landmark_fragment_key> fkey(64);
    std::vector<uint64_t> sids(4, 0), svers(4, 0);
    std::vector<uint32_t> sptrs(5, 0), sidx(64, 0);
    size_t nsegs = 0, nsidx = 0;
    sr_bounded_attribution at;
    at.row_frag_index = fidx.data();
    at.row_frag_key = fkey.data();
    at.rows_cap = 64;
    at.seg_ids = sids.data();
    at.seg_versions = svers.data();
    at.segs_cap = 4;
    at.out_n_segs = &nsegs;
    at.seg_ptrs = sptrs.data();
    at.seg_indices = sidx.data();
    at.seg_idx_cap = 64;
    at.out_n_seg_idx = &nsidx;
    bounded_single_out o;
    {
        o.rows.resize(64);
        o.scores.resize(8);
        size_t nr = 0, ns = 0;
        bool ok = select_sr_query_bounded(qz, views.data(), views.size(), one, nullptr, 0,
            32, nullptr, 0x1, scratch.data(), scratch.size(), o.rows.data(), o.rows.size(),
            &nr, o.scores.data(), o.scores.size(), &ns, &o.refined, &o.cap, nullptr, table,
            2, &at);
        assert(ok);
        o.rows.resize(nr);
        o.scores.resize(ns);
    }
    assert(o.rows.size() == 4);
    for (const auto & r : o.rows) assert(r.segment_id == 100);
    for (size_t k = 0; k < o.rows.size(); ++k) assert(fidx[k] == 0);
    assert(nsegs == 1 && sids[0] == 100);
    assert(sptrs[0] == 0 && sptrs[1] == 4);
    // budget all: both segments, seg CSR reconstructs per-segment rows
    sr_selection_config cfg_all;
    cfg_all.sr_budget = 0;
    nsegs = 0;
    nsidx = 0;
    {
        o.rows.resize(64);
        o.scores.resize(8);
        size_t nr = 0, ns = 0;
        bool ok = select_sr_query_bounded(qz, views.data(), views.size(), cfg_all, nullptr, 0,
            32, nullptr, 0x1, scratch.data(), scratch.size(), o.rows.data(), o.rows.size(),
            &nr, o.scores.data(), o.scores.size(), &ns, &o.refined, &o.cap, nullptr, table,
            2, &at);
        assert(ok);
        o.rows.resize(nr);
    }
    assert(o.rows.size() == 16);
    assert(nsegs == 2 && sids[0] == 100 && sids[1] == 200);
    assert(sptrs[0] == 0 && sptrs[1] == 8 && sptrs[2] == 16);
    for (uint32_t k = sptrs[0]; k < sptrs[1]; ++k) assert(o.rows[sidx[k]].segment_id == 100);
    for (uint32_t k = sptrs[1]; k < sptrs[2]; ++k) assert(o.rows[sidx[k]].segment_id == 200);
    // src-index attribution: segA rows from frags 0,1; segB rows from frags 2,3
    for (size_t k = 0; k < o.rows.size(); ++k) {
        if (o.rows[k].segment_id == 100) assert(fidx[k] <= 1);
        else assert(fidx[k] >= 2);
        assert(fkey[k].segment_id == o.rows[k].segment_id);
    }
    // partial across segments via table: cutoff splits segB frag1 (pos 104..107)
    sr_query qc;
    qc.head_dim = 32;
    qc.causal_limit_pos = 105;
    qc.q_vec.assign(32, 0.3f);
    nsegs = 0;
    nsidx = 0;
    {
        o.rows.resize(64);
        o.scores.resize(8);
        size_t nr = 0, ns = 0;
        bool ok = select_sr_query_bounded(qc, views.data(), views.size(), cfg_all, nullptr, 0,
            32, nullptr, 0x1, scratch.data(), scratch.size(), o.rows.data(), o.rows.size(),
            &nr, o.scores.data(), o.scores.size(), &ns, &o.refined, &o.cap, nullptr, table,
            2, &at);
        assert(ok);
        o.rows.resize(nr);
    }
    for (const auto & r : o.rows) {
        if (r.segment_id == 200) assert(r.row <= 5); // rows 6,7 (pos 106,107) excluded
    }
    bool saw_partial_key = false;
    for (size_t k = 0; k < o.rows.size(); ++k) {
        if (o.rows[k].segment_id == 200 && o.rows[k].row >= 4) {
            assert(fkey[k].causal_cutoff == 105); // effective partial key
            saw_partial_key = true;
        }
        if (o.rows[k].segment_id == 100) assert(fkey[k].causal_cutoff == -1);
    }
    assert(saw_partial_key);
    // unresolvable owner fails closed: table without segB + partial in segB
    {
        const xkv_segment * partial_table[1] = {segA.get()};
        o.rows.resize(64);
        size_t nr = 0, ns = 0;
        o.scores.resize(8);
        assert(!select_sr_query_bounded(qc, views.data(), views.size(), cfg_all, nullptr, 0,
            32, nullptr, 0x1, scratch.data(), scratch.size(), o.rows.data(), o.rows.size(),
            &nr, o.scores.data(), o.scores.size(), &ns, &o.refined, &o.cap, nullptr,
            partial_table, 1, nullptr));
    }
    // table miss without rebuild need still succeeds
    {
        const xkv_segment * partial_table[1] = {segA.get()};
        sr_query qfull;
        qfull.head_dim = 32;
        qfull.q_vec.assign(32, 0.3f);
        o.rows.resize(64);
        size_t nr = 0, ns = 0;
        o.scores.resize(8);
        assert(select_sr_query_bounded(qfull, views.data(), views.size(), cfg_all, nullptr, 0,
            32, nullptr, 0x1, scratch.data(), scratch.size(), o.rows.data(), o.rows.size(),
            &nr, o.scores.data(), o.scores.size(), &ns, &o.refined, &o.cap, nullptr,
            partial_table, 1, nullptr));
        assert(nr == 16);
    }
    std::cout << "[test_multisegment_attribution] passed!" << std::endl;
}
static void test_head_dim_mismatch_guards() {
    std::cout << "[test_head_dim_mismatch_guards] starting..." << std::endl;
    auto seg = create_test_segment(8, 16, 64, 1);
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 8; ++i) {
        rows.push_back({i, 1000 + i, 1, 0, (int64_t) i, (int64_t) i, 0, true, true});
    }
    auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x1, -1);
    for (auto & f : frags) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
    auto views = test_views(frags);
    sr_selection_config cfg;
    cfg.sr_budget = 1;
    cfg.refine_mode = LLAMA_XKV_LANDMARK_REFINE_BOUNDARY;
    sr_query q;
    q.head_dim = 16; // != feature_dim 32 with segment set: must fail closed
    q.q_vec.assign(16, 0.5f);
    bool threw = false;
    try {
        select_sr_query(q, frags, cfg, seg.get(), 0, 32, nullptr, 0x1);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
    std::vector<uint8_t> scratch = test_make_scratch(2, 1, 16, 4, test_padded_rank(*seg));
    std::vector<segment_row_ref> br(32);
    std::vector<float> bs(4);
    size_t nr = 0, ns = 0;
    uint32_t rf = 0;
    bool ch = false;
    assert(!select_sr_query_bounded(q, views.data(), views.size(), cfg, seg.get(), 0, 32,
        nullptr, 0x1, scratch.data(), scratch.size(), br.data(), br.size(), &nr, bs.data(),
        bs.size(), &ns, &rf, &ch, nullptr));
    std::cout << "[test_head_dim_mismatch_guards] passed!" << std::endl;
}
static void test_future_draft_isolation_bounded() {
    std::cout << "[test_future_draft_isolation_bounded] starting..." << std::endl;
    auto seg = create_test_segment(16, 16, 64, 42);
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 16; ++i) {
        rows.push_back({i, 1000 + i, 100 + i, 0, (int64_t) i, (int64_t) i, 0, true, true});
    }
    auto frags = build_legal_fragments(*seg, stamp, 42, 100, 0, 0, rows, 4, 0x42, -1);
    for (auto & f : frags) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x42, GGML_TYPE_Q8_0);
    auto views = test_views(frags);
    auto phase_tx = [](const float * src, int64_t, uint32_t, float * dst) {
        for (int i = 0; i < 32; ++i) dst[i] = src[i];
    };
    sr_query q0, q1;
    q0.query_index = 0;
    q0.causal_limit_pos = 5;
    q0.head_dim = 32;
    q0.q_vec.assign(32, 1.0f);
    q1.query_index = 1;
    q1.causal_limit_pos = 15;
    q1.head_dim = 32;
    q1.q_vec.assign(32, 1000.0f);
    sr_selection_config cfg;
    cfg.sr_budget = 2;
    // reference batch first
    auto bref = select_sr_batch({q0, q1}, frags, cfg, seg.get(), 0, 32, phase_tx, 0x42);
    for (const auto & r : bref.per_query[0].selected_rows) assert(r.row <= 5);
    // bounded batch with caller CSR must match exactly
    uint32_t p = test_padded_rank(*seg);
    landmark_workspace_config wcfg;
    wcfg.max_fragments = 4;
    wcfg.max_rows_per_fragment = 4;
    wcfg.max_rows_total = 16;
    wcfg.max_queries = 2;
    wcfg.max_q_heads_per_query = 1;
    wcfg.max_head_dim = 32;
    wcfg.max_protected_rows = 0;
    wcfg.max_padded_rank = p;
    landmark_workspace ws;
    assert(ws.init(wcfg, nullptr));
    std::vector<sr_query> qs = {q0, q1};
    std::vector<segment_row_ref> gather(64);
    std::vector<uint32_t> ptrs(3, 0), cidx(64, 0);
    size_t ng = 0, nc = 0;
    uint32_t tr = 0, tc = 0;
    std::string err;
    bool bok = select_sr_batch_bounded(qs.data(), qs.size(), views.data(), views.size(), cfg,
        seg.get(), 0, 32, phase_tx, 0x42, ws.data(), ws.capacity_bytes(), gather.data(),
        gather.size(), &ng, ptrs.data(), ptrs.size(), cidx.data(), cidx.size(), &nc, &tr,
        &tc, &err);
    if (!bok) std::cerr << "batch bounded failed: " << err << std::endl;
    assert(bok);
    gather.resize(ng);
    assert(gather == bref.gather_rows);
    assert(ptrs == bref.csr_ptrs);
    assert(cidx.size() >= nc);
    for (size_t i = 0; i < nc; ++i) assert(cidx[i] == bref.csr_indices[i]);
    // per-query CSR slices reconstruct reference rows
    for (size_t qi = 0; qi < 2; ++qi) {
        std::vector<segment_row_ref> rec;
        for (uint32_t k = ptrs[qi]; k < ptrs[qi + 1]; ++k) rec.push_back(gather[cidx[k]]);
        assert(rec == bref.per_query[qi].selected_rows);
    }
    // one-byte-short batch scratch refuses
    assert(!select_sr_batch_bounded(qs.data(), qs.size(), views.data(), views.size(), cfg,
        seg.get(), 0, 32, phase_tx, 0x42, ws.data(), 0, gather.data(), 64, &ng, ptrs.data(),
        ptrs.size(), cidx.data(), cidx.size(), &nc, &tr, &tc, nullptr));
    std::cout << "[test_future_draft_isolation_bounded] passed!" << std::endl;
}
static void test_shared_prefix_csr() {
    std::cout << "[test_shared_prefix_csr] starting..." << std::endl;
    auto seg = create_test_segment(12, 16, 64, 1);
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 12; ++i) {
        rows.push_back({i, 1000 + i, 1, 0, (int64_t) i, (int64_t) i, 0, true, true});
    }
    auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x1, -1);
    assert(frags.size() == 3);
    for (auto & f : frags) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x1, GGML_TYPE_F32);
    auto views = test_views(frags);
    // shared prefix rows 0..7 visible to both; qA stops at 7, qB extends to 11
    sr_query qA, qB;
    qA.head_dim = 32;
    qA.causal_limit_pos = 7;
    qA.q_vec.assign(32, 0.5f);
    qB.head_dim = 32;
    qB.causal_limit_pos = 11;
    qB.q_vec.assign(32, 0.5f);
    sr_selection_config cfg;
    cfg.sr_budget = 4;
    std::vector<uint8_t> scratch =
        test_make_scratch(3, 1, 32, 4, test_padded_rank(*seg));
    auto a = run_bounded_single(qA, views, cfg, seg.get(), 0, 32, nullptr, 0x1, scratch);
    auto b = run_bounded_single(qB, views, cfg, seg.get(), 0, 32, nullptr, 0x1, scratch);
    for (const auto & r : a.rows) assert((int64_t) r.row <= 7);
    assert(a.rows.size() == 8); // frags 0,1 fully legal
    assert(b.rows.size() == 12);
    // shared prefix rows appear in both, suffix only in qB
    for (const auto & r : a.rows) assert(std::binary_search(b.rows.begin(), b.rows.end(), r));
    std::cout << "[test_shared_prefix_csr] passed!" << std::endl;
}
static void test_nan_zero_query() {
    std::cout << "[test_nan_zero_query] starting..." << std::endl;
    auto seg = create_test_segment(8, 16, 64, 1);
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 8; ++i) {
        rows.push_back({i, 1000 + i, 1, 0, (int64_t) i, (int64_t) i, 0, true, true});
    }
    auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 2, 0x1, -1);
    for (auto & f : frags) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
    auto views = test_views(frags);
    std::vector<uint8_t> scratch =
        test_make_scratch((uint32_t) views.size(), 1, 32, 2, test_padded_rank(*seg));
    sr_selection_config cfg;
    cfg.sr_budget = 2;
    // NaN query fails closed in both paths
    sr_query qn;
    qn.head_dim = 32;
    qn.q_vec.assign(32, std::numeric_limits<float>::quiet_NaN());
    bool threw = false;
    try {
        select_sr_query(qn, frags, cfg);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
    std::vector<segment_row_ref> br(32);
    std::vector<float> bs(8);
    size_t nr = 0, ns = 0;
    uint32_t rf = 0;
    bool ch = false;
    assert(!select_sr_query_bounded(qn, views.data(), views.size(), cfg, nullptr, 0, 32,
        nullptr, 0, scratch.data(), scratch.size(), br.data(), br.size(), &nr, bs.data(),
        bs.size(), &ns, &rf, &ch, nullptr));
    // infinite scale fails closed too
    sr_query qi = qn;
    qi.q_vec.assign(32, 1.0f);
    qi.scale = std::numeric_limits<float>::infinity();
    threw = false;
    try {
        select_sr_query(qi, frags, cfg);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
    // zero query: exact tie, stable src-identity order, deterministic
    sr_query qz;
    qz.head_dim = 32;
    qz.q_vec.assign(32, 0.0f);
    qz.scale = 1.0f;
    auto ref = select_sr_query(qz, frags, cfg, seg.get(), 0, 32, nullptr, 0x1);
    for (float s : ref.scores) assert(s == 0.0f);
    auto out = run_bounded_single(qz, views, cfg, nullptr, 0, 32, nullptr, 0, scratch);
    assert(out.rows == ref.selected_rows);
    assert(out.scores == ref.scores);
    // rows belong to the first two fragments in src order (frag 0: rows 0,1; frag 1: rows 2,3)
    assert(out.rows.size() == 4);
    assert(out.rows[0].row == 0 && out.rows[3].row == 3);
    std::cout << "[test_nan_zero_query] passed!" << std::endl;
}
static void test_phase_stamp_invalidation() {
    std::cout << "[test_phase_stamp_invalidation] starting..." << std::endl;
    landmark_table table(1024 * 1024);
    auto seg = create_test_segment(4, 16, 64, 1);
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    stamp.binding_epoch = 2;
    stamp.view.topology_epoch = 3;
    std::vector<row_meta> rows = {{0, 1000, 1, 0, 0, 0, 0, true, true}};
    auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x11, -1);
    assert(frags[0].key.binding_epoch == 2);
    assert(frags[0].key.view_topology_epoch == 3);
    encode_fragment_landmark(frags[0], *seg, 0, 32, nullptr, 0x11, GGML_TYPE_Q8_0);
    auto fp = std::make_shared<const legal_fragment>(std::move(frags[0]));
    assert(table.insert(fp));
    // view change evicts
    xkv_snapshot_stamp moved = stamp;
    moved.view.layout_epoch = 99;
    table.invalidate_stamp(moved);
    assert(table.fragment_count() == 0);
    assert(table.insert(fp));
    // binding change evicts
    table.invalidate_binding_epoch(seg->segment_id, 77);
    assert(table.fragment_count() == 0);
    assert(table.insert(fp));
    // phase/source change evicts
    table.invalidate_phase_source(seg->segment_id, 0x22, fp->key.source_fingerprint);
    assert(table.fragment_count() == 0);
    assert(table.insert(fp));
    table.invalidate_phase_source(seg->segment_id, 0x11, 0xDEAD);
    assert(table.fragment_count() == 0);
    assert(table.insert(fp));
    // can_cache rejects stale stamp / oversize
    std::string err;
    assert(table.can_cache(*fp, stamp, 0x11, &err));
    assert(!table.can_cache(*fp, moved, 0x11, nullptr));
    assert(!table.can_cache(*fp, stamp, 0x22, nullptr));
    landmark_table tiny(1);
    assert(!tiny.can_cache(*fp, stamp, 0x11, nullptr));
    assert(!tiny.insert(fp));
    std::cout << "[test_phase_stamp_invalidation] passed!" << std::endl;
}
static void test_lru_pinned_atomic() {
    std::cout << "[test_lru_pinned_atomic] starting..." << std::endl;
    auto mk = [](uint64_t seg, size_t bytes) {
        auto f = std::make_shared<legal_fragment>();
        f->key.segment_id = seg;
        f->row_indices = {0};
        f->landmark_matrix.bytes.resize(bytes, 0);
        return f;
    };
    size_t entry = mk(1, 100)->allocated_bytes();
    landmark_table tbl(entry + 50); // room for exactly one entry
    auto f1 = mk(1, 100);
    auto k1 = f1->key;
    assert(tbl.insert(f1));
    f1.reset();
    auto held = tbl.find(k1); // sole external lease
    assert(held != nullptr);
    size_t before_bytes = tbl.total_allocated_bytes();
    auto f2 = mk(2, 100);
    auto k2 = f2->key;
    assert(!tbl.insert(f2)); // pinned: refuse with zero mutation
    assert(tbl.fragment_count() == 1);
    assert(tbl.total_allocated_bytes() == before_bytes);
    assert(tbl.find(k1) != nullptr);
    assert(tbl.find(k2) == nullptr);
    held.reset();
    assert(tbl.insert(f2)); // lease released: evict + insert atomically
    f2.reset();
    assert(tbl.fragment_count() == 1);
    assert(tbl.find(k1) == nullptr);
    assert(tbl.find(k2) != nullptr);
    // in-place replace preflights: same key, bigger but fitting after replace
    auto f3 = mk(2, 100);
    assert(tbl.insert(f3));
    assert(tbl.fragment_count() == 1);
    std::cout << "[test_lru_pinned_atomic] passed!" << std::endl;
}
static void test_bounded_parity_all_types() {
    std::cout << "[test_bounded_parity_all_types] starting..." << std::endl;
    for (ggml_type lt : {GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0, GGML_TYPE_F32}) {
        auto seg = create_test_segment(16, 16, 64, 1);
        xkv_snapshot_stamp stamp;
        stamp.live_epoch = 1;
        std::vector<row_meta> rows;
        for (uint32_t i = 0; i < 16; ++i) {
            rows.push_back({i, 1000 + i, 7 + i, 0, (int64_t) i, (int64_t) i, 0, true, true});
        }
        auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x1, -1);
        assert(frags.size() == 4);
        for (auto & f : frags) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x1, lt);
        auto views = test_views(frags);
        // GQA query with protected rows (incl. duplicate + distinct generations)
        sr_query q;
        q.head_dim = 32;
        q.q_head_indices = {0, 1, 2};
        q.q_vec.resize(96);
        for (size_t i = 0; i < 96; ++i) q.q_vec[i] = 0.05f * (float) ((int) (i % 11) - 5);
        q.scale = 0.11f;
        sr_selection_config cfg;
        cfg.sr_budget = 2;
        cfg.hot_rows = {{100, 1, 7, 0}, {100, 1, 9, 2}, {100, 1, 7, 0}};
        cfg.recent_rows = {{100, 1, 21, 14}};
        cfg.outlier_rows = {{100, 1, 22, 15}};
        auto ref = select_sr_query(q, frags, cfg);
        std::vector<uint8_t> scratch =
            test_make_scratch((uint32_t) views.size(), 3, 32, test_rmax(frags), test_padded_rank(*seg));
        auto out = run_bounded_single(q, views, cfg, nullptr, 0, 32, nullptr, 0, scratch);
        assert(out.rows == ref.selected_rows);
        assert(out.scores.size() == ref.scores.size());
        for (size_t i = 0; i < out.scores.size(); ++i) assert(out.scores[i] == ref.scores[i]);
        assert(out.refined == ref.rows_refined);
        assert(out.cap == ref.refine_cap_hit);
        // budget 0 means all; budget overrun clamps
        sr_selection_config all = cfg;
        all.sr_budget = 0;
        assert(select_sr_query(q, frags, all).selected_rows ==
               run_bounded_single(q, views, all, nullptr, 0, 32, nullptr, 0, scratch).rows);
        all.sr_budget = 99;
        assert(select_sr_query(q, frags, all).selected_rows ==
               run_bounded_single(q, views, all, nullptr, 0, 32, nullptr, 0, scratch).rows);
    }
    std::cout << "[test_bounded_parity_all_types] passed!" << std::endl;
}
static void test_zero_heap_production() {
    std::cout << "[test_zero_heap_production] starting..." << std::endl;
    auto seg = create_test_segment(16, 16, 64, 1);
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 16; ++i) {
        rows.push_back({i, 1000 + i, 1, 0, (int64_t) i, (int64_t) i, 0, true, true});
    }
    auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x1, -1);
    for (ggml_type lt : {GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0, GGML_TYPE_F32}) {
        for (auto & f : frags) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x1, lt);
        auto views = test_views(frags);
        sr_query q;
        q.head_dim = 32;
        q.q_vec.resize(32, 0.3f);
        q.scale = 0.2f;
        sr_selection_config cfg;
        cfg.sr_budget = 3;
        cfg.refine_mode = LLAMA_XKV_LANDMARK_REFINE_BOUNDARY;
        cfg.refine_max_rows = 64;
        uint32_t p = test_padded_rank(*seg);
        landmark_workspace_config wcfg;
        wcfg.max_fragments = (uint32_t) views.size();
        wcfg.max_rows_per_fragment = test_rmax(frags);
        wcfg.max_rows_total = 16;
        wcfg.max_queries = 2;
        wcfg.max_q_heads_per_query = 1;
        wcfg.max_head_dim = 32;
        wcfg.max_protected_rows = 0;
        wcfg.max_padded_rank = p;
        landmark_workspace ws;
        assert(ws.init(wcfg, nullptr));
        // warm up once (codec traits lazy init etc.) without counting
        std::vector<uint8_t> warm(ws.capacity_bytes(), 0);
        run_bounded_single(q, views, cfg, seg.get(), 0, 32, nullptr, 0x1, warm);
        std::vector<uint8_t> scratch(ws.capacity_bytes(), 0);
        std::vector<segment_row_ref> orows(64);
        std::vector<float> oscores(8);
        g_alloc_count = 0;
        {
            alloc_scope on(true);
            for (int rep = 0; rep < 3; ++rep) {
                size_t nr = 0, ns = 0;
                uint32_t rf = 0;
                bool ch = false;
                bool ok = select_sr_query_bounded(q, views.data(), views.size(), cfg,
                    seg.get(), 0, 32, nullptr, 0x1, scratch.data(), scratch.size(),
                    orows.data(), orows.size(), &nr, oscores.data(), oscores.size(), &ns,
                    &rf, &ch, nullptr);
                assert(ok);
            }
        }
        assert(g_alloc_count == 0);
    }
    std::cout << "[test_zero_heap_production] passed!" << std::endl;
}
static void test_partial_causal_bounded() {
    std::cout << "[test_partial_causal_bounded] starting..." << std::endl;
    auto seg = create_test_segment(16, 16, 64, 1);
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 16; ++i) {
        rows.push_back({i, 1000 + i, 1, 0, (int64_t) i, (int64_t) i, 0, true, true});
    }
    // F32 landmarks: bounded partial agrees bit-exactly with re-quantized reference.
    {
        auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x1, -1);
        for (auto & f : frags) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x1, GGML_TYPE_F32);
        auto views = test_views(frags);
        sr_query q;
        q.head_dim = 32;
        q.causal_limit_pos = 9; // splits frag 2 (rows 8..11) into legal 8,9 + future 10,11
        q.q_vec.resize(32, 0.4f);
        sr_selection_config cfg;
        cfg.sr_budget = 4;
        cfg.landmark_type = GGML_TYPE_F32; // reference re-quantizes losslessly: parity
        auto ref = select_sr_query(q, frags, cfg, seg.get(), 0, 32, nullptr, 0x1);
        std::vector<uint8_t> scratch =
            test_make_scratch((uint32_t) views.size(), 1, 32, test_rmax(frags), test_padded_rank(*seg));
        auto out = run_bounded_single(q, views, cfg, seg.get(), 0, 32, nullptr, 0x1, scratch);
        assert(out.rows == ref.selected_rows);
        assert(out.scores.size() == ref.scores.size());
        for (size_t i = 0; i < out.scores.size(); ++i) assert(out.scores[i] == ref.scores[i]);
        for (const auto & r : out.rows) assert((int64_t) r.row <= 9); // future never leaks
    }
    // Q8 landmarks: legality-identical, deterministic, future-free (higher-fidelity mean).
    {
        auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x1, -1);
        for (auto & f : frags) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
        auto views = test_views(frags);
        sr_query q;
        q.head_dim = 32;
        q.causal_limit_pos = 5; // splits frag 1 (rows 4..7)
        q.q_vec.resize(32, -0.25f);
        sr_selection_config cfg;
        cfg.sr_budget = 4;
        std::vector<uint8_t> scratch =
            test_make_scratch((uint32_t) views.size(), 1, 32, test_rmax(frags), test_padded_rank(*seg));
        auto o1 = run_bounded_single(q, views, cfg, seg.get(), 0, 32, nullptr, 0x1, scratch);
        auto o2 = run_bounded_single(q, views, cfg, seg.get(), 0, 32, nullptr, 0x1, scratch);
        assert(o1.rows == o2.rows);
        for (const auto & r : o1.rows) assert((int64_t) r.row <= 5);
        assert(!o1.rows.empty());
        // partial without segment fails closed in both paths
        bool threw = false;
        try {
            select_sr_query(q, frags, cfg, nullptr, 0, 32, nullptr, 0x1);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        assert(threw);
        std::vector<segment_row_ref> br(64);
        std::vector<float> bs(8);
        size_t nr = 0, ns = 0;
        uint32_t rf = 0;
        bool ch = false;
        assert(!select_sr_query_bounded(q, views.data(), views.size(), cfg, nullptr, 0, 32,
            nullptr, 0x1, scratch.data(), scratch.size(), br.data(), br.size(), &nr,
            bs.data(), bs.size(), &ns, &rf, &ch, nullptr));
    }
    std::cout << "[test_partial_causal_bounded] passed!" << std::endl;
}
static void test_variable_group_slices() {
    std::cout << "[test_variable_group_slices] starting..." << std::endl;
    auto seg = create_test_segment(8, 16, 64, 1);
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 8; ++i) {
        rows.push_back({i, 1000 + i, 1, 0, (int64_t) i, (int64_t) i, 0, true, true});
    }
    // layer 0 owns [0,32), layer 1 owns [32,64)
    auto frags0 = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x1, -1);
    auto frags1 = build_legal_fragments(*seg, stamp, 1, 1, 1, 0, rows, 4, 0x1, -1);
    for (auto & f : frags0) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
    for (auto & f : frags1) encode_fragment_landmark(f, *seg, 32, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
    assert(!frags0.empty() && !frags1.empty());
    // cross-layer slice inside total width but outside owning layer must fail
    bool threw = false;
    try {
        encode_fragment_landmark(frags0[0], *seg, 16, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
    } catch (const std::out_of_range &) {
        threw = true;
    }
    assert(threw);
    // unknown owning layer must fail
    auto frags_bad = build_legal_fragments(*seg, stamp, 1, 1, 9, 0, rows, 4, 0x1, -1);
    threw = false;
    try {
        encode_fragment_landmark(frags_bad[0], *seg, 0, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
    std::cout << "[test_variable_group_slices] passed!" << std::endl;
}
static std::shared_ptr<xkv_segment> create_test_segment(uint32_t n_rows, uint32_t rank_k, uint32_t total_dim_k, uint64_t seg_version = 1) {
    auto seg = std::make_shared<xkv_segment>();
    seg->segment_id = 100;
    seg->segment_version = seg_version;
    seg->n_rows = n_rows;
    seg->n_live_rows = n_rows;
    seg->row_payload_ids.resize(n_rows);
    seg->live_rows.resize(n_rows, true);
    for (uint32_t i = 0; i < n_rows; ++i) {
        seg->row_payload_ids[i] = 1000 + i; // strictly unique stable payload_ids
    }

    xkv_factor_group_payload g;
    g.group_index = 0;
    g.owning_layers = {0, 1};
    g.total_dim_k = total_dim_k;
    g.total_dim_v = total_dim_k;
    g.layer_feature_offsets_k = {0, total_dim_k / 2};
    g.layer_feature_dims_k = {total_dim_k / 2, total_dim_k / 2};
    g.layer_feature_offsets_v = {0, total_dim_k / 2};
    g.layer_feature_dims_v = {total_dim_k / 2, total_dim_k / 2};

    // A_K: n_rows x rank_k, token_major F32
    codec_desc desc_a = make_codec_desc(factor_role::a_k, GGML_TYPE_F32, orientation::token_major, {n_rows, rank_k});
    std::vector<float> a_data(n_rows * rank_k);
    for (size_t i = 0; i < a_data.size(); ++i) {
        int v = static_cast<int>(i % 17) - 8; // signed int subtraction
        a_data[i] = static_cast<float>(v) * 0.1f;
    }
    g.a_k = encode_matrix(desc_a, a_data.data(), a_data.size());

    // B_K: total_dim_k x rank_k, feature_major_transposed F32
    codec_desc desc_b = make_codec_desc(factor_role::b_k, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim_k, rank_k});
    std::vector<float> b_data(total_dim_k * rank_k);
    for (size_t i = 0; i < b_data.size(); ++i) {
        int v = static_cast<int>(i % 13) - 6;
        b_data[i] = static_cast<float>(v) * 0.1f;
    }
    g.set_b_k(encode_matrix(desc_b, b_data.data(), b_data.size()));

    // Also add dummy A_V and B_V for descriptor completeness
    codec_desc desc_av = make_codec_desc(factor_role::a_v, GGML_TYPE_F32, orientation::token_major, {n_rows, rank_k});
    g.a_v = encode_matrix(desc_av, a_data.data(), a_data.size());
    codec_desc desc_bv = make_codec_desc(factor_role::b_v, GGML_TYPE_F32, orientation::feature_major_transposed, {total_dim_k, rank_k});
    g.set_b_v(encode_matrix(desc_bv, b_data.data(), b_data.size()));

    g.refresh_descriptor_fingerprint();
    g.update_byte_counters();
    seg->groups.push_back(std::move(g));
    seg->update_byte_counters();
    return seg;
}

// Test 1: Fragment partitioning across visibility, phase change, sparse gaps, and causal cutoff
static void test_fragment_partitioning() {
    std::cout << "[test_fragment_partitioning] starting..." << std::endl;
    auto seg = create_test_segment(16, 16, 64, 1);

    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    stamp.content_epoch = 1;
    stamp.codec_epoch = 1;
    stamp.view.topology_epoch = 1;
    stamp.view.publish_epoch = 1;
    stamp.view.layout_epoch = 1;

    std::vector<row_meta> rows = {
        // Run 0: normal visibility, phase_delta = 0
        {0, 1000, 1, 0, 10, 10, 0, static_cast<uint8_t>(llama_rerot_visibility::normal), true, true},
        {1, 1001, 1, 0, 11, 11, 0, static_cast<uint8_t>(llama_rerot_visibility::normal), true, true},
        {2, 1002, 1, 0, 12, 12, 0, static_cast<uint8_t>(llama_rerot_visibility::normal), true, true},

        // Boundary: visibility change to public_live
        {3, 1003, 1, 0, 13, 13, 0, static_cast<uint8_t>(llama_rerot_visibility::public_live), true, true},
        {4, 1004, 1, 0, 14, 14, 0, static_cast<uint8_t>(llama_rerot_visibility::public_live), true, true},

        // Boundary: phase change (virtual_pos offset change: storage_pos 15, virt 20 -> phase -5)
        {5, 1005, 1, 0, 15, 20, 0, static_cast<uint8_t>(llama_rerot_visibility::public_live), true, true},
        {6, 1006, 1, 0, 16, 21, 0, static_cast<uint8_t>(llama_rerot_visibility::public_live), true, true},

        // Boundary: sparse gap (storage_pos jumps from 16 to 19)
        {7, 1007, 1, 0, 19, 24, 0, static_cast<uint8_t>(llama_rerot_visibility::public_live), true, true},
        {8, 1008, 1, 0, 20, 25, 0, static_cast<uint8_t>(llama_rerot_visibility::public_live), true, true},

        // Boundary: reader_visible false
        {9, 1009, 1, 0, 21, 26, 0, static_cast<uint8_t>(llama_rerot_visibility::public_live), true, false},
        {10, 1010, 1, 0, 22, 27, 0, static_cast<uint8_t>(llama_rerot_visibility::public_live), true, true},
    };

    uint32_t chunk_size = 4;
    auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, chunk_size, 0x1234, -1);

    assert(frags.size() == 5);
    assert(frags[0].row_count == 3);
    assert(frags[1].row_count == 2);
    assert(frags[2].row_count == 2);
    assert(frags[3].row_count == 2);
    assert(frags[4].row_count == 1);

    // Verify key fields match snapshot stamp and segment version
    assert(frags[0].key.segment_version == 1);
    assert(frags[0].key.live_epoch == stamp.live_epoch);
    assert(frags[0].key.content_epoch == stamp.content_epoch);
    assert(frags[0].key.phase_tx_fingerprint == 0x1234);
    assert(frags[0].key.live_set_fingerprint != 0);

    // Now test causal cutoff at storage_pos 14
    auto causal_frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, chunk_size, 0x1234, 14);
    assert(causal_frags.size() == 2);
    assert(causal_frags[0].row_count == 3);
    assert(causal_frags[1].row_count == 2);

    std::cout << "[test_fragment_partitioning] passed!" << std::endl;
}

// Test 2: Multi-query isolation and 4-field row ref inspection (nonzero rows, generations, segment_version)
static void test_multi_query_isolation_and_4field_refs() {
    std::cout << "[test_multi_query_isolation_and_4field_refs] starting..." << std::endl;
    uint64_t seg_version = 42;
    auto seg = create_test_segment(16, 16, 64, seg_version);
    uint32_t feature_dim = 32;

    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;

    // Use heterogeneous generations: gen = 100 + i
    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 16; ++i) {
        rows.push_back({i, 1000 + i, 100 + i, 0, static_cast<int64_t>(i), static_cast<int64_t>(i), 0, true, true});
    }

    auto frags = build_legal_fragments(*seg, stamp, seg_version, 100, 0, 0, rows, 4, 0x42, -1);
    assert(frags.size() == 4);

    auto phase_tx = [](const float * src, int64_t, uint32_t, float * dst) {
        for (int i = 0; i < 32; ++i) dst[i] = src[i];
    };

    for (auto & f : frags) {
        encode_fragment_landmark(f, *seg, 0, feature_dim, phase_tx, 0x42, GGML_TYPE_Q8_0);
    }

    // Query 0: causal limit 5
    // Query 1: future query with gigantic query vector
    sr_query q0;
    q0.query_index = 0;
    q0.query_virtual_pos = 5;
    q0.causal_limit_pos = 5;
    q0.head_dim = feature_dim;
    q0.q_vec.resize(feature_dim, 1.0f);
    q0.scale = 1.0f;

    sr_query q1;
    q1.query_index = 1;
    q1.query_virtual_pos = 15;
    q1.causal_limit_pos = 15;
    q1.head_dim = feature_dim;
    q1.q_vec.resize(feature_dim, 1000.0f);
    q1.scale = 1.0f;

    sr_selection_config cfg;
    cfg.sr_budget = 2;

    auto batch_res = select_sr_batch({q0, q1}, frags, cfg, seg.get(), 0, feature_dim, phase_tx, 0x42);

    // Verify q0 did not select any rows beyond causal limit (rows <= 5)
    for (const auto & r_ref : batch_res.per_query[0].selected_rows) {
        assert(r_ref.row <= 5);
        assert(r_ref.segment_id == seg->segment_id);
        assert(r_ref.segment_version == seg_version);
        // Verify heterogeneous generation is preserved (gen = 100 + row)
        assert(r_ref.storage_generation == 100 + r_ref.row);
    }

    // Inspect nonzero row index in selected rows of q1
    bool found_nonzero_row = false;
    for (const auto & r_ref : batch_res.per_query[1].selected_rows) {
        assert(r_ref.segment_id == seg->segment_id);
        assert(r_ref.segment_version == seg_version);
        assert(r_ref.storage_generation == 100 + r_ref.row);
        if (r_ref.row > 0) {
            found_nonzero_row = true;
        }
    }
    assert(found_nonzero_row); // Proves row is not silently mapped to 0

    // Verify CSR representation exactly matches gather_rows
    assert(batch_res.csr_ptrs.size() == 3);
    assert(batch_res.csr_ptrs[0] == 0);
    assert(batch_res.csr_ptrs[2] == batch_res.csr_indices.size());

    for (size_t q_idx = 0; q_idx < 2; ++q_idx) {
        uint32_t start = batch_res.csr_ptrs[q_idx];
        uint32_t end = batch_res.csr_ptrs[q_idx + 1];
        std::vector<segment_row_ref> reconstructed;
        for (uint32_t idx = start; idx < end; ++idx) {
            reconstructed.push_back(batch_res.gather_rows[batch_res.csr_indices[idx]]);
        }
        assert(reconstructed == batch_res.per_query[q_idx].selected_rows);
    }

    std::cout << "[test_multi_query_isolation_and_4field_refs] passed!" << std::endl;
}

// Test 3: Keeper dedup at selection input and mandatory future rejection
static void test_keeper_dedup_and_mandatory_future_rejection() {
    std::cout << "[test_keeper_dedup_and_mandatory_future_rejection] starting..." << std::endl;
    uint64_t seg_version = 1;
    auto seg = create_test_segment(8, 16, 64, seg_version);
    uint32_t feature_dim = 32;

    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;

    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 8; ++i) {
        rows.push_back({i, 1000 + i, 1, 0, static_cast<int64_t>(i), static_cast<int64_t>(i), 0, true, true});
    }

    auto frags = build_legal_fragments(*seg, stamp, seg_version, 1, 0, 0, rows, 4, 0x1, -1);
    for (auto & f : frags) {
        encode_fragment_landmark(f, *seg, 0, feature_dim, nullptr, 0x1, GGML_TYPE_Q8_0);
    }

    sr_query q;
    q.head_dim = feature_dim;
    q.causal_limit_pos = 4; // Causal cutoff at row 4
    q.q_vec.resize(feature_dim, 0.5f);

    sr_selection_config cfg;
    cfg.sr_budget = 1;

    segment_row_ref ref2 = {seg->segment_id, seg_version, 1, 2};
    segment_row_ref ref3 = {seg->segment_id, seg_version, 1, 3};
    segment_row_ref ref7 = {seg->segment_id, seg_version, 1, 7}; // FUTURE row (storage_pos 7 > causal 4)

    cfg.hot_rows = {ref2, ref3, ref3}; // duplicate keeper ref
    cfg.outlier_rows = {ref7};         // illegal future row

    auto res = select_sr_query(q, frags, cfg, seg.get(), 0, feature_dim, nullptr, 0x1);

    // Selected rows must be strictly sorted and contain no duplicates
    for (size_t i = 1; i < res.selected_rows.size(); ++i) {
        assert(res.selected_rows[i - 1] < res.selected_rows[i]);
    }
    // Must contain legal keeper protected refs
    assert(std::binary_search(res.selected_rows.begin(), res.selected_rows.end(), ref2));
    assert(std::binary_search(res.selected_rows.begin(), res.selected_rows.end(), ref3));
    // Must NOT contain future row ref7
    assert(!std::binary_search(res.selected_rows.begin(), res.selected_rows.end(), ref7));

    std::cout << "[test_keeper_dedup_and_mandatory_future_rejection] passed!" << std::endl;
}

// Test 4: GQA normalized max pooling with per-Q-head z-score normalization
static void test_gqa_max_pooling() {
    std::cout << "[test_gqa_max_pooling] starting..." << std::endl;
    auto seg = create_test_segment(8, 16, 64, 1);
    uint32_t feature_dim = 16;

    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;

    std::vector<row_meta> rows = {
        {0, 1000, 1, 0, 0, 0, 0, true, true},
        {1, 1001, 1, 0, 1, 1, 0, true, true},
        {2, 1002, 1, 0, 2, 2, 0, true, true},
        {3, 1003, 1, 0, 3, 3, 0, true, true},
    };

    auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 2, 0x1, -1);
    assert(frags.size() == 2);
    for (auto & f : frags) {
        encode_fragment_landmark(f, *seg, 0, feature_dim, nullptr, 0x1, GGML_TYPE_Q8_0);
    }

    sr_query q;
    q.head_dim = feature_dim;
    q.q_head_indices = {0, 1};
    q.q_vec.resize(feature_dim * 2, 0.0f);

    for (uint32_t d = 0; d < feature_dim; ++d) q.q_vec[d] = 1.0f;
    for (uint32_t d = 0; d < feature_dim; ++d) q.q_vec[feature_dim + d] = -1.0f;

    sr_selection_config cfg;
    cfg.sr_budget = 1;
    auto res = select_sr_query(q, frags, cfg);

    // Compute exact oracle scores
    // Decode landmarks for frags 0 and 1
    auto lm0 = decode_matrix(frags[0].landmark_matrix, value_domain::canonical);
    auto lm1 = decode_matrix(frags[1].landmark_matrix, value_domain::canonical);
    float scale = 1.0f / std::sqrt(float(feature_dim));

    // For qh = 0: dot with lm0, dot with lm1
    float dot0_0 = 0.0f, dot0_1 = 0.0f;
    for (uint32_t d = 0; d < feature_dim; ++d) {
        dot0_0 += 1.0f * lm0[d];
        dot0_1 += 1.0f * lm1[d];
    }
    float raw0_0 = dot0_0 * scale;
    float raw0_1 = dot0_1 * scale;

    // z-score normalize for qh=0
    float mean0 = 0.5f * (raw0_0 + raw0_1);
    float var0 = 0.5f * ((raw0_0 - mean0)*(raw0_0 - mean0) + (raw0_1 - mean0)*(raw0_1 - mean0));
    float std0 = std::max(std::sqrt(var0), 1e-10f);
    float norm0_0 = (raw0_0 - mean0) / std0;
    float norm0_1 = (raw0_1 - mean0) / std0;

    // For qh = 1: dot with lm0, dot with lm1
    float dot1_0 = 0.0f, dot1_1 = 0.0f;
    for (uint32_t d = 0; d < feature_dim; ++d) {
        dot1_0 += -1.0f * lm0[d];
        dot1_1 += -1.0f * lm1[d];
    }
    float raw1_0 = dot1_0 * scale;
    float raw1_1 = dot1_1 * scale;

    float mean1 = 0.5f * (raw1_0 + raw1_1);
    float var1 = 0.5f * ((raw1_0 - mean1)*(raw1_0 - mean1) + (raw1_1 - mean1)*(raw1_1 - mean1));
    float std1 = std::max(std::sqrt(var1), 1e-10f);
    float norm1_0 = (raw1_0 - mean1) / std1;
    float norm1_1 = (raw1_1 - mean1) / std1;

    // Max pooled scores across Q heads:
    float pool0 = std::max(norm0_0, norm1_0);
    float pool1 = std::max(norm0_1, norm1_1);
    float expected_top_score = std::max(pool0, pool1);

    assert(!res.scores.empty());
    assert(std::fabs(res.scores[0] - expected_top_score) < 1e-4f);

    std::cout << "[test_gqa_max_pooling] passed!" << std::endl;
}

// Test 5: Landmark quantization recall (Q8_0 and Turbo4_0) with valid 128/256 dimensions
static void test_landmark_quantization_recall() {
    std::cout << "[test_landmark_quantization_recall] starting..." << std::endl;
    uint32_t feature_dim = 128; // valid 128 for Turbo4_0
    uint32_t total_dim_k = 256;
    auto seg = create_test_segment(16, 16, total_dim_k, 1);

    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;

    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 16; ++i) {
        rows.push_back({i, 1000 + i, 1, 0, static_cast<int64_t>(i), static_cast<int64_t>(i), 0, true, true});
    }

    auto frags_fp = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x10, -1);
    auto frags_q8 = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x10, -1);
    auto frags_t4 = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x10, -1);

    for (auto & f : frags_fp) {
        encode_fragment_landmark(f, *seg, 0, feature_dim, nullptr, 0x10, GGML_TYPE_F32);
    }
    for (auto & f : frags_q8) {
        encode_fragment_landmark(f, *seg, 0, feature_dim, nullptr, 0x10, GGML_TYPE_Q8_0);
        assert(f.landmark_matrix.desc.type == GGML_TYPE_Q8_0);
        assert(f.error_bound > 0.0f);
        assert(f.source_fingerprint != 0);
    }

    for (auto & f : frags_t4) {
        encode_fragment_landmark(f, *seg, 0, feature_dim, nullptr, 0x10, GGML_TYPE_TURBO4_0);
        assert(f.landmark_matrix.desc.type == GGML_TYPE_TURBO4_0);
        assert(f.error_bound > 0.0f);
        assert(f.source_fingerprint != 0);
    }

    sr_query q;
    q.head_dim = feature_dim;
    q.q_vec.resize(feature_dim, 0.2f);
    q.scale = 1.0f / std::sqrt(float(feature_dim));

    sr_selection_config cfg;
    cfg.sr_budget = 2;

    auto res_fp = select_sr_query(q, frags_fp, cfg);
    auto res_q8 = select_sr_query(q, frags_q8, cfg);
    auto res_t4 = select_sr_query(q, frags_t4, cfg);

    // High recall vs unquantized FP32 landmark oracle:
    // Measure intersection of selected rows between FP and Q8 / Turbo4
    size_t match_q8 = 0;
    for (const auto & r : res_q8.selected_rows) {
        if (std::binary_search(res_fp.selected_rows.begin(), res_fp.selected_rows.end(), r)) {
            ++match_q8;
    }
    }
    assert(match_q8 == res_fp.selected_rows.size()); // 100% top-k recall for Q8

    size_t match_t4 = 0;
    for (const auto & r : res_t4.selected_rows) {
        if (std::binary_search(res_fp.selected_rows.begin(), res_fp.selected_rows.end(), r)) {
            ++match_t4;
    }
    }
    assert(match_t4 == res_fp.selected_rows.size()); // 100% top-k recall for Turbo4

    std::cout << "[test_landmark_quantization_recall] passed!" << std::endl;
}

// Test 6: Invalid bounds, bad segment_version, and unsupported codec explicit rejection
static void test_invalid_bounds_and_codec_rejection() {
    std::cout << "[test_invalid_bounds_and_codec_rejection] starting..." << std::endl;
    auto seg = create_test_segment(8, 16, 64, 1);
    xkv_snapshot_stamp stamp;

    // 1. Segment version mismatch must throw
    bool caught_version_mismatch = false;
    try {
        build_legal_fragments(*seg, stamp, 999, 1, 0, 0, {}, 4, 0x0, -1); // 999 != 1
    } catch (const std::invalid_argument &) {
        caught_version_mismatch = true;
    }
    // (empty rows returns early, test with non-empty)
    std::vector<row_meta> single_row = {{0, 1000, 1, 0, 10, 10, 0, true, true}};
    try {
        build_legal_fragments(*seg, stamp, 999, 1, 0, 0, single_row, 4, 0x0, -1);
    } catch (const std::invalid_argument &) {
        caught_version_mismatch = true;
    }
    assert(caught_version_mismatch);

    // 2. Unsorted rows must throw
    std::vector<row_meta> unsorted_rows = {
        {1, 1001, 1, 0, 20, 20, 0, true, true},
        {0, 1000, 1, 0, 10, 10, 0, true, true},
    };
    bool caught_unsorted = false;
    try {
        build_legal_fragments(*seg, stamp, 1, 1, 0, 0, unsorted_rows, 4, 0x0, -1);
    } catch (const std::invalid_argument &) {
        caught_unsorted = true;
    }
    assert(caught_unsorted);

    // 3. Mismatched payload_id must throw
    std::vector<row_meta> mismatched_payload = {
        {0, 9999, 1, 0, 10, 10, 0, true, true}, // payload 9999 != 1000
    };
    bool caught_mismatch = false;
    try {
        build_legal_fragments(*seg, stamp, 1, 1, 0, 0, mismatched_payload, 4, 0x0, -1);
    } catch (const std::invalid_argument &) {
        caught_mismatch = true;
    }
    assert(caught_mismatch);

    // 4. Out of bounds slice in encode_fragment_landmark must throw
    legal_fragment frag;
    frag.row_count = 1;
    frag.row_indices = {0};
    frag.storage_positions = {0};
    bool caught_oob = false;
    try {
        encode_fragment_landmark(frag, *seg, 60, 10, nullptr, 0); // 60 + 10 = 70 > 64
    } catch (const std::out_of_range &) {
        caught_oob = true;
    }
    assert(caught_oob);

    // 5. Unsupported landmark codec must throw explicitly
    bool caught_bad_codec = false;
    try {
        encode_fragment_landmark(frag, *seg, 0, 32, nullptr, 0, GGML_TYPE_Q4_0);
    } catch (const std::invalid_argument &) {
        caught_bad_codec = true;
    }
    assert(caught_bad_codec);

    std::cout << "[test_invalid_bounds_and_codec_rejection] passed!" << std::endl;
}

// Test 7: landmark_table deterministic LRU eviction with held external lease protection
static void test_landmark_table_lease_protection() {
    std::cout << "[test_landmark_table_lease_protection] starting..." << std::endl;
    auto make_frag = [](uint64_t seg_id, uint64_t epoch, size_t bytes_len) {
        auto f = std::make_shared<legal_fragment>();
        f->key.segment_id = seg_id;
        f->key.live_epoch = epoch;
        f->row_indices = {0};
        f->landmark_matrix.bytes.resize(bytes_len, 0);
        return f;
    };

    // Part A: Ordinary unleased deterministic LRU eviction
    {
        auto f_sample = make_frag(1, 1, 100);
        size_t entry_sz = f_sample->allocated_bytes();
        size_t budget = entry_sz * 2 + 50; // Holds exactly 2 entries
    landmark_table tbl(budget);

        auto e1 = make_frag(1, 1, 100);
        auto e2 = make_frag(2, 1, 100);
        landmark_fragment_key k1 = e1->key;
        landmark_fragment_key k2 = e2->key;

        assert(tbl.insert(e1));
        assert(tbl.insert(e2));
        assert(tbl.fragment_count() == 2);

        // Explicitly reset creator handles so tbl holds the sole reference (use_count == 1)
        e1.reset();
        e2.reset();

        // Access e1 in a narrow scope to promote to MRU, then immediately drop the lease
        {
            std::shared_ptr<const legal_fragment> promo = tbl.find(k1);
            assert(promo != nullptr);
        }

        // Insert e3: both e1 and e2 are completely unleased, so e2 (LRU) is evicted!
        auto e3 = make_frag(3, 1, 100);
        landmark_fragment_key k3 = e3->key;
        assert(tbl.insert(e3));
        e3.reset();
        assert(tbl.fragment_count() == 2);
        assert(tbl.find(k1) != nullptr);
        assert(tbl.find(k2) == nullptr);
        assert(tbl.find(k3) != nullptr);
}

    // Part B: Held external lease prevents eviction, causing insert rejection when over capacity.
    // Then releasing the lease allows successful insertion.
    {
        auto f_sample = make_frag(1, 1, 100);
        size_t entry_sz = f_sample->allocated_bytes();
        size_t budget = entry_sz + 50; // Holds only 1 entry
    landmark_table tbl(budget);

        auto f1 = make_frag(1, 1, 100);
        landmark_fragment_key k1 = f1->key;
    assert(tbl.insert(f1));
        assert(tbl.fragment_count() == 1);

        // Hold external lease to f1
        std::shared_ptr<const legal_fragment> held_lease = tbl.find(f1->key);
    assert(held_lease != nullptr);
        // Reset creator f1 so held_lease is the ONLY external lease (use_count == 2: tbl + held_lease)
        f1.reset();
    assert(held_lease.use_count() > 1);

        // Attempt to insert f2: requires evicting f1, but f1 is leased!
        // Must fail, roll back f2, keep f1, and respect hard max_bytes bound.
        auto f2 = make_frag(2, 1, 100);
        landmark_fragment_key k2 = f2->key;
        bool inserted = tbl.insert(f2);
        assert(!inserted);
        assert(tbl.fragment_count() == 1);
        assert(tbl.find(k1) != nullptr);
        assert(tbl.find(k2) == nullptr);
        assert(tbl.total_allocated_bytes() <= budget);

        // Now release the external lease to f1
        held_lease.reset();
        // f1 is now unleased in tbl (sole owner: use_count == 1), so inserting f2 can successfully evict f1!
        bool inserted_after_release = tbl.insert(f2);
        f2.reset();
        assert(inserted_after_release);
        assert(tbl.fragment_count() == 1);
        assert(tbl.find(k1) == nullptr);
        assert(tbl.find(k2) != nullptr);
        assert(tbl.total_allocated_bytes() <= budget);
}

    std::cout << "[test_landmark_table_lease_protection] passed!" << std::endl;
}
static void test_normalized_boundary_refinement() {
    std::cout << "[test_normalized_boundary_refinement] starting..." << std::endl;
    uint32_t feature_dim = 32;
    auto seg = create_test_segment(16, 16, 64, 1);

    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;

    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 16; ++i) {
        rows.push_back({i, 1000 + i, 1, 0, static_cast<int64_t>(i), static_cast<int64_t>(i), 0, true, true});
    }

    auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x1, -1);
    assert(frags.size() == 4);
    for (auto & f : frags) {
        encode_fragment_landmark(f, *seg, 0, feature_dim, nullptr, 0x1, GGML_TYPE_Q8_0);
    }

    sr_query q;
    q.head_dim = feature_dim;
    q.q_vec.resize(feature_dim, 0.2f);
    q.scale = 1.0f / std::sqrt(float(feature_dim));

    // Case A: Sufficient refinement budget (refine_max_rows = 64)
    sr_selection_config cfg_ok;
    cfg_ok.sr_budget = 2;
    cfg_ok.refine_mode = LLAMA_XKV_LANDMARK_REFINE_BOUNDARY;
    cfg_ok.refine_max_rows = 64;

    auto res_ok = select_sr_query(q, frags, cfg_ok, seg.get(), 0, feature_dim, nullptr, 0x1);
    assert(!res_ok.refine_cap_hit);
    assert(res_ok.selected_rows.size() == 8);

    // Case B: Insufficient refinement budget (refine_max_rows = 0)
    // Forces refine_cap_hit = true when candidates exist around boundary
    sr_selection_config cfg_cap;
    cfg_cap.sr_budget = 2;
    cfg_cap.refine_mode = LLAMA_XKV_LANDMARK_REFINE_BOUNDARY;
    cfg_cap.refine_max_rows = 0; // zero budget forces cap hit

    auto res_cap = select_sr_query(q, frags, cfg_cap, seg.get(), 0, feature_dim, nullptr, 0x1);
    assert(res_cap.refine_cap_hit);
    assert(res_cap.rows_refined == 0);
    assert(res_cap.selected_rows.size() == 8);

    std::cout << "[test_normalized_boundary_refinement] passed!" << std::endl;
}

static void test_bounded_csr_groups() {
    std::cout << "[test_bounded_csr_groups] starting..." << std::endl;
    auto seg = create_test_segment(16, 16, 64, 1);
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 16; ++i) {
        rows.push_back({i, 1000 + i, 1, 0, (int64_t) i, (int64_t) i, 0, true, true});
    }
    auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x1, -1);
    assert(frags.size() == 4);
    for (auto & f : frags) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
    // Expanded per-(query,group) views: frags 0,1 in group 7; frags 2,3 in group 9.
    auto views = test_views(frags);
    views[0].group_index = 7;
    views[1].group_index = 7;
    views[2].group_index = 9;
    views[3].group_index = 9;
    sr_query q0, q1;
    q0.head_dim = 32;
    q0.q_vec.assign(32, 0.3f);
    q1.head_dim = 32;
    q1.causal_limit_pos = 9; // partial on frag 2 (rows 8,9 legal; 10,11 future), frag3 cut
    q1.q_vec.assign(32, -0.4f);
    sr_selection_config cfg;
    cfg.sr_budget = 4;
    std::vector<uint8_t> scratch =
        test_make_scratch(4, 1, 32, 4, test_padded_rank(*seg));
    // Single: per-row groups ride parallel to rows; scores unchanged by group tracking.
    auto plain = run_bounded_single(q0, views, cfg, nullptr, 0, 32, nullptr, 0, scratch);
    std::vector<uint32_t> groups(64, 0xDEADu);
    bounded_single_out o;
    {
        o.rows.resize(64);
        o.scores.resize(8);
        size_t nr = 0, ns = 0;
        bool ok = select_sr_query_bounded(q0, views.data(), views.size(), cfg, nullptr, 0,
            32, nullptr, 0, scratch.data(), scratch.size(), o.rows.data(), o.rows.size(),
            &nr, o.scores.data(), o.scores.size(), &ns, &o.refined, &o.cap, nullptr,
            nullptr, 0, nullptr, groups.data(), groups.size());
        assert(ok);
        o.rows.resize(nr);
        o.scores.resize(ns);
    }
    assert(o.rows == plain.rows);
    assert(o.scores == plain.scores); // no score duplication/change from group metadata
    assert(o.rows.size() == 16);
    for (size_t k = 0; k < o.rows.size(); ++k) {
        uint32_t expect = (o.rows[k].row < 8) ? 7u : 9u;
        assert(groups[k] == expect); // exactly its effective group
    }
    // Partial rows keep their group (causal split never regroups).
    std::fill(groups.begin(), groups.end(), 0xDEADu);
    {
        o.rows.resize(64);
        o.scores.resize(8);
        size_t nr = 0, ns = 0;
        bool ok = select_sr_query_bounded(q1, views.data(), views.size(), cfg, seg.get(), 0,
            32, nullptr, 0x1, scratch.data(), scratch.size(), o.rows.data(), o.rows.size(),
            &nr, o.scores.data(), o.scores.size(), &ns, &o.refined, &o.cap, nullptr,
            nullptr, 0, nullptr, groups.data(), groups.size());
        assert(ok);
        o.rows.resize(nr);
    }
    for (size_t k = 0; k < o.rows.size(); ++k) {
        assert(o.rows[k].row <= 9);
        uint32_t expect = (o.rows[k].row < 8) ? 7u : 9u;
        assert(groups[k] == expect);
    }
    // Batch: csr_group_indices parallel csr_indices; merge-back preserves groups.
    std::vector<sr_query> qs = {q0, q1};
    landmark_workspace_config wcfg;
    wcfg.max_fragments = 4;
    wcfg.max_rows_per_fragment = 4;
    wcfg.max_rows_total = 16;
    wcfg.max_queries = 2;
    wcfg.max_q_heads_per_query = 1;
    wcfg.max_head_dim = 32;
    wcfg.max_protected_rows = 0;
    wcfg.max_padded_rank = test_padded_rank(*seg);
    landmark_workspace batch_ws;
    assert(batch_ws.init(wcfg, nullptr));
    std::vector<segment_row_ref> gather(64);
    std::vector<uint32_t> ptrs(3, 0), cidx(64, 0), cgrp(64, 0xDEADu);
    size_t ng = 0, nc = 0;
    uint32_t tr = 0, tc = 0;
    assert(select_sr_batch_bounded(qs.data(), qs.size(), views.data(), views.size(), cfg,
        seg.get(), 0, 32, nullptr, 0x1, batch_ws.data(), batch_ws.capacity_bytes(), gather.data(),
        gather.size(), &ng, ptrs.data(), ptrs.size(), cidx.data(), cidx.size(), &nc, &tr,
        &tc, nullptr, nullptr, 0, nullptr, cgrp.data(), cgrp.size()));
    gather.resize(ng);
    for (size_t k = 0; k < nc; ++k) {
        const segment_row_ref & r = gather[cidx[k]];
        uint32_t expect = (r.row < 8) ? 7u : 9u;
        assert(cgrp[k] == expect);
    }
    // Same physical row in both queries keeps exactly its group (no loss/duplication).
    for (uint32_t k = ptrs[0]; k < ptrs[1]; ++k) {
        for (uint32_t j = ptrs[1]; j < ptrs[2]; ++j) {
            if (gather[cidx[k]] == gather[cidx[j]]) assert(cgrp[k] == cgrp[j]);
        }
    }
    // Short group capacity fails closed.
    assert(!select_sr_batch_bounded(qs.data(), qs.size(), views.data(), views.size(), cfg,
        seg.get(), 0, 32, nullptr, 0x1, batch_ws.data(), batch_ws.capacity_bytes(), gather.data(), 64,
        &ng, ptrs.data(), ptrs.size(), cidx.data(), cidx.size(), &nc, &tr, &tc, nullptr,
        nullptr, 0, nullptr, cgrp.data(), 0));
    // Unset views pass UINT32_MAX through verbatim.
    auto views_plain = test_views(frags);
    std::fill(groups.begin(), groups.end(), 0xDEADu);
    {
        o.rows.resize(64);
        o.scores.resize(8);
        size_t nr = 0, ns = 0;
        bool ok = select_sr_query_bounded(q0, views_plain.data(), views_plain.size(), cfg,
            nullptr, 0, 32, nullptr, 0, scratch.data(), scratch.size(), o.rows.data(),
            o.rows.size(), &nr, o.scores.data(), o.scores.size(), &ns, &o.refined, &o.cap,
            nullptr, nullptr, 0, nullptr, groups.data(), groups.size());
        assert(ok);
        for (size_t k = 0; k < nr; ++k) assert(groups[k] == UINT32_MAX);
    }
    std::cout << "[test_bounded_csr_groups] passed!" << std::endl;
}
static void test_fault_atomic_rescore() {
    std::cout << "[test_fault_atomic_rescore] starting..." << std::endl;
    auto seg = create_test_segment(16, 16, 64, 1);
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 16; ++i) {
        rows.push_back({i, 1000 + i, 1, 0, (int64_t) i, (int64_t) i, 0, true, true});
    }
    auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x1, -1);
    for (auto & f : frags) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
    auto views = test_views(frags);
    sr_query q;
    q.head_dim = 32;
    q.q_vec.assign(32, 0.2f);
    q.scale = 1.0f / std::sqrt(32.0f);
    sr_selection_config cfg;
    cfg.sr_budget = 2;
    cfg.refine_mode = LLAMA_XKV_LANDMARK_REFINE_BOUNDARY;
    cfg.refine_max_rows = 64;
    auto clean = select_sr_query(q, frags, cfg, seg.get(), 0, 32, nullptr, 0x1);
    assert(clean.rows_refined > 0); // at least one rescore event exists to inject into
    std::vector<uint8_t> scratch =
        test_make_scratch(4, 1, 32, 4, test_padded_rank(*seg));
    // Reference: injected fault throws before result emission (nothing committed).
    sr_selection_config faulty = cfg;
    faulty.fault.stage = landmark_fault_stage::refine_rescore;
    faulty.fault.occurrence = 0;
    bool threw = false;
    try {
        select_sr_query(q, frags, faulty, seg.get(), 0, 32, nullptr, 0x1);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    assert(threw);
    // Beyond the executed rebuilds: never triggers, selection identical (field inert).
    faulty.fault.occurrence = 99;
    auto same = select_sr_query(q, frags, faulty, seg.get(), 0, 32, nullptr, 0x1);
    assert(same.selected_rows == clean.selected_rows);
    assert(same.scores == clean.scores);
    // Bounded: failure leaves counts zero and payload arrays bit-untouched.
    faulty.fault.occurrence = 0;
    std::vector<segment_row_ref> br(64);
    std::vector<float> bs(8);
    for (auto & r : br) r = segment_row_ref{0xFFu, 0xFFu, 0xFFu, 0xFFu};
    for (auto & s : bs) s = -1.25f;
    std::vector<segment_row_ref> br_snap = br;
    std::vector<float> bs_snap = bs;
    size_t nr = 12345, ns = 12345;
    uint32_t rf = 999;
    bool ch = true;
    bool ok = select_sr_query_bounded(q, views.data(), views.size(), faulty, seg.get(), 0,
        32, nullptr, 0x1, scratch.data(), scratch.size(), br.data(), br.size(), &nr,
        bs.data(), bs.size(), &ns, &rf, &ch, nullptr);
    assert(!ok);
    assert(nr == 0 && ns == 0 && rf == 0 && !ch);
    assert(br == br_snap); // payload rows untouched
    assert(bs == bs_snap); // payload scores untouched
    std::cout << "[test_fault_atomic_rescore] passed!" << std::endl;
}

static void test_zero_fragment_hard_keeps() {
    std::cout << "[test_zero_fragment_hard_keeps] starting..." << std::endl;
    auto seg = create_test_segment(8, 16, 64, 1); // id 100, version 1, 8 live rows
    std::vector<legal_fragment> nofrags;
    // Zero-candidate core need is small but nonzero (q norms/stddevs, rebuild slot, slack).
    std::vector<uint8_t> scratch(4096, 0);
    segment_row_ref h1{100, 1, 1, 1}, h3{100, 1, 1, 3};
    segment_row_ref r3{100, 1, 1, 3}, r5{100, 1, 1, 5};
    segment_row_ref o7{100, 1, 1, 7};
    std::vector<segment_row_ref> want = {h1, h3, r5, o7}; // union, deduped, sorted
    auto check_union = [&](const sr_selection_config & cfg) {
        sr_query q;
        q.head_dim = 32;
        q.q_vec.assign(32, 0.1f);
        auto ref = select_sr_query(q, nofrags, cfg, seg.get());
        assert(ref.selected_rows == want);
        assert(ref.scores.empty());
        std::vector<segment_row_ref> br(8);
        std::vector<float> bs(4);
        std::vector<uint32_t> bg(8, 0xDEADu);
        size_t nr = 0, ns = 0;
        uint32_t rf = 0;
        bool ch = false;
        bool ok = select_sr_query_bounded(q, nullptr, 0, cfg, seg.get(), 0, 32, nullptr,
            0, scratch.data(), scratch.size(), br.data(), br.size(), &nr, bs.data(),
            bs.size(), &ns, &rf, &ch, nullptr, nullptr, 0, nullptr, bg.data(), bg.size());
        assert(ok);
        br.resize(nr);
        assert(br == want);
        assert(ns == 0 && rf == 0 && !ch);
        for (size_t k = 0; k < nr; ++k) assert(bg[k] == UINT32_MAX); // no source group
        return q;
    };
    // each protected class alone, then combined (dedup across classes)
    {
        sr_selection_config c;
        c.hot_rows = {h1};
        sr_query q;
        q.head_dim = 32;
        q.q_vec.assign(32, 0.1f);
        assert(select_sr_query(q, nofrags, c, seg.get()).selected_rows ==
               std::vector<segment_row_ref>({h1}));
    }
    {
        sr_selection_config c;
        c.recent_rows = {r5};
        sr_query q;
        q.head_dim = 32;
        q.q_vec.assign(32, 0.1f);
        assert(select_sr_query(q, nofrags, c, seg.get()).selected_rows ==
               std::vector<segment_row_ref>({r5}));
    }
    {
        sr_selection_config c;
        c.outlier_rows = {o7};
        sr_query q;
        q.head_dim = 32;
        q.q_vec.assign(32, 0.1f);
        assert(select_sr_query(q, nofrags, c, seg.get()).selected_rows ==
               std::vector<segment_row_ref>({o7}));
    }
    sr_selection_config all;
    all.hot_rows = {h1, h3};
    all.recent_rows = {r3, r5};
    all.outlier_rows = {o7};
    sr_query qbase = check_union(all);
    (void) qbase;
    // fail-closed: causal cutoff (unprovable), no segment, wrong owner, dead/OOB rows
    {
        sr_query qc = qbase;
        qc.causal_limit_pos = 100;
        bool threw = false;
        try {
            select_sr_query(qc, nofrags, all, seg.get());
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        assert(threw);
        std::vector<segment_row_ref> br(8);
        std::vector<float> bs(4);
        size_t nr = 0, ns = 0;
        uint32_t rf = 0;
        bool ch = false;
        assert(!select_sr_query_bounded(qc, nullptr, 0, all, seg.get(), 0, 32, nullptr, 0,
            scratch.data(), scratch.size(), br.data(), br.size(), &nr, bs.data(), bs.size(),
            &ns, &rf, &ch, nullptr));
    }
    {
        bool threw = false;
        try {
            select_sr_query(qbase, nofrags, all, nullptr);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        assert(threw);
        std::vector<segment_row_ref> br(8);
        std::vector<float> bs(4);
        size_t nr = 0, ns = 0;
        uint32_t rf = 0;
        bool ch = false;
        assert(!select_sr_query_bounded(qbase, nullptr, 0, all, nullptr, 0, 32, nullptr, 0,
            scratch.data(), scratch.size(), br.data(), br.size(), &nr, bs.data(), bs.size(),
            &ns, &rf, &ch, nullptr));
    }
    {
        sr_selection_config bad = all;
        bad.hot_rows = {segment_row_ref{999, 1, 1, 1}}; // unknown owner
        bool threw = false;
        try {
            select_sr_query(qbase, nofrags, bad, seg.get());
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        assert(threw);
    }
    {
        auto dead = create_test_segment(8, 16, 64, 1);
        dead->live_rows[5] = false;
        dead->n_live_rows = 7;
        sr_selection_config bad = all; // r5 dead in this segment
        bool threw = false;
        try {
            select_sr_query(qbase, nofrags, bad, dead.get());
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        assert(threw);
        std::vector<segment_row_ref> br(8);
        std::vector<float> bs(4);
        size_t nr = 0, ns = 0;
        uint32_t rf = 0;
        bool ch = false;
        assert(!select_sr_query_bounded(qbase, nullptr, 0, bad, dead.get(), 0, 32, nullptr,
            0, scratch.data(), scratch.size(), br.data(), br.size(), &nr, bs.data(),
            bs.size(), &ns, &rf, &ch, nullptr));
    }
    {
        sr_selection_config bad = all;
        bad.outlier_rows = {segment_row_ref{100, 1, 1, 99}}; // out of range
        bool threw = false;
        try {
            select_sr_query(qbase, nofrags, bad, seg.get());
        } catch (const std::exception &) {
            threw = true;
        }
        assert(threw);
    }
    // capacity: exact fits, one less refuses (single + batch)
    {
        std::vector<segment_row_ref> br(4);
        std::vector<float> bs(4);
        size_t nr = 0, ns = 0;
        uint32_t rf = 0;
        bool ch = false;
        assert(select_sr_query_bounded(qbase, nullptr, 0, all, seg.get(), 0, 32, nullptr, 0,
            scratch.data(), scratch.size(), br.data(), br.size(), &nr, bs.data(), bs.size(),
            &ns, &rf, &ch, nullptr));
        assert(nr == 4);
        std::vector<segment_row_ref> br3(3);
        nr = 0;
        assert(!select_sr_query_bounded(qbase, nullptr, 0, all, seg.get(), 0, 32, nullptr, 0,
            scratch.data(), scratch.size(), br3.data(), br3.size(), &nr, bs.data(), bs.size(),
            &ns, &rf, &ch, nullptr));
    }
    // batch parity: keeps query + empty query; reference groups invalid, bounded same rows
    {
        sr_query qempty;
        qempty.head_dim = 32;
        qempty.q_vec.assign(32, 0.1f);
        sr_selection_config none;
        std::vector<sr_query> qs = {qbase, qempty};
        // per-query configs differ: run reference per query and compare slices
        auto r0 = select_sr_query(qs[0], nofrags, all, seg.get());
        auto r1 = select_sr_query(qs[1], nofrags, none, seg.get());
        assert(r0.selected_rows == want);
        assert(r1.selected_rows.empty());
        std::vector<segment_row_ref> gather(8);
        std::vector<uint32_t> ptrs(3, 0), cidx(8, 0), cgrp(8, 0);
        size_t ng = 0, nc = 0;
        uint32_t tr = 0, tc = 0;
        // batch shares one config; use all-keeps for q0-shaped check via single-config batch
        std::vector<sr_query> qs0 = {qbase};
        assert(select_sr_batch_bounded(qs0.data(), qs0.size(), nullptr, 0, all, seg.get(),
            0, 32, nullptr, 0, scratch.data(), scratch.size(), gather.data(), gather.size(),
            &ng, ptrs.data(), ptrs.size(), cidx.data(), cidx.size(), &nc, &tr, &tc,
            nullptr, nullptr, 0, nullptr, cgrp.data(), cgrp.size()));
        gather.resize(ng);
        assert(gather == want);
        assert(nc == 4);
        for (size_t k = 0; k < nc; ++k) assert(cgrp[k] == UINT32_MAX);
        auto bref = select_sr_batch(qs0, nofrags, all, seg.get());
        assert(bref.gather_rows == want);
        assert(bref.csr_group_indices.size() == bref.csr_indices.size());
        for (auto g : bref.csr_group_indices) assert(g == UINT32_MAX);
    }
    std::cout << "[test_zero_fragment_hard_keeps] passed!" << std::endl;
}

static void test_base_table_bind() {
    std::cout << "[test_base_table_bind] starting..." << std::endl;
    auto seg = create_test_segment(8, 16, 64, 1);
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    std::vector<row_meta> rows;
    for (uint32_t i = 0; i < 8; ++i) {
        rows.push_back({i, 1000 + i, 1, 0, (int64_t) i, (int64_t) i, 0, true, true});
    }
    auto ref = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x1, -1);
    auto bare = ref;
    assert(ref.size() == 2);
    for (auto & f : ref) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
    for (auto & f : bare) {
        f.landmark_matrix.bytes.clear(); // runtime shape: empty data, intact identity
        f.error_bound = 0.0f;
    }
    auto lm0 = decode_matrix(ref[0].landmark_matrix, value_domain::canonical);
    auto lm1 = decode_matrix(ref[1].landmark_matrix, value_domain::canonical);
    std::vector<float> base_data = lm0;
    base_data.insert(base_data.end(), lm1.begin(), lm1.end());
    codec_desc bdesc = make_codec_desc(factor_role::landmark, GGML_TYPE_Q8_0,
        orientation::token_major, {2, 32}, 0, 0x5);
    auto base = std::make_shared<const encoded_matrix>(encode_matrix(bdesc, base_data.data(), base_data.size()));
    std::vector<uint64_t> fpids(8), fgens(8);
    std::vector<int64_t> fpos(8);
    for (uint32_t i = 0; i < 8; ++i) {
        fpids[i] = 1000 + i;
        fgens[i] = 1;
        fpos[i] = (int64_t) i;
    }
    std::vector<uint32_t> offs = {0, 4, 8};
    float bounds[2] = {ref[0].error_bound, ref[1].error_bound};
    uint64_t source_fps[2] = {ref[0].source_fingerprint, ref[1].source_fingerprint};
    landmark_base_table table;
    table.landmark = base;
    table.row_payload_ids = fpids.data();
    table.row_generations = fgens.data();
    table.row_positions = fpos.data();
    table.chunk_error_bounds = bounds;
    table.chunk_source_fingerprints = source_fps;
    table.chunk_row_offsets = offs.data();
    table.n_rows_total = 8;
    table.n_chunks = 2;
    table.stamp = stamp;
    table.phase_tx_fingerprint = 0x1;
    table.bounds_fingerprint = compute_base_table_fingerprint(table);
    size_t nb = 999;
    assert(bind_base_landmarks(bare.data(), bare.size(), table, &nb, nullptr));
    assert(nb == 2);
    assert(bare[0].uses_base_landmark() && bare[1].uses_base_landmark());
    assert(bare[0].base_landmark_row == 0 && bare[1].base_landmark_row == 1);
    assert(bare[0].error_bound == bounds[0] && bare[1].error_bound == bounds[1]); // persisted seal bounds
    assert(bare[0].key.landmark_codec_fp == base->desc.fingerprint());
    // Selection parity vs the duplicate-encoded set (reference and bounded).
    sr_query q;
    q.head_dim = 32;
    q.q_vec.assign(32, 0.6f);
    sr_selection_config cfg;
    cfg.sr_budget = 1;
    auto want = select_sr_query(q, ref, cfg);
    auto got = select_sr_query(q, bare, cfg);
    assert(got.selected_rows == want.selected_rows);
    assert(got.scores == want.scores);
    auto views = test_views(bare);
    std::vector<uint8_t> scratch =
        test_make_scratch(2, 1, 32, 4, test_padded_rank(*seg));
    auto out = run_bounded_single(q, views, cfg, nullptr, 0, 32, nullptr, 0, scratch);
    assert(out.rows == want.selected_rows);
    // Partial fragment left for scratch rebuild (not an error).
    bare[1].row_count = 3;
    bare[1].row_indices.pop_back();
    bare[1].payload_ids.pop_back();
    bare[1].generations.pop_back();
    bare[1].storage_positions.pop_back();
    bare[1].base_landmark.reset();
    nb = 999;
    assert(bind_base_landmarks(bare.data(), bare.size(), table, &nb, nullptr));
    assert(nb == 0); // frag0 already bound (kept, uncounted), frag1 partial
    assert(bare[0].uses_base_landmark());
    assert(!bare[1].uses_base_landmark());
    // Stale table fails closed: epoch drift breaks the closure fingerprint, so a
    // stale table can never bind silently (no silent all-scratch degradation).
    landmark_base_table stale = table;
    stale.stamp.content_epoch = 77;
    std::vector<legal_fragment> fresh = ref;
    for (auto & f : fresh) {
        f.landmark_matrix.bytes.clear();
        f.base_landmark.reset();
    }
    nb = 999;
    assert(!bind_base_landmarks(fresh.data(), fresh.size(), stale, &nb, nullptr));
    // Structural defects fail closed.
    landmark_base_table bad = table;
    bad.landmark.reset();
    assert(!bind_base_landmarks(fresh.data(), fresh.size(), bad, &nb, nullptr));
    // Missing closure fingerprint fails closed even with valid data.
    {
        landmark_base_table nofp = table;
        nofp.bounds_fingerprint = 0;
        assert(!bind_base_landmarks(fresh.data(), fresh.size(), nofp, &nb, nullptr));
    }
    // Tampered bound under a stale fingerprint: bounds check passes, closure catches it.
    {
        float tbad[2] = {bounds[0], bounds[1] + 0.5f};
        landmark_base_table tamp = table;
        tamp.chunk_error_bounds = tbad;
        assert(!bind_base_landmarks(fresh.data(), fresh.size(), tamp, &nb, nullptr));
    }
    {
        uint64_t sbad[2] = {source_fps[0], source_fps[1] ^ 1u};
        landmark_base_table tamp = table;
        tamp.chunk_source_fingerprints = sbad;
        assert(!bind_base_landmarks(fresh.data(), fresh.size(), tamp, &nb, nullptr));
    }
    // Non-finite / negative bounds fail closed (fingerprint recomputed so the bounds
    // check itself is what fires).
    {
        float nbad[2] = {bounds[0], std::numeric_limits<float>::quiet_NaN()};
        landmark_base_table nant = table;
        nant.chunk_error_bounds = nbad;
        nant.bounds_fingerprint = compute_base_table_fingerprint(nant);
        assert(!bind_base_landmarks(fresh.data(), fresh.size(), nant, &nb, nullptr));
    }
    {
        float vbad[2] = {bounds[0], -1.0f};
        landmark_base_table negt = table;
        negt.chunk_error_bounds = vbad;
        negt.bounds_fingerprint = compute_base_table_fingerprint(negt);
        assert(!bind_base_landmarks(fresh.data(), fresh.size(), negt, &nb, nullptr));
    }
    // Empty middle chunk fails closed.
    {
        std::vector<uint32_t> eo = {0, 0, 8};
        landmark_base_table et = table;
        et.chunk_row_offsets = eo.data();
        et.bounds_fingerprint = compute_base_table_fingerprint(et);
        assert(!bind_base_landmarks(fresh.data(), fresh.size(), et, &nb, nullptr));
    }
    // Final offset must equal n_rows_total exactly.
    {
        std::vector<uint32_t> fo = {0, 4, 7};
        landmark_base_table ft = table;
        ft.chunk_row_offsets = fo.data();
        ft.bounds_fingerprint = compute_base_table_fingerprint(ft);
        assert(!bind_base_landmarks(fresh.data(), fresh.size(), ft, &nb, nullptr));
    }
    // Landmark rows must equal n_chunks exactly (2-row base, 1-chunk claim).
    {
        landmark_base_table rc = table;
        rc.n_chunks = 1;
        rc.bounds_fingerprint = compute_base_table_fingerprint(rc);
        assert(!bind_base_landmarks(fresh.data(), fresh.size(), rc, &nb, nullptr));
    }
    // Fragments owning bytes are left untouched (seal path uses attach instead).
    nb = 999;
    assert(bind_base_landmarks(ref.data(), ref.size(), table, &nb, nullptr));
    assert(nb == 0);
    assert(!ref[0].uses_base_landmark());
    std::cout << "[test_base_table_bind] passed!" << std::endl;
}

// Defined after main to keep the runner call list readable.
static void test_quant_row_encoder();
static void test_bounded_builder_q8();
static void test_landmark_advanced_lifecycle();

int main() {
    std::cout << "Running test-xkv-landmark..." << std::endl;
    test_fragment_partitioning();
    test_multi_query_isolation_and_4field_refs();
    test_keeper_dedup_and_mandatory_future_rejection();
    test_gqa_max_pooling();
    test_landmark_quantization_recall();
    test_invalid_bounds_and_codec_rejection();
    test_landmark_table_lease_protection();
    test_normalized_boundary_refinement();

    // Test 9: Immediate epoch & stamp invalidation in landmark_table
    {
        std::cout << "[test_landmark_table_invalidation] starting..." << std::endl;
        landmark_table table(1024 * 1024);
        auto seg = create_test_segment(8, 16, 64, 1);
        xkv_snapshot_stamp stamp;
        stamp.live_epoch = 10;
        stamp.content_epoch = 20;
        stamp.codec_epoch = 30;

        std::vector<row_meta> rows = {
            {0, 1000, 1, 0, 0, 0, static_cast<uint8_t>(llama_rerot_visibility::normal), true, true},
        };
        auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x1, -1);
        assert(!frags.empty());
        encode_fragment_landmark(frags[0], *seg, 0, 32, nullptr, 0x1, GGML_TYPE_Q8_0);

        auto frag_ptr = std::make_shared<const legal_fragment>(std::move(frags[0]));
        assert(table.insert(frag_ptr));
        assert(table.fragment_count() == 1);

        // Invalidate by content_epoch
        table.invalidate_content_epoch(seg->segment_id, 999); // Stale content epoch
        assert(table.fragment_count() == 0);

        // Re-insert and test invalidate_stamp
        assert(table.insert(frag_ptr));
        assert(table.fragment_count() == 1);

        xkv_snapshot_stamp new_stamp = stamp;
        new_stamp.live_epoch = 11; // Live epoch advanced
        table.invalidate_stamp(new_stamp);
        assert(table.fragment_count() == 0);
        std::cout << "[test_landmark_table_invalidation] passed!" << std::endl;
    }

    // Test 10: Strict weak ordering and cyclic near-epsilon score regression
    {
        std::cout << "[test_strict_weak_ordering_and_tie_breaking] starting..." << std::endl;
        auto seg = create_test_segment(16, 16, 64, 1);
        xkv_snapshot_stamp stamp;
        stamp.live_epoch = 1;

        std::vector<row_meta> rows;
        for (uint32_t i = 0; i < 16; ++i) {
            rows.push_back({i, 1000 + i, 1, 0, (int64_t)i, (int64_t)i, static_cast<uint8_t>(llama_rerot_visibility::normal), true, true});
        }
        auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 2, 0x1, -1);
        assert(frags.size() == 8);
        for (auto & f : frags) {
            encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0x1, GGML_TYPE_Q8_0);
        }

        sr_query q;
        q.head_dim = 32;
        q.q_vec.resize(32, 0.0f); // Zero query generates exact identical raw scores across all fragments
        q.scale = 1.0f;

        sr_selection_config cfg;
        cfg.sr_budget = 4;

        // Must not crash or cycle and must produce deterministic strictly increasing order by frag_index on tie
        auto res = select_sr_query(q, frags, cfg, seg.get(), 0, 32, nullptr, 0x1);
        assert(res.selected_rows.size() == 8); // 4 chunks * 2 rows
        std::cout << "[test_strict_weak_ordering_and_tie_breaking] passed!" << std::endl;
    }
    test_workspace_bytes_and_short_refusal();
    test_planner_parity();
    test_bounded_parity_all_types();
    test_zero_heap_production();
    test_partial_causal_bounded();
    test_variable_group_slices();
    test_future_draft_isolation_bounded();
    test_shared_prefix_csr();
    test_nan_zero_query();
    test_phase_stamp_invalidation();
    test_lru_pinned_atomic();
    test_seed_source_fidelity();
    test_refine_bounded_parity();
    test_budget_all_equals_dense();
    test_multisegment_attribution();
    test_head_dim_mismatch_guards();
    test_base_landmark_reference();
    test_bounded_builder();
    test_bounded_csr_groups();
    test_fault_atomic_rescore();
    test_zero_fragment_hard_keeps();
    test_quant_row_encoder();
    test_bounded_builder_q8();
    test_base_table_bind();
    test_landmark_advanced_lifecycle();

    std::cout << "All test-xkv-landmark tests passed successfully!" << std::endl;
    return 0;
}
static void test_quant_row_encoder() {
    std::cout << "[test_quant_row_encoder] starting..." << std::endl;
    const uint32_t dims[] = {1, 7, 31, 32, 33, 63, 64, 100, 127, 128, 129, 200, 255, 256};
    const ggml_type types[] = {GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0};
    const uint64_t seed = 0x1234ABCDu;
    for (uint32_t dim : dims) {
        std::vector<float> mean(dim);
        for (uint32_t i = 0; i < dim; ++i) {
            float v = (float) ((int) (i * 37u % 101u) - 50) * 0.11f;
            mean[i] = (i % 9 == 0) ? 0.0f : v; // zeros, signs, magnitudes
        }
        for (ggml_type t : types) {
            uint32_t group = (t == GGML_TYPE_TURBO4_0) ? 128 : 0;
            codec_desc ref_desc =
                make_codec_desc(factor_role::landmark, t, orientation::token_major, {1, dim}, group, seed);
            encoded_matrix ref = encode_matrix(ref_desc, mean.data(), mean.size());
            size_t nq = 0;
            codec_desc qd;
            float eb = 0.0f;
            assert(landmark_encode_quant_row(nullptr, dim, t, seed, nullptr, 0, &nq, &qd, &eb, nullptr));
            assert(nq == ref.bytes.size());
            assert(qd == ref_desc);
            assert(qd.fingerprint() == ref_desc.fingerprint());
            std::vector<uint8_t> out(nq, 0xAA);
            assert(landmark_encode_quant_row(mean.data(), dim, t, seed, out.data(), out.size(),
                &nq, &qd, &eb, nullptr));
            assert(nq == ref.bytes.size());
            assert(out == ref.bytes); // bit-identical to encode_matrix
            encoded_matrix mine;
            mine.desc = qd;
            mine.bytes = out;
            assert(decode_matrix(mine, value_domain::canonical) ==
                   decode_matrix(ref, value_domain::canonical));
            // Conservative bound: independent actual L2 over logical dim, same order.
            {
                std::vector<float> dec = decode_matrix(mine, value_domain::canonical);
                double ss = 0.0;
                for (uint32_t d = 0; d < dim; ++d) {
                    double diff = (double) dec[d] - (double) mean[d];
                    ss += diff * diff;
                }
                double actual = std::sqrt(ss);
                assert(actual <= (double) eb); // 10% margin + floor make this exact-safe
                if (t == GGML_TYPE_F32) {
                    assert(eb == 0.0f && actual == 0.0);
                } else {
                    assert(eb > 0.0f); // floor convention: never a silent zero bound
                }
            }
            if (nq > 0) { // exact T-1 refusal
                std::vector<uint8_t> short_out(nq - 1, 0);
                size_t nn = 0;
                codec_desc cd2;
                float eb2 = 0.0f;
                assert(!landmark_encode_quant_row(mean.data(), dim, t, seed, short_out.data(),
                    short_out.size(), &nn, &cd2, &eb2, nullptr));
            }
        }
    }
    std::vector<float> m32(32, 0.5f);
    std::vector<uint8_t> obuf(256, 0);
    // Provably lossy row: amax = 1.0 -> d = 1/127, and 0.1/d = 12.7 rounds to 13.
    {
        std::vector<float> lossy(32, 0.0f);
        lossy[0] = 0.1f;
        lossy[1] = 1.0f;
        for (ggml_type t : {GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0}) {
            size_t n = 0;
            codec_desc cd;
            float eb = 0.0f;
            assert(landmark_encode_quant_row(lossy.data(), 32, t, seed, nullptr, 0, &n,
                &cd, &eb, nullptr));
            std::vector<uint8_t> ob(n, 0);
            assert(landmark_encode_quant_row(lossy.data(), 32, t, seed, ob.data(), ob.size(),
                &n, &cd, &eb, nullptr));
            assert(eb > 0.0f); // nonzero bound, never silent zero
            encoded_matrix em;
            em.desc = cd;
            em.bytes = ob;
            std::vector<float> dec = decode_matrix(em, value_domain::canonical);
            double ss = 0.0;
            for (int d = 0; d < 32; ++d) {
                double diff = (double) dec[d] - (double) lossy[d];
                ss += diff * diff;
            }
            double actual = std::sqrt(ss);
            assert(actual <= (double) eb); // actual error within the reported bound
        }
        // Q8 lossiness is provable here (0.1 mis-rounds at d = 1/127), so the bound
        // must be strictly driven by real error, not just the floor.
        size_t n = 0;
        codec_desc cd;
        float eb = 0.0f;
        std::vector<uint8_t> ob(64, 0);
        assert(landmark_encode_quant_row(lossy.data(), 32, GGML_TYPE_Q8_0, seed, ob.data(),
            ob.size(), &n, &cd, &eb, nullptr));
        ob.resize(n);
        encoded_matrix em;
        em.desc = cd;
        em.bytes = ob;
        std::vector<float> dec = decode_matrix(em, value_domain::canonical);
        double ss = 0.0;
        for (int d = 0; d < 32; ++d) {
            double diff = (double) dec[d] - (double) lossy[d];
            ss += diff * diff;
        }
        assert(std::sqrt(ss) > 0.0); // genuinely lossy row
        assert(eb >= std::sqrt(ss)); // bound covers it (with margin)
    }
    for (ggml_type t : {GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0}) {
        size_t n = 0;
        codec_desc cd;
        float eb = 0.0f;
        assert(!landmark_encode_quant_row(m32.data(), 32, t, seed, obuf.data(), obuf.size(),
            &n, &cd, &eb, nullptr));
        assert(!landmark_encode_quant_row(nullptr, 32, t, seed, nullptr, 0, &n, &cd, &eb, nullptr));
    }
    {
        size_t n = 0;
        codec_desc cd;
        float eb = 0.0f;
        assert(!landmark_encode_quant_row(nullptr, 32, GGML_TYPE_Q8_0, seed, obuf.data(),
            obuf.size(), &n, &cd, &eb, nullptr)); // null mean with real output
        assert(!landmark_encode_quant_row(m32.data(), 0, GGML_TYPE_Q8_0, seed, obuf.data(),
            obuf.size(), &n, &cd, &eb, nullptr)); // zero dim
    }
    std::cout << "[test_quant_row_encoder] passed!" << std::endl;
}
static void test_bounded_builder_q8() {
    std::cout << "[test_bounded_builder_q8] starting..." << std::endl;
    auto seg = create_test_segment(10, 16, 64, 1);
    const encoded_matrix * a_k = &seg->groups[0].a_k;
    const encoded_matrix * b_k = seg->groups[0].b_k.get();
    std::vector<uint32_t> srows(10);
    std::vector<int64_t> spos(10);
    for (uint32_t i = 0; i < 10; ++i) {
        srows[i] = i;
        spos[i] = (int64_t) i;
    }
    landmark_build_layer layers[2] = {{0, 0, 32, 0}, {0, 32, 32, 0}};
    landmark_build_spec spec;
    spec.a_k = a_k;
    spec.b_k = b_k;
    spec.surviving_rows = srows.data();
    spec.n_rows = 10;
    spec.storage_positions = spos.data();
    spec.layers = layers;
    spec.n_layers = 2;
    spec.chunk_tokens = 4;
    spec.landmark_type = GGML_TYPE_Q8_0;
    spec.phase_tx_fingerprint = 0;
    spec.seed_extra = 0xBEEF;
    size_t ws = 0, ar = 0, nc = 0;
    assert(landmark_build_workspace_bytes(spec, landmark_encode_quant_row, ws, ar, nc, nullptr));
    assert(nc == 3);
    std::vector<uint8_t> scratch(ws, 0);
    std::vector<uint8_t> arena(ar, 0);
    std::vector<landmark_build_output> outs(3);
    size_t no = 0;
    assert(build_landmarks_bounded(spec, nullptr, landmark_encode_quant_row, scratch.data(),
        scratch.size(), arena.data(), arena.size(), outs.data(), outs.size(), &no, nullptr));
    assert(no == 3);
    for (int c = 0; c < 3; ++c) {
        assert(outs[c].byte_size > 0);
        assert(std::isfinite(outs[c].error_bound));
        assert(outs[c].error_bound >= 1e-4f); // measured Q8 bound under the floor convention
        assert(outs[c].desc.type == GGML_TYPE_Q8_0);
    }
    assert(outs[1].byte_offset == outs[0].byte_offset + outs[0].byte_size);
    // Zero production heap after init (streamed decode + quant encode, identity phase).
    g_alloc_count = 0;
    {
        alloc_scope on(true);
        for (int rep = 0; rep < 3; ++rep) {
            size_t n = 0;
            bool ok = build_landmarks_bounded(spec, nullptr, landmark_encode_quant_row,
                scratch.data(), scratch.size(), arena.data(), arena.size(), outs.data(),
                outs.size(), &n, nullptr);
            assert(ok && n == 3);
        }
    }
    assert(g_alloc_count == 0);
    // Parity: single-slice Q8 builder chunks equal per-fragment Q8 landmarks bit-exactly.
    {
        landmark_build_layer one[1] = {{0, 0, 32, 0}};
        landmark_build_spec s1 = spec;
        s1.layers = one;
        s1.n_layers = 1;
        std::vector<uint32_t> r8(8);
        std::vector<int64_t> p8(8);
        for (uint32_t i = 0; i < 8; ++i) {
            r8[i] = i;
            p8[i] = (int64_t) i;
        }
        s1.surviving_rows = r8.data();
        s1.n_rows = 8;
        s1.storage_positions = p8.data();
        size_t w1 = 0, a1 = 0, n1 = 0;
        assert(landmark_build_workspace_bytes(s1, landmark_encode_quant_row, w1, a1, n1, nullptr));
        assert(n1 == 2);
        std::vector<uint8_t> sc1(w1, 0), ar1(a1, 0);
        std::vector<landmark_build_output> o1(2);
        size_t nn = 0;
        assert(build_landmarks_bounded(s1, nullptr, landmark_encode_quant_row, sc1.data(),
            sc1.size(), ar1.data(), ar1.size(), o1.data(), o1.size(), &nn, nullptr));
        assert(nn == 2);
        xkv_snapshot_stamp stamp;
        stamp.live_epoch = 1;
        std::vector<row_meta> mrows;
        for (uint32_t i = 0; i < 8; ++i) {
            mrows.push_back({i, 1000 + i, 1, 0, (int64_t) i, (int64_t) i, 0, true, true});
        }
        auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, mrows, 4, 0, -1);
        assert(frags.size() == 2);
        for (auto & f : frags) encode_fragment_landmark(f, *seg, 0, 32, nullptr, 0, GGML_TYPE_Q8_0);
        for (int c = 0; c < 2; ++c) {
            assert(o1[c].byte_size == frags[c].landmark_matrix.bytes.size());
            const uint8_t * got = ar1.data() + o1[c].byte_offset;
            assert(0 == std::memcmp(got, frags[c].landmark_matrix.bytes.data(), o1[c].byte_size));
            assert(o1[c].source_fingerprint == frags[c].source_fingerprint);
        }
    }
    std::cout << "[test_bounded_builder_q8] passed!" << std::endl;
}

static void test_landmark_advanced_lifecycle() {
    std::cout << "[test_landmark_advanced_lifecycle] starting..." << std::endl;
    // 1. Invalidation: phase transform fingerprint mismatch, source fingerprint mismatch,
    // binding epoch, view stamp, and codec epoch.
    landmark_table tbl(1024 * 1024);
    auto seg = create_test_segment(8, 16, 64, 1);
    // Segment-level source identity: the table key carries this (not the
    // encoded content fingerprint), so the fixture must set it explicitly.
    seg->source_fingerprint = 0x5EED;
    xkv_snapshot_stamp stamp;
    stamp.live_epoch = 1;
    stamp.content_epoch = 1;
    stamp.codec_epoch = 1;
    stamp.binding_epoch = 1;
    stamp.view.topology_epoch = 1;
    stamp.view.publish_epoch = 1;
    stamp.view.layout_epoch = 1;

    std::vector<row_meta> rows = {
        {0, 1000, 1, 0, 0, 0, 0, static_cast<uint8_t>(llama_rerot_visibility::normal), true, true},
    };
    auto frags = build_legal_fragments(*seg, stamp, 1, 1, 0, 0, rows, 4, 0x111, -1);
    assert(!frags.empty());
    encode_fragment_landmark(frags[0], *seg, 0, 32, nullptr, 0x111, GGML_TYPE_Q8_0);
    uint64_t orig_src_fp = frags[0].source_fingerprint;
    auto fp0 = std::make_shared<const legal_fragment>(frags[0]);
    assert(tbl.insert(fp0));
    assert(tbl.fragment_count() == 1);

    // Phase/source invalidation keys on the fragment key identity: matching
    // phase + matching key source keeps the entry.
    tbl.invalidate_phase_source(seg->segment_id, 0x111, frags[0].key.source_fingerprint);
    assert(tbl.fragment_count() == 1);
    // Mismatch source evicts even when the phase matches.
    tbl.invalidate_phase_source(seg->segment_id, 0x111, frags[0].key.source_fingerprint ^ 0xFFu);
    assert(tbl.fragment_count() == 0);
    // Re-insert: mismatch phase evicts even when the source matches.
    assert(tbl.insert(fp0));
    assert(tbl.fragment_count() == 1);
    tbl.invalidate_phase_source(seg->segment_id, 0x999, frags[0].key.source_fingerprint);
    assert(tbl.fragment_count() == 0);

    // Re-insert and test binding epoch invalidation
    assert(tbl.insert(fp0));
    assert(tbl.fragment_count() == 1);
    tbl.invalidate_binding_epoch(seg->segment_id, 999);
    assert(tbl.fragment_count() == 0);

    // Re-insert and test view stamp invalidation
    assert(tbl.insert(fp0));
    assert(tbl.fragment_count() == 1);
    xkv_snapshot_stamp diff_view = stamp;
    diff_view.view.topology_epoch = 999;
    tbl.invalidate_view(seg->segment_id, diff_view);
    assert(tbl.fragment_count() == 0);

    // Re-insert and test codec epoch invalidation
    assert(tbl.insert(fp0));
    assert(tbl.fragment_count() == 1);
    tbl.invalidate_codec_epoch(seg->segment_id, 999);
    assert(tbl.fragment_count() == 0);

    // 2. DEVICE_OWNED empty-host-bytes intact landmark binding
    codec_desc bdesc = make_codec_desc(factor_role::landmark, GGML_TYPE_TURBO4_0,
        orientation::token_major, {1, 32}, 128, 0x777);
    auto empty_base = std::make_shared<encoded_matrix>();
    empty_base->desc = bdesc;
    assert(empty_base->bytes.empty());

    std::vector<uint64_t> pids = {1000};
    std::vector<uint64_t> gens = {1};
    std::vector<int64_t> poss = {0};
    std::vector<uint32_t> coffs = {0, 1};
    float eb[1] = {0.02f};
    uint64_t sfp[1] = {orig_src_fp};

    landmark_base_table btab;
    btab.landmark = empty_base;
    btab.row_payload_ids = pids.data();
    btab.row_generations = gens.data();
    btab.row_positions = poss.data();
    btab.chunk_error_bounds = eb;
    btab.chunk_source_fingerprints = sfp;
    btab.chunk_row_offsets = coffs.data();
    btab.n_rows_total = 1;
    btab.n_chunks = 1;
    btab.stamp = stamp;
    btab.phase_tx_fingerprint = 0x111;
    btab.bounds_fingerprint = compute_base_table_fingerprint(btab);

    legal_fragment intact_frag;
    intact_frag.row_count = 1;
    intact_frag.row_indices = {0};
    intact_frag.payload_ids = {1000};
    intact_frag.generations = {1};
    intact_frag.storage_positions = {0};
    intact_frag.key.live_epoch = 1; intact_frag.key.content_epoch = 1; intact_frag.key.codec_epoch = 1;
    intact_frag.key.binding_epoch = 1; intact_frag.key.view_topology_epoch = 1;
    intact_frag.key.view_publish_epoch = 1; intact_frag.key.view_layout_epoch = 1;
    intact_frag.key.phase_tx_fingerprint = 0x111;

    size_t nb = 0;
    assert(bind_base_landmarks(&intact_frag, 1, btab, &nb, nullptr));
    assert(nb == 1);
    assert(intact_frag.uses_base_landmark());
    assert(intact_frag.error_bound == eb[0]);
    assert(intact_frag.source_fingerprint == orig_src_fp);
    assert(intact_frag.key.landmark_codec_fp == bdesc.fingerprint());
    assert(intact_frag.landmark_matrix.bytes.empty());

    // 3. Turbo4 top-k scoring behavior
    std::vector<float> mean(32, 0.4f);
    legal_fragment t4_frag;
    t4_frag.row_count = 1;
    t4_frag.row_indices = {0};
    t4_frag.storage_positions = {0};
    t4_frag.payload_ids = {1000};
    t4_frag.generations = {1};
    t4_frag.key.storage_generation = 1;
    t4_frag.key.storage_pos0 = 0;
    encode_fragment_landmark(t4_frag, *seg, 0, 32, nullptr, 0x111, GGML_TYPE_TURBO4_0);
    assert(t4_frag.landmark_matrix.desc.type == GGML_TYPE_TURBO4_0);
    assert(t4_frag.error_bound > 0.0f);
    assert(t4_frag.source_fingerprint != 0);

    sr_query q;
    q.head_dim = 32;
    q.q_vec.assign(32, 0.5f);
    sr_selection_config cfg;
    cfg.sr_budget = 1;
    cfg.landmark_type = GGML_TYPE_TURBO4_0;
    auto res = select_sr_query(q, {t4_frag}, cfg, seg.get(), 0, 32, nullptr, 0x111);
    assert(res.selected_rows.size() == 1);
    assert(res.selected_rows[0].row == 0);
    assert(res.scores.size() == 1);
    assert(std::isfinite(res.scores[0]));

    std::cout << "[test_landmark_advanced_lifecycle] passed!" << std::endl;
}

