// Focused regression: the real bounded seal path retains no decoded mirror.
// Runs seal_segment_bundle over canonical staging, drops every source/staging
// copy, then proves the absence of retained decoded A/B/KV/landmark matrices
// with store-independent observations only:
//   (a) store arena live bytes return to zero (no held staging leases);
//   (b) a strict-workspace read performs exactly one heap allocation (the
//       accounted output) — reconstruction happens on demand from code streams;
//   (c) weak_ptr witnesses to the published segment and its shared B streams
//       expire after full payload removal + reclamation (nothing else retains
//       decoded or encoded artifacts).
// Store accounting equality is kept as a secondary consistency check, never as
// the mirror proof (self-reported counters cannot detect uncounted mirrors).
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-cache.h"
#include "llama-xkv-codec.h"
#include "llama-xkv-reader.h"
#include "llama-cparams.h"
#include "ggml.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <vector>

using namespace llama_xkv;

// Heap allocation counter: observes C++ allocations inside a scope,
// independent of any store-reported counter.
namespace {
std::atomic<size_t> g_heap_new_count{0};
thread_local bool g_heap_counting = false;
struct heap_scope {
    heap_scope() {
        g_heap_new_count.store(0, std::memory_order_relaxed);
        g_heap_counting = true;
    }
    ~heap_scope() { g_heap_counting = false; }
    size_t count() const { return g_heap_new_count.load(std::memory_order_relaxed); }
};
} // namespace
void * operator new(std::size_t n) {
    if (g_heap_counting) {
        g_heap_new_count.fetch_add(1, std::memory_order_relaxed);
    }
    if (void * p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }
void * operator new[](std::size_t n) {
    if (g_heap_counting) {
        g_heap_new_count.fetch_add(1, std::memory_order_relaxed);
    }
    if (void * p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete[](void * p, std::size_t) noexcept { std::free(p); }

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

// Deterministic xorshift64 PRNG (uniform [-1,1]) for genuine low-rank fixtures.
static uint64_t xkv_mirror_rng_next(uint64_t & s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

// Genuine rank-true_rank fixture: K = U.V^T + noise, V = 0.5.K.
// All true_rank components carry signal, so a rank-true_rank seal measures
// real factorization/quantization error at the quality gate. (A sin/cos outer
// product is exactly rank-1 and leaves degenerate zero-signal columns.)
static void build_mirror_low_rank(uint64_t rows, uint64_t cols, uint32_t true_rank,
                                  float noise_scale, uint64_t seed,
                                  std::vector<float> & k_out, std::vector<float> & v_out) {
    uint64_t s = seed ? seed : 0x243F6A8885A308D3ULL;
    auto rnd = [&]() -> float {
        xkv_mirror_rng_next(s);
        return ((float)(s & 0x7FFFFFFF) / (float)0x7FFFFFFF) * 2.0f - 1.0f;
    };
    std::vector<float> u((size_t)rows * true_rank), vv((size_t)cols * true_rank);
    for (float & x : u) x = rnd();
    for (float & x : vv) x = rnd();
    k_out.assign((size_t)rows * cols, 0.0f);
    for (uint64_t r = 0; r < rows; ++r) {
        for (uint64_t c = 0; c < cols; ++c) {
            double sum = 0.0;
            for (uint32_t k = 0; k < true_rank; ++k) {
                sum += (double)u[(size_t)r * true_rank + k] * (double)vv[(size_t)c * true_rank + k];
            }
            k_out[(size_t)r * cols + c] = (float)sum;
        }
    }
    if (noise_scale > 0.0f) {
        for (float & x : k_out) x += rnd() * noise_scale;
    }
    v_out.resize((size_t)rows * cols);
    for (size_t i = 0; i < v_out.size(); ++i) v_out[i] = k_out[i] * 0.5f;
}

// Exact measured source row bytes per owning layer: ggml_row_size over the
// live hot tensor type/ne for each layer feature width. The types must match
// the seal params' flat_type_k/v (F16 in every seal below); never hand-entered.
static void fill_hot_row_bytes(xkv_factor_group_input & g, ggml_type type_k, ggml_type type_v) {
    g.hot_bytes_per_row_k.clear();
    g.hot_bytes_per_row_v.clear();
    for (uint32_t d : g.layer_feature_dims_k) {
        g.hot_bytes_per_row_k.push_back((uint64_t) ggml_row_size(type_k, (int64_t) d));
    }
    for (uint32_t d : g.layer_feature_dims_v) {
        g.hot_bytes_per_row_v.push_back((uint64_t) ggml_row_size(type_v, (int64_t) d));
    }
}

int main() {
    std::cout << "=== test-xkv-no-dense-mirror ===" << std::endl;

    llama_xkv_cache_store store(test_cparams());

    // Real bounded seal path: canonical staging -> factorize -> encode -> gate.
    const uint32_t n_tokens = 256;
    const uint32_t total_dim = 64;
    const uint32_t rank = 8;

    std::vector<uint64_t> pids(n_tokens);
    std::vector<uint64_t> gens(n_tokens, 1);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        pids[i] = 5000 + i;
        CHECK(store.register_hot_payload(pids[i], i, gens[i], xkv_state::hot_committed));
    }

    std::vector<float> k_data((size_t) n_tokens * total_dim);
    std::vector<float> v_data((size_t) n_tokens * total_dim);
    // Genuine rank-8 data (true_rank == requested rank): passes the quality
    // gate on real factorization/quantization error (~few %).
    build_mirror_low_rank(n_tokens, total_dim, rank, 0.001f, 1001, k_data, v_data);

    xkv_factor_group_input g0;
    g0.group_index = 0;
    g0.owning_layers = {0, 1, 2, 3};
    g0.rank_k = rank;
    g0.rank_v = rank;
    g0.total_dim_k = total_dim;
    g0.total_dim_v = total_dim;
    g0.layer_feature_offsets_k = {0, 16, 32, 48};
    g0.layer_feature_dims_k = {16, 16, 16, 16};
    g0.layer_feature_offsets_v = {0, 16, 32, 48};
    g0.layer_feature_dims_v = {16, 16, 16, 16};
    g0.canonical_k_data = k_data.data();
    g0.k_rows = n_tokens;
    g0.k_cols = total_dim;
    g0.canonical_v_data = v_data.data();
    g0.v_rows = n_tokens;
    g0.v_cols = total_dim;
    g0.row_positions.resize(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) g0.row_positions[i] = (int64_t) i;
    fill_hot_row_bytes(g0, GGML_TYPE_F16, GGML_TYPE_F16);

    xkv_bundle_sealing_params sparams;
    sparams.profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
    sparams.source = LLAMA_XKV_SOURCE_DECODED_HOT;
    sparams.rank_k = rank;
    sparams.rank_v = rank;
    sparams.max_relative_error = 0.99;
    sparams.min_saving_ratio = 0.10;
    sparams.flat_type_k = GGML_TYPE_F16;
    sparams.flat_type_v = GGML_TYPE_F16;

    xkv_sealing_result res = store.seal_segment_bundle({g0}, pids, gens, sparams);
    if (!res.success) {
        std::fprintf(stderr, "no-dense-mirror base seal failed: %s (skip=%d)\n", res.message.c_str(), (int) res.skip_reason);
        return 1;
    }
    CHECK(res.success);
    CHECK(res.segment_id > 0);
    CHECK(res.release_plan.dense_bytes_freed > 0); // seal freed the dense source
    const uint64_t seg_id = res.segment_id;

    // Drop every test-side source/staging copy: the store must stand alone on
    // its published code streams from here on.
    k_data.clear();
    k_data.shrink_to_fit();
    v_data.clear();
    v_data.shrink_to_fit();
    CHECK(k_data.capacity() == 0);
    CHECK(v_data.capacity() == 0);

    auto seg = store.get_segment(seg_id);
    CHECK(seg != nullptr);
    if (seg == nullptr) return 1;
    CHECK(seg->n_live_rows == n_tokens);

    // (a) No held staging leases after the real seal path completes.
    CHECK(store.get_arena().get_live_bytes() == 0);

    // Secondary consistency only: accounting matches the segment's own bytes.
    {
        const xkv_accounting acc = store.get_accounting();
        CHECK(acc.active_segments == 1);
        CHECK(acc.allocated_bytes == seg->total_allocated_bytes);
        CHECK(acc.factored_bytes == seg->total_allocated_bytes);
    }

    // (b) Strict-workspace read over all sealed rows: exactly one heap
    // allocation (the accounted output). Views/query are built outside the
    // measured scope; the read itself must reconstruct from code streams
    // with zero auxiliary heap (no retained-mirror shortcut, no temp vectors).
    xkv_reader_workspace ws;
    xkv_reader_workspace_config ws_cfg; // 16 MiB default cap covers these maxima
    std::string werr;
    CHECK(ws.warmup(ws_cfg, &werr));
    xkv_query_input query;
    query.query_index = 0;
    query.n_q_heads = 1;
    query.head_dim_k = 16;
    query.head_dim_v = 16;
    query.q_vec.assign(16, 0.1f);
    query.scale = 1.0f / std::sqrt(16.0f);
    {
        xkv_segment_read_view view;
        view.pin = store.pin_segment(seg_id);
        CHECK(bool(view.pin));
        view.segment_version_id = view.pin->segment_version;
        view.storage_generation = 1;
        view.owning_layer = 0;
        view.factor_group_index = 0;
        for (uint32_t i = 0; i < n_tokens; ++i) {
            view.selected_rows.push_back(i);
            view.storage_positions.push_back((int64_t) i);
            view.group_indices.push_back(0);
        }
        std::vector<xkv_segment_read_view> views;
        views.push_back(std::move(view));
        const xkv_snapshot_stamp stamp = store.current_stamp();
        xkv_reader_config rcfg;
        rcfg.workspace = &ws;
        {
            heap_scope scope;
            xkv_read_result r = xkv_read_attention(query, {}, views, nullptr, stamp, &store, rcfg);
            CHECK(r.status == xkv_read_status::success);
            CHECK(r.output.size() == 16);
            for (float v : r.output) {
                CHECK(std::isfinite(v));
            }
            CHECK(scope.count() == 1);
        }
        CHECK(ws.live_bytes() == 0);
    }
    CHECK(store.get_segment(seg_id)->pin_count.load() == 0);

    // (c) Lifetime proof: weak witnesses to the segment and its shared B
    // streams must expire once every payload is removed and retired versions
    // are reclaimed — no hidden cache retains decoded or encoded artifacts.
    std::weak_ptr<const xkv_segment> wseg;
    std::weak_ptr<const encoded_matrix> wbk;
    std::weak_ptr<const encoded_matrix> wbv;
    {
        auto h = store.get_segment(seg_id);
        CHECK(h != nullptr);
        wseg = h;
        wbk = h->groups[0].b_k;
        wbv = h->groups[0].b_v;
        CHECK(!wseg.expired());
    }
    seg.reset(); // drop the test's own handle: only the store may retain now
    {
        std::string err;
        for (uint64_t pid : pids) {
            CHECK(store.remove_payload(pid, &err));
        }
    }
    store.reclaim_retired_segments();
    CHECK(wseg.expired());
    CHECK(wbk.expired());
    CHECK(wbv.expired());
    {
        const xkv_accounting acc = store.get_accounting();
        CHECK(acc.active_segments == 0);
        CHECK(acc.allocated_bytes == 0);
    }

    // ---- Live/reserved separation on a fresh seal (store-independent + accounting) ----
    {
        llama_xkv_cache_store store2(test_cparams());
        // 192 rows: fixed B-stream cost amortizes to ~29% saving, clearing the
        // 10% saving gate. (32 rows cannot: fixed B bytes exceed the flat source.)
        const uint32_t n2 = 192;
        std::vector<uint64_t> p2(n2);
        std::vector<uint64_t> g2(n2, 1);
        for (uint32_t i = 0; i < n2; ++i) {
            p2[i] = 8000 + i;
            CHECK(store2.register_hot_payload(p2[i], i, g2[i], xkv_state::hot_committed));
        }
        std::vector<float> k2((size_t) n2 * total_dim), v2((size_t) n2 * total_dim);
        build_mirror_low_rank(n2, total_dim, rank, 0.001f, 1002, k2, v2);
        xkv_factor_group_input gi;
        gi.group_index = 0;
        gi.owning_layers = {0, 1, 2, 3};
        gi.rank_k = rank;
        gi.rank_v = rank;
        gi.total_dim_k = total_dim;
        gi.total_dim_v = total_dim;
        gi.layer_feature_offsets_k = {0, 16, 32, 48};
        gi.layer_feature_dims_k = {16, 16, 16, 16};
        gi.layer_feature_offsets_v = {0, 16, 32, 48};
        gi.layer_feature_dims_v = {16, 16, 16, 16};
        gi.canonical_k_data = k2.data();
        gi.k_rows = n2;
        gi.k_cols = total_dim;
        gi.canonical_v_data = v2.data();
        gi.v_rows = n2;
        gi.v_cols = total_dim;
        gi.row_positions.resize(n2);
        for (uint32_t i = 0; i < n2; ++i) gi.row_positions[i] = (int64_t) i;
        fill_hot_row_bytes(gi, GGML_TYPE_F16, GGML_TYPE_F16);
        xkv_bundle_sealing_params sp;
        sp.profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
        sp.source = LLAMA_XKV_SOURCE_DECODED_HOT;
        sp.rank_k = rank;
        sp.rank_v = rank;
        sp.max_relative_error = 0.99;
        sp.min_saving_ratio = 0.10;
        sp.flat_type_k = GGML_TYPE_F16;
        sp.flat_type_v = GGML_TYPE_F16;
        xkv_sealing_result r2 = store2.seal_segment_bundle({gi}, p2, g2, sp);
        if (!r2.success) {
            std::fprintf(stderr, "no-dense-mirror live/reserved seal failed: %s (skip=%d)\n", r2.message.c_str(), (int) r2.skip_reason);
            return 1;
        }
        CHECK(r2.success);
        auto s2 = store2.get_segment(r2.segment_id);
        CHECK(s2 != nullptr);
        if (s2 == nullptr) return 1;
        // No held staging leases: live arena bytes are zero while reserved capacity stays.
        CHECK(store2.get_arena().get_live_bytes() == 0);
        CHECK(store2.get_arena().get_capacity_bytes() == (size_t) 16 * 1024 * 1024);
        const xkv_accounting acc = store2.get_accounting();
        CHECK(acc.active_segments == 1);
        CHECK(acc.allocated_bytes == s2->total_allocated_bytes);
        CHECK(acc.arena_live_bytes == 0);
        CHECK(acc.reserved_bytes == acc.allocated_bytes + acc.arena_reserved_bytes + acc.dedup_scratch_bytes);
        CHECK(acc.workspace_budget_bytes == acc.arena_capacity_bytes);
        CHECK(acc.arena_capacity_bytes == (size_t) 16 * 1024 * 1024);
    }

    // ---- Multi-group tail (full 8-layer group + 1-layer tail), non-divisible 100 rows ----
    {
        llama_xkv_cache_store store3(test_cparams());
        const uint32_t n3 = 100; // not divisible by 64/32/8: exercises tail tiles
        // 128-wide full group (8x16 layers): two A-stream sets cost ~272 B/row
        // against a ~576 B/row flat source, so the 10% gate passes (~50%).
        // A 64+16 two-group shape cannot amortize the fixed B-stream cost.
        const uint32_t dim3 = 128;
        std::vector<uint64_t> p3(n3);
        std::vector<uint64_t> g3(n3, 1);
        for (uint32_t i = 0; i < n3; ++i) {
            p3[i] = 9000 + i;
            CHECK(store3.register_hot_payload(p3[i], i, g3[i], xkv_state::hot_committed));
        }
        std::vector<float> k3((size_t) n3 * dim3), v3((size_t) n3 * dim3);
        build_mirror_low_rank(n3, dim3, rank, 0.001f, 1003, k3, v3);
        auto mk_input = [&](uint32_t gidx, std::vector<uint32_t> owners,
                              std::vector<uint32_t> offs, std::vector<uint32_t> dims, uint32_t total) {
            xkv_factor_group_input gi;
            gi.group_index = gidx;
            gi.owning_layers = std::move(owners);
            gi.rank_k = rank;
            gi.rank_v = rank;
            gi.total_dim_k = total;
            gi.total_dim_v = total;
            gi.layer_feature_offsets_k = offs;
            gi.layer_feature_dims_k = dims;
            gi.layer_feature_offsets_v = offs;
            gi.layer_feature_dims_v = dims;
            gi.canonical_k_data = k3.data();
            gi.k_rows = n3;
            gi.k_cols = dim3;
            gi.canonical_v_data = v3.data();
            gi.v_rows = n3;
            gi.v_cols = dim3;
            gi.row_positions.resize(n3);
            for (uint32_t i = 0; i < n3; ++i) gi.row_positions[i] = (int64_t) i;
            fill_hot_row_bytes(gi, GGML_TYPE_F16, GGML_TYPE_F16);
            return gi;
        };
        xkv_factor_group_input g_full = mk_input(0, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 16, 32, 48, 64, 80, 96, 112}, {16, 16, 16, 16, 16, 16, 16, 16}, dim3);
        // Tail group: single owning layer with its own 16-wide canonical slice.
        std::vector<float> k3t((size_t) n3 * 16), v3t((size_t) n3 * 16);
        build_mirror_low_rank(n3, 16, rank, 0.001f, 1004, k3t, v3t);
        xkv_factor_group_input g_tail;
        g_tail.group_index = 1;
        g_tail.owning_layers = {8};
        g_tail.rank_k = rank;
        g_tail.rank_v = rank;
        g_tail.total_dim_k = 16;
        g_tail.total_dim_v = 16;
        g_tail.layer_feature_offsets_k = {0};
        g_tail.layer_feature_dims_k = {16};
        g_tail.layer_feature_offsets_v = {0};
        g_tail.layer_feature_dims_v = {16};
        g_tail.canonical_k_data = k3t.data();
        g_tail.k_rows = n3;
        g_tail.k_cols = 16;
        g_tail.canonical_v_data = v3t.data();
        g_tail.v_rows = n3;
        g_tail.v_cols = 16;
        g_tail.row_positions.resize(n3);
        for (uint32_t i = 0; i < n3; ++i) g_tail.row_positions[i] = (int64_t) i;
        fill_hot_row_bytes(g_tail, GGML_TYPE_F16, GGML_TYPE_F16);
        xkv_bundle_sealing_params sp3;
        sp3.profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
        sp3.source = LLAMA_XKV_SOURCE_DECODED_HOT;
        sp3.rank_k = rank;
        sp3.rank_v = rank;
        sp3.max_relative_error = 0.99;
        sp3.min_saving_ratio = 0.10;
        sp3.flat_type_k = GGML_TYPE_F16;
        sp3.flat_type_v = GGML_TYPE_F16;
        xkv_sealing_result r3 = store3.seal_segment_bundle({g_full, g_tail}, p3, g3, sp3);
        if (!r3.success) {
            std::fprintf(stderr, "no-dense-mirror tail seal failed: %s (skip=%d)\n", r3.message.c_str(), (int) r3.skip_reason);
            return 1;
        }
        CHECK(r3.success);
        const uint64_t seg3 = r3.segment_id;
        k3.clear();
        k3.shrink_to_fit();
        v3.clear();
        v3.shrink_to_fit();
        k3t.clear();
        k3t.shrink_to_fit();
        v3t.clear();
        v3t.shrink_to_fit();
        CHECK(k3t.capacity() == 0);
        CHECK(v3t.capacity() == 0);
        auto segh3 = store3.get_segment(seg3);
        CHECK(segh3 != nullptr);
        if (segh3 == nullptr) return 1;
        CHECK(segh3->groups.size() == 2);
        CHECK(segh3->n_live_rows == n3);
        CHECK(store3.get_arena().get_live_bytes() == 0);
        // Strict-workspace read over all 100 rows: on-demand reconstruct, zero live after.
        xkv_reader_workspace ws3;
        xkv_reader_workspace_config cfg3;
        std::string e3;
        CHECK(ws3.warmup(cfg3, &e3));
        xkv_query_input q3;
        q3.query_index = 0;
        q3.n_q_heads = 1;
        q3.head_dim_k = 16;
        q3.head_dim_v = 16;
        q3.q_vec.assign(16, 0.1f);
        q3.scale = 1.0f / std::sqrt(16.0f);
        {
            xkv_segment_read_view view;
            view.pin = store3.pin_segment(seg3);
            CHECK(bool(view.pin));
            view.segment_version_id = view.pin->segment_version;
            view.storage_generation = 1;
            view.owning_layer = 0;
            view.factor_group_index = 0;
            for (uint32_t i = 0; i < n3; ++i) {
                view.selected_rows.push_back(i);
                view.storage_positions.push_back((int64_t) i);
                view.group_indices.push_back(0);
            }
            std::vector<xkv_segment_read_view> views;
            views.push_back(std::move(view));
            xkv_reader_config rcfg;
            rcfg.workspace = &ws3;
            {
                heap_scope scope;
                xkv_read_result rr = xkv_read_attention(q3, {}, views, nullptr, store3.current_stamp(), &store3, rcfg);
                CHECK(rr.status == xkv_read_status::success);
                CHECK(rr.output.size() == 16);
                for (float v : rr.output) CHECK(std::isfinite(v));
                CHECK(scope.count() == 1);
            }
            CHECK(ws3.live_bytes() == 0);
        }
        // Lifetime: segment + both groups' shared B streams expire after full removal.
        std::weak_ptr<const xkv_segment> w3;
        std::weak_ptr<const encoded_matrix> w3bk0, w3bk1;
        {
            auto h = store3.get_segment(seg3);
            w3 = h;
            w3bk0 = h->groups[0].b_k;
            w3bk1 = h->groups[1].b_k;
        }
        segh3.reset();
        {
            std::string err;
            for (uint64_t pid : p3) CHECK(store3.remove_payload(pid, &err));
        }
        store3.reclaim_retired_segments();
        CHECK(w3.expired());
        CHECK(w3bk0.expired());
        CHECK(w3bk1.expired());
    }

    // ---- One-row boundary: single-token seal leaves no mirror ----
    {
        llama_xkv_cache_store store1(test_cparams());
        const uint32_t n1 = 1;
        std::vector<uint64_t> p1 = {7001};
        std::vector<uint64_t> g1 = {1};
        CHECK(store1.register_hot_payload(p1[0], 0, g1[0], xkv_state::hot_committed));
        // Single row is trivially rank-1: request rank 1 (rank 8 would exceed the
        // 1-row dimension and be rejected by shape validation before any gating).
        std::vector<float> k1, v1;
        build_mirror_low_rank(n1, total_dim, 1, 0.001f, 1005, k1, v1);
        xkv_factor_group_input gi1;
        gi1.group_index = 0;
        gi1.owning_layers = {0, 1, 2, 3};
        gi1.rank_k = 1;
        gi1.rank_v = 1;
        gi1.total_dim_k = total_dim;
        gi1.total_dim_v = total_dim;
        gi1.layer_feature_offsets_k = {0, 16, 32, 48};
        gi1.layer_feature_dims_k = {16, 16, 16, 16};
        gi1.layer_feature_offsets_v = {0, 16, 32, 48};
        gi1.layer_feature_dims_v = {16, 16, 16, 16};
        gi1.canonical_k_data = k1.data();
        gi1.k_rows = n1;
        gi1.k_cols = total_dim;
        gi1.canonical_v_data = v1.data();
        gi1.v_rows = n1;
        gi1.v_cols = total_dim;
        gi1.row_positions.resize(n1);
        gi1.row_positions[0] = 0;
        fill_hot_row_bytes(gi1, GGML_TYPE_F16, GGML_TYPE_F16);
        xkv_bundle_sealing_params sp1;
        sp1.profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS;
        sp1.source = LLAMA_XKV_SOURCE_DECODED_HOT;
        sp1.rank_k = 1;
        sp1.rank_v = 1;
        sp1.max_relative_error = 0.99;
        sp1.min_saving_ratio = 0.10;
        sp1.flat_type_k = GGML_TYPE_F16;
        sp1.flat_type_v = GGML_TYPE_F16;
        xkv_sealing_result r1 = store1.seal_segment_bundle({gi1}, p1, g1, sp1);
        // A single row can never amortize the fixed B-stream cost (B alone is
        // 2*64 Turbo4 rows vs one 256-byte flat row), so the production saving
        // gate must refuse fail-closed. Asserting the exact reason proves the
        // shape validation AND the quality gate both passed (any other refusal
        // reason would name a different skip code), with zero state mutation.
        if (!r1.success && r1.skip_reason != xkv_skip_reason::no_saving) {
            std::fprintf(stderr, "no-dense-mirror one-row seal failed with unexpected reason: %s (skip=%d)\n", r1.message.c_str(), (int) r1.skip_reason);
        }
        CHECK(!r1.success);
        CHECK(r1.skip_reason == xkv_skip_reason::no_saving);
        {
            xkv_state st = xkv_state::hot_writing;
            CHECK(store1.find_payload_state(p1[0], st));
            CHECK(st == xkv_state::hot_committed);
        }
        CHECK(store1.get_arena().get_live_bytes() == 0);
        {
            const xkv_accounting acc = store1.get_accounting();
            CHECK(acc.active_segments == 0);
            CHECK(acc.allocated_bytes == 0);
        }
        k1.clear();
        k1.shrink_to_fit();
        v1.clear();
        v1.shrink_to_fit();
        CHECK(k1.capacity() == 0);
        CHECK(v1.capacity() == 0);
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED: %d checks\n", g_failures);
        return 1;
    }
    std::cout << "=== ALL NO-DENSE-MIRROR CHECKS PASSED ===" << std::endl;
    return 0;
}
