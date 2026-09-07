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
    for (uint32_t r = 0; r < n_tokens; ++r) {
        for (uint32_t c = 0; c < total_dim; ++c) {
            const float v = std::sin((float) r * 0.3f) * std::cos((float) c * 0.2f);
            k_data[(size_t) r * total_dim + c] = v;
            v_data[(size_t) r * total_dim + c] = v * 0.5f;
        }
    }

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

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED: %d checks\n", g_failures);
        return 1;
    }
    std::cout << "=== ALL NO-DENSE-MIRROR CHECKS PASSED ===" << std::endl;
    return 0;
}
