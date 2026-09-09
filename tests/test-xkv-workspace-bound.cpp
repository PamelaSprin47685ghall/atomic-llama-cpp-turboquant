// Focused regression, two layers:
// (A) The bounded host workspace arena enforces its hard cap (peak never
//     exceeds capacity, over-capacity acquire/preflight refuse, scale-out
//     stays within budget).
// (B) The real factor/landmark/reader runtime actually uses and counts the
//     arena: a LANDMARKS-profile evaluate_segment_bundle succeeds at exact
//     measured peak T but refuses with preflight_oom at T minus one
//     accounting unit and at zero (a bypass bug would succeed everywhere);
//     peak grows with history length; strict reads over the published segment
//     succeed at exact-fit workspace bytes, refuse a one-byte-short warmup,
//     scale across growing row selections, and fail closed (empty output,
//     zero live bytes) on an undersized workspace.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-cache.h"
#include "llama-xkv-factor.h"
#include "llama-xkv-reader.h"
#include "llama-cparams.h"

#include <algorithm>
#include <cmath>
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

static uint64_t xkv_test_rng_next(uint64_t & s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

// Genuine rank-true_rank fixture: K = U.V^T + noise, V = 0.5.K.
// All true_rank components carry signal, so a rank-true_rank seal measures
// real factorization/quantization error at the quality gate. (A sin/cos outer
// product is exactly rank-1 and leaves degenerate zero-signal columns.)
static void build_low_rank_canonical(uint64_t rows, uint64_t cols, uint32_t true_rank,
                                     float noise_scale, uint64_t seed,
                                     std::vector<float> & k_out, std::vector<float> & v_out) {
    uint64_t s = seed ? seed : 0x243F6A8885A308D3ULL;
    auto rnd = [&]() -> float {
        xkv_test_rng_next(s);
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

static void build_canonical(uint64_t rows, uint64_t cols,
                            std::vector<float> & k_out, std::vector<float> & v_out) {
    build_low_rank_canonical(rows, cols, 8, 0.001f, 777, k_out, v_out);
}

// Exact measured source row bytes per owning layer: ggml_row_size over the
// live hot tensor type/ne for each layer feature width. The types must match
// landmark_params()' flat_type_k/v (F16); never hand-entered.
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

static xkv_factor_group_input make_input(const float * k_data, const float * v_data, uint64_t rows) {
    xkv_factor_group_input g;
    g.group_index = 0;
    g.owning_layers = {0, 1, 2, 3};
    g.rank_k = 8;
    g.rank_v = 8;
    g.total_dim_k = 64;
    g.total_dim_v = 64;
    g.layer_feature_offsets_k = {0, 16, 32, 48};
    g.layer_feature_dims_k = {16, 16, 16, 16};
    g.layer_feature_offsets_v = {0, 16, 32, 48};
    g.layer_feature_dims_v = {16, 16, 16, 16};
    g.canonical_k_data = k_data;
    g.k_rows = rows;
    g.k_cols = 64;
    g.canonical_v_data = v_data;
    g.v_rows = rows;
    g.v_cols = 64;
    g.row_positions.resize((size_t) rows);
    for (uint64_t i = 0; i < rows; ++i) {
        g.row_positions[(size_t) i] = (int64_t) i;
    }
    fill_hot_row_bytes(g, GGML_TYPE_F16, GGML_TYPE_F16);
    return g;
}

// Real post-factor landmark factory for the current encoded-stream contract:
// chunk-mean landmarks over reconstructed K, streamed in row tiles from the
// FINAL ENCODED A_K/B_K streams using only the supplied scratch span.
// Zero heap allocation (no std::vector temporaries); carve failure or any
// shape/domain violation fails closed.
static xkv_group_landmark_factory_fn test_landmark_factory() {
    return [](
        uint32_t /*group_index*/,
        const encoded_matrix & enc_a_k,
        const encoded_matrix & enc_b_k,
        const int64_t * row_positions,
        uint64_t n_rows,
        factor_workspace_span scratch,
        encoded_matrix & out_lm,
        std::vector<xkv_landmark_chunk> & out_chunks,
        std::string * err
    ) -> bool {
        auto fail = [&](const char * msg) -> bool {
            if (err) *err = msg;
            return false;
        };
        if (row_positions == nullptr) return fail("landmark factory: null row_positions");
        if (n_rows == 0) return fail("landmark factory: zero rows");
        if (!scratch.valid()) return fail("landmark factory: insufficient scratch");
        if (enc_a_k.desc.role != factor_role::a_k) return fail("landmark factory: A_K role mismatch");
        if (enc_b_k.desc.role != factor_role::b_k) return fail("landmark factory: B_K role mismatch");
        const uint64_t rank = enc_a_k.desc.logical_shape.cols;
        if (rank == 0 || rank != enc_b_k.desc.logical_shape.cols) {
            return fail("landmark factory: rank mismatch");
        }
        if (enc_a_k.desc.logical_shape.rows != n_rows) {
            return fail("landmark factory: A_K row mismatch");
        }
        const uint64_t feat = enc_b_k.desc.logical_shape.rows;
        if (feat == 0) return fail("landmark factory: zero feature width");
        const uint64_t pad_a = enc_a_k.desc.padded_shape.cols;
        const uint64_t pad_b = enc_b_k.desc.padded_shape.cols;
        if (pad_a < rank || pad_b < rank || pad_a == 0 || pad_b == 0) {
            return fail("landmark factory: bad padded shape");
        }
        const uint64_t max_u64 = (uint64_t) SIZE_MAX;
        if (n_rows > max_u64 / sizeof(uint64_t)) return fail("landmark factory: index overflow");
        if (n_rows > max_u64 / pad_a / sizeof(float)) return fail("landmark factory: A decode overflow");
        if (feat > max_u64 / pad_b / sizeof(float)) return fail("landmark factory: B decode overflow");
        // Row-tile streaming: A is decoded 32 rows at a time and each K row is
        // accumulated straight into its chunk sum, so no n x feat K tile exists.
        // Peak stays B_full + one A tile + sums, fitting the production-sized
        // factor/shadow lease at every exercised geometry (no oversized buffer).
        constexpr uint64_t kTileRows = 32;
        if (kTileRows > max_u64 / pad_a / sizeof(float)) return fail("landmark factory: A tile overflow");
        uint8_t * p = (uint8_t *) scratch.data;
        size_t remain = scratch.size_bytes;
        auto carve = [&](size_t need, void *& out) -> bool {
            const size_t aligned = (need + (size_t) 63) & ~(size_t) 63;
            if (aligned > remain) return false;
            out = (void *) p;
            p += aligned;
            remain -= aligned;
            return true;
        };
        // Row index array 0..n-1 (shared by the A and B decodes).
        void * idx_mem = nullptr;
        if (!carve((size_t) (n_rows * sizeof(uint64_t)), idx_mem)) {
            return fail("landmark factory: scratch too small for indices");
        }
        uint64_t * idx = (uint64_t *) idx_mem;
        for (uint64_t i = 0; i < n_rows; ++i) idx[i] = i;
        // Decode B_K^T [feat x pad_b] once (reused by every row tile).
        void * b_mem = nullptr;
        if (!carve((size_t) (feat * pad_b * sizeof(float)), b_mem)) {
            return fail("landmark factory: scratch too small for B tile");
        }
        float * b_dec = (float *) b_mem;
        decode_rows(enc_b_k, idx, (size_t) feat, b_dec, (size_t) (feat * pad_b));
        // Phase-aware chunking: chunk = storage_position / 8. Two passes so
        // the sums/counts carve is exact.
        uint64_t n_chunks = 0;
        for (uint64_t i = 0; i < n_rows; ++i) {
            if (row_positions[i] < 0) return fail("landmark factory: negative position");
            const uint64_t ch = (uint64_t) row_positions[i] / 8u;
            if (ch >= 8u * n_rows) return fail("landmark factory: position out of range");
            if (ch + 1 > n_chunks) n_chunks = ch + 1;
        }
        if (n_chunks == 0) return fail("landmark factory: no chunks");
        if (n_chunks > max_u64 / feat / sizeof(float)) {
            return fail("landmark factory: landmark overflow");
        }
        if (n_chunks > max_u64 / sizeof(uint64_t)) {
            return fail("landmark factory: counts overflow");
        }
        void * sums_mem = nullptr;
        void * counts_mem = nullptr;
        if (!carve((size_t) (n_chunks * feat * sizeof(float)), sums_mem) ||
            !carve((size_t) (n_chunks * sizeof(uint64_t)), counts_mem)) {
            return fail("landmark factory: scratch too small for landmarks");
        }
        float * sums = (float *) sums_mem;
        uint64_t * counts = (uint64_t *) counts_mem;
        std::memset(sums, 0, (size_t) (n_chunks * feat * sizeof(float)));
        std::memset(counts, 0, (size_t) (n_chunks * sizeof(uint64_t)));
        // Stream A_K tile by tile; each reconstructed K row accumulates
        // straight into its chunk sum in row order (bit-identical sums).
        void * a_mem = nullptr;
        if (!carve((size_t) (kTileRows * pad_a * sizeof(float)), a_mem)) {
            return fail("landmark factory: scratch too small for A tile");
        }
        float * a_tile = (float *) a_mem;
        for (uint64_t t = 0; t < n_rows; t += kTileRows) {
            const uint64_t tn = (t + kTileRows <= n_rows) ? kTileRows : (n_rows - t);
            decode_rows(enc_a_k, idx + t, (size_t) tn, a_tile, (size_t) (tn * pad_a));
            for (uint64_t ii = 0; ii < tn; ++ii) {
                const uint64_t i = t + ii;
                const uint64_t ch = (uint64_t) row_positions[i] / 8u;
                counts[ch] += 1;
                for (uint64_t j = 0; j < feat; ++j) {
                    double sum = 0.0;
                    for (uint64_t k = 0; k < rank; ++k) {
                        sum += (double) a_tile[ii * pad_a + k] * (double) b_dec[j * pad_b + k];
                    }
                    sums[ch * feat + j] += (float) sum;
                }
            }
        }
        for (uint64_t ch = 0; ch < n_chunks; ++ch) {
            if (counts[ch] == 0) continue;
            const float inv = 1.0f / (float) counts[ch];
            for (uint64_t j = 0; j < feat; ++j) {
                sums[ch * feat + j] *= inv;
            }
        }
        codec_desc desc = make_codec_desc(factor_role::landmark, GGML_TYPE_Q8_0,
                                          orientation::token_major,
                                          {n_chunks, feat}, 0, 777);
        out_lm = encode_matrix(desc, sums, (size_t) (n_chunks * feat));
        // Chunk table: contiguous positions give exact row ranges; counts were
        // verified nonzero above for populated chunks (sparse holes stay empty
        // with zero error bound, still correctly delimited).
        out_chunks.clear();
        for (uint64_t ch = 0; ch < n_chunks; ++ch) {
            xkv_landmark_chunk lmc;
            lmc.row_begin = (uint32_t) (ch * 8u);
            lmc.row_count = (uint32_t) (counts[ch] & 0xFFFFFFFFu);
            lmc.error_bound = 0.0f;
            lmc.source_fingerprint = (uint64_t)ch + 1;
            out_chunks.push_back(lmc);
        }
        return true;
    };
}

static xkv_bundle_sealing_params landmark_params() {
    xkv_bundle_sealing_params p;
    p.profile = LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS;
    p.source = LLAMA_XKV_SOURCE_DECODED_HOT;
    p.rank_k = 8;
    p.rank_v = 8;
    p.max_relative_error = 0.99;
    p.min_saving_ratio = 0.10;
    p.flat_type_k = GGML_TYPE_F16;
    p.flat_type_v = GGML_TYPE_F16;
    p.landmark_factory = test_landmark_factory();
    return p;
}

static void register_payloads(llama_xkv_cache_store & store, uint64_t n, uint64_t base) {
    for (uint64_t i = 0; i < n; ++i) {
        CHECK(store.register_hot_payload(base + i, (uint32_t) i, 1, xkv_state::hot_committed));
    }
}

static std::vector<uint64_t> make_pids(uint64_t n, uint64_t base) {
    std::vector<uint64_t> pids(n);
    for (uint64_t i = 0; i < n; ++i) pids[i] = base + i;
    return pids;
}

int main() {
    std::cout << "=== test-xkv-workspace-bound ===" << std::endl;

    // ---- Layer A: arena arithmetic (standalone 1 MiB hard cap) ----
    {
        llama_xkv_cache_store store(test_cparams());
        const size_t cap = store.get_arena().get_capacity_bytes();
        CHECK(cap == (size_t) 16 * 1024 * 1024);
        CHECK(store.get_arena().get_live_bytes() == 0);
        CHECK(store.get_arena().get_peak_bytes() == 0);

        xkv_workspace_arena arena(1u << 20);
        CHECK(arena.get_capacity_bytes() == (1u << 20));
        CHECK(!arena.acquire(0));
        CHECK(!arena.preflight(0));
        CHECK(!arena.preflight((1u << 20) + 64));
        {
            xkv_arena_lease no = arena.acquire((1u << 20) + 64);
            CHECK(!no);
        }
        CHECK(arena.get_live_bytes() == 0);
        CHECK(arena.get_peak_bytes() == 0);

        {
            xkv_arena_lease a = arena.acquire(512u << 10);
            CHECK(bool(a));
            {
                xkv_arena_lease b = arena.acquire(512u << 10);
                CHECK(bool(b));
                CHECK(arena.get_live_bytes() == (1u << 20));
                CHECK(arena.get_peak_bytes() == (1u << 20));
                CHECK(!arena.preflight(64));
                xkv_arena_lease c = arena.acquire(64);
                CHECK(!c);
                CHECK(arena.get_peak_bytes() <= arena.get_capacity_bytes());
            }
            CHECK(arena.get_live_bytes() == (512u << 10));
        }
        CHECK(arena.get_live_bytes() == 0);
        CHECK(arena.get_peak_bytes() == (1u << 20));

        for (int i = 0; i < 128; ++i) {
            xkv_arena_lease t = arena.acquire(8u << 10);
            CHECK(bool(t));
            CHECK(arena.get_peak_bytes() <= arena.get_capacity_bytes());
        }
        CHECK(arena.get_live_bytes() == 0);

        {
            std::vector<xkv_arena_lease> readers;
            for (int i = 0; i < 16; ++i) {
                readers.push_back(arena.acquire(64u << 10));
                CHECK(bool(readers.back()));
            }
            CHECK(arena.get_live_bytes() == (1u << 20));
            xkv_arena_lease extra = arena.acquire(64u << 10);
            CHECK(!extra);
            CHECK(arena.get_peak_bytes() <= arena.get_capacity_bytes());
        }
        CHECK(arena.get_live_bytes() == 0);

        factor_config cfg;
        cfg.rank_k = 8;
        cfg.rank_v = 8;
        uint64_t small_need = 0, large_need = 0;
        std::string err;
        matrix xk_small(16, 32), xv_small(16, 32);
        matrix xk_large(256, 32), xv_large(256, 32);
        CHECK(estimate_factorize_kv_workspace_bytes(xk_small, xv_small, cfg, &small_need, &err));
        CHECK(estimate_factorize_kv_workspace_bytes(xk_large, xv_large, cfg, &large_need, &err));
        CHECK(large_need > small_need);
        CHECK(arena.preflight((size_t) small_need));
        CHECK(!arena.preflight(static_cast<size_t>(-1)));
    }

    // ---- Layer B: real seal + landmark path under exact T / T-1 / 0 ----
    std::vector<float> k256, v256;
    build_canonical(256, 64, k256, v256);

    auto run_evaluate = [&](llama_xkv_cache_store & store, uint64_t rows,
                            const std::vector<uint64_t> & pids) {
        xkv_factor_group_input g = make_input(k256.data(), v256.data(), rows);
        std::vector<uint64_t> gens(pids.size(), 1);
        return store.evaluate_segment_bundle({g}, pids, gens, landmark_params());
    };

    // Generous arena: LANDMARKS evaluate succeeds; peak proves acquisition.
    llama_xkv_cache_store store_big(test_cparams());
    register_payloads(store_big, 256, 1000);
    xkv_sealing_result ok_big = run_evaluate(store_big, 256, make_pids(256, 1000));
    if (!ok_big.success) {
        std::fprintf(stderr, "workspace-bound evaluate-256 failed: %s (skip=%d)\n", ok_big.message.c_str(), (int) ok_big.skip_reason);
        return 1;
    }
    CHECK(ok_big.success);
    const size_t P = store_big.get_arena().get_peak_bytes();
    CHECK(P > 64); // the seal+landmark path really acquired arena scratch
    CHECK(store_big.get_arena().get_live_bytes() == 0); // and released it

    // Exact T: capacity == measured peak still succeeds (deterministic need).
    llama_xkv_cache_store store_T(test_cparams());
    CHECK(store_T.get_arena().set_capacity_bytes(P));
    register_payloads(store_T, 256, 1000);
    {
        xkv_sealing_result rT = run_evaluate(store_T, 256, make_pids(256, 1000));
        if (!rT.success) {
            std::fprintf(stderr, "workspace-bound exact-T evaluate failed: %s (skip=%d)\n", rT.message.c_str(), (int) rT.skip_reason);
            return 1;
        }
        CHECK(rT.success);
    }

    // T minus one accounting unit: the same production op must refuse.
    // (A bypass bug that never counts arena bytes would succeed here.)
    llama_xkv_cache_store store_Tm(test_cparams());
    CHECK(store_Tm.get_arena().set_capacity_bytes(P - 64));
    register_payloads(store_Tm, 256, 1000);
    {
        xkv_sealing_result r = run_evaluate(store_Tm, 256, make_pids(256, 1000));
        CHECK(!r.success);
        CHECK(r.skip_reason == xkv_skip_reason::preflight_oom);
    }

    // Zero arena: refusal as well, with zero mutation.
    llama_xkv_cache_store store_0(test_cparams());
    CHECK(store_0.get_arena().set_capacity_bytes(0));
    register_payloads(store_0, 256, 1000);
    {
        const xkv_snapshot_stamp before = store_0.current_stamp();
        xkv_sealing_result r = run_evaluate(store_0, 256, make_pids(256, 1000));
        CHECK(!r.success);
        CHECK(r.skip_reason == xkv_skip_reason::preflight_oom);
        CHECK(store_0.current_stamp() == before);
        CHECK(store_0.get_accounting().active_segments == 0);
    }

    // History scaling: shorter history succeeds with strictly smaller peak.
    llama_xkv_cache_store store_128(test_cparams());
    register_payloads(store_128, 128, 2000);
    xkv_sealing_result ok_128 = run_evaluate(store_128, 128, make_pids(128, 2000));
    if (!ok_128.success) {
        std::fprintf(stderr, "workspace-bound evaluate-128 failed: %s (skip=%d)\n", ok_128.message.c_str(), (int) ok_128.skip_reason);
        return 1;
    }
    CHECK(ok_128.success);
    const size_t P128 = store_128.get_arena().get_peak_bytes();
    CHECK(P128 > 0 && P128 < P);

    // ---- Layer C: strict reads over a published landmark segment ----
    llama_xkv_cache_store store_pub(test_cparams());
    register_payloads(store_pub, 256, 3000);
    xkv_factor_group_input gpub = make_input(k256.data(), v256.data(), 256);
    std::vector<uint64_t> pub_pids = make_pids(256, 3000);
    std::vector<uint64_t> pub_gens(256, 1);
    xkv_sealing_result sealed = store_pub.seal_segment_bundle({gpub}, pub_pids, pub_gens, landmark_params());
    if (!sealed.success) {
        std::fprintf(stderr, "workspace-bound publish-256 seal failed: %s (skip=%d)\n", sealed.message.c_str(), (int) sealed.skip_reason);
        return 1;
    }
    CHECK(sealed.success);
    const uint64_t seg_id = sealed.segment_id;
    {
        auto seg = store_pub.get_segment(seg_id);
        CHECK(seg != nullptr);
        if (seg == nullptr) return 1;
        CHECK(seg->groups[0].bytes_landmark > 0); // landmark path produced output
    }

    xkv_query_input query;
    query.query_index = 0;
    query.n_q_heads = 1;
    query.head_dim_k = 16;
    query.head_dim_v = 16;
    query.q_vec.assign(16, 0.1f);
    query.scale = 1.0f / std::sqrt(16.0f);

    auto make_views = [&](std::initializer_list<uint32_t> row_counts) {
        std::vector<xkv_segment_read_view> views;
        for (uint32_t n : row_counts) {
            xkv_segment_read_view view;
            view.pin = store_pub.pin_segment(seg_id);
            CHECK(bool(view.pin));
            view.segment_version_id = view.pin->segment_version;
            view.storage_generation = 1;
            view.owning_layer = 0;
            view.factor_group_index = 0;
            for (uint32_t i = 0; i < n; ++i) {
                view.selected_rows.push_back(i);
                view.storage_positions.push_back((int64_t) i);
                view.group_indices.push_back(0);
            }
            views.push_back(std::move(view));
        }
        return views;
    };

    // Exact-fit workspace for default maxima: T succeeds, T-1 warmup refuses.
    xkv_reader_workspace_config fit_cfg;
    size_t layout_total = 0;
    std::string lerr;
    CHECK(xkv_estimate_workspace_layout(fit_cfg, layout_total, &lerr));
    CHECK(layout_total > 0);
    fit_cfg.capacity_bytes = layout_total;
    xkv_reader_workspace ws_fit;
    CHECK(ws_fit.warmup(fit_cfg, &lerr));
    CHECK(ws_fit.capacity_bytes() == layout_total);
    {
        xkv_reader_workspace_config short_cfg = fit_cfg;
        short_cfg.capacity_bytes = layout_total - 1;
        xkv_reader_workspace ws_short;
        CHECK(!ws_short.warmup(short_cfg, &lerr));
        CHECK(!lerr.empty());
        CHECK(!ws_short.is_warmed_up());
    }

    // History scaling reads: 32/128/256 selected rows all succeed; peak bounded.
    {
        std::vector<xkv_segment_read_view> views = make_views({32, 128, 256});
        const xkv_snapshot_stamp stamp = store_pub.current_stamp();
        for (auto & view : views) {
            std::vector<xkv_segment_read_view> one;
            one.push_back(std::move(view));
            xkv_reader_config rcfg;
            rcfg.workspace = &ws_fit;
            rcfg.tile_size = 32;
            xkv_read_result r = xkv_read_attention(query, {}, one, nullptr, stamp, &store_pub, rcfg);
            CHECK(r.status == xkv_read_status::success);
            CHECK(r.output.size() == 16);
            for (float v : r.output) {
                CHECK(std::isfinite(v));
            }
            CHECK(ws_fit.live_bytes() == 0);
        }
        CHECK(ws_fit.peak_bytes() <= ws_fit.capacity_bytes());
        // Repeat reads (reader fan-out over time) never grow past the cap.
        for (int rep = 0; rep < 4; ++rep) {
            std::vector<xkv_segment_read_view> views2 = make_views({256});
            xkv_reader_config rcfg;
            rcfg.workspace = &ws_fit;
            rcfg.tile_size = 32;
            xkv_read_result r = xkv_read_attention(query, {}, views2, nullptr, stamp, &store_pub, rcfg);
            CHECK(r.status == xkv_read_status::success);
            CHECK(ws_fit.peak_bytes() <= ws_fit.capacity_bytes());
        }
        CHECK(ws_fit.live_bytes() == 0);
    }

    // Undersized workspace (reduced maxima, exact-fit cap): full-size read
    // fails closed — empty output, zero live bytes, no partial heap use.
    {
        xkv_reader_workspace_config small_cfg;
        small_cfg.capacity_bytes = 0; // estimated below
        small_cfg.max_queries = 1;
        small_cfg.max_q_heads = 1;
        small_cfg.max_head_dim_k = 16;
        small_cfg.max_head_dim_v = 16;
        small_cfg.max_tile_size = 4;
        small_cfg.max_rank_k = 8;
        small_cfg.max_rank_v = 8;
        small_cfg.max_csr_entries = 64;
        size_t small_total = 0;
        CHECK(xkv_estimate_workspace_layout(small_cfg, small_total, &lerr));
        small_cfg.capacity_bytes = small_total;
        xkv_reader_workspace ws_small;
        CHECK(ws_small.warmup(small_cfg, &lerr));
        std::vector<xkv_segment_read_view> views = make_views({256});
        xkv_reader_config rcfg;
        rcfg.workspace = &ws_small;
        rcfg.tile_size = 4; // legal under ws maxima: the 256-row CSR need alone must exceed
        xkv_read_result r = xkv_read_attention(query, {}, views, nullptr,
                                              store_pub.current_stamp(), &store_pub, rcfg);
        CHECK(r.status == xkv_read_status::workspace_exceeded);
        CHECK(r.output.empty());
        CHECK(ws_small.live_bytes() == 0);
    }

    // ---- Layer D: workspace/cache independent of segment count and reader fan-out ----
    {
        llama_xkv_cache_store store_multi(test_cparams());
        // 128 rows: fixed B-stream cost amortizes to ~17% saving, clearing the
        // 10% production saving gate. (64 rows cannot: fixed B bytes exceed the
        // 64-row flat source regardless of data quality.)
        const uint32_t seg_rows = 128;
        std::vector<uint64_t> seg_ids;
        for (uint64_t s = 0; s < 3; ++s) {
            const uint64_t base = 5000 + s * 1000;
            register_payloads(store_multi, seg_rows, base);
            xkv_factor_group_input g = make_input(k256.data(), v256.data(), seg_rows);
            std::vector<uint64_t> pids = make_pids(seg_rows, base);
            std::vector<uint64_t> gens(seg_rows, 1);
            xkv_sealing_result sr = store_multi.seal_segment_bundle({g}, pids, gens, landmark_params());
            if (!sr.success) {
                std::fprintf(stderr, "workspace-bound multi-seg seal failed: %s (skip=%d)\n", sr.message.c_str(), (int) sr.skip_reason);
                return 1;
            }
            CHECK(sr.success);
            seg_ids.push_back(sr.segment_id);
        }
        CHECK(store_multi.get_arena().get_live_bytes() == 0);
        {
            const xkv_accounting acc = store_multi.get_accounting();
            CHECK(acc.active_segments == 3);
            CHECK(acc.arena_live_bytes == 0);
            CHECK(acc.reserved_bytes == acc.allocated_bytes + acc.arena_reserved_bytes + acc.dedup_scratch_bytes);
            CHECK(acc.workspace_budget_bytes == acc.arena_capacity_bytes);
        }
        auto make_multi_views = [&](uint32_t rows_per_seg) {
            std::vector<xkv_segment_read_view> views;
            for (uint64_t sid : seg_ids) {
                xkv_segment_read_view view;
                view.pin = store_multi.pin_segment(sid);
                CHECK(bool(view.pin));
                view.segment_version_id = view.pin->segment_version;
                view.storage_generation = 1;
                view.owning_layer = 0;
                view.factor_group_index = 0;
                for (uint32_t i = 0; i < rows_per_seg; ++i) {
                    view.selected_rows.push_back(i);
                    view.storage_positions.push_back((int64_t) i);
                    view.group_indices.push_back(0);
                }
                views.push_back(std::move(view));
            }
            return views;
        };
        // Chained read across all 3 segments in one call: tile streaming reuses
        // the same bounded workspace; peak never exceeds the pre-warmed cap.
        {
            std::vector<xkv_segment_read_view> views = make_multi_views(seg_rows);
            xkv_reader_config rcfg;
            rcfg.workspace = &ws_fit;
            rcfg.tile_size = 32;
            const size_t peak_before = ws_fit.peak_bytes();
            xkv_read_result r = xkv_read_attention(query, {}, views, nullptr,
                                                  store_multi.current_stamp(), &store_multi, rcfg);
            CHECK(r.status == xkv_read_status::success);
            CHECK(r.output.size() == 16);
            for (float v : r.output) CHECK(std::isfinite(v));
            CHECK(ws_fit.live_bytes() == 0);
            CHECK(ws_fit.peak_bytes() <= ws_fit.capacity_bytes());
            CHECK(ws_fit.peak_bytes() >= peak_before);
        }
        // Reader fan-out over time (sequential readers, concurrent-head query):
        // peak stays bounded, live returns to zero after every reader.
        xkv_query_input q4 = query;
        q4.n_q_heads = 4;
        q4.q_vec.assign((size_t) 4 * 16, 0.1f);
        for (int rep = 0; rep < 4; ++rep) {
            std::vector<xkv_segment_read_view> views = make_multi_views(seg_rows);
            xkv_reader_config rcfg;
            rcfg.workspace = &ws_fit;
            rcfg.tile_size = 32;
            xkv_read_result r = xkv_read_attention(q4, {}, views, nullptr,
                                                  store_multi.current_stamp(), &store_multi, rcfg);
            CHECK(r.status == xkv_read_status::success);
            CHECK(r.output.size() == (size_t) 4 * 16);
            CHECK(ws_fit.live_bytes() == 0);
            CHECK(ws_fit.peak_bytes() <= ws_fit.capacity_bytes());
        }
        CHECK(store_multi.get_arena().get_live_bytes() == 0);
    }

    // ---- Layer E: non-divisible / one-row reads, head geometries, zero-capacity refusal ----
    {
        // Non-divisible row selections over the published 256-row segment.
        for (uint32_t n : {7u, 19u, 100u, 1u}) {
            std::vector<xkv_segment_read_view> views = make_views({n});
            xkv_reader_config rcfg;
            rcfg.workspace = &ws_fit;
            rcfg.tile_size = 32; // 7/19/100/1 all leave tail tiles
            xkv_read_result r = xkv_read_attention(query, {}, views, nullptr,
                                                  store_pub.current_stamp(), &store_pub, rcfg);
            CHECK(r.status == xkv_read_status::success);
            CHECK(r.output.size() == 16);
            for (float v : r.output) CHECK(std::isfinite(v));
            CHECK(ws_fit.live_bytes() == 0);
            CHECK(ws_fit.peak_bytes() <= ws_fit.capacity_bytes());
        }
        // Workspace layout grows with head geometry (16 < 128 < 256 partial-IMRoPE width).
        {
            xkv_reader_workspace_config c16, c128, c256;
            c16.max_head_dim_k = 16;
            c16.max_head_dim_v = 16;
            c128.max_head_dim_k = 128;
            c128.max_head_dim_v = 128;
            c256.max_head_dim_k = 256;
            c256.max_head_dim_v = 256;
            size_t t16 = 0, t128 = 0, t256 = 0;
            CHECK(xkv_estimate_workspace_layout(c16, t16, &lerr));
            CHECK(xkv_estimate_workspace_layout(c128, t128, &lerr));
            CHECK(xkv_estimate_workspace_layout(c256, t256, &lerr));
            CHECK(t16 > 0 && t128 > t16 && t256 > t128);
            c256.capacity_bytes = t256;
            xkv_reader_workspace ws256;
            CHECK(ws256.warmup(c256, &lerr));
            CHECK(ws256.capacity_bytes() == t256);
            // Zero-capacity reader workspace must refuse warmup (empty boundary).
            xkv_reader_workspace_config cz = c16;
            cz.capacity_bytes = 0;
            xkv_reader_workspace wsz;
            CHECK(!wsz.warmup(cz, &lerr));
            CHECK(!wsz.is_warmed_up());
        }
        // Group tail seal (full 8-layer group + 1-layer tail) still preflights exactly.
        {
            llama_xkv_cache_store store_tail(test_cparams());
            // 132 rows: non-divisible by 8 (partial 4-row tail chunk) while the
            // fixed B-stream cost still amortizes past the 10% saving gate.
            // 128-wide full group (8x16 layers): two A-stream sets cost ~272 B/row
            // against a ~576 B/row flat source (~48% saving). A 64+16 two-group
            // shape caps at ~9% and can never clear the 10% gate.
            const uint32_t n = 132;
            register_payloads(store_tail, n, 9000);
            std::vector<float> kw, vw;
            build_low_rank_canonical(n, 128, 8, 0.001f, 779, kw, vw);
            xkv_factor_group_input g0;
            g0.group_index = 0;
            g0.owning_layers = {0, 1, 2, 3, 4, 5, 6, 7};
            g0.rank_k = 8;
            g0.rank_v = 8;
            g0.total_dim_k = 128;
            g0.total_dim_v = 128;
            g0.layer_feature_offsets_k = {0, 16, 32, 48, 64, 80, 96, 112};
            g0.layer_feature_dims_k = {16, 16, 16, 16, 16, 16, 16, 16};
            g0.layer_feature_offsets_v = {0, 16, 32, 48, 64, 80, 96, 112};
            g0.layer_feature_dims_v = {16, 16, 16, 16, 16, 16, 16, 16};
            g0.canonical_k_data = kw.data();
            g0.k_rows = n;
            g0.k_cols = 128;
            g0.canonical_v_data = vw.data();
            g0.v_rows = n;
            g0.v_cols = 128;
            g0.row_positions.resize((size_t) n);
            for (uint64_t i = 0; i < n; ++i) g0.row_positions[(size_t) i] = (int64_t) i;
            fill_hot_row_bytes(g0, GGML_TYPE_F16, GGML_TYPE_F16);
            std::vector<float> kt, vt;
            build_low_rank_canonical(n, 16, 8, 0.001f, 778, kt, vt);
            xkv_factor_group_input g1;
            g1.group_index = 1;
            g1.owning_layers = {8};
            g1.rank_k = 8;
            g1.rank_v = 8;
            g1.total_dim_k = 16;
            g1.total_dim_v = 16;
            g1.layer_feature_offsets_k = {0};
            g1.layer_feature_dims_k = {16};
            g1.layer_feature_offsets_v = {0};
            g1.layer_feature_dims_v = {16};
            g1.canonical_k_data = kt.data();
            g1.k_rows = n;
            g1.k_cols = 16;
            g1.canonical_v_data = vt.data();
            g1.v_rows = n;
            g1.v_cols = 16;
            g1.row_positions.resize((size_t) n);
            for (uint64_t i = 0; i < n; ++i) g1.row_positions[(size_t) i] = (int64_t) i;
            fill_hot_row_bytes(g1, GGML_TYPE_F16, GGML_TYPE_F16);
            xkv_sealing_result tr = store_tail.seal_segment_bundle({g0, g1}, make_pids(n, 9000),
                                                                   std::vector<uint64_t>(n, 1), landmark_params());
            if (!tr.success) {
                std::fprintf(stderr, "workspace-bound tail seal failed: %s (skip=%d)\n", tr.message.c_str(), (int) tr.skip_reason);
            }
            CHECK(tr.success);
            if (!tr.success) return 1;
            CHECK(store_tail.get_arena().get_live_bytes() == 0);
            auto segt = store_tail.get_segment(tr.segment_id);
            CHECK(segt != nullptr);
            if (segt == nullptr) return 1;
            CHECK(segt->groups.size() == 2);
        }
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED: %d checks\n", g_failures);
        return 1;
    }
    std::cout << "=== ALL WORKSPACE-BOUND CHECKS PASSED ===" << std::endl;
    return 0;
}
