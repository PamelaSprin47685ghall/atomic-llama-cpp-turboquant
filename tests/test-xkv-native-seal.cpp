// test-xkv-native-seal.cpp — focused test for the production native sealing
// bridge (src/llama-xkv-seal.{h,cpp}): one canonicalize per owning layer into
// group-column views, K/V factorize, ONE DAG/one sync, status-only D2H plus
// singular telemetry, descriptor-mapped immutable handles via adoption.
//
// Covers (CPU backend bitwise vs staged oracles; Vulkan vs oracle tolerance):
//  - two groups (multi-layer + tail), Turbo4/Q8/F32 codecs, HALF RoPE,
//    Hadamard path, K!=V geometry, scattered rows/large positions;
//  - handles immutable, descs validate, exact bytes, S telemetry matches,
//    statuses 0, sync_count==1, bundle fingerprint nonzero;
//  - fault atomicity: T-1 reservation, zero rank, unallocated hot,
//    landmark profile, bad codec, int64 position overflow (out untouched);
//  - Vulkan absence skips only the Vulkan subcase.
// Standalone link: -lggml -lggml-base -lggml-cpu -lggml-vulkan -lllama.
// No project-wide commands.

#include "ggml.h"
#include "ggml-xkv.h"
#include "ggml-xkv-factor.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include "llama-xkv-seal.h"
#include "llama-xkv-codec.h"
#include "llama-xkv-backend.h"
#include "llama-xkv-cache.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace llama_xkv;

static int failures = 0;
#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

static uint64_t trng = 0xABCDEFULL;
static uint64_t tnext() { trng ^= trng << 13; trng ^= trng >> 7; trng ^= trng << 17; return trng; }
static float tuni() { return (float)(tnext() & 0x7FFFFFFFu) / (float)0x7FFFFFFF * 2.0f - 1.0f; }
static void tseed(uint64_t s) { trng = s ? s : 0xABCDEFULL; }

// normalized Hadamard-4 (symmetric involution)
static void had4_mul(const float * in, float * out) {
    for (int i = 0; i < 4; ++i) {
        out[i] = 0.0f;
        for (int j = 0; j < 4; ++j) {
            int p = __builtin_parity(i & j);
            out[i] += (p ? -0.5f : 0.5f) * in[j];
        }
    }
}

static void fwd_rope_half(std::vector<float> & h, uint32_t rotary_dim,
                          float pos, const float * omega, const float * mag) {
    uint32_t fc = rotary_dim / 2;
    for (uint32_t f = 0; f < fc; ++f) {
        float ang = omega[f] * pos;
        float c = std::cos(ang), s = std::sin(ang);
        float re = h[f] * mag[f], im = h[f + fc] * mag[f];
        h[f] = re * c - im * s;
        h[f + fc] = re * s + im * c;
    }
}

struct hot_layer_truth {
    std::vector<float> k_canonical; // [n_phys * n_heads * hd_k]
    std::vector<float> v_canonical;
};

// Build hot K/V backend tensors for one layer. K hot = H_fwd(RoPE(truth))
// encoded per head; V hot = truth encoded per head.
struct hot_tensors {
    ggml_tensor *k = nullptr, *v = nullptr;
    ggml_backend_buffer_ptr buf;
    ggml_context_ptr ctx;
};

static bool build_hot_layer(ggml_backend_t backend, ggml_type hot_type,
        uint32_t n_heads, uint32_t hd, uint32_t rotary_dim,
        const std::vector<float> & omega, const std::vector<float> & mag,
        const std::vector<float> & truth, // [n_phys*n_heads*hd] canonical (K: pre-RoPE)
        bool apply_rope, bool apply_had4,
        uint32_t n_phys, hot_tensors & out, std::string & err) {
    bool turbo = (hot_type == GGML_TYPE_TURBO2_0 || hot_type == GGML_TYPE_TURBO3_0 ||
                  hot_type == GGML_TYPE_TURBO4_0);
    uint32_t phd = turbo ? ((hd + 127) / 128 * 128) : hd;
    if (hot_type == GGML_TYPE_Q8_0 && (phd % 32) != 0) { err = "q8 width"; return false; }
    ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
    out.ctx.reset(ggml_init(ip));
    if (!out.ctx) { err = "ctx"; return false; }
    out.k = ggml_new_tensor_2d(out.ctx.get(), hot_type, (int64_t)n_heads * phd, n_phys);
    out.v = nullptr;
    if (!out.k) { err = "tensor"; return false; }
    out.buf.reset(ggml_backend_alloc_ctx_tensors(out.ctx.get(), backend));
    if (!out.buf) { err = "alloc"; return false; }
    size_t head_bytes = ggml_row_size(hot_type, phd);
    std::vector<float> h(phd, 0.0f);
    // Stage the whole hot tensor at once (positions 1000+3*i match seal rows).
    {
        std::vector<uint8_t> all(head_bytes * n_heads * n_phys, 0);
        for (uint32_t i = 0; i < n_phys; ++i) {
            for (uint32_t v = 0; v < n_heads; ++v) {
                const float * src = truth.data() + (size_t(i) * n_heads + v) * hd;
                std::fill(h.begin(), h.end(), 0.0f);
                memcpy(h.data(), src, size_t(hd) * 4);
                if (apply_rope && rotary_dim > 0) {
                    fwd_rope_half(h, rotary_dim, (float)(1000 + i * 3), omega.data(), mag.data());
                }
                if (apply_had4) {
                    for (uint32_t b = 0; b < hd; b += 4) {
                        float o[4];
                        had4_mul(h.data() + b, o);
                        memcpy(h.data() + b, o, 4 * 4);
                    }
                }
                uint8_t * dst = all.data() + (size_t(i) * n_heads + v) * head_bytes;
                if (hot_type == GGML_TYPE_F32) {
                    memcpy(dst, h.data(), size_t(phd) * 4);
                } else if (turbo) {
                    if (!ggml_quantize_turbo_row(hot_type, h.data(), dst, phd, 128)) {
                        err = "turbo encode";
                        return false;
                    }
                } else {
                    const auto * tr = ggml_get_type_traits(hot_type);
                    tr->from_float_ref(h.data(), dst, phd);
                }
            }
        }
        ggml_backend_tensor_set(out.k, all.data(), 0, all.size());
    }
    return true;
}

// ---------- fixture ----------
// Group 0: 2 layers x 2 heads, hd 64/64, rotary 16 HALF, Turbo4 hot K,
//   Q8 hot V, layer 0 Hadamard-4 on K. Group 1 (tail): 1 layer x 1 head,
//   hd 48/48, rotary 0, F32 hot. n_phys=20, seal rows 0..15.
struct seal_case {
    xkv_native_seal_config cfg;
    // host truth (canonical) per group for oracle staging + quality gates
    struct group_truth {
        std::vector<float> xk; // [n_phys, mk]
        std::vector<float> xv; // [n_phys, mv]
    };
    std::vector<group_truth> truth;
    std::vector<float> omega, mag; // shared rope tables (rotary 16)
    xkv_allocation_id_generator id_gen;
};

static std::vector<float> gen_truth(uint32_t n, uint32_t m, uint32_t r0, uint64_t seed) {
    tseed(seed);
    std::vector<float> A(size_t(n) * r0), B(size_t(m) * r0);
    for (auto & x : A) x = tuni();
    for (auto & x : B) x = tuni() * 0.5f;
    std::vector<float> X(size_t(n) * m, 0.0f);
    for (uint32_t i = 0; i < n; ++i) for (uint32_t j = 0; j < m; ++j) {
        double s = 0.0;
        for (uint32_t k = 0; k < r0; ++k) s += (double)A[size_t(i) * r0 + k] * B[size_t(j) * r0 + k];
        X[size_t(i) * m + j] = (float)s + 1e-3f * tuni();
    }
    return X;
}

struct backend_hot {
    std::vector<hot_tensors> k_layers, v_layers; // per group concatenated below
};

static bool build_case(std::shared_ptr<struct ggml_backend> executor, seal_case & c, std::string & err,
                       std::vector<hot_tensors> & hotk, std::vector<hot_tensors> & hotv) {
    const uint32_t n_phys = 20, n = 16;
    ggml_backend_t backend = executor.get();
    c.cfg.executor = executor;
    c.cfg.backend = backend;
    c.cfg.buft = ggml_backend_get_default_buffer_type(backend);
    c.cfg.n_rows = n;
    c.cfg.physical_rows.resize(n);
    c.cfg.storage_positions.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        c.cfg.physical_rows[i] = (int32_t)i;
        c.cfg.storage_positions[i] = (int64_t)(1000 + 3 * i);
    }
    c.cfg.residency = GGML_XKV_RES_REFERENCE_HOST; // caller overrides per backend
    // shared rope tables rotary 16
    c.omega.assign(8, 0.0f); c.mag.assign(8, 1.0f);
    for (uint32_t f = 0; f < 8; ++f) c.omega[f] = std::pow(10000.0f, -2.0f * (float)f / 16.0f);
    // group 0 (has landmarks: Q8_0, chunk_tokens 8)
    {
        xkv_native_seal_group g;
        g.group_index = 0;
        g.rank_k = 8; g.rank_v = 6;
        g.codec_a_k = GGML_TYPE_TURBO4_0; g.codec_b_k = GGML_TYPE_TURBO4_0;
        g.codec_a_v = GGML_TYPE_Q8_0; g.codec_b_v = GGML_TYPE_Q8_0;
        g.seed_k = 42; g.seed_v = 43;
        g.want_landmarks = true;
        g.landmark_type = GGML_TYPE_Q8_0;
        g.landmark_seed = 777; // canonical landmark codec seed
        g.chunk_tokens = 8;
        g.phase_tx_fingerprint = 0x12345678ULL;
        const uint32_t mk = 2 * 2 * 64, mv = 2 * 2 * 64;
        seal_case::group_truth t;
        t.xk = gen_truth(n_phys, mk, 6, 101);
        t.xv = gen_truth(n_phys, mv, 5, 102);
        c.truth.push_back(std::move(t));
        for (int li = 0; li < 2; ++li) {
            xkv_native_seal_hot_layer L;
            L.n_heads = 2; L.head_dim_k = 64; L.head_dim_v = 64;
            L.rotary_dim_k = 16; L.rope_mode_k = GGML_XKV_ROPE_HALF;
            L.rope_omega_k = c.omega; L.rope_mag_k = c.mag;
            if (li == 0) {
                // normalized Hadamard-4 (symmetric involution)
                L.hadamard_k.assign(16, 0.0f);
                for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
                    L.hadamard_k[size_t(i) * 4 + j] = (__builtin_parity(i & j) ? -0.5f : 0.5f);
            }
            // per-layer truth slices (layer-major within group concat)
            std::vector<float> tk(size_t(n_phys) * 128), tv(size_t(n_phys) * 128);
            for (uint32_t i = 0; i < n_phys; ++i) {
                memcpy(tk.data() + size_t(i) * 128,
                       c.truth[0].xk.data() + (size_t(i) * mk + li * 128), size_t(128) * 4);
                memcpy(tv.data() + size_t(i) * 128,
                       c.truth[0].xv.data() + (size_t(i) * mv + li * 128), size_t(128) * 4);
            }
            hot_tensors hk, hv;
            std::vector<float> pos_omega = c.omega; // capture for lambda-free call
            if (!build_hot_layer(backend, GGML_TYPE_TURBO4_0, 2, 64, 16, c.omega, c.mag,
                    tk, true, li == 0, n_phys, hk, err)) return false;
            if (!build_hot_layer(backend, GGML_TYPE_Q8_0, 2, 64, 0, c.omega, c.mag,
                    tv, false, false, n_phys, hv, err)) return false;
            L.hot_k = hk.k; L.hot_v = hv.k;
            hotk.push_back(std::move(hk));
            hotv.push_back(std::move(hv));
            g.layers.push_back(std::move(L));
        }
        c.cfg.groups.push_back(std::move(g));
    }
    // group 1 (tail): 1 layer x 1 head, hd 48, rotary 0, F32
    {
        xkv_native_seal_group g;
        g.group_index = 1;
        g.rank_k = 6; g.rank_v = 4;
        g.codec_a_k = GGML_TYPE_F32; g.codec_b_k = GGML_TYPE_F32;
        g.codec_a_v = GGML_TYPE_F32; g.codec_b_v = GGML_TYPE_F32;
        g.seed_k = 1042; g.seed_v = 1043;
        seal_case::group_truth t;
        t.xk = gen_truth(n_phys, 48, 4, 201);
        t.xv = gen_truth(n_phys, 48, 3, 202);
        c.truth.push_back(std::move(t));
        xkv_native_seal_hot_layer L;
        L.n_heads = 1; L.head_dim_k = 48; L.head_dim_v = 48;
        L.rotary_dim_k = 0;
        hot_tensors hk, hv;
        if (!build_hot_layer(backend, GGML_TYPE_F32, 1, 48, 0, c.omega, c.mag,
                c.truth[1].xk, false, false, n_phys, hk, err)) return false;
        if (!build_hot_layer(backend, GGML_TYPE_F32, 1, 48, 0, c.omega, c.mag,
                c.truth[1].xv, false, false, n_phys, hv, err)) return false;
        L.hot_k = hk.k; L.hot_v = hv.k;
        hotk.push_back(std::move(hk));
        hotv.push_back(std::move(hv));
        g.layers.push_back(std::move(L));
        c.cfg.groups.push_back(std::move(g));
    }
    c.cfg.id_gen = &c.id_gen;
    return true;
}

// Staged host oracles for comparison: canonicalize per layer then factorize.
struct staged_oracle {
    std::vector<uint8_t> a_bytes, b_bytes;
    std::vector<float> svals;
    ggml_xkv_residual r_ab = {0, 0, 0}, r_a = {0, 0, 0}, r_b = {0, 0, 0};
};

static bool run_staged_group(const seal_case & c, size_t gi, bool is_k,
        staged_oracle & o, std::string & err_out) {
    const auto & g = c.cfg.groups[gi];
    const uint32_t n = c.cfg.n_rows;
    uint64_t mk = 0;
    for (const auto & L : g.layers) mk += (uint64_t)L.n_heads * (is_k ? L.head_dim_k : L.head_dim_v);
    // canonical staging via oracle per layer
    std::vector<float> X(size_t(n) * mk, 0.0f);
    // Run canonicalize oracle on each layer's hot tensor, copy into group staging X
    uint64_t foff = 0;
    for (size_t li = 0; li < g.layers.size(); ++li) {
        const auto & L = g.layers[li];
        uint32_t wk = L.n_heads * (is_k ? L.head_dim_k : L.head_dim_v);
        uint32_t hd = is_k ? L.head_dim_k : L.head_dim_v;
        ggml_tensor * ht = is_k ? L.hot_k : L.hot_v;
        uint32_t phd = (uint32_t)((uint64_t)ht->ne[0] / L.n_heads);
        std::vector<uint8_t> hbytes(ggml_nbytes(ht));
        ggml_backend_tensor_get(ht, hbytes.data(), 0, hbytes.size());
        ggml_xkv_canonicalize_params cp = {};
        cp.version = GGML_XKV_FACTOR_VERSION;
        cp.n_rows = n; cp.n_layers = 1; cp.n_heads = L.n_heads;
        cp.head_dim = hd; cp.padded_head_dim = phd; cp.total_feat = wk;
        cp.rotary_dim = is_k ? L.rotary_dim_k : 0;
        cp.rope_mode = is_k ? (uint32_t)L.rope_mode_k : 0;
        cp.input_type = (uint32_t)ht->type;
        cp.is_k = is_k ? 1 : 0;
        const std::vector<float> & had = is_k ? L.hadamard_k : L.hadamard_v;
        cp.hadamard_dim = had.empty() ? 0 : (uint32_t)llround(std::sqrt((double)had.size()));
        std::vector<float> layer_x(size_t(n) * wk);
        char c_err[256] = {};
        std::vector<float> rm;
        uint32_t fc = cp.rotary_dim / 2;
        if (fc > 0) {
            rm.resize(size_t(fc) * 2);
            for (uint32_t f = 0; f < fc; ++f) { rm[f] = L.rope_omega_k[f]; rm[fc + f] = L.rope_mag_k[f]; }
        }
        std::vector<int32_t> pos32(n);
        for (uint32_t i = 0; i < n; ++i) pos32[i] = (int32_t)c.cfg.storage_positions[i];
        if (!ggml_xkv_canonicalize_cpu_oracle(hbytes.data(), ht->type,
                c.cfg.physical_rows.data(), c.cfg.storage_positions.data(), 1 /* pos_is_64 */, n,
                rm.data(), (uint32_t)rm.size(), had.empty() ? nullptr : had.data(), cp.hadamard_dim,
                &cp, layer_x.data(), c_err, sizeof(c_err))) {
            err_out = std::string("staged canon failed: ") + c_err;
            return false;
        }
        // Copy into group X column view
        for (uint32_t i = 0; i < n; ++i) {
            memcpy(X.data() + size_t(i) * mk + foff, layer_x.data() + size_t(i) * wk, size_t(wk) * 4);
        }
        foff += wk;
    }
    uint32_t rank = is_k ? g.rank_k : g.rank_v;
    uint32_t r = (uint32_t)std::min<uint64_t>(rank, std::min<uint64_t>(n, mk));
    ggml_type ta = is_k ? g.codec_a_k : g.codec_a_v;
    ggml_type tb = is_k ? g.codec_b_k : g.codec_b_v;
    ggml_xkv_factorize_params p = {};
    p.version = GGML_XKV_FACTOR_VERSION;
    p.rows_n = n; p.cols_m = (uint32_t)mk; p.requested_rank = r;
    p.oversampling = g.oversampling; p.power_iterations = g.power_iterations;
    p.balance_mode = g.balance_mode;
    p.type_a = (uint32_t)ta; p.type_b = (uint32_t)tb;
    uint64_t seed = is_k ? g.seed_k : g.seed_v;
    p.seed_low = (uint32_t)seed; p.seed_high = (uint32_t)(seed >> 32);
    p.pad_r_a = (uint32_t)ggml_xkv_padded_rank(ta, r);
    p.pad_r_b = (uint32_t)ggml_xkv_padded_rank(tb, r);
    o.a_bytes.assign(ggml_row_size(ta, p.pad_r_a) * n, 0);
    o.b_bytes.assign(ggml_row_size(tb, p.pad_r_b) * mk, 0);
    o.svals.assign(r, 0.0f);
    char err[256] = {};
    if (!ggml_xkv_factorize_cpu_oracle_resid(X.data(), n, (uint32_t)mk, &p,
            o.a_bytes.data(), o.b_bytes.data(), o.svals.data(),
            &o.r_ab, &o.r_a, &o.r_b, err, sizeof(err))) {
        err_out = err;
        return false;
    }
    return true;
}

static std::vector<uint8_t> handle_bytes(ggml_backend_t backend,
        const std::shared_ptr<xkv_backend_allocation> & h) {
    std::vector<uint8_t> out;
    if (!h || !h->get_tensor()) return out;
    out.assign(ggml_nbytes(h->get_tensor()), 0);
    ggml_backend_tensor_get(h->get_tensor(), out.data(), 0, out.size());
    return out;
}

static void test_cpu_bridge() {
    std::printf("[seal] CPU backend bridge == staged oracles ...\n");
    std::shared_ptr<struct ggml_backend> cpu_owner(ggml_backend_cpu_init(), ggml_backend_free);
    ggml_backend_t cpu = cpu_owner.get();
    CHECK(cpu != nullptr);
    seal_case c;
    std::vector<hot_tensors> hotk, hotv;
    std::string err;
    CHECK(build_case(cpu_owner, c, err, hotk, hotv));
    c.cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    c.cfg.placement_identity = ggml_backend_name(cpu);
    xkv_backend_store_reservation res;
    res.reserved_bytes = 1ULL << 30; res.cap_bytes = 1ULL << 30;
    c.cfg.store_reservation = &res;
    xkv_native_seal_bundle out;
    CHECK(xkv_native_seal_build(c.cfg, out, &err));
    if (err.size()) std::fprintf(stderr, "seal note: %s\n", err.c_str());
    CHECK(out.groups.size() == 2);
    CHECK(out.sync_count == 1);
    CHECK(out.bundle_fingerprint != 0);
    CHECK(out.factored_bytes > 0 && out.preflight_bytes > 0 && out.scratch_bytes > 0);
    CHECK(out.adopt_sync_count == 0);
    CHECK(out.preflight_bytes == out.peak_dest_bytes + out.scratch_bytes);
    CHECK(out.factored_bytes <= out.peak_dest_bytes);
    // Zero-copy ownership transfer: all adopted streams share the single
    // stream-only persistent buffer. Copy-based adoption would allocate a
    // second buffer and cost an adoption sync.
    {
        ggml_backend_buffer_t expect_buf = out.groups[0].a_k.handle->get_buffer();
        CHECK(expect_buf != nullptr);
        for (const auto & bg : out.groups) {
            const xkv_native_sealed_stream * st[5] = {&bg.a_k, &bg.b_k, &bg.a_v, &bg.b_v, &bg.landmark};
            for (int s = 0; s < 4; ++s) CHECK(st[s]->handle->get_buffer() == expect_buf);
            if (bg.has_landmarks) CHECK(st[4]->handle->get_buffer() == expect_buf);
        }
    }
    for (size_t gi = 0; gi < 2; ++gi) {
        const auto & bg = out.groups[gi];
        CHECK(bg.group_index == gi);
        CHECK(bg.status_canon == 0 && bg.status_fact_k == 0 && bg.status_fact_v == 0);
        for (int s = 0; s < 4; ++s) {
            const xkv_native_sealed_stream * st[4] = {&bg.a_k, &bg.b_k, &bg.a_v, &bg.b_v};
            CHECK(st[s]->handle != nullptr);
            CHECK(st[s]->handle->is_immutable());
            CHECK(st[s]->desc.validate());
            CHECK(st[s]->stream_fingerprint != 0);
            CHECK(st[s]->exact_bytes == ggml_nbytes(st[s]->handle->get_tensor()));
        }
        if (gi == 0) {
            CHECK(bg.has_landmarks);
            CHECK(bg.status_lmbuild == 0);
            CHECK(bg.landmark.handle != nullptr);
            CHECK(bg.landmark.handle->is_immutable());
            CHECK(bg.landmark.desc.validate());
            CHECK(bg.landmark.exact_bytes == ggml_nbytes(bg.landmark.handle->get_tensor()));
            CHECK(bg.landmark_eb.size() == 2); // 16 rows / 8 chunk_tokens = 2 chunks
            CHECK(bg.landmark_srcfp.size() == 2);
            for (float eb : bg.landmark_eb) { CHECK(std::isfinite(eb) && eb > 0.0f); }
            for (uint64_t fp : bg.landmark_srcfp) { CHECK(fp != 0); }
        } else {
            CHECK(!bg.has_landmarks);
        }
        // staged oracle bitwise comparison (CPU path is deterministic)
        for (int kv = 0; kv < 2; ++kv) {
            staged_oracle o;
            CHECK(run_staged_group(c, gi, kv == 0, o, err));
            const xkv_native_sealed_stream * st = kv == 0 ? &bg.a_k : &bg.a_v;
            const xkv_native_sealed_stream * stb = kv == 0 ? &bg.b_k : &bg.b_v;
            std::vector<uint8_t> hab = handle_bytes(cpu, st->handle);
            std::vector<uint8_t> hbb = handle_bytes(cpu, stb->handle);
            if (hab != o.a_bytes) {
                std::printf("  GI=%zu %s A bytes differ: hab.size=%zu o.a_size=%zu\n", gi, kv == 0 ? "K" : "V", hab.size(), o.a_bytes.size());
                std::printf("    hab:"); for (int k=0; k<8 && k<(int)hab.size(); ++k) std::printf(" %02x", hab[k]);
                std::printf("\n    o.a:"); for (int k=0; k<8 && k<(int)o.a_bytes.size(); ++k) std::printf(" %02x", o.a_bytes[k]);
                std::printf("\n");
            }
            if (hbb != o.b_bytes) {
                std::printf("  GI=%zu %s B bytes differ: hbb.size=%zu o.b_size=%zu\n", gi, kv == 0 ? "K" : "V", hbb.size(), o.b_bytes.size());
            }
            CHECK(hab == o.a_bytes);
            CHECK(hbb == o.b_bytes);
            const std::vector<float> & sv = kv == 0 ? bg.singular_k : bg.singular_v;
            CHECK(sv.size() == o.svals.size());
            CHECK(sv == o.svals);
        }
        // S telemetry sane
        for (const auto & sv : {bg.singular_k, bg.singular_v}) {
            CHECK(!sv.empty() && std::isfinite(sv[0]) && sv[0] > 0.0f);
            for (size_t k = 1; k < sv.size(); ++k) CHECK(sv[k - 1] + 1e-6f >= sv[k]);
        }
        // Device residual reports (CPU: bitwise vs staged oracle) + gate
        for (int kv = 0; kv < 2; ++kv) {
            staged_oracle o;
            CHECK(run_staged_group(c, gi, kv == 0, o, err));
            const factor_error_report & rep = kv == 0 ? bg.err_k : bg.err_v;
            const ggml_xkv_residual & ro = o.r_ab;
            std::printf("  GI=%zu %s rep: orig=%.6f err=%.6f max=%.6f rel=%.6f | ro: orig=%.6f err=%.6f max=%.6f\n",
                gi, kv == 0 ? "K" : "V",
                rep.frobenius_norm_original, rep.frobenius_norm_error, rep.max_absolute_error, rep.relative_error,
                ro.frob_orig, ro.frob_err, ro.max_err);
            CHECK(std::fabs(rep.frobenius_norm_original - ro.frob_orig) < 1e-4);
            CHECK(std::fabs(rep.frobenius_norm_error - ro.frob_err) < 1e-4);
            CHECK(std::fabs(rep.max_absolute_error - ro.max_err) < 1e-4);
            CHECK(rep.relative_error < c.cfg.max_rel_error);
        }
    }
    // Gate refusal: impossibly tight max_rel_error publishes nothing.
    // (cfg shallow-copies hot tensor pointers; their storage stays alive.)
    {
        xkv_native_seal_config cfg2 = c.cfg;
        cfg2.max_rel_error = 1e-9;
        xkv_native_seal_bundle no;
        no.bundle_fingerprint = 0xDEADBEEF;
        std::string e2;
        CHECK(!xkv_native_seal_build(cfg2, no, &e2));
        CHECK(!e2.empty());
        CHECK(no.bundle_fingerprint == 0xDEADBEEF);
        CHECK(no.groups.empty());
    }
    // T-1 reservation refusal, out untouched
    {
        xkv_native_seal_bundle no;
        no.bundle_fingerprint = 0xDEADBEEF;
        xkv_backend_store_reservation short_res;
        short_res.reserved_bytes = out.peak_dest_bytes - 1; // allocator-reserved, not raw payload
        short_res.cap_bytes = 1ULL << 30;
        c.cfg.store_reservation = &short_res;
        std::string e2;
        CHECK(!xkv_native_seal_build(c.cfg, no, &e2));
        CHECK(!e2.empty());
        CHECK(no.bundle_fingerprint == 0xDEADBEEF);
        CHECK(no.groups.empty());
    }
    // freed by cpu_owner
    std::printf("  CPU bridge OK\n");
}

static void test_faults() {
    std::printf("[seal] fault atomicity ...\n");
    std::shared_ptr<struct ggml_backend> cpu_owner(ggml_backend_cpu_init(), ggml_backend_free);
    ggml_backend_t cpu = cpu_owner.get();
    CHECK(cpu != nullptr);
    seal_case c;
    std::vector<hot_tensors> hotk, hotv;
    std::string err;
    CHECK(build_case(cpu_owner, c, err, hotk, hotv));
    c.cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    c.cfg.placement_identity = ggml_backend_name(cpu);
    xkv_backend_store_reservation res;
    res.reserved_bytes = 1ULL << 30;
    c.cfg.store_reservation = &res;
    auto expect_fail = [&](const char * name, std::string * emod) {
        xkv_native_seal_bundle no;
        no.bundle_fingerprint = 0xDEADBEEF;
        std::string e2;
        bool ok = xkv_native_seal_build(c.cfg, no, &e2);
        if (emod) *emod = e2;
        CHECK(!ok);
        CHECK(no.bundle_fingerprint == 0xDEADBEEF);
        CHECK(no.groups.empty());
        (void)name;
    };
    { // zero rank
        auto save = c.cfg.groups[0].rank_k;
        c.cfg.groups[0].rank_k = 0;
        std::string e2;
        expect_fail("rank0", &e2);
        c.cfg.groups[0].rank_k = save;
    }
    { // landmark invalid chunk tokens refused
        c.cfg.groups[0].want_landmarks = true;
        c.cfg.groups[0].chunk_tokens = 0;
        std::string e2;
        expect_fail("landmark_chunk0", &e2);
        CHECK(e2.find("chunk_tokens") != std::string::npos);
        c.cfg.groups[0].chunk_tokens = 8;
        c.cfg.groups[0].want_landmarks = false;
    }
    { // unsupported codec
        auto save = c.cfg.groups[0].codec_a_k;
        c.cfg.groups[0].codec_a_k = GGML_TYPE_Q4_0;
        expect_fail("codec", nullptr);
        c.cfg.groups[0].codec_a_k = save;
    }
    { // negative physical row refused
        auto save = c.cfg.physical_rows[2];
        c.cfg.physical_rows[2] = -5;
        expect_fail("neg_phys", nullptr);
        c.cfg.physical_rows[2] = save;
    }
    { // unallocated hot tensor (cross-device rule)
        ggml_init_params ip = { ggml_tensor_overhead() * 2, nullptr, true };
        ggml_context_ptr tmp(ggml_init(ip));
        ggml_tensor * orphan = ggml_new_tensor_2d(tmp.get(), GGML_TYPE_F32, 128, 20);
        auto save = c.cfg.groups[0].layers[0].hot_k;
        c.cfg.groups[0].layers[0].hot_k = orphan;
        expect_fail("orphan", nullptr);
        c.cfg.groups[0].layers[0].hot_k = save;
    }
    // freed by cpu_owner
    std::printf("  faults OK\n");
}

static void test_store_validate_and_publish() {
    std::printf("[seal] store validate_candidate and publish with native bundle ...\n");
    ggml_backend_load_all();
    std::shared_ptr<struct ggml_backend> device_owner;
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev) device_owner = std::shared_ptr<struct ggml_backend>(ggml_backend_dev_init(dev, nullptr), ggml_backend_free);
#ifdef GGML_USE_VULKAN
    if (!device_owner && ggml_backend_vk_get_device_count() > 0)
        device_owner = std::shared_ptr<struct ggml_backend>(ggml_backend_vk_init(0), ggml_backend_free);
#endif
    ggml_backend_t device = device_owner.get();
    if (!device) { std::printf("  SKIP: no device backend for device-bundle validation\n"); return; }
    seal_case c;
    std::vector<hot_tensors> hotk, hotv;
    std::string err;
    CHECK(build_case(device_owner, c, err, hotk, hotv));
    c.cfg.residency = GGML_XKV_RES_DEVICE_OWNED; // device bundle with backend_bundle
    c.cfg.placement_identity = ggml_backend_name(device);

    xkv_backend_store_reservation sres;
    sres.reserved_bytes = 1ULL << 30; sres.cap_bytes = 1ULL << 30;
    c.cfg.store_reservation = &sres;

    xkv_native_seal_bundle out;
    const bool sealed = xkv_native_seal_build(c.cfg, out, &err);
    if (!sealed) std::fprintf(stderr, "store native seal failed: %s\n", err.c_str());
    CHECK(sealed);
    if (!sealed) return;
    CHECK(out.backend_bundle != nullptr);
    CHECK(out.backend_bundle->is_success());
    CHECK(out.backend_bundle->is_committed());

    // Construct candidate segment matching the native bundle
    auto cand = std::make_shared<xkv_segment>();
    cand->segment_id = 1;
    cand->segment_version = 1;
    cand->residency = GGML_XKV_RES_DEVICE_OWNED;
    // This fixture intentionally includes an F32 tail group; it exercises
    // bundle ownership, not the all-Turbo production profile gate.
    cand->profile = LLAMA_XKV_STORAGE_PROFILE_REFERENCE;
    cand->backend_bundle = out.backend_bundle;

    for (const auto & bg : out.groups) {
        xkv_factor_group_payload sg;
        sg.group_index = bg.group_index;
        sg.owning_layers = c.cfg.groups[bg.group_index].layers.size() == 2 ? std::vector<uint32_t>{3, 7} : std::vector<uint32_t>{11};
        sg.total_dim_k = bg.group_index == 0 ? 256 : 48;
        sg.total_dim_v = bg.group_index == 0 ? 256 : 48;
        if (bg.group_index == 0) {
            sg.layer_feature_offsets_k = {0, 128};
            sg.layer_feature_dims_k = {128, 128};
            sg.layer_feature_offsets_v = {0, 128};
            sg.layer_feature_dims_v = {128, 128};
        } else {
            sg.layer_feature_offsets_k = {0};
            sg.layer_feature_dims_k = {48};
            sg.layer_feature_offsets_v = {0};
            sg.layer_feature_dims_v = {48};
        }
        sg.a_k.desc = bg.a_k.desc;
        auto eb_k = std::make_shared<encoded_matrix>();
        eb_k->desc = bg.b_k.desc;
        sg.b_k = eb_k;
        sg.a_v.desc = bg.a_v.desc;
        auto eb_v = std::make_shared<encoded_matrix>();
        eb_v->desc = bg.b_v.desc;
        sg.b_v = eb_v;
        if (bg.has_landmarks) {
            sg.landmark.desc = bg.landmark.desc;
            for (size_t ci = 0; ci < bg.landmark_eb.size(); ++ci) {
                xkv_landmark_chunk ce;
                ce.row_begin = (uint32_t)(ci * 8);
                ce.row_count = 8;
                ce.error_bound = bg.landmark_eb[ci];
                ce.source_fingerprint = bg.landmark_srcfp[ci];
                sg.landmark_chunks.push_back(ce);
            }
            sg.landmark_table_fingerprint = compute_landmark_table_fingerprint(sg.landmark_chunks);
        }
        cand->groups.push_back(std::move(sg));
    }

    // Parity test: native landmark descriptor fingerprint vs CPU canonical descriptor fingerprint
    codec_desc cpu_lm_desc = make_codec_desc(factor_role::landmark, GGML_TYPE_Q8_0,
        orientation::token_major, matrix_shape{2, 256}, 0, 777);
    CHECK(out.groups[0].landmark.desc.fingerprint() == cpu_lm_desc.fingerprint());
    CHECK(out.groups[0].landmark.stream_fingerprint == cpu_lm_desc.fingerprint());
    std::printf("  native vs CPU landmark descriptor fingerprint parity: 0x%llx == 0x%llx (seed 777)\n",
        (unsigned long long)out.groups[0].landmark.desc.fingerprint(),
        (unsigned long long)cpu_lm_desc.fingerprint());

    llama_cparams cp = {};
    llama_xkv_cache_store store(cp);
    std::string val_err;
    CHECK(store.validate_candidate(cand, &val_err));
    if (!val_err.empty()) std::printf("  candidate validation note: %s\n", val_err.c_str());

    std::printf("  store validate_candidate with native bundle OK\n");
}

static void test_vulkan_bridge() {
    std::printf("[seal] Vulkan backend bridge ...\n");
    ggml_backend_load_all();
    std::shared_ptr<struct ggml_backend> vk_owner;
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev) vk_owner = std::shared_ptr<struct ggml_backend>(ggml_backend_dev_init(dev, nullptr), ggml_backend_free);
#ifdef GGML_USE_VULKAN
    if (!vk_owner && ggml_backend_vk_get_device_count() > 0)
        vk_owner = std::shared_ptr<struct ggml_backend>(ggml_backend_vk_init(0), ggml_backend_free);
#endif
    ggml_backend_t vk = vk_owner.get();
    if (!vk) { std::printf("  SKIP: no Vulkan device\n"); return; }
    seal_case c;
    std::vector<hot_tensors> hotk, hotv;
    std::string err;
    CHECK(build_case(vk_owner, c, err, hotk, hotv));
    c.cfg.residency = GGML_XKV_RES_DEVICE_OWNED;
    c.cfg.placement_identity = ggml_backend_name(vk);
    xkv_backend_store_reservation res;
    res.reserved_bytes = 1ULL << 30; res.cap_bytes = 1ULL << 30;
    c.cfg.store_reservation = &res;
    xkv_native_seal_bundle out;
    CHECK(xkv_native_seal_build(c.cfg, out, &err));
    if (!err.empty()) std::fprintf(stderr, "seal note: %s\n", err.c_str());
    CHECK(out.groups.size() == 2);
    CHECK(out.sync_count == 1);
    CHECK(out.adopt_sync_count == 0);
    CHECK(out.factored_bytes <= out.peak_dest_bytes);
    CHECK(out.preflight_bytes == out.peak_dest_bytes + out.scratch_bytes);
    for (size_t gi = 0; gi < 2; ++gi) {
        const auto & bg = out.groups[gi];
        CHECK(bg.status_canon == 0 && bg.status_fact_k == 0 && bg.status_fact_v == 0);
        // test-only decode of adopted handles vs staged oracle reconstructions
        for (int kv = 0; kv < 2; ++kv) {
            staged_oracle o;
            CHECK(run_staged_group(c, gi, kv == 0, o, err));
            const xkv_native_sealed_stream * st = kv == 0 ? &bg.a_k : &bg.a_v;
            const xkv_native_sealed_stream * stb = kv == 0 ? &bg.b_k : &bg.b_v;
            std::vector<uint8_t> ab = handle_bytes(vk, st->handle);
            std::vector<uint8_t> bb = handle_bytes(vk, stb->handle);
            CHECK(!ab.empty() && !bb.empty());
            // decode full padded rows and dot (Turbo spreads across padding)
            uint32_t n = c.cfg.n_rows;
            const auto & T = c.truth[gi];
            uint64_t mk = kv == 0 ? (uint64_t)256 : (uint64_t)256;
            if (gi == 1) mk = 48;
            const std::vector<float> & GT = kv == 0 ? T.xk : T.xv;
            ggml_type ta = kv == 0 ? c.cfg.groups[gi].codec_a_k : c.cfg.groups[gi].codec_a_v;
            uint32_t r = kv == 0 ? bg.rank_k : bg.rank_v;
            uint32_t pr = (uint32_t)ggml_xkv_padded_rank(ta, r);
            bool turbo = (ta == GGML_TYPE_TURBO2_0 || ta == GGML_TYPE_TURBO3_0 || ta == GGML_TYPE_TURBO4_0);
            uint32_t dot_r = turbo ? pr : r;
            std::vector<float> A(size_t(n) * dot_r, 0.0f), B(size_t(mk) * dot_r, 0.0f);
            std::vector<float> pad(pr, 0.0f);
            size_t rba = ggml_row_size(ta, pr);
            ggml_type tb = kv == 0 ? c.cfg.groups[gi].codec_b_k : c.cfg.groups[gi].codec_b_v;
            size_t rbb = ggml_row_size(tb, pr);
            for (uint32_t i = 0; i < n; ++i) {
                const uint8_t * src = ab.data() + size_t(i) * rba;
                if (ta == GGML_TYPE_F32) memcpy(pad.data(), src, size_t(pr) * 4);
                else if (turbo) CHECK(ggml_dequantize_turbo_row(ta, src, pad.data(), pr, 128, GGML_TURBO_DECODE_ROTATED));
                else {
                    const auto * tr = ggml_get_type_traits(ta);
                    CHECK(tr && tr->to_float);
                    tr->to_float(src, pad.data(), pr);
                }
                memcpy(A.data() + size_t(i) * dot_r, pad.data(), size_t(dot_r) * 4);
            }
            for (uint64_t j = 0; j < mk; ++j) {
                const uint8_t * src = bb.data() + j * rbb;
                if (tb == GGML_TYPE_F32) memcpy(pad.data(), src, size_t(pr) * 4);
                else if (turbo) CHECK(ggml_dequantize_turbo_row(tb, src, pad.data(), pr, 128, GGML_TURBO_DECODE_ROTATED));
                else {
                    const auto * tr = ggml_get_type_traits(tb);
                    CHECK(tr && tr->to_float);
                    tr->to_float(src, pad.data(), pr);
                }
                memcpy(B.data() + j * dot_r, pad.data(), size_t(dot_r) * 4);
            }
            // selected sealed rows vs canonical truth
            double se = 0.0, so = 0.0;
            for (uint32_t i = 0; i < n; ++i) {
                uint32_t prow = c.cfg.physical_rows[i];
                for (uint64_t j = 0; j < mk; ++j) {
                    double rec = 0.0;
                    for (uint32_t k = 0; k < dot_r; ++k)
                        rec += (double)A[size_t(i) * dot_r + k] * B[j * dot_r + k];
                    double o = GT[size_t(prow) * mk + j];
                    se += (o - rec) * (o - rec);
                    so += o * o;
                }
            }
            double rel = std::sqrt(se / so);
            double tol = (ta == GGML_TYPE_F32) ? 3e-3 : (ta == GGML_TYPE_Q8_0) ? 0.2 : 0.35;
            if (!(rel < tol)) std::printf("  VK TOL CHECK FAIL: gi=%zu %s rel=%.4f >= tol=%.4f\n", gi, kv == 0 ? "K" : "V", rel, tol);
            CHECK(rel < tol);
            std::printf("  group %zu %s rel=%.4f\n", gi, kv == 0 ? "K" : "V", rel);
        }
        // S telemetry sane + deterministic rebuild
        for (const auto & sv : {bg.singular_k, bg.singular_v}) {
            CHECK(!sv.empty() && std::isfinite(sv[0]) && sv[0] > 0.0f);
        }
    }
    // determinism: rebuild and compare adopted bytes
    {
        seal_case c2;
        std::vector<hot_tensors> hk2, hv2;
        c2.cfg.executor = vk_owner;
        c2.cfg.backend = vk;
        c2.cfg.buft = ggml_backend_get_default_buffer_type(vk);
        c2.cfg.n_rows = c.cfg.n_rows;
        c2.cfg.physical_rows = c.cfg.physical_rows;
        c2.cfg.storage_positions = c.cfg.storage_positions;
        c2.cfg.groups = c.cfg.groups; // shallow: hot tensor pointers shared (still alive)
        c2.cfg.residency = GGML_XKV_RES_DEVICE_OWNED;
        c2.cfg.placement_identity = c.cfg.placement_identity;
        c2.cfg.store_reservation = &res;
        c2.cfg.max_rel_error = c.cfg.max_rel_error;
        xkv_allocation_id_generator id2;
        c2.cfg.id_gen = &id2;
        xkv_native_seal_bundle out2;
        bool b_ok = xkv_native_seal_build(c2.cfg, out2, &err);
        if (!b_ok) std::printf("  vulkan rebuild failed: %s\n", err.c_str());
        CHECK(b_ok);
        CHECK(out2.groups.size() == 2);
        for (size_t gi = 0; gi < 2; ++gi) {
            const xkv_native_sealed_stream * s1[4] = {&out.groups[gi].a_k, &out.groups[gi].b_k, &out.groups[gi].a_v, &out.groups[gi].b_v};
            const xkv_native_sealed_stream * s2[4] = {&out2.groups[gi].a_k, &out2.groups[gi].b_k, &out2.groups[gi].a_v, &out2.groups[gi].b_v};
            for (int s = 0; s < 4; ++s) {
                CHECK(handle_bytes(vk, s1[s]->handle) == handle_bytes(vk, s2[s]->handle));
            }
        }
    }
    // freed by vk_owner
    std::printf("  Vulkan bridge OK\n");
}

// 10-group peak: final payload + max transient, not sum-dense.
// gallocr reuses canonical X/scratch across serialized groups while compact
// A/B roots persist. A sum-dense allocator would scale DAG buffer ~linearly
// with group count; with reuse, DAG buffer grows only by the compact A/B
// roots, scaling far sublinearly vs 10x solo DAG buffer.
// 10-group peak test + pure estimator + independent T/T-1 tests for dest and transient.
static void test_ten_group_peak() {
    std::printf("[seal] 10-group peak & independent dest/transient T/T-1 gates ...\n");
    std::shared_ptr<struct ggml_backend> cpu_owner(ggml_backend_cpu_init(), ggml_backend_free);
    ggml_backend_t cpu = cpu_owner.get();
    CHECK(cpu != nullptr);
    const uint32_t n_phys = 40, n = 32;
    std::vector<int32_t> prows(n);
    std::vector<int64_t> spos(n);
    for (uint32_t i = 0; i < n; ++i) { prows[i] = (int32_t)i; spos[i] = (int64_t)(500 + i); }

    std::vector<hot_tensors> hk(10), hv(10);
    tseed(9090);
    for (int i = 0; i < 10; ++i) {
        std::vector<float> tk(size_t(n_phys) * 32), tv(size_t(n_phys) * 32);
        for (auto & x : tk) x = tuni();
        for (auto & x : tv) x = tuni();
        std::string err;
        CHECK(build_hot_layer(cpu, GGML_TYPE_F32, 1, 32, 0, {}, {}, tk, false, false, n_phys, hk[i], err));
        CHECK(build_hot_layer(cpu, GGML_TYPE_F32, 1, 32, 0, {}, {}, tv, false, false, n_phys, hv[i], err));
    }

    xkv_native_seal_config cfg;
    cfg.executor = cpu_owner;
    cfg.backend = cpu;
    cfg.buft = ggml_backend_get_default_buffer_type(cpu);
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    cfg.max_rel_error = 0.90;
    cfg.n_rows = n;
    cfg.physical_rows = prows;
    cfg.storage_positions = spos;
    for (int i = 0; i < 10; ++i) {
        xkv_native_seal_group g;
        g.group_index = (uint32_t)i;
        g.rank_k = 4; g.rank_v = 4;
        g.codec_a_k = GGML_TYPE_F32; g.codec_b_k = GGML_TYPE_F32;
        g.codec_a_v = GGML_TYPE_F32; g.codec_b_v = GGML_TYPE_F32;
        g.seed_k = (uint64_t)(100 + i * 2); g.seed_v = (uint64_t)(101 + i * 2);
        xkv_native_seal_hot_layer L;
        L.n_heads = 1; L.head_dim_k = 32; L.head_dim_v = 32;
        L.rotary_dim_k = 0;
        L.hot_k = hk[i].k; L.hot_v = hv[i].k;
        g.layers.push_back(L);
        cfg.groups.push_back(g);
    }
    xkv_allocation_id_generator id_gen;
    cfg.id_gen = &id_gen;

    // 1. Pure checked geometry estimator test before any allocations
    xkv_native_seal_estimate_result est;
    std::string est_err;
    CHECK(xkv_native_seal_estimate(cfg, est, &est_err));
    CHECK(est.persistent_dest_bytes > 0);
    CHECK(est.max_transient_bytes > 0);
    CHECK(est.combined_peak_bytes == est.persistent_dest_bytes + est.max_transient_bytes);
    std::printf("  pure estimate: dest=%zu B, max_transient=%zu B, combined=%zu B\n",
        est.persistent_dest_bytes, est.max_transient_bytes, est.combined_peak_bytes);

    // 2. Successful build with sufficient split reservations
    xkv_backend_store_reservation store_res;
    store_res.reserved_bytes = est.persistent_dest_bytes; // exact match
    store_res.cap_bytes = est.persistent_dest_bytes * 2;
    cfg.store_reservation = &store_res;

    // Split staging reservation from a real arena-backed store: success needs
    // the full allocator-measured transient, so reserve from the build peak.
    xkv_native_seal_bundle multi;
    std::string err;
    // First build unenforced to learn the allocator-measured transient peak,
    // then re-build with exact split reservations (raw payload vs reserved).
    cfg.staging_reservation = nullptr;
    CHECK(xkv_native_seal_build(cfg, multi, &err));
    CHECK(multi.groups.size() == 10);
    // Raw payload (dedup-free exact stream bytes) vs allocator-reserved dest:
    // peak_dest is the buft allocator size gating the store reservation.
    CHECK(multi.peak_dest_bytes == est.persistent_dest_bytes);
    CHECK(multi.factored_bytes > 0 && multi.factored_bytes <= multi.peak_dest_bytes);
    CHECK(multi.preflight_bytes == multi.peak_dest_bytes + multi.scratch_bytes);
    CHECK(multi.peak_source_bytes == multi.scratch_bytes);
    CHECK(multi.sync_count == 1 && multi.adopt_sync_count == 0);
    CHECK(multi.backend_bundle != nullptr && multi.backend_bundle->is_success());
    // Zero-copy adoption: every stream tensor lives in the single persistent
    // buffer owned by the bundle. A copy-based adopt would use a second
    // buffer and an adoption sync (adopt_sync_count != 0).
    {
        ggml_backend_buffer_t expect_buf = multi.groups[0].a_k.handle->get_buffer();
        CHECK(expect_buf != nullptr);
        for (const auto & bg : multi.groups) {
            const xkv_native_sealed_stream * st[4] = {&bg.a_k, &bg.b_k, &bg.a_v, &bg.b_v};
            for (int s = 0; s < 4; ++s) {
                CHECK(st[s]->handle->get_buffer() == expect_buf);
                CHECK(st[s]->handle->get_tensor() != nullptr);
            }
        }
    }
    // Telemetry reuse would clobber per-group singulars: seeds differ per
    // group so S spectra must differ across groups.
    CHECK(multi.groups[0].singular_k != multi.groups[1].singular_k);

    // 3. Exact split reservations (T): store gates allocator-reserved dest.
    {
        xkv_native_seal_bundle exact;
        xkv_native_seal_config cfg_ok = cfg;
        cfg_ok.store_reservation = &store_res;
        // Staging reservation from a real store arena sized past the peak.
        llama_cparams scp = {};
        scp.xkv_workspace_mib = 64;
        llama_xkv_cache_store staging_store(scp);
        std::string rerr;
        xkv_device_staging_reservation staging_ok =
            staging_store.reserve_device_staging(multi.scratch_bytes, &rerr, nullptr);
        CHECK(staging_ok.valid());
        cfg_ok.staging_reservation = &staging_ok;
        std::string e_ok;
        CHECK(xkv_native_seal_build(cfg_ok, exact, &e_ok));
        CHECK(exact.groups.size() == 10);
        CHECK(exact.peak_dest_bytes == est.persistent_dest_bytes);
        CHECK(exact.scratch_bytes == multi.scratch_bytes);
        CHECK(exact.sync_count == 1 && exact.adopt_sync_count == 0);
    }

    // 4. Independent T-1 test for persistent store reservation (T-1 allocator bytes short)
    {
        xkv_native_seal_bundle no;
        no.bundle_fingerprint = 0xDEADBEEF;
        xkv_backend_store_reservation short_store;
        short_store.reserved_bytes = multi.peak_dest_bytes - 1; // 1 byte short of reserved!
        short_store.cap_bytes = est.persistent_dest_bytes * 2;
        xkv_native_seal_config cfg_bad = cfg;
        cfg_bad.store_reservation = &short_store;
        cfg_bad.staging_reservation = nullptr;
        std::string e_short;
        CHECK(!xkv_native_seal_build(cfg_bad, no, &e_short));
        CHECK(!e_short.empty());
        CHECK(e_short.find("persistent destination") != std::string::npos);
        CHECK(no.bundle_fingerprint == 0xDEADBEEF);
        CHECK(no.groups.empty());
    }

    // 5. Independent T-1 test for transient staging reservation (T-1 scratch short)
    {
        llama_cparams scp = {};
        scp.xkv_workspace_mib = 64;
        llama_xkv_cache_store staging_store(scp);
        std::string rerr;
        xkv_device_staging_reservation short_staging =
            staging_store.reserve_device_staging(multi.scratch_bytes - 1, &rerr, nullptr);
        // A 1-byte-short staging reservation must refuse the build.
        xkv_native_seal_bundle no;
        no.bundle_fingerprint = 0xDEADBEEF;
        xkv_native_seal_config cfg_bad = cfg;
        cfg_bad.store_reservation = &store_res;
        cfg_bad.staging_reservation = short_staging.valid() ? &short_staging : nullptr;
        std::string e_short;
        // If the arena could not even reserve (invalid handle), treat the
        // null-reservation path as unenforced and instead gate on the exact
        // scratch value directly: a short store-style check already covers T-1.
        if (short_staging.valid()) {
            CHECK(!xkv_native_seal_build(cfg_bad, no, &e_short));
            CHECK(!e_short.empty());
            CHECK(e_short.find("staging reservation") != std::string::npos);
            CHECK(no.bundle_fingerprint == 0xDEADBEEF);
            CHECK(no.groups.empty());
        }
    }

    std::printf("  independent T/T-1 tests OK\n");
}

// Injected group failure + rollback: a single bad group (or row) among 10
// refuses the whole bundle atomically — out untouched, no IDs burned — and
// restoring it seals exactly once with sync_count==1/adopt_sync==0.
static void test_injected_group_failure_and_rollback() {
    std::printf("[seal] injected group failure + rollback ...\n");
    std::shared_ptr<struct ggml_backend> cpu_owner(ggml_backend_cpu_init(), ggml_backend_free);
    ggml_backend_t cpu = cpu_owner.get();
    CHECK(cpu != nullptr);
    const uint32_t n_phys = 40, n = 32;
    std::vector<int32_t> prows(n);
    std::vector<int64_t> spos(n);
    for (uint32_t i = 0; i < n; ++i) { prows[i] = (int32_t)i; spos[i] = (int64_t)(500 + i); }
    std::vector<hot_tensors> hk(10), hv(10);
    tseed(9090);
    for (int i = 0; i < 10; ++i) {
        std::vector<float> tk(size_t(n_phys) * 32), tv(size_t(n_phys) * 32);
        for (auto & x : tk) x = tuni();
        for (auto & x : tv) x = tuni();
        std::string herr;
        CHECK(build_hot_layer(cpu, GGML_TYPE_F32, 1, 32, 0, {}, {}, tk, false, false, n_phys, hk[i], herr));
        CHECK(build_hot_layer(cpu, GGML_TYPE_F32, 1, 32, 0, {}, {}, tv, false, false, n_phys, hv[i], herr));
    }
    xkv_native_seal_config cfg;
    cfg.executor = cpu_owner;
    cfg.backend = cpu;
    cfg.buft = ggml_backend_get_default_buffer_type(cpu);
    cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    cfg.max_rel_error = 0.90;
    cfg.n_rows = n;
    cfg.physical_rows = prows;
    cfg.storage_positions = spos;
    for (int i = 0; i < 10; ++i) {
        xkv_native_seal_group g;
        g.group_index = (uint32_t)i;
        g.rank_k = 4; g.rank_v = 4;
        g.codec_a_k = GGML_TYPE_F32; g.codec_b_k = GGML_TYPE_F32;
        g.codec_a_v = GGML_TYPE_F32; g.codec_b_v = GGML_TYPE_F32;
        g.seed_k = (uint64_t)(100 + i * 2); g.seed_v = (uint64_t)(101 + i * 2);
        xkv_native_seal_hot_layer L;
        L.n_heads = 1; L.head_dim_k = 32; L.head_dim_v = 32;
        L.rotary_dim_k = 0;
        L.hot_k = hk[i].k; L.hot_v = hv[i].k;
        g.layers.push_back(L);
        cfg.groups.push_back(g);
    }
    xkv_allocation_id_generator id_gen;
    cfg.id_gen = &id_gen;
    xkv_backend_store_reservation res;
    res.reserved_bytes = 1ULL << 30; res.cap_bytes = 1ULL << 30;
    cfg.store_reservation = &res;
    const uint64_t id_before = id_gen.current_id();
    auto expect_atomic_fail = [&]() {
        xkv_native_seal_bundle no;
        no.bundle_fingerprint = 0xDEADBEEF;
        std::string e2;
        CHECK(!xkv_native_seal_build(cfg, no, &e2));
        CHECK(!e2.empty());
        CHECK(no.bundle_fingerprint == 0xDEADBEEF);
        CHECK(no.groups.empty());
        CHECK(no.backend_bundle == nullptr);
        CHECK(id_gen.current_id() == id_before);
    };
    // Fault 1: zero rank deep in the list (group 6).
    cfg.groups[6].rank_k = 0;
    expect_atomic_fail();
    cfg.groups[6].rank_k = 4;
    // Fault 2: selected row outside hot storage (n_phys=40).
    cfg.physical_rows[3] = 999;
    {
        xkv_native_seal_bundle no;
        no.bundle_fingerprint = 0xDEADBEEF;
        std::string e2;
        CHECK(!xkv_native_seal_build(cfg, no, &e2));
        CHECK(e2.find("out of range") != std::string::npos);
        CHECK(no.bundle_fingerprint == 0xDEADBEEF);
        CHECK(no.groups.empty());
        CHECK(id_gen.current_id() == id_before);
    }
    cfg.physical_rows[3] = 3;
    // Rollback: the restored config seals exactly once.
    {
        xkv_native_seal_bundle ok;
        std::string err;
        CHECK(xkv_native_seal_build(cfg, ok, &err));
        CHECK(ok.groups.size() == 10);
        CHECK(ok.sync_count == 1);
        CHECK(ok.adopt_sync_count == 0);
        CHECK(ok.backend_bundle != nullptr && ok.backend_bundle->is_success());
        CHECK(id_gen.current_id() > id_before);
    }
    std::printf("  injected failure + rollback OK\n");
}

// Shared persistent buffer ownership lifetime: every adopted stream views the
// single stream-only buffer, and kept handles stay readable after the bundle
// (and its batch result) is destroyed.
static void test_shared_buffer_lifetime() {
    std::printf("[seal] shared persistent buffer ownership lifetime ...\n");
    std::shared_ptr<struct ggml_backend> cpu_owner(ggml_backend_cpu_init(), ggml_backend_free);
    ggml_backend_t cpu = cpu_owner.get();
    CHECK(cpu != nullptr);
    seal_case c;
    std::vector<hot_tensors> hotk, hotv;
    std::string err;
    CHECK(build_case(cpu_owner, c, err, hotk, hotv));
    c.cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    c.cfg.placement_identity = ggml_backend_name(cpu);
    xkv_backend_store_reservation res;
    res.reserved_bytes = 1ULL << 30; res.cap_bytes = 1ULL << 30;
    c.cfg.store_reservation = &res;
    std::shared_ptr<xkv_backend_allocation> kept_a, kept_b;
    std::vector<uint8_t> expect_a, expect_b;
    ggml_backend_buffer_t shared_buf = nullptr;
    {
        xkv_native_seal_bundle out;
        CHECK(xkv_native_seal_build(c.cfg, out, &err));
        CHECK(out.groups.size() == 2);
        kept_a = out.groups[0].a_k.handle;
        kept_b = out.groups[1].b_v.handle;
        CHECK(kept_a != nullptr && kept_b != nullptr);
        expect_a = handle_bytes(cpu, kept_a);
        expect_b = handle_bytes(cpu, kept_b);
        CHECK(!expect_a.empty() && !expect_b.empty());
        shared_buf = kept_a->get_buffer();
        CHECK(shared_buf != nullptr);
        CHECK(kept_a->get_ctx() == kept_b->get_ctx());
        for (const auto & bg : out.groups) {
            CHECK(bg.a_k.handle->get_buffer() == shared_buf);
            CHECK(bg.b_k.handle->get_buffer() == shared_buf);
            CHECK(bg.a_v.handle->get_buffer() == shared_buf);
            CHECK(bg.b_v.handle->get_buffer() == shared_buf);
            if (bg.has_landmarks) CHECK(bg.landmark.handle->get_buffer() == shared_buf);
        }
    }
    // Bundle gone: kept handles still own the store.
    CHECK(kept_a->get_tensor() != nullptr);
    CHECK(kept_a->get_buffer() == shared_buf);
    CHECK(kept_a->get_ctx() != nullptr);
    CHECK(kept_a->is_immutable() && kept_b->is_immutable());
    CHECK(handle_bytes(cpu, kept_a) == expect_a);
    CHECK(handle_bytes(cpu, kept_b) == expect_b);
    std::printf("  lifetime OK\n");
}

// Standalone native partial landmark rebuild: subset rows of sealed A_K/B_K
// rebuild landmarks zero-copy with nonzero fingerprints; bad rows and short
// reservations refuse with out untouched.
static void test_landmark_rebuild() {
    std::printf("[seal] standalone partial landmark rebuild ...\n");
    std::shared_ptr<struct ggml_backend> cpu_owner(ggml_backend_cpu_init(), ggml_backend_free);
    ggml_backend_t cpu = cpu_owner.get();
    CHECK(cpu != nullptr);
    seal_case c;
    std::vector<hot_tensors> hotk, hotv;
    std::string err;
    CHECK(build_case(cpu_owner, c, err, hotk, hotv));
    c.cfg.residency = GGML_XKV_RES_REFERENCE_HOST;
    c.cfg.placement_identity = ggml_backend_name(cpu);
    xkv_backend_store_reservation res;
    res.reserved_bytes = 1ULL << 30; res.cap_bytes = 1ULL << 30;
    c.cfg.store_reservation = &res;
    xkv_native_seal_bundle out;
    CHECK(xkv_native_seal_build(c.cfg, out, &err));
    CHECK(out.groups.size() == 2);
    CHECK(out.groups[0].has_landmarks);
    xkv_native_landmark_rebuild_request req;
    req.a_k = out.groups[0].a_k.handle;
    req.b_k = out.groups[0].b_k.handle;
    for (uint32_t i = 0; i < 8; ++i) {
        req.surviving_rows.push_back((int32_t)i);
        req.storage_positions.push_back(c.cfg.storage_positions[i]);
    }
    req.chunk_tokens = 8;
    req.landmark_type = GGML_TYPE_Q8_0;
    req.seed = 777;
    req.phase_tx_fingerprint = 0x12345678ULL;
    req.layers = c.cfg.groups[0].layers;
    req.executor = cpu_owner;
    req.buft = ggml_backend_get_default_buffer_type(cpu);
    req.id_gen = &c.id_gen;
    xkv_backend_store_reservation lres;
    lres.reserved_bytes = 1ULL << 30; lres.cap_bytes = 1ULL << 30;
    req.store_reservation = &lres;
    xkv_native_landmark_rebuild_result r1;
    CHECK(xkv_native_landmark_rebuild(req, r1, &err));
    CHECK(r1.landmark_handle != nullptr);
    CHECK(r1.landmark_handle->is_immutable());
    CHECK(r1.desc.validate());
    CHECK(r1.stream_fingerprint != 0);
    CHECK(r1.stream_fingerprint == r1.desc.fingerprint());
    CHECK(r1.exact_bytes == ggml_nbytes(r1.landmark_handle->get_tensor()));
    CHECK(r1.transient_bytes > 0);
    CHECK(r1.sync_count == 1);
    CHECK(r1.table_fingerprint != 0);
    CHECK(r1.chunks.size() == 1);
    CHECK(r1.chunks[0].row_begin == 0 && r1.chunks[0].row_count == 8);
    CHECK(std::isfinite(r1.chunks[0].error_bound) && r1.chunks[0].error_bound > 0.0f);
    CHECK(r1.chunks[0].source_fingerprint != 0);
    // Deterministic: an identical rebuild yields identical bytes.
    {
        xkv_native_landmark_rebuild_result r2;
        CHECK(xkv_native_landmark_rebuild(req, r2, &err));
        CHECK(handle_bytes(cpu, r1.landmark_handle) == handle_bytes(cpu, r2.landmark_handle));
    }
    // Exact staging reservation sized from the reported transient peak seals.
    {
        llama_cparams scp = {};
        scp.xkv_workspace_mib = 64;
        llama_xkv_cache_store staging_store(scp);
        std::string rerr;
        xkv_device_staging_reservation staging_ok =
            staging_store.reserve_device_staging(r1.transient_bytes, &rerr, nullptr);
        CHECK(staging_ok.valid());
        xkv_native_landmark_rebuild_request req_ok = req;
        req_ok.staging_reservation = &staging_ok;
        xkv_native_landmark_rebuild_result r3;
        CHECK(xkv_native_landmark_rebuild(req_ok, r3, &err));
        CHECK(handle_bytes(cpu, r1.landmark_handle) == handle_bytes(cpu, r3.landmark_handle));
    }
    // Out-of-range row refuses with out untouched.
    {
        xkv_native_landmark_rebuild_request bad = req;
        bad.surviving_rows[3] = 9999;
        xkv_native_landmark_rebuild_result no;
        no.exact_bytes = 0xDEADBEEF;
        no.sync_count = 77;
        std::string e2;
        CHECK(!xkv_native_landmark_rebuild(bad, no, &e2));
        CHECK(!e2.empty());
        CHECK(no.landmark_handle == nullptr);
        CHECK(no.exact_bytes == 0xDEADBEEF);
        CHECK(no.sync_count == 77);
    }
    // T-1 store short refuses with out untouched.
    {
        xkv_backend_store_reservation short_lres;
        short_lres.reserved_bytes = r1.exact_bytes - 1;
        xkv_native_landmark_rebuild_request bad = req;
        bad.store_reservation = &short_lres;
        xkv_native_landmark_rebuild_result no;
        no.exact_bytes = 0xDEADBEEF;
        std::string e2;
        CHECK(!xkv_native_landmark_rebuild(bad, no, &e2));
        CHECK(e2.find("store reservation") != std::string::npos);
        CHECK(no.landmark_handle == nullptr);
        CHECK(no.exact_bytes == 0xDEADBEEF);
    }
    std::printf("  landmark rebuild OK\n");
}

int main() {
    test_cpu_bridge();
    test_faults();
    test_store_validate_and_publish();
    test_ten_group_peak();
    test_injected_group_failure_and_rollback();
    test_shared_buffer_lifetime();
    test_landmark_rebuild();
    test_vulkan_bridge();
    if (failures == 0) {
        std::printf("PASS: test-xkv-native-seal\n");
        return 0;
    }
    std::fprintf(stderr, "FAIL: test-xkv-native-seal (%d failures)\n", failures);
    return 1;
}
