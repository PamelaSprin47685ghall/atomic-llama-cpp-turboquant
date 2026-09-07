// test-xkv-vulkan-attention.cpp — backend tests for GGML_OP_XKV_ATTENTION.
//
// Dual-source indexed attention (Stage 2, one op per KV head): single FP32
// online softmax over bounded hot rows + reconstructed cold rows.
//
// Coverage (CPU oracle vs independent dense ref vs CPU graph vs Vulkan graph):
//  - ordinary causal, RERoT variable DDVR, two queries sharing a key in
//    different groups, multi-GQA, hot+cold global normalization, sink once,
//    softcap, all-masked (empty + valid==0) exact zeros, zero-range Vulkan
//    dummy bindings, current hot write
//    dependency, Dk!=Dv, capacity padding (garbage ignored), future draft
//    isolation, malformed offsets/source/index/group (+dupe-reject), F16 cold,
//    Q8_0 hot, Turbo4 K/Turbo2 V canonical decode,
//    entry strides 2 and 4, multi-stream configured stream, zero D2H/upload
//    structure (device-side hot write via in-graph set_rows; only dst/status ever
//    read back).
// Vulkan unavailable skips only the Vulkan subcase; claimed support but
// execution failure fails. No project-wide commands; standalone binary.

#include "ggml.h"
#include "ggml-xkv.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int failures = 0;
#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static bool is_nan_row(const std::vector<float> & v, size_t base, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (v[base + i] != v[base + i]) return true;
    }
    return false;
}

// Encode one logical row into a packed hot/cold stream row.
static bool encode_row(ggml_type type, const float * logical, uint32_t padded, void * out_row) {
    if (type == GGML_TYPE_F32) {
        memcpy(out_row, logical, (size_t)padded * 4);
        return true;
    }
    if (type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0) {
        const auto * tr = ggml_get_type_traits(type);
        if (!tr || !tr->from_float_ref) return false;
        std::vector<float> pad(padded, 0.0f);
        // caller passes full padded width; direct convert
        tr->from_float_ref(logical, (char *)out_row, padded);
        return true;
    }
    if (type == GGML_TYPE_TURBO2_0 || type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0) {
        return ggml_quantize_turbo_row(type, logical, out_row, padded, 128);
    }
    return false;
}

// Independent dense reference (fresh implementation, no oracle code reuse):
// per (query, gqa head) softmax over hot F32 rows + cold F32 rows.
static std::vector<float> ref_dense(
        const std::vector<float> & q, uint32_t dk, uint32_t gqa, uint32_t ng,
        const std::vector<float> & kh, // [hot_rows][dk] canonical
        const std::vector<float> & vh, // [hot_rows][dv] canonical
        const std::vector<float> & kc, // [n_cold][dk]
        const std::vector<float> & vc, // [n_cold][dv]
        uint32_t dv, uint32_t hot_rows,
        const std::vector<int32_t> & entries, int stride, // (source,row,group[,valid]) or (key,group)
        const std::vector<int32_t> & offsets,
        float scale, float softcap,
        const std::vector<float> & sinks) { // empty = none
    CHECK(q.size() == (size_t) dk * gqa * ng);
    const uint32_t nq = (uint32_t)offsets.size() - 1;
    std::vector<float> out((size_t)dv * gqa * nq, 0.0f);
    const float eff = softcap > 0.0f ? scale / softcap : scale;
    for (uint32_t qq = 0; qq < nq; ++qq) {
        for (uint32_t g = 0; g < gqa; ++g) {
            float mx = -INFINITY, sum = 0.0f;
            std::vector<float> acc(dv, 0.0f);
            for (int32_t e = offsets[qq]; e < offsets[qq + 1]; ++e) {
                int source, row, group;
                if (stride == 4) {
                    if (entries[(size_t)e * 4 + 3] == 0) continue;
                    source = entries[(size_t)e * 4 + 0];
                    row    = entries[(size_t)e * 4 + 1];
                    group  = entries[(size_t)e * 4 + 2];
                } else {
                    int key = entries[(size_t)e * 2 + 0];
                    group = entries[(size_t)e * 2 + 1];
                    if ((uint32_t)key < hot_rows) { source = 1; row = key; }
                    else { source = 2; row = key - (int32_t)hot_rows; }
                }
                const float * kv = (source == 1) ? kh.data() + (size_t)row * dk
                                                 : kc.data() + (size_t)row * dk;
                const float * vv = (source == 1) ? vh.data() + (size_t)row * dv
                                                 : vc.data() + (size_t)row * dv;
                const float * qv = q.data() + ((size_t)group * gqa + g) * dk;
                double dot = 0.0;
                for (uint32_t d = 0; d < dk; ++d) dot += (double)qv[d] * (double)kv[d];
                float s = (float)(dot * eff);
                if (softcap > 0.0f) s = softcap * tanhf(s);
                if (mx <= -INFINITY) { mx = s; sum = 1.0f; acc.assign(vv, vv + dv); }
                else if (s > mx) {
                    float f = expf(mx - s); mx = s;
                    for (uint32_t d = 0; d < dv; ++d) acc[d] = acc[d] * f + vv[d];
                    sum = sum * f + 1.0f;
                } else {
                    float f = expf(s - mx); sum += f;
                    for (uint32_t d = 0; d < dv; ++d) acc[d] += f * vv[d];
                }
            }
            if (!sinks.empty()) {
                float ss = sinks[g];
                if (mx <= -INFINITY) { mx = ss; sum = 1.0f; }
                else if (ss > mx) {
                    float f = expf(mx - ss); mx = ss;
                    for (uint32_t d = 0; d < dv; ++d) acc[d] *= f;
                    sum = sum * f + 1.0f;
                } else sum += expf(ss - mx);
            }
            if (sum > 0.0f) {
                for (uint32_t d = 0; d < dv; ++d)
                    out[((size_t)qq * gqa + g) * dv + d] = acc[d] / sum;
            }
        }
    }
    return out;
}

struct graph_out {
    bool ok = false;
    std::vector<float> dst;
    int32_t status = -999;
};

static ggml_xkv_attention_params base_params(uint32_t nq, uint32_t ng, uint32_t gqa,
        uint32_t dk, uint32_t dv, uint32_t hot_rows, uint32_t n_cold, uint32_t n_entries);

// Build + run one attention graph on the given backend. When requested, the
// attention reads functional set_rows outputs so the hot-row write is an
// explicit dependency in the same graph.
static graph_out run_graph(ggml_backend_t backend,
        ggml_type hot_k_type, const std::vector<uint8_t> & kh_b, uint32_t dk,
        ggml_type hot_v_type, const std::vector<uint8_t> & vh_b, uint32_t dv,
        uint32_t n_kv_heads, uint32_t hot_rows, uint32_t n_stream,
        ggml_type cold_k_type, const std::vector<uint8_t> & kc_b,
        ggml_type cold_v_type, const std::vector<uint8_t> & vc_b, uint32_t n_cold,
        const std::vector<float> & q, uint32_t gqa, uint32_t ng,
        const std::vector<int32_t> & entries, int stride, uint32_t n_entries,
        const std::vector<int32_t> & offsets, const std::vector<float> & sinks,
        const ggml_xkv_attention_params & p,
        bool do_hot_write, const std::vector<uint8_t> & write_row, uint32_t write_row_idx) {
    graph_out r;
    const uint32_t nq = p.n_queries;
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor * tq = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, dk, gqa, ng);
    ggml_tensor * tk = ggml_new_tensor_4d(ctx.get(), hot_k_type, dk, n_kv_heads, hot_rows, n_stream);
    ggml_tensor * tv = ggml_new_tensor_4d(ctx.get(), hot_v_type, dv, n_kv_heads, hot_rows, n_stream);
    ggml_tensor * ck = ggml_new_tensor_2d(ctx.get(), cold_k_type, dk, n_cold);
    ggml_tensor * cv = ggml_new_tensor_2d(ctx.get(), cold_v_type, dv, n_cold);
    ggml_tensor * en = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, stride, n_entries);
    ggml_tensor * off = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, nq + 1);
    ggml_tensor * sn = sinks.empty() ? nullptr
        : ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, gqa);
    ggml_tensor * st = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_tensor * stage_k = nullptr;
    ggml_tensor * stage_v = nullptr;
    ggml_tensor * write_idx = nullptr;
    ggml_tensor * tk_attn = tk;
    ggml_tensor * tv_attn = tv;
    const size_t write_k_bytes = ggml_row_size(hot_k_type, dk);
    const size_t write_v_bytes = ggml_row_size(hot_v_type, dv);
    if (do_hot_write) {
        if (write_row_idx >= hot_rows || write_row.size() != write_k_bytes + write_v_bytes) return r;
        stage_k = ggml_new_tensor_2d(ctx.get(), hot_k_type, dk, 1);
        stage_v = ggml_new_tensor_2d(ctx.get(), hot_v_type, dv, 1);
        write_idx = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 1);
        ggml_tensor * tk_flat = ggml_view_2d(ctx.get(), tk, dk,
            (int64_t)n_kv_heads * hot_rows * n_stream, tk->nb[1], 0);
        ggml_tensor * tv_flat = ggml_view_2d(ctx.get(), tv, dv,
            (int64_t)n_kv_heads * hot_rows * n_stream, tv->nb[1], 0);
        ggml_tensor * tk_written = ggml_set_rows(ctx.get(), tk_flat, stage_k, write_idx);
        ggml_tensor * tv_written = ggml_set_rows(ctx.get(), tv_flat, stage_v, write_idx);
        tk_attn = ggml_view_4d(ctx.get(), tk_written, dk, n_kv_heads, hot_rows, n_stream,
            tk->nb[1], tk->nb[2], tk->nb[3], 0);
        tv_attn = ggml_view_4d(ctx.get(), tv_written, dv, n_kv_heads, hot_rows, n_stream,
            tv->nb[1], tv->nb[2], tv->nb[3], 0);
    }
    ggml_tensor * out = ggml_xkv_attention(ctx.get(), tq, tk_attn, tv_attn, ck, cv, en, off, sn, st, nullptr, &p);
    if (!out) return r;
    if (!ggml_backend_supports_op(backend, out)) return r; // caller decides: valid cases must be supported
    r.ok = true; // supported; execution checked below
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    ggml_backend_tensor_set(tq, q.data(), 0, q.size() * 4);
    if (!kh_b.empty()) ggml_backend_tensor_set(tk, kh_b.data(), 0, kh_b.size());
    if (!vh_b.empty()) ggml_backend_tensor_set(tv, vh_b.data(), 0, vh_b.size());
    if (!kc_b.empty()) ggml_backend_tensor_set(ck, kc_b.data(), 0, kc_b.size());
    if (!vc_b.empty()) ggml_backend_tensor_set(cv, vc_b.data(), 0, vc_b.size());
    if (!entries.empty()) ggml_backend_tensor_set(en, entries.data(), 0, entries.size() * 4);
    ggml_backend_tensor_set(off, offsets.data(), 0, offsets.size() * 4);
    if (sn) ggml_backend_tensor_set(sn, sinks.data(), 0, sinks.size() * 4);
    if (do_hot_write) {
        ggml_backend_tensor_set(stage_k, write_row.data(), 0, write_k_bytes);
        ggml_backend_tensor_set(stage_v, write_row.data() + write_k_bytes, 0, write_v_bytes);
        const int64_t flat_row = (int64_t)p.stream * n_kv_heads * hot_rows +
            (int64_t)write_row_idx * n_kv_heads + p.kv_head;
        ggml_backend_tensor_set(write_idx, &flat_row, 0, sizeof(flat_row));
    }
    int32_t zero = 0;
    ggml_backend_tensor_set(st, &zero, 0, 4);
    ggml_cgraph * g = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(g, out);
    if (ggml_backend_graph_compute(backend, g) != GGML_STATUS_SUCCESS) { r.ok = false; return r; }
    ggml_backend_synchronize(backend);
    std::vector<float> state(ggml_nelements(out), 0.0f);
    ggml_backend_tensor_get(out, state.data(), 0, state.size() * sizeof(float));
    r.dst.resize((size_t)dv * gqa * nq);
    for (uint32_t qq = 0; qq < nq; ++qq) {
        for (uint32_t h = 0; h < gqa; ++h) {
            const size_t si = ((size_t)qq * gqa + h) * (dv + 2);
            const size_t di = ((size_t)qq * gqa + h) * dv;
            std::copy_n(state.data() + si, dv, r.dst.data() + di);
        }
    }
    ggml_backend_tensor_get(st, &r.status, 0, 4);
    return r;
}

// Build one explicit three-tile DAG. Tile 0 consumes hot rows, every tile
// consumes two local cold rows, and only tile 0 sees the sink. Each dst is the
// next tile's immutable carry; expanding only the final node must schedule the
// full chain and produce one global softmax without a dense cold union.
static graph_out run_three_tile_graph(
        ggml_backend_t backend,
        const std::vector<float> & q,
        const std::vector<float> & kh,
        const std::vector<float> & vh,
        const std::vector<float> & kc,
        const std::vector<float> & vc,
        uint32_t dk, uint32_t dv, uint32_t gqa,
        const std::vector<float> & sinks) {
    graph_out r;
    constexpr uint32_t n_tiles = 3;
    constexpr uint32_t cold_per_tile = 2;
    const uint32_t hot_rows = (uint32_t)(kh.size() / dk);
    if (hot_rows == 0 || vh.size() != (size_t)hot_rows * dv ||
        kc.size() != (size_t)n_tiles * cold_per_tile * dk ||
        vc.size() != (size_t)n_tiles * cold_per_tile * dv ||
        q.size() != (size_t)dk * gqa || sinks.size() != gqa) {
        return r;
    }

    ggml_init_params ip = {
        ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false), nullptr, true
    };
    ggml_context_ptr ctx(ggml_init(ip));
    ggml_tensor * tq = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, dk, gqa, 1);
    ggml_tensor * tk = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, dk, 1, hot_rows, 1);
    ggml_tensor * tv = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, dv, 1, hot_rows, 1);
    ggml_tensor * sn = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, gqa);
    ggml_tensor * st = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);

    std::vector<ggml_tensor *> cks, cvs, ens, offs, states;
    cks.reserve(n_tiles); cvs.reserve(n_tiles); ens.reserve(n_tiles);
    offs.reserve(n_tiles); states.reserve(n_tiles);
    std::vector<std::vector<int32_t>> tile_entries(n_tiles);
    ggml_tensor * carry = nullptr;
    for (uint32_t ti = 0; ti < n_tiles; ++ti) {
        ggml_tensor * ck = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, dk, cold_per_tile);
        ggml_tensor * cv = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, dv, cold_per_tile);
        auto & e = tile_entries[ti];
        if (ti == 0) {
            for (uint32_t row = 0; row < hot_rows; ++row) {
                e.insert(e.end(), {GGML_XKV_ATTN_SOURCE_HOT, (int32_t)row, 0, 1});
            }
        }
        e.insert(e.end(), {GGML_XKV_ATTN_SOURCE_COLD, 0, 0, 1});
        e.insert(e.end(), {GGML_XKV_ATTN_SOURCE_COLD, 1, 0, 1});
        ggml_tensor * en = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, e.size() / 4);
        ggml_tensor * off = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 2);

        ggml_xkv_attention_params p = base_params(1, 1, gqa, dk, dv, hot_rows,
            cold_per_tile, (uint32_t)e.size() / 4);
        p.flags = ti == 0 ? GGML_XKV_ATTN_FLAG_FIRST_TILE : 0;
        if (ti + 1 == n_tiles) p.flags |= GGML_XKV_ATTN_FLAG_FINAL_TILE;
        ggml_tensor * out = ggml_xkv_attention(ctx.get(), tq, tk, tv, ck, cv, en, off,
            ti == 0 ? sn : nullptr, st, carry, &p);
        if (!out || !ggml_backend_supports_op(backend, out)) return r;
        cks.push_back(ck); cvs.push_back(cv); ens.push_back(en); offs.push_back(off);
        states.push_back(out);
        carry = out;
    }

    r.ok = true;
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) { r.ok = false; return r; }
    ggml_backend_tensor_set(tq, q.data(), 0, q.size() * sizeof(float));
    ggml_backend_tensor_set(tk, kh.data(), 0, kh.size() * sizeof(float));
    ggml_backend_tensor_set(tv, vh.data(), 0, vh.size() * sizeof(float));
    ggml_backend_tensor_set(sn, sinks.data(), 0, sinks.size() * sizeof(float));
    int32_t zero = 0;
    ggml_backend_tensor_set(st, &zero, 0, sizeof(zero));
    for (uint32_t ti = 0; ti < n_tiles; ++ti) {
        ggml_backend_tensor_set(cks[ti], kc.data() + (size_t)ti * cold_per_tile * dk,
            0, (size_t)cold_per_tile * dk * sizeof(float));
        ggml_backend_tensor_set(cvs[ti], vc.data() + (size_t)ti * cold_per_tile * dv,
            0, (size_t)cold_per_tile * dv * sizeof(float));
        ggml_backend_tensor_set(ens[ti], tile_entries[ti].data(), 0,
            tile_entries[ti].size() * sizeof(int32_t));
        const int32_t ov[2] = {0, (int32_t)(tile_entries[ti].size() / 4)};
        ggml_backend_tensor_set(offs[ti], ov, 0, sizeof(ov));
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, states.back());
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        r.ok = false;
        return r;
    }
    ggml_backend_synchronize(backend);
    std::vector<float> state(ggml_nelements(states.back()), 0.0f);
    ggml_backend_tensor_get(states.back(), state.data(), 0, state.size() * sizeof(float));
    r.dst.resize((size_t)dv * gqa);
    for (uint32_t h = 0; h < gqa; ++h) {
        std::copy_n(state.data() + (size_t)h * (dv + 2), dv, r.dst.data() + (size_t)h * dv);
    }
    ggml_backend_tensor_get(st, &r.status, 0, sizeof(r.status));
    return r;
}

// Direct oracle run (raw pointers) for fault/edge checks.
static bool run_oracle_direct(
        const std::vector<float> & q, uint32_t dk, uint32_t gqa, uint32_t ng,
        const void * kh, ggml_type tkh, const void * vh, ggml_type tvh, uint32_t dv,
        uint32_t hot_rows, uint32_t n_stream,
        const void * kc, ggml_type tck, const void * vc, ggml_type tcv,
        uint32_t n_cold, const std::vector<int32_t> & entries, int stride,
        const std::vector<int32_t> & offsets, const std::vector<float> & sinks,
        const ggml_xkv_attention_params & p, std::vector<float> & out, int32_t & status) {
    const uint32_t nq = p.n_queries;
    std::vector<float> state((size_t)(dv + 2) * gqa * nq, 0.0f);
    size_t kh_rs = ggml_row_size(tkh, dk), vh_rs = ggml_row_size(tvh, dv);
    size_t kc_rs = ggml_row_size(tck, dk), vc_rs = ggml_row_size(tcv, dv);
    std::vector<float> tmp((size_t)dk + dv, 0.0f);
    char err[256] = {0};
    bool ok = ggml_xkv_attention_oracle(
        q.data(), dk, gqa, ng, 4, (size_t)dk * 4, (size_t)dk * gqa * 4,
        kh, tkh, dk, 1, hot_rows, n_stream, ggml_type_size(tkh), dk * ggml_type_size(tkh), kh_rs, kh_rs * hot_rows,
        vh, tvh, dv, 1, hot_rows, n_stream, ggml_type_size(tvh), dv * ggml_type_size(tvh), vh_rs, vh_rs * hot_rows,
        kc, tck, dk, n_cold, ggml_type_size(tck), kc_rs,
        vc, tcv, dv, n_cold, ggml_type_size(tcv), vc_rs,
        entries.data(), stride, (int64_t)entries.size() / stride,
        offsets.data(), (int64_t)offsets.size(),
        sinks.empty() ? nullptr : sinks.data(), &status, nullptr, &p,
        state.data(), dv + 2, gqa, nq, 4, (size_t)(dv + 2) * 4, (size_t)(dv + 2) * gqa * 4,
        tmp.data(), tmp.size(), err, sizeof(err));
    out.resize((size_t)dv * gqa * nq);
    for (uint32_t qq = 0; qq < nq; ++qq) {
        for (uint32_t h = 0; h < gqa; ++h) {
            const size_t si = ((size_t)qq * gqa + h) * (dv + 2);
            const size_t di = ((size_t)qq * gqa + h) * dv;
            std::copy_n(state.data() + si, dv, out.data() + di);
        }
    }
    if (!ok) std::fprintf(stderr, "oracle: %s\n", err);
    return ok;
}

static ggml_xkv_attention_params base_params(uint32_t nq, uint32_t ng, uint32_t gqa,
        uint32_t dk, uint32_t dv, uint32_t hot_rows, uint32_t n_cold, uint32_t n_entries) {
    ggml_xkv_attention_params p = {};
    p.version = GGML_XKV_ATTN_VERSION;
    p.scale = 1.0f / sqrtf((float)dk);
    p.logit_softcap = 0.0f;
    p.n_queries = nq; p.n_groups = ng; p.gqa_ratio = gqa;
    p.dim_k = dk; p.dim_v = dv;
    p.hot_rows = hot_rows; p.n_cold = n_cold; p.n_entries = n_entries;
    p.kv_head = 0; p.stream = 0;
    p.flags = GGML_XKV_ATTN_FLAG_FIRST_TILE | GGML_XKV_ATTN_FLAG_FINAL_TILE;
    return p;
}

// Pack helpers: F32 canonical hot/cold matrices.
static void fill_mat(std::mt19937 & rng, std::vector<float> & m, float s = 0.5f) {
    std::normal_distribution<float> d(0.0f, s);
    for (auto & x : m) x = d(rng);
}

static void check_case(ggml_backend_t cpu_b, ggml_backend_t vk_b, const char * name,
        ggml_type hot_k_type, const std::vector<uint8_t> & kh_b,
        ggml_type hot_v_type, const std::vector<uint8_t> & vh_b,
        ggml_type cold_k_type, const std::vector<uint8_t> & kc_b,
        ggml_type cold_v_type, const std::vector<uint8_t> & vc_b,
        const std::vector<float> & q, uint32_t dk, uint32_t dv,
        uint32_t gqa, uint32_t ng, uint32_t n_kv_heads, uint32_t hot_rows,
        uint32_t n_stream, uint32_t n_cold,
        const std::vector<int32_t> & entries, int stride, uint32_t n_entries,
        const std::vector<int32_t> & offsets,
        const std::vector<float> & sinks, const ggml_xkv_attention_params & p,
        const std::vector<float> & ref, float tol_cpu, float tol_vk) {
    const uint32_t nq = p.n_queries;
    std::vector<float> orc;
    int32_t ost = -999;
    // oracle needs raw hot/cold: decode packed streams to F32 for the direct call is
    // unnecessary — pass packed pointers with real types/strides.
    bool ok = run_oracle_direct(q, dk, gqa, ng, kh_b.data(), hot_k_type, vh_b.data(), hot_v_type,
        dv, hot_rows, n_stream, kc_b.data(), cold_k_type, vc_b.data(), cold_v_type, n_cold,
        entries, stride, offsets, sinks, p, orc, ost);
    CHECK(ok && ost == 0);
    float e_or = max_abs_diff(orc, ref);
    std::fprintf(stderr, "%s: oracle-vs-ref = %.9g (tol %.9g)\n", name, e_or, tol_cpu);
    CHECK(e_or < tol_cpu);

    graph_out c = run_graph(cpu_b, hot_k_type, kh_b, dk, hot_v_type, vh_b, dv,
        n_kv_heads, hot_rows, n_stream, cold_k_type, kc_b, cold_v_type, vc_b, n_cold,
        q, gqa, ng, entries, stride, n_entries, offsets, sinks, p, false, {}, 0);
    CHECK(c.ok && c.status == 0);
    if (c.ok) {
        float e = max_abs_diff(c.dst, ref);
        std::fprintf(stderr, "%s: cpu-vs-ref = %.9g (tol %.9g)\n", name, e, tol_cpu);
        CHECK(e < tol_cpu);
    }
    if (vk_b) {
        graph_out v = run_graph(vk_b, hot_k_type, kh_b, dk, hot_v_type, vh_b, dv,
            n_kv_heads, hot_rows, n_stream, cold_k_type, kc_b, cold_v_type, vc_b, n_cold,
            q, gqa, ng, entries, stride, n_entries, offsets, sinks, p, false, {}, 0);
        CHECK(v.ok); // claimed support but execution failure fails
        if (v.ok) {
            CHECK(v.status == 0);
            float e = max_abs_diff(v.dst, ref);
            std::fprintf(stderr, "%s: vk-vs-ref = %.9g (tol %.9g)\n", name, e, tol_vk);
            CHECK(e < tol_vk);
        }
    }
    (void)nq;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t cpu_b = ggml_backend_cpu_init();
    CHECK(cpu_b != nullptr);
    ggml_backend_t vk_b = nullptr;
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev) {
        vk_b = ggml_backend_dev_init(gpu_dev, nullptr);
        if (!vk_b) std::puts("SKIP: GPU device present but init failed (Vulkan subcases skipped)");
    }
#ifdef GGML_USE_VULKAN
    if (!vk_b && ggml_backend_vk_get_device_count() > 0) {
        vk_b = ggml_backend_vk_init(0);
    }
#endif
    if (!vk_b) {
        std::puts("SKIP: no GPU backend; Vulkan subcases skipped, CPU oracle checks still run");
    }

    // ---- 1. ordinary causal (F32, 1 group, gqa=2) ----
    {
        const uint32_t dk = 32, dv = 32, gqa = 2, ng = 1, nq = 4, hot_rows = 6, n_cold = 4;
        std::mt19937 rng(11);
        std::vector<float> q((size_t)dk * gqa * ng, 0), kh((size_t)hot_rows * dk, 0),
            vh((size_t)hot_rows * dv, 0), kc((size_t)n_cold * dk, 0), vc((size_t)n_cold * dv, 0);
        fill_mat(rng, q); fill_mat(rng, kh); fill_mat(rng, vh); fill_mat(rng, kc); fill_mat(rng, vc);
        // causal: query i sees hot rows [0..i] + cold row (i % n_cold)
        std::vector<int32_t> entries, offsets = {0};
        for (uint32_t i = 0; i < nq; ++i) {
            for (uint32_t r = 0; r <= i; ++r) {
                entries.insert(entries.end(), {1, (int32_t)r, 0, 1});
            }
            entries.insert(entries.end(), {2, (int32_t)(i % n_cold), 0, 1});
            offsets.push_back((int32_t)entries.size() / 4);
        }
        auto p = base_params(nq, ng, gqa, dk, dv, hot_rows, n_cold, (uint32_t)entries.size() / 4);
        std::vector<float> ref = ref_dense(q, dk, gqa, ng, kh, vh, kc, vc, dv, hot_rows,
            entries, 4, offsets, p.scale, 0.0f, {});
        check_case(cpu_b, vk_b, "causal", GGML_TYPE_F32, std::vector<uint8_t>((char *)kh.data(), (char *)kh.data() + kh.size() * 4),
            GGML_TYPE_F32, std::vector<uint8_t>((char *)vh.data(), (char *)vh.data() + vh.size() * 4),
            GGML_TYPE_F32, std::vector<uint8_t>((char *)kc.data(), (char *)kc.data() + kc.size() * 4),
            GGML_TYPE_F32, std::vector<uint8_t>((char *)vc.data(), (char *)vc.data() + vc.size() * 4),
            q, dk, dv, gqa, ng, 1, hot_rows, 1, n_cold, entries, 4,
            (uint32_t)entries.size() / 4, offsets, {}, p, ref, 1e-5f, 1e-5f);
    }

    // ---- 2. variable DDVR (3 groups) + multi-GQA(4) + sink + softcap ----
    {
        const uint32_t dk = 24, dv = 24, gqa = 4, ng = 3, nq = 3, hot_rows = 5, n_cold = 6;
        std::mt19937 rng(22);
        std::vector<float> q((size_t)dk * gqa * ng, 0), kh((size_t)hot_rows * dk, 0),
            vh((size_t)hot_rows * dv, 0), kc((size_t)n_cold * dk, 0), vc((size_t)n_cold * dv, 0);
        fill_mat(rng, q); fill_mat(rng, kh); fill_mat(rng, vh); fill_mat(rng, kc); fill_mat(rng, vc);
        // q0: groups {0,2} hot rows + cold; q1: group {1}; q2: all groups
        std::vector<int32_t> entries = {
            1,0,0,1, 1,1,0,1, 2,0,2,1, 2,1,2,1,
            1,2,1,1, 2,2,1,1, 2,3,1,1,
            1,3,0,1, 1,4,1,1, 2,4,2,1, 2,5,0,1, 1,0,2,1,
        };
        std::vector<int32_t> offsets = {0, 4, 7, 12};
        std::vector<float> sinks = {0.5f, -0.25f, 1.0f, 0.0f};
        auto p = base_params(nq, ng, gqa, dk, dv, hot_rows, n_cold, 12);
        p.logit_softcap = 20.0f;
        std::vector<float> ref = ref_dense(q, dk, gqa, ng, kh, vh, kc, vc, dv, hot_rows,
            entries, 4, offsets, p.scale, 20.0f, sinks);
        auto f2b = [](const std::vector<float> & v) {
            return std::vector<uint8_t>((const char *)v.data(), (const char *)v.data() + v.size() * 4);
        };
        check_case(cpu_b, vk_b, "ddvr-sink-softcap", GGML_TYPE_F32, f2b(kh),
            GGML_TYPE_F32, f2b(vh), GGML_TYPE_F32, f2b(kc), GGML_TYPE_F32, f2b(vc),
            q, dk, dv, gqa, ng, 1, hot_rows, 1, n_cold, entries, 4, 12, offsets, sinks, p, ref, 1e-5f, 1e-4f);
    }

    // ---- 3. shared key in different groups, 2-wide entries, 2 streams ----
    {
        const uint32_t dk = 16, dv = 16, gqa = 2, ng = 2, nq = 2, hot_rows = 4, n_cold = 2;
        std::mt19937 rng(33);
        std::vector<float> q((size_t)dk * gqa * ng, 0), kh((size_t)hot_rows * dk, 0),
            vh((size_t)hot_rows * dv, 0), kc((size_t)n_cold * dk, 0), vc((size_t)n_cold * dv, 0);
        fill_mat(rng, q); fill_mat(rng, kh); fill_mat(rng, vh); fill_mat(rng, kc); fill_mat(rng, vc);
        // same hot key 2 via group 0 (q0) and group 1 (q1); same cold row via both
        std::vector<int32_t> entries = {2,0, 4,0, 2,1, 5,1};
        std::vector<int32_t> offsets = {0, 2, 4};
        auto p = base_params(nq, ng, gqa, dk, dv, hot_rows, n_cold, 4);
        p.stream = 1;
        // stream 1 holds a copy of kh/vh (2-stream cache, configured stream selected)
        std::vector<float> kh2 = kh, vh2 = vh;
        std::vector<float> ref = ref_dense(q, dk, gqa, ng, kh, vh, kc, vc, dv, hot_rows,
            entries, 2, offsets, p.scale, 0.0f, {});
        auto f2b = [](const std::vector<float> & v) {
            return std::vector<uint8_t>((const char *)v.data(), (const char *)v.data() + v.size() * 4);
        };
        // interleave two streams: [stream0=kh, stream1=kh2]
        std::vector<uint8_t> khs(f2b(kh)), vhs(f2b(vh));
        const std::vector<uint8_t> kh2b = f2b(kh2);
        const std::vector<uint8_t> vh2b = f2b(vh2);
        khs.insert(khs.end(), kh2b.begin(), kh2b.end());
        vhs.insert(vhs.end(), vh2b.begin(), vh2b.end());
        check_case(cpu_b, vk_b, "shared-key-2wide-2stream", GGML_TYPE_F32, khs,
            GGML_TYPE_F32, vhs, GGML_TYPE_F32, f2b(kc), GGML_TYPE_F32, f2b(vc),
            q, dk, dv, gqa, ng, 1, hot_rows, 2, n_cold, entries, 2, 4, offsets, {}, p, ref, 1e-5f, 1e-5f);
    }

    // ---- 4. all-masked exact zeros (empty range + valid==0) ----
    {
        const uint32_t dk = 16, dv = 16, gqa = 2, ng = 1, nq = 3, hot_rows = 3, n_cold = 2;
        std::mt19937 rng(44);
        std::vector<float> q((size_t)dk * gqa * ng, 0), kh((size_t)hot_rows * dk, 0),
            vh((size_t)hot_rows * dv, 0), kc((size_t)n_cold * dk, 0), vc((size_t)n_cold * dv, 0);
        fill_mat(rng, q); fill_mat(rng, kh); fill_mat(rng, vh); fill_mat(rng, kc); fill_mat(rng, vc);
        std::vector<int32_t> entries = {1,0,0,0, 2,0,0,0, 1,1,0,1};
        std::vector<int32_t> offsets = {0, 0, 2, 3};
        auto p = base_params(nq, ng, gqa, dk, dv, hot_rows, n_cold, 3);
        std::vector<float> ref = ref_dense(q, dk, gqa, ng, kh, vh, kc, vc, dv, hot_rows,
            entries, 4, offsets, p.scale, 0.0f, {});
        // q0 empty -> 0; q1 all valid==0 -> 0; q2 one hot row
        CHECK(ref[0] == 0.0f && ref[1] == 0.0f);
        auto f2b = [](const std::vector<float> & v) {
            return std::vector<uint8_t>((const char *)v.data(), (const char *)v.data() + v.size() * 4);
        };
        check_case(cpu_b, vk_b, "all-masked", GGML_TYPE_F32, f2b(kh),
            GGML_TYPE_F32, f2b(vh), GGML_TYPE_F32, f2b(kc), GGML_TYPE_F32, f2b(vc),
            q, dk, dv, gqa, ng, 1, hot_rows, 1, n_cold, entries, 4, 3, offsets, {}, p, ref, 1e-6f, 1e-6f);
    }

    // ---- 5. zero hot/cold/entry ranges use nonzero Vulkan dummy bindings ----
    {
        const uint32_t dk = 16, dv = 12, gqa = 2, ng = 1, nq = 1;
        std::mt19937 rng(45);
        std::vector<float> q((size_t)dk * gqa), kh((size_t)2 * dk), vh((size_t)2 * dv),
            kc((size_t)2 * dk), vc((size_t)2 * dv);
        fill_mat(rng, q); fill_mat(rng, kh); fill_mat(rng, vh); fill_mat(rng, kc); fill_mat(rng, vc);
        auto f2b = [](const std::vector<float> & v) {
            return std::vector<uint8_t>((const char *)v.data(), (const char *)v.data() + v.size() * sizeof(float));
        };
        const std::vector<uint8_t> empty_bytes;
        const std::vector<float> empty_floats;

        std::vector<int32_t> hot_entries = {GGML_XKV_ATTN_SOURCE_HOT, 0, 0, 1,
                                             GGML_XKV_ATTN_SOURCE_HOT, 1, 0, 1};
        std::vector<int32_t> offsets = {0, 2};
        auto ph = base_params(nq, ng, gqa, dk, dv, 2, 0, 2);
        auto refh = ref_dense(q, dk, gqa, ng, kh, vh, empty_floats, empty_floats, dv,
            2, hot_entries, 4, offsets, ph.scale, 0.0f, {});
        check_case(cpu_b, vk_b, "hot-only", GGML_TYPE_F32, f2b(kh), GGML_TYPE_F32,
            f2b(vh), GGML_TYPE_F32, empty_bytes, GGML_TYPE_F32, empty_bytes, q, dk,
            dv, gqa, ng, 1, 2, 1, 0, hot_entries, 4, 2, offsets, {}, ph, refh, 1e-5f, 1e-5f);

        std::vector<int32_t> cold_entries = {GGML_XKV_ATTN_SOURCE_COLD, 0, 0, 1,
                                              GGML_XKV_ATTN_SOURCE_COLD, 1, 0, 1};
        auto pc = base_params(nq, ng, gqa, dk, dv, 0, 2, 2);
        auto refc = ref_dense(q, dk, gqa, ng, empty_floats, empty_floats, kc, vc, dv,
            0, cold_entries, 4, offsets, pc.scale, 0.0f, {});
        check_case(cpu_b, vk_b, "cold-only", GGML_TYPE_F32, empty_bytes, GGML_TYPE_F32,
            empty_bytes, GGML_TYPE_F32, f2b(kc), GGML_TYPE_F32, f2b(vc), q, dk,
            dv, gqa, ng, 1, 0, 1, 2, cold_entries, 4, 2, offsets, {}, pc, refc, 1e-5f, 1e-5f);

        std::vector<int32_t> no_entries;
        std::vector<int32_t> no_offsets = {0, 0};
        const std::vector<float> sinks = {0.5f, -0.25f};
        auto pe = base_params(nq, ng, gqa, dk, dv, 2, 2, 0);
        auto refe = ref_dense(q, dk, gqa, ng, kh, vh, kc, vc, dv, 2, no_entries, 4,
            no_offsets, pe.scale, 0.0f, sinks);
        check_case(cpu_b, vk_b, "sink-only-empty-entries", GGML_TYPE_F32, f2b(kh),
            GGML_TYPE_F32, f2b(vh), GGML_TYPE_F32, f2b(kc), GGML_TYPE_F32, f2b(vc),
            q, dk, dv, gqa, ng, 1, 2, 1, 2, no_entries, 4, 0, no_offsets, sinks,
            pe, refe, 1e-6f, 1e-6f);
    }

    // ---- 6. Dk!=Dv + Q8_0 hot + F16 cold ----
    {
        const uint32_t dk = 32, dv = 64, gqa = 2, ng = 2, nq = 2, hot_rows = 4, n_cold = 3;
        std::mt19937 rng(55);
        std::vector<float> q((size_t)dk * gqa * ng, 0), kh((size_t)hot_rows * dk, 0),
            vh((size_t)hot_rows * dv, 0), kc((size_t)n_cold * dk, 0), vc((size_t)n_cold * dv, 0);
        fill_mat(rng, q); fill_mat(rng, kh, 0.3f); fill_mat(rng, vh, 0.3f);
        fill_mat(rng, kc, 0.3f); fill_mat(rng, vc, 0.3f);
        auto enc = [&](ggml_type t, const std::vector<float> & v, uint32_t w) {
            size_t rs = ggml_row_size(t, w);
            std::vector<uint8_t> b(rs * (v.size() / w), 0);
            for (size_t r = 0; r < v.size() / w; ++r)
                CHECK(encode_row(t, v.data() + r * w, w, b.data() + r * rs));
            return b;
        };
        std::vector<int32_t> entries = {1,0,0,1, 2,0,1,1, 1,2,0,1, 2,1,0,1, 1,3,1,1, 2,2,1,1};
        std::vector<int32_t> offsets = {0, 3, 6};
        auto p = base_params(nq, ng, gqa, dk, dv, hot_rows, n_cold, 6);
        std::vector<float> ref = ref_dense(q, dk, gqa, ng, kh, vh, kc, vc, dv, hot_rows,
            entries, 4, offsets, p.scale, 0.0f, {});
        check_case(cpu_b, vk_b, "dkdv-q8-f16", GGML_TYPE_Q8_0, enc(GGML_TYPE_Q8_0, kh, dk),
            GGML_TYPE_Q8_0, enc(GGML_TYPE_Q8_0, vh, dv),
            GGML_TYPE_F16, enc(GGML_TYPE_F16, kc, dk),
            GGML_TYPE_F16, enc(GGML_TYPE_F16, vc, dv),
            q, dk, dv, gqa, ng, 1, hot_rows, 1, n_cold, entries, 4, 6, offsets, {}, p, ref, 2e-2f, 2e-2f);
    }

    // ---- 6. capacity padding: garbage beyond offsets ignored ----
    {
        const uint32_t dk = 16, dv = 16, gqa = 1, ng = 1, nq = 2, hot_rows = 3, n_cold = 2;
        std::mt19937 rng(66);
        std::vector<float> q((size_t)dk * gqa * ng, 0), kh((size_t)hot_rows * dk, 0),
            vh((size_t)hot_rows * dv, 0), kc((size_t)n_cold * dk, 0), vc((size_t)n_cold * dv, 0);
        fill_mat(rng, q); fill_mat(rng, kh); fill_mat(rng, vh); fill_mat(rng, kc); fill_mat(rng, vc);
        // valid: 3 entries; capacity 7 with malformed-looking garbage past offsets[2]=3
        std::vector<int32_t> entries = {1,0,0,1, 2,1,0,1, 1,2,0,1, 99,99,99,99, 1,99,0,1, 2,2,0,0, 3,0,5,1};
        std::vector<int32_t> offsets = {0, 2, 3};
        auto p = base_params(nq, ng, gqa, dk, dv, hot_rows, n_cold, 7);
        std::vector<float> ref = ref_dense(q, dk, gqa, ng, kh, vh, kc, vc, dv, hot_rows,
            entries, 4, offsets, p.scale, 0.0f, {});
        auto f2b = [](const std::vector<float> & v) {
            return std::vector<uint8_t>((const char *)v.data(), (const char *)v.data() + v.size() * 4);
        };
        check_case(cpu_b, vk_b, "capacity-pad", GGML_TYPE_F32, f2b(kh),
            GGML_TYPE_F32, f2b(vh), GGML_TYPE_F32, f2b(kc), GGML_TYPE_F32, f2b(vc),
            q, dk, dv, gqa, ng, 1, hot_rows, 1, n_cold, entries, 4, 7, offsets, {}, p, ref, 1e-5f, 1e-5f);
    }

    // ---- 7. future draft isolation: q1 extras never affect q0 ----
    {
        const uint32_t dk = 16, dv = 16, gqa = 1, ng = 1, nq = 2, hot_rows = 5, n_cold = 1;
        std::mt19937 rng(77);
        std::vector<float> q((size_t)dk * gqa * ng, 0), kh((size_t)hot_rows * dk, 0),
            vh((size_t)hot_rows * dv, 0), kc((size_t)n_cold * dk, 0), vc((size_t)n_cold * dv, 0);
        fill_mat(rng, q); fill_mat(rng, kh); fill_mat(rng, vh); fill_mat(rng, kc); fill_mat(rng, vc);
        std::vector<int32_t> entries = {1,0,0,1, 1,1,0,1, 1,3,0,1, 1,4,0,1, 2,0,0,1};
        std::vector<int32_t> offsets = {0, 2, 5};
        auto p = base_params(nq, ng, gqa, dk, dv, hot_rows, n_cold, 5);
        std::vector<float> ref = ref_dense(q, dk, gqa, ng, kh, vh, kc, vc, dv, hot_rows,
            entries, 4, offsets, p.scale, 0.0f, {});
        // solo reference for q0 alone must match batch q0 output exactly
        std::vector<int32_t> solo_entries = {1,0,0,1, 1,1,0,1};
        std::vector<int32_t> solo_offsets = {0, 2};
        std::vector<float> q0only(q.begin(), q.begin() + dk);
        std::vector<float> ref0 = ref_dense(q0only, dk, gqa, ng, kh, vh, kc, vc, dv, hot_rows,
            solo_entries, 4, solo_offsets, p.scale, 0.0f, {});
        for (uint32_t d = 0; d < dv; ++d) CHECK(ref[d] == ref0[d]);
        auto f2b = [](const std::vector<float> & v) {
            return std::vector<uint8_t>((const char *)v.data(), (const char *)v.data() + v.size() * 4);
        };
        check_case(cpu_b, vk_b, "draft-isolation", GGML_TYPE_F32, f2b(kh),
            GGML_TYPE_F32, f2b(vh), GGML_TYPE_F32, f2b(kc), GGML_TYPE_F32, f2b(vc),
            q, dk, dv, gqa, ng, 1, hot_rows, 1, n_cold, entries, 4, 5, offsets, {}, p, ref, 1e-5f, 1e-5f);
    }

    // ---- 8. Turbo4 K / Turbo2 V hot decode into canonical attention ----
    {
        const uint32_t dk = 128, dv = 128, gqa = 2, ng = 1, nq = 2, hot_rows = 4, n_cold = 3;
        std::mt19937 rng(88);
        std::vector<float> q((size_t)dk * gqa * ng, 0), kh((size_t)hot_rows * dk, 0),
            vh((size_t)hot_rows * dv, 0), kc((size_t)n_cold * dk, 0), vc((size_t)n_cold * dv, 0);
        fill_mat(rng, q, 0.2f); fill_mat(rng, kh, 0.2f); fill_mat(rng, vh, 0.2f);
        fill_mat(rng, kc, 0.2f); fill_mat(rng, vc, 0.2f);
        auto enc = [&](ggml_type t, const std::vector<float> & v, uint32_t w) {
            size_t rs = ggml_row_size(t, w);
            std::vector<uint8_t> b(rs * (v.size() / w), 0);
            for (size_t r = 0; r < v.size() / w; ++r)
                CHECK(encode_row(t, v.data() + r * w, w, b.data() + r * rs));
            return b;
        };
        std::vector<int32_t> entries = {1,0,0,1, 2,0,0,1, 1,1,0,1, 1,2,0,1, 2,1,0,1, 2,2,0,1};
        std::vector<int32_t> offsets = {0, 3, 6};
        auto p = base_params(nq, ng, gqa, dk, dv, hot_rows, n_cold, 6);
        const std::vector<uint8_t> kh_t = enc(GGML_TYPE_TURBO4_0, kh, dk);
        const std::vector<uint8_t> vh_t = enc(GGML_TYPE_TURBO2_0, vh, dv);
        std::vector<float> kh_dec(kh.size()), vh_dec(vh.size());
        for (uint32_t r = 0; r < hot_rows; ++r) {
            CHECK(ggml_dequantize_turbo_row(GGML_TYPE_TURBO4_0,
                kh_t.data() + r * ggml_row_size(GGML_TYPE_TURBO4_0, dk),
                kh_dec.data() + r * dk, dk, 128, GGML_TURBO_DECODE_CANONICAL));
            CHECK(ggml_dequantize_turbo_row(GGML_TYPE_TURBO2_0,
                vh_t.data() + r * ggml_row_size(GGML_TYPE_TURBO2_0, dv),
                vh_dec.data() + r * dv, dv, 128, GGML_TURBO_DECODE_CANONICAL));
        }
        const std::vector<float> ref = ref_dense(q, dk, gqa, ng, kh_dec, vh_dec, kc, vc,
            dv, hot_rows, entries, 4, offsets, p.scale, 0.0f, {});
        check_case(cpu_b, vk_b, "turbo-canonical", GGML_TYPE_TURBO4_0, kh_t,
            GGML_TYPE_TURBO2_0, vh_t, GGML_TYPE_F32, enc(GGML_TYPE_F32, kc, dk),
            GGML_TYPE_F32, enc(GGML_TYPE_F32, vc, dv), q, dk, dv, gqa, ng, 1,
            hot_rows, 1, n_cold, entries, 4, 6, offsets, {}, p, ref, 1e-5f, 1e-4f);
    }

    // ---- 9. hot write dependency (device-side in-graph visibility) ----
    {
        const uint32_t dk = 16, dv = 16, gqa = 1, ng = 1, nq = 1, hot_rows = 3, n_cold = 1;
        std::mt19937 rng(99);
        std::vector<float> q((size_t)dk * gqa * ng, 0), kh((size_t)hot_rows * dk, 0),
            vh((size_t)hot_rows * dv, 0), kc((size_t)n_cold * dk, 0), vc((size_t)n_cold * dv, 0);
        fill_mat(rng, q); fill_mat(rng, kh); fill_mat(rng, vh); fill_mat(rng, kc); fill_mat(rng, vc);
        std::vector<int32_t> entries = {1,2,0,1, 2,0,0,1};
        std::vector<int32_t> offsets = {0, 2};
        auto p = base_params(nq, ng, gqa, dk, dv, hot_rows, n_cold, 2);
        auto f2b = [](const std::vector<float> & v) {
            return std::vector<uint8_t>((const char *)v.data(), (const char *)v.data() + v.size() * 4);
        };
        const std::vector<uint8_t> kh_initial = f2b(kh);
        const std::vector<uint8_t> vh_initial = f2b(vh);
        std::vector<float> ref0 = ref_dense(q, dk, gqa, ng, kh, vh, kc, vc, dv, hot_rows,
            entries, 4, offsets, p.scale, 0.0f, {});
        check_case(cpu_b, vk_b, "hot-read", GGML_TYPE_F32, kh_initial,
            GGML_TYPE_F32, vh_initial, GGML_TYPE_F32, f2b(kc), GGML_TYPE_F32, f2b(vc),
            q, dk, dv, gqa, ng, 1, hot_rows, 1, n_cold, entries, 4, 2, offsets, {}, p, ref0, 1e-5f, 1e-5f);
        // Stage a replacement, but keep the cache inputs at their old bytes.
        // set_rows and attention execute in one graph through an explicit edge.
        for (uint32_t d = 0; d < dk; ++d) kh[(size_t)2 * dk + d] = 1.5f;
        for (uint32_t d = 0; d < dv; ++d) vh[(size_t)2 * dv + d] = -0.75f;
        std::vector<float> ref1 = ref_dense(q, dk, gqa, ng, kh, vh, kc, vc, dv, hot_rows,
            entries, 4, offsets, p.scale, 0.0f, {});
        CHECK(max_abs_diff(ref0, ref1) > 1e-3f); // write is observable
        std::vector<uint8_t> write_row;
        const uint8_t * kp = reinterpret_cast<const uint8_t *>(kh.data() + (size_t)2 * dk);
        const uint8_t * vp = reinterpret_cast<const uint8_t *>(vh.data() + (size_t)2 * dv);
        write_row.insert(write_row.end(), kp, kp + dk * sizeof(float));
        write_row.insert(write_row.end(), vp, vp + dv * sizeof(float));
        graph_out c = run_graph(cpu_b, GGML_TYPE_F32, kh_initial, dk, GGML_TYPE_F32,
            vh_initial, dv, 1, hot_rows, 1, GGML_TYPE_F32, f2b(kc), GGML_TYPE_F32,
            f2b(vc), n_cold, q, gqa, ng, entries, 4, 2, offsets, {}, p, true, write_row, 2);
        CHECK(c.ok && c.status == 0);
        if (c.ok) CHECK(max_abs_diff(c.dst, ref1) < 1e-5f);
        if (vk_b) {
            graph_out v = run_graph(vk_b, GGML_TYPE_F32, kh_initial, dk, GGML_TYPE_F32,
                vh_initial, dv, 1, hot_rows, 1, GGML_TYPE_F32, f2b(kc), GGML_TYPE_F32,
                f2b(vc), n_cold, q, gqa, ng, entries, 4, 2, offsets, {}, p, true, write_row, 2);
            CHECK(v.ok && v.status == 0);
            if (v.ok) CHECK(max_abs_diff(v.dst, ref1) < 1e-5f);
        }
    }

    // ---- 10. explicit three-tile DAG + one global softmax/sink ----
    {
        const uint32_t dk = 32, dv = 24, gqa = 2, hot_rows = 3, n_cold = 6;
        std::mt19937 rng(100);
        std::vector<float> q((size_t)dk * gqa), kh((size_t)hot_rows * dk),
            vh((size_t)hot_rows * dv), kc((size_t)n_cold * dk), vc((size_t)n_cold * dv);
        fill_mat(rng, q); fill_mat(rng, kh); fill_mat(rng, vh); fill_mat(rng, kc); fill_mat(rng, vc);
        std::vector<int32_t> entries;
        for (uint32_t row = 0; row < hot_rows; ++row) {
            entries.insert(entries.end(), {GGML_XKV_ATTN_SOURCE_HOT, (int32_t)row, 0, 1});
        }
        for (uint32_t row = 0; row < n_cold; ++row) {
            entries.insert(entries.end(), {GGML_XKV_ATTN_SOURCE_COLD, (int32_t)row, 0, 1});
        }
        const std::vector<int32_t> offsets = {0, (int32_t)(entries.size() / 4)};
        const std::vector<float> sinks = {0.75f, -0.5f};
        const auto p = base_params(1, 1, gqa, dk, dv, hot_rows, n_cold,
            (uint32_t)entries.size() / 4);
        const std::vector<float> ref = ref_dense(q, dk, gqa, 1, kh, vh, kc, vc, dv,
            hot_rows, entries, 4, offsets, p.scale, 0.0f, sinks);
        graph_out c = run_three_tile_graph(cpu_b, q, kh, vh, kc, vc, dk, dv, gqa, sinks);
        CHECK(c.ok && c.status == 0);
        if (c.ok) CHECK(max_abs_diff(c.dst, ref) < 1e-5f);
        if (vk_b) {
            graph_out v = run_three_tile_graph(vk_b, q, kh, vh, kc, vc, dk, dv, gqa, sinks);
            CHECK(v.ok && v.status == 0);
            if (v.ok) CHECK(max_abs_diff(v.dst, ref) < 1e-4f);
        }
    }

    // ---- 11. faults: malformed offsets/source/index/group + dupe-reject ----
    {
        const uint32_t dk = 16, dv = 16, gqa = 1, ng = 1, nq = 2, hot_rows = 3, n_cold = 2;
        std::vector<float> q((size_t)dk * gqa * ng, 1.0f), kh((size_t)hot_rows * dk, 0.1f),
            vh((size_t)hot_rows * dv, 0.1f), kc((size_t)n_cold * dk, 0.1f), vc((size_t)n_cold * dv, 0.1f);
        auto f2b = [](const std::vector<float> & v) {
            return std::vector<uint8_t>((const char *)v.data(), (const char *)v.data() + v.size() * 4);
        };
        struct fault { const char * name; std::vector<int32_t> entries; std::vector<int32_t> offsets; int32_t code; uint32_t flags; };
        std::vector<fault> faults = {
            {"offsets-end-lt-begin", {1,0,0,1}, {0, 1, 0}, 1, 0},
            {"offsets-beyond-cap",   {1,0,0,1}, {0, 1, 5}, 1, 0},
            {"bad-source",           {3,0,0,1, 1,0,0,1}, {0, 1, 2}, 2, 0},
            {"hot-index-oob",        {1,3,0,1, 1,0,0,1}, {0, 1, 2}, 3, 0},
            {"cold-index-oob",       {2,2,0,1, 1,0,0,1}, {0, 1, 2}, 3, 0},
            {"bad-group",            {1,0,1,1, 1,0,0,1}, {0, 1, 2}, 4, 0},
            {"dupe-reject",          {1,0,0,1, 1,0,0,1}, {0, 2, 2}, 5, GGML_XKV_ATTN_FLAG_REJECT_DUPES},
        };
        for (const auto & f : faults) {
            auto p = base_params(nq, ng, gqa, dk, dv, hot_rows, n_cold,
                (uint32_t)f.entries.size() / 4);
            p.flags |= f.flags;
            std::vector<float> out;
            int32_t st = -999;
            bool ok = run_oracle_direct(q, dk, gqa, ng, kh.data(), GGML_TYPE_F32,
                vh.data(), GGML_TYPE_F32, dv, hot_rows, 1, kc.data(), GGML_TYPE_F32,
                vc.data(), GGML_TYPE_F32, n_cold, f.entries, 4, f.offsets, {}, p, out, st);
            CHECK(!ok && st == f.code);
            bool oracle_poisoned = false;
            for (uint32_t row = 0; row < nq * gqa; ++row) {
                oracle_poisoned |= is_nan_row(out, (size_t)row * dv, dv);
            }
            CHECK(oracle_poisoned);
            // CPU graph: same status + full-dst NaN
            graph_out c = run_graph(cpu_b, GGML_TYPE_F32, f2b(kh), dk, GGML_TYPE_F32, f2b(vh), dv,
                1, hot_rows, 1, GGML_TYPE_F32, f2b(kc), GGML_TYPE_F32, f2b(vc), n_cold,
                q, gqa, ng, f.entries, 4, (uint32_t)f.entries.size() / 4, f.offsets, {}, p, false, {}, 0);
            CHECK(c.ok && c.status == f.code);
            if (c.ok) CHECK(is_nan_row(c.dst, 0, c.dst.size()));
            if (vk_b) {
                graph_out v = run_graph(vk_b, GGML_TYPE_F32, f2b(kh), dk, GGML_TYPE_F32, f2b(vh), dv,
                    1, hot_rows, 1, GGML_TYPE_F32, f2b(kc), GGML_TYPE_F32, f2b(vc), n_cold,
                    q, gqa, ng, f.entries, 4, (uint32_t)f.entries.size() / 4, f.offsets, {}, p, false, {}, 0);
                CHECK(v.ok); // shader handles faults as status (exec failure would fail here)
                if (v.ok) {
                    CHECK(v.status == f.code);
                    bool vulkan_poisoned = false;
                    for (uint32_t row = 0; row < nq * gqa; ++row) {
                        vulkan_poisoned |= is_nan_row(v.dst, (size_t)row * dv, dv);
                    }
                    CHECK(vulkan_poisoned);
                }
            }
            std::fprintf(stderr, "fault %s: status=%d OK\n", f.name, f.code);
        }
        // duplicates WITHOUT reject flag accumulate deterministically (2x weight check)
        {
            std::vector<int32_t> entries = {1,0,0,1, 1,0,0,1};
            std::vector<int32_t> offsets = {0, 2, 2};
            auto p = base_params(nq, ng, gqa, dk, dv, hot_rows, n_cold, 2);
            std::vector<float> ref = ref_dense(q, dk, gqa, ng, kh, vh, kc, vc, dv, hot_rows,
                entries, 4, offsets, p.scale, 0.0f, {});
            check_case(cpu_b, vk_b, "dupe-accumulate", GGML_TYPE_F32, f2b(kh),
                GGML_TYPE_F32, f2b(vh), GGML_TYPE_F32, f2b(kc), GGML_TYPE_F32, f2b(vc),
                q, dk, dv, gqa, ng, 1, hot_rows, 1, n_cold, entries, 4, 2, offsets, {}, p, ref, 1e-5f, 1e-5f);
        }
        // supports() rejects: bad version, q/K width mismatch, bad entry stride, Q8 dim
        {
            char err[256];
            ggml_init_params ip = { ggml_tensor_overhead() * 16, nullptr, true };
            ggml_context_ptr cx(ggml_init(ip));
            ggml_tensor * tq = ggml_new_tensor_3d(cx.get(), GGML_TYPE_F32, dk, gqa, ng);
            ggml_tensor * tk = ggml_new_tensor_4d(cx.get(), GGML_TYPE_F32, dk, 1, hot_rows, 1);
            ggml_tensor * tv = ggml_new_tensor_4d(cx.get(), GGML_TYPE_F32, dv, 1, hot_rows, 1);
            ggml_tensor * ck = ggml_new_tensor_2d(cx.get(), GGML_TYPE_F32, dk, n_cold);
            ggml_tensor * cv = ggml_new_tensor_2d(cx.get(), GGML_TYPE_F32, dv, n_cold);
            ggml_tensor * en = ggml_new_tensor_2d(cx.get(), GGML_TYPE_I32, 4, 2);
            ggml_tensor * en3 = ggml_new_tensor_2d(cx.get(), GGML_TYPE_I32, 3, 2);
            ggml_tensor * off = ggml_new_tensor_1d(cx.get(), GGML_TYPE_I32, nq + 1);
            ggml_tensor * st = ggml_new_tensor_1d(cx.get(), GGML_TYPE_I32, 1);
            ggml_tensor * dst = ggml_new_tensor_4d(cx.get(), GGML_TYPE_F32, dv + 2, gqa, nq, 1);
            auto pp = base_params(nq, ng, gqa, dk, dv, hot_rows, n_cold, 2);
            CHECK(ggml_xkv_attention_supports(tq, tk, tv, ck, cv, en, off, nullptr, st, nullptr, dst, &pp, err, sizeof(err)));
            { auto q = pp; q.version = 99; CHECK(!ggml_xkv_attention_supports(tq, tk, tv, ck, cv, en, off, nullptr, st, nullptr, dst, &q, err, sizeof(err))); }
            CHECK(!ggml_xkv_attention_supports(tq, tk, tv, ck, cv, en3, off, nullptr, st, nullptr, dst, &pp, err, sizeof(err)));
            { auto q = pp; q.dim_k = dk + 1; CHECK(!ggml_xkv_attention_supports(tq, tk, tv, ck, cv, en, off, nullptr, st, nullptr, dst, &q, err, sizeof(err))); }
            { auto q = pp; q.kv_head = 5; CHECK(!ggml_xkv_attention_supports(tq, tk, tv, ck, cv, en, off, nullptr, st, nullptr, dst, &q, err, sizeof(err))); }
            size_t nf = 0;
            CHECK(ggml_xkv_attn_tmp_floats(&pp, &nf, err, sizeof(err)) && nf == dk + dv);
            { auto q = pp; q.dim_k = 0; CHECK(!ggml_xkv_attn_tmp_floats(&q, &nf, err, sizeof(err))); }
        }
    }

    if (failures != 0) {
        std::fprintf(stderr, "test-xkv-vulkan-attention: %d FAILURES\n", failures);
        return 1;
    }
    std::puts("test-xkv-vulkan-attention: all tests OK");
    return 0;
}
