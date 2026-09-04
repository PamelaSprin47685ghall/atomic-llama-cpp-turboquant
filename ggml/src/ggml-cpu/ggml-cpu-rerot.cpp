// Correctness-only CPU segmented DDVR attention for RERoT (experimental).
//
// This file adds no GGML_OP and touches no ordinary attention path. The
// production indexed op (GGML_OP_FLASH_ATTN_EXT_REROT) lives in ops.cpp and
// consumes pre-rotated Q groups; the helpers here implement the full DDVR
// math (raw Q/K + per-span Q-phase + IMRoPE + GQA + strong/lag1 frontier
// filtering + ONE global softmax) as an F32 reference for tests and for
// future CPU graph integration. Ordinary FA behavior is unchanged when this
// file is linked in.
//
// Layering: ggml-cpu must not depend on src/llama-rerot.h, so the rope/span
// structs below mirror the core config without including it. Tests that need
// both sides convert explicitly.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ggml_cpu_rerot {

enum class RopeLayout : uint8_t {
    Half = 0,
    Interleaved = 1,
};

enum class FrontierMode : uint8_t {
    Strong = 0,
    Lag1 = 1,
};

struct RopeConfig {
    uint32_t head_dim = 0;
    uint32_t rotary_dim = 0;
    double theta = 10000.0;
    double freq_scale = 1.0;
    RopeLayout layout = RopeLayout::Half;
    uint32_t axis_pair_count[4] = {0, 0, 0, 0};
};

struct Span {
    uint32_t key_begin = 0;
    uint32_t key_count = 0;
    int64_t storage_pos0 = 0;
    int64_t virtual_pos0 = 0;
};

namespace detail {

uint32_t configured_pairs(const RopeConfig & cfg) {
    return cfg.axis_pair_count[0] + cfg.axis_pair_count[1] +
           cfg.axis_pair_count[2] + cfg.axis_pair_count[3];
}

uint32_t pair_axis(const RopeConfig & cfg, uint32_t pair) {
    const uint32_t n_pairs = cfg.rotary_dim / 2;
    const uint32_t configured = configured_pairs(cfg);
    if (configured == 0) {
        return 0;
    }
    if (configured != n_pairs) {
        return 0; // validated by callers; defensive fallback
    }
    uint32_t offset = 0;
    for (uint32_t axis = 0; axis < 4; ++axis) {
        const uint32_t next = offset + cfg.axis_pair_count[axis];
        if (pair < next) {
            return axis;
        }
        offset = next;
    }
    return 0;
}

bool valid_config(const RopeConfig & cfg) {
    if (cfg.head_dim == 0) {
        return false;
    }
    if (cfg.rotary_dim == 0 || cfg.rotary_dim > cfg.head_dim || (cfg.rotary_dim % 2) != 0) {
        return false;
    }
    if (!std::isfinite(cfg.theta) || cfg.theta <= 0.0) {
        return false;
    }
    if (!std::isfinite(cfg.freq_scale) || cfg.freq_scale <= 0.0) {
        return false;
    }
    const uint32_t configured = configured_pairs(cfg);
    if (configured != 0 && configured != cfg.rotary_dim / 2) {
        return false;
    }
    return true;
}

float dot_f64(const float * a, const float * b, uint32_t n) {
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        sum += double(a[i]) * double(b[i]);
    }
    return float(sum);
}

} // namespace detail

// Applies RoPE in place for one head vector. Position is given as four
// explicit axis coordinates; text callers pass (p, p, p, 0) so the delta
// affects only the first three axes. Non-rotary tail dims are preserved.
// Returns false on invalid config (vector left untouched).
bool RopeApply(float * vec, const RopeConfig & cfg, const int64_t pos[4]) {
    if (vec == nullptr || !detail::valid_config(cfg)) {
        return false;
    }
    const uint32_t n_pairs = cfg.rotary_dim / 2;
    for (uint32_t pair = 0; pair < n_pairs; ++pair) {
        const uint32_t axis = detail::pair_axis(cfg, pair);
        const double freq = std::pow(cfg.theta, -2.0 * double(pair) / double(cfg.rotary_dim)) *
                            cfg.freq_scale;
        const double angle = double(pos[axis]) * freq;
        const float c = float(std::cos(angle));
        const float s = float(std::sin(angle));
        uint32_t first;
        uint32_t second;
        if (cfg.layout == RopeLayout::Interleaved) {
            first = pair * 2;
            second = first + 1;
        } else {
            first = pair;
            second = pair + n_pairs;
        }
        const float x = vec[first];
        const float y = vec[second];
        vec[first] = x * c - y * s;
        vec[second] = x * s + y * c;
    }
    return true;
}

bool RopeApplyText(float * vec, const RopeConfig & cfg, int64_t pos) {
    const int64_t p[4] = {pos, pos, pos, 0};
    return RopeApply(vec, cfg, p);
}

// Validates span coverage: every key in [0, n_keys) covered exactly once,
// no empty spans, ranges in bounds. Returns false on violation.
bool SpansCoverAll(const Span * spans, size_t n_spans, uint32_t n_keys) {
    if (spans == nullptr && n_spans != 0) {
        return false;
    }
    if (n_keys == 0) {
        return false;
    }
    std::vector<uint8_t> covered(n_keys, 0);
    for (size_t s = 0; s < n_spans; ++s) {
        const Span & sp = spans[s];
        if (sp.key_count == 0) {
            return false;
        }
        if (sp.key_begin >= n_keys || sp.key_count > n_keys - sp.key_begin) {
            return false;
        }
        for (uint32_t i = 0; i < sp.key_count; ++i) {
            if (++covered[sp.key_begin + i] != 1) {
                return false;
            }
        }
    }
    for (uint8_t c : covered) {
        if (c != 1) {
            return false;
        }
    }
    return true;
}

// Frontier visibility for one key. Keys behind the query frontier are always
// visible; current-frontier peers are visible only in strong mode (lag1 hides
// them until the next frontier); future keys are never visible. The extra
// mask (nullptr = all pass) models private-gap filtering on top.
bool KeyVisible(uint64_t key_frontier,
                uint64_t query_frontier,
                FrontierMode mode,
                bool extra_mask_pass) {
    if (!extra_mask_pass) {
        return false;
    }
    if (key_frontier < query_frontier) {
        return true;
    }
    if (key_frontier == query_frontier) {
        return mode == FrontierMode::Strong;
    }
    return false;
}

// Single-head materialized baseline: K rephased storage -> virtual, Q at
// query_virtual, ONE global softmax over visible keys. Layouts:
//   raw_q[head_dim], raw_k[key * head_dim + d], values[key * value_dim + d].
//   key_frontiers[key], extra_mask[key] (nullptr = all pass).
// Returns false on invalid input; out[value_dim] zeroed when nothing visible.
bool DdvrMaterializedSingle(const float * raw_q,
                            const float * raw_k,
                            const float * values,
                            uint32_t n_keys,
                            uint32_t value_dim,
                            int64_t query_virtual_pos,
                            const Span * spans,
                            size_t n_spans,
                            const RopeConfig & cfg,
                            float scale,
                            const uint64_t * key_frontiers,
                            uint64_t query_frontier,
                            FrontierMode mode,
                            const uint8_t * extra_mask,
                            float * out) {
    if (!raw_q || !raw_k || !values || !spans || !out) {
        return false;
    }
    if (!detail::valid_config(cfg) || n_keys == 0 || value_dim == 0) {
        return false;
    }
    if (!SpansCoverAll(spans, n_spans, n_keys)) {
        return false;
    }
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(float(cfg.head_dim));
    }

    std::vector<float> q(raw_q, raw_q + cfg.head_dim);
    if (!RopeApplyText(q.data(), cfg, query_virtual_pos)) {
        return false;
    }

    std::vector<float> scores;
    scores.reserve(n_keys);
    std::vector<uint32_t> order;
    order.reserve(n_keys);
    std::vector<float> k(cfg.head_dim);
    for (size_t s = 0; s < n_spans; ++s) {
        const Span & sp = spans[s];
        for (uint32_t local = 0; local < sp.key_count; ++local) {
            const uint32_t key = sp.key_begin + local;
            const bool pass = extra_mask ? extra_mask[key] != 0 : true;
            const uint64_t kf = key_frontiers ? key_frontiers[key] : 0;
            const uint64_t qf = key_frontiers ? query_frontier : 0;
            // When no frontier array is given, every key is visible.
            const bool visible = key_frontiers ? KeyVisible(kf, qf, mode, pass) : pass;
            if (!visible) {
                continue;
            }
            std::memcpy(k.data(), raw_k + size_t(key) * cfg.head_dim, size_t(cfg.head_dim) * sizeof(float));
            if (!RopeApplyText(k.data(), cfg, sp.virtual_pos0 + int64_t(local))) {
                return false;
            }
            scores.push_back(scale * detail::dot_f64(q.data(), k.data(), cfg.head_dim));
            order.push_back(key);
        }
    }

    std::fill(out, out + value_dim, 0.0f);
    if (scores.empty()) {
        return true;
    }
    const float m = *std::max_element(scores.begin(), scores.end());
    std::vector<double> w(scores.size());
    double denom = 0.0;
    for (size_t i = 0; i < scores.size(); ++i) {
        w[i] = std::exp(double(scores[i] - m));
        denom += w[i];
    }
    for (size_t i = 0; i < scores.size(); ++i) {
        const double weight = w[i] / denom;
        const uint32_t key = order[i];
        for (uint32_t d = 0; d < value_dim; ++d) {
            out[d] += float(weight * double(values[size_t(key) * value_dim + d]));
        }
    }
    return true;
}

// Single-head Q-side DDVR: K stays at storage phase, one Q per span at
// query_virtual + storage_pos0 - virtual_pos0, ONE global softmax.
bool DdvrQsideSingle(const float * raw_q,
                     const float * raw_k,
                     const float * values,
                     uint32_t n_keys,
                     uint32_t value_dim,
                     int64_t query_virtual_pos,
                     const Span * spans,
                     size_t n_spans,
                     const RopeConfig & cfg,
                     float scale,
                     const uint64_t * key_frontiers,
                     uint64_t query_frontier,
                     FrontierMode mode,
                     const uint8_t * extra_mask,
                     float * out) {
    if (!raw_q || !raw_k || !values || !spans || !out) {
        return false;
    }
    if (!detail::valid_config(cfg) || n_keys == 0 || value_dim == 0) {
        return false;
    }
    if (!SpansCoverAll(spans, n_spans, n_keys)) {
        return false;
    }
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(float(cfg.head_dim));
    }

    std::vector<float> scores;
    scores.reserve(n_keys);
    std::vector<uint32_t> order;
    order.reserve(n_keys);
    std::vector<float> q(cfg.head_dim);
    std::vector<float> k(cfg.head_dim);
    for (size_t s = 0; s < n_spans; ++s) {
        const Span & sp = spans[s];
        std::memcpy(q.data(), raw_q, size_t(cfg.head_dim) * sizeof(float));
        const int64_t q_frame = query_virtual_pos + sp.storage_pos0 - sp.virtual_pos0;
        if (!RopeApplyText(q.data(), cfg, q_frame)) {
            return false;
        }
        for (uint32_t local = 0; local < sp.key_count; ++local) {
            const uint32_t key = sp.key_begin + local;
            const bool pass = extra_mask ? extra_mask[key] != 0 : true;
            const bool visible = key_frontiers ? KeyVisible(key_frontiers[key], query_frontier, mode, pass) : pass;
            if (!visible) {
                continue;
            }
            std::memcpy(k.data(), raw_k + size_t(key) * cfg.head_dim, size_t(cfg.head_dim) * sizeof(float));
            if (!RopeApplyText(k.data(), cfg, sp.storage_pos0 + int64_t(local))) {
                return false;
            }
            scores.push_back(scale * detail::dot_f64(q.data(), k.data(), cfg.head_dim));
            order.push_back(key);
        }
    }

    std::fill(out, out + value_dim, 0.0f);
    if (scores.empty()) {
        return true;
    }
    const float m = *std::max_element(scores.begin(), scores.end());
    std::vector<double> w(scores.size());
    double denom = 0.0;
    for (size_t i = 0; i < scores.size(); ++i) {
        w[i] = std::exp(double(scores[i] - m));
        denom += w[i];
    }
    for (size_t i = 0; i < scores.size(); ++i) {
        const double weight = w[i] / denom;
        const uint32_t key = order[i];
        for (uint32_t d = 0; d < value_dim; ++d) {
            out[d] += float(weight * double(values[size_t(key) * value_dim + d]));
        }
    }
    return true;
}

// GQA wrappers. Layouts (row-major, matching the indexed-op test):
//   raw_q[(hq * head_dim)], raw_k[(hkv * n_keys * head_dim)], values[(hkv * n_keys * value_dim)]
// Head mapping: kv = hq_idx / (hq / hkv) (hq must be a multiple of hkv).
// Each query head gets its own global softmax over the same visible set.
bool DdvrMaterializedGqa(const float * raw_q,
                         const float * raw_k,
                         const float * values,
                         uint32_t n_keys,
                         uint32_t value_dim,
                         uint32_t n_head_q,
                         uint32_t n_head_kv,
                         int64_t query_virtual_pos,
                         const Span * spans,
                         size_t n_spans,
                         const RopeConfig & cfg,
                         float scale,
                         const uint64_t * key_frontiers,
                         uint64_t query_frontier,
                         FrontierMode mode,
                         const uint8_t * extra_mask,
                         float * out) {
    if (!raw_q || !raw_k || !values || !spans || !out) {
        return false;
    }
    if (n_head_q == 0 || n_head_kv == 0 || n_head_q % n_head_kv != 0) {
        return false;
    }
    const uint32_t group = n_head_q / n_head_kv;
    // Slice K/V per KV head into contiguous single-head buffers to reuse the
    // single-head path without changing its layout contract.
    std::vector<float> k_slice(size_t(n_keys) * cfg.head_dim);
    std::vector<float> v_slice(size_t(n_keys) * value_dim);
    for (uint32_t hq = 0; hq < n_head_q; ++hq) {
        const uint32_t hkv = hq / group;
        for (uint32_t key = 0; key < n_keys; ++key) {
            std::memcpy(k_slice.data() + size_t(key) * cfg.head_dim,
                        raw_k + (size_t(hkv) * n_keys + key) * cfg.head_dim,
                        size_t(cfg.head_dim) * sizeof(float));
            std::memcpy(v_slice.data() + size_t(key) * value_dim,
                        values + (size_t(hkv) * n_keys + key) * value_dim,
                        size_t(value_dim) * sizeof(float));
        }
        if (!DdvrMaterializedSingle(raw_q + size_t(hq) * cfg.head_dim,
                                    k_slice.data(), v_slice.data(), n_keys, value_dim,
                                    query_virtual_pos, spans, n_spans, cfg, scale,
                                    key_frontiers, query_frontier, mode, extra_mask,
                                    out + size_t(hq) * value_dim)) {
            return false;
        }
    }
    return true;
}

bool DdvrQsideGqa(const float * raw_q,
                  const float * raw_k,
                  const float * values,
                  uint32_t n_keys,
                  uint32_t value_dim,
                  uint32_t n_head_q,
                  uint32_t n_head_kv,
                  int64_t query_virtual_pos,
                  const Span * spans,
                  size_t n_spans,
                  const RopeConfig & cfg,
                  float scale,
                  const uint64_t * key_frontiers,
                  uint64_t query_frontier,
                  FrontierMode mode,
                  const uint8_t * extra_mask,
                  float * out) {
    if (!raw_q || !raw_k || !values || !spans || !out) {
        return false;
    }
    if (n_head_q == 0 || n_head_kv == 0 || n_head_q % n_head_kv != 0) {
        return false;
    }
    const uint32_t group = n_head_q / n_head_kv;
    std::vector<float> k_slice(size_t(n_keys) * cfg.head_dim);
    std::vector<float> v_slice(size_t(n_keys) * value_dim);
    for (uint32_t hq = 0; hq < n_head_q; ++hq) {
        const uint32_t hkv = hq / group;
        for (uint32_t key = 0; key < n_keys; ++key) {
            std::memcpy(k_slice.data() + size_t(key) * cfg.head_dim,
                        raw_k + (size_t(hkv) * n_keys + key) * cfg.head_dim,
                        size_t(cfg.head_dim) * sizeof(float));
            std::memcpy(v_slice.data() + size_t(key) * value_dim,
                        values + (size_t(hkv) * n_keys + key) * value_dim,
                        size_t(value_dim) * sizeof(float));
        }
        if (!DdvrQsideSingle(raw_q + size_t(hq) * cfg.head_dim,
                             k_slice.data(), v_slice.data(), n_keys, value_dim,
                             query_virtual_pos, spans, n_spans, cfg, scale,
                             key_frontiers, query_frontier, mode, extra_mask,
                             out + size_t(hq) * value_dim)) {
            return false;
        }
    }
    return true;
}

} // namespace ggml_cpu_rerot
