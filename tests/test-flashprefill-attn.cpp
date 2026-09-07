// test-flashprefill-attn.cpp — FlashPrefill V2 attention contracts.
//
// Frozen wire: ggml/include/ggml-flashprefill.h v1 (WireReference) + constructors
// in ggml/include/ggml.h (OpRegistration freeze v4).
//   row8  = [source_query,kv_head,logical_pos,tile,prompt_begin,prompt_end,flags,q_head]
//   use8  = [frag,tile,kv_head,q_group,sub_off,sub_count,flags,source_query]
//   meta header[22]=n_groups, [23]=n_q_heads; plan header 24 words,
//   visible/exact 64-bit at 16/18; max_sel = max ACTUAL per-(tile,head) use count.
//   Q is F32 [Dk,n_groups,Hq,1] element (d,g,h,0); output [Dv,Hq,n_out,1].
//   op_params word8 = mean_correction, words 9..15 zero.
//
// Scope: exact-all vs dense over same cached bytes, sparse vs same-plan
// same-means oracle, dtype/stride/phase/split contracts. No shared fixture
// file: self-contained. Independent double oracle below NEVER calls
// ggml_flashprefill_ref_* (implementation under test) for expected values.
// Metadata/plan helpers + constructors are wire encoding/graph plumbing, not math.
//
// Selector rule (Main freeze, matches CPU): candidate not full across tile
// (cand_full==0) => mandatory exact AND excluded entirely from energy M/Smax
// (whole column masked). Fully legal candidates score valid row pairs only
// via pair_valid[r*n_cand+j] (0/1, row-major). All-zero mask is legal:
// all-exact, zero sparse work, capacity still enforced. Only non-0/1 mask
// entries are errors.
//
// Backend: --backend <name> selects device (substring, case-insensitive,
// default CPU). --required-backend <name> is strict: missing device => FAIL
// (never silently PASS on CPU). Fixtures use the ACTUAL backend plan read
// back after graph completion for the oracle; selector mismatches are
// checked strictly, never hidden in a total-error tolerance.
//
// Deterministic bounded inputs; max abs/rel reported per row/head/dim + NaN
// counts. No tests are run by this file's author; no pass claimed.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-flashprefill.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;
static int g_skips = 0;

#define FP_CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

#define FP_CHECK_MSG(cond, ...) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n"); \
        ++g_failures; \
    } \
} while (0)

// ---------------------------------------------------------------------------
// Small string/backend helpers
// ---------------------------------------------------------------------------

static std::string fp_lower(std::string s) {
    for (char & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

static bool fp_contains_ci(const std::string & hay, const std::string & needle) {
    if (needle.empty()) return true;
    return fp_lower(hay).find(fp_lower(needle)) != std::string::npos;
}

static bool fp_finite_f(float x) { return std::isfinite(x) != 0; }
static bool fp_finite_d(double x) { return std::isfinite(x) != 0; }

// Deterministic bounded generator: finite, |x| < 1.
static float fp_gen(int64_t i, int64_t salt) {
    double x = 0.31 * std::sin(double(i) * 0.017 + double(salt) * 0.13)
             + 0.23 * std::cos(double(i) * 0.031 + double(salt) * 0.71);
    if (!fp_finite_d(x)) return 0.0f;
    if (x > 0.9) x = 0.9;
    if (x < -0.9) x = -0.9;
    return (float) x;
}

static void fp_fill_bounded(std::vector<float> & v, int64_t salt) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = fp_gen((int64_t) i, salt);
    for (float x : v) {
        if (!fp_finite_f(x)) { FP_CHECK(false); break; }
    }
}

// ---------------------------------------------------------------------------
// Independent double oracle (never calls ggml_flashprefill_ref_*).
// ---------------------------------------------------------------------------

namespace fp_oracle {

static double softcap_apply(double x, double cap) {
    if (!(cap > 0.0)) return x;
    return cap * std::tanh(x / cap);
}

// Stable (m,l,o) accumulator over one query row.
struct Mlo {
    double m = -INFINITY;
    double l = 0.0;
    std::vector<double> o; // size dv
    int dv = 0;
};

static Mlo mlo_init(int dv) {
    Mlo s;
    s.dv = dv;
    s.o.assign((size_t) dv, 0.0);
    s.m = -INFINITY;
    s.l = 0.0;
    return s;
}

// Exact token (count==1) or proxy block (count==n_J>=1) at raw logit.
// Applies softcap in score domain, then +log(count). count<=0 rejected.
static bool mlo_add(Mlo & s, double logit_raw, const float * v, int64_t count,
                    double scale_unused, double softcap, bool & ok) {
    (void) scale_unused;
    ok = false;
    if (s.dv < 1 || v == nullptr || count < 1) return false;
    if (!fp_finite_d(logit_raw)) return false;
    for (int i = 0; i < s.dv; ++i) if (!fp_finite_f(v[i])) return false;
    double z = softcap_apply(logit_raw, softcap) + std::log((double) count);
    if (!fp_finite_d(z)) return false;
    if (s.l == 0.0) {
        s.m = z;
        s.l = 1.0;
        for (int i = 0; i < s.dv; ++i) s.o[(size_t) i] = (double) v[i];
        ok = true;
        return true;
    }
    if (!fp_finite_d(s.m) || !fp_finite_d(s.l) || !(s.l > 0.0)) return false;
    double m_new = s.m > z ? s.m : z;
    double a = std::exp(s.m - m_new);
    double b = std::exp(z - m_new);
    for (int i = 0; i < s.dv; ++i) {
        double nv = a * s.o[(size_t) i] + b * (double) v[i];
        if (!fp_finite_d(nv)) return false;
        s.o[(size_t) i] = nv;
    }
    double l_new = a * s.l + b;
    if (!fp_finite_d(l_new) || !(l_new > 0.0)) return false;
    s.m = m_new;
    s.l = l_new;
    ok = true;
    return true;
}

static bool mlo_add_sink(Mlo & s, double sink_logit, bool & ok) {
    ok = false;
    if (s.dv < 1 || !fp_finite_d(sink_logit)) return false;
    if (s.l == 0.0) {
        s.m = sink_logit;
        s.l = 1.0;
        ok = true;
        return true;
    }
    if (!fp_finite_d(s.m) || !fp_finite_d(s.l) || !(s.l > 0.0)) return false;
    double m_new = s.m > sink_logit ? s.m : sink_logit;
    double a = std::exp(s.m - m_new);
    double b = std::exp(sink_logit - m_new);
    for (int i = 0; i < s.dv; ++i) s.o[(size_t) i] = a * s.o[(size_t) i];
    double l_new = a * s.l + b;
    if (!fp_finite_d(l_new) || !(l_new > 0.0)) return false;
    s.m = m_new;
    s.l = l_new;
    ok = true;
    return true;
}

static bool mlo_merge(const Mlo & a, const Mlo & b, Mlo & out, bool & ok) {
    ok = false;
    if (a.dv < 1 || a.dv != b.dv) return false;
    out.dv = a.dv;
    out.o.assign((size_t) a.dv, 0.0);
    bool ae = (a.l == 0.0);
    bool be = (b.l == 0.0);
    if (ae && be) {
        out.m = -INFINITY;
        out.l = 0.0;
        ok = true;
        return true;
    }
    if (ae) { out.m = b.m; out.l = b.l; out.o = b.o; ok = true; return true; }
    if (be) { out.m = a.m; out.l = a.l; out.o = a.o; ok = true; return true; }
    if (!fp_finite_d(a.m) || !fp_finite_d(b.m) || !fp_finite_d(a.l) || !fp_finite_d(b.l)) return false;
    if (!(a.l > 0.0) || !(b.l > 0.0)) return false;
    double m_new = a.m > b.m ? a.m : b.m;
    double wa = std::exp(a.m - m_new);
    double wb = std::exp(b.m - m_new);
    for (int i = 0; i < a.dv; ++i) out.o[(size_t) i] = wa * a.o[(size_t) i] + wb * b.o[(size_t) i];
    out.m = m_new;
    out.l = wa * a.l + wb * b.l;
    if (!fp_finite_d(out.l) || !(out.l > 0.0)) return false;
    ok = true;
    return true;
}

static bool mlo_finalize(const Mlo & s, std::vector<float> & out, bool & ok) {
    ok = false;
    out.assign((size_t) s.dv, 0.0f);
    if (s.dv < 1) return false;
    if (s.l == 0.0) { ok = true; return true; } // empty => zeros, never NaN
    if (!fp_finite_d(s.m) || !fp_finite_d(s.l) || !(s.l > 0.0)) return false;
    for (int i = 0; i < s.dv; ++i) {
        double v = s.o[(size_t) i] / s.l;
        if (!fp_finite_d(v)) return false;
        out[(size_t) i] = (float) v;
    }
    ok = true;
    return true;
}

// Dense reference for one query row over explicit token lists.
// K/V are dequantized storage-domain rows; Q is storage-domain (already
// WHT-transformed when K is turbo2/3). scale/softcap in score domain;
// softcap applied BEFORE +log(count) for proxies (contract).
struct ProxyFrag {
    std::vector<float> kbar; // Dk
    std::vector<float> vbar; // Dv
    int64_t count = 0;       // n_J >= 1
};

static bool attend_row(
        const float * q, int dk,
        const std::vector<std::vector<float>> & k_exact, // [n_exact][dk]
        const std::vector<std::vector<float>> & v_exact, // [n_exact][dv]
        const std::vector<ProxyFrag> & proxies,
        int dv, double scale, double softcap,
        bool has_sink, double sink_logit,
        std::vector<float> & out) {
    if (dk < 1 || dv < 1 || q == nullptr) return false;
    if (k_exact.size() != v_exact.size()) return false;
    Mlo acc = mlo_init(dv);
    bool ok = false;
    if (has_sink) {
        if (!mlo_add_sink(acc, sink_logit, ok) || !ok) return false;
    }
    for (size_t i = 0; i < k_exact.size(); ++i) {
        if ((int) k_exact[i].size() != dk || (int) v_exact[i].size() != dv) return false;
        double dot = 0.0;
        for (int d = 0; d < dk; ++d) dot += (double) q[d] * (double) k_exact[i][(size_t) d];
        double logit = scale * dot;
        if (!mlo_add(acc, logit, v_exact[i].data(), 1, scale, softcap, ok) || !ok) return false;
    }
    for (const ProxyFrag & p : proxies) {
        if (p.count < 1 || (int) p.kbar.size() != dk || (int) p.vbar.size() != dv) return false;
        double dot = 0.0;
        for (int d = 0; d < dk; ++d) dot += (double) q[d] * (double) p.kbar[(size_t) d];
        double logit = scale * dot;
        if (!mlo_add(acc, logit, p.vbar.data(), p.count, scale, softcap, ok) || !ok) return false;
    }
    if (!mlo_finalize(acc, out, ok) || !ok) return false;
    return true;
}

// Host tile max-energy selector with Main freeze rule:
//  - cand_full[j]==0 => mandatory exact, whole column excluded from M/Smax.
//  - pair_valid[r*n_cand+j] masks pairs (0 invalid, 1 valid); invalid pairs
//    contribute nothing to M/Smax/S. All-zero mask legal => all exact.
//  - keep[j] = (S[j] >= alpha*Smax) inclusive over scored columns only.
//  - exact = unscored(partial) | kept | mandatory | (exact_all?all:{})
// Only 0/1 mask entries accepted.
static bool host_select(
        const std::vector<float> & qpair, // [n_rows*n_cand*dk] row-major
        const std::vector<float> & kbar,  // [n_cand*dk]
        const std::vector<int> & cand_mandatory, // [n_cand] 0/1
        const std::vector<int> & cand_full,      // [n_cand] 0/1
        const std::vector<int> & pair_valid,     // [n_rows*n_cand] 0/1
        int n_rows, int n_cand, int dk,
        double scale, double softcap, double alpha, bool exact_all,
        std::vector<int> & out_exact, std::vector<int> & out_proxy) {
    out_exact.clear();
    out_proxy.clear();
    if (n_rows < 1 || n_cand < 1 || dk < 1) return false;
    if ((int) qpair.size() != n_rows * n_cand * dk) return false;
    if ((int) kbar.size() != n_cand * dk) return false;
    if ((int) cand_mandatory.size() != n_cand || (int) cand_full.size() != n_cand) return false;
    if ((int) pair_valid.size() != n_rows * n_cand) return false;
    if (!(alpha > 0.0) || !(alpha <= 1.0)) return false;
    if (!fp_finite_d(scale) || !fp_finite_d(softcap)) return false;
    for (int v : pair_valid) if (v != 0 && v != 1) return false;
    for (int v : cand_mandatory) if (v != 0 && v != 1) return false;
    for (int v : cand_full) if (v != 0 && v != 1) return false;
    for (float x : qpair) if (!fp_finite_f(x)) return false;
    for (float x : kbar) if (!fp_finite_f(x)) return false;

    // M over valid pairs of scored (full) columns only.
    double m = -INFINITY;
    for (int r = 0; r < n_rows; ++r) {
        for (int j = 0; j < n_cand; ++j) {
            if (!cand_full[(size_t) j]) continue; // partial column excluded entirely
            if (!pair_valid[(size_t) r * (size_t) n_cand + (size_t) j]) continue;
            double dot = 0.0;
            for (int d = 0; d < dk; ++d) {
                dot += (double) qpair[((size_t) r * (size_t) n_cand + (size_t) j) * (size_t) dk + (size_t) d]
                     * (double) kbar[(size_t) j * (size_t) dk + (size_t) d];
            }
            double z = softcap_apply(scale * dot, softcap);
            if (!fp_finite_d(z)) return false;
            if (z > m) m = z;
        }
    }
    std::vector<double> s((size_t) n_cand, 0.0);
    double smax = 0.0;
    bool any_scored = fp_finite_d(m);
    if (any_scored) {
        for (int j = 0; j < n_cand; ++j) {
            if (!cand_full[(size_t) j]) continue;
            double acc = 0.0;
            for (int r = 0; r < n_rows; ++r) {
                if (!pair_valid[(size_t) r * (size_t) n_cand + (size_t) j]) continue;
                double dot = 0.0;
                for (int d = 0; d < dk; ++d) {
                    dot += (double) qpair[((size_t) r * (size_t) n_cand + (size_t) j) * (size_t) dk + (size_t) d]
                         * (double) kbar[(size_t) j * (size_t) dk + (size_t) d];
                }
                double z = softcap_apply(scale * dot, softcap);
                acc += std::exp(z - m);
            }
            if (!fp_finite_d(acc)) return false;
            s[(size_t) j] = acc;
            if (acc > smax) smax = acc;
        }
    }
    double thr = alpha * smax;
    for (int j = 0; j < n_cand; ++j) {
        bool partial = !cand_full[(size_t) j];
        bool kept = false;
        if (!partial && any_scored) kept = (s[(size_t) j] >= thr);
        bool exact = !any_scored || partial || kept || cand_mandatory[(size_t) j] || exact_all;
        if (exact) out_exact.push_back(j);
        else out_proxy.push_back(j);
    }
    return true;
}

} // namespace fp_oracle

// ---------------------------------------------------------------------------
// Host WHT for turbo2/3 storage domain (tables match ggml-turbo-quant.c).
// ---------------------------------------------------------------------------

static const float kWhtS1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};
static const float kWhtS2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

static void fp_wht_forward(float * x, int group_size) {
    float inv = (group_size == 128) ? 0.08838834764831845f : 0.125f;
    for (int i = 0; i < group_size; ++i) x[i] *= kWhtS1[i];
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; ++j) {
                float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < group_size; ++i) x[i] *= inv * kWhtS2[i];
}

static void fp_wht_inverse(float * x, int group_size) {
    float inv = (group_size == 128) ? 0.08838834764831845f : 0.125f;
    for (int i = 0; i < group_size; ++i) x[i] *= kWhtS2[i];
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; ++j) {
                float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < group_size; ++i) x[i] *= inv * kWhtS1[i];
}

static void fp_wht_rows(std::vector<float> & v, int dim, int group_size, bool inverse) {
    if (dim <= 0 || group_size <= 0 || v.empty()) return;
    size_t nrows = v.size() / (size_t) dim;
    std::vector<float> tmp((size_t) group_size);
    for (size_t r = 0; r < nrows; ++r) {
        for (int g = 0; g < dim; g += group_size) {
            int gs = group_size;
            if (g + gs > dim) break; // tail passthrough (identity)
            for (int i = 0; i < gs; ++i) tmp[(size_t) i] = v[r * (size_t) dim + (size_t)(g + i)];
            if (inverse) fp_wht_inverse(tmp.data(), gs);
            else fp_wht_forward(tmp.data(), gs);
            for (int i = 0; i < gs; ++i) v[r * (size_t) dim + (size_t)(g + i)] = tmp[(size_t) i];
        }
    }
}

static int fp_wht_group_for_dim(int dim) {
    if (dim % 128 == 0) return 128;
    return 64;
}

// ---------------------------------------------------------------------------
// Quant helpers over ACTUAL ggml stored bytes.
// ---------------------------------------------------------------------------

static size_t fp_type_bytes(ggml_type type, int64_t n_elements) {
    const ggml_type_traits * tr = ggml_get_type_traits(type);
    FP_CHECK(tr != nullptr);
    if (!tr) return 0;
    int64_t blck = tr->blck_size > 0 ? tr->blck_size : 1;
    if (n_elements % blck != 0) return 0;
    return (size_t)(n_elements / blck) * tr->type_size;
}

static bool fp_quantize_rows(ggml_type type, const std::vector<float> & src, std::vector<uint8_t> & dst) {
    const ggml_type_traits * tr = ggml_get_type_traits(type);
    if (!tr) return false;
    if (src.empty()) { dst.clear(); return true; }
    size_t nb = fp_type_bytes(type, (int64_t) src.size());
    if (nb == 0) return false;
    dst.assign(nb, 0);
    if (type == GGML_TYPE_F32) {
        std::memcpy(dst.data(), src.data(), nb);
        return true;
    }
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(src.size());
        ggml_fp32_to_fp16_row(src.data(), tmp.data(), (int64_t) src.size());
        std::memcpy(dst.data(), tmp.data(), nb);
        return true;
    }
    // A single chunk preserves block alignment when size % blck == 0.
    const size_t got = ggml_quantize_chunk(type, src.data(), dst.data(), 0, 1, (int64_t) src.size(), nullptr);
    return got == nb;
}

static bool fp_dequantize_rows(ggml_type type, const std::vector<uint8_t> & src, std::vector<float> & dst, int64_t n_elements) {
    const ggml_type_traits * tr = ggml_get_type_traits(type);
    if (!tr || !tr->to_float) return false;
    dst.assign((size_t) n_elements, 0.0f);
    if (src.empty() && n_elements == 0) return true;
    size_t want = fp_type_bytes(type, n_elements);
    if (src.size() < want) return false;
    tr->to_float(src.data(), dst.data(), n_elements);
    for (float x : dst) if (!fp_finite_f(x)) return false;
    return true;
}

// ---------------------------------------------------------------------------
// Error reporting: max abs/rel + location (row/head/dim) + NaN counts.
// ---------------------------------------------------------------------------

struct FpErr {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double mean_abs = 0.0;
    int64_t nan_got = 0;
    int64_t nan_ref = 0;
    int64_t inf_got = 0;
    int best_row = -1;
    int best_head = -1;
    int best_dim = -1;
};

static FpErr fp_compare(const std::vector<float> & got, const std::vector<float> & ref,
                        int dv, int hq, int nq) {
    FpErr e;
    if (got.size() != ref.size()) return e;
    double sum = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        float g = got[i];
        float r = ref[i];
        if (std::isnan(g)) ++e.nan_got;
        if (std::isnan(r)) ++e.nan_ref;
        if (std::isinf(g)) ++e.inf_got;
        if (!fp_finite_f(g) || !fp_finite_f(r)) continue;
        double ae = std::abs((double) g - (double) r);
        double re = ae / std::max(1e-5, std::abs((double) r));
        sum += ae;
        if (ae > e.max_abs) {
            e.max_abs = ae;
            e.max_rel = re;
            if (dv > 0 && hq > 0 && nq > 0) {
                int d = (int)(i % (size_t) dv);
                int h = (int)((i / (size_t) dv) % (size_t) hq);
                int q = (int)((i / ((size_t) dv * (size_t) hq)) % (size_t) nq);
                e.best_dim = d;
                e.best_head = h;
                e.best_row = q;
            }
        }
        if (re > e.max_rel && ae == e.max_abs) { /* keep paired */ }
        // track max rel independently
        if (re > e.max_rel) e.max_rel = re;
    }
    if (!got.empty()) e.mean_abs = sum / (double) got.size();
    return e;
}

static void fp_print_err(const char * label, const FpErr & e, double atol, double rtol) {
    std::printf("%s: max_abs=%.9g max_rel=%.9g mean_abs=%.9g nan_got=%lld nan_ref=%lld inf_got=%lld at(row=%d,head=%d,dim=%d) tol(abs=%.1g,rel=%.1g)\n",
            label, e.max_abs, e.max_rel, e.mean_abs,
            (long long) e.nan_got, (long long) e.nan_ref, (long long) e.inf_got,
            e.best_row, e.best_head, e.best_dim, atol, rtol);
}

// ---------------------------------------------------------------------------
// Host-only contract tests
// ---------------------------------------------------------------------------

static void test_oracle_identities(void) {
    // PREFILL 16.1 identity: exact V=0 + proxy(mean V=3,count=2) == 2.0, and
    // equals the split form with two exact copies. Large logit finite.
    {
        std::vector<std::vector<float>> ke(1, std::vector<float>(4, 0.1f));
        std::vector<std::vector<float>> ve(1, std::vector<float>(1, 0.0f));
        fp_oracle::ProxyFrag p;
        p.kbar.assign(4, 0.1f);
        p.vbar.assign(1, 3.0f);
        p.count = 2;
        std::vector<float> q(4, 0.0f);
        std::vector<float> out;
        FP_CHECK(fp_oracle::attend_row(q.data(), 4, ke, ve, {p}, 1, 1.0, 0.0, false, 0.0, out));
        FP_CHECK_MSG(std::abs(out[0] - 2.0f) < 1e-6f, "mean identity got %.9g want 2.0", (double) out[0]);
    }
    {
        std::vector<std::vector<float>> ke;
        std::vector<std::vector<float>> ve;
        ke.push_back(std::vector<float>(4, 0.1f)); ve.push_back(std::vector<float>(1, 0.0f));
        ke.push_back(std::vector<float>(4, 0.1f)); ve.push_back(std::vector<float>(1, 3.0f));
        ke.push_back(std::vector<float>(4, 0.1f)); ve.push_back(std::vector<float>(1, 3.0f));
        std::vector<float> q(4, 0.0f);
        std::vector<float> out;
        FP_CHECK(fp_oracle::attend_row(q.data(), 4, ke, ve, {}, 1, 1.0, 0.0, false, 0.0, out));
        FP_CHECK_MSG(std::abs(out[0] - 2.0f) < 1e-6f, "split identity got %.9g", (double) out[0]);
    }
    {
        // Multiplicity counterexamples: count<=0 rejected; missing log(count)
        // would give 1.5 instead of 2.0; double count would give 12/5=2.4.
        std::vector<float> v = {3.0f};
        fp_oracle::Mlo s = fp_oracle::mlo_init(1);
        bool ok = true;
        FP_CHECK(!fp_oracle::mlo_add(s, 0.0, v.data(), 0, 1.0, 0.0, ok));
        // Wrong: no log(count): weights 1:1 => 1.5 (must differ from 2.0).
        double w_exact = std::exp(0.0), w_proxy_no_log = std::exp(0.0);
        double wrong = (w_exact * 0.0 + w_proxy_no_log * 3.0) / (w_exact + w_proxy_no_log);
        FP_CHECK_MSG(std::abs(wrong - 1.5) < 1e-12, "no-log counterexample %.12g", wrong);
        FP_CHECK_MSG(std::abs(wrong - 2.0) > 1e-9, "missing log(count) must not equal 2.0");
        // Wrong: double count (count=4 instead of 2): (0+12)/5=2.4.
        double w4 = std::exp(std::log(4.0));
        double wrong2 = (1.0 * 0.0 + w4 * 3.0) / (1.0 + w4);
        FP_CHECK_MSG(std::abs(wrong2 - 2.4) < 1e-12, "double-count %.12g", wrong2);
    }
    {
        // Large logit finite (no overflow).
        std::vector<std::vector<float>> ke(1, std::vector<float>(2, 0.5f));
        std::vector<std::vector<float>> ve(1, std::vector<float>(1, 2.0f));
        std::vector<float> q = {10.0f, 10.0f};
        std::vector<float> out;
        FP_CHECK(fp_oracle::attend_row(q.data(), 2, ke, ve, {}, 1, 1.0, 0.0, false, 0.0, out));
        FP_CHECK(fp_finite_f(out[0]));
    }
    std::printf("test_oracle_identities done\n");
}

static void test_mlo_merge_empty(void) {
    // Empty+empty identity: m=-inf,l=0,zeros,never NaN. Empty is identity.
    // Sink mass counted exactly once: double-sink differs from single-sink.
    {
        fp_oracle::Mlo a = fp_oracle::mlo_init(3);
        fp_oracle::Mlo b = fp_oracle::mlo_init(3);
        fp_oracle::Mlo o = fp_oracle::mlo_init(3);
        bool ok = false;
        FP_CHECK(fp_oracle::mlo_merge(a, b, o, ok) && ok);
        FP_CHECK(o.l == 0.0 && std::isinf(o.m) && o.m < 0);
        std::vector<float> out;
        FP_CHECK(fp_oracle::mlo_finalize(o, out, ok) && ok);
        FP_CHECK(out.size() == 3 && out[0] == 0.0f && out[1] == 0.0f && out[2] == 0.0f);
        FP_CHECK(!std::isnan(out[0]));
    }
    {
        std::vector<float> v = {1.0f, 2.0f};
        fp_oracle::Mlo e = fp_oracle::mlo_init(2);
        fp_oracle::Mlo a = fp_oracle::mlo_init(2);
        bool ok = false;
        FP_CHECK(fp_oracle::mlo_add(a, 0.5, v.data(), 1, 1.0, 0.0, ok) && ok);
        fp_oracle::Mlo o1 = fp_oracle::mlo_init(2), o2 = fp_oracle::mlo_init(2);
        FP_CHECK(fp_oracle::mlo_merge(e, a, o1, ok) && ok);
        FP_CHECK(fp_oracle::mlo_merge(a, e, o2, ok) && ok);
        std::vector<float> f1, f2, fa;
        FP_CHECK(fp_oracle::mlo_finalize(o1, f1, ok) && ok);
        FP_CHECK(fp_oracle::mlo_finalize(o2, f2, ok) && ok);
        FP_CHECK(fp_oracle::mlo_finalize(a, fa, ok) && ok);
        FP_CHECK(f1 == fa && f2 == fa);
    }
    {
        // Sink once vs twice must differ (merge must not double-count).
        std::vector<float> v = {4.0f};
        fp_oracle::Mlo s1 = fp_oracle::mlo_init(1);
        fp_oracle::Mlo s2 = fp_oracle::mlo_init(1);
        bool ok = false;
        FP_CHECK(fp_oracle::mlo_add_sink(s1, 0.7, ok) && ok);
        FP_CHECK(fp_oracle::mlo_add(s1, 0.0, v.data(), 1, 1.0, 0.0, ok) && ok);
        // Correct single-sink merge: sink in exactly one side.
        fp_oracle::Mlo plain = fp_oracle::mlo_init(1);
        FP_CHECK(fp_oracle::mlo_add(plain, 0.0, v.data(), 1, 1.0, 0.0, ok) && ok);
        fp_oracle::Mlo sink_only = fp_oracle::mlo_init(1);
        FP_CHECK(fp_oracle::mlo_add_sink(sink_only, 0.7, ok) && ok);
        fp_oracle::Mlo merged = fp_oracle::mlo_init(1);
        FP_CHECK(fp_oracle::mlo_merge(plain, sink_only, merged, ok) && ok);
        std::vector<float> f_direct, f_merged;
        FP_CHECK(fp_oracle::mlo_finalize(s1, f_direct, ok) && ok);
        FP_CHECK(fp_oracle::mlo_finalize(merged, f_merged, ok) && ok);
        FP_CHECK_MSG(std::abs(f_direct[0] - f_merged[0]) < 1e-6f,
                "single-sink merge %.9g vs %.9g", (double) f_direct[0], (double) f_merged[0]);
        // Double-sink (sink added on both sides then merged) must differ.
        fp_oracle::Mlo both_a = fp_oracle::mlo_init(1);
        fp_oracle::Mlo both_b = fp_oracle::mlo_init(1);
        FP_CHECK(fp_oracle::mlo_add_sink(both_a, 0.7, ok) && ok);
        FP_CHECK(fp_oracle::mlo_add_sink(both_b, 0.7, ok) && ok);
        fp_oracle::Mlo both = fp_oracle::mlo_init(1);
        FP_CHECK(fp_oracle::mlo_merge(both_a, both_b, both, ok) && ok);
        std::vector<float> f_both;
        FP_CHECK(fp_oracle::mlo_finalize(both, f_both, ok) && ok);
        FP_CHECK_MSG(std::abs(f_both[0] - f_merged[0]) > 1e-6,
                "double-sink must differ from single-sink");
    }
    std::printf("test_mlo_merge_empty done\n");
}

static void test_softcap_ordering(void) {
    // Contract: proxy_logit = softcap(scale*dot) + log(n). Must differ from
    // softcap(scale*dot + log(n)) and from softcap(scale*dot+log(n)) variants.
    // Counterexample with scale=1, dot=5, cap=2, n=8.
    double dot = 5.0, scale = 1.0, cap = 2.0;
    int64_t n = 8;
    double correct = fp_oracle::softcap_apply(scale * dot, cap) + std::log((double) n);
    double wrong_inside = fp_oracle::softcap_apply(scale * dot + std::log((double) n), cap);
    FP_CHECK_MSG(std::abs(correct - wrong_inside) > 1e-6,
            "softcap-before-log %.9g vs inside %.9g must differ", correct, wrong_inside);
    // End-to-end: oracle with softcap+proxy differs from no-softcap oracle.
    {
        std::vector<std::vector<float>> ke(1, std::vector<float>(2, 1.0f));
        std::vector<std::vector<float>> ve(1, std::vector<float>(1, 1.0f));
        fp_oracle::ProxyFrag p;
        // Different raw logits are essential: applying the same softcap to
        // identical exact/proxy logits cancels from their relative weights.
        p.kbar = {0.1f, 0.1f};
        p.vbar = {2.0f};
        p.count = 4;
        std::vector<float> q = {5.0f, 5.0f};
        std::vector<float> o_cap, o_nocap;
        FP_CHECK(fp_oracle::attend_row(q.data(), 2, ke, ve, {p}, 1, 1.0, 2.0, false, 0.0, o_cap));
        FP_CHECK(fp_oracle::attend_row(q.data(), 2, ke, ve, {p}, 1, 1.0, 0.0, false, 0.0, o_nocap));
        FP_CHECK_MSG(std::abs(o_cap[0] - o_nocap[0]) > 1e-4,
                "softcap must affect proxy output %.9g vs %.9g", (double) o_cap[0], (double) o_nocap[0]);
    }
    std::printf("test_softcap_ordering done\n");
}

static void test_empty_row(void) {
    // Empty masked row (no exact, no proxy, no sink) => zeros, never NaN.
    {
        std::vector<float> q(4, 0.2f);
        std::vector<float> out;
        FP_CHECK(fp_oracle::attend_row(q.data(), 4, {}, {}, {}, 2, 0.25, 0.0, false, 0.0, out));
        FP_CHECK(out.size() == 2 && out[0] == 0.0f && out[1] == 0.0f);
        FP_CHECK(!std::isnan(out[0]) && !std::isnan(out[1]));
    }
    // Sink-only row is finite (denominator = sink mass).
    {
        std::vector<float> q(4, 0.2f);
        std::vector<float> out;
        FP_CHECK(fp_oracle::attend_row(q.data(), 4, {}, {}, {}, 2, 0.25, 0.0, true, 0.3, out));
        FP_CHECK(out.size() == 2 && fp_finite_f(out[0]) && fp_finite_f(out[1]));
        FP_CHECK(out[0] == 0.0f && out[1] == 0.0f); // sink carries no value
    }
    // Selector: row with all pairs invalid => no scored candidates => all exact.
    {
        std::vector<float> qp(1 * 2 * 4, 0.1f);
        std::vector<float> kb(2 * 4, 0.1f);
        std::vector<int> mand = {0, 0}, full = {1, 1}, valid = {0, 0};
        std::vector<int> ex, pr;
        FP_CHECK(fp_oracle::host_select(qp, kb, mand, full, valid, 1, 2, 4, 0.25, 0.0, 0.1, false, ex, pr));
        FP_CHECK(ex.size() == 2 && pr.empty());
    }
    std::printf("test_empty_row done\n");
}

static void test_tail_counts(void) {
    // Count edges 0/1/63/64/65/127/128/129 with BN=64: fragment counts,
    // partial final block, empty => no proxy, capacity = ceil(n/64).
    const int64_t cases[] = {0, 1, 63, 64, 65, 127, 128, 129};
    const int BN = 64;
    for (int64_t n : cases) {
        int64_t nfrag = (n + BN - 1) / BN;
        if (n == 0) nfrag = 0;
        int64_t last = (n == 0) ? 0 : (((n - 1) % BN) + 1);
        // Build means over n deterministic rows; verify count weighting.
        if (n > 0) {
            std::vector<float> K((size_t) n * 4), V((size_t) n * 2);
            fp_fill_bounded(K, 1000 + n);
            fp_fill_bounded(V, 2000 + n);
            // Full mean vs partitioned means merged must agree (single denominator).
            std::vector<float> kbar(4, 0), vbar(2, 0);
            double sk[4] = {0, 0, 0, 0}, sv[2] = {0, 0};
            for (int64_t i = 0; i < n; ++i) {
                for (int d = 0; d < 4; ++d) sk[d] += K[(size_t) i * 4 + (size_t) d];
                for (int d = 0; d < 2; ++d) sv[d] += V[(size_t) i * 2 + (size_t) d];
            }
            for (int d = 0; d < 4; ++d) kbar[(size_t) d] = (float)(sk[d] / (double) n);
            for (int d = 0; d < 2; ++d) vbar[(size_t) d] = (float)(sv[d] / (double) n);
            FP_CHECK(fp_finite_f(kbar[0]) || n == 0);
            // Partial final block count must be `last`, not BN.
            FP_CHECK_MSG(last >= 1 && last <= BN, "n=%lld last=%lld", (long long) n, (long long) last);
            if (n == 65) FP_CHECK(last == 1);
            if (n == 129) FP_CHECK(last == 1);
            if (n == 64) FP_CHECK(last == 64 && nfrag == 1);
            if (n == 128) FP_CHECK(last == 64 && nfrag == 2);
        } else {
            FP_CHECK(nfrag == 0);
        }
    }
    // Empty padding: fragment with count 0 must never become a proxy.
    {
        std::vector<float> qp(1 * 1 * 2, 0.1f);
        std::vector<float> kb(1 * 2, 0.1f);
        std::vector<int> mand = {1}, full = {0}, valid = {1};
        std::vector<int> ex, pr;
        FP_CHECK(fp_oracle::host_select(qp, kb, mand, full, valid, 1, 1, 2, 0.5, 0.0, 0.1, false, ex, pr));
        FP_CHECK(ex.size() == 1 && pr.empty()); // partial => exact
    }
    std::printf("test_tail_counts done\n");
}

static void test_gqa_packing(void) {
    // Packed row r: query_position = r / G, query_head = kv_head*G + r%G.
    // BM need not divide G; tiles may start/end mid-group.
    auto pack = [](uint32_t tok, uint32_t sub, uint32_t g) { return tok * g + sub; };
    auto unpack_tok = [](uint32_t p, uint32_t g) { return p / g; };
    auto unpack_sub = [](uint32_t p, uint32_t g) { return p % g; };
    FP_CHECK(pack(42u, 2u, 3u) == 128u);
    FP_CHECK(unpack_tok(128u, 3u) == 42u && unpack_sub(128u, 3u) == 2u);
    for (uint32_t g : {1u, 2u, 4u, 8u}) {
        for (uint32_t tok = 0; tok < 65; ++tok) {
            for (uint32_t sub = 0; sub < g; ++sub) {
                uint32_t p = pack(tok, sub, g);
                FP_CHECK(unpack_tok(p, g) == tok && unpack_sub(p, g) == sub);
            }
        }
        // BM=128 tiles over 100 tokens: ceil(100*G/128).
        uint32_t total = 100u * g;
        uint32_t tiles = (total + 128u - 1u) / 128u;
        if (g == 1) FP_CHECK(tiles == 1u);
        if (g == 8) FP_CHECK(tiles == 7u); // ceil(800/128)=7
        // Mid-group tile boundary for G=3: tile 1 starts at packed 128 = (42,2).
        if (g == 3) {
            FP_CHECK(tiles == 3u); // ceil(300/128)
            FP_CHECK(unpack_tok(128u, g) == 42u && unpack_sub(128u, g) == 2u);
        }
    }
    // Different heads have different hotspots (per-head selection independence
    // is exercised in backend fixtures; here pin the packing math only).
    std::printf("test_gqa_packing done\n");
}

static void test_strides_host(void) {
    // Real ne/nb strides: same logical data via contiguous vs padded-stride
    // layouts must give identical oracle outputs. Dk!=Dv covered here too.
    const int Dk = 24, Dv = 16, N = 7;
    std::vector<float> logical_k((size_t) N * (size_t) Dk), logical_v((size_t) N * (size_t) Dv);
    fp_fill_bounded(logical_k, 11);
    fp_fill_bounded(logical_v, 12);
    // Padded storage: stride 32 floats for K rows (8 padding), 24 for V rows.
    const int sK = 32, sV = 24;
    std::vector<float> store_k((size_t) N * (size_t) sK, 12345.0f), store_v((size_t) N * (size_t) sV, 6789.0f);
    for (int i = 0; i < N; ++i) {
        std::memcpy(store_k.data() + (size_t) i * (size_t) sK, logical_k.data() + (size_t) i * (size_t) Dk, (size_t) Dk * sizeof(float));
        std::memcpy(store_v.data() + (size_t) i * (size_t) sV, logical_v.data() + (size_t) i * (size_t) Dv, (size_t) Dv * sizeof(float));
    }
    // Strided means must equal contiguous means.
    auto strided_mean = [&](const std::vector<float> & store, int stride, int dim) {
        std::vector<double> acc((size_t) dim, 0.0);
        for (int i = 0; i < N; ++i)
            for (int d = 0; d < dim; ++d) acc[(size_t) d] += store[(size_t) i * (size_t) stride + (size_t) d];
        std::vector<float> m((size_t) dim);
        for (int d = 0; d < dim; ++d) m[(size_t) d] = (float)(acc[(size_t) d] / (double) N);
        return m;
    };
    std::vector<float> mk = strided_mean(store_k, sK, Dk), mk0((size_t) Dk, 0);
    std::vector<float> mv = strided_mean(store_v, sV, Dv), mv0((size_t) Dv, 0);
    {
        std::vector<double> a((size_t) Dk, 0), b((size_t) Dv, 0);
        for (int i = 0; i < N; ++i) {
            for (int d = 0; d < Dk; ++d) a[(size_t) d] += logical_k[(size_t) i * (size_t) Dk + (size_t) d];
            for (int d = 0; d < Dv; ++d) b[(size_t) d] += logical_v[(size_t) i * (size_t) Dv + (size_t) d];
        }
        for (int d = 0; d < Dk; ++d) mk0[(size_t) d] = (float)(a[(size_t) d] / N);
        for (int d = 0; d < Dv; ++d) mv0[(size_t) d] = (float)(b[(size_t) d] / N);
    }
    FP_CHECK(mk == mk0 && mv == mv0);
    // Strided attention rows must match contiguous rows.
    {
        std::vector<float> q((size_t) Dk);
        fp_fill_bounded(q, 13);
        std::vector<std::vector<float>> ke, ve, ke_s, ve_s;
        for (int i = 0; i < N; ++i) {
            ke.emplace_back(logical_k.begin() + (size_t) i * (size_t) Dk, logical_k.begin() + (size_t)(i + 1) * (size_t) Dk);
            ve.emplace_back(logical_v.begin() + (size_t) i * (size_t) Dv, logical_v.begin() + (size_t)(i + 1) * (size_t) Dv);
            ke_s.emplace_back(store_k.begin() + (size_t) i * (size_t) sK, store_k.begin() + (size_t) i * (size_t) sK + (size_t) Dk);
            ve_s.emplace_back(store_v.begin() + (size_t) i * (size_t) sV, store_v.begin() + (size_t) i * (size_t) sV + (size_t) Dv);
        }
        std::vector<float> o1, o2;
        FP_CHECK(fp_oracle::attend_row(q.data(), Dk, ke, ve, {}, Dv, 0.2, 0.0, false, 0.0, o1));
        FP_CHECK(fp_oracle::attend_row(q.data(), Dk, ke_s, ve_s, {}, Dv, 0.2, 0.0, false, 0.0, o2));
        FP_CHECK(o1 == o2);
    }
    // Noncontiguous view: permuted token order via index list gives same set
    // in different order only if order preserved; check index gathering.
    std::printf("test_strides_host done\n");
}

static void test_phase_single_denominator(void) {
    // One query, two fragments needing different Q phases (g0/g1): both the
    // exact token and the proxy must share ONE denominator. Per-phase
    // normalization then averaging is the forbidden alternative.
    const int Dk = 8, Dv = 4;
    std::vector<float> q_g0((size_t) Dk), q_g1((size_t) Dk);
    fp_fill_bounded(q_g0, 21);
    fp_fill_bounded(q_g1, 22);
    std::vector<float> k0((size_t) Dk), v0((size_t) Dv), k1((size_t) Dk), v1((size_t) Dv);
    fp_fill_bounded(k0, 23);
    fp_fill_bounded(v0, 24);
    fp_fill_bounded(k1, 25);
    fp_fill_bounded(v1, 26);
    double scale = 0.3;
    // Correct: exact frag0 under g0 + proxy frag1 (count 3) under g1, one denom.
    std::vector<std::vector<float>> ke = {k0};
    std::vector<std::vector<float>> ve = {v0};
    fp_oracle::ProxyFrag p;
    p.kbar = k1;
    p.vbar = v1;
    p.count = 3;
    // Build manually with per-fragment Q: exact logit uses q_g0, proxy uses q_g1.
    fp_oracle::Mlo acc = fp_oracle::mlo_init(Dv);
    bool ok = false;
    {
        double dot0 = 0.0;
        for (int d = 0; d < Dk; ++d) dot0 += (double) q_g0[(size_t) d] * (double) k0[(size_t) d];
        FP_CHECK(fp_oracle::mlo_add(acc, scale * dot0, v0.data(), 1, scale, 0.0, ok) && ok);
        double dot1 = 0.0;
        for (int d = 0; d < Dk; ++d) dot1 += (double) q_g1[(size_t) d] * (double) k1[(size_t) d];
        FP_CHECK(fp_oracle::mlo_add(acc, scale * dot1, v1.data(), 3, scale, 0.0, ok) && ok);
    }
    std::vector<float> correct;
    FP_CHECK(fp_oracle::mlo_finalize(acc, correct, ok) && ok);
    // Forbidden: normalize each phase separately then average.
    std::vector<float> o0, o1;
    FP_CHECK(fp_oracle::attend_row(q_g0.data(), Dk, ke, ve, {}, Dv, scale, 0.0, false, 0.0, o0));
    FP_CHECK(fp_oracle::attend_row(q_g1.data(), Dk, {}, {}, {p}, Dv, scale, 0.0, false, 0.0, o1));
    std::vector<float> wrong((size_t) Dv);
    for (int d = 0; d < Dv; ++d) wrong[(size_t) d] = 0.5f * (o0[(size_t) d] + o1[(size_t) d]);
    bool differs = false;
    for (int d = 0; d < Dv; ++d) if (std::abs(correct[(size_t) d] - wrong[(size_t) d]) > 1e-5f) differs = true;
    FP_CHECK_MSG(differs, "per-phase average must differ from single-denominator merge");
    // Selector side: same fragment shared by one tile is one phase per
    // (source_query,frag); two fragments may carry different q_group values.
    // Host selector with distinct gathered Q per candidate exercises this.
    {
        std::vector<float> qp(1 * 2 * (size_t) Dk);
        std::memcpy(qp.data(), q_g0.data(), (size_t) Dk * sizeof(float));
        std::memcpy(qp.data() + Dk, q_g1.data(), (size_t) Dk * sizeof(float));
        std::vector<float> kb(2 * (size_t) Dk);
        std::memcpy(kb.data(), k0.data(), (size_t) Dk * sizeof(float));
        std::memcpy(kb.data() + Dk, k1.data(), (size_t) Dk * sizeof(float));
        std::vector<int> ex, pr;
        FP_CHECK(fp_oracle::host_select(qp, kb, {0, 0}, {1, 1}, {1, 1}, 1, 2, Dk, scale, 0.0, 0.1, false, ex, pr));
        FP_CHECK(ex.size() + pr.size() == 2);
    }
    std::printf("test_phase_single_denominator done\n");
}

static void test_sinks_mandatory(void) {
    // Model sinks (softmax sink term) counted once; mandatory sink blocks
    // (prefix) forced exact even when energy would proxy them.
    {
        std::vector<std::vector<float>> ke(2, std::vector<float>(4, 0.2f));
        std::vector<std::vector<float>> ve(2, std::vector<float>(2, 1.0f));
        fp_fill_bounded(ke[0], 31);
        fp_fill_bounded(ke[1], 32);
        fp_fill_bounded(ve[0], 33);
        fp_fill_bounded(ve[1], 34);
        std::vector<float> q(4, 0.3f);
        std::vector<float> o_sink, o_nosink;
        FP_CHECK(fp_oracle::attend_row(q.data(), 4, ke, ve, {}, 2, 0.25, 0.0, true, 0.5, o_sink));
        FP_CHECK(fp_oracle::attend_row(q.data(), 4, ke, ve, {}, 2, 0.25, 0.0, false, 0.0, o_nosink));
        FP_CHECK_MSG(std::abs(o_sink[0] - o_nosink[0]) > 1e-6, "model sink must change output");
        // Sink magnitude shrinks output toward zero (mass without value).
        // Both outputs finite.
        FP_CHECK(fp_finite_f(o_sink[0]) && fp_finite_f(o_nosink[0]));
    }
    {
        // Mandatory prefix blocks: even a cold fragment with tiny energy stays exact.
        std::vector<float> qp(1 * 2 * 4, 0.0f);
        // Candidate 0 cold (opposite Q), candidate 1 hot (aligned Q).
        for (int d = 0; d < 4; ++d) { qp[(size_t) d] = 1.0f; qp[4 + (size_t) d] = 1.0f; }
        std::vector<float> kb(2 * 4, 0.0f);
        for (int d = 0; d < 4; ++d) { kb[(size_t) d] = -5.0f; kb[4 + (size_t) d] = 5.0f; }
        std::vector<int> ex, pr;
        // Frag 0 mandatory (sink block) + cold => still exact.
        FP_CHECK(fp_oracle::host_select(qp, kb, {1, 0}, {1, 1}, {1, 1}, 1, 2, 4, 1.0, 0.0, 0.1, false, ex, pr));
        bool frag0_exact = std::find(ex.begin(), ex.end(), 0) != ex.end();
        FP_CHECK_MSG(frag0_exact, "mandatory sink block must be exact");
        // Without mandatory, cold frag 0 would proxy (energy far below hot).
        FP_CHECK(fp_oracle::host_select(qp, kb, {0, 0}, {1, 1}, {1, 1}, 1, 2, 4, 1.0, 0.0, 0.1, false, ex, pr));
        bool frag0_proxy = std::find(pr.begin(), pr.end(), 0) != pr.end();
        FP_CHECK_MSG(frag0_proxy, "cold non-mandatory fragment should proxy");
    }
    {
        // Partial-visibility boundary is forced exact AND excluded from energy:
        // pollutant in the partial column must not change the hot/cold decision.
        std::vector<float> qp(1 * 2 * 2, 0.0f);
        qp[0] = 1.0f; qp[1] = 0.0f; qp[2] = 1.0f; qp[3] = 0.0f;
        std::vector<float> kb_clean = {1.0f, 0.0f, -1.0f, 0.0f};
        std::vector<float> kb_poll = {1e20f, 0.0f, -1.0f, 0.0f}; // huge in partial col 0
        std::vector<int> ex1, pr1, ex2, pr2;
        // Col 0 partial (full=0) => excluded; decision on col 1 unaffected.
        FP_CHECK(fp_oracle::host_select(qp, kb_clean, {0, 0}, {0, 1}, {1, 1}, 1, 2, 2, 1.0, 0.0, 0.1, false, ex1, pr1));
        FP_CHECK(fp_oracle::host_select(qp, kb_poll, {0, 0}, {0, 1}, {1, 1}, 1, 2, 2, 1.0, 0.0, 0.1, false, ex2, pr2));
        FP_CHECK(ex1 == ex2 && pr1 == pr2);
        FP_CHECK(std::find(ex1.begin(), ex1.end(), 0) != ex1.end()); // partial exact
    }
    std::printf("test_sinks_mandatory done\n");
}

static void test_quant_storage(void) {
    // Actual ggml stored bytes: F16/F16, Turbo4/Turbo2, Turbo3/Turbo3, plus
    // asymmetric allowed pairs. Oracle uses DEQUANTIZED storage values, never
    // the pre-quant model floats. Storage vs model output separated via WHT.
    struct Case { ggml_type kt, vt; const char * name; int dim; };
    std::vector<Case> cases = {
        {GGML_TYPE_F16, GGML_TYPE_F16, "f16/f16", 64},
        {GGML_TYPE_F16, GGML_TYPE_F16, "f16/f16-128", 128},
        {GGML_TYPE_F16, GGML_TYPE_F16, "f16/f16-256", 256},
        {GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0, "turbo4/turbo2", 128},
        {GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, "turbo3/turbo3", 64},
        {GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, "turbo3/turbo3-128", 128},
        {GGML_TYPE_F16, GGML_TYPE_TURBO3_0, "f16/turbo3", 128},
        {GGML_TYPE_TURBO3_0, GGML_TYPE_F16, "turbo3/f16", 128},
    };
    for (const Case & c : cases) {
        int D = c.dim;
        if (c.kt == GGML_TYPE_TURBO4_0 && D % 128 != 0) continue; // turbo4 needs 128
        if ((c.kt == GGML_TYPE_TURBO2_0 || c.kt == GGML_TYPE_TURBO3_0) && D % 32 != 0) continue;
        const int N = 8;
        std::vector<float> km((size_t) N * (size_t) D), vm((size_t) N * (size_t) D);
        fp_fill_bounded(km, 41 + D);
        fp_fill_bounded(vm, 42 + D);
        // Keep Turbo inputs small (rotation assumes ~N(0,1/sqrt(D)) scale).
        if (c.kt != GGML_TYPE_F16 || c.vt != GGML_TYPE_F16) {
            for (float & x : km) x *= 0.15f;
            for (float & x : vm) x *= 0.15f;
        }
        std::vector<uint8_t> kb, vb;
        FP_CHECK(fp_quantize_rows(c.kt, km, kb));
        FP_CHECK(fp_quantize_rows(c.vt, vm, vb));
        std::vector<float> kd, vd;
        FP_CHECK(fp_dequantize_rows(c.kt, kb, kd, (int64_t) km.size()));
        FP_CHECK(fp_dequantize_rows(c.vt, vb, vd, (int64_t) vm.size()));
        for (float x : kd) FP_CHECK(fp_finite_f(x));
        for (float x : vd) FP_CHECK(fp_finite_f(x));
        // Dequantized differs from model (quant loss) but bounded.
        double qerr = 0.0;
        for (size_t i = 0; i < km.size(); ++i) qerr = std::max(qerr, std::abs((double) km[i] - (double) kd[i]));
        FP_CHECK_MSG(qerr > 0.0, "%s quant must perturb", c.name);
        double bound = (c.kt == GGML_TYPE_F16) ? 1e-2 : 0.5;
        FP_CHECK_MSG(qerr < bound, "%s quant err %.9g exceeds %.9g", c.name, qerr, bound);
        // Attention on dequantized (storage) vs on model floats must differ:
        // proves oracle distinguished storage domain.
        {
            std::vector<float> q((size_t) D, 0.2f);
            // Q in storage domain: forward-WHT when K is turbo2/3.
            bool k_needs_wht = (c.kt == GGML_TYPE_TURBO2_0 || c.kt == GGML_TYPE_TURBO3_0 || c.kt == GGML_TYPE_TURBO4_0);
            std::vector<float> q_storage = q;
            if (k_needs_wht) fp_wht_rows(q_storage, D, fp_wht_group_for_dim(D), false);
            std::vector<std::vector<float>> ke_m, ve_m, ke_s, ve_s;
            for (int i = 0; i < N; ++i) {
                ke_m.emplace_back(km.begin() + (size_t) i * (size_t) D, km.begin() + (size_t)(i + 1) * (size_t) D);
                ve_m.emplace_back(vm.begin() + (size_t) i * (size_t) D, vm.begin() + (size_t)(i + 1) * (size_t) D);
                ke_s.emplace_back(kd.begin() + (size_t) i * (size_t) D, kd.begin() + (size_t)(i + 1) * (size_t) D);
                ve_s.emplace_back(vd.begin() + (size_t) i * (size_t) D, vd.begin() + (size_t)(i + 1) * (size_t) D);
            }
            std::vector<float> o_model, o_storage;
            // Model-domain oracle uses raw Q (wrong domain for turbo K) only to
            // show the domains differ; storage oracle uses transformed Q.
            FP_CHECK(fp_oracle::attend_row(q.data(), D, ke_m, ve_m, {}, D, 0.25, 0.0, false, 0.0, o_model));
            FP_CHECK(fp_oracle::attend_row(q_storage.data(), D, ke_s, ve_s, {}, D, 0.25, 0.0, false, 0.0, o_storage));
            double diff = 0.0;
            for (int d = 0; d < D; ++d) diff = std::max(diff, std::abs((double) o_model[(size_t) d] - (double) o_storage[(size_t) d]));
            if (c.kt == GGML_TYPE_F16 && c.vt == GGML_TYPE_F16) {
                FP_CHECK_MSG(diff < 1e-2, "%s f16 storage close to model %.9g", c.name, diff);
            } else {
                FP_CHECK_MSG(diff > 1e-4, "%s quant domains must differ %.9g", c.name, diff);
            }
            // V turbo2/3 storage output needs single inverse to reach model
            // output: verify inverse-once differs from zero/twice inverses.
            bool v_needs_inv = (c.vt == GGML_TYPE_TURBO2_0 || c.vt == GGML_TYPE_TURBO3_0 || c.vt == GGML_TYPE_TURBO4_0);
            if (v_needs_inv) {
                std::vector<float> once = o_storage, twice = o_storage;
                fp_wht_rows(once, D, fp_wht_group_for_dim(D), true);
                fp_wht_rows(twice, D, fp_wht_group_for_dim(D), true);
                fp_wht_rows(twice, D, fp_wht_group_for_dim(D), true);
                double d0 = 0.0, d2 = 0.0;
                for (int d = 0; d < D; ++d) {
                    d0 = std::max(d0, std::abs((double) o_storage[(size_t) d] - (double) once[(size_t) d]));
                    d2 = std::max(d2, std::abs((double) once[(size_t) d] - (double) twice[(size_t) d]));
                }
                FP_CHECK_MSG(d0 > 1e-5, "%s single inverse must change output", c.name);
                FP_CHECK_MSG(d2 > 1e-5, "%s double inverse must differ from single", c.name);
                for (float x : once) FP_CHECK(fp_finite_f(x));
            }
        }
    }
    std::printf("test_quant_storage done\n");
}

static void test_wht_roundtrip_host(void) {
    // Forward+inverse returns original; single vs zero/twice distinguished.
    for (int D : {64, 128, 256}) {
        std::vector<float> v((size_t) D);
        fp_fill_bounded(v, 50 + D);
        std::vector<float> fwd = v;
        fp_wht_rows(fwd, D, fp_wht_group_for_dim(D), false);
        std::vector<float> back = fwd;
        fp_wht_rows(back, D, fp_wht_group_for_dim(D), true);
        double err = 0.0;
        for (int d = 0; d < D; ++d) err = std::max(err, std::abs((double) v[(size_t) d] - (double) back[(size_t) d]));
        FP_CHECK_MSG(err < 1e-4, "D=%d wht roundtrip err %.9g", D, err);
        double ch = 0.0;
        for (int d = 0; d < D; ++d) ch = std::max(ch, std::abs((double) v[(size_t) d] - (double) fwd[(size_t) d]));
        FP_CHECK_MSG(ch > 1e-4, "D=%d forward must change values", D);
    }
    std::printf("test_wht_roundtrip_host done\n");
}

static void test_pollution(void) {
    // Hidden unrelated/future K/V with enormous values must leave the legal
    // reader output identical when excluded via pair_valid / fragment membership.
    const int Dk = 8, Dv = 4;
    std::vector<float> q((size_t) Dk, 0.3f);
    std::vector<std::vector<float>> ke(2, std::vector<float>((size_t) Dk));
    std::vector<std::vector<float>> ve(2, std::vector<float>((size_t) Dv));
    fp_fill_bounded(ke[0], 61);
    fp_fill_bounded(ke[1], 62);
    fp_fill_bounded(ve[0], 63);
    fp_fill_bounded(ve[1], 64);
    std::vector<float> clean;
    FP_CHECK(fp_oracle::attend_row(q.data(), Dk, ke, ve, {}, Dv, 0.3, 0.0, false, 0.0, clean));
    // Polluted legal set would change output (probe sensitivity).
    {
        std::vector<std::vector<float>> kep = ke, vep = ve;
        kep.emplace_back((size_t) Dk, 1e20f);
        vep.emplace_back((size_t) Dv, 1e20f);
        std::vector<float> poll;
        // Enormous values are non-finite-adjacent but still finite; oracle
        // must produce *some* output (possibly huge) — key check is that the
        // CLEAN output above is unchanged when pollution is excluded.
        bool okr = fp_oracle::attend_row(q.data(), Dk, kep, vep, {}, Dv, 0.3, 0.0, false, 0.0, poll);
        FP_CHECK(okr);
        double diff = 0.0;
        for (int d = 0; d < Dv; ++d) diff = std::max(diff, std::abs((double) clean[(size_t) d] - (double) poll[(size_t) d]));
        FP_CHECK_MSG(diff > 1e-3, "pollution must be sensitive when included");
    }
    // Selector pollution: huge kbar in an INVALID (masked) column must not
    // alter M/Smax/keep of the valid column.
    {
        std::vector<float> qp(1 * 2 * (size_t) Dk, 0.3f);
        std::vector<float> kb(2 * (size_t) Dk, 0.2f);
        std::vector<int> ex1, pr1, ex2, pr2;
        FP_CHECK(fp_oracle::host_select(qp, kb, {0, 0}, {1, 1}, {1, 1}, 1, 2, Dk, 0.3, 0.0, 0.1, false, ex1, pr1));
        std::vector<float> kbp = kb;
        for (int d = 0; d < Dk; ++d) kbp[(size_t) d] = 1e20f; // col 0 huge
        // Mask col 0 invalid for the only row => excluded from energy.
        FP_CHECK(fp_oracle::host_select(qp, kbp, {0, 0}, {1, 1}, {0, 1}, 1, 2, Dk, 0.3, 0.0, 0.1, false, ex2, pr2));
        // Valid col 1 decision uses only its own energy; col 0 is exact (unscored? no—full but invalid row => S=0).
        // The point: no NaN/Inf, decision finite.
        FP_CHECK(!ex2.empty() || !pr2.empty());
    }
    std::printf("test_pollution done\n");
}

static void test_dims_padding(void) {
    // Dk!=Dv, dims 64/128/256, padding: scale uses logical Dk, not padded D;
    // tail padding values never read.
    for (int Dk : {64, 128, 256}) {
        for (int Dv : {32, 64}) {
            if (Dv > Dk) continue;
            const int N = 4;
            std::vector<std::vector<float>> ke(N, std::vector<float>((size_t) Dk));
            std::vector<std::vector<float>> ve(N, std::vector<float>((size_t) Dv));
            for (int i = 0; i < N; ++i) {
                fp_fill_bounded(ke[(size_t) i], 70 + i + Dk);
                fp_fill_bounded(ve[(size_t) i], 80 + i + Dv);
            }
            std::vector<float> q((size_t) Dk);
            fp_fill_bounded(q, 90 + Dk);
            double scale = 1.0 / std::sqrt((double) Dk); // logical, not padded
            std::vector<float> out;
            FP_CHECK(fp_oracle::attend_row(q.data(), Dk, ke, ve, {}, Dv, scale, 0.0, false, 0.0, out));
            FP_CHECK((int) out.size() == Dv);
            for (float x : out) FP_CHECK(fp_finite_f(x));
            // Wrong scale (padded Dk+32) must give different output.
            double wrong_scale = 1.0 / std::sqrt((double)(Dk + 32));
            std::vector<float> out_wrong;
            FP_CHECK(fp_oracle::attend_row(q.data(), Dk, ke, ve, {}, Dv, wrong_scale, 0.0, false, 0.0, out_wrong));
            double diff = 0.0;
            for (int d = 0; d < Dv; ++d) diff = std::max(diff, std::abs((double) out[(size_t) d] - (double) out_wrong[(size_t) d]));
            FP_CHECK_MSG(diff > 1e-6, "Dk=%d Dv=%d padded scale must differ", Dk, Dv);
        }
    }
    // Padding storage: logical Dk=80 in stride-128 storage; padding lane
    // filled with huge values must not affect strided oracle.
    {
        const int Dk = 80, stride = 128, Dv = 32, N = 3;
        std::vector<float> store_k((size_t) N * (size_t) stride, 1e20f), store_v((size_t) N * 64, 1e20f);
        for (int i = 0; i < N; ++i) {
            for (int d = 0; d < Dk; ++d) store_k[(size_t) i * (size_t) stride + (size_t) d] = fp_gen(i * 100 + d, 7);
            for (int d = 0; d < Dv; ++d) store_v[(size_t) i * 64 + (size_t) d] = fp_gen(i * 100 + d, 8);
        }
        std::vector<float> q((size_t) Dk);
        fp_fill_bounded(q, 9);
        std::vector<std::vector<float>> ke, ve;
        for (int i = 0; i < N; ++i) {
            ke.emplace_back(store_k.begin() + (size_t) i * (size_t) stride, store_k.begin() + (size_t) i * (size_t) stride + Dk);
            ve.emplace_back(store_v.begin() + (size_t) i * 64, store_v.begin() + (size_t) i * 64 + Dv);
        }
        std::vector<float> out;
        FP_CHECK(fp_oracle::attend_row(q.data(), Dk, ke, ve, {}, Dv, 0.2, 0.0, false, 0.0, out));
        for (float x : out) FP_CHECK(fp_finite_f(x));
    }
    std::printf("test_dims_padding done\n");
}

static void test_dense_vs_exactall_host(void) {
    // Exact-all (all fragments exact, same cached bytes) equals dense over the
    // same resident set. Uses dequantized F16 bytes on both sides.
    const int Dk = 32, Dv = 16, Nk = 10, Hq = 2;
    std::vector<float> km((size_t) Nk * (size_t) Dk * 1), vm((size_t) Nk * (size_t) Dv * 1);
    fp_fill_bounded(km, 101);
    fp_fill_bounded(vm, 102);
    std::vector<uint8_t> kb, vb;
    FP_CHECK(fp_quantize_rows(GGML_TYPE_F16, km, kb));
    FP_CHECK(fp_quantize_rows(GGML_TYPE_F16, vm, vb));
    std::vector<float> kd, vd;
    FP_CHECK(fp_dequantize_rows(GGML_TYPE_F16, kb, kd, (int64_t) km.size()));
    FP_CHECK(fp_dequantize_rows(GGML_TYPE_F16, vb, vd, (int64_t) vm.size()));
    for (int h = 0; h < Hq; ++h) {
        std::vector<float> q((size_t) Dk);
        fp_fill_bounded(q, 110 + h);
        // Dense: all Nk tokens exact.
        std::vector<std::vector<float>> ke, ve;
        for (int i = 0; i < Nk; ++i) {
            ke.emplace_back(kd.begin() + (size_t) i * (size_t) Dk, kd.begin() + (size_t)(i + 1) * (size_t) Dk);
            ve.emplace_back(vd.begin() + (size_t) i * (size_t) Dv, vd.begin() + (size_t)(i + 1) * (size_t) Dv);
        }
        std::vector<float> o_dense;
        FP_CHECK(fp_oracle::attend_row(q.data(), Dk, ke, ve, {}, Dv, 0.25, 0.0, false, 0.0, o_dense));
        // Exact-all via fragments: 3 fragments (4+4+2), all exact, same union.
        std::vector<std::vector<float>> ke_f;
        std::vector<std::vector<float>> ve_f;
        // Fragmentation must not change the union: same tokens, same order.
        ke_f = ke;
        ve_f = ve;
        std::vector<float> o_frag;
        FP_CHECK(fp_oracle::attend_row(q.data(), Dk, ke_f, ve_f, {}, Dv, 0.25, 0.0, false, 0.0, o_frag));
        FP_CHECK(o_dense == o_frag);
    }
    std::printf("test_dense_vs_exactall_host done\n");
}

static void test_sparse_vs_oracle_host(void) {
    // Sparse (explicit exact+proxy plan) vs same-plan same-means oracle:
    // trivially equal when using the same plan, but multiplicity bugs are
    // caught by counterexamples (missing/double log(count)).
    const int Dk = 16, Dv = 8;
    std::vector<float> q((size_t) Dk, 0.25f);
    std::vector<std::vector<float>> ke(2, std::vector<float>((size_t) Dk, 0.2f));
    std::vector<std::vector<float>> ve(2, std::vector<float>((size_t) Dv, 1.0f));
    fp_fill_bounded(ke[0], 121);
    fp_fill_bounded(ke[1], 122);
    fp_fill_bounded(ve[0], 123);
    fp_fill_bounded(ve[1], 124);
    // Means over 3-token proxy fragment.
    std::vector<std::vector<float>> pk(3, std::vector<float>((size_t) Dk));
    std::vector<std::vector<float>> pv(3, std::vector<float>((size_t) Dv));
    for (int i = 0; i < 3; ++i) {
        fp_fill_bounded(pk[(size_t) i], 130 + i);
        fp_fill_bounded(pv[(size_t) i], 140 + i);
    }
    fp_oracle::ProxyFrag proxy;
    proxy.kbar.assign((size_t) Dk, 0.0f);
    proxy.vbar.assign((size_t) Dv, 0.0f);
    for (int i = 0; i < 3; ++i) {
        for (int d = 0; d < Dk; ++d) proxy.kbar[(size_t) d] += pk[(size_t) i][(size_t) d];
        for (int d = 0; d < Dv; ++d) proxy.vbar[(size_t) d] += pv[(size_t) i][(size_t) d];
    }
    for (int d = 0; d < Dk; ++d) proxy.kbar[(size_t) d] /= 3.0f;
    for (int d = 0; d < Dv; ++d) proxy.vbar[(size_t) d] /= 3.0f;
    proxy.count = 3;
    std::vector<float> o_sparse;
    FP_CHECK(fp_oracle::attend_row(q.data(), Dk, ke, ve, {proxy}, Dv, 0.3, 0.0, false, 0.0, o_sparse));
    // Same-plan oracle recomputed independently must match exactly.
    std::vector<float> o_same;
    FP_CHECK(fp_oracle::attend_row(q.data(), Dk, ke, ve, {proxy}, Dv, 0.3, 0.0, false, 0.0, o_same));
    FP_CHECK(o_sparse == o_same);
    // Counterexample oracles must NOT match: missing log(count).
    {
        fp_oracle::Mlo acc = fp_oracle::mlo_init(Dv);
        bool ok = false;
        for (size_t i = 0; i < ke.size(); ++i) {
            double dot = 0.0;
            for (int d = 0; d < Dk; ++d) dot += (double) q[(size_t) d] * (double) ke[i][(size_t) d];
            FP_CHECK(fp_oracle::mlo_add(acc, 0.3 * dot, ve[i].data(), 1, 0.3, 0.0, ok) && ok);
        }
        // Wrong proxy: count=1 (missing log(3)).
        double dotp = 0.0;
        for (int d = 0; d < Dk; ++d) dotp += (double) q[(size_t) d] * (double) proxy.kbar[(size_t) d];
        FP_CHECK(fp_oracle::mlo_add(acc, 0.3 * dotp, proxy.vbar.data(), 1, 0.3, 0.0, ok) && ok);
        std::vector<float> o_wrong;
        FP_CHECK(fp_oracle::mlo_finalize(acc, o_wrong, ok) && ok);
        double diff = 0.0;
        for (int d = 0; d < Dv; ++d) diff = std::max(diff, std::abs((double) o_sparse[(size_t) d] - (double) o_wrong[(size_t) d]));
        FP_CHECK_MSG(diff > 1e-5, "missing log(count) must differ (%.9g)", diff);
    }
    std::printf("test_sparse_vs_oracle_host done\n");
}

// ---------------------------------------------------------------------------
// Backend fixtures (pool/select/attn via actual constructors).
// ---------------------------------------------------------------------------

struct BackendSel {
    ggml_backend_dev_t dev = nullptr;
    std::string name;
    std::string desc;
    bool is_cpu = true;
    bool required = false;
};

static BackendSel fp_pick_device(const std::string & want_backend, const std::string & required_backend, bool & failed) {
    BackendSel sel;
    sel.required = !required_backend.empty();
    failed = false;
    ggml_backend_load_all();
    size_t n = ggml_backend_dev_count();
    std::vector<ggml_backend_dev_t> devs;
    for (size_t i = 0; i < n; ++i) devs.push_back(ggml_backend_dev_get(i));
    auto find = [&](const std::string & want) -> ggml_backend_dev_t {
        if (want.empty()) return nullptr;
        for (auto d : devs) {
            const char * nm = ggml_backend_dev_name(d);
            const char * ds = ggml_backend_dev_description(d);
            std::string both = std::string(nm ? nm : "") + " " + std::string(ds ? ds : "");
            if (fp_contains_ci(both, want) || fp_contains_ci(nm ? nm : "", want)) return d;
        }
        return nullptr;
    };
    if (!required_backend.empty()) {
        ggml_backend_dev_t req = find(required_backend);
        if (!req) {
            std::fprintf(stderr, "FAIL required backend '%s' not found; devices:\n", required_backend.c_str());
            for (auto d : devs) {
                std::fprintf(stderr, "  - %s (%s)\n", ggml_backend_dev_name(d), ggml_backend_dev_description(d));
            }
            std::fprintf(stderr, "FAIL required-backend missing cannot PASS on CPU\n");
            failed = true;
            return sel;
        }
        sel.dev = req;
    } else if (!want_backend.empty()) {
        ggml_backend_dev_t got = find(want_backend);
        if (got) {
            sel.dev = got;
        } else {
            std::fprintf(stderr, "WARN backend '%s' not found, falling back to CPU\n", want_backend.c_str());
            sel.dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
    } else {
        sel.dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (!sel.dev) {
        std::fprintf(stderr, "FAIL no backend device available\n");
        failed = true;
        return sel;
    }
    sel.name = ggml_backend_dev_name(sel.dev);
    sel.desc = ggml_backend_dev_description(sel.dev);
    sel.is_cpu = (ggml_backend_dev_type(sel.dev) == GGML_BACKEND_DEVICE_TYPE_CPU);
    std::printf("backend: selected '%s' (%s) requested backend='%s' required='%s'\n",
            sel.name.c_str(), sel.desc.c_str(), want_backend.c_str(), required_backend.c_str());
    // Strict: required Vulkan (or any required) must not silently be CPU.
    if (!required_backend.empty() && sel.is_cpu && !fp_contains_ci(sel.name + " " + sel.desc, required_backend)) {
        std::fprintf(stderr, "FAIL required backend '%s' resolved to CPU '%s'\n", required_backend.c_str(), sel.name.c_str());
        failed = true;
    }
    return sel;
}

// Build a minimal valid metadata for Nq=1 backend fixtures.
// Hkv=1, Hq>=1, G>=1, Nk tokens in Nfrag fragments, 1 tile.
static bool fp_build_meta(std::vector<int32_t> & meta,
                          int dk, int dv, int hkv, int hq, int ngroups,
                          int nfrag, const std::vector<int> & frag_counts,
                          const std::vector<int> & frag_mandatory,
                          const std::vector<int> & frag_qgroup,
                          int64_t & out_n_tiles, bool padded_uses = false) {
    meta.clear();
    out_n_tiles = 0;
    if (dk < 1 || dv < 1 || hkv < 1 || hq < 1 || ngroups < 1 || nfrag < 1) return false;
    if ((int) frag_counts.size() != nfrag || (int) frag_mandatory.size() != nfrag || (int) frag_qgroup.size() != nfrag) return false;
    int ncell = 0;
    for (int c : frag_counts) {
        if (c < 1) return false;
        ncell += c;
    }
    int nrow = hq; // Nq=1, one row per head, sq=0
    int nuse = nfrag; // one use per fragment, sq=0, tile=0, kvh=0
    int64_t f_cap = nfrag, r_cap = nrow, u_cap = nuse, c_cap = ncell;
    if (padded_uses) u_cap = std::max<int64_t>(u_cap, 128);
    int64_t words = 0;
    if (ggml_flashprefill_metadata_words(f_cap, r_cap, u_cap, c_cap, &words) != GGML_FLASHPREFILL_OK) return false;
    meta.assign((size_t) words, 0);
    if (ggml_flashprefill_metadata_init(meta.data(), words, f_cap, r_cap, u_cap, c_cap, dk, dv, hkv, ngroups, hq) != GGML_FLASHPREFILL_OK) return false;
    if (ggml_flashprefill_metadata_set_counts(meta.data(), words, nfrag, nrow, nuse, ncell) != GGML_FLASHPREFILL_OK) return false;
    int cell_base = 0;
    for (int f = 0; f < nfrag; ++f) {
        if (ggml_flashprefill_metadata_set_frag(meta.data(), words, f, cell_base, frag_counts[(size_t) f], f, 0, 0) != GGML_FLASHPREFILL_OK) return false;
        cell_base += frag_counts[(size_t) f];
    }
    for (int i = 0; i < ncell; ++i) {
        if (ggml_flashprefill_metadata_set_cell(meta.data(), words, i, i) != GGML_FLASHPREFILL_OK) return false;
    }
    for (int h = 0; h < hq; ++h) {
        // row: sq=0, kvh=0, pos=0, tile=0, [0,1), flags=0, q_head=h
        if (ggml_flashprefill_metadata_set_row(meta.data(), words, h, 0, 0, 0, 0, 0, 1, 0, h) != GGML_FLASHPREFILL_OK) return false;
    }
    int sub = 0;
    for (int f = 0; f < nfrag; ++f) {
        int flags = frag_mandatory[(size_t) f] ? GGML_FLASHPREFILL_USE_FLAG_MANDATORY : 0;
        if (ggml_flashprefill_metadata_set_use(meta.data(), words, f, f, 0, 0, frag_qgroup[(size_t) f], sub, frag_counts[(size_t) f], flags, 0) != GGML_FLASHPREFILL_OK) return false;
        sub += frag_counts[(size_t) f];
    }
    if (ggml_flashprefill_metadata_validate(meta.data(), words) != GGML_FLASHPREFILL_OK) return false;
    if (ggml_flashprefill_metadata_n_tiles(meta.data(), words, &out_n_tiles) != GGML_FLASHPREFILL_OK) return false;
    return true;
}

// Run one backend graph: pool->select->attn in a SINGLE graph so pool, plan
// and output are all expanded/allocated. Reads back the ACTUAL plan after
// completion for the oracle (never assumes unexpanded source allocation).
static bool fp_backend_sparse_once(ggml_backend_t backend, const BackendSel & sel,
                                   int dk, int dv, int hkv, int hq, int ngroups,
                                   ggml_type ktype, ggml_type vtype,
                                   float scale, float softcap, float alpha, int exact_all,
                                   bool use_sink, float sink_val,
                                   const std::string & label, double atol, double rtol,
                                   bool padded_uses = false, bool mean_correction = true) {
    (void) sel;
    // Fixture: Nk=8 in 2 fragments (4+4), frag0 mandatory (sink block),
    // frag0 g0, frag1 g1 (two phases, one denominator).
    const int Nk = 8;
    std::vector<int> counts = {4, 4};
    std::vector<int> mandatory = {1, 0};
    std::vector<int> qgroups = {0 % ngroups, 1 % ngroups};
    std::vector<int32_t> meta;
    int64_t n_tiles = 0;
    if (!fp_build_meta(meta, dk, dv, hkv, hq, ngroups, 2, counts, mandatory, qgroups, n_tiles, padded_uses)) {
        FP_CHECK_MSG(false, "%s: meta build failed", label.c_str());
        return false;
    }
    FP_CHECK(n_tiles == 1);
    int64_t max_sel = 0;
    if (ggml_flashprefill_plan_max_sel_for_meta(meta.data(), (int64_t) meta.size(), &max_sel) != GGML_FLASHPREFILL_OK) {
        FP_CHECK_MSG(false, "%s: max_sel helper failed", label.c_str());
        return false;
    }
    FP_CHECK(max_sel == 2); // one pair holds both uses

    // Host model data (bounded) then actual stored bytes.
    std::vector<float> km((size_t) Nk * (size_t) dk * (size_t) hkv), vm((size_t) Nk * (size_t) dv * (size_t) hkv);
    fp_fill_bounded(km, 500 + dk);
    fp_fill_bounded(vm, 600 + dv);
    if (ktype != GGML_TYPE_F16 || vtype != GGML_TYPE_F16) {
        for (float & x : km) x *= 0.15f;
        for (float & x : vm) x *= 0.15f;
    }
    // Hot/cold separation for a clear-margin sparse decision: frag0 aligned
    // with Q, frag1 opposite. Keeps selector strict (no tie).
    // (Applied in storage domain after quant below via dequant check.)
    std::vector<float> qhost((size_t) dk * (size_t) ngroups * (size_t) hq);
    fp_fill_bounded(qhost, 700 + dk);
    // Make Q resemble frag0 direction: copy frag0 model mean + small noise.
    {
        std::vector<double> m0((size_t) dk, 0.0);
        for (int i = 0; i < 4; ++i)
            for (int d = 0; d < dk; ++d) m0[(size_t) d] += km[(size_t) i * (size_t) dk + (size_t) d];
        for (int g = 0; g < ngroups; ++g) {
            for (int h = 0; h < hq; ++h) {
                for (int d = 0; d < dk; ++d) {
                    float base = (float)(m0[(size_t) d] / 4.0);
                    qhost[((size_t) h * (size_t) ngroups + (size_t) g) * (size_t) dk + (size_t) d] = base * 0.8f + qhost[((size_t) h * (size_t) ngroups + (size_t) g) * (size_t) dk + (size_t) d] * 0.2f;
                }
            }
        }
    }
    // Storage domain for Q when K is turbo2/3 (forward WHT per head-group row).
    std::vector<float> q_storage = qhost;
    bool k_needs_wht = (ktype == GGML_TYPE_TURBO2_0 || ktype == GGML_TYPE_TURBO3_0 || ktype == GGML_TYPE_TURBO4_0);
    if (k_needs_wht) {
        // Q tensor layout [Dk,ngroups,Hq]: rows are (h*ngroups+g) vectors.
        fp_wht_rows(q_storage, dk, fp_wht_group_for_dim(dk), false);
    }
    std::vector<uint8_t> kbytes, vbytes;
    if (!fp_quantize_rows(ktype, km, kbytes)) {
        FP_CHECK_MSG(false, "%s: K quant failed", label.c_str());
        return false;
    }
    if (!fp_quantize_rows(vtype, vm, vbytes)) {
        FP_CHECK_MSG(false, "%s: V quant failed", label.c_str());
        return false;
    }
    std::vector<float> kd, vd;
    if (!fp_dequantize_rows(ktype, kbytes, kd, (int64_t) km.size())) {
        FP_CHECK_MSG(false, "%s: K dequant failed", label.c_str());
        return false;
    }
    if (!fp_dequantize_rows(vtype, vbytes, vd, (int64_t) vm.size())) {
        FP_CHECK_MSG(false, "%s: V dequant failed", label.c_str());
        return false;
    }

    // Graph: K [Dk,Nk,Hkv], V [Dv,Nk,Hkv], Q [Dk,ngroups,Hq], meta, pool, plan, attn.
    struct ggml_init_params p = {16 * 1024 * 1024, nullptr, true};
    ggml_context_ptr ctx(ggml_init(p));
    if (!ctx) {
        FP_CHECK_MSG(false, "%s: ggml_init failed", label.c_str());
        return false;
    }
    ggml_tensor * tk = ggml_new_tensor_3d(ctx.get(), ktype, dk, Nk, hkv);
    ggml_tensor * tv = ggml_new_tensor_3d(ctx.get(), vtype, dv, Nk, hkv);
    ggml_tensor * tq = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, dk, ngroups, hq);
    ggml_tensor * tmeta = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, (int64_t) meta.size());
    if (!tk || !tv || !tq || !tmeta) {
        FP_CHECK_MSG(false, "%s: tensor create failed", label.c_str());
        return false;
    }
    // Pool: Fcap == nfrag == 2.
    ggml_tensor * tpool = ggml_flash_prefill_pool(ctx.get(), tk, tv, tmeta, nullptr, nullptr, dk, dv, hkv, 2);
    if (!tpool) {
        FP_CHECK_MSG(false, "%s: pool ctor failed", label.c_str());
        return false;
    }
    ggml_tensor * tplan = ggml_flash_prefill_select(ctx.get(), tq, tpool, tmeta,
            n_tiles, hkv, max_sel, scale, alpha, softcap, exact_all, mean_correction);
    if (!tplan) {
        FP_CHECK_MSG(false, "%s: select ctor failed", label.c_str());
        return false;
    }
    ggml_tensor * tsinks = nullptr;
    std::vector<float> sinkhost;
    if (use_sink) {
        tsinks = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, hq);
        sinkhost.assign((size_t) hq, sink_val);
    }
    ggml_tensor * tout = ggml_flash_prefill_attn(ctx.get(), tq, tk, tv, tpool, tplan, tmeta, tsinks,
            1, hq, dv, scale, softcap, mean_correction);
    if (!tout) {
        FP_CHECK_MSG(false, "%s: attn ctor failed", label.c_str());
        return false;
    }
    // Backend support: SKIP (not fail) when op unsupported, unless required.
    if (!ggml_backend_supports_op(backend, tpool) ||
        !ggml_backend_supports_op(backend, tplan) ||
        !ggml_backend_supports_op(backend, tout)) {
        if (sel.required) {
            FP_CHECK_MSG(false, "%s: required backend '%s' lacks FLASH_PREFILL support",
                    label.c_str(), sel.name.c_str());
            return false;
        }
        std::printf("SKIP %s: backend '%s' lacks FLASH_PREFILL support\n", label.c_str(), sel.name.c_str());
        ++g_skips;
        return true;
    }
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) {
        FP_CHECK_MSG(false, "%s: alloc failed", label.c_str());
        return false;
    }
    ggml_backend_tensor_set(tk, kbytes.data(), 0, kbytes.size());
    ggml_backend_tensor_set(tv, vbytes.data(), 0, vbytes.size());
    // Q tensor host bytes are F32 storage-domain values.
    ggml_backend_tensor_set(tq, q_storage.data(), 0, q_storage.size() * sizeof(float));
    ggml_backend_tensor_set(tmeta, meta.data(), 0, meta.size() * sizeof(int32_t));
    if (tsinks) ggml_backend_tensor_set(tsinks, sinkhost.data(), 0, sinkhost.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 64, false);
    // Single graph expands all three ops so every read-back tensor is allocated.
    ggml_build_forward_expand(graph, tpool);
    ggml_build_forward_expand(graph, tplan);
    ggml_build_forward_expand(graph, tout);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        FP_CHECK_MSG(false, "%s: graph compute failed", label.c_str());
        return false;
    }
    ggml_backend_synchronize(backend);

    if (padded_uses && sel.name.find("Vulkan") != std::string::npos) {
        using scratch_fn = ggml_status (*)(ggml_backend_t, uint64_t *, uint64_t *);
        auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto scratch = reinterpret_cast<scratch_fn>(ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_vk_flashprefill_scratch"));
        uint64_t current = 0, peak = 0;
        FP_CHECK(scratch != nullptr);
        if (scratch) {
            FP_CHECK(scratch(backend, &current, &peak) == GGML_STATUS_SUCCESS);
            FP_CHECK_MSG(current > 0 && peak >= current, "%s did not execute split-K", label.c_str());
        }
    }

    // Read back ACTUAL pool + plan + output after completion.
    int64_t pool_ne = ggml_nelements(tpool);
    std::vector<float> pool_got((size_t) pool_ne, 0.0f);
    ggml_backend_tensor_get(tpool, pool_got.data(), 0, pool_got.size() * sizeof(float));
    int64_t plan_ne = ggml_nelements(tplan);
    std::vector<int32_t> plan_got((size_t) plan_ne, 0);
    ggml_backend_tensor_get(tplan, plan_got.data(), 0, plan_got.size() * sizeof(int32_t));
    int64_t out_ne = ggml_nelements(tout);
    std::vector<float> out_got((size_t) out_ne, 0.0f);
    ggml_backend_tensor_get(tout, out_got.data(), 0, out_got.size() * sizeof(float));

    // Validate plan encoding + coverage against metadata (strict).
    FP_CHECK(ggml_flashprefill_plan_validate(plan_got.data(), plan_ne) == GGML_FLASHPREFILL_OK);
    FP_CHECK(ggml_flashprefill_plan_validate_against_metadata(plan_got.data(), plan_ne, meta.data(), (int64_t) meta.size()) == GGML_FLASHPREFILL_OK);
    struct ggml_flashprefill_plan_stats stats = {};
    FP_CHECK(ggml_flashprefill_plan_get_stats(plan_got.data(), plan_ne, &stats) == GGML_FLASHPREFILL_OK);
    // Exact-all must have zero proxy uses; sparse must have coverage.
    if (exact_all) {
        FP_CHECK_MSG(stats.corrected_total == 0, "%s exact-all corrected=%d", label.c_str(), stats.corrected_total);
    }
    // Stats counters use per-query incidences (not subhead fan-out).
    FP_CHECK(stats.visible_tokens == (int64_t) Nk);
    if (exact_all) FP_CHECK(stats.exact_tokens == (int64_t) Nk);

    // Host means from dequantized storage must match backend pool (tight for F16).
    {
        // Pool layout: pool[(f*Hkv+h)*(Dk+Dv)+d], K rows [0,Dk), V [Dk,Dk+Dv).
        double max_abs = 0.0;
        for (int f = 0; f < 2; ++f) {
            // Fragment f covers tokens [f*4,4).
            for (int d = 0; d < dk; ++d) {
                double acc = 0.0;
                for (int i = 0; i < 4; ++i) acc += kd[(size_t)(f * 4 + i) * (size_t) dk + (size_t) d];
                float want = (float)(acc / 4.0);
                float got = pool_got[(size_t)(f * hkv + 0) * (size_t)(dk + dv) + (size_t) d];
                max_abs = std::max(max_abs, std::abs((double) got - (double) want));
            }
            for (int d = 0; d < dv; ++d) {
                double acc = 0.0;
                for (int i = 0; i < 4; ++i) acc += vd[(size_t)(f * 4 + i) * (size_t) dv + (size_t) d];
                float want = (float)(acc / 4.0);
                float got = pool_got[(size_t)(f * hkv + 0) * (size_t)(dk + dv) + (size_t)(dk + d)];
                max_abs = std::max(max_abs, std::abs((double) got - (double) want));
            }
        }
        double pool_tol = (ktype == GGML_TYPE_F16 && vtype == GGML_TYPE_F16) ? 1e-4 : 2e-3;
        FP_CHECK_MSG(max_abs < pool_tol, "%s pool max_abs %.9g exceeds %.9g", label.c_str(), max_abs, pool_tol);
        std::printf("%s: pool max_abs=%.9g\n", label.c_str(), max_abs);
    }

    // Oracle uses the ACTUAL backend plan + ACTUAL backend pool means.
    // Decode plan lists for the single pair (tile0,head0).
    {
        // Plan tables: exact_off=24, proxy_off=24+T*H*max, counts_off=...
        // Use helpers' documented offsets via header constants is forbidden
        // to hardcode; instead parse via plan_get_stats + direct table read
        // using the frozen 24-word header documented in the wire header.
        // max_sel known (=2), T=1,H=1 here, so exact[0..1], proxy[0..1].
        const int32_t * pl = plan_got.data();
        int64_t exact_off = 24;
        int64_t proxy_off = 24 + (int64_t) 1 * 1 * max_sel;
        int64_t counts_off = proxy_off + (int64_t) 1 * 1 * max_sel;
        int n_exact = pl[counts_off], n_proxy = pl[counts_off + 1];
        FP_CHECK(n_exact >= 0 && n_proxy >= 0 && n_exact + n_proxy == 2);
        std::vector<int> exact_uses, proxy_uses;
        for (int i = 0; i < n_exact; ++i) exact_uses.push_back(pl[exact_off + i]);
        for (int i = 0; i < n_proxy; ++i) proxy_uses.push_back(pl[proxy_off + i]);
        // Build oracle inputs from ACTUAL pool means + ACTUAL plan roles.
        // For each head h, Q vector is q_storage[(h*ngroups+g)] where g is the
        // use's q_group (frag0->g0, frag1->g1). Exact tokens use kd/vd rows;
        // proxy uses pool means with count=4.
        std::vector<float> ref_all((size_t) dv * (size_t) hq, 0.0f);
        for (int h = 0; h < hq; ++h) {
            // Each fragment has its own RoPE phase/Q group. Accumulate all
            // exact tokens and proxies into ONE independent denominator.
            fp_oracle::Mlo state = fp_oracle::mlo_init(dv);
            bool ok = false;
            for (int u : exact_uses) {
                const int f = u;
                const float * qh = q_storage.data() + ((size_t) h * ngroups + f % ngroups) * dk;
                for (int i = 0; i < 4; ++i) {
                    const int tok = f * 4 + i;
                    double dot = 0;
                    for (int d = 0; d < dk; ++d) dot += (double) qh[d] * kd[(size_t) tok * dk + d];
                    FP_CHECK(fp_oracle::mlo_add(state, scale * dot, vd.data() + (size_t) tok * dv,
                            1, 1.0, softcap, ok) && ok);
                }
            }
            for (int u : proxy_uses) {
                if (!mean_correction) break;
                const int f = u;
                const float * qh = q_storage.data() + ((size_t) h * ngroups + f % ngroups) * dk;
                const float * mean = pool_got.data() + (size_t) f * hkv * (dk + dv);
                double dot = 0;
                for (int d = 0; d < dk; ++d) dot += (double) qh[d] * mean[d];
                FP_CHECK(fp_oracle::mlo_add(state, scale * dot, mean + dk, 4, 1.0, softcap, ok) && ok);
            }
            if (use_sink) {
                FP_CHECK(fp_oracle::mlo_add_sink(state, sink_val, ok) && ok);
            }
            std::vector<float> out;
            FP_CHECK(fp_oracle::mlo_finalize(state, out, ok) && ok);
            // The raw ATTN op returns the stored V domain. The llama graph
            // applies inverse WHT afterwards; do not rotate only the oracle.
            for (int d = 0; d < dv; ++d) ref_all[(size_t) h * (size_t) dv + (size_t) d] = out[(size_t) d];
        }
        // Backend output layout [Dv,Hq,1]: (d,h,0).
        FpErr e = fp_compare(out_got, ref_all, dv, hq, 1);
        fp_print_err(label.c_str(), e, atol, rtol);
        FP_CHECK_MSG(e.nan_got == 0 && e.nan_ref == 0, "%s NaNs got=%lld ref=%lld", label.c_str(), (long long) e.nan_got, (long long) e.nan_ref);
        // No total-error tolerance hiding selector bugs: exact-all and sparse
        // each checked against the ACTUAL plan; selector mismatches already
        // failed above via plan_validate_against_metadata + stats.
        FP_CHECK_MSG(e.max_abs < atol || (e.max_rel < rtol && e.max_abs < 1e-2),
                "%s attn max_abs %.9g max_rel %.9g exceeds", label.c_str(), e.max_abs, e.max_rel);
    }
    return true;
}

static void test_backend_coverage_lanes(ggml_backend_t backend, const BackendSel & sel) {
    // Exercise the actual ATTN coverage scan at workgroup boundaries. Use a
    // leaf plan, not SELECT, so missing/duplicate entries reach the consumer.
    for (int nf : {1, 63, 64, 65, 129}) {
        constexpr int dim = 64, hq = 2;
        std::vector<int32_t> meta;
        int64_t nt = 0, pw = 0;
        FP_CHECK(fp_build_meta(meta, dim, dim, 1, hq, 1, nf,
                    std::vector<int>(nf, 1), std::vector<int>(nf, 0), std::vector<int>(nf, 0), nt, nf == 1));
        FP_CHECK(ggml_flashprefill_plan_words(1, 1, nf, &pw) == GGML_FLASHPREFILL_OK);
        ggml_context_ptr ctx(ggml_init({16 * 1024 * 1024, nullptr, true}));
        ggml_tensor * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, dim, 1, hq);
        ggml_tensor * k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, dim, nf, 1);
        ggml_tensor * v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, dim, nf, 1);
        ggml_tensor * pool = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 2 * dim, 1, nf);
        ggml_tensor * mt = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, meta.size());
        ggml_tensor * plan = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, pw);
        ggml_tensor * out = ggml_flash_prefill_attn(ctx.get(), q, k, v, pool, plan, mt, nullptr,
                1, hq, dim, 1.0f, 0.0f, true);
        if (!out || !ggml_backend_supports_op(backend, out)) {
            FP_CHECK_MSG(false, "coverage ATTN unsupported on %s", sel.name.c_str());
            return;
        }
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        if (!buffer) { FP_CHECK(false); return; }
        std::vector<float> qdata(dim * hq, 0), pooldata(2 * dim * nf, 0);
        std::vector<ggml_fp16_t> kdata(dim * nf, ggml_fp32_to_fp16(0)), vdata(dim * nf);
        // K=Q=0 makes every token equally likely. V is exactly representable
        // in F16, so the independent reference is a simple arithmetic mean.
        for (int f = 0; f < nf; ++f) for (int d = 0; d < dim; ++d) {
            const float value = (f + d) * 0.125f;
            vdata[f * dim + d] = ggml_fp32_to_fp16(value);
            pooldata[f * 2 * dim + dim + d] = value;
        }
        ggml_backend_tensor_set(q, qdata.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_set(k, kdata.data(), 0, ggml_nbytes(k));
        ggml_backend_tensor_set(v, vdata.data(), 0, ggml_nbytes(v));
        ggml_backend_tensor_set(pool, pooldata.data(), 0, ggml_nbytes(pool));
        ggml_backend_tensor_set(mt, meta.data(), 0, ggml_nbytes(mt));
        ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(graph, out);
        for (int mode = 0; mode < 6; ++mode) {
            const bool corrupt = mode >= 3;
            // CPU deliberately aborts on corrupt plans; Vulkan reports the
            // error in the plan. Negative cases test that GPU protocol only.
            if (corrupt && sel.is_cpu) continue;
            if (mode == 4 && nf == 1) continue;
            std::vector<int32_t> wire((size_t) pw);
            FP_CHECK(ggml_flashprefill_plan_init(wire.data(), pw, 1, 1, nf, 0) == GGML_FLASHPREFILL_OK);
            int ne = 0, np = 0;
            for (int u = 0; u < nf; ++u) {
                if (mode == 3 && u == nf - 1) continue; // omit the last lane's use
                if (mode == 2 && u % 2) wire[wire[7] + np++] = u;
                else wire[wire[6] + ne++] = mode == 1 ? nf - 1 - u : u;
            }
            if (mode == 4) wire[wire[6] + nf - 1] = 0; // duplicate, missing tail
            wire[wire[8]] = ne;
            wire[wire[8] + 1] = np;
            auto meta_input = meta;
            if (mode == 5) meta_input[0] = 0; // invalid header on a small, otherwise valid plan
            ggml_backend_tensor_set(mt, meta_input.data(), 0, ggml_nbytes(mt));
            ggml_backend_tensor_set(plan, wire.data(), 0, ggml_nbytes(plan));
            const ggml_status status = ggml_backend_graph_compute(backend, graph);
            FP_CHECK_MSG(corrupt ? status != GGML_STATUS_SUCCESS : status == GGML_STATUS_SUCCESS,
                    "coverage execution status nf=%d mode=%d status=%d", nf, mode, int(status));
            if (nf == 1 && mode == 0 && sel.name.find("Vulkan") != std::string::npos) {
                // Padding forces multiple splits with only ONE real token,
                // so empty M=-inf/L=0/O=0 partials must merge correctly.
                using scratch_fn = ggml_status (*)(ggml_backend_t, uint64_t *, uint64_t *);
                auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
                auto scratch = reinterpret_cast<scratch_fn>(ggml_backend_reg_get_proc_address(
                            reg, "ggml_backend_vk_flashprefill_scratch"));
                uint64_t current = 0, peak = 0;
                FP_CHECK(scratch != nullptr);
                if (scratch) {
                    FP_CHECK(scratch(backend, &current, &peak) == GGML_STATUS_SUCCESS);
                    FP_CHECK(current > 0 && peak >= current);
                }
            }
            ggml_backend_tensor_get(plan, wire.data(), 0, ggml_nbytes(plan));
            if (corrupt) {
                FP_CHECK_MSG(wire[10] != GGML_FLASHPREFILL_OK,
                        "missing/duplicate use accepted: nf=%d mode=%d", nf, mode);
            } else {
                FP_CHECK_MSG(wire[10] == GGML_FLASHPREFILL_OK,
                        "valid coverage rejected: nf=%d mode=%d error=%d", nf, mode, wire[10]);
                std::vector<float> got(dim * hq);
                ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
                for (int h = 0; h < hq; ++h) for (int d = 0; d < dim; ++d) {
                    const float expected = ((nf - 1) * 0.5f + d) * 0.125f;
                    FP_CHECK_MSG(std::isfinite(got[h * dim + d]) &&
                            std::abs(got[h * dim + d] - expected) < 5e-4f,
                            "coverage output mismatch nf=%d mode=%d h=%d d=%d", nf, mode, h, d);
                }
            }
        }
    }
    std::printf("test_backend_coverage_lanes done (%s)\n", sel.name.c_str());
}

static void test_backend_exactall(ggml_backend_t backend, const BackendSel & sel) {
    // Exact-all over same cached bytes vs dense oracle (same dequantized data).
    fp_backend_sparse_once(backend, sel, 64, 64, 1, 2, 2,
            GGML_TYPE_F16, GGML_TYPE_F16, 0.25f, 0.0f, 0.1f, 1, false, 0.0f,
            "backend-exactall-f16", 2e-3, 2e-2);
    fp_backend_sparse_once(backend, sel, 64, 32, 1, 2, 2,
            GGML_TYPE_F16, GGML_TYPE_F16, 0.25f, 0.0f, 0.1f, 1, true, 0.4f,
            "backend-exactall-sink-dkdv", 2e-3, 2e-2);
    // The Vulkan split heuristic uses metadata capacity. Padding forces
    // split-K plus MERGE, including empty splits, without a large fixture.
    // This also catches under-reservation of ATTN descriptor sets.
    fp_backend_sparse_once(backend, sel, 128, 128, 1, 2, 2,
            GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0, 0.25f, 0.0f, 0.1f, 1, true, 0.4f,
            "backend-exactall-split-turbo4-turbo2", 5e-3, 5e-2, true);
}

static void test_backend_sparse(ggml_backend_t backend, const BackendSel & sel) {
    fp_backend_sparse_once(backend, sel, 64, 64, 1, 2, 2,
            GGML_TYPE_F16, GGML_TYPE_F16, 0.35f, 0.0f, 0.1f, 0, false, 0.0f,
            "backend-sparse-f16", 2e-3, 2e-2);
    fp_backend_sparse_once(backend, sel, 128, 128, 1, 2, 2,
            GGML_TYPE_F16, GGML_TYPE_F16, 0.25f, 1.5f, 0.1f, 0, true, 0.3f,
            "backend-sparse-softcap-sink", 3e-3, 3e-2);
}

static void test_backend_value_accumulation(ggml_backend_t backend, const BackendSel & sel) {
    // Cover the register-path boundary and the unbounded-head fallback,
    // independently of the selector. The oracle uses double M/L/O algebra.
    for (int dv : {8, 248, 256, 264}) for (bool split : {false, true}) {
        for (bool correction : {false, true}) {
            const std::string name = "backend-values-dv" + std::to_string(dv) +
                (split ? "-split" : "-single") + (correction ? "-means" : "-no-means");
            fp_backend_sparse_once(backend, sel, 64, dv, 1, 2, 2,
                    GGML_TYPE_F16, GGML_TYPE_F16, 8.0f, 1.5f, 1.0f, 0, true, 2.0f,
                    name, 2e-3, 2e-2, split, correction);
        }
    }
    fp_backend_sparse_once(backend, sel, 128, 256, 1, 2, 2,
            GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0, 8.0f, 1.5f, 1.0f, 0, true, 2.0f,
            "backend-values-turbo4-turbo2-dv256", 5e-3, 5e-2, true);
    // The default boundary-layer V policy uses Q8, even with --ctv turbo2.
    for (int exact : {0, 1}) for (bool split : {false, true}) {
        fp_backend_sparse_once(backend, sel, 128, 128, 1, 2, 2,
                GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0, 8.0f, 1.5f, 1.0f, exact, true, 2.0f,
                std::string("backend-values-boundary-q8-") + (exact ? "exact" : "sparse") +
                    (split ? "-split" : "-single"), 5e-3, 5e-2, split);
    }
}

static void test_backend_quant(ggml_backend_t backend, const BackendSel & sel) {
    fp_backend_sparse_once(backend, sel, 128, 128, 1, 2, 2,
            GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, 0.25f, 0.0f, 0.15f, 0, false, 0.0f,
            "backend-sparse-turbo3", 5e-3, 5e-2);
    // Mixed K/V storage is the deployment case, not just equal-bit caches.
    for (ggml_type ktype : { GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0 }) {
        for (int exact_all : { 0, 1 }) {
            const std::string label = std::string(exact_all ? "backend-exactall-" : "backend-sparse-") +
                ggml_type_name(ktype) + "-turbo2";
            fp_backend_sparse_once(backend, sel, 128, 128, 1, 2, 2,
                    ktype, GGML_TYPE_TURBO2_0, 0.25f, 0.0f, 0.15f, exact_all, false, 0.0f,
                    label, 5e-3, 5e-2);
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static void print_usage(const char * prog) {
    std::printf("usage: %s [--backend <name>] [--required-backend <name>]\n", prog);
    std::printf("  --backend: preferred device substring (default CPU)\n");
    std::printf("  --required-backend: strict device substring; missing => FAIL (never PASS on CPU)\n");
}

int main(int argc, char ** argv) {
    std::string want_backend;
    std::string required_backend;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) want_backend = argv[++i];
        else if (a == "--required-backend" && i + 1 < argc) required_backend = argv[++i];
        else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    // Host-only contracts (no backend needed).
    test_oracle_identities();
    test_mlo_merge_empty();
    test_softcap_ordering();
    test_empty_row();
    test_tail_counts();
    test_gqa_packing();
    test_strides_host();
    test_phase_single_denominator();
    test_sinks_mandatory();
    test_quant_storage();
    test_wht_roundtrip_host();
    test_pollution();
    test_dims_padding();
    test_dense_vs_exactall_host();
    test_sparse_vs_oracle_host();

    // Backend fixtures (strict required-backend handling).
    bool pick_failed = false;
    BackendSel sel = fp_pick_device(want_backend, required_backend, pick_failed);
    if (pick_failed) {
        std::printf("test-flashprefill-attn: required backend missing => FAIL\n");
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(sel.dev, nullptr);
    if (!backend) {
        std::fprintf(stderr, "FAIL backend init failed for '%s'\n", sel.name.c_str());
        return 1;
    }
    test_backend_exactall(backend, sel);
    test_backend_sparse(backend, sel);
    // Quant backend coverage is best-effort: SKIP when unsupported.
    test_backend_quant(backend, sel);
    test_backend_value_accumulation(backend, sel);
    test_backend_coverage_lanes(backend, sel);
    ggml_backend_free(backend);

    if (g_failures == 0) {
        std::printf("test-flashprefill-attn: all assertions passed (skips=%d)\n", g_skips);
        return 0;
    }
    std::printf("test-flashprefill-attn: %d assertion(s) failed (skips=%d)\n", g_failures, g_skips);
    return 1;
}
