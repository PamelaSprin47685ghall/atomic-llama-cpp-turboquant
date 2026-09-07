// test-flashprefill-select.cpp — selector behavior + SELECT plan contract.
//
// Covers the FlashPrefill V2 tile-energy selector (PREFILL.md section 3.2)
// and the GGML SELECT wire contract (ggml/include/ggml-flashprefill.h):
//
//   - PackGQA mapping for G=1/2/4/6/8, BM both dividing and NOT dividing G,
//     including tiles that split a GQA group and partial tail tiles.
//   - Tile-energy selection with a GLOBAL max over all (row, candidate)
//     pairs (not per-head top-k, not mean-logit averaging).
//   - Threshold ties are inclusive (>=): equal energies are all kept, even
//     at alpha=1.
//   - Overflow stability for extreme but finite logits (max-subtraction,
//     no inf/nan, correct keep sets).
//   - Mandatory sets: sink prefix blocks, local window blocks, partial
//     (non-full) fragments, and DENSE_FORCE dense-tail rows; per-head
//     hotspots that deliberately differ between KV heads.
//   - Selector legality invariant (frozen single policy): pair_valid says
//     which (row, candidate) pairs may legally see a whole mean; partial
//     candidates (cand_full==0) are NEVER read nor scored (entire column
//     excluded) and forced exact, so pollutant values at unscored slots
//     cannot shift M/Smax/keep sets. Full candidates score valid pairs
//     only; all-zero/all-excluded is legal (all exact, capacity enforced);
//     mask entries are strictly 0/1. Every ref call passes an explicit
//     mask (all-ones, zeroed partial columns, or the all-zero boundary).
//     Production backend SELECT enforces the same rule from descriptors.
//   - Multi-phase (q_group) gathering: each (row, candidate) uses the Q
//     vector of the row's head under the candidate's effective phase.
//   - Metadata legality across token counts 0/1/63/64/65/127/128/129 with
//     padding excluded, duplicate/orphan/dense rejection, plan role
//     rejections (partial/mandatory must never proxy), and
//     capacity-overflow errors that never truncate.
//   - exact-all (every legal use exact, proxy lists explicitly empty) and
//     correction-off (complement still explicitly listed) plan contracts.
//   - Backend SELECT parity (CPU always; --backend NAME for GPU) with
//     --required-backend turning a missing/unsupported backend into a
//     failure instead of a silent CPU fallback or quiet pass.
//
// Method: every expected selected set is computed by an independent
// test-local oracle (sel_oracle, plain double math, never calls the
// ggml_flashprefill_* implementation under test). The oracle is compared
// against BOTH the ggml reference (ggml_flashprefill_ref_select) AND the
// backend SELECT op output. Fixtures use only the frozen helper API to
// size/fill/validate tensors (metadata_words/init/set_*, plan_words/
// plan_max_sel_for_meta/init/set_*/accumulate_stats/get_stats); stats come
// from get_stats and plan tables are decoded through the plan's own
// self-describing offset words, never hardcoded counter indices.
//
// Self-contained: no shared fixture file. Math tests always run; backend
// tests run on CPU by default and additionally/alternatively on --backend.
//
// Registered in tests/CMakeLists.txt; GPU checks are explicitly requested.

#include "ggml-flashprefill.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

static int g_failures = 0;

#define SELECT_CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

#define SELECT_CHECK_MSG(cond, ...) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n"); \
        ++g_failures; \
    } \
} while (0)

// ---------------------------------------------------------------------------
// Independent oracle: tile-energy selector in plain double math.
// Never calls ggml_flashprefill_*; this is the spec the implementation and
// the backend op are both checked against.
//
// Legality rule (frozen selector invariant): candidates with cand_full==0
// are partial-boundary uses whose whole-fragment mean may include
// future/invisible tokens for some tile rows. They are EXCLUDED from the
// energy (no contribution to global M, per-fragment S, or smax) and forced
// exact. Full candidates (including mandatory sink/local ones, whose whole
// means are legal) score normally. This makes pollutant isolation
// structural: foreign values in a partial mean cannot shift any legal
// keep set.
// ---------------------------------------------------------------------------
namespace sel_oracle {

static constexpr int OK = 0;
static constexpr int OVER_CAPACITY = 1;
static constexpr int BAD_MASK = 2;

static double softcap_apply(double x, double cap) {
    if (!(cap > 0.0)) {
        return x;
    }
    return cap * std::tanh(x / cap);
}

struct select_result {
    int rc = OK;
    std::vector<int32_t> exact; // candidate positions, increasing
    std::vector<int32_t> proxy; // candidate positions, increasing
    double m = 0.0;             // global max over scored (full-only) pairs
    double smax = 0.0;
    std::vector<double> energies; // per candidate; -1 for excluded partials
};

// qpair[(r*n_cand+j)*stride_qp .. +dk]: Q of packed row r under candidate
// j's effective phase. kbar[j*stride_kb .. +dk]: pooled K mean of candidate
// j. pair_valid[r*n_cand+j] = 1 iff row r may legally see candidate j's
// whole mean; strict 0/1 (else BAD_MASK). Single rule: a pair is scored
// iff the candidate is full AND the mask is 1; scored pairs alone feed M
// and S. Partial candidates (cand_full==0) mask their entire column out of
// the energy and are forced exact, so pollutant values at unscored slots
// (invalid mask or partial column) are never read and cannot shift M,
// smax, or any keep decision. Unscored candidates classify exact.
static select_result select_tile(
        const float * qpair, int64_t stride_qp, int32_t n_rows,
        const float * kbar, int64_t stride_kb,
        const int32_t * cand_mandatory, const int32_t * cand_full,
        const int32_t * pair_valid, int32_t n_cand,
        int32_t dk, double scale, double softcap, double alpha, int32_t exact_all,
        int32_t cap_exact, int32_t cap_proxy) {
    select_result out;
    out.energies.assign((size_t) n_cand, -1.0);
    if (!pair_valid) {
        out.rc = BAD_MASK;
        return out;
    }
    for (int64_t k = 0; k < (int64_t) n_rows * n_cand; ++k) {
        if (pair_valid[k] != 0 && pair_valid[k] != 1) {
            out.rc = BAD_MASK;
            out.exact.clear();
            out.proxy.clear();
            return out;
        }
    }
    const auto scored = [&](int32_t r, int32_t j) {
        return cand_full[j] && pair_valid[(int64_t) r * n_cand + j];
    };
    double m = -INFINITY;
    for (int32_t r = 0; r < n_rows; ++r) {
        for (int32_t j = 0; j < n_cand; ++j) {
            if (!scored(r, j)) {
                continue; // never read: no float load, no max effect
            }
            const float * qr = qpair + ((int64_t) r * n_cand + j) * stride_qp;
            const float * kb = kbar + (int64_t) j * stride_kb;
            double dot = 0.0;
            for (int32_t d = 0; d < dk; ++d) {
                dot += (double) qr[d] * (double) kb[d];
            }
            const double z = softcap_apply(scale * dot, softcap);
            if (z > m) {
                m = z;
            }
        }
    }
    out.m = m;
    double smax = 0.0;
    std::vector<char> has_valid((size_t) n_cand, 0);
    for (int32_t j = 0; j < n_cand; ++j) {
        if (!cand_full[j]) {
            continue; // partial column: no energy, forced exact below
        }
        const float * kb = kbar + (int64_t) j * stride_kb;
        double s = 0.0;
        bool any = false;
        for (int32_t r = 0; r < n_rows; ++r) {
            if (!pair_valid[(int64_t) r * n_cand + j]) {
                continue;
            }
            any = true;
            const float * qr = qpair + ((int64_t) r * n_cand + j) * stride_qp;
            double dot = 0.0;
            for (int32_t d = 0; d < dk; ++d) {
                dot += (double) qr[d] * (double) kb[d];
            }
            s += std::exp(softcap_apply(scale * dot, softcap) - m);
        }
        if (!any) {
            continue; // fully masked column: unscored, exact below
        }
        has_valid[(size_t) j] = 1;
        out.energies[(size_t) j] = s;
        if (s > smax) {
            smax = s;
        }
    }
    out.smax = smax;
    const double thr = alpha * smax;
    std::vector<int32_t> exact, proxy;
    for (int32_t j = 0; j < n_cand; ++j) {
        if (!cand_full[j] || !has_valid[(size_t) j]) {
            exact.push_back(j); // excluded partial / unscored: always exact
            continue;
        }
        const bool keep = out.energies[(size_t) j] >= thr; // inclusive tie
        if (keep || cand_mandatory[j] || exact_all) {
            exact.push_back(j);
        } else {
            proxy.push_back(j);
        }
    }
    if ((int32_t) exact.size() > cap_exact || (int32_t) proxy.size() > cap_proxy) {
        out.rc = OVER_CAPACITY; // error, never a truncated list
        out.exact.clear();
        out.proxy.clear();
        return out;
    }
    out.exact = exact;
    out.proxy = proxy;
    return out;
}

} // namespace sel_oracle

// ---------------------------------------------------------------------------
// Small diagnostics: specific set comparisons, never bare "nonempty".
// ---------------------------------------------------------------------------
static void report_int_sets(
        const char * what, const std::vector<int32_t> & got, const std::vector<int32_t> & want) {
    std::fprintf(stderr, "  %s mismatch:\n    got  [", what);
    for (size_t i = 0; i < got.size(); ++i) {
        std::fprintf(stderr, "%s%d", i ? "," : "", got[i]);
    }
    std::fprintf(stderr, "] (%zu)\n    want [", got.size());
    for (size_t i = 0; i < want.size(); ++i) {
        std::fprintf(stderr, "%s%d", i ? "," : "", want[i]);
    }
    std::fprintf(stderr, "] (%zu)\n", want.size());
    const size_t n = std::max(got.size(), want.size());
    int shown = 0;
    for (size_t i = 0; i < n && shown < 8; ++i) {
        const int32_t g = i < got.size() ? got[i] : INT32_MIN;
        const int32_t w = i < want.size() ? want[i] : INT32_MIN;
        if (g != w) {
            std::fprintf(stderr, "    slot %zu: got %d want %d\n", i, g, w);
            ++shown;
        }
    }
}

static bool check_int_sets(const char * what,
        const std::vector<int32_t> & got, const std::vector<int32_t> & want) {
    if (got == want) {
        return true;
    }
    std::fprintf(stderr, "FAIL %s:%d: int-set %s differs\n", __FILE__, __LINE__, what);
    report_int_sets(what, got, want);
    ++g_failures;
    return false;
}

// ---------------------------------------------------------------------------
// PackGQA helpers (pure mapping; mirror of the frozen packing rule).
// packed r (position-major within one KV head): pos = r / G, sub = r % G.
// ---------------------------------------------------------------------------
static int64_t packgqa_rows(int64_t n_tokens, int64_t g) { return n_tokens * g; }

static void test_packgqa_mapping() {
    std::puts("--- PackGQA mapping G=1/2/4/6/8, BM dividing and non-dividing ---");
    const int64_t groups[] = {1, 2, 4, 6, 8}; // Nanbeige has GQA ratio 6
    const int64_t bms[] = {1, 2, 3, 4, 5, 6, 7, 8, 16, 100, 127, 128};
    const int64_t token_counts[] = {1, 3, 5, 33};
    for (int64_t g : groups) {
        for (int64_t bm : bms) {
            for (int64_t nt : token_counts) {
                const int64_t npack = packgqa_rows(nt, g);
                const int64_t ntiles = (npack + bm - 1) / bm;
                // Every packed row maps to exactly one (pos, sub, tile) with
                // an exact position-major round trip (never BM/G truncation).
                for (int64_t r = 0; r < npack; ++r) {
                    const int64_t pos = r / g;
                    const int64_t sub = r % g;
                    SELECT_CHECK_MSG(pos >= 0 && pos < nt && sub >= 0 && sub < g,
                            "G=%lld BM=%lld r=%lld maps out of range pos=%lld sub=%lld",
                            (long long) g, (long long) bm, (long long) r, (long long) pos, (long long) sub);
                    SELECT_CHECK_MSG(pos * g + sub == r, "G=%lld pack round trip r=%lld",
                            (long long) g, (long long) r);
                }
                // Starting mid-group means sharing a position with the
                // preceding tile, not necessarily spanning two positions
                // within this tile (BM=1 is an immediate counterexample).
                for (int64_t t = 0; t < ntiles; ++t) {
                    const int64_t start = t * bm;
                    const int64_t end = std::min(start + bm, npack);
                    const bool predicted_split = (t > 0) && ((start % g) != 0);
                    const bool observed_split = start > 0 && start / g == (start - 1) / g;
                    SELECT_CHECK(predicted_split == observed_split);
                    const int64_t positions = (start % g + end - start + g - 1) / g;
                    SELECT_CHECK(positions == (end - 1) / g - start / g + 1);
                }
                // Partial tail tile holds exactly the remainder rows.
                const int64_t tail = npack - (ntiles - 1) * bm;
                SELECT_CHECK_MSG(tail >= 1 && tail <= bm && (ntiles - 1) * bm + tail == npack,
                        "G=%lld BM=%lld nt=%lld tail=%lld", (long long) g, (long long) bm,
                        (long long) nt, (long long) tail);
            }
        }
    }
    // Head mapping inside KV heads: packed sub s of kv head kh belongs to
    // query_head = kh * G + s, and different KV heads are disjoint.
    for (int64_t g : groups) {
        for (int64_t kh = 0; kh < 2; ++kh) {
            for (int64_t s = 0; s < g; ++s) {
                SELECT_CHECK_MSG((kh * g + s) / g == kh, "G=%lld sub/head parent", (long long) g);
            }
        }
    }
    std::puts("PackGQA mapping: PASS");
}

// ---------------------------------------------------------------------------
// ref_select call wrapper (implementation under test).
// Frozen single policy: partial candidates (cand_full==0) are never read
// nor scored (entire column excluded; exact regardless), full candidates
// score valid mask pairs only, all-zero/all-excluded is legal (all exact,
// capacity still enforced), mask entries are strictly 0/1. Production
// backend SELECT derives the same legality from descriptors (partial uses
// are exact-only and excluded from the energy); the rectangular mask
// exists only in the reference. Every ref call below passes an explicit
// mask: all-ones for wholly-legal cases, zeroed partial columns for
// partial-boundary cases.
// ---------------------------------------------------------------------------
struct ref_out {
    int32_t rc = 0;
    std::vector<int32_t> exact; // frag ids, increasing candidate order
    std::vector<int32_t> proxy; // frag ids, increasing candidate order
};

static ref_out call_ref_select(
        const float * qpair, int64_t stride_qp, int32_t n_rows,
        const float * kbar, int64_t stride_kb,
        const std::vector<int32_t> & cand_frag,
        const std::vector<int32_t> & cand_mandatory,
        const std::vector<int32_t> & cand_full,
        const std::vector<int32_t> & pair_valid,
        int32_t dk, float scale, float softcap, float alpha, int32_t exact_all,
        int32_t cap_exact, int32_t cap_proxy) {
    ref_out o;
    const int32_t n_cand = (int32_t) cand_frag.size();
    o.exact.assign((size_t) n_cand, -1);
    o.proxy.assign((size_t) n_cand, -1);
    int32_t n_exact = -1, n_proxy = -1;
    o.rc = ggml_flashprefill_ref_select(qpair, stride_qp, n_rows, kbar, stride_kb,
            cand_frag.data(), cand_mandatory.data(), cand_full.data(), pair_valid.data(), n_cand,
            dk, scale, softcap, alpha, exact_all, cap_exact, cap_proxy,
            o.exact.data(), &n_exact, o.proxy.data(), &n_proxy);
    if (o.rc == GGML_FLASHPREFILL_OK) {
        o.exact.resize((size_t) n_exact);
        o.proxy.resize((size_t) n_proxy);
    } else {
        o.exact.clear();
        o.proxy.clear();
    }
    return o;
}

static bool check_oracle_vs_ref(const char * what,
        const sel_oracle::select_result & oracle, const ref_out & ref,
        const std::vector<int32_t> & cand_frag) {
    // Oracle reports candidate positions; ref reports frag ids. Map first.
    std::vector<int32_t> want_exact, want_proxy;
    for (int32_t p : oracle.exact) {
        want_exact.push_back(cand_frag[(size_t) p]);
    }
    for (int32_t p : oracle.proxy) {
        want_proxy.push_back(cand_frag[(size_t) p]);
    }
    if (oracle.rc != sel_oracle::OK || ref.rc != GGML_FLASHPREFILL_OK) {
        std::fprintf(stderr, "FAIL %s:%d: %s rc oracle=%d ref=%d(%s)\n",
                __FILE__, __LINE__, what, oracle.rc, ref.rc, ggml_flashprefill_strerror(ref.rc));
        ++g_failures;
        return false;
    }
    bool ok = true;
    ok &= check_int_sets(what, ref.exact, want_exact);
    ok &= check_int_sets(what, ref.proxy, want_proxy);
    return ok;
}

// Global-max discriminator: spiky fragment (strong in one row, weak in the
// other) beats a uniformly mediocre fragment under tile-energy, while a
// per-row-top-1-union would also retain a weak candidate.
static void test_energy_global_max_not_head_topk() {
    std::puts("--- energy: global max over rows, not head top-k / mean ---");
    constexpr int32_t dk = 4;
    constexpr int32_t n_rows = 2;
    constexpr int32_t n_cand = 3; // A spiky, B flat, C weak
    // row0 Q = +8*e0, row1 Q = -4*e0; kbarA=+8*e0, kbarB=0, kbarC=-8*e0.
    // scale=1/8: z(A)=(8,-4), z(C)=(-8,4), M=8.
    // S[A]=1+e^-12, S[B]=2e^-8, S[C]=e^-16+e^-4. alpha=.5 keeps A
    // only, unlike the row-top-1 union {A,C}. Symmetric +/-8 queries would
    // give A and C equal energy and cannot discriminate these algorithms.
    std::vector<float> qpair((size_t) n_rows * n_cand * dk, 0.0f);
    std::vector<float> kbar((size_t) n_cand * dk, 0.0f);
    for (int32_t j = 0; j < n_cand; ++j) {
        qpair[((size_t) 0 * n_cand + j) * dk + 0] = 8.0f;
        qpair[((size_t) 1 * n_cand + j) * dk + 0] = -4.0f;
    }
    kbar[0 * dk + 0] = 8.0f;   // A: +e0
    kbar[2 * dk + 0] = -8.0f;  // C: -e0 (B stays zero)
    const std::vector<int32_t> frag = {0, 1, 2};
    const std::vector<int32_t> mand = {0, 0, 0};
    const std::vector<int32_t> full = {1, 1, 1};
    constexpr float scale = 1.0f / 8.0f;
    const std::vector<int32_t> emask((size_t) n_rows * n_cand, 1);
    const sel_oracle::select_result oracle = sel_oracle::select_tile(
            qpair.data(), dk, n_rows, kbar.data(), dk, mand.data(), full.data(), emask.data(), n_cand,
            dk, scale, 0.0f, 0.5, 0, 8, 8);
    SELECT_CHECK(oracle.rc == sel_oracle::OK);
    // Global max M=8 comes from row0/A (not a per-row or per-head max).
    SELECT_CHECK_MSG(std::fabs(oracle.m - 8.0) < 1e-9, "global M=%g want 8", oracle.m);
    SELECT_CHECK_MSG(oracle.exact == std::vector<int32_t>{0},
            "energy keeps spiky A only, exact size %zu", oracle.exact.size());
    SELECT_CHECK_MSG(oracle.proxy == (std::vector<int32_t>{1, 2}),
            "flat B and row1-top-1 C are both proxy, proxy size %zu", oracle.proxy.size());
    const ref_out ref = call_ref_select(qpair.data(), dk, n_rows, kbar.data(), dk,
            frag, mand, full, emask, dk, scale, 0.0f, 0.5f, 0, 8, 8);
    check_oracle_vs_ref("energy-global-max", oracle, ref, frag);
    std::fprintf(stderr, "  M=%.6g S=[%.6g %.6g %.6g] smax=%.6g\n",
            oracle.m, oracle.energies[0], oracle.energies[1], oracle.energies[2], oracle.smax);
    std::puts("energy global max: PASS");
}

static void test_threshold_ties_inclusive() {
    std::puts("--- threshold ties: equal scores kept inclusively (>=) ---");
    constexpr int32_t dk = 4;
    // Two identical fragments: bit-identical S, ratio exactly 1. At
    // alpha=1.0 inclusive->= keeps both; strict-> would keep NEITHER, and a
    // top-1 rule would keep only one. A third weaker fragment is dropped.
    std::vector<float> qpair(2 * 3 * dk, 0.0f);
    std::vector<float> kbar(3 * dk, 0.0f);
    for (int r = 0; r < 2; ++r) {
        for (int j = 0; j < 3; ++j) {
            qpair[((size_t) r * 3 + j) * dk + 0] = 2.0f;
        }
    }
    kbar[0 * dk + 0] = 1.0f;
    kbar[1 * dk + 0] = 1.0f; // identical to frag 0
    kbar[2 * dk + 0] = -1.0f;
    const std::vector<int32_t> frag = {0, 1, 2};
    const std::vector<int32_t> mand = {0, 0, 0};
    const std::vector<int32_t> full = {1, 1, 1};
    const std::vector<int32_t> tmask(6, 1);
    const sel_oracle::select_result oracle = sel_oracle::select_tile(
            qpair.data(), dk, 2, kbar.data(), dk, mand.data(), full.data(), tmask.data(), 3,
            dk, 1.0f, 0.0f, 1.0, 0, 8, 8);
    SELECT_CHECK(oracle.rc == sel_oracle::OK);
    SELECT_CHECK_MSG(oracle.energies[0] == oracle.energies[1],
            "tie fragments must have identical energy %g vs %g",
            oracle.energies[0], oracle.energies[1]);
    SELECT_CHECK_MSG(oracle.exact == (std::vector<int32_t>{0, 1}),
            "alpha=1 keeps both tied fragments");
    SELECT_CHECK_MSG(oracle.proxy == (std::vector<int32_t>{2}), "weak fragment proxies");
    const ref_out ref = call_ref_select(qpair.data(), dk, 2, kbar.data(), dk,
            frag, mand, full, tmask, dk, 1.0f, 0.0f, 1.0f, 0, 8, 8);
    check_oracle_vs_ref("threshold-tie", oracle, ref, frag);
    const sel_oracle::select_result oracle_lo = sel_oracle::select_tile(
            qpair.data(), dk, 2, kbar.data(), dk, mand.data(), full.data(), tmask.data(), 3,
            dk, 1.0f, 0.0f, 0.1, 0, 8, 8);
    const ref_out ref_lo = call_ref_select(qpair.data(), dk, 2, kbar.data(), dk,
            frag, mand, full, tmask, dk, 1.0f, 0.0f, 0.1f, 0, 8, 8);
    check_oracle_vs_ref("threshold-tie-lo", oracle_lo, ref_lo, frag);
    std::puts("threshold ties: PASS");
}

static void test_extreme_finite_logits_stability() {
    std::puts("--- extreme finite logits: max-subtraction stability ---");
    constexpr int32_t dk = 8;
    // All-equal huge logits: dot = 8 * 1e3 * 1e3 = 8e6, finite. Naive exp()
    // would overflow to inf; max-subtraction gives exp(0)=1, all kept.
    std::vector<float> qpair(2 * 2 * dk, 1000.0f);
    std::vector<float> kbar(2 * dk, 1000.0f);
    const std::vector<int32_t> frag = {0, 1};
    const std::vector<int32_t> mand = {0, 0};
    const std::vector<int32_t> full = {1, 1};
    const std::vector<int32_t> xmask2x2(4, 1);
    const std::vector<int32_t> xmask1x2(2, 1);
    sel_oracle::select_result oracle = sel_oracle::select_tile(
            qpair.data(), dk, 2, kbar.data(), dk, mand.data(), full.data(), xmask2x2.data(), 2,
            dk, 1.0f, 0.0f, 0.1, 0, 8, 8);
    SELECT_CHECK(oracle.rc == sel_oracle::OK);
    SELECT_CHECK_MSG(std::isfinite(oracle.m) && oracle.m > 7.9e6 && oracle.m < 8.1e6,
            "M=%g want ~8e6 finite", oracle.m);
    SELECT_CHECK_MSG(std::isfinite(oracle.smax), "smax finite, got %g", oracle.smax);
    SELECT_CHECK_MSG(oracle.exact == (std::vector<int32_t>{0, 1}), "all huge-equal kept");
    SELECT_CHECK_MSG(oracle.proxy.empty(), "none proxied");
    ref_out ref = call_ref_select(qpair.data(), dk, 2, kbar.data(), dk,
            frag, mand, full, xmask2x2, dk, 1.0f, 0.0f, 0.1f, 0, 8, 8);
    check_oracle_vs_ref("extreme-equal", oracle, ref, frag);
    // Disparate huge logits: +1e6 vs -1e6 in a single row. e^-2e6
    // underflows to 0 (fine), nothing overflows, weak drops at alpha=0.5.
    std::vector<float> qp2(1 * 2 * dk, 0.0f); // (r*n_cand+j) layout, r=0
    std::vector<float> k2(2 * dk, 0.0f);
    for (int32_t d = 0; d < dk; ++d) {
        qp2[0 * dk + d] = 125000.0f; // candidate 0's Q slice
        qp2[1 * dk + d] = 125000.0f; // candidate 1's Q slice
        k2[0 * dk + d] = 1.0f;       // dot = 8*125000 = +1e6
        k2[1 * dk + d] = -1.0f;      // dot = -1e6
    }
    oracle = sel_oracle::select_tile(
            qp2.data(), dk, 1, k2.data(), dk, mand.data(), full.data(), xmask1x2.data(), 2,
            dk, 1.0f, 0.0f, 0.5, 0, 8, 8);
    SELECT_CHECK(oracle.rc == sel_oracle::OK);
    SELECT_CHECK_MSG(std::isfinite(oracle.m) && std::fabs(oracle.m - 1e6) < 1.0,
            "M=%g want 1e6", oracle.m);
    SELECT_CHECK_MSG(oracle.exact == (std::vector<int32_t>{0}), "only +1e6 kept");
    SELECT_CHECK_MSG(oracle.proxy == (std::vector<int32_t>{1}), "-1e6 proxied, not nan");
    ref = call_ref_select(qp2.data(), dk, 1, k2.data(), dk,
            frag, mand, full, xmask1x2, dk, 1.0f, 0.0f, 0.5f, 0, 8, 8);
    check_oracle_vs_ref("extreme-disparate", oracle, ref, frag);
    // Softcap bounds the same huge dots into [-cap, cap]: stability must
    // hold in the capped domain too, with identical keep sets both sides.
    oracle = sel_oracle::select_tile(
            qp2.data(), dk, 1, k2.data(), dk, mand.data(), full.data(), xmask1x2.data(), 2,
            dk, 1.0f, 30.0, 0.5, 0, 8, 8);
    ref = call_ref_select(qp2.data(), dk, 1, k2.data(), dk,
            frag, mand, full, xmask1x2, dk, 1.0f, 30.0f, 0.5f, 0, 8, 8);
    SELECT_CHECK(oracle.rc == sel_oracle::OK);
    SELECT_CHECK_MSG(std::fabs(oracle.m - 30.0) < 1e-9, "capped M=%g want 30", oracle.m);
    check_oracle_vs_ref("extreme-softcap", oracle, ref, frag);
    std::puts("extreme logits: PASS");
}

static void test_mandatory_partial_exactall_ref() {
    std::puts("--- mandatory / partial-exclusion / exact_all at ref level ---");
    constexpr int32_t dk = 4;
    // One row; kbar magnitudes order full-candidate energies S0 > S1 > S3:
    // z = 16/12/-16 over candidates 0/1/3. Candidate 2 is partial: its
    // column is zero-masked, never read, never scored (see also
    // test_partial_pollutant_isolation for the pollutant variant).
    std::vector<float> qpair(1 * 4 * dk, 0.0f);
    std::vector<float> kbar(4 * dk, 0.0f);
    for (int32_t j = 0; j < 4; ++j) {
        qpair[(size_t) j * dk + 0] = 4.0f;
    }
    kbar[0 * dk + 0] = 4.0f;  // z=16
    kbar[1 * dk + 0] = 3.0f;  // z=12
    kbar[2 * dk + 0] = 0.0f;  // partial column: masked out, never read
    kbar[3 * dk + 0] = -4.0f; // z=-16
    const std::vector<int32_t> frag = {10, 11, 12, 13};
    // Case A: nothing forced. Partial column 2 is caller-excluded from the
    // energy via its zero mask (never read, never scored). Full
    // S=[1, e^-4, e^-32], smax=1. alpha=0.05 keeps {10}; proxy={11,13}.
    const std::vector<int32_t> part_mask = {1, 1, 0, 1};
    const std::vector<int32_t> ones_mask = {1, 1, 1, 1};
    {
        const std::vector<int32_t> mand = {0, 0, 0, 0};
        const std::vector<int32_t> full = {1, 1, 0, 1};
        const sel_oracle::select_result oracle = sel_oracle::select_tile(
                qpair.data(), dk, 1, kbar.data(), dk, mand.data(), full.data(), part_mask.data(), 4,
                dk, 1.0f, 0.0f, 0.05, 0, 8, 8);
        SELECT_CHECK_MSG(oracle.exact == (std::vector<int32_t>{0, 2}), "caseA exact");
        SELECT_CHECK_MSG(oracle.proxy == (std::vector<int32_t>{1, 3}), "caseA proxy");
        const ref_out ref = call_ref_select(qpair.data(), dk, 1, kbar.data(), dk,
                frag, mand, full, part_mask, dk, 1.0f, 0.0f, 0.05f, 0, 8, 8);
        check_oracle_vs_ref("mandatory-A", oracle, ref, frag);
    }
    // Case B: candidate 3 mandatory (sink). exact={0,2,3}, proxy={1}: the
    // complement is explicit, nothing lost.
    {
        const std::vector<int32_t> mand = {0, 0, 0, 1};
        const std::vector<int32_t> full = {1, 1, 0, 1};
        const sel_oracle::select_result oracle = sel_oracle::select_tile(
                qpair.data(), dk, 1, kbar.data(), dk, mand.data(), full.data(), part_mask.data(), 4,
                dk, 1.0f, 0.0f, 0.05, 0, 8, 8);
        SELECT_CHECK_MSG(oracle.exact == (std::vector<int32_t>{0, 2, 3}), "caseB exact");
        SELECT_CHECK_MSG(oracle.proxy == (std::vector<int32_t>{1}), "caseB proxy");
        const ref_out ref = call_ref_select(qpair.data(), dk, 1, kbar.data(), dk,
                frag, mand, full, part_mask, dk, 1.0f, 0.0f, 0.05f, 0, 8, 8);
        check_oracle_vs_ref("mandatory-B", oracle, ref, frag);
    }
    // Case C: exact_all classifies every legal candidate exact; proxy is
    // explicitly empty (never an inferred complement).
    {
        const std::vector<int32_t> mand = {0, 0, 0, 0};
        const std::vector<int32_t> full = {1, 1, 1, 1};
        const sel_oracle::select_result oracle = sel_oracle::select_tile(
                qpair.data(), dk, 1, kbar.data(), dk, mand.data(), full.data(), ones_mask.data(), 4,
                dk, 1.0f, 0.0f, 0.05, 1, 8, 8);
        SELECT_CHECK_MSG(oracle.exact == (std::vector<int32_t>{0, 1, 2, 3}), "exact-all");
        SELECT_CHECK_MSG(oracle.proxy.empty(), "exact-all proxy empty");
        const ref_out ref = call_ref_select(qpair.data(), dk, 1, kbar.data(), dk,
                frag, mand, full, ones_mask, dk, 1.0f, 0.0f, 0.05f, 1, 8, 8);
        check_oracle_vs_ref("mandatory-C-exactall", oracle, ref, frag);
    }
    std::puts("mandatory/partial/exact_all: PASS");
}

// Pollutant isolation: a partial candidate carrying extreme foreign values
// (simulating future/invisible tokens folded into a whole-fragment mean)
// must leave the global max, every legal energy, and every legal keep set
// bit-identical, while the partial itself stays exact.
static void test_partial_pollutant_isolation() {
    std::puts("--- partial exclusion: pollutant isolation ---");
    constexpr int32_t dk = 4;
    std::vector<float> qpair(1 * 4 * dk, 0.0f);
    std::vector<float> kbar_clean(4 * dk, 0.0f);
    std::vector<float> kbar_polluted(4 * dk, 0.0f);
    for (int32_t j = 0; j < 4; ++j) {
        qpair[(size_t) j * dk + 0] = 4.0f;
    }
    kbar_clean[0 * dk + 0] = 4.0f;
    kbar_clean[1 * dk + 0] = 3.0f;
    kbar_clean[2 * dk + 0] = 0.0f; // legal sub-mean of the partial
    kbar_clean[3 * dk + 0] = -4.0f;
    kbar_polluted = kbar_clean;
    for (int32_t d = 0; d < dk; ++d) {
        kbar_polluted[2 * dk + d] = 250000.0f; // foreign/future values: z=+4e6
    }
    const std::vector<int32_t> mand = {0, 0, 0, 0};
    const std::vector<int32_t> full = {1, 1, 0, 1}; // candidate 2 partial
    // Callers exclude the partial column from the energy via the mask, so
    // the polluted slot is never read by either implementation.
    const std::vector<int32_t> pmask = {1, 1, 0, 1};
    const sel_oracle::select_result clean = sel_oracle::select_tile(
            qpair.data(), dk, 1, kbar_clean.data(), dk, mand.data(), full.data(), pmask.data(), 4,
            dk, 1.0f, 0.0f, 0.05, 0, 8, 8);
    const sel_oracle::select_result polluted = sel_oracle::select_tile(
            qpair.data(), dk, 1, kbar_polluted.data(), dk, mand.data(), full.data(), pmask.data(), 4,
            dk, 1.0f, 0.0f, 0.05, 0, 8, 8);
    SELECT_CHECK(clean.rc == sel_oracle::OK && polluted.rc == sel_oracle::OK);
    SELECT_CHECK_MSG(clean.m == polluted.m && clean.m == 16.0,
            "pollutant must not shift M: clean=%g polluted=%g", clean.m, polluted.m);
    SELECT_CHECK_MSG(clean.smax == polluted.smax, "pollutant must not shift smax");
    SELECT_CHECK_MSG(clean.exact == polluted.exact, "pollutant must not change exact set");
    SELECT_CHECK_MSG(clean.proxy == polluted.proxy, "pollutant must not change proxy set");
    SELECT_CHECK_MSG(polluted.exact == (std::vector<int32_t>{0, 2}),
            "partial stays exact under pollution");
    SELECT_CHECK_MSG(polluted.energies[2] < 0.0, "excluded partial carries no energy");
    // Reference agrees on the clean inputs, and — mask excluding the
    // polluted column — on the polluted inputs too: invalid slots are never
    // read, so foreign values cannot leak into M/Smax/keep sets.
    const std::vector<int32_t> frag = {10, 11, 12, 13};
    const ref_out ref = call_ref_select(qpair.data(), dk, 1, kbar_clean.data(), dk,
            frag, mand, full, pmask, dk, 1.0f, 0.0f, 0.05f, 0, 8, 8);
    check_oracle_vs_ref("pollutant-clean-ref", clean, ref, frag);
    const ref_out ref_pol = call_ref_select(qpair.data(), dk, 1, kbar_polluted.data(), dk,
            frag, mand, full, pmask, dk, 1.0f, 0.0f, 0.05f, 0, 8, 8);
    check_oracle_vs_ref("pollutant-masked-ref", polluted, ref_pol, frag);
    std::puts("pollutant isolation: PASS");
}

static void test_capacity_overflow_no_truncation() {
    std::puts("--- capacity overflow errors, never truncation ---");
    constexpr int32_t dk = 4;
    std::vector<float> qpair(1 * 3 * dk, 1.0f);
    std::vector<float> kbar(3 * dk, 1.0f);
    const std::vector<int32_t> frag = {0, 1, 2};
    const std::vector<int32_t> mand = {0, 0, 0};
    const std::vector<int32_t> full = {1, 1, 1};
    const std::vector<int32_t> cmask(3, 1);
    // All three tie -> all exact; cap_exact=2 must ERROR, not keep two.
    {
        const sel_oracle::select_result oracle = sel_oracle::select_tile(
                qpair.data(), dk, 1, kbar.data(), dk, mand.data(), full.data(), cmask.data(), 3,
                dk, 1.0f, 0.0f, 0.1, 0, 2, 8);
        SELECT_CHECK_MSG(oracle.rc == sel_oracle::OVER_CAPACITY, "oracle over-capacity");
        SELECT_CHECK_MSG(oracle.exact.empty() && oracle.proxy.empty(), "oracle emits no truncated list");
        const ref_out ref = call_ref_select(qpair.data(), dk, 1, kbar.data(), dk,
                frag, mand, full, cmask, dk, 1.0f, 0.0f, 0.1f, 0, 2, 8);
        SELECT_CHECK_MSG(ref.rc == GGML_FLASHPREFILL_ERR_OVER_CAPACITY,
                "ref OVER_CAPACITY, got %d (%s)", ref.rc, ggml_flashprefill_strerror(ref.rc));
        SELECT_CHECK_MSG(ref.exact.empty() && ref.proxy.empty(), "ref emits no truncated list");
    }
    // Proxy-side overflow: one hot candidate exact, two weak proxy; cap 1 fails.
    {
        std::vector<float> qp(1 * 3 * dk, 0.0f);
        std::vector<float> kb(3 * dk, 0.0f);
        qp[0 * dk + 0] = 8.0f;
        kb[0 * dk + 0] = 8.0f; // z=64, others z=0
        const sel_oracle::select_result oracle = sel_oracle::select_tile(
                qp.data(), dk, 1, kb.data(), dk, mand.data(), full.data(), cmask.data(), 3,
                dk, 1.0f, 0.0f, 1.0, 0, 8, 1);
        SELECT_CHECK_MSG(oracle.rc == sel_oracle::OVER_CAPACITY, "oracle proxy over-capacity");
        const ref_out ref = call_ref_select(qp.data(), dk, 1, kb.data(), dk,
                frag, mand, full, cmask, dk, 1.0f, 0.0f, 1.0f, 0, 8, 1);
        SELECT_CHECK_MSG(ref.rc == GGML_FLASHPREFILL_ERR_OVER_CAPACITY,
                "ref proxy OVER_CAPACITY, got %d", ref.rc);
    }
    // Per-role boundary: OVER_CAPACITY iff n_exact > max_sel OR
    // n_proxy > max_sel. exact=2/proxy=2 with caps 2/2 is VALID even though
    // the pair totals 4 > 2 across roles.
    {
        std::vector<float> qp(1 * 4 * dk, 0.0f);
        std::vector<float> kb(4 * dk, 0.0f);
        for (int32_t j = 0; j < 4; ++j) {
            qp[(size_t) j * dk + 0] = 4.0f;
        }
        kb[0 * dk + 0] = 4.0f; // z=16
        kb[1 * dk + 0] = 3.0f; // z=12
        kb[2 * dk + 0] = 0.0f; // z=0
        kb[3 * dk + 0] = -4.0f; // z=-16
        const std::vector<int32_t> frag4 = {0, 1, 2, 3};
        const std::vector<int32_t> mand4 = {0, 0, 0, 0};
        const std::vector<int32_t> full4 = {1, 1, 1, 1};
        const std::vector<int32_t> mask4(4, 1);
        // alpha=0.001: thr=0.001; e^-4~=0.0183 kept, e^-16 proxied.
        const sel_oracle::select_result oracle = sel_oracle::select_tile(
                qp.data(), dk, 1, kb.data(), dk, mand4.data(), full4.data(), mask4.data(), 4,
                dk, 1.0f, 0.0f, 0.001, 0, 2, 2);
        SELECT_CHECK_MSG(oracle.rc == sel_oracle::OK, "per-role 2/2 within caps");
        SELECT_CHECK_MSG(oracle.exact == (std::vector<int32_t>{0, 1}), "per-role exact");
        SELECT_CHECK_MSG(oracle.proxy == (std::vector<int32_t>{2, 3}), "per-role proxy");
        const ref_out ref = call_ref_select(qp.data(), dk, 1, kb.data(), dk,
                frag4, mand4, full4, mask4, dk, 1.0f, 0.0f, 0.001f, 0, 2, 2);
        check_oracle_vs_ref("capacity-per-role", oracle, ref, frag4);
    }
    // Plan-level capacity: a per-list count beyond max_sel is CAP_EXCEEDED,
    // while a (2,2) split at max_sel=2 is accepted (per-role rule).
    {
        int64_t words = 0;
        SELECT_CHECK(ggml_flashprefill_plan_words(2, 2, 3, &words) == GGML_FLASHPREFILL_OK);
        std::vector<int32_t> plan((size_t) words, 0);
        SELECT_CHECK(ggml_flashprefill_plan_init(plan.data(), words, 2, 2, 3, 0) == GGML_FLASHPREFILL_OK);
        SELECT_CHECK_MSG(ggml_flashprefill_plan_set_counts(plan.data(), words, 0, 0, 4, 0) ==
                        GGML_FLASHPREFILL_ERR_CAP_EXCEEDED,
                "4 > max_sel=3 must exceed, not truncate");
        SELECT_CHECK(ggml_flashprefill_plan_set_counts(plan.data(), words, 0, 0, 2, 1) == GGML_FLASHPREFILL_OK);
        int64_t words2 = 0;
        SELECT_CHECK(ggml_flashprefill_plan_words(1, 1, 2, &words2) == GGML_FLASHPREFILL_OK);
        std::vector<int32_t> plan2((size_t) words2, 0);
        SELECT_CHECK(ggml_flashprefill_plan_init(plan2.data(), words2, 1, 1, 2, 0) == GGML_FLASHPREFILL_OK);
        SELECT_CHECK_MSG(ggml_flashprefill_plan_set_counts(plan2.data(), words2, 0, 0, 2, 2) ==
                        GGML_FLASHPREFILL_OK,
                "per-role (2,2) at max_sel=2 accepted despite total 4");
    }
    std::puts("capacity overflow: PASS");
}

// All-zero mask boundary over mandatory-partial candidates: legal, every
// candidate exact, zero sparse work, capacity still enforced; non-0/1 mask
// entries are BAD_ARG on both sides.
static void test_mask_allzero_boundary() {
    std::puts("--- all-zero mask boundary + strict 0/1 ---");
    constexpr int32_t dk = 4;
    std::vector<float> qpair(2 * 3 * dk, 1.0f);
    std::vector<float> kbar(3 * dk, 1.0f);
    const std::vector<int32_t> frag = {7, 8, 9};
    const std::vector<int32_t> mand = {1, 1, 1}; // mandatory-partial rows
    const std::vector<int32_t> full = {0, 0, 0}; // ...with no legal mean
    const std::vector<int32_t> zero(6, 0);
    const sel_oracle::select_result oracle = sel_oracle::select_tile(
            qpair.data(), dk, 2, kbar.data(), dk, mand.data(), full.data(), zero.data(), 3,
            dk, 1.0f, 0.0f, 0.1, 0, 8, 8);
    SELECT_CHECK_MSG(oracle.rc == sel_oracle::OK, "all-zero mask legal");
    SELECT_CHECK_MSG(oracle.exact == (std::vector<int32_t>{0, 1, 2}), "all exact");
    SELECT_CHECK_MSG(oracle.proxy.empty(), "zero sparse work");
    const ref_out ref = call_ref_select(qpair.data(), dk, 2, kbar.data(), dk,
            frag, mand, full, zero, dk, 1.0f, 0.0f, 0.1f, 0, 8, 8);
    check_oracle_vs_ref("mask-allzero", oracle, ref, frag);
    // Capacity still enforced when nothing is scored: 3 exact > cap 2.
    const sel_oracle::select_result oracle_cap = sel_oracle::select_tile(
            qpair.data(), dk, 2, kbar.data(), dk, mand.data(), full.data(), zero.data(), 3,
            dk, 1.0f, 0.0f, 0.1, 0, 2, 8);
    SELECT_CHECK_MSG(oracle_cap.rc == sel_oracle::OVER_CAPACITY, "all-zero still capacity-checked");
    const ref_out ref_cap = call_ref_select(qpair.data(), dk, 2, kbar.data(), dk,
            frag, mand, full, zero, dk, 1.0f, 0.0f, 0.1f, 0, 2, 8);
    SELECT_CHECK_MSG(ref_cap.rc == GGML_FLASHPREFILL_ERR_OVER_CAPACITY,
            "ref all-zero OVER_CAPACITY, got %d", ref_cap.rc);
    // Strict 0/1: a 2 entry is BAD_ARG (ref) / BAD_MASK (oracle).
    std::vector<int32_t> badmask(6, 1);
    badmask[4] = 2;
    const sel_oracle::select_result oracle_bad = sel_oracle::select_tile(
            qpair.data(), dk, 2, kbar.data(), dk, mand.data(), full.data(), badmask.data(), 3,
            dk, 1.0f, 0.0f, 0.1, 0, 8, 8);
    SELECT_CHECK_MSG(oracle_bad.rc == sel_oracle::BAD_MASK, "oracle rejects non-0/1 mask");
    const std::vector<int32_t> full_ok = {1, 1, 1};
    const std::vector<int32_t> mand_ok = {0, 0, 0};
    const ref_out ref_bad = call_ref_select(qpair.data(), dk, 2, kbar.data(), dk,
            frag, mand_ok, full_ok, badmask, dk, 1.0f, 0.0f, 0.1f, 0, 8, 8);
    SELECT_CHECK_MSG(ref_bad.rc == GGML_FLASHPREFILL_ERR_BAD_ARG,
            "ref rejects non-0/1 mask, got %d (%s)", ref_bad.rc, ggml_flashprefill_strerror(ref_bad.rc));
    std::puts("all-zero mask: PASS");
}

// ---------------------------------------------------------------------------
// Metadata builders (helper API only) and legality tests.
// Single-source-query fixtures: sq=0, one row per q_head, tile=r/BM.
// ---------------------------------------------------------------------------
struct sq_fixture {
    int32_t dk = 8, dv = 4, hkv = 2, ng = 1, hq = 4;
    int32_t n_frag = 4, cells_per_frag = 2, bm = 4;
    std::vector<int32_t> row_flags;   // per q_head
    std::vector<int32_t> use_mand;    // per (pair, frag): pair=t*hkv+kh
    std::vector<int32_t> use_sub;     // per (pair, frag): sub_count (full=cells_per_frag)
    std::vector<int32_t> use_group;   // per (pair, frag): q_group
    std::vector<int32_t> meta;
    int64_t meta_words = 0;
    int64_t n_tiles = 0;
    // use index of (pair, frag); -1 when the pair owns no rows (such pairs
    // get no uses at all, so no orphan can exist).
    std::vector<int32_t> use_index;
};

static int32_t build_sq_meta(sq_fixture & fx) {
    const int32_t g = fx.hq / fx.hkv;
    if (fx.hq % fx.hkv != 0 || g < 1) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    const int32_t n_row = fx.hq;
    std::vector<int32_t> row_tile((size_t) n_row);
    int64_t max_tile = 0;
    for (int32_t qh = 0; qh < n_row; ++qh) {
        row_tile[(size_t) qh] = qh / fx.bm;
        max_tile = std::max<int64_t>(max_tile, row_tile[(size_t) qh]);
    }
    const int64_t n_tiles = max_tile + 1;
    const int64_t n_pair = n_tiles * fx.hkv;
    if ((int64_t) fx.use_mand.size() != n_pair * fx.n_frag ||
        (int64_t) fx.use_sub.size() != n_pair * fx.n_frag ||
        (int64_t) fx.use_group.size() != n_pair * fx.n_frag ||
        (int64_t) fx.row_flags.size() != n_row) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    std::vector<std::vector<int32_t>> pair_rows((size_t) n_pair);
    for (int32_t qh = 0; qh < n_row; ++qh) {
        pair_rows[(size_t)(row_tile[(size_t) qh] * fx.hkv + qh / g)].push_back(qh);
    }
    struct use_rec { int32_t frag, tile, kh, grp, sub, mand; };
    std::vector<use_rec> uses;
    fx.use_index.assign((size_t)(n_pair * fx.n_frag), -1);
    for (int64_t p = 0; p < n_pair; ++p) {
        if (pair_rows[(size_t) p].empty()) {
            continue;
        }
        const int32_t tile = (int32_t)(p / fx.hkv);
        const int32_t kh = (int32_t)(p % fx.hkv);
        for (int32_t f = 0; f < fx.n_frag; ++f) {
            const size_t k = (size_t)(p * fx.n_frag + f);
            fx.use_index[k] = (int32_t) uses.size();
            uses.push_back({f, tile, kh, fx.use_group[k], fx.use_sub[k], fx.use_mand[k]});
        }
    }
    const int64_t n_use = (int64_t) uses.size();
    const int64_t n_cell = (int64_t) fx.n_frag * fx.cells_per_frag;
    int64_t words = 0;
    int32_t rc = ggml_flashprefill_metadata_words(fx.n_frag, n_row, n_use, n_cell, &words);
    if (rc) {
        return rc;
    }
    fx.meta.assign((size_t) words, 0);
    rc = ggml_flashprefill_metadata_init(fx.meta.data(), words, fx.n_frag, n_row, n_use, n_cell,
            fx.dk, fx.dv, fx.hkv, fx.ng, fx.hq);
    if (rc) {
        return rc;
    }
    rc = ggml_flashprefill_metadata_set_counts(fx.meta.data(), words, fx.n_frag, n_row, n_use, n_cell);
    if (rc) {
        return rc;
    }
    for (int32_t f = 0; f < fx.n_frag; ++f) {
        rc = ggml_flashprefill_metadata_set_frag(fx.meta.data(), words, f,
                (int64_t) f * fx.cells_per_frag, fx.cells_per_frag, f, 0, 0);
        if (rc) {
            return rc;
        }
    }
    for (int32_t qh = 0; qh < n_row; ++qh) {
        rc = ggml_flashprefill_metadata_set_row(fx.meta.data(), words, qh,
                0, qh / g, qh, row_tile[(size_t) qh], 0, 1, fx.row_flags[(size_t) qh], qh);
        if (rc) {
            return rc;
        }
    }
    for (size_t i = 0; i < uses.size(); ++i) {
        const use_rec & u = uses[i];
        rc = ggml_flashprefill_metadata_set_use(fx.meta.data(), words, (int64_t) i,
                u.frag, u.tile, u.kh, u.grp, (int64_t) u.frag * fx.cells_per_frag, u.sub, u.mand, 0);
        if (rc) {
            return rc;
        }
    }
    for (int64_t c = 0; c < n_cell; ++c) {
        rc = ggml_flashprefill_metadata_set_cell(fx.meta.data(), words, c, (int32_t) c);
        if (rc) {
            return rc;
        }
    }
    rc = ggml_flashprefill_metadata_validate(fx.meta.data(), words);
    if (rc) {
        return rc;
    }
    fx.meta_words = words;
    rc = ggml_flashprefill_metadata_n_tiles(fx.meta.data(), words, &fx.n_tiles);
    if (rc) {
        return rc;
    }
    if (fx.n_tiles != n_tiles) {
        return GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
    }
    return GGML_FLASHPREFILL_OK;
}

static void test_metadata_counts_and_padding() {
    std::puts("--- metadata counts 0/1/63/64/65/127/128/129, padding excluded ---");
    const int64_t counts[] = {0, 1, 63, 64, 65, 127, 128, 129};
    constexpr int32_t BN = 64;
    for (int64_t n : counts) {
        const int64_t nf = (n + BN - 1) / BN; // 0 when n==0
        int64_t words = 0;
        SELECT_CHECK(ggml_flashprefill_metadata_words(nf, n, n, n, &words) == GGML_FLASHPREFILL_OK);
        std::vector<int32_t> meta((size_t) words, 0);
        SELECT_CHECK(ggml_flashprefill_metadata_init(meta.data(), words, nf, n, n, n,
                8, 4, 1, 1, 1) == GGML_FLASHPREFILL_OK);
        SELECT_CHECK(ggml_flashprefill_metadata_set_counts(meta.data(), words, nf, n, n, n) ==
                GGML_FLASHPREFILL_OK);
        // Token-granular uses: token t lives in frag t/BN at offset t%BN;
        // each use covers exactly its own cell (sub_count=1). Padding cells
        // are never entered in the table.
        for (int64_t f = 0; f < nf; ++f) {
            const int64_t cnt = std::min<int64_t>(BN, n - f * BN);
            SELECT_CHECK(ggml_flashprefill_metadata_set_frag(meta.data(), words, f,
                    f * BN, cnt, (int32_t) f, 0, 0) == GGML_FLASHPREFILL_OK);
        }
        for (int64_t t = 0; t < n; ++t) {
            // BM=128 packed tiles over the single-head packed rows.
            SELECT_CHECK(ggml_flashprefill_metadata_set_row(meta.data(), words, t,
                    (int32_t) t, 0, (int32_t) t, (int32_t)(t / 128), 0, 1, 0, 0) ==
                    GGML_FLASHPREFILL_OK);
            SELECT_CHECK(ggml_flashprefill_metadata_set_use(meta.data(), words, t,
                    (int32_t)(t / BN), (int32_t)(t / 128), 0, 0, t, 1, 0, (int32_t) t) ==
                    GGML_FLASHPREFILL_OK);
            SELECT_CHECK(ggml_flashprefill_metadata_set_cell(meta.data(), words, t, (int32_t) t) ==
                    GGML_FLASHPREFILL_OK);
        }
        SELECT_CHECK_MSG(ggml_flashprefill_metadata_validate(meta.data(), words) == GGML_FLASHPREFILL_OK,
                "count %lld validates", (long long) n);
        int64_t ntiles = -1;
        SELECT_CHECK(ggml_flashprefill_metadata_n_tiles(meta.data(), words, &ntiles) == GGML_FLASHPREFILL_OK);
        const int64_t want_tiles = n == 0 ? 0 : (n - 1) / 128 + 1;
        SELECT_CHECK_MSG(ntiles == want_tiles, "count %lld tiles %lld want %lld",
                (long long) n, (long long) ntiles, (long long) want_tiles);
        // max_sel is the actual per-pair max (fullest tile occupancy = 128,
        // or n when smaller), never the global n_use.
        int64_t max_sel = -1;
        SELECT_CHECK(ggml_flashprefill_plan_max_sel_for_meta(meta.data(), words, &max_sel) ==
                GGML_FLASHPREFILL_OK);
        const int64_t want_max = n == 0 ? 0 : std::min<int64_t>(n, 128);
        SELECT_CHECK_MSG(max_sel == want_max, "count %lld max_sel %lld want %lld",
                (long long) n, (long long) max_sel, (long long) want_max);
        // All-exact plan: visible/exact token incidences must equal the
        // legal count n (padding excluded), via the stats accessor.
        int64_t pwords = 0;
        SELECT_CHECK(ggml_flashprefill_plan_words(ntiles, 1, max_sel, &pwords) == GGML_FLASHPREFILL_OK);
        std::vector<int32_t> plan((size_t) pwords, 0);
        SELECT_CHECK(ggml_flashprefill_plan_init(plan.data(), pwords, ntiles, 1, max_sel, 0) ==
                GGML_FLASHPREFILL_OK);
        for (int64_t t = 0; t < ntiles; ++t) {
            std::vector<int64_t> members;
            for (int64_t u = 0; u < n; ++u) {
                if (u / 128 == t) {
                    members.push_back(u);
                }
            }
            SELECT_CHECK(ggml_flashprefill_plan_set_counts(plan.data(), pwords, t, 0,
                    (int64_t) members.size(), 0) == GGML_FLASHPREFILL_OK);
            for (size_t s = 0; s < members.size(); ++s) {
                SELECT_CHECK(ggml_flashprefill_plan_set_entry(plan.data(), pwords, t, 0,
                        GGML_FLASHPREFILL_ROLE_EXACT, (int64_t) s, members[s]) == GGML_FLASHPREFILL_OK);
            }
        }
        SELECT_CHECK(ggml_flashprefill_plan_validate(plan.data(), pwords) == GGML_FLASHPREFILL_OK);
        SELECT_CHECK_MSG(ggml_flashprefill_plan_validate_against_metadata(
                plan.data(), pwords, meta.data(), words) == GGML_FLASHPREFILL_OK,
                "count %lld plan covers metadata exactly once", (long long) n);
        SELECT_CHECK(ggml_flashprefill_plan_accumulate_stats(plan.data(), pwords, meta.data(), words) ==
                GGML_FLASHPREFILL_OK);
        struct ggml_flashprefill_plan_stats st = {};
        SELECT_CHECK(ggml_flashprefill_plan_get_stats(plan.data(), pwords, &st) == GGML_FLASHPREFILL_OK);
        SELECT_CHECK_MSG(st.visible_tokens == n && st.exact_tokens == n,
                "count %lld visible=%lld exact=%lld (padding excluded)",
                (long long) n, (long long) st.visible_tokens, (long long) st.exact_tokens);
        SELECT_CHECK_MSG(st.selected_total == n && st.corrected_total == 0,
                "count %lld selected=%d corrected=%d", (long long) n, st.selected_total, st.corrected_total);
        SELECT_CHECK_MSG(st.sparse_rows == 0 && st.dense_rows == (int32_t) n,
                "count %lld sparse=%d dense=%d", (long long) n, st.sparse_rows, st.dense_rows);
        std::fprintf(stderr, "  count %lld: tiles=%lld max_sel=%lld visible=%lld PASS\n",
                (long long) n, (long long) ntiles, (long long) max_sel, (long long) st.visible_tokens);
    }
    std::puts("metadata counts: PASS");
}

static void test_metadata_rejections() {
    std::puts("--- metadata rejection boundaries: dup/orphan/dense ---");
    // G=2 (hq=2, hkv=1), 2 frags x 2 cells, BM=2 -> 1 tile, 1 pair.
    sq_fixture fx;
    fx.dk = 8;
    fx.dv = 4;
    fx.hkv = 1;
    fx.ng = 1;
    fx.hq = 2;
    fx.n_frag = 2;
    fx.cells_per_frag = 2;
    fx.bm = 2;
    fx.row_flags = {0, 0};
    fx.use_mand = {0, 0};
    fx.use_sub = {2, 2};
    fx.use_group = {0, 0};
    SELECT_CHECK(build_sq_meta(fx) == GGML_FLASHPREFILL_OK);
    // Duplicate use key: rewrite use 1 with frag 0's key.
    {
        std::vector<int32_t> bad = fx.meta;
        SELECT_CHECK(ggml_flashprefill_metadata_set_use(bad.data(), fx.meta_words, 1,
                0, 0, 0, 0, 0, 2, 0, 0) == GGML_FLASHPREFILL_OK);
        SELECT_CHECK_MSG(ggml_flashprefill_metadata_validate(bad.data(), fx.meta_words) ==
                        GGML_FLASHPREFILL_ERR_DUP_KEY,
                "dup use key rejected");
    }
    // Duplicate row key: rewrite row 1 with q_head 0.
    {
        std::vector<int32_t> bad = fx.meta;
        SELECT_CHECK(ggml_flashprefill_metadata_set_row(bad.data(), fx.meta_words, 1,
                0, 0, 1, 0, 0, 1, 0, 0) == GGML_FLASHPREFILL_OK);
        SELECT_CHECK_MSG(ggml_flashprefill_metadata_validate(bad.data(), fx.meta_words) ==
                        GGML_FLASHPREFILL_ERR_DUP_KEY,
                "dup row key rejected");
    }
    // Duplicate physical token in one triple while keeping each use inside
    // its own fragment's legal cell-index range.
    {
        std::vector<int32_t> bad = fx.meta;
        SELECT_CHECK(ggml_flashprefill_metadata_set_cell(bad.data(), fx.meta_words, 2, 0) ==
                GGML_FLASHPREFILL_OK);
        SELECT_CHECK_MSG(ggml_flashprefill_metadata_validate(bad.data(), fx.meta_words) ==
                        GGML_FLASHPREFILL_ERR_DUP_TOKEN,
                "dup token rejected");
    }
    // Orphan use: move use 1 to a tile with no rows.
    {
        std::vector<int32_t> bad = fx.meta;
        SELECT_CHECK(ggml_flashprefill_metadata_set_use(bad.data(), fx.meta_words, 1,
                1, 7, 0, 0, 2, 2, 0, 0) == GGML_FLASHPREFILL_OK);
        SELECT_CHECK_MSG(ggml_flashprefill_metadata_validate(bad.data(), fx.meta_words) ==
                        GGML_FLASHPREFILL_ERR_ORPHAN_USE,
                "orphan use rejected");
    }
    // DENSE_FORCE row with a non-mandatory matched use.
    {
        std::vector<int32_t> bad = fx.meta;
        SELECT_CHECK(ggml_flashprefill_metadata_set_row(bad.data(), fx.meta_words, 0,
                0, 0, 0, 0, 0, 1, GGML_FLASHPREFILL_ROW_FLAG_DENSE_FORCE, 0) ==
                GGML_FLASHPREFILL_OK);
        SELECT_CHECK_MSG(ggml_flashprefill_metadata_validate(bad.data(), fx.meta_words) ==
                        GGML_FLASHPREFILL_ERR_DENSE_ROW_PROXY,
                "dense row with proxy use rejected");
    }
    std::puts("metadata rejections: PASS");
}

// ---------------------------------------------------------------------------
// Hotspot fixtures: Q/pool with deliberately different hotspots per KV
// head. Q layout [(hq*ng)*dk], element (h*ng+g)*dk+d == tensor (d,g,h,0).
// Pool layout [(F*hkv)*(dk+dv)], element ((f*hkv+h)*(dk+dv)+d).
// ---------------------------------------------------------------------------
struct hotspot_case {
    sq_fixture fx;
    std::vector<float> Q;
    std::vector<float> pool;
    float scale = 1.0f;
    float alpha = 0.1f;
};

static const float * hotspot_qvec(const hotspot_case & hc, int32_t qh, int32_t grp) {
    return hc.Q.data() + ((size_t) qh * hc.fx.ng + grp) * hc.fx.dk;
}

static const float * hotspot_kbar(const hotspot_case & hc, int32_t frag, int32_t kh) {
    return hc.pool.data() + ((size_t) frag * hc.fx.hkv + kh) * (hc.fx.dk + hc.fx.dv);
}

// Pool K means: frag f hot in direction (f % dk) for every kv head, except
// frag 1 under kv head 1 is negated (kh1 hates frag 1's direction). Q:
// kv-head-0 subheads point at frag 0, kv-head-1 subheads at frag 2.
static void hotspot_fill(hotspot_case & hc) {
    const int32_t dk = hc.fx.dk, dv = hc.fx.dv, hq = hc.fx.hq, hkv = hc.fx.hkv, ng = hc.fx.ng;
    hc.Q.assign((size_t) hq * ng * dk, 0.0f);
    hc.pool.assign((size_t) hc.fx.n_frag * hkv * (dk + dv), 0.0f);
    for (int32_t f = 0; f < hc.fx.n_frag; ++f) {
        for (int32_t kh = 0; kh < hkv; ++kh) {
            float * kb = const_cast<float *>(hotspot_kbar(hc, f, kh));
            for (int32_t d = 0; d < dk; ++d) {
                kb[d] = (d == f % dk) ? 3.0f : -0.5f;
            }
            if (f == 1 && kh == 1) {
                for (int32_t d = 0; d < dk; ++d) {
                    kb[d] = -kb[d];
                }
            }
            float * vb = kb + dk;
            for (int32_t d = 0; d < dv; ++d) {
                vb[d] = 0.25f * (f + 1) + 0.1f * kh;
            }
        }
    }
    const int32_t g = hq / hkv;
    for (int32_t qh = 0; qh < hq; ++qh) {
        const int32_t kh = qh / g;
        const int32_t hot_frag = (kh == 0) ? 0 : 2;
        for (int32_t grp = 0; grp < ng; ++grp) {
            float * qv = const_cast<float *>(hotspot_qvec(hc, qh, grp));
            const float * kb = hotspot_kbar(hc, hot_frag, kh);
            for (int32_t d = 0; d < dk; ++d) {
                qv[d] = kb[d];
            }
        }
    }
}

struct pair_expect {
    int64_t tile = 0, kh = 0;
    std::vector<int32_t> rows_qh;
    std::vector<int32_t> cand_frag; // per candidate, increasing use index
    std::vector<int32_t> cand_use;
    std::vector<int32_t> cand_mand;
    std::vector<int32_t> cand_full;
    std::vector<int32_t> cand_grp;
    std::vector<int32_t> want_exact; // use indices
    std::vector<int32_t> want_proxy; // use indices
};

// Gather one pair's oracle inputs (per-(row,candidate) Q under the
// candidate's q_group), run oracle + ref, compare, and record use-index
// expectations for plan/backend checks.
static bool solve_pair(const hotspot_case & hc, pair_expect & pe, const char * name) {
    const int32_t dk = hc.fx.dk;
    const int32_t n_rows = (int32_t) pe.rows_qh.size();
    const int32_t n_cand = (int32_t) pe.cand_frag.size();
    if (n_rows < 1 || n_cand < 1) {
        std::fprintf(stderr, "FAIL %s:%d: %s empty pair\n", __FILE__, __LINE__, name);
        ++g_failures;
        return false;
    }
    std::vector<float> qpair((size_t) n_rows * n_cand * dk);
    std::vector<float> kbar((size_t) n_cand * dk);
    for (int32_t r = 0; r < n_rows; ++r) {
        for (int32_t j = 0; j < n_cand; ++j) {
            const float * qv = hotspot_qvec(hc, pe.rows_qh[(size_t) r], pe.cand_grp[(size_t) j]);
            float * dst = qpair.data() + ((size_t) r * n_cand + j) * dk;
            for (int32_t d = 0; d < dk; ++d) {
                dst[d] = qv[d];
            }
        }
    }
    for (int32_t j = 0; j < n_cand; ++j) {
        const float * kb = hotspot_kbar(hc, pe.cand_frag[(size_t) j], (int32_t) pe.kh);
        float * dst = kbar.data() + (size_t) j * dk;
        for (int32_t d = 0; d < dk; ++d) {
            dst[d] = kb[d];
        }
    }
    // Caller excludes partial columns from the energy via the mask (frozen
    // single policy); all hotspot/dense-tail/multiphase pairs here are
    // all-full, so this mask is all-ones.
    std::vector<int32_t> pvmask((size_t) n_rows * n_cand, 1);
    for (int32_t r = 0; r < n_rows; ++r) {
        for (int32_t j = 0; j < n_cand; ++j) {
            if (!pe.cand_full[(size_t) j]) {
                pvmask[(size_t) r * n_cand + j] = 0;
            }
        }
    }
    const sel_oracle::select_result oracle = sel_oracle::select_tile(
            qpair.data(), dk, n_rows, kbar.data(), dk,
            pe.cand_mand.data(), pe.cand_full.data(), pvmask.data(), n_cand,
            dk, hc.scale, 0.0f, hc.alpha, 0, 64, 64);
    if (oracle.rc != sel_oracle::OK) {
        std::fprintf(stderr, "FAIL %s:%d: %s oracle rc=%d\n", __FILE__, __LINE__, name, oracle.rc);
        ++g_failures;
        return false;
    }
    pe.want_exact.clear();
    pe.want_proxy.clear();
    for (int32_t p : oracle.exact) {
        pe.want_exact.push_back(pe.cand_use[(size_t) p]);
    }
    for (int32_t p : oracle.proxy) {
        pe.want_proxy.push_back(pe.cand_use[(size_t) p]);
    }
    const ref_out ref = call_ref_select(qpair.data(), dk, n_rows, kbar.data(), dk,
            pe.cand_frag, pe.cand_mand, pe.cand_full, pvmask, dk, hc.scale, 0.0f, hc.alpha, 0, 64, 64);
    return check_oracle_vs_ref(name, oracle, ref, pe.cand_frag);
}

static void collect_pair(const hotspot_case & hc, int64_t tile, int64_t kh, pair_expect & pe) {
    const int32_t g = hc.fx.hq / hc.fx.hkv;
    pe.tile = tile;
    pe.kh = kh;
    pe.rows_qh.clear();
    for (int32_t qh = 0; qh < hc.fx.hq; ++qh) {
        if (qh / hc.fx.bm == tile && qh / g == kh) {
            pe.rows_qh.push_back(qh);
        }
    }
    pe.cand_frag.clear();
    pe.cand_use.clear();
    pe.cand_mand.clear();
    pe.cand_full.clear();
    pe.cand_grp.clear();
    const int64_t p = tile * hc.fx.hkv + kh;
    for (int32_t f = 0; f < hc.fx.n_frag; ++f) {
        const int32_t u = hc.fx.use_index[(size_t)(p * hc.fx.n_frag + f)];
        if (u < 0) {
            continue;
        }
        pe.cand_frag.push_back(f);
        pe.cand_use.push_back(u);
        const size_t k = (size_t)(p * hc.fx.n_frag + f);
        pe.cand_mand.push_back(hc.fx.use_mand[k] ? 1 : 0);
        pe.cand_full.push_back(hc.fx.use_sub[k] == hc.fx.cells_per_frag ? 1 : 0);
        pe.cand_grp.push_back(hc.fx.use_group[k]);
    }
}

// Fill a plan from solved pairs (increasing use-index order is preserved by
// collect_pair's frag order), accumulate stats, and return them.
static bool fill_plan_from_pairs(const hotspot_case & hc,
        const std::vector<pair_expect> & pairs, int32_t req_exact_all,
        std::vector<int32_t> & plan, struct ggml_flashprefill_plan_stats & st) {
    int64_t max_sel = 0;
    if (ggml_flashprefill_plan_max_sel_for_meta(hc.fx.meta.data(), hc.fx.meta_words, &max_sel) !=
        GGML_FLASHPREFILL_OK) {
        return false;
    }
    int64_t pwords = 0;
    if (ggml_flashprefill_plan_words(hc.fx.n_tiles, hc.fx.hkv, max_sel, &pwords) != GGML_FLASHPREFILL_OK) {
        return false;
    }
    plan.assign((size_t) pwords, 0);
    if (ggml_flashprefill_plan_init(plan.data(), pwords, hc.fx.n_tiles, hc.fx.hkv, max_sel, req_exact_all) !=
        GGML_FLASHPREFILL_OK) {
        return false;
    }
    // Default every pair to empty lists (covers row-less pairs); solved
    // pairs overwrite with their explicit complement.
    for (int64_t t = 0; t < hc.fx.n_tiles; ++t) {
        for (int64_t h = 0; h < hc.fx.hkv; ++h) {
            if (ggml_flashprefill_plan_set_counts(plan.data(), pwords, t, h, 0, 0) != GGML_FLASHPREFILL_OK) {
                return false;
            }
        }
    }
    for (const pair_expect & pe : pairs) {
        const std::vector<int32_t> ex = req_exact_all ? pe.cand_use : pe.want_exact;
        const std::vector<int32_t> pr = req_exact_all ? std::vector<int32_t>() : pe.want_proxy;
        // Union of both roles must be exactly the pair's use set, disjoint.
        std::vector<int32_t> both = ex;
        both.insert(both.end(), pr.begin(), pr.end());
        std::sort(both.begin(), both.end());
        std::vector<int32_t> sorted_uses = pe.cand_use;
        std::sort(sorted_uses.begin(), sorted_uses.end());
        if (both != sorted_uses) {
            return false;
        }
        if (ggml_flashprefill_plan_set_counts(plan.data(), pwords, pe.tile, pe.kh,
                (int64_t) ex.size(), (int64_t) pr.size()) != GGML_FLASHPREFILL_OK) {
            return false;
        }
        for (size_t s = 0; s < ex.size(); ++s) {
            if (ggml_flashprefill_plan_set_entry(plan.data(), pwords, pe.tile, pe.kh,
                    GGML_FLASHPREFILL_ROLE_EXACT, (int64_t) s, ex[s]) != GGML_FLASHPREFILL_OK) {
                return false;
            }
        }
        for (size_t s = 0; s < pr.size(); ++s) {
            if (ggml_flashprefill_plan_set_entry(plan.data(), pwords, pe.tile, pe.kh,
                    GGML_FLASHPREFILL_ROLE_PROXY, (int64_t) s, pr[s]) != GGML_FLASHPREFILL_OK) {
                return false;
            }
        }
    }
    if (ggml_flashprefill_plan_validate(plan.data(), pwords) != GGML_FLASHPREFILL_OK) {
        return false;
    }
    if (ggml_flashprefill_plan_validate_against_metadata(plan.data(), pwords,
            hc.fx.meta.data(), hc.fx.meta_words) != GGML_FLASHPREFILL_OK) {
        return false;
    }
    if (ggml_flashprefill_plan_accumulate_stats(plan.data(), pwords,
            hc.fx.meta.data(), hc.fx.meta_words) != GGML_FLASHPREFILL_OK) {
        return false;
    }
    return ggml_flashprefill_plan_get_stats(plan.data(), pwords, &st) == GGML_FLASHPREFILL_OK;
}

static void test_hotspot_mandatory_plan() {
    std::puts("--- hotspots differ per head; sink/local mandatory; plan stats ---");
    // All-full uses (no partials): energy rule is unambiguous. frag0 = sink
    // mandatory, frag3 = local mandatory on both pairs.
    hotspot_case hc;
    hc.fx.dk = 8;
    hc.fx.dv = 4;
    hc.fx.hkv = 2;
    hc.fx.ng = 1;
    hc.fx.hq = 4;
    hc.fx.n_frag = 4;
    hc.fx.cells_per_frag = 2;
    hc.fx.bm = 4; // 1 tile: pairs (0,0) rows{qh0,qh1}, (0,1) rows{qh2,qh3}
    hc.fx.row_flags = {0, 0, 0, 0};
    hc.fx.use_mand = {
        1, 0, 0, 1, // pair (0,0): sink frag0, local frag3
        1, 0, 0, 1, // pair (0,1)
    };
    hc.fx.use_sub = {2, 2, 2, 2, 2, 2, 2, 2};
    hc.fx.use_group = {0, 0, 0, 0, 0, 0, 0, 0};
    SELECT_CHECK(build_sq_meta(hc.fx) == GGML_FLASHPREFILL_OK);
    SELECT_CHECK(hc.fx.n_tiles == 1);
    hotspot_fill(hc);
    pair_expect p0, p1;
    collect_pair(hc, 0, 0, p0);
    collect_pair(hc, 0, 1, p1);
    SELECT_CHECK(solve_pair(hc, p0, "hotspot-p00"));
    SELECT_CHECK(solve_pair(hc, p1, "hotspot-p01"));
    // Hand-verified energies (dk=8, |X|^2=10.75, alpha=0.1):
    // pair0 rows -> frag0: exact uses {u0(f0),u3(f3)}, proxy {u1,u2}.
    // pair1 rows -> frag2: exact {u4(f0,mandatory-over-low-energy),u6(f2),u7(f3)}, proxy {u5}.
    SELECT_CHECK_MSG(p0.want_exact == (std::vector<int32_t>{0, 3}), "p0 exact");
    SELECT_CHECK_MSG(p0.want_proxy == (std::vector<int32_t>{1, 2}), "p0 proxy");
    SELECT_CHECK_MSG(p1.want_exact == (std::vector<int32_t>{4, 6, 7}), "p1 exact");
    SELECT_CHECK_MSG(p1.want_proxy == (std::vector<int32_t>{5}), "p1 proxy");
    // Deliberately different hotspots: frag2's use is proxy under kv head 0
    // but exact under kv head 1.
    SELECT_CHECK_MSG(std::find(p0.want_proxy.begin(), p0.want_proxy.end(), 2) != p0.want_proxy.end(),
            "frag2 proxied under head 0");
    SELECT_CHECK_MSG(std::find(p1.want_exact.begin(), p1.want_exact.end(), 6) != p1.want_exact.end(),
            "frag2 exact under head 1");
    // Mandatory sink frag0 is exact under head 1 even though its energy is
    // far below threshold there (mandatory overrides energy, never the
    // reverse). Prove the energy is low: without the flag it would proxy.
    {
        pair_expect p1_nomand = p1;
        p1_nomand.cand_mand = {0, 0, 0, 0};
        const int32_t dk = hc.fx.dk;
        std::vector<float> qpair((size_t) 2 * 4 * dk);
        std::vector<float> kbar((size_t) 4 * dk);
        for (int32_t r = 0; r < 2; ++r) {
            for (int32_t j = 0; j < 4; ++j) {
                const float * qv = hotspot_qvec(hc, p1_nomand.rows_qh[(size_t) r], 0);
                float * dst = qpair.data() + ((size_t) r * 4 + j) * dk;
                for (int32_t d = 0; d < dk; ++d) {
                    dst[d] = qv[d];
                }
                const float * kb = hotspot_kbar(hc, j, 1);
                for (int32_t d = 0; d < dk; ++d) {
                    kbar[(size_t) j * dk + d] = kb[d];
                }
            }
        }
        const std::vector<int32_t> nmask(8, 1);
        const sel_oracle::select_result o = sel_oracle::select_tile(qpair.data(), dk, 2,
                kbar.data(), dk, p1_nomand.cand_mand.data(), p1_nomand.cand_full.data(), nmask.data(), 4,
                dk, hc.scale, 0.0, hc.alpha, 0, 64, 64);
        SELECT_CHECK_MSG(std::find(o.proxy.begin(), o.proxy.end(), 0) != o.proxy.end(),
                "unflagged sink frag would proxy under head 1 (energy low)");
    }
    // max_sel is the actual per-pair max (4), not global n_use (8).
    int64_t max_sel = -1;
    SELECT_CHECK(ggml_flashprefill_plan_max_sel_for_meta(hc.fx.meta.data(), hc.fx.meta_words, &max_sel) ==
            GGML_FLASHPREFILL_OK);
    SELECT_CHECK_MSG(max_sel == 4, "max_sel %lld want 4 (not global 8)", (long long) max_sel);
    std::vector<int32_t> plan;
    struct ggml_flashprefill_plan_stats st = {};
    SELECT_CHECK_MSG(fill_plan_from_pairs(hc, {p0, p1}, 0, plan, st), "plan fill+validate+stats");
    SELECT_CHECK_MSG(st.error == 0, "plan error %d", st.error);
    SELECT_CHECK_MSG(st.selected_total == 5 && st.corrected_total == 3,
            "selected=%d corrected=%d want 5/3", st.selected_total, st.corrected_total);
    SELECT_CHECK_MSG(st.visible_tokens == 16 && st.exact_tokens == 10,
            "visible=%lld exact=%lld want 16/10", (long long) st.visible_tokens, (long long) st.exact_tokens);
    SELECT_CHECK_MSG(st.sparse_rows == 4 && st.dense_rows == 0,
            "sparse=%d dense=%d want 4/0", st.sparse_rows, st.dense_rows);
    std::puts("hotspot plan: PASS");
}

static void test_dense_tail_plan() {
    std::puts("--- dense tail: BM-split tiles, DENSE_FORCE rows all-exact ---");
    // BM=2 with G=2: tile=qh/2 gives tile0={qh0,qh1} under kvh0 and
    // tile1={qh2,qh3} under kvh1. Pairs (0,0),(1,1) own rows; (0,1),(1,0)
    // are row-less with explicitly empty plan lists.
    hotspot_case hc;
    hc.fx.dk = 8;
    hc.fx.dv = 4;
    hc.fx.hkv = 2;
    hc.fx.ng = 1;
    hc.fx.hq = 4;
    hc.fx.n_frag = 4;
    hc.fx.cells_per_frag = 2;
    hc.fx.bm = 2;
    hc.fx.row_flags = {0, 0, GGML_FLASHPREFILL_ROW_FLAG_DENSE_FORCE, GGML_FLASHPREFILL_ROW_FLAG_DENSE_FORCE};
    // pairs (0,0),(0,1),(1,0),(1,1) x 4 frags; row-less pairs' flags unused.
    hc.fx.use_mand = {
        1, 0, 0, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
        1, 1, 1, 1, // dense tail triple: every matched use mandatory
    };
    hc.fx.use_sub = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    hc.fx.use_group = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    SELECT_CHECK(build_sq_meta(hc.fx) == GGML_FLASHPREFILL_OK);
    SELECT_CHECK(hc.fx.n_tiles == 2);
    hotspot_fill(hc);
    pair_expect p00, p11;
    collect_pair(hc, 0, 0, p00);
    collect_pair(hc, 1, 1, p11);
    SELECT_CHECK(solve_pair(hc, p00, "densetail-p00"));
    SELECT_CHECK(solve_pair(hc, p11, "densetail-p11"));
    SELECT_CHECK_MSG(p00.want_exact == (std::vector<int32_t>{0, 3}), "tail p00 exact");
    SELECT_CHECK_MSG(p00.want_proxy == (std::vector<int32_t>{1, 2}), "tail p00 proxy");
    SELECT_CHECK_MSG(p11.want_exact == (std::vector<int32_t>{4, 5, 6, 7}), "dense tail all exact");
    SELECT_CHECK_MSG(p11.want_proxy.empty(), "dense tail proxy empty");
    std::vector<int32_t> plan;
    struct ggml_flashprefill_plan_stats st = {};
    SELECT_CHECK_MSG(fill_plan_from_pairs(hc, {p00, p11}, 0, plan, st), "tail plan fill");
    SELECT_CHECK_MSG(st.sparse_rows == 2 && st.dense_rows == 2,
            "sparse=%d dense=%d want 2/2", st.sparse_rows, st.dense_rows);
    SELECT_CHECK_MSG(st.selected_total == 6 && st.corrected_total == 2,
            "selected=%d corrected=%d want 6/2", st.selected_total, st.corrected_total);
    SELECT_CHECK_MSG(st.visible_tokens == 16 && st.exact_tokens == 12,
            "visible=%lld exact=%lld want 16/12", (long long) st.visible_tokens, (long long) st.exact_tokens);
    std::puts("dense tail: PASS");
}

static void test_multiphase_qgroup() {
    std::puts("--- multiphase: per-use q_group selects the Q phase ---");
    // hq=1, hkv=1 (G=1), ng=2 phases, F=2 frags, single row qh0. Uses carry
    // different q_groups; Q(h0,0)=+X matches both kbars while Q(h0,1)=-X
    // mismatches kbarB=+X. Flipping use B's group flips its keep decision.
    hotspot_case hc;
    hc.fx.dk = 8;
    hc.fx.dv = 4;
    hc.fx.hkv = 1;
    hc.fx.ng = 2;
    hc.fx.hq = 1;
    hc.fx.n_frag = 2;
    hc.fx.cells_per_frag = 2;
    hc.fx.bm = 4;
    hc.fx.row_flags = {0};
    hc.fx.use_mand = {0, 0};
    hc.fx.use_sub = {2, 2};
    hc.fx.use_group = {0, 1}; // fragA phase 0, fragB phase 1
    hc.alpha = 0.1f;
    SELECT_CHECK(build_sq_meta(hc.fx) == GGML_FLASHPREFILL_OK);
    const int32_t dk = hc.fx.dk, dv = hc.fx.dv;
    hc.Q.assign((size_t) 1 * 2 * dk, 0.0f);
    hc.pool.assign((size_t) 2 * 1 * (dk + dv), 0.0f);
    for (int32_t d = 0; d < dk; ++d) {
        const float x = (d == 0) ? 3.0f : -0.5f;
        hc.Q[(0 * 2 + 0) * dk + d] = x;  // Q(h0,grp0)=+X
        hc.Q[(0 * 2 + 1) * dk + d] = -x; // Q(h0,grp1)=-X
        hc.pool[(0 * 1 + 0) * (dk + dv) + d] = x; // kbarA=+X
        hc.pool[(1 * 1 + 0) * (dk + dv) + d] = x; // kbarB=+X
    }
    pair_expect pe;
    collect_pair(hc, 0, 0, pe);
    SELECT_CHECK(solve_pair(hc, pe, "multiphase-mixed"));
    // z_A = +10.75 (phase 0), z_B = -10.75 (phase 1): A exact, B proxy.
    SELECT_CHECK_MSG(pe.want_exact == (std::vector<int32_t>{0}), "mixed phases exact={A}");
    SELECT_CHECK_MSG(pe.want_proxy == (std::vector<int32_t>{1}), "mixed phases proxy={B}");
    // Flip use B to phase 0: both +10.75 tie -> both exact. The q_group
    // mapping observably routes the Q phase per use.
    hc.fx.use_group = {0, 0};
    SELECT_CHECK(build_sq_meta(hc.fx) == GGML_FLASHPREFILL_OK);
    pair_expect pe2;
    collect_pair(hc, 0, 0, pe2);
    SELECT_CHECK(solve_pair(hc, pe2, "multiphase-flipped"));
    SELECT_CHECK_MSG(pe2.want_exact == (std::vector<int32_t>{0, 1}), "flipped phases both exact");
    SELECT_CHECK_MSG(pe2.want_proxy.empty(), "flipped phases proxy empty");
    std::puts("multiphase: PASS");
}

static void test_exactall_correctionoff_plan() {
    std::puts("--- exact-all and correction-off plan complements ---");
    hotspot_case hc;
    hc.fx.dk = 8;
    hc.fx.dv = 4;
    hc.fx.hkv = 2;
    hc.fx.ng = 1;
    hc.fx.hq = 4;
    hc.fx.n_frag = 4;
    hc.fx.cells_per_frag = 2;
    hc.fx.bm = 4;
    hc.fx.row_flags = {0, 0, 0, 0};
    hc.fx.use_mand = {1, 0, 0, 1, 1, 0, 0, 1};
    hc.fx.use_sub = {2, 2, 2, 2, 2, 2, 2, 2};
    hc.fx.use_group = {0, 0, 0, 0, 0, 0, 0, 0};
    SELECT_CHECK(build_sq_meta(hc.fx) == GGML_FLASHPREFILL_OK);
    hotspot_fill(hc);
    pair_expect p0, p1;
    collect_pair(hc, 0, 0, p0);
    collect_pair(hc, 0, 1, p1);
    SELECT_CHECK(solve_pair(hc, p0, "exactall-p00"));
    SELECT_CHECK(solve_pair(hc, p1, "exactall-p01"));
    // exact-all: every legal use exact, proxy lists explicitly empty, echo
    // bit set; stats show zero sparse rows and full exact tokens.
    std::vector<int32_t> plan_all;
    struct ggml_flashprefill_plan_stats st_all = {};
    SELECT_CHECK_MSG(fill_plan_from_pairs(hc, {p0, p1}, 1, plan_all, st_all), "exact-all fill");
    SELECT_CHECK_MSG(st_all.req_exact_all == 1, "exact-all echo %d", st_all.req_exact_all);
    SELECT_CHECK_MSG(st_all.corrected_total == 0 && st_all.selected_total == 8,
            "exact-all selected=%d corrected=%d want 8/0", st_all.selected_total, st_all.corrected_total);
    SELECT_CHECK_MSG(st_all.visible_tokens == 16 && st_all.exact_tokens == 16,
            "exact-all visible=%lld exact=%lld", (long long) st_all.visible_tokens,
            (long long) st_all.exact_tokens);
    SELECT_CHECK_MSG(st_all.sparse_rows == 0 && st_all.dense_rows == 4,
            "exact-all sparse=%d dense=%d", st_all.sparse_rows, st_all.dense_rows);
    // correction-off: the plan contract is unchanged — the complement is
    // still explicitly listed (ATTN skips the proxy correction, the plan
    // does not drop it). Same solved sets validate identically.
    std::vector<int32_t> plan_off;
    struct ggml_flashprefill_plan_stats st_off = {};
    SELECT_CHECK_MSG(fill_plan_from_pairs(hc, {p0, p1}, 0, plan_off, st_off), "correction-off fill");
    SELECT_CHECK_MSG(st_off.selected_total == 5 && st_off.corrected_total == 3,
            "correction-off still lists complement: %d/%d", st_off.selected_total, st_off.corrected_total);
    // Flag accessors on a constructed (not yet executed) SELECT tensor.
    {
        ggml_init_params params = {(size_t) 16 * 1024 * 1024, nullptr, true};
        ggml_context_ptr ctx(ggml_init(params));
        SELECT_CHECK(bool(ctx));
        ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, hc.fx.dk, 1, hc.fx.hq, 1);
        ggml_tensor * pool = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32,
                hc.fx.dk + hc.fx.dv, hc.fx.hkv, hc.fx.n_frag);
        ggml_tensor * meta = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, hc.fx.meta_words);
        SELECT_CHECK(q && pool && meta);
        ggml_tensor * sel_on = ggml_flash_prefill_select(ctx.get(), q, pool, meta,
                hc.fx.n_tiles, hc.fx.hkv, 4, 1.0f, 0.1f, 0.0f, 0, true);
        SELECT_CHECK(sel_on);
        SELECT_CHECK_MSG(ggml_flash_prefill_select_get_mean_correction(sel_on), "ctor true sticks");
        ggml_flash_prefill_select_set_mean_correction(sel_on, false);
        SELECT_CHECK_MSG(!ggml_flash_prefill_select_get_mean_correction(sel_on), "setter clears");
        ggml_tensor * sel_off = ggml_flash_prefill_select(ctx.get(), q, pool, meta,
                hc.fx.n_tiles, hc.fx.hkv, 4, 1.0f, 0.1f, 0.0f, 0, false);
        SELECT_CHECK(sel_off);
        SELECT_CHECK_MSG(!ggml_flash_prefill_select_get_mean_correction(sel_off), "ctor false sticks");
    }
    std::puts("exact-all/correction-off: PASS");
}

static void test_plan_role_rejections() {
    std::puts("--- plan role rejections: partial/mandatory never proxy ---");
    // Partial use listed as proxy -> PROXY_NOT_FULL; mandatory use listed
    // as proxy -> MANDATORY_AS_PROXY.
    sq_fixture fx;
    fx.dk = 8;
    fx.dv = 4;
    fx.hkv = 1;
    fx.ng = 1;
    fx.hq = 2;
    fx.n_frag = 2;
    fx.cells_per_frag = 2;
    fx.bm = 2;
    fx.row_flags = {0, 0};
    fx.use_mand = {1, 0}; // frag0 mandatory, frag1 energy
    fx.use_sub = {2, 1};  // frag1 partial (sub-cell only)
    fx.use_group = {0, 0};
    SELECT_CHECK(build_sq_meta(fx) == GGML_FLASHPREFILL_OK);
    int64_t max_sel = 0;
    SELECT_CHECK(ggml_flashprefill_plan_max_sel_for_meta(fx.meta.data(), fx.meta_words, &max_sel) ==
            GGML_FLASHPREFILL_OK);
    SELECT_CHECK(max_sel == 2);
    int64_t pwords = 0;
    SELECT_CHECK(ggml_flashprefill_plan_words(1, 1, max_sel, &pwords) == GGML_FLASHPREFILL_OK);
    // Bad plan 1: partial use 1 as proxy.
    {
        std::vector<int32_t> plan((size_t) pwords, 0);
        SELECT_CHECK(ggml_flashprefill_plan_init(plan.data(), pwords, 1, 1, max_sel, 0) == GGML_FLASHPREFILL_OK);
        SELECT_CHECK(ggml_flashprefill_plan_set_counts(plan.data(), pwords, 0, 0, 1, 1) == GGML_FLASHPREFILL_OK);
        SELECT_CHECK(ggml_flashprefill_plan_set_entry(plan.data(), pwords, 0, 0,
                GGML_FLASHPREFILL_ROLE_EXACT, 0, 0) == GGML_FLASHPREFILL_OK);
        SELECT_CHECK(ggml_flashprefill_plan_set_entry(plan.data(), pwords, 0, 0,
                GGML_FLASHPREFILL_ROLE_PROXY, 0, 1) == GGML_FLASHPREFILL_OK);
        SELECT_CHECK_MSG(ggml_flashprefill_plan_validate_against_metadata(
                plan.data(), pwords, fx.meta.data(), fx.meta_words) == GGML_FLASHPREFILL_ERR_PROXY_NOT_FULL,
                "partial-as-proxy rejected");
    }
    // Bad plan 2: mandatory use 0 as proxy (make frag0 full-coverage proxy
    // attempt by listing use 0 proxy and use 1 exact).
    {
        std::vector<int32_t> plan((size_t) pwords, 0);
        SELECT_CHECK(ggml_flashprefill_plan_init(plan.data(), pwords, 1, 1, max_sel, 0) == GGML_FLASHPREFILL_OK);
        SELECT_CHECK(ggml_flashprefill_plan_set_counts(plan.data(), pwords, 0, 0, 1, 1) == GGML_FLASHPREFILL_OK);
        SELECT_CHECK(ggml_flashprefill_plan_set_entry(plan.data(), pwords, 0, 0,
                GGML_FLASHPREFILL_ROLE_EXACT, 0, 1) == GGML_FLASHPREFILL_OK);
        SELECT_CHECK(ggml_flashprefill_plan_set_entry(plan.data(), pwords, 0, 0,
                GGML_FLASHPREFILL_ROLE_PROXY, 0, 0) == GGML_FLASHPREFILL_OK);
        const int32_t rc = ggml_flashprefill_plan_validate_against_metadata(
                plan.data(), pwords, fx.meta.data(), fx.meta_words);
        SELECT_CHECK_MSG(rc == GGML_FLASHPREFILL_ERR_MANDATORY_AS_PROXY,
                "mandatory-as-proxy rejected, got %d (%s)", rc, ggml_flashprefill_strerror(rc));
    }
    std::puts("plan role rejections: PASS");
}

// ---------------------------------------------------------------------------
// Backend SELECT parity: oracle sets vs reference vs executed SELECT plan.
// ---------------------------------------------------------------------------
static std::string to_lower(std::string s) {
    for (char & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

static bool dev_name_matches(ggml_backend_dev_t dev, const std::string & needle) {
    const std::string name = to_lower(ggml_backend_dev_name(dev));
    return name.find(to_lower(needle)) != std::string::npos;
}

// Decode one executed plan through its own self-describing offset words
// ([3..9]: dims + exact/proxy/counts offsets) and compare every pair's
// explicit complement against the oracle's use-index sets.
static bool check_backend_plan(const char * dev_name,
        const std::vector<int32_t> & plan, const std::vector<pair_expect> & pairs,
        const struct ggml_flashprefill_plan_stats & want_st) {
    if (plan.size() < 24) {
        std::fprintf(stderr, "FAIL: %s plan too small (%zu words)\n", dev_name, plan.size());
        ++g_failures;
        return false;
    }
    const int64_t T = plan[3], H = plan[4], MS = plan[5];
    const int64_t eo = plan[6], po = plan[7], co = plan[8];
    bool ok = true;
    // Header self-consistency (frozen layout: header 24 + 2*T*H*MS + 2*T*H).
    // Dimension sanity first so the products below cannot overflow int64.
    if (T < 0 || T > 1024 || H < 0 || H > 1024 || MS < 0 || MS > 1048576 ||
        eo != 24 || po != 24 + T * H * MS ||
        co != po + T * H * MS || co + 2 * T * H != (int64_t) plan.size()) {
        std::fprintf(stderr, "FAIL %s:%d: %s plan header inconsistent T=%lld H=%lld MS=%lld eo=%lld po=%lld co=%lld size=%zu\n",
                __FILE__, __LINE__, dev_name, (long long) T, (long long) H, (long long) MS,
                (long long) eo, (long long) po, (long long) co, plan.size());
        ++g_failures;
        return false;
    }
    for (const pair_expect & pe : pairs) {
        const int64_t p = pe.tile * H + pe.kh;
        const int64_t ne = plan[(size_t)(co + p * 2)];
        const int64_t np = plan[(size_t)(co + p * 2 + 1)];
        if (ne < 0 || ne > MS || np < 0 || np > MS ||
            eo + p * MS + ne > (int64_t) plan.size() || po + p * MS + np > (int64_t) plan.size()) {
            std::fprintf(stderr, "FAIL %s:%d: %s pair(t=%lld,h=%lld) counts out of bounds ne=%lld np=%lld MS=%lld\n",
                    __FILE__, __LINE__, dev_name, (long long) pe.tile, (long long) pe.kh,
                    (long long) ne, (long long) np, (long long) MS);
            ++g_failures;
            ok = false;
            continue;
        }
        std::vector<int32_t> got_exact, got_proxy;
        for (int64_t s = 0; s < ne; ++s) {
            got_exact.push_back(plan[(size_t)(eo + p * MS + s)]);
        }
        for (int64_t s = 0; s < np; ++s) {
            got_proxy.push_back(plan[(size_t)(po + p * MS + s)]);
        }
        // Backend lists must be the same SETS (order-insensitive here: the
        // contract pins increasing order, checked right after).
        std::vector<int32_t> gx = got_exact, wx = pe.want_exact;
        std::vector<int32_t> gp = got_proxy, wp = pe.want_proxy;
        std::sort(gx.begin(), gx.end());
        std::sort(wx.begin(), wx.end());
        std::sort(gp.begin(), gp.end());
        std::sort(wp.begin(), wp.end());
        char tag[128];
        std::snprintf(tag, sizeof(tag), "%s pair(t=%lld,h=%lld) exact", dev_name,
                (long long) pe.tile, (long long) pe.kh);
        if (gx != wx) {
            report_int_sets(tag, got_exact, pe.want_exact);
            std::fprintf(stderr, "FAIL %s:%d: backend %s\n", __FILE__, __LINE__, tag);
            ++g_failures;
            ok = false;
        }
        std::snprintf(tag, sizeof(tag), "%s pair(t=%lld,h=%lld) proxy", dev_name,
                (long long) pe.tile, (long long) pe.kh);
        if (gp != wp) {
            report_int_sets(tag, got_proxy, pe.want_proxy);
            std::fprintf(stderr, "FAIL %s:%d: backend %s\n", __FILE__, __LINE__, tag);
            ++g_failures;
            ok = false;
        }
        if (!std::is_sorted(got_exact.begin(), got_exact.end()) ||
            !std::is_sorted(got_proxy.begin(), got_proxy.end())) {
            std::fprintf(stderr, "FAIL %s:%d: %s lists not in increasing use-index order\n",
                    __FILE__, __LINE__, dev_name);
            ++g_failures;
            ok = false;
        }
        (void) T;
    }
    struct ggml_flashprefill_plan_stats got_st = {};
    if (ggml_flashprefill_plan_get_stats(plan.data(), (int64_t) plan.size(), &got_st) !=
        GGML_FLASHPREFILL_OK) {
        std::fprintf(stderr, "FAIL %s:%d: %s get_stats failed\n", __FILE__, __LINE__, dev_name);
        ++g_failures;
        return false;
    }
    if (got_st.error != 0 || got_st.selected_total != want_st.selected_total ||
        got_st.corrected_total != want_st.corrected_total ||
        got_st.sparse_rows != want_st.sparse_rows || got_st.dense_rows != want_st.dense_rows ||
        got_st.visible_tokens != want_st.visible_tokens || got_st.exact_tokens != want_st.exact_tokens) {
        std::fprintf(stderr,
                "FAIL %s:%d: %s stats got err=%d sel=%d corr=%d sparse=%d dense=%d vis=%lld ex=%lld"
                " want err=0 sel=%d corr=%d sparse=%d dense=%d vis=%lld ex=%lld\n",
                __FILE__, __LINE__, dev_name, got_st.error, got_st.selected_total, got_st.corrected_total,
                got_st.sparse_rows, got_st.dense_rows,
                (long long) got_st.visible_tokens, (long long) got_st.exact_tokens,
                want_st.selected_total, want_st.corrected_total, want_st.sparse_rows, want_st.dense_rows,
                (long long) want_st.visible_tokens, (long long) want_st.exact_tokens);
        ++g_failures;
        ok = false;
    }
    return ok;
}

// Build the hotspot bm=4 graphs and run SELECT on one device. Returns true
// when the device executed and matched; sets skipped when the device cannot
// run SELECT (missing or unsupported op).
static void test_backend_repeated_fragments(ggml_backend_t backend) {
    // Two queries share all three fragments. The first two are proxies,
    // the last is exact. Writing proxy uses over the score scratch used to
    // overwrite the last fragment's score before the second query read it.
    const int dk = 8, dv = 8, nq = 2, hq = 2, nf = 3, nu = nq * nf;
    int64_t mw = 0;
    SELECT_CHECK(ggml_flashprefill_metadata_words(nf, nq * hq, nu, nf, &mw) == 0);
    std::vector<int32_t> meta((size_t) mw);
    SELECT_CHECK(ggml_flashprefill_metadata_init(meta.data(), mw, nf, nq * hq, nu, nf,
                dk, dv, 1, nq, hq) == 0);
    SELECT_CHECK(ggml_flashprefill_metadata_set_counts(meta.data(), mw, nf, nq * hq, nu, nf) == 0);
    for (int f = 0; f < nf; ++f) {
        SELECT_CHECK(ggml_flashprefill_metadata_set_frag(meta.data(), mw, f, f, 1, f, 0, 0) == 0);
        SELECT_CHECK(ggml_flashprefill_metadata_set_cell(meta.data(), mw, f, f) == 0);
    }
    for (int q = 0; q < nq; ++q) {
        for (int h = 0; h < hq; ++h) {
            SELECT_CHECK(ggml_flashprefill_metadata_set_row(meta.data(), mw, q * hq + h,
                        q, 0, q, 0, 0, nq, 0, h) == 0);
        }
        for (int f = 0; f < nf; ++f) {
            SELECT_CHECK(ggml_flashprefill_metadata_set_use(meta.data(), mw, q * nf + f,
                        f, 0, 0, q, f, 1, 0, q) == 0);
        }
    }
    SELECT_CHECK(ggml_flashprefill_metadata_validate(meta.data(), mw) == 0);
    ggml_context_ptr ctx(ggml_init({1024 * 1024, nullptr, true}));
    ggml_tensor * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, dk, nq, hq);
    ggml_tensor * pool = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, dk + dv, 1, nf);
    ggml_tensor * mt = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, mw);
    ggml_tensor * out = ggml_flash_prefill_select(ctx.get(), q, pool, mt, 1, 1, nu,
            1.0f, 0.5f, 0.0f, 0, true);
    if (!out || !ggml_backend_supports_op(backend, out)) {
        SELECT_CHECK_MSG(false, "repeated-fragment SELECT is unsupported");
        return;
    }
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buf) {
        SELECT_CHECK_MSG(false, "repeated-fragment allocation failed");
        return;
    }
    std::vector<float> qdata(dk * nq * hq, 0.0f), pdata((dk + dv) * nf, 0.0f);
    for (int i = 0; i < nq * hq; ++i) qdata[i * dk] = 1.0f;
    pdata[0] = -10.0f; pdata[dk + dv] = -1.0f; pdata[2 * (dk + dv)] = 1.0f;
    ggml_backend_tensor_set(q, qdata.data(), 0, qdata.size() * sizeof(float));
    ggml_backend_tensor_set(pool, pdata.data(), 0, pdata.size() * sizeof(float));
    ggml_backend_tensor_set(mt, meta.data(), 0, meta.size() * sizeof(int32_t));
    std::vector<int32_t> got((size_t) ggml_nelements(out), (int32_t) 0xa5a5a5a5u);
    ggml_backend_tensor_set(out, got.data(), 0, got.size() * sizeof(int32_t));
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, out);
    // Reuse the same graph and dirty output: neither header nor counters
    // may depend on caller initialization, including a change to exact-all.
    for (int run = 0; run < 3; ++run) {
        if (run == 2) {
            std::fill(qdata.begin(), qdata.end(), 0.0f); // all scores tie => all exact
            ggml_backend_tensor_set(q, qdata.data(), 0, qdata.size() * sizeof(float));
        }
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            SELECT_CHECK_MSG(false, "repeated-fragment graph compute failed");
            return;
        }
        ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(int32_t));
        if (ggml_flashprefill_plan_validate(got.data(), (int64_t) got.size()) != 0) {
            SELECT_CHECK_MSG(false, "repeated-fragment plan invalid (run=%d, error=%d)", run, got[10]);
            continue;
        }
        SELECT_CHECK(ggml_flashprefill_plan_validate_against_metadata(got.data(),
                    (int64_t) got.size(), meta.data(), mw) == 0);
        const std::vector<int32_t> exact = run == 2 ? std::vector<int32_t>{0, 1, 2, 3, 4, 5} :
                                                                    std::vector<int32_t>{2, 5};
        const std::vector<int32_t> proxy = run == 2 ? std::vector<int32_t>{} :
                                                                    std::vector<int32_t>{0, 1, 3, 4};
        SELECT_CHECK(got[got[8]] == (int32_t) exact.size());
        SELECT_CHECK(got[got[8] + 1] == (int32_t) proxy.size());
        SELECT_CHECK(std::equal(exact.begin(), exact.end(), got.begin() + got[6]));
        SELECT_CHECK(std::equal(proxy.begin(), proxy.end(), got.begin() + got[7]));
        ggml_flashprefill_plan_stats stats{};
        SELECT_CHECK(ggml_flashprefill_plan_get_stats(got.data(), (int64_t) got.size(), &stats) == 0);
        SELECT_CHECK(stats.visible_tokens == nu);
        SELECT_CHECK(stats.exact_tokens == (int64_t) exact.size());
        SELECT_CHECK(stats.sparse_rows == (run == 2 ? 0 : nq * hq));
    }
    // An invalid input must not leave yesterday's valid output header.
    // A subsequent valid run must recover without caller-side plan reset.
    const int32_t saved_magic = meta[0];
    meta[0] = 0;
    ggml_backend_tensor_set(mt, meta.data(), 0, meta.size() * sizeof(int32_t));
    // CPU exposes invalid input through the plan header. Vulkan also
    // rejects the graph at its mandatory plan-error readback boundary.
    const bool is_vulkan = std::string(ggml_backend_name(backend)).find("Vulkan") != std::string::npos;
    SELECT_CHECK(ggml_backend_graph_compute(backend, graph) ==
            (is_vulkan ? GGML_STATUS_FAILED : GGML_STATUS_SUCCESS));
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(int32_t));
    SELECT_CHECK(got[10] != GGML_FLASHPREFILL_OK);
    meta[0] = saved_magic;
    ggml_backend_tensor_set(mt, meta.data(), 0, meta.size() * sizeof(int32_t));
    SELECT_CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(int32_t));
    SELECT_CHECK(ggml_flashprefill_plan_validate_against_metadata(got.data(),
                (int64_t) got.size(), meta.data(), mw) == GGML_FLASHPREFILL_OK);
}

static bool run_backend_select(ggml_backend_dev_t dev, bool & skipped, std::string & skip_why) {
    skipped = false;
    const char * dev_name = ggml_backend_dev_name(dev);
    hotspot_case hc;
    hc.fx.dk = 8;
    hc.fx.dv = 4;
    hc.fx.hkv = 2;
    hc.fx.ng = 1;
    hc.fx.hq = 4;
    hc.fx.n_frag = 4;
    hc.fx.cells_per_frag = 2;
    hc.fx.bm = 4;
    hc.fx.row_flags = {0, 0, 0, 0};
    hc.fx.use_mand = {1, 0, 0, 1, 1, 0, 0, 1};
    hc.fx.use_sub = {2, 2, 2, 2, 2, 2, 2, 2};
    hc.fx.use_group = {0, 0, 0, 0, 0, 0, 0, 0};
    if (build_sq_meta(hc.fx) != GGML_FLASHPREFILL_OK) {
        std::fprintf(stderr, "FAIL %s:%d: backend fixture meta\n", __FILE__, __LINE__);
        ++g_failures;
        return false;
    }
    hotspot_fill(hc);
    pair_expect p0, p1;
    collect_pair(hc, 0, 0, p0);
    collect_pair(hc, 0, 1, p1);
    if (!solve_pair(hc, p0, "backend-p00") || !solve_pair(hc, p1, "backend-p01")) {
        return false;
    }
    int64_t max_sel = 0;
    if (ggml_flashprefill_plan_max_sel_for_meta(hc.fx.meta.data(), hc.fx.meta_words, &max_sel) !=
        GGML_FLASHPREFILL_OK) {
        std::fprintf(stderr, "FAIL %s:%d: max_sel\n", __FILE__, __LINE__);
        ++g_failures;
        return false;
    }
    ggml_init_params params = {(size_t) 64 * 1024 * 1024, nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        std::fprintf(stderr, "FAIL %s:%d: ggml_init\n", __FILE__, __LINE__);
        ++g_failures;
        return false;
    }
    // Agreed shapes: Q F32 [Dk, n_groups, Hq, 1]; pool F32 [Dk+Dv, Hkv, F].
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, hc.fx.dk, hc.fx.ng, hc.fx.hq, 1);
    ggml_tensor * pool = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32,
            hc.fx.dk + hc.fx.dv, hc.fx.hkv, hc.fx.n_frag);
    ggml_tensor * meta = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, hc.fx.meta_words);
    if (!q || !pool || !meta) {
        std::fprintf(stderr, "FAIL %s:%d: tensor alloc\n", __FILE__, __LINE__);
        ++g_failures;
        return false;
    }
    ggml_tensor * out = ggml_flash_prefill_select(ctx.get(), q, pool, meta,
            hc.fx.n_tiles, hc.fx.hkv, max_sel, hc.scale, hc.alpha, 0.0f, 0, true);
    if (!out) {
        std::fprintf(stderr, "FAIL %s:%d: select ctor\n", __FILE__, __LINE__);
        ++g_failures;
        return false;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        skipped = true;
        skip_why = "device init failed";
        return false;
    }
    if (!ggml_backend_supports_op(backend, out)) {
        skipped = true;
        skip_why = "SELECT op not supported (kernel landing later)";
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        std::fprintf(stderr, "FAIL %s:%d: %s buffer alloc\n", __FILE__, __LINE__, dev_name);
        ++g_failures;
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_tensor_set(q, hc.Q.data(), 0, hc.Q.size() * sizeof(float));
    ggml_backend_tensor_set(pool, hc.pool.data(), 0, hc.pool.size() * sizeof(float));
    ggml_backend_tensor_set(meta, hc.fx.meta.data(), 0, hc.fx.meta.size() * sizeof(int32_t));
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, out);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL %s:%d: %s graph compute\n", __FILE__, __LINE__, dev_name);
        ++g_failures;
        buffer.reset();
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_synchronize(backend);
    std::vector<int32_t> plan((size_t) ggml_nelements(out), 0);
    ggml_backend_tensor_get(out, plan.data(), 0, plan.size() * sizeof(int32_t));
    buffer.reset();
    test_backend_repeated_fragments(backend);
    ggml_backend_free(backend);
    struct ggml_flashprefill_plan_stats want_st = {};
    want_st.selected_total = 5;
    want_st.corrected_total = 3;
    want_st.sparse_rows = 4;
    want_st.dense_rows = 0;
    want_st.visible_tokens = 16;
    want_st.exact_tokens = 10;
    const bool ok = check_backend_plan(dev_name, plan, {p0, p1}, want_st);
    std::fprintf(stderr, "  backend %s: max_sel=%lld plan_words=%zu %s\n",
            dev_name, (long long) max_sel, plan.size(), ok ? "MATCH" : "MISMATCH");
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    std::string backend_name;
    bool required_backend = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (a == "--required-backend") {
            required_backend = true;
        } else if (a == "--help" || a == "-h") {
            std::printf("Usage: %s [--backend NAME] [--required-backend]\n"
                        "  --backend NAME: also verify SELECT on the backend device whose\n"
                        "    name contains NAME (e.g. Vulkan). Default is CPU only.\n"
                        "  --required-backend: a missing/unsupported selected backend is a\n"
                        "    failure (exit 2). Otherwise it is an explicit SKIP (exit 0)\n"
                        "    and CPU math/contract tests still run; CPU is never silently\n"
                        "    substituted for a requested GPU backend.\n",
                    argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }

    test_packgqa_mapping();
    test_energy_global_max_not_head_topk();
    test_threshold_ties_inclusive();
    test_extreme_finite_logits_stability();
    test_mandatory_partial_exactall_ref();
    test_partial_pollutant_isolation();
    test_capacity_overflow_no_truncation();
    test_mask_allzero_boundary();
    test_metadata_counts_and_padding();
    test_metadata_rejections();
    test_hotspot_mandatory_plan();
    test_dense_tail_plan();
    test_multiphase_qgroup();
    test_exactall_correctionoff_plan();
    test_plan_role_rejections();

    // Backend parity. Default: CPU device only. With --backend NAME: only
    // the named device (never a silent CPU substitution).
    ggml_backend_load_all();
    bool required_unmet = false;
    if (backend_name.empty()) {
        ggml_backend_dev_t cpu = nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                cpu = dev;
                break;
            }
        }
        if (!cpu) {
            std::printf("SKIP: no CPU backend device found\n");
            if (required_backend) {
                required_unmet = true;
            }
        } else {
            bool skipped = false;
            std::string why;
            std::puts("--- backend SELECT parity: CPU ---");
            if (!run_backend_select(cpu, skipped, why)) {
                if (skipped) {
                    std::printf("SKIP: CPU SELECT: %s\n", why.c_str());
                    if (required_backend) {
                        required_unmet = true;
                    }
                }
            } else {
                std::puts("backend CPU parity: PASS");
            }
        }
    } else {
        ggml_backend_dev_t chosen = nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (dev_name_matches(dev, backend_name)) {
                chosen = dev;
                break;
            }
        }
        if (!chosen) {
            std::printf("SKIP: backend '%s' not available (no matching device, %zu devices present)\n",
                    backend_name.c_str(), ggml_backend_dev_count());
            required_unmet = required_backend;
            if (required_backend) {
                std::printf("FAIL: --required-backend set and backend '%s' absent\n", backend_name.c_str());
            }
        } else {
            bool skipped = false;
            std::string why;
            std::printf("--- backend SELECT parity: %s ---\n", ggml_backend_dev_name(chosen));
            if (!run_backend_select(chosen, skipped, why)) {
                if (skipped) {
                    std::printf("SKIP: backend '%s' cannot run SELECT yet: %s\n",
                            ggml_backend_dev_name(chosen), why.c_str());
                    if (required_backend) {
                        std::printf("FAIL: --required-backend set and backend '%s' unsupported\n",
                                ggml_backend_dev_name(chosen));
                        required_unmet = true;
                    }
                }
            } else {
                std::printf("backend %s parity: PASS\n", ggml_backend_dev_name(chosen));
            }
        }
    }

    if (g_failures != 0) {
        std::printf("=== Results: %d failure(s) ===\n", g_failures);
        return 1;
    }
    if (required_unmet) {
        std::printf("=== Results: required backend unmet ===\n");
        return 2;
    }
    std::puts("=== Results: all selector tests passed (backend parity where supported) ===");
    return 0;
}
