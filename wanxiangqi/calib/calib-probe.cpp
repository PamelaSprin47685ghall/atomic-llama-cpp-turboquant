// Offline structural probe for the routed experts.
//
// This does not run the model. It reads the GGUF weights directly, dequantises
// selected expert slabs, and answers the questions that decide whether a codec
// can beat IQ by a first-order margin. IQ's coder is memoryless within a row, so
// every large win has to come from a dependence it cannot see:
//
//   1. along the input dim   -> already settled from the calibration dump alone:
//                               (1/2d)*log2(prod A_jj / det A) = 0.410 bpw, which
//                               is exactly what OBQ/GPTQ error feedback cashes.
//   2. across rows (neurons) -> low rank of W in the functional metric. Measured
//                               here as the spectrum of W Abar W^T.
//   3. across experts        -> measured here permutation-invariantly. The earlier
//                               "experts are near-orthogonal, max cos 0.0213" test
//                               compared matrices entry by entry, but neuron order
//                               inside an expert is arbitrary, so that test could
//                               not have detected shared structure even if it were
//                               total. W^T W is invariant to row permutation.
//
// Bias note: the source weights are already iq2_xs/iq3_xxs, so they carry
// broadband quantisation noise which fills in small singular values. Every
// low-rank number here is therefore pessimistic - if it looks low rank through
// that noise, it is low rank.

#include "ggml.h"
#include "gguf.h"

#include "calib-dump.h"
#include "gguf-slab.h"
#include "linalg.h"
#include "parallel.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <thread>
#include <vector>

using wxq::gemm_AB;
using wxq::gemm_ABt;
using wxq::syrk_AtA;
using wxq::eig_sym;
using wxq::chol_upper;

// ---------------------------------------------------------------------------
// analysis helpers
// ---------------------------------------------------------------------------

// rank at which the cumulative eigenvalue mass reaches `frac`
static int64_t rank_at(const std::vector<double> & ev, double frac) {
    double tot = 0.0;
    for (double v : ev) if (v > 0.0) tot += v;
    if (tot <= 0.0) return 0;
    double acc = 0.0;
    for (size_t i = 0; i < ev.size(); ++i) {
        if (ev[i] > 0.0) acc += ev[i];
        if (acc >= frac*tot) return (int64_t) i + 1;
    }
    return (int64_t) ev.size();
}

static double part_ratio(const std::vector<double> & ev) {
    double s = 0.0, s2 = 0.0;
    for (double v : ev) { if (v <= 0.0) continue; s += v; s2 += v*v; }
    return s2 > 0.0 ? s*s/s2 : 0.0;
}

// tail energy fraction beyond rank r
static double tail_frac(const std::vector<double> & ev, int64_t r) {
    double tot = 0.0, head = 0.0;
    for (size_t i = 0; i < ev.size(); ++i) {
        const double v = ev[i] > 0.0 ? ev[i] : 0.0;
        tot += v;
        if ((int64_t) i < r) head += v;
    }
    return tot > 0.0 ? 1.0 - head/tot : 0.0;
}

// relative quantiser MSE at a given bpw. One tunable; roughly matches Lloyd-Max
// scalar and the measured behaviour of the IQ codebooks.
static double delta_bpw(double b) { return std::min(1.0, 1.9*std::pow(2.0, -2.0*b)); }

static double kurtosis(const float * x, size_t n) {
    double m = 0.0; for (size_t i = 0; i < n; ++i) m += x[i]; m /= (double) n;
    double v = 0.0, q = 0.0;
    for (size_t i = 0; i < n; ++i) { const double d = x[i]-m; v += d*d; q += d*d*d*d; }
    v /= (double) n; q /= (double) n;
    return v > 0.0 ? q/(v*v) : 0.0;
}

static double maxrms(const float * x, size_t n) {
    double s = 0.0, mx = 0.0;
    for (size_t i = 0; i < n; ++i) { s += (double) x[i]*x[i]; mx = std::max(mx, (double) std::fabs(x[i])); }
    const double rms = std::sqrt(s/(double) n);
    return rms > 0.0 ? mx/rms : 0.0;
}

// 32-point Walsh-Hadamard with a fixed sign flip, applied blockwise. This is the
// incoherence step the TQ types already do and the IQ types do not.
static void wht32(float * x, size_t n) {
    static const uint32_t signs = 0x9E3779B9u;
    for (size_t off = 0; off + 32 <= n; off += 32) {
        float * b = x + off;
        for (int i = 0; i < 32; ++i) if ((signs >> (i & 31)) & 1u) b[i] = -b[i];
        for (int len = 1; len < 32; len <<= 1)
            for (int i = 0; i < 32; i += len<<1)
                for (int j = i; j < i+len; ++j) {
                    const float u = b[j], v = b[j+len];
                    b[j] = u+v; b[j+len] = u-v;
                }
        const float sc = 1.0f/std::sqrt(32.0f);
        for (int i = 0; i < 32; ++i) b[i] *= sc;
    }
}

static double pearson_log(const std::vector<double> & a, const std::vector<double> & b) {
    std::vector<double> x, y;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] > 0.0 && b[i] > 0.0) { x.push_back(std::log(a[i])); y.push_back(std::log(b[i])); }
    if (x.size() < 4) return 0.0;
    double mx = 0, my = 0;
    for (size_t i = 0; i < x.size(); ++i) { mx += x[i]; my += y[i]; }
    mx /= (double) x.size(); my /= (double) y.size();
    double sxy = 0, sxx = 0, syy = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        const double dx = x[i]-mx, dy = y[i]-my;
        sxy += dx*dy; sxx += dx*dx; syy += dy*dy;
    }
    return (sxx > 0 && syy > 0) ? sxy/std::sqrt(sxx*syy) : 0.0;
}

int main(int argc, char ** argv) {
    std::string model, calib = "/tmp/opencode/calib-full2.bin";
    std::vector<int> blocks = {2, 18, 30, 38};
    int n_exp_sample = 16;
    wxq::set_num_threads((int) std::max(1u, std::thread::hardware_concurrency()));

    for (int i = 1; i < argc-1; ++i) {
        if (!std::strcmp(argv[i], "--model"))   model = argv[++i];
        else if (!std::strcmp(argv[i], "--calib")) calib = argv[++i];
        else if (!std::strcmp(argv[i], "--threads")) wxq::set_num_threads(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--experts")) n_exp_sample = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--blocks")) {
            blocks.clear();
            for (char * p = std::strtok(argv[++i], ","); p; p = std::strtok(nullptr, ",")) blocks.push_back(std::atoi(p));
        }
    }
    if (model.empty()) { fprintf(stderr, "usage: %s --model X.gguf [--calib f] [--blocks 2,18] [--experts 16]\n", argv[0]); return 1; }

    printf("threads %d, blocks:", wxq::num_threads());
    for (int b : blocks) printf(" %d", b);
    printf(", experts sampled %d\n\n", n_exp_sample);

    // the spectrum pass needs the full pooled covariance, so cov is requested
    wxq::calib_dump dump;
    std::string cal_err;
    if (!wxq::load_calib(calib, blocks, /* want_cov */ true, dump, cal_err)) {
        fprintf(stderr, "%s\n", cal_err.c_str());
        return 1;
    }
    const wxq::calib_meta & cm = dump.meta;
    printf("calib: %u blocks, flags 0x%x, decoded %" PRIu64 "/%" PRIu64 ", max_ctx %u, %s\n",
           cm.n_entries, cm.flags, cm.n_decoded, cm.n_total, cm.max_ctx, cm.complete ? "complete" : "PARTIAL");
    const uint32_t flags = cm.flags;
    std::map<int, wxq::calib_layer> & cal = dump.layers;

    wxq::gguf_slab_reader gr;
    if (!gr.open(model)) return 1;

    printf("\n=== A. functional low-rank spectrum of W (metric = measured Abar / diag h) ===\n");
    printf("%-5s %-6s %-7s %8s %8s %8s %8s %8s %8s\n",
           "blk", "proj", "expert", "r90", "r95", "r99", "PR", "r90_unw", "PR_unw");

    struct acc {
        double r90=0, r95=0, r99=0, pr=0, r90u=0, pru=0;
        int n=0;
        int64_t n_in=0, n_out=0;
        std::vector<double> spec;  // mean spectrum, normalised to sum 1
    };
    std::map<std::string, acc> summary;

    for (int il : blocks) {
        auto it = cal.find(il);
        if (it == cal.end()) { printf("blk %d: no calib entry\n", il); continue; }
        wxq::calib_layer & lc = it->second;
        const int64_t d = lc.n_embd, nff = lc.n_ff, ne = lc.n_expert;

        // pick experts spread across the routing-frequency order
        std::vector<int> order(ne);
        for (int64_t i = 0; i < ne; ++i) order[i] = (int) i;
        std::sort(order.begin(), order.end(), [&](int a, int b){ return lc.counts[a] > lc.counts[b]; });
        std::vector<int> pick;
        for (int k = 0; k < n_exp_sample; ++k) pick.push_back(order[(size_t) k*(ne-1)/std::max(1, n_exp_sample-1)]);

        // Abar, normalised, upper triangle live -> full symmetric
        std::vector<float> Abar;
        if (!lc.cov.empty() && lc.n_cov > 0) {
            Abar.resize((size_t) d*d);
            const double s = 1.0/(double) lc.n_cov;
            for (int64_t i = 0; i < d; ++i)
                for (int64_t j = 0; j < d; ++j)
                    Abar[i*d+j] = (float) (lc.cov[(j>=i? i*d+j : j*d+i)] * s);
        }

        const char * projs[3] = { "gate", "up", "down" };
        for (int p = 0; p < 3; ++p) {
            const std::string tname = "blk." + std::to_string(il) + ".ffn_" + projs[p] + "_exps.weight";
            for (int ei : pick) {
                std::vector<float> W;
                int64_t ne0, nrows; ggml_type ty;
                if (!gr.slab(tname, ei, W, ne0, nrows, ty)) { p = 3; break; }

                std::vector<double> G, ev, evu;
                if (p < 2) {
                    // W is [nrows=n_ff, ne0=n_embd] math-wise: row = neuron
                    if (Abar.empty()) continue;
                    std::vector<float> T((size_t) nrows*ne0);
                    gemm_AB(T.data(), W.data(), Abar.data(), nrows, ne0, ne0);
                    std::vector<float> Gf((size_t) nrows*nrows);
                    gemm_ABt(Gf.data(), T.data(), W.data(), nrows, ne0, nrows);
                    G.assign(Gf.begin(), Gf.end());
                    eig_sym(G, nrows, ev);
                    gemm_ABt(Gf.data(), W.data(), W.data(), nrows, ne0, nrows);
                    G.assign(Gf.begin(), Gf.end());
                    eig_sym(G, nrows, evu);
                } else {
                    // down: [nrows=n_embd, ne0=n_ff], contraction is the hidden dim,
                    // so the metric is the measured per-expert per-neuron energy
                    std::vector<float> U((size_t) ne0*ne0);
                    syrk_AtA(U.data(), W.data(), nrows, ne0);
                    std::vector<double> h(ne0, 1.0);
                    if (!lc.hid_sum.empty() && lc.hid_n[ei] > 0) {
                        const double inv = 1.0/(double) lc.hid_n[ei];
                        for (int64_t j = 0; j < ne0; ++j) h[j] = std::sqrt(std::max(0.0, lc.hid_sum[(size_t) ei*ne0 + j]*inv));
                    }
                    G.resize((size_t) ne0*ne0);
                    for (int64_t i = 0; i < ne0; ++i)
                        for (int64_t j = 0; j < ne0; ++j) G[i*ne0+j] = (double) U[i*ne0+j]*h[i]*h[j];
                    eig_sym(G, ne0, ev);
                    G.assign(U.begin(), U.end());
                    eig_sym(G, ne0, evu);
                }

                const int64_t n = (int64_t) ev.size();
                const std::string key = std::string(projs[p]) + ":" + std::to_string(il);
                acc & A = summary[key];
                A.r90 += (double) rank_at(ev,0.90); A.r95 += (double) rank_at(ev,0.95);
                A.r99 += (double) rank_at(ev,0.99); A.pr += part_ratio(ev);
                A.r90u += (double) rank_at(evu,0.90); A.pru += part_ratio(evu); A.n++;
                A.n_in  = (p < 2) ? ne0   : nrows;   // contraction dim
                A.n_out = (p < 2) ? nrows : ne0;
                {
                    // accumulate the normalised spectrum so B can build a real
                    // rate-distortion frontier instead of a rank threshold
                    double tot = 0.0;
                    for (double v : ev) if (v > 0.0) tot += v;
                    if (tot > 0.0) {
                        if (A.spec.size() != (size_t) n) A.spec.assign((size_t) n, 0.0);
                        for (int64_t i = 0; i < n; ++i) A.spec[i] += (ev[i] > 0.0 ? ev[i] : 0.0)/tot;
                    }
                }
                if (ei == pick.front() || ei == pick.back())
                    printf("%-5d %-6s %-7d %8" PRId64 " %8" PRId64 " %8" PRId64 " %8.1f %8" PRId64 " %8.1f   (n=%" PRId64 ")\n",
                           il, projs[p], ei, rank_at(ev,0.90), rank_at(ev,0.95), rank_at(ev,0.99),
                           part_ratio(ev), rank_at(evu,0.90), part_ratio(evu), n);
                (void) nff;
            }
        }
    }

    printf("\n--- mean over sampled experts ---\n");
    printf("%-12s %8s %8s %8s %8s %8s %8s\n", "proj:blk", "r90", "r95", "r99", "PR", "r90_unw", "PR_unw");
    for (auto & [k, A] : summary)
        if (A.n) printf("%-12s %8.0f %8.0f %8.0f %8.1f %8.0f %8.1f\n",
                        k.c_str(), A.r90/A.n, A.r95/A.n, A.r99/A.n, A.pr/A.n, A.r90u/A.n, A.pru/A.n);

    printf("\n=== B. rank x bpw rate-distortion frontier vs full-rank IQ ===\n");
    printf("low-rank eq-bpw = r*(n_in+n_out)*bpw / (n_in*n_out)\n");
    printf("rel-err = tail(r) + 2*delta(bpw)*(1-tail(r))   [two factors, so 2x]\n");
    printf("full-rank rel-err = delta(bpw),  delta(b) = 1.9*2^-2b\n");
    printf("verdict compares the best low-rank point at <= the IQ eq-bpw.\n\n");
    printf("%-12s %8s %8s %6s %6s %9s %9s %8s\n",
           "proj:blk", "IQ bpw", "IQ err", "rank", "bpw", "eq-bpw", "LR err", "win");
    for (auto & [k, A] : summary) {
        if (!A.n || A.spec.empty()) continue;
        std::vector<double> sp = A.spec;
        double tot = 0.0;
        for (double v : sp) tot += v;
        if (tot <= 0.0) continue;
        for (double & v : sp) v /= tot;
        const double npar = (double) A.n_in * (double) A.n_out;
        const double nfac = (double) A.n_in + (double) A.n_out;

        for (double iqb : {2.3125, 3.0625}) {
            const double iq_err = delta_bpw(iqb);
            double best_err = 1e30; int64_t best_r = 0; double best_b = 0, best_eq = 0;
            for (int64_t r = 16; r <= (int64_t) sp.size(); r += 16) {
                for (double b : {2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0}) {
                    const double eq = r*nfac*b/npar;
                    if (eq > iqb) continue;
                    const double tf = tail_frac(sp, r);
                    const double err = tf + 2.0*delta_bpw(b)*(1.0 - tf);
                    if (err < best_err) { best_err = err; best_r = r; best_b = b; best_eq = eq; }
                }
            }
            if (best_r)
                printf("%-12s %8.4f %8.4f %6" PRId64 " %6.1f %9.4f %9.4f %8s\n",
                       k.c_str(), iqb, iq_err, best_r, best_b, best_eq, best_err,
                       best_err < iq_err ? "LOW-RANK" : "IQ");
            else
                printf("%-12s %8.4f %8.4f %6s %6s %9s %9s %8s\n",
                       k.c_str(), iqb, iq_err, "-", "-", "-", "-", "IQ");
        }
    }

    printf("\n=== C. cross-expert similarity, permutation invariant ===\n");
    printf("S_e = W_e^T W_e is invariant to neuron reordering. cos_raw includes the\n");
    printf("isotropic part; cos_iso removes tr(S)/d * I, which is what makes the\n");
    printf("comparison informative.\n\n");
    printf("%-5s %-6s %10s %10s %10s\n", "blk", "proj", "cos_raw", "cos_iso", "vs_shexp");

    for (int il : blocks) {
        auto it = cal.find(il);
        if (it == cal.end()) continue;
        wxq::calib_layer & lc = it->second;
        const int64_t ne = lc.n_expert;
        std::vector<int> order(ne);
        for (int64_t i = 0; i < ne; ++i) order[i] = (int) i;
        std::sort(order.begin(), order.end(), [&](int a, int b){ return lc.counts[a] > lc.counts[b]; });
        const int k_pair = std::min(8, n_exp_sample);
        std::vector<int> pick;
        for (int k = 0; k < k_pair; ++k) pick.push_back(order[(size_t) k*(ne-1)/std::max(1, k_pair-1)]);

        for (int p = 0; p < 2; ++p) {
            const char * pn = p == 0 ? "gate" : "up";
            const std::string tname = "blk." + std::to_string(il) + ".ffn_" + pn + "_exps.weight";
            std::vector<std::vector<float>> S(pick.size());
            int64_t dd = 0;
            bool ok = true;
            for (size_t i = 0; i < pick.size() && ok; ++i) {
                std::vector<float> W; int64_t ne0, nrows; ggml_type ty;
                if (!gr.slab(tname, pick[i], W, ne0, nrows, ty)) { ok = false; break; }
                dd = ne0;
                S[i].resize((size_t) ne0*ne0);
                syrk_AtA(S[i].data(), W.data(), nrows, ne0);
            }
            if (!ok) continue;

            auto cosine = [&](const std::vector<float> & a, const std::vector<float> & b, bool iso) {
                double ta = 0, tb = 0;
                if (iso) { for (int64_t i = 0; i < dd; ++i) { ta += a[i*dd+i]; tb += b[i*dd+i]; } ta /= (double) dd; tb /= (double) dd; }
                double ab = 0, aa = 0, bb = 0;
                for (int64_t i = 0; i < dd; ++i)
                    for (int64_t j = 0; j < dd; ++j) {
                        const double x = a[i*dd+j] - (iso && i==j ? ta : 0.0);
                        const double y = b[i*dd+j] - (iso && i==j ? tb : 0.0);
                        ab += x*y; aa += x*x; bb += y*y;
                    }
                return (aa > 0 && bb > 0) ? ab/std::sqrt(aa*bb) : 0.0;
            };

            double cr = 0, ci = 0; int np = 0;
            for (size_t i = 0; i < S.size(); ++i)
                for (size_t j = i+1; j < S.size(); ++j) { cr += cosine(S[i],S[j],false); ci += cosine(S[i],S[j],true); np++; }

            double vs = 0.0;
            {
                const ggml_tensor * ts = gr.find("blk." + std::to_string(il) + ".ffn_" + pn + "_shexp.weight");
                if (ts) {
                    std::vector<float> Ws; int64_t ne0, nrows; ggml_type ty;
                    if (gr.slab("blk." + std::to_string(il) + ".ffn_" + pn + "_shexp.weight", 0, Ws, ne0, nrows, ty) && ne0 == dd) {
                        std::vector<float> Ss((size_t) dd*dd);
                        syrk_AtA(Ss.data(), Ws.data(), nrows, ne0);
                        for (auto & Se : S) vs += cosine(Se, Ss, true);
                        vs /= (double) S.size();
                    }
                }
            }
            if (np) printf("%-5d %-6s %10.4f %10.4f %10.4f\n", il, pn, cr/np, ci/np, vs);
        }
    }

    printf("\n=== D. weight distribution before / after 32-point WHT ===\n");
    printf("IQ uses a hardcoded universal codebook and no rotation. kurtosis 3.0 and\n");
    printf("a small max/rms is what a codebook wants to see.\n\n");
    printf("%-5s %-6s %10s %10s %10s %10s\n", "blk", "proj", "kurt", "max/rms", "kurt_wht", "mr_wht");
    for (int il : blocks) {
        const char * projs[3] = { "gate", "up", "down" };
        for (int p = 0; p < 3; ++p) {
            const std::string tname = "blk." + std::to_string(il) + ".ffn_" + projs[p] + "_exps.weight";
            std::vector<float> W; int64_t ne0, nrows; ggml_type ty;
            if (!gr.slab(tname, 0, W, ne0, nrows, ty)) continue;
            const double k0 = kurtosis(W.data(), W.size()), m0 = maxrms(W.data(), W.size());
            wht32(W.data(), W.size());
            printf("%-5d %-6s %10.3f %10.3f %10.3f %10.3f\n", il, projs[p], k0, m0,
                   kurtosis(W.data(), W.size()), maxrms(W.data(), W.size()));
        }
    }

    printf("\n=== E. is a weight-norm proxy enough for neuron importance? ===\n");
    printf("pearson on logs, measured E[h_j^2] vs weight-norm proxies. High r means the\n");
    printf("activation measurement is not needed and the choice generalises off-corpus.\n\n");
    printf("%-5s %10s %10s %10s\n", "blk", "r_down", "r_gate", "r_up");
    for (int il : blocks) {
        auto it = cal.find(il);
        if (it == cal.end() || it->second.hid_sum.empty()) continue;
        wxq::calib_layer & lc = it->second;
        const int64_t nff = lc.n_ff;
        double rd = 0, rg = 0, ru = 0; int n = 0;
        std::vector<int> order(lc.n_expert);
        for (uint32_t i = 0; i < lc.n_expert; ++i) order[i] = (int) i;
        std::sort(order.begin(), order.end(), [&](int a, int b){ return lc.counts[a] > lc.counts[b]; });
        for (int k = 0; k < std::min(4, (int) lc.n_expert); ++k) {
            const int ei = order[k];
            if (lc.hid_n[ei] == 0) continue;
            std::vector<double> h(nff), nd(nff), ng(nff), nu(nff);
            const double inv = 1.0/(double) lc.hid_n[ei];
            for (int64_t j = 0; j < nff; ++j) h[j] = lc.hid_sum[(size_t) ei*nff + j]*inv;
            std::vector<float> W; int64_t ne0, nrows; ggml_type ty;
            if (gr.slab("blk." + std::to_string(il) + ".ffn_down_exps.weight", ei, W, ne0, nrows, ty))
                for (int64_t j = 0; j < ne0; ++j) { double s = 0; for (int64_t i = 0; i < nrows; ++i) { const double v = W[i*ne0+j]; s += v*v; } nd[j] = s; }
            if (gr.slab("blk." + std::to_string(il) + ".ffn_gate_exps.weight", ei, W, ne0, nrows, ty))
                for (int64_t j = 0; j < nrows; ++j) { double s = 0; for (int64_t i = 0; i < ne0; ++i) { const double v = W[j*ne0+i]; s += v*v; } ng[j] = s; }
            if (gr.slab("blk." + std::to_string(il) + ".ffn_up_exps.weight", ei, W, ne0, nrows, ty))
                for (int64_t j = 0; j < nrows; ++j) { double s = 0; for (int64_t i = 0; i < ne0; ++i) { const double v = W[j*ne0+i]; s += v*v; } nu[j] = s; }
            rd += pearson_log(h, nd); rg += pearson_log(h, ng); ru += pearson_log(h, nu); n++;
        }
        if (n) printf("%-5d %10.3f %10.3f %10.3f\n", il, rd/n, rg/n, ru/n);
    }

    printf("\n=== F. OBQ error-feedback bound per block (from the dump alone) ===\n");
    printf("dR_OBQ = (1/2d) * log2( prod_j A_jj / det A ) = -(1/2d) log2 det R\n\n");
    printf("%-5s %12s %12s %12s\n", "blk", "diag_dR", "eigen_dR", "OBQ_dR");
    for (int il : blocks) {
        auto it = cal.find(il);
        if (it == cal.end() || it->second.cov.empty()) continue;
        wxq::calib_layer & lc = it->second;
        const int64_t d = lc.n_embd;
        const double s = 1.0/(double) lc.n_cov;
        double tr = 0.0, logdiag = 0.0;
        for (int64_t i = 0; i < d; ++i) { const double v = lc.cov[i*d+i]*s; tr += v; if (v > 0) logdiag += std::log2(v); }
        const double am = tr/(double) d;
        std::vector<double> work((size_t) d*d, 0.0);
        for (int64_t i = 0; i < d; ++i) {
            for (int64_t j = i; j < d; ++j) work[i*d+j] = lc.cov[i*d+j]*s;
            work[i*d+i] += 1e-9*am;
        }
        double logdet = 0.0;
        if (chol_upper(work, d)) {
            for (int64_t k = 0; k < d; ++k) logdet += 2.0*std::log2(work[k*d+k]);
            const double diag_dR = 0.5*std::log2(am/std::exp2(logdiag/(double) d));
            const double eig_dR  = 0.5*std::log2(am/std::exp2(logdet /(double) d));
            printf("%-5d %12.3f %12.3f %12.3f\n", il, diag_dR, eig_dR, eig_dR - diag_dR);
        }
    }

    printf("\ndone\n");
    return 0;
}
