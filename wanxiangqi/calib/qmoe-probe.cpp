// Does QMoE's sub-1-bit result transfer to a modern top-k MoE?
//
// QMoE (Frantar & Alistarh 2023) compresses SwitchTransformer-c2048 to 0.8 bpw
// by quantising experts to ternary and then entropy-coding the ternary stream
// with a dictionary code. The rate is therefore set by the entropy of the
// ternary symbol distribution, which is set by the dead-zone threshold. GPTQ
// error feedback improves the distortion at a given rate; it barely moves the
// symbol distribution. So the rate question can be answered without any Hessian,
// any GPTQ, and any decode kernel - which is what this tool does.
//
// Switch-c2048 is top-1 over 2048 experts, so each expert sees ~1/2048 of the
// tokens. This model is top-8 over 256, so each expert sees ~1/32 - 64x more
// training signal per expert, and correspondingly less redundancy. Measured
// kurtosis here is 3.5, i.e. essentially Gaussian, and an error-optimal ternary
// quantiser on a Gaussian leaves ~60% zeros, which is ~1.37 bpw of order-0
// entropy, not 0.8. This tool checks that prediction against the real weights.
//
// Reported per (block, expert, threshold):
//   zero%      fraction of ternary symbols that are zero
//   H1         order-0 entropy of single symbols            [bpw]
//   H4, H8     order-0 entropy of 4- and 8-symbol groups    [bpw]
//              these stand in for what a dictionary code can reach: a dictionary
//              exploits exactly the joint skew of short runs.
//   rel-err    Hessian-weighted reconstruction error, using the measured
//              per-expert input diagonal as the metric
//
// Bias note: source weights are already iq2_xs/iq3_xxs, so they carry broadband
// quantisation noise. Noise raises entropy, so the rate numbers here are
// pessimistic and the real fp16 source will code slightly better.

#include "ggml.h"
#include "gguf.h"

#include "calib-dump.h"
#include "gguf-slab.h"
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
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// ternary quantisation and entropy
// ---------------------------------------------------------------------------

// Ternary a row with dead zone t*mean|w|. The non-zero magnitude is the
// conditional mean of |w| over the retained set, which is the error-optimal
// single-level reconstruction.
static void ternary_row(const float * w, int64_t n, double t, int8_t * q, double & scale) {
    double mabs = 0.0;
    for (int64_t i = 0; i < n; ++i) mabs += std::fabs(w[i]);
    mabs /= (double) n;
    const double thr = t * mabs;
    double s = 0.0; int64_t k = 0;
    for (int64_t i = 0; i < n; ++i) if (std::fabs(w[i]) > thr) { s += std::fabs(w[i]); ++k; }
    scale = k ? s/(double) k : 0.0;
    for (int64_t i = 0; i < n; ++i)
        q[i] = (std::fabs(w[i]) > thr) ? (w[i] > 0.0f ? 1 : -1) : 0;
}

// order-0 entropy of k-symbol groups, in bits per weight
static double group_entropy(const std::vector<int8_t> & q, int64_t n, int k) {
    std::unordered_map<uint32_t, uint64_t> hist;
    hist.reserve(4096);
    uint64_t tot = 0;
    for (int64_t i = 0; i + k <= n; i += k) {
        uint32_t code = 0;
        for (int j = 0; j < k; ++j) code = code*3u + (uint32_t)(q[i+j] + 1);
        ++hist[code]; ++tot;
    }
    if (!tot) return 0.0;
    double H = 0.0;
    for (const auto & [c, cnt] : hist) {
        const double p = (double) cnt / (double) tot;
        H -= p * std::log2(p);
    }
    return H / (double) k;   // bits per weight
}

int main(int argc, char ** argv) {
    std::string model, calib = "/tmp/opencode/calib-full2.bin";
    std::vector<int> blocks = {2, 18, 38};
    int n_exp = 4;
    wxq::set_num_threads((int) std::max(1u, std::thread::hardware_concurrency()));

    for (int i = 1; i < argc-1; ++i) {
        if      (!std::strcmp(argv[i], "--model"))   model = argv[++i];
        else if (!std::strcmp(argv[i], "--calib"))   calib = argv[++i];
        else if (!std::strcmp(argv[i], "--threads")) wxq::set_num_threads(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--experts")) n_exp = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--blocks")) {
            blocks.clear();
            for (char * p = std::strtok(argv[++i], ","); p; p = std::strtok(nullptr, ",")) blocks.push_back(std::atoi(p));
        }
    }
    if (model.empty()) { fprintf(stderr, "usage: %s --model X.gguf [--calib f] [--blocks 2,18] [--experts 4]\n", argv[0]); return 1; }

    printf("threads %d, experts/block %d, blocks:", wxq::num_threads(), n_exp);
    for (int b : blocks) printf(" %d", b);
    printf("\n\n");

    // only the per-expert input diagonals are needed here, so cov is skipped
    wxq::calib_dump dump;
    std::string cal_err;
    if (!wxq::load_calib(calib, blocks, /* want_cov */ false, dump, cal_err)) {
        fprintf(stderr, "%s\n", cal_err.c_str());
        return 1;
    }
    printf("calib: %u blocks, flags 0x%x, %" PRIu64 "/%" PRIu64 " nodes, %s\n\n",
           dump.meta.n_entries, dump.meta.flags, dump.meta.n_decoded, dump.meta.n_total,
           dump.meta.complete ? "complete" : "PARTIAL");
    std::map<int, wxq::calib_layer> & cal = dump.layers;

    wxq::gguf_slab_reader gr;
    if (!gr.open(model)) return 1;

    printf("QMoE reference: SwitchTransformer-c2048 -> 0.8 bpw (ternary + dictionary code)\n");
    printf("order-0 entropy is the floor any dictionary code must respect; H4/H8 show\n");
    printf("how much joint skew a dictionary could additionally exploit.\n\n");

    const double thrs[] = { 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0 };
    const char * projs[3] = { "gate", "up", "down" };

    printf("%-5s %-6s %6s %8s %8s %8s %8s %9s\n",
           "blk", "proj", "thr", "zero%", "H1", "H4", "H8", "rel-err");

    struct agg { double h1=0,h4=0,h8=0,z=0,e=0; int n=0; };
    std::map<double, agg> byt;

    for (int il : blocks) {
        auto it = cal.find(il);
        if (it == cal.end()) continue;
        wxq::calib_layer & lc = it->second;

        std::vector<int> order(lc.n_expert);
        for (uint32_t i = 0; i < lc.n_expert; ++i) order[i] = (int) i;
        std::sort(order.begin(), order.end(), [&](int a, int b){ return lc.counts[a] > lc.counts[b]; });
        std::vector<int> pick;
        for (int k = 0; k < n_exp; ++k) pick.push_back(order[(size_t) k*(lc.n_expert-1)/std::max(1, n_exp-1)]);

        for (int p = 0; p < 3; ++p) {
            const std::string tname = "blk." + std::to_string(il) + ".ffn_" + projs[p] + "_exps.weight";

            for (double t : thrs) {
                double z = 0, h1 = 0, h4 = 0, h8 = 0, num = 0, den = 0;
                int cnt = 0;
                for (int ei : pick) {
                    std::vector<float> W;
                    int64_t ne0, nrows;
                    if (!gr.slab(tname, ei, W, ne0, nrows)) { p = 3; break; }

                    // metric: per-expert input diagonal for gate/up, per-expert
                    // per-neuron hidden energy for down (its contraction dim is n_ff)
                    std::vector<double> m(ne0, 1.0);
                    if (p < 2 && !lc.exp_diag.empty() && lc.exp_n[ei] > 0) {
                        const double inv = 1.0/(double) lc.exp_n[ei];
                        for (int64_t j = 0; j < ne0; ++j) m[j] = lc.exp_diag[(size_t) ei*ne0 + j]*inv;
                    } else if (p == 2 && !lc.hid_sum.empty() && lc.hid_n[ei] > 0) {
                        const double inv = 1.0/(double) lc.hid_n[ei];
                        for (int64_t j = 0; j < ne0; ++j) m[j] = lc.hid_sum[(size_t) ei*ne0 + j]*inv;
                    }

                    std::vector<int8_t> q((size_t) ne0*nrows);
                    double lz = 0, ln = 0, ld = 0;
                    for (int64_t r = 0; r < nrows; ++r) {
                        double sc = 0;
                        ternary_row(W.data() + r*ne0, ne0, t, q.data() + r*ne0, sc);
                        for (int64_t j = 0; j < ne0; ++j) {
                            const double w  = W[r*ne0 + j];
                            const double wh = sc * (double) q[r*ne0 + j];
                            const double d  = w - wh;
                            ln += m[j]*d*d;
                            ld += m[j]*w*w;
                            if (q[r*ne0 + j] == 0) lz += 1.0;
                        }
                    }
                    z   += lz/((double) ne0*nrows);
                    num += ln; den += ld;
                    h1  += group_entropy(q, (int64_t) q.size(), 1);
                    h4  += group_entropy(q, (int64_t) q.size(), 4);
                    h8  += group_entropy(q, (int64_t) q.size(), 8);
                    ++cnt;
                }
                if (!cnt) break;
                const double relerr = den > 0 ? num/den : 0.0;
                printf("%-5d %-6s %6.1f %8.1f %8.3f %8.3f %8.3f %9.4f\n",
                       il, projs[p], t, 100.0*z/cnt, h1/cnt, h4/cnt, h8/cnt, relerr);
                agg & A = byt[t];
                A.z += z/cnt; A.h1 += h1/cnt; A.h4 += h4/cnt; A.h8 += h8/cnt; A.e += relerr; A.n++;
            }
        }
    }

    printf("\n--- aggregate over all (block, proj) ---\n");
    printf("%6s %8s %8s %8s %8s %9s %12s\n", "thr", "zero%", "H1", "H4", "H8", "rel-err", "vs QMoE 0.8");
    for (auto & [t, A] : byt) {
        if (!A.n) continue;
        const double h8 = A.h8/A.n;
        printf("%6.1f %8.1f %8.3f %8.3f %8.3f %9.4f %12s\n",
               t, 100.0*A.z/A.n, A.h1/A.n, A.h4/A.n, h8, A.e/A.n,
               h8 <= 0.8 ? "REACHED" : "above");
    }
    printf("\ndone\n");
    return 0;
}
