// Apply a linear input transform to the MoE expert weights, in place.
//
// The experts compute y = W x. Replacing the input by x_hat = M x is the same as
// replacing the weights by W' = W M, so the whole tail ablation can be done
// offline with no change to the graph and no runtime cost. Shapes and types are
// preserved, which means the output file is a byte-for-byte copy of the source
// with only the gate/up expert slabs overwritten at their existing offsets --
// no GGUF rebuild, no metadata surgery.
//
// Only gate and up are touched. They are the two tensors that read x. The down
// projection reads the hidden state and is left alone, and the router
// (ffn_gate_inp) is a separate tensor that is deliberately left alone too, so
// routing stays exactly as it was and the ablation isolates the expert function
// rather than confounding it with a routing change.
//
// Requantisation uses the same type the tensor already had, with the per-expert
// importance matrix, so the only difference from the source is the transform
// plus one extra round of quantisation error. Run the identity transform to
// measure that error on its own -- without that control the ablation cannot be
// separated from the requantisation.

#include "ggml.h"
#include "gguf.h"

#include "parallel.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// transform file (WXQXFM01): per layer, an n_embd x n_embd row-major f32 M
// with x_hat[a] = sum_b M[a][b] x[b]
// ---------------------------------------------------------------------------

// Fitted replacement weights (WXQW0001): the width-m solution is written back
// zero-padded into the original n_ff slots. SiLU(0)*0 = 0, so the padded units
// contribute nothing and the file keeps its exact shape and size -- no metadata
// edit, no llama.cpp change. Real units go in slots 0..m-1 so every quantisation
// block is homogeneous (all-real or all-zero) rather than half of each.
struct fitted {
    uint32_t layer=0, m=0, ne=0, d=0;
    std::vector<float> g,u,dn;
    bool load(const std::string & p){
        std::ifstream f(p,std::ios::binary);
        if(!f){fprintf(stderr,"cannot open %s\n",p.c_str());return false;}
        char mg[8]; f.read(mg,8);
        if(std::memcmp(mg,"WXQW0001",8)!=0){fprintf(stderr,"bad weight magic\n");return false;}
        f.read((char*)&layer,4); f.read((char*)&m,4); f.read((char*)&ne,4); f.read((char*)&d,4);
        const size_t a=(size_t)ne*m*d;
        g.resize(a); u.resize(a); dn.resize(a);
        f.read((char*)g.data(),a*4); f.read((char*)u.data(),a*4); f.read((char*)dn.data(),a*4);
        printf("fitted weights: blk.%u m=%u experts=%u d=%u\n",layer,m,ne,d);
        return (bool)f;
    }
};

struct transforms {
    uint32_t n_embd = 0;
    std::map<int, std::vector<float>> M;

    bool load(const std::string & path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
        char magic[8];
        f.read(magic, 8);
        if (std::memcmp(magic, "WXQXFM01", 8) != 0) { fprintf(stderr, "bad transform magic\n"); return false; }
        uint32_t n_layers = 0;
        f.read((char *) &n_layers, 4);
        f.read((char *) &n_embd,   4);
        for (uint32_t i = 0; i < n_layers; ++i) {
            uint32_t layer = 0, pad = 0;
            f.read((char *) &layer, 4);
            f.read((char *) &pad,   4);
            std::vector<float> m((size_t) n_embd * n_embd);
            f.read((char *) m.data(), (std::streamsize) m.size() * 4);
            if (!f) { fprintf(stderr, "truncated transform at layer %u\n", layer); return false; }
            M[(int) layer] = std::move(m);
        }
        printf("transforms: %zu layers, n_embd %u\n", M.size(), n_embd);
        return true;
    }
};

// ---------------------------------------------------------------------------
// imatrix: <tensor>.in_sum2 [ne0, n_expert] + <tensor>.counts [1, n_expert],
// stored as raw sums; llama.cpp divides on load, so do the same here
// ---------------------------------------------------------------------------

struct imatrix {
    std::map<std::string, std::vector<float>> per_tensor;   // [n_expert * ne0], normalised

    bool load(const std::string & path) {
        ggml_context * mctx = nullptr;
        gguf_init_params gp{};
        gp.no_alloc = false;
        gp.ctx      = &mctx;
        gguf_context * g = gguf_init_from_file(path.c_str(), gp);
        if (!g) { fprintf(stderr, "cannot open imatrix %s\n", path.c_str()); return false; }

        std::map<std::string, ggml_tensor *> sums, counts;
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            const char * nm = gguf_get_tensor_name(g, i);
            ggml_tensor * t = ggml_get_tensor(mctx, nm);
            std::string s(nm);
            const std::string a = ".in_sum2", b = ".counts";
            if (s.size() > a.size() && s.compare(s.size()-a.size(), a.size(), a) == 0) sums[s.substr(0, s.size()-a.size())] = t;
            else if (s.size() > b.size() && s.compare(s.size()-b.size(), b.size(), b) == 0) counts[s.substr(0, s.size()-b.size())] = t;
        }
        for (auto & [name, st] : sums) {
            auto it = counts.find(name);
            if (it == counts.end()) continue;
            const int64_t ne0 = st->ne[0], nmat = st->ne[1];
            const float * sv = (const float *) st->data;
            const float * cv = (const float *) it->second->data;
            std::vector<float> v((size_t) ne0 * nmat);
            for (int64_t e = 0; e < nmat; ++e) {
                const float c = cv[e] > 0.0f ? cv[e] : 1.0f;
                for (int64_t j = 0; j < ne0; ++j) v[(size_t) e*ne0 + j] = sv[(size_t) e*ne0 + j] / c;
            }
            per_tensor[name] = std::move(v);
        }
        printf("imatrix: %zu tensors\n", per_tensor.size());
        gguf_free(g);
        ggml_free(mctx);
        return true;
    }
};

static int layer_of(const std::string & n) {
    int il = -1;
    return std::sscanf(n.c_str(), "blk.%d.", &il) == 1 ? il : -1;
}

int main(int argc, char ** argv) {
    std::string src, dst, xfm, imat;
    std::vector<std::string> wfiles;
    wxq::set_num_threads((int) std::max(1u, std::thread::hardware_concurrency()));

    for (int i = 1; i < argc - 1; ++i) {
        if      (!std::strcmp(argv[i], "--src"))     src  = argv[++i];
        else if (!std::strcmp(argv[i], "--dst"))     dst  = argv[++i];
        else if (!std::strcmp(argv[i], "--xfm"))     xfm  = argv[++i];
        else if (!std::strcmp(argv[i], "--imatrix")) imat = argv[++i];
        else if (!std::strcmp(argv[i], "--weights")) wfiles.push_back(argv[++i]);
        else if (!std::strcmp(argv[i], "--threads")) wxq::set_num_threads(std::atoi(argv[++i]));
    }
    std::vector<fitted> FWs;
    const bool use_w = !wfiles.empty();
    for (const auto & wf : wfiles) { fitted f; if (!f.load(wf)) return 1; FWs.push_back(std::move(f)); }
    if (use_w) xfm = "none";
    if (src.empty() || dst.empty() || xfm.empty()) {
        fprintf(stderr, "usage: %s --src IN.gguf --dst OUT.gguf --xfm XFM.bin [--imatrix I.gguf] [--threads N]\n", argv[0]);
        return 1;
    }

    const bool identity = (xfm == "none");
    transforms T;
    if (identity) {
        // control: dequantise and requantise with no transform, to price the
        // extra round of quantisation error on its own
        T.n_embd = 0;
        printf("identity mode: no transform, requantisation only\n");
    } else if (!T.load(xfm)) return 1;
    imatrix IM;
    if (!imat.empty() && !IM.load(imat)) return 1;

    printf("copying %s -> %s\n", src.c_str(), dst.c_str());
    std::error_code ec;
    std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing, ec);
    if (ec) { fprintf(stderr, "copy failed: %s\n", ec.message().c_str()); return 1; }

    ggml_context * mctx = nullptr;
    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx      = &mctx;
    gguf_context * g = gguf_init_from_file(src.c_str(), gp);
    if (!g) { fprintf(stderr, "cannot read %s\n", src.c_str()); return 1; }
    const size_t data_off = gguf_get_data_offset(g);

    std::fstream out(dst, std::ios::binary | std::ios::in | std::ios::out);
    if (!out) { fprintf(stderr, "cannot open %s for update\n", dst.c_str()); return 1; }

    const int64_t n_tensors = gguf_get_n_tensors(g);
    int n_done = 0;
    double sum_rel = 0.0;
    int    n_rel   = 0;

    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const std::string name = gguf_get_tensor_name(g, ti);
        const bool is_gate = name.find("ffn_gate_exps.weight") != std::string::npos;
        const bool is_up   = name.find("ffn_up_exps.weight")   != std::string::npos;
        const bool is_down = name.find("ffn_down_exps.weight") != std::string::npos;
        const bool is_gate_up = is_gate || is_up;
        const fitted * FWp = nullptr;
        if (use_w) {
            if (!(is_gate_up || is_down)) continue;
            for (const auto & f : FWs) if (layer_of(name) == (int) f.layer) { FWp = &f; break; }
            if (!FWp) continue;
        } else if (!is_gate_up) continue;
        const int il = layer_of(name);
        auto mit = T.M.find(il);
        if (il < 0 || (!identity && mit == T.M.end())) continue;

        ggml_tensor * t = ggml_get_tensor(mctx, name.c_str());
        const ggml_type type = t->type;
        const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2];
        if (!identity && (uint32_t) ne0 != T.n_embd) {
            fprintf(stderr, "%s: ne0 %" PRId64 " != transform dim %u, skipping\n", name.c_str(), ne0, T.n_embd);
            continue;
        }
        const size_t row_bytes  = ggml_row_size(type, ne0);
        const size_t slab_bytes = row_bytes * (size_t) ne1;
        const size_t base       = data_off + gguf_get_tensor_offset(g, ti);
        const float * M         = identity ? nullptr : mit->second.data();

        const auto * tr = ggml_get_type_traits(type);
        if (!tr->to_float) { fprintf(stderr, "%s: no dequantiser for %s\n", name.c_str(), ggml_type_name(type)); continue; }

        const float * imv = nullptr;
        auto iit = IM.per_tensor.find(name);
        if (iit != IM.per_tensor.end() && (int64_t) iit->second.size() == ne0*ne2) imv = iit->second.data();
        else if (ggml_type_size(type) && !imat.empty()) {
            fprintf(stderr, "%s: no imatrix entry, quantising blind\n", name.c_str());
        }

        std::vector<uint8_t> slab(slab_bytes * (size_t) ne2);
        out.seekg((std::streamoff) base);
        out.read((char *) slab.data(), (std::streamsize) slab.size());
        if (!out) { fprintf(stderr, "%s: short read\n", name.c_str()); return 1; }

        std::atomic<double> num{0.0}, den{0.0};
        const int nth = wxq::num_threads();
        std::vector<std::thread> pool;
        pool.reserve(nth);
        for (int th = 0; th < nth; ++th) {
            pool.emplace_back([&, th]() {
                std::vector<float> W((size_t) ne1 * ne0), Wp((size_t) ne1 * ne0);
                double ln = 0.0, ld = 0.0;
                for (int64_t e = th; e < ne2; e += nth) {
                    uint8_t * eslab = slab.data() + (size_t) e * slab_bytes;
                    for (int64_t r = 0; r < ne1; ++r) tr->to_float(eslab + (size_t) r*row_bytes, W.data() + (size_t) r*ne0, ne0);

                    // W' = W M, accumulated as an axpy over the contiguous rows of M
                    if (use_w) {
                        std::fill(Wp.begin(), Wp.end(), 0.0f);
                        const fitted & FW = *FWp;
                        const size_t off = (size_t) e * FW.m * FW.d;
                        if (is_down) {
                            for (int64_t r = 0; r < ne1; ++r)
                                for (uint32_t j = 0; j < FW.m; ++j)
                                    Wp[(size_t) r*ne0 + j] = FW.dn[off + (size_t) r*FW.m + j];
                        } else {
                            const float * srcw = (is_gate ? FW.g.data() : FW.u.data()) + off;
                            for (uint32_t r = 0; r < FW.m; ++r)
                                std::memcpy(Wp.data() + (size_t) r*ne0, srcw + (size_t) r*FW.d, (size_t) ne0*4);
                        }
                    } else if (identity) { Wp = W; } else {
                    std::fill(Wp.begin(), Wp.end(), 0.0f);
                    for (int64_t r = 0; r < ne1; ++r) {
                        const float * wr = W.data()  + (size_t) r*ne0;
                        float       * op = Wp.data() + (size_t) r*ne0;
                        for (int64_t a = 0; a < ne0; ++a) {
                            const float c = wr[a];
                            if (c == 0.0f) continue;
                            const float * mr = M + (size_t) a*ne0;
                            for (int64_t b = 0; b < ne0; ++b) op[b] += c * mr[b];
                        }
                    } }
                    for (size_t i = 0; i < W.size(); ++i) {
                        const double dd = (double) Wp[i] - (double) W[i];
                        ln += dd*dd; ld += (double) W[i] * (double) W[i];
                    }
                    ggml_quantize_chunk(type, Wp.data(), eslab, 0, ne1, ne0,
                                        imv ? imv + (size_t) e*ne0 : nullptr);
                }
                double x = num.load(); while (!num.compare_exchange_weak(x, x + ln)) {}
                double y = den.load(); while (!den.compare_exchange_weak(y, y + ld)) {}
            });
        }
        for (auto & p : pool) p.join();

        out.seekp((std::streamoff) base);
        out.write((const char *) slab.data(), (std::streamsize) slab.size());
        if (!out) { fprintf(stderr, "%s: short write\n", name.c_str()); return 1; }

        const double rel = den.load() > 0.0 ? num.load()/den.load() : 0.0;
        sum_rel += rel; ++n_rel;
        printf("  %-34s %-9s %4" PRId64 " experts   weight rel-err %.4f\n",
               name.c_str(), ggml_type_name(type), ne2, rel);
        fflush(stdout);
        ++n_done;
    }

    out.flush();
    out.close();
    printf("rewrote %d tensors; mean weight-space rel-err %.4f\n", n_done, n_rel ? sum_rel/n_rel : 0.0);
    gguf_free(g);
    ggml_free(mctx);
    return 0;
}
