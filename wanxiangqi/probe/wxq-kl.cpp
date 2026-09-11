// Per-position KL between two models on the calibration trie.
//
// Why not tools/perplexity --kl-divergence: that path eats a text file, walks it
// in fixed chunks, and reports aggregates. Three things are needed here that it
// cannot give:
//   * the trie population itself (270,952 unique nodes, prefix-shared) rather
//     than a detokenised approximation of it
//   * per-position output, so aggregation is a post-hoc decision: mean, CVaR95,
//     or weighted by role (think-prose vs final answer vs tool call)
//   * a stable node id per record, so two runs are compared point-for-point
//
// Modes:
//   --dump  FILE   run the model, write per-position reference distributions
//   --cmp   FILE   run the model, compare against a reference dump
//
// The reference keeps only the top-K of each distribution. KL is then evaluated
// exactly on that support (the candidate is running live, so its probabilities
// on those tokens are exact) and the uncovered reference mass is reported per
// record, so the truncation error is visible rather than assumed small.

#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"

#include "trie.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using wxq::trie;
using wxq::load_trie;

// ---------------------------------------------------------------------------
// reference dump
// ---------------------------------------------------------------------------

static const char * WXQKL_MAGIC = "WXQKL001";

struct ref_rec {
    uint64_t             node = 0;
    std::vector<int32_t> tok;
    std::vector<float>   lp;     // log p, normalised over the full vocabulary
    float                mass = 0.0f;
};

struct ref_writer {
    std::ofstream out;
    uint64_t      n = 0;

    void open(const std::string & path, int32_t n_vocab, int32_t top_k, uint32_t stride) {
        out.open(path, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write " + path);
        out.write(WXQKL_MAGIC, 8);
        out.write((const char *) &n_vocab, 4);
        out.write((const char *) &top_k,   4);
        out.write((const char *) &stride,  4);
        const uint32_t pad = 0;
        out.write((const char *) &pad, 4);
        out.write((const char *) &n, 8);   // patched on close
    }
    void put(const ref_rec & r) {
        const uint32_t k = (uint32_t) r.tok.size();
        out.write((const char *) &r.node, 8);
        out.write((const char *) &k, 4);
        out.write((const char *) &r.mass, 4);
        out.write((const char *) r.tok.data(), (std::streamsize) k*4);
        out.write((const char *) r.lp.data(),  (std::streamsize) k*4);
        ++n;
    }
    void close() {
        out.seekp(24);
        out.write((const char *) &n, 8);
        out.close();
    }
};

struct ref_reader {
    std::ifstream in;
    int32_t  n_vocab = 0, top_k = 0;
    uint32_t stride = 0;
    uint64_t n = 0, read_i = 0;

    void open(const std::string & path) {
        in.open(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open " + path);
        char m[8];
        in.read(m, 8);
        if (std::memcmp(m, WXQKL_MAGIC, 8) != 0) throw std::runtime_error("bad magic in " + path);
        uint32_t pad = 0;
        in.read((char *) &n_vocab, 4);
        in.read((char *) &top_k,   4);
        in.read((char *) &stride,  4);
        in.read((char *) &pad,     4);
        in.read((char *) &n,       8);
    }
    bool next(ref_rec & r) {
        if (read_i >= n) return false;
        uint32_t k = 0;
        in.read((char *) &r.node, 8);
        in.read((char *) &k, 4);
        in.read((char *) &r.mass, 4);
        r.tok.resize(k); r.lp.resize(k);
        in.read((char *) r.tok.data(), (std::streamsize) k*4);
        in.read((char *) r.lp.data(),  (std::streamsize) k*4);
        ++read_i;
        return (bool) in;
    }
};

// ---------------------------------------------------------------------------
// per-position results
// ---------------------------------------------------------------------------

struct kl_point {
    uint64_t node = 0;
    float    kl   = 0.0f;   // nats, on the reference top-K support, renormalised
    float    mass = 0.0f;   // reference mass covered by that support
    uint8_t  top1 = 0;      // argmax agreement
};

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

struct driver {
    llama_context * ctx = nullptr;
    llama_batch     batch{};
    int32_t         n_batch = 0;
    int32_t         n_vocab = 0;
    int32_t         top_k = 64;
    uint32_t        stride = 8;
    uint32_t        max_ctx = 0;
    uint64_t        max_nodes = 0;

    bool            cmp_mode = false;
    ref_writer *    wr = nullptr;
    ref_reader *    rd = nullptr;
    std::vector<kl_point> * pts = nullptr;

    uint64_t n_decoded = 0, n_scored = 0, n_ctx_resets = 0, n_total = 0, n_mismatch = 0;
    uint64_t live_snapshot_bytes = 0, peak_snapshot_bytes = 0;
    llama_pos cur_pos = 0;
    bool stop = false;
    int64_t t_start_us = 0, t_last_report_us = 0, report_period_us = 5000000;

    // scratch: batch slot -> node id, for the slots that asked for logits
    std::vector<std::pair<int32_t, uint64_t>> want;

    void reset_window() {
        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        cur_pos = 0;
        ++n_ctx_resets;
    }

    // log-softmax over the row, then the top-k by logit (== top-k by probability)
    void take_topk(const float * logits, ref_rec & r) const {
        float mx = logits[0];
        for (int32_t i = 1; i < n_vocab; ++i) mx = std::max(mx, logits[i]);
        double se = 0.0;
        for (int32_t i = 0; i < n_vocab; ++i) se += std::exp((double) (logits[i] - mx));
        const double lse = (double) mx + std::log(se);

        std::vector<int32_t> idx(n_vocab);
        for (int32_t i = 0; i < n_vocab; ++i) idx[i] = i;
        const int32_t k = std::min(top_k, n_vocab);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
        r.tok.resize(k); r.lp.resize(k);
        double mass = 0.0;
        for (int32_t j = 0; j < k; ++j) {
            const double lp = (double) logits[idx[j]] - lse;
            r.tok[j] = idx[j];
            r.lp[j]  = (float) lp;
            mass    += std::exp(lp);
        }
        r.mass = (float) mass;
    }

    // KL(ref || cand) restricted to the reference support and renormalised there
    void score(const float * logits, const ref_rec & ref, kl_point & out) const {
        float mx = logits[0];
        int32_t am = 0;
        for (int32_t i = 1; i < n_vocab; ++i) if (logits[i] > mx) { mx = logits[i]; am = i; }
        double se = 0.0;
        for (int32_t i = 0; i < n_vocab; ++i) se += std::exp((double) (logits[i] - mx));
        const double lse = (double) mx + std::log(se);

        double kl = 0.0, z = 0.0;
        for (size_t j = 0; j < ref.tok.size(); ++j) {
            const double p  = std::exp((double) ref.lp[j]);
            const double lq = (double) logits[ref.tok[j]] - lse;
            kl += p * ((double) ref.lp[j] - lq);
            z  += p;
        }
        out.node = ref.node;
        out.kl   = (float) (z > 0.0 ? kl / z : 0.0);
        out.mass = ref.mass;
        out.top1 = (!ref.tok.empty() && ref.tok[0] == am) ? 1 : 0;
    }

    bool flush_batch() {
        if (batch.n_tokens == 0) return true;
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: llama_decode failed at pos %d\n", __func__, (int) cur_pos);
            return false;
        }
        for (const auto & [slot, node] : want) {
            const float * lg = llama_get_logits_ith(ctx, slot);
            if (lg == nullptr) { LOG_ERR("%s: no logits for slot %d\n", __func__, slot); return false; }
            if (cmp_mode) {
                ref_rec ref;
                if (!rd->next(ref)) { stop = true; break; }
                if (ref.node != node) { ++n_mismatch; continue; }
                kl_point p;
                score(lg, ref, p);
                pts->push_back(p);
            } else {
                ref_rec r;
                r.node = node;
                take_topk(lg, r);
                wr->put(r);
            }
            ++n_scored;
        }
        want.clear();
        common_batch_clear(batch);
        return true;
    }

    bool decode_span(const std::vector<llama_token> & toks, const std::vector<uint64_t> & ids) {
        for (size_t i = 0; i < toks.size(); ++i) {
            if (max_nodes != 0 && n_decoded >= max_nodes) { stop = true; return flush_batch(); }

            if (batch.n_tokens >= n_batch) { if (!flush_batch()) return false; }
            if (max_ctx != 0 && cur_pos + 1 > (llama_pos) max_ctx) {
                if (!flush_batch()) return false;
                reset_window();
            }

            // one node per position; score every stride-th node
            const bool score_it = (ids[i] % stride) == 0;
            common_batch_add(batch, toks[i], cur_pos, { 0 }, score_it);
            if (score_it) want.emplace_back(batch.n_tokens - 1, ids[i]);

            ++cur_pos;
            ++n_decoded;

            const int64_t t1 = ggml_time_us();
            if (t1 - t_last_report_us >= report_period_us) {
                const double el   = (double) (t1 - t_start_us) / 1e6;
                const double rate = (double) n_decoded / std::max(el, 1e-6);
                const double frac = n_total ? (double) n_decoded / (double) n_total : 0.0;
                LOG_INF("%s: pos %7d | %7" PRIu64 "/%" PRIu64 " (%4.1f%%) | %7.1f tok/s | scored %" PRIu64 " | elapsed %5.0fs eta %5.0fs\n",
                        __func__, (int) cur_pos, n_decoded, n_total, 100.0*frac, rate, n_scored, el,
                        rate > 0.0 && n_total > n_decoded ? (double) (n_total - n_decoded)/rate : 0.0);
                fflush(stderr);
                t_last_report_us = t1;
            }
        }
        return true;
    }

    // collapse each straight-line run, exactly as the calibration traversal does,
    // so the visited node sequence is identical between runs
    bool visit(const trie & t, uint64_t node) {
        if (stop) return true;
        std::vector<llama_token> toks;
        std::vector<uint64_t>    ids;
        uint64_t cur = node;
        while (true) {
            toks.push_back(t.nodes[cur - 1].token);
            ids.push_back(cur);
            if (t.children[cur].size() != 1) break;
            cur = t.children[cur][0];
        }
        if (!decode_span(toks, ids)) return false;

        const auto & kids = t.children[cur];
        if (kids.empty() || stop) return true;

        // Branch: park the sequence on the host and replay it for every sibling
        // after the first. seq_rm would not do - it cannot restore the
        // GatedDeltaNet recurrent state, and with a bounded window the position
        // at a branch point is not derivable from trie depth, so cur_pos has to
        // travel with the blob. The batch is drained first so the snapshot
        // covers every token that has actually been decoded.
        if (!flush_batch()) return false;

        const size_t snap_size = llama_state_seq_get_size(ctx, 0);
        std::vector<uint8_t> snap(snap_size);
        if (llama_state_seq_get_data(ctx, snap.data(), snap.size(), 0) != snap_size) {
            LOG_ERR("%s: state_seq_get_data short read at pos %d\n", __func__, (int) cur_pos);
            return false;
        }
        const llama_pos snap_pos = cur_pos;
        live_snapshot_bytes += snap_size;
        peak_snapshot_bytes  = std::max(peak_snapshot_bytes, live_snapshot_bytes);

        bool ok = true;
        for (size_t i = 0; i < kids.size() && ok; ++i) {
            if (i > 0) {
                if (!flush_batch()) { ok = false; break; }
                if (llama_state_seq_set_data(ctx, snap.data(), snap.size(), 0) != snap_size) {
                    LOG_ERR("%s: state_seq_set_data short write at pos %d\n", __func__, (int) cur_pos);
                    ok = false;
                    break;
                }
                cur_pos = snap_pos;
            }
            ok = visit(t, kids[i]);
            if (stop) break;
        }

        live_snapshot_bytes -= snap_size;
        return ok;
    }
};

// ---------------------------------------------------------------------------

static void summarise(std::vector<kl_point> & p, const std::string & csv) {
    if (p.empty()) { LOG_INF("no scored positions\n"); return; }
    std::vector<float> v; v.reserve(p.size());
    double mass = 0.0, top1 = 0.0;
    for (const auto & q : p) { v.push_back(q.kl); mass += q.mass; top1 += q.top1; }
    std::sort(v.begin(), v.end());
    const auto q = [&](double f) { return v[std::min(v.size() - 1, (size_t) (f * (double) v.size())) ]; };
    double sum = 0.0;
    for (float x : v) sum += x;
    // CVaR95: mean of the worst 5%, which is what actually breaks a deployment
    const size_t c0 = (size_t) (0.95 * (double) v.size());
    double cv = 0.0;
    for (size_t i = c0; i < v.size(); ++i) cv += v[i];
    cv /= std::max<size_t>(1, v.size() - c0);

    LOG_INF("\n--- KL(ref || cand), nats, %zu positions ---\n", v.size());
    LOG_INF("  mean    %10.6f\n", sum / (double) v.size());
    LOG_INF("  median  %10.6f\n", q(0.50));
    LOG_INF("  p90     %10.6f\n", q(0.90));
    LOG_INF("  p99     %10.6f\n", q(0.99));
    LOG_INF("  max     %10.6f\n", v.back());
    LOG_INF("  CVaR95  %10.6f\n", cv);
    LOG_INF("  top1 agree %6.2f%%\n", 100.0 * top1 / (double) v.size());
    LOG_INF("  ref mass covered by top-K: %6.4f (mean)\n", mass / (double) v.size());

    if (!csv.empty()) {
        std::ofstream o(csv, std::ios::trunc);
        o << "node,kl,mass,top1\n";
        for (const auto & r : p) o << r.node << ',' << r.kl << ',' << r.mass << ',' << (int) r.top1 << '\n';
        LOG_INF("  per-position -> %s\n", csv.c_str());
    }
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_batch = 1024;
    params.n_ubatch = 512;
    params.n_ctx = 16384;

    std::string trie_dir, dump_path, cmp_path, csv_path;
    int32_t  top_k   = 64;
    uint32_t stride  = 8;
    uint32_t max_ctx = 16384;
    uint64_t max_nodes = 0;

    // strip our flags before handing the rest to common_params_parse
    std::vector<char *> rest;
    rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        auto val = [&]() -> const char * { if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + a); return argv[++i]; };
        if      (!std::strcmp(a, "--trie"))      trie_dir  = val();
        else if (!std::strcmp(a, "--dump"))      dump_path = val();
        else if (!std::strcmp(a, "--cmp"))       cmp_path  = val();
        else if (!std::strcmp(a, "--csv"))       csv_path  = val();
        else if (!std::strcmp(a, "--kl-top-k"))  top_k     = std::atoi(val());
        else if (!std::strcmp(a, "--kl-stride")) stride    = (uint32_t) std::atoi(val());
        else if (!std::strcmp(a, "--kl-max-ctx"))max_ctx   = (uint32_t) std::atoi(val());
        else if (!std::strcmp(a, "--kl-max-nodes")) max_nodes = std::strtoull(val(), nullptr, 10);
        else rest.push_back(argv[i]);
    }
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) return 1;
    if (trie_dir.empty() || (dump_path.empty() == cmp_path.empty())) {
        LOG_ERR("usage: %s -m MODEL --trie DIR (--dump REF.bin | --cmp REF.bin) [--csv OUT.csv]\n"
                "          [--kl-stride N] [--kl-top-k K] [--kl-max-ctx N] [--kl-max-nodes N]\n", argv[0]);
        return 1;
    }
    if (stride == 0) stride = 1;

    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result_ptr ir = common_init_from_params(params);
    if (!ir || !ir->model() || !ir->context()) { LOG_ERR("failed to load model\n"); return 1; }

    llama_model   * model = ir->model();
    llama_context * ctx   = ir->context();
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    trie t = load_trie(trie_dir, n_vocab);
    LOG_INF("%s: trie %" PRIu64 " nodes, %zu requests, stride %u -> ~%" PRIu64 " scored\n",
            __func__, t.n_nodes(), t.leaves.size(), stride, t.n_nodes() / stride);

    ref_writer wr; ref_reader rd;
    std::vector<kl_point> pts;

    driver d;
    d.ctx = ctx;
    d.n_batch = (int32_t) params.n_batch;
    d.batch = llama_batch_init(d.n_batch, 0, 1);
    d.n_vocab = n_vocab;
    d.top_k = top_k;
    d.stride = stride;
    d.max_ctx = max_ctx;
    d.max_nodes = max_nodes;
    d.n_total = t.n_nodes();
    d.cmp_mode = !cmp_path.empty();
    d.t_start_us = d.t_last_report_us = ggml_time_us();

    if (d.cmp_mode) {
        rd.open(cmp_path);
        if (rd.stride != stride) {
            LOG_ERR("reference was built with stride %u, run asked for %u\n", rd.stride, stride);
            return 1;
        }
        d.rd = &rd;
        d.pts = &pts;
        LOG_INF("%s: reference %s, %" PRIu64 " records, top-%d\n", __func__, cmp_path.c_str(), rd.n, rd.top_k);
    } else {
        wr.open(dump_path, n_vocab, top_k, stride);
        d.wr = &wr;
    }

    bool ok = true;
    for (uint64_t root_child : t.children[0]) {
        d.reset_window();
        if (!d.visit(t, root_child)) { ok = false; break; }
        if (d.stop) break;
    }
    if (ok) ok = d.flush_batch();

    LOG_INF("%s: decoded %" PRIu64 ", scored %" PRIu64 ", resets %" PRIu64 ", mismatches %" PRIu64 "\n",
            __func__, d.n_decoded, d.n_scored, d.n_ctx_resets, d.n_mismatch);

    if (d.cmp_mode) summarise(pts, csv_path);
    else            { wr.close(); LOG_INF("%s: wrote %s\n", __func__, dump_path.c_str()); }

    llama_batch_free(d.batch);
    llama_backend_free();
    return ok ? 0 : 1;
}
