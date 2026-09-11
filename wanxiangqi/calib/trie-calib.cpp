// Trie-driven MoE calibration collector.
//
// Feeds a versioned request-token prefix trie (the format written by
// `llama-server --run-dump`, see wanxiangqi/docs/request-corpus-dump.md) to the
// model as *token ids* and collects two statistics that a rate-distortion plan
// for the routed experts needs:
//
//   1. per-layer, per-expert routing counts
//   2. per-layer expert-input second moment  E[x_j^2]
//
// Why token ids and not text: the trie captures a live server session, so most
// of it is *generated* tokens. Detokenising and re-tokenising does not round
// trip (measured: 2.1% of positions survive on this corpus), so a text-driven
// collector would measure a token stream that never occurs in deployment.
//
// Why a DFS: the 42 request leaves of this corpus share prefixes 12.9:1
// (3,487,264 leaf positions over 270,952 unique nodes). Walking the trie depth
// first and snapshotting the sequence state at branch points evaluates every
// unique node exactly once. Snapshots go to host memory via
// llama_state_seq_get_data(), which carries the recurrent (GatedDeltaNet) state
// as well as the attention KV.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "calib-dump.h"
#include "linalg.h"
#include "trie.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// collector
// ---------------------------------------------------------------------------

using wxq::trie;
using wxq::load_trie;
using wxq::accum_cov;

using layer_stats = wxq::calib_layer;

struct collector {
    std::map<int, layer_stats> layers;  // keyed by block index
    std::vector<uint8_t>       buf_ids;
    std::vector<uint8_t>       buf_src1;
    std::vector<uint8_t>       buf_hid;
    uint64_t                   n_nodes_seen = 0;

    // Routing counts come from the tiny `ids` tensor and are collected on every
    // ubatch. The expert-input second moment needs src1, which is orders of
    // magnitude larger, so it is sampled: every `act_stride` ubatches, and within
    // those only the first `act_tokens` positions. Restricting to a *prefix* keeps
    // the readback a single contiguous range, which matters because there is no
    // synchronous strided get - copying the whole tensor to reach one slice of
    // ne[1] would move n_expert_used times more bytes than the statistic needs.
    // E[x_j^2] over n_embd dimensions converges long before the corpus does; the
    // routing histogram is the statistic that wants every position.
    uint32_t act_stride = 8;
    uint32_t act_tokens = 64;
    // The routing histogram is cheap per byte but not per stall: see want().
    // 0 or 1 means every ubatch.
    uint32_t route_stride = 1;
    // The full covariance needs samples, not positions: a d x d second moment
    // estimated from n < d samples is rank deficient by construction and its
    // spectrum is Marchenko-Pastur noise rather than the model's. d is 2048 here,
    // so aim for n in the tens of thousands, which means act_stride 1.
    bool     want_cov       = false;
    bool     want_hidden    = false;
    bool     want_expert_in = false;
    int      n_threads      = 1;
    uint64_t ubatch_idx = 0;
    // ubatch width drives everything: each observe() costs one synchronous
    // readback per layer, so cost per token scales as 1/n_tokens. Log every
    // distinct width rather than just the first one - the first span of this trie
    // is only 4 tokens (first branch point is at depth 4), which is not
    // representative of the steady state.
    std::set<int64_t> seen_ntok;

    // The gate projection carries the routing decision for the whole block
    // (gate/up/down share one `ids`) and its src1 is the pre-routing hidden state,
    // i.e. the expert input. The down projection is the only node whose src1 is the
    // post-nonlinearity hidden vector, which is where neuron-level death lives.
    bool route_due() const { return route_stride <= 1 || ubatch_idx % route_stride == 0; }
    bool act_due()   const { return act_stride   == 0 || ubatch_idx % act_stride   == 0; }

    enum class kind { none, gate, down };

    static kind kind_of(const ggml_tensor * t) {
        if (t->op != GGML_OP_MUL_MAT_ID) {
            return kind::none;
        }
        const char * name = t->src[0]->name;
        if (std::strstr(name, "ffn_gate_exps") != nullptr) {
            return kind::gate;
        }
        if (std::strstr(name, "ffn_down_exps") != nullptr) {
            return kind::down;
        }
        return kind::none;
    }

    // Returning true makes the scheduler cut the graph at this node and
    // ggml_backend_synchronize() before handing it over, so every observed node
    // costs one GPU stall - 41 of them per ubatch with all blocks observed, which
    // serialises what would otherwise be a single submission. On ubatches where
    // neither statistic is due, decline and let the whole graph go in one piece.
    bool want(const ggml_tensor * t) const {
        switch (kind_of(t)) {
            case kind::gate: return route_due() || act_due();
            // the hidden state only exists on the down projection, and it is
            // n_expert_used times wider than the input, so it is never worth
            // reading on a ubatch that is not sampling anyway
            case kind::down: return want_hidden && act_due();
            case kind::none: return false;
        }
        return false;
    }

    static int layer_of(const char * name) {
        int il = -1;
        if (std::sscanf(name, "blk.%d.", &il) != 1) {
            return -1;
        }
        return il;
    }

    void observe(ggml_tensor * t) {
        switch (kind_of(t)) {
            case kind::gate: observe_gate(t); break;
            case kind::down: observe_down(t); break;
            case kind::none: break;
        }
    }

    // Stage a contiguous prefix of `n_act` positions of src1 on the host. There is
    // no synchronous strided get, so reaching one slice of ne[1] any other way
    // would move the whole tensor.
    const char * stage_prefix(const ggml_tensor * src1, int64_t n_act, std::vector<uint8_t> & buf) {
        if (ggml_backend_buffer_is_host(src1->buffer)) {
            return (const char *) src1->data;
        }
        const size_t nbytes = (size_t) n_act * src1->nb[2];
        buf.resize(nbytes);
        ggml_backend_tensor_get(src1, buf.data(), 0, nbytes);
        return (const char *) buf.data();
    }

    void observe_gate(ggml_tensor * t) {
        const ggml_tensor * w    = t->src[0];  // [n_embd, n_ff, n_expert]
        const ggml_tensor * src1 = t->src[1];  // [n_embd, 1, n_tokens] f32, broadcast
        const ggml_tensor * ids  = t->src[2];  // [n_expert_used, n_tokens] i32

        const int il = layer_of(w->name);
        if (il < 0 || src1->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
            return;
        }

        const int64_t n_expert = w->ne[2];
        const int64_t n_embd   = src1->ne[0];
        const int64_t n_tokens = src1->ne[2];
        const int64_t n_used   = ids->ne[0];

        if (ids->ne[1] != n_tokens) {
            return;
        }

        auto & e = layers[il];
        if (e.counts.empty()) {
            e.n_expert = (uint32_t) n_expert;
            e.n_embd   = (uint32_t) n_embd;
            e.counts.assign(n_expert, 0);
            e.act_sum.assign(n_embd, 0.0);
            if (want_cov) {
                e.cov.assign((size_t) n_embd * (size_t) n_embd, 0.0);
            }
            if (want_expert_in) {
                e.exp_mean.assign((size_t) n_expert * (size_t) n_embd, 0.0);
                e.exp_diag.assign((size_t) n_expert * (size_t) n_embd, 0.0);
                e.exp_n.assign(n_expert, 0);
            }
        }
        if (e.n_expert != n_expert || e.n_embd != n_embd) {
            return;
        }

        if (seen_ntok.size() < 12 && seen_ntok.insert(src1->ne[2]).second) {
            LOG_INF("%s: ubatch width %" PRId64 " | src1 %s [%" PRId64 ",%" PRId64 ",%" PRId64 "] nb2=%zu | ids [%" PRId64 ",%" PRId64 "] | n_expert=%" PRId64 "\n",
                    __func__, src1->ne[2], ggml_type_name(src1->type),
                    src1->ne[0], src1->ne[1], src1->ne[2],
                    (size_t) src1->nb[2], ids->ne[0], ids->ne[1], n_expert);
        }

        const bool do_route = route_due();
        const bool do_act   = act_due();

        // ids is small; both statistics need it
        buf_ids.resize(ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, buf_ids.data(), 0, ggml_nbytes(ids));

        if (do_route) {
            for (int64_t i1 = 0; i1 < n_tokens; ++i1) {
                const char * row = (const char *) buf_ids.data() + i1*ids->nb[1];
                for (int64_t i0 = 0; i0 < n_used; ++i0) {
                    int32_t ex = 0;
                    std::memcpy(&ex, row + i0*ids->nb[0], sizeof(ex));
                    if (ex >= 0 && ex < n_expert) {
                        e.counts[ex]++;
                    }
                }
            }
            e.n_route += n_tokens;
        }

        if (!do_act) {
            return;
        }

        const int64_t n_act = (act_tokens == 0) ? n_tokens : std::min<int64_t>(n_tokens, act_tokens);
        const char *  data  = stage_prefix(src1, n_act, buf_src1);
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));

        for (int64_t i2 = 0; i2 < n_act; ++i2) {
            const float * x = (const float *) (data + i2*src1->nb[2]);
            for (int64_t j = 0; j < n_embd; ++j) {
                e.act_sum[j] += (double) x[j] * (double) x[j];
            }
            if (want_expert_in) {
                const char * row = (const char *) buf_ids.data() + i2*ids->nb[1];
                for (int64_t i0 = 0; i0 < n_used; ++i0) {
                    int32_t ex = 0;
                    std::memcpy(&ex, row + i0*ids->nb[0], sizeof(ex));
                    if (ex < 0 || ex >= n_expert) {
                        continue;
                    }
                    double * m = e.exp_mean.data() + (size_t) ex * (size_t) n_embd;
                    double * d = e.exp_diag.data() + (size_t) ex * (size_t) n_embd;
                    for (int64_t j = 0; j < n_embd; ++j) {
                        const double v = (double) x[j];
                        m[j] += v;
                        d[j] += v*v;
                    }
                    e.exp_n[ex]++;
                }
            }
        }
        e.n_pos += n_act;

        if (want_cov) {
            const size_t ld = src1->nb[2] / sizeof(float);
            accum_cov(e.cov.data(), (const float *) data, ld, n_act, n_embd, n_threads);
            e.n_cov += (uint64_t) n_act;
        }
    }

    void observe_down(ggml_tensor * t) {
        const ggml_tensor * w    = t->src[0];  // [n_ff, n_embd, n_expert]
        const ggml_tensor * src1 = t->src[1];  // [n_ff, n_expert_used, n_tokens] f32
        const ggml_tensor * ids  = t->src[2];  // [n_expert_used, n_tokens] i32

        const int il = layer_of(w->name);
        if (il < 0 || src1->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
            return;
        }

        const int64_t n_expert = w->ne[2];
        const int64_t n_ff     = src1->ne[0];
        const int64_t n_used   = src1->ne[1];
        const int64_t n_tokens = src1->ne[2];

        if (ids->ne[1] != n_tokens || ids->ne[0] != n_used) {
            return;
        }

        auto it = layers.find(il);
        if (it == layers.end()) {
            return;  // gate for this block has not been seen yet
        }
        auto & e = it->second;
        if (e.n_expert != n_expert) {
            return;
        }
        if (e.hid_sum.empty()) {
            e.n_ff = (uint32_t) n_ff;
            e.hid_sum.assign((size_t) n_expert * (size_t) n_ff, 0.0);
            e.hid_n.assign(n_expert, 0);
            LOG_INF("%s: blk.%d hidden [%" PRId64 ",%" PRId64 ",%" PRId64 "] nb1=%zu nb2=%zu\n",
                    __func__, il, src1->ne[0], src1->ne[1], src1->ne[2],
                    (size_t) src1->nb[1], (size_t) src1->nb[2]);
        }
        if (e.n_ff != n_ff) {
            return;
        }

        const int64_t n_act = (act_tokens == 0) ? n_tokens : std::min<int64_t>(n_tokens, act_tokens);
        const char *  data  = stage_prefix(src1, n_act, buf_hid);

        buf_ids.resize(ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, buf_ids.data(), 0, ggml_nbytes(ids));

        for (int64_t i2 = 0; i2 < n_act; ++i2) {
            const char * idrow = (const char *) buf_ids.data() + i2*ids->nb[1];
            for (int64_t i1 = 0; i1 < n_used; ++i1) {
                int32_t ex = 0;
                std::memcpy(&ex, idrow + i1*ids->nb[0], sizeof(ex));
                if (ex < 0 || ex >= n_expert) {
                    continue;
                }
                const float * h = (const float *) (data + i2*src1->nb[2] + i1*src1->nb[1]);
                double * acc = e.hid_sum.data() + (size_t) ex * (size_t) n_ff;
                for (int64_t j = 0; j < n_ff; ++j) {
                    acc[j] += (double) h[j] * (double) h[j];
                }
                e.hid_n[ex]++;
            }
        }
    }
};

static bool cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto * c = (collector *) user_data;
    if (ask) {
        return c->want(t);
    }
    c->observe(t);
    return true;
}

// The dump format, its metadata and both directions of the codec live in
// lib/calib-dump.h: writer and readers have to agree byte for byte, and the
// readers are separate binaries.
using dump_meta = wxq::calib_meta;

using wxq::CALIB_COV;
using wxq::CALIB_HIDDEN;
using wxq::CALIB_EXPERT_IN;

static void write_dump(const std::string & path, const collector & c, const dump_meta & m);

// ---------------------------------------------------------------------------
// depth-first traversal
// ---------------------------------------------------------------------------

struct dfs_driver {
    llama_context * ctx = nullptr;
    llama_batch     batch{};
    int32_t         n_batch = 0;
    collector *     coll = nullptr;
    uint64_t        n_decoded = 0;
    uint64_t        max_nodes = 0;   // 0 = no limit
    bool            stop = false;
    uint64_t        peak_snapshot_bytes = 0;
    uint64_t        live_snapshot_bytes = 0;
    uint64_t        n_total = 0;      // trie nodes, for ETA
    int64_t         t_start_us = 0;
    int64_t         t_last_report_us = 0;
    int64_t         report_period_us = 5000000;

    // Attention window. The measured cost of one 4096-token batch on this box is
    // 3.0s + 0.374ms per token of context already in the cache: at the 133k-deep
    // end of this trie a batch takes ~53s against ~3s at the shallow end, so the
    // traversal spends most of its life in the 10 full-attention blocks and the
    // total is O(sum of node depths), not O(nodes). Capping the window makes it
    // O(nodes * max_ctx/2) instead.
    //
    // This is an approximation, and not a small one: the reset drops the
    // GatedDeltaNet recurrent state as well as the KV, so every segment restarts
    // cold. Routing at 16k of context is not routing at 133k of context. Off by
    // default for that reason.
    uint32_t        max_ctx = 0;
    uint64_t        n_ctx_resets = 0;
    llama_pos       cur_pos = 0;

    // Periodic dump. An 18-minute traversal that writes nothing until the end
    // loses everything to one driver reset, which is how the first full run died
    // at 74.5%.
    std::string     dump_path;
    int64_t         dump_period_us = 30000000;
    int64_t         t_last_dump_us = 0;
    dump_meta       meta{};

    void reset_window() {
        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        cur_pos = 0;
        ++n_ctx_resets;
    }

    void maybe_dump(int64_t t_now, bool force) {
        if (dump_path.empty()) {
            return;
        }
        if (!force && t_now - t_last_dump_us < dump_period_us) {
            return;
        }
        meta.n_decoded    = n_decoded;
        meta.n_total      = n_total;
        meta.n_ctx_resets = n_ctx_resets;
        write_dump(dump_path, *coll, meta);
        t_last_dump_us = t_now;
    }

    bool decode_span(const std::vector<llama_token> & toks) {
        for (size_t off = 0; off < toks.size(); off += (size_t) n_batch) {
            if (max_nodes != 0 && n_decoded >= max_nodes) {
                stop = true;
                return true;
            }
            const size_t n = std::min((size_t) n_batch, toks.size() - off);
            if (max_ctx != 0 && cur_pos + (llama_pos) n > (llama_pos) max_ctx) {
                reset_window();
            }
            common_batch_clear(batch);
            for (size_t i = 0; i < n; ++i) {
                // logits only on the last position of the chunk: nothing here reads
                // them, but a batch with no outputs is not worth relying on and one
                // row of a 248k vocabulary is cheap
                common_batch_add(batch, toks[off + i], cur_pos + (llama_pos) i, { 0 }, i + 1 == n);
            }
            coll->ubatch_idx++;
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("%s: llama_decode failed at pos %d\n", __func__, (int) cur_pos);
                return false;
            }
            const int64_t t1 = ggml_time_us();
            cur_pos   += (llama_pos) n;
            n_decoded += n;
            coll->n_nodes_seen += n;

            // time-based so something always shows up, and so the rate is readable
            // as a function of depth: the 10 full-attention blocks make the
            // per-token cost grow with context, which is what separates inherent
            // attention growth from a constant collector overhead
            if (t1 - t_last_report_us >= report_period_us) {
                const double el   = (double) (t1 - t_start_us) / 1e6;
                const double rate = (double) n_decoded / std::max(el, 1e-6);
                const double frac = n_total ? (double) n_decoded / (double) n_total : 0.0;
                const double eta  = (rate > 0.0 && n_total > n_decoded)
                                  ? (double) (n_total - n_decoded) / rate : 0.0;
                LOG_INF("%s: pos %7d | %7" PRIu64 "/%" PRIu64 " (%4.1f%%) | %7.1f tok/s | elapsed %5.0fs eta %5.0fs\n",
                        __func__, (int) cur_pos, n_decoded, n_total, 100.0*frac, rate, el, eta);
                fflush(stderr);
                t_last_report_us = t1;
            }
            maybe_dump(t1, false);
        }
        return true;
    }

    bool visit(const trie & t, uint64_t node) {
        if (stop) {
            return true;
        }
        // collapse the straight-line run starting at `node`; it ends on the first
        // node that is a leaf or a branch
        std::vector<llama_token> span;
        uint64_t cur = node;
        while (true) {
            span.push_back(t.nodes[cur - 1].token);
            const auto & kids = t.children[cur];
            if (kids.size() == 1) {
                cur = kids[0];
                continue;
            }
            break;
        }

        if (!decode_span(span)) {
            return false;
        }

        const auto & kids = t.children[cur];
        if (kids.empty() || stop) {
            return true;
        }

        // branch: park the state on the host, then replay it for every sibling
        // after the first. cur_pos travels with the blob - with a bounded window
        // the position at a branch point is not derivable from trie depth.
        const size_t snap_size = llama_state_seq_get_size(ctx, 0);
        std::vector<uint8_t> snap(snap_size);
        if (llama_state_seq_get_data(ctx, snap.data(), snap.size(), 0) != snap_size) {
            LOG_ERR("%s: state_seq_get_data short read at pos %d\n", __func__, (int) cur_pos);
            return false;
        }
        const llama_pos snap_pos = cur_pos;
        live_snapshot_bytes += snap_size;
        peak_snapshot_bytes = std::max(peak_snapshot_bytes, live_snapshot_bytes);

        bool ok = true;
        for (size_t i = 0; i < kids.size() && ok; ++i) {
            if (i > 0) {
                if (llama_state_seq_set_data(ctx, snap.data(), snap.size(), 0) != snap_size) {
                    LOG_ERR("%s: state_seq_set_data short write at pos %d\n", __func__, (int) cur_pos);
                    ok = false;
                    break;
                }
                cur_pos = snap_pos;
            }
            ok = visit(t, kids[i]);
        }

        live_snapshot_bytes -= snap_size;
        return ok;
    }
};

// ---------------------------------------------------------------------------
// reporting
// ---------------------------------------------------------------------------

// Optimal reverse water-filling over a set of per-unit weights beats a flat
// allocation by 0.5*log2(AM/GM) bits per weight at equal distortion, so AM/GM of
// the measured statistic is exactly the rate that the skew is worth.
static void am_gm(const std::vector<double> & v, double & ratio, double & delta_bpw, size_t & n_zero) {
    double sum = 0.0;
    double log_sum = 0.0;
    size_t n = 0;
    n_zero = 0;
    for (double x : v) {
        if (x > 0.0) {
            sum += x;
            log_sum += std::log(x);
            ++n;
        } else {
            ++n_zero;
        }
    }
    if (n == 0) {
        ratio = 1.0;
        delta_bpw = 0.0;
        return;
    }
    const double am = sum / (double) n;
    const double gm = std::exp(log_sum / (double) n);
    ratio = (gm > 0.0) ? am / gm : 1.0;
    delta_bpw = 0.5 * std::log2(ratio);
}

// log|A| by in-place Cholesky over the upper triangle; `a` is destroyed.
// Right-looking so the trailing update parallelises; rows are interleaved for the
// same reason as accum_cov. Returns false on a non-positive pivot, which is what
// a covariance estimated from fewer samples than dimensions looks like.
static bool logdet_chol(std::vector<double> & a, int64_t d, int nth, double & logdet) {
    logdet = 0.0;
    for (int64_t k = 0; k < d; ++k) {
        double * ak = a.data() + (size_t) k * (size_t) d;
        if (!(ak[k] > 0.0)) {
            return false;
        }
        const double dk = std::sqrt(ak[k]);
        logdet += 2.0 * std::log(dk);
        const double inv = 1.0 / dk;
        for (int64_t j = k; j < d; ++j) {
            ak[j] *= inv;
        }
        auto upd = [&](int64_t off, int64_t step) {
            for (int64_t i = k + 1 + off; i < d; i += step) {
                double * ai = a.data() + (size_t) i * (size_t) d;
                const double aki = ak[i];
                if (aki == 0.0) {
                    continue;
                }
                for (int64_t j = i; j < d; ++j) {
                    ai[j] -= aki * ak[j];
                }
            }
        };
        if (nth > 1 && d - k - 1 > 256) {
            std::vector<std::thread> pool;
            pool.reserve(nth);
            for (int t = 0; t < nth; ++t) {
                pool.emplace_back(upd, (int64_t) t, (int64_t) nth);
            }
            for (auto & th : pool) {
                th.join();
            }
        } else {
            upd(0, 1);
        }
    }
    return true;
}

// The diagonal of A says nothing about rank. These three numbers do:
//
//   eig AM/GM   AM/GM of the *eigenvalues*, so dR = 0.5*log2 of it is the rate a
//               shared orthogonal basis is worth. The diagonal-only version of
//               this number is what the earlier passes reported, and it is blind
//               to a flat-diagonal-but-rank-deficient matrix.
//   PR          tr(A)^2 / tr(A^2), the participation ratio: d when isotropic,
//               1 when rank one. A cheap effective rank that needs no eigenvalues.
//   cone        mean over experts of ||mu_e||^2 / tr(A_e). This is the number that
//               decides whether cross-expert prototypes can work: prototypes need
//               W_e to be accurate only on the input cone the router actually
//               sends to expert e, and cone -> 1 means that cone is one direction.
static void report_spectrum(const collector & c, int nth) {
    LOG_INF("\n");
    LOG_INF("%-6s %8s %12s %10s %10s %10s %10s %10s\n",
            "blk", "n_cov", "eig AM/GM", "dR bpw", "PR", "PR/d", "cone", "S_B/tr(A)");

    std::vector<double> work;
    double sum_dbpw = 0.0;
    size_t n_seen   = 0;

    for (const auto & [il, e] : c.layers) {
        if (e.cov.empty() || e.n_cov == 0) {
            continue;
        }
        const int64_t d = (int64_t) e.n_embd;
        const double  s = 1.0 / (double) e.n_cov;

        double tr = 0.0;
        double tr2 = 0.0;
        for (int64_t i = 0; i < d; ++i) {
            const double * ci = e.cov.data() + (size_t) i * (size_t) d;
            tr += ci[i] * s;
            tr2 += (ci[i]*s) * (ci[i]*s);
            for (int64_t j = i + 1; j < d; ++j) {
                const double v = ci[j] * s;
                tr2 += 2.0 * v * v;
            }
        }
        const double am = tr / (double) d;
        const double pr = (tr2 > 0.0) ? (tr*tr) / tr2 : 0.0;

        // ridge, escalated until the factorisation survives; it only ever biases
        // the geometric mean up, i.e. it under-reports the available rate
        double logdet = 0.0;
        bool   ok     = false;
        double eps    = 1e-10;
        for (int attempt = 0; attempt < 6 && !ok; ++attempt, eps *= 100.0) {
            work.assign((size_t) d * (size_t) d, 0.0);
            for (int64_t i = 0; i < d; ++i) {
                const double * ci = e.cov.data() + (size_t) i * (size_t) d;
                double * wi = work.data() + (size_t) i * (size_t) d;
                for (int64_t j = i; j < d; ++j) {
                    wi[j] = ci[j] * s;
                }
                wi[i] += eps * am;
            }
            ok = logdet_chol(work, d, nth, logdet);
        }

        double ratio = 1.0;
        double dbpw  = 0.0;
        if (ok) {
            const double gm = std::exp(logdet / (double) d);
            ratio = (gm > 0.0) ? am / gm : 1.0;
            dbpw  = 0.5 * std::log2(std::max(ratio, 1.0));
            sum_dbpw += dbpw;
            ++n_seen;
        }

        // cone concentration and between-expert scatter, both from the per-expert
        // first moment
        double cone = 0.0;
        double sb   = 0.0;
        if (!e.exp_mean.empty()) {
            std::vector<double> mu(d, 0.0);
            uint64_t tot = 0;
            for (uint32_t ex = 0; ex < e.n_expert; ++ex) {
                tot += e.exp_n[ex];
                const double * m = e.exp_mean.data() + (size_t) ex * (size_t) d;
                for (int64_t j = 0; j < d; ++j) {
                    mu[j] += m[j];
                }
            }
            if (tot > 0) {
                for (int64_t j = 0; j < d; ++j) {
                    mu[j] /= (double) tot;
                }
                size_t n_live = 0;
                for (uint32_t ex = 0; ex < e.n_expert; ++ex) {
                    if (e.exp_n[ex] == 0) {
                        continue;
                    }
                    const double   inv = 1.0 / (double) e.exp_n[ex];
                    const double * m   = e.exp_mean.data() + (size_t) ex * (size_t) d;
                    const double * dg  = e.exp_diag.data() + (size_t) ex * (size_t) d;
                    double nrm = 0.0, tre = 0.0, dev = 0.0;
                    for (int64_t j = 0; j < d; ++j) {
                        const double mj = m[j] * inv;
                        nrm += mj * mj;
                        tre += dg[j] * inv;
                        const double t = mj - mu[j];
                        dev += t * t;
                    }
                    if (tre > 0.0) {
                        cone += nrm / tre;
                        ++n_live;
                    }
                    sb += ((double) e.exp_n[ex] / (double) tot) * dev;
                }
                if (n_live) {
                    cone /= (double) n_live;
                }
            }
        }

        char s_ratio[32];
        char s_dbpw[32];
        if (ok) {
            std::snprintf(s_ratio, sizeof(s_ratio), "%.3f", ratio);
            std::snprintf(s_dbpw,  sizeof(s_dbpw),  "%.3f", dbpw);
        } else {
            std::snprintf(s_ratio, sizeof(s_ratio), "singular");
            std::snprintf(s_dbpw,  sizeof(s_dbpw),  "-");
        }
        LOG_INF("%-6d %8" PRIu64 " %12s %10s %10.1f %10.4f %10.4f %10.4f\n",
                il, e.n_cov, s_ratio, s_dbpw,
                pr, pr / (double) d, cone, (tr > 0.0) ? sb / tr : 0.0);
    }

    if (n_seen) {
        LOG_INF("\n");
        LOG_INF("mean over blocks: eigen-basis %.3f bpw (diagonal-only estimate was the 'act' column)\n",
                sum_dbpw / (double) n_seen);
    }
}

// Per-expert, per-neuron hidden energy. This is the tier the input moment cannot
// reach: the gate nonlinearity is what kills a neuron on a given domain, so the
// death is only visible after it.
static void report_hidden(const collector & c) {
    LOG_INF("\n");
    LOG_INF("%-6s %8s %12s %10s %14s %14s\n",
            "blk", "n_ff", "neur AM/GM", "dR bpw", "dead<1e-3", "dead<1e-2");

    double sum_dbpw = 0.0;
    size_t n_seen   = 0;
    for (const auto & [il, e] : c.layers) {
        if (e.hid_sum.empty()) {
            continue;
        }
        const size_t nff = e.n_ff;
        double sum_ratio = 0.0;
        double sum_dr    = 0.0;
        size_t n_live    = 0;
        uint64_t dead3 = 0, dead2 = 0, tot = 0;
        std::vector<double> v(nff);
        for (uint32_t ex = 0; ex < e.n_expert; ++ex) {
            if (e.hid_n[ex] == 0) {
                continue;
            }
            const double   inv = 1.0 / (double) e.hid_n[ex];
            const double * h   = e.hid_sum.data() + (size_t) ex * nff;
            double mean = 0.0;
            for (size_t j = 0; j < nff; ++j) {
                v[j]  = h[j] * inv;
                mean += v[j];
            }
            mean /= (double) nff;
            double ratio, dr;
            size_t nz;
            am_gm(v, ratio, dr, nz);
            sum_ratio += ratio;
            sum_dr    += dr;
            ++n_live;
            for (size_t j = 0; j < nff; ++j) {
                if (v[j] < 1e-3 * mean) { ++dead3; }
                if (v[j] < 1e-2 * mean) { ++dead2; }
                ++tot;
            }
        }
        if (!n_live) {
            continue;
        }
        LOG_INF("%-6d %8u %12.3f %10.3f %13.2f%% %13.2f%%\n",
                il, e.n_ff, sum_ratio / (double) n_live, sum_dr / (double) n_live,
                100.0 * (double) dead3 / (double) std::max<uint64_t>(tot, 1),
                100.0 * (double) dead2 / (double) std::max<uint64_t>(tot, 1));
        sum_dbpw += sum_dr / (double) n_live;
        ++n_seen;
    }
    if (n_seen) {
        LOG_INF("\n");
        LOG_INF("mean over blocks: per-neuron energy skew %.3f bpw\n", sum_dbpw / (double) n_seen);
    }
}

// Written repeatedly during the run, so the swap has to be atomic: a consumer
// polling the path must never observe a half-written header. write_calib does the
// rename; the format itself is documented in lib/calib-dump.h.
static void write_dump(const std::string & path, const collector & c, const dump_meta & m) {
    std::string err;
    if (!wxq::write_calib(path, m, c.layers, err)) {
        LOG_ERR("%s: %s\n", __func__, err.c_str());
    }
}

static void report(const collector & c) {
    LOG_INF("\n");
    LOG_INF("%-6s %11s %8s %10s %11s %8s %12s %10s\n",
            "blk", "route AM/GM", "dR bpw", "dead exp", "act AM/GM", "dR bpw", "route pos", "act pos");

    double sum_route = 0.0;
    double sum_act   = 0.0;
    for (const auto & [il, e] : c.layers) {
        std::vector<double> counts(e.counts.begin(), e.counts.end());

        double r_ratio, r_dbpw, a_ratio, a_dbpw;
        size_t r_zero, a_zero;
        am_gm(counts,    r_ratio, r_dbpw, r_zero);
        am_gm(e.act_sum, a_ratio, a_dbpw, a_zero);

        LOG_INF("%-6d %11.3f %8.3f %6zu/%-4u %11.3f %8.3f %12" PRIu64 " %10" PRIu64 "\n",
                il, r_ratio, r_dbpw, r_zero, e.n_expert, a_ratio, a_dbpw, e.n_route, e.n_pos);

        sum_route += r_dbpw;
        sum_act   += a_dbpw;
    }

    const double n = (double) std::max<size_t>(c.layers.size(), 1);
    LOG_INF("\n");
    LOG_INF("mean over blocks: routing %.3f bpw, activation %.3f bpw, combined %.3f bpw\n",
            sum_route / n, sum_act / n, (sum_route + sum_act) / n);
    LOG_INF("note: the activation column is E[x_j^2] only. Multiply by the per-column\n");
    LOG_INF("      weight variance before reading it as the full importance skew.\n");
}

// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    common_params params;

    std::string trie_dir = "wanxiangqi/calibration";
    std::string out_path = "wanxiangqi-trie-calib.bin";
    uint32_t    act_stride   = 8;
    uint32_t    act_tokens   = 64;
    uint32_t    route_stride = 1;
    uint32_t    max_ctx      = 0;
    uint32_t    dump_secs    = 30;
    uint32_t    want_cov     = 0;
    uint32_t    want_hidden  = 0;
    int         n_threads    = (int) std::max(1u, std::thread::hardware_concurrency());
    uint64_t    max_nodes    = 0;
    for (int i = 1; i < argc; ++i) {
        const char * val = (i + 1 < argc) ? argv[i + 1] : nullptr;
        if (val == nullptr) {
            continue;
        }
        bool taken = true;
        if (std::strcmp(argv[i], "--trie") == 0) {
            trie_dir = val;
        } else if (std::strcmp(argv[i], "--calib-out") == 0) {
            out_path = val;
        } else if (std::strcmp(argv[i], "--calib-act-stride") == 0) {
            act_stride = (uint32_t) std::strtoul(val, nullptr, 10);
        } else if (std::strcmp(argv[i], "--calib-act-tokens") == 0) {
            act_tokens = (uint32_t) std::strtoul(val, nullptr, 10);
        } else if (std::strcmp(argv[i], "--calib-route-stride") == 0) {
            route_stride = (uint32_t) std::strtoul(val, nullptr, 10);
        } else if (std::strcmp(argv[i], "--calib-max-ctx") == 0) {
            max_ctx = (uint32_t) std::strtoul(val, nullptr, 10);
        } else if (std::strcmp(argv[i], "--calib-dump-interval") == 0) {
            dump_secs = (uint32_t) std::strtoul(val, nullptr, 10);
        } else if (std::strcmp(argv[i], "--calib-cov") == 0) {
            want_cov = (uint32_t) std::strtoul(val, nullptr, 10);
        } else if (std::strcmp(argv[i], "--calib-hidden") == 0) {
            want_hidden = (uint32_t) std::strtoul(val, nullptr, 10);
        } else if (std::strcmp(argv[i], "--calib-threads") == 0) {
            n_threads = (int) std::max(1L, std::strtol(val, nullptr, 10));
        } else if (std::strcmp(argv[i], "--calib-max-nodes") == 0) {
            max_nodes = std::strtoull(val, nullptr, 10);
        } else {
            taken = false;
        }
        if (taken) {
            argv[i] = argv[i + 1] = const_cast<char *>("--ignore");
            ++i;
        }
    }
    // strip the options we consumed so the common parser does not see them
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--ignore") != 0) {
            args.push_back(argv[i]);
        }
    }

    collector coll;
    coll.act_stride          = act_stride;
    coll.act_tokens          = act_tokens;
    coll.route_stride        = route_stride;
    coll.want_cov            = want_cov    != 0;
    coll.want_hidden         = want_hidden != 0;
    // the per-expert first moment is what makes the covariance interpretable
    // (cone concentration, between-expert scatter), so it rides along with it
    coll.want_expert_in      = want_cov    != 0;
    coll.n_threads           = n_threads;
    params.cb_eval           = cb_eval;
    params.cb_eval_user_data = &coll;
    params.warmup            = false;

    common_init();

    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    // Recurrent-state rollback snapshots must be off. When any MTP/EAGLE3 draft
    // type is active, common_params::need_n_rs_seq() returns draft.n_max, and the
    // recurrent memory then forces every ubatch down to (n_rs_seq + 1) tokens so
    // that a sequence's rollback tail stays in one ubatch. Measured on this model
    // that is 4 tokens per graph launch: ~128x more graph launches than the 512
    // this tool wants, which is what made a first attempt look hung.
    //
    // This collector never uses the rollback ring - the DFS parks state with
    // llama_state_seq_get_data()/set_data() instead - so the snapshots are pure
    // overhead here.
    params.speculative.types.clear();
    params.speculative.draft.n_max = 0;

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init model/context\n", __func__);
        return 1;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    trie t;
    try {
        t = load_trie(trie_dir, n_vocab);
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        return 1;
    }

    size_t n_branch = 0;
    uint64_t max_depth = 0;
    {
        std::vector<uint64_t> depth(t.n_nodes() + 1, 0);
        for (uint64_t i = 0; i < t.n_nodes(); ++i) {
            depth[i + 1] = depth[t.nodes[i].parent] + 1;
            max_depth = std::max(max_depth, depth[i + 1]);
        }
        for (const auto & kids : t.children) {
            if (kids.size() > 1) {
                ++n_branch;
            }
        }
    }

    LOG_INF("%s: trie %s\n", __func__, trie_dir.c_str());
    LOG_INF("%s:   nodes %" PRIu64 ", leaves %zu, branch points %zu, max depth %" PRIu64 "\n",
            __func__, t.n_nodes(), t.leaves.size(), n_branch, max_depth);
    LOG_INF("%s:   leaf positions %" PRIu64 " -> DFS decodes %" PRIu64 " (%.1fx saving)\n",
            __func__, t.n_leaf_positions, t.n_nodes(),
            (double) t.n_leaf_positions / (double) std::max<uint64_t>(t.n_nodes(), 1));

    const uint32_t n_ctx   = llama_n_ctx(ctx);
    const uint32_t n_batch = llama_n_batch(ctx);
    if (max_ctx != 0 && max_ctx < n_batch) {
        LOG_ERR("%s: --calib-max-ctx %u is below n_batch %u; a single batch would not fit the window\n",
                __func__, max_ctx, n_batch);
        return 1;
    }
    // A bounded run stops after max_nodes decodes, and depth never exceeds the
    // number of positions decoded, so it can legitimately use a context far
    // smaller than the full trie needs. This is what makes it possible to A/B a
    // short prefix against the production server's -c 32768. --calib-max-ctx caps
    // it outright by restarting the sequence, which is the only way this fits in a
    // context smaller than the 133k-deep path.
    uint64_t depth_needed = max_nodes ? std::min<uint64_t>(max_depth, max_nodes) : max_depth;
    if (max_ctx != 0) {
        depth_needed = std::min<uint64_t>(depth_needed, max_ctx);
    }
    if (depth_needed > n_ctx) {
        LOG_ERR("%s: context %u is smaller than the deepest path this run reaches %" PRIu64 "; pass -c %" PRIu64 " or --calib-max-ctx %u\n",
                __func__, n_ctx, depth_needed, depth_needed, n_ctx);
        return 1;
    }

    LOG_INF("%s: n_ctx %u, n_batch %u, n_ubatch %u, n_rs_seq %u\n", __func__,
            n_ctx, n_batch, llama_n_ubatch(ctx), llama_n_rs_seq(ctx));
    LOG_INF("%s: act_stride %u, act_tokens %u, route_stride %u, max_ctx %u, dump every %us -> %s\n",
            __func__, act_stride, act_tokens, route_stride, max_ctx, dump_secs, out_path.c_str());
    LOG_INF("%s: cov %s, hidden %s, threads %d\n", __func__,
            want_cov ? "on" : "off", want_hidden ? "on" : "off", n_threads);
    if (want_cov) {
        LOG_WRN("%s: the covariance spectrum needs many more samples than dimensions. n_embd is\n", __func__);
        LOG_WRN("%s: 2048 here, so anything under ~20k sampled positions gives a Marchenko-Pastur\n", __func__);
        LOG_WRN("%s: spectrum rather than the model's: use --calib-act-stride 1 and a large\n", __func__);
        LOG_WRN("%s: --calib-act-tokens.\n", __func__);
    }
    if (max_ctx != 0) {
        LOG_WRN("%s: --calib-max-ctx %u restarts the sequence every %u tokens. This drops the\n",
                __func__, max_ctx, max_ctx);
        LOG_WRN("%s: GatedDeltaNet recurrent state as well as the KV, so routing is measured at\n", __func__);
        LOG_WRN("%s: <=%u tokens of context and not at the %" PRIu64 " this corpus reaches.\n",
                __func__, max_ctx, max_depth);
    }

    dfs_driver drv;
    drv.ctx        = ctx;
    drv.n_batch    = (int32_t) n_batch;
    drv.coll       = &coll;
    drv.max_nodes  = max_nodes;
    drv.max_ctx    = max_ctx;
    drv.n_total    = max_nodes ? std::min<uint64_t>(max_nodes, t.n_nodes()) : t.n_nodes();
    drv.t_start_us = ggml_time_us();
    drv.t_last_report_us = drv.t_start_us;
    drv.t_last_dump_us   = drv.t_start_us;
    drv.dump_path      = out_path;
    drv.dump_period_us = (int64_t) dump_secs * 1000000;
    drv.meta.max_ctx      = max_ctx;
    drv.meta.act_stride   = act_stride;
    drv.meta.act_tokens   = act_tokens;
    drv.meta.route_stride = route_stride;
    drv.meta.flags        = (want_cov    ? (CALIB_COV | CALIB_EXPERT_IN) : 0u)
                          | (want_hidden ?  CALIB_HIDDEN                : 0u);
    drv.batch      = llama_batch_init((int32_t) n_batch, 0, 1);

    // A Vulkan device reset surfaces as a thrown vk::DeviceLostError out of
    // llama_decode. That killed the first full run at 74.5% with nothing on disk;
    // whatever has been accumulated is still worth keeping, so land it before
    // rethrowing the failure into the exit code.
    bool ok = true;
    try {
        for (uint64_t root_child : t.children[0]) {
            llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
            drv.cur_pos = 0;
            ok = drv.visit(t, root_child);
            if (!ok) {
                break;
            }
        }
    } catch (const std::exception & e) {
        LOG_ERR("%s: traversal aborted: %s\n", __func__, e.what());
        ok = false;
    }

    llama_batch_free(drv.batch);

    LOG_INF("%s: decoded %" PRIu64 " positions, %" PRIu64 " context resets, peak host snapshot %.2f GiB\n",
            __func__, drv.n_decoded, drv.n_ctx_resets,
            (double) drv.peak_snapshot_bytes / (1024.0*1024.0*1024.0));

    if (coll.layers.empty()) {
        LOG_ERR("%s: no MoE layers observed - is this a routed-expert model?\n", __func__);
        return 1;
    }

    report(coll);
    if (want_cov) {
        const int64_t t0 = ggml_time_us();
        report_spectrum(coll, n_threads);
        LOG_INF("%s: spectrum (Cholesky log-det, %d threads) took %.1fs\n",
                __func__, n_threads, (double) (ggml_time_us() - t0) / 1e6);
    }
    if (want_hidden) {
        report_hidden(coll);
    }
    drv.meta.complete = ok ? 1 : 0;
    drv.maybe_dump(ggml_time_us(), true);
    LOG_INF("%s: wrote %s (%s)\n", __func__, out_path.c_str(),
            ok ? "complete" : "PARTIAL - traversal did not finish");

    return ok ? 0 : 1;
}
