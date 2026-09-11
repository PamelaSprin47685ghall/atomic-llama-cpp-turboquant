// MoE calibration dump (WXQCAL03): the shared on-disk contract between the
// collector that produces it (calib/trie-calib.cpp) and everything that reads it
// (the offline probes, the imatrix converter, the transform builder).
//
// Layout, all little endian:
//
//   magic     char[8]  "WXQCAL03"
//   header    u32 n_entries, max_ctx, act_stride, act_tokens, route_stride, flags
//             u64 n_decoded, n_total, n_ctx_resets
//             u8  complete, u8 pad[7]
//   entry[]   u32 layer, n_expert, n_embd, n_ff
//             u64 n_pos, n_route, n_cov
//             u64 counts[n_expert]
//             f64 act_sum[n_embd]
//             if CALIB_EXPERT_IN: u64 exp_n[n_expert], f64 exp_mean[n_expert*n_embd],
//                                 f64 exp_diag[n_expert*n_embd]
//             if CALIB_HIDDEN:    u64 hid_n[n_expert], f64 hid_sum[n_expert*n_ff]
//             if CALIB_COV:       f64 cov[n_embd*n_embd]   (full square on disk)
//
// The dump is rewritten while a run is in flight, so `complete` and
// n_decoded/n_total are load-bearing: a consumer that cannot tell a finished
// corpus from a crashed one has no basis for trusting the tail of the expert
// histogram.
//
// Python readers of the same format: probe/wxqcal.py.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace wxq {

enum calib_flag {
    CALIB_COV       = 1u << 0,
    CALIB_HIDDEN    = 1u << 1,
    CALIB_EXPERT_IN = 1u << 2,
};

struct calib_meta {
    uint32_t n_entries    = 0;
    uint32_t max_ctx      = 0;   // 0 = unbounded, else attention window in tokens
    uint32_t act_stride   = 0;
    uint32_t act_tokens   = 0;
    uint32_t route_stride = 0;
    uint32_t flags        = 0;
    uint64_t n_decoded    = 0;
    uint64_t n_total      = 0;
    uint64_t n_ctx_resets = 0;
    uint8_t  complete     = 0;
};

// One layer's routed-expert statistics. Used by both ends: the collector fills
// it in as it accumulates, the readers receive it as parsed.
//
// `cov` is the one field whose in-memory meaning differs by direction. The
// collector keeps only the upper triangle live (accumulation touches j >= i);
// write_calib mirrors it on the way out, so a reader always gets a full square.
struct calib_layer {
    uint32_t              n_expert = 0;
    uint32_t              n_embd   = 0;
    uint32_t              n_ff     = 0;

    std::vector<uint64_t> counts;    // [n_expert]  times this expert was routed to
    std::vector<double>   act_sum;   // [n_embd]    sum of x_j^2 over positions
    uint64_t              n_pos   = 0;
    uint64_t              n_route = 0;

    // Pooled input second moment. The diagonal alone cannot see rank -- a flat
    // diagonal can still be badly rank deficient -- and rank is what decides
    // whether a shared basis buys anything.
    std::vector<double>   cov;       // [n_embd*n_embd]
    uint64_t              n_cov = 0;

    // Per-expert input first moment and diagonal. Full per-expert covariance is
    // 256*2048^2*8 = 8.6 GiB per layer, so it is out.
    std::vector<double>   exp_mean;  // [n_expert*n_embd]
    std::vector<double>   exp_diag;  // [n_expert*n_embd]
    std::vector<uint64_t> exp_n;     // [n_expert]

    // Per-expert per-neuron hidden energy E[h_j^2], h = act(W_gate x) * (W_up x).
    // The input moment cannot see neuron death: it happens after the gate.
    std::vector<double>   hid_sum;   // [n_expert*n_ff]
    std::vector<uint64_t> hid_n;     // [n_expert]
};

struct calib_dump {
    calib_meta                     meta;
    std::map<int, calib_layer>     layers;
};

// Reads `path`. `want` selects layers; empty means all. Layers outside `want` are
// skipped without allocating. `CALIB_COV` is only materialised when want_cov,
// because it is n_embd^2 doubles per layer (32 MiB at n_embd 2048).
// Returns false and sets `err` on a malformed dump.
bool load_calib(const std::string & path, const std::vector<int> & want, bool want_cov,
                calib_dump & out, std::string & err);

// Writes atomically via `path`.tmp + rename. Only the fields enabled in
// meta.flags are emitted; a flag set with empty data is skipped, matching what
// the collector does when a statistic was not requested.
// Returns false and sets `err` on any I/O failure.
bool write_calib(const std::string & path, const calib_meta & meta,
                 const std::map<int, calib_layer> & layers, std::string & err);

} // namespace wxq
