#pragma once
//
// llama-prefab-hotpath.h — ggml-free hot-path seam of the A x B whole-graph
// MTP slot lane (src/llama-prefab-mtp.cpp).
//
// This header carries the production contracts that must stay testable and
// observable without ggml:
//
//   - the padded KV window the whole-graph slot is composed for (the reuse
//     contract of the lane) and the fit judgement the slot reuse gate uses;
//   - the organic counter events POD (llama_prefab_stats), moved here so a
//     CPU mock drives the identical production decision code and counters.
//
// Every counter write in the lane is armed only by the exact-"1" gate
// (GGML_TP5_PREFAB_STATS, llama_prefab_stats_enabled, defined in
// llama-prefab-mtp.cpp); with the gate off no hot-path write happens and the
// stock branches are behaviourally identical to the unpatched builder.
//
#include <cstdint>

#include <algorithm>

// exact-"1" arm gate of every build/reuse counter write in the lane
bool llama_prefab_stats_enabled();

// the padded KV width a whole-graph slot is composed for, derived from the
// live n_kv at compose time: a 1024-aligned window (>= 1024) that absorbs
// n_kv growth across steps so the composed graph is reused instead of
// rebuilt while the live width stays inside the window. kv_size (the
// physical cache size; 0 skips the clamp) bounds the window.
inline uint32_t llama_prefab_nkv_window(uint32_t n_kv, uint32_t kv_size) {
    const uint32_t w = n_kv <= 1024u ? 1024u : ((n_kv + 1023u) / 1024u) * 1024u;
    return kv_size ? std::min(w, kv_size) : w;
}

// the slot reuse judgement: the composed graph stays valid while the live
// width fits the window it was composed for (pad columns are -INF-masked by
// the prefab mask writer, so attention never reads beyond the live width)
inline bool llama_prefab_nkv_fits(uint32_t n_kv_pad, uint32_t n_kv) {
    return n_kv_pad >= n_kv;
}

// armed counter writes: the ONLY sanctioned mutation of llama_prefab_stats in
// the lane. Both no-op unless the exact-"1" gate (GGML_TP5_PREFAB_STATS) is
// on, so stats-off never writes -- stock hot-path parity is structural, not
// observational.
inline void llama_prefab_stat_tick(uint64_t & counter) {
    if (llama_prefab_stats_enabled()) {
        ++counter;
    }
}

inline void llama_prefab_stat_add(uint64_t & counter, uint64_t v) {
    if (llama_prefab_stats_enabled()) {
        counter += v;
    }
}

// per-context counters (all organic: incremented only where the event happens;
// armed by llama_prefab_stats_enabled, the exact-"1" gate above)
struct llama_prefab_stats {
    uint64_t n_ubatch_prefab;       // ubatches executed on an envelope graph
    uint64_t n_ubatch_stock;        // ubatches executed by the stock builder path
    uint64_t n_stock_builds;        // stock model.build_graph() invocations
    uint64_t n_stock_reuses;        // stock graph reuses (can_reuse == true)
    uint64_t n_envelope_builds;     // envelope graph compositions
    uint64_t n_envelope_reuses;     // envelope graph reused without rebuild
    uint64_t n_envelope_allocs;     // ggml_backend_sched_alloc_graph on an envelope sched
    uint64_t n_sub_envelope;        // ubatches with n_tokens < n_env (runtime padded)
    uint64_t n_pad_rows;            // total pad rows executed
    uint64_t n_factor_compiles;     // factor modules compiled
    uint64_t n_factor_cache_hits;   // factor modules taken from the cache while composing
    uint64_t n_factor_splices;      // factor modules spliced into an envelope
    uint64_t n_factor_clones;       // tensors cloned by splicing
    uint64_t n_cache_resets;        // topology-key invalidations of the factor cache
    uint64_t n_fallback_no_scratch; // runtime fallback to stock: no free scratch KV cell
    uint64_t n_fallback_runtime;    // runtime fallback to stock: adapters / samplers / attention views / ...
    uint64_t last_n_env;            // envelope width of the last prefab ubatch
    uint64_t last_n_tokens;         // real n_tokens of the last prefab ubatch
    uint64_t last_graph_nodes;      // node count of the last executed envelope graph
    uint64_t last_inp_tokens_ne0;   // ne[0] of the token input of the executed envelope graph (== last_n_env)
    uint64_t n_decouple_declines;    // decouple iface: ubatches declined by the compose layer (gate on)
    uint64_t n_decouple_handled;     // decouple iface: ubatches executed by the compose layer (gate on)
    // MTP factor compose (GGML_TP5_PREFAB_IN_GRAPH=1), all organic:
    uint64_t n_mtp_compose_hits;     // DECODER_MTP ubatches executed on a composed A x B envelope graph
    uint64_t n_mtp_a_compiles;       // A (head prologue) modules built
    uint64_t n_mtp_a_reuses;         // A modules taken from the cache while composing
    uint64_t n_mtp_b_builds;         // B (block + epilogue) parts emitted into an envelope
    uint64_t n_mtp_a_clones;         // tensors cloned from A while splicing it into an envelope
    uint64_t n_mtp_declines;         // DECODER_MTP ubatches the compose layer declined (stock fallback)
    uint64_t build_ns;              // wall time in instrumented request graph builds, excluding allocation/compute
    uint64_t n_nodes_built;         // nodes in successfully emitted request graphs, plus compiled MTP A graphs
    uint64_t template_hits;         // MTP A template cache hits, not whole-graph hits
};
