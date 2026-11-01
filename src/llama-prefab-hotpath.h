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
//   - the armed counter-write helpers (llama_prefab_stat_tick/add); the POD
//     itself (llama_prefab_stats) lives in the public header include/llama.h
//     so external consumers read it through the llama_prefab_* C API, and a
//     CPU mock still drives the identical production decision code.
//
// Every counter write in the lane is armed only by the exact-"1" gate
// (GGML_TP5_PREFAB_STATS, llama_prefab_stats_enabled, defined in
// llama-prefab-mtp.cpp); with the gate off no hot-path write happens and the
// stock branches are behaviourally identical to the unpatched builder.
//
#include "llama.h" // struct llama_prefab_stats (public C API POD)

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
// struct llama_prefab_stats moved to the public header (include/llama.h): the
// counters are externally consumable through the llama_prefab_* C API, so the
// POD lives where external consumers can see its layout.
