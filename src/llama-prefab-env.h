#pragma once

// Default-off env gates for the prefab n_kv pad (this layer, not the upstream
// LLAMA_PREFAB_ENABLE gate). Each value is read once (function-local static) and counts
// as enabled only for the exact string "1": unset, "0", "true", "01", "1 " all leave
// stock behaviour. When both are unset the only cost is one boolean check per call site.
#include <cstdlib>

static inline bool llama_prefab_nkv_pad_enabled() {
    static const bool on = []() {
        const char * s = getenv("LLAMA_PREFAB_NKV_PAD");
        return s != nullptr && s[0] == '1' && s[1] == '\0';
    }();
    return on;
}

static inline bool llama_prefab_nkv_obs_enabled() {
    static const bool on = []() {
        const char * s = getenv("LLAMA_PREFAB_NKV_OBS");
        return s != nullptr && s[0] == '1' && s[1] == '\0';
    }();
    return on;
}

// production A+B compose lane (r7). Same exact-"1" contract as the gates above:
// unset, "0", "true", "01", "1 " all leave stock behaviour. When exactly "1" it selects the
// whole-graph MTP lane with the A x B compose as the production path on its own (no
// LLAMA_PREFAB_MTP / LLAMA_PREFAB_ENABLE / LLAMA_PREFAB_EXPERIMENTAL needed): eligible
// Qwen4Exp MTP contexts cache one whole graph per envelope width and reuse it across
// requests while the graph inputs allow (exact can_reuse, no pad / headroom / bucket),
// and each rebuild splices the cached A prologue under the fresh B block.
static inline bool llama_prefab_ab_enabled() {
    static const bool on = []() {
        const char * s = getenv("GGML_TP5_PREFAB_AB");
        return s != nullptr && s[0] == '1' && s[1] == '\0';
    }();
    return on;
}
