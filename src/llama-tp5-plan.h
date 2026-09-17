// TP5 plan for qwen4exp: immutable role tables, per-rank slice rules and
// memory budget, shared by the model split-state adapter and the CPU tests.
//
// This is a pure computation module: no Vulkan, no model loading, no GPU.
// It mirrors tools/tp5/tp5-inspect-model.py; both must stay consistent
// (the Python tool emits the manifest, this header drives the runtime).
//
// Design references: TP5.md sections 3.3, 4.4, 6, 7.1, 8.1, 11, 12.

#pragma once

#include "ggml-backend.h"
#include "llama-hparams.h"

#include <cstdlib>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

// Maximum TP ranks this plan supports (target machine has 5 RX 6800).
#define LLAMA_TP5_MAX_RANKS 8

struct llama_tp5_error {
    std::string code;   // e.g. "TP5_E_QUANT_SLICE"
    std::string detail; // human-readable fields
};

// How a tensor is laid out across ranks.
enum class llama_tp5_layout {
    MIRRORED,          // every rank holds the full tensor
    SPLIT_AXIS0,       // contiguous ranges on ne[0]; per-rank partial sums
    SPLIT_AXIS1,       // contiguous ranges on ne[1]; per-rank output rows
    SPLIT_AXIS1_REPL,  // axis-1 ranges, declared cross-rank replicas (KV)
    CPU_RESIDENT,      // never uploaded (PLE n-gram table)
};

// Semantic class of a qwen4exp tensor under the TP5 A-F plan.
enum class llama_tp5_semantic {
    HC,             // hyper-connection weights, mirrored (A-F)
    MOE_GATE_UP,    // routed expert gate/up: split output rows
    MOE_DOWN,       // routed expert down: split input channels
    MOE_ROUTER,     // router, mirrored
    SHEXP_GATE_UP,  // shared expert gate/up: split output rows
    SHEXP_DOWN,     // shared expert down: split input channels
    SHEXP_ROUTER,   // shared gate, mirrored
    QSA_Q_GATE,     // interleaved q|gate projection: split whole heads
    QSA_KV,         // k/v projection: replicated head rows per role
    QSA_OUT,        // attention output: split head columns
    QSA_NORM,       // per-head norms, mirrored
    INDEXER,        // indexer projections, mirrored
    GDN_QKV,        // GDN q/k/v projection: prearranged V-head rows
    GDN_GATE,       // GDN output gate: V-head rows
    GDN_OUT,        // GDN output projection: V-head columns
    GDN_SCALAR,     // beta/alpha: V-head columns
    GDN_PARAM,      // ssm_a/dt/norm: mirrored
    GDN_CONV,       // conv1d: prearranged channels
    PLE,            // PLE weights, mirrored
    TOKEN_EMBD,     // embedding, mirrored (first version)
    LM_HEAD,        // output projection: vocab row split
    PLE_TABLE,      // per-layer token embedding table, CPU
    OTHER,          // anything unrecognized: mirrored
};

struct llama_tp5_tensor_plan {
    llama_tp5_semantic semantic;
    llama_tp5_layout   layout;
    int                split_axis;          // -1 for mirrored/cpu
    // per-rank length along split_axis (elements, not bytes)
    std::array<int64_t, LLAMA_TP5_MAX_RANKS> per_rank_len{};
    // for SPLIT_AXIS1_REPL: which global head ranges each rank needs
    std::vector<std::array<int64_t, 2>> head_ranges;
};

// The full immutable plan for one model + rank count.
struct llama_tp5_plan {
    uint32_t ranks = 0;
    bool     replicate_attention = false;
    // Opt-in nonuniform QSA head map (GGML_TP5_QSA_HEADMAP=1): Q counts
    // [5,5,5,5,4] with starts [0,5,10,15,20] over KV counts [1,1,2,1,1] with
    // starts [0,0,0,1,1]. The rank-local Q/KV integer ratio is NOT the true
    // global GQA assignment (rank 2 holds Q heads of two different KV groups),
    // so FLASH_ATTN_EXT consumers must use the explicit local q->kv map.
    bool     qsa_headmap = false;

    // hparams-derived sizes (validated copies)
    int64_t H = 0, L = 0, C = 0, R = 0;
    int64_t E = 0, K = 0, F = 0, Fs = 0;
    int64_t Nq = 0, Nkv = 0, da = 0;
    int64_t Nk = 0, Nv = 0, ds = 0;
    int64_t Ni = 0, di = 0;
    int64_t n_vocab = 0;

    // per-layer type (true = recurrent/GDN, false = full attention)
    std::vector<bool> is_recr;
    uint32_t n_full = 0;

    // QSA role table (TP5.md 7.1)
    std::array<int32_t, LLAMA_TP5_MAX_RANKS> q_role_counts{};
    std::array<int32_t, LLAMA_TP5_MAX_RANKS> kv_role_counts{};
    std::array<int32_t, LLAMA_TP5_MAX_RANKS> kv_head_starts{};
    // KV head instances across full layers, with the same fixed weight/cache roles.
    std::array<int32_t, LLAMA_TP5_MAX_RANKS> kv_instances{};

    // Per-rank local q->kv map: local_qkv_map[r][local_q] = local KV index
    // (rank 2: [0,0,1,1,1]; every other rank: all zeros).
    // Sized by the shared GGML_TP5_HEADMAP_MAX_ENTRIES ABI.
    std::vector<std::array<int32_t, GGML_TP5_HEADMAP_MAX_ENTRIES>> local_qkv_map;

    // GDN V/state heads per rank (TP5.md 8.1): [10,10,10,10,8]
    std::array<int32_t, LLAMA_TP5_MAX_RANKS> gdn_v_heads{};
    // global V head ranges per rank: [first, last)
    std::vector<std::array<int64_t, 2>> gdn_v_ranges;
    // Sharded prearrangement: Q/K count equals V count; replicas use native Nk.
    std::array<int32_t, LLAMA_TP5_MAX_RANKS> gdn_qk_heads{};

    // --- TP5 balanced GDN head map (opt-in GGML_TP5_GDN_HEADMAP=1) ---
    // Main-chosen paired-head assignment: each original V head h uses original
    // QK head (h % Nk); adjacent 128-dim V head PAIRS move together so the
    // 256-element ssm_out quant blocks stay whole. Per-rank global V head order
    // and unique QK head lists (V [10,10,10,10,8], unique QK [4,6,4,4,4],
    // 22 QK copies total vs 16 native: +15% QKV projection rows).
    // Populated only by llama_tp5_plan_build when the mapping is enabled;
    // all-zero gdn_headmap_enabled means native repeated-segment layout.
    static constexpr int LLAMA_TP5_GDN_MAX_V_HEADS   = 10; // max V heads per rank
    static constexpr int LLAMA_TP5_GDN_MAX_QK_HEADS  = 6;  // max unique QK heads per rank
    bool gdn_headmap_enabled = false;
    // gdn_v_global[r][i]: original global V head index at rank-local V slot i
    std::array<std::array<int32_t, LLAMA_TP5_GDN_MAX_V_HEADS>, LLAMA_TP5_MAX_RANKS> gdn_v_global{};
    // gdn_qk_unique[r][i]: original global QK head index at rank-local unique-QK slot i
    std::array<std::array<int32_t, LLAMA_TP5_GDN_MAX_QK_HEADS>, LLAMA_TP5_MAX_RANKS> gdn_qk_unique{};
    // number of valid entries per rank in the arrays above
    std::array<int32_t, LLAMA_TP5_MAX_RANKS> gdn_v_global_count{};
    std::array<int32_t, LLAMA_TP5_MAX_RANKS> gdn_qk_unique_count{};

    // Local QKV channel offsets (elements along the projection row axis):
    // local row layout is [Q unique | K unique | V ordered] with
    // q_off = 0, k_off = ds * gdn_qk_unique_count[r],
    // v_off = ds * (2 * gdn_qk_unique_count[r]).
    int64_t gdn_local_qk_rows(uint32_t rank) const { return ds * gdn_qk_unique_count[rank]; }
    int64_t gdn_local_v_off(uint32_t rank) const { return 2 * gdn_local_qk_rows(rank); }

    // One FFN reduction per layer, plus attention when it is sharded.
    uint32_t expected_events = 0;

    // Plan for one named tensor; returns error via ok=false.
    llama_tp5_tensor_plan plan_tensor(const std::string & name, const int64_t ne[4],
                                      uint32_t type_id, int64_t blck_size,
                                      bool & ok, llama_tp5_error & err) const;

    // Per-rank static weight bytes for the given tensor set.
    // tensors: (name, ne[4], type_id, blck_size, row_size_bytes, n_outer)
    struct tensor_info {
        std::string name;
        int64_t ne[4];
        uint32_t type_id;
        int64_t blck_size;
    };
    std::vector<int64_t> weight_bytes_per_rank(const std::vector<tensor_info> & tensors,
                                               bool & ok, llama_tp5_error & err) const;

    // Validate the plan against the model hparams; fills err on failure.
    bool validate(const llama_hparams & hp, uint32_t n_devices, llama_tp5_error & err) const;
};

// Build the plan from hparams (no GGUF access; shapes come from hp).
// Returns ok=false with a structured error when the configuration is
// unsupported (wrong rank count, illegal head splits, ...).
bool llama_tp5_plan_build(const llama_hparams & hp,
                          uint32_t              n_devices,
                          int64_t               n_vocab,
                          bool                  replicate_attention,
                          llama_tp5_plan &      out,
                          llama_tp5_error &     err);

// TP5.md §8.2: global V head h uses Q/K head (h % Nk) after prearrangement.
int64_t llama_tp5_gdn_qk_global_head(const llama_tp5_plan & plan, int64_t global_v_head);

// True when the QSA headmap opt-in is active for this model+rank count
// True when the QSA headmap opt-in is requested via GGML_TP5_QSA_HEADMAP=1.
// The plan build fails closed (TP5_E_QSA_HEADMAP) unless the geometry is
// exactly the balanced QWEN4EXP 5-rank Q24/KV2 assignment.
bool llama_tp5_qsa_headmap_enabled(void);

// Stamp the per-local-Q-head -> local-KV-head map of the given rank onto a
// rank-local GGML_OP_FLASH_ATTN_EXT node using the shared ggml_tp5_headmap_set
// ABI (op_params i32 slots 8/9/10). The global graph node is never stamped;
// only META-produced rank-local clones carry the map. No-op returning false
// when the headmap is disabled or rank is out of range.
bool llama_tp5_qsa_headmap_stamp(const llama_tp5_plan & plan, uint32_t rank, struct ggml_tensor * fa_node);

// Global Q head h -> global KV head (uniform global GQA ratio).
int64_t llama_tp5_qsa_global_kv_head(const llama_tp5_plan & plan, int64_t global_q_head);

// Balanced head-map helpers (opt-in GGML_TP5_GDN_HEADMAP=1):
//
// Rank-local QK head index used by rank-local V slot v_idx on the given rank
// (position of the original QK head (gdn_v_global[rank][v_idx] % Nk) inside
// gdn_qk_unique[rank]). Returns -1 when the head map is disabled or v_idx is
// out of range.
int32_t llama_tp5_gdn_local_qk(const llama_tp5_plan & plan, uint32_t rank, int32_t v_idx);

// Pack the per-rank local V->QK map into the op_params head-map ABI
// (ggml_tp5_headmap_set): entries llama_tp5_gdn_local_qk(plan, rank, i) for
// i in [0, gdn_v_global_count[rank]). No-op returning false when the head map
// is disabled or the map does not fit the 3-bit packing (count > 10 or an
// index >= 8).
bool llama_tp5_gdn_headmap_stamp(const llama_tp5_plan & plan, uint32_t rank, struct ggml_tensor * gdn_node);

// Map a named tensor to meta split_state via the immutable plan.
// Sharded native GDN weights/caches retain the caller's repeated-segment layout;
// the loader does not implement this plan's hypothetical Q/K prearrangement.
// Replicated attention instead mirrors its native weights and complete caches.
// Returns false for delegated tensors or when the plan has no applicable rule.
// indexer_head_size: pass hparams.indexer_head_size for cache_k/v indexer detection.
bool llama_tp5_try_apply_split_state(
        const llama_tp5_plan & plan,
        const char * tensor_name,
        const struct ggml_tensor * tensor,
        int64_t indexer_head_size,
        struct ggml_backend_meta_split_state & out);
