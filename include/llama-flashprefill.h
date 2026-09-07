// llama-flashprefill.h — public FlashPrefill V2 policy/role contract (v1).
//
// Public llama-side policy configuration plus per-row execution roles.
// The GGML wire schema + math live in ggml/include/ggml-flashprefill.h and
// MUST NOT be included here (and vice versa). Scalar policy fields are shared
// by value semantics only; no wire conversion is declared here.
//
// Contract facts:
//   - C-compatible (C99 + C++), no dependency on llama.h (forward declarations
//     and integer ids only) to avoid circularity.
//   - Every config/exec type is version + struct_size tagged (v1).
//   - Defaults are always OFF; OFF path helpers never allocate or do work.
//   - No decode-extension declaration yet.
//   - Every declared function is implemented in src/llama-flashprefill.cpp.
//
// Lifecycle / ownership semantics:
//   - llama_flashprefill_config and llama_flashprefill_row are plain values.
//     Caller owns them; they are memcpy-able; reserved bytes must be zero.
//   - llama_flashprefill_exec borrows: `rows` points to a caller-owned
//     contiguous array of `n_rows` entries that must stay alive and unchanged
//     for the duration of the call that consumes the exec view. The callee
//     never stores the pointer beyond return, never frees it, never writes
//     through it (const). `n_rows == 0` allows `rows == NULL` (empty view).
//   - All helpers are pure: no allocation, no globals, no thread-local state,
//     safe to call concurrently. Invalid input yields a precise error or a
//     conservative dense route, never sparse on doubt.
//   - Wire packing for GPU exec must fit GGML_MAX_SRC == 10 (integer
//     descriptors packed into an I32 metadata tensor + views); this header
//     does not define that packing, it only keeps descriptors translatable.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Standalone export macro mirroring the llama.h convention. Self-contained:
// ConfigIntegration includes this header before llama.h, so LLAMA_API is not
// relied upon. Honors the existing LLAMA_SHARED / LLAMA_BUILD compile
// definitions as well as the independent LLAMA_FLASHPREFILL_* ones.
#if defined(LLAMA_FLASHPREFILL_SHARED) || defined(LLAMA_SHARED)
#    if defined(_WIN32) && !defined(__MINGW32__)
#        if defined(LLAMA_FLASHPREFILL_BUILD) || defined(LLAMA_BUILD)
#            define LLAMA_FLASHPREFILL_API __declspec(dllexport)
#        else
#            define LLAMA_FLASHPREFILL_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_FLASHPREFILL_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAMA_FLASHPREFILL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations only. Never include llama.h from this header.
struct llama_model;
struct llama_context;

// ---------------------------------------------------------------------------
// Frozen v1 identity
// ---------------------------------------------------------------------------

#define LLAMA_FLASHPREFILL_VERSION 1
#define LLAMA_FLASHPREFILL_CONFIG_VERSION 1
#define LLAMA_FLASHPREFILL_ROW_VERSION 1
#define LLAMA_FLASHPREFILL_EXEC_VERSION 1

// Frozen v1 defaults (see PREFILL.md section 12; OFF is mandatory default).
#define LLAMA_FLASHPREFILL_DEFAULT_ALPHA 0.1f
#define LLAMA_FLASHPREFILL_DEFAULT_BLOCK_Q 128u
#define LLAMA_FLASHPREFILL_DEFAULT_BLOCK_K 128u
#define LLAMA_FLASHPREFILL_DEFAULT_SINK_BLOCKS 2u
#define LLAMA_FLASHPREFILL_DEFAULT_WINDOW_BLOCKS 4u
#define LLAMA_FLASHPREFILL_DEFAULT_DENSE_TAIL_TILES 8u
#define LLAMA_FLASHPREFILL_DEFAULT_MIN_KV 1024u
#define LLAMA_FLASHPREFILL_DEFAULT_FULL_ATTN_LAYERS 0u

// Validation bounds (frozen v1).
#define LLAMA_FLASHPREFILL_BLOCK_Q_MAX 256u
#define LLAMA_FLASHPREFILL_BLOCK_K_MIN 64u
#define LLAMA_FLASHPREFILL_BLOCK_K_MAX 1024u
#define LLAMA_FLASHPREFILL_BLOCK_K_STEP 64u
#define LLAMA_FLASHPREFILL_EXEC_ROWS_MAX ((uint32_t) (1u << 24))

// Opaque identity sentinels (integer ids; no llama.h dependency).
#define LLAMA_FLASHPREFILL_SEQ_UNKNOWN ((int32_t) -1)
#define LLAMA_FLASHPREFILL_READER_NONE ((uint32_t) 0xFFFFFFFFu)
#define LLAMA_FLASHPREFILL_POS_UNKNOWN ((int32_t) -1)

// ---------------------------------------------------------------------------
// Enums (fixed values are part of the frozen contract)
// ---------------------------------------------------------------------------

typedef enum llama_flashprefill_mode {
    LLAMA_FLASHPREFILL_MODE_OFF      = 0,
    LLAMA_FLASHPREFILL_MODE_AUTO     = 1,
    LLAMA_FLASHPREFILL_MODE_REQUIRED = 2,
} llama_flashprefill_mode;

typedef enum llama_flashprefill_tail_scope {
    LLAMA_FLASHPREFILL_TAIL_CALL           = 0, // dense tail vs. this call's query range (reference alignment)
    LLAMA_FLASHPREFILL_TAIL_LOGICAL_PROMPT = 1, // dense tail vs. frozen logical prefill range (production)
} llama_flashprefill_tail_scope;

// Execution role of one source row. Only PREFILL and REROT_TEACHER_FORCED are
// ever sparse-eligible; every other role is a conservative dense route.
typedef enum llama_flashprefill_role {
    LLAMA_FLASHPREFILL_ROLE_UNKNOWN            = 0,
    LLAMA_FLASHPREFILL_ROLE_PREFILL            = 1,
    LLAMA_FLASHPREFILL_ROLE_REROT_TEACHER_FORCED = 2,
    LLAMA_FLASHPREFILL_ROLE_DECODE             = 3,
    LLAMA_FLASHPREFILL_ROLE_MTP_DRAFT          = 4,
    LLAMA_FLASHPREFILL_ROLE_MTP_VERIFY         = 5,
    LLAMA_FLASHPREFILL_ROLE_SPECULATIVE_REPLAY = 6,
    LLAMA_FLASHPREFILL_ROLE_REROT_FRONTIER     = 7,
    LLAMA_FLASHPREFILL_ROLE_EMBEDDING          = 8,
    LLAMA_FLASHPREFILL_ROLE_RERANK             = 9,
    LLAMA_FLASHPREFILL_ROLE_MULTIMODAL         = 10,
} llama_flashprefill_role;

// Precise validation / sizing errors. Order of checks is documented on each
// validate function; the first failure wins (deterministic).
typedef enum llama_flashprefill_error {
    LLAMA_FLASHPREFILL_OK              = 0,
    LLAMA_FLASHPREFILL_ERR_NULL        = 1,
    LLAMA_FLASHPREFILL_ERR_VERSION     = 2,
    LLAMA_FLASHPREFILL_ERR_SIZE        = 3,
    LLAMA_FLASHPREFILL_ERR_MODE        = 4,
    LLAMA_FLASHPREFILL_ERR_TAIL_SCOPE  = 5,
    LLAMA_FLASHPREFILL_ERR_ALPHA       = 6,
    LLAMA_FLASHPREFILL_ERR_BLOCK_Q     = 7,
    LLAMA_FLASHPREFILL_ERR_BLOCK_K     = 8,
    LLAMA_FLASHPREFILL_ERR_ROLE        = 9,
    LLAMA_FLASHPREFILL_ERR_INTERVAL    = 10,
    LLAMA_FLASHPREFILL_ERR_COUNT       = 11,
    LLAMA_FLASHPREFILL_ERR_FLAG        = 12,
    LLAMA_FLASHPREFILL_ERR_OVERFLOW    = 13,
} llama_flashprefill_error;

// Routing outcome per row. SPARSE / EXACT_ALL take the new prefill path;
// every DENSE_* value keeps the pre-existing attention path and doubles as
// the metrics/fallback reason. Stable values; do not renumber.
typedef enum llama_flashprefill_route {
    LLAMA_FLASHPREFILL_ROUTE_SPARSE                = 0,
    LLAMA_FLASHPREFILL_ROUTE_EXACT_ALL             = 1,
    LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF             = 2,
    LLAMA_FLASHPREFILL_ROUTE_DENSE_ROLE            = 3,
    LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED     = 4,
    LLAMA_FLASHPREFILL_ROUTE_DENSE_SHORT_CONTEXT   = 5,
    LLAMA_FLASHPREFILL_ROUTE_DENSE_TAIL            = 6,
    LLAMA_FLASHPREFILL_ROUTE_DENSE_UNKNOWN_BOUNDARY = 7,
    LLAMA_FLASHPREFILL_ROUTE_DENSE_CAPACITY        = 8,
} llama_flashprefill_route;

// ---------------------------------------------------------------------------
// Frozen v1 structs (exact layout is contractual; do not reorder/pack)
// ---------------------------------------------------------------------------

// Policy configuration. Scalar fields are semantically shared with the GGML
// wire schema (same names, units, ranges) without either header including
// the other. `mean_correction=true` is the only production mode; `false` is
// an ablation. `exact_all=true` is a debug gate through the new path.
typedef struct llama_flashprefill_config {
    uint32_t version;           // == LLAMA_FLASHPREFILL_CONFIG_VERSION
    uint32_t struct_size;       // == sizeof(llama_flashprefill_config)
    int32_t  mode;              // llama_flashprefill_mode
    int32_t  tail_scope;        // llama_flashprefill_tail_scope
    float    alpha;             // (0, 1] finite; energy threshold factor, not a sparsity ratio
    uint32_t block_q;           // BM: packed Q rows, 1..256 (any value; non-pow2 tests nondividing GQA)
    uint32_t block_k;           // BN: logical K block, multiples of 64 up to 1024
    uint32_t sink_blocks;       // mandatory exact prefix blocks (any uint32)
    uint32_t window_blocks;     // mandatory exact local/partial-visibility blocks (any uint32)
    uint32_t dense_tail_tiles;  // trailing packed-Q tiles kept dense (any uint32)
    uint32_t min_kv;            // eligibility floor on resident-visible tokens (any uint32)
    uint32_t full_attn_layers;  // first N *eligible full-attention* layers kept dense (any uint32)
    bool     mean_correction;   // production must be true
    bool     exact_all;         // debug: new path, all legal fragments exact
    uint8_t  reserved[2];       // must be zero
} llama_flashprefill_config;

// Per-source-row descriptor: who this query row belongs to, where it sits in
// its logical prefill, and which frozen interval scopes its dense tail.
typedef struct llama_flashprefill_row {
    uint32_t version;        // == LLAMA_FLASHPREFILL_ROW_VERSION
    uint32_t struct_size;    // == sizeof(llama_flashprefill_row)
    int32_t  role;           // llama_flashprefill_role
    int32_t  seq_id;         // logical sequence id, or LLAMA_FLASHPREFILL_SEQ_UNKNOWN
    uint32_t reader_id;      // RERoT reader identity, or LLAMA_FLASHPREFILL_READER_NONE
    int32_t  logical_pos;    // logical query coordinate, or LLAMA_FLASHPREFILL_POS_UNKNOWN
    int32_t  prefill_begin;  // frozen logical prefill interval begin (inclusive); valid iff prefill_known
    int32_t  prefill_end;    // frozen logical prefill interval end (exclusive); valid iff prefill_known
    bool     prefill_known;  // explicit known flag: false forbids guessing the logical range
    uint8_t  reserved[3];    // must be zero
} llama_flashprefill_row;

// Borrowed batch view for one planning call. See lifecycle rules above.
typedef struct llama_flashprefill_exec {
    uint32_t version;       // == LLAMA_FLASHPREFILL_EXEC_VERSION
    uint32_t struct_size;   // == sizeof(llama_flashprefill_exec)
    uint32_t n_rows;        // <= LLAMA_FLASHPREFILL_EXEC_ROWS_MAX
    uint32_t reserved0;     // must be zero
    const llama_flashprefill_row * rows; // NULL iff n_rows == 0; else non-NULL
} llama_flashprefill_exec;

// ---------------------------------------------------------------------------
// Implemented API (every declaration below has a definition; no stubs)
// ---------------------------------------------------------------------------

// Default policy: always OFF, all other fields at frozen v1 defaults.
LLAMA_FLASHPREFILL_API llama_flashprefill_config llama_flashprefill_default_config(void);

// Precise validators (pure, no allocation). Check order: NULL -> version ->
// size -> mode -> tail_scope -> alpha -> block_q -> block_k ->
// reserved/flags. First failure wins.
LLAMA_FLASHPREFILL_API llama_flashprefill_error llama_flashprefill_validate_config(const llama_flashprefill_config * cfg);
LLAMA_FLASHPREFILL_API llama_flashprefill_error llama_flashprefill_validate_row(const llama_flashprefill_row * row);
LLAMA_FLASHPREFILL_API llama_flashprefill_error llama_flashprefill_validate_exec(const llama_flashprefill_exec * exec);

// True iff cfg is non-NULL, validates OK, and mode != OFF. No work on OFF.
LLAMA_FLASHPREFILL_API bool llama_flashprefill_is_enabled(const llama_flashprefill_config * cfg);

// Stable deterministic policy fingerprint (FNV-1a 64 over a fixed field order
// plus explicit model/adapter identity bytes). Covers every approximation
// policy field: version, mode, tail_scope, alpha bits, block_q/block_k,
// sink/window/tail, min_kv, full_attn_layers, mean_correction, exact_all,
// then model_id and adapter_id (NULL == distinct empty domain, "" == empty
// string domain). Pure, no allocation, no host KV readback. Writes *out_fingerprint.
// Returns OK, ERR_NULL (cfg/out NULL), or the first config validation error.
LLAMA_FLASHPREFILL_API llama_flashprefill_error llama_flashprefill_fingerprint(
    const llama_flashprefill_config * cfg,
    const char * model_id,
    const char * adapter_id,
    uint64_t * out_fingerprint);

// Static display names (never NULL; "?" for out-of-range). No allocation.
LLAMA_FLASHPREFILL_API const char * llama_flashprefill_error_name(int32_t error);
LLAMA_FLASHPREFILL_API const char * llama_flashprefill_mode_name(int32_t mode);
LLAMA_FLASHPREFILL_API const char * llama_flashprefill_role_name(int32_t role);
LLAMA_FLASHPREFILL_API const char * llama_flashprefill_route_name(int32_t route);

// True for any DENSE_* route (including DENSE_OFF). False for SPARSE/EXACT_ALL
// and for out-of-range values (conservative: unknown is not trusted dense).
LLAMA_FLASHPREFILL_API bool llama_flashprefill_route_is_dense(int32_t route);

#ifdef __cplusplus
}
#endif
