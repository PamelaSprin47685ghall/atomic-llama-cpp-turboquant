// llama-flashprefill.cpp — implementation of the public policy API and the
// internal allocation-free reference policy (routing / packing / sizing).
//
// No llama.h / ggml.h dependency. No allocation, no globals, no thread-local
// state. OFF paths short-circuit before any arithmetic.

#include "llama-flashprefill.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

// --- checked u64 arithmetic ------------------------------------------------

bool ck_mul_u64(uint64_t a, uint64_t b, uint64_t * out) {
    if (a != 0 && b > UINT64_MAX / a) {
        return false;
    }
    *out = a * b;
    return true;
}

bool ck_add_u64(uint64_t a, uint64_t b, uint64_t * out) {
    if (b > UINT64_MAX - a) {
        return false;
    }
    *out = a + b;
    return true;
}

uint64_t align64_u64(uint64_t x) {
    return (x + 63u) & ~uint64_t(63u);
}

// --- FNV-1a 64 --------------------------------------------------------------

const uint64_t kFnvOffset = 14695981039346656037ull;
const uint64_t kFnvPrime  = 1099511628211ull;

void fnv_feed(uint64_t * h, const void * data, size_t len) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < len; ++i) {
        *h ^= uint64_t(p[i]);
        *h *= kFnvPrime;
    }
}

void fnv_feed_u32(uint64_t * h, uint32_t v) {
    uint8_t b[4];
    b[0] = uint8_t(v & 0xFFu);
    b[1] = uint8_t((v >> 8) & 0xFFu);
    b[2] = uint8_t((v >> 16) & 0xFFu);
    b[3] = uint8_t((v >> 24) & 0xFFu);
    fnv_feed(h, b, 4);
}

void fnv_feed_u64(uint64_t * h, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) {
        b[i] = uint8_t((v >> (8 * i)) & 0xFFu);
    }
    fnv_feed(h, b, 8);
}

// NULL == distinct "absent" domain (tag 0x4E); "" == empty string (tag 0x53).
void fnv_feed_id(uint64_t * h, const char * s) {
    if (s == nullptr) {
        const uint8_t tag = 0x4Eu;
        fnv_feed(h, &tag, 1);
        fnv_feed_u64(h, 0);
        return;
    }
    const uint8_t tag = 0x53u;
    fnv_feed(h, &tag, 1);
    const size_t n = strlen(s);
    fnv_feed_u64(h, uint64_t(n));
    fnv_feed(h, s, n);
}

bool role_sparse_eligible(int32_t role) {
    return role == LLAMA_FLASHPREFILL_ROLE_PREFILL ||
           role == LLAMA_FLASHPREFILL_ROLE_REROT_TEACHER_FORCED;
}

} // namespace

// --- layout contracts -------------------------------------------------------

static_assert(sizeof(llama_flashprefill_config) == 52, "v1 config layout frozen");
static_assert(sizeof(llama_flashprefill_row) == 36, "v1 row layout: 8xu32 + bool + 3xpad");

// --- public C API -----------------------------------------------------------

extern "C" {

llama_flashprefill_config llama_flashprefill_default_config(void) {
    llama_flashprefill_config cfg;
    cfg.version            = LLAMA_FLASHPREFILL_CONFIG_VERSION;
    cfg.struct_size        = uint32_t(sizeof(llama_flashprefill_config));
    cfg.mode               = LLAMA_FLASHPREFILL_MODE_OFF;
    cfg.tail_scope         = LLAMA_FLASHPREFILL_TAIL_LOGICAL_PROMPT;
    cfg.alpha              = LLAMA_FLASHPREFILL_DEFAULT_ALPHA;
    cfg.block_q            = LLAMA_FLASHPREFILL_DEFAULT_BLOCK_Q;
    cfg.block_k            = LLAMA_FLASHPREFILL_DEFAULT_BLOCK_K;
    cfg.sink_blocks        = LLAMA_FLASHPREFILL_DEFAULT_SINK_BLOCKS;
    cfg.window_blocks      = LLAMA_FLASHPREFILL_DEFAULT_WINDOW_BLOCKS;
    cfg.dense_tail_tiles   = LLAMA_FLASHPREFILL_DEFAULT_DENSE_TAIL_TILES;
    cfg.min_kv             = LLAMA_FLASHPREFILL_DEFAULT_MIN_KV;
    cfg.full_attn_layers   = LLAMA_FLASHPREFILL_DEFAULT_FULL_ATTN_LAYERS;
    cfg.mean_correction    = true;
    cfg.exact_all          = false;
    cfg.reserved[0]        = 0;
    cfg.reserved[1]        = 0;
    return cfg;
}

llama_flashprefill_error llama_flashprefill_validate_config(const llama_flashprefill_config * cfg) {
    if (cfg == nullptr) {
        return LLAMA_FLASHPREFILL_ERR_NULL;
    }
    if (cfg->version != LLAMA_FLASHPREFILL_CONFIG_VERSION) {
        return LLAMA_FLASHPREFILL_ERR_VERSION;
    }
    if (cfg->struct_size != uint32_t(sizeof(llama_flashprefill_config))) {
        return LLAMA_FLASHPREFILL_ERR_SIZE;
    }
    if (cfg->mode < LLAMA_FLASHPREFILL_MODE_OFF || cfg->mode > LLAMA_FLASHPREFILL_MODE_REQUIRED) {
        return LLAMA_FLASHPREFILL_ERR_MODE;
    }
    if (cfg->tail_scope < LLAMA_FLASHPREFILL_TAIL_CALL ||
        cfg->tail_scope > LLAMA_FLASHPREFILL_TAIL_LOGICAL_PROMPT) {
        return LLAMA_FLASHPREFILL_ERR_TAIL_SCOPE;
    }
    if (!std::isfinite(cfg->alpha) || !(cfg->alpha > 0.0f) || !(cfg->alpha <= 1.0f)) {
        return LLAMA_FLASHPREFILL_ERR_ALPHA;
    }
    if (cfg->block_q == 0 || cfg->block_q > LLAMA_FLASHPREFILL_BLOCK_Q_MAX) {
        return LLAMA_FLASHPREFILL_ERR_BLOCK_Q;
    }
    if (cfg->block_k < LLAMA_FLASHPREFILL_BLOCK_K_MIN ||
        cfg->block_k > LLAMA_FLASHPREFILL_BLOCK_K_MAX ||
        (cfg->block_k % LLAMA_FLASHPREFILL_BLOCK_K_STEP) != 0) {
        return LLAMA_FLASHPREFILL_ERR_BLOCK_K;
    }
    if (cfg->reserved[0] != 0 || cfg->reserved[1] != 0) {
        return LLAMA_FLASHPREFILL_ERR_FLAG;
    }
    if (cfg->mean_correction != false && cfg->mean_correction != true) {
        return LLAMA_FLASHPREFILL_ERR_FLAG;
    }
    if (cfg->exact_all != false && cfg->exact_all != true) {
        return LLAMA_FLASHPREFILL_ERR_FLAG;
    }
    return LLAMA_FLASHPREFILL_OK;
}

llama_flashprefill_error llama_flashprefill_validate_row(const llama_flashprefill_row * row) {
    if (row == nullptr) {
        return LLAMA_FLASHPREFILL_ERR_NULL;
    }
    if (row->version != LLAMA_FLASHPREFILL_ROW_VERSION) {
        return LLAMA_FLASHPREFILL_ERR_VERSION;
    }
    if (row->struct_size != uint32_t(sizeof(llama_flashprefill_row))) {
        return LLAMA_FLASHPREFILL_ERR_SIZE;
    }
    if (row->role < LLAMA_FLASHPREFILL_ROLE_UNKNOWN || row->role > LLAMA_FLASHPREFILL_ROLE_MULTIMODAL) {
        return LLAMA_FLASHPREFILL_ERR_ROLE;
    }
    if (row->logical_pos != LLAMA_FLASHPREFILL_POS_UNKNOWN && row->logical_pos < 0) {
        return LLAMA_FLASHPREFILL_ERR_INTERVAL;
    }
    if (row->prefill_known != false && row->prefill_known != true) {
        return LLAMA_FLASHPREFILL_ERR_FLAG;
    }
    if (row->prefill_known) {
        if (row->prefill_begin < 0 || row->prefill_end < row->prefill_begin) {
            return LLAMA_FLASHPREFILL_ERR_INTERVAL;
        }
    }
    if (row->reserved[0] != 0 || row->reserved[1] != 0 || row->reserved[2] != 0) {
        return LLAMA_FLASHPREFILL_ERR_FLAG;
    }
    return LLAMA_FLASHPREFILL_OK;
}

llama_flashprefill_error llama_flashprefill_validate_exec(const llama_flashprefill_exec * exec) {
    if (exec == nullptr) {
        return LLAMA_FLASHPREFILL_ERR_NULL;
    }
    if (exec->version != LLAMA_FLASHPREFILL_EXEC_VERSION) {
        return LLAMA_FLASHPREFILL_ERR_VERSION;
    }
    if (exec->struct_size != uint32_t(sizeof(llama_flashprefill_exec))) {
        return LLAMA_FLASHPREFILL_ERR_SIZE;
    }
    if (exec->reserved0 != 0) {
        return LLAMA_FLASHPREFILL_ERR_FLAG;
    }
    if (exec->n_rows > LLAMA_FLASHPREFILL_EXEC_ROWS_MAX) {
        return LLAMA_FLASHPREFILL_ERR_COUNT;
    }
    if (exec->n_rows == 0) {
        return LLAMA_FLASHPREFILL_OK; // rows may be NULL (empty view)
    }
    if (exec->rows == nullptr) {
        return LLAMA_FLASHPREFILL_ERR_NULL;
    }
    for (uint32_t i = 0; i < exec->n_rows; ++i) {
        const llama_flashprefill_error rc = llama_flashprefill_validate_row(&exec->rows[i]);
        if (rc != LLAMA_FLASHPREFILL_OK) {
            return rc;
        }
    }
    return LLAMA_FLASHPREFILL_OK;
}

bool llama_flashprefill_is_enabled(const llama_flashprefill_config * cfg) {
    if (cfg == nullptr) {
        return false;
    }
    if (llama_flashprefill_validate_config(cfg) != LLAMA_FLASHPREFILL_OK) {
        return false;
    }
    return cfg->mode != LLAMA_FLASHPREFILL_MODE_OFF;
}

llama_flashprefill_error llama_flashprefill_fingerprint(
    const llama_flashprefill_config * cfg,
    const char * model_id,
    const char * adapter_id,
    uint64_t * out_fingerprint) {
    if (cfg == nullptr || out_fingerprint == nullptr) {
        return LLAMA_FLASHPREFILL_ERR_NULL;
    }
    const llama_flashprefill_error rc = llama_flashprefill_validate_config(cfg);
    if (rc != LLAMA_FLASHPREFILL_OK) {
        return rc;
    }
    uint32_t alpha_bits = 0;
    static_assert(sizeof(alpha_bits) == sizeof(cfg->alpha), "float width");
    memcpy(&alpha_bits, &cfg->alpha, sizeof(alpha_bits));

    uint64_t h = kFnvOffset;
    fnv_feed_u32(&h, LLAMA_FLASHPREFILL_VERSION);
    fnv_feed_u32(&h, uint32_t(cfg->mode));
    fnv_feed_u32(&h, uint32_t(cfg->tail_scope));
    fnv_feed_u32(&h, alpha_bits);
    fnv_feed_u32(&h, cfg->block_q);
    fnv_feed_u32(&h, cfg->block_k);
    fnv_feed_u32(&h, cfg->sink_blocks);
    fnv_feed_u32(&h, cfg->window_blocks);
    fnv_feed_u32(&h, cfg->dense_tail_tiles);
    fnv_feed_u32(&h, cfg->min_kv);
    fnv_feed_u32(&h, cfg->full_attn_layers);
    {
        const uint8_t flags[2] = {
            uint8_t(cfg->mean_correction ? 1 : 0),
            uint8_t(cfg->exact_all ? 1 : 0),
        };
        fnv_feed(&h, flags, 2);
    }
    fnv_feed_id(&h, model_id);
    fnv_feed_id(&h, adapter_id);

    *out_fingerprint = h;
    return LLAMA_FLASHPREFILL_OK;
}

const char * llama_flashprefill_error_name(int32_t error) {
    switch (error) {
        case LLAMA_FLASHPREFILL_OK:              return "ok";
        case LLAMA_FLASHPREFILL_ERR_NULL:        return "null";
        case LLAMA_FLASHPREFILL_ERR_VERSION:     return "bad_version";
        case LLAMA_FLASHPREFILL_ERR_SIZE:        return "bad_size";
        case LLAMA_FLASHPREFILL_ERR_MODE:        return "bad_mode";
        case LLAMA_FLASHPREFILL_ERR_TAIL_SCOPE:  return "bad_tail_scope";
        case LLAMA_FLASHPREFILL_ERR_ALPHA:       return "bad_alpha";
        case LLAMA_FLASHPREFILL_ERR_BLOCK_Q:     return "bad_block_q";
        case LLAMA_FLASHPREFILL_ERR_BLOCK_K:     return "bad_block_k";
        case LLAMA_FLASHPREFILL_ERR_ROLE:        return "bad_role";
        case LLAMA_FLASHPREFILL_ERR_INTERVAL:    return "bad_interval";
        case LLAMA_FLASHPREFILL_ERR_COUNT:       return "bad_count";
        case LLAMA_FLASHPREFILL_ERR_FLAG:        return "bad_flag";
        case LLAMA_FLASHPREFILL_ERR_OVERFLOW:    return "overflow";
        default:                                 return "?";
    }
}

const char * llama_flashprefill_mode_name(int32_t mode) {
    switch (mode) {
        case LLAMA_FLASHPREFILL_MODE_OFF:      return "off";
        case LLAMA_FLASHPREFILL_MODE_AUTO:     return "auto";
        case LLAMA_FLASHPREFILL_MODE_REQUIRED: return "required";
        default:                               return "?";
    }
}

const char * llama_flashprefill_role_name(int32_t role) {
    switch (role) {
        case LLAMA_FLASHPREFILL_ROLE_UNKNOWN:             return "unknown";
        case LLAMA_FLASHPREFILL_ROLE_PREFILL:             return "prefill";
        case LLAMA_FLASHPREFILL_ROLE_REROT_TEACHER_FORCED: return "rerot_teacher_forced";
        case LLAMA_FLASHPREFILL_ROLE_DECODE:              return "decode";
        case LLAMA_FLASHPREFILL_ROLE_MTP_DRAFT:           return "mtp_draft";
        case LLAMA_FLASHPREFILL_ROLE_MTP_VERIFY:          return "mtp_verify";
        case LLAMA_FLASHPREFILL_ROLE_SPECULATIVE_REPLAY:  return "speculative_replay";
        case LLAMA_FLASHPREFILL_ROLE_REROT_FRONTIER:      return "rerot_frontier";
        case LLAMA_FLASHPREFILL_ROLE_EMBEDDING:           return "embedding";
        case LLAMA_FLASHPREFILL_ROLE_RERANK:              return "rerank";
        case LLAMA_FLASHPREFILL_ROLE_MULTIMODAL:          return "multimodal";
        default:                                          return "?";
    }
}

const char * llama_flashprefill_route_name(int32_t route) {
    switch (route) {
        case LLAMA_FLASHPREFILL_ROUTE_SPARSE:                 return "sparse";
        case LLAMA_FLASHPREFILL_ROUTE_EXACT_ALL:              return "exact_all";
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF:              return "off";
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_ROLE:             return "role";
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED:      return "unsupported";
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_SHORT_CONTEXT:    return "short_context";
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_TAIL:             return "dense_tail";
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_UNKNOWN_BOUNDARY: return "unknown_boundary";
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_CAPACITY:         return "capacity";
        default:                                              return "?";
    }
}

bool llama_flashprefill_route_is_dense(int32_t route) {
    switch (route) {
        case LLAMA_FLASHPREFILL_ROUTE_SPARSE:
        case LLAMA_FLASHPREFILL_ROUTE_EXACT_ALL:
            return false;
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF:
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_ROLE:
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED:
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_SHORT_CONTEXT:
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_TAIL:
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_UNKNOWN_BOUNDARY:
        case LLAMA_FLASHPREFILL_ROUTE_DENSE_CAPACITY:
            return true;
        default:
            return false; // unknown route: never trusted as either path
    }
}

} // extern "C"

// --- internal C++ reference policy -------------------------------------------

namespace llama_flashprefill {

llama_flashprefill_error packed_layout_checked(
    uint32_t n_tokens,
    uint32_t gqa,
    uint32_t block_q,
    uint32_t * out_total_packed,
    uint32_t * out_tiles) {
    if (out_total_packed == nullptr || out_tiles == nullptr) {
        return LLAMA_FLASHPREFILL_ERR_NULL;
    }
    if (gqa == 0 || block_q == 0) {
        return LLAMA_FLASHPREFILL_ERR_COUNT;
    }
    const uint64_t total = uint64_t(n_tokens) * uint64_t(gqa);
    if (total > uint64_t(UINT32_MAX)) {
        return LLAMA_FLASHPREFILL_ERR_OVERFLOW;
    }
    const uint64_t tiles = (total + uint64_t(block_q) - 1u) / uint64_t(block_q);
    if (tiles > uint64_t(UINT32_MAX)) {
        return LLAMA_FLASHPREFILL_ERR_OVERFLOW;
    }
    *out_total_packed = uint32_t(total);
    *out_tiles        = uint32_t(tiles);
    return LLAMA_FLASHPREFILL_OK;
}

bool packed_row_in_dense_tail(
    uint32_t packed_row,
    uint32_t total_packed,
    uint32_t block_q,
    uint32_t tail_tiles) {
    if (tail_tiles == 0 || total_packed == 0 || block_q == 0) {
        return false;
    }
    if (packed_row >= total_packed) {
        return false;
    }
    const uint64_t tiles = (uint64_t(total_packed) + uint64_t(block_q) - 1u) / uint64_t(block_q);
    const uint64_t keep  = tail_tiles < tiles ? uint64_t(tail_tiles) : tiles;
    const uint64_t cutoff = tiles - keep;
    return (uint64_t(packed_row) / uint64_t(block_q)) >= cutoff;
}

llama_flashprefill_route route_for_role(
    const llama_flashprefill_config * cfg,
    int32_t role,
    uint32_t resident_visible_tokens,
    bool backend_supported) {
    if (cfg == nullptr || llama_flashprefill_validate_config(cfg) != LLAMA_FLASHPREFILL_OK) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF;
    }
    if (cfg->mode == LLAMA_FLASHPREFILL_MODE_OFF) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF;
    }
    if (role < LLAMA_FLASHPREFILL_ROLE_UNKNOWN ||
        role > LLAMA_FLASHPREFILL_ROLE_MULTIMODAL ||
        !role_sparse_eligible(role)) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_ROLE;
    }
    if (cfg->exact_all) {
        // Debug bypasses the length gate but never a missing backend.
        return backend_supported
            ? LLAMA_FLASHPREFILL_ROUTE_EXACT_ALL
            : LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED;
    }
    if (resident_visible_tokens < cfg->min_kv) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_SHORT_CONTEXT;
    }
    if (!backend_supported) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED;
    }
    return LLAMA_FLASHPREFILL_ROUTE_SPARSE;
}

llama_flashprefill_route route_row_call(
    const llama_flashprefill_config * cfg,
    const llama_flashprefill_row * row,
    uint32_t resident_visible_tokens,
    bool backend_supported,
    uint32_t packed_row,
    uint32_t total_packed) {
    if (cfg == nullptr || llama_flashprefill_validate_config(cfg) != LLAMA_FLASHPREFILL_OK) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF;
    }
    if (cfg->mode == LLAMA_FLASHPREFILL_MODE_OFF) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF;
    }
    if (row == nullptr || llama_flashprefill_validate_row(row) != LLAMA_FLASHPREFILL_OK) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_UNKNOWN_BOUNDARY;
    }
    if (!role_sparse_eligible(row->role)) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_ROLE;
    }
    if (cfg->exact_all) {
        return backend_supported
            ? LLAMA_FLASHPREFILL_ROUTE_EXACT_ALL
            : LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED;
    }
    if (resident_visible_tokens < cfg->min_kv) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_SHORT_CONTEXT;
    }
    if (packed_row_in_dense_tail(packed_row, total_packed, cfg->block_q, cfg->dense_tail_tiles)) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_TAIL;
    }
    if (!backend_supported) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED;
    }
    return LLAMA_FLASHPREFILL_ROUTE_SPARSE;
}

llama_flashprefill_route route_row_logical(
    const llama_flashprefill_config * cfg,
    const llama_flashprefill_row * row,
    uint32_t resident_visible_tokens,
    bool backend_supported,
    uint32_t packed_row_in_prefill,
    uint32_t total_prefill_packed) {
    if (cfg == nullptr || llama_flashprefill_validate_config(cfg) != LLAMA_FLASHPREFILL_OK) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF;
    }
    if (cfg->mode == LLAMA_FLASHPREFILL_MODE_OFF) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF;
    }
    if (row == nullptr || llama_flashprefill_validate_row(row) != LLAMA_FLASHPREFILL_OK) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_UNKNOWN_BOUNDARY;
    }
    if (!role_sparse_eligible(row->role)) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_ROLE;
    }
    if (!row->prefill_known) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_UNKNOWN_BOUNDARY;
    }
    if (cfg->exact_all) {
        return backend_supported
            ? LLAMA_FLASHPREFILL_ROUTE_EXACT_ALL
            : LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED;
    }
    if (resident_visible_tokens < cfg->min_kv) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_SHORT_CONTEXT;
    }
    if (packed_row_in_dense_tail(
            packed_row_in_prefill, total_prefill_packed, cfg->block_q, cfg->dense_tail_tiles)) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_TAIL;
    }
    if (!backend_supported) {
        return LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED;
    }
    return LLAMA_FLASHPREFILL_ROUTE_SPARSE;
}

llama_flashprefill_error scratch_bytes_checked(
    const llama_flashprefill_config * cfg,
    const scratch_inputs * in,
    uint64_t * out_bytes) {
    if (cfg == nullptr || in == nullptr || out_bytes == nullptr) {
        return LLAMA_FLASHPREFILL_ERR_NULL;
    }
    const llama_flashprefill_error vc = llama_flashprefill_validate_config(cfg);
    if (vc != LLAMA_FLASHPREFILL_OK) {
        return vc;
    }
    if (cfg->mode == LLAMA_FLASHPREFILL_MODE_OFF) {
        *out_bytes = 0; // OFF: no allocation, no further work
        return LLAMA_FLASHPREFILL_OK;
    }
    // Generic capacity guards (catch garbage before arithmetic).
    if (in->n_fragments > (1u << 24) || in->n_kv_heads > 256 || in->n_packed_rows > (1u << 28) ||
        in->n_splits > 1024 || in->d_k > 4096 || in->d_v > 4096) {
        return LLAMA_FLASHPREFILL_ERR_COUNT;
    }
    if (in->mean_bytes != 2 && in->mean_bytes != 4) {
        return LLAMA_FLASHPREFILL_ERR_FLAG;
    }

    const uint64_t F = in->n_fragments;
    const uint64_t H = in->n_kv_heads;
    const uint64_t Dk = in->d_k;
    const uint64_t Dv = in->d_v;
    const uint64_t P = in->n_packed_rows;
    const uint64_t S = in->n_splits;
    const uint64_t MW = in->mean_bytes;
    const uint64_t BQ = cfg->block_q;

    const uint64_t tiles = BQ == 0 ? 0 : (P + BQ - 1u) / BQ;
    const uint64_t R = tiles * H; // work segments; bounded inputs, no overflow

    uint64_t dv_sum = 0, fh = 0, pool = 0;
    if (!ck_add_u64(Dk, Dv, &dv_sum) || !ck_mul_u64(F, H, &fh) || !ck_mul_u64(fh, dv_sum, &pool) ||
        !ck_mul_u64(pool, MW, &pool)) {
        return LLAMA_FLASHPREFILL_ERR_OVERFLOW;
    }

    uint64_t worst_entries = 0, index_body = 0, index = 0, t = 0;
    if (!ck_mul_u64(tiles, F, &worst_entries) || !ck_mul_u64(worst_entries, 4u, &index_body)) {
        return LLAMA_FLASHPREFILL_ERR_OVERFLOW;
    }
    // offsets (F+1)*4 + per-fragment counts F*4 + worst exact/correction entries + error/count words.
    if (!ck_add_u64((F + 1u) * 4u, F * 4u, &t) || !ck_add_u64(t, index_body, &t) ||
        !ck_add_u64(t, 64u, &index)) {
        return LLAMA_FLASHPREFILL_ERR_OVERFLOW;
    }

    uint64_t desc = 0;
    if (!ck_mul_u64(F, 32u, &desc)) { // 8 x i32 descriptor stride (documented, not VRAM)
        return LLAMA_FLASHPREFILL_ERR_OVERFLOW;
    }

    uint64_t split = 0;
    if (S > 1 && R > 0 && Dv > 0) {
        uint64_t per = 0;
        if (!ck_add_u64(Dv, 2u, &per) || !ck_mul_u64(S, R, &split) || !ck_mul_u64(split, per, &split) ||
            !ck_mul_u64(split, 4u, &split)) {
            return LLAMA_FLASHPREFILL_ERR_OVERFLOW;
        }
    }

    uint64_t total = 0;
    if (!ck_add_u64(align64_u64(pool), align64_u64(index), &total) ||
        !ck_add_u64(total, align64_u64(desc), &total) ||
        !ck_add_u64(total, align64_u64(split), &total)) {
        return LLAMA_FLASHPREFILL_ERR_OVERFLOW;
    }
    *out_bytes = total;
    return LLAMA_FLASHPREFILL_OK;
}

} // namespace llama_flashprefill
