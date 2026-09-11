// FlashPrefill V2 GGML wire schema + reference math — implementation.
// See ggml/include/ggml-flashprefill.h for the frozen v1 contract.

#include "ggml-flashprefill.h"

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace {

const int64_t FP_I32_MAX = 2147483647LL;

int32_t fp_add_ok(int64_t a, int64_t b, int64_t * out) {
    if (a < 0 || b < 0 || a > FP_I32_MAX - b) {
        return GGML_FLASHPREFILL_ERR_OVERFLOW;
    }
    *out = a + b;
    return GGML_FLASHPREFILL_OK;
}

int32_t fp_mul_ok(int64_t a, int64_t b, int64_t * out) {
    if (a < 0 || b < 0) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (a != 0 && b > FP_I32_MAX / a) {
        return GGML_FLASHPREFILL_ERR_OVERFLOW;
    }
    *out = a * b;
    return GGML_FLASHPREFILL_OK;
}

bool fp_finite_d(double x) {
    return x > -DBL_MAX && x < DBL_MAX;
}

bool fp_finite_f(float x) {
    return x > -FLT_MAX && x < FLT_MAX;
}

// Parsed metadata header view (host order, range-checked on parse).
struct FpMetaView {
    const int32_t * base;
    int64_t n_words;
    int64_t n_frag, f_cap, n_row, r_cap, n_use, u_cap, n_cell, c_cap;
    int64_t frag_off, row_off, use_off, cell_off, total;
    int32_t dk, dv, hkv, ngroups, nqheads;
};

int32_t fp_parse_meta(const int32_t * meta, int64_t n_words, FpMetaView * v) {
    if (!meta || !v || n_words < GGML_FLASHPREFILL_HEADER_WORDS) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (meta[0] != GGML_FLASHPREFILL_METADATA_MAGIC) {
        return GGML_FLASHPREFILL_ERR_BAD_MAGIC;
    }
    if (meta[1] != GGML_FLASHPREFILL_VERSION) {
        return GGML_FLASHPREFILL_ERR_BAD_VERSION;
    }
    if (meta[2] != GGML_FLASHPREFILL_HEADER_WORDS ||
        meta[15] != GGML_FLASHPREFILL_FRAG_WORDS ||
        meta[16] != GGML_FLASHPREFILL_ROW_WORDS ||
        meta[17] != GGML_FLASHPREFILL_USE_WORDS) {
        return GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
    }
    const int64_t vals[8] = {meta[3], meta[4], meta[5], meta[6], meta[7], meta[8], meta[9], meta[10]};
    for (int i = 0; i < 8; i++) {
        if (vals[i] < 0 || vals[i] > FP_I32_MAX) {
            return GGML_FLASHPREFILL_ERR_BAD_RANGE;
        }
    }
    v->base    = meta;
    v->n_words = n_words;
    v->n_frag = vals[0]; v->f_cap = vals[1];
    v->n_row  = vals[2]; v->r_cap = vals[3];
    v->n_use  = vals[4]; v->u_cap = vals[5];
    v->n_cell = vals[6]; v->c_cap = vals[7];
    if (v->n_frag > v->f_cap || v->n_row > v->r_cap ||
        v->n_use > v->u_cap || v->n_cell > v->c_cap) {
        return GGML_FLASHPREFILL_ERR_CAP_EXCEEDED;
    }
    // Recompute canonical offsets; stored ones are cross-checked.
    int64_t row_off = 0, use_off = 0, cell_off = 0, total = 0, t = 0;
    int32_t rc = GGML_FLASHPREFILL_OK;
    rc = fp_mul_ok(v->f_cap, GGML_FLASHPREFILL_FRAG_WORDS, &t); if (rc) return rc;
    rc = fp_add_ok(GGML_FLASHPREFILL_HEADER_WORDS, t, &row_off); if (rc) return rc;
    rc = fp_mul_ok(v->r_cap, GGML_FLASHPREFILL_ROW_WORDS, &t); if (rc) return rc;
    rc = fp_add_ok(row_off, t, &use_off); if (rc) return rc;
    rc = fp_mul_ok(v->u_cap, GGML_FLASHPREFILL_USE_WORDS, &t); if (rc) return rc;
    rc = fp_add_ok(use_off, t, &cell_off); if (rc) return rc;
    rc = fp_add_ok(cell_off, v->c_cap, &total); if (rc) return rc;
    if (meta[11] != (int32_t)GGML_FLASHPREFILL_HEADER_WORDS ||
        meta[12] != (int32_t)row_off ||
        meta[13] != (int32_t)use_off ||
        meta[14] != (int32_t)cell_off ||
        meta[21] != (int32_t)total) {
        return GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
    }
    if (n_words != total) {
        return GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
    }
    v->frag_off = GGML_FLASHPREFILL_HEADER_WORDS;
    v->row_off  = row_off;
    v->use_off  = use_off;
    v->cell_off = cell_off;
    v->total    = total;
    v->dk  = meta[18];
    v->dv  = meta[19];
    v->hkv = meta[20];
    v->ngroups = meta[22];
    v->nqheads = meta[23];
    if (v->dk < 1 || v->dk > (int32_t)GGML_FLASHPREFILL_HEAD_DIM_MAX ||
        v->dv < 1 || v->dv > (int32_t)GGML_FLASHPREFILL_HEAD_DIM_MAX ||
        v->hkv < 1 || v->ngroups < 1 || v->nqheads < 1) {
        return GGML_FLASHPREFILL_ERR_BAD_RANGE;
    }
    for (int i = 24; i < 32; i++) {
        if (meta[i] != 0) {
            return GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
        }
    }
    return GGML_FLASHPREFILL_OK;
}

struct FpRowKey {
    int32_t sq, tile, kvh, qh;
    int32_t dense;
    int64_t idx;
};

struct FpUseKey {
    int32_t sq, tile, kvh, frag;
    int32_t group;
    int64_t sub_off, sub_count;
    int32_t mandatory;
    int64_t idx;
};

int fp_cmp_u64(const void * a, const void * b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

int fp_cmp_i32(const void * a, const void * b) {
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

int fp_cmp_rowkey(const void * a, const void * b) {
    const FpRowKey * x = (const FpRowKey *)a, * y = (const FpRowKey *)b;
    if (x->sq != y->sq) return (x->sq > y->sq) - (x->sq < y->sq);
    if (x->qh != y->qh) return (x->qh > y->qh) - (x->qh < y->qh);
    return 0;
}

int fp_cmp_rowtriple(const void * a, const void * b) {
    const FpRowKey * x = (const FpRowKey *)a, * y = (const FpRowKey *)b;
    if (x->sq != y->sq) return (x->sq > y->sq) - (x->sq < y->sq);
    if (x->tile != y->tile) return (x->tile > y->tile) - (x->tile < y->tile);
    if (x->kvh != y->kvh) return (x->kvh > y->kvh) - (x->kvh < y->kvh);
    return 0;
}

int fp_cmp_usekey(const void * a, const void * b) {
    const FpUseKey * x = (const FpUseKey *)a, * y = (const FpUseKey *)b;
    if (x->sq != y->sq) return (x->sq > y->sq) - (x->sq < y->sq);
    if (x->tile != y->tile) return (x->tile > y->tile) - (x->tile < y->tile);
    if (x->kvh != y->kvh) return (x->kvh > y->kvh) - (x->kvh < y->kvh);
    if (x->frag != y->frag) return (x->frag > y->frag) - (x->frag < y->frag);
    return 0;
}

int fp_cmp_usetriple(const void * a, const void * b) {
    const FpUseKey * x = (const FpUseKey *)a, * y = (const FpUseKey *)b;
    if (x->sq != y->sq) return (x->sq > y->sq) - (x->sq < y->sq);
    if (x->tile != y->tile) return (x->tile > y->tile) - (x->tile < y->tile);
    if (x->kvh != y->kvh) return (x->kvh > y->kvh) - (x->kvh < y->kvh);
    return 0;
}

// Index search for (sq, tile, kvh) in an array sorted by fp_cmp_rowtriple.
bool fp_triple_find(const FpRowKey * rows, int64_t n, int32_t sq, int32_t tile, int32_t kvh,
        int64_t * pos) {
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        const FpRowKey * m = &rows[mid];
        int c = 0;
        if (m->sq != sq) c = (m->sq > sq) - (m->sq < sq);
        else if (m->tile != tile) c = (m->tile > tile) - (m->tile < tile);
        else if (m->kvh != kvh) c = (m->kvh > kvh) - (m->kvh < kvh);
        if (c == 0) {
            if (pos) *pos = mid;
            return true;
        }
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    return false;
}

// Binary search for (sq, tile, kvh) in an array sorted by fp_cmp_rowtriple.
bool fp_triple_has_row(const FpRowKey * rows, int64_t n, int32_t sq, int32_t tile, int32_t kvh) {
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        const FpRowKey * m = &rows[mid];
        int c = 0;
        if (m->sq != sq) c = (m->sq > sq) - (m->sq < sq);
        else if (m->tile != tile) c = (m->tile > tile) - (m->tile < tile);
        else if (m->kvh != kvh) c = (m->kvh > kvh) - (m->kvh < kvh);
        if (c == 0) return true;
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    return false;
}

struct FpPlanView {
    const int32_t * base;
    int64_t n_tiles, n_heads, max_sel;
    int64_t exact_off, proxy_off, counts_off, total;
};

int32_t fp_parse_plan(const int32_t * plan, int64_t n_words, FpPlanView * v) {
    if (!plan || !v || n_words < GGML_FLASHPREFILL_PLAN_HEADER_WORDS) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (plan[0] != GGML_FLASHPREFILL_PLAN_MAGIC) {
        return GGML_FLASHPREFILL_ERR_BAD_MAGIC;
    }
    if (plan[1] != GGML_FLASHPREFILL_VERSION) {
        return GGML_FLASHPREFILL_ERR_BAD_VERSION;
    }
    if (plan[2] != GGML_FLASHPREFILL_PLAN_HEADER_WORDS) {
        return GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
    }
    for (int i = 3; i <= 5; i++) {
        if (plan[i] < 0) return GGML_FLASHPREFILL_ERR_BAD_RANGE;
    }
    int64_t th = 0, pair = 0, t = 0;
    int64_t proxy_off = 0, counts_off = 0, total = 0;
    const int64_t exact_off = GGML_FLASHPREFILL_PLAN_HEADER_WORDS;
    int32_t rc = fp_mul_ok(plan[3], plan[4], &th);
    if (rc) return rc;
    rc = fp_mul_ok(th, plan[5], &pair); if (rc) return rc;
    rc = fp_add_ok(exact_off, pair, &proxy_off); if (rc) return rc;
    rc = fp_mul_ok(th, 2, &t); if (rc) return rc;
    // Both exact and proxy tables contain th * max_sel entries. Counts
    // contain th * 2 entries; using that size for the proxy table only
    // happened to work when max_sel==2.
    rc = fp_add_ok(proxy_off, pair, &counts_off); if (rc) return rc;
    rc = fp_add_ok(counts_off, t, &total); if (rc) return rc;
    if (plan[6] != (int32_t)exact_off || plan[7] != (int32_t)proxy_off ||
        plan[8] != (int32_t)counts_off || plan[9] != (int32_t)total) {
        return GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
    }
    if (n_words != total) {
        return GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
    }
    for (int i = 20; i < 24; i++) {
        if (plan[i] != 0) return GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
    }
    v->base = plan;
    v->n_tiles = plan[3]; v->n_heads = plan[4]; v->max_sel = plan[5];
    v->exact_off = exact_off; v->proxy_off = proxy_off;
    v->counts_off = counts_off; v->total = total;
    return GGML_FLASHPREFILL_OK;
}

} // namespace

extern "C" {

const char * ggml_flashprefill_strerror(int32_t err) {
    switch (err) {
        case GGML_FLASHPREFILL_OK: return "ok";
        case GGML_FLASHPREFILL_ERR_BAD_ARG: return "bad argument";
        case GGML_FLASHPREFILL_ERR_OVERFLOW: return "offset/size overflow";
        case GGML_FLASHPREFILL_ERR_BAD_MAGIC: return "bad magic";
        case GGML_FLASHPREFILL_ERR_BAD_VERSION: return "bad version";
        case GGML_FLASHPREFILL_ERR_BAD_LAYOUT: return "bad layout";
        case GGML_FLASHPREFILL_ERR_CAP_EXCEEDED: return "capacity exceeded";
        case GGML_FLASHPREFILL_ERR_BAD_RANGE: return "value out of range";
        case GGML_FLASHPREFILL_ERR_BAD_FLAG: return "unknown flag bit";
        case GGML_FLASHPREFILL_ERR_DUP_KEY: return "duplicate key";
        case GGML_FLASHPREFILL_ERR_DUP_TOKEN: return "repeated physical token";
        case GGML_FLASHPREFILL_ERR_ORPHAN_USE: return "use matches no row";
        case GGML_FLASHPREFILL_ERR_PARTIAL_AS_PROXY: return "partial use as proxy";
        case GGML_FLASHPREFILL_ERR_PROXY_NOT_FULL: return "proxy not full fragment";
        case GGML_FLASHPREFILL_ERR_MANDATORY_AS_PROXY: return "mandatory use as proxy";
        case GGML_FLASHPREFILL_ERR_DENSE_ROW_PROXY: return "dense row not fully exact";
        case GGML_FLASHPREFILL_ERR_BAD_PLAN_COVERAGE: return "plan coverage mismatch";
        case GGML_FLASHPREFILL_ERR_PLAN_ROLE_MISMATCH: return "plan role mismatch";
        case GGML_FLASHPREFILL_ERR_BAD_CONFIG: return "bad config";
        case GGML_FLASHPREFILL_ERR_OVER_CAPACITY: return "selection over capacity";
        case GGML_FLASHPREFILL_ERR_BAD_INPUT: return "bad numeric input";
        default: return "unknown error";
    }
}

struct ggml_flashprefill_config ggml_flashprefill_config_default(void) {
    struct ggml_flashprefill_config cfg;
    cfg.mode              = GGML_FLASHPREFILL_DEFAULT_MODE;
    cfg.alpha             = GGML_FLASHPREFILL_DEFAULT_ALPHA;
    cfg.block_q           = GGML_FLASHPREFILL_DEFAULT_BLOCK_Q;
    cfg.block_k           = GGML_FLASHPREFILL_DEFAULT_BLOCK_K;
    cfg.sink_blocks       = GGML_FLASHPREFILL_DEFAULT_SINK_BLOCKS;
    cfg.window_blocks     = GGML_FLASHPREFILL_DEFAULT_WINDOW_BLOCKS;
    cfg.dense_tail_tiles  = GGML_FLASHPREFILL_DEFAULT_DENSE_TAIL_TILES;
    cfg.min_kv            = GGML_FLASHPREFILL_DEFAULT_MIN_KV;
    cfg.full_attn_layers  = GGML_FLASHPREFILL_DEFAULT_FULL_ATTN_LAYERS;
    cfg.tail_scope        = GGML_FLASHPREFILL_DEFAULT_TAIL_SCOPE;
    cfg.mean_correction   = true;
    cfg.exact_all         = false;
    return cfg;
}

int32_t ggml_flashprefill_config_validate(const struct ggml_flashprefill_config * cfg, int32_t * bad_field) {
    int32_t field = GGML_FLASHPREFILL_FIELD_NONE;
    int32_t rc = GGML_FLASHPREFILL_OK;
    if (!cfg) {
        rc = GGML_FLASHPREFILL_ERR_BAD_ARG;
    } else if (cfg->mode != GGML_FLASHPREFILL_MODE_OFF &&
               cfg->mode != GGML_FLASHPREFILL_MODE_AUTO &&
               cfg->mode != GGML_FLASHPREFILL_MODE_REQUIRED) {
        field = GGML_FLASHPREFILL_FIELD_MODE;
        rc = GGML_FLASHPREFILL_ERR_BAD_CONFIG;
    } else if (!fp_finite_f(cfg->alpha) || !(cfg->alpha > 0.0f) || !(cfg->alpha <= 1.0f)) {
        field = GGML_FLASHPREFILL_FIELD_ALPHA;
        rc = GGML_FLASHPREFILL_ERR_BAD_CONFIG;
    } else if (cfg->block_q < GGML_FLASHPREFILL_BLOCK_Q_MIN ||
               cfg->block_q > GGML_FLASHPREFILL_BLOCK_Q_MAX) {
        field = GGML_FLASHPREFILL_FIELD_BLOCK_Q;
        rc = GGML_FLASHPREFILL_ERR_BAD_CONFIG;
    } else if (cfg->block_k < GGML_FLASHPREFILL_BLOCK_K_MIN ||
               cfg->block_k > GGML_FLASHPREFILL_BLOCK_K_MAX ||
               (cfg->block_k % GGML_FLASHPREFILL_BLOCK_K_STEP) != 0) {
        field = GGML_FLASHPREFILL_FIELD_BLOCK_K;
        rc = GGML_FLASHPREFILL_ERR_BAD_CONFIG;
    } else if (cfg->sink_blocks > GGML_FLASHPREFILL_COUNT_WINDOW_MAX) {
        field = GGML_FLASHPREFILL_FIELD_SINK_BLOCKS;
        rc = GGML_FLASHPREFILL_ERR_BAD_CONFIG;
    } else if (cfg->window_blocks > GGML_FLASHPREFILL_COUNT_WINDOW_MAX) {
        field = GGML_FLASHPREFILL_FIELD_WINDOW_BLOCKS;
        rc = GGML_FLASHPREFILL_ERR_BAD_CONFIG;
    } else if (cfg->dense_tail_tiles > GGML_FLASHPREFILL_COUNT_WINDOW_MAX) {
        field = GGML_FLASHPREFILL_FIELD_DENSE_TAIL_TILES;
        rc = GGML_FLASHPREFILL_ERR_BAD_CONFIG;
    } else if (cfg->tail_scope != GGML_FLASHPREFILL_TAIL_SCOPE_CALL &&
               cfg->tail_scope != GGML_FLASHPREFILL_TAIL_SCOPE_LOGICAL_PROMPT) {
        field = GGML_FLASHPREFILL_FIELD_TAIL_SCOPE;
        rc = GGML_FLASHPREFILL_ERR_BAD_CONFIG;
    }
    if (bad_field) {
        *bad_field = field;
    }
    return rc;
}

int32_t ggml_flashprefill_metadata_words(
        int64_t f_cap, int64_t r_cap, int64_t u_cap, int64_t c_cap,
        int64_t * out_words) {
    if (!out_words || f_cap < 0 || r_cap < 0 || u_cap < 0 || c_cap < 0 ||
        f_cap > FP_I32_MAX || r_cap > FP_I32_MAX ||
        u_cap > FP_I32_MAX || c_cap > FP_I32_MAX) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    int64_t t = 0, total = GGML_FLASHPREFILL_HEADER_WORDS;
    int32_t rc = fp_mul_ok(f_cap, GGML_FLASHPREFILL_FRAG_WORDS, &t);
    if (rc) return rc;
    rc = fp_add_ok(total, t, &total); if (rc) return rc;
    rc = fp_mul_ok(r_cap, GGML_FLASHPREFILL_ROW_WORDS, &t); if (rc) return rc;
    rc = fp_add_ok(total, t, &total); if (rc) return rc;
    rc = fp_mul_ok(u_cap, GGML_FLASHPREFILL_USE_WORDS, &t); if (rc) return rc;
    rc = fp_add_ok(total, t, &total); if (rc) return rc;
    rc = fp_add_ok(total, c_cap, &total); if (rc) return rc;
    *out_words = total;
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_metadata_init(
        int32_t * meta, int64_t n_words,
        int64_t f_cap, int64_t r_cap, int64_t u_cap, int64_t c_cap,
        int32_t dk, int32_t dv, int32_t n_kv_heads,
        int32_t n_groups, int32_t n_q_heads) {
    if (!meta) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    int64_t want = 0;
    int32_t rc = ggml_flashprefill_metadata_words(f_cap, r_cap, u_cap, c_cap, &want);
    if (rc) return rc;
    if (n_words != want) {
        return GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
    }
    if (dk < 1 || dk > (int32_t)GGML_FLASHPREFILL_HEAD_DIM_MAX ||
        dv < 1 || dv > (int32_t)GGML_FLASHPREFILL_HEAD_DIM_MAX ||
        n_kv_heads < 1 || n_groups < 1 || n_q_heads < 1) {
        return GGML_FLASHPREFILL_ERR_BAD_RANGE;
    }
    int64_t row_off = 0, use_off = 0, cell_off = 0, t = 0;
    rc = fp_mul_ok(f_cap, GGML_FLASHPREFILL_FRAG_WORDS, &t); if (rc) return rc;
    rc = fp_add_ok(GGML_FLASHPREFILL_HEADER_WORDS, t, &row_off); if (rc) return rc;
    rc = fp_mul_ok(r_cap, GGML_FLASHPREFILL_ROW_WORDS, &t); if (rc) return rc;
    rc = fp_add_ok(row_off, t, &use_off); if (rc) return rc;
    rc = fp_mul_ok(u_cap, GGML_FLASHPREFILL_USE_WORDS, &t); if (rc) return rc;
    rc = fp_add_ok(use_off, t, &cell_off); if (rc) return rc;
    for (int64_t i = 0; i < n_words; i++) meta[i] = 0;
    meta[0]  = GGML_FLASHPREFILL_METADATA_MAGIC;
    meta[1]  = GGML_FLASHPREFILL_VERSION;
    meta[2]  = GGML_FLASHPREFILL_HEADER_WORDS;
    meta[4]  = (int32_t)f_cap;
    meta[6]  = (int32_t)r_cap;
    meta[8]  = (int32_t)u_cap;
    meta[10] = (int32_t)c_cap;
    meta[11] = GGML_FLASHPREFILL_HEADER_WORDS;
    meta[12] = (int32_t)row_off;
    meta[13] = (int32_t)use_off;
    meta[14] = (int32_t)cell_off;
    meta[15] = GGML_FLASHPREFILL_FRAG_WORDS;
    meta[16] = GGML_FLASHPREFILL_ROW_WORDS;
    meta[17] = GGML_FLASHPREFILL_USE_WORDS;
    meta[18] = dk;
    meta[19] = dv;
    meta[20] = n_kv_heads;
    meta[21] = (int32_t)want;
    meta[22] = n_groups;
    meta[23] = n_q_heads;
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_metadata_set_counts(
        int32_t * meta, int64_t n_words,
        int64_t n_frag, int64_t n_row, int64_t n_use, int64_t n_cell) {
    if (!meta) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    FpMetaView v;
    int32_t rc = fp_parse_meta(meta, n_words, &v);
    if (rc) return rc;
    if (n_frag < 0 || n_frag > v.f_cap || n_row < 0 || n_row > v.r_cap ||
        n_use < 0 || n_use > v.u_cap || n_cell < 0 || n_cell > v.c_cap) {
        return GGML_FLASHPREFILL_ERR_CAP_EXCEEDED;
    }
    meta[3] = (int32_t)n_frag;
    meta[5] = (int32_t)n_row;
    meta[7] = (int32_t)n_use;
    meta[9] = (int32_t)n_cell;
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_metadata_set_frag(
        int32_t * meta, int64_t n_words, int64_t idx,
        int64_t cell_off, int64_t cell_count,
        int32_t logical_block, int32_t domain, int32_t flags) {
    if (!meta) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    FpMetaView v;
    int32_t rc = fp_parse_meta(meta, n_words, &v);
    if (rc) return rc;
    if (idx < 0 || idx >= v.f_cap) {
        return GGML_FLASHPREFILL_ERR_BAD_RANGE;
    }
    if (cell_off < 0 || cell_count < 0 || cell_off + cell_count < cell_off ||
        cell_off + cell_count > v.c_cap) {
        return GGML_FLASHPREFILL_ERR_BAD_RANGE;
    }
    if ((flags & ~GGML_FLASHPREFILL_FRAG_FLAG_BOUNDARY_PARTIAL) != 0) {
        return GGML_FLASHPREFILL_ERR_BAD_FLAG;
    }
    int32_t * r = meta + v.frag_off + idx * GGML_FLASHPREFILL_FRAG_WORDS;
    r[0] = (int32_t)cell_off;
    r[1] = (int32_t)cell_count;
    r[2] = logical_block;
    r[3] = domain;
    r[4] = flags;
    r[5] = 0; r[6] = 0; r[7] = 0;
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_metadata_set_row(
        int32_t * meta, int64_t n_words, int64_t idx,
        int32_t source_query, int32_t kv_head, int32_t logical_pos,
        int32_t tile_id, int32_t prompt_begin, int32_t prompt_end,
        int32_t flags, int32_t q_head) {
    if (!meta) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    FpMetaView v;
    int32_t rc = fp_parse_meta(meta, n_words, &v);
    if (rc) return rc;
    if (idx < 0 || idx >= v.r_cap) {
        return GGML_FLASHPREFILL_ERR_BAD_RANGE;
    }
    if (source_query < 0 || kv_head < 0 || kv_head >= v.hkv ||
        logical_pos < 0 || tile_id < 0 ||
        prompt_begin < 0 || prompt_begin >= prompt_end ||
        q_head < 0 || q_head >= v.nqheads) {
        return GGML_FLASHPREFILL_ERR_BAD_RANGE;
    }
    if ((flags & ~GGML_FLASHPREFILL_ROW_FLAG_DENSE_FORCE) != 0) {
        return GGML_FLASHPREFILL_ERR_BAD_FLAG;
    }
    int32_t * r = meta + v.row_off + idx * GGML_FLASHPREFILL_ROW_WORDS;
    r[0] = source_query;
    r[1] = kv_head;
    r[2] = logical_pos;
    r[3] = tile_id;
    r[4] = prompt_begin;
    r[5] = prompt_end;
    r[6] = flags;
    r[7] = q_head;
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_metadata_set_use(
        int32_t * meta, int64_t n_words, int64_t idx,
        int32_t frag_id, int32_t tile_id, int32_t kv_head, int32_t q_group,
        int64_t sub_off, int64_t sub_count, int32_t flags, int32_t source_query) {
    if (!meta) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    FpMetaView v;
    int32_t rc = fp_parse_meta(meta, n_words, &v);
    if (rc) return rc;
    if (idx < 0 || idx >= v.u_cap) {
        return GGML_FLASHPREFILL_ERR_BAD_RANGE;
    }
    if (frag_id < 0 || frag_id >= v.f_cap || tile_id < 0 ||
        kv_head < 0 || kv_head >= v.hkv ||
        q_group < 0 || q_group >= v.ngroups || source_query < 0) {
        return GGML_FLASHPREFILL_ERR_BAD_RANGE;
    }
    if (sub_off < 0 || sub_count <= 0 || sub_off + sub_count < sub_off ||
        sub_off + sub_count > FP_I32_MAX) {
        return GGML_FLASHPREFILL_ERR_BAD_RANGE;
    }
    if ((flags & ~GGML_FLASHPREFILL_USE_FLAG_MANDATORY) != 0) {
        return GGML_FLASHPREFILL_ERR_BAD_FLAG;
    }
    int32_t * r = meta + v.use_off + idx * GGML_FLASHPREFILL_USE_WORDS;
    r[0] = frag_id;
    r[1] = tile_id;
    r[2] = kv_head;
    r[3] = q_group;
    r[4] = (int32_t)sub_off;
    r[5] = (int32_t)sub_count;
    r[6] = flags;
    r[7] = source_query;
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_metadata_set_cell(
        int32_t * meta, int64_t n_words, int64_t idx, int32_t cell) {
    if (!meta) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    FpMetaView v;
    int32_t rc = fp_parse_meta(meta, n_words, &v);
    if (rc) return rc;
    if (idx < 0 || idx >= v.c_cap) {
        return GGML_FLASHPREFILL_ERR_BAD_RANGE;
    }
    if (cell < 0) {
        return GGML_FLASHPREFILL_ERR_BAD_RANGE;
    }
    int32_t * c = meta + v.cell_off;
    c[idx] = cell;
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_metadata_n_tiles(
        const int32_t * meta, int64_t n_words, int64_t * out_n_tiles) {
    if (!out_n_tiles) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    FpMetaView v;
    int32_t rc = fp_parse_meta(meta, n_words, &v);
    if (rc) return rc;
    int64_t hi = -1;
    for (int64_t i = 0; i < v.n_row; i++) {
        const int32_t * r = v.base + v.row_off + i * GGML_FLASHPREFILL_ROW_WORDS;
        if (r[3] > hi) hi = r[3];
    }
    *out_n_tiles = hi + 1;
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_metadata_validate(const int32_t * meta, int64_t n_words) {
    FpMetaView v;
    int32_t rc = fp_parse_meta(meta, n_words, &v);
    if (rc) return rc;

    // --- fragment ranges / flags ---
    for (int64_t i = 0; i < v.n_frag; i++) {
        const int32_t * r = v.base + v.frag_off + i * GGML_FLASHPREFILL_FRAG_WORDS;
        int64_t off = r[0], cnt = r[1];
        if (off < 0 || cnt < 0 || off + cnt < off || off + cnt > v.n_cell) {
            return GGML_FLASHPREFILL_ERR_BAD_RANGE;
        }
        if ((r[4] & ~GGML_FLASHPREFILL_FRAG_FLAG_BOUNDARY_PARTIAL) != 0) {
            return GGML_FLASHPREFILL_ERR_BAD_FLAG;
        }
        if (r[5] != 0 || r[6] != 0 || r[7] != 0) {
            return GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
        }
    }
    // --- row ranges / flags ---
    for (int64_t i = 0; i < v.n_row; i++) {
        const int32_t * r = v.base + v.row_off + i * GGML_FLASHPREFILL_ROW_WORDS;
        if (r[0] < 0 || r[1] < 0 || r[1] >= v.hkv || r[2] < 0 || r[3] < 0 ||
            r[4] < 0 || r[4] >= r[5] || r[7] < 0 || r[7] >= v.nqheads) {
            return GGML_FLASHPREFILL_ERR_BAD_RANGE;
        }
        if ((r[6] & ~GGML_FLASHPREFILL_ROW_FLAG_DENSE_FORCE) != 0) {
            return GGML_FLASHPREFILL_ERR_BAD_FLAG;
        }
    }
    // --- use ranges / flags / subset inside fragment ---
    for (int64_t i = 0; i < v.n_use; i++) {
        const int32_t * r = v.base + v.use_off + i * GGML_FLASHPREFILL_USE_WORDS;
        if (r[0] < 0 || r[0] >= v.n_frag || r[1] < 0 ||
            r[2] < 0 || r[2] >= v.hkv ||
            r[3] < 0 || r[3] >= v.ngroups || r[7] < 0) {
            return GGML_FLASHPREFILL_ERR_BAD_RANGE;
        }
        if ((r[6] & ~GGML_FLASHPREFILL_USE_FLAG_MANDATORY) != 0) {
            return GGML_FLASHPREFILL_ERR_BAD_FLAG;
        }
        int64_t so = r[4], sc = r[5];
        if (so < 0 || sc <= 0 || so + sc < so) {
            return GGML_FLASHPREFILL_ERR_BAD_RANGE;
        }
        const int32_t * f = v.base + v.frag_off + (int64_t)r[0] * GGML_FLASHPREFILL_FRAG_WORDS;
        int64_t fo = f[0], fc = f[1];
        if (so < fo || so + sc > fo + fc) {
            return GGML_FLASHPREFILL_ERR_BAD_RANGE;
        }
    }
    for (int64_t i = 0; i < v.n_cell; i++) {
        if ((v.base + v.cell_off)[i] < 0) {
            return GGML_FLASHPREFILL_ERR_BAD_RANGE;
        }
    }
    if (v.n_row == 0 || v.n_use == 0) {
        return GGML_FLASHPREFILL_OK; // nothing further to cross-check
    }

    FpRowKey * rows = (FpRowKey *)malloc(
            (size_t)(v.n_row > 0 ? v.n_row : 1) * sizeof(FpRowKey));
    FpUseKey * uses = (FpUseKey *)malloc(
            (size_t)(v.n_use > 0 ? v.n_use : 1) * sizeof(FpUseKey));
    if (!rows || !uses) {
        free(rows);
        free(uses);
        return GGML_FLASHPREFILL_ERR_OVERFLOW;
    }
    for (int64_t i = 0; i < v.n_row; i++) {
        const int32_t * r = v.base + v.row_off + i * GGML_FLASHPREFILL_ROW_WORDS;
        rows[i].sq = r[0]; rows[i].kvh = r[1]; rows[i].tile = r[3];
        rows[i].qh = r[7];
        rows[i].dense = (r[6] & GGML_FLASHPREFILL_ROW_FLAG_DENSE_FORCE) != 0;
        rows[i].idx = i;
    }
    for (int64_t i = 0; i < v.n_use; i++) {
        const int32_t * r = v.base + v.use_off + i * GGML_FLASHPREFILL_USE_WORDS;
        uses[i].sq = r[7]; uses[i].tile = r[1]; uses[i].kvh = r[2];
        uses[i].frag = r[0]; uses[i].group = r[3];
        uses[i].sub_off = r[4]; uses[i].sub_count = r[5];
        uses[i].mandatory = (r[6] & GGML_FLASHPREFILL_USE_FLAG_MANDATORY) != 0;
        uses[i].idx = i;
    }
    // Row key (source_query, q_head) unique.
    {
        FpRowKey * by_key = (FpRowKey *)malloc((size_t)v.n_row * sizeof(FpRowKey));
        if (!by_key) {
            free(rows); free(uses);
            return GGML_FLASHPREFILL_ERR_OVERFLOW;
        }
        memcpy(by_key, rows, (size_t)v.n_row * sizeof(FpRowKey));
        qsort(by_key, (size_t)v.n_row, sizeof(FpRowKey), fp_cmp_rowkey);
        rc = GGML_FLASHPREFILL_OK;
        for (int64_t i = 1; i < v.n_row; i++) {
            if (by_key[i].sq == by_key[i-1].sq && by_key[i].qh == by_key[i-1].qh) {
                rc = GGML_FLASHPREFILL_ERR_DUP_KEY;
                break;
            }
        }
        free(by_key);
        if (rc) {
            free(rows); free(uses);
            return rc;
        }
    }
    // Use key (source_query, tile, kv_head, frag) unique.
    {
        FpUseKey * by_key = (FpUseKey *)malloc((size_t)v.n_use * sizeof(FpUseKey));
        if (!by_key) {
            free(rows); free(uses);
            return GGML_FLASHPREFILL_ERR_OVERFLOW;
        }
        memcpy(by_key, uses, (size_t)v.n_use * sizeof(FpUseKey));
        qsort(by_key, (size_t)v.n_use, sizeof(FpUseKey), fp_cmp_usekey);
        rc = GGML_FLASHPREFILL_OK;
        for (int64_t i = 1; i < v.n_use; i++) {
            if (by_key[i].sq == by_key[i-1].sq && by_key[i].tile == by_key[i-1].tile &&
                by_key[i].kvh == by_key[i-1].kvh && by_key[i].frag == by_key[i-1].frag) {
                rc = GGML_FLASHPREFILL_ERR_DUP_KEY;
                break;
            }
        }
        free(by_key);
        if (rc) {
            free(rows); free(uses);
            return rc;
        }
    }
    // Every use matches >= 1 row on (source_query, tile, kv_head).
    {
        FpRowKey * by_triple = (FpRowKey *)malloc((size_t)v.n_row * sizeof(FpRowKey));
        if (!by_triple) {
            free(rows); free(uses);
            return GGML_FLASHPREFILL_ERR_OVERFLOW;
        }
        memcpy(by_triple, rows, (size_t)v.n_row * sizeof(FpRowKey));
        qsort(by_triple, (size_t)v.n_row, sizeof(FpRowKey), fp_cmp_rowtriple);
        rc = GGML_FLASHPREFILL_OK;
        for (int64_t i = 0; i < v.n_use; i++) {
            if (!fp_triple_has_row(by_triple, v.n_row, uses[i].sq, uses[i].tile, uses[i].kvh)) {
                rc = GGML_FLASHPREFILL_ERR_ORPHAN_USE;
                break;
            }
        }
        // DENSE_FORCE rows match only MANDATORY uses.
        if (rc == GGML_FLASHPREFILL_OK) {
            FpUseKey * by_triple_u = (FpUseKey *)malloc((size_t)v.n_use * sizeof(FpUseKey));
            if (!by_triple_u) {
                free(by_triple); free(rows); free(uses);
                return GGML_FLASHPREFILL_ERR_OVERFLOW;
            }
            memcpy(by_triple_u, uses, (size_t)v.n_use * sizeof(FpUseKey));
            qsort(by_triple_u, (size_t)v.n_use, sizeof(FpUseKey), fp_cmp_usetriple);
            for (int64_t i = 0; i < v.n_row && rc == GGML_FLASHPREFILL_OK; i++) {
                if (!rows[i].dense) continue;
                for (int64_t j = 0; j < v.n_use; j++) {
                    if (by_triple_u[j].sq == rows[i].sq &&
                        by_triple_u[j].tile == rows[i].tile &&
                        by_triple_u[j].kvh == rows[i].kvh &&
                        !by_triple_u[j].mandatory) {
                        rc = GGML_FLASHPREFILL_ERR_DENSE_ROW_PROXY;
                        break;
                    }
                }
            }
            free(by_triple_u);
        }
        free(by_triple);
        if (rc) {
            free(rows); free(uses);
            return rc;
        }
    }
    // No repeated physical token per matched (source_query, tile, kv_head).
    {
        FpUseKey * by_triple = (FpUseKey *)malloc((size_t)v.n_use * sizeof(FpUseKey));
        if (!by_triple) {
            free(rows); free(uses);
            return GGML_FLASHPREFILL_ERR_OVERFLOW;
        }
        memcpy(by_triple, uses, (size_t)v.n_use * sizeof(FpUseKey));
        qsort(by_triple, (size_t)v.n_use, sizeof(FpUseKey), fp_cmp_usetriple);
        rc = GGML_FLASHPREFILL_OK;
        const int32_t * cells = v.base + v.cell_off;
        int64_t start = 0;
        while (start < v.n_use && rc == GGML_FLASHPREFILL_OK) {
            int64_t end = start + 1;
            int64_t total_refs = by_triple[start].sub_count;
            while (end < v.n_use &&
                   by_triple[end].sq == by_triple[start].sq &&
                   by_triple[end].tile == by_triple[start].tile &&
                   by_triple[end].kvh == by_triple[start].kvh) {
                if (total_refs > FP_I32_MAX - by_triple[end].sub_count) {
                    rc = GGML_FLASHPREFILL_ERR_OVERFLOW;
                    break;
                }
                total_refs += by_triple[end].sub_count;
                end++;
            }
            if (rc == GGML_FLASHPREFILL_OK && total_refs > 0) {
                int32_t * vals = (int32_t *)malloc((size_t)total_refs * sizeof(int32_t));
                if (!vals) {
                    rc = GGML_FLASHPREFILL_ERR_OVERFLOW;
                } else {
                    int64_t k = 0;
                    for (int64_t j = start; j < end; j++) {
                        for (int64_t c = 0; c < by_triple[j].sub_count; c++) {
                            vals[k++] = cells[by_triple[j].sub_off + c];
                        }
                    }
                    qsort(vals, (size_t)total_refs, sizeof(int32_t), fp_cmp_i32);
                    for (int64_t k2 = 1; k2 < total_refs; k2++) {
                        if (vals[k2] == vals[k2-1]) {
                            rc = GGML_FLASHPREFILL_ERR_DUP_TOKEN;
                            break;
                        }
                    }
                    free(vals);
                }
            }
            start = end;
        }
        free(by_triple);
        if (rc) {
            free(rows); free(uses);
            return rc;
        }
    }
    free(rows);
    free(uses);
    return GGML_FLASHPREFILL_OK;
}

int64_t ggml_flashprefill_pool_nelements(
        int32_t dk, int32_t dv, int64_t n_heads, int64_t f_cap) {
    if (dk < 1 || dv < 1 || n_heads < 0 || f_cap < 0) {
        return -1;
    }
    int64_t dd = (int64_t)dk + (int64_t)dv;
    if (dd > FP_I32_MAX) {
        return -1;
    }
    int64_t t = 0;
    if (fp_mul_ok(dd, n_heads, &t)) return -1;
    int64_t out = 0;
    if (fp_mul_ok(t, f_cap, &out)) return -1;
    return out;
}

int32_t ggml_flashprefill_plan_words(
        int64_t n_tiles, int64_t n_heads, int64_t max_sel_pair, int64_t * out_words) {
    if (!out_words || n_tiles < 0 || n_heads < 0 || max_sel_pair < 0 ||
        n_tiles > FP_I32_MAX || n_heads > FP_I32_MAX || max_sel_pair > FP_I32_MAX) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    int64_t th = 0, pair = 0, lists = 0, counts = 0, total = 0;
    int32_t rc = fp_mul_ok(n_tiles, n_heads, &th);
    if (rc) return rc;
    rc = fp_mul_ok(th, max_sel_pair, &pair); if (rc) return rc;
    rc = fp_add_ok(pair, pair, &lists); if (rc) return rc;
    rc = fp_mul_ok(th, 2, &counts); if (rc) return rc;
    total = GGML_FLASHPREFILL_PLAN_HEADER_WORDS;
    rc = fp_add_ok(total, lists, &total); if (rc) return rc;
    rc = fp_add_ok(total, counts, &total); if (rc) return rc;
    *out_words = total;
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_plan_max_sel_for_meta(
        const int32_t * meta, int64_t meta_words, int64_t * out_max) {
    if (!out_max) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    FpMetaView v;
    int32_t rc = fp_parse_meta(meta, meta_words, &v);
    if (rc) return rc;
    if (v.n_use == 0) {
        *out_max = 0;
        return GGML_FLASHPREFILL_OK;
    }
    // Count uses per (tile_id, kv_head) via a sorted copy.
    struct Pair {
        int32_t tile, head;
    };
    Pair * pairs = (Pair *)malloc((size_t)v.n_use * sizeof(Pair));
    if (!pairs) {
        return GGML_FLASHPREFILL_ERR_OVERFLOW;
    }
    for (int64_t i = 0; i < v.n_use; i++) {
        const int32_t * r = v.base + v.use_off + i * GGML_FLASHPREFILL_USE_WORDS;
        pairs[i].tile = r[1];
        pairs[i].head = r[2];
    }
    // Insertion-free sort via qsort on 64-bit combined keys.
    uint64_t * keys = (uint64_t *)malloc((size_t)v.n_use * sizeof(uint64_t));
    if (!keys) {
        free(pairs);
        return GGML_FLASHPREFILL_ERR_OVERFLOW;
    }
    for (int64_t i = 0; i < v.n_use; i++) {
        keys[i] = ((uint64_t)(uint32_t)pairs[i].tile << 32) | (uint32_t)pairs[i].head;
    }
    free(pairs);
    qsort(keys, (size_t)v.n_use, sizeof(uint64_t), fp_cmp_u64);
    int64_t best = 1, run = 1;
    for (int64_t i = 1; i < v.n_use; i++) {
        if (keys[i] == keys[i-1]) {
            run++;
        } else {
            if (run > best) best = run;
            run = 1;
        }
    }
    if (run > best) best = run;
    free(keys);
    *out_max = best;
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_plan_init(
        int32_t * plan, int64_t n_words,
        int64_t n_tiles, int64_t n_heads, int64_t max_sel_pair, int32_t req_exact_all) {
    if (!plan) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (req_exact_all != 0 && req_exact_all != 1) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    int64_t want = 0;
    int32_t rc = ggml_flashprefill_plan_words(n_tiles, n_heads, max_sel_pair, &want);
    if (rc) return rc;
    if (n_words != want) {
        return GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
    }
    int64_t th = n_tiles * n_heads; // checked products fit: words() passed
    int64_t pair = th * max_sel_pair;
    int64_t exact_off = GGML_FLASHPREFILL_PLAN_HEADER_WORDS;
    int64_t proxy_off = exact_off + pair;
    int64_t counts_off = proxy_off + pair;
    for (int64_t i = 0; i < n_words; i++) plan[i] = 0;
    plan[0] = GGML_FLASHPREFILL_PLAN_MAGIC;
    plan[1] = GGML_FLASHPREFILL_VERSION;
    plan[2] = GGML_FLASHPREFILL_PLAN_HEADER_WORDS;
    plan[3] = (int32_t)n_tiles;
    plan[4] = (int32_t)n_heads;
    plan[5] = (int32_t)max_sel_pair;
    plan[6] = (int32_t)exact_off;
    plan[7] = (int32_t)proxy_off;
    plan[8] = (int32_t)counts_off;
    plan[9] = (int32_t)want;
    plan[15] = req_exact_all;
    for (int64_t i = 0; i < pair; i++) {
        plan[exact_off + i] = -1;
        plan[proxy_off + i] = -1;
    }
    (void)th;
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_plan_set_counts(
        int32_t * plan, int64_t n_words,
        int64_t tile, int64_t head, int64_t n_exact, int64_t n_proxy) {
    if (!plan) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    FpPlanView v;
    int32_t rc = fp_parse_plan(plan, n_words, &v);
    if (rc) return rc;
    if (tile < 0 || tile >= v.n_tiles || head < 0 || head >= v.n_heads) {
        return GGML_FLASHPREFILL_ERR_BAD_RANGE;
    }
    if (n_exact < 0 || n_exact > v.max_sel || n_proxy < 0 || n_proxy > v.max_sel) {
        return GGML_FLASHPREFILL_ERR_CAP_EXCEEDED;
    }
    int64_t slot = tile * v.n_heads + head;
    int32_t * p = plan + v.counts_off + slot * 2;
    p[0] = (int32_t)n_exact;
    p[1] = (int32_t)n_proxy;
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_plan_set_entry(
        int32_t * plan, int64_t n_words,
        int64_t tile, int64_t head, int32_t role, int64_t slot, int64_t use_idx) {
    if (!plan) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    FpPlanView v;
    int32_t rc = fp_parse_plan(plan, n_words, &v);
    if (rc) return rc;
    if (tile < 0 || tile >= v.n_tiles || head < 0 || head >= v.n_heads) {
        return GGML_FLASHPREFILL_ERR_BAD_RANGE;
    }
    if (role != GGML_FLASHPREFILL_ROLE_EXACT && role != GGML_FLASHPREFILL_ROLE_PROXY) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (slot < 0 || slot >= v.max_sel) {
        return GGML_FLASHPREFILL_ERR_BAD_RANGE;
    }
    if (use_idx < 0 || use_idx > FP_I32_MAX) {
        return GGML_FLASHPREFILL_ERR_BAD_RANGE;
    }
    int64_t base = (role == GGML_FLASHPREFILL_ROLE_EXACT) ? v.exact_off : v.proxy_off;
    int32_t * p = plan + base + (tile * v.n_heads + head) * v.max_sel + slot;
    *p = (int32_t)use_idx;
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_plan_validate(const int32_t * plan, int64_t n_words) {
    FpPlanView v;
    int32_t rc = fp_parse_plan(plan, n_words, &v);
    if (rc) return rc;
    int64_t npair = v.n_tiles * v.n_heads; // products checked in parse
    for (int64_t p = 0; p < npair; p++) {
        int64_t ne = v.base[v.counts_off + p * 2];
        int64_t np = v.base[v.counts_off + p * 2 + 1];
        if (ne < 0 || ne > v.max_sel || np < 0 || np > v.max_sel) {
            return GGML_FLASHPREFILL_ERR_CAP_EXCEEDED;
        }
        const int32_t * e = v.base + v.exact_off + p * v.max_sel;
        const int32_t * x = v.base + v.proxy_off + p * v.max_sel;
        for (int64_t i = 0; i < ne; i++) {
            if (e[i] < 0) return GGML_FLASHPREFILL_ERR_BAD_RANGE;
        }
        for (int64_t i = ne; i < v.max_sel; i++) {
            if (e[i] != -1) return GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
        }
        for (int64_t i = 0; i < np; i++) {
            if (x[i] < 0) return GGML_FLASHPREFILL_ERR_BAD_RANGE;
        }
        for (int64_t i = np; i < v.max_sel; i++) {
            if (x[i] != -1) return GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
        }
        // No use index twice within the pair (across both lists).
        for (int64_t i = 0; i < ne; i++) {
            for (int64_t j = 0; j < np; j++) {
                if (e[i] == x[j]) return GGML_FLASHPREFILL_ERR_DUP_KEY;
            }
            for (int64_t j = i + 1; j < ne; j++) {
                if (e[i] == e[j]) return GGML_FLASHPREFILL_ERR_DUP_KEY;
            }
        }
        for (int64_t i = 0; i < np; i++) {
            for (int64_t j = i + 1; j < np; j++) {
                if (x[i] == x[j]) return GGML_FLASHPREFILL_ERR_DUP_KEY;
            }
        }
    }
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_plan_accumulate_stats(
        int32_t * plan, int64_t plan_words,
        const int32_t * meta, int64_t meta_words) {
    if (!plan) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    FpPlanView pv;
    int32_t rc = fp_parse_plan(plan, plan_words, &pv);
    if (rc) return rc;
    rc = ggml_flashprefill_plan_validate(plan, plan_words);
    if (rc) return rc;
    FpMetaView mv;
    rc = fp_parse_meta(meta, meta_words, &mv);
    if (rc) return rc;
    // Mark listed uses: 1 = exact, 2 = proxy. Reject out-of-range indices.
    int8_t * mark = nullptr;
    if (mv.n_use > 0) {
        mark = (int8_t *)malloc((size_t)mv.n_use);
        if (!mark) return GGML_FLASHPREFILL_ERR_OVERFLOW;
        memset(mark, 0, (size_t)mv.n_use);
    }
    int64_t npair = pv.n_tiles * pv.n_heads;
    int64_t selected = 0, corrected = 0;
    int64_t visible = 0, exact_tok = 0;
    rc = GGML_FLASHPREFILL_OK;
    for (int64_t p = 0; p < npair && rc == GGML_FLASHPREFILL_OK; p++) {
        int64_t ne = pv.base[pv.counts_off + p * 2];
        int64_t np = pv.base[pv.counts_off + p * 2 + 1];
        const int32_t * e = pv.base + pv.exact_off + p * pv.max_sel;
        const int32_t * x = pv.base + pv.proxy_off + p * pv.max_sel;
        for (int64_t i = 0; i < ne && rc == GGML_FLASHPREFILL_OK; i++) {
            int64_t u = e[i];
            if (u >= mv.n_use) {
                rc = GGML_FLASHPREFILL_ERR_BAD_RANGE;
                break;
            }
            if (mark[u] != 0) {
                rc = GGML_FLASHPREFILL_ERR_DUP_KEY;
                break;
            }
            mark[u] = 1;
            const int32_t * ur = mv.base + mv.use_off + u * GGML_FLASHPREFILL_USE_WORDS;
            exact_tok += ur[5];
            selected++;
        }
        for (int64_t i = 0; i < np && rc == GGML_FLASHPREFILL_OK; i++) {
            int64_t u = x[i];
            if (u >= mv.n_use) {
                rc = GGML_FLASHPREFILL_ERR_BAD_RANGE;
                break;
            }
            if (mark[u] != 0) {
                rc = GGML_FLASHPREFILL_ERR_DUP_KEY;
                break;
            }
            mark[u] = 2;
            corrected++;
        }
    }
    // visible_tokens: legal membership incidences over ALL metadata uses,
    // independent of plan coverage.
    if (rc == GGML_FLASHPREFILL_OK) {
        for (int64_t i = 0; i < mv.n_use; i++) {
            const int32_t * ur = mv.base + mv.use_off + i * GGML_FLASHPREFILL_USE_WORDS;
            if (ur[5] <= 0 || ur[1] < 0 || ur[2] < 0 || ur[7] < 0) {
                rc = GGML_FLASHPREFILL_ERR_BAD_RANGE;
                break;
            }
            visible += ur[5];
        }
    }
    // A packed row is sparse iff >= 1 matched metadata use is proxy-listed
    // or missing from both plan lists. exact_all, all-exact, and
    // no-matched-use rows are dense. Unlisted uses fail visible as sparse,
    // never as fake-dense.
    int64_t sparse = 0;
    if (rc == GGML_FLASHPREFILL_OK && mv.n_row > 0 && mv.n_use > 0) {
        FpRowKey * triples = (FpRowKey *)malloc((size_t)mv.n_row * sizeof(FpRowKey));
        int8_t * tflag = (int8_t *)malloc((size_t)(mv.n_row > 0 ? mv.n_row : 1));
        if (!triples || !tflag) {
            free(triples);
            free(tflag);
            free(mark);
            return GGML_FLASHPREFILL_ERR_OVERFLOW;
        }
        for (int64_t i = 0; i < mv.n_row; i++) {
            const int32_t * r = mv.base + mv.row_off + i * GGML_FLASHPREFILL_ROW_WORDS;
            triples[i].sq = r[0]; triples[i].tile = r[3]; triples[i].kvh = r[1];
            triples[i].qh = 0; triples[i].dense = 0; triples[i].idx = 0;
            tflag[i] = 0;
        }
        qsort(triples, (size_t)mv.n_row, sizeof(FpRowKey), fp_cmp_rowtriple);
        for (int64_t i = 0; i < mv.n_use; i++) {
            if (mark[i] == 1) continue; // fully-exact uses keep rows dense
            const int32_t * ur = mv.base + mv.use_off + i * GGML_FLASHPREFILL_USE_WORDS;
            int64_t pos = -1;
            if (fp_triple_find(triples, mv.n_row, ur[7], ur[1], ur[2], &pos)) {
                int64_t k = pos; // flag every duplicate triple entry
                while (k >= 0 && triples[k].sq == ur[7] &&
                       triples[k].tile == ur[1] && triples[k].kvh == ur[2]) {
                    tflag[k] = 1;
                    k--;
                }
                k = pos + 1;
                while (k < mv.n_row && triples[k].sq == ur[7] &&
                       triples[k].tile == ur[1] && triples[k].kvh == ur[2]) {
                    tflag[k] = 1;
                    k++;
                }
            }
        }
        for (int64_t i = 0; i < mv.n_row; i++) {
            const int32_t * r = mv.base + mv.row_off + i * GGML_FLASHPREFILL_ROW_WORDS;
            int64_t pos = -1;
            if (fp_triple_find(triples, mv.n_row, r[0], r[3], r[1], &pos) && tflag[pos]) {
                sparse++;
            }
        }
        free(triples);
        free(tflag);
    }
    free(mark);
    if (rc) return rc;
    if (selected > FP_I32_MAX || corrected > FP_I32_MAX || sparse > FP_I32_MAX ||
        mv.n_row - sparse < 0 || mv.n_row - sparse > FP_I32_MAX) {
        return GGML_FLASHPREFILL_ERR_OVERFLOW;
    }
    plan[11] = (int32_t)sparse;
    plan[12] = (int32_t)(mv.n_row - sparse);
    plan[13] = (int32_t)selected;
    plan[14] = (int32_t)corrected;
    plan[16] = (int32_t)(visible & 0xFFFFFFFFLL);
    plan[17] = (int32_t)((visible >> 32) & 0xFFFFFFFFLL);
    plan[18] = (int32_t)(exact_tok & 0xFFFFFFFFLL);
    plan[19] = (int32_t)((exact_tok >> 32) & 0xFFFFFFFFLL);
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_plan_get_stats(
        const int32_t * plan, int64_t n_words,
        struct ggml_flashprefill_plan_stats * out) {
    if (!out) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    FpPlanView v;
    int32_t rc = fp_parse_plan(plan, n_words, &v);
    if (rc) return rc;
    out->error           = plan[10];
    out->sparse_rows     = plan[11];
    out->dense_rows      = plan[12];
    out->selected_total  = plan[13];
    out->corrected_total = plan[14];
    out->req_exact_all   = plan[15];
    uint32_t vlo = (uint32_t)plan[16], vhi = (uint32_t)plan[17];
    uint32_t elo = (uint32_t)plan[18], ehi = (uint32_t)plan[19];
    out->visible_tokens = (int64_t)(((uint64_t)vhi << 32) | vlo);
    out->exact_tokens   = (int64_t)(((uint64_t)ehi << 32) | elo);
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_plan_validate_against_metadata(
        const int32_t * plan, int64_t plan_words,
        const int32_t * meta, int64_t meta_words) {
    FpPlanView pv;
    int32_t rc = fp_parse_plan(plan, plan_words, &pv);
    if (rc) return rc;
    rc = ggml_flashprefill_plan_validate(plan, plan_words);
    if (rc) return rc;
    rc = ggml_flashprefill_metadata_validate(meta, meta_words);
    if (rc) return rc;
    FpMetaView mv;
    rc = fp_parse_meta(meta, meta_words, &mv);
    if (rc) return rc;
    if (mv.n_use == 0) {
        // Empty metadata: every plan list must be empty.
        int64_t npair = pv.n_tiles * pv.n_heads;
        for (int64_t p = 0; p < npair; p++) {
            if (pv.base[pv.counts_off + p * 2] != 0 ||
                pv.base[pv.counts_off + p * 2 + 1] != 0) {
                return GGML_FLASHPREFILL_ERR_BAD_PLAN_COVERAGE;
            }
        }
        return GGML_FLASHPREFILL_OK;
    }
    // n_tiles of the plan must cover the metadata tile span. Metadata tiles
    // are 0-based via rows; an empty row set needs plan n_tiles == 0.
    int64_t meta_tiles = 0;
    rc = ggml_flashprefill_metadata_n_tiles(meta, meta_words, &meta_tiles);
    if (rc) return rc;
    if (pv.n_tiles != meta_tiles || pv.n_heads != mv.hkv) {
        return GGML_FLASHPREFILL_ERR_BAD_PLAN_COVERAGE;
    }
    // Per (tile,head): plan exact+proxy sets must equal the metadata use set
    // exactly once each, with role rules.
    int8_t * seen = (int8_t *)malloc((size_t)mv.n_use);
    if (!seen) return GGML_FLASHPREFILL_ERR_OVERFLOW;
    memset(seen, 0, (size_t)mv.n_use);
    rc = GGML_FLASHPREFILL_OK;
    int64_t npair = pv.n_tiles * pv.n_heads;
    for (int64_t p = 0; p < npair && rc == GGML_FLASHPREFILL_OK; p++) {
        int64_t tile = (pv.n_heads == 0) ? 0 : p / pv.n_heads;
        int64_t head = (pv.n_heads == 0) ? 0 : p % pv.n_heads;
        int64_t ne = pv.base[pv.counts_off + p * 2];
        int64_t np = pv.base[pv.counts_off + p * 2 + 1];
        const int32_t * e = pv.base + pv.exact_off + p * pv.max_sel;
        const int32_t * x = pv.base + pv.proxy_off + p * pv.max_sel;
        // Expected set: metadata uses with this (tile, kv_head).
        for (int64_t i = 0; i < mv.n_use && rc == GGML_FLASHPREFILL_OK; i++) {
            const int32_t * ur = mv.base + mv.use_off + i * GGML_FLASHPREFILL_USE_WORDS;
            if (ur[1] != (int32_t)tile || ur[2] != (int32_t)head) continue;
            bool in_exact = false, in_proxy = false;
            for (int64_t k = 0; k < ne; k++) if (e[k] == i) in_exact = true;
            for (int64_t k = 0; k < np; k++) if (x[k] == i) in_proxy = true;
            if (in_exact == in_proxy) {
                rc = GGML_FLASHPREFILL_ERR_BAD_PLAN_COVERAGE;
                break;
            }
            bool mandatory = (ur[6] & GGML_FLASHPREFILL_USE_FLAG_MANDATORY) != 0;
            const int32_t * fr = mv.base + mv.frag_off +
                    (int64_t)ur[0] * GGML_FLASHPREFILL_FRAG_WORDS;
            bool full = (ur[4] == fr[0] && ur[5] == fr[1]);
            if (in_proxy) {
                if (!full) {
                    rc = GGML_FLASHPREFILL_ERR_PROXY_NOT_FULL;
                    break;
                }
                if (mandatory) {
                    rc = GGML_FLASHPREFILL_ERR_MANDATORY_AS_PROXY;
                    break;
                }
            } else {
                if (mandatory) {
                    // ok: mandatory must be exact
                } else if (!full) {
                    // ok: partial uses are exact-only
                }
                // non-mandatory full uses may be either role: no check.
            }
            seen[i] = 1;
        }
        // No listed index may fall outside the metadata set of this pair.
        for (int64_t k = 0; k < ne && rc == GGML_FLASHPREFILL_OK; k++) {
            if (e[k] >= mv.n_use) {
                rc = GGML_FLASHPREFILL_ERR_BAD_RANGE;
                break;
            }
            const int32_t * ur = mv.base + mv.use_off + (int64_t)e[k] * GGML_FLASHPREFILL_USE_WORDS;
            if (ur[1] != (int32_t)tile || ur[2] != (int32_t)head) {
                rc = GGML_FLASHPREFILL_ERR_BAD_PLAN_COVERAGE;
                break;
            }
        }
        for (int64_t k = 0; k < np && rc == GGML_FLASHPREFILL_OK; k++) {
            if (x[k] >= mv.n_use) {
                rc = GGML_FLASHPREFILL_ERR_BAD_RANGE;
                break;
            }
            const int32_t * ur = mv.base + mv.use_off + (int64_t)x[k] * GGML_FLASHPREFILL_USE_WORDS;
            if (ur[1] != (int32_t)tile || ur[2] != (int32_t)head) {
                rc = GGML_FLASHPREFILL_ERR_BAD_PLAN_COVERAGE;
                break;
            }
        }
    }
    if (rc == GGML_FLASHPREFILL_OK) {
        for (int64_t i = 0; i < mv.n_use; i++) {
            if (!seen[i]) {
                rc = GGML_FLASHPREFILL_ERR_BAD_PLAN_COVERAGE;
                break;
            }
        }
    }
    free(seen);
    return rc;
}

namespace {
struct FpOpWords {
    int32_t op;
    float   alpha, scale, softcap;
    int32_t exact_all, dk, dv, version, mean_correction;
    int32_t reserved[7];
};
} // namespace

static_assert(sizeof(FpOpWords) == 64, "op params must be 16 int32 words");
static_assert(sizeof(struct ggml_flashprefill_op_params) == 64, "op params must fit GGML_MAX_OP_PARAMS");

int32_t ggml_flashprefill_op_params_pack(
        const struct ggml_flashprefill_op_params * p, int32_t out16[16]) {
    if (!p || !out16) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (p->version != GGML_FLASHPREFILL_VERSION) {
        return GGML_FLASHPREFILL_ERR_BAD_VERSION;
    }
    if (p->op != GGML_FLASHPREFILL_OP_POOL &&
        p->op != GGML_FLASHPREFILL_OP_SELECT &&
        p->op != GGML_FLASHPREFILL_OP_ATTN) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (p->exact_all != 0 && p->exact_all != 1) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (p->mean_correction != 0 && p->mean_correction != 1) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    memcpy(out16, p, 64);
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_op_params_unpack(
        const int32_t in16[16], struct ggml_flashprefill_op_params * p) {
    if (!in16 || !p) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    memcpy(p, in16, 64);
    if (p->version != GGML_FLASHPREFILL_VERSION) {
        return GGML_FLASHPREFILL_ERR_BAD_VERSION;
    }
    if (p->op != GGML_FLASHPREFILL_OP_POOL &&
        p->op != GGML_FLASHPREFILL_OP_SELECT &&
        p->op != GGML_FLASHPREFILL_OP_ATTN) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (p->exact_all != 0 && p->exact_all != 1) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (p->mean_correction != 0 && p->mean_correction != 1) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    for (int i = 0; i < 7; i++) {
        if (p->reserved[i] != 0) {
            return GGML_FLASHPREFILL_ERR_BAD_LAYOUT;
        }
    }
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_ref_means(
        const float * K, int64_t stride_k,
        const float * V, int64_t stride_v,
        const int32_t * idx, int64_t n,
        int32_t dk, int32_t dv,
        float * kbar, float * vbar) {
    if (!K || !V || !idx || !kbar || !vbar) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (n < 1 || dk < 1 || dv < 1 || stride_k < dk || stride_v < dv) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    double * acc = (double *)malloc((size_t)((int64_t)dk + (int64_t)dv) * sizeof(double));
    if (!acc) {
        return GGML_FLASHPREFILL_ERR_OVERFLOW;
    }
    for (int64_t d = 0; d < (int64_t)dk + (int64_t)dv; d++) acc[d] = 0.0;
    int32_t rc = GGML_FLASHPREFILL_OK;
    for (int64_t i = 0; i < n && rc == GGML_FLASHPREFILL_OK; i++) {
        if (idx[i] < 0) {
            rc = GGML_FLASHPREFILL_ERR_BAD_RANGE;
            break;
        }
        const float * kr = K + (int64_t)idx[i] * stride_k;
        const float * vr = V + (int64_t)idx[i] * stride_v;
        for (int32_t d = 0; d < dk; d++) {
            if (!fp_finite_f(kr[d])) {
                rc = GGML_FLASHPREFILL_ERR_BAD_INPUT;
                break;
            }
            acc[d] += (double)kr[d];
        }
        for (int32_t d = 0; d < dv && rc == GGML_FLASHPREFILL_OK; d++) {
            if (!fp_finite_f(vr[d])) {
                rc = GGML_FLASHPREFILL_ERR_BAD_INPUT;
                break;
            }
            acc[dk + d] += (double)vr[d];
        }
    }
    if (rc == GGML_FLASHPREFILL_OK) {
        for (int32_t d = 0; d < dk; d++) kbar[d] = (float)(acc[d] / (double)n);
        for (int32_t d = 0; d < dv; d++) vbar[d] = (float)(acc[dk + d] / (double)n);
    }
    free(acc);
    return rc;
}

int32_t ggml_flashprefill_ref_select(
        const float * qpair, int64_t stride_qp, int32_t n_rows,
        const float * kbar, int64_t stride_kb,
        const int32_t * cand_frag, const int32_t * cand_mandatory,
        const int32_t * cand_full, const int32_t * pair_valid, int32_t n_cand,
        int32_t dk, float scale, float softcap, float alpha, int32_t exact_all,
        int32_t cap_exact, int32_t cap_proxy,
        int32_t * out_exact, int32_t * n_exact_out,
        int32_t * out_proxy, int32_t * n_proxy_out) {
    if (!qpair || !kbar || !cand_frag || !cand_mandatory || !cand_full || !pair_valid ||
        !out_exact || !n_exact_out || !out_proxy || !n_proxy_out) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (n_rows < 1 || n_cand < 1 || dk < 1 ||
        stride_qp < dk || stride_kb < dk ||
        cap_exact < 0 || cap_proxy < 0) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (!fp_finite_f(alpha) || !(alpha > 0.0f) || !(alpha <= 1.0f) ||
        !fp_finite_f(scale) || !fp_finite_f(softcap)) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (exact_all != 0 && exact_all != 1) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    for (int32_t j = 0; j < n_cand; j++) {
        if (cand_frag[j] < 0) return GGML_FLASHPREFILL_ERR_BAD_RANGE;
        if ((cand_mandatory[j] != 0 && cand_mandatory[j] != 1) ||
            (cand_full[j] != 0 && cand_full[j] != 1)) {
            return GGML_FLASHPREFILL_ERR_BAD_ARG;
        }
    }
    // All-zero mask is legitimate (e.g. all-mandatory partial rows with no
    // legal mean): every candidate classifies exact below, capacity still
    // checked. Only the 0/1 domain is enforced here.
    for (int64_t r = 0; r < (int64_t)n_rows; r++) {
        for (int64_t j = 0; j < (int64_t)n_cand; j++) {
            const int32_t mv = pair_valid[r * (int64_t)n_cand + j];
            if (mv != 0 && mv != 1) {
                return GGML_FLASHPREFILL_ERR_BAD_ARG;
            }
        }
    }
    double * ssum = (double *)malloc((size_t)n_cand * sizeof(double));
    int8_t * keep = nullptr;
    int8_t * has = nullptr;
    if (ssum) {
        keep = (int8_t *)malloc((size_t)n_cand);
    }
    if (keep) {
        has = (int8_t *)malloc((size_t)n_cand);
    }
    if (!ssum || !keep || !has) {
        free(ssum);
        free(keep);
        free(has);
        return GGML_FLASHPREFILL_ERR_OVERFLOW;
    }
    // Pass 1: global tile max over scored pairs only: full candidates' valid
    // pairs. Partial columns and invalid pairs are never read, so pollutant
    // values there cannot move M.
    double m = -INFINITY;
    int32_t rc = GGML_FLASHPREFILL_OK;
    const double cap = (double)softcap;
    for (int32_t r = 0; r < n_rows && rc == GGML_FLASHPREFILL_OK; r++) {
        for (int32_t j = 0; j < n_cand && rc == GGML_FLASHPREFILL_OK; j++) {
            if (!cand_full[j]) continue;
            if (!pair_valid[(int64_t)r * (int64_t)n_cand + j]) continue;
            const float * qr = qpair + ((int64_t)r * (int64_t)n_cand + j) * stride_qp;
            const float * kb = kbar + (int64_t)j * stride_kb;
            double dot = 0.0;
            for (int32_t d = 0; d < dk; d++) {
                if (!fp_finite_f(qr[d]) || !fp_finite_f(kb[d])) {
                    rc = GGML_FLASHPREFILL_ERR_BAD_INPUT;
                    break;
                }
                dot += (double)qr[d] * (double)kb[d];
            }
            if (rc) break;
            double z = ggml_flashprefill_softcap_apply((double)scale * dot, cap);
            if (!fp_finite_d(z)) {
                rc = GGML_FLASHPREFILL_ERR_BAD_INPUT;
                break;
            }
            if (z > m) m = z;
        }
    }
    // Pass 2: per-fragment energies over scored pairs only. Partial columns
    // and invalid pairs are never read, so pollutants cannot move Smax.
    double smax = 0.0;
    if (rc == GGML_FLASHPREFILL_OK) {
        for (int32_t j = 0; j < n_cand && rc == GGML_FLASHPREFILL_OK; j++) {
            const float * kb = kbar + (int64_t)j * stride_kb;
            double s = 0.0;
            has[j] = 0;
            if (!cand_full[j]) {
                // Partial candidates are mandatory-exact and excluded from
                // energy entirely: never read, never scored.
                ssum[j] = 0.0;
                continue;
            }
            for (int32_t r = 0; r < n_rows; r++) {
                if (!pair_valid[(int64_t)r * (int64_t)n_cand + j]) continue;
                has[j] = 1;
                const float * qr = qpair + ((int64_t)r * (int64_t)n_cand + j) * stride_qp;
                double dot = 0.0;
                for (int32_t d = 0; d < dk; d++) dot += (double)qr[d] * (double)kb[d];
                double z = ggml_flashprefill_softcap_apply((double)scale * dot, cap);
                s += exp(z - m);
            }
            if (has[j]) {
                if (!fp_finite_d(s)) {
                    rc = GGML_FLASHPREFILL_ERR_BAD_INPUT;
                    break;
                }
                if (s > smax) smax = s;
            }
            ssum[j] = s;
        }
    }
    int32_t n_exact = 0, n_proxy = 0;
    if (rc == GGML_FLASHPREFILL_OK) {
        const double thr = (double)alpha * smax;
        for (int32_t j = 0; j < n_cand; j++) {
            keep[j] = (has[j] && ssum[j] >= thr) ? 1 : 0;
            bool exact = !has[j] || keep[j] || cand_mandatory[j] || !cand_full[j] || exact_all;
            if (exact) {
                n_exact++;
            } else {
                n_proxy++;
            }
        }
        if (n_exact > cap_exact || n_proxy > cap_proxy) {
            rc = GGML_FLASHPREFILL_ERR_OVER_CAPACITY;
        } else {
            int32_t ie = 0, ip = 0;
            for (int32_t j = 0; j < n_cand; j++) {
                bool exact = !has[j] || keep[j] || cand_mandatory[j] || !cand_full[j] || exact_all;
                if (exact) {
                    out_exact[ie++] = cand_frag[j];
                } else {
                    out_proxy[ip++] = cand_frag[j];
                }
            }
            *n_exact_out = n_exact;
            *n_proxy_out = n_proxy;
        }
    }
    free(ssum);
    free(keep);
    free(has);
    return rc;
}

double ggml_flashprefill_softcap_apply(double x, double cap) {
    if (!(cap > 0.0)) {
        return x;
    }
    return cap * tanh(x / cap);
}

int32_t ggml_flashprefill_mlo_init(struct ggml_flashprefill_mlo_state * s, float * o, int32_t dv) {
    if (!s || !o || dv < 1) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    s->m = -INFINITY;
    s->l = 0.0;
    s->dv = dv;
    for (int32_t i = 0; i < dv; i++) o[i] = 0.0f;
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_mlo_add(
        struct ggml_flashprefill_mlo_state * s, float * o,
        double logit, const float * v, int64_t count) {
    if (!s || !o || !v || s->dv < 1) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (!fp_finite_d(logit) || count < 1) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    for (int32_t i = 0; i < s->dv; i++) {
        if (!fp_finite_f(v[i])) {
            return GGML_FLASHPREFILL_ERR_BAD_INPUT;
        }
    }
    const double z = logit + log((double)count);
    if (!fp_finite_d(z)) {
        return GGML_FLASHPREFILL_ERR_BAD_INPUT;
    }
    if (s->l == 0.0) {
        s->m = z;
        s->l = 1.0;
        for (int32_t i = 0; i < s->dv; i++) o[i] = v[i];
        return GGML_FLASHPREFILL_OK;
    }
    if (!fp_finite_d(s->m) || !fp_finite_d(s->l) || !(s->l > 0.0)) {
        return GGML_FLASHPREFILL_ERR_BAD_INPUT;
    }
    const double m_new = s->m > z ? s->m : z;
    const double a = exp(s->m - m_new);
    const double b = exp(z - m_new);
    for (int32_t i = 0; i < s->dv; i++) {
        double ov = a * (double)o[i] + b * (double)v[i];
        if (!fp_finite_d(ov)) {
            return GGML_FLASHPREFILL_ERR_BAD_INPUT;
        }
        o[i] = (float)ov;
    }
    const double l_new = a * s->l + b;
    if (!fp_finite_d(l_new) || !(l_new > 0.0)) {
        return GGML_FLASHPREFILL_ERR_BAD_INPUT;
    }
    s->m = m_new;
    s->l = l_new;
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_mlo_add_sink(
        struct ggml_flashprefill_mlo_state * s, float * o, double sink_logit) {
    if (!s || !o || s->dv < 1) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (!fp_finite_d(sink_logit)) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (s->l == 0.0) {
        s->m = sink_logit;
        s->l = 1.0;
        return GGML_FLASHPREFILL_OK;
    }
    if (!fp_finite_d(s->m) || !fp_finite_d(s->l) || !(s->l > 0.0)) {
        return GGML_FLASHPREFILL_ERR_BAD_INPUT;
    }
    const double m_new = s->m > sink_logit ? s->m : sink_logit;
    const double a = exp(s->m - m_new);
    const double b = exp(sink_logit - m_new);
    for (int32_t i = 0; i < s->dv; i++) {
        double ov = a * (double)o[i];
        if (!fp_finite_d(ov)) {
            return GGML_FLASHPREFILL_ERR_BAD_INPUT;
        }
        o[i] = (float)ov;
    }
    const double l_new = a * s->l + b;
    if (!fp_finite_d(l_new) || !(l_new > 0.0)) {
        return GGML_FLASHPREFILL_ERR_BAD_INPUT;
    }
    s->m = m_new;
    s->l = l_new;
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_mlo_merge(
        struct ggml_flashprefill_mlo_state * sa, const float * oa,
        const struct ggml_flashprefill_mlo_state * sb, const float * ob,
        struct ggml_flashprefill_mlo_state * s_out, float * o_out) {
    if (!sa || !oa || !sb || !ob || !s_out || !o_out) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (sa->dv < 1 || sa->dv != sb->dv) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    const int32_t dv = sa->dv;
    const bool a_empty = (sa->l == 0.0);
    const bool b_empty = (sb->l == 0.0);
    if (a_empty && b_empty) {
        s_out->m = -INFINITY;
        s_out->l = 0.0;
        s_out->dv = dv;
        for (int32_t i = 0; i < dv; i++) o_out[i] = 0.0f;
        return GGML_FLASHPREFILL_OK;
    }
    if (a_empty) {
        s_out->m = sb->m;
        s_out->l = sb->l;
        s_out->dv = dv;
        for (int32_t i = 0; i < dv; i++) {
            if (!fp_finite_f(ob[i])) return GGML_FLASHPREFILL_ERR_BAD_INPUT;
            o_out[i] = ob[i];
        }
        return GGML_FLASHPREFILL_OK;
    }
    if (b_empty) {
        s_out->m = sa->m;
        s_out->l = sa->l;
        s_out->dv = dv;
        for (int32_t i = 0; i < dv; i++) {
            if (!fp_finite_f(oa[i])) return GGML_FLASHPREFILL_ERR_BAD_INPUT;
            o_out[i] = oa[i];
        }
        return GGML_FLASHPREFILL_OK;
    }
    if (!fp_finite_d(sa->m) || !fp_finite_d(sb->m) ||
        !fp_finite_d(sa->l) || !fp_finite_d(sb->l) ||
        !(sa->l > 0.0) || !(sb->l > 0.0)) {
        return GGML_FLASHPREFILL_ERR_BAD_INPUT;
    }
    const double m_new = sa->m > sb->m ? sa->m : sb->m;
    const double a = exp(sa->m - m_new);
    const double b = exp(sb->m - m_new);
    for (int32_t i = 0; i < dv; i++) {
        if (!fp_finite_f(oa[i]) || !fp_finite_f(ob[i])) {
            return GGML_FLASHPREFILL_ERR_BAD_INPUT;
        }
        double ov = a * (double)oa[i] + b * (double)ob[i];
        if (!fp_finite_d(ov)) {
            return GGML_FLASHPREFILL_ERR_BAD_INPUT;
        }
        o_out[i] = (float)ov;
    }
    const double l_new = a * sa->l + b * sb->l;
    if (!fp_finite_d(l_new) || !(l_new > 0.0)) {
        return GGML_FLASHPREFILL_ERR_BAD_INPUT;
    }
    s_out->m = m_new;
    s_out->l = l_new;
    s_out->dv = dv;
    return GGML_FLASHPREFILL_OK;
}

int32_t ggml_flashprefill_mlo_finalize(
        const struct ggml_flashprefill_mlo_state * s, const float * o, float * out) {
    if (!s || !o || !out || s->dv < 1) {
        return GGML_FLASHPREFILL_ERR_BAD_ARG;
    }
    if (s->l == 0.0) {
        for (int32_t i = 0; i < s->dv; i++) out[i] = 0.0f;
        return GGML_FLASHPREFILL_OK;
    }
    if (!fp_finite_d(s->m) || !fp_finite_d(s->l) || !(s->l > 0.0)) {
        return GGML_FLASHPREFILL_ERR_BAD_INPUT;
    }
    for (int32_t i = 0; i < s->dv; i++) {
        if (!fp_finite_f(o[i])) {
            return GGML_FLASHPREFILL_ERR_BAD_INPUT;
        }
        double v = (double)o[i] / s->l;
        if (!fp_finite_d(v)) {
            return GGML_FLASHPREFILL_ERR_BAD_INPUT;
        }
        out[i] = (float)v;
    }
    return GGML_FLASHPREFILL_OK;
}

} // extern "C"
