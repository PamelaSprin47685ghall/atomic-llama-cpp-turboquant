// Routing contract tests for FlashPrefill V2 context integration.
//
// Policy-helper checks over the frozen PolicyCore contract
// (src/llama-flashprefill.h, namespace llama_flashprefill) plus observable
// real-API checks on a model-free test-only context (no weights, backend,
// GPU, or threads; this file never executes inference):
//
//   - role gating (only PREFILL / REROT_TEACHER_FORCED ever sparse-eligible)
//   - OFF / short-context / unsupported-backend dense precedence
//   - call-scope vs logical-prompt dense-tail edges (chunked-prefill safety)
//   - GQA packing exactness when BM does not divide G
//   - version/count validation edges + fingerprint determinism
//   - real decode_with_flashprefill count-mismatch rejection (-1) and legacy
//     empty-batch early-out (-1) on a test-only context
//
// Deliberately NOT tested here: source-row slice plumbing and the corrupt/
// missing-map failure code (-3). Those live in src/llama-context.{h,cpp}
// (StatePolicy owner); a local mirror would only re-pin a copy, and the
// previous mirror pinned the old silent-dense behavior.
//
// Not wired into tests/CMakeLists.txt yet (later owner).

#include "../src/llama-flashprefill.h"
#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "../src/llama-graph.h"
#include "../src/llama-flashprefill-pack.h"
#include "../src/llama-flashprefill-fixture.h"
#include "../src/llama-kv-cells.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <map>
#include <tuple>

namespace {

static int g_failures = 0;

#define TEST_ASSERT(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

using llama_flashprefill::pack_index;
using llama_flashprefill::packed_layout_checked;
using llama_flashprefill::packed_row_in_dense_tail;
using llama_flashprefill::route_for_role;
using llama_flashprefill::route_row_call;
using llama_flashprefill::route_row_logical;
using llama_flashprefill::unpack_subhead;
using llama_flashprefill::unpack_token;

static llama_flashprefill_config make_auto(int32_t tail_scope) {
    llama_flashprefill_config cfg = llama_flashprefill_default_config();
    cfg.mode       = LLAMA_FLASHPREFILL_MODE_AUTO;
    cfg.tail_scope = tail_scope;
    return cfg;
}

static llama_flashprefill_row make_row(
        int32_t role, int32_t seq_id, int32_t logical_pos,
        int32_t begin, int32_t end, bool known) {
    llama_flashprefill_row row;
    std::memset(&row, 0, sizeof(row));
    row.version       = LLAMA_FLASHPREFILL_ROW_VERSION;
    row.struct_size   = (uint32_t) sizeof(row);
    row.role          = role;
    row.seq_id        = seq_id;
    row.reader_id     = LLAMA_FLASHPREFILL_READER_NONE;
    row.logical_pos   = logical_pos;
    row.prefill_begin = begin;
    row.prefill_end   = end;
    row.prefill_known = known;
    return row;
}

// Test-only model following the established tests/ pattern
// (test-rerot-runtime.cpp, test-rerot-recurrent.cpp): bare weights, no-op
// loader/graph overrides. Only early validation/early-out paths run against
// it below; no inference is executed.
struct test_stub_model : public llama_model {
    test_stub_model() : llama_model(llama_model_default_params()) {}
    void load_stats(llama_model_loader &) override {}
    void load_hparams(llama_model_loader &) override {}
    void load_vocab(llama_model_loader &) override {}
    bool load_tensors(llama_model_loader &) override { return true; }
    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override {
        return nullptr;
    }
};

static void test_defaults(void) {
    const llama_flashprefill_config cfg = llama_flashprefill_default_config();
    TEST_ASSERT(cfg.version   == LLAMA_FLASHPREFILL_CONFIG_VERSION);
    TEST_ASSERT(cfg.struct_size == sizeof(cfg));
    TEST_ASSERT(cfg.mode      == LLAMA_FLASHPREFILL_MODE_OFF);
    TEST_ASSERT(cfg.tail_scope == LLAMA_FLASHPREFILL_TAIL_LOGICAL_PROMPT);
    TEST_ASSERT(cfg.alpha     == LLAMA_FLASHPREFILL_DEFAULT_ALPHA);
    TEST_ASSERT(cfg.block_q   == 128u);
    TEST_ASSERT(cfg.block_k   == 128u);
    TEST_ASSERT(cfg.sink_blocks  == 2u);
    TEST_ASSERT(cfg.window_blocks == 4u);
    TEST_ASSERT(cfg.dense_tail_tiles == 8u);
    TEST_ASSERT(cfg.min_kv    == 1024u);
    TEST_ASSERT(cfg.full_attn_layers == 0u);
    TEST_ASSERT(cfg.mean_correction);
    TEST_ASSERT(!cfg.exact_all);
    TEST_ASSERT(llama_flashprefill_validate_config(&cfg) == LLAMA_FLASHPREFILL_OK);
    TEST_ASSERT(!llama_flashprefill_is_enabled(&cfg));
}

static void test_role_gate(void) {
    const llama_flashprefill_config off = llama_flashprefill_default_config();
    TEST_ASSERT(route_for_role(&off, LLAMA_FLASHPREFILL_ROLE_PREFILL, 4096, true) ==
            LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF);

    llama_flashprefill_config cfg = make_auto(LLAMA_FLASHPREFILL_TAIL_CALL);
    TEST_ASSERT(llama_flashprefill_validate_config(&cfg) == LLAMA_FLASHPREFILL_OK);
    TEST_ASSERT(llama_flashprefill_is_enabled(&cfg));

    // Only these two roles may leave the dense path.
    TEST_ASSERT(!llama_flashprefill_route_is_dense(
            route_for_role(&cfg, LLAMA_FLASHPREFILL_ROLE_PREFILL, 4096, true)));
    TEST_ASSERT(!llama_flashprefill_route_is_dense(
            route_for_role(&cfg, LLAMA_FLASHPREFILL_ROLE_REROT_TEACHER_FORCED, 4096, true)));

    const int32_t dense_roles[] = {
        LLAMA_FLASHPREFILL_ROLE_UNKNOWN,
        LLAMA_FLASHPREFILL_ROLE_DECODE,
        LLAMA_FLASHPREFILL_ROLE_MTP_DRAFT,
        LLAMA_FLASHPREFILL_ROLE_MTP_VERIFY,
        LLAMA_FLASHPREFILL_ROLE_SPECULATIVE_REPLAY,
        LLAMA_FLASHPREFILL_ROLE_REROT_FRONTIER,
        LLAMA_FLASHPREFILL_ROLE_EMBEDDING,
        LLAMA_FLASHPREFILL_ROLE_RERANK,
        LLAMA_FLASHPREFILL_ROLE_MULTIMODAL,
    };
    for (int32_t role : dense_roles) {
        TEST_ASSERT(route_for_role(&cfg, role, 4096, true) == LLAMA_FLASHPREFILL_ROUTE_DENSE_ROLE);
    }

    // Precedence edges: unsupported backend, short resident-visible context.
    TEST_ASSERT(route_for_role(&cfg, LLAMA_FLASHPREFILL_ROLE_PREFILL, 4096, false) ==
            LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED);
    TEST_ASSERT(route_for_role(&cfg, LLAMA_FLASHPREFILL_ROLE_PREFILL, 100, true) ==
            LLAMA_FLASHPREFILL_ROUTE_DENSE_SHORT_CONTEXT);

    // Debug gate: exact-all for eligible roles.
    cfg.exact_all = true;
    TEST_ASSERT(route_for_role(&cfg, LLAMA_FLASHPREFILL_ROLE_PREFILL, 4096, true) ==
            LLAMA_FLASHPREFILL_ROUTE_EXACT_ALL);
    TEST_ASSERT(route_for_role(&cfg, LLAMA_FLASHPREFILL_ROLE_DECODE, 4096, true) ==
            LLAMA_FLASHPREFILL_ROUTE_DENSE_ROLE);
}

static void test_call_tail_edges(void) {
    llama_flashprefill_config cfg = make_auto(LLAMA_FLASHPREFILL_TAIL_CALL);
    // block_q = 128, dense_tail_tiles = 8: 2048 packed rows = 16 tiles,
    // tiles [0, 8) sparse-eligible, tiles [8, 16) dense.
    const llama_flashprefill_row row = make_row(
            LLAMA_FLASHPREFILL_ROLE_PREFILL, 0, LLAMA_FLASHPREFILL_POS_UNKNOWN, 0, 0, false);
    TEST_ASSERT(route_row_call(&cfg, &row, 4096, true, 0, 2048) ==
            LLAMA_FLASHPREFILL_ROUTE_SPARSE);
    TEST_ASSERT(route_row_call(&cfg, &row, 4096, true, 7 * 128 + 127, 2048) ==
            LLAMA_FLASHPREFILL_ROUTE_SPARSE);
    TEST_ASSERT(route_row_call(&cfg, &row, 4096, true, 8 * 128, 2048) ==
            LLAMA_FLASHPREFILL_ROUTE_DENSE_TAIL);
    TEST_ASSERT(route_row_call(&cfg, &row, 4096, true, 2047, 2048) ==
            LLAMA_FLASHPREFILL_ROUTE_DENSE_TAIL);

    // Whole small call covered by the tail: dense, not a silent no-op later.
    TEST_ASSERT(route_row_call(&cfg, &row, 4096, true, 0, 256) ==
            LLAMA_FLASHPREFILL_ROUTE_DENSE_TAIL);

    // Tail predicate edges: never sparse on doubt.
    TEST_ASSERT(!packed_row_in_dense_tail(5, 100, 128, 0));     // no tail
    TEST_ASSERT(!packed_row_in_dense_tail(5, 0, 128, 8));       // empty call
    TEST_ASSERT(!packed_row_in_dense_tail(5000, 2048, 128, 8)); // out of range
    TEST_ASSERT(!packed_row_in_dense_tail(0, 2048, 128, 8));    // head tile
    TEST_ASSERT(packed_row_in_dense_tail(2047, 2048, 128, 8));  // tail tile
}

static void test_gqa_packing(void) {
    // BM need not divide G: tile 128 starts mid-group for G = 3.
    TEST_ASSERT(pack_index(42u, 2u, 3u) == 128u);
    TEST_ASSERT(unpack_token(128u, 3u) == 42u);
    TEST_ASSERT(unpack_subhead(128u, 3u) == 2u);
    for (uint32_t tok = 0; tok < 65; ++tok) {
        for (uint32_t sub = 0; sub < 3; ++sub) {
            const uint32_t p = pack_index(tok, sub, 3u);
            TEST_ASSERT(unpack_token(p, 3u) == tok);
            TEST_ASSERT(unpack_subhead(p, 3u) == sub);
        }
    }

    uint32_t total = 0, tiles = 0;
    TEST_ASSERT(packed_layout_checked(100u, 3u, 128u, &total, &tiles) == LLAMA_FLASHPREFILL_OK);
    TEST_ASSERT(total == 300u);
    TEST_ASSERT(tiles == 3u); // ceil(300 / 128)
    TEST_ASSERT(packed_layout_checked(100u, 0u, 128u, &total, &tiles) ==
            LLAMA_FLASHPREFILL_ERR_COUNT);
    TEST_ASSERT(packed_layout_checked(100u, 3u, 0u, &total, &tiles) ==
            LLAMA_FLASHPREFILL_ERR_COUNT);
    TEST_ASSERT(packed_layout_checked(0xFFFFFFFFu, 2u, 128u, &total, &tiles) ==
            LLAMA_FLASHPREFILL_ERR_OVERFLOW);
}

static void test_logical_stability(void) {
    llama_flashprefill_config cfg = make_auto(LLAMA_FLASHPREFILL_TAIL_LOGICAL_PROMPT);
    // Frozen logical prefill [0, 2048), GQA = 1: 16 packed tiles, tail 8.
    const llama_flashprefill_row head = make_row(
            LLAMA_FLASHPREFILL_ROLE_PREFILL, 0, 100, 0, 2048, true);
    const llama_flashprefill_row tail = make_row(
            LLAMA_FLASHPREFILL_ROLE_PREFILL, 0, 2000, 0, 2048, true);
    TEST_ASSERT(route_row_logical(&cfg, &head, 4096, true, 100, 2048) ==
            LLAMA_FLASHPREFILL_ROUTE_SPARSE);
    TEST_ASSERT(route_row_logical(&cfg, &tail, 4096, true, 2000, 2048) ==
            LLAMA_FLASHPREFILL_ROUTE_DENSE_TAIL);

    // Retry/chunking safety: the logical route never depends on the call
    // slicing, so a halved retry batch keeps identical routing. The same row
    // under call scope would flip dense for a 256-row retry (tail covers it).
    llama_flashprefill_config call = make_auto(LLAMA_FLASHPREFILL_TAIL_CALL);
    const llama_flashprefill_row bare = make_row(
            LLAMA_FLASHPREFILL_ROLE_PREFILL, 0, LLAMA_FLASHPREFILL_POS_UNKNOWN, 0, 0, false);
    TEST_ASSERT(route_row_call(&call, &bare, 4096, true, 100, 256) ==
            LLAMA_FLASHPREFILL_ROUTE_DENSE_TAIL);
    TEST_ASSERT(route_row_logical(&cfg, &head, 4096, true, 100, 2048) ==
            LLAMA_FLASHPREFILL_ROUTE_SPARSE);

    // Unknown boundary: logical scope must refuse, never guess.
    const llama_flashprefill_row unknown = make_row(
            LLAMA_FLASHPREFILL_ROLE_PREFILL, 0, 100, 0, 0, false);
    TEST_ASSERT(route_row_logical(&cfg, &unknown, 4096, true, 100, 2048) ==
            LLAMA_FLASHPREFILL_ROUTE_DENSE_UNKNOWN_BOUNDARY);
}

static void test_real_api_mismatch(void) {
    // Observable real-API checks on a test-only context built from the
    // stub model above: no weights, default cparams (normalized to OFF),
    // no backends, no sched, no memory. Only early validation/early-out
    // paths run here.
    test_stub_model model;
    llama_cparams cp{};
    llama_context ctx(model, cp, true);

    // OFF policy carries no isolation identity.
    TEST_ASSERT(ctx.flashprefill_context_serial() == 0);
    TEST_ASSERT(ctx.flashprefill_policy_fingerprint() == ctx.flashprefill_policy_fingerprint());

    // n_rows != n_tokens is rejected (-1) before any state mutates or the
    // model/memory is touched (the row contents below are never read).
    llama_token toks[2] = {0, 0};
    llama_batch batch{};
    batch.n_tokens = 2;
    batch.token    = toks;

    llama_flashprefill_row rows[3];
    for (int i = 0; i < 3; ++i) {
        rows[i] = make_row(LLAMA_FLASHPREFILL_ROLE_PREFILL, 0, i, 0, 3, true);
    }
    llama_flashprefill_exec exec{};
    exec.version     = LLAMA_FLASHPREFILL_EXEC_VERSION;
    exec.struct_size = (uint32_t) sizeof(exec);
    exec.n_rows      = 3;
    exec.rows        = rows;
    TEST_ASSERT(ctx.decode_with_flashprefill(batch, &exec) == -1);

    // Legacy dense path with an empty batch early-outs (-1) via the
    // no-memory encode fallback, still without touching weights/backends.
    llama_token dummy = 0;
    llama_batch empty{};
    empty.n_tokens = 0;
    empty.token    = &dummy;
    TEST_ASSERT(ctx.decode(empty) == -1);
}

static void test_validation_edges(void) {
    llama_flashprefill_row row = make_row(
            LLAMA_FLASHPREFILL_ROLE_PREFILL, 0, 0, 0, 4, true);
    TEST_ASSERT(llama_flashprefill_validate_row(&row) == LLAMA_FLASHPREFILL_OK);

    llama_flashprefill_row bad_role = row;
    bad_role.role = 99;
    TEST_ASSERT(llama_flashprefill_validate_row(&bad_role) == LLAMA_FLASHPREFILL_ERR_ROLE);

    // Inverted interval is rejected (empty [x, x) is accepted by the row
    // validator; the context attach layer additionally refuses empty
    // intervals because no dense tail can be scoped from them).
    llama_flashprefill_row bad_interval = row;
    bad_interval.prefill_begin = 4;
    bad_interval.prefill_end   = 2;
    TEST_ASSERT(llama_flashprefill_validate_row(&bad_interval) == LLAMA_FLASHPREFILL_ERR_INTERVAL);

    llama_flashprefill_row bad_version = row;
    bad_version.version = 0;
    TEST_ASSERT(llama_flashprefill_validate_row(&bad_version) == LLAMA_FLASHPREFILL_ERR_VERSION);

    const llama_flashprefill_row rows[2] = {row, row};
    llama_flashprefill_exec exec;
    std::memset(&exec, 0, sizeof(exec));
    exec.version     = LLAMA_FLASHPREFILL_EXEC_VERSION;
    exec.struct_size = (uint32_t) sizeof(exec);
    exec.n_rows      = 2;
    exec.rows        = rows;
    TEST_ASSERT(llama_flashprefill_validate_exec(&exec) == LLAMA_FLASHPREFILL_OK);

    llama_flashprefill_exec bad_exec_version = exec;
    bad_exec_version.version = 0;
    TEST_ASSERT(llama_flashprefill_validate_exec(&bad_exec_version) ==
            LLAMA_FLASHPREFILL_ERR_VERSION);

    llama_flashprefill_exec null_rows = exec;
    null_rows.rows = nullptr;
    TEST_ASSERT(llama_flashprefill_validate_exec(&null_rows) == LLAMA_FLASHPREFILL_ERR_NULL);

    llama_flashprefill_exec bad_size = exec;
    bad_size.struct_size = 0;
    TEST_ASSERT(llama_flashprefill_validate_exec(&bad_size) == LLAMA_FLASHPREFILL_ERR_SIZE);

    // Context layer contract (checked in decode_with_flashprefill, before any
    // state mutates): n_rows must equal batch.n_tokens exactly. Empty views
    // (n_rows == 0, rows == NULL) validate as dense-by-construction.
    llama_flashprefill_exec empty;
    std::memset(&empty, 0, sizeof(empty));
    empty.version     = LLAMA_FLASHPREFILL_EXEC_VERSION;
    empty.struct_size = (uint32_t) sizeof(empty);
    TEST_ASSERT(llama_flashprefill_validate_exec(&empty) == LLAMA_FLASHPREFILL_OK);
}

static void test_fingerprint_and_names(void) {
    const llama_flashprefill_config cfg = llama_flashprefill_default_config();
    uint64_t a = 0, b = 0;
    TEST_ASSERT(llama_flashprefill_fingerprint(&cfg, "model", nullptr, &a) ==
            LLAMA_FLASHPREFILL_OK);
    TEST_ASSERT(llama_flashprefill_fingerprint(&cfg, "model", nullptr, &b) ==
            LLAMA_FLASHPREFILL_OK);
    TEST_ASSERT(a == b); // deterministic

    llama_flashprefill_config changed = cfg;
    changed.alpha = 0.2f;
    uint64_t c = 0;
    TEST_ASSERT(llama_flashprefill_fingerprint(&changed, "model", nullptr, &c) ==
            LLAMA_FLASHPREFILL_OK);
    TEST_ASSERT(c != a); // every approximation field is covered

    uint64_t d = 0;
    TEST_ASSERT(llama_flashprefill_fingerprint(&cfg, "other-model", nullptr, &d) ==
            LLAMA_FLASHPREFILL_OK);
    TEST_ASSERT(d != a); // model identity domain separates

    uint64_t e = 0, f = 0;
    TEST_ASSERT(llama_flashprefill_fingerprint(&cfg, "model", nullptr, &e) ==
            LLAMA_FLASHPREFILL_OK);
    TEST_ASSERT(llama_flashprefill_fingerprint(&cfg, "model", "", &f) ==
            LLAMA_FLASHPREFILL_OK);
    TEST_ASSERT(e != f); // NULL != "" adapter domain

    TEST_ASSERT(llama_flashprefill_fingerprint(nullptr, "model", nullptr, &a) ==
            LLAMA_FLASHPREFILL_ERR_NULL);
    TEST_ASSERT(llama_flashprefill_fingerprint(&cfg, "model", nullptr, nullptr) ==
            LLAMA_FLASHPREFILL_ERR_NULL);

    TEST_ASSERT(llama_flashprefill_error_name(LLAMA_FLASHPREFILL_OK) != nullptr);
    TEST_ASSERT(llama_flashprefill_mode_name(LLAMA_FLASHPREFILL_MODE_AUTO) != nullptr);
    TEST_ASSERT(llama_flashprefill_role_name(LLAMA_FLASHPREFILL_ROLE_PREFILL) != nullptr);
    TEST_ASSERT(llama_flashprefill_route_name(LLAMA_FLASHPREFILL_ROUTE_SPARSE) != nullptr);

    TEST_ASSERT(!llama_flashprefill_route_is_dense(LLAMA_FLASHPREFILL_ROUTE_SPARSE));
    TEST_ASSERT(!llama_flashprefill_route_is_dense(LLAMA_FLASHPREFILL_ROUTE_EXACT_ALL));
    TEST_ASSERT(llama_flashprefill_route_is_dense(LLAMA_FLASHPREFILL_ROUTE_DENSE_OFF));
    TEST_ASSERT(llama_flashprefill_route_is_dense(LLAMA_FLASHPREFILL_ROUTE_DENSE_ROLE));
    TEST_ASSERT(llama_flashprefill_route_is_dense(LLAMA_FLASHPREFILL_ROUTE_DENSE_UNSUPPORTED));
    TEST_ASSERT(llama_flashprefill_route_is_dense(LLAMA_FLASHPREFILL_ROUTE_DENSE_SHORT_CONTEXT));
    TEST_ASSERT(llama_flashprefill_route_is_dense(LLAMA_FLASHPREFILL_ROUTE_DENSE_TAIL));
    TEST_ASSERT(llama_flashprefill_route_is_dense(LLAMA_FLASHPREFILL_ROUTE_DENSE_UNKNOWN_BOUNDARY));
    TEST_ASSERT(llama_flashprefill_route_is_dense(LLAMA_FLASHPREFILL_ROUTE_DENSE_CAPACITY));
    TEST_ASSERT(!llama_flashprefill_route_is_dense(99)); // unknown: not trusted
}

} // namespace

static void test_real_graph_pack() {
    // Real owner planner AND graph packer, without model weights. Nonzero
    // storage offsets, sparse holes and multiple fragments expose address
    // mistakes that zero-based, one-fragment fixtures cannot detect.
    constexpr int nq = 128, phys_base = 7;
    for (bool split_domains : {false, true}) {
        llama_kv_cells cells;
        cells.resize(512);
        for (int q = 0; q < nq; ++q) {
            if (q == 40 || q == 41) continue;
            cells.pos_set(phys_base + q, q);
            cells.seq_add(phys_base + q, 0);
            if (split_domains) cells.seq_add(phys_base + q, 1);
        }
        std::vector<llama_pos> pos(nq);
        std::vector<int32_t> nseq(nq, 1);
        std::vector<llama_seq_id> seq(nq);
        std::vector<llama_seq_id *> seqptr(nq);
        std::vector<llama_flashprefill_row> rows;
        for (int q = 0; q < nq; ++q) {
            pos[q] = q;
            seq[q] = split_domains ? q % 2 : 0;
            seqptr[q] = &seq[q];
            rows.push_back(make_row(LLAMA_FLASHPREFILL_ROLE_PREFILL, seq[q], q, 0, nq, true));
        }
        llama_ubatch ub{};
        ub.n_tokens = nq; ub.pos = pos.data(); ub.n_seq_id = nseq.data(); ub.seq_id = seqptr.data();
        llama_flashprefill_layout_params params;
        params.block_k = 64;
        params.want_exact_rows = true;
        llama_flashprefill_layout layout;
        std::string error;
        if (!llama_flashprefill_fixture_build_ordinary({cells}, {0, 0}, ub, params, layout, &error)) {
            std::fprintf(stderr, "graph-pack fixture: %s\n", error.c_str());
            TEST_ASSERT(false);
            continue;
        }
        TEST_ASSERT(layout.n_groups() == nq);
        for (uint32_t gqa : {1u, 6u, 8u}) for (uint32_t hkv : {1u, 8u}) {
            for (uint32_t bm : {5u, 64u, 128u}) {
                auto cfg = make_auto(LLAMA_FLASHPREFILL_TAIL_CALL);
                cfg.block_q = bm; cfg.block_k = 64; cfg.exact_all = true;
                llm_fp_pack pack;
                const auto rc = llm_fp_pack_live(layout, rows, cfg, gqa*hkv, hkv, gqa, nq, &pack, &error);
                if (rc != LL_FP_PACK_OK) {
                    std::fprintf(stderr, "graph pack G=%u Hkv=%u BM=%u: %s\n", gqa, hkv, bm, error.c_str());
                    TEST_ASSERT(rc == LL_FP_PACK_OK);
                    continue;
                }
                TEST_ASSERT(pack.qpos.empty()); // ordinary Q is already roped
                TEST_ASSERT(pack.n_groups == nq);
                TEST_ASSERT(pack.cells.size() == nq - 2);
                TEST_ASSERT(pack.rows.size() == nq * gqa * hkv);
                const int64_t tiles_per_domain = ((split_domains ? nq / 2 : nq) * gqa + bm - 1) / bm;
                TEST_ASSERT(pack.n_tiles == tiles_per_domain * (split_domains ? 2 : 1));
                std::map<std::tuple<int32_t,int32_t,int32_t>, std::vector<int32_t>> addressed;
                for (const auto & u : pack.uses) {
                    TEST_ASSERT(u.tile >= 0 && u.tile < pack.n_tiles);
                    TEST_ASSERT(u.q_group == u.src_q);
                    if (u.sub_off < 0 || u.sub_count <= 0 ||
                        uint64_t(u.sub_off + u.sub_count) > pack.cells.size()) {
                        TEST_ASSERT(false);
                        continue;
                    }
                    auto & keys = addressed[{u.src_q, u.tile, u.kv_head}];
                    keys.insert(keys.end(), pack.cells.begin() + u.sub_off,
                                pack.cells.begin() + u.sub_off + u.sub_count);
                }
                for (const auto & r : pack.rows) {
                    const int local_q = split_domains ? r.src_q / 2 : r.src_q;
                    const int domain = split_domains ? r.src_q % 2 : 0;
                    const int64_t tile = domain * tiles_per_domain +
                        (local_q * gqa + (uint32_t) r.q_head % gqa) / bm;
                    TEST_ASSERT(r.tile == tile);
                    TEST_ASSERT(r.kv_head == r.q_head / (int32_t) gqa);
                    std::vector<int32_t> expected;
                    for (int p = 0; p <= r.src_q; ++p) {
                        if (p != 40 && p != 41) expected.push_back(phys_base + p);
                    }
                    // Exact key coverage per actual row, with no duplicates,
                    // missing keys, foreign-domain keys, or resurrected holes.
                    TEST_ASSERT((addressed[{r.src_q, r.tile, r.kv_head}] == expected));
                }
            }
        }
        auto cfg = make_auto(LLAMA_FLASHPREFILL_TAIL_CALL);
        cfg.exact_all = true;
        cfg.block_k = 64;
        llm_fp_pack pack;
        auto bad = layout;
        bad.groups[0].query_index = 1;
        TEST_ASSERT(llm_fp_pack_live(bad, rows, cfg, 48, 8, 6, nq, &pack, &error) == LL_FP_PACK_CORRUPT);
        bad = layout;
        bad.uses[0].group = 1;
        TEST_ASSERT(llm_fp_pack_live(bad, rows, cfg, 48, 8, 6, nq, &pack, &error) == LL_FP_PACK_CORRUPT);
        bad = layout;
        bad.uses[0].sub_off = bad.fragments[bad.uses[0].fragment].token_count;
        TEST_ASSERT(llm_fp_pack_live(bad, rows, cfg, 48, 8, 6, nq, &pack, &error) == LL_FP_PACK_CORRUPT);
        // Phase-bearing graphs still need their Q positions; suppressing
        // ordinary identity uploads must not discard RERoT's phase data.
        auto phased = layout;
        phased.is_rerot = 1;
        TEST_ASSERT(llm_fp_pack_live(phased, rows, cfg, 48, 8, 6, nq, &pack, &error) == LL_FP_PACK_OK);
        TEST_ASSERT(pack.qpos == pos);
    }
}

int main(void) {
    test_real_graph_pack();
    // One shared metadata input must support the actual Nanbeige boundary
    // V policy without disabling detection of genuine per-layer changes.
    llm_graph_fp_key key;
    key.k_type = GGML_TYPE_TURBO4_0;
    key.v_type = GGML_TYPE_Q8_0; // first layer is not the interior format
    key.layer_kv_types = {{GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0},
                          {GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0}, {-1, -1}};
    TEST_ASSERT(key.matches_layer_types(0, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0));
    TEST_ASSERT(key.matches_layer_types(1, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0));
    TEST_ASSERT(!key.matches_layer_types(0, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0));
    TEST_ASSERT(!key.matches_layer_types(1, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0));
    TEST_ASSERT(!key.matches_layer_types(2, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0));
    TEST_ASSERT(!key.matches_layer_types(-1, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0));
    TEST_ASSERT(!key.matches_layer_types(3, GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0));
    auto changed = key;
    TEST_ASSERT(changed == key);
    changed.layer_kv_types[1].second = GGML_TYPE_Q8_0;
    TEST_ASSERT(changed != key);

    test_defaults();
    test_role_gate();
    test_call_tail_edges();
    test_gqa_packing();
    test_logical_stability();
    test_real_api_mismatch();
    test_validation_edges();
    test_fingerprint_and_names();

    if (g_failures == 0) {
        std::printf("test-flashprefill-routing: all assertions passed\n");
        return 0;
    }
    std::printf("test-flashprefill-routing: %d assertion(s) failed\n", g_failures);
    return 1;
}
