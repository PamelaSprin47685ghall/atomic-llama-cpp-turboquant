#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"
#include "llama-nout-state.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>   // unsetenv: the fixture suppresses LLAMA_GDN_REPLAY
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

//
// test-rs-ring-relay
//
// The recurrent rollback ring (n_rs_seq, the "RS ring") crossed with the
// tp5-relay path, driven through the REAL graph builder. No model file, no
// llama_context, no backend device: the test builds a minimal
// llama_memory_recurrent (one layer, `mem_size` cells, CPU buffer type) and a
// real llama_memory_recurrent_context over it, then calls the production
// llm_graph_context::build_rs() and llm_graph_input_rs::set_input() /
// can_reuse().
//
// WHAT IT PROVES
//
//   The bound zero row is a ROW of the state tensor, not a cell index, and it
//   is the row the gathers read:
//
//       bound row = rs_idx[seq] * mem_size + rs_z          (get_rs_zero_row)
//
//   with rs_z a cell in [0, mem_size) and rs_idx[seq] the sequence's rollback
//   depth in [0, n_rs_seq]. The state tensor holds mem_size*(1+n_rs_seq) rows,
//   so the row is always inside it. The zero-row op clears exactly that row,
//   and both state gathers read through the op's output (zero before gather is
//   a graph edge, not a build-order convention).
//
// ARMS (each is a different path, and each asserts the others' observable is
// absent)
//
//   R0  n_rs_seq == 0, relay form, rs_idx == 0.
//       Bound row == rs_z (group 0 is the only group).
//       Catches: an implementation that widens the state tensor without a
//       ring (height must be exactly mem_size, NOT 3*mem_size).
//
//   R1  n_rs_seq == 2, relay form, rs_idx[0] == 2.
//       Bound row == 2*mem_size + rs_z, inside the 3*mem_size-row tensor.
//       Catches: the pre-merge binder value, which wrote the CELL index rs_z
//       into the zero-row tensor. The gathers read group 2, so the cell index
//       names a row nobody reads; the arm asserts the written row, the row
//       s_copy() reads, and that the two are equal.
//       Catches: a group factor of 1 (row == rs_z + ...) or an off-by-one
//       (row == (rs_idx+1)*mem_size + rs_z).
//
//   R2  n_rs_seq == 2, stock (non-relay) form.
//       llama_rsz_rt_armable() must refuse the ring for every rs_z, and its
//       positive control -- the same call with n_rs_seq == 0 -- must arm.
//       The gate argument is identical in both calls, so the refusal is the
//       ring's doing, not the gate's. The stock build also carries no
//       union_rs_zero_row node: rs_z stays a topology key there.
//       Catches: an implementation that lets the stock lane arm on a ring.
//
//   R3  n_rs_seq == 2, relay form, MIXED depths in one ubatch
//       (rs_idx[0] == 2, rs_idx[1] == 0).
//       One graph carries ONE zero row, so the bind must refuse:
//       set_input() sets llm_graph_input_rs::rs_zero_row_refused, the row
//       tensor keeps its previous value, and can_reuse() returns false so the
//       next build is stock. The context counts the refusal by name as
//       c_rsz_rt_refused_mixed_depth (llama-context.cpp, before set_inputs()).
//       Catches: an implementation that asserts instead of refusing (a crash
//       on a legal batch) or that silently writes one sequence's row and
//       computes with it for both.
//
//   SWEEP  mem_size in {1,2,3,4,7,8} x n_rs_seq in {0,1,2}.
//       For every (rs_z, depth) pair: the bound row is
//       depth*mem_size + rs_z, it is inside mem_size*(1+n_rs_seq), the row the
//       gathers read equals the row the op zeroes, and the relay row rule
//       accepts the cell and refuses the first cell past the capacity.
//       Catches: an off-by-one in the group arithmetic -- a missing group
//       factor, a "+1", a head offset, or a capacity bound that uses
//       mem_size instead of mem_size*(1+n_rs_seq).
//
// TWO REAL DEFECTS FIXED HERE (found by the merge survey)
//
//   (a) llm_graph_params::res was nullptr. The llm_graph_context constructor
//       dereferences it immediately (res->get_ctx(), res->get_gf(),
//       res->set_params(); llama-graph.cpp:1891-1896), so the old fixture was
//       a null dereference. The fixture now owns a real llm_graph_result and
//       passes it.
//   (b) The index tensor had no backend buffer. ggml_init(no_alloc=false)
//       gives tensors inline host memory but buffer == nullptr, and
//       union_rsz_bind_index() asserts ggml_backend_buffer_is_host(index->
//       buffer) (llama-graph.cpp:542). The fixture now creates its graph
//       tensors in a no_alloc=true context and allocates them on
//       ggml_backend_cpu_buffer_type() with
//       ggml_backend_alloc_ctx_tensors_from_buft(), so the production assert
//       is the one exercised.
//
// ENVIRONMENT
//
//   The arms differ by constructor arguments and by
//   llm_graph_params::rs_row_armed, so a runner that exports
//   GGML_TP5_RSZ_RUNTIME_PARAM, GGML_TP5_UNION_BINDER,
//   LLAMA_GRAPH_REUSE_DISABLE or GGML_TP5_UNION_BINDER_EXACT cannot collapse
//   two arms into one path. The one variable this file WRITES is
//   LLAMA_GDN_REPLAY, which it unsets in its own process (see below); it reads
//   no other environment variable and changes none that any assertion depends
//   on.
//
//   The one variable the fixture's memory does read is LLAMA_GDN_REPLAY
//   (src/llama-memory-recurrent.cpp:44-48). The fixture unsets it immediately
//   before constructing the memory, so gdn_replay is false whatever the caller
//   exported. The suppression is NOT achieved by a zero ingredient count: the
//   fixture needs a real state tensor (n_embd_s() = ssm_d_state * ssm_d_inner
//   = 32, the state_size build_rs() is given), and ssm_d_inner 8 makes
//   n_embd_s_ingredient() 32 > 0, which is exactly the condition under which
//   the env var WOULD arm the ingredient ring. unsetenv() is process-local, so
//   it touches nothing outside this test binary, and
//   CHECK(!f.recr->gdn_replay()) in the arms proves the suppression worked.
//
// Exits 0 on pass, 1 on any failed check.

namespace {

int g_fail = 0;
int g_checks_run = 0;

#define CHECK(cond, ...) do { \
    ++g_checks_run; \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        std::fprintf(stderr, __VA_ARGS__); \
        std::fprintf(stderr, "\n"); \
        ++g_fail; \
    } \
} while (0)

// A model that exists only to carry hparams. llama_model's constructor and
// destructor are safe on an empty impl (llama-model.cpp:1464-1481); the pure
// virtual loaders are never called because nothing is loaded.
struct stub_model : llama_model {
    explicit stub_model(const llama_model_params & params) : llama_model(params) {}

    void load_stats        (struct llama_model_loader &) override;
    void load_hparams      (struct llama_model_loader &) override;
    void load_vocab        (struct llama_model_loader &) override;
    bool load_tensors      (struct llama_model_loader &) override;
    void load_arch_hparams (struct llama_model_loader &) override;
    void load_arch_tensors (struct llama_model_loader &) override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override {
        return nullptr;
    }
};

void stub_model::load_stats        (struct llama_model_loader &) {}
void stub_model::load_hparams      (struct llama_model_loader &) {}
void stub_model::load_vocab        (struct llama_model_loader &) {}
bool stub_model::load_tensors      (struct llama_model_loader &) { return false; }
void stub_model::load_arch_hparams (struct llama_model_loader &) {}
void stub_model::load_arch_tensors (struct llama_model_loader &) {}

// one layer, `mem_size` cells, an explicit ring depth. The state tensors are
// sized mem_size*(1+n_rs_seq) rows by the cache itself
// (llama-memory-recurrent.cpp:121-128), so the ring is observable through
// get_n_rs_seq() and through the tensor the zero-row op is applied to.
//
// The graph tensors live in a no_alloc=true ggml context with a real
// ggml_backend_cpu_buffer_type() buffer, which is what
// union_rsz_bind_index()'s host-buffer assert requires.
struct fixture {
    stub_model              * model  = nullptr;
    llama_memory_recurrent  * mem    = nullptr;
    llama_memory_recurrent_context * recr = nullptr;
    ggml_context            * ctx0   = nullptr;   // graph tensors, no_alloc
    ggml_backend_buffer_t     buf    = nullptr;   // CPU buffer over ctx0
    ggml_tensor             * row    = nullptr;   // I32 [1], the zero-row index
    // the copy tensors can_reuse() dereferences (llama-graph.cpp:594-596),
    // sized exactly as build_rs_inp_impl() sizes them (llama-graph.cpp:4274-4283)
    ggml_tensor             * s_copy = nullptr;   // I32 [n_rs]
    ggml_tensor             * s_main = nullptr;   // I32 [n_seqs]
    ggml_tensor             * s_extra= nullptr;   // I32 [n_rs - n_seqs]
    std::unique_ptr<llm_graph_result> res;        // the real graph result

    // the ubatch the bind reads: one token per sequence
    llama_batch_allocr        balloc;
    llama_ubatch              ub;
    std::vector<llama_seq_id> seqs;

    // llama_model_params::devices is a borrowed, NULL-terminated pointer and
    // the model keeps it, so it must outlive the model: a member here.
    ggml_backend_dev_t devices[1] = { nullptr };

    fixture(uint32_t mem_size, uint32_t n_rs_seq, uint32_t n_seqs)
        : balloc(1), ub(balloc.ubatch_reserve(1, n_seqs)) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.vocab_only   = true;   // no weights are ever read
        mp.devices      = devices;

        model = new stub_model(mp);
        model->hparams.n_layer_all   = 1;
        model->hparams.n_embd        = 8;
        model->hparams.n_embd_r_impl = 8;   // n_embd_r() = 8 (explicit override)
        model->hparams.ssm_d_state   = 4;
        // A REAL state tensor: n_embd_s() = ssm_d_state * ssm_d_inner = 32
        // (src/llama-hparams.cpp:228), which is exactly the state_size this
        // file passes to build_rs(), so ggml_reshape_2d's element-count
        // assert holds. With ssm_d_inner 0 the tensor is zero-element and the
        // first build_rs() call aborts.
        model->hparams.ssm_d_inner   = 8;
        model->hparams.ssm_d_conv    = 4;   // keeps n_embd_r() non-zero if the override goes

        // gdn_replay is suppressed EXPLICITLY, not by a zero ingredient count:
        // with ssm_d_inner 8 n_embd_s_ingredient() is 32 > 0, so an inherited
        // LLAMA_GDN_REPLAY would turn the ingredient ring on
        // (src/llama-memory-recurrent.cpp:44-48) and this fixture tests the K
        // snapshot ring, not the ingredient ring. setenv/unsetenv are
        // process-local, so this affects only this test binary.
        unsetenv("LLAMA_GDN_REPLAY");

        mem = new llama_memory_recurrent(*model,
                GGML_TYPE_F32, GGML_TYPE_F32, /*offload=*/false,
                mem_size, /*n_seq_max=*/2, n_rs_seq,
                /*gdn_replay_req=*/false, /*filter=*/nullptr);

        std::vector<llama_ubatch> ubatches;   // empty: is_full == false
        recr = new llama_memory_recurrent_context(mem, std::move(ubatches));

        ggml_init_params gp = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 8 + 4096,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx0 = ggml_init(gp);
        if (ctx0 != nullptr) {
            // The recurrent sub-cache has not run find_slot() yet, so its n is
            // still 0 and get_n_rs() (is_full is a hard `const bool false`)
            // reports 0. s_copy would then be a zero-byte tensor and
            // ggml_view_1d(s_copy, n_seqs, 0) trips ggml.c's view-bounds
            // assert (data_size + view_offs <= ggml_nbytes(view_src) with
            // nbytes 0). This fixture gives the cache mem_size cells, which is
            // what a full-cache context would report -- the same workaround
            // tests/test-rsz-hybrid-selection-write.cpp:246 documents.
            mem->n = mem_size;

            const int64_t n_rs    = recr->get_n_rs();
            const int64_t n_seqs_ = (int64_t) n_seqs;

            row     = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
            s_copy  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
            s_main  = ggml_view_1d(ctx0, s_copy, n_seqs_, 0);
            s_extra = ggml_view_1d(ctx0, s_copy, n_rs - n_seqs_, n_seqs_ * s_copy->nb[0]);
            ggml_set_input(row);
            ggml_set_input(s_copy);
            // the production assert is ggml_backend_buffer_is_host(), which
            // needs a real buffer: allocate the context's tensors on the CPU
            // buffer type
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx0, ggml_backend_cpu_buffer_type());
        }

        // one token per sequence, each naming its own sequence
        for (uint32_t s = 0; s < n_seqs; ++s) {
            seqs.push_back((llama_seq_id) s);
        }
        for (uint32_t s = 0; s < n_seqs; ++s) {
            ub.seq_id[s]   = &seqs[s];
            ub.n_seq_id[s] = 1;
            ub.pos[s]      = 0;
        }
    }

    ~fixture() {
        res.reset();
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx0 != nullptr) {
            ggml_free(ctx0);
        }
        delete recr;
        delete mem;
        delete model;
    }

    bool ok() const {
        return model != nullptr && mem != nullptr && recr != nullptr &&
               ctx0 != nullptr && buf != nullptr && row != nullptr &&
               s_copy != nullptr && s_main != nullptr && s_extra != nullptr;
    }

    fixture(const fixture &) = delete;
    fixture & operator=(const fixture &) = delete;
};

// the value the row holds, or a sentinel when there is no row
int32_t row_value(const ggml_tensor * t) {
    return t == nullptr ? std::numeric_limits<int32_t>::min() : ((const int32_t *) t->data)[0];
}

// Drives the REAL builder. llm_graph_context's constructor stores its
// parameters and takes ctx0/gf from the llm_graph_result it is handed
// (llama-graph.cpp:1891-1896), so the fixture must supply a real result: a
// null one is a null dereference, not a shortcut.
//
// The get_rows callback is ggml_get_rows, the same default the production
// wrapper uses (llama-graph.h:1490), so the copy node is the production node.
struct graph_driver : llm_graph_context {
    graph_driver(const llm_graph_params & params) : llm_graph_context(params) {}

    // const member: build_rs() is const and reads ctx0/gf without writing
    // either. Its input parameter is non-const, so this wrapper's is too.
    ggml_tensor * rs(llm_graph_input_rs * inp, ggml_tensor * s, int32_t state_size,
                     int32_t n_seqs) const {
        return build_rs(inp, s, state_size, n_seqs, ggml_get_rows);
    }
};

// the tensors build_rs() created, found by name in the result's context
struct rs_nodes {
    ggml_tensor * zero_row = nullptr;   // union_rs_zero_row (the in-place op)
    ggml_tensor * states   = nullptr;   // the 2D reshape the gathers consume
    bool          found    = false;
};

rs_nodes find_rs_nodes(ggml_context * ctx) {
    rs_nodes n;
    for (ggml_tensor * x = ggml_get_first_tensor(ctx); x != nullptr; x = ggml_get_next_tensor(ctx, x)) {
        if (std::strcmp(x->name, "union_rs_zero_row") == 0) {
            n.zero_row = x;
        }
    }
    if (n.zero_row == nullptr) {
        return n;
    }
    for (ggml_tensor * x = ggml_get_first_tensor(ctx); x != nullptr; x = ggml_get_next_tensor(ctx, x)) {
        if (x->op == GGML_OP_RESHAPE && x->src[0] == n.zero_row) {
            n.states = x;
            break;
        }
    }
    n.found = n.states != nullptr;
    return n;
}

// One build of the production graph over the fixture's state tensor.
//
// `armed` is llm_graph_params::rs_row_armed, the ONE flag that decides whether
// the build carries the runtime zero-row index (llama-graph.cpp:4276). With it
// false the build is the stock view/scale form, which is R2's shape.
struct build_out {
    rs_nodes      nodes;
    ggml_tensor * out_states = nullptr;   // build_rs()'s return (the main gather)
    bool          ok          = false;
};

build_out build_once(fixture & f, bool armed, int32_t state_size, int32_t n_seqs) {
    build_out b;

    // a fresh graph result per build: reset() re-creates the compute context
    // and the graph, so two builds never share nodes
    f.res.reset(new llm_graph_result(1024));

    llm_graph_params params = {};
    params.arch         = LLM_ARCH_QWEN35;   // an arch with rs rollback support
    params.hparams      = f.model->hparams;
    params.cparams      = {};
    params.ubatch       = {};
    params.mctx         = f.recr;
    params.res          = f.res.get();       // a REAL result: the ctor dereferences it
    params.rs_row_armed = armed;

    graph_driver g(params);

    llm_graph_input_rs inp(f.recr);
    inp.s_copy       = nullptr;             // the copy loop is skipped
    inp.s_copy_main  = nullptr;
    inp.s_copy_extra = nullptr;
    inp.head         = f.recr->get_head();
    inp.rs_z         = -2;                  // "never bound" sentinel
    inp.s_zero       = armed ? f.row : nullptr;
    inp.replay_len   = 0;

    b.out_states = g.rs(&inp, f.mem->s_l[0], state_size, n_seqs);
    b.nodes      = find_rs_nodes(f.res->get_ctx());
    b.ok         = b.out_states != nullptr;
    return b;
}

// Stages the fixture's single-sequence ubatch into a NON-ZERO cell, so that
// rs_z (which find_slot() sets to the staged cell) is non-zero AND the cell
// the gathers read is the cell rs_z names.
//
// Why this is needed: find_slot() stages src0 = rs_z for every cell with no
// prior state (llama-memory-recurrent.cpp:772-780), and s_copy() returns
// rs_idx*mem_size + src0 (:1482-1501). So `src0 == rs_z` holds by
// construction, but rs_z is 0 on a fresh cache. With rs_z == 0 the row
// `rs_idx*mem_size + rs_z` collapses to `rs_idx*mem_size`, and an
// implementation that drops rs_z entirely is indistinguishable from the
// correct one.
//
// Occupying cell 0 with a sequence that is NOT in the batch pushes the
// batch's cell to index 1: find_slot() starts its empty-cell search at
// `head` (0) and advances past the occupied cell (:654-661), so min == 1.
//
// Returns the staged cell (== rs_z), or -1 when the staging did not land on
// a non-zero cell.
int32_t stage_nonzero_cell(fixture & f) {
    // Both placements go through a LOCAL context that owns the ubatches.
    //
    // f.recr was built with an EMPTY ubatch vector, so its apply() takes the
    // "no ubatches -> update" branch and asserts
    // status == LLAMA_MEMORY_STATUS_NO_UPDATE (llama-memory-recurrent.cpp:
    // 1344-1349); its status is SUCCESS, so that assert fires. The local
    // contexts below carry real ubatches and take the find_slot() branch
    // instead. They mutate `mem`, which is what the fixture's own context
    // reads back through get_head()/get_rs_z()/get_rs_zero_row().
    llama_batch_allocr balloc(1);

    // 1. seq 1 alone takes cell 0
    {
        llama_ubatch ub = balloc.ubatch_reserve(1, 1);
        std::vector<llama_seq_id> ids = { 1 };
        ub.seq_id[0]   = &ids[0];
        ub.n_seq_id[0] = 1;
        ub.pos[0]      = 0;

        llama_memory_recurrent_context ctx(f.mem, { ub });
        if (!ctx.apply()) {
            return -1;
        }
    }

    // 2. the fixture's own ubatch (seq 0) must now land past cell 0
    {
        llama_memory_recurrent_context ctx(f.mem, { f.ub });
        if (!ctx.apply()) {
            return -1;
        }
    }

    const int32_t cell = f.mem->cells[f.mem->head].src0;
    if (cell <= 0 || cell >= (int32_t) f.mem->size) {
        return -1;
    }
    f.mem->rs_z = cell;
    return cell;
}

// one set_input() pass over the fixture's ubatch, through the production
// llm_graph_input_rs::set_input(). Returns the input so the caller can read
// rs_zero_row_refused.
struct bind_out {
    int32_t row      = 0;
    bool    refused  = false;
    bool    reuse_ok = false;
};

bind_out bind_once(fixture & f, int32_t seed) {
    bind_out b;

    llm_graph_input_rs inp(f.recr);
    inp.s_copy       = nullptr;   // the copy loop is skipped: s_copy consumes rs_idx
    inp.s_copy_main  = nullptr;
    inp.s_copy_extra = nullptr;
    inp.head         = f.recr->get_head();
    inp.rs_z         = f.recr->get_rs_z();
    inp.s_zero       = f.row;
    inp.replay_len   = 0;

    ((int32_t *) f.row->data)[0] = seed;
    inp.set_input(&f.ub);

    b.row     = row_value(f.row);
    b.refused = inp.rs_zero_row_refused;

    // can_reuse() is the production reuse decision, and it DEREFERENCES
    // s_copy / s_copy_main / s_copy_extra (llama-graph.cpp:594-596). They are
    // null during set_input() on purpose -- the copy loop is the one consumer
    // of rs_idx, and wiring it would consume the depth this test drives -- so
    // they are wired here, from the fixture's real tensors, for the reuse
    // call only. Passing them null would be a null dereference, not a
    // shortcut.
    inp.s_copy       = f.s_copy;
    inp.s_copy_main  = f.s_main;
    inp.s_copy_extra = f.s_extra;

    llm_graph_params params = {};
    params.mctx   = f.recr;
    params.ubatch = f.ub;
    b.reuse_ok = inp.can_reuse(params);
    return b;
}

} // namespace

int main() {
    llama_backend_init();

    const bool relay_ok = true;   // the relay lane is live and its scheduler is one split
    const bool gate_ok  = true;   // the stock lane's umbrella gate
    const bool has_recr = true;

    const int32_t state_size = 32;   // n_embd_s() of the fixture

    // ======================================================================
    // R0: n_rs_seq == 0, relay form -- group 0 is the only group
    // ======================================================================
    {
        const uint32_t mem_size = 4;
        fixture f(mem_size, /*n_rs_seq=*/0, /*n_seqs=*/1);
        if (!f.ok()) {
            std::fprintf(stderr, "FAIL: cannot build the R0 fixture\n");
            llama_backend_free();
            return 1;
        }

        CHECK(f.recr->get_n_rs_seq() == 0,
              "R0: cache reports n_rs_seq %u, expected 0", f.recr->get_n_rs_seq());
        CHECK(!f.recr->gdn_replay(),
              "R0: gdn_replay is on; LLAMA_GDN_REPLAY must not change this fixture");

        CHECK(llama_rsz_rt_relay_armable(relay_ok, has_recr, 0, 1),
              "R0: relay form refused an empty ring (n_rs_seq=0, rs_z=1)");
        CHECK(llama_rsz_rt_armable(gate_ok, has_recr, 0, 1),
              "R0: stock form refused an empty ring (n_rs_seq=0, rs_z=1)");

        const uint32_t expect_rows = mem_size;
        CHECK(f.mem->s_l[0]->ne[1] == (int64_t) expect_rows,
              "R0: state tensor has %lld rows, expected mem_size %u",
              (long long) f.mem->s_l[0]->ne[1], expect_rows);

        const build_out b = build_once(f, /*armed=*/true, state_size, /*n_seqs=*/1);
        CHECK(b.ok, "R0: build_rs() returned no states");
        CHECK(b.nodes.found, "R0: build_rs() created no union_rs_zero_row / reshape pair");
        CHECK(b.nodes.zero_row != nullptr && b.nodes.zero_row->op == GGML_OP_MAP_CUSTOM2,
              "R0: the zero node is not the MAP_CUSTOM2 zero-row op");
        CHECK(b.nodes.zero_row != nullptr && b.nodes.zero_row->src[1] == f.row,
              "R0: the zero-row op does not consume the runtime index tensor");
        CHECK(b.nodes.states != nullptr && b.nodes.states->ne[1] == (int64_t) expect_rows,
              "R0: the gathers see %lld rows, expected %u",
              b.nodes.states == nullptr ? -1LL : (long long) b.nodes.states->ne[1], expect_rows);

        // R1's observable is absent: the state is NOT widened
        CHECK(b.nodes.states != nullptr && b.nodes.states->ne[1] != (int64_t) (3 * mem_size),
              "R0: the gathers see %lld rows, which is the R1 ring height",
              b.nodes.states == nullptr ? -1LL : (long long) b.nodes.states->ne[1]);

        const uint32_t rows = (uint32_t) f.mem->s_l[0]->ne[1];
        for (int32_t rs_z = 0; rs_z < (int32_t) mem_size; ++rs_z) {
            CHECK(llama_rsz_rt_check_relay_row(rs_z, mem_size, rows) == llama_rsz_rt_row_check::ok,
                  "R0: relay row check refused rs_z=%d inside the %u-cell capacity",
                  rs_z, mem_size);
        }
        CHECK(llama_rsz_rt_check_relay_row((int32_t) mem_size, mem_size, rows) ==
                  llama_rsz_rt_row_check::envelope,
              "R0: relay row check accepted rs_z=%u, past the %u-cell capacity",
              mem_size, mem_size);
        CHECK(llama_rsz_rt_check_relay_row(-1, mem_size, rows) ==
                  llama_rsz_rt_row_check::cell,
              "R0: relay row check did not report rs_z=-1 as _cell");
        CHECK(llama_rsz_rt_check_relay_row(0, mem_size, rows + 1) ==
                  llama_rsz_rt_row_check::shape,
              "R0: relay row check accepted a tensor that is not a whole number of groups");

        // the write: seed a value that is not the expected one, so a missing
        // write cannot pass by coincidence. With n_rs_seq == 0 the bound row
        // IS the cell index (rs_idx is 0), so R0's row equals rs_z.
        f.mem->rs_z = 3;
        CHECK(f.recr->get_rs_z() == 3,
              "R0: context reports rs_z %d, expected 3", f.recr->get_rs_z());

        const bind_out bo = bind_once(f, /*seed=*/-7);
        CHECK(!bo.refused, "R0: the bind refused a single-depth ubatch");
        CHECK(bo.row == 3,
              "R0: row holds %d after set_input(), expected the cell index 3", bo.row);
        // the row the gathers read, from the library's own accessor. With
        // rs_idx == 0 this is the cell index, so it must equal the literal 3
        // the test set -- not merely agree with the value it just wrote.
        CHECK(f.recr->get_rs_zero_row(0) == 3,
              "R0: get_rs_zero_row(0) is %d, expected the cell index 3",
              f.recr->get_rs_zero_row(0));
        CHECK(bo.row == f.recr->get_rs_zero_row(0),
              "R0: the written row %d != get_rs_zero_row(0) %d",
              bo.row, f.recr->get_rs_zero_row(0));
        CHECK(bo.reuse_ok, "R0: can_reuse() refused a graph whose row was bound");
    }

    // ======================================================================
    // R1: n_rs_seq == 2, relay form -- group 2
    // ======================================================================
    {
        const uint32_t mem_size = 4;
        const uint32_t n_rs_seq = 2;
        fixture f(mem_size, n_rs_seq, /*n_seqs=*/1);
        if (!f.ok()) {
            std::fprintf(stderr, "FAIL: cannot build the R1 fixture\n");
            llama_backend_free();
            return 1;
        }

        CHECK(f.recr->get_n_rs_seq() == n_rs_seq,
              "R1: cache reports n_rs_seq %u, expected %u",
              f.recr->get_n_rs_seq(), n_rs_seq);
        CHECK(!f.recr->gdn_replay(),
              "R1: gdn_replay is on; the fixture must exercise the K snapshot ring");

        // the relay form carries the ring. On pre-merge code the only rule was
        // llama_rsz_rt_armable(), which refuses n_rs_seq > 0, so this fails.
        CHECK(llama_rsz_rt_relay_armable(relay_ok, has_recr, n_rs_seq, 1),
              "R1: relay form refused the ring (n_rs_seq=%u, rs_z=1)", n_rs_seq);

        const uint32_t expect_rows = mem_size * (1 + n_rs_seq);
        CHECK(f.mem->s_l[0]->ne[1] == (int64_t) expect_rows,
              "R1: state tensor has %lld rows, expected mem_size*(1+n_rs_seq) = %u",
              (long long) f.mem->s_l[0]->ne[1], expect_rows);

        const build_out b = build_once(f, /*armed=*/true, state_size, /*n_seqs=*/1);
        CHECK(b.ok, "R1: build_rs() returned no states");
        CHECK(b.nodes.found, "R1: build_rs() created no union_rs_zero_row / reshape pair");
        CHECK(b.nodes.zero_row != nullptr && b.nodes.zero_row->op == GGML_OP_MAP_CUSTOM2,
              "R1: the zero node is not the MAP_CUSTOM2 zero-row op");
        CHECK(b.nodes.zero_row != nullptr && b.nodes.zero_row->src[1] == f.row,
              "R1: the zero-row op does not consume the runtime index tensor");
        CHECK(b.nodes.states != nullptr && b.nodes.states->ne[1] == (int64_t) expect_rows,
              "R1: the gathers see %lld rows, expected %u",
              b.nodes.states == nullptr ? -1LL : (long long) b.nodes.states->ne[1], expect_rows);

        // R0's observable is absent: the state IS widened
        CHECK(b.nodes.states != nullptr && b.nodes.states->ne[1] != (int64_t) mem_size,
              "R1: the gathers see %lld rows, which is the R0 height (the ring is not widened)",
              b.nodes.states == nullptr ? -1LL : (long long) b.nodes.states->ne[1]);

        const uint32_t rows = (uint32_t) f.mem->s_l[0]->ne[1];
        for (int32_t rs_z = 0; rs_z < (int32_t) mem_size; ++rs_z) {
            CHECK(llama_rsz_rt_check_relay_row(rs_z, mem_size, rows) == llama_rsz_rt_row_check::ok,
                  "R1: relay row check refused rs_z=%d inside the %u-cell capacity",
                  rs_z, mem_size);
        }
        CHECK(llama_rsz_rt_check_relay_row((int32_t) mem_size, mem_size, rows) ==
                  llama_rsz_rt_row_check::envelope,
              "R1: relay row check accepted rs_z=%u, which is snapshot group 1, not a cell",
              mem_size);

        // ---- the row is rs_idx*mem_size + rs_z, not the cell index ---------
        //
        // rs_idx is driven through the public API: set_rs_idx(seq, depth)
        // clamps to n_rs_seq (llama-memory-recurrent.cpp:454-458), which is
        // the same clamp seq_rm()'s partial rollback goes through (:236-246).
        CHECK(f.recr->get_status() == LLAMA_MEMORY_STATUS_SUCCESS,
              "R1: memory context status is not success");

        // Stage the batch into a NON-ZERO cell, so rs_z != 0 and the cell the
        // gathers read is the cell rs_z names (see stage_nonzero_cell()).
        // Both facts are required: with rs_z == 0 the row
        // `rs_idx*mem_size + rs_z` collapses to `rs_idx*mem_size` and an
        // implementation that drops rs_z passes; with src0 != rs_z the
        // gathers read a different row from the one the op zeroes, so the
        // arm's central claim would be false for a correct implementation.
        const int32_t rs_z = stage_nonzero_cell(f);
        CHECK(rs_z > 0,
              "R1: could not stage the batch into a non-zero cell (got %d); the arm "
              "cannot separate rs_z from 0", rs_z);
        CHECK(rs_z < (int32_t) mem_size,
              "R1: the staged cell %d is outside the %u-cell capacity", rs_z, mem_size);
        CHECK(f.mem->head == rs_z && f.mem->n == 1,
              "R1: after staging, head=%u n=%u, expected %d/1",
              f.mem->head, f.mem->n, rs_z);
        CHECK(f.mem->cells[f.mem->head].src0 == rs_z,
              "R1: the gathers' source cell %d is not the cell rs_z names %d",
              f.mem->cells[f.mem->head].src0, rs_z);

        for (uint32_t depth = 0; depth <= n_rs_seq; ++depth) {
            f.mem->set_rs_idx(0, depth);
            CHECK(f.mem->rs_idx[0] == depth,
                  "R1: set_rs_idx(0, %u) left rs_idx %u", depth, f.mem->rs_idx[0]);

            const int32_t expect_row = (int32_t) (depth * mem_size) + rs_z;

            const int32_t ctx_row = f.recr->get_rs_zero_row(0);
            CHECK(ctx_row == expect_row,
                  "R1: get_rs_zero_row(0) = %d, expected rs_idx*mem_size + rs_z = %u*%u+%d = %d",
                  ctx_row, depth, mem_size, rs_z, expect_row);

            // what the production bind writes (set_input -> union_rsz_bind_index)
            const bind_out bo = bind_once(f, /*seed=*/-7);
            CHECK(!bo.refused, "R1: the bind refused a single-depth ubatch");
            CHECK(bo.row == expect_row,
                  "R1: the bound row is %d, expected %d (rs_idx=%u, mem_size=%u, rs_z=%d); "
                  "the cell index %d is the pre-merge value and names a row nobody reads",
                  bo.row, expect_row, depth, mem_size, rs_z, rs_z);
            CHECK(bo.reuse_ok, "R1: can_reuse() refused a graph whose row was bound");

            // what the gathers read for the cell that copies from rs_z:
            // s_copy() returns rs_idx*mem_size + src0 and CONSUMES rs_idx, so
            // it is called last and re-armed before the next depth
            const int32_t read_row = f.recr->s_copy(0);
            CHECK(read_row == expect_row,
                  "R1: s_copy(0) reads row %d, expected %d (the row the op zeroed)",
                  read_row, expect_row);
            CHECK(read_row == bo.row,
                  "R1: the gathers read row %d but the op zeroed row %d", read_row, bo.row);
            CHECK(read_row < (int32_t) expect_rows,
                  "R1: the read row %d is outside the %u-row state tensor",
                  read_row, expect_rows);

            // the cell index the pre-merge binder wrote. With rs_z != 0 this
            // is a real discriminator at every depth, including depth 0.
            CHECK(bo.row != rs_z || depth == 0,
                  "R1: depth %u bound the cell index %d, not a group row", depth, rs_z);
            // and the row must carry rs_z: an implementation that drops it
            // (row = rs_idx*mem_size) is distinguishable only because rs_z != 0
            CHECK(bo.row % (int32_t) mem_size == rs_z,
                  "R1: depth %u bound row %d, whose cell is %d, expected rs_z %d",
                  depth, bo.row, bo.row % (int32_t) mem_size, rs_z);
        }

        // the clamp: a depth past n_rs_seq is clamped, so it cannot name a
        // row outside the tensor
        f.mem->set_rs_idx(0, n_rs_seq + 5);
        CHECK(f.mem->rs_idx[0] == n_rs_seq,
              "R1: rs_idx %u was not clamped to n_rs_seq %u", f.mem->rs_idx[0], n_rs_seq);
        CHECK(f.recr->get_rs_zero_row(0) < (int32_t) expect_rows,
              "R1: the clamped row %d is outside the %u-row state tensor",
              f.recr->get_rs_zero_row(0), expect_rows);
        CHECK(f.recr->get_rs_zero_row(0) == (int32_t) (n_rs_seq * mem_size + rs_z),
              "R1: the clamped row %d, expected n_rs_seq*mem_size + rs_z = %u",
              f.recr->get_rs_zero_row(0), n_rs_seq * mem_size + rs_z);
    }

    // ======================================================================
    // R2: n_rs_seq == 2, stock (non-relay) form -- refused, with a control
    // ======================================================================
    {
        const uint32_t mem_size = 4;
        const uint32_t n_rs_seq = 2;
        fixture f(mem_size, n_rs_seq, /*n_seqs=*/1);
        if (!f.ok()) {
            std::fprintf(stderr, "FAIL: cannot build the R2 fixture\n");
            llama_backend_free();
            return 1;
        }

        for (int32_t rs_z = 0; rs_z < (int32_t) mem_size; ++rs_z) {
            CHECK(!llama_rsz_rt_armable(gate_ok, has_recr, n_rs_seq, rs_z),
                  "R2: the stock form armed the ring (n_rs_seq=%u, rs_z=%d): the stock lane "
                  "names cells, not snapshot rows, and must refuse", n_rs_seq, rs_z);
        }
        // POSITIVE CONTROL: the same call, same gate, same rs_z, arms once the
        // ring is gone. That is what makes the ring -- not the gate, not rs_z
        // -- the reason R2 refuses.
        CHECK(llama_rsz_rt_armable(gate_ok, has_recr, 0, 1),
              "R2: the stock form refused an empty ring, so the refusal above is not the ring's doing");
        // and the relay form on the SAME fixture arms: the ring is carried
        // there, so the refusal is the stock lane's, not the fixture's
        CHECK(llama_rsz_rt_relay_armable(relay_ok, has_recr, n_rs_seq, 1),
              "R2: the relay form refused the same fixture, so the fixture cannot carry a ring at all");
        // a gate that is off refuses even the empty ring: the control above
        // is not passing because the predicate ignores its gate argument
        CHECK(!llama_rsz_rt_armable(/*gate_ok=*/false, has_recr, 0, 1),
              "R2: the stock form armed with the gate off, so the gate argument is ignored");

        // the stock build carries no runtime index and keeps the stock
        // view/scale zero node, so rs_z stays a topology key there
        const build_out b = build_once(f, /*armed=*/false, state_size, /*n_seqs=*/1);
        CHECK(b.ok, "R2: the stock build returned no states");
        CHECK(b.nodes.zero_row == nullptr,
              "R2: the stock build created a union_rs_zero_row node");
        CHECK(b.nodes.states != nullptr &&
                  b.nodes.states->ne[1] == (int64_t) (mem_size * (1 + n_rs_seq)),
              "R2: the stock build changed the state height to %lld, expected %u",
              b.nodes.states == nullptr ? -1LL : (long long) b.nodes.states->ne[1],
              mem_size * (1 + n_rs_seq));
    }

    // ======================================================================
    // R3: n_rs_seq == 2, relay form, MIXED depths in one ubatch -- refused
    // ======================================================================
    {
        const uint32_t mem_size = 4;
        const uint32_t n_rs_seq = 2;
        fixture f(mem_size, n_rs_seq, /*n_seqs=*/2);
        if (!f.ok()) {
            std::fprintf(stderr, "FAIL: cannot build the R3 fixture\n");
            llama_backend_free();
            return 1;
        }

        // Place the two-sequence ubatch through a LOCAL context: f.recr was
        // built with an empty ubatch vector, so its apply() asserts
        // status == NO_UPDATE (llama-memory-recurrent.cpp:1344-1349) and
        // would abort. The local context takes the find_slot() branch and
        // mutates `mem`, which f.recr reads back.
        {
            llama_memory_recurrent_context ctx(f.mem, { f.ub });
            CHECK(ctx.apply(), "R3: apply() failed");
        }
        CHECK(f.mem->n == 2,
              "R3: after a two-sequence apply, n=%u, expected 2", f.mem->n);

        // the two sequences of the ubatch name different rows:
        //   seq 0 -> 2*mem_size + rs_z   (group 2)
        //   seq 1 -> 0*mem_size + rs_z   (group 0)
        f.mem->rs_z = 1;
        f.mem->set_rs_idx(0, 2);
        f.mem->set_rs_idx(1, 0);
        CHECK(f.recr->get_rs_zero_row(0) == 9,
              "R3: seq 0 row %d, expected 2*4+1 = 9", f.recr->get_rs_zero_row(0));
        CHECK(f.recr->get_rs_zero_row(1) == 1,
              "R3: seq 1 row %d, expected 0*4+1 = 1", f.recr->get_rs_zero_row(1));
        CHECK(f.recr->get_rs_zero_row(0) != f.recr->get_rs_zero_row(1),
              "R3: the two sequences agree on the row, so the mixed case is not exercised");

        // the production bind refuses: the row tensor keeps its seeded value
        // and the input records the refusal
        const bind_out bo = bind_once(f, /*seed=*/-7);
        CHECK(bo.refused,
              "R3: set_input() did not record a refusal for a mixed-depth ubatch");
        CHECK(bo.row == -7,
              "R3: the refused bind wrote row %d, expected the seed -7 to survive", bo.row);

        // ... and the graph must not be reused: the row it would clear is not
        // the row this batch reads
        CHECK(!bo.reuse_ok,
              "R3: can_reuse() accepted a graph whose zero row could not be bound");

        // the same graph with a single depth is reusable again: the refusal
        // is the mixed batch's, not a permanent state
        f.mem->set_rs_idx(1, 2);
        const bind_out bo2 = bind_once(f, /*seed=*/-7);
        CHECK(!bo2.refused, "R3: the bind refused a single-depth ubatch");
        CHECK(bo2.row == 9, "R3: the bound row is %d, expected 9", bo2.row);
        CHECK(bo2.reuse_ok, "R3: can_reuse() refused a graph whose row was bound");
    }

    // ======================================================================
    // SWEEP: mem_size x n_rs_seq, the row arithmetic
    // ======================================================================
    {
        const uint32_t sizes[] = { 1, 2, 3, 4, 7, 8 };
        for (uint32_t mem_size : sizes) {
            for (uint32_t n_rs_seq : { 0u, 1u, 2u }) {
                fixture f(mem_size, n_rs_seq, /*n_seqs=*/1);
                if (!f.ok()) {
                    std::fprintf(stderr, "FAIL sweep: cannot build mem_size=%u n_rs_seq=%u\n",
                                 mem_size, n_rs_seq);
                    ++g_fail;
                    continue;
                }
                const uint32_t rows = (uint32_t) f.mem->s_l[0]->ne[1];
                CHECK(rows == mem_size * (1 + n_rs_seq),
                      "sweep mem_size=%u n_rs_seq=%u: state tensor has %u rows, expected %u",
                      mem_size, n_rs_seq, rows, mem_size * (1 + n_rs_seq));

                // Stage into a NON-ZERO cell so rs_z != 0 and src0 == rs_z
                // (see stage_nonzero_cell()). With rs_z == 0 the row
                // `rs_idx*mem_size + rs_z` collapses to `rs_idx*mem_size`
                // and an implementation that drops rs_z passes every cell.
                //
                // mem_size == 1 has no non-zero cell, so that combination
                // CANNOT separate rs_z from 0: it still checks the row
                // arithmetic and the capacity bound, but it says nothing
                // about the rs_z term. The check below states that limit
                // instead of hiding it.
                const int32_t rs_z = stage_nonzero_cell(f);
                if (mem_size == 1) {
                    CHECK(rs_z < 0,
                          "sweep mem_size=1: a non-zero cell was staged (%d); mem_size 1 "
                          "cannot hold one", rs_z);
                    // stage cell 0 through a local context so the remaining
                    // arithmetic is still exercised. The rs_z term is
                    // degenerate here by construction; the mem_size >= 2 cells
                    // cover it.
                    {
                        llama_memory_recurrent_context ctx(f.mem, { f.ub });
                        CHECK(ctx.apply(), "sweep mem_size=1: apply() failed");
                    }
                    f.mem->rs_z = 0;
                    CHECK(f.mem->head == 0 && f.mem->n == 1,
                          "sweep mem_size=1: head=%u n=%u, expected 0/1",
                          f.mem->head, f.mem->n);
                } else {
                    CHECK(rs_z > 0,
                          "sweep mem_size=%u n_rs_seq=%u: could not stage a non-zero cell (%d)",
                          mem_size, n_rs_seq, rs_z);
                    CHECK(f.mem->cells[f.mem->head].src0 == rs_z,
                          "sweep mem_size=%u n_rs_seq=%u: the gathers' source cell %d is not "
                          "the cell rs_z names %d",
                          mem_size, n_rs_seq, f.mem->cells[f.mem->head].src0, rs_z);
                }

                const int32_t rs_z_used = f.mem->rs_z;

                for (uint32_t depth = 0; depth <= n_rs_seq; ++depth) {
                    f.mem->set_rs_idx(0, depth);

                    const int32_t expect_row = (int32_t) (depth * mem_size) + rs_z_used;

                    const bind_out bo = bind_once(f, /*seed=*/-7);
                    CHECK(!bo.refused,
                          "sweep mem_size=%u n_rs_seq=%u rs_z=%d depth=%u: the bind refused",
                          mem_size, n_rs_seq, rs_z_used, depth);
                    CHECK(bo.row == expect_row,
                          "sweep mem_size=%u n_rs_seq=%u rs_z=%d depth=%u: bound row %d, "
                          "expected rs_idx*mem_size + rs_z = %d",
                          mem_size, n_rs_seq, rs_z_used, depth, bo.row, expect_row);
                    CHECK(bo.row < (int32_t) (mem_size * (1 + n_rs_seq)),
                          "sweep mem_size=%u n_rs_seq=%u rs_z=%d depth=%u: bound row %d is "
                          "outside mem_size*(1+n_rs_seq) = %u",
                          mem_size, n_rs_seq, rs_z_used, depth, bo.row, mem_size * (1 + n_rs_seq));
                    CHECK(bo.row >= 0,
                          "sweep mem_size=%u n_rs_seq=%u rs_z=%d depth=%u: bound row %d is "
                          "negative",
                          mem_size, n_rs_seq, rs_z_used, depth, bo.row);

                    // the row the gathers read, read from s_copy() itself --
                    // not recomputed here. s_copy() consumes rs_idx, so it is
                    // called last and re-armed by the next set_rs_idx().
                    const int32_t read_row = f.recr->s_copy(0);
                    CHECK(read_row == bo.row,
                          "sweep mem_size=%u n_rs_seq=%u rs_z=%d depth=%u: the gathers read "
                          "row %d but the op zeroed row %d",
                          mem_size, n_rs_seq, rs_z_used, depth, read_row, bo.row);

                    // the row rule depends on rs_z and the capacity only: it
                    // must accept this cell and refuse the first cell past the
                    // capacity, at every depth
                    CHECK(llama_rsz_rt_check_relay_row(rs_z_used, mem_size, rows) ==
                              llama_rsz_rt_row_check::ok,
                          "sweep mem_size=%u n_rs_seq=%u: relay row check refused cell %d",
                          mem_size, n_rs_seq, rs_z_used);
                    CHECK(llama_rsz_rt_check_relay_row((int32_t) mem_size, mem_size, rows) ==
                              llama_rsz_rt_row_check::envelope,
                          "sweep mem_size=%u n_rs_seq=%u: relay row check accepted cell %u",
                          mem_size, n_rs_seq, mem_size);
                }
            }
        }
    }

    llama_backend_free();

    if (g_checks_run == 0) {
        std::fprintf(stderr, "RS_RING_RELAY_TEST UNPROVEN reason=no_check_ran checks=0\n");
        return 2;
    }
    if (g_fail != 0) {
        std::printf("RS_RING_RELAY_TEST FAIL %d\n", g_fail);
        return 1;
    }
    std::printf("RS_RING_RELAY_TEST PASS\n");
    return 0;
}
