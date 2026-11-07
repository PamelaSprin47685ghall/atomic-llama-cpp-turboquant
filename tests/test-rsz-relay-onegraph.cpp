#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-memory.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>   // unsetenv: the fixture suppresses LLAMA_GDN_REPLAY
#include <cstring>
#include <memory>
#include <vector>

//
// test-rsz-relay-onegraph
//
// The ticket's "rebuild_count == 0 across a moving rs_z" claim, pinned on the
// REUSE DECISION the relay lane actually takes, not on a counter that needs a
// live relay backend.
//
// WHY THIS IS NOT A COUNTER TEST. A relay context needs params.tp5_relay and a
// Meta backend with GGML_RELAY_RANKS ranks (src/llama-context.cpp:747-765), so
// `llama_graph_build_count()` cannot be read on this host. What CAN be pinned
// is the decision that produces the count: llama_context::process_ubatch()
// calls res->can_reuse(gparams) (src/llama-context.cpp:5139) and rebuilds iff
// it returns false. On the relay lane there is no binder to bind anything
// (union_rebind_apply refuses relay by name, src/llama-context.cpp:4424-4432),
// so can_reuse() is the WHOLE story: true means one graph, zero rebuilds.
//
// WHAT THE RELAY LANE CAN AND CANNOT DO -- stated here, not assumed:
//
//   rs_z moves          can_reuse() stays true. The zero row is a runtime I32
//                       input written per set_inputs() (src/llama-graph.cpp:578)
//                       and the rs_z key is skipped whenever s_zero is wired
//                       (:603-608). This is the ticket's property.
//
//   n_kv moves          can_reuse() depends on the KQ mask width, and on a
//                       relay context the mask is built at the envelope while
//                       can_reuse_kq_mask() compares it against
//                       get_n_kv_target() == get_n_kv() == the true used span
//                       (src/llama-graph.cpp:108-117, llama-kv-cache.cpp:3784-3789).
//                       The two agree only on the step whose used span equals
//                       the envelope, so a growing n_kv DOES rebuild on relay.
//                       There is NO live arm for it here: the only
//                       llama_kv_cache_context that yields an n_kv without a
//                       llama_context is the full-cache constructor
//                       (llama-kv-cache.cpp:3666-3681), which pins n_kv to the
//                       capacity and therefore cannot move. The rule is
//                       asserted where it is reachable, on the union binder
//                       (tests/test-rsz-nkv-rsz-topology.cpp), and the
//                       three-parameter one-graph claim is proved there and by
//                       test-triparam-matrix (rebuilds=0 over 300 cells).
//                       Consequence, stated plainly: the ticket's "one graph
//                       across n_kv" holds on the union binder, NOT on relay.
//
//   n_tokens moves      can_reuse() is false BEFORE any input is asked:
//                       llm_graph_params::allow_reuse() compares ubatch.n_tokens
//                       (src/llama-graph.h:931) inside llm_graph_result::can_reuse()
//                       (src/llama-graph.cpp:1762). A moved ntok is a rebuild by
//                       construction on every lane, relay included.
//
//   n_outputs moves     can_reuse() is driven by llm_graph_input_out_ids::
//                       can_reuse(), which compares bound_outputs against
//                       params.n_outputs (src/llama-graph.cpp:389-394). The
//                       params-level n_outputs key is skipped (ignore_runtime),
//                       so the input's bound width is the whole rule.
//
// ARMS
//
//   R1  relay form, rs_z moves, everything else held: can_reuse() TRUE.
//       Catches: rs_z re-entering a reuse key; a bind that runs once per graph.
//   R3  n_tokens moves: can_reuse() FALSE (allow_reuse, before any input).
//       Catches: a reuse rule that ignores the ubatch shape.
//   R4  n_outputs moves: can_reuse() FALSE; unchanged: TRUE. This is
//       llm_graph_input_out_ids::can_reuse(), the one place n_out is keyed.
//       Catches: the n_out key disappearing, and the control keeps it from
//       passing vacuously.
//   R5  control: the same R1 pair with the rs_z key FORCED back on (s_zero
//       nulled) is refused. Without this, R1 could pass because can_reuse()
//       never refuses anything.
//
// The labels are a stable history, not an index: there is no R2 arm. An n_kv
// arm would need a movable llama_kv_cache_context and none is constructible
// without a llama_context (see "WHAT THE RELAY LANE CAN AND CANNOT DO").
//
// No model file, no llama_context, no backend device. Exits 0 on pass, 1 on
// any failed check. Prints one RSZ_RELAY_ONEGRAPH_STATS line, then
// RSZ_RELAY_ONEGRAPH_TEST PASS|FAIL n.

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

// A model that exists only to carry hparams; nothing is ever loaded.
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

// The recurrent half of the relay lane: one layer, `mem_size` cells, ring
// depth 2 so the state tensor is widened to mem_size*(1+n_rs_seq) rows and the
// zero row is rs_idx*mem_size + rs_z, not the cell index.
struct fixture {
    stub_model              * model  = nullptr;
    llama_memory_recurrent  * mem    = nullptr;
    llama_memory_recurrent_context * recr = nullptr;
    ggml_context            * ctx0   = nullptr;
    ggml_backend_buffer_t     buf    = nullptr;
    ggml_tensor             * row    = nullptr;   // I32 [1]
    ggml_tensor             * s_copy = nullptr;   // I32 [n_rs]
    ggml_tensor             * s_main = nullptr;
    ggml_tensor             * s_extra= nullptr;

    llama_batch_allocr        balloc;
    llama_ubatch              ub;
    std::vector<llama_seq_id> seqs;

    ggml_backend_dev_t devices[1] = { nullptr };

    fixture(uint32_t mem_size, uint32_t n_rs_seq, uint32_t n_seqs)
        : balloc(1), ub(balloc.ubatch_reserve(1, n_seqs)) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.vocab_only   = true;
        mp.devices      = devices;

        model = new stub_model(mp);
        model->hparams.n_layer_all   = 1;
        model->hparams.n_embd        = 8;
        model->hparams.n_embd_r_impl = 8;   // n_embd_r() = 8 (explicit override)
        model->hparams.ssm_d_state   = 4;
        // A REAL state tensor: n_embd_s() = ssm_d_state * ssm_d_inner = 32
        // (src/llama-hparams.cpp:228), the same width the inherited
        // test-rs-ring-relay passes to build_rs() as state_size. With
        // ssm_d_inner 0 the tensor is zero-element and any build_rs() on it
        // aborts at ggml_reshape_2d's element-count assert.
        model->hparams.ssm_d_inner   = 8;
        model->hparams.ssm_d_conv    = 4;   // keeps n_embd_r() non-zero if the override goes

        // This file owns LLAMA_GDN_REPLAY inside its own process: setenv and
        // unsetenv are process-local. With ssm_d_inner 8 the ingredient count
        // is 32 > 0, so an inherited LLAMA_GDN_REPLAY would enable gdn_replay
        // (src/llama-memory-recurrent.cpp:44-48) and change the state shape.
        unsetenv("LLAMA_GDN_REPLAY");

        mem = new llama_memory_recurrent(*model,
                GGML_TYPE_F32, GGML_TYPE_F32, /*offload=*/false,
                mem_size, /*n_seq_max=*/2, n_rs_seq,
                /*gdn_replay_req=*/false, /*filter=*/nullptr);

        std::vector<llama_ubatch> ubatches;   // empty: is_full == false
        recr = new llama_memory_recurrent_context(mem, std::move(ubatches));

        // The context was built with an EMPTY ubatch vector, so is_full is
        // false and get_n_rs() would be mem->n, which the cache never sets
        // outside find_slot() -- it stays 0. A zero-length s_copy makes
        // ggml_view_1d(s_copy, 1, ...) trip ggml.c:1867
        // (data_size + view_offs <= ggml_nbytes(view_src) with nbytes == 0),
        // and can_reuse() would compare a 0-length tensor against a non-zero
        // n_rs. A fixture that owns the whole cache holds mem_size cells, which
        // is exactly what the full-cache context reports, so state it here.
        mem->n = mem_size;

        ggml_init_params gp = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 8 + 4096,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx0 = ggml_init(gp);
        if (ctx0 != nullptr) {
            const int64_t n_rs    = recr->get_n_rs();
            const int64_t n_seqs_ = (int64_t) n_seqs;

            row     = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
            s_copy  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
            s_main  = ggml_view_1d(ctx0, s_copy, n_seqs_, 0);
            s_extra = ggml_view_1d(ctx0, s_copy, n_rs - n_seqs_, n_seqs_ * s_copy->nb[0]);
            ggml_set_input(row);
            ggml_set_input(s_copy);
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx0, ggml_backend_cpu_buffer_type());
        }

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

// The relay lane's recurrent input, at the shape build_rs_inp_impl() gives it
// (src/llama-graph.cpp:4273-4294). `armed` is llm_graph_params::rs_row_armed.
std::unique_ptr<llm_graph_input_rs> make_rs_input(fixture & f, bool armed) {
    auto inp = std::make_unique<llm_graph_input_rs>(f.recr);
    inp->s_copy       = f.s_copy;
    inp->s_copy_main  = f.s_main;
    inp->s_copy_extra = f.s_extra;
    inp->head         = f.recr->get_head();
    inp->rs_z         = f.recr->get_rs_z();
    inp->s_zero       = armed ? f.row : nullptr;
    inp->replay_len   = 0;
    return inp;
}

// llm_graph_result::can_reuse() over one input, which is the whole decision on
// the relay lane (no binder). `params` carries the ubatch and the runtime
// widths the caller wants to test.
bool reuse_ok(llm_graph_input_rs * inp, const llm_graph_params & params) {
    return inp->can_reuse(params);
}

} // namespace

int main() {
    llama_backend_init();

    // ---- R1: relay form, rs_z moves, one graph ----------------------------
    bool r1_first  = false;
    bool r1_second = false;
    int32_t r1_row_first  = -99;
    int32_t r1_row_second = -99;
    {
        fixture f(/*mem_size=*/4, /*n_rs_seq=*/2, /*n_seqs=*/1);
        if (!f.ok()) {
            std::fprintf(stderr, "FAIL: cannot build the R1 fixture\n");
            llama_backend_free();
            return 1;
        }
        // the live cell the graph is built at; it must be set BEFORE the
        // input exists, because inp.rs_z caches the build-time value
        f.mem->rs_z = 1;
        auto inp = make_rs_input(f, /*armed=*/true);

        llm_graph_params p = {};
        p.mctx    = f.recr;
        p.ubatch  = f.ub;
        p.n_outputs = 1;

        ((int32_t *) f.row->data)[0] = -7;
        inp->set_input(&f.ub);
        r1_row_first = ((const int32_t *) f.row->data)[0];
        r1_first = reuse_ok(inp.get(), p);
        CHECK(r1_first, "R1: can_reuse() refused the relay form at rs_z=1");

        // move rs_z; the SAME input object must still be reusable, and the row
        // must follow
        f.mem->rs_z = 3;
        inp->set_input(&f.ub);
        r1_row_second = ((const int32_t *) f.row->data)[0];
        r1_second = reuse_ok(inp.get(), p);
        CHECK(r1_second,
              "R1: can_reuse() refused the relay form after the rs_z move "
              "(rs_z is a reuse key again)");
        CHECK(r1_row_first != r1_row_second,
              "R1: the row did not follow rs_z (%d -> %d); the arm is vacuous",
              r1_row_first, r1_row_second);
        // with n_rs_seq == 2 and rs_idx == 0 the row is 0*mem_size + rs_z
        CHECK(r1_row_second == 3,
              "R1: the row after the move is %d, expected 3", r1_row_second);
    }

    // ---- R3: n_tokens moves -> refused by allow_reuse ---------------------
    bool r3_input_reuse = false;
    bool r3_params_refused = false;
    bool r3_params_control = false;
    {
        fixture f(/*mem_size=*/4, /*n_rs_seq=*/2, /*n_seqs=*/1);
        if (!f.ok()) {
            std::fprintf(stderr, "FAIL: cannot build the R3 fixture\n");
            llama_backend_free();
            return 1;
        }
        f.mem->rs_z = 1;   // build-time value, set before the input exists
        auto inp = make_rs_input(f, /*armed=*/true);

        llm_graph_params p = {};
        p.mctx   = f.recr;
        p.ubatch = f.ub;
        p.n_outputs = 1;

        inp->set_input(&f.ub);

        // the same ubatch: reusable (the control)
        CHECK(reuse_ok(inp.get(), p), "R3: the control (same ubatch) was refused");

        // llm_graph_input_rs::can_reuse() compares n_seqs and the copy widths,
        // not n_tokens (src/llama-graph.cpp:596-600), so the input alone
        // accepts a moved n_tokens. The refusal lives one level up, in
        // llm_graph_params::allow_reuse() (src/llama-graph.h:931), which
        // llm_graph_result::can_reuse() calls first
        // (src/llama-graph.cpp:1762). Assert both halves and name which is
        // which.
        llm_graph_params p2 = p;
        p2.ubatch.n_tokens = p.ubatch.n_tokens + 1;
        r3_input_reuse = reuse_ok(inp.get(), p2);
        CHECK(r3_input_reuse,
              "R3: the input-level can_reuse() refused a moved n_tokens; it must not -- "
              "the n_tokens key lives in llm_graph_params::allow_reuse()");

        llm_graph_params a = {};
        llm_graph_params b = {};
        a.ubatch = f.ub;
        b.ubatch = f.ub;
        b.ubatch.n_tokens = a.ubatch.n_tokens + 1;
        r3_params_refused = !a.allow_reuse(b, /*ignore_runtime=*/true);
        CHECK(r3_params_refused,
              "R3: allow_reuse() accepted a moved ubatch.n_tokens; the n_tokens key is gone");
        llm_graph_params c = {};
        c.ubatch = f.ub;
        r3_params_control = a.allow_reuse(c, /*ignore_runtime=*/true);
        CHECK(r3_params_control,
              "R3: allow_reuse() refused an identical ubatch; the control is broken");
    }

    // ---- R4: n_outputs moves -> llm_graph_input_out_ids::can_reuse --------
    bool r4_moved    = false;
    bool r4_unchanged= false;
    {
        // llm_graph_input_out_ids::can_reuse() is bound_outputs != n_outputs
        // (src/llama-graph.cpp:389-394), and union_bind_runtime() refreshes
        // bound_outputs from out_ids->ne[0] (:396-399). It needs no context and
        // no backend: the input reads only its own tensor.
        llm_graph_input_out_ids out(llama_hparams{}, llama_cparams{}, /*n_outputs=*/4);

        ggml_init_params gp = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 4 + 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(gp);
        if (ctx == nullptr) {
            std::fprintf(stderr, "FAIL: R4: cannot allocate a ggml context\n");
            llama_backend_free();
            return 1;
        }
        out.out_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
        out.union_bind_runtime();   // bound_outputs = out_ids->ne[0] = 4

        llm_graph_params p = {};
        p.n_outputs = 4;
        r4_unchanged = out.can_reuse(p);
        CHECK(r4_unchanged,
              "R4: out_ids refused an unchanged n_outputs; the control is broken");

        p.n_outputs = 5;
        r4_moved = out.can_reuse(p);
        CHECK(!r4_moved,
              "R4: out_ids REUSED a graph across an n_outputs move; the n_out key is gone");

        ggml_free(ctx);
    }

    // ---- R5: control -- the rs_z key forced back on is refused ------------
    bool r5_reuse = false;
    {
        fixture f(/*mem_size=*/4, /*n_rs_seq=*/2, /*n_seqs=*/1);
        if (!f.ok()) {
            std::fprintf(stderr, "FAIL: cannot build the R5 fixture\n");
            llama_backend_free();
            return 1;
        }
        // s_zero nulled: the stock form, whose zero node is a view at that
        // offset, so rs_z IS a key (src/llama-graph.cpp:606-608). Built at
        // rs_z = 1 BEFORE the input exists, so the unchanged control below
        // compares the cached 1 against the live 1 and the moved check compares
        // that same cached 1 against 3.
        f.mem->rs_z = 1;
        auto inp = make_rs_input(f, /*armed=*/false);

        llm_graph_params p = {};
        p.mctx   = f.recr;
        p.ubatch = f.ub;
        p.n_outputs = 1;

        inp->set_input(&f.ub);
        CHECK(inp->can_reuse(p), "R5: the stock form refused an unchanged rs_z; the control is broken");

        f.mem->rs_z = 3;
        inp->set_input(&f.ub);
        r5_reuse = inp->can_reuse(p);
        CHECK(!r5_reuse,
              "R5: the stock form REUSED across an rs_z move; without this control "
              "R1 could pass because can_reuse() never refuses anything");
    }

    std::printf("RSZ_RELAY_ONEGRAPH_STATS r1_reuse=%d/%d r1_row=%d->%d "
                "r3_input_reuse=%d r3_params_refused=%d r3_params_control=%d "
                "r4_reuse_unchanged=%d r4_reuse_moved=%d r5_control_reuse=%d\n",
                int(r1_first), int(r1_second), r1_row_first, r1_row_second,
                int(r3_input_reuse), int(r3_params_refused), int(r3_params_control),
                int(r4_unchanged), int(r4_moved), int(r5_reuse));

    llama_backend_free();

    if (g_checks_run == 0) {
        std::fprintf(stderr, "RSZ_RELAY_ONEGRAPH_TEST UNPROVEN reason=no_check_ran checks=0\n");
        return 2;
    }
    if (g_fail != 0) {
        std::printf("RSZ_RELAY_ONEGRAPH_TEST FAIL %d\n", g_fail);
        return 1;
    }
    std::printf("RSZ_RELAY_ONEGRAPH_TEST PASS\n");
    return 0;
}
