#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

//
// test-rsz-selection-write
//
// Pins the contract of the ONE definition of the rs_z selection write,
// llm_graph_input_rs::set_input() (src/llama-graph.cpp:560), through the
// helper itself. No model file, no llama_context, no backend device: the test
// builds a minimal llama_memory_recurrent (one layer, a few cells) on the CPU
// buffer type and a real llama_memory_recurrent_context over it, so get_rs_z()
// and get_rs_zero_row() return values the test drives.
//
// The write used to live in set_rs_z_input(); it is now the tail of
// set_input(): union_rsz_bind_index() writes the ROW the gathers read
// (rs_idx[seq]*mem_size + rs_z) into the I32 [1] tensor s_zero, and returns
// false when the ubatch's sequences disagree on it.
//
// ARMS (each a different path; each asserts the others' observable is absent)
//
//   (i)   s_zero == nullptr (stock build) is a no-op: the cached rs_z is left
//         alone, no context is needed, and the tensor stays null.
//         Catches: a write that runs unconditionally and would need a row on
//         the stock path.
//
//   (ii)  s_zero != nullptr receives the live row, and the cached rs_z
//         follows the live cell. With n_rs_seq == 0 the row IS the cell, so
//         this arm also pins "group 0 is the identity".
//         Catches: a write that copies a stale or build-time value.
//
//   (iii) a second call with a DIFFERENT live rs_z rewrites the row. This is
//         the property the hybrid defect violated: the hybrid set_input()
//         sites reused the graph but never rewrote the row, so it kept the
//         previous value.
//         Catches: a write that happens once per graph rather than once per
//         set_inputs().
//
//   (iv)  with n_rs_seq == 2 and rs_idx[0] == 2 the row is
//         2*mem_size + rs_z, not rs_z.
//         Catches: the pre-merge binder value (the CELL index), which names a
//         row nobody reads under the ring.
//
//   (v)   a MIXED-depth ubatch (rs_idx[0] != rs_idx[1]) is refused:
//         rs_zero_row_refused is set, the row keeps its seeded value, and
//         can_reuse() returns false.
//         Catches: a bind that writes one sequence's row and lets the batch
//         compute with it for both; and a refusal that is recorded but not
//         acted on (can_reuse() must refuse).
//
//   (vi)  a null row on a LIVE context is still a no-op: the cached rs_z is
//         not taken from the context and the tensor keeps its value.
//         Catches: a write that dereferences s_zero without the null check.
//
// ENVIRONMENT
//
//   This test reads no environment variable and writes none. The fixture's
//   memory reads LLAMA_GDN_REPLAY (src/llama-memory-recurrent.cpp:46-48), but
//   only when hparams.n_embd_s_ingredient() > 0; the fixture leaves
//   ssm_d_inner == 0, so that product is 0 and gdn_replay stays false whatever
//   the caller exported. Asserted in arm (ii), not assumed.
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
// virtual loaders are never called because nothing is loaded. The parameter
// type is spelled `struct llama_model_loader` exactly as the forward
// declaration in llama-model.h does, and the definitions are out of line so
// the incomplete type is never dereferenced here.
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

// one layer, `mem_size` cells, an explicit ring depth. The graph tensors live
// in a no_alloc=true context with a real CPU buffer, because the production
// bind asserts ggml_backend_buffer_is_host(s_zero->buffer)
// (llama-graph.cpp:542).
//
// s_copy / s_copy_main / s_copy_extra are REAL tensors, sized exactly as
// build_rs_inp_impl() sizes them (llama-graph.cpp:4274-4283): s_copy is
// [n_rs], main is its [n_seqs] prefix, extra is its [n_rs - n_seqs] tail.
// They must exist because llm_graph_input_rs::can_reuse() dereferences all
// three (llama-graph.cpp:594-596); passing nullptr there is a null
// dereference, and can_reuse() is exactly the reuse decision this test pins.
// They are all nullptr-free here but s_copy itself is left unwired in
// set_input() (see run_set_input), so the copy loop that CONSUMES rs_idx does
// not run and the rs_idx values stay readable.
struct fixture {
    stub_model              * model  = nullptr;
    llama_memory_recurrent  * mem    = nullptr;
    llama_memory_recurrent_context * recr = nullptr;
    ggml_context            * ctx0   = nullptr;   // graph tensors, no_alloc
    ggml_backend_buffer_t     buf    = nullptr;   // CPU buffer over ctx0
    ggml_tensor             * row    = nullptr;   // I32 [1], the zero-row index
    ggml_tensor             * s_copy = nullptr;   // I32 [n_rs]
    ggml_tensor             * s_main = nullptr;   // I32 [n_seqs]
    ggml_tensor             * s_extra= nullptr;   // I32 [n_rs - n_seqs]

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
        model->hparams.n_embd_r_impl = 8;   // explicit conv-state width
        model->hparams.ssm_d_state   = 4;   // n_embd_s() = 4 * ssm_d_inner
        model->hparams.ssm_d_inner   = 0;   // keeps n_embd_s_ingredient() == 0

        mem = new llama_memory_recurrent(*model,
                GGML_TYPE_F32, GGML_TYPE_F32, /*offload=*/false,
                mem_size, /*n_seq_max=*/2, n_rs_seq,
                /*gdn_replay_req=*/false, /*filter=*/nullptr);

        std::vector<llama_ubatch> ubatches;   // empty: is_full == false
        recr = new llama_memory_recurrent_context(mem, std::move(ubatches));

        ggml_init_params gp = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 8 + 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx0 = ggml_init(gp);
        if (ctx0 != nullptr) {
            // The context was built with an EMPTY ubatch vector, so is_full is
            // false and get_n_rs() is mem->n, which the cache only sets inside
            // find_slot(). It is 0 here, which would make s_copy a zero-byte
            // tensor and ggml_view_1d(s_copy, 1, 0) trip ggml.c:1867
            // (data_size + view_offs <= ggml_nbytes(view_src) with nbytes 0).
            // This fixture owns the whole cache, so it holds mem_size cells --
            // exactly what the full-cache context reports.
            mem->n = mem_size;

            // the live recurrent state length the graph would be built at
            const int64_t n_rs   = recr->get_n_rs();
            const int64_t n_seqs_ = (int64_t) n_seqs;

            row     = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
            s_copy  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
            s_main  = ggml_view_1d(ctx0, s_copy, n_seqs_, 0);
            s_extra = ggml_view_1d(ctx0, s_copy, n_rs - n_seqs_, n_seqs_ * s_copy->nb[0]);
            ggml_set_input(row);
            ggml_set_input(s_copy);
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx0, ggml_backend_cpu_buffer_type());
        }

        // the ubatch's seq_id pointers point into `seqs`, which outlives them
        // (the same pattern llama-context.cpp:6806-6808 uses for its own
        // reserve ubatch)
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

// the value the row holds, or a sentinel when there is no row
int32_t row_value(const ggml_tensor * t) {
    return t == nullptr ? std::numeric_limits<int32_t>::min() : ((const int32_t *) t->data)[0];
}

// one set_input() pass. `row` is the tensor to wire as s_zero (nullptr = the
// stock build); `copy`/`copy_main`/`copy_extra` are the real copy tensors
// can_reuse() reads. Returns what the caller can read back.
//
// s_copy is deliberately NOT wired into the input (`inp.s_copy = nullptr`):
// the copy loop is the one consumer of rs_idx (llama-memory-recurrent.cpp:
// 1482-1501), so wiring it would consume the depth the test is driving and
// the next get_rs_zero_row() would read 0. The tensors still exist and are
// still handed to can_reuse(), which is what needs them.
struct pass {
    int32_t row       = 0;
    bool    refused   = false;
    bool    reuse_ok  = false;
    int32_t cached_rs = 0;
};

struct copy_tensors {
    ggml_tensor * s_copy  = nullptr;
    ggml_tensor * s_main  = nullptr;
    ggml_tensor * s_extra = nullptr;
};

pass run_set_input(llama_memory_recurrent_context * recr, const llama_ubatch * ub,
                   ggml_tensor * row, int32_t seed, int32_t cached_rs_z,
                   const copy_tensors & copy) {
    pass p;

    llm_graph_input_rs inp(recr);
    inp.s_copy       = nullptr;   // the copy loop is skipped: it consumes rs_idx
    inp.s_copy_main  = copy.s_main;
    inp.s_copy_extra = copy.s_extra;
    inp.head         = recr != nullptr ? recr->get_head() : 0;
    inp.rs_z         = cached_rs_z;
    inp.s_zero       = row;
    inp.replay_len   = 0;

    if (row != nullptr) {
        ((int32_t *) row->data)[0] = seed;
    }
    inp.set_input(ub);

    p.row       = row_value(row);
    p.refused   = inp.rs_zero_row_refused;
    p.cached_rs = inp.rs_z;

    // can_reuse() dereferences s_copy (llama-graph.cpp:594). The input's own
    // s_copy is null by design, so the width check is made against the real
    // tensor here: the production input would have it wired, and the tensor
    // is sized exactly as build_rs_inp_impl() sizes it.
    inp.s_copy = copy.s_copy;
    llm_graph_params params = {};
    params.mctx   = recr;
    params.ubatch = *ub;
    p.reuse_ok = inp.can_reuse(params);
    return p;
}

} // namespace

int main() {
    llama_backend_init();

    // ---- (i) stock build: s_zero == nullptr is a no-op --------------------
    {
        // a null mctx is legal for this arm: the bind returns before it reads
        // the context
        llm_graph_input_rs inp(nullptr);
        inp.s_copy       = nullptr;
        inp.s_copy_main  = nullptr;
        inp.s_copy_extra = nullptr;
        inp.head         = 0;
        inp.rs_z         = -2;               // "never bound" sentinel
        inp.s_zero       = nullptr;
        inp.replay_len   = 0;

        llama_batch_allocr balloc(1);
        const llama_ubatch ub = balloc.ubatch_reserve(1, 1);
        inp.set_input(&ub);

        CHECK(inp.rs_z == -2,
              "(i) null row must not touch the cached rs_z, got %d", inp.rs_z);
        CHECK(inp.s_zero == nullptr, "(i) null row must stay null");
        CHECK(!inp.rs_zero_row_refused,
              "(i) a null row is not a refusal: nothing was asked for");
    }

    // ---- (ii) the row receives the live row ------------------------------
    {
        const uint32_t mem_size = 4;
        fixture f(mem_size, /*n_rs_seq=*/0, /*n_seqs=*/1);
        if (!f.ok()) {
            std::fprintf(stderr, "FAIL: cannot build the recurrent fixture\n");
            llama_backend_free();
            return 1;
        }
        CHECK(!f.recr->gdn_replay(),
              "(ii) gdn_replay is on; LLAMA_GDN_REPLAY must not change this fixture");

        // the live selection the bind must copy
        f.mem->rs_z = 3;
        CHECK(f.recr->get_rs_z() == 3,
              "(ii) fixture context reports rs_z %d, expected 3 (is_full must be false)",
              f.recr->get_rs_z());

        const copy_tensors cp{f.s_copy, f.s_main, f.s_extra};
        const pass p = run_set_input(f.recr, &f.ub, f.row, /*seed=*/-7, /*cached_rs_z=*/-2, cp);
        CHECK(p.row == 3,
              "(ii) row holds %d after set_input(), expected the live rs_z 3", p.row);
        CHECK(p.cached_rs == 3,
              "(ii) cached rs_z is %d after set_input(), expected 3", p.cached_rs);
        CHECK(!p.refused, "(ii) the bind refused a single-depth ubatch");
        CHECK(p.reuse_ok, "(ii) can_reuse() refused a graph whose row was bound");

        // ---- (iii) a moved rs_z rewrites the row -------------------------
        //
        // This is the assertion the hybrid defect violated: the hybrid
        // set_input() sites copied s_copy and never wrote the row, so a moved
        // rs_z left the previous value in place while can_reuse() accepted the
        // graph. Here the same shape of call is made twice with a different
        // live rs_z and the row must follow.
        f.mem->rs_z = 1;
        const pass q = run_set_input(f.recr, &f.ub, f.row, /*seed=*/3, /*cached_rs_z=*/3, cp);
        CHECK(q.row == 1,
              "(iii) row holds %d after the second write, expected the moved rs_z 1 "
              "(a stale row is the hybrid defect)", q.row);
        CHECK(q.cached_rs == 1,
              "(iii) cached rs_z is %d after the second write, expected 1", q.cached_rs);
        CHECK(q.row != 3,
              "(iii) the second write left the first arm's value 3 in place");

        // ---- (vi) null row on a LIVE context is still a no-op ------------
        f.mem->rs_z = 2;
        const pass r = run_set_input(f.recr, &f.ub, /*row=*/nullptr,
                                     /*seed=*/0, /*cached_rs_z=*/1, cp);
        CHECK(r.cached_rs == 1,
              "(vi) null row on a live context changed the cached rs_z to %d", r.cached_rs);
        CHECK(row_value(f.row) == 1,
              "(vi) null row on a live context wrote the row (%d)", row_value(f.row));
        CHECK(!r.refused, "(vi) a null row is not a refusal");
    }

    // ---- (iv) n_rs_seq == 2: the row is rs_idx*mem_size + rs_z ------------
    {
        const uint32_t mem_size = 4;
        const uint32_t n_rs_seq = 2;
        fixture f(mem_size, n_rs_seq, /*n_seqs=*/1);
        if (!f.ok()) {
            std::fprintf(stderr, "FAIL: cannot build the ring fixture\n");
            llama_backend_free();
            return 1;
        }

        CHECK(f.recr->get_n_rs_seq() == n_rs_seq,
              "(iv) cache reports n_rs_seq %u, expected %u",
              f.recr->get_n_rs_seq(), n_rs_seq);

        const copy_tensors cp{f.s_copy, f.s_main, f.s_extra};

        f.mem->rs_z = 1;
        for (uint32_t depth = 0; depth <= n_rs_seq; ++depth) {
            f.mem->set_rs_idx(0, depth);
            CHECK(f.mem->rs_idx[0] == depth,
                  "(iv) set_rs_idx(0, %u) left rs_idx %u", depth, f.mem->rs_idx[0]);

            const int32_t expect = (int32_t) (depth * mem_size) + 1;
            const pass p = run_set_input(f.recr, &f.ub, f.row, /*seed=*/-7,
                                         /*cached_rs_z=*/f.recr->get_rs_z(), cp);
            CHECK(!p.refused, "(iv) the bind refused a single-depth ubatch");
            CHECK(p.row == expect,
                  "(iv) depth %u bound row %d, expected rs_idx*mem_size + rs_z = %d "
                  "(the cell index 1 is the pre-merge value)", depth, p.row, expect);
            CHECK(p.row == f.recr->get_rs_zero_row(0),
                  "(iv) the written row %d != get_rs_zero_row(0) %d",
                  p.row, f.recr->get_rs_zero_row(0));
            if (depth > 0) {
                CHECK(p.row != 1,
                      "(iv) depth %u bound the cell index, not a group row", depth);
            }
        }
    }

    // ---- (v) a mixed-depth ubatch is refused, and the graph is not reused -
    {
        const uint32_t mem_size = 4;
        const uint32_t n_rs_seq = 2;
        fixture f(mem_size, n_rs_seq, /*n_seqs=*/2);
        if (!f.ok()) {
            std::fprintf(stderr, "FAIL: cannot build the mixed-depth fixture\n");
            llama_backend_free();
            return 1;
        }

        const copy_tensors cp{f.s_copy, f.s_main, f.s_extra};

        f.mem->rs_z = 1;
        f.mem->set_rs_idx(0, 2);
        f.mem->set_rs_idx(1, 0);
        CHECK(f.recr->get_rs_zero_row(0) == 9,
              "(v) seq 0 row %d, expected 2*4+1 = 9", f.recr->get_rs_zero_row(0));
        CHECK(f.recr->get_rs_zero_row(1) == 1,
              "(v) seq 1 row %d, expected 0*4+1 = 1", f.recr->get_rs_zero_row(1));

        const pass p = run_set_input(f.recr, &f.ub, f.row, /*seed=*/-7,
                                     /*cached_rs_z=*/f.recr->get_rs_z(), cp);
        CHECK(p.refused,
              "(v) set_input() did not record a refusal for a mixed-depth ubatch");
        CHECK(p.row == -7,
              "(v) the refused bind wrote row %d, expected the seed -7 to survive", p.row);
        CHECK(!p.reuse_ok,
              "(v) can_reuse() accepted a graph whose zero row could not be bound");

        // the refusal is the mixed batch's, not a permanent state: one depth
        // binds and reuses again
        f.mem->set_rs_idx(1, 2);
        const pass q = run_set_input(f.recr, &f.ub, f.row, /*seed=*/-7,
                                     /*cached_rs_z=*/f.recr->get_rs_z(), cp);
        CHECK(!q.refused, "(v) the bind refused a single-depth ubatch");
        CHECK(q.row == 9, "(v) the bound row is %d, expected 9", q.row);
        CHECK(q.reuse_ok, "(v) can_reuse() refused a graph whose row was bound");
    }

    llama_backend_free();

    if (g_checks_run == 0) {
        std::fprintf(stderr, "RSZ_SELECTION_WRITE_TEST UNPROVEN reason=no_check_ran checks=0\n");
        return 2;
    }
    if (g_fail != 0) {
        std::printf("RSZ_SELECTION_WRITE_TEST FAIL %d\n", g_fail);
        return 1;
    }
    std::printf("RSZ_SELECTION_WRITE_TEST PASS\n");
    return 0;
}
