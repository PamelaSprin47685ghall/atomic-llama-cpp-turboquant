#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"
#include "llama-union-width-table.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>   // unsetenv: the fixture suppresses LLAMA_GDN_REPLAY
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

//
// test-rsz-key-injection
//
// The ticket's literal requirement: putting rs_z into a reuse key or a
// topology key must be DETECTED as a failure. This file is that detector. It
// fails the moment anyone re-introduces rs_z as a key, either by removing the
// `if (s_zero == nullptr)` guard in the four can_reuse() sites
// (src/llama-graph.cpp:606-608, :1414-1416, :1482-1484, :1595-1597) or by
// letting the zero-row index's VALUE reach the topology hash.
//
// No model file, no llama_context, no backend device. The reuse arms drive the
// production llm_graph_input_rs::set_input() / can_reuse() over a real
// llama_memory_recurrent_context; the hash arms drive the production
// llama_union_width_table over a graph shaped like build_rs()'s.
//
// ARMS
//
//   K1  armed (s_zero wired), rs_z moves between two set_input()/can_reuse()
//       pairs. The row tensor must be REWRITTEN on the second call and
//       can_reuse() must be TRUE both times. This is the property the ticket
//       wants: a moved rs_z is an input write, not a rebuild.
//       Catches: a bind that runs once per graph instead of once per
//       set_inputs(); a can_reuse() that reads rs_z even when the row is
//       wired.
//
//   K2  the injection itself: the same two calls with s_zero == nullptr (the
//       stock form). can_reuse() must be FALSE when rs_z moved and TRUE when
//       it did not. This is the stock key, and it is the arm that fails if the
//       guard is removed -- a graph whose row is not wired MUST key on rs_z,
//       because its zero node is a view at that offset.
//       Catches: "rs_z is not a reuse key anywhere", which would be wrong:
//       the stock lane bakes it into the topology.
//
//   K3a the zero-row index's VALUE is not a shape. Two graphs identical except
//       for the int32 written into the index tensor hash to the SAME value.
//       Catches: hashing the index tensor's data, or naming the row in the
//       node set.
//
//   K3b positive control: the state tensor's row count is a shape. Changing it
//       must move the hash. Catches: a hash that ignores the tensor it is
//       supposed to describe, which would make K3a pass for the wrong reason.
//
//   K3c the table excludes exactly the dimension it names, and nothing more.
//       union_rs_copy_dst carries n_rs in ne[1] and the table names that
//       dimension (llama-union-width-table.h:169-176), so moving ne[1] must
//       NOT move the hash. A node the table does not name (union_rs_plain)
//       must move it. Catches: a table that excludes too much (the hash stops
//       being a detector) or too little (a runtime move would rebuild).
//
// Exits 0 on pass, 1 on any failed check. Prints exactly one
// RSZ_KEY_INJECTION_STATS line and then RSZ_KEY_INJECTION_TEST PASS|FAIL n.

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

// A model that exists only to carry hparams. Same shape as
// tests/test-rsz-selection-write.cpp: llama_model's constructor and destructor
// are safe on an empty impl, and no loader is ever called because nothing is
// loaded.
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

// One recurrent layer, `mem_size` cells, an explicit ring depth. The graph
// tensors live in a no_alloc ggml context with a real CPU buffer, because
// union_rsz_bind_index() asserts ggml_backend_buffer_is_host() on the index
// tensor (src/llama-graph.cpp:542).
struct fixture {
    stub_model              * model  = nullptr;
    llama_memory_recurrent  * mem    = nullptr;
    llama_memory_recurrent_context * recr = nullptr;
    ggml_context            * ctx0   = nullptr;
    ggml_backend_buffer_t     buf    = nullptr;
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
        model->hparams.n_embd_r_impl = 8;   // n_embd_r() = 8 (explicit override)
        model->hparams.ssm_d_state   = 4;
        // A REAL state tensor: n_embd_s() = ssm_d_state * ssm_d_inner = 32
        // (src/llama-hparams.cpp:228). With ssm_d_inner 0 the tensor is
        // zero-element, the cache's own allocation makes a 0-byte buffer, and
        // any build_rs() on it aborts at ggml_reshape_2d's element-count
        // assert. ssm_d_conv is set so n_embd_r() stays non-zero even if the
        // n_embd_r_impl override is ever removed (src/llama-hparams.cpp:211).
        model->hparams.ssm_d_inner   = 8;
        model->hparams.ssm_d_conv    = 4;   // keeps n_embd_r() non-zero if the override goes

        // The fixture owns LLAMA_GDN_REPLAY inside its own process: setenv and
        // unsetenv are process-local, and this file never reads the variable
        // for its own purposes. With ssm_d_inner 8 the ingredient count is
        // 4 * 8 = 32 > 0, so the env var WOULD enable gdn_replay
        // (src/llama-memory-recurrent.cpp:44-48) and the state tensor would
        // change shape. Suppress it explicitly instead of relying on a zero
        // ingredient count; CHECK(!recr->gdn_replay()) in the arms proves it.
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

        // the ubatch's seq_id pointers point into `seqs`, which outlives them
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

int32_t row_value(const ggml_tensor * t) {
    return t == nullptr ? std::numeric_limits<int32_t>::min() : ((const int32_t *) t->data)[0];
}

// One BUILT graph (an llm_graph_input_rs) held across a runtime-parameter
// move. Holding it is the whole point: the reuse key is a property of the
// graph that was built, compared against the live state. Creating a fresh
// input per call and re-reading rs_z would make both arms vacuous -- an armed
// graph would look reusable because it was never asked about a different
// value, and the stock form would look reusable for the same reason.
struct rs_probe {
    fixture &                    f;
    llm_graph_input_rs           inp;

    rs_probe(fixture & f_, bool armed) : f(f_), inp(f_.recr) {
        inp.s_copy       = nullptr;   // the copy loop is skipped: s_copy consumes rs_idx
        inp.s_copy_main  = nullptr;
        inp.s_copy_extra = nullptr;
        inp.head         = f.recr->get_head();
        inp.rs_z         = f.recr->get_rs_z();   // the BUILD-time value
        inp.s_zero       = armed ? f.row : nullptr;
        inp.replay_len   = 0;
    }

    // one set_inputs() pass on this graph; returns the row it left behind
    int32_t write(int32_t seed) {
        ((int32_t *) f.row->data)[0] = seed;
        inp.set_input(&f.ub);
        return row_value(f.row);
    }

    bool refused() const { return inp.rs_zero_row_refused; }

    // can_reuse() DEREFERENCES s_copy / s_copy_main / s_copy_extra
    // (src/llama-graph.cpp:594-604), so they are wired here, not during
    // set_input(), where the copy loop would consume rs_idx.
    bool reusable() {
        inp.s_copy       = f.s_copy;
        inp.s_copy_main  = f.s_main;
        inp.s_copy_extra = f.s_extra;
        llm_graph_params params = {};
        params.mctx     = f.recr;
        params.ubatch   = f.ub;
        params.n_outputs = 1;
        return inp.can_reuse(params);
    }
};

// ---------------------------------------------------------------------------
// K3: the synthetic graph build_rs() produces, at the shape the width table
// expects. Node names are the ones llama-union-width-table.h keys on.
// ---------------------------------------------------------------------------

struct synth {
    ggml_context * ctx  = nullptr;
    ggml_tensor  * idx  = nullptr;   // I32 [1]  the zero-row index (union_rs_zero_index)
    ggml_tensor  * state= nullptr;   // F32 [state_size, rows]
    ggml_tensor  * zero = nullptr;   // ggml_map_custom2_inplace(state, idx)  union_rs_zero_row
    ggml_tensor  * st2  = nullptr;   // reshape_2d(zero, state_size, rows)
    ggml_tensor  * copy = nullptr;   // I32 [n_rs]  union_rs_copy
    ggml_tensor  * extra= nullptr;   // view_1d(copy, n_rs - n_seqs, ...)  union_rs_copy_extra
    ggml_tensor  * dst  = nullptr;   // view_2d(state, state_size, n_rs - n_seqs, ...)  union_rs_copy_dst
    ggml_tensor  * head = nullptr;   // view_1d(state, state_size, ...)  union_rs_head_view
    ggml_tensor  * plain= nullptr;   // F32 [state_size]  union_rs_plain: NOT a runtime root
    std::vector<ggml_tensor *> nodes;

    int64_t state_size = 8;
    int64_t rows       = 4;    // mem_size * (1 + n_rs_seq)
    int64_t n_rs       = 4;
    int64_t n_seqs     = 1;
    int64_t head_rows  = 1;

    bool build() {
        // no_alloc = false: every tensor gets inline host memory in this
        // context, so idx->data is writable with no backend allocator at all.
        // The hash reads only op / flags / op_params / ne / name / src[] /
        // view_src, never `buffer` or `data`, so no buffer type is involved.
        // Data actually allocated: idx 4 B, state 8*4*4 = 128 B, copy 4*4 = 16 B,
        // plain 8*4 = 32 B; the reshape and the three views are views of these.
        // The pool is far larger than that on purpose.
        ggml_init_params gp = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 32 + (1 << 16),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        ctx = ggml_init(gp);
        if (ctx == nullptr) {
            return false;
        }
        idx   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_name(idx, "union_rs_zero_index");
        ggml_set_input(idx);

        state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, state_size, rows);
        ggml_set_name(state, "state");

        // the production zero-row op: in-place over the state, reading the
        // index as its second source (src/llama-graph.cpp:4228-4230). The
        // callback is never executed here; only the node is hashed.
        zero = ggml_map_custom2_inplace(ctx, state, idx, nullptr, GGML_N_TASKS_MAX, nullptr);
        ggml_set_name(zero, "union_rs_zero_row");

        st2 = ggml_reshape_2d(ctx, zero, state_size, rows);
        ggml_set_name(st2, "union_rs_state");

        copy = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_rs);
        ggml_set_name(copy, "union_rs_copy");
        ggml_set_input(copy);

        extra = ggml_view_1d(ctx, copy, n_rs - n_seqs, (size_t) n_seqs * copy->nb[0]);
        ggml_set_name(extra, "union_rs_copy_extra");

        dst = ggml_view_2d(ctx, state, state_size, n_rs - n_seqs, state->nb[1],
                           (size_t) n_seqs * state->nb[1]);
        ggml_set_name(dst, "union_rs_copy_dst");

        head = ggml_view_1d(ctx, state, state_size, 0);
        ggml_set_name(head, "union_rs_head_view");

        // a node the width table does NOT name: its shape is a plain graph
        // property, so any change to it must move the hash. This is the
        // falsifiability control for K3c's exclusion half.
        plain = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, state_size);
        ggml_set_name(plain, "union_rs_plain");

        // graph order, source before consumer
        nodes = { idx, state, zero, st2, copy, extra, dst, head, plain };
        return true;
    }

    ~synth() {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }

    synth() = default;
    synth(const synth &) = delete;
    synth & operator=(const synth &) = delete;
};

// Hash `nodes` under a table built at (n_kv, n_outputs, n_rs, head). The n_kv
// and n_outputs roots are absent from this graph, so a zero value means "no
// such root" (llama-union-width-table.h:90-93) and is accepted.
bool synth_hash(const synth & g, uint64_t & out) {
    llama_union_width_table table;
    // The last three arguments are the token lane's, which this graph does
    // not arm: build_n_tokens = 0 means "no token-width runtime dimension"
    // (llama-union-width-table.h), and n_pos_per_embd / n_embd_v_gqa_max are
    // read only when it is non-zero. The synthetic graph has no token-lane
    // root, so 0/1/1 is the same table the five-argument intent described.
    if (!table.build(g.nodes, /*build_n_kv=*/1, /*build_n_outputs=*/0,
                     (uint32_t) g.n_rs, (uint32_t) g.head_rows,
                     /*build_n_tokens=*/0, /*n_pos_per_embd=*/1, /*n_embd_v_gqa_max=*/1)) {
        return false;
    }
    return table.topology_hash(g.nodes, out);
}

} // namespace

int main() {
    llama_backend_init();

    // ---- K1: armed, rs_z moves, reuse must hold ---------------------------
    bool k1_reuse_first  = false;
    bool k1_reuse_second = false;
    int32_t k1_row_first  = -99;
    int32_t k1_row_second = -99;
    {
        const uint32_t mem_size = 4;
        fixture f(mem_size, /*n_rs_seq=*/0, /*n_seqs=*/1);
        if (!f.ok()) {
            std::fprintf(stderr, "FAIL: cannot build the K1 fixture\n");
            llama_backend_free();
            return 1;
        }

        // rs_z = 1: a non-zero cell, so a stale row cannot coincide with the
        // live one. The probe is built HERE, at rs_z = 1, and then held.
        f.mem->rs_z = 1;
        rs_probe p(f, /*armed=*/true);

        k1_row_first = p.write(/*seed=*/-7);
        CHECK(!p.refused(), "K1: the armed bind refused a single-sequence ubatch");
        CHECK(k1_row_first == 1, "K1: the armed bind wrote row %d, expected 1", k1_row_first);
        k1_reuse_first = p.reusable();
        CHECK(k1_reuse_first, "K1: can_reuse() refused an armed graph at rs_z=1 "
                              "(rs_z is a reuse key again)");

        // move rs_z on the live cache; the graph built at 1 must stay reusable
        // and set_input() must rewrite the row
        f.mem->rs_z = 3;
        k1_row_second = p.write(/*seed=*/-7);
        CHECK(!p.refused(), "K1: the armed bind refused after the rs_z move");
        CHECK(k1_row_second == 3,
              "K1: the row was not rewritten after the rs_z move (row=%d, expected 3): "
              "the bind runs once per graph, not once per set_inputs()", k1_row_second);
        k1_reuse_second = p.reusable();
        CHECK(k1_reuse_second,
              "K1: can_reuse() refused a graph built at rs_z=1 after the live rs_z "
              "moved to 3 (rs_z is a reuse key again)");

        CHECK(k1_row_first != k1_row_second,
              "K1: the row did not move (%d -> %d); the arm is vacuous",
              k1_row_first, k1_row_second);
    }

    // ---- K2: the injection -- stock form MUST key on rs_z ------------------
    bool k2_reuse_moved    = false;
    bool k2_reuse_unchanged= false;
    {
        const uint32_t mem_size = 4;
        fixture f(mem_size, /*n_rs_seq=*/0, /*n_seqs=*/1);
        if (!f.ok()) {
            std::fprintf(stderr, "FAIL: cannot build the K2 fixture\n");
            llama_backend_free();
            return 1;
        }

        // the same SHAPE of probe, but stock: s_zero == nullptr, so the zero
        // node is a view at the live offset (src/llama-graph.cpp:4238) and
        // rs_z is part of the graph identity.
        f.mem->rs_z = 1;
        rs_probe p(f, /*armed=*/false);
        p.write(/*seed=*/0);
        CHECK(!p.refused(), "K2: the stock bind refused a single-sequence ubatch");

        // unchanged rs_z: the stock graph is reusable. Without this control a
        // blanket refusal would make the moved-rs_z assertion below vacuous.
        k2_reuse_unchanged = p.reusable();
        CHECK(k2_reuse_unchanged,
              "K2: the stock form refused an unchanged rs_z; the control is broken, "
              "so the moved-rs_z refusal below would prove nothing");

        // moved rs_z: the SAME graph, built at 1, must be refused at live 3
        f.mem->rs_z = 3;
        k2_reuse_moved = p.reusable();
        CHECK(!k2_reuse_moved,
              "K2: the stock form REUSED a graph built at rs_z=1 with the live rs_z "
              "at 3 -- rs_z is not a reuse key where it must be one");
    }

    // ---- K3a: the index tensor's VALUE is not a shape ----------------------
    // (the arm's verdict is its CHECKs; no separate flag is kept)
    uint64_t k3a_hash_value0 = 0;
    uint64_t k3a_hash_value1 = 0;
    {
        synth g;
        if (!g.build()) {
            std::fprintf(stderr, "FAIL: cannot build the K3a graph\n");
            llama_backend_free();
            return 1;
        }
        // The synth context allocates inline host memory (no_alloc = false),
        // so idx->data is already writable: no backend allocator, no buffer,
        // and no failure branch to get wrong.
        ((int32_t *) g.idx->data)[0] = 0;
        const bool h0 = synth_hash(g, k3a_hash_value0);
        ((int32_t *) g.idx->data)[0] = 3;
        const bool h1 = synth_hash(g, k3a_hash_value1);

        CHECK(h0 && h1, "K3a: the topology hash could not be taken (h0=%d h1=%d)",
              int(h0), int(h1));
        CHECK(h0 && h1 && k3a_hash_value0 == k3a_hash_value1,
              "K3a: the zero-row index VALUE moved the topology hash "
              "(%016llx -> %016llx); the row is a runtime value, not a shape",
              (unsigned long long) k3a_hash_value0, (unsigned long long) k3a_hash_value1);
    }

    // ---- K3b: the state row count IS a shape (positive control) ------------
    // (the arm's verdict is its CHECKs; no separate flag is kept)
    uint64_t k3b_hash_base = 0;
    uint64_t k3b_hash_wider= 0;
    {
        synth g;
        if (!g.build()) {
            std::fprintf(stderr, "FAIL: cannot build the K3b graph\n");
            llama_backend_free();
            return 1;
        }
        const bool h0 = synth_hash(g, k3b_hash_base);
        // widen the state tensor: 4 -> 8 rows. The reshape and the two views
        // built on it keep their own ne[], so only the state node moves.
        g.state->ne[1] = 8;
        g.state->nb[2] = g.state->nb[1] * 8;
        g.state->nb[3] = g.state->nb[2];
        const bool h1 = synth_hash(g, k3b_hash_wider);

        CHECK(h0 && h1, "K3b: the topology hash could not be taken (h0=%d h1=%d)",
              int(h0), int(h1));
        CHECK(h0 && h1 && k3b_hash_base != k3b_hash_wider,
              "K3b: widening the state tensor did not move the topology hash "
              "(%016llx -> %016llx); the hash is not a detector and K3a proves nothing",
              (unsigned long long) k3b_hash_base, (unsigned long long) k3b_hash_wider);

    }

    // ---- K3c: the table excludes exactly the dimension it names ------------
    bool k3c_ok = false;
    {
        synth g;
        if (!g.build()) {
            std::fprintf(stderr, "FAIL: cannot build the K3c graph\n");
            llama_backend_free();
            return 1;
        }
        uint64_t base = 0;
        const bool hb = synth_hash(g, base);

        // union_rs_copy_dst carries n_rs in ne[1] and the table names that
        // dimension (llama-union-width-table.h:174-181). The table's rule is
        // ne[1] <= build_n_rs, so a smaller ne[1] is the legal runtime move
        // (the live state length can sit below the build envelope) and must NOT
        // move the hash. It was built at n_rs - n_seqs == 3, so the probe must
        // move it to a DIFFERENT in-range value; setting it to n_rs - 1 would
        // be a no-op and the arm could never fail.
        g.dst->ne[1] = 2;
        uint64_t excluded = 0;
        const bool he = synth_hash(g, excluded);
        CHECK(hb && he, "K3c: the topology hash could not be taken (base=%d excl=%d)",
              int(hb), int(he));
        CHECK(hb && he && base == excluded,
              "K3c: the runtime n_rs dimension of union_rs_copy_dst moved the hash "
              "(%016llx -> %016llx); the table does not exclude what it names",
              (unsigned long long) base, (unsigned long long) excluded);

        // a node the table does NOT name must move it. union_rs_plain is not a
        // runtime root, so its shape is a plain graph property; a hash that
        // ignored it would pass the exclusion check above for the wrong reason.
        g.plain->ne[0] = g.state_size + 1;
        g.plain->nb[1] = g.plain->nb[0] * (g.state_size + 1);
        uint64_t other = 0;
        const bool ho = synth_hash(g, other);
        CHECK(hb && ho && base != other,
              "K3c: a non-runtime node's shape did not move the hash "
              "(%016llx -> %016llx); the hash is not a detector",
              (unsigned long long) base, (unsigned long long) other);

        k3c_ok = hb && he && ho && base == excluded && base != other;
    }

    std::printf("RSZ_KEY_INJECTION_STATS k1_reuse=%d/%d k1_row=%d->%d "
                "k2_reuse_unchanged=%d k2_reuse_moved=%d "
                "k3a_value_hash=%016llx k3b_wider_hash=%016llx k3c_excluded=%d\n",
                int(k1_reuse_first), int(k1_reuse_second), k1_row_first, k1_row_second,
                int(k2_reuse_unchanged), int(k2_reuse_moved),
                (unsigned long long) k3a_hash_value0, (unsigned long long) k3b_hash_wider,
                int(k3c_ok));

    llama_backend_free();

    if (g_checks_run == 0) {
        std::fprintf(stderr, "RSZ_KEY_INJECTION_TEST UNPROVEN reason=no_check_ran checks=0\n");
        return 2;
    }
    if (g_fail != 0) {
        std::printf("RSZ_KEY_INJECTION_TEST FAIL %d\n", g_fail);
        return 1;
    }
    std::printf("RSZ_KEY_INJECTION_TEST PASS\n");
    return 0;
}
