#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-model.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

//
// test-rsz-hybrid-selection-write
//
// Drives the HYBRID set_input() sites of the rs_z runtime lane and asserts
// that each one writes the selection row. No model file, no llama_context, no
// backend device: the test builds a minimal llama_memory_hybrid (one attention
// layer, one recurrent layer, a few cells) on the CPU buffer type, takes a real
// llama_memory_hybrid_context over it, and calls set_input() on both hybrid
// input classes that wrap that context:
//
//   - llm_graph_input_mem_hybrid   (attn_kv flavour), set_input() at
//                                  src/llama-graph.cpp:1354
//   - llm_graph_input_mem_hybrid_k (attn_k  flavour), set_input() at
//                                  src/llama-graph.cpp:1443
//
// The third hybrid site, llm_graph_input_mem_hybrid_iswa::set_input()
// (src/llama-graph.cpp:1478), is NOT driven here: the lane never arms on an
// iswa context (union_is_hybrid_iswa() excludes llama_memory_hybrid_iswa
// contexts from rsz_gate_ok), so s_zero is never created for it and its bind
// is a no-op by construction. Covering it here would mean fabricating an armed
// state the production gate cannot produce.
//
// THE DEFECT THIS GUARDS
//
//   Each hybrid set_input() must write the zero row through the ONE definition
//   (union_rsz_bind_index, src/llama-graph.cpp:537, called at :1386 and :1460).
//   If a site omits the call, the row keeps the value of the previous call
//   while can_reuse() accepts the graph (the rs_z key is skipped whenever
//   s_zero != nullptr, src/llama-graph.cpp:614-616), so a moved rs_z never
//   reaches the graph. The row is seeded with a value DIFFERENT from the live
//   row before every call, so a missing write shows up as the seeded value,
//   not as a coincidence with the expected one.
//
// ARMS (each a different path; each asserts the others' observable is absent)
//
//   (1) the row receives the live row on the first call, at both sites.
//       Catches: a site that never calls the bind.
//   (2) after rs_z moves, a second call rewrites the row -- the assertion that
//       fails on a hybrid set_input() that omits the call.
//       Catches: a bind that runs once per graph instead of once per
//       set_inputs().
//   (3) the two sites agree on the row for the same live rs_z, so the two
//       flavours cannot drift apart.
//       Catches: one site being fixed and the other not.
//   (4) with n_rs_seq == 2 and rs_idx == 2 the row is 2*mem_size + rs_z.
//       Catches: the pre-merge CELL-index write on the hybrid path.
//   (5) a MIXED-depth ubatch is refused at both sites: rs_zero_row_refused is
//       set, the row keeps its seed, and can_reuse() returns false.
//       Catches: a site that asserts instead of refusing, or that records the
//       refusal but still lets the graph be reused.
//   (6) with s_zero == nullptr (stock build) the row is untouched and no value
//       is invented.
//       Catches: a bind that dereferences s_zero unconditionally.
//
// ENVIRONMENT
//
//   This test reads no environment variable and writes none. The fixture's
//   recurrent sub-cache reads LLAMA_GDN_REPLAY
//   (src/llama-memory-recurrent.cpp:46-48), but only when
//   hparams.n_embd_s_ingredient() > 0; the fixture leaves ssm_d_inner == 0, so
//   that product is 0 and gdn_replay stays false whatever the caller exported.
//   Asserted in arm (1), not assumed.
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

// A model that exists only to carry hparams and arch. llama_model's
// constructor and destructor are safe on an empty impl (llama-model.cpp:1464-1481);
// the pure virtual loaders are never called because nothing is loaded. The
// parameter type is spelled `struct llama_model_loader` exactly as the forward
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

constexpr int N_ATTN_LAYERS = 1;
constexpr int N_RS_LAYERS   = 1;
constexpr int N_LAYERS      = N_ATTN_LAYERS + N_RS_LAYERS;
constexpr int N_CTX         = 4;   // recurrent cells and KV cells
constexpr int N_SEQ_MAX     = 2;
constexpr int N_HEAD        = 2;
constexpr int N_EMBD_HEAD   = 4;

// one attention layer (layer 0) and one recurrent layer (layer 1), so the
// hybrid memory allocates both sub-caches and both filters select one layer
struct fixture {
    stub_model              * model  = nullptr;
    llama_memory_hybrid     * mem    = nullptr;
    llama_memory_hybrid_context * hyb = nullptr;
    ggml_context            * ctx0   = nullptr;   // graph tensors, no_alloc
    ggml_backend_buffer_t     buf    = nullptr;   // CPU buffer over ctx0
    ggml_tensor             * row    = nullptr;   // I32 [1], the zero-row index
    ggml_tensor             * s_copy = nullptr;   // I32 [n_rs]
    ggml_tensor             * s_main = nullptr;   // I32 [n_seqs]
    ggml_tensor             * s_extra= nullptr;   // I32 [n_rs - n_seqs]
    ggml_tensor             * k_idxs = nullptr;   // I64 [n_tokens]
    ggml_tensor             * v_idxs = nullptr;   // I64 [n_tokens]
    ggml_tensor             * kq_mask = nullptr;  // F32 [n_kv, n_tokens, 1, 1]
    uint32_t                  n_seqs_mem = 0;     // the ctor's n_seqs, for prepare()
    bool                      prepared = false;   // prepare() ran
    bool                      applied  = false;   // hyb->apply() succeeded
    uint32_t                  mask_target = 0;    // the live width the mask was built at

    llama_hparams hparams;
    llama_cparams cparams;

    // the ubatch and the seq_id storage its pointers reference
    llama_batch_allocr        balloc;
    llama_ubatch              ub;
    std::vector<llama_seq_id> seq_ids;
    std::vector<llama_seq_id> seq_id_data;
    std::vector<int32_t>      n_seq_id;
    std::vector<llama_pos>    pos;
    std::vector<llama_token>  tok;
    std::vector<int8_t>       output;
    std::vector<llama_seq_id *> seq_id_ptrs;

    // llama_model_params::devices is a borrowed, NULL-terminated pointer and
    // the model keeps it, so it must outlive the model: a member here.
    ggml_backend_dev_t devices[1] = { nullptr };

    fixture(uint32_t n_rs_seq, uint32_t n_seqs)
        : balloc(1), ub(balloc.ubatch_reserve(1, n_seqs)), n_seqs_mem(n_seqs) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.vocab_only   = true;   // no weights are ever read
        mp.devices      = devices;

        model = new stub_model(mp);

        // the attention sub-cache reads these; the recurrent one reads
        // n_embd_r()/n_embd_s(). has_kv(il) is true for every layer by
        // default; is_recr(il) reads is_recr_impl, so layer 1 is marked
        // recurrent and the default filters split the two layers.
        model->hparams.n_layer_all        = N_LAYERS;
        model->hparams.n_embd             = 8;
        model->hparams.n_embd_head_k_full = N_EMBD_HEAD;
        model->hparams.n_embd_head_v_full = N_EMBD_HEAD;
        model->hparams.n_embd_r_impl      = 8;   // explicit conv-state width
        model->hparams.ssm_d_state        = 4;   // n_embd_s() = 4 * ssm_d_inner
        model->hparams.ssm_d_inner        = 0;   // keeps n_embd_s_ingredient() == 0

        for (uint32_t il = 0; il < N_LAYERS; ++il) {
            model->hparams.n_head_arr[il]    = N_HEAD;
            model->hparams.n_head_kv_arr[il] = N_HEAD;
        }

        // the layer-1 (recurrent) marker the default filters read
        model->hparams.is_recr_impl.fill(0);
        model->hparams.is_recr_impl[N_ATTN_LAYERS] = 1;

        mem = new llama_memory_hybrid(*model,
                /* attn */ GGML_TYPE_F32, GGML_TYPE_F32, /*v_trans=*/false,
                /*kv_size=*/N_CTX, /*n_pad=*/1, /*n_swa=*/0, LLAMA_SWA_TYPE_NONE,
                /* recurrent */ GGML_TYPE_F32, GGML_TYPE_F32, /*rs_size=*/N_CTX,
                /* common */ N_SEQ_MAX, n_rs_seq, /*gdn_replay_req=*/false,
                /*offload=*/false, /*unified=*/true,
                /*filter_attn=*/nullptr, /*filter_recr=*/nullptr);

        // the ubatch: one token per sequence, each naming its own sequence.
        // The seq_id pointers point into `seq_id_data`, which outlives them.
        seq_id_data.reserve(n_seqs);
        for (uint32_t s = 0; s < n_seqs; ++s) {
            seq_ids.push_back((llama_seq_id) s);
            seq_id_data.push_back((llama_seq_id) s);
        }
        for (uint32_t s = 0; s < n_seqs; ++s) {
            ub.seq_id[s]   = &seq_id_data[s];
            ub.n_seq_id[s] = 1;
            ub.pos[s]      = 0;
        }

        std::vector<llama_ubatch> ubatches = { ub };

        llama_kv_cache::slot_info_vec_t sinfos(1);
        sinfos[0].s0 = 0;
        sinfos[0].s1 = 0;
        sinfos[0].strm.push_back(0);
        sinfos[0].idxs.resize(1);
        sinfos[0].idxs[0].resize(n_seqs, 0);

        // The hybrid memory is built with unified = true, so cparams must say
        // so: can_reuse_kq_mask() divides the mask's row count by
        // n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq, and
        // llama_cparams is a plain aggregate with no initializer for this
        // field, so an unset value is indeterminate.
        cparams.kv_unified = true;
        cparams.n_seq_max  = n_seqs;
        cparams.n_batch    = ub.n_tokens;
        cparams.n_ubatch   = ub.n_tokens;
        hyb = new llama_memory_hybrid_context(mem, std::move(sinfos), std::move(ubatches));

        ggml_init_params gp = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 8 + 4096,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx0 = ggml_init(gp);
        if (ctx0 != nullptr) {
            // The recurrent sub-cache has not run find_slot() yet, so its n is
            // still 0 and get_n_rs() (is_full == false) would report 0. That
            // makes s_copy a zero-byte tensor and ggml_view_1d(s_copy, 1, 0)
            // trips ggml.c:1867 (data_size + view_offs <= ggml_nbytes(view_src)
            // with nbytes 0). This fixture gives the cache N_CTX cells, which is
            // what a full-cache context would report.
            mem->get_mem_recr()->n = N_CTX;

            // The copy tensors are NOT built here either: can_reuse() compares
            // their lengths against get_n_rs(), and get_n_rs() is the number of
            // USED cells (mem->n), which find_slot() sets during apply(). Before
            // apply() it is the capacity the fixture wrote by hand, after it is
            // the batch's cell count, so a tensor sized here is sized for the
            // wrong number. prepare() builds them at the live value.
            row     = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
            k_idxs  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_seqs);
            v_idxs  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_seqs);
            // The mask is NOT built here. Its width must be the width
            // can_reuse_kq_mask() compares it against
            // (mctx->get_n_kv_target(), llama-graph.cpp:1406), and that value
            // is only known after the context applies a ubatch. The fixture
            // therefore exposes prepare(), which applies the ubatch, reads the
            // live target, builds the mask at exactly that width and then
            // allocates the context's tensors. Sizing the mask here would be a
            // guess: it matched only while the target happened to equal
            // N_CTX, and the target is the ubatch's token count.
            ggml_set_input(row);
            buf = nullptr;
        }
    }

    // Apply the ubatch, then build the mask at the width the reuse check will
    // compare against, then allocate. Idempotent: a second call is a no-op.
    bool prepare() {
        if (prepared) {
            return ok();
        }
        if (ctx0 == nullptr || hyb == nullptr) {
            return false;
        }
        applied = hyb->apply();
        if (!applied) {
            return false;
        }
        const uint32_t target = hyb->get_attn() != nullptr ? hyb->get_attn()->get_n_kv_target() : 0;
        if (target == 0) {
            return false;
        }
        // the copy tensors can_reuse() dereferences (llama-graph.cpp:1547-1550),
        // sized at the live n_rs and n_seqs -- after apply(), which is when
        // get_n_rs() reports the batch's cell count rather than the capacity
        const int64_t n_rs    = hyb->get_recr()->get_n_rs();
        const int64_t n_seqs_ = (int64_t) n_seqs_mem;
        s_copy  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
        s_main  = ggml_view_1d(ctx0, s_copy, n_seqs_, 0);
        s_extra = ggml_view_1d(ctx0, s_copy, n_rs - n_seqs_, n_seqs_ * s_copy->nb[0]);
        if (s_copy == nullptr || s_main == nullptr || s_extra == nullptr) {
            return false;
        }
        ggml_set_input(s_copy);

        mask_target = target;
        kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, (int64_t) target, n_seqs_mem, 1, 1);
        if (kq_mask == nullptr) {
            return false;
        }
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx0, ggml_backend_cpu_buffer_type());
        prepared = true;
        return buf != nullptr;
    }

    ~fixture() {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx0 != nullptr) {
            ggml_free(ctx0);
        }
        delete hyb;
        delete mem;
        delete model;
    }

    bool ok() const {
        return model != nullptr && mem != nullptr && hyb != nullptr &&
               ctx0 != nullptr && buf != nullptr && row != nullptr &&
               s_copy != nullptr && s_main != nullptr && s_extra != nullptr &&
               k_idxs != nullptr && v_idxs != nullptr && kq_mask != nullptr;
    }

    fixture(const fixture &) = delete;
    fixture & operator=(const fixture &) = delete;
};

int32_t row_value(const ggml_tensor * t) {
    return t == nullptr ? std::numeric_limits<int32_t>::min() : ((const int32_t *) t->data)[0];
}

// one hybrid site under test: the input object plus the attention input it
// wraps, so the test can call set_input() exactly as the graph does
struct site {
    const char * label;
    std::unique_ptr<llm_graph_input_mem_hybrid>   kv;   // attn_kv flavour
    std::unique_ptr<llm_graph_input_mem_hybrid_k> k;    // attn_k flavour

    llm_graph_input_i * input() const { return kv != nullptr ? (llm_graph_input_i *) kv.get()
                                                             : (llm_graph_input_i *) k.get(); }
    llm_graph_input_rs * recr() const { return kv != nullptr ? kv->get_recr() : k->get_recr(); }

    // the attn side's tensors, for the reuse diagnostic
    // The attn input the wrapper took ownership of; make_sites() records it
    // before the move so the diagnostic can read the tensors can_reuse() reads.
    // The fixture's cparams, which the real graph build passes through
    // llm_graph_params. can_reuse_kq_mask() reads cparams.kv_unified to decide
    // n_stream, so a params struct without it compares the mask against the
    // wrong row count.
    const llama_cparams * cparams = nullptr;
    const llm_graph_input_attn_kv * attn_kv = nullptr;
    const llm_graph_input_attn_k *  attn_k  = nullptr;
    int64_t attn_k_idxs_ne0() const { return attn_kv != nullptr ? attn_kv->self_k_idxs->ne[0]
                                                                : attn_k->self_k_idxs->ne[0]; }
    int64_t attn_v_idxs_ne0() const { return attn_kv != nullptr && attn_kv->self_v_idxs != nullptr
                                             ? attn_kv->self_v_idxs->ne[0] : -1; }
    int64_t attn_mask_ne(int d) const {
        const ggml_tensor * m = attn_kv != nullptr ? attn_kv->self_kq_mask : attn_k->self_kq_mask;
        return m != nullptr ? m->ne[d] : -1;
    }
};

// what one set_input() pass at one site observed
struct pass {
    int32_t  row        = 0;
    bool     refused    = false;
    bool     reuse_ok   = false;
    uint32_t nkv_target = 0;   // the width can_reuse_kq_mask() compared against
};

// One set_input() pass at one site.
//
// s_copy is left UNWIRED (nullptr) so the copy loop -- the one consumer of
// rs_idx (llama-memory-recurrent.cpp:1482-1501) -- does not run and the
// depths the test drives stay readable. The tensors are wired just before
// can_reuse(), which dereferences all three (llama-graph.cpp:1412-1414).
pass run_site(site & s, const llama_memory_hybrid_context * hyb, const llama_ubatch * ub,
              ggml_tensor * row, int32_t seed,
              ggml_tensor * s_copy, ggml_tensor * s_main, ggml_tensor * s_extra) {
    pass p;
    s.recr()->s_zero = row;
    if (row != nullptr) {
        ((int32_t *) row->data)[0] = seed;
    }
    s.input()->set_input(ub);
    p.row     = row_value(row);
    p.refused = s.recr()->rs_zero_row_refused;

    s.recr()->s_copy       = s_copy;
    s.recr()->s_copy_main  = s_main;
    s.recr()->s_copy_extra = s_extra;

    llm_graph_params params = {};
    params.mctx    = hyb;
    params.ubatch  = *ub;
    if (s.cparams != nullptr) {
        params.cparams = *s.cparams;
    }

    // the width can_reuse_kq_mask() will compare the mask against
    p.nkv_target = hyb->get_attn() != nullptr ? hyb->get_attn()->get_n_kv_target() : 0;
    p.reuse_ok   = s.input()->can_reuse(params);
    if (!p.reuse_ok) {
        // Print every conjunct can_reuse() reads, so the refusing one is
        // measured rather than guessed. RS side: llama-graph.cpp:733-763.
        // attn side: llama-graph.cpp:99-119 and :925-933.
        const auto * recr = hyb->get_recr();
        const auto * attn = hyb->get_attn();
        const uint32_t n_stream = params.cparams.kv_unified ? 1 : ub->n_seqs_unq;
        std::fprintf(stderr,
                "REUSE_DIAG site=%s refused=%d\n"
                "  rs: s_copy.ne0=%lld get_n_rs=%u | n_seqs=%u ub.n_seqs=%u"
                " | s_main.ne0=%lld | s_extra.ne0=%lld want=%u"
                " | head=%d get_head=%u | replay=%d get_replay=%u"
                " | rs_zero_row_refused=%d | s_zero=%p rs_z=%d get_rs_z=%d\n"
                "  attn: k_idxs.ne0=%lld ub.n_tokens=%u v_idxs.ne0=%lld"
                " | mask.ne=[%lld,%lld,%lld,%lld] n_kv_target=%u n_tokens=%u n_stream=%u"
                " | kv_unified=%d\n",
                s.label, int(p.reuse_ok),
                (long long) s_copy->ne[0], recr->get_n_rs(),
                s.recr()->n_seqs, ub->n_seqs,
                (long long) s_main->ne[0],
                (long long) s_extra->ne[0], recr->get_n_rs() - ub->n_seqs,
                s.recr()->head, recr->get_head(),
                s.recr()->replay_len, recr->get_replay_len(),
                int(s.recr()->rs_zero_row_refused),
                (void *) s.recr()->s_zero, s.recr()->rs_z, recr->get_rs_z(),
                (long long) s.attn_k_idxs_ne0(), ub->n_tokens, (long long) s.attn_v_idxs_ne0(),
                (long long) s.attn_mask_ne(0), (long long) s.attn_mask_ne(1), (long long) s.attn_mask_ne(2), (long long) s.attn_mask_ne(3),
                attn != nullptr ? attn->get_n_kv_target() : 0, ub->n_tokens, n_stream,
                int(params.cparams.kv_unified));
    }

    // back to the unwired state for the next set_input()
    s.recr()->s_copy       = nullptr;
    s.recr()->s_copy_main  = nullptr;
    s.recr()->s_copy_extra = nullptr;
    return p;
}

// builds the two sites over one hybrid context. s_copy/s_copy_main/
// s_copy_extra/head/rs_z have NO default initializer
// (src/llama-graph.h:308-325), so every one is set explicitly: leaving them
// unset would make `if (inp_rs->s_copy)` read an indeterminate pointer.
std::vector<site> make_sites(fixture & f) {
    std::vector<site> sites;

    {
        auto inp_attn = std::make_unique<llm_graph_input_attn_kv>(f.hparams, f.cparams, f.hyb->get_attn());
        inp_attn->self_k_idxs     = f.k_idxs;
        inp_attn->self_v_idxs     = f.v_idxs;
        inp_attn->self_kq_mask    = f.kq_mask;
        inp_attn->self_kq_mask_cnv = f.kq_mask;

        auto inp_rs = std::make_unique<llm_graph_input_rs>(f.hyb->get_recr());
        inp_rs->s_copy       = nullptr;   // wired per call, see run_site()
        inp_rs->s_copy_main  = nullptr;
        inp_rs->s_copy_extra = nullptr;
        inp_rs->head         = f.hyb->get_recr()->get_head();
        inp_rs->rs_z         = f.hyb->get_recr()->get_rs_z();
        inp_rs->replay_len   = 0;
        inp_rs->s_zero       = f.row;

        site s;
        s.label = "mem_hybrid";
        s.cparams = &f.cparams;
        s.attn_kv = inp_attn.get();
        s.kv = std::make_unique<llm_graph_input_mem_hybrid>(f.cparams,
                std::move(inp_attn), std::move(inp_rs), f.hyb);
        sites.push_back(std::move(s));
    }
    {
        auto inp_attn = std::make_unique<llm_graph_input_attn_k>(f.hparams, f.cparams, f.hyb->get_attn());
        inp_attn->self_k_idxs     = f.k_idxs;
        inp_attn->self_kq_mask    = f.kq_mask;
        inp_attn->self_kq_mask_cnv = f.kq_mask;

        auto inp_rs = std::make_unique<llm_graph_input_rs>(f.hyb->get_recr());
        inp_rs->s_copy       = nullptr;   // wired per call, see run_site()
        inp_rs->s_copy_main  = nullptr;
        inp_rs->s_copy_extra = nullptr;
        inp_rs->head         = f.hyb->get_recr()->get_head();
        inp_rs->rs_z         = f.hyb->get_recr()->get_rs_z();
        inp_rs->replay_len   = 0;
        inp_rs->s_zero       = f.row;

        site s;
        s.label = "mem_hybrid_k";
        s.cparams = &f.cparams;
        s.attn_k = inp_attn.get();
        s.k = std::make_unique<llm_graph_input_mem_hybrid_k>(f.cparams,
                std::move(inp_attn), std::move(inp_rs), f.hyb);
        sites.push_back(std::move(s));
    }
    return sites;
}

} // namespace

int main() {
    llama_backend_init();

    // ======================================================================
    // (1)-(3), (6): n_rs_seq == 0, both sites, one live sequence
    // ======================================================================
    {
        fixture f(/*n_rs_seq=*/0, /*n_seqs=*/1);
        if (!f.prepare() || !f.ok()) {
            std::fprintf(stderr, "FAIL: cannot build the hybrid fixture\n");
            llama_backend_free();
            return 1;
        }
        CHECK(!f.hyb->get_recr()->gdn_replay(),
              "(1) gdn_replay is on; LLAMA_GDN_REPLAY must not change this fixture");

        std::vector<site> sites = make_sites(f);
        llama_memory_recurrent * recr_mem = f.mem->get_mem_recr();

        // Apply the ubatch so the attention sub-cache has a live width: until
        // then llama_kv_cache_context::get_n_kv_target() is 0
        // (llama-kv-cache.cpp:3784 -> get_n_kv() -> n_kv, set only by
        // apply_ubatch at :3728), and can_reuse_kq_mask() would compare the
        // mask against 0 and refuse every graph for a reason that is about
        // the fixture, not about the row.
        CHECK(f.prepare(), "(1) prepare() failed (apply + mask at the live width)");
        CHECK(f.applied, "(1) apply() failed");
        const llama_ubatch ub = f.hyb->get_ubatch();

        // The fixture fact worth pinning: the mask the test builds is exactly
        // the width the reuse check compares against. If this ever diverges,
        // every reuse_ok assertion below would be testing the mask width and
        // not the row, so it is asserted once, loudly.
        {
            const uint32_t target = f.hyb->get_attn()->get_n_kv_target();
            CHECK(target == (uint32_t) f.kq_mask->ne[0],
                  "(1) fixture: the live n_kv target is %u but the mask is built at %lld; "
                  "can_reuse_kq_mask() would then decide reuse on the mask width, not on the row",
                  target, (long long) f.kq_mask->ne[0]);
            CHECK(f.mask_target == (uint32_t) f.kq_mask->ne[0],
                  "(1) fixture: the mask was built at %u but records %u",
                  (unsigned) f.kq_mask->ne[0], f.mask_target);
        }

        // ---- (1) the row receives the live row ---------------------------
        recr_mem->rs_z = 3;
        CHECK(f.hyb->get_recr()->get_rs_z() == 3,
              "(1) fixture context reports rs_z %d, expected 3", f.hyb->get_recr()->get_rs_z());

        for (site & s : sites) {
            const pass p = run_site(s, f.hyb, &ub, f.row, /*seed=*/-7, f.s_copy, f.s_main, f.s_extra);
            CHECK(p.row == 3,
                  "(1) %s: row holds %d after set_input(), expected the live rs_z 3 "
                  "(the hybrid set_input() did not write the selection row)",
                  s.label, p.row);
            CHECK(!p.refused, "(1) %s: the bind refused a single-depth ubatch", s.label);
            CHECK(p.reuse_ok, "(1) %s: can_reuse() refused a graph whose row was bound", s.label);
        }

        // ---- (2) a moved rs_z rewrites the row ---------------------------
        recr_mem->rs_z = 1;
        CHECK(f.hyb->get_recr()->get_rs_z() == 1,
              "(2) fixture context reports rs_z %d after the move, expected 1",
              f.hyb->get_recr()->get_rs_z());

        for (site & s : sites) {
            // seed the STALE value a missing write would leave behind
            const pass p = run_site(s, f.hyb, &ub, f.row, /*seed=*/3, f.s_copy, f.s_main, f.s_extra);
            CHECK(p.row == 1,
                  "(2) %s: row holds %d after the second set_input(), expected the moved rs_z 1 "
                  "(a stale row is the hybrid defect: the graph was reused but the row was not "
                  "rewritten)", s.label, p.row);
            CHECK(p.row != 3,
                  "(2) %s: the second write left the first arm's value 3 in place", s.label);
        }

        // ---- (3) both sites agree on the row -----------------------------
        recr_mem->rs_z = 2;
        {
            const pass a = run_site(sites[0], f.hyb, &ub, f.row, /*seed=*/-7, f.s_copy, f.s_main, f.s_extra);
            const pass b = run_site(sites[1], f.hyb, &ub, f.row, /*seed=*/-7, f.s_copy, f.s_main, f.s_extra);
            CHECK(a.row == b.row,
                  "(3) the two hybrid sites disagree on the row: %s wrote %d, %s wrote %d",
                  sites[0].label, a.row, sites[1].label, b.row);
            CHECK(a.row == 2, "(3) the agreed row is %d, expected the live rs_z 2", a.row);
        }

        // ---- (6) stock build: s_zero == nullptr is a no-op ---------------
        //
        // With the lane off the graph carries no row and nothing may be
        // written. The seed must survive both sites unchanged, and the cached
        // rs_z must not be taken from the context either.
        recr_mem->rs_z = 3;
        for (site & s : sites) {
            s.recr()->rs_z = -2;   // "never bound" sentinel
            // The lane is turned off by passing no row, so the value to read
            // back is the fixture's own row tensor, seeded here. Reading the
            // pass's `row` would read a null pointer: run_site() only reports
            // the row it was given.
            ((int32_t *) f.row->data)[0] = -7;
            const pass p = run_site(s, f.hyb, &ub, /*row=*/nullptr, /*seed=*/-7, f.s_copy, f.s_main, f.s_extra);
            const int32_t after = ((const int32_t *) f.row->data)[0];
            CHECK(after == -7,
                  "(6) %s: stock build (s_zero == nullptr) wrote the row (%d)", s.label, after);
            CHECK(s.recr()->rs_z == -2,
                  "(6) %s: stock build changed the cached rs_z to %d, expected the sentinel -2",
                  s.label, s.recr()->rs_z);
            CHECK(!p.refused, "(6) %s: a null row is not a refusal", s.label);
        }
    }

    // ======================================================================
    // (4) n_rs_seq == 2: the row is rs_idx*mem_size + rs_z at both sites
    // ======================================================================
    {
        const uint32_t mem_size = N_CTX;
        const uint32_t n_rs_seq = 2;
        fixture f(n_rs_seq, /*n_seqs=*/1);
        if (!f.prepare() || !f.ok()) {
            std::fprintf(stderr, "FAIL: cannot build the ring hybrid fixture\n");
            llama_backend_free();
            return 1;
        }
        CHECK(f.hyb->get_recr()->get_n_rs_seq() == n_rs_seq,
              "(4) recurrent sub-cache reports n_rs_seq %u, expected %u",
              f.hyb->get_recr()->get_n_rs_seq(), n_rs_seq);

        std::vector<site> sites = make_sites(f);
        llama_memory_recurrent * recr_mem = f.mem->get_mem_recr();

        CHECK(f.prepare(), "(4) prepare() failed (apply + mask at the live width)");
        CHECK(f.applied, "(4) apply() failed");
        const llama_ubatch ub = f.hyb->get_ubatch();
        CHECK(f.hyb->get_attn()->get_n_kv_target() == (uint32_t) f.kq_mask->ne[0],
              "(4) fixture: live n_kv target %u != the mask width %lld",
              f.hyb->get_attn()->get_n_kv_target(), (long long) f.kq_mask->ne[0]);

        recr_mem->rs_z = 1;
        for (uint32_t depth = 0; depth <= n_rs_seq; ++depth) {
            recr_mem->set_rs_idx(0, depth);
            CHECK(recr_mem->rs_idx[0] == depth,
                  "(4) set_rs_idx(0, %u) left rs_idx %u", depth, recr_mem->rs_idx[0]);

            const int32_t expect = (int32_t) (depth * mem_size) + 1;
            for (site & s : sites) {
                const pass p = run_site(s, f.hyb, &ub, f.row, /*seed=*/-7, f.s_copy, f.s_main, f.s_extra);
                CHECK(!p.refused, "(4) %s: the bind refused a single-depth ubatch", s.label);
                CHECK(p.row == expect,
                      "(4) %s: depth %u bound row %d, expected rs_idx*mem_size + rs_z = %d "
                      "(the cell index 1 is the pre-merge value)", s.label, depth, p.row, expect);
                if (depth > 0) {
                    CHECK(p.row != 1,
                          "(4) %s: depth %u bound the cell index, not a group row",
                          s.label, depth);
                }
            }
        }
    }

    // ======================================================================
    // (5) a mixed-depth ubatch is refused at both sites, and not reused
    // ======================================================================
    {
        const uint32_t mem_size = N_CTX;
        const uint32_t n_rs_seq = 2;
        fixture f(n_rs_seq, /*n_seqs=*/2);
        if (!f.prepare() || !f.ok()) {
            std::fprintf(stderr, "FAIL: cannot build the mixed-depth hybrid fixture\n");
            llama_backend_free();
            return 1;
        }

        std::vector<site> sites = make_sites(f);
        llama_memory_recurrent * recr_mem = f.mem->get_mem_recr();

        CHECK(f.prepare(), "(5) prepare() failed (apply + mask at the live width)");
        CHECK(f.applied, "(5) apply() failed");
        const llama_ubatch ub = f.hyb->get_ubatch();
        CHECK(f.hyb->get_attn()->get_n_kv_target() == (uint32_t) f.kq_mask->ne[0],
              "(5) fixture: live n_kv target %u != the mask width %lld",
              f.hyb->get_attn()->get_n_kv_target(), (long long) f.kq_mask->ne[0]);

        recr_mem->rs_z = 1;
        recr_mem->set_rs_idx(0, 2);
        recr_mem->set_rs_idx(1, 0);
        CHECK(f.hyb->get_recr()->get_rs_zero_row(0) == (int32_t) (2 * mem_size + 1),
              "(5) seq 0 row %d, expected 2*%u+1", f.hyb->get_recr()->get_rs_zero_row(0), mem_size);
        CHECK(f.hyb->get_recr()->get_rs_zero_row(1) == 1,
              "(5) seq 1 row %d, expected 0*%u+1", f.hyb->get_recr()->get_rs_zero_row(1), mem_size);

        for (site & s : sites) {
            const pass p = run_site(s, f.hyb, &ub, f.row, /*seed=*/-7, f.s_copy, f.s_main, f.s_extra);
            CHECK(p.refused,
                  "(5) %s: set_input() did not record a refusal for a mixed-depth ubatch", s.label);
            CHECK(p.row == -7,
                  "(5) %s: the refused bind wrote row %d, expected the seed -7 to survive",
                  s.label, p.row);
            CHECK(!p.reuse_ok,
                  "(5) %s: can_reuse() accepted a graph whose zero row could not be bound",
                  s.label);
        }

        // the refusal is the mixed batch's, not a permanent state
        recr_mem->set_rs_idx(1, 2);
        for (site & s : sites) {
            const pass p = run_site(s, f.hyb, &ub, f.row, /*seed=*/-7, f.s_copy, f.s_main, f.s_extra);
            CHECK(!p.refused, "(5) %s: the bind refused a single-depth ubatch", s.label);
            CHECK(p.row == (int32_t) (2 * mem_size + 1),
                  "(5) %s: the bound row is %d, expected %d",
                  s.label, p.row, (int32_t) (2 * mem_size + 1));
            CHECK(p.reuse_ok, "(5) %s: can_reuse() refused a graph whose row was bound", s.label);
        }
    }

    llama_backend_free();

    if (g_checks_run == 0) {
        std::fprintf(stderr, "RSZ_HYBRID_SELECTION_WRITE_TEST UNPROVEN reason=no_check_ran checks=0\n");
        return 2;
    }
    if (g_fail != 0) {
        std::printf("RSZ_HYBRID_SELECTION_WRITE_TEST FAIL %d\n", g_fail);
        return 1;
    }
    std::printf("RSZ_HYBRID_SELECTION_WRITE_TEST PASS\n");
    return 0;
}
