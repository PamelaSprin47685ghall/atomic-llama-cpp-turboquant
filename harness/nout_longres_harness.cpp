// nout_longres_harness.cpp - long-resident single-context CPU harness for the
// GGML_TP5_NOUT_REUSE output-capacity reuse lane (src/llama-nout-bind.cpp,
// llm_graph_result::bind_outputs, llama_context::process_ubatch bind path).
//
// Part 1 ("ctx", long-resident): one llama_model shared by three llama
// contexts kept alive across the whole run, decoded step by step over the
// SAME token/position/KV sequence:
//   ctx_exact : gate OFF (stock graphs rebuilt at the exact active count).
//   ctx_reuse : gate ON  (GGML_TP5_NOUT_REUSE=1 at construction; the graph is
//               built once over the output envelope n_outputs_max=16 and
//               narrowed in place per decode by the bind path).
//   ctx_ref   : gate OFF, every token flagged output (24 per step): the
//               row-to-token anchor. Row j of any narrower run at the same
//               position history must be bit-identical to row j here, because
//               the gather sits inside the last block (src/models/llama.cpp:
//               the attention of the last layer runs at full token count
//               before the gather) and every accepted op is row-independent.
// Step sequence {0,1,3,7,16,1,1,0,16,0,3,7,16,1,0} covers shrink, grow,
// repeat-equal and 0<->full-envelope jumps against the same resident graph.
// Zero outputs is exercised through non-empty all-zero output flags (never a
// null pointer); no logits pointer is dereferenced on such steps.
//
// Counting (all real, no simulated counters):
//   - per decode: delta of llama_perf_context().n_reused (the runtime's own
//     graph-reuse counter);
//   - at context destruction: the runtime's own "NOUT_REUSE_STATS binds=..
//     fallbacks=.." line captured via llama_log_set;
//   - per decode executed shapes (gather count, out_ids width, logits shape,
//     logits node pointer) through the scheduler eval callback.
// Pointer equality is printed under trace only as corroboration; assertions
// never rely on it (a rebuild can land on the same address). Reuse
// expectations are derived from the documented n_kv 256-bucketing
// (llama-kv-cache.cpp get_n_kv), which is a reuse key through
// llm_graph_input_attn_kv::can_reuse; any divergence fails loudly with the
// actual numbers.
//
// Output correctness: after llama_synchronize, the public logits rows
// (exactly n_outputs x n_vocab floats via llama_get_logits_ith) are copied
// into guard-page-protected staging buffers and compared BYTE-EXACT between
// the three contexts. --dump-prefix additionally writes raw float bytes per
// context for cross-process (stock-vs-patch) cmp comparison.
//
// Boundaries (honest scope):
//   - the harness never writes outside the public logits region (no sentinel
//     beyond the logits pointer length);
//   - for k=0 steps the public buffer (>= 1 row by n_seq_max=1) is snapshotted
//     and required byte-unchanged across the decode;
//   - staging buffers carry guard pages, so harness-side overruns fault;
//   - part "plan" exercises the planner's fail-closed contract directly:
//     over-envelope, softmax/reshape consumers, envelope mismatch and
//     no-gather graphs must reject with ALL node ne/nb AND data unchanged;
//   - what this still cannot prove without a sanitizer build of libllama:
//     backend writes beyond narrowed tensor extents into unrelated internal
//     buffers. Build the library AND this harness with ASan on the acceptance
//     machine for that layer; the harness reports its own ASan status so the
//     report cannot overclaim.
//
// Part "plan" mirrors the real LLAMA-arch output closure
// (2 x get_rows -> add -> rms_norm -> mul -> mul_mat up/gate -> swiglu ->
// mul_mat down -> add -> rms_norm -> mul -> mul_mat head -> scale -> cont)
// on ONE ggml graph, binds 0/1/3/7/16 with restores and grow-backs on the
// SAME graph, compares every bind bit-exactly against a fresh exact-count
// build, and finally re-checks the envelope output after a restore.
//
// Build (acceptance machine ONLY; every compilation goes through
// /home/hermes/.local/bin/build-slot - it does not exist on the authoring
// machine and must not be bypassed):
//   /home/hermes/.local/bin/build-slot g++ -std=c++17 -O2 -g -Wall -Wextra \
//     -fsanitize=address,undefined \
//     -I<wt>/include -I<wt>/ggml/include -I<wt>/src \
//     harness/nout_longres_harness.cpp -o nout_longres_harness \
//     -L<wt>/build/src -lllama -Wl,-rpath,<wt>/build/src
// Run: ./nout_longres_harness [--mode all|ctx|plan] [--model PATH.gguf]
//      [--save-gguf PATH] [--dump-prefix P] [--log PATH] [--trace]
// GGML_TP5_HARNESS_TRACE=1 enables per-step trace lines (gated instrumentation;
// assertions run regardless). Exit codes: 0 pass, 1 failures, 2 setup error.
//
// Honesty note: this source was written and reviewed but NEVER compiled or
// executed on the authoring machine (no build-slot, compilation forbidden);
// all runtime claims in the docs are recorded as not-run.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include "llama-nout-bind.h"

#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// failure accounting + gated trace
// ---------------------------------------------------------------------------

static int  g_failures = 0;
static bool g_trace    = false;

#define CHECK(cond, ...)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            ++g_failures;                                                       \
            fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__);               \
            fprintf(stderr, __VA_ARGS__);                                       \
            fprintf(stderr, "\n");                                              \
        }                                                                       \
    } while (0)

#define TRACE(...)                                                             \
    do {                                                                        \
        if (g_trace) {                                                          \
            fprintf(stderr, __VA_ARGS__);                                       \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// log capture: forward everything unchanged, keep lines so assertions can use
// the runtime's own counters (NOUT_REUSE_STATS / first-bind / gate lines).
// ---------------------------------------------------------------------------

struct log_capture {
    std::mutex        mx;
    std::vector<std::string> lines;

    void add(const char * text) {
        std::string s(text ? text : "");
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
            s.pop_back();
        }
        std::lock_guard<std::mutex> lk(mx);
        lines.push_back(s);
        fprintf(stderr, "%s\n", s.c_str());  // full transcript stays visible
    }

    size_t count_contains(const char * needle) {
        std::lock_guard<std::mutex> lk(mx);
        size_t n = 0;
        for (const auto & l : lines) {
            if (l.find(needle) != std::string::npos) {
                ++n;
            }
        }
        return n;
    }

    bool parse_last(const char * prefix, const char * fmt, ...) {
        std::lock_guard<std::mutex> lk(mx);
        for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
            if (it->rfind(prefix, 0) == 0) {
                va_list ap;
                va_start(ap, fmt);
                const int rc = vsscanf(it->c_str(), fmt, ap);
                va_end(ap);
                return rc > 0;
            }
        }
        return false;
    }
} g_log;

static void harness_log_cb(ggml_log_level /*level*/, const char * text, void * /*user*/) {
    g_log.add(text);
}

// ---------------------------------------------------------------------------
// guard-page staging buffer: writes past cap fault (SIGSEGV) instead of
// silently corrupting; protects the harness's own copies of public rows.
// ---------------------------------------------------------------------------

struct guarded_buf {
    void * base = nullptr;
    size_t cap  = 0;   // usable bytes (request rounded up to page size)
    char * ptr   = nullptr;
    size_t page = 0;

    void alloc(size_t bytes) {
        page = (size_t) sysconf(_SC_PAGESIZE);
        cap  = ((bytes + page - 1) / page) * page;
        base = mmap(nullptr, cap + 2 * page, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) {
            base = nullptr;
            fprintf(stderr, "harness: mmap failed for %zu bytes\n", bytes);
            exit(2);
        }
        mprotect(base, page, PROT_NONE);                     // underflow guard
        mprotect((char *) base + page + cap, page, PROT_NONE);  // overflow guard
        ptr = (char *) base + page;
    }

    void free_buf() {
        if (base) {
            munmap(base, cap + 2 * page);
            base = nullptr;
            ptr = nullptr;
            cap = 0;
        }
    }

    void set(const void * src, size_t bytes) {
        if (bytes > cap) {
            fprintf(stderr, "harness: guarded_buf overflow %zu > %zu\n", bytes, cap);
            exit(2);
        }
        memcpy(ptr, src, bytes);
    }
};

// ---------------------------------------------------------------------------
// tiny LLAMA-arch model built in memory (deterministic F32 weights), in the
// style of tests/test-llama-archs.cpp. --model PATH loads a real GGUF
// instead; --save-gguf PATH dumps the fixture for cross-machine reproduction.
// ---------------------------------------------------------------------------

struct harness_model_cfg {
    uint32_t n_vocab = 128;
    uint32_t n_embd  = 64;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 128;
    uint32_t n_layer = 2;
    uint32_t n_ctx   = 512;
    size_t   seed    = 1234;
};

static void harness_set_tensor_data(ggml_tensor * t, void * ud) {
    size_t seed = *(size_t *) ud;
    std::hash<std::string> hasher;
    seed ^= hasher(t->name);
    std::mt19937 gen((uint32_t) seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        std::vector<float> tmp((size_t) ne);
        for (int64_t i = 0; i < ne; ++i) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, ggml_nbytes(t));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp((size_t) ne);
        for (int64_t i = 0; i < ne; ++i) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, ggml_nbytes(t));
    } else {
        GGML_ABORT("unsupported tensor type for synthetic weights: %s", ggml_type_name(t->type));
    }
}

static gguf_context * harness_make_gguf(const harness_model_cfg & cfg) {
    gguf_context * gctx = gguf_init_empty();

    auto kv_u32 = [&](const char * k, uint32_t v) { gguf_set_val_u32(gctx, k, v); };
    auto kv_f32 = [&](const char * k, float v)    { gguf_set_val_f32(gctx, k, v); };
    auto kv_str = [&](const char * k, const char * v) { gguf_set_val_str(gctx, k, v); };

    kv_str("general.architecture", "llama");
    kv_str("general.name", "nout-longres-harness");

    kv_u32("llama.vocab_size",          cfg.n_vocab);
    kv_u32("llama.context_length",      cfg.n_ctx);
    kv_u32("llama.embedding_length",    cfg.n_embd);
    kv_u32("llama.feed_forward_length", cfg.n_ff);
    kv_u32("llama.block_count",         cfg.n_layer);
    kv_u32("llama.attention.head_count", cfg.n_head);
    kv_u32("llama.attention.head_count_kv", cfg.n_head);
    kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
    kv_f32("llama.rope.freq_base", 10000.0f);
    kv_u32("llama.rope.dimension_count", cfg.n_embd / cfg.n_head);
    kv_u32("llama.attention.key_length", cfg.n_embd / cfg.n_head);
    kv_u32("llama.attention.value_length", cfg.n_embd / cfg.n_head);

    // tokenizer: SPM with a flat token list, enough for llama_model_load
    kv_str("tokenizer.ggml.model", "llama");
    std::vector<std::string> tokens;
    tokens.reserve(cfg.n_vocab);
    for (uint32_t i = 0; i < cfg.n_vocab; ++i) {
        tokens.push_back("t" + std::to_string(i));
    }
    std::vector<const char *> tok_ptrs;
    tok_ptrs.reserve(tokens.size());
    for (const auto & s : tokens) {
        tok_ptrs.push_back(s.c_str());
    }
    gguf_set_arr_str(gctx, "tokenizer.ggml.tokens", tok_ptrs.data(), tokens.size());
    std::vector<int32_t> types(cfg.n_vocab, 1 /* normal */);
    types[0] = 2; // unknown
    types[1] = 3; // control (bos)
    types[2] = 3; // control (eos)
    gguf_set_arr_data(gctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    std::vector<float> scores(cfg.n_vocab, 0.0f);
    gguf_set_arr_data(gctx, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    kv_u32("tokenizer.ggml.bos_token_id", 1);
    kv_u32("tokenizer.ggml.eos_token_id", 2);
    kv_u32("tokenizer.ggml.unknown_token_id", 0);

    // tensor skeleton (F32); data filled by harness_set_tensor_data
    auto add = [&](const char * name, int64_t ne0, int64_t ne1) {
        ggml_tensor t = {};
        snprintf(t.name, sizeof(t.name), "%s", name);
        t.type = GGML_TYPE_F32;
        t.ne[0] = ne0;
        t.ne[1] = ne1;
        t.ne[2] = 1;
        t.ne[3] = 1;
        t.nb[0] = sizeof(float);
        t.nb[1] = t.nb[0] * t.ne[0];
        t.nb[2] = t.nb[1] * t.ne[1];
        t.nb[3] = t.nb[2] * t.ne[2];
        gguf_add_tensor(gctx, &t);
    };

    add("token_embd.weight", cfg.n_embd, cfg.n_vocab);
    for (uint32_t il = 0; il < cfg.n_layer; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        add((p + "attn_norm.weight").c_str(), cfg.n_embd, 1);
        add((p + "attn_q.weight").c_str(),    cfg.n_embd, cfg.n_embd);
        add((p + "attn_k.weight").c_str(),    cfg.n_embd, cfg.n_embd);
        add((p + "attn_v.weight").c_str(),    cfg.n_embd, cfg.n_embd);
        add((p + "attn_output.weight").c_str(), cfg.n_embd, cfg.n_embd);
        add((p + "ffn_norm.weight").c_str(),  cfg.n_embd, 1);
        add((p + "ffn_gate.weight").c_str(),  cfg.n_embd, cfg.n_ff);
        add((p + "ffn_down.weight").c_str(),  cfg.n_ff,   cfg.n_embd);
        add((p + "ffn_up.weight").c_str(),    cfg.n_embd, cfg.n_ff);
    }
    add("output_norm.weight", cfg.n_embd, 1);
    add("output.weight",      cfg.n_embd, cfg.n_vocab);

    return gctx;
}

static llama_model * harness_load_model(const harness_model_cfg & cfg, const char * save_path) {
    gguf_context * gctx = harness_make_gguf(cfg);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.progress_callback = [](float, void *) { return true; };

    size_t seed = cfg.seed;
    llama_model * model = llama_model_init_from_user(gctx, harness_set_tensor_data, &seed, mparams);
    gguf_free(gctx);
    if (!model) {
        fprintf(stderr, "harness: llama_model_init_from_user failed\n");
        return nullptr;
    }
    if (save_path) {
        llama_model_save_to_file(model, save_path);
    }
    return model;
}

// ---------------------------------------------------------------------------
// per-decode graph observation through the scheduler eval callback.
// The scheduler calls the callback per node: ask=true before the node's
// compute range, ask=false after it (ggml/src/ggml-backend.cpp). CPU decode is
// synchronous, so the harness pushes an empty pass before each llama_decode
// and records that decode's nodes into it; the pass is finalized by the
// harness right after decode returns. out_ids is an INPUT leaf (never a
// node), so its executed width is observed through the gather's src[1].
// Shapes recorded in ask=false callbacks reflect the post-bind executed
// shapes: they are the per-step evidence for the narrowing.
// ---------------------------------------------------------------------------

struct pass_rec {
    int64_t gathers      = 0;
    int64_t out_ids_ne0  = -1;
    int64_t logits_ne0   = -1;
    int64_t logits_ne1   = -1;
    int64_t logits_ne2   = -1;
    int64_t logits_ne3   = -1;
    size_t  logits_nb0   = 0;
    size_t  logits_nb1   = 0;
    const void * logits_node = nullptr;
    bool     saw_logits  = false;
};

struct cb_state {
    std::mutex       mx;
    std::vector<pass_rec> passes;
};

static bool harness_cb_eval(ggml_tensor * t, bool ask, void * ud) {
    if (ask) {
        return true;  // always compute; recording happens post-node below
    }
    auto * st = (cb_state *) ud;
    std::lock_guard<std::mutex> lk(st->mx);
    if (st->passes.empty()) {
        return true;  // a compute outside a tracked decode (e.g. probes)
    }
    pass_rec & r = st->passes.back();
    if (t->op == GGML_OP_GET_ROWS && t->src[1] != nullptr && t->src[1]->type == GGML_TYPE_I32) {
        ++r.gathers;
        r.out_ids_ne0 = t->src[1]->ne[0];
    }
    if (t->name[0] != '\0' && strstr(t->name, "result_output") == t->name) {
        r.saw_logits  = true;
        r.logits_ne0  = t->ne[0];
        r.logits_ne1  = t->ne[1];
        r.logits_ne2  = t->ne[2];
        r.logits_ne3  = t->ne[3];
        r.logits_nb0  = t->nb[0];
        r.logits_nb1  = t->nb[1];
        r.logits_node = t;
    }
    return true;
}

// ---------------------------------------------------------------------------
// part 1: long-resident contexts, 0/1/3/7/16 alternating
// ---------------------------------------------------------------------------

static const uint32_t K_STEPS[] = { 0, 1, 3, 7, 16, 1, 1, 0, 16, 0, 3, 7, 16, 1, 0 };
static const uint32_t N_STEPS   = (uint32_t)(sizeof(K_STEPS) / sizeof(K_STEPS[0]));
static const uint32_t N_TOK     = 24;  // tokens per step; single ubatch (n_ubatch 32)
static const uint32_t N_ENV     = 16;  // output envelope = n_outputs_max of the reuse ctx

// The KV cache pads n_kv so the attention mask width stays constant within
// buckets of 256 (llama-kv-cache.cpp get_n_kv: max(1,256) padding), and the
// mask width is part of the reuse key (llm_graph_input_attn_kv::can_reuse).
// The step grid is 24 tokens/step, so bucket 256 covers steps 0..9 and 512
// covers 10..14 (n_ctx 512 = physical cell count cap). The harness derives
// expectations from this documented mechanic; any divergence fails loudly.
static uint32_t derived_mask_ne0(uint32_t step) {
    const uint32_t used = N_TOK * (step + 1);  // cells populated incl. current step
    const uint32_t pad  = 256;
    uint32_t nkv = GGML_PAD(used, pad);
    if (nkv < pad) {
        nkv = pad;
    }
    const uint32_t phys = 512;  // N_CTX_PHYS below; kept in sync
    return nkv > phys ? phys : nkv;
}

static const uint32_t N_CTX_PHYS = 512;  // text context (already 256-padded)

struct ctx_run {
    const char *    name = nullptr;
    llama_context * ctx  = nullptr;
    cb_state        cb;
    llama_batch     batch {};
    std::vector<int64_t> reused_delta;
    int64_t         n_reused_total = 0;
};

static void ctx_run_init_batch(ctx_run & r) {
    r.batch = llama_batch_init(N_TOK, 0, 1);
}

static void ctx_run_fill(ctx_run & r, uint32_t step, uint32_t k_out, bool all_outputs) {
    llama_batch & b = r.batch;
    const int32_t base = (int32_t)(step * N_TOK);
    for (uint32_t i = 0; i < N_TOK; ++i) {
        b.token[i]        = (llama_token)((11 + 7 * (int) i + 3 * base) % 97);
        b.pos[i]          = base + (int32_t) i;
        b.n_seq_id[i]     = 1;
        b.seq_id[i][0]    = 0;
        // k == 0: non-empty all-zero flags (never a null pointer)
        b.logits[i]       = all_outputs ? 1 : (i < k_out ? 1 : 0);
    }
    b.n_tokens = (int32_t) N_TOK;
}

static int run_ctx_part(llama_model * model, uint32_t n_vocab, const char * dump_prefix) {
    int failures0 = g_failures;

    if (getenv("GGML_TP5_NOUT_REUSE")) {
        fprintf(stderr, "harness: GGML_TP5_NOUT_REUSE is set in the environment; the harness "
                        "controls the gate per-context via setenv - unset it for a clean run\n");
        return 2;
    }

    auto make_params = [&](uint32_t n_outputs_max, cb_state * cb) {
        llama_context_params p = llama_context_default_params();
        p.n_ctx            = N_CTX_PHYS;
        p.n_batch          = 32;
        p.n_ubatch         = 32;  // single ubatch per decode (runbook requirement)
        p.n_seq_max        = 1;
        p.n_outputs_max    = n_outputs_max;
        p.n_threads        = 1;
        p.n_threads_batch  = 1;
        p.embeddings       = false;
        p.no_perf          = false;  // n_reused must be tracked
        p.cb_eval          = cb ? &harness_cb_eval : nullptr;
        p.cb_eval_user_data = cb;
        return p;
    };

    ctx_run run_exact; run_exact.name = "exact";
    ctx_run run_reuse; run_reuse.name = "reuse";
    ctx_run run_ref;   run_ref.name   = "ref";

    // the gate is read at context-construction time from the environment;
    // setenv/unsetenv around each construction gives per-context control
    setenv("GGML_TP5_NOUT_REUSE", "0", 1);
    run_exact.ctx = llama_init_from_model(model, make_params(N_ENV, &run_exact.cb));
    setenv("GGML_TP5_NOUT_REUSE", "1", 1);
    run_reuse.ctx = llama_init_from_model(model, make_params(N_ENV, &run_reuse.cb));
    setenv("GGML_TP5_NOUT_REUSE", "0", 1);
    run_ref.ctx   = llama_init_from_model(model, make_params(N_TOK, &run_ref.cb));
    unsetenv("GGML_TP5_NOUT_REUSE");

    if (!run_exact.ctx || !run_reuse.ctx || !run_ref.ctx) {
        fprintf(stderr, "harness: context creation failed\n");
        return 2;
    }

    ctx_run_init_batch(run_exact);
    ctx_run_init_batch(run_reuse);
    ctx_run_init_batch(run_ref);

    llama_perf_context_reset(run_exact.ctx);
    llama_perf_context_reset(run_reuse.ctx);
    llama_perf_context_reset(run_ref.ctx);

    const size_t row_bytes = (size_t) n_vocab * sizeof(float);
    guarded_buf stage_exact, stage_reuse, stage_ref, row0_snap;
    stage_exact.alloc(row_bytes * N_ENV);
    stage_reuse.alloc(row_bytes * N_ENV);
    stage_ref.alloc(row_bytes * N_TOK);
    row0_snap.alloc(row_bytes);

    for (uint32_t s = 0; s < N_STEPS; ++s) {
        const uint32_t k = K_STEPS[s];
        pass_rec rec_exact, rec_reuse, rec_ref;
        int64_t d_exact = 0, d_reuse = 0, d_ref = 0;

        // ---- ctx_exact (stock semantics, exact-count graph) ----
    {
        const int32_t before = llama_perf_context(run_exact.ctx).n_reused;
        float * pub_before = nullptr;
        if (k == 0) {
            // pre-decode snapshot of the single public row (capacity >= 1 row
            // by n_seq_max=1); compared after the decode below
            pub_before = llama_get_logits(run_exact.ctx);
            CHECK(pub_before != nullptr, "exact step %" PRIu32 ": public logits base null", s);
            if (pub_before) {
                memcpy(row0_snap.ptr, pub_before, row_bytes);
            }
        }
        ctx_run_fill(run_exact, s, k, false);
        run_exact.cb.mx.lock();
        run_exact.cb.passes.push_back({});
        run_exact.cb.mx.unlock();
        const int rc = llama_decode(run_exact.ctx, run_exact.batch);
        CHECK(rc == 0, "exact step %" PRIu32 " (k=%" PRIu32 "): decode rc=%d", s, k, rc);
        run_exact.cb.mx.lock();
        rec_exact = run_exact.cb.passes.empty() ? pass_rec{} : run_exact.cb.passes.back();
        run_exact.cb.mx.unlock();
        llama_synchronize(run_exact.ctx);
        if (k == 0 && pub_before) {
            CHECK(memcmp(row0_snap.ptr, pub_before, row_bytes) == 0,
                  "exact step %" PRIu32 ": zero-output decode modified the public buffer", s);
        }
        d_exact = llama_perf_context(run_exact.ctx).n_reused - before;
    }
        // ---- ctx_reuse (gate on, envelope graph narrowed in place) ----
    {
        const int32_t before = llama_perf_context(run_reuse.ctx).n_reused;
        float * pub_before = nullptr;
        if (k == 0) {
            pub_before = llama_get_logits(run_reuse.ctx);
            CHECK(pub_before != nullptr, "reuse step %" PRIu32 ": public logits base null", s);
            if (pub_before) {
                memcpy(row0_snap.ptr, pub_before, row_bytes);
            }
        }
        ctx_run_fill(run_reuse, s, k, false);
        run_reuse.cb.mx.lock();
        run_reuse.cb.passes.push_back({});
        run_reuse.cb.mx.unlock();
        const int rc = llama_decode(run_reuse.ctx, run_reuse.batch);
        CHECK(rc == 0, "reuse step %" PRIu32 " (k=%" PRIu32 "): decode rc=%d", s, k, rc);
        run_reuse.cb.mx.lock();
        rec_reuse = run_reuse.cb.passes.empty() ? pass_rec{} : run_reuse.cb.passes.back();
        run_reuse.cb.mx.unlock();
        llama_synchronize(run_reuse.ctx);
        if (k == 0 && pub_before) {
            CHECK(memcmp(row0_snap.ptr, pub_before, row_bytes) == 0,
                  "reuse step %" PRIu32 ": zero-output decode modified the public buffer", s);
        }
        d_reuse = llama_perf_context(run_reuse.ctx).n_reused - before;
    }
        // ---- ctx_ref (all outputs; row-to-token anchor) ----
    {
        const int32_t before = llama_perf_context(run_ref.ctx).n_reused;
        ctx_run_fill(run_ref, s, k, true);
        run_ref.cb.mx.lock();
        run_ref.cb.passes.push_back({});
        run_ref.cb.mx.unlock();
        const int rc = llama_decode(run_ref.ctx, run_ref.batch);
        CHECK(rc == 0, "ref step %" PRIu32 " (k=%" PRIu32 "): decode rc=%d", s, k, rc);
        run_ref.cb.mx.lock();
        rec_ref = run_ref.cb.passes.empty() ? pass_rec{} : run_ref.cb.passes.back();
        run_ref.cb.mx.unlock();
        llama_synchronize(run_ref.ctx);
        d_ref = llama_perf_context(run_ref.ctx).n_reused - before;
    }

        // ---- executed shapes (real observations, not derivations) ----
        const pass_rec * recs[3]  = { &rec_exact, &rec_reuse, &rec_ref };
        const char *     names[3] = { "exact", "reuse", "ref" };
        for (int i = 0; i < 3; ++i) {
            const pass_rec & rec = *recs[i];
            const uint32_t want_rows = (i == 2) ? N_TOK : k;
            CHECK(rec.gathers == 2, "ctx_%s step %" PRIu32 ": gathers=%" PRId64 " != 2",
                  names[i], s, rec.gathers);
            CHECK(rec.out_ids_ne0 == (int64_t) want_rows,
                  "ctx_%s step %" PRIu32 ": out_ids ne[0]=%" PRId64 " != %" PRIu32,
                  names[i], s, rec.out_ids_ne0, want_rows);
            CHECK(rec.saw_logits, "ctx_%s step %" PRIu32 ": result_output not observed", names[i], s);
            CHECK(rec.logits_ne0 == (int64_t) n_vocab,
                  "ctx_%s step %" PRIu32 ": logits ne[0]=%" PRId64 " != %u", names[i], s,
                  rec.logits_ne0, (unsigned) n_vocab);
            CHECK(rec.logits_ne1 == (int64_t) want_rows,
                  "ctx_%s step %" PRIu32 ": logits ne[1]=%" PRId64 " != %" PRIu32,
                  names[i], s, rec.logits_ne1, want_rows);
            CHECK(rec.logits_ne2 == 1 && rec.logits_ne3 == 1,
                  "ctx_%s step %" PRIu32 ": logits ne[2..3] = %" PRId64 "/%" PRId64,
                  names[i], s, rec.logits_ne2, rec.logits_ne3);
        }

        // ---- copy the public rows only (n_outputs x n_vocab floats) ----
        auto copy_rows = [&](const char * nm, llama_context * cx, uint32_t rows, guarded_buf & dst) {
            if (rows == 0) {
                return;
            }
            for (uint32_t j = 0; j < rows; ++j) {
                const float * p = llama_get_logits_ith(cx, (int32_t) j);
                CHECK(p != nullptr, "ctx_%s step %" PRIu32 ": logits row %" PRIu32 " null",
                      nm, s, j);
                if (!p) {
                    continue;
                }
                memcpy(dst.ptr + (size_t) j * row_bytes, p, row_bytes);
            }
        };

        if (k > 0) {
            copy_rows("exact", run_exact.ctx, k, stage_exact);
            copy_rows("reuse", run_reuse.ctx, k, stage_reuse);
            copy_rows("ref",   run_ref.ctx, k, stage_ref);
            CHECK(memcmp(stage_exact.ptr, stage_reuse.ptr, (size_t) k * row_bytes) == 0,
                  "step %" PRIu32 " (k=%" PRIu32 "): reuse vs exact logits BYTE MISMATCH (%zu bytes compared)",
                  s, k, (size_t) k * row_bytes);
            CHECK(memcmp(stage_ref.ptr, stage_exact.ptr, (size_t) k * row_bytes) == 0,
                  "step %" PRIu32 " (k=%" PRIu32 "): exact vs all-output reference BYTE MISMATCH",
                  s, k);
        }
    // k == 0: the public-buffer-unchanged checks ran inside the per-context
    // blocks above (pre-decode snapshot vs post-sync compare). No logits row
    // is dereferenced anywhere for zero-output steps.

        run_exact.reused_delta.push_back(d_exact);
        run_reuse.reused_delta.push_back(d_reuse);
        run_ref.reused_delta.push_back(d_ref);

        if (s > 0) {
            const bool same_bucket = derived_mask_ne0(s) == derived_mask_ne0(s - 1);
            if (same_bucket) {
                // inside one n_kv bucket the only changing key is the output
                // count: the lane must reuse across it
                CHECK(d_reuse == 1,
                      "step %" PRIu32 ": ctx_reuse n_reused delta %" PRId64 " != 1 "
                      "(cross-count reuse failed)", s, d_reuse);
                if (K_STEPS[s] != K_STEPS[s - 1]) {
                    CHECK(d_exact == 0,
                          "step %" PRIu32 ": ctx_exact reused across an output-count change", s);
                }
            }
        }

        if (dump_prefix) {
            const ctx_run * runs[3]     = { &run_exact, &run_reuse, &run_ref };
            const guarded_buf * bufs[3] = { &stage_exact, &stage_reuse, &stage_ref };
            for (int i = 0; i < 3; ++i) {
                char path[512];
                snprintf(path, sizeof(path), "%s.%s.bin", dump_prefix, runs[i]->name);
                FILE * f = fopen(path, s == 0 ? "wb" : "ab");  // truncate at the first step
                if (!f) {
                    fprintf(stderr, "harness: cannot open dump %s\n", path);
                    return 2;
                }
                const uint32_t rows = (i == 2) ? N_TOK : k;
                const uint32_t hdr[3] = { s, rows, n_vocab };
                fwrite(hdr, sizeof(hdr), 1, f);
                if (rows > 0) {
                    fwrite(bufs[i]->ptr, row_bytes, rows, f);
                }
                fclose(f);
            }
        }

        TRACE("step %2" PRIu32 " k=%2" PRIu32 " | delta e/r/f = %" PRId64 "/%" PRId64 "/%" PRId64
              " | bucket %" PRIu32 " | logits ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]"
              " nb0 %zu nb1 %zu | out_ids ne0=%" PRId64 " | node=%p\n",
              s, k, d_exact, d_reuse, d_ref, derived_mask_ne0(s), rec_reuse.logits_ne0,
              rec_reuse.logits_ne1, rec_reuse.logits_ne2, rec_reuse.logits_ne3,
              rec_reuse.logits_nb0, rec_reuse.logits_nb1, rec_reuse.out_ids_ne0,
              rec_reuse.logits_node);
    }

    run_exact.n_reused_total = llama_perf_context(run_exact.ctx).n_reused;
    run_reuse.n_reused_total = llama_perf_context(run_reuse.ctx).n_reused;
    run_ref.n_reused_total   = llama_perf_context(run_ref.ctx).n_reused;

    // derived totals: reuse lane reuses on every same-bucket pair; exact only
    // when the count also repeats (the 1->1 pair at steps 5->6)
    {
        int64_t exp_reuse = 0;
        int64_t exp_exact = 0;
        for (uint32_t s = 1; s < N_STEPS; ++s) {
            if (derived_mask_ne0(s) == derived_mask_ne0(s - 1)) {
                ++exp_reuse;
                if (K_STEPS[s] == K_STEPS[s - 1]) {
                    ++exp_exact;
                }
            }
        }
        CHECK(run_reuse.n_reused_total == exp_reuse,
              "ctx_reuse total reused %" PRId64 " != derived %" PRId64,
              run_reuse.n_reused_total, exp_reuse);
        CHECK(run_exact.n_reused_total == exp_exact,
              "ctx_exact total reused %" PRId64 " != derived %" PRId64,
              run_exact.n_reused_total, exp_exact);
        CHECK(run_reuse.n_reused_total > run_exact.n_reused_total,
              "lane must reuse strictly more than exact-count graphs (%" PRId64 " vs %" PRId64 ")",
              run_reuse.n_reused_total, run_exact.n_reused_total);
    }

    llama_batch_free(run_exact.batch);
    llama_batch_free(run_reuse.batch);
    llama_batch_free(run_ref.batch);
    llama_free(run_exact.ctx);
    llama_free(run_ref.ctx);
    llama_free(run_reuse.ctx);

    // only the gate-on context ever emits the stats line; attribution is
    // unambiguous regardless of free order
    CHECK(g_log.count_contains("NOUT_REUSE_STATS") == 1,
          "expected exactly one NOUT_REUSE_STATS line, found %zu",
          g_log.count_contains("NOUT_REUSE_STATS"));
    {
        size_t binds = (size_t)-1;
        size_t fallbacks = (size_t)-1;
        const bool ok = g_log.parse_last("NOUT_REUSE_STATS ",
                                         "NOUT_REUSE_STATS binds=%zu fallbacks=%zu",
                                         &binds, &fallbacks);
        CHECK(ok, "could not parse the NOUT_REUSE_STATS line");
        if (ok) {
            CHECK(binds == N_STEPS, "binds=%zu != steps=%u (every decode must bind 0..envelope)",
                  binds, (unsigned) N_STEPS);
            CHECK(fallbacks == 0,
                  "fallbacks=%zu != 0 (a fallback means the planner rejected the real closure)",
                  fallbacks);
        }
    }

    CHECK(g_log.count_contains("runtime n_outputs binding enabled (GGML_TP5_NOUT_REUSE=1") == 1,
          "gate-on enable line count != 1");
    CHECK(g_log.count_contains("first bind active=0 envelope=16") == 1,
          "first-bind ACTIVE line (active=0 envelope=16) not found exactly once");
    CHECK(g_log.count_contains("binding DISABLED") == 0, "unexpected backend-sampler disable");

    stage_exact.free_buf();
    stage_reuse.free_buf();
    stage_ref.free_buf();
    row0_snap.free_buf();

    return g_failures > failures0 ? 1 : 0;
}

// ---------------------------------------------------------------------------
// part 2: planner fail-closed contract on the real closure shape
// ---------------------------------------------------------------------------

// Pseudo-random fixed active token subset (< N_TOK); out_ids content for
// bind-and-compare rounds. The real runtime writes flagged-token indices
// ascending; the planner is content-agnostic, so a scrambled subset also
// exercises non-identity gathers.
static const int32_t ACTIVE_IDS[] = { 3, 17, 5, 22, 0, 9, 14, 2, 19, 6, 11, 21, 1, 15, 8, 4 };

struct plan_weights {
    int64_t E = 0, V = 0, FF = 0;
    std::vector<float> h0, h1;          // hidden states [E x T]
    std::vector<float> w_up, w_gate;    // [FF x E]
    std::vector<float> w_down;          // [E x FF]
    std::vector<float> w_out;           // [V x E]
    std::vector<float> w_norm, w_outn;  // [E] broadcast row
};

static plan_weights make_plan_weights(int64_t E, int64_t V, int64_t FF, int64_t T) {
    plan_weights w;
    w.E = E; w.V = V; w.FF = FF;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    auto fill = [](std::vector<float> & v, size_t n, std::mt19937 & rng,
                   std::uniform_real_distribution<float> & dis) {
        v.resize(n);
        for (auto & x : v) {
            x = dis(rng);
        }
    };
    fill(w.w_up,   (size_t)(FF * E), rng, dis);   // [FF, E]
    fill(w.w_gate, (size_t)(FF * E), rng, dis);   // [FF, E]
    fill(w.w_down, (size_t)(E * FF), rng, dis);   // [E, FF]
    fill(w.w_out,  (size_t)(V * E),  rng, dis);   // [V, E]
    fill(w.w_norm, (size_t) E,       rng, dis);   // [E] broadcast
    fill(w.w_outn, (size_t) E,       rng, dis);   // [E] broadcast
    fill(w.h0,     (size_t)(E * T),  rng, dis);   // [E, T] hidden states
    fill(w.h1,     (size_t)(E * T),  rng, dis);
    return w;
}

// closure shape mirrored from src/models/llama.cpp (gather inside the last
// block, then ffn/norm/lm_head): every op is in the planner's accepted set
// (llama-nout-bind.h) except where noted. head_mode 0 = plain head (scale +
// cont cover the accepted SCALE/CONT ops); 1 = softmax consumer (unsupported:
// must reject); 2 = reshape consumer (unsupported: must reject).
struct plan_graph {
    ggml_context *  ctx  = nullptr;
    ggml_cgraph *   gf   = nullptr;
    ggml_tensor *   oids = nullptr;  // I32 [ENV]
    ggml_tensor *   lg   = nullptr;  // F32 [n_vocab, ENV] (post SCALE+CONT head)
};

// one shared graph-building lambda; rows == T builds the envelope graph,
// rows < T builds an exact-count reference. weights are materialized inside
// the given context (same bytes), mirroring llama's per-graph weight views.
static plan_graph build_closure(ggml_context * ctx, const plan_weights & w, int64_t T, int64_t rows,
                                int head_mode) {
    plan_graph g;
    g.ctx = ctx;

    ggml_tensor * hstates0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.E, T);
    ggml_tensor * hstates1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.E, T);
    ggml_tensor * w_up     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.FF, w.E);
    ggml_tensor * w_gate   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.FF, w.E);
    ggml_tensor * w_down   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.E, w.FF);
    ggml_tensor * w_out    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w.V, w.E);
    ggml_tensor * w_norm   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, w.E);
    ggml_tensor * w_outn   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, w.E);

    ggml_backend_tensor_set(hstates0, w.h0.data(), 0, ggml_nbytes(hstates0));
    ggml_backend_tensor_set(hstates1, w.h1.data(), 0, ggml_nbytes(hstates1));
    ggml_backend_tensor_set(w_up,   w.w_up.data(),   0, ggml_nbytes(w_up));
    ggml_backend_tensor_set(w_gate, w.w_gate.data(), 0, ggml_nbytes(w_gate));
    ggml_backend_tensor_set(w_down, w.w_down.data(), 0, ggml_nbytes(w_down));
    ggml_backend_tensor_set(w_out,  w.w_out.data(),  0, ggml_nbytes(w_out));
    ggml_backend_tensor_set(w_norm, w.w_norm.data(), 0, ggml_nbytes(w_norm));
    ggml_backend_tensor_set(w_outn, w.w_outn.data(), 0, ggml_nbytes(w_outn));

    g.oids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rows);
    ggml_set_input(g.oids);

    ggml_tensor * g0 = ggml_get_rows(ctx, hstates0, g.oids);
    ggml_tensor * g1 = ggml_get_rows(ctx, hstates1, g.oids);

    ggml_tensor * cur = g0;
    if (head_mode == 1) {
        cur = ggml_soft_max(ctx, g0);  // unsupported consumer -> must reject
    } else if (head_mode == 2) {
        cur = ggml_reshape_1d(ctx, g0, w.E * rows);  // unsupported layout op
    }
    ggml_tensor * inpSA = g1;

    ggml_tensor * ffnin = ggml_add(ctx, cur, inpSA);                  // ADD
    ggml_tensor * n1    = ggml_rms_norm(ctx, ffnin, 1e-5f);           // RMS_NORM
    n1 = ggml_mul(ctx, n1, w_norm);                                   // MUL broadcast
    ggml_tensor * up    = ggml_mul_mat(ctx, w_up, n1);                // MUL_MAT
    ggml_tensor * gate  = ggml_mul_mat(ctx, w_gate, n1);              // MUL_MAT
    ggml_tensor * glu   = ggml_swiglu_split(ctx, up, gate);           // GLU (split)
    ggml_tensor * down  = ggml_mul_mat(ctx, w_down, glu);             // MUL_MAT
    ggml_tensor * res   = ggml_add(ctx, down, ffnin);                 // ADD (residual)
    ggml_tensor * n2    = ggml_rms_norm(ctx, res, 1e-5f);             // RMS_NORM
    n2 = ggml_mul(ctx, n2, w_outn);                                   // MUL broadcast
    ggml_tensor * lg    = ggml_mul_mat(ctx, w_out, n2);               // MUL_MAT (head)
    lg = ggml_scale(ctx, lg, 1.0f);                                   // SCALE
    lg = ggml_cont(ctx, lg);                                          // CONT

    g.lg = lg;
    g.gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(g.gf, lg);
    return g;
}

static void fill_oids(plan_graph & g, uint32_t rows) {
    if (rows == 0) {
        return;  // nothing to write; avoids a zero-size set with a null source
    }
    std::vector<int32_t> ids(rows);
    for (uint32_t i = 0; i < rows; ++i) {
        ids[i] = ACTIVE_IDS[i % (sizeof(ACTIVE_IDS) / sizeof(ACTIVE_IDS[0]))];
    }
    ggml_backend_tensor_set(g.oids, ids.data(), 0, rows * sizeof(int32_t));
}

struct shape_snap {
    ggml_tensor * t;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    std::vector<char> data;
};

static std::vector<shape_snap> snapshot_graph(ggml_cgraph * gf) {
    std::vector<shape_snap> snaps;
    const int n = ggml_graph_n_nodes(gf);
    snaps.reserve((size_t) n);
    for (int i = 0; i < n; ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        shape_snap s;
        s.t = t;
        memcpy(s.ne, t->ne, sizeof(s.ne));
        memcpy(s.nb, t->nb, sizeof(s.nb));
        if (t->data) {
            const size_t bytes = ggml_nbytes(t);
            s.data.resize(bytes);
            memcpy(s.data.data(), t->data, bytes);
        }
        snaps.push_back(std::move(s));
    }
    return snaps;
}

static void check_snapshots_untouched(const std::vector<shape_snap> & snaps, const char * what) {
    for (const auto & s : snaps) {
        CHECK(memcmp(s.t->ne, s.ne, sizeof(s.ne)) == 0 && memcmp(s.t->nb, s.nb, sizeof(s.nb)) == 0,
              "%s: tensor %p ne/nb modified by a rejected bind", what, (const void *) s.t);
        if (!s.data.empty() && s.t->data) {
            CHECK(memcmp(s.t->data, s.data.data(), s.data.size()) == 0,
                  "%s: tensor %p data modified by a rejected bind", what, (const void *) s.t);
        }
    }
}

static int run_plan_part() {
    int failures0 = g_failures;

    const int64_t E = 64, V = 128, FF = 128, T = N_TOK, ENV = N_ENV;
    plan_weights w = make_plan_weights(E, V, FF, T);

    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) {
        fprintf(stderr, "harness: ggml_backend_cpu_init failed\n");
        return 2;
    }

    // ---- envelope graph + compute (identity ids over the active subset) ----
    ggml_init_params ip { 64 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    plan_graph g = build_closure(ctx, w, T, ENV, 0);

    fill_oids(g, (uint32_t) ENV);
    ggml_backend_graph_compute(be, g.gf);
    guarded_buf env_out;
    env_out.alloc((size_t) V * ENV * sizeof(float));
    ggml_backend_tensor_get(g.lg, env_out.ptr, 0, (size_t) V * ENV * sizeof(float));

    // ---- union envelope record: mirrors llama-graph.cpp bind_outputs' side
    // table. env_ne/env_nb recorded at first plan-add (pristine envelope);
    // restore = memcpy back before every new bind. ----
    std::vector<llama_nout_plan_entry> env_union;

    auto restore_envelope = [&]() {
        for (const auto & e : env_union) {
            memcpy(e.t->ne, e.env_ne, sizeof(e.env_ne));
            memcpy(e.t->nb, e.env_nb, sizeof(e.env_nb));
        }
    };

    auto merge_plan = [&](const llama_nout_plan & plan) {
        for (size_t i = 0; i < plan.n_entries; ++i) {
            const llama_nout_plan_entry & e = plan.entries[i];
            bool found = false;
            for (const auto & u : env_union) {
                if (u.t == e.t) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                env_union.push_back(e);
            }
        }
    };

    auto verify_narrowed = [&](uint32_t rows) {
        CHECK(g.oids->ne[0] == (int64_t) rows, "oids ne[0]=%" PRId64 " != %" PRIu32, g.oids->ne[0], rows);
        CHECK(g.oids->nb[0] == sizeof(int32_t), "oids nb[0] modified");
        CHECK(ggml_is_contiguous(g.oids), "oids not contiguous after bind");
        for (const auto & e : env_union) {
            if (e.t == g.oids) {
                continue;
            }
            CHECK(e.t->ne[1] == (int64_t) rows, "tensor %p ne[1]=%" PRId64 " != %" PRIu32,
                  (const void *) e.t, e.t->ne[1], rows);
            // contiguous nb recomputation (nout_narrow_nb contract)
            CHECK(e.t->nb[1] == e.t->nb[0] * (size_t) e.t->ne[0],
                  "tensor %p nb[1] not contiguous", (const void *) e.t);
            for (int d = 2; d < GGML_MAX_DIMS; ++d) {
                CHECK(e.t->nb[d] == e.t->nb[d - 1] * (size_t) e.t->ne[d - 1],
                      "tensor %p nb[%d] not contiguous", (const void *) e.t, d);
            }
            CHECK(e.t->ne[2] == e.env_ne[2] && e.t->ne[3] == e.env_ne[3],
                  "tensor %p ne[2..3] disturbed", (const void *) e.t);
        }
    };

    // exact-count reference: fresh graph built at `rows` (stock path), computed
    // on the same backend with the same ids; compared bit-exactly.
    auto exact_reference = [&](uint32_t rows, guarded_buf & out) {
        ggml_init_params rip { 64 * 1024 * 1024, nullptr, false };
        ggml_context * rctx = ggml_init(rip);
        plan_graph rg = build_closure(rctx, w, T, (int64_t) rows, 0);
        fill_oids(rg, rows);
        const enum ggml_status st = ggml_backend_graph_compute(be, rg.gf);
        CHECK(st == GGML_STATUS_SUCCESS, "exact reference rows=%" PRIu32 " compute status %d", rows, (int) st);
        out.alloc((size_t) V * rows * sizeof(float) + 64);
        if (rows > 0) {
            ggml_backend_tensor_get(rg.lg, out.ptr, 0, (size_t) V * rows * sizeof(float));
        }
        ggml_free(rctx);
    };

    // ---- bind sweep on the SAME graph: 0/1/3/7/16 + grow-back + re-shrink ----
    const uint32_t SWEEP[] = { 0, 1, 3, 7, 16, 1, 16, 0, 7, 3, 16, 1, 0, 16 };
    for (uint32_t si = 0; si < sizeof(SWEEP) / sizeof(SWEEP[0]); ++si) {
        const uint32_t rows = SWEEP[si];

        restore_envelope();

        llama_nout_plan plan;
        memset(&plan, 0, sizeof(plan));
        const bool bound = llama_nout_plan_bind(g.gf, g.oids, (uint32_t) ENV, rows, &plan);
        CHECK(bound, "sweep %" PRIu32 " (rows=%" PRIu32 "): planner rejected a bindable closure", si, rows);
        if (!bound) {
            llama_nout_plan_free(&plan);
            continue;
        }
        merge_plan(plan);

        llama_nout_plan_apply(&plan);
        verify_narrowed(rows);

        // idempotency: a second apply must leave the narrowed shapes as-is
        llama_nout_plan_apply(&plan);
        verify_narrowed(rows);
        llama_nout_plan_free(&plan);

        fill_oids(g, rows);
        const enum ggml_status st = ggml_backend_graph_compute(be, g.gf);
        CHECK(st == GGML_STATUS_SUCCESS, "sweep %" PRIu32 " (rows=%" PRIu32 "): compute status %d",
              si, rows, (int) st);

        guarded_buf ref;
        exact_reference(rows, ref);
        if (rows > 0) {
            CHECK(memcmp(ref.ptr, g.lg->data, (size_t) V * rows * sizeof(float)) == 0,
                  "sweep %" PRIu32 " (rows=%" PRIu32 "): narrowed compute != exact-count build (bit-exact)",
                  si, rows);
        } else {
            // real zero-row execution: with the transitive closure walk
            // (4f416f5) the whole chain incl. the head is planned, so the
            // head tensor must be empty; against a depth-1-only walk this check
            // correctly fails (the head would stay at envelope width).
            // Grow-back binds later in the sweep prove the graph survives.
            CHECK(ggml_is_empty(g.lg), "sweep %" PRIu32 ": logits not empty at rows=0", si);
        }
        ref.free_buf();

        TRACE("plan sweep %2" PRIu32 " rows=%2" PRIu32 " ok (union entries=%zu)\n",
              si, rows, env_union.size());
    }

    // ---- restore to envelope and re-check the original envelope output ----
    restore_envelope();
    {
        llama_nout_plan plan;
        memset(&plan, 0, sizeof(plan));
        const bool bound = llama_nout_plan_bind(g.gf, g.oids, (uint32_t) ENV, (uint32_t) ENV, &plan);
        CHECK(bound, "final envelope re-bind rejected");
        if (bound) {
            merge_plan(plan);
            llama_nout_plan_apply(&plan);
            llama_nout_plan_free(&plan);
            fill_oids(g, (uint32_t) ENV);
            ggml_backend_graph_compute(be, g.gf);
            CHECK(memcmp(env_out.ptr, g.lg->data, (size_t) V * ENV * sizeof(float)) == 0,
                  "envelope restore does not reproduce the original envelope output (bit-exact)");
        }
    }

    // ---- failure modes: every reject must leave ALL nodes' ne/nb AND data
    // byte-identical (fail-closed contract) ----
    {
        // over-envelope
        restore_envelope();
        const auto snap = snapshot_graph(g.gf);
        llama_nout_plan plan;
        memset(&plan, 0, sizeof(plan));
        const bool bound = llama_nout_plan_bind(g.gf, g.oids, (uint32_t) ENV, (uint32_t) ENV + 1, &plan);
        CHECK(!bound, "over-envelope bind (env+1) was accepted");
        CHECK(plan.entries == nullptr && plan.n_entries == 0, "rejected over-envelope plan not zeroed");
        check_snapshots_untouched(snap, "over-envelope");

        // zero plan on a zeroed struct is safe to free
        llama_nout_plan_free(&plan);
    }
    {
        // unsupported consumer: softmax after the gather (sampler-shaped graph)
        ggml_init_params sip { 32 * 1024 * 1024, nullptr, false };
        ggml_context * sctx = ggml_init(sip);
        plan_graph sg = build_closure(sctx, w, T, ENV, 1);
        fill_oids(sg, (uint32_t) ENV);
        const auto snap = snapshot_graph(sg.gf);
        llama_nout_plan plan;
        memset(&plan, 0, sizeof(plan));
        const bool bound = llama_nout_plan_bind(sg.gf, sg.oids, (uint32_t) ENV, 7, &plan);
        CHECK(!bound, "softmax consumer in the closure was accepted");
        CHECK(plan.entries == nullptr, "rejected softmax plan not zeroed");
        check_snapshots_untouched(snap, "softmax-consumer");
        ggml_free(sctx);
    }
    {
        // unsupported layout op: reshape in the closure
        ggml_init_params rip { 32 * 1024 * 1024, nullptr, false };
        ggml_context * rctx = ggml_init(rip);
        plan_graph rg = build_closure(rctx, w, T, ENV, 2);
        fill_oids(rg, (uint32_t) ENV);
        const auto snap = snapshot_graph(rg.gf);
        llama_nout_plan plan;
        memset(&plan, 0, sizeof(plan));
        const bool bound = llama_nout_plan_bind(rg.gf, rg.oids, (uint32_t) ENV, 7, &plan);
        CHECK(!bound, "reshape consumer in the closure was accepted");
        check_snapshots_untouched(snap, "reshape-consumer");
        ggml_free(rctx);
    }
    {
        // envelope mismatch: graph built at 8 rows, bind claims env=16
        ggml_init_params mip { 32 * 1024 * 1024, nullptr, false };
        ggml_context * mctx8 = ggml_init(mip);
        plan_graph mg = build_closure(mctx8, w, T, 8, 0);
        std::vector<int32_t> ids(8);
        for (int i = 0; i < 8; ++i) {
            ids[i] = ACTIVE_IDS[i];
        }
        ggml_backend_tensor_set(mg.oids, ids.data(), 0, 8 * sizeof(int32_t));
        const auto snap = snapshot_graph(mg.gf);
        llama_nout_plan plan;
        memset(&plan, 0, sizeof(plan));
        const bool bound = llama_nout_plan_bind(mg.gf, mg.oids, (uint32_t) ENV, 8, &plan);
        CHECK(!bound, "bind on a graph built below the claimed envelope was accepted");
        check_snapshots_untouched(snap, "envelope-mismatch");
        ggml_free(mctx8);
    }
    {
        // no gather root: out_ids present but never read by GET_ROWS
        ggml_init_params nip { 16 * 1024 * 1024, nullptr, false };
        ggml_context * nctx = ggml_init(nip);
        ggml_tensor * oids = ggml_new_tensor_1d(nctx, GGML_TYPE_I32, ENV);
        ggml_set_input(oids);
        ggml_tensor * w = ggml_new_tensor_2d(nctx, GGML_TYPE_F32, V, E);
        ggml_tensor * x = ggml_new_tensor_2d(nctx, GGML_TYPE_F32, E, T);
        ggml_tensor * lg = ggml_mul_mat(nctx, w, x);
        ggml_cgraph * gf = ggml_new_graph_custom(nctx, GGML_DEFAULT_GRAPH_SIZE, false);
        ggml_build_forward_expand(gf, lg);
        llama_nout_plan plan;
        memset(&plan, 0, sizeof(plan));
        const bool bound = llama_nout_plan_bind(gf, oids, (uint32_t) ENV, 4, &plan);
        CHECK(!bound, "bind without a gather root was accepted");
        ggml_free(nctx);
    }

    env_out.free_buf();
    ggml_free(ctx);
    ggml_backend_free(be);

    return g_failures > failures0 ? 1 : 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static void usage() {
    fprintf(stderr,
            "nout_longres_harness [--mode=all|ctx|plan] [--model=PATH] [--save-gguf=PATH]\n"
            "                     [--dump-prefix=PATH] [--log=PATH] [--trace]\n");
}

static bool compiled_with_asan() {
#if defined(__SANITIZE_ADDRESS__)
    return true;
#elif defined(__has_feature)
# if __has_feature(address_sanitizer)
    return true;
# endif
#endif
    return false;
}

int main(int argc, char ** argv) {
    std::string mode = "all";
    const char * model_path = nullptr;
    const char * save_gguf  = nullptr;
    const char * dump_prefix = nullptr;
    const char * log_path   = nullptr;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--mode=", 0) == 0) {
            mode = a.substr(7);
        } else if (a.rfind("--model=", 0) == 0) {
            model_path = a.c_str() + 8;
        } else if (a.rfind("--save-gguf=", 0) == 0) {
            save_gguf = a.c_str() + 12;
        } else if (a.rfind("--dump-prefix=", 0) == 0) {
            dump_prefix = a.c_str() + 14;
        } else if (a.rfind("--log=", 0) == 0) {
            log_path = a.c_str() + 6;
        } else if (a == "--trace") {
            g_trace = true;
        } else {
            usage();
            return 2;
        }
    }
    if (mode != "all" && mode != "ctx" && mode != "plan") {
        usage();
        return 2;
    }
    if (getenv("GGML_TP5_HARNESS_TRACE")) {
        g_trace = true;
    }

    // keep the unmodified runtime semantics; only the nout gate is controlled
    unsetenv("LLAMA_GRAPH_UNION_NKV_RUNTIME");
    unsetenv("LLAMA_GRAPH_REUSE_DISABLE");

    llama_log_set(&harness_log_cb, nullptr);

    harness_model_cfg cfg;
    uint32_t n_vocab = cfg.n_vocab;

    llama_model * model = nullptr;
    if (model_path) {
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;
        model = llama_model_load_from_file(model_path, mparams);
        if (!model) {
            fprintf(stderr, "harness: failed to load %s\n", model_path);
            return 2;
        }
        n_vocab = (uint32_t) llama_vocab_n_tokens(llama_model_get_vocab(model));
    } else {
        model = harness_load_model(cfg, save_gguf);
        if (!model) {
            return 2;
        }
    }

    fprintf(stderr, "nout_longres_harness: mode=%s n_vocab=%u steps=%u tokens/step=%u envelope=%u "
            "asan=%s\n", mode.c_str(), (unsigned) n_vocab, (unsigned) N_STEPS, (unsigned) N_TOK,
            (unsigned) N_ENV, compiled_with_asan() ? "on" : "off");

    if (mode == "ctx" || mode == "all") {
        run_ctx_part(model, n_vocab, dump_prefix);
    }
    if (mode == "plan" || mode == "all") {
        run_plan_part();
    }

    llama_model_free(model);

    if (log_path) {
        FILE * f = fopen(log_path, "w");
        if (f) {
            for (const auto & l : g_log.lines) {
                fprintf(f, "%s\n", l.c_str());
            }
            fclose(f);
        }
    }

    if (g_failures) {
        fprintf(stderr, "nout_longres_harness: %d FAILURES\n", g_failures);
        return 1;
    }
    fprintf(stderr, "nout_longres_harness: PASS (mode=%s: reuse counted from the runtime, "
            "outputs bit-exact, boundaries fail-closed)\n", mode.c_str());
    return 0;
}
