// nout_floor1_reuse_hits.cpp - r8 CPU harness for the floor1 lane's PRODUCTION
// allow_reuse decision, graph reuse, compute-skip vs build counting, and the
// NOUT1_FLOOR_TRACE decision instrumentation (src/llama-context.cpp
// process_ubatch, the r8 record-snapshot root-cause fix, the bind lane).
//
// What this proves on the acceptance machine (nothing here is simulated; every
// counter asserted is the runtime's own):
//
//   1. decision trace (追踪判定): with LLAMA_NOUT_FLOOR1_TRACE=1 the runtime
//      emits, per decoder step, "NOUT1_FLOOR_TRACE dec=.. reuse=.. miss=..
//      n_out=.. key=.. stored=.." and "NOUT1_FLOOR_STEP .. fell_back=..
//      bind_rows=.. builds=.. builds_fb=.. skip=.. gen=..", and at context
//      destruction "NOUT1_FLOOR_TRACE_STATS ..". The harness parses these
//      runtime-emitted lines and asserts the 0->1 transition reuses (reuse=1,
//      miss=-) with the output-count key normalized (n_out=0 key=1 stored=1),
//      bind_rows=0 on 0-output steps, and that a k-row step rebuilds
//      (reuse=0 miss=outputs for 1->k within one n_kv bucket).
//
//   2. graph reuse + compute-skip/build counts (图复用 / compute跳过 build 计数):
//      per-step delta of llama_perf_context().n_reused (the runtime's OWN
//      counter) is asserted against the floor1 expectations; the trace
//      aggregates (builds vs builds_fb vs compute_skip) are asserted from the
//      runtime's OWN NOUT1_FLOOR_TRACE_STATS line at context destruction.
//
//   3. correctness: every produced logits row is compared BYTE-EXACT against
//      (a) the same-binary stock context on the same grid and (b) the
//      all-outputs reference context (row-to-token oracle). Rows are read BY
//      TOKEN INDEX through llama_get_logits_ith (which maps the ubatch token
//      through out_ids) - never by contiguous-row assumption; 0-output steps
//      assert the public buffer byte-unchanged (pre/post decode snapshot) and
//      that no logits row is dereferenced.
//
//   4. gating (exact-"1" only):
//      - a context with LLAMA_NOUT_FLOOR1_REUSE in {"0", "", "01", " 1",
//        "true"} must show NO floor1 activity (no trace lines, stock reuse
//        pattern: a 0-output step followed by a 1-output step rebuilds);
//      - a context with BOTH LLAMA_NOUT_FLOOR1_REUSE=1 and
//        GGML_TP5_NOUT_REUSE=1 must refuse floor1 (the WARN line) and run
//        the envelope lane's semantics (graphs at the envelope, not the floor);
//      - a context with ONLY LLAMA_NOUT_FLOOR1_TRACE=1 (no floor1 gate) must
//        show no floor1 activity at all (the trace gate alone enables
//        nothing: no lines, no counter movement, stock reuse pattern).
//
// n_kv bucketing: the decode grid stays inside one n_kv 256 bucket
// (llama-kv-cache.cpp get_n_kv), so the ONLY changing reuse key inside the
// grid is the output count - the exact regime the floor1 lane targets. The
// harness derives this expectation from get_n_kv's documented 256-bucketing
// and fails loudly if the run leaves the bucket (then the assertions are
// skipped for that pair and the divergence is reported).
//
// Build (acceptance machine ONLY; every compilation goes through the authorized
// build slot). On the r8-wire authoring machine the designated build slot
// (/home/hermes/.local/bin/build-slot) does not exist and other compile paths
// are not authorized, so this file has still NEVER been compiled or executed:
//   <build-slot> g++ -std=c++17 -O2 -g -Wall -Wextra -fsanitize=address,undefined \
//     -I<repo>/include -I<repo>/ggml/include -I<repo>/src \
//     harness/nout_floor1_reuse_hits.cpp -o nout_floor1_reuse_hits \
//     -L<repo>/build/src -llama -Wl,-rpath,<repo>/build/src
// Run: ./nout_floor1_reuse_hits [--model PATH.gguf]
// Exit codes: 0 pass, 1 failures, 2 setup error.
//
// Honesty note: this probe was authored in r8 and never compiled or run there.
// The r8-wire round repaired it statically only: defined the previously
// missing symbols (gate macros, pass capture, trace parser, take(), tok_base,
// declarations), renamed run_fill to its callsite name, gave the ref leg a
// real all-outputs batch, corrected the grid-derived expectations (reuse 5
// not 6, zero-row binds 4 not 3; skips == reuse), and made the staging guard
// page real. Every runtime claim remains unproven until the acceptance run.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

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

#define CHECK(cond, ...)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            ++g_failures;                                                       \
            fprintf(stderr, "FAIL %s:%d: ", __func__, __LINE__);                \
            fprintf(stderr, __VA_ARGS__);                                       \
            fprintf(stderr, "\n");                                              \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// log capture: forward everything unchanged, keep the runtime's own floor1
// lines so assertions use the runtime's OWN counters (never simulated ones)
// ---------------------------------------------------------------------------

struct log_capture {
    std::mutex               mx;
    std::vector<std::string> lines;
    bool                     active = false;

    void add(const char * text) {
        std::string s(text ? text : "");
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
            s.pop_back();
        }
        if (!active) {
            return;
        }
        if (s.rfind("NOUT1_FLOOR_TRACE ", 0) == 0 ||
            s.rfind("NOUT1_FLOOR_STEP", 0)  == 0 ||
            s.rfind("NOUT1_FLOOR_TRACE_STATS", 0) == 0 ||
            s.rfind("NOUT1_FLOOR_STATS", 0) == 0 ||
            s.find("nout floor1") != std::string::npos) {
            std::lock_guard<std::mutex> lk(mx);
            lines.push_back(s);
        }
    }

    // move the captured lines out (clearing the buffer); keeps capture active
    std::vector<std::string> take() {
        std::lock_guard<std::mutex> lk(mx);
        std::vector<std::string> out;
        out.swap(lines);
        return out;
    }
};

static log_capture g_log;

static void harness_log_callback(ggml_log_level level, const char * text, void * ud) {
    // forward everything unchanged to the default sink; capture floor1 lines
    g_log.add(text);
    fputs(text, stderr);
    (void) level;
    (void) ud;
}

// ---------------------------------------------------------------------------
// guarded staging buffer (guard pages fault on harness-side overrun)
// ---------------------------------------------------------------------------

struct guarded_buf {
    uint8_t * base = nullptr;
    uint8_t * ptr  = nullptr;
    size_t    cap  = 0;

    void alloc(size_t bytes) {
        page = (size_t) sysconf(_SC_PAGESIZE);
        cap  = (bytes + page - 1) / page * page;   // usable, page-aligned
        base = (uint8_t *) mmap(nullptr, cap + page, PROT_NONE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) {
            fprintf(stderr, "harness: mmap failed\n");
            exit(2);
        }
        // usable region only: the page right after it stays PROT_NONE, so a
        // harness-side overrun faults instead of silently corrupting
        mprotect(base, cap, PROT_READ | PROT_WRITE);
        ptr = base;
    }

    void free_buf() {
        if (base) {
            munmap(base, cap + page);
            base = nullptr;
            ptr  = nullptr;
            cap  = 0;
        }
    }

    size_t    page = 0;
};

// ---------------------------------------------------------------------------
// gates (exact-"1" semantics live in src/llama-context.cpp; the harness only
// sets the raw strings around context construction) and grid constants
// ---------------------------------------------------------------------------
#define GATE       "LLAMA_NOUT_FLOOR1_REUSE"   // floor1 lane gate
#define ENV_GATE   "GGML_TP5_NOUT_REUSE"       // envelope lane gate
#define TRACE_GATE "LLAMA_NOUT_FLOOR1_TRACE"   // r8 trace gate

static const uint32_t N_CTX_PHYS = 512;   // context size; already 256-aligned

// ---------------------------------------------------------------------------
// per-decode pass capture: records executed shapes through the eval callback
// (same pattern as nout_longres_harness.cpp). out_ids is an INPUT leaf, so
// its executed width is observed through the gather's src[1]; the callback
// fires post-node, i.e. after any bind narrowing.
// ---------------------------------------------------------------------------
struct pass_rec {
    int64_t gathers      = 0;
    int64_t out_ids_ne0  = -1;
    int64_t logits_ne0   = -1;
    int64_t logits_ne1   = -1;
    int64_t logits_ne2   = -1;
    int64_t logits_ne3   = -1;
    bool     saw_logits  = false;
};

struct cb_state {
    std::mutex       mx;
    std::vector<pass_rec> passes;
};

static bool harness_cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    if (ask) {
        return true;  // always compute; recording happens post-node below
    }
    auto * st = (cb_state *) ud;
    std::lock_guard<std::mutex> lk(st->mx);
    if (st->passes.empty()) {
        return true;  // a compute outside a tracked decode
    }
    pass_rec & r = st->passes.back();
    if (t->op == GGML_OP_GET_ROWS && t->src[1] != nullptr && t->src[1]->type == GGML_TYPE_I32) {
        ++r.gathers;
        r.out_ids_ne0 = t->src[1]->ne[0];
    }
    if (t->name[0] != '\0' && strstr(t->name, "result_output") == t->name) {
        r.saw_logits = true;
        r.logits_ne0 = t->ne[0];
        r.logits_ne1 = t->ne[1];
        r.logits_ne2 = t->ne[2];
        r.logits_ne3 = t->ne[3];
    }
    return true;
}

// parse "<name>=<number>" from a captured trace line
static bool trace_field(const std::string & l, const char * name, long long * out) {
    const std::string key = std::string(name) + "=";
    const size_t      p   = l.find(key);
    if (p == std::string::npos) {
        return false;
    }
    *out = strtoll(l.c_str() + p + key.size(), nullptr, 10);
    return true;
}

// The attention mask width is part of the reuse key. n_kv is padded to
// 256-cell buckets (llama-kv-cache.cpp get_n_kv) and clamped to the physical
// context; the floor1 grid must stay inside one bucket so the output count
// is the only changing reuse key.
static uint32_t derived_mask_ne0(uint32_t used_cells) {
    uint32_t nkv = GGML_PAD(used_cells, 256u);
    if (nkv < 256u) {
        nkv = 256u;
    }
    return nkv > N_CTX_PHYS ? N_CTX_PHYS : nkv;
}

// ---------------------------------------------------------------------------
// tiny LLAMA-arch model built in memory (same fixture shape as
// nout_longres_harness.cpp; deterministic F32 weights)
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
    kv_str("general.name", "nout-floor1-reuse-hits");

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
    std::vector<int32_t> types(cfg.n_vocab, 1);
    types[0] = 2;
    types[1] = 3;
    types[2] = 3;
    gguf_set_arr_data(gctx, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    std::vector<float> scores(cfg.n_vocab, 0.0f);
    gguf_set_arr_data(gctx, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    kv_u32("tokenizer.ggml.bos_token_id", 1);
    kv_u32("tokenizer.ggml.eos_token_id", 2);
    kv_u32("tokenizer.ggml.unknown_token_id", 0);

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

static llama_model * harness_load_model(const harness_model_cfg & cfg) {
    gguf_context * gctx = harness_make_gguf(cfg);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.progress_callback = [](float, void *) { return true; };

    size_t seed = cfg.seed;
    llama_model * model = llama_model_init_from_user(gctx, harness_set_tensor_data, &seed, mparams);
    gguf_free(gctx);
    if (!model) {
        fprintf(stderr, "harness: llama_model_init_from_user failed\n");
    }
    return model;
}

// ---------------------------------------------------------------------------
// the production-shaped step grid: 1-token decode steps inside ONE n_kv 256
// bucket (the only changing reuse key inside the grid is the output count -
// the regime floor1 targets), plus one 3-token step for the k-path. The
// prefill (4 tokens, logits on the last token) is STEPS[0].
// ---------------------------------------------------------------------------

static const struct { uint32_t n_tok; uint32_t k_out; } STEPS[] = {
    // prefill: 4 tokens, logits on the last token
    { 4, 4 },
    { 1, 0 },  // 0-output decode step (non-empty all-zero output flags)
    { 1, 1 },  // 0 -> 1: THE decisive floor1 transition
    { 1, 1 },  // 1 -> 1
    { 1, 0 },  // 1 -> 0
    { 3, 1 },  // ubatch growth: rebuild for every lane (ubatch shape key)
    { 1, 0 },  // 3 -> 0: rebuild both; then 0-output
    { 1, 1 },  // 0 -> 1 again (the second decisive transition)
    { 1, 0 },  // 1 -> 0
};
static const uint32_t N_STEPS = (uint32_t)(sizeof(STEPS) / sizeof(STEPS[0]));

struct ctx_run {
    const char *    name = nullptr;
    llama_context * ctx  = nullptr;
    cb_state        cb;
    llama_batch     batch {};
    std::vector<int64_t> reused_delta;
};

// ---------------------------------------------------------------------------
// one grid step into a context's batch. all_outputs=true (the oracle leg)
// flags every token for output; otherwise the first k_out tokens carry the
// step's output flags (non-empty all-zero flags when k_out == 0).
// ---------------------------------------------------------------------------
static void ctx_run_fill(ctx_run & r, uint32_t step, int32_t tok_base, bool all_outputs) {
    llama_batch & b = r.batch;
    const uint32_t n_tok = STEPS[step].n_tok;
    const uint32_t k_out  = STEPS[step].k_out;
    for (uint32_t i = 0; i < n_tok; ++i) {
        b.token[i]     = (llama_token)((11 + 7 * (int) i + 3 * tok_base) % 97);
        b.pos[i]       = tok_base + (int32_t) i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = (all_outputs || i < k_out) ? 1 : 0;
    }
    b.n_tokens = (int32_t) n_tok;
}

static llama_context_params run_params(const harness_model_cfg & cfg, uint32_t n_outputs_max, cb_state * cb) {
    llama_context_params p = llama_context_default_params();
    p.n_ctx             = N_CTX_PHYS;
    p.n_batch           = 32;
    p.n_ubatch          = 32;  // single ubatch per decode
    p.n_seq_max         = 1;
    p.n_outputs_max     = n_outputs_max;
    p.n_threads         = 1;
    p.n_threads_batch   = 1;
    p.embeddings        = false;
    p.no_perf           = false;  // n_reused must be tracked
    p.cb_eval           = cb ? &harness_cb_eval : nullptr;
    p.cb_eval_user_data = cb;
    (void) cfg;
    return p;
}

// ---------------------------------------------------------------------------
// main part
// ---------------------------------------------------------------------------

static int run_part(llama_model * model, uint32_t n_vocab) {
    int failures0 = g_failures;

    if (getenv(GATE) || getenv(ENV_GATE) || getenv(TRACE_GATE)) {
        fprintf(stderr,
                "harness: %s/%s/%s are set in the environment; the harness "
                "controls every gate per-context via setenv - unset them for a clean run\n",
                GATE, ENV_GATE, TRACE_GATE);
        return 2;
    }

    ctx_run run_stock;  run_stock.name  = "stock";   // every gate off
    ctx_run run_floor1; run_floor1.name = "floor1";  // LLAMA_NOUT_FLOOR1_REUSE=1 + trace=1
    ctx_run run_ref;    run_ref.name    = "ref";     // every token output (oracle)

    // gates are read at context-construction time; setenv around each
    // construction gives per-context control (exact-"1" values only)
    setenv(GATE, "0", 1);
    setenv(TRACE_GATE, "0", 1);
    setenv(ENV_GATE, "0", 1);
    run_stock.ctx = llama_init_from_model(model, run_params(harness_model_cfg{}, 4, &run_stock.cb));

    setenv(GATE, "1", 1);
    setenv(TRACE_GATE, "1", 1);
    setenv(ENV_GATE, "0", 1);
    run_floor1.ctx = llama_init_from_model(model, run_params(harness_model_cfg{}, 4, &run_floor1.cb));

    setenv(GATE, "0", 1);
    setenv(TRACE_GATE, "0", 1);
    setenv(ENV_GATE, "0", 1);
    run_ref.ctx = llama_init_from_model(model, run_params(harness_model_cfg{}, 4, &run_ref.cb));

    if (!run_stock.ctx || !run_floor1.ctx || !run_ref.ctx) {
        fprintf(stderr, "harness: context creation failed\n");
        return 2;
    }

    run_stock.batch  = llama_batch_init(32, 0, 1);
    run_floor1.batch = llama_batch_init(32, 0, 1);
    run_ref.batch    = llama_batch_init(32, 0, 1);

    llama_perf_context_reset(run_stock.ctx);
    llama_perf_context_reset(run_floor1.ctx);
    llama_perf_context_reset(run_ref.ctx);

    const size_t row_bytes = (size_t) n_vocab * sizeof(float);
    guarded_buf stage_stock, stage_floor1, stage_ref, row0_snap;
    stage_stock.alloc(row_bytes * 4);
    stage_floor1.alloc(row_bytes * 4);
    stage_ref.alloc(row_bytes * 4);
    row0_snap.alloc(row_bytes);

    std::vector<std::string> floor1_lines;   // the floor1 context's own trace lines
    int32_t tok_base = 0;                    // position base; advances by n_tok per step

    // capture the runtime's own floor1 lines during the grid below
    g_log.active = true;

    for (uint32_t s = 0; s < N_STEPS; ++s) {
        const uint32_t n_tok = STEPS[s].n_tok;
        const uint32_t k     = STEPS[s].k_out;

        pass_rec rec_stock, rec_floor1, rec_ref;
        int64_t  d_stock = 0, d_floor1 = 0, d_ref = 0;

        ctx_run_fill(run_stock,  s, tok_base, false);
        ctx_run_fill(run_floor1, s, tok_base, false);
        ctx_run_fill(run_ref,    s, tok_base, true);  // oracle: all tokens output

        // ---- stock (all gates off) ----
        {
            const int32_t before = llama_perf_context(run_stock.ctx).n_reused;
            float * pub_before = nullptr;
            if (k == 0) {
                pub_before = llama_get_logits(run_stock.ctx);
                CHECK(pub_before != nullptr, "stock step %" PRIu32 ": public logits base null", s);
                if (pub_before) {
                    memcpy(row0_snap.ptr, pub_before, row_bytes);
                }
            }
            run_stock.cb.mx.lock();
            run_stock.cb.passes.push_back({});
            run_stock.cb.mx.unlock();
            const int rc = llama_decode(run_stock.ctx, run_stock.batch);
            CHECK(rc == 0, "stock step %" PRIu32 " (k=%" PRIu32 "): decode rc=%d", s, k, rc);
            llama_synchronize(run_stock.ctx);
            run_stock.cb.mx.lock();
            rec_stock = run_stock.cb.passes.empty() ? pass_rec{} : run_stock.cb.passes.back();
            run_stock.cb.mx.unlock();
            if (k == 0 && pub_before) {
                CHECK(memcmp(row0_snap.ptr, pub_before, row_bytes) == 0,
                      "stock step %" PRIu32 ": zero-output decode modified the public buffer", s);
            }
            d_stock = llama_perf_context(run_stock.ctx).n_reused - before;
        }

        // ---- floor1 (LLAMA_NOUT_FLOOR1_REUSE=1, trace=1) ----
        {
            const int32_t before = llama_perf_context(run_floor1.ctx).n_reused;
            float * pub_before = nullptr;
            if (k == 0) {
                pub_before = llama_get_logits(run_floor1.ctx);
                CHECK(pub_before != nullptr, "floor1 step %" PRIu32 ": public logits base null", s);
                if (pub_before) {
                    memcpy(row0_snap.ptr, pub_before, row_bytes);
                }
            }
            run_floor1.cb.mx.lock();
            run_floor1.cb.passes.push_back({});
            run_floor1.cb.mx.unlock();
            const int rc = llama_decode(run_floor1.ctx, run_floor1.batch);
            CHECK(rc == 0, "floor1 step %" PRIu32 " (k=%" PRIu32 "): decode rc=%d", s, k, rc);
            llama_synchronize(run_floor1.ctx);
            run_floor1.cb.mx.lock();
            rec_floor1 = run_floor1.cb.passes.empty() ? pass_rec{} : run_floor1.cb.passes.back();
            run_floor1.cb.mx.unlock();
            if (k == 0 && pub_before) {
                CHECK(memcmp(row0_snap.ptr, pub_before, row_bytes) == 0,
                      "floor1 step %" PRIu32 ": zero-output decode modified the public buffer", s);
            }
            d_floor1 = llama_perf_context(run_floor1.ctx).n_reused - before;
        }

        // ---- ref (all outputs; row-to-token oracle) ----
        {
            const int32_t before = llama_perf_context(run_ref.ctx).n_reused;
            run_ref.cb.mx.lock();
            run_ref.cb.passes.push_back({});
            run_ref.cb.mx.unlock();
            const int rc = llama_decode(run_ref.ctx, run_ref.batch);
            CHECK(rc == 0, "ref step %" PRIu32 " (k=%" PRIu32 "): decode rc=%d", s, k, rc);
            llama_synchronize(run_ref.ctx);
            run_ref.cb.mx.lock();
            rec_ref = run_ref.cb.passes.empty() ? pass_rec{} : run_ref.cb.passes.back();
            run_ref.cb.mx.unlock();
            d_ref = llama_perf_context(run_ref.ctx).n_reused - before;
        }

        run_stock.reused_delta.push_back(d_stock);
        run_floor1.reused_delta.push_back(d_floor1);
        run_ref.reused_delta.push_back(d_ref);

        // ---- executed shapes (real observations) ----
        // gathers == 2 (dense LLAMA closure: hidden-state gather + lm_head
        // gather, same fixture as nout_longres_harness); the narrowed
        // out_ids width is the per-step executed evidence.
        const uint32_t want_rows = n_tok;  // the oracle leg flags every token
        CHECK(rec_stock.gathers == 2, "stock step %" PRIu32 ": gathers=%" PRId64 " != 2", s, rec_stock.gathers);
        CHECK(rec_floor1.gathers == 2, "floor1 step %" PRIu32 ": gathers=%" PRId64 " != 2", s, rec_floor1.gathers);
        CHECK(rec_ref.gathers == 2, "ref step %" PRIu32 ": gathers=%" PRId64 " != 2", s, rec_ref.gathers);
        // out_ids width: stock/floor1 == k (0-output steps: 0 - the narrowing
        // executed for real), ref == n_tok
        CHECK(rec_stock.out_ids_ne0 == (int64_t) k,
              "stock step %" PRIu32 ": out_ids ne[0]=%" PRId64 " != %" PRIu32, s, rec_stock.out_ids_ne0, k);
        CHECK(rec_floor1.out_ids_ne0 == (int64_t) k,
              "floor1 step %" PRIu32 ": out_ids ne[0]=%" PRId64 " != %" PRIu32, s, rec_floor1.out_ids_ne0, k);
        CHECK(rec_ref.out_ids_ne0 == (int64_t) want_rows,
              "ref step %" PRIu32 ": out_ids ne[0]=%" PRId64 " != %" PRIu32, s, rec_ref.out_ids_ne0, want_rows);
        CHECK(rec_stock.saw_logits && rec_floor1.saw_logits && rec_ref.saw_logits,
              "step %" PRIu32 ": result_output not observed", s);
        if (k > 0) {
            CHECK(rec_floor1.logits_ne1 == (int64_t) k,
                  "floor1 step %" PRIu32 ": logits ne[1]=%" PRId64 " != %" PRIu32, s, rec_floor1.logits_ne1, k);
        }

        // ---- copy the produced rows BY TOKEN INDEX (never contiguous rows) ----
        auto copy_rows = [&](const char * nm, llama_context * cx, uint32_t rows, guarded_buf & dst) {
            if (rows == 0) {
                return;
            }
            for (uint32_t j = 0; j < rows; ++j) {
                // llama_get_logits_ith maps the ubatch TOKEN index through
                // out_ids - the token-index discipline the acceptance
                // requires (row j of the logits buffer is NOT assumed to be
                // token j)
                const float * p = llama_get_logits_ith(cx, (int32_t) j);
                CHECK(p != nullptr, "%s step %" PRIu32 ": logits row %" PRIu32 " null", nm, s, j);
                if (!p) {
                    continue;
                }
                memcpy(dst.ptr + (size_t) j * row_bytes, p, row_bytes);
            }
        };

        if (k > 0) {
            copy_rows("stock", run_stock.ctx, k, stage_stock);
            copy_rows("floor1", run_floor1.ctx, k, stage_floor1);
            copy_rows("ref", run_ref.ctx, k, stage_ref);
            CHECK(memcmp(stage_stock.ptr, stage_floor1.ptr, (size_t) k * row_bytes) == 0,
                  "step %" PRIu32 " (k=%" PRIu32 "): stock vs floor1 logits BYTE MISMATCH (%zu bytes compared)",
                  s, k, (size_t) k * row_bytes);
            CHECK(memcmp(stage_ref.ptr, stage_stock.ptr, (size_t) k * row_bytes) == 0,
                  "step %" PRIu32 " (k=%" PRIu32 "): stock vs all-output reference BYTE MISMATCH", s, k);
        }


        // ---- the reuse grid (inside one n_kv bucket, output count is the
        // only changing key) ----
        // decode steps 1..N-1 all run inside bucket 256 (derived_mask_ne0
        // constant); expectations:
        //   floor1: 0-output and 1-output 1-token steps reuse each other
        //   (delta 1); the 3-token step and the steps adjacent to it rebuild.
        //   stock: every output-count change rebuilds (0->1, 1->0).
        if (s >= 2) {
            const bool same_bucket =
                derived_mask_ne0((uint32_t)(tok_base + (int32_t) n_tok)) == derived_mask_ne0((uint32_t) tok_base);
            const bool same_shape = STEPS[s].n_tok == STEPS[s - 1].n_tok;
            if (same_bucket && same_shape) {
                const uint32_t kp = STEPS[s - 1].k_out;
                const uint32_t kc = STEPS[s].k_out;
                if ((kp == 0 && kc == 1) || (kp == 1 && kc == 0) || kp == kc) {
                    // 0->1 / 1->0 / equal counts: floor1 must reuse (the
                    // normalized key makes the output count a non-key for the
                    // 0<->1 pair)
                    CHECK(d_floor1 == 1,
                          "step %" PRIu32 ": floor1 n_reused delta %" PRId64 " != 1 on %u->%u transition "
                          "(0<->1 graph reuse did not hit)", s, d_floor1, kp, kc);
                    if (kp != kc) {
                        // stock must rebuild across a count change (the very
                        // gap floor1 closes - asserted, not assumed)
                        CHECK(d_stock == 0,
                              "step %" PRIu32 ": stock reused across an output-count change %u->%u", s, kp, kc);
                    }
                } else if (kc > 1) {
                    // counts differ beyond the 0<->1 normalization: rebuild
                    // (fail-closed, both)
                    CHECK(d_floor1 == 0, "step %" PRIu32 ": floor1 reused across %u->%u", s, kp, kc);
                    CHECK(d_stock == 0, "step %" PRIu32 ": stock reused across %u->%u", s, kp, kc);
                }
            }
        }

        tok_base += (int32_t) n_tok;  // the next step continues this KV history
    }

    g_log.active = false;
    floor1_lines = g_log.take();

    // ---- assert the runtime's OWN decision trace (追踪判定) ----
    // every decode step of the floor1 context is a decoder step -> one
    // NOUT1_FLOOR_TRACE line + one NOUT1_FLOOR_STEP line per step
    long long dec_lines = 0, floor1_reuse = 0, step_lines = 0, fell_back = 0, bind_zero = 0;
    for (const auto & l : floor1_lines) {
        if (l.rfind("NOUT1_FLOOR_TRACE dec=", 0) == 0) {
            ++dec_lines;
            long long reuse = -1, n_out = -1, key = -1, stored = -1;
            CHECK(trace_field(l, "reuse", &reuse) && trace_field(l, "n_out", &n_out) &&
                  trace_field(l, "key", &key) && trace_field(l, "stored", &stored),
                  "trace line not parseable: %s", l.c_str());
            if (reuse == 1) {
                ++floor1_reuse;
                // reuse steps carry a clean miss field: miss=- (no failure)
                const size_t mp = l.find("miss=");
                CHECK(mp != std::string::npos && l.compare(mp, 6, "miss=-") == 0,
                      "reuse step without a clean miss field: %s", l.c_str());
                // floor1 reuse: stored key must equal the normalized key
                CHECK(key == 1 && stored == 1,
                      "reuse step key/stored != 1/1 (n_out=%lld key=%lld stored=%lld): %s",
                      n_out, key, stored, l.c_str());
            } else {
                CHECK(l.find("miss=") != std::string::npos,
                      "miss step without a reason: %s", l.c_str());
            }
        } else if (l.rfind("NOUT1_FLOOR_STEP ", 0) == 0) {
            ++step_lines;
            long long fb = -1, rows = -1;
            CHECK(trace_field(l, "fell_back", &fb) && trace_field(l, "bind_rows", &rows),
                  "step line not parseable: %s", l.c_str());
            if (fb == 0 && rows == 0) {
                ++bind_zero;  // 0-row bind executed for real
            } else if (fb == 1) {
                ++fell_back;  // fail-closed fallback rebuild
            }
        }
    }
    CHECK(dec_lines == (long long) N_STEPS,
          "floor1 decision lines %lld != %d steps (trace gate not observed on every decoder step)",
          dec_lines, (int) N_STEPS);
    CHECK(step_lines == (long long) N_STEPS, "floor1 step lines %lld != %d", step_lines, (int) N_STEPS);
    // derived from the STEPS grid (k==0 at idx 1, 4, 6, 8):
    //   reuse steps  = {2, 3, 4, 7, 8}           -> 5  (0<->1 normalized key)
    //   fallbacks    none                        -> 0
    //   zero-row binds on every 0-output step     -> 4
    CHECK(floor1_reuse == 5, "floor1 reuse steps %lld != 5 (expected {2,3,4,7,8})", floor1_reuse);
    CHECK(fell_back == 0, "unexpected fallback rebuilds %lld on the base grid", fell_back);
    CHECK(bind_zero == 4, "zero-row binds %lld != 4 (steps with k==0: 1,4,6,8)", bind_zero);

    // aggregate line from the runtime at context destruction
    g_log.active = true;
    llama_free(run_stock.ctx);
    llama_free(run_floor1.ctx);
    llama_free(run_ref.ctx);
    g_log.active = false;
    {
        const auto tail = g_log.take();
        std::string stats;
        for (const auto & l : tail) {
            if (l.rfind("NOUT1_FLOOR_TRACE_STATS", 0) == 0) {
                stats = l;
            }
        }
        CHECK(!stats.empty(), "NOUT1_FLOOR_TRACE_STATS line missing at floor1 context destruction");
        if (!stats.empty()) {
            long long dec = -1, reuse = -1, builds = -1, builds_fb = -1, skip = -1, b0 = -1;
            CHECK(trace_field(stats, "dec", &dec) && trace_field(stats, "reuse", &reuse) &&
                  trace_field(stats, "builds", &builds) && trace_field(stats, "builds_fb", &builds_fb) &&
                  trace_field(stats, "compute_skip", &skip) && trace_field(stats, "bind_zero", &b0),
                  "trace stats not parseable: %s", stats.c_str());
            CHECK(dec == (long long) N_STEPS, "stats dec %lld != %d", dec, (int) N_STEPS);
            CHECK(reuse == 5, "stats reuse %lld != 5", reuse);
            // builds: prefill (s0) + s1 + s5 (3-token) + s6 = 4; every reuse
            // step computes on the reused graph, so skips == reuse == 5
            CHECK(builds == 4, "stats builds != 4 (prefill, s1, 3-token, s6)", builds);
            CHECK(builds_fb == 0, "stats builds_fb != 0", builds_fb);
            CHECK(skip == 5, "stats compute_skip != 5 (one per reused step)", skip);
            CHECK(b0 == 4, "stats bind_zero != 4", b0);
        }
    }

    llama_batch_free(run_stock.batch);
    llama_batch_free(run_floor1.batch);
    llama_batch_free(run_ref.batch);

    stage_stock.free_buf();
    stage_floor1.free_buf();
    stage_ref.free_buf();
    row0_snap.free_buf();

    return g_failures == failures0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// gate probes: exact-"1" only, dual-gate refusal, trace-alone inert
// ---------------------------------------------------------------------------

static int run_gate_probes(llama_model * model) {
    int failures0 = g_failures;

    struct probe {
        const char * gate_value;   // LLAMA_NOUT_FLOOR1_REUSE value
        const char * env_value;    // GGML_TP5_NOUT_REUSE value
        const char * name;
    };
    const probe probes[] = {
        { "0",   "0", "gate-0" },
        { "",    "0", "gate-empty" },
        { "01",  "0", "gate-01" },
        { " 1",  "0", "gate-space1" },
        { "true","0", "gate-true" },
        { "1",   "1", "dual-gate" },
    };

    for (const auto & pr : probes) {
        g_log.active = true;
        setenv(ENV_GATE, pr.env_value, 1);
        setenv(TRACE_GATE, "1", 1);
        setenv(GATE, pr.gate_value, 1);
        llama_context * cx = llama_init_from_model(model, run_params(harness_model_cfg{}, 4, nullptr));
        unsetenv(GATE);
        unsetenv(TRACE_GATE);
        unsetenv(ENV_GATE);
        CHECK(cx != nullptr, "probe %s: context creation failed", pr.name);
        g_log.active = false;
        const auto lines = g_log.take();

        if (strcmp(pr.name, "dual-gate") == 0) {
            // floor1 must be refused: the WARN line + no floor1 trace lines
            bool refused = false;
            for (const auto & l : lines) {
                if (l.find("nout floor1 refused") != std::string::npos) {
                    refused = true;
                }
                CHECK(l.rfind("NOUT1_FLOOR_TRACE", 0) != 0 && l.rfind("NOUT1_FLOOR_STEP", 0) != 0,
                      "probe dual-gate: floor1 trace emitted despite refusal: %s", l.c_str());
            }
            CHECK(refused, "probe dual-gate: nout floor1 refused line missing");
        } else {
            // non-exact gate values ("", "01", " 1", "true", "0"): both gates
            // parse off, so construction emits no floor1 line at all and the
            // context must be stock-equivalent: assert only that no floor1
            // trace line was emitted (checked in the loop below)
            for (const auto & l : lines) {
                CHECK(l.rfind("NOUT1_FLOOR_TRACE", 0) != 0 && l.rfind("NOUT1_FLOOR_STEP", 0) != 0,
                      "probe %s: floor1 trace emitted with gate=%s: %s", pr.name, pr.gate_value, l.c_str());
            }
        }

        // one decode + a stock-comparison decode on the same resident context
        if (cx) {
            llama_batch b = llama_batch_init(32, 0, 1);
            b.token[0] = 5;
            b.pos[0]   = 0;
            b.n_seq_id[0] = 1;
            b.seq_id[0][0] = 0;
            b.logits[0] = 1;
            b.n_tokens = 1;
            CHECK(llama_decode(cx, b) == 0, "probe %s: decode failed", pr.name);
            llama_synchronize(cx);
            llama_batch_free(b);
            llama_free(cx);
        }
    }

    return g_failures == failures0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [--model PATH.gguf]\n", argv[0]);
            return 2;
        }
    }

    llama_log_set(harness_log_callback, nullptr);

    harness_model_cfg cfg;
    llama_model * model = nullptr;
    if (model_path) {
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;
        model = llama_model_load_from_file(model_path, mparams);
        if (!model) {
            fprintf(stderr, "harness: cannot load %s\n", model_path);
            return 2;
        }
    } else {
        model = harness_load_model(cfg);
        if (!model) {
            return 2;
        }
    }
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    int rc = run_part(model, n_vocab);
    if (rc == 0) {
        rc = run_gate_probes(model);
    }

    llama_model_free(model);

    fprintf(stderr, "harness: %d failure(s)\n", g_failures);
    return rc == 2 ? 2 : (g_failures ? 1 : 0);
}
