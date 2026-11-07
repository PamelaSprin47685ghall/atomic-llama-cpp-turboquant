// graph_reuse_probe — engine-level measurement of the llama graph-reuse guard.
//
// Ported from the previous lane's devops/guard_accept_probe.c (read-only
// reference) with two harness fixes:
//   1. the original hard-coded n_seq_max = 1, but the `seqs` trace feeds
//      seq_id 1 on its odd steps. llama_batch_allocr::init() rejected that
//      batch ("invalid seq_id ... >= 1") and llama_decode() returned -1
//      before any graph decision, so the trace never measured graph reuse.
//      This copy sizes the context for the trace (seqs needs two sequence
//      slots; every other trace keeps 1 so its KV/mask widths are unchanged)
//      and exposes PROBE_N_SEQ_MAX to reproduce both sides with one binary.
//   2. the original `seqs` trace used one global n_past for every sequence,
//      so seq 0 skipped a position whenever an odd step handed a token to
//      seq 1. Batch validation then failed with "sequence positions must
//      remain consecutive" at the next even step. This copy tracks one next
//      position per sequence.
//
//   PROBE_N_SEQ_MAX=1 ./graph_reuse_probe tiny.gguf seqs 2 8   # rc=-1
//   PROBE_N_SEQ_MAX=2 ./graph_reuse_probe tiny.gguf seqs 16 8  # rc=0
//
// Original probe documentation (unchanged):
//
// Public llama API only. CPU backend, no GPU/HIP. It runs a real model and
// prints, for every llama_decode() call, whether the previous compute graph was
// reused (llama_perf_context().n_reused delta). That delta is exactly the
// decision of llm_graph_result::can_reuse() -> llm_graph_input_*::can_reuse()
// -> can_reuse_kq_mask().
//
// Traces:
//   grow    : prefill P tokens, then N single-token decodes at growing pos.
//             n_kv grows by one every step, so the KQ mask width changes.
//             stock guard: reject every step (accept 0).
//   repeat  : prefill P, then N times { decode 1 token; seq_rm that token }.
//             Every decode has the same shape as the previous graph.
//             Positive control: stock guard must accept here.
//   mix13   : prefill P, then alternate 1-token and 3-token decodes.
//             n_tokens changes -> must reject before and after any widening.
//   outids  : prefill P, then alternate 2-token decodes with 1 vs 2 outputs.
//             n_outputs changes -> must reject.
//   seqs    : prefill P, then alternate 2-token decodes with 1 vs 2 sequences.
//             n_seqs changes -> must reject.
//
// Output: one line per decode:
//   STEP <trace> <i> n_tokens=<t> n_kv=<kv> reused=<0|1> n_reused_total=<r> rc=<rc>
// and a final:
//   SUMMARY <trace> decodes=<d> reused=<r> accept_rate=<r/d> n_reused_total=<n>
// Exit code: 0 only when every decode returned rc=0; a decode failure sets
// the exit code to 1 even though the per-step line and SUMMARY are printed.
// The A/B wrapper must record the exit code, not just the SUMMARY.
//
// Build (stock narrow prefix, no GPU):
//   build-slot ccache cc -O2 -std=c11 -I<narrow>/include graph_reuse_probe.c
//     -L<narrow>/lib -lllama -lggml -lggml-base -lggml-cpu -lpthread -ldl -lm
//     -Wl,-rpath,<narrow>/lib -o graph_reuse_probe

#include "llama.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SEQ 4

static const char * g_trace = "grow";
static int          g_steps = 32;
static int          g_prefill = 8;

static int32_t reused_now(struct llama_context * ctx) {
    return llama_perf_context(ctx).n_reused;
}

// Build one batch. tokens/pos/seq per token are given by the caller.
// logits[i] = 1 means "produce output for token i".
static int decode_batch(
        struct llama_context * ctx,
        struct llama_batch   * batch,
        const llama_token    * toks,
        const llama_pos      * poss,
        const llama_seq_id   * seqs,
        const int8_t         * logits,
        int32_t                n) {
    batch->n_tokens = n;
    for (int32_t i = 0; i < n; ++i) {
        batch->token[i]     = toks[i];
        batch->pos[i]       = poss[i];
        batch->n_seq_id[i]  = 1;
        batch->seq_id[i][0] = seqs[i];
        batch->logits[i]    = logits[i];
    }
    return llama_decode(ctx, *batch);
}

static void report(
        int step, int32_t n_tokens, llama_pos n_kv, int reused_delta,
        int32_t total, int rc) {
    printf("STEP %s %d n_tokens=%d n_kv=%lld reused=%d n_reused_total=%d rc=%d\n",
           g_trace, step, (int) n_tokens, (long long) n_kv,
           reused_delta ? 1 : 0, (int) total, rc);
    fflush(stdout);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <model.gguf> [grow|repeat|mix13|outids|seqs] [steps] [prefill]\n",
                argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    if (argc > 2) g_trace   = argv[2];
    if (argc > 3) g_steps   = atoi(argv[3]);
    if (argc > 4) g_prefill = atoi(argv[4]);

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "FAIL model load: %s\n", model_path);
        return 1;
    }
    fprintf(stderr, "INFO model loaded: %s\n", model_path);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 2048;
    cparams.n_batch         = 64;
    cparams.n_ubatch        = 64;
    // The previous lane's probe hard-coded n_seq_max = 1; its `seqs` trace
    // then failed in batch validation (seq_id 1 >= 1) and never reached the
    // graph guard. Size the context for the trace and keep PROBE_N_SEQ_MAX as
    // the A/B override (1 reproduces the old rc=-1, 2 measures the guard).
    uint32_t probe_n_seq_max = strcmp(g_trace, "seqs") == 0 ? 2 : 1;
    if (const char * env = getenv("PROBE_N_SEQ_MAX")) {
        probe_n_seq_max = (uint32_t) atoi(env);
    }
    cparams.n_seq_max       = probe_n_seq_max;
    cparams.n_threads       = 2;
    cparams.n_threads_batch = 2;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.no_perf         = false;
    cparams.embeddings      = false;

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "FAIL context init\n");
        return 1;
    }

    struct llama_batch batch = llama_batch_init(16, 0, 1);

    // fixed token ids, well inside the vocabulary of any test model
    llama_token ptok[64];
    llama_pos   ppos[64];
    llama_seq_id pseq[64];
    int8_t       plog[64];
    for (int i = 0; i < g_prefill; ++i) {
        ptok[i] = (llama_token) (10 + i);
        ppos[i] = i;
        pseq[i] = 0;
        plog[i] = (i == g_prefill - 1) ? 1 : 0;
    }

    int rc = decode_batch(ctx, &batch, ptok, ppos, pseq, plog, g_prefill);
    if (rc != 0) {
        fprintf(stderr, "FAIL prefill rc=%d\n", rc);
        return 1;
    }
    llama_pos n_past = g_prefill;
    int32_t reused_total = reused_now(ctx);
    printf("PREFILL %s n_tokens=%d n_kv=%lld n_reused_total=%d rc=%d\n",
           g_trace, g_prefill, (long long) n_past, (int) reused_total, rc);
    fflush(stdout);

    int decodes = 0;
    int reused  = 0;
    int failed  = 0;

    if (strcmp(g_trace, "grow") == 0) {
        for (int i = 0; i < g_steps; ++i) {
            llama_token t = (llama_token) (20 + (i % 50));
            llama_pos   p = n_past;
            llama_seq_id s = 0;
            int8_t l = 1;
            int32_t before = reused_now(ctx);
            int r = decode_batch(ctx, &batch, &t, &p, &s, &l, 1);
            int32_t after = reused_now(ctx);
            n_past += 1;
            report(i, 1, n_past, after - before, after, r);
            if (r != 0) { failed = 1; break; }
            decodes++;
            reused += (after > before);
        }
    } else if (strcmp(g_trace, "repeat") == 0) {
        for (int i = 0; i < g_steps; ++i) {
            llama_token t = (llama_token) (20 + (i % 50));
            llama_pos   p = n_past;
            llama_seq_id s = 0;
            int8_t l = 1;
            int32_t before = reused_now(ctx);
            int r = decode_batch(ctx, &batch, &t, &p, &s, &l, 1);
            int32_t after = reused_now(ctx);
            report(i, 1, p + 1, after - before, after, r);
            if (r != 0) { failed = 1; break; }
            decodes++;
            reused += (after > before);
            // drop the token so the next decode has the same shape
            if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, p, -1)) {
                fprintf(stderr, "FAIL seq_rm pos=%d\n", (int) p);
                break;
            }
        }
    } else if (strcmp(g_trace, "mix13") == 0) {
        for (int i = 0; i < g_steps; ++i) {
            int32_t n = (i % 2 == 0) ? 1 : 3;
            llama_token tk[3];
            llama_pos   ps[3];
            llama_seq_id sq[3];
            int8_t       lg[3];
            for (int32_t j = 0; j < n; ++j) {
                tk[j] = (llama_token) (30 + (i * 3 + j) % 60);
                ps[j] = n_past + j;
                sq[j] = 0;
                lg[j] = (j == n - 1) ? 1 : 0;
            }
            int32_t before = reused_now(ctx);
            int r = decode_batch(ctx, &batch, tk, ps, sq, lg, n);
            int32_t after = reused_now(ctx);
            n_past += n;
            report(i, n, n_past, after - before, after, r);
            if (r != 0) { failed = 1; break; }
            decodes++;
            reused += (after > before);
        }
    } else if (strcmp(g_trace, "outids") == 0) {
        for (int i = 0; i < g_steps; ++i) {
            int32_t n = 2;
            llama_token tk[2];
            llama_pos   ps[2];
            llama_seq_id sq[2];
            int8_t       lg[2];
            for (int32_t j = 0; j < n; ++j) {
                tk[j] = (llama_token) (40 + (i * 2 + j) % 60);
                ps[j] = n_past + j;
                sq[j] = 0;
                // even step: 1 output; odd step: 2 outputs
                lg[j] = (i % 2 == 0) ? (j == n - 1) : 1;
            }
            int32_t before = reused_now(ctx);
            int r = decode_batch(ctx, &batch, tk, ps, sq, lg, n);
            int32_t after = reused_now(ctx);
            n_past += n;
            report(i, n, n_past, after - before, after, r);
            if (r != 0) { failed = 1; break; }
            decodes++;
            reused += (after > before);
        }
    } else if (strcmp(g_trace, "seqs") == 0) {
        // Per-sequence positions: llama_batch_allocr::init() requires the
        // positions of each sequence to stay consecutive. A shared n_past
        // skips a position when an odd step hands one token to seq 1 and the
        // next even step resumes seq 0 two positions later, which fails batch
        // validation before any graph decision.
        llama_pos next_pos[2] = { n_past, n_past };
        for (int i = 0; i < g_steps; ++i) {
            int32_t n = 2;
            llama_token tk[2];
            llama_pos   ps[2];
            llama_seq_id sq[2];
            int8_t       lg[2];
            int two = (i % 2 == 1);
            for (int32_t j = 0; j < n; ++j) {
                tk[j] = (llama_token) (50 + (i * 2 + j) % 60);
                sq[j] = two ? (llama_seq_id) j : 0;
                ps[j] = next_pos[sq[j]]++;
                lg[j] = (j == n - 1) ? 1 : 0;
            }
            int32_t before = reused_now(ctx);
            int r = decode_batch(ctx, &batch, tk, ps, sq, lg, n);
            int32_t after = reused_now(ctx);
            n_past += n; // n_kv estimate for the report only
            report(i, n, n_past, after - before, after, r);
            if (r != 0) { failed = 1; break; }
            decodes++;
            reused += (after > before);
        }
    } else {
        fprintf(stderr, "FAIL unknown trace: %s\n", g_trace);
        return 2;
    }

    printf("SUMMARY %s decodes=%d reused=%d accept_rate=%.4f n_reused_total=%d\n",
           g_trace, decodes, reused,
           decodes ? (double) reused / (double) decodes : 0.0,
           (int) reused_now(ctx));

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return failed ? 1 : 0;
}
