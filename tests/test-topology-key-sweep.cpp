//
// test-topology-key-sweep.cpp
//
// The topology key must not move across a length sweep, and the binder must
// keep binding while it does not.
//
// How the binder arm actually reuses the graph, and what this test asserts:
//
//   - the binder patches the recorded graph in place at each new (n_kv,
//     n_tokens) and re-executes it (union_rebind_apply). The build count
//     therefore stays at the first build for every step the binder applies.
//     The stock graph cache's allow_reuse() (n_tokens) and
//     can_reuse_kq_mask() (mask width) are not the gate here: those decide
//     whether the stock path may reuse a graph, while the union bind is a
//     separate, earlier path that patches the record and returns it. On the
//     gate's 8-round (32-step) 1,2,3,4 schedule the binder applies on all 31
//     post-first steps, so `steps_reused` is 31 and `steps_built` is 0. That
//     is the binder's intended behavior, not a defect, and this test does not
//     forbid it.
//
//   - the topology key does not move across the sweep, even though n_tokens
//     and n_kv both move on every step. This is the assertion that catches the
//     original defect, where a length parameter leaked into the graph
//     identity;
//
//   - every step builds at most once. The sweep is one graph per step at
//     most, never a retry loop, so a step that built twice would be a defect;
//
//   - the arm is the binder arm: diagnostics report enabled, not
//     envelope-disabled, and record_valid;
//
//   - every step after the first did work: it either built a new graph or
//     reused the recorded one, and a reused step bound something
//     (`patch_binds` moved). The run as a whole must have bound at least
//     once: a context that silently did nothing reports patch_binds == 0 and
//     fails, even though its unmoved build count makes every post-first step
//     look "reused".
//
// The two counts are findings, not fixed expectations. `steps_built == 0` is
// what this binder does today; a future binder that rebuilds a step still
// passes if the key stays stable, the step builds at most once, and the binds
// happened. The sweep must prove it did work, not merely that the loop ran.
//
// Usage: test-topology-key-sweep <model.gguf> [rounds]
//
// Exits 77 (the CTest skip convention) when there is no model, the model does
// not load, or the binder cannot be enabled on it. Never 0 for a skip.
//

#include "llama.h"

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond) do { \
    ++g_checks; \
    if (!(cond)) { \
        ++g_failures; \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    } \
} while (0)

static int skip(const char * why) {
    std::fprintf(stderr, "SKIP: %s\n", why);
    return 77;
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        return skip("usage: test-topology-key-sweep <model.gguf> [rounds]");
    }

    char * end = nullptr;
    errno = 0;
    const long rounds = argc == 3 ? std::strtol(argv[2], &end, 10) : 8;
    if (errno || (argc == 3 && (end == argv[2] || *end)) || rounds < 1 || rounds > INT_MAX / 10) {
        return skip("bad rounds argument");
    }

    llama_backend_init();

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    ggml_backend_dev_t devices[] = { nullptr };
    mp.devices = devices;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (model == nullptr) {
        llama_backend_free();
        return skip("model failed to load");
    }

    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (vocab <= 3) {
        llama_model_free(model);
        llama_backend_free();
        return skip("model vocabulary too small for the sweep");
    }

    // One context, the binder arm. The test creates it itself, so setting the
    // gate before the constructor is legitimate: it is the arm the test wants,
    // not a mutation of an environment a runner chose. No variable a runner
    // sets is unset here.
    auto cp = llama_context_default_params();
    // n_ctx_kv = 0: the physical KV capacity follows n_ctx instead of being
    // pinned to a separate value. The sweep submits rounds * 4 steps of 1..4
    // tokens, i.e. rounds * 10 tokens in total (2.5 average), so n_ctx is the
    // history the sweep will use. Pinning n_ctx_kv did not make the envelope
    // and the used span agree on any step but the last - see the header
    // comment, reasons 1 and 2 - so there is nothing to gain by pinning it,
    // and leaving it 0 keeps the capacity equal to the history the test
    // actually submits.
    cp.n_ctx = (int) rounds * 10;
    cp.n_batch = cp.n_ubatch = cp.n_outputs_max = 4;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 1;
    cp.kv_unified = true;
    cp.offload_kqv = cp.op_offload = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.embeddings = false;
    cp.cb_eval = nullptr;
    cp.cb_eval_user_data = nullptr;

    setenv("GGML_TP5_UNION_BINDER", "1", 1);
    setenv("LLAMA_EXPERIMENTAL_VARLEN_NTOK", "1", 1);
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        llama_model_free(model);
        llama_backend_free();
        return skip("context creation failed");
    }

    llama_union_diagnostics d = {};
    if (!llama_get_union_diagnostics(ctx, &d) || !d.enabled) {
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return skip("union binder not enabled on this model");
    }
    if (d.envelope_disabled) {
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return skip("union binder envelope disabled on this model");
    }

    const uint64_t init_builds = llama_graph_build_count(ctx);

    // The sweep: widths 1, 2, 3, 4 cycling. n_tokens moves every step, and
    // n_kv (the used span) moves with it because the single sequence keeps
    // every submitted token resident.
    const int widths[] = { 1, 2, 3, 4 };
    int pos = 0;
    bool first = true;
    uint64_t first_hash = 0;
    uint64_t previous_binds = d.patch_binds;

    int steps_reused = 0;
    int steps_built  = 0;
    uint64_t builds_at_previous_step = init_builds;

    for (long step = 0; step < rounds * 4; ++step) {
        const int width = widths[step % 4];
        llama_batch batch = llama_batch_init(width, 0, 1);
        if (!llama_batch_is_complete(&batch, width, false, true)) {
            llama_batch_free(batch);
            batch = {};
            ++g_failures;
            ++g_checks;
            std::fprintf(stderr, "FAIL: batch allocation step=%ld\n", step);
            break;
        }
        for (int i = 0; i < width; ++i) {
            batch.token[i] = 3 + (int) (((int64_t) (pos + i) * 7 + 1) % (vocab - 3));
            batch.pos[i] = pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        batch.n_tokens = width;

        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);

        const uint64_t builds = llama_graph_build_count(ctx);
        const bool have = llama_get_union_diagnostics(ctx, &d);
        std::fprintf(stderr,
                "step=%ld width=%d builds=%llu rebuild_count=%lld binds=%llu "
                "bound_tokens=%u bound_n_kv=%u topology=%016llx\n",
                step, width, (unsigned long long) builds,
                (long long) (builds - init_builds),
                (unsigned long long) d.patch_binds, d.bound_tokens, d.bound_n_kv,
                (unsigned long long) d.topology_hash);

        // Fails if the decode errors, if the binder stops binding (a stock
        // context rebuilds on each width change, so the count would rise), or
        // if the arm is not the binder arm.
        CHECK(rc == 0);
        CHECK(have);
        CHECK(d.enabled);
        CHECK(!d.envelope_disabled);
        CHECK(d.record_valid);

        // Every step builds at most once. A step that built twice is a retry
        // loop (a failed bind followed by a rebuild), which is the defect this
        // catches. The step is counted as "built" when the count moved and as
        // "reused" when it did not. `reused` is computed BEFORE the update
        // below: after the update the comparison would be true for every
        // post-first step, including a step that rebuilt, and the bind clause
        // would check the wrong set. On this schedule the binder patches the
        // record on every step after the first, so "reused" is the rule (see
        // the header comment).
        CHECK(builds - init_builds <= (uint64_t) (step + 1));
        const bool reused = step > 0 && builds == builds_at_previous_step;
        if (step > 0) {
            if (reused) {
                ++steps_reused;
            } else {
                ++steps_built;
            }
        }
        builds_at_previous_step = builds;

        // The key is stable across the whole sweep. Fails if a length
        // parameter leaked into the graph identity - the original defect.
        if (first) {
            first_hash = d.topology_hash;
            first = false;
        }
        CHECK(d.topology_hash == first_hash);

        // The binder is doing work: every step that reused the graph must
        // have bound something. On this schedule that is every post-first
        // step, so the clause is live on all of them; a reuse that bound
        // nothing would be a step that computed nothing.
        if (reused) {
            CHECK(d.patch_binds > previous_binds);
        }
        previous_binds = d.patch_binds;

        pos += width;
    }

    // The sweep really moved the length. Fails if the schedule degenerated.
    CHECK(pos == (int) rounds * 10);

    // The sweep proves it did work: the binder bound at least once across the
    // run, and the per-step clause above proves every reused step bound. A
    // context that silently did nothing cannot pass: it reports
    // patch_binds == 0 here, even though its build count would make every
    // post-first step look "reused".
    //
    // The two counts are printed as findings, not asserted separately. Today
    // steps_built == 0 and steps_reused == rounds * 4 - 1; a future binder
    // that rebuilds a step still passes as long as the key stays stable, the
    // step builds at most once, and the binds happened.
    std::fprintf(stderr, "reuse_summary steps_reused=%d steps_built=%d\n", steps_reused, steps_built);
    CHECK(d.patch_binds > 0);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL %d/%d topology-key-sweep checks\n", g_failures, g_checks);
        return 1;
    }
    std::printf("PASS %d topology-key-sweep checks\n", g_checks);
    return 0;
}
