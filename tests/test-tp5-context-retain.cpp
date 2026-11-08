// TP5 WS4 pending-context and model ownership test (CPU only).
//
// e3c9ba1e8 retained a pending llama_context by releasing individual members;
// memory/kv_stream_phase_arena/cvec/loras/balloc/sampling were still freed by
// the destructor and the model was caller-owned without a retained reference.
// The fix retains the whole context (llama_free returns without deleting it)
// and increments a model retained-context count that defers llama_model_free.
//
// This test observes:
//   * llama_model_free defers while a live or retained context reference
//     exists; the last clean llama_free deletes a model whose free was
//     already requested,
//   * a duplicate llama_free does not increment the retained count,
//   * retain_for_process clears the CPU backend's external threadpool pointer,
//   * a sampler borrow may die after retention without a dereference,
//   * a prefab envelope slot built by a real decode is part of the context's
//     unproven state when its scheduler is faulted,
//   * a clean model still frees normally,
//   * with LLAMA_TEST_MODEL (or argv[1]), a real CPU context marked unproven is
//     retained by llama_free and the model free is deferred; the context is
//     never destructed, so memory/arena/balloc/sampling cannot be freed behind
//     an in-flight copy,
//   * without a model the context part is skipped; the model part still runs.
//
// Build (ctest registers it; Release or ASAN):
//   cmake --build build-ws4-sched-ctest -j --target test-tp5-context-retain
//   LLAMA_TEST_MODEL=/path/to/model.gguf ctest -R test-tp5-context-retain

#include "llama.h"
#include "llama-ext.h"
#include "ggml-cpu.h"

#include "../src/llama-context.h"
#include "../src/llama-model.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            ++g_failures;                                                                \
        }                                                                                \
    } while (0)

llama_model * make_mock_model() {
    llama_quant_model_desc desc = {};
    desc.architecture  = "llama";
    desc.n_embd        = 64;
    desc.n_ff          = 128;
    desc.n_layer       = 2;
    desc.n_head        = 4;
    desc.n_head_kv     = 4;
    desc.n_expert      = 0;
    desc.n_embd_head_k = 16;
    desc.n_embd_head_v = 16;
    return llama_quant_model_from_metadata(&desc);
}

void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value != nullptr) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// A model retained by a pending context must not be deleted by
// llama_model_free. The retained model object stays readable.
void test_model_retention() {
    llama_model * retained = make_mock_model();
    CHECK(retained != nullptr);
    if (retained == nullptr) {
        return;
    }
    CHECK(retained->retained_contexts() == 0);
    retained->retain_context();
    CHECK(retained->retained_contexts() == 1);

    llama_model_free(retained);
    CHECK(retained->retained_contexts() == 1); // deferred, not deleted
    CHECK(!retained->name.empty());            // object still readable
    // The test owns the mock pointer after observing the deferred request;
    // dropping it directly avoids an intentional test-only leak.
    delete retained;

    // A clean model frees normally. Do not read it after the free.
    llama_model * clean = make_mock_model();
    CHECK(clean != nullptr);
    if (clean != nullptr) {
        CHECK(clean->retained_contexts() == 0);
        llama_model_free(clean);
    }
}

// Model lifetime is a reference count, not a check-then-delete:
// llama_model_free while a context is live defers, and the last clean context
// release owns the delete. The reference and the free request are serialized
// by the model lifetime mutex.
void test_model_lifetime_reference() {
    llama_model * model = make_mock_model();
    CHECK(model != nullptr);
    if (model == nullptr) {
        return;
    }
    CHECK(model->live_contexts() == 0);
    CHECK(!model->free_requested());
    CHECK(model->acquire_context());
    CHECK(model->live_contexts() == 1);

    llama_model_free(model); // live context owns the model: defer
    CHECK(model->free_requested());
    CHECK(model->live_contexts() == 1);
    CHECK(!model->name.empty());
    CHECK(!model->acquire_context()); // a new context is refused once free is requested

    CHECK(model->release_context()); // last live + free requested -> caller deletes
    delete model;

    // Without a free request, releasing the last reference only clears the
    // count; the caller still owns the model and frees it explicitly.
    llama_model * normal = make_mock_model();
    CHECK(normal != nullptr);
    if (normal != nullptr) {
        CHECK(normal->acquire_context());
        CHECK(!normal->release_context());
        CHECK(normal->live_contexts() == 0);
        llama_model_free(normal); // no live/retained contexts -> immediate delete
    }
}

struct open_ctx {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
};

// Load a CPU model and create one default context. Returns false when the
// path is unset, the model cannot load, or context creation fails; on failure
// no handle is left behind.
bool open_context(const char * model_path, open_ctx & o) {
    if (model_path == nullptr || model_path[0] == '\0') {
        return false;
    }
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    o.model = llama_model_load_from_file(model_path, mparams);
    if (o.model == nullptr) {
        return false;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx            = 64;
    cparams.n_batch          = 4;
    cparams.n_threads        = 1;
    cparams.n_threads_batch  = 1;
    try {
        o.ctx = llama_init_from_model(o.model, cparams);
    } catch (...) {
        o.ctx = nullptr;
    }
    if (o.ctx == nullptr) {
        llama_model_free(o.model);
        o.model = nullptr;
        return false;
    }
    return true;
}

// llama_model_free before llama_free is reachable through the public API. It
// must defer while the context is live, and the last clean context release
// must delete the model. The pending arm retains the context and the model.
void test_model_free_before_context(const char * model_path) {
    {
        open_ctx c;
        if (!open_context(model_path, c)) {
            return;
        }
        CHECK(c.model->live_contexts() == 1);
        llama_model_free(c.model); // must defer, not delete under the live context
        CHECK(c.model->free_requested());
        CHECK(c.model->live_contexts() == 1);
        CHECK(!c.model->name.empty());

        llama_free(c.ctx); // clean release: last live ref + requested -> delete model
        // c.ctx and c.model are gone; do not touch either pointer.
    }
    {
        open_ctx c;
        if (!open_context(model_path, c)) {
            return;
        }
        CHECK(c.model->live_contexts() == 1);
        c.ctx->debug_mark_unproven();
        llama_model_free(c.model); // must defer while the live context exists
        CHECK(c.model->free_requested());
        CHECK(c.model->live_contexts() == 1);
        CHECK(!c.model->name.empty());

        llama_free(c.ctx); // retain: the live reference and the model stay
        CHECK(c.ctx->is_retained());
        CHECK(c.model->live_contexts() == 1);
        CHECK(c.model->retained_contexts() == 1);
        CHECK(!c.model->name.empty());
    }
}

// retain_for_process clears the CPU backend's raw threadpool pointer. The
// caller may then free the threadpool without leaving a dangling pointer in a
// process-owned backend.
void test_cpu_threadpool_detached(const char * model_path) {
    open_ctx c;
    if (!open_context(model_path, c)) {
        return;
    }

    ggml_threadpool_params tpp = ggml_threadpool_params_default(1);
    ggml_threadpool_t tp = ggml_threadpool_new(&tpp);
    CHECK(tp != nullptr);
    c.ctx->debug_set_cpu_backend_threadpool(tp);
    CHECK(c.ctx->debug_cpu_backend_threadpool() == tp);

    c.ctx->debug_mark_unproven();
    llama_free(c.ctx); // retain whole context
    CHECK(c.ctx->is_retained());
    CHECK(c.ctx->debug_cpu_backend_threadpool() == nullptr);

    if (tp != nullptr) {
        ggml_threadpool_free(tp);
    }
    CHECK(c.model->live_contexts() == 1);
    CHECK(c.model->retained_contexts() == 1);
    llama_model_free(c.model);
    CHECK(!c.model->name.empty());
}

// A sampler is an external borrow. After retention the context never samples
// again, so the caller may free it; a duplicate llama_free must neither
// re-evaluate nor dereference it.
void test_sampler_borrow_after_retention(const char * model_path) {
    open_ctx c;
    if (!open_context(model_path, c)) {
        return;
    }
    llama_sampler * smpl = llama_sampler_init_greedy();
    CHECK(smpl != nullptr);
    if (smpl != nullptr) {
        CHECK(llama_set_sampler(c.ctx, 0, smpl));
    }

    c.ctx->debug_mark_unproven();
    llama_free(c.ctx); // retain
    CHECK(c.ctx->is_retained());

    if (smpl != nullptr) {
        llama_sampler_free(smpl); // caller-owned borrow dies after retention
    }
    llama_free(c.ctx); // duplicate free is ignored; no stale sampler deref
    CHECK(c.ctx->is_retained());
    CHECK(c.model->retained_contexts() == 1);
    llama_model_free(c.model);
    CHECK(!c.model->name.empty());
}

// A prefab envelope slot owns a scheduler separate from the main scheduler.
// The scheduler is created only when an envelope ubatch runs (the legacy init
// path builds the slot result, not the scheduler), so the fixture must decode
// one token first. An unproven slot alone must then retain the context and
// defer the model free.
void test_prefab_slot_unproven(const char * model_path) {
    if (model_path == nullptr || model_path[0] == '\0') {
        return;
    }
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        return;
    }

    const char * old_enable = std::getenv("LLAMA_PREFAB_ENABLE");
    const char * old_experimental = std::getenv("LLAMA_PREFAB_EXPERIMENTAL");
    const char * old_envelopes = std::getenv("LLAMA_PREFAB_ENVELOPES");
    const std::string save_enable = old_enable ? old_enable : "";
    const std::string save_experimental = old_experimental ? old_experimental : "";
    const std::string save_envelopes = old_envelopes ? old_envelopes : "";
    set_env("LLAMA_PREFAB_ENABLE", "1");
    set_env("LLAMA_PREFAB_EXPERIMENTAL", "1");
    // Pin a one-row envelope: the fixture decodes exactly one token, and the
    // slot arm is about slot retention, not sub-envelope padding. Without this
    // an exported LLAMA_PREFAB_ENVELOPES would change the path under test.
    set_env("LLAMA_PREFAB_ENVELOPES", "1");

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx            = 64;
    cparams.n_batch          = 4;
    cparams.n_threads        = 1;
    cparams.n_threads_batch  = 1;
    llama_context * ctx = nullptr;
    try {
        ctx = llama_init_from_model(model, cparams);
    } catch (...) {
        ctx = nullptr;
    }
    // The constructor has read both flags; leave the process environment as it
    // was for the other tests.
    set_env("LLAMA_PREFAB_ENABLE", old_enable ? save_enable.c_str() : nullptr);
    set_env("LLAMA_PREFAB_EXPERIMENTAL", old_experimental ? save_experimental.c_str() : nullptr);
    set_env("LLAMA_PREFAB_ENVELOPES", old_envelopes ? save_envelopes.c_str() : nullptr);

    if (ctx == nullptr) {
        llama_model_free(model);
        std::printf("TP5_CONTEXT_RETAIN_TEST prefab-skip (context init refused)\n");
        return;
    }
    const llama_prefab_state * prefab = ctx->get_prefab();
    if (prefab == nullptr || prefab->slots.empty()) {
        llama_free(ctx);
        llama_model_free(model);
        std::printf("TP5_CONTEXT_RETAIN_TEST prefab-skip (no envelope slot)\n");
        return;
    }

    // Build the smallest envelope on the real path: one token selects slot 0
    // and creates its scheduler, graph result and proof of completion.
    CHECK(prefab->slots[0].n_env == 1);
    llama_token token = 1;
    llama_batch batch = llama_batch_get_one(&token, 1);
    CHECK(llama_decode(ctx, batch) == 0);
    CHECK(prefab->slots[0].sched != nullptr);
    CHECK(prefab->slots[0].built);
    CHECK(!ctx->has_unproven()); // the envelope compute proved its scheduler

    CHECK(ctx->debug_mark_prefab_slot_unproven(0));
    CHECK(ctx->has_unproven()); // main scheduler is clean; the slot is not
    llama_free(ctx);
    CHECK(ctx->is_retained());
    CHECK(model->retained_contexts() == 1);
    llama_model_free(model);
    CHECK(!model->name.empty());
}

// With a real model, create a CPU context, force the scheduler's unproven
// state, and free the context. llama_free must retain the whole context and
// the model free must defer. The context destructor is never reached (a direct
// delete aborts by design).
void test_pending_context_retention(const char * model_path) {
    if (model_path == nullptr || model_path[0] == '\0') {
        std::printf("TP5_CONTEXT_RETAIN_TEST context-skip (no LLAMA_TEST_MODEL)\n");
        return;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        std::printf("TP5_CONTEXT_RETAIN_TEST context-skip (model load failed)\n");
        return;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx            = 64;
    cparams.n_batch          = 4;
    cparams.n_threads        = 1;
    cparams.n_threads_batch  = 1;

    llama_context * ctx = nullptr;
    try {
        ctx = llama_init_from_model(model, cparams);
    } catch (...) {
        ctx = nullptr;
    }
    if (ctx == nullptr) {
        std::printf("TP5_CONTEXT_RETAIN_TEST context-skip (context creation failed)\n");
        llama_model_free(model);
        return;
    }

    CHECK(model->retained_contexts() == 0);
    CHECK(model->live_contexts() == 1);
    ctx->debug_mark_unproven();
    CHECK(ctx->has_unproven());

    llama_free(ctx); // must retain the whole context, not delete it
    CHECK(ctx->is_retained());
    CHECK(model->live_contexts() == 1); // a retained context never releases
    CHECK(model->retained_contexts() == 1);

    // A duplicate llama_free must not increment the count or re-evaluate the
    // retained context.
    llama_free(ctx);
    CHECK(model->retained_contexts() == 1);

    llama_model_free(model); // must defer while the context is retained
    CHECK(model->live_contexts() == 1);
    CHECK(model->retained_contexts() == 1);
    CHECK(!model->name.empty()); // model object still alive
}

// A clean context takes the normal destruction path: the scheduler is freed
// before the backend members, and the model can then be freed.
void test_clean_context_retention(const char * model_path) {
    if (model_path == nullptr || model_path[0] == '\0') {
        return;
    }
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        return;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 64;
    cparams.n_batch         = 4;
    cparams.n_threads       = 1;
    cparams.n_threads_batch = 1;
    llama_context * ctx = nullptr;
    try {
        ctx = llama_init_from_model(model, cparams);
    } catch (...) {
        ctx = nullptr;
    }
    if (ctx == nullptr) {
        llama_model_free(model);
        return;
    }

    CHECK(model->retained_contexts() == 0);
    CHECK(model->live_contexts() == 1);
    llama_free(ctx); // clean: deletes the context
    llama_model_free(model); // clean: deletes the model
}

} // namespace

int main(int argc, char ** argv) {
    llama_backend_init();

    const char * model_path = argc > 1 ? argv[1] : std::getenv("LLAMA_TEST_MODEL");

    test_model_retention();
    test_model_lifetime_reference();
    test_model_free_before_context(model_path);
    test_cpu_threadpool_detached(model_path);
    test_sampler_borrow_after_retention(model_path);
    test_pending_context_retention(model_path);
    test_prefab_slot_unproven(model_path);
    test_clean_context_retention(model_path);

    llama_backend_free();

    if (g_failures != 0) {
        std::fprintf(stderr, "TP5_CONTEXT_RETAIN_TEST FAIL failures=%d\n", g_failures);
        return 1;
    }
    std::printf("TP5_CONTEXT_RETAIN_TEST PASS\n");
    return 0;
}
