// TP5 staging owner-runtime CPU test (real Qwen4Exp target/draft contexts).
//
// The retainer must keep a busy run's target/MTP contexts alive when a later
// run retains its own owner, keep them on owner_release failure, and reap a
// delayed completion from the background retry thread. The staging provider is
// a real pool with controllable events, not an echo.
//
// usage: test-tp5-staging-owner-runtime <target.gguf> <draft.gguf>

#include "llama.h"
#include "staging_owner_retain.h"
#include "../ggml/src/ggml-cuda/staging.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

using namespace ggml_cuda_staging;

namespace {

int g_failures = 0;
ggml_backend_meta_split_state split_meta(const ggml_tensor *, void *) {
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}
#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            ++g_failures;                                                                \
        }                                                                                \
    } while (0)

struct fake_event {
    bool complete = false;
};

int g_host_allocs = 0;
int g_host_frees = 0;
int g_events = 0;
int g_event_destroys = 0;
bool g_release_fail = false;
std::vector<fake_event *> g_recorded;

int fake_host_alloc(size_t bytes, void ** out) {
    void * p = std::malloc(bytes != 0 ? bytes : 1);
    if (p == nullptr) {
        return 1;
    }
    ++g_host_allocs;
    *out = p;
    return 0;
}

void fake_host_free(void * p) {
    ++g_host_frees;
    std::free(p);
}

int fake_event_create(void ** out) {
    *out = new fake_event();
    ++g_events;
    return 0;
}

int fake_event_record(void * opaque, void *) {
    fake_event * event = static_cast<fake_event *>(opaque);
    event->complete = false;
    g_recorded.push_back(event);
    return 0;
}

int fake_event_sync(void * opaque) {
    static_cast<fake_event *>(opaque)->complete = true;
    return 0;
}

int fake_event_query(void * opaque, bool * complete) {
    *complete = static_cast<fake_event *>(opaque)->complete;
    return 0;
}

void fake_event_destroy(void * opaque) {
    ++g_event_destroys;
    delete static_cast<fake_event *>(opaque);
}

int fake_copy_h2d(void * dst, const void * src, size_t bytes, void *) {
    std::memcpy(dst, src, bytes);
    return 0;
}

int fake_current_device() { return 0; }
bool fake_select_device(int) { return true; }
bool fake_restore_device(int) { return true; }

backend_ops fake_ops() {
    backend_ops ops;
    ops.host_alloc = fake_host_alloc;
    ops.host_free = fake_host_free;
    ops.event_create = fake_event_create;
    ops.event_record = fake_event_record;
    ops.event_sync = fake_event_sync;
    ops.event_query = fake_event_query;
    ops.event_destroy = fake_event_destroy;
    ops.copy_h2d = fake_copy_h2d;
    ops.current_device = fake_current_device;
    ops.select_device = fake_select_device;
    ops.restore_device = fake_restore_device;
    return ops;
}

// Leaked: the retainer's atexit retry and background thread may run after main.
pool * g_pool = nullptr;

bool api_enabled() { return true; }
bool api_merge_wait() { return true; }
bool api_acquire(uint32_t window, kind k, size_t bytes, lease * out) {
    return g_pool->acquire(window, k, bytes, out);
}
bool api_acquire_nowait(uint32_t window, kind k, size_t bytes, lease * out) {
    return g_pool->acquire(window, k, bytes, out, false);
}
int api_upload(const lease * l, const void * src, size_t bytes) {
    return int(g_pool->upload(*l, src, bytes));
}
int api_upload_rows(const lease * l, const void * src, size_t src_stride, size_t dst_stride,
                    size_t row_bytes, size_t rows) {
    return int(g_pool->upload_rows(*l, src, src_stride, dst_stride, row_bytes, rows));
}
bool api_wait(const lease * l) { return g_pool->wait(*l); }
void api_release(const lease * l) { g_pool->release(*l); }
bool api_owns(const void * p, size_t bytes) { return g_pool->owns(p, bytes); }
void * api_device_ptr(const lease * l, int rank) { return g_pool->device_ptr(*l, rank); }
void api_dump() { g_pool->dump(stderr); }
bool api_can_enqueue_async(ggml_backend_t, ggml_tensor *, size_t, size_t) { return false; }
int api_enqueue_async(const lease *, ggml_backend_t, ggml_tensor *, size_t, size_t) {
    return int(submit_result::rejected);
}
bool api_can_enqueue_sync(ggml_tensor *, size_t, size_t) { return false; }
int api_enqueue_sync(const lease *, ggml_tensor *, size_t, size_t) {
    return int(submit_result::rejected);
}
int api_first_error() { return g_pool->first_error(); }
bool api_enter_submit() { return g_pool->enter_submit(); }
void api_leave_submit() { g_pool->leave_submit(); }
size_t api_unknown_count() { return g_pool->unknown_count(); }
bool api_submission_pending() { return g_pool->submission_pending(); }
bool api_configured() { return true; }
bool api_usable() { return g_pool->usable(); }
bool api_zc_enabled() { return false; }
void api_mark_partial(const lease * l, const void * dst) { g_pool->mark_partial(*l, dst); }
bool api_owner_acquire(uint64_t run_id, uint32_t window_id, uint64_t * owner, uint64_t * gen) {
    return g_pool->owner_acquire(run_id, window_id, owner, gen);
}
bool api_acquire_owner(uint64_t owner, uint64_t gen, kind k, size_t bytes,
                       lease * out, bool wait_pending) {
    return g_pool->acquire_owner(owner, gen, k, bytes, out, wait_pending);
}
int api_owner_drain(uint64_t owner, uint64_t gen) {
    return int(g_pool->owner_drain(owner, gen));
}
bool api_owner_release(uint64_t owner, uint64_t gen) {
    if (g_release_fail) {
        return false;
    }
    return g_pool->owner_release(owner, gen);
}

const api g_api = {
    /* .abi_version       = */ kApiVersion,
    /* .enabled           = */ api_enabled,
    /* .merge_wait        = */ api_merge_wait,
    /* .acquire           = */ api_acquire,
    /* .acquire_nowait    = */ api_acquire_nowait,
    /* .upload            = */ api_upload,
    /* .upload_rows       = */ api_upload_rows,
    /* .wait              = */ api_wait,
    /* .release           = */ api_release,
    /* .owns              = */ api_owns,
    /* .device_ptr        = */ api_device_ptr,
    /* .dump              = */ api_dump,
    /* .can_enqueue_async = */ api_can_enqueue_async,
    /* .enqueue_async     = */ api_enqueue_async,
    /* .can_enqueue_sync  = */ api_can_enqueue_sync,
    /* .enqueue_sync      = */ api_enqueue_sync,
    /* .first_error       = */ api_first_error,
    /* .enter_submit      = */ api_enter_submit,
    /* .leave_submit      = */ api_leave_submit,
    /* .unknown_count     = */ api_unknown_count,
    /* .submission_pending= */ api_submission_pending,
    /* .configured        = */ api_configured,
    /* .usable            = */ api_usable,
    /* .zc_enabled        = */ api_zc_enabled,
    /* .mark_partial      = */ api_mark_partial,
    /* .kind_count        = */ kKindCount,
    /* .owner_acquire     = */ api_owner_acquire,
    /* .acquire_owner     = */ api_acquire_owner,
    /* .owner_drain       = */ api_owner_drain,
    /* .owner_release     = */ api_owner_release,
};

int g_context_frees = 0;
struct counting_deleter {
    void operator()(llama_context * p) const {
        ++g_context_frees;
        llama_free(p);
    }
};
using test_context_ptr = std::unique_ptr<llama_context, counting_deleter>;

int g_model_frees = 0;
int g_order_failures = 0;
int g_expected_contexts = 0;
struct model_counting_deleter {
    void operator()(llama_model * p) const {
        if (g_context_frees < g_expected_contexts) {
            ++g_order_failures;
        }
        ++g_model_frees;
        llama_model_free(p);
    }
};
using test_model_ptr = std::unique_ptr<llama_model, model_counting_deleter>;

struct contexts {
    std::vector<test_context_ptr> target;
    std::vector<test_context_ptr> mtp;
};

bool make_contexts(llama_model * model_tgt, llama_model * model_dft, contexts & out) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 64;
    cparams.n_batch         = 4;
    cparams.n_ubatch        = 4;
    cparams.n_seq_max       = 1;
    cparams.n_threads       = 2;
    cparams.n_threads_batch = 2;
    cparams.kv_unified      = true;

    llama_context * tgt = llama_init_from_model(model_tgt, cparams);
    CHECK(tgt != nullptr);
    if (tgt == nullptr) {
        return false;
    }
    llama_context_params dparams = cparams;
    dparams.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
    dparams.ctx_other = tgt;
    llama_context * dft = llama_init_from_model(model_dft, dparams);
    CHECK(dft != nullptr);
    if (dft == nullptr) {
        llama_free(tgt);
        return false;
    }
    out.target.emplace_back(tgt);
    out.mtp.emplace_back(dft);
    return true;
}

bool enqueue_pending(uint64_t owner, uint64_t generation, fake_event ** event_out) {
    lease l;
    if (!g_pool->acquire_owner(owner, generation, kind::tokens, 32, &l, true)) {
        return false;
    }
    uint8_t src[32] = {};
    uint8_t dst[32] = {};
    if (g_pool->upload(l, src, sizeof(src)) != submit_result::ok ||
            g_pool->enqueue(l, 0, dst, sizeof(src), nullptr) != submit_result::ok) {
        g_pool->release(l);
        return false;
    }
    g_pool->release(l);
    if (event_out != nullptr) {
        *event_out = g_recorded.empty() ? nullptr : g_recorded.back();
    }
    return true;
}

void mark_complete(fake_event * event) {
    if (event != nullptr) {
        event->complete = true;
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "need target and draft GGUFs\n");
        return 77;
    }
    llama_backend_init();
    ggml_backend_reg_t cpu_reg = ggml_backend_cpu_reg();
    CHECK(cpu_reg != nullptr);
    CHECK(ggml_backend_meta_staging_register_for_reg(cpu_reg, &g_api));
    g_pool = new pool(fake_ops());

    // The standard tensor-split path exposes a Meta outer device. Its staging
    // provider must resolve to the common CPU child registry, not the Meta
    // registry, and _for_device must acquire through that child provider.
    ggml_backend_dev_t cpu_dev = cpu_reg != nullptr ? ggml_backend_reg_dev_get(cpu_reg, 0) : nullptr;
    CHECK(cpu_dev != nullptr);
    if (cpu_dev != nullptr) {
        ggml_backend_dev_t children[2] = { cpu_dev, cpu_dev };
        ggml_backend_dev_t meta_dev = ggml_backend_meta_device(children, 2, split_meta, nullptr);
        CHECK(meta_dev != nullptr);
        CHECK(ggml_backend_meta_staging_provider_reg_for_device(meta_dev) == cpu_reg);
        uint64_t meta_owner = 0;
        uint64_t meta_generation = 0;
        CHECK(ggml_backend_meta_staging_owner_acquire_for_device(
                meta_dev, 1, 0, &meta_owner, &meta_generation));
        CHECK(ggml_backend_meta_staging_owner_drain(meta_owner, meta_generation) ==
              int(owner_drain_result::drained));
        CHECK(ggml_backend_meta_staging_owner_release(meta_owner, meta_generation));
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    test_model_ptr model_tgt(llama_model_load_from_file(argv[1], mparams));
    llama_model_params dparams = mparams;
    dparams.load_mtp = true;
    test_model_ptr model_dft(llama_model_load_from_file(argv[2], dparams));
    CHECK(model_tgt != nullptr && model_dft != nullptr);
    if (model_tgt == nullptr || model_dft == nullptr) {
        return 1;
    }

    auto * retainer = new tq::tp5::staging_owner_retainer<test_context_ptr, test_model_ptr>();
    const std::array<test_model_ptr *, tq::tp5::kMaxRetainedModels> no_models = {};

    contexts a;
    CHECK(make_contexts(model_tgt.get(), model_dft.get(), a));
    llama_context * raw_tgt_a = a.target[0].get();
    llama_context * raw_dft_a = a.mtp[0].get();
    CHECK(llama_tp5_context_staging_drain(raw_tgt_a) ==
          LLAMA_TP5_CONTEXT_DRAIN_DRAINED);
    uint64_t public_owner = 0, public_generation = 0;
    CHECK(!llama_tp5_context_staging_owner(
            raw_tgt_a, &public_owner, &public_generation));
    uint64_t owner_a = 0, gen_a = 0;
    CHECK(g_pool->owner_acquire(100, 0, &owner_a, &gen_a));
    fake_event * event_a = nullptr;
    CHECK(enqueue_pending(owner_a, gen_a, &event_a));
    std::array<test_model_ptr *, tq::tp5::kMaxRetainedModels> models_a = {
        &model_tgt, &model_dft, nullptr
    };
    CHECK(retainer->retain_run(&a.target, &a.mtp, nullptr, models_a, &owner_a, &gen_a, 1));
    CHECK(retainer->retained_runs() == 1);
    CHECK(retainer->retained_tokens() == 1);
    CHECK(a.target.empty() && a.mtp.empty());

    // A later run with a different owner and no contexts must not overwrite
    // or destroy the older run's contexts/models.
    std::vector<test_context_ptr> empty_b_t;
    std::vector<test_context_ptr> empty_b_m;
    uint64_t owner_b = 0, gen_b = 0;
    CHECK(g_pool->owner_acquire(101, 1, &owner_b, &gen_b));
    fake_event * event_b = nullptr;
    CHECK(enqueue_pending(owner_b, gen_b, &event_b));
    CHECK(retainer->retain_run(&empty_b_t, &empty_b_m, nullptr, no_models, &owner_b, &gen_b, 1));
    CHECK(retainer->retained_runs() == 2);

    CHECK(raw_tgt_a != nullptr && llama_n_ctx(raw_tgt_a) > 0);
    CHECK(raw_dft_a != nullptr && llama_get_model(raw_dft_a) != nullptr);
    CHECK(g_model_frees == 0);

    // Release failure keeps the token and the run's contexts.
    const int frees_before = g_context_frees;
    g_release_fail = true;
    mark_complete(event_a);
    retainer->retry();
    CHECK(retainer->retained_tokens() >= 1);
    CHECK(g_context_frees == frees_before);
    CHECK(llama_n_ctx(raw_tgt_a) > 0);

    // A successful retry releases A; contexts are destroyed before models.
    g_release_fail = false;
    g_expected_contexts = g_context_frees + 2;
    mark_complete(event_a);
    retainer->retry();
    CHECK(g_context_frees >= frees_before + 2);
    CHECK(g_model_frees == 2);
    CHECK(g_order_failures == 0);

    // Delayed completion with no manual retry: the background thread reaps B.
    mark_complete(event_b);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (retainer->retained_tokens() != 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    CHECK(retainer->retained_tokens() == 0);
    CHECK(g_context_frees >= frees_before + 2);

    // Record capacity: four token-only runs fill the retainer; the fifth
    // retain must fail without losing the token.
    uint64_t owners[4] = {};
    uint64_t gens[4] = {};
    fake_event * events[4] = {};
    for (int i = 0; i < 4; ++i) {
        CHECK(g_pool->owner_acquire(200 + uint64_t(i), uint32_t(i), &owners[i], &gens[i]));
        CHECK(enqueue_pending(owners[i], gens[i], &events[i]));
        std::vector<test_context_ptr> empty_t;
        std::vector<test_context_ptr> empty_m;
        CHECK(retainer->retain_run(&empty_t, &empty_m, nullptr, no_models, &owners[i], &gens[i], 1));
    }
    CHECK(retainer->retained_tokens() == 4);
    CHECK(retainer->retained_runs() == 4);

    // The global credit bound is four: a fifth legitimate acquire fails.
    uint64_t fifth_owner = 0;
    uint64_t fifth_generation = 0;
    CHECK(!g_pool->owner_acquire(500, 0, &fifth_owner, &fifth_generation));

    // An independent invalid handle is rejected by the provider and is not a
    // fifth retained token.
    CHECK(g_pool->owner_drain(999, 999) == owner_drain_result::invalid);
    CHECK(!g_pool->owner_release(999, 999));
    CHECK(retainer->retained_tokens() == 4);

    for (int i = 0; i < 4; ++i) {
        mark_complete(events[i]);
    }
    retainer->retry();
    CHECK(retainer->retained_tokens() == 0);
    retainer->stop();

    // A retain after stop must restart the worker, not enqueue an owner that
    // no thread will re-probe.
    uint64_t late_owner = 0;
    uint64_t late_generation = 0;
    CHECK(g_pool->owner_acquire(300, 0, &late_owner, &late_generation));
    fake_event * late_event = nullptr;
    CHECK(enqueue_pending(late_owner, late_generation, &late_event));
    std::vector<test_context_ptr> late_t;
    std::vector<test_context_ptr> late_m;
    CHECK(retainer->retain_run(&late_t, &late_m, nullptr, no_models,
                               &late_owner, &late_generation, 1));
    CHECK(retainer->retained_tokens() == 1);
    mark_complete(late_event);
    retainer->retry();
    CHECK(retainer->retained_tokens() == 0);
    retainer->stop();

    CHECK(g_model_frees == 2);
    CHECK(g_order_failures == 0);
    llama_backend_free();

    if (g_failures != 0) {
        std::fprintf(stderr, "TP5_STAGING_OWNER_RUNTIME_TEST FAIL (%d)\n", g_failures);
        return 1;
    }
    std::printf("TP5_STAGING_OWNER_RUNTIME_TEST PASS\n");
    return 0;
}
