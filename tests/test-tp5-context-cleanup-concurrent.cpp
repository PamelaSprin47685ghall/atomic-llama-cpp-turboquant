// TP5 multi-provider CPU Meta-lane test: two real CPU Meta backends whose
// child devices carry different registries. The context must enumerate both
// lanes, bind one token per provider, share target/draft references, and
// refuse a partial set. No GPU and no mock echo.

#include "llama.h"
#include "ggml-cpu.h"

#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-cuda/staging.h"
#include "../src/llama-context.h"
#include "../../fork/tp5-compose/src/staging_owner_retain.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <chrono>

namespace {

int g_failures = 0;
extern "C" void llama_tp5_test_fail_after_bind_count(int);
extern "C" void ggml_backend_meta_test_fail_clear_count(int);
extern "C" void ggml_backend_meta_test_fail_unbind_count(int);
extern "C" size_t ggml_backend_meta_staging_owner_orphan_count(void);
extern "C" void llama_tp5_test_force_bind(int);
extern "C" void llama_tp5_test_barrier_arm(int);
extern "C" bool llama_tp5_test_cleanup_idle(void);
extern "C" void llama_tp5_test_fail_after_bind_count(int);
extern "C" void llama_tp5_test_force_bind(int);
extern "C" void llama_tp5_test_barrier_arm(int);
extern "C" bool llama_tp5_test_cleanup_idle(void);
#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            ++g_failures;                                                                \
        }                                                                                \
    } while (0)

ggml_backend_reg g_reg_a = {};
ggml_backend_reg g_reg_b = {};
ggml_backend_device g_dev_a = {};
ggml_backend_device g_dev_b = {};
uint64_t g_owner_a = 0x1000;
uint64_t g_owner_b = 0x2000;
std::atomic<int> g_busy_once{0};

bool api_enabled() { return true; }
bool api_merge_wait() { return false; }
bool api_configured() { return true; }
bool api_usable() { return true; }
void api_mark_partial(const ggml_cuda_staging::lease *, const void *) {}
bool api_owner_acquire_a(uint64_t, uint32_t, uint64_t * owner, uint64_t * generation) {
    *owner = g_owner_a++;
    *generation = 1;
    return true;
}
bool api_owner_acquire_b(uint64_t, uint32_t, uint64_t * owner, uint64_t * generation) {
    *owner = g_owner_b++;
    *generation = 1;
    return true;
}
bool api_acquire_owner(uint64_t, uint64_t, ggml_cuda_staging::kind, size_t,
                       ggml_cuda_staging::lease *, bool) { return false; }
int api_owner_drain(uint64_t, uint64_t) {
    if (g_busy_once.exchange(0) != 0) {
        return 1; // temporary busy, then recover
    }
    return int(ggml_cuda_staging::owner_drain_result::drained);
}
bool api_owner_release(uint64_t, uint64_t) { return true; }
bool api_can_enqueue_async(ggml_backend_t, ggml_tensor *, size_t, size_t) { return false; }
int api_enqueue_async(const ggml_cuda_staging::lease *, ggml_backend_t, ggml_tensor *, size_t, size_t) {
    return int(ggml_cuda_staging::submit_result::rejected);
}
bool api_can_enqueue_sync(ggml_tensor *, size_t, size_t) { return false; }
int api_enqueue_sync(const ggml_cuda_staging::lease *, ggml_tensor *, size_t, size_t) {
    return int(ggml_cuda_staging::submit_result::rejected);
}
int api_first_error() { return 0; }
bool api_enter_submit() { return true; }
void api_leave_submit() {}
size_t api_unknown_count() { return 0; }
bool api_submission_pending() { return false; }

ggml_cuda_staging::api make_api(bool provider_b) {
    ggml_cuda_staging::api a = {};
    a.abi_version = ggml_cuda_staging::kApiVersion;
    a.enabled = api_enabled;
    a.merge_wait = api_merge_wait;
    a.configured = api_configured;
    a.usable = api_usable;
    a.mark_partial = api_mark_partial;
    a.kind_count = ggml_cuda_staging::kKindCount;
    a.owner_acquire = provider_b ? api_owner_acquire_b : api_owner_acquire_a;
    a.acquire_owner = api_acquire_owner;
    a.owner_drain = api_owner_drain;
    a.owner_release = api_owner_release;
    a.can_enqueue_async = api_can_enqueue_async;
    a.enqueue_async = api_enqueue_async;
    a.can_enqueue_sync = api_can_enqueue_sync;
    a.enqueue_sync = api_enqueue_sync;
    a.first_error = api_first_error;
    a.enter_submit = api_enter_submit;
    a.leave_submit = api_leave_submit;
    a.unknown_count = api_unknown_count;
    a.submission_pending = api_submission_pending;
    return a;
}

ggml_cuda_staging::api g_api_a = make_api(false);
ggml_cuda_staging::api g_api_b = make_api(true);

ggml_backend_meta_split_state split_meta(const ggml_tensor *, void *) {
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) return 77;
    ggml_backend_reg_t cpu_reg = ggml_backend_cpu_reg();
    CHECK(cpu_reg != nullptr);
    ggml_backend_dev_t cpu_dev = cpu_reg ? ggml_backend_reg_dev_get(cpu_reg, 0) : nullptr;
    CHECK(cpu_dev != nullptr);
    if (!cpu_dev) return 1;
    g_reg_a = *cpu_reg; g_reg_b = *cpu_reg;
    g_dev_a = *cpu_dev; g_dev_b = *cpu_dev;
    g_dev_a.reg = &g_reg_a; g_dev_b.reg = &g_reg_b;
    CHECK(ggml_backend_meta_staging_register_for_reg(&g_reg_a, &g_api_a));
    CHECK(ggml_backend_meta_staging_register_for_reg(&g_reg_b, &g_api_b));

    ggml_backend_dev_t children_a[1] = { &g_dev_a };
    ggml_backend_dev_t children_b[1] = { &g_dev_b };
    ggml_backend_dev_t meta_a = ggml_backend_meta_device(children_a, 1, split_meta, nullptr);
    ggml_backend_dev_t meta_b = ggml_backend_meta_device(children_b, 1, split_meta, nullptr);
    CHECK(meta_a && meta_b);
    if (!meta_a || !meta_b) return 1;

    auto load_model = [&]() {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
        ggml_backend_dev_t devs[3] = { meta_a, meta_b, nullptr };
        mp.devices = devs;
        return llama_model_load_from_file(argv[1], mp);
    };
    llama_model * model_a = load_model();
    llama_model * model_b = load_model();
    CHECK(model_a && model_b);
    if (!model_a || !model_b) return 1;

    uint64_t owner_a = 0, gen_a = 0, owner_b = 0, gen_b = 0;
    CHECK(ggml_backend_meta_staging_owner_acquire_for_reg(&g_reg_a, 100, 0, &owner_a, &gen_a));
    CHECK(ggml_backend_meta_staging_owner_acquire_for_reg(&g_reg_b, 200, 0, &owner_b, &gen_b));
    CHECK(owner_a && owner_b);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64; cp.n_batch = 4; cp.n_ubatch = 4;
    cp.n_threads = 1; cp.n_threads_batch = 1; cp.no_perf = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; cp.kv_unified = true;
    cp.tp5_owner = owner_a; cp.tp5_owner_generation = gen_a;

    llama_tp5_test_fail_after_bind_count(2);
    ggml_backend_meta_test_fail_clear_count(4);
    ggml_backend_meta_test_fail_unbind_count(4);

    std::atomic<int> nulls{0};
    std::thread t1([&] {
        llama_context_params p = cp;
        llama_context * c = llama_init_from_model(model_a, p);
        if (!c) nulls.fetch_add(1); else llama_free(c);
    });
    std::thread t2([&] {
        llama_context_params p = cp;
        p.tp5_owner = owner_b; p.tp5_owner_generation = gen_b;
        llama_context * c = llama_init_from_model(model_b, p);
        if (!c) nulls.fetch_add(1); else llama_free(c);
    });
    t1.join(); t2.join();
    CHECK(nulls.load() == 2);

    bool ra = false, rb = false;
    for (int i = 0; i < 200000 && !(ra && rb); ++i) {
        ra = ra || ggml_backend_meta_staging_owner_release(owner_a, gen_a);
        rb = rb || ggml_backend_meta_staging_owner_release(owner_b, gen_b);
        if (!(ra && rb)) std::this_thread::yield();
    }
    CHECK(ra && rb);
    CHECK(ggml_backend_meta_staging_owner_orphan_count() == 0);

    llama_model_free(model_a);
    llama_model_free(model_b);
    bool had = false;
    CHECK(ggml_backend_meta_staging_unregister_reg(&g_reg_a, &had)); CHECK(had);
    had = false;
    CHECK(ggml_backend_meta_staging_unregister_reg(&g_reg_b, &had)); CHECK(had);

    if (g_failures) { std::fprintf(stderr, "concurrent cleanup FAIL (%d)\n", g_failures); return 1; }
    std::puts("concurrent_context_cleanup_failure: PASS");
    return 0;
}
