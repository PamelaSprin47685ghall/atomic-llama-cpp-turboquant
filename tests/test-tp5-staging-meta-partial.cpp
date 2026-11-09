// TP5 staging Meta partial-enqueue CPU test (no GPU, no HIP runtime).
//
// A MIRRORED Meta submission can queue rank 0 and then get a clean rejection
// from rank 1. The destination is partially written; the Meta layer must call
// the v3 staging mark_partial entry instead of returning a silent success.
//
// Both lanes are exercised:
//   * the synchronous buffer set path (merge_wait + enqueue_sync), and
//   * the asynchronous backend path (acquire_nowait + enqueue_async).
//
// The test builds a real Meta backend over two CPU child devices and injects a
// host-only staging api table. No CUDA symbol is linked.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include "../ggml/src/ggml-cuda/staging.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace ggml_cuda_staging;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            ++g_failures;                                                                \
        }                                                                                \
    } while (0)

struct fake_staging_state {
    int  mark_partial_calls = 0;
    int  enqueue_calls      = 0;
    int  sync_enqueue_calls = 0;
    int  release_calls      = 0;
    int  enter_submit_calls = 0;
    bool pending            = false;
    bool released           = false;
    bool merge_wait_mode    = false;
};

fake_staging_state g_fake;

bool fake_enabled()   { return !g_fake.pending; }
bool fake_usable()    { return !g_fake.pending; }
bool fake_configured(){ return true; }
bool fake_merge_wait(){ return g_fake.merge_wait_mode; }

bool fake_acquire_common(lease * out) {
    if (g_fake.pending || out == nullptr) {
        return false;
    }
    *out = {};
    return true;
}

bool fake_acquire(uint32_t, kind, size_t, lease * out) {
    return fake_acquire_common(out);
}

bool fake_acquire_nowait(uint32_t, kind, size_t, lease * out) {
    return fake_acquire_common(out);
}

int fake_upload(const lease *, const void * src, size_t bytes) {
    return src != nullptr && bytes != 0
        ? int(submit_result::ok) : int(submit_result::rejected);
}

bool fake_wait(const lease *) {
    return !g_fake.pending;
}

void fake_release(const lease *) {
    CHECK(!g_fake.released);
    g_fake.released = true;
    ++g_fake.release_calls;
}

bool fake_can_enqueue_async(ggml_backend_t, ggml_tensor *, size_t, size_t) {
    return !g_fake.pending;
}

int fake_enqueue_async(const lease *, ggml_backend_t, ggml_tensor *, size_t, size_t) {
    ++g_fake.enqueue_calls;
    // Rank 0 is queued; rank 1 is a clean rejection before any copy.
    return g_fake.enqueue_calls == 1
        ? int(submit_result::ok) : int(submit_result::rejected);
}

bool fake_can_enqueue_sync(ggml_tensor *, size_t, size_t) {
    return !g_fake.pending;
}

int fake_enqueue_sync(const lease *, ggml_tensor *, size_t, size_t) {
    ++g_fake.sync_enqueue_calls;
    return g_fake.sync_enqueue_calls == 1
        ? int(submit_result::ok) : int(submit_result::rejected);
}

int fake_first_error() {
    return 0; // a rejection is not a native error
}

bool fake_enter_submit() {
    ++g_fake.enter_submit_calls;
    return !g_fake.pending;
}

void fake_leave_submit() {
}

size_t fake_unknown_count() {
    return g_fake.pending ? 1 : 0;
}

bool fake_submission_pending() {
    return g_fake.pending;
}

void fake_mark_partial(const lease * l, const void * dst) {
    CHECK(l != nullptr);
    CHECK(dst != nullptr);
    CHECK(!g_fake.released); // mark_partial must run before the lease release
    ++g_fake.mark_partial_calls;
    g_fake.pending = true;
}

const api g_fake_api = {
    /* .abi_version       = */ kApiVersion,
    /* .enabled           = */ fake_enabled,
    /* .merge_wait        = */ fake_merge_wait,
    /* .acquire           = */ fake_acquire,
    /* .acquire_nowait    = */ fake_acquire_nowait,
    /* .upload            = */ fake_upload,
    /* .upload_rows       = */ nullptr,
    /* .wait              = */ fake_wait,
    /* .release           = */ fake_release,
    /* .owns              = */ nullptr,
    /* .device_ptr        = */ nullptr,
    /* .dump              = */ nullptr,
    /* .can_enqueue_async = */ fake_can_enqueue_async,
    /* .enqueue_async     = */ fake_enqueue_async,
    /* .can_enqueue_sync  = */ fake_can_enqueue_sync,
    /* .enqueue_sync      = */ fake_enqueue_sync,
    /* .first_error       = */ fake_first_error,
    /* .enter_submit      = */ fake_enter_submit,
    /* .leave_submit      = */ fake_leave_submit,
    /* .unknown_count     = */ fake_unknown_count,
    /* .submission_pending= */ fake_submission_pending,
    /* .configured        = */ fake_configured,
    /* .usable            = */ fake_usable,
    /* .zc_enabled        = */ nullptr,
    /* .mark_partial      = */ fake_mark_partial,
    /* .kind_count       = */ kKindCount,
};

ggml_backend_meta_split_state split_mirrored(const ggml_tensor *, void *) {
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

struct meta_fixture {
    ggml_backend_t        backend = nullptr;
    ggml_context *        ctx     = nullptr;
    ggml_backend_buffer_t buffer  = nullptr;
    ggml_tensor *         tensor  = nullptr;
};

bool make_fixture(ggml_backend_dev_t cpu, meta_fixture & fixture) {
    ggml_backend_dev_t children[] = { cpu, cpu };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(children, 2, split_mirrored, nullptr);
    CHECK(meta_dev != nullptr);
    if (meta_dev == nullptr) {
        return false;
    }
    fixture.backend = ggml_backend_dev_init(meta_dev, nullptr);
    CHECK(fixture.backend != nullptr);
    if (fixture.backend == nullptr) {
        return false;
    }

    ggml_init_params params = { /*.mem_size=*/ 64 * 1024, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    fixture.ctx = ggml_init(params);
    CHECK(fixture.ctx != nullptr);
    if (fixture.ctx == nullptr) {
        return false;
    }

    fixture.tensor = ggml_new_tensor_2d(fixture.ctx, GGML_TYPE_F32, 4, 4);
    CHECK(fixture.tensor != nullptr);
    if (fixture.tensor == nullptr) {
        return false;
    }
    fixture.tensor->flags |= GGML_TENSOR_FLAG_INPUT; // the staging lane only rotates graph inputs

    fixture.buffer = ggml_backend_alloc_ctx_tensors_from_buft(
            fixture.ctx, ggml_backend_dev_buffer_type(meta_dev));
    CHECK(fixture.buffer != nullptr);
    CHECK(fixture.tensor->data != nullptr);
    return fixture.buffer != nullptr && fixture.tensor->data != nullptr;
}

} // namespace

int main() {
    // The registration gate accepts only a v3 table with mark_partial and a
    // kind_count covering the host's kind domain. A v1 table cannot publish a
    // partial submission; a v2 table cannot advertise its kind domain. Both
    // are rejected and the staging path falls back instead of aborting later.
    // An unknown future version is rejected before any new field is read.
    const void * before = ggml_backend_meta_staging_get();
    api v1 = {};
    v1.abi_version = 1;
    CHECK(!ggml_backend_meta_staging_register(&v1));
    CHECK(ggml_backend_meta_staging_get() == before);

    api future = {};
    future.abi_version = kApiVersion + 1;
    CHECK(!ggml_backend_meta_staging_register(&future));
    CHECK(ggml_backend_meta_staging_get() == before);

    api v2 = g_fake_api;
    v2.abi_version = 2;
    CHECK(!ggml_backend_meta_staging_register(&v2));
    CHECK(ggml_backend_meta_staging_get() == before);

    api broken = g_fake_api;
    broken.mark_partial = nullptr;
    CHECK(!ggml_backend_meta_staging_register(&broken));
    CHECK(ggml_backend_meta_staging_get() == before);

    api short_kinds = g_fake_api;
    short_kinds.kind_count = uint32_t(kKindCount - 1);
    CHECK(!ggml_backend_meta_staging_register(&short_kinds));
    CHECK(ggml_backend_meta_staging_get() == before);

    CHECK(ggml_backend_meta_staging_register(&g_fake_api));
    CHECK(ggml_backend_meta_staging_get() == &g_fake_api);

    ggml_backend_reg_t cpu_reg = ggml_backend_cpu_reg();
    CHECK(cpu_reg != nullptr);
    ggml_backend_dev_t cpu = cpu_reg != nullptr ? ggml_backend_reg_dev_get(cpu_reg, 0) : nullptr;
    CHECK(cpu != nullptr);
    if (cpu == nullptr) {
        return 1;
    }

    // Synchronous buffer-set lane: merge_wait is enabled, rank 0 is queued and
    // rank 1 rejects. The void setter cannot return the fault, but mark_partial
    // must publish it and the next operations must see the fail-stop state.
    {
        g_fake = {};
        g_fake.merge_wait_mode = true;

        meta_fixture sync_fixture;
        if (!make_fixture(cpu, sync_fixture)) {
            return 1;
        }
        std::vector<uint8_t> source(ggml_nbytes(sync_fixture.tensor), 0x5a);
        ggml_backend_tensor_set(sync_fixture.tensor, source.data(), 0, source.size());

        CHECK(g_fake.sync_enqueue_calls == 2);
        CHECK(g_fake.mark_partial_calls == 1);
        CHECK(g_fake.release_calls == 1);
        CHECK(g_fake.pending);
        CHECK(g_fake.enter_submit_calls == 0); // partial returns before the fallback
    }

    // Asynchronous backend lane: the same partial publication, then the next
    // set hits the fatal submit gate without another enqueue or fallback write.
    {
        g_fake = {};

        meta_fixture async_fixture;
        if (!make_fixture(cpu, async_fixture)) {
            return 1;
        }
        std::vector<uint8_t> source(ggml_nbytes(async_fixture.tensor), 0x5a);
        ggml_backend_tensor_set_async(
                async_fixture.backend, async_fixture.tensor, source.data(), 0, source.size());

        CHECK(g_fake.enqueue_calls == 2);          // rank 0 ok, rank 1 rejected
        CHECK(g_fake.mark_partial_calls == 1);     // the partial state was published
        CHECK(g_fake.release_calls == 1);          // the lease was released after marking
        CHECK(g_fake.pending);                     // submission_pending() is now true

        const int enqueues_before = g_fake.enqueue_calls;
        const int enters_before   = g_fake.enter_submit_calls;
        ggml_backend_tensor_set_async(
                async_fixture.backend, async_fixture.tensor, source.data(), 0, source.size());
        CHECK(g_fake.enqueue_calls == enqueues_before);
        CHECK(g_fake.enter_submit_calls == enters_before + 1);
    }

    // The destinations and Meta owners are intentionally retained: freeing
    // them here would contradict the unknown-submission contract. The process
    // owns them until exit (run with ASAN_OPTIONS=detect_leaks=0).
    if (g_failures != 0) {
        std::fprintf(stderr, "TP5_STAGING_META_PARTIAL_TEST FAIL (%d)\n", g_failures);
        return 1;
    }
    std::printf("TP5_STAGING_META_PARTIAL_TEST PASS\n");
    return 0;
}
