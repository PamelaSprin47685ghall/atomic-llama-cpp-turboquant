// TP5 staging Meta partial-enqueue CPU test (no GPU, no HIP runtime).
//
// A MIRRORED Meta set can queue rank 0 and then get a clean rejection from
// rank 1. The destination is partially written; the Meta layer must call the
// v2 staging mark_partial entry instead of returning a silent success.
//
// The test builds a real Meta backend over two CPU child devices and injects a
// host-only staging api table. Its enqueue_async returns ok once then rejected.
// No CUDA symbol is linked.

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
    int  release_calls      = 0;
    int  enter_submit_calls = 0;
    bool pending            = false;
};

fake_staging_state g_fake;

bool fake_enabled()   { return !g_fake.pending; }
bool fake_usable()    { return !g_fake.pending; }
bool fake_configured(){ return true; }
bool fake_merge_wait(){ return false; }

bool fake_acquire_nowait(uint32_t, kind, size_t, lease * out) {
    if (g_fake.pending || out == nullptr) {
        return false;
    }
    *out = {};
    return true;
}

int fake_upload(const lease *, const void * src, size_t bytes) {
    return src != nullptr && bytes != 0
        ? int(submit_result::ok) : int(submit_result::rejected);
}

bool fake_wait(const lease *) {
    return !g_fake.pending;
}

void fake_release(const lease *) {
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
    ++g_fake.mark_partial_calls;
    g_fake.pending = true;
}

const api g_fake_api = {
    /* .abi_version       = */ kApiVersion,
    /* .enabled           = */ fake_enabled,
    /* .merge_wait        = */ fake_merge_wait,
    /* .acquire           = */ nullptr,
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
    /* .can_enqueue_sync  = */ nullptr,
    /* .enqueue_sync      = */ nullptr,
    /* .first_error       = */ fake_first_error,
    /* .enter_submit      = */ fake_enter_submit,
    /* .leave_submit      = */ fake_leave_submit,
    /* .unknown_count     = */ fake_unknown_count,
    /* .submission_pending= */ fake_submission_pending,
    /* .configured        = */ fake_configured,
    /* .usable            = */ fake_usable,
    /* .zc_enabled        = */ nullptr,
    /* .mark_partial      = */ fake_mark_partial,
};

ggml_backend_meta_split_state split_mirrored(const ggml_tensor *, void *) {
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

} // namespace

int main() {
    // The registration gate accepts the frozen v1 prefix and the v2 table, and
    // rejects an unknown future version before it can be read.
    api v1 = {};
    v1.abi_version = 1;
    CHECK(ggml_backend_meta_staging_register(&v1));
    CHECK(ggml_backend_meta_staging_get() == &v1);

    api future = {};
    future.abi_version = kApiVersion + 1;
    CHECK(!ggml_backend_meta_staging_register(&future));
    CHECK(ggml_backend_meta_staging_get() == &v1);

    CHECK(ggml_backend_meta_staging_register(&g_fake_api));
    CHECK(ggml_backend_meta_staging_get() == &g_fake_api);

    ggml_backend_reg_t cpu_reg = ggml_backend_cpu_reg();
    CHECK(cpu_reg != nullptr);
    ggml_backend_dev_t cpu = cpu_reg != nullptr ? ggml_backend_reg_dev_get(cpu_reg, 0) : nullptr;
    CHECK(cpu != nullptr);
    if (cpu == nullptr) {
        return 1;
    }

    ggml_backend_dev_t children[] = { cpu, cpu };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(children, 2, split_mirrored, nullptr);
    CHECK(meta_dev != nullptr);
    ggml_backend_t meta = meta_dev != nullptr ? ggml_backend_dev_init(meta_dev, nullptr) : nullptr;
    CHECK(meta != nullptr);
    if (meta == nullptr) {
        return 1;
    }

    ggml_init_params params = { /*.mem_size=*/ 64 * 1024, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);
    if (ctx == nullptr) {
        return 1;
    }

    ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
    CHECK(tensor != nullptr);
    if (tensor == nullptr) {
        return 1;
    }
    tensor->flags |= GGML_TENSOR_FLAG_INPUT; // the staging lane only rotates graph inputs

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx, ggml_backend_dev_buffer_type(meta_dev));
    CHECK(buffer != nullptr);
    CHECK(tensor->data != nullptr);
    if (buffer == nullptr) {
        return 1;
    }

    std::vector<uint8_t> source(ggml_nbytes(tensor), 0x5a);
    ggml_backend_tensor_set_async(meta, tensor, source.data(), 0, source.size());

    CHECK(g_fake.enqueue_calls == 2);          // rank 0 ok, rank 1 rejected
    CHECK(g_fake.mark_partial_calls == 1);     // the partial state was published
    CHECK(g_fake.release_calls == 1);          // the lease was released after marking
    CHECK(g_fake.pending);                     // submission_pending() is now true

    // The pool is fail-stop: the next set hits enter_submit() and returns
    // without another enqueue or a fallback write.
    const int enqueues_before = g_fake.enqueue_calls;
    const int enters_before   = g_fake.enter_submit_calls;
    ggml_backend_tensor_set_async(meta, tensor, source.data(), 0, source.size());
    CHECK(g_fake.enqueue_calls == enqueues_before);
    CHECK(g_fake.enter_submit_calls == enters_before + 1);

    // The destination and Meta owner are intentionally retained: freeing them
    // here would contradict the unknown-submission contract. The process owns
    // them until exit (run with ASAN_OPTIONS=detect_leaks=0).
    if (g_failures != 0) {
        std::fprintf(stderr, "TP5_STAGING_META_PARTIAL_TEST FAIL (%d)\n", g_failures);
        return 1;
    }
    std::printf("TP5_STAGING_META_PARTIAL_TEST PASS\n");
    return 0;
}
