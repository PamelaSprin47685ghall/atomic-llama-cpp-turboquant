// TP5 WS4: HIP/CUDA adapter for the persistent pinned staging pool.

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/staging.h"

#include <cstdlib>

namespace ggml_cuda_staging {
namespace {

int hip_host_alloc(size_t bytes, void ** out) {
    void * host = nullptr;
    const cudaError_t err = cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) {
        return int(err);
    }
    *out = host;
    return 0;
}

void hip_host_free(void * ptr) {
    if (ptr != nullptr) {
        (void) cudaFreeHost(ptr);
    }
}

int hip_event_create(void ** out) {
    cudaEvent_t event = nullptr;
    const cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        return int(err);
    }
    *out = reinterpret_cast<void *>(event);
    return 0;
}

int hip_event_record(void * event, void * stream) {
    return int(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),
                               reinterpret_cast<cudaStream_t>(stream)));
}

int hip_event_sync(void * event) {
    return int(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event)));
}

int hip_event_query(void * event, bool * complete) {
    const cudaError_t err = cudaEventQuery(reinterpret_cast<cudaEvent_t>(event));
    if (err == cudaSuccess) {
        *complete = true;
        return 0;
    }
    if (err == cudaErrorNotReady) {
        // NotReady is a state, not a fault. Do not call cudaGetLastError()
        // here: clearing the sticky error would hide a real asynchronous
        // failure from the next CUDA_CHECK(cudaGetLastError()).
        *complete = false;
        return 0;
    }
    return int(err);
}

void hip_event_destroy(void * event) {
    if (event != nullptr) {
        (void) cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event));
    }
}

int hip_copy_h2d(void * dst, const void * src, size_t bytes, void * stream) {
    return int(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                               reinterpret_cast<cudaStream_t>(stream)));
}

int hip_device_ptr(void * host, void ** out) {
    void * dev = nullptr;
    const cudaError_t err = cudaHostGetDevicePointer(&dev, host, 0);
    if (err != cudaSuccess) {
        return int(err);
    }
    *out = dev;
    return 0;
}

int hip_current_device() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return -1;
    }
    return device;
}

bool hip_select_device(int rank) {
    if (rank < 0 || rank >= GGML_CUDA_MAX_DEVICES) {
        return false;
    }
    // ggml_cuda_set_device translates the virtual rank to the physical GPU.
    ggml_cuda_set_device(rank);
    return true;
}

bool hip_restore_device(int physical) {
    if (physical < 0) {
        return true;
    }
    if (cudaSetDevice(physical) != cudaSuccess) {
        return false;
    }
    return true;
}

const backend_ops kHipOps = {
    /* .host_alloc   = */ hip_host_alloc,
    /* .host_free    = */ hip_host_free,
    /* .event_create = */ hip_event_create,
    /* .event_record = */ hip_event_record,
    /* .event_sync   = */ hip_event_sync,
    /* .event_query  = */ hip_event_query,
    /* .event_destroy= */ hip_event_destroy,
    /* .copy_h2d     = */ hip_copy_h2d,
    /* .device_ptr   = */ hip_device_ptr,
    /* .current_device = */ hip_current_device,
    /* .select_device  = */ hip_select_device,
    /* .restore_device = */ hip_restore_device,
};

bool api_acquire(uint32_t window, kind k, size_t bytes, lease * out) {
    return global_pool().acquire(window, k, bytes, out);
}

bool api_acquire_nowait(uint32_t window, kind k, size_t bytes, lease * out) {
    return global_pool().acquire(window, k, bytes, out, /*wait_pending =*/ false);
}

int api_upload(const lease * l, const void * src, size_t bytes) {
    if (l == nullptr) {
        return int(submit_result::rejected);
    }
    return int(global_pool().upload(*l, src, bytes));
}

int api_upload_rows(const lease * l, const void * src, size_t src_stride,
                    size_t dst_stride, size_t row_bytes, size_t rows) {
    if (l == nullptr) {
        return int(submit_result::rejected);
    }
    return int(global_pool().upload_rows(*l, src, src_stride, dst_stride, row_bytes, rows));
}

bool api_wait(const lease * l) {
    return l != nullptr && global_pool().wait(*l);
}

void api_release(const lease * l) {
    if (l != nullptr) {
        global_pool().release(*l);
    }
}

bool api_owns(const void * p, size_t bytes) {
    return global_pool().owns(p, bytes);
}

void * api_device_ptr(const lease * l, int rank) {
    return l == nullptr ? nullptr : global_pool().device_ptr(*l, rank);
}

void api_dump() {
    dump();
}

bool api_zc_enabled() {
    return zc_enabled();
}

int api_enqueue_async(const lease * l, ggml_backend_t backend, ggml_tensor * tensor,
                      size_t offset, size_t size) {
    if (l == nullptr) {
        return int(submit_result::rejected);
    }
    return int(enqueue_async_impl(*l, backend, tensor, offset, size));
}

int api_enqueue_sync(const lease * l, ggml_tensor * tensor, size_t offset, size_t size) {
    if (l == nullptr) {
        return int(submit_result::rejected);
    }
    return int(enqueue_sync_impl(*l, tensor, offset, size));
}

int api_first_error() {
    return global_pool().first_error();
}

bool api_enter_submit() {
    return global_pool().enter_submit();
}

void api_leave_submit() {
    global_pool().leave_submit();
}

size_t api_unknown_count() {
    return global_pool().unknown_count();
}

bool api_submission_pending() {
    return global_pool().submission_pending();
}

void api_mark_partial(const lease * l, const void * dst) {
    GGML_ASSERT(l != nullptr);
    global_pool().mark_partial(*l, dst);
}

const api kApi = {
    /* .abi_version       = */ kApiVersion,
    /* .enabled           = */ enabled,
    /* .merge_wait        = */ merge_wait_enabled,
    /* .acquire           = */ api_acquire,
    /* .acquire_nowait    = */ api_acquire_nowait,
    /* .upload            = */ api_upload,
    /* .upload_rows       = */ api_upload_rows,
    /* .wait              = */ api_wait,
    /* .release           = */ api_release,
    /* .owns              = */ api_owns,
    /* .device_ptr        = */ api_device_ptr,
    /* .dump              = */ api_dump,
    /* .can_enqueue_async = */ can_enqueue_async_impl,
    /* .enqueue_async     = */ api_enqueue_async,
    /* .can_enqueue_sync  = */ can_enqueue_sync_impl,
    /* .enqueue_sync      = */ api_enqueue_sync,
    /* .first_error       = */ api_first_error,
    /* .enter_submit      = */ api_enter_submit,
    /* .leave_submit      = */ api_leave_submit,
    /* .unknown_count     = */ api_unknown_count,
    /* .submission_pending= */ api_submission_pending,
    /* .configured        = */ configured,
    /* .usable            = */ usable,
    /* .zc_enabled        = */ api_zc_enabled,
    /* .mark_partial      = */ api_mark_partial,
    /* .kind_count       = */ kKindCount,
};

}  // namespace

pool & global_pool() {
    // Never destroyed: HIP may be unloaded before static destructors run.
    static pool * instance = new pool(kHipOps);
    return *instance;
}

bool enabled() {
    return configured() && usable();
}

bool configured() {
    static const bool flag = [] {
        const char * v = std::getenv("GGML_TP5_STAGING");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return flag;
}

bool usable() {
    return global_pool().usable();
}

bool merge_wait_enabled() {
    static const bool flag = [] {
        const char * v = std::getenv("GGML_TP5_STAGING_MERGE_WAIT");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return flag && enabled();
}

bool zc_enabled() {
    static const bool flag = [] {
        const char * v = std::getenv("GGML_TP5_STAGING_ZC");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return flag && enabled();
}

void dump() {
    const char * diag = std::getenv("GGML_TP5_STAGING_DIAG");
    if (diag == nullptr || std::strcmp(diag, "1") != 0) {
        return;
    }
    global_pool().dump(stderr);
    ggml_backend_meta_staging_diag_dump();
}

void init() {
    static std::atomic<bool> done{false};
    if (done.exchange(true, std::memory_order_relaxed)) {
        return;
    }
    // A rejected table does not replace an existing registration; if none is
    // registered, staging stays unregistered and the original copy path is
    // used. The provider keeps running and does not receive a failure state.
    (void) ggml_backend_meta_staging_register(&kApi);
    static const bool diag = [] {
        const char * v = std::getenv("GGML_TP5_STAGING_DIAG");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    if (diag) {
        std::atexit(dump);
    }
}

// The pool is never destroyed: HIP may be unloaded before static destructors.
// Events and pinned slots live until process exit, which is bounded by design.

}  // namespace ggml_cuda_staging
