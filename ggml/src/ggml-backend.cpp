// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-backend-module.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"
#include "ggml-backend-moe-cache.h"

#include <algorithm>
#include <functional>
#include <new>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <unordered_set>

// MoE expert cache providers, keyed by backend registration object. Multiple
// backends (CUDA, Metal, Vulkan, HIP) can register; each scheduler picks the
// provider that owns a device in its backend set.
static std::mutex g_moe_cache_registry_mu;
static std::vector<ggml_moe_cache_api> g_moe_cache_providers;

void ggml_moe_cache_register(const ggml_moe_cache_api * api) {
    if (!api || !api->owner || !api->session_create || !api->session_destroy ||
        !api->session_enter || !api->session_leave) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_moe_cache_registry_mu);
    for (ggml_moe_cache_api & existing : g_moe_cache_providers) {
        if (existing.owner == api->owner) {
            existing = *api; // refresh: a rebuilt library re-registers
            return;
        }
    }
    g_moe_cache_providers.push_back(*api);
}

void ggml_moe_cache_unregister(const void * owner) {
    std::lock_guard<std::mutex> lock(g_moe_cache_registry_mu);
    g_moe_cache_providers.erase(
        std::remove_if(g_moe_cache_providers.begin(), g_moe_cache_providers.end(),
            [owner](const ggml_moe_cache_api & api) { return api.owner == owner; }),
        g_moe_cache_providers.end());
}

ggml_moe_cache_api ggml_moe_cache_get(const void * owner) {
    std::lock_guard<std::mutex> lock(g_moe_cache_registry_mu);
    for (const ggml_moe_cache_api & api : g_moe_cache_providers) {
        if (api.owner == owner) {
            return api;
        }
    }
    return ggml_moe_cache_api{};
}

// The provider table the current thread is executing under. Set by the
// scheduler scope while a graph with a cache session runs; the CPU backend
// reads it to reach the session's provider callbacks.
static thread_local ggml_moe_cache_api t_moe_cache_active;

ggml_moe_cache_api ggml_moe_cache_active(void) {
    if (t_moe_cache_active.owner) {
        return t_moe_cache_active;
    }
    // Direct CPU users outside a scheduler scope fall back to the first
    // registered provider.
    std::lock_guard<std::mutex> lock(g_moe_cache_registry_mu);
    for (const ggml_moe_cache_api & api : g_moe_cache_providers) {
        if (api.session_create && api.session_destroy) {
            return api;
        }
    }
    return ggml_moe_cache_api{};
}

// Providers whose backend owns at least one device in the scheduler set, in
// registration order. Returns copies: the scheduler keeps the selected copy
// with its session so later registration changes cannot alter the callbacks.
static std::vector<ggml_moe_cache_api> ggml_moe_cache_provider_candidates(
        void * const * backends, int n_backends) {
    std::lock_guard<std::mutex> lock(g_moe_cache_registry_mu);
    std::vector<ggml_moe_cache_api> result;
    for (const ggml_moe_cache_api & api : g_moe_cache_providers) {
        if (!api.session_create || !api.session_destroy ||
            !api.session_enter || !api.session_leave) {
            continue;
        }
        for (int b = 0; b < n_backends; b++) {
            ggml_backend_dev_t device = ggml_backend_get_device((ggml_backend_t) backends[b]);
            if (device && (const void *) ggml_backend_dev_backend_reg(device) == api.owner) {
                result.push_back(api);
                break;
            }
        }
    }
    return result;
}

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <mutex>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));
        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

static void ggml_backend_moe_cache_invalidate_buffer(
        ggml_backend_buffer_t buffer, const void * address, size_t size) {
    if (!buffer || !address || size == 0 ||
        buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
        !ggml_backend_buffer_is_host(buffer)) {
        return;
    }
    // Each registered provider invalidates its own sessions.
    std::vector<ggml_moe_cache_api> providers;
    {
        std::lock_guard<std::mutex> lock(g_moe_cache_registry_mu);
        providers = g_moe_cache_providers;
    }
    for (const ggml_moe_cache_api & api : providers) {
        if (api.invalidate) {
            api.invalidate(address, size);
        }
    }
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    // Host weight buffers can back queued cache-fill jobs.
    if (buffer->iface.get_base) {
        void * base = ggml_backend_buffer_get_base(buffer);
        if (base) {
            ggml_backend_moe_cache_invalidate_buffer(
                    buffer, base, ggml_backend_buffer_get_size(buffer));
        }
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    if (buffer->iface.get_base) {
        ggml_backend_moe_cache_invalidate_buffer(
                buffer, ggml_backend_buffer_get_base(buffer), buffer->size);
    }
    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    if (usage != buffer->usage && buffer->iface.get_base) {
        ggml_backend_moe_cache_invalidate_buffer(
                buffer, ggml_backend_buffer_get_base(buffer), buffer->size);
    }
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        if (buffer->iface.get_base) {
            ggml_backend_moe_cache_invalidate_buffer(
                    buffer, ggml_backend_buffer_get_base(buffer), buffer->size);
        }
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        ggml_backend_moe_cache_invalidate_buffer(
                dst_buf, dst->data, ggml_nbytes(dst));
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

bool ggml_backend_free_and_report(ggml_backend_t backend) {
    if (backend == NULL) {
        return true;
    }

    // A scheduler retained under this backend owns copy slots whose native
    // work may still be in flight. Drain it while the backend is still alive;
    // if it cannot be proven quiescent, keep the backend handle reachable
    // instead of freeing its native object underneath the retained scheduler.
    if (ggml_backend_sched_retained_count(backend) != 0) {
        (void) ggml_backend_sched_drain_retained(backend);
        if (ggml_backend_sched_retained_count(backend) != 0) {
            GGML_LOG_ERROR("%s: backend owner still holds a retained scheduler; not freeing\n", __func__);
            return false;
        }
    }

    if (!ggml_backend_quiesce(backend)) {
        GGML_LOG_ERROR("%s: backend owner could not prove begin/pending quiescence; not freeing\n", __func__);
        return false;
    }

    // The host-owned pin is independent of whether the provider deletes the
    // backend object. Take it before the provider call so a provider that
    // retains the backend (or does not delete it) cannot leave users/module
    // state behind.
    ggml_backend_module::ggml_backend_pin pin;
    const bool pinned = ggml_backend_module::take_backend(backend, &pin);
    if (!pinned && ggml_backend_module::backend_free_in_progress(backend)) {
        // A retry pass owns this backend's single-flight marker. It already
        // took (or will take) the pin; do not dereference or free behind it.
        return true;
    }
    // Save the registry identity before provider free: a retained branch must
    // not dereference a backend the provider may have partially deleted.
    ggml_backend_dev_t saved_device = backend->device;
    const ggml_backend_reg_t saved_reg =
        saved_device != nullptr ? saved_device->reg : nullptr;

    // The provider's free function may delete the backend, which drops the
    // module reference. If that is the last reference, dlclose must not run
    // inside provider code; the shared-handle deleter queues it instead.
    ggml_backend_module::close_defer_scope defer_close;
    bool retained = false;
    try {
        if (ggml_backend_is_meta(backend)) {
            // Meta reports whether completion was proved. A false result means
            // the object was not deleted and must keep its pin. Clear the
            // cached owner first so a retained token can be unbound/released
            // by the cleanup owner; a failed clear keeps the pin.
            const bool cleared = ggml_backend_meta_clear_staging_owner(backend);
            retained = !cleared || !ggml_backend_meta_free_and_report(backend);
        } else {
            backend->iface.free(backend);
            retained = ggml_backend_module::consume_retained(backend);
        }
    } catch (...) {
        // A provider free exception leaves completion unproved. Keep the pin
        // rather than releasing users/module behind a partially freed object.
        retained = true;
    }
    if (pinned) {
        if (retained) {
            ggml_backend_module::put_back(backend, std::move(pin));
        } else {
            ggml_backend_module::release_pin(std::move(pin));
        }
    } else if (retained && saved_reg != nullptr) {
        // Direct-init providers have no pin yet. A retained object still
        // needs a user count and module reference so unload cannot remove its
        // registry entry. Use the pre-free identity, not backend->device.
        ggml_backend_module::ggml_backend_pin late;
        if (ggml_backend_module::hold_user(saved_reg, &late)) {
            ggml_backend_module::put_back(backend, std::move(late));
        } else {
            GGML_LOG_ERROR("%s: cannot pin a retained direct-init backend\n", __func__);
        }
    }
    return !retained;
}

void ggml_backend_free(ggml_backend_t backend) {
    (void) ggml_backend_free_and_report(backend);
}

bool ggml_backend_quiesce(ggml_backend_t backend) {
    if (backend == nullptr) {
        return true;
    }
    if (backend->device != nullptr && backend->device->reg != nullptr) {
        using quiesce_t = bool (*)(ggml_backend_t);
        auto fn = (quiesce_t) ggml_backend_reg_get_proc_address(
                backend->device->reg, "ggml_backend_submission_quiesce");
        if (fn != nullptr) {
            return fn(backend);
        }
    }
    return true;
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        ggml_backend_moe_cache_invalidate_buffer(
                buf, (const char *) tensor->data + offset, size);
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    ggml_backend_moe_cache_invalidate_buffer(
            buf, (const char *) tensor->data + offset,
            (n_copies - 1)*stride_tensor + size);
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    ggml_backend_moe_cache_invalidate_buffer(
            buf, (const char *) tensor->data + offset, size);
    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    ggml_backend_moe_cache_invalidate_buffer(
            buf, (const char *) tensor->data + offset,
            (n_copies - 1)*stride_tensor + size);
    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    ggml_backend_moe_cache_invalidate_buffer(
            buf, (const char *) tensor->data + offset, size);
    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);
    for (int i = 0; i < ggml_graph_n_nodes(cgraph); ++i) {
        const ggml_tensor * node = ggml_graph_node(cgraph, i);
        if ((node->flags & GGML_TENSOR_FLAG_ATTN_ORDERED) &&
                !ggml_backend_supports_op(backend, node)) {
            return NULL;
        }
    }

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    // A backend that ignores the explicit mask cannot compute this node.
    for (int i = 0; i < ggml_graph_n_nodes(cgraph); ++i) {
        const ggml_tensor * node = ggml_graph_node(cgraph, i);
        if ((node->flags & GGML_TENSOR_FLAG_ATTN_ORDERED) &&
                !ggml_backend_supports_op(backend, node)) {
            return GGML_STATUS_FAILED;
        }
    }
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

// Optional backend capability negotiated through the existing reg proc-address
// hook. Backends without it report no error and no pending submission.
typedef int  (*ggml_backend_submission_error_t)(ggml_backend_t);
typedef bool (*ggml_backend_submission_pending_t)(ggml_backend_t);

static int ggml_backend_sched_submission_error(ggml_backend_t backend) {
    if (backend == nullptr || backend->device == nullptr || backend->device->reg == nullptr) {
        return 0;
    }
    auto fn = (ggml_backend_submission_error_t)
        ggml_backend_reg_get_proc_address(backend->device->reg, "ggml_backend_submission_error");
    return fn != nullptr ? fn(backend) : 0;
}

static bool ggml_backend_sched_submission_pending(ggml_backend_t backend) {
    if (backend == nullptr || backend->device == nullptr || backend->device->reg == nullptr) {
        return false;
    }
    auto fn = (ggml_backend_submission_pending_t)
        ggml_backend_reg_get_proc_address(backend->device->reg, "ggml_backend_submission_pending");
    return fn != nullptr ? fn(backend) : false;
}

// Either capability answer means the submission is unproven: a native error
// was recorded, or the destination may still be in flight (a pool can fail
// before it records the error).
static bool ggml_backend_submission_unproven(ggml_backend_t backend) {
    return ggml_backend_sched_submission_error(backend) != 0 ||
           ggml_backend_sched_submission_pending(backend);
}

static const ggml_backend_completion_api * ggml_backend_sched_completion_api(
        ggml_backend_t backend) {
    if (backend == nullptr || backend->device == nullptr ||
            backend->device->reg == nullptr) {
        return nullptr;
    }
    auto * api = (const ggml_backend_completion_api *)
        ggml_backend_reg_get_proc_address(backend->device->reg,
                                          "ggml_backend_completion_v1");
    if (api == nullptr || api->abi_version != GGML_BACKEND_COMPLETION_API_VERSION ||
            api->attempt == nullptr || api->wait == nullptr) {
        return nullptr;
    }
    return api;
}

// A full lifetime view is either an explicit pending_full proc or the
// versioned completion API. The latter names the exact attempts
// (begin/end/wait/drain/release), so a backend that publishes it can be
// drained without a separate pending_full proc.
static bool ggml_backend_sched_has_full_pending_view(ggml_backend_t backend) {
    if (backend == nullptr || backend->device == nullptr || backend->device->reg == nullptr) {
        return false;
    }
    if (ggml_backend_reg_get_proc_address(
            backend->device->reg, "ggml_backend_submission_pending_full") != nullptr) {
        return true;
    }
    return ggml_backend_sched_completion_api(backend) != nullptr;
}

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    // Empty copies are a no-op. Do not reach the fallback's malloc(0), whose
    // result is implementation-defined and may be nullptr.
    if (ggml_nbytes(src) == 0) {
        return;
    }

    ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    // A CPU Meta buffer reports is_host=true when every child is host, but the
    // top-level tensor data is a virtual sentinel. Only the source sentinel
    // matters for the source host fast path, and only the destination sentinel
    // matters for the destination host fast path. A Meta buffer on the other
    // side is safe because that fast path calls the Meta get/set hook, not the
    // sentinel pointer.
    const bool src_meta = ggml_backend_buffer_is_meta(src_buf);
    const bool dst_meta = ggml_backend_buffer_is_meta(dst_buf);
    // A Meta destination needs the source tensor identity for staging slot
    // selection. cpy_tensor is the only buffer hook that receives src; the
    // blocking set path below sees only the scheduler's decorated copy. A
    // false return is a pure preflight rejection, so the original path still
    // applies.
    if (ggml_backend_buffer_is_host(src_buf) && ggml_backend_buffer_is_meta(dst_buf) &&
            ggml_backend_buffer_copy_tensor(src, dst)) {
        return;
    }
    if (ggml_backend_buffer_is_host(src_buf) && !src_meta) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst_buf) && !dst_meta) {
        ggml_backend_moe_cache_invalidate_buffer(
                dst_buf, dst->data, ggml_nbytes(dst));
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        if (data == nullptr) {
            GGML_ABORT("%s: failed to allocate %zu bytes for the copy fallback\n", __func__, nbytes);
        }
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

bool ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    GGML_ASSERT(backend_dst);
    // Do not submit more work to an owner that already reported an unknown
    // submission: the blocking fallback below must never overwrite a
    // destination that an in-flight copy may still write. Return false without
    // touching the destination or either backend's stream. This also applies to
    // a self-copy: true is not a completion proof for an earlier submission.
    if (ggml_backend_submission_unproven(backend_src) ||
            ggml_backend_submission_unproven(backend_dst)) {
        return false;
    }

    if (src == dst) {
        return true;
    }

    if (backend_dst->iface.cpy_tensor_async != NULL) {
        ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
        ggml_backend_moe_cache_invalidate_buffer(
                dst_buf, dst->data, ggml_nbytes(dst));
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            // Accepted. Re-query both backends: the callback may have published
            // the fault, and either operand can make completion unprovable.
            // Report failure instead of replaying the destination.
            return !ggml_backend_submission_unproven(backend_src) &&
                   !ggml_backend_submission_unproven(backend_dst);
        }
        // Rejected with no side effects: the blocking copy below is safe.
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
    // A fault published while the blocking copy ran is not a completion proof.
    return !ggml_backend_submission_unproven(backend_src) &&
           !ggml_backend_submission_unproven(backend_dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    const ggml_backend_reg_t reg = device->reg;
    ggml_backend_module::ggml_backend_pin pin;
    if (!ggml_backend_module::hold_user(reg, &pin)) {
        // Unknown/unloading/unloaded regs are not usable. Never fall through
        // to init_backend without a pin.
        return nullptr;
    }
    ggml_backend_t backend = nullptr;
    try {
        backend = device->iface.init_backend(device, params);
    } catch (...) {
        ggml_backend_module::release_pin(std::move(pin));
        return nullptr;
    }
    if (backend == nullptr) {
        ggml_backend_module::release_pin(std::move(pin));
        return nullptr;
    }
    bool attached = false;
    try {
        attached = ggml_backend_module::attach_backend(backend, &pin);
    } catch (...) {
        attached = false;
    }
    if (!attached) {
        // The side table allocation failed. Free the backend and release the
        // user slot; never publish an unpinned backend.
        ggml_backend_free(backend);
        ggml_backend_module::release_pin(std::move(pin));
        return nullptr;
    }
    return backend;
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (op->flags & GGML_TENSOR_FLAG_ATTN_ORDERED) {
        // A backend opts in with bool(device, op), including its layout checks.
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        typedef bool (*ordered_attention_support_t)(ggml_backend_dev_t, const struct ggml_tensor *);
        auto support = reg ? (ordered_attention_support_t) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_supports_ordered_attention") : nullptr;
        return support && support(device, op);
    }
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
// 30 upstream. Gemma 4 E-series feeds a per-layer embedding input into every
// block; with partial offload (small-VRAM Vulkan + CPU) each of them becomes
// a split input, so a 30+-layer model trips the assert at split creation.
// 128 covers the deepest current target (31B dense) with headroom.
#define GGML_SCHED_MAX_SPLIT_INPUTS 128
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_sched_copy_record {
    ggml_tensor * source;
    ggml_tensor * copy;
    ggml_tensor * dependency;
    size_t bytes;
    size_t alloc_size;
    ggml_backend_buffer_t buffer;
    void * data;
    enum ggml_type type;
    int backend_id;
    int copy_id;
    bool valid;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;
    // Nonzero after an asynchronous submission whose completion is unprovable.
    // The destination copy slots stay owned and no further graph is submitted.
    int submission_failed = 0;
    // Cached result of a submission_pending query. Set while the backends are
    // alive, so ggml_backend_sched_free never has to dereference them (the
    // caller may destroy the backends before the scheduler).
    bool submission_unproven = false;
    // Cached at creation: the backend set has an async entry point without a
    // synchronize hook and without a full pending view. sched_free must not
    // probe the backends (they may already be freed), so this capability is
    // recorded while they are alive and used as a retention reason.
    bool async_no_barrier = false;
    // Monotonic id for explicit completion receipts. Normal graph_compute
    // calls do not use it.
    uint64_t receipt_attempt = 0;
    // Reachable process-level owner for a scheduler that sched_free could
    // not release. owner is the context or backend that will retry the
    // drain; the retained list keeps the scheduler reachable.
    void * owner = nullptr;
    struct ggml_backend_sched * next_retained_sched = nullptr;
    bool retained_registered = false;
    bool draining = false;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]
    struct ggml_sched_copy_record * copy_records;
    size_t n_copy_records;
    size_t copy_records_capacity;
    bool copy_tracking;
    bool copy_envelopes_valid;

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;

    struct ggml_context * ctx;
    void * moe_cache_session;
    // The provider table selected for this session. Copied at creation so a
    // later registration change cannot alter the session's callbacks.
    struct ggml_moe_cache_api moe_cache_api;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

struct ggml_backend_sched_receipt_resource {
    ggml_backend_t        backend;
    uint64_t              attempt;
    ggml_backend_event_t  event;
    bool                  event_only;
    bool                  drain_only;
};

struct ggml_backend_sched_receipt_impl {
    uint64_t attempt;
    int      state;
    int      error;
    bool     submitted;
    bool     failed_early;
    bool     in_flight;
    std::vector<ggml_backend_sched_receipt_resource> resources;
};

static void ggml_backend_sched_receipt_cancel(
        ggml_backend_sched_receipt_resource * res) {
    if (res->event_only) {
        ggml_backend_event_free(res->event);
        return;
    }
    const ggml_backend_completion_api * api = ggml_backend_sched_completion_api(res->backend);
    if (api != nullptr) {
        if (api->end != nullptr) api->end(res->backend);
        if (api->release != nullptr) api->release(res->backend, res->attempt);
    }
}

static bool ggml_backend_sched_receipt_reserve(
        ggml_backend_sched_receipt_impl * impl, ggml_backend_t backend,
        ggml_backend_sched_receipt_resource * out) {
    *out = {};
    out->backend = backend;
    const ggml_backend_completion_api * api = ggml_backend_sched_completion_api(backend);
    if (api != nullptr && api->begin != nullptr) {
        const uint64_t token = api->begin(backend);
        if (token != 0) {
            out->attempt = token;
            return true;
        }
    }
    if (backend->device != nullptr && backend->device->iface.event_new != nullptr &&
            backend->iface.event_record != nullptr) {
        ggml_backend_event_t event = ggml_backend_event_new(backend->device);
        if (event != nullptr) {
            out->event = event;
            out->event_only = true;
            return true;
        }
    }
    impl->state = GGML_BACKEND_SCHED_RECEIPT_UNKNOWN;
    impl->in_flight = true;
    impl->error = -1;
    return false;
}

static void ggml_backend_sched_receipt_finish(
        ggml_backend_sched_receipt_impl * impl, ggml_backend_sched_receipt_resource * res) {
    const ggml_backend_completion_api * api = ggml_backend_sched_completion_api(res->backend);
    if (!res->event_only) {
        if (api != nullptr && api->end != nullptr) {
            api->end(res->backend);
        }
    } else {
        ggml_backend_event_record(res->event, res->backend);
    }
    try {
        impl->resources.push_back(*res);
    } catch (...) {
        if (res->event_only) {
            ggml_backend_event_free(res->event);
        }
        impl->state = GGML_BACKEND_SCHED_RECEIPT_UNKNOWN;
        impl->in_flight = true;
        impl->error = -1;
    }
}

static bool ggml_backend_sched_any_pending(ggml_backend_sched_t sched);
static bool ggml_backend_sched_has_async_no_barrier(ggml_backend_sched_t sched);

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static void ggml_backend_sched_split_inputs_grow(struct ggml_backend_sched_split * split) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_WARN("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) split->inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

static void ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_WARN("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) sched->graph_inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
            }
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;
    sched->copy_envelopes_valid = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    // Preserve a caller-assigned uid: it identifies one graph definition
    // across re-splits. A zero uid keeps the historical fresh-per-split id.
    if (graph->uid == 0) {
        graph->uid = ggml_graph_next_uid();
    }

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split->n_inputs >= split->inputs_capacity) {
                        const size_t id = hash_id(src);
                        int src_backend_id = sched->hv_tensor_backend_ids[id];
                        bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                        if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    int old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                    for (int k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            ggml_set_input(tensor_copy);
                            ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_graph_inputs = sched->n_graph_inputs++;
                        if (n_graph_inputs >= sched->graph_inputs_capacity) {
                            ggml_backend_sched_graph_inputs_grow(sched);
                        }
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    int total_inputs = sched->n_graph_inputs;
    for (int i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // Optimize this split of the graph. This needs to happen before we make graph_copy,
        // so they are in sync.
        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits. A single split is the whole graph, so it keeps
    // the caller's uid and backend caches can key on the definition.
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = (sched->n_splits == 1 && graph->uid != 0)
            ? graph->uid : ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    if (ggml_gallocr_is_failed(sched->galloc)) {
        return false;
    }
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

// Fold one backend's optional submission state into the scheduler's cached
// fail-stop state. True means the submission is unproven: the backend either
// reported a native error or still reports outstanding work. Pending without
// an error is a real state (a disabled staging pool can have no recorded
// native error), so both answers must be consulted.
static bool ggml_backend_sched_note_submission_state(
        ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    const int err = ggml_backend_sched_submission_error(backend);
    if (err != 0) {
        if (sched->submission_failed == 0) {
            sched->submission_failed = err;
        }
        sched->submission_unproven = true;
        return true;
    }
    if (ggml_backend_sched_submission_pending(backend)) {
        sched->submission_unproven = true;
        return true;
    }
    return false;
}

static enum ggml_status ggml_backend_sched_compute_splits(
        ggml_backend_sched_t sched, bool retained_inputs,
        ggml_backend_sched_receipt_impl * receipt) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    struct moe_cache_scope {
        void * session;
        void (*leave)(void *);
        ggml_moe_cache_api previous_active;

        explicit moe_cache_scope(void * session, void (*enter)(void *),
                                 void (*leave_fn)(void *),
                                 const ggml_moe_cache_api & active_api)
            : session(session), leave(leave_fn), previous_active(t_moe_cache_active) {
            t_moe_cache_active = active_api;
            if (enter && leave) {
                enter(session);
            } else {
                leave = nullptr;
            }
        }

        ~moe_cache_scope() {
            t_moe_cache_active = previous_active;
            if (leave) {
                leave(session);
            }
        }
    } cache_scope(sched->moe_cache_session,
                  sched->moe_cache_api.session_enter,
                  sched->moe_cache_api.session_leave,
                  sched->moe_cache_api);

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;

    struct input_copy_scope {
        ggml_backend_t backend;
        bool pending;
        // Set when the submission is unprovable. The destructor must then
        // skip the flush: synchronizing would contradict the no-drain
        // contract and could make the caller reuse the source.
        bool retain;

        void flush() {
            if (pending && !retain) {
                ggml_backend_synchronize(backend);
                pending = false;
            }
        }

        ~input_copy_scope() {
            flush();
        }
    };

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        const int64_t d0=ggml_time_us();
        input_copy_scope input_copies = { split_backend, false, false };
        bool batch_inputs = split->n_inputs > 0 && split_backend->iface.cpy_tensor_async != nullptr;
        bool has_user_input = false;
        for (int input_id = 0; batch_inputs && input_id < split->n_inputs; input_id++) {
            const ggml_tensor * input = split->inputs[input_id];
            ggml_backend_buffer_t buffer = input->view_src ? input->view_src->buffer : input->buffer;
            const bool empty = ggml_nbytes(input) == 0;
            has_user_input |= !empty && (input->flags & GGML_TENSOR_FLAG_INPUT) != 0;
            batch_inputs = empty ||
                    (buffer && ggml_backend_buffer_is_host(buffer) && ggml_is_contiguous(input) &&
                     ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        batch_inputs &= has_user_input;
        bool batch_slot_waited = false;

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            // Empty views neither read source bytes nor overwrite a copy slot.
            if (ggml_nbytes(split->inputs[input_id]) == 0) {
                continue;
            }
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            if ((input->flags & GGML_TENSOR_FLAG_INPUT) || batch_inputs) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                // A host-input batch waits for the previous copy slot once, then completes all uploads before dispatch.
                if (!batch_inputs || !batch_slot_waited) {
                    if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                        ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                    } else {
                        ggml_backend_synchronize(split_backend);
                    }
                    batch_slot_waited = true;
                }
                if (batch_inputs) {
                    GGML_ASSERT(ggml_are_same_layout(input, input_cpy) && "cannot copy tensors with different layouts");
                    const size_t nbytes = ggml_nbytes(input);
                    GGML_ASSERT((nbytes == 0 || input_cpy->data != NULL) && "tensor not allocated");
                    GGML_ASSERT(nbytes <= ggml_nbytes(input_cpy) && "tensor write out of bounds");
                    ggml_backend_buffer_t buffer = input_cpy->view_src ? input_cpy->view_src->buffer : input_cpy->buffer;
                    GGML_ASSERT(buffer != NULL && "tensor buffer not set");
                    ggml_backend_moe_cache_invalidate_buffer(buffer, input_cpy->data, nbytes);

                    // A callback can throw after queuing part of an upload. Keep caller memory alive until it drains.
                    const bool was_pending = input_copies.pending;
                    input_copies.pending = true;
                    if (split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        if (ggml_backend_sched_note_submission_state(sched, split_backend)) {
                            // Unknown submission: keep the destination and
                            // stop. Do not flush or replay the copy.
                            input_copies.retain = true;
                            return GGML_STATUS_FAILED;
                        }
                        continue;
                    }
                    // Rejection has no side effects, but earlier accepted uploads must finish before a blocking copy.
                    input_copies.pending = was_pending;
                    input_copies.flush();
                    batch_inputs = false;
                }
                ggml_backend_tensor_copy(input, input_cpy);
            } else {
                // wait for the split backend to finish using the input before overwriting it
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = split->graph.nodes[0];
                if (split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
                    )) {

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    ggml_backend_synchronize(input_backend);

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = input_id + 1; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    if (ids_tensor != prev_ids_tensor) {
                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                        ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                        ggml_backend_synchronize(ids_backend);

                        // find the used experts
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                GGML_ASSERT(id >= 0 && id < n_expert);
                                ggml_bitset_set(used_ids.data(), id);
                            }
                        }

                        prev_ids_tensor = ids_tensor;
                    }

                    // group consecutive experts and copy them together
                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        ggml_backend_tensor_set_async(split_backend,
                            input_cpy,
                            (const uint8_t *)input->data + expert_offset, expert_offset,
                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                            // this is necessary for MMQ in the CUDA backend
                            expert_size_copy + padding_end);
                    };

                    int id = 0;
                    while (!ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    int32_t first_id = id;
                    int32_t last_id = first_id;

                    for (++id; id < n_expert; ++id) {
                        if (!ggml_bitset_get(used_ids.data(), id)) {
                            continue;
                        }

                        if (id == last_id + 1) {
                            last_id = id;
                            continue;
                        }

                        copy_experts(first_id, last_id);

                        first_id = id;
                        last_id = id;
                    }
                    copy_experts(first_id, last_id);
                } else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    ggml_backend_sched_receipt_resource copy_res = {};
                    const bool async_copy_available = split_backend->iface.cpy_tensor_async != nullptr;
                    if (receipt != nullptr && async_copy_available) {
                        if (!ggml_backend_sched_receipt_reserve(receipt, split_backend, &copy_res)) {
                            return GGML_STATUS_FAILED;
                        }
                        receipt->submitted = true;
                    }
                    const bool copied = async_copy_available &&
                        split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy);
                    if (receipt != nullptr && async_copy_available) {
                        if (copied) {
                            ggml_backend_sched_receipt_finish(receipt, &copy_res);
                        } else {
                            ggml_backend_sched_receipt_cancel(&copy_res);
                        }
                    }
                    if (copied) {
                        if (ggml_backend_sched_note_submission_state(sched, split_backend)) {
                            input_copies.retain = true;
                            return GGML_STATUS_FAILED;
                        }
                    }
                    if (!copied) {
                        ggml_backend_synchronize(input_backend);
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                    }
                }
            }
        }

        // The default API preserves immediate host-input reuse. The explicit
        // retained-input API can leave uploads ordered before compute on each
        // backend stream; a rejected/partial batch keeps the original contract.
        const int64_t dcopied=ggml_time_us();
        const bool defer_input_flush = retained_inputs && sched->n_splits == 1 &&
                sched->callback_eval == nullptr && batch_inputs;
        if (!defer_input_flush) {
            input_copies.flush();
        }

        const int64_t dflushed=ggml_time_us();
        if (!sched->callback_eval) {
            ggml_backend_sched_receipt_resource compute_res = {};
            if (receipt != nullptr) {
                if (!ggml_backend_sched_receipt_reserve(receipt, split_backend, &compute_res)) {
                    return GGML_STATUS_FAILED;
                }
                receipt->submitted = true;
            }
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (receipt != nullptr) {
                ggml_backend_sched_receipt_finish(receipt, &compute_res);
            }
            if (ec != GGML_STATUS_SUCCESS) {
                // A compute failure may be the first observation of a staging
                // fault that happened outside the scheduler copy path. Query
                // every backend, not only the failed split: the capability is
                // process-global for the Meta owner. Turn any answer into the
                // scheduler's fail-stop state so reset/reserve/alloc/release
                // cannot free the owners.
                if (ggml_backend_sched_any_pending(sched)) {
                    // The compute failure revealed an unknown submission. The
                    // scope must not flush it on the return path.
                    input_copies.retain = true;
                }
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                ggml_backend_sched_receipt_resource compute_res = {};
                if (receipt != nullptr) {
                    if (!ggml_backend_sched_receipt_reserve(receipt, split_backend, &compute_res)) {
                        return GGML_STATUS_FAILED;
                    }
                    receipt->submitted = true;
                }
                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (receipt != nullptr) {
                    ggml_backend_sched_receipt_finish(receipt, &compute_res);
                }
                if (ec != GGML_STATUS_SUCCESS) {
                    if (ggml_backend_sched_any_pending(sched)) {
                        input_copies.retain = true;
                    }
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        const int64_t dcomputed=ggml_time_us();
        // record the event of this copy
        if (split->n_inputs > 0) {
            if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
            }
        }
        if (defer_input_flush) {
            // Transfer source lifetime only after successful graph submission.
            // Every earlier return/exception still drains through the scope.
            input_copies.pending = false;
        }
        if (std::getenv("TP5_DIAG_GRAPH")) GGML_LOG_INFO("TP5_DIAG_SCHED backend=%s split=%d inputs=%d batched=%d retained=%d defer=%d copy_us=%lld flush_us=%lld compute_us=%lld event_us=%lld\n",ggml_backend_name(split_backend),split_id,split->n_inputs,int(batch_inputs),int(retained_inputs),int(defer_input_flush),(long long)(dcopied-d0),(long long)(dflushed-dcopied),(long long)(dcomputed-dflushed),(long long)(ggml_time_us()-dcomputed));
    }

    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (struct ggml_tensor **) calloc(sched->graph_inputs_capacity, sizeof(struct ggml_tensor *));

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    // Record the async-without-barrier capability while the backends are
    // alive; sched_free uses the cached flag and never dereferences them.
    sched->async_no_barrier = ggml_backend_sched_has_async_no_barrier(sched);

    {
        void * cache_backends[GGML_SCHED_MAX_BACKENDS];
        for (int b = 0; b < n_backends; b++) {
            cache_backends[b] = backends[b];
        }
        for (const ggml_moe_cache_api & api :
                ggml_moe_cache_provider_candidates(cache_backends, n_backends)) {
            void * session = api.session_create(cache_backends, n_backends, nullptr);
            if (session) {
                sched->moe_cache_session = session;
                sched->moe_cache_api = api;
                break;
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

enum ggml_moe_cache_set_result ggml_backend_sched_set_moe_cache(
        ggml_backend_sched_t sched, enum ggml_moe_cache_mode mode, size_t budget_mib) {
    GGML_ASSERT(sched);
    if (mode == GGML_MOE_CACHE_MODE_UNSPECIFIED) {
        return GGML_MOE_CACHE_SET_APPLIED;
    }
    if (ggml_backend_sched_any_pending(sched)) {
        // Destroying or replacing the session would free provider resources
        // that an unproven submission may still use. Keep the current session.
        GGML_LOG_ERROR("%s: refusing to change the MoE cache session: an async submission is still unproven\n", __func__);
        return GGML_MOE_CACHE_SET_REFUSED;
    }

    if (sched->moe_cache_session && sched->moe_cache_api.session_destroy) {
        sched->moe_cache_api.session_destroy(sched->moe_cache_session);
    }
    sched->moe_cache_session = nullptr;
    sched->moe_cache_api = ggml_moe_cache_api{};
    if (mode == GGML_MOE_CACHE_MODE_OFF) {
        return GGML_MOE_CACHE_SET_APPLIED;
    }

    void * cache_backends[GGML_SCHED_MAX_BACKENDS];
    for (int index = 0; index < sched->n_backends; index++) {
        cache_backends[index] = sched->backends[index];
    }

    // Probe the providers against this scheduler's backend set and pick the
    // first one that can create a session, retrying another provider when
    // session_create returns null.
    const int automatic = mode == GGML_MOE_CACHE_MODE_AUTO ? 1 : 0;
    bool provider_tried = false;
    for (const ggml_moe_cache_api & api :
            ggml_moe_cache_provider_candidates(cache_backends, sched->n_backends)) {
        ggml_moe_cache_config config = {};
        if (!api.query_config || !api.query_config(automatic, budget_mib, &config)) {
            continue;
        }
        provider_tried = true;
        void * session = api.session_create(cache_backends, sched->n_backends, &config);
        if (session) {
            sched->moe_cache_session = session;
            sched->moe_cache_api = api;
            break;
        }
    }
    if (provider_tried && !sched->moe_cache_session) {
        GGML_LOG_INFO("[moe-cache] session creation failed: all candidate providers returned null\n");
    }
    // No session is a dormant request, not a fallback to the environment
    // default: the default session was destroyed above and is not recreated.
    return sched->moe_cache_session != nullptr ? GGML_MOE_CACHE_SET_APPLIED : GGML_MOE_CACHE_SET_DORMANT;
}

static bool ggml_backend_sched_any_pending(ggml_backend_sched_t sched) {
    if (sched == nullptr) {
        return false;
    }
    if (sched->submission_failed != 0) {
        sched->submission_unproven = true;
        return true;
    }
    if (sched->submission_unproven) {
        return true;
    }
    for (int b = 0; b < sched->n_backends; ++b) {
        const int err = ggml_backend_sched_submission_error(sched->backends[b]);
        if (err != 0) {
            if (sched->submission_failed == 0) {
                sched->submission_failed = err;
            }
            sched->submission_unproven = true;
            return true;
        }
        if (ggml_backend_sched_submission_pending(sched->backends[b])) {
            sched->submission_unproven = true;
            return true;
        }
    }
    return false;
}

bool ggml_backend_sched_has_unproven(ggml_backend_sched_t sched) {
    return ggml_backend_sched_any_pending(sched);
}

void ggml_backend_sched_mark_unproven(ggml_backend_sched_t sched) {
    if (sched != nullptr) {
        sched->submission_unproven = true;
    }
}

static bool ggml_backend_sched_has_standard_pending_view(ggml_backend_t backend) {
    if (backend == nullptr || backend->device == nullptr || backend->device->reg == nullptr) {
        return false;
    }
    return ggml_backend_reg_get_proc_address(
            backend->device->reg, "ggml_backend_submission_pending") != nullptr;
}

static bool ggml_backend_sched_has_async_no_barrier(ggml_backend_sched_t sched) {
    for (int i = 0; i < sched->n_backends; ++i) {
        ggml_backend_t backend = sched->backends[i];
        const bool async_iface = backend != nullptr &&
                (backend->iface.cpy_tensor_async != nullptr ||
                 backend->iface.set_tensor_async != nullptr ||
                 backend->iface.get_tensor_async != nullptr ||
                 ggml_backend_sched_has_standard_pending_view(backend));
        if (async_iface && backend->iface.synchronize == nullptr &&
                !ggml_backend_sched_has_full_pending_view(backend)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_sched_has_no_full_barrier(ggml_backend_sched_t sched) {
    for (int i = 0; i < sched->n_backends; ++i) {
        ggml_backend_t backend = sched->backends[i];
        if (backend != nullptr && backend->iface.synchronize != nullptr &&
                !ggml_backend_sched_has_full_pending_view(backend)) {
            return true;
        }
    }
    return false;
}

static bool ggml_backend_sched_has_no_full_async(ggml_backend_sched_t sched) {
    for (int i = 0; i < sched->n_backends; ++i) {
        ggml_backend_t backend = sched->backends[i];
        const bool async_iface = backend != nullptr &&
                (backend->iface.synchronize != nullptr ||
                 backend->iface.cpy_tensor_async != nullptr ||
                 backend->iface.set_tensor_async != nullptr ||
                 backend->iface.get_tensor_async != nullptr ||
                 ggml_backend_sched_has_standard_pending_view(backend));
        if (async_iface && !ggml_backend_sched_has_full_pending_view(backend)) {
            return true;
        }
    }
    return false;
}

static std::mutex & ggml_backend_sched_retired_mutex() {
    static std::mutex mutex;
    return mutex;
}

static ggml_backend_sched *& ggml_backend_sched_retired_head() {
    static ggml_backend_sched * head = nullptr;
    return head;
}

static void ggml_backend_sched_retire(ggml_backend_sched_t sched) {
    std::lock_guard<std::mutex> lock(ggml_backend_sched_retired_mutex());
    if (sched->retained_registered) {
        return;
    }
    sched->next_retained_sched = ggml_backend_sched_retired_head();
    ggml_backend_sched_retired_head() = sched;
    sched->retained_registered = true;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    if (sched->retained_registered || sched->draining) {
        GGML_LOG_ERROR("%s: scheduler is retained/draining; drain_retained must release it\n", __func__);
        return;
    }
    // Use cached state only. The caller may destroy the backends before the
    // scheduler (llama_context does), so probing them here would be a
    // use-after-free on a normal shutdown.
    const bool unproven = sched->submission_failed != 0 || sched->submission_unproven ||
                          sched->async_no_barrier;
    if (unproven) {
        if (sched->owner == nullptr && sched->n_backends > 0) {
            // The first backend is the caller's teardown authority; its
            // ggml_backend_free drains this scheduler before the native free.
            sched->owner = sched->backends[0];
        }
        GGML_LOG_ERROR("%s: retaining scheduler, an async submission or receipt is still unproven\n", __func__);
        ggml_backend_sched_retire(sched);
        return;
    }
    if (sched->moe_cache_session && sched->moe_cache_api.session_destroy) {
        sched->moe_cache_api.session_destroy(sched->moe_cache_session);
        sched->moe_cache_session = NULL;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    for (int i = 0; i < sched->splits_capacity; i++) {
        free(sched->splits[i].inputs);
    }
    free(sched->splits);
    free(sched->graph_inputs);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->copy_records);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

void ggml_backend_sched_set_owner(ggml_backend_sched_t sched, void * owner) {
    if (sched != nullptr) {
        sched->owner = owner;
    }
}

int ggml_backend_sched_drain_retained(void * owner) {
    int freed = 0;
    for (;;) {
        ggml_backend_sched * sched = nullptr;
        {
            std::lock_guard<std::mutex> lock(ggml_backend_sched_retired_mutex());
            for (ggml_backend_sched * it = ggml_backend_sched_retired_head(); it != nullptr;
                    it = it->next_retained_sched) {
                if (it->owner == owner && !it->draining) {
                    it->draining = true;
                    sched = it;
                    break;
                }
            }
        }
        if (sched == nullptr) {
            break;
        }
        bool drained = false;
        if (ggml_backend_sched_has_async_no_barrier(sched)) {
            // Async entry points without a synchronize hook: no barrier exists,
            // so the copy slots cannot be proven quiescent.
            GGML_LOG_ERROR("%s: scheduler has async work without a synchronize barrier; keeping it retained\n",
                    __func__);
        } else if (ggml_backend_sched_has_no_full_async(sched) && sched->submission_failed != 0) {
            // A recorded native failure has no success receipt to clear it, and
            // a synchronize barrier is a quiescence proof, not a success proof.
            GGML_LOG_ERROR("%s: scheduler has a failed submission without a full pending view; keeping it retained\n",
                    __func__);
        } else {
            if (ggml_backend_sched_has_no_full_barrier(sched)) {
                for (int b = 0; b < sched->n_backends; ++b) {
                    ggml_backend_t backend = sched->backends[b];
                    if (backend != nullptr && backend->iface.synchronize != nullptr &&
                            !ggml_backend_sched_has_full_pending_view(backend)) {
                        ggml_backend_synchronize(backend);
                    }
                }
            }
            int error = 0;
            drained = ggml_backend_sched_drain_pending(sched, &error);
            if (drained && sched->submission_failed == 0) {
                // A successful drain proves the backends report no pending or
                // error submission; the cached flag was the only remaining
                // unproven state. A recorded native failure stays fail-stop.
                sched->submission_unproven = false;
            }
        }
        if (!drained) {
            std::lock_guard<std::mutex> lock(ggml_backend_sched_retired_mutex());
            sched->draining = false;
            break;
        }
        if (sched->submission_failed != 0 || sched->submission_unproven) {
            std::lock_guard<std::mutex> lock(ggml_backend_sched_retired_mutex());
            sched->draining = false;
            break;
        }
        bool removed = false;
        {
            std::lock_guard<std::mutex> lock(ggml_backend_sched_retired_mutex());
            ggml_backend_sched ** link = &ggml_backend_sched_retired_head();
            while (*link != nullptr) {
                if (*link == sched) {
                    *link = sched->next_retained_sched;
                    sched->next_retained_sched = nullptr;
                    sched->retained_registered = false;
                    sched->draining = false;
                    removed = true;
                    break;
                }
                link = &(*link)->next_retained_sched;
            }
        }
        if (!removed) {
            GGML_LOG_ERROR("%s: drained scheduler disappeared from owner list\n", __func__);
            break;
        }
        ggml_backend_sched_free(sched);
        ++freed;
    }
    return freed;
}

size_t ggml_backend_sched_retained_count(void * owner) {
    std::lock_guard<std::mutex> lock(ggml_backend_sched_retired_mutex());
    size_t n = 0;
    for (ggml_backend_sched * it = ggml_backend_sched_retired_head(); it != nullptr;
            it = it->next_retained_sched) {
        if (it->owner == owner) {
            ++n;
        }
    }
    return n;
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    if (ggml_backend_sched_any_pending(sched)) {
        // Keep every copy slot and its destination: an unknown submission may
        // still read them.
        return;
    }
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
    sched->copy_envelopes_valid = false;
    sched->n_copy_records = 0;
}

void ggml_backend_sched_release_buffers(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    if (ggml_backend_sched_any_pending(sched)) {
        // The gallocr owns the copy-slot destinations. Releasing it here would
        // free them behind an in-flight native copy.
        GGML_LOG_ERROR("%s: refusing to release buffers, an async submission is still unproven\n", __func__);
        return;
    }
    ggml_backend_sched_synchronize(sched);

    // Keep the scheduler and its provider session; only the allocator owns
    // the transient compute buffers whose high-water mark is being released.
    ggml_gallocr_t replacement = ggml_gallocr_new_n(sched->bufts, sched->n_backends);
    ggml_backend_sched_reset(sched);
    ggml_gallocr_free(sched->galloc);
    sched->galloc = replacement;

    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;
    sched->cur_copy = 0;
    sched->next_copy = 0;
    // The next graph is a deliberate cold allocation, not an unexpected
    // same-shape reallocation for GGML_SCHED_DEBUG_REALLOC diagnostics.
    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    if (ggml_backend_sched_any_pending(sched)) {
        return;
    }
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

void ggml_backend_sched_reserve_size_preserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    if (ggml_backend_sched_any_pending(sched)) {
        return;
    }
    GGML_ASSERT(!ggml_gallocr_is_failed(sched->galloc));
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);
    ggml_backend_sched_reset(sched);
    ggml_backend_sched_synchronize(sched);
    ggml_backend_sched_split_graph(sched, measure_graph);
    ggml_gallocr_t measure = ggml_gallocr_new_n(sched->bufts, sched->n_backends);
    ggml_gallocr_reserve_n_size(measure, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
    ggml_gallocr_free(measure);
}

// TP5 max-footprint share: reserve the measure graph while guaranteeing each
// backend buffer's total capacity is at least min_sizes[backend_index]. The
// graph is placed normally; only a shortfall against the caller's measured
// max-footprint envelope is added to the last chunk. This total floor does not
// guarantee that later placements fit without reallocation.
bool ggml_backend_sched_reserve_min(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, const size_t * min_sizes) {
    GGML_ASSERT(sched);
    if (ggml_backend_sched_any_pending(sched)) {
        return false;
    }
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(min_sizes);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    const bool ok = ggml_gallocr_reserve_n_min(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, min_sizes);

    ggml_backend_sched_reset(sched);
    return ok;
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    if (ggml_backend_sched_any_pending(sched)) {
        return false;
    }
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

// Compute a non-overlapping layout span without overflowing ggml_nbytes or
// ggml_nelements. Empty and permuted layouts are not admitted for rebinding.
static bool ggml_sched_layout_span(const ggml_tensor * t, size_t * span) {
    if (t->type < 0 || t->type >= GGML_TYPE_COUNT) {
        return false;
    }
    const size_t block = (size_t) ggml_blck_size(t->type);
    const size_t unit = ggml_type_size(t->type);
    if (block == 0 || unit == 0 || t->nb[0] != unit || t->ne[0] <= 0 ||
            (uint64_t) t->ne[0] % block != 0 || (uint64_t) t->ne[0] > SIZE_MAX / unit) {
        return false;
    }
    size_t row = (size_t) t->ne[0] / block * unit;
    size_t bytes = row;
    uint64_t elements = (uint64_t) t->ne[0];
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] <= 0 || (uint64_t) t->ne[i] > (uint64_t) INT64_MAX / elements || t->nb[i] < row) {
            return false;
        }
        elements *= (uint64_t) t->ne[i];
        const size_t count = (size_t) t->ne[i] - 1;
        if (count && t->nb[i] > (SIZE_MAX - bytes) / count) {
            return false;
        }
        bytes += count * t->nb[i];
        if ((size_t) t->ne[i] > SIZE_MAX / t->nb[i]) {
            return false;
        }
        row = t->nb[i] * (size_t) t->ne[i];
    }
    *span = bytes;
    return true;
}

static void ggml_backend_sched_capture_copy_envelopes(ggml_backend_sched_t sched) {
    sched->copy_envelopes_valid = false;
    sched->n_copy_records = 0;

    size_t count = 0;
    for (int i = 0; i < sched->n_splits; ++i) {
        const int inputs = sched->splits[i].n_inputs;
        if (inputs < 0 || (size_t) inputs > (SIZE_MAX - count) / (size_t) sched->n_copies) {
            return;
        }
        count += (size_t) inputs * (size_t) sched->n_copies;
    }
    const size_t extra_copies = (size_t) sched->n_copies - 1;
    if (extra_copies && (size_t) sched->n_graph_inputs > (SIZE_MAX - count) / extra_copies) {
        return;
    }
    count += (size_t) sched->n_graph_inputs * extra_copies;
    if (count > SIZE_MAX / sizeof(sched->copy_records[0])) {
        return;
    }
    if (count > sched->copy_records_capacity) {
        auto * records = (ggml_sched_copy_record *) malloc(count * sizeof(sched->copy_records[0]));
        if (!records) {
            return;
        }
        free(sched->copy_records);
        sched->copy_records = records;
        sched->copy_records_capacity = count;
    }

    size_t n = 0;
    auto add_copy = [&](ggml_tensor * source, ggml_tensor * copy, ggml_tensor * dependency, int b, int c) {
        if (!source || !copy || copy == source || n >= count) {
            return false;
        }
        ggml_sched_copy_record & record = sched->copy_records[n++];
        record = {};
        record.source = source;
        record.copy = copy;
        record.dependency = dependency;
        record.backend_id = b;
        record.copy_id = c;
        if (!copy->view_src && copy->buffer && copy->data && ggml_sched_layout_span(copy, &record.bytes)) {
            record.alloc_size = ggml_backend_buft_get_alloc_size(sched->bufts[b], copy);
            record.buffer = copy->buffer;
            record.data = copy->data;
            record.type = copy->type;
            record.valid = true;
        }
        return true;
    };

    if (sched->graph.n_nodes < 0) {
        return;
    }
    size_t node = 0;
    const size_t graph_nodes = (size_t) sched->graph.n_nodes;
    for (int i = 0; i < sched->n_splits; ++i) {
        const ggml_backend_sched_split & split = sched->splits[i];
        const int b = split.backend_id;
        if (b < 0 || b >= sched->n_backends || split.i_end < split.i_start) {
            return;
        }
        for (int j = 0; j < split.n_inputs; ++j) {
            ggml_tensor * input = split.inputs[j];
            if (!input || node > graph_nodes || graph_nodes - node < 2) {
                return;
            }
            const size_t id = ggml_hash_find(&sched->hash_set, input);
            if (id == GGML_HASHSET_FULL || !ggml_bitset_get(sched->hash_set.used, id)) {
                return;
            }
            ggml_tensor * dependency = sched->graph.nodes[node++];
            ggml_tensor * active_copy = sched->graph.nodes[node++];
            if (!dependency || dependency->src[0] != input || active_copy != tensor_id_copy(id, b, sched->cur_copy)) {
                return;
            }
            for (int c = 0; c < sched->n_copies; ++c) {
                if (!add_copy(input, tensor_id_copy(id, b, c), dependency, b, c)) {
                    return;
                }
            }
        }
        const size_t split_nodes = (size_t) (split.i_end - split.i_start);
        if (node > graph_nodes || split_nodes > graph_nodes - node) {
            return;
        }
        node += split_nodes;
    }
    if (node != graph_nodes) {
        return;
    }
    for (int i = 0; i < sched->n_graph_inputs; ++i) {
        ggml_tensor * input = sched->graph_inputs[i];
        if (!input) {
            return;
        }
        const size_t id = ggml_hash_find(&sched->hash_set, input);
        if (id == GGML_HASHSET_FULL || !ggml_bitset_get(sched->hash_set.used, id)) {
            return;
        }
        const int b = sched->hv_tensor_backend_ids[id];
        if (b < 0 || b >= sched->n_backends) {
            return;
        }
        for (int c = 0; c < sched->n_copies; ++c) {
            ggml_tensor * copy = tensor_id_copy(id, b, c);
            if (c == sched->cur_copy) {
                if (copy != input) {
                    return;
                }
            } else if (!add_copy(input, copy, nullptr, b, c)) {
                return;
            }
        }
    }
    if (n != count) {
        return;
    }
    if (n > 1) {
        std::sort(sched->copy_records, sched->copy_records + n,
                [](const ggml_sched_copy_record & a, const ggml_sched_copy_record & b) {
                    return std::less<const ggml_tensor *>()(a.source, b.source);
                });
    }
    sched->n_copy_records = n;
    sched->copy_envelopes_valid = true;
}

bool ggml_backend_sched_set_split_copy_tracking(ggml_backend_sched_t sched, bool enabled) {
    if (!sched) {
        return false;
    }
    if (sched->copy_tracking == enabled) {
        return true;
    }
    if (sched->is_alloc) {
        return false;
    }
    sched->copy_tracking = enabled;
    sched->copy_envelopes_valid = false;
    sched->n_copy_records = 0;
    if (!enabled) {
        free(sched->copy_records);
        sched->copy_records = nullptr;
        sched->copy_records_capacity = 0;
    }
    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (ggml_backend_sched_any_pending(sched)) {
        return false;
    }
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }
    if (sched->copy_tracking) {
        ggml_backend_sched_capture_copy_envelopes(sched);
    }

    sched->is_alloc = true;

    return true;
}

bool ggml_backend_sched_get_tensor_plan_size(ggml_backend_sched_t sched,
        const ggml_tensor * tensor, size_t * size) {
    return sched && sched->is_alloc &&
        ggml_gallocr_get_tensor_plan_size(sched->galloc, &sched->graph, tensor, size);
}

bool ggml_backend_sched_replan_graph(ggml_backend_sched_t sched,
        ggml_tensor * const * tensors, const uint8_t * fixed, size_t count,
        const ggml_backend_sched_tensor_layout * layouts, const size_t * view_offsets, size_t n_layouts) {
    // Re-splitting a graph can free scheduler copies still referenced by its
    // nodes. Only replan an already-placed host graph without such copies.
    if (!sched || ggml_backend_sched_any_pending(sched) ||
            !sched->is_alloc || !tensors || !fixed || count == 0 ||
            (n_layouts != 0 && (!layouts || !view_offsets)) || sched->n_backends != 1 ||
            sched->n_splits != 1 || sched->n_copies != 1 ||
            sched->splits[0].n_inputs != 0 || !ggml_backend_buft_is_host(sched->bufts[0])) return false;

    std::unordered_map<const ggml_tensor *, size_t> all;
    all.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        ggml_tensor * t = tensors[i];
        if (!t || !all.emplace(t, i).second) return false;
        if (fixed[i]) {
            if (t->view_src || ggml_gallocr_owns_buffer(sched->galloc, t->buffer)) return false;
            continue;
        }
        // An arena-owned constant is not refreshed after relocation. Refuse it
        // rather than silently lose its contents; external fixed roots are safe.
        if (!t->view_src && t->op == GGML_OP_NONE &&
                !(t->flags & GGML_TENSOR_FLAG_INPUT)) return false;
        if (t->extra != nullptr) return false;
        if (!t->view_src && ((t->buffer && !ggml_gallocr_owns_buffer(sched->galloc, t->buffer)) ||
                            (t->data && !t->buffer))) return false;
        if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) return false;
    }
    // The list includes external roots as well as all graph-owned views. This
    // prevents stale external views and stale pointers into a relocated arena.
    auto covered = [&](const ggml_tensor * t) { return !t || all.count(t) != 0; };
    for (size_t i = 0; i < count; ++i) if (!covered(tensors[i]->view_src)) return false;
    for (int i = 0; i < sched->graph.n_leafs; ++i) {
        if (!covered(sched->graph.leafs[i])) return false;
    }
    for (int i = 0; i < sched->graph.n_nodes; ++i) {
        const ggml_tensor * node = sched->graph.nodes[i];
        if (!covered(node) || !covered(node->view_src)) return false;
        for (int j = 0; j < GGML_MAX_SRC; ++j) if (!covered(node->src[j])) return false;
    }

    std::unordered_set<const ggml_tensor *> changed;
    for (size_t i = 0; i < n_layouts; ++i) {
        const ggml_tensor * t = layouts[i].tensor;
        const auto found = all.find(t);
        if (!t || found == all.end() || !changed.insert(t).second || fixed[found->second]) return false;
    }

    ggml_backend_sched_synchronize(sched);
    // Keep the old allocation alive until the new plan is fully allocated.
    // A failed attempt restores all live metadata and addresses.
    std::vector<ggml_tensor> saved;
    saved.reserve(count);
    for (size_t i = 0; i < count; ++i) saved.push_back(*tensors[i]);
    auto restore = [&]() {
        for (size_t i = 0; i < count; ++i) *tensors[i] = saved[i];
    };
    for (size_t i = 0; i < n_layouts; ++i) {
        ggml_tensor * t = layouts[i].tensor;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            t->ne[d] = layouts[i].ne[d];
            t->nb[d] = layouts[i].nb[d];
        }
        t->view_offs = view_offsets[i];
    }

    // Check byte spans and ultimate roots before an allocator can initialize
    // a view. A zero-size view may point at the end, but not beyond it.
    auto span = [](const ggml_tensor * t, size_t & bytes) {
        const int64_t block = ggml_blck_size(t->type);
        if (block <= 0 || t->ne[0] < 0 || t->ne[0] % block) return false;
        bool empty = false;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (t->ne[d] < 0) return false;
            empty = empty || t->ne[d] == 0;
        }
        if (empty) { bytes = 0; return true; }
        bytes = ggml_type_size(t->type);
        size_t part;
        if (__builtin_mul_overflow(size_t(t->ne[0] / block - 1), t->nb[0], &part) ||
                __builtin_add_overflow(bytes, part, &bytes)) return false;
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
            if (__builtin_mul_overflow(size_t(t->ne[d] - 1), t->nb[d], &part) ||
                    __builtin_add_overflow(bytes, part, &bytes)) return false;
        }
        return true;
    };
    for (size_t i = 0; i < count; ++i) {
        const ggml_tensor * t = tensors[i];
        size_t bytes;
        if (!span(t, bytes)) { restore(); return false; }
        if (!t->view_src) {
            if (t->view_offs != 0) { restore(); return false; }
            continue;
        }
        size_t offset = 0, depth = 0;
        const ggml_tensor * root = t;
        while (root->view_src) {
            if (++depth > count || __builtin_add_overflow(offset, root->view_offs, &offset)) {
                restore(); return false;
            }
            root = root->view_src;
        }
        size_t root_bytes;
        if (!span(root, root_bytes) || offset > root_bytes || bytes > root_bytes - offset) {
            restore(); return false;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        if (!fixed[i]) {
            tensors[i]->data = nullptr;
            tensors[i]->buffer = nullptr;
        }
    }
    ggml_gallocr_t next = ggml_gallocr_new_n(sched->bufts, sched->n_backends);
    if (!ggml_gallocr_reserve_n(next, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids) ||
            !ggml_gallocr_alloc_graph(next, &sched->graph)) {
        restore();
        ggml_gallocr_free(next);
        return false;
    }
    // A fixed root can have a host data pointer but no backend buffer. The
    // graph allocator deliberately skips those views; restore their address.
    for (size_t i = 0; i < count; ++i) {
        ggml_tensor * t = tensors[i];
        if (fixed[i] || !t->view_src || t->data) continue;
        size_t offset = 0;
        ggml_tensor * root = t;
        while (root->view_src) {
            offset += root->view_offs; // overflow was checked above
            root = root->view_src;
        }
        if (root->data && !root->buffer) {
            t->data = static_cast<char *>(root->data) + offset;
        } else if (ggml_nbytes(t) != 0) {
            restore();
            ggml_gallocr_free(next);
            return false;
        }
    }
    ggml_gallocr_t old = sched->galloc;
    sched->galloc = next;
    sched->copy_envelopes_valid = false;
    if (sched->copy_tracking) ggml_backend_sched_capture_copy_envelopes(sched);
    ggml_backend_sched_invalidate_graph_cache(sched);
    ggml_gallocr_free(old);
    return true;
}

void ggml_backend_sched_invalidate_graph_cache(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // Backend executables (CUDA/HIP graph captures, meta subgraphs) key their
    // cache on the split graph uid. A fresh uid makes the next compute
    // re-check the node and source properties, which is the only way a
    // metadata-only patch of the live tensors becomes visible to them.
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

bool ggml_backend_sched_tensor_has_split_copy(ggml_backend_sched_t sched, const struct ggml_tensor * tensor) {
    GGML_ASSERT(sched);
    if (tensor == nullptr) {
        return false;
    }
    for (int i = 0; i < sched->n_splits; ++i) {
        const ggml_backend_sched_split & split = sched->splits[i];
        for (int j = 0; j < split.n_inputs; ++j) {
            if (split.inputs[j] == tensor) {
                return true;
            }
        }
    }
    return false;
}

static size_t ggml_sched_copy_lower_bound(ggml_backend_sched_t sched, const ggml_tensor * source) {
    size_t first = 0;
    size_t count = sched->n_copy_records;
    while (count) {
        const size_t step = count / 2;
        const size_t middle = first + step;
        if (std::less<const ggml_tensor *>()(sched->copy_records[middle].source, source)) {
            first = middle + 1;
            count -= step + 1;
        } else {
            count = step;
        }
    }
    return first;
}

bool ggml_backend_sched_is_tensor_copy(ggml_backend_sched_t sched, const ggml_tensor * original, const ggml_tensor * candidate) {
    if (!sched || !sched->is_alloc || !sched->copy_tracking || !sched->copy_envelopes_valid ||
            !original || !candidate || original == candidate) {
        return false;
    }
    for (size_t i = ggml_sched_copy_lower_bound(sched, original);
            i < sched->n_copy_records && sched->copy_records[i].source == original; ++i) {
        const ggml_sched_copy_record & record = sched->copy_records[i];
        if (record.copy_id == sched->cur_copy && record.copy == candidate) {
            return true;
        }
    }
    return false;
}

struct ggml_tensor * ggml_backend_sched_debug_copy_dependency(
        ggml_backend_sched_t sched, const ggml_tensor * source) {
    if (!sched || !source || !sched->is_alloc || !sched->copy_tracking ||
            !sched->copy_envelopes_valid) {
        return nullptr;
    }
    for (size_t i = ggml_sched_copy_lower_bound(sched, source);
            i < sched->n_copy_records && sched->copy_records[i].source == source; ++i) {
        if (sched->copy_records[i].copy_id == sched->cur_copy) {
            return sched->copy_records[i].dependency;
        }
    }
    return nullptr;
}

bool ggml_backend_sched_update_split_copies(ggml_backend_sched_t sched,
        const ggml_backend_sched_tensor_layout * layouts, size_t n_layouts, bool apply) {
    if (!sched || ggml_backend_sched_any_pending(sched) ||
            !sched->is_alloc || !sched->copy_tracking || !sched->copy_envelopes_valid ||
            (n_layouts && !layouts)) {
        return false;
    }

    // Validate the complete transaction before changing any copy or view.
    for (size_t i = 0; i < n_layouts; ++i) {
        const ggml_backend_sched_tensor_layout & layout = layouts[i];
        if (!layout.tensor) {
            return false;
        }
        const size_t first = ggml_sched_copy_lower_bound(sched, layout.tensor);
        if (first == sched->n_copy_records || sched->copy_records[first].source != layout.tensor) {
            continue;
        }
        for (size_t j = 0; j < i; ++j) {
            if (layout.tensor == layouts[j].tensor) {
                return false;
            }
        }
        const size_t id = ggml_hash_find(&sched->hash_set, layout.tensor);
        if (id == GGML_HASHSET_FULL || !ggml_bitset_get(sched->hash_set.used, id)) {
            return false;
        }
        ggml_tensor proposed = *layout.tensor;
        memcpy(proposed.ne, layout.ne, sizeof(proposed.ne));
        memcpy(proposed.nb, layout.nb, sizeof(proposed.nb));
        size_t bytes;
        if (!ggml_sched_layout_span(&proposed, &bytes)) {
            return false;
        }
        for (size_t j = first; j < sched->n_copy_records && sched->copy_records[j].source == layout.tensor; ++j) {
            const ggml_sched_copy_record & record = sched->copy_records[j];
            ggml_tensor * copy = record.copy;
            if (!record.valid || tensor_id_copy(id, record.backend_id, record.copy_id) != copy ||
                    (record.dependency && record.dependency->src[0] != layout.tensor) ||
                    copy->view_src || copy->buffer != record.buffer || copy->data != record.data ||
                    copy->type != record.type || !ggml_are_same_layout(layout.tensor, copy) || bytes > record.bytes) {
                return false;
            }
            ggml_tensor copy_proposed = *copy;
            memcpy(copy_proposed.ne, layout.ne, sizeof(copy_proposed.ne));
            memcpy(copy_proposed.nb, layout.nb, sizeof(copy_proposed.nb));
            if (ggml_backend_buft_get_alloc_size(sched->bufts[record.backend_id], &copy_proposed) > record.alloc_size) {
                return false;
            }
        }
    }
    if (!apply) {
        return true;
    }

    for (size_t i = 0; i < n_layouts; ++i) {
        const ggml_backend_sched_tensor_layout & layout = layouts[i];
        for (size_t j = ggml_sched_copy_lower_bound(sched, layout.tensor);
                j < sched->n_copy_records && sched->copy_records[j].source == layout.tensor; ++j) {
            ggml_tensor * copy = sched->copy_records[j].copy;
            memcpy(copy->ne, layout.ne, sizeof(copy->ne));
            memcpy(copy->nb, layout.nb, sizeof(copy->nb));
        }
    }
    ggml_backend_sched_invalidate_graph_cache(sched);
    return true;
}

void ggml_backend_sched_refresh_split_copy_views(ggml_backend_sched_t sched) {
    if (!sched || !sched->is_alloc || !sched->copy_tracking || !sched->copy_envelopes_valid) {
        return;
    }
    if (ggml_backend_sched_any_pending(sched)) {
        // The dependency views may describe a destination an unproven copy is
        // still writing. Refreshing them would publish metadata for work that
        // has not completed.
        GGML_LOG_ERROR("%s: refusing to refresh split copy views: an async submission is still unproven\n", __func__);
        return;
    }
    for (size_t i = 0; i < sched->n_copy_records; ++i) {
        const ggml_sched_copy_record & record = sched->copy_records[i];
        if (!record.dependency || record.copy_id != sched->cur_copy) {
            continue;
        }
        ggml_tensor * input = record.source;
        ggml_tensor * dep = record.dependency;
        memcpy(dep->ne, input->ne, sizeof(dep->ne));
        memcpy(dep->nb, input->nb, sizeof(dep->nb));
        dep->view_src = input->view_src ? input->view_src : input;
        dep->view_offs = input->view_src ? input->view_offs : 0;
        dep->data = input->data;
        dep->buffer = dep->view_src->buffer;
    }
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    if (ggml_backend_sched_any_pending(sched)) {
        // Do not synchronize an unproven submission: a drain is not claimed.
        // A successful async result is not proof when a backend reports
        // outstanding work, so never return SUCCESS here.
        return err == GGML_STATUS_SUCCESS ? GGML_STATUS_FAILED : err;
    }
    ggml_backend_sched_synchronize(sched);
    return err;
}

static enum ggml_status ggml_backend_sched_graph_compute_async_impl(
        ggml_backend_sched_t sched, struct ggml_cgraph * graph, bool retained_inputs,
        ggml_backend_sched_receipt_impl * receipt) {
    GGML_ASSERT(sched);
    if (ggml_backend_sched_any_pending(sched)) {
        return GGML_STATUS_FAILED;
    }
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched, retained_inputs, receipt);
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    return ggml_backend_sched_graph_compute_async_impl(sched, graph, false, nullptr);
}

enum ggml_status ggml_backend_sched_graph_compute_async_retained_inputs(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    return ggml_backend_sched_graph_compute_async_impl(sched, graph, true, nullptr);
}

bool ggml_backend_sched_graph_compute_receipt(
        ggml_backend_sched_t sched, struct ggml_cgraph * graph,
        struct ggml_backend_sched_receipt * receipt) {
    if (sched == nullptr || graph == nullptr || receipt == nullptr) {
        return false;
    }
    memset(receipt, 0, sizeof(*receipt));
    receipt->version = GGML_BACKEND_SCHED_RECEIPT_API_VERSION;
    receipt->state   = GGML_BACKEND_SCHED_RECEIPT_COMPLETED_FAILURE;
    receipt->error   = -1;

    if (ggml_backend_sched_any_pending(sched)) {
        receipt->state = GGML_BACKEND_SCHED_RECEIPT_UNKNOWN;
        receipt->in_flight = true;
        return false;
    }

    auto * impl = new (std::nothrow) ggml_backend_sched_receipt_impl();
    if (impl == nullptr) {
        receipt->state = GGML_BACKEND_SCHED_RECEIPT_UNKNOWN;
        return false;
    }
    impl->attempt = ++sched->receipt_attempt;
    impl->state   = GGML_BACKEND_SCHED_RECEIPT_ACCEPTED;
    impl->error   = 0;

    receipt->attempt = impl->attempt;
    receipt->state   = impl->state;
    receipt->priv    = impl;

    const enum ggml_status status =
        ggml_backend_sched_graph_compute_async_impl(sched, graph, false, impl);
    if (status != GGML_STATUS_SUCCESS) {
        impl->failed_early = true;
        impl->error = status;
        if (impl->resources.empty()) {
            impl->state = GGML_BACKEND_SCHED_RECEIPT_COMPLETED_FAILURE;
        }
    }
    receipt->state = impl->state;
    receipt->error = impl->error;
    receipt->submitted = impl->submitted;
    receipt->in_flight = impl->in_flight;
    return impl->submitted;
}

int ggml_backend_sched_receipt_wait(
        ggml_backend_sched_t sched, struct ggml_backend_sched_receipt * receipt) {
    if (sched == nullptr || receipt == nullptr || receipt->priv == nullptr ||
            receipt->version != GGML_BACKEND_SCHED_RECEIPT_API_VERSION) {
        return GGML_BACKEND_SCHED_RECEIPT_UNKNOWN;
    }
    auto * impl = static_cast<ggml_backend_sched_receipt_impl *>(receipt->priv);
    const bool started_unknown = impl->state == GGML_BACKEND_SCHED_RECEIPT_UNKNOWN;
    if (impl->state != GGML_BACKEND_SCHED_RECEIPT_ACCEPTED &&
            !(impl->state == GGML_BACKEND_SCHED_RECEIPT_UNKNOWN && impl->in_flight)) {
        receipt->state = impl->state;
        receipt->error = impl->error;
        receipt->in_flight = impl->in_flight;
        return impl->state;
    }

    bool failed = impl->failed_early;
    bool unknown = false;
    bool event_only_seen = false;
    bool drained_unknown = false;
    int  error = impl->error;
    for (const ggml_backend_sched_receipt_resource & res : impl->resources) {
        if (res.event_only) {
            // A recorded backend event is a real completion fence but carries
            // no success/error status. It proves drained, not success.
            ggml_backend_event_synchronize(res.event);
            event_only_seen = true;
            continue;
        }
        const ggml_backend_completion_api * api =
            ggml_backend_sched_completion_api(res.backend);
        if (api == nullptr) {
            unknown = true;
            continue;
        }
        int wait_error = 0;
        int result;
        if (res.drain_only) {
            result = api->drain != nullptr ? api->drain(res.backend, &wait_error)
                                           : GGML_BACKEND_COMPLETION_UNKNOWN;
        } else {
            result = api->wait(res.backend, res.attempt, &wait_error);
        }
        if (result == GGML_BACKEND_COMPLETION_FAILED) {
            failed = true;
            if (error == 0) {
                error = wait_error != 0 ? wait_error : -1;
            }
        } else if (result != GGML_BACKEND_COMPLETION_SUCCESS) {
            if (api->drain != nullptr) {
                int drain_error = 0;
                if (api->drain(res.backend, &drain_error) == GGML_BACKEND_COMPLETION_SUCCESS) {
                    drained_unknown = true;
                } else {
                    unknown = true;
                    if (error == 0) error = drain_error != 0 ? drain_error : -1;
                }
            } else {
                unknown = true;
            }
        }
    }
    if (unknown) {
        // Keep the attempt quarantined: an accepted split may still be in
        // flight, so this is not a terminal failure the caller may free.
        impl->state = GGML_BACKEND_SCHED_RECEIPT_UNKNOWN;
        impl->error = error != 0 ? error : -1;
        impl->in_flight = !impl->resources.empty();
    } else if (failed || drained_unknown) {
        impl->state = GGML_BACKEND_SCHED_RECEIPT_COMPLETED_FAILURE;
        impl->error = error != 0 ? error : -1;
        impl->in_flight = false;
    } else if (event_only_seen) {
        // Drained, but not provable as success or failure.
        impl->state = GGML_BACKEND_SCHED_RECEIPT_UNKNOWN;
        impl->error = error != 0 ? error : -1;
        impl->in_flight = false;
    } else if (started_unknown && !impl->resources.empty()) {
        // All registered streams/events drained, but the original exact
        // attempt could not be named. This is a completed failure, not a
        // success; host undo may now be restored and the receipt released.
        impl->state = GGML_BACKEND_SCHED_RECEIPT_COMPLETED_FAILURE;
        impl->error = error != 0 ? error : -1;
        impl->in_flight = false;
    } else if (started_unknown) {
        // No resource was registered for an unknown attempt: keep it
        // quarantined rather than claiming a drain.
        impl->state = GGML_BACKEND_SCHED_RECEIPT_UNKNOWN;
        impl->error = error != 0 ? error : -1;
        impl->in_flight = true;
    } else {
        impl->state = GGML_BACKEND_SCHED_RECEIPT_COMPLETED_SUCCESS;
        impl->error = 0;
    }
    receipt->state = impl->state;
    receipt->error = impl->error;
    receipt->in_flight = impl->in_flight;
    return impl->state;
}

bool ggml_backend_sched_receipt_release(
        ggml_backend_sched_t sched, struct ggml_backend_sched_receipt * receipt) {
    (void) sched;
    if (receipt == nullptr || receipt->priv == nullptr) {
        return true;
    }
    auto * impl = static_cast<ggml_backend_sched_receipt_impl *>(receipt->priv);
    if (impl->state == GGML_BACKEND_SCHED_RECEIPT_ACCEPTED ||
            (impl->state == GGML_BACKEND_SCHED_RECEIPT_UNKNOWN && impl->in_flight)) {
        // Do not free an unproven in-flight attempt. The caller must drain or
        // keep it quarantined; deleting here would hide live backend work.
        return false;
    }
    for (const auto & res : impl->resources) {
        if (res.event_only) {
            ggml_backend_event_free(res.event);
            continue;
        }
        const ggml_backend_completion_api * api =
            ggml_backend_sched_completion_api(res.backend);
        if (api != nullptr && api->release != nullptr) {
            api->release(res.backend, res.attempt);
        }
    }
    delete impl;
    receipt->priv = nullptr;
    return true;
}

bool ggml_backend_sched_drain_pending(ggml_backend_sched_t sched, int * error) {
    if (sched == nullptr) {
        if (error != nullptr) *error = -1;
        return false;
    }
    bool any = false;
    for (int i = 0; i < sched->n_backends; ++i) {
        ggml_backend_t backend = sched->backends[i];
        const int submission_error = ggml_backend_sched_submission_error(backend);
        const bool pending = ggml_backend_sched_submission_pending(backend);
        if (submission_error == 0 && !pending) {
            continue;
        }
        any = true;
        const ggml_backend_completion_api * api = ggml_backend_sched_completion_api(backend);
        if (api != nullptr && api->drain != nullptr) {
            int drain_error = 0;
            if (api->drain(backend, &drain_error) != GGML_BACKEND_COMPLETION_SUCCESS) {
                if (error != nullptr) *error = drain_error != 0 ? drain_error : -1;
                return false;
            }
            continue;
        }
        if (backend->device != nullptr && backend->device->iface.event_new != nullptr &&
                backend->iface.event_record != nullptr) {
            ggml_backend_event_t event = ggml_backend_event_new(backend->device);
            if (event == nullptr) {
                if (error != nullptr) *error = -1;
                return false;
            }
            ggml_backend_event_record(event, backend);
            ggml_backend_event_synchronize(event);
            ggml_backend_event_free(event);
            continue;
        }
        if (error != nullptr) *error = submission_error != 0 ? submission_error : -1;
        return false;
    }
    if (!any) {
        if (error != nullptr) *error = 0;
        return true;
    }
    if (error != nullptr) *error = 0;
    return true;
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
