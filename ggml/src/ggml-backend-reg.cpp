#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-dl.h"
#include "ggml-backend-module.h"
#include "ggml-backend-moe-cache.h"
#include "ggml-impl.h"
#include <algorithm>
#include <atomic>
#include <mutex>
#include <cstring>
#include <filesystem>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <vector>
#include <cctype>

extern "C" GGML_BASE_API bool ggml_backend_meta_staging_unregister_reg(
        ggml_backend_reg_t reg, bool * had_provider);

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#elif defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#else
#    include <dlfcn.h>
#    include <unistd.h>
#endif

// Backend registry
#ifdef GGML_USE_CPU
#include "ggml-cpu.h"
#endif

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef GGML_USE_SYCL
#include "ggml-sycl.h"
#endif

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef GGML_USE_WEBGPU
#include "ggml-webgpu.h"
#endif

#ifdef GGML_USE_ZDNN
#include "ggml-zdnn.h"
#endif

#ifdef GGML_USE_OPENCL
#include "ggml-opencl.h"
#endif

#ifdef GGML_USE_HEXAGON
#include "ggml-hexagon.h"
#endif

#ifdef GGML_USE_BLAS
#include "ggml-blas.h"
#endif

#ifdef GGML_USE_RPC
#include "ggml-rpc.h"
#endif

#ifdef GGML_USE_VIRTGPU_FRONTEND
#include "ggml-virtgpu.h"
#endif

#ifdef GGML_USE_CANN
#include "ggml-cann.h"
#endif

#ifdef GGML_USE_ZENDNN
#include "ggml-zendnn.h"
#endif

#ifdef GGML_USE_OPENVINO
#include "ggml-openvino.h"
#endif

#ifdef GGML_USE_ET
#include "ggml-et.h"
#endif

namespace fs = std::filesystem;

static std::string path_str(const fs::path & path) {
    try {
#if defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        const std::u8string u8str = path.u8string();
        return std::string(reinterpret_cast<const char *>(u8str.data()), u8str.size());
#else
        // C++17: u8string() returns std::string
        return path.u8string();
#endif
    } catch (...) {
        return std::string();
    }
}

struct ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    bool unloading = false;
    // True while this entry was created by backend_init_fn and the loader
    // has not yet attached its handle. A self-registered entry must never be
    // mistaken for a static backend.
    bool loading = false;
};

struct ggml_backend_registry {
    std::mutex module_mutex;
    // Serializes load and unload transactions. Provider self-registration
    // during backend_init_fn runs under this lock, so a concurrent unload
    // cannot observe a half-published backend.
    std::recursive_mutex load_mutex;
    std::vector<ggml_backend_reg_entry> backends;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_registry() {
#ifdef GGML_USE_CUDA
        register_backend(ggml_backend_cuda_reg());
#endif
#ifdef GGML_USE_METAL
        register_backend(ggml_backend_metal_reg());
#endif
#ifdef GGML_USE_SYCL
        register_backend(ggml_backend_sycl_reg());
#endif
#ifdef GGML_USE_VULKAN
    // Add runtime disable check
    if (getenv("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } else {
        GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n");
    }
#endif
#ifdef GGML_USE_WEBGPU
        register_backend(ggml_backend_webgpu_reg());
#endif
#ifdef GGML_USE_ZDNN
        register_backend(ggml_backend_zdnn_reg());
#endif
#ifdef GGML_USE_VIRTGPU_FRONTEND
        register_backend(ggml_backend_virtgpu_reg());
#endif

#ifdef GGML_USE_OPENCL
        register_backend(ggml_backend_opencl_reg());
#endif
#ifdef GGML_USE_ZENDNN
        register_backend(ggml_backend_zendnn_reg());
#endif
#ifdef GGML_USE_HEXAGON
        register_backend(ggml_backend_hexagon_reg());
#endif
#ifdef GGML_USE_CANN
        register_backend(ggml_backend_cann_reg());
#endif
#ifdef GGML_USE_BLAS
        register_backend(ggml_backend_blas_reg());
#endif
#ifdef GGML_USE_RPC
        register_backend(ggml_backend_rpc_reg());
#endif
#ifdef GGML_USE_OPENVINO
        register_backend(ggml_backend_openvino_reg());
#endif
#ifdef GGML_USE_ET
        register_backend(ggml_backend_et_reg());
#endif
#ifdef GGML_USE_CPU
        register_backend(ggml_backend_cpu_reg());
#endif
    }

    ~ggml_backend_registry() {
        // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
        // since backend threads may still be running and accessing resources from the dynamic library
        // shared handles close when the last registry/provider/cleanup
        // reference is gone; do not force-release them here.
    }

    static std::shared_ptr<void> module_ref(dl_handle_shared & handle) {
        if (!handle) {
            return {};
        }
        return std::shared_ptr<void>(handle, handle.get());
    }

    bool register_backend(ggml_backend_reg_t reg, dl_handle_shared handle = {},
                          dl_handle_shared * rejected = nullptr) {
        if (rejected != nullptr) {
            *rejected = {};
        }
        if (!reg) {
            if (rejected != nullptr) {
                *rejected = std::move(handle);
            }
            return false;
        }
        // Serialize the module publication with load/unload. A rejected
        // handle is returned to the caller, which drops it after releasing
        // this lock.
        std::lock_guard<std::recursive_mutex> load_lock(load_mutex);
        bool added = false;
        bool ok = true;
        std::shared_ptr<void> undo_module;
        {
            std::lock_guard<std::mutex> lock(module_mutex);
            auto it = std::find_if(backends.begin(), backends.end(),
                                   [reg](const ggml_backend_reg_entry & entry) {
                                       return entry.reg == reg;
                                   });
            if (it != backends.end()) {
                if (it->unloading || ggml_backend_module::is_unloading(reg)) {
                    ok = false;
                } else if (handle) {
                    // The loader is attaching its handle to an entry that the
                    // DSO registered from backend_init_fn.
                    std::shared_ptr<void> module = module_ref(handle);
                    if (!ggml_backend_module::attach(reg, module, false)) {
                        ok = false;
                    } else {
                        it->loading = false;
                    }
                } else if (it->loading) {
                    // A self-registered entry has no loader handle here.
                    // Clear the stale loading marker; the module table may
                    // already hold the loading handle.
                    std::shared_ptr<void> none;
                    if (!ggml_backend_module::attach(reg, none, false)) {
                        ok = false;
                    } else {
                        it->loading = false;
                    }
                } else {
                    // A static backend: publish a null-handle record so
                    // module_hold can distinguish it from an unknown reg.
                    std::shared_ptr<void> none;
                    if (!ggml_backend_module::attach(reg, none, false)) {
                        ok = false;
                    }
                }
            } else {
                const bool loading = !handle && ggml_backend_module::loading_active();
                const bool has_loader = static_cast<bool>(handle) || loading;
                if (loading && !ggml_backend_module::loading_bind(reg)) {
                    // A different reg already owns this load transaction.
                    ok = false;
                } else {
                    const ggml_backend_module_state state = ggml_backend_module::state(reg);
                    if (state == ggml_backend_module_state::unloading_backend ||
                            (state == ggml_backend_module_state::unloaded_backend &&
                             !has_loader)) {
                        ok = false;
                    } else {
                        std::shared_ptr<void> module = module_ref(handle);
                        if (loading && !module) {
                            // Publish the loader handle immediately. Provider
                            // registrations that happen before this entry exists
                            // already hold a copy through module_hold().
                            module = ggml_backend_module::loading_handle();
                        }
                        if (!ggml_backend_module::attach(reg, module, false)) {
                            ok = false;
                        } else {
                            try {
                                backends.push_back({ reg, false, loading });
                                added = true;
                            } catch (...) {
                                // attach() already published the module record;
                                // undo it so an unregistered reg cannot keep
                                // answering module_hold().
                                bool was_dynamic = false;
                                if (!ggml_backend_module::detach(
                                        reg, &undo_module, &was_dynamic)) {
                                    // A concurrent backend init took a user pin.
                                    // Keep the record rather than erasing a live
                                    // module; the failed registration remains
                                    // visible to unload/reap.
                                    GGML_LOG_ERROR("%s: cannot undo module record for %p: users are live\n",
                                                   __func__, (void *) reg);
                                }
                                ok = false;
                            }
                        }
                    }
                }
            }
        }
        undo_module.reset();
        if (!ok) {
            if (rejected != nullptr) {
                *rejected = std::move(handle);
            }
            return false;
        }

        // A provider may have registered before the registry entry existed.
        // Attach the handle to those provider entries now.
        const ggml_backend_module_ref module = ggml_backend_module::hold(reg);
        if (module.module) {
            ggml_backend_meta_staging_attach_module(reg, module.module);
        }

        if (added) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
                __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
#endif
        }
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
        if (rejected != nullptr && handle) {
            *rejected = std::move(handle);
        }
        return true;
    }

    void register_device(ggml_backend_dev_t device) {
        if (!device) {
            return;
        }
        std::lock_guard<std::recursive_mutex> load_lock(load_mutex);
        {
            std::lock_guard<std::mutex> lock(module_mutex);
            const ggml_backend_reg_t reg = device->reg;
            const auto entry = std::find_if(backends.begin(), backends.end(),
                                            [reg](const ggml_backend_reg_entry & candidate) {
                                                return candidate.reg == reg;
                                            });
            if (entry == backends.end() || entry->unloading ||
                    ggml_backend_module::is_unloading(reg)) {
                return;
            }
            if (std::find(devices.begin(), devices.end(), device) != devices.end()) {
                return;
            }
            try {
                devices.push_back(device);
            } catch (...) {
                return;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
    }

    // Snapshot a device and its registry under one lock. The caller pins the
    // module before calling into the device; an unload between the snapshot
    // and the pin is visible as unloading/unloaded.
    bool device_snapshot(size_t index, ggml_backend_dev_t * device,
                         ggml_backend_reg_t * reg) {
        if (device == nullptr || reg == nullptr) {
            return false;
        }
        std::lock_guard<std::mutex> lock(module_mutex);
        if (index >= devices.size()) {
            return false;
        }
        *device = devices[index];
        *reg = *device != nullptr ? (*device)->reg : nullptr;
        return *device != nullptr;
    }

    // A failed backend_init_fn may have self-registered a registry entry
    // before returning. Remove exactly the entries created by this load
    // transaction; the loader's handle is still alive, so dropping the
    // table reference here cannot unmap the DSO.
    void rollback_backend_load(const void * module_identity,
                               std::vector<std::shared_ptr<void>> * released) {
        if (module_identity == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> lock(module_mutex);
        for (auto it = backends.begin(); it != backends.end(); ) {
            if (!it->loading ||
                    !ggml_backend_module::matches(it->reg, module_identity)) {
                ++it;
                continue;
            }
            const ggml_backend_reg_t reg = it->reg;
            bool was_dynamic = false;
            std::shared_ptr<void> module;
            if (!ggml_backend_module::detach(reg, &module, &was_dynamic)) {
                // A backend instance already holds a user pin. Leave the
                // registry entry and its module mapped; the caller cannot
                // safely roll it back.
                GGML_LOG_ERROR("%s: cannot roll back %p: live backend pins\n",
                               __func__, (void *) reg);
                ++it;
                continue;
            }
            devices.erase(
                std::remove_if(devices.begin(), devices.end(),
                               [reg](ggml_backend_dev_t dev) {
                                   return dev != nullptr && dev->reg == reg;
                               }),
                devices.end());
            it = backends.erase(it);
            (void) was_dynamic;
            if (module && released != nullptr) {
                try {
                    released->push_back(std::move(module));
                } catch (...) {
                    // The local module drops under the loader's close-defer
                    // scope, so the DSO is not unmapped inside this lock.
                }
            }
        }
    }

    // Device-bearing DSOs stay mapped for process lifetime. A device, buffer
    // type, or model pointer obtained before unload must never point into
    // unmapped code. This is the conservative side of the quiescent-unload
    // contract, not a deferred dlclose list: the registry entry and provider
    // are gone, and a reload registers a fresh provider tag.
    struct retained_node {
        std::shared_ptr<void> module;
        retained_node * next = nullptr;
    };

    static void retain_module(std::shared_ptr<void> module, retained_node * node) {
        if (!module || node == nullptr) {
            delete node;
            return;
        }
        static std::mutex * mutex = new std::mutex();
        static retained_node * retained = nullptr;
        node->module = std::move(module);
        node->next = nullptr;
        std::lock_guard<std::mutex> lock(*mutex);
        node->next = retained;
        retained = node;
    }

    ggml_backend_reg_t load_backend(const fs::path & path, bool silent) {
        // Provider free code and registry transactions must not dlclose.
        ggml_backend_module::close_defer_scope defer_close;
        std::unique_lock<std::recursive_mutex> load_lock(load_mutex);
        dl_handle_shared handle;
        if (dl_handle * raw = dl_load_library(path)) {
            handle = dl_handle_shared::adopt(raw);
        }
        if (!handle) {
            if (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            load_lock.unlock();
            return nullptr;
        }

        std::shared_ptr<void> module = module_ref(handle);
        const void * module_identity = handle.get();
        auto finish_failure = [&]() -> ggml_backend_reg_t {
            load_lock.unlock();
            module.reset();
            handle.reset();
            dl_handle_drain_deferred();
            return nullptr;
        };

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        if (score_fn && score_fn() == 0) {
            if (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            return finish_failure();
        }

        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        if (!backend_init_fn) {
            if (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            return finish_failure();
        }

        // Everything the provider registers from now on belongs to this
        // transaction. A failure removes exactly those entries; a repeated
        // load of the same DSO keeps the provider that was registered before.
        const uint32_t tag_floor = ggml_backend_meta_staging_tag_floor();

        // Publish the DSO to provider self-registration for the duration of
        // backend_init_fn. The loader owns the handle, so the DSO cannot be
        // unloaded while init runs. A nested load restores the outer handle.
        ggml_backend_module::loading_state previous_loading =
            ggml_backend_module::loading_begin(module);

        ggml_backend_reg_t reg = backend_init_fn();
        if (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            if (!silent) {
                if (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } else {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            // Fill null-module provider entries from this transaction before
            // rollback decides whether to quarantine them. A quarantined
            // entry must keep the DSO mapped.
            ggml_backend_meta_staging_attach_load_module(
                    reg, tag_floor, module_identity, module);
            ggml_backend_meta_staging_rollback_load(reg, tag_floor, module_identity);
            std::vector<std::shared_ptr<void>> released;
            rollback_backend_load(module_identity, &released);
            load_lock.unlock();
            released.clear();
            ggml_backend_module::loading_end(std::move(previous_loading));
            module.reset();
            handle.reset();
            dl_handle_drain_deferred();
            return nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        dl_handle_shared rejected;
        if (!register_backend(reg, std::move(handle), &rejected)) {
            GGML_LOG_ERROR("%s: failed to register backend from %s\n",
                           __func__, path_str(path).c_str());
            ggml_backend_meta_staging_attach_load_module(
                    reg, tag_floor, module_identity, module);
            ggml_backend_meta_staging_rollback_load(reg, tag_floor, module_identity);
            std::vector<std::shared_ptr<void>> released;
            rollback_backend_load(module_identity, &released);
            load_lock.unlock();
            released.clear();
            rejected.reset();
            ggml_backend_module::loading_end(std::move(previous_loading));
            module.reset();
            dl_handle_drain_deferred();
            return nullptr;
        }

        load_lock.unlock();
        rejected.reset();
        ggml_backend_module::loading_end(std::move(previous_loading));
        module.reset();
        dl_handle_drain_deferred();
        return reg;
    }

    void unload_backend(ggml_backend_reg_t reg, bool silent) {
        ggml_backend_module::close_defer_scope defer_close;
        std::unique_lock<std::recursive_mutex> load_lock(load_mutex);
        // Reserve the terminal owner node before any unload side effect. If
        // this allocation fails, refuse the unload instead of aborting or
        // dropping a device module later.
        std::unique_ptr<retained_node> retain_slot(
            new (std::nothrow) retained_node{});
        if (!retain_slot) {
            GGML_LOG_ERROR("%s: cannot reserve retained module record\n", __func__);
            load_lock.unlock();
            return;
        }

        // Mark before touching the provider: module_hold keeps returning the
        // handle for cleanup records, while new provider/backend
        // registrations are refused until the unload either commits or rolls
        // back.
        {
            std::lock_guard<std::mutex> lock(module_mutex);
            auto it = std::find_if(backends.begin(), backends.end(),
                                   [reg](const ggml_backend_reg_entry & entry) {
                                       return entry.reg == reg;
                                   });
            if (it == backends.end()) {
                if (!silent) {
                    GGML_LOG_ERROR("%s: backend not found\n", __func__);
                }
                load_lock.unlock();
                return;
            }
            if (it->unloading) {
                if (!silent) {
                    GGML_LOG_DEBUG("%s: unload already in progress\n", __func__);
                }
                load_lock.unlock();
                return;
            }
            it->unloading = true;
            if (!ggml_backend_module::mark_unloading(reg)) {
                it->unloading = false;
                load_lock.unlock();
                return;
            }
        }

        if (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        auto clear_unloading = [&]() {
            ggml_backend_module::unmark_unloading(reg);
            std::lock_guard<std::mutex> lock(module_mutex);
            auto it = std::find_if(backends.begin(), backends.end(),
                                   [reg](const ggml_backend_reg_entry & entry) {
                                       return entry.reg == reg;
                                   });
            if (it != backends.end()) {
                it->unloading = false;
            }
        };

        if (ggml_backend_module::users(reg) != 0) {
            // A live backend instance still calls into this DSO. The module
            // pin keeps the code mapped, but removing the registry entry
            // would strand its device/buffer lookups.
            clear_unloading();
            GGML_LOG_WARN("%s: refusing to unload %s while backend instances are live\n",
                          __func__, ggml_backend_reg_name(reg));
            load_lock.unlock();
            return;
        }

        bool had_staging = false;
        if (!ggml_backend_meta_staging_unregister_reg(reg, &had_staging)) {
            // Retained owners, live tokens, or an in-flight acquire still
            // point into this DSO. Refuse the unload; the caller can retry
            // after draining/releasing the owners.
            clear_unloading();
            GGML_LOG_WARN("%s: refusing to unload %s while staging owners or calls are live\n",
                          __func__, ggml_backend_reg_name(reg));
            load_lock.unlock();
            return;
        }

        ggml_moe_cache_unregister(reg);

        std::shared_ptr<void> module;
        std::size_t device_count = 0;
        {
            std::lock_guard<std::mutex> lock(module_mutex);
            auto it = std::find_if(backends.begin(), backends.end(),
                                   [reg](const ggml_backend_reg_entry & entry) {
                                       return entry.reg == reg;
                                   });
            if (it == backends.end()) {
                ggml_backend_module::unmark_unloading(reg);
                load_lock.unlock();
                return;
            }

            // remove devices
            for (ggml_backend_dev_t dev : devices) {
                if (dev != nullptr && dev->reg == reg) {
                    ++device_count;
                }
            }
            bool was_dynamic = false;
            if (!ggml_backend_module::detach(reg, &module, &was_dynamic)) {
                // The users check above should make this unreachable. Keep
                // the entry and module mapped rather than erasing a live
                // backend's registry record.
                it->unloading = false;
                ggml_backend_module::unmark_unloading(reg);
                load_lock.unlock();
                GGML_LOG_ERROR("%s: refusing to detach %p: live backend pins\n",
                               __func__, (void *) reg);
                return;
            }
            devices.erase(
                std::remove_if(devices.begin(), devices.end(),
                                [reg](ggml_backend_dev_t dev) {
                                    return dev != nullptr && dev->reg == reg;
                                }),
                devices.end());

            backends.erase(it);
            (void) was_dynamic;
        }
        (void) had_staging;
        load_lock.unlock();

        // Device-bearing and staging DSOs keep their mapping for process
        // lifetime. A model, buffer type, or device pointer obtained before
        // unload must not become dangling code. Non-device providers are
        // released here, outside every registry lock; the shared-handle
        // deleter closes them from host code.
        if (module && (device_count != 0 || had_staging)) {
            retain_module(std::move(module), retain_slot.release());
        } else {
            module.reset();
        }
        dl_handle_drain_deferred();
    }
};

static ggml_backend_registry & get_reg() {
    // Process-lifetime registry: intentionally not destroyed. Module handles
    // must outlive static destructors and provider atexit callbacks; freeing
    // them from a static destructor can dlclose code that still has live
    // function pointers. The OS reclaims the allocation at process exit.
    static ggml_backend_registry * reg = new ggml_backend_registry();
    return *reg;
}

// Internal API
void ggml_backend_register(ggml_backend_reg_t reg) {
    (void) get_reg().register_backend(reg);
}

void ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
static bool striequals(const char * a, const char * b) {
    for (; *a && *b; a++, b++) {
        if (std::tolower(*a) != std::tolower(*b)) {
            return false;
        }
    }
    return *a == *b;
}

size_t ggml_backend_reg_count() {
    auto & registry = get_reg();
    std::lock_guard<std::mutex> lock(registry.module_mutex);
    return registry.backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    auto & registry = get_reg();
    std::lock_guard<std::mutex> lock(registry.module_mutex);
    if (index >= registry.backends.size()) {
        // count() and get() are separate calls; an unload between them must
        // not turn a shrinking table into an assertion.
        return nullptr;
    }
    return registry.backends[index].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(const char * name) {
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (reg == nullptr) {
            continue;
        }
        const ggml_backend_module_ref hold = ggml_backend_reg_module_hold(reg);
        if (hold.state == ggml_backend_module_state::unloading_backend ||
                hold.state == ggml_backend_module_state::unloaded_backend) {
            continue;
        }
        if (striequals(ggml_backend_reg_name(reg), name)) {
            return reg;
        }
    }
    return nullptr;
}

// Device enumeration
size_t ggml_backend_dev_count() {
    auto & registry = get_reg();
    std::lock_guard<std::mutex> lock(registry.module_mutex);
    return registry.devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    auto & registry = get_reg();
    std::lock_guard<std::mutex> lock(registry.module_mutex);
    if (index >= registry.devices.size()) {
        return nullptr;
    }
    return registry.devices[index];
}

ggml_backend_dev_t ggml_backend_dev_by_name(const char * name) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = nullptr;
        ggml_backend_reg_t reg = nullptr;
        if (!get_reg().device_snapshot(i, &dev, &reg)) {
            continue;
        }
        const ggml_backend_module_ref hold = ggml_backend_reg_module_hold(reg);
        if (hold.state == ggml_backend_module_state::unloading_backend ||
                hold.state == ggml_backend_module_state::unloaded_backend) {
            continue;
        }
        if (striequals(ggml_backend_dev_name(dev), name)) {
            return dev;
        }
    }
    return nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = nullptr;
        ggml_backend_reg_t reg = nullptr;
        if (!get_reg().device_snapshot(i, &dev, &reg)) {
            continue;
        }
        const ggml_backend_module_ref hold = ggml_backend_reg_module_hold(reg);
        if (hold.state == ggml_backend_module_state::unloading_backend ||
                hold.state == ggml_backend_module_state::unloaded_backend) {
            continue;
        }
        if (ggml_backend_dev_type(dev) == type) {
            return dev;
        }
    }
    return nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(const char * path) {
    return get_reg().load_backend(path, false);
}

void ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, true);
}

static fs::path get_executable_path() {
#if defined(__APPLE__)
    // get executable path
    std::vector<char> path;
    uint32_t size;
    while (true) {
        size = path.size();
        if (_NSGetExecutablePath(path.data(), &size) == 0) {
            break;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    if (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    return base_path + "/";
#elif defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<char> path(1024);
    while (true) {
        // get executable path
#    if defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        if (len == -1) {
            break;
        }
        if (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            if (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            break;
        }
        path.resize(path.size() * 2);
    }

    return base_path + "/";
#elif defined(_WIN32)
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(NULL, path.data(), path.size());
    if (len == 0) {
        return {};
    }
    std::wstring base_path(path.data(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    if (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    return base_path + L"\\";
#else
    return {};
#endif
}

static fs::path backend_filename_prefix() {
#ifdef _WIN32
    return fs::u8path("ggml-");
#else
    return fs::u8path("libggml-");
#endif
}

static fs::path backend_filename_extension() {
#ifdef _WIN32
    return fs::u8path(".dll");
#else
    return fs::u8path(".so");
#endif
}

static ggml_backend_reg_t ggml_backend_load_best(const char * name, bool silent, const char * user_search_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    const fs::path name_path = fs::u8path(name);
    const fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    const fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> search_paths;
    if (user_search_path == nullptr) {
#ifdef GGML_BACKEND_DIR
        search_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif
        // default search paths: executable directory, current directory
        search_paths.push_back(get_executable_path());
        search_paths.push_back(fs::current_path());
    } else {
        search_paths.push_back(fs::u8path(user_search_path));
    }

    int best_score = 0;
    fs::path best_path;
    std::error_code ec;

    for (const auto & search_path : search_paths) {
        if (!fs::exists(search_path, ec)) {
            if (ec) {
                GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(search_path).c_str(), ec.message().c_str());
            } else {
                GGML_LOG_DEBUG("%s: search path %s does not exist\n", __func__, path_str(search_path).c_str());
            }
            continue;
        }
        fs::directory_iterator dir_it(search_path, fs::directory_options::skip_permission_denied);
        for (const auto & entry : dir_it) {
            if (entry.is_regular_file(ec)) {
                auto filename = entry.path().filename();
                auto ext = entry.path().extension();
                if (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_load_library(entry) };
                    if (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(entry.path()).c_str(), dl_error());
                    }
                    if (handle) {
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        if (score_fn) {
                            int s = score_fn();
#ifndef NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif
                            if (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } else {
                            if (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }
    }

    if (best_score == 0) {
        // try to load the base backend
        for (const auto & search_path : search_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = search_path / filename;
            if (std::error_code ec; fs::exists(path, ec)) {
                return get_reg().load_backend(path, silent);
            } else {
                if (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
        return nullptr;
    }

    return get_reg().load_backend(best_path, silent);
}

void ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

void ggml_backend_load_all_from_path(const char * dir_path) {
#ifdef NDEBUG
    bool silent = true;
#else
    bool silent = false;
#endif

    ggml_backend_load_best("blas", silent, dir_path);
    ggml_backend_load_best("zdnn", silent, dir_path);
    ggml_backend_load_best("zendnn", silent, dir_path);
    ggml_backend_load_best("cann", silent, dir_path);
    ggml_backend_load_best("cuda", silent, dir_path);
    ggml_backend_load_best("hip", silent, dir_path);
    ggml_backend_load_best("metal", silent, dir_path);
    ggml_backend_load_best("rpc", silent, dir_path);
    ggml_backend_load_best("sycl", silent, dir_path);
    ggml_backend_load_best("vulkan", silent, dir_path);
    ggml_backend_load_best("virtgpu", silent, dir_path);
    ggml_backend_load_best("opencl", silent, dir_path);
    ggml_backend_load_best("hexagon", silent, dir_path);
    ggml_backend_load_best("musa", silent, dir_path);
    ggml_backend_load_best("openvino", silent, dir_path);
    ggml_backend_load_best("cpu", silent, dir_path);
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    const char * backend_path = std::getenv("GGML_BACKEND_PATH");
    if (backend_path) {
        ggml_backend_load(backend_path);
    }
}
