#pragma once

// Process-wide DSO module table. It lives in ggml-base so both the registry
// (ggml) and ggml-base callers (Meta provider, backend instances) can pin a
// module without a ggml-base -> ggml reverse library dependency.

#include "ggml-backend-impl.h"

#ifdef __cplusplus

#include <memory>

namespace ggml_backend_module {

// Pin the module behind reg. The returned shared_ptr keeps the DSO mapped;
// state tells the caller whether the handle is still usable.
GGML_BASE_API ggml_backend_module_ref hold(ggml_backend_reg_t reg);

// State without taking a reference. Used to re-check a registration after it
// was inserted: an unload may have started between lookup and insertion.
GGML_BASE_API ggml_backend_module_state state(ggml_backend_reg_t reg);

// Publish the handle for reg. On success the caller's shared_ptr is moved
// into the table. On failure the caller keeps ownership. loading marks a
// self-registered entry whose loader handle has not been attached yet.
GGML_BASE_API bool attach(ggml_backend_reg_t reg, std::shared_ptr<void> & module, bool loading);

// Remove reg from the table only when users==0. On success the table's module
// reference is written to *module; the caller releases it outside registry
// locks. On failure the record is left in place. A dynamic reg is recorded so
// a late provider registration is rejected instead of treated as static.
GGML_BASE_API bool detach(ggml_backend_reg_t reg, std::shared_ptr<void> * module,
                          bool * was_dynamic);

GGML_BASE_API bool mark_unloading(ggml_backend_reg_t reg);
GGML_BASE_API void unmark_unloading(ggml_backend_reg_t reg);
GGML_BASE_API bool is_unloading(ggml_backend_reg_t reg);

// True when reg's table entry holds module_identity. Used by the failed-load
// rollback to remove exactly the registry entry created by this transaction.
GGML_BASE_API bool matches(ggml_backend_reg_t reg, const void * module_identity);

// Live backend instances. unload refuses while this is nonzero, and a new
// backend init cannot start once the module is marked unloading.
GGML_BASE_API uint32_t users(ggml_backend_reg_t reg);

// One pin per backend instance. hold_user takes the module reference and the
// user slot under one lock; attach_backend publishes it in the host-owned
// side table after init_backend succeeds. release_pin drops both.
struct ggml_backend_pin {
    std::shared_ptr<void> module;
    ggml_backend_reg_t reg = nullptr;
    uint64_t epoch = 0;
    bool valid = false;
    // Preallocated spill node for put_back after a retained free. Opaque to
    // callers; owned by this pin until consumed.
    void * spill = nullptr;
};

GGML_BASE_API bool hold_user(ggml_backend_reg_t reg, ggml_backend_pin * pin);
GGML_BASE_API bool attach_backend(ggml_backend_t backend, ggml_backend_pin * pin);
GGML_BASE_API bool take_backend(ggml_backend_t backend, ggml_backend_pin * pin);
GGML_BASE_API void put_back(ggml_backend_t backend, ggml_backend_pin pin);
GGML_BASE_API void release_pin(ggml_backend_pin pin);
GGML_BASE_API bool backend_free_in_progress(ggml_backend_t backend);
GGML_BASE_API void retry_retained_pins();
GGML_BASE_API bool begin_registration(ggml_backend_reg_t reg);
GGML_BASE_API void end_registration(ggml_backend_reg_t reg);

// Provider free callbacks that intentionally retain the backend object call
// mark_retained before returning. ggml_backend_free consumes the mark on the
// same thread and keeps the pin instead of releasing it.
GGML_BASE_API void mark_retained(ggml_backend_t backend);
GGML_BASE_API bool consume_retained(ggml_backend_t backend);

// Loader transaction. loading_begin returns the previous handle/reg so nested
// loads restore them. A self-registering provider must bind its reg before
// asking for the handle; another reg during the same load is rejected.
struct loading_state {
    std::shared_ptr<void> module;
    ggml_backend_reg_t reg = nullptr;
};
GGML_BASE_API loading_state loading_begin(std::shared_ptr<void> module);
GGML_BASE_API void loading_end(loading_state previous);
GGML_BASE_API bool loading_bind(ggml_backend_reg_t reg);
GGML_BASE_API bool loading_active();
GGML_BASE_API std::shared_ptr<void> loading_handle();

// True while the current thread must not dlclose/FreeLibrary: provider free
// code or a registry transaction may still be on the stack. The shared-handle
// deleter queues the handle instead of closing it inline.
GGML_BASE_API bool close_deferred();
GGML_BASE_API void close_defer_begin();
GGML_BASE_API void close_defer_end(bool drain = true);
GGML_BASE_API void set_close_drain(void (*drain)());

struct close_defer_scope {
    bool drain;
    explicit close_defer_scope(bool drain_on_end = true) : drain(drain_on_end) {
        close_defer_begin();
    }
    ~close_defer_scope() { close_defer_end(drain); }
    close_defer_scope(const close_defer_scope &) = delete;
    close_defer_scope & operator=(const close_defer_scope &) = delete;
};

} // namespace ggml_backend_module

#endif
