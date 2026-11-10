#include "ggml-backend-module.h"
#include "ggml-impl.h"

#include <atomic>
#include <mutex>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ggml_backend_module {

namespace {

struct module_record {
    std::shared_ptr<void> module;
    uint32_t users = 0;
    uint32_t registrations = 0;
    uint64_t epoch = 0;
    bool dynamic = false;
    bool loading = false;
    bool unloading = false;
};

struct module_table {
    std::mutex mutex;
    std::unordered_map<ggml_backend_reg_t, module_record> records;
    std::unordered_set<ggml_backend_reg_t> unloaded_dynamic;
};

module_table & get_table() {
    // Process lifetime: provider static destructors and atexit callbacks may
    // still ask for a module after the registry is gone. The OS reclaims it.
    static module_table * table = new module_table();
    return *table;
}

thread_local std::shared_ptr<void> t_loading;
thread_local ggml_backend_reg_t t_loading_reg = nullptr;
thread_local uint32_t t_close_defer = 0;
thread_local ggml_backend_t t_retained_backend = nullptr;
std::atomic<uint64_t> g_module_epoch{1};
void (*g_close_drain)() = nullptr;

ggml_backend_module_state state_locked(const module_table & table, ggml_backend_reg_t reg) {
    const auto it = table.records.find(reg);
    if (it != table.records.end()) {
        const module_record & record = it->second;
        if (record.unloading) {
            return ggml_backend_module_state::unloading_backend;
        }
        if (record.loading && !record.module) {
            return (t_loading && t_loading_reg == reg)
                ? ggml_backend_module_state::dynamic_backend
                : ggml_backend_module_state::unknown;
        }
        return record.module ? ggml_backend_module_state::dynamic_backend
                             : ggml_backend_module_state::static_backend;
    }
    if (t_loading && t_loading_reg == reg) {
        return ggml_backend_module_state::dynamic_backend;
    }
    if (table.unloaded_dynamic.count(reg) != 0) {
        return ggml_backend_module_state::unloaded_backend;
    }
    return ggml_backend_module_state::unknown;
}

void release_user(ggml_backend_reg_t reg, uint64_t epoch) {
    if (reg == nullptr) {
        return;
    }
    module_table & table = get_table();
    std::lock_guard<std::mutex> lock(table.mutex);
    const auto it = table.records.find(reg);
    if (it != table.records.end() && it->second.epoch == epoch &&
            it->second.users > 0) {
        --it->second.users;
    }
}

struct pin_node {
    ggml_backend_t backend = nullptr;
    ggml_backend_pin pin;
    std::uint32_t attempts = 0;
    std::chrono::steady_clock::time_point next_retry{};
    pin_node * next = nullptr;
};

std::mutex & g_pin_mutex() {
    static std::mutex * mutex = new std::mutex();
    return *mutex;
}

std::unordered_map<ggml_backend_t, ggml_backend_pin> & g_pins() {
    static auto * pins = new std::unordered_map<ggml_backend_t, ggml_backend_pin>();
    return *pins;
}

pin_node *& g_retained_pins() {
    static pin_node * head = nullptr;
    return head;
}

struct emergency_pin {
    ggml_backend_t backend = nullptr;
    ggml_backend_pin pin;
    std::uint32_t attempts = 0;
    std::chrono::steady_clock::time_point next_retry{};
};

constexpr std::size_t kEmergencyPins = 8;

emergency_pin * g_emergency_pins() {
    static emergency_pin pins[kEmergencyPins];
    return pins;
}

// Backends currently owned by one retry/free pass. Guarded by g_pin_mutex.
std::unordered_set<ggml_backend_t> g_freeing;

bool begin_backend_free(ggml_backend_t backend) {
    std::lock_guard<std::mutex> lock(g_pin_mutex());
    if (g_freeing.count(backend) != 0) {
        return false;
    }
    try {
        g_freeing.insert(backend);
        return true;
    } catch (...) {
        return false;
    }
}

void end_backend_free(ggml_backend_t backend) {
    std::lock_guard<std::mutex> lock(g_pin_mutex());
    g_freeing.erase(backend);
}

} // namespace

ggml_backend_module_ref hold(ggml_backend_reg_t reg) {
    if (reg == nullptr) {
        return {};
    }
    module_table & table = get_table();
    std::lock_guard<std::mutex> lock(table.mutex);
    const auto it = table.records.find(reg);
    if (it != table.records.end()) {
        const module_record & record = it->second;
        if (record.unloading) {
            return {record.module, ggml_backend_module_state::unloading_backend};
        }
        if (record.loading && !record.module) {
            if (t_loading && t_loading_reg == reg) {
                return {t_loading, ggml_backend_module_state::dynamic_backend};
            }
            return {nullptr, ggml_backend_module_state::unknown};
        }
        return {record.module, record.module ? ggml_backend_module_state::dynamic_backend
                                             : ggml_backend_module_state::static_backend};
    }
    if (t_loading && t_loading_reg == reg) {
        return {t_loading, ggml_backend_module_state::dynamic_backend};
    }
    if (table.unloaded_dynamic.count(reg) != 0) {
        return {nullptr, ggml_backend_module_state::unloaded_backend};
    }
    return {nullptr, ggml_backend_module_state::unknown};
}

ggml_backend_module_state state(ggml_backend_reg_t reg) {
    if (reg == nullptr) {
        return ggml_backend_module_state::unknown;
    }
    module_table & table = get_table();
    std::lock_guard<std::mutex> lock(table.mutex);
    return state_locked(table, reg);
}

bool attach(ggml_backend_reg_t reg, std::shared_ptr<void> & module, bool loading) {
    if (reg == nullptr) {
        return false;
    }
    module_table & table = get_table();
    std::lock_guard<std::mutex> lock(table.mutex);
    try {
        // Allocate the node before moving the caller's reference. If the map
        // allocation throws, the caller still owns module.
        const auto inserted = table.records.try_emplace(reg);
        module_record & record = inserted.first->second;
        bool changed = inserted.second;
        bool identity_changed = inserted.second;
        const bool tombstone = table.unloaded_dynamic.count(reg) != 0;
        if (record.unloading) {
            if (inserted.second) {
                table.records.erase(inserted.first);
            }
            return false;
        }
        if (module) {
            if (!record.module || record.module.get() != module.get()) {
                record.module = std::move(module);
                changed = true;
                identity_changed = true;
            }
            if (record.loading) {
                record.loading = false;
            }
            record.dynamic = true;
            record.unloading = false;
            // Only a real loader handle may resurrect an unloaded dynamic
            // reg or clear the unloading mark.
            table.unloaded_dynamic.erase(reg);
        } else if (loading) {
            if (!record.module && !record.loading) {
                record.loading = true;
                changed = true;
            }
        } else {
            if (tombstone) {
                // A static provider must not reuse the address of an unloaded
                // dynamic DSO.
                if (inserted.second) {
                    table.records.erase(inserted.first);
                }
                return false;
            }
            if (record.loading) {
                // A static registration supersedes a stale loading marker.
                record.loading = false;
                changed = true;
            }
        }
        if (changed && identity_changed) {
            // A new pin epoch invalidates old release tokens only when the
            // record identity actually changed. Duplicate static/self
            // registrations must not strand live users.
            record.epoch = g_module_epoch.fetch_add(1, std::memory_order_release);
        } else if (changed && record.epoch == 0) {
            record.epoch = g_module_epoch.fetch_add(1, std::memory_order_release);
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool detach(ggml_backend_reg_t reg, std::shared_ptr<void> * module_out,
            bool * was_dynamic) {
    if (module_out != nullptr) {
        module_out->reset();
    }
    if (was_dynamic != nullptr) {
        *was_dynamic = false;
    }
    if (reg == nullptr) {
        return false;
    }
    module_table & table = get_table();
    std::lock_guard<std::mutex> lock(table.mutex);
    const auto it = table.records.find(reg);
    if (it == table.records.end()) {
        return false;
    }
    if (it->second.users != 0 || it->second.registrations != 0) {
        // A live backend instance or an in-flight provider registration
        // still points into this module. Keep the entry.
        return false;
    }
    const bool dynamic = it->second.dynamic || it->second.module != nullptr;
    if (dynamic) {
        try {
            table.unloaded_dynamic.insert(reg);
        } catch (...) {
            // Leave the record in place; an unloaded reg must never fall back
            // to "unknown" and be re-published as static.
            return false;
        }
    }
    std::shared_ptr<void> module = std::move(it->second.module);
    table.records.erase(it);
    if (module_out != nullptr) {
        *module_out = std::move(module);
    }
    if (was_dynamic != nullptr) {
        *was_dynamic = dynamic;
    }
    return true;
}

bool mark_unloading(ggml_backend_reg_t reg) {
    if (reg == nullptr) {
        return false;
    }
    module_table & table = get_table();
    std::lock_guard<std::mutex> lock(table.mutex);
    const auto it = table.records.find(reg);
    if (it == table.records.end()) {
        // Every registered backend has a record. A missing record means the
        // reg was never published or was already removed; fail closed.
        return false;
    }
    it->second.unloading = true;
    return true;
}

void unmark_unloading(ggml_backend_reg_t reg) {
    if (reg == nullptr) {
        return;
    }
    module_table & table = get_table();
    std::lock_guard<std::mutex> lock(table.mutex);
    const auto it = table.records.find(reg);
    if (it != table.records.end()) {
        it->second.unloading = false;
    }
}

bool is_unloading(ggml_backend_reg_t reg) {
    if (reg == nullptr) {
        return false;
    }
    module_table & table = get_table();
    std::lock_guard<std::mutex> lock(table.mutex);
    const auto it = table.records.find(reg);
    return it != table.records.end() && it->second.unloading;
}

bool matches(ggml_backend_reg_t reg, const void * module_identity) {
    if (reg == nullptr || module_identity == nullptr) {
        return false;
    }
    module_table & table = get_table();
    std::lock_guard<std::mutex> lock(table.mutex);
    const auto it = table.records.find(reg);
    return it != table.records.end() && it->second.module.get() == module_identity;
}

uint32_t users(ggml_backend_reg_t reg) {
    if (reg == nullptr) {
        return 0;
    }
    module_table & table = get_table();
    std::lock_guard<std::mutex> lock(table.mutex);
    const auto it = table.records.find(reg);
    return it != table.records.end() ? it->second.users : 0;
}

bool hold_user(ggml_backend_reg_t reg, ggml_backend_pin * pin) {
    if (reg == nullptr || pin == nullptr) {
        return false;
    }
    module_table & table = get_table();
    std::lock_guard<std::mutex> lock(table.mutex);
    const auto it = table.records.find(reg);
    if (it == table.records.end()) {
        // Unknown is not a usable module. Never let a backend init race an
        // unload that removed the record.
        return false;
    }
    module_record & record = it->second;
    if (record.unloading) {
        return false;
    }
    std::shared_ptr<void> module = record.module;
    if (record.loading && !module) {
        if (!t_loading || t_loading_reg != reg) {
            return false;
        }
        module = t_loading;
    }
    if (record.dynamic && !module) {
        return false;
    }
    ++record.users;
    pin->module = std::move(module);
    pin->reg = reg;
    pin->epoch = record.epoch;
    pin->valid = true;
    return true;
}

bool attach_backend(ggml_backend_t backend, ggml_backend_pin * pin) {
    if (backend == nullptr || pin == nullptr || !pin->valid) {
        return false;
    }
    {
        module_table & table = get_table();
        std::lock_guard<std::mutex> lock(table.mutex);
        const auto it = table.records.find(pin->reg);
        if (it == table.records.end() || it->second.epoch != pin->epoch ||
                it->second.unloading) {
            // The reg was reloaded or unloaded while init_backend ran. Do not
            // publish a pin that no longer belongs to the record.
            return false;
        }
    }
    // Reserve the spill node before publishing the pin. A later retained free
    // must not need an allocation.
    pin_node * spill = new (std::nothrow) pin_node{};
    if (spill == nullptr) {
        return false;
    }
    pin->spill = spill;
    std::lock_guard<std::mutex> lock(g_pin_mutex());
    try {
        const auto inserted = g_pins().try_emplace(backend);
        if (!inserted.second) {
            pin->spill = nullptr;
            delete spill;
            return false;
        }
        inserted.first->second = std::move(*pin);
        pin->valid = false;
        pin->spill = nullptr;
        return true;
    } catch (...) {
        pin->spill = nullptr;
        delete spill;
        return false;
    }
}

bool begin_registration(ggml_backend_reg_t reg) {
    if (reg == nullptr) {
        return false;
    }
    module_table & table = get_table();
    std::lock_guard<std::mutex> lock(table.mutex);
    const auto it = table.records.find(reg);
    if (it == table.records.end()) {
        // register_provider publishes static/loading records before this
        // call; an absent record is not a usable provider target.
        return false;
    }
    module_record & record = it->second;
    if (record.unloading) {
        return false;
    }
    if (record.loading && !record.module && !t_loading) {
        return false;
    }
    ++record.registrations;
    return true;
}

void end_registration(ggml_backend_reg_t reg) {
    if (reg == nullptr) {
        return;
    }
    module_table & table = get_table();
    std::lock_guard<std::mutex> lock(table.mutex);
    const auto it = table.records.find(reg);
    if (it != table.records.end() && it->second.registrations > 0) {
        --it->second.registrations;
    }
}

bool backend_free_in_progress(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_pin_mutex());
    return g_freeing.count(backend) != 0;
}

bool take_backend(ggml_backend_t backend, ggml_backend_pin * pin) {
    if (backend == nullptr || pin == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_pin_mutex());
    const auto it = g_pins().find(backend);
    if (it == g_pins().end()) {
        // A provider-retained backend is parked in the spill list; a later
        // free takes its pin from there.
        pin_node ** link = &g_retained_pins();
        while (*link != nullptr && (*link)->backend != backend) {
            link = &(*link)->next;
        }
        if (*link != nullptr) {
            pin_node * node = *link;
            *link = node->next;
            *pin = std::move(node->pin);
            // Reuse the unlinked node as the pin's spill for the next
            // retained free; no allocation is needed.
            pin->spill = node;
            node->backend = nullptr;
            node->pin = {};
            node->next = nullptr;
            return true;
        }
        for (std::size_t i = 0; i < kEmergencyPins; ++i) {
            emergency_pin & slot = g_emergency_pins()[i];
            if (slot.backend == backend && slot.pin.valid) {
                *pin = std::move(slot.pin);
                slot.pin = {};
                slot.backend = nullptr;
                return true;
            }
        }
        return false;
    }
    if (it->second.spill == nullptr) {
        // Replenish the spill node for a retained free on this take. A failed
        // allocation is recoverable in put_back's fallback path.
        it->second.spill = new (std::nothrow) pin_node{};
    }
    *pin = std::move(it->second);
    g_pins().erase(it);
    return true;
}

void put_back(ggml_backend_t backend, ggml_backend_pin pin) {
    if (backend == nullptr || !pin.valid) {
        release_pin(std::move(pin));
        return;
    }
    pin_node * spill = static_cast<pin_node *>(pin.spill);
    // A retained pin always lives in the retained list, never in g_pins: the
    // retry pass must be able to find it.
    pin.spill = nullptr;
    pin_node * node = spill;
    if (node == nullptr) {
        node = new (std::nothrow) pin_node{};
    }
    if (node == nullptr) {
        // No heap storage for a second spill. Use the bounded emergency
        // slots so the pin is still retained; never release a retained
        // backend's users/module.
        {
            std::lock_guard<std::mutex> lock(g_pin_mutex());
            for (std::size_t i = 0; i < kEmergencyPins; ++i) {
                emergency_pin & slot = g_emergency_pins()[i];
                if (!slot.pin.valid) {
                    slot.backend = backend;
                    slot.pin = std::move(pin);
                    slot.attempts = 0;
                    slot.next_retry = {};
                    return;
                }
            }
        }
        GGML_LOG_ERROR("%s: emergency pin slots exhausted; retaining raw pin state\n", __func__);
        release_pin(std::move(pin));
        return;
    }
    node->backend = backend;
    node->pin = std::move(pin);
    {
        std::lock_guard<std::mutex> lock(g_pin_mutex());
        node->next = g_retained_pins();
        g_retained_pins() = node;
    }
}

void release_pin(ggml_backend_pin pin) {
    if (pin.spill != nullptr) {
        delete static_cast<pin_node *>(pin.spill);
        pin.spill = nullptr;
    }
    if (!pin.valid) {
        return;
    }
    release_user(pin.reg, pin.epoch);
    pin.module.reset();
}

void mark_retained(ggml_backend_t backend) {
    t_retained_backend = backend;
}

bool consume_retained(ggml_backend_t backend) {
    if (t_retained_backend == backend) {
        t_retained_backend = nullptr;
        return true;
    }
    return false;
}

void retry_retained_pins() {
    ggml_backend_t target = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_pin_mutex());
        const auto now = std::chrono::steady_clock::now();
        for (pin_node * node = g_retained_pins(); node != nullptr; node = node->next) {
            if (g_freeing.count(node->backend) == 0 && now >= node->next_retry) {
                target = node->backend;
                break;
            }
        }
        if (target == nullptr) {
            for (std::size_t i = 0; i < kEmergencyPins; ++i) {
                emergency_pin & slot = g_emergency_pins()[i];
                if (slot.pin.valid && g_freeing.count(slot.backend) == 0 &&
                        now >= slot.next_retry) {
                    target = slot.backend;
                    break;
                }
            }
        }
        if (target == nullptr) {
            return;
        }
        try {
            g_freeing.insert(target);
        } catch (...) {
            return;
        }
    }
    // We own the single-flight marker for target. ggml_backend_free_and_report
    // takes the pin before touching the backend; if another free already took
    // it, it returns without dereferencing. One target per call avoids a raw
    // pointer snapshot race.
    try {
        (void) ggml_backend_free_and_report(target);
    } catch (...) {
    }
    std::lock_guard<std::mutex> lock(g_pin_mutex());
    const auto now = std::chrono::steady_clock::now();
    for (pin_node * node = g_retained_pins(); node != nullptr; node = node->next) {
        if (node->backend == target) {
            ++node->attempts;
            const std::uint32_t shift = std::min<std::uint32_t>(node->attempts, 6);
            node->next_retry = now + std::chrono::milliseconds(
                std::min<std::uint64_t>(5000, std::uint64_t(100) << shift));
            break;
        }
    }
    for (std::size_t i = 0; i < kEmergencyPins; ++i) {
        emergency_pin & slot = g_emergency_pins()[i];
        if (slot.backend == target && slot.pin.valid) {
            ++slot.attempts;
            const std::uint32_t shift = std::min<std::uint32_t>(slot.attempts, 6);
            slot.next_retry = now + std::chrono::milliseconds(
                std::min<std::uint64_t>(5000, std::uint64_t(100) << shift));
            break;
        }
    }
    g_freeing.erase(target);
}

loading_state loading_begin(std::shared_ptr<void> module) {
    loading_state previous{t_loading, t_loading_reg};
    t_loading = std::move(module);
    t_loading_reg = nullptr;
    return previous;
}

void loading_end(loading_state previous) {
    t_loading = std::move(previous.module);
    t_loading_reg = previous.reg;
}

bool loading_bind(ggml_backend_reg_t reg) {
    if (!t_loading || reg == nullptr) {
        return false;
    }
    if (t_loading_reg == nullptr) {
        t_loading_reg = reg;
        return true;
    }
    return t_loading_reg == reg;
}

bool loading_active() {
    return static_cast<bool>(t_loading);
}

std::shared_ptr<void> loading_handle() {
    return t_loading;
}

bool close_deferred() {
    return t_close_defer != 0;
}

void close_defer_begin() {
    ++t_close_defer;
}

void close_defer_end(bool drain) {
    if (t_close_defer > 0) {
        --t_close_defer;
        if (drain && t_close_defer == 0 && g_close_drain != nullptr) {
            // Queue entries created inside provider free / registry
            // transactions are closed now that the last scope is gone.
            g_close_drain();
        }
    }
}

void set_close_drain(void (*drain)()) {
    g_close_drain = drain;
}

} // namespace ggml_backend_module

GGML_BASE_API ggml_backend_module_ref ggml_backend_reg_module_hold(ggml_backend_reg_t reg) {
    return ggml_backend_module::hold(reg);
}

extern "C" GGML_BASE_API void ggml_backend_mark_retained(ggml_backend_t backend) {
    ggml_backend_module::mark_retained(backend);
}

extern "C" GGML_BASE_API void ggml_backend_retry_retained(void) {
    ggml_backend_module::retry_retained_pins();
}
