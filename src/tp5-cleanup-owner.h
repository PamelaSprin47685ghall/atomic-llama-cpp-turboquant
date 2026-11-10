#pragma once

// Fixed-capacity is deliberately avoided.  A cleanup owner reserves all
// records it may need before native resources are created; the failure path
// only moves already-owned resources into those records.  When the current
// capacity is exhausted, reserve() grows it before returning (never on the
// failure path) and blocks the caller until the reservation succeeds.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <utility>
#include <vector>

namespace tq::tp5 {

template <typename Traits>
class cleanup_owner {
public:
    using backend_type = typename Traits::backend_type;
    using backend_ptr  = typename Traits::backend_ptr;

    struct retained_raw_node {
        // Declared before backend: reverse destruction destroys backend
        // first, then the module reference that keeps its code mapped.
        std::shared_ptr<void> module;
        backend_ptr backend{};
        const char * why = nullptr;
        retained_raw_node * next = nullptr;
    };
    struct retained_token_node {
        std::uint64_t owner = 0;
        std::uint64_t generation = 0;
        retained_token_node * next = nullptr;
    };

    struct reservation {
        std::vector<std::size_t> slot;
        // armed[pos] is true once this reservation has handed the slot to a
        // worker. commit() must not reset an armed slot: it may already be
        // FREE and reused by another reservation.
        std::vector<bool> armed;
        std::size_t count = 0;
        // Nodes are exclusively bound to this reservation. They are allocated
        // before any native side effect and moved into a slot when armed.
        std::vector<std::unique_ptr<retained_raw_node>> raw_nodes;
        std::vector<std::unique_ptr<retained_token_node>> token_nodes;
    };

    cleanup_owner() = default;
    cleanup_owner(const cleanup_owner &) = delete;
    cleanup_owner & operator=(const cleanup_owner &) = delete;
    ~cleanup_owner() noexcept {
        try {
            shutdown();
        } catch (...) {
            // A destructor must not terminate on a lock/system error.
        }
    }

    // Reserve n records before native side effects.  The vector growth and
    // the worker start are the only allocations, and both happen here.
    bool reserve(std::size_t n, reservation & out) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (stopping_) {
            return false;
        }
        if (n == 0) {
            out.count = 0;
            return true;
        }
        // Exclusive error storage for this reservation, allocated before any
        // native owner exists. Concurrent reservations cannot share it.
        try {
            out.raw_nodes.clear();
            out.token_nodes.clear();
            out.raw_nodes.reserve(n);
            out.token_nodes.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                auto raw = std::unique_ptr<retained_raw_node>(
                    new (std::nothrow) retained_raw_node{});
                auto token = std::unique_ptr<retained_token_node>(
                    new (std::nothrow) retained_token_node{});
                if (!raw || !token) {
                    out.raw_nodes.clear();
                    out.token_nodes.clear();
                    return false;
                }
                out.raw_nodes.push_back(std::move(raw));
                out.token_nodes.push_back(std::move(token));
            }
        } catch (...) {
            out.raw_nodes.clear();
            out.token_nodes.clear();
            return false;
        }
        // Grow before returning: the caller performs this reservation before
        // any native side effect, so allocation failure is a clean refusal,
        // not a dropped record.  Existing reservations keep their indices.
        std::size_t need = free_slots_locked() < n ? n - free_slots_locked() : 0;
        try {
            if (need != 0) {
                slots_.resize(slots_.size() + need);
            }
            out.slot.resize(n);
            out.armed.assign(n, false);
            std::size_t found = 0;
            for (std::size_t i = 0; i < slots_.size() && found < n; ++i) {
                if (slots_[i].state == slot::FREE) {
                    slots_[i].state = slot::RESERVED;
                    out.slot[found++] = i;
                }
            }
            out.count = found;
            if (found != n) {
                for (std::size_t i = 0; i < found; ++i) {
                    slots_[out.slot[i]] = slot{};
                }
                out.count = 0;
                out.slot.clear();
                out.armed.clear();
                return false;
            }
        } catch (...) {
            out.count = 0;
            out.slot.clear();
            out.armed.clear();
            return false;
        }
        if (!worker_.joinable()) {
            try {
                worker_ = std::thread([this] { worker_loop(); });
            } catch (...) {
                for (std::size_t i = 0; i < out.count; ++i) {
                    slots_[out.slot[i]] = slot{};
                }
                out.count = 0;
                out.slot.clear();
                return false;
            }
        }
        return true;
    }

    // Failure path: move backend ownership into a reserved record.  No
    // allocation and no lock held across the move.
#ifdef TP5_TEST_HOOKS
    void test_fail_next_arm(int n) { test_arm_fail_.store(n); }
    void test_fail_next_token(int n) { test_token_fail_.store(n); }
#endif

    bool arm_backend(reservation & res, std::size_t pos, backend_ptr && backend,
                     bool retain_only = false,
                     std::shared_ptr<std::atomic<bool>> release_gate = {},
                     std::shared_ptr<void> module = {},
                     bool module_unloaded = false) {
#ifdef TP5_TEST_HOOKS
        if (test_arm_fail_.fetch_sub(1) > 0) {
            return false;
        }
#endif
        if (pos >= res.count) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            return false;
        }
        slot & s = slots_[res.slot[pos]];
        if (s.state != slot::RESERVED) {
            return false;
        }
        s.backend = std::move(backend);
        s.has_backend = true;
        s.retain_only = retain_only;
        s.release_gate = std::move(release_gate);
        s.module = std::move(module);
        s.module_unloaded = module_unloaded;
        if (pos < res.raw_nodes.size()) {
            s.raw_node = std::move(res.raw_nodes[pos]);
        }
        s.next_retry = {};
        s.release_attempts = 0;
        s.state = slot::READY;
        if (pos < res.armed.size()) {
            res.armed[pos] = true;
        }
        cv_.notify_all();
        return true;
    }

    // Failure path: retain the complete owner token (owner, generation and
    // the provider route encoded in owner) for a later unbind.
    bool arm_token(reservation & res, std::size_t pos, std::uint64_t owner,
                   std::uint64_t generation) {
#ifdef TP5_TEST_HOOKS
        if (test_token_fail_.fetch_sub(1) > 0) {
            return false;
        }
#endif
        if (pos >= res.count) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            return false;
        }
        slot & s = slots_[res.slot[pos]];
        if (s.state != slot::RESERVED) {
            return false;
        }
        s.owner = owner;
        s.generation = generation;
        s.has_token = true;
        if (pos < res.token_nodes.size()) {
            s.token_node = std::move(res.token_nodes[pos]);
        }
        s.next_retry = {};
        s.release_attempts = 0;
        s.state = slot::READY;
        if (pos < res.armed.size()) {
            res.armed[pos] = true;
        }
        cv_.notify_all();
        return true;
    }

    // No-throw forced transfer into a slot already owned by the reservation.
    // Used when a normal arm unexpectedly rejects a reserved index.
    bool force_backend(reservation & res, std::size_t pos, backend_ptr && backend,
                       bool retain_only, std::shared_ptr<std::atomic<bool>> release_gate,
                       std::shared_ptr<void> module = {},
                       bool module_unloaded = false) {
        if (pos >= res.count) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            return false;
        }
        slot & s = slots_[res.slot[pos]];
        if (s.state != slot::RESERVED) {
            // Never overwrite READY/BUSY/RETAINED: that would destroy an
            // owner the worker may still be processing.
            return false;
        }
        s.backend = std::move(backend);
        s.has_backend = true;
        s.retain_only = retain_only;
        s.release_gate = std::move(release_gate);
        s.module = std::move(module);
        s.module_unloaded = module_unloaded;
        if (pos < res.raw_nodes.size()) {
            s.raw_node = std::move(res.raw_nodes[pos]);
        }
        s.next_retry = {};
        s.release_attempts = 0;
        s.state = slot::READY;
        if (pos < res.armed.size()) {
            res.armed[pos] = true;
        }
        cv_.notify_all();
        return true;
    }

    bool force_token(reservation & res, std::size_t pos, std::uint64_t owner,
                     std::uint64_t generation) {
        if (pos >= res.count) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            return false;
        }
        slot & s = slots_[res.slot[pos]];
        if (s.state != slot::RESERVED) {
            return false;
        }
        s.owner = owner;
        s.generation = generation;
        s.has_token = true;
        if (pos < res.token_nodes.size()) {
            s.token_node = std::move(res.token_nodes[pos]);
        }
        s.next_retry = {};
        s.release_attempts = 0;
        s.state = slot::READY;
        if (pos < res.armed.size()) {
            res.armed[pos] = true;
        }
        cv_.notify_all();
        return true;
    }

    // Ownership-preserving fallback when a reserved slot is unexpectedly not
    // usable. Grow the table by one record and let the worker retry. The
    // caller keeps the backend on allocation failure.
    bool retain_backend(backend_ptr && backend,
                        std::shared_ptr<std::atomic<bool>> release_gate,
                        std::shared_ptr<void> module = {},
                        bool module_unloaded = false) {
        if (!backend) {
            return true;
        }
        std::unique_ptr<retained_raw_node> raw(
            new (std::nothrow) retained_raw_node{});
        if (!raw) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            slots_.emplace_back();
        } catch (...) {
            return false;
        }
        slot & s = slots_.back();
        s.raw_node = std::move(raw);
        s.backend = std::move(backend);
        s.has_backend = true;
        s.release_gate = std::move(release_gate);
        s.module = std::move(module);
        s.module_unloaded = module_unloaded;
        s.next_retry = {};
        s.release_attempts = 0;
        s.state = slot::READY;
        cv_.notify_all();
        return true;
    }

    bool retain_token(std::uint64_t owner, std::uint64_t generation) {
        std::unique_ptr<retained_token_node> token(
            new (std::nothrow) retained_token_node{});
        if (!token) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            slots_.emplace_back();
        } catch (...) {
            return false;
        }
        slot & s = slots_.back();
        s.token_node = std::move(token);
        s.owner = owner;
        s.generation = generation;
        s.has_token = true;
        s.next_retry = {};
        s.release_attempts = 0;
        s.state = slot::READY;
        cv_.notify_all();
        return true;
    }

    // Success path: retire only records that were never armed.
    void commit(reservation & res) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t i = 0; i < res.count; ++i) {
            slot & s = slots_[res.slot[i]];
            if (!res.armed[i] && s.state == slot::RESERVED) {
                s.reset();
            }
        }
        res.count = 0;
        res.slot.clear();
        res.armed.clear();
        cv_.notify_all();
    }

    bool idle() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const slot & s : slots_) {
            if (s.state != slot::FREE) {
                return false;
            }
        }
        return !has_retained_tokens();
    }

    void retry_now() {
        std::lock_guard<std::mutex> lock(mutex_);
        cv_.notify_all();
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                return;
            }
            stopping_ = true;
            cv_.notify_all();
        }
        if (worker_.joinable()) {
            worker_.join();
        }
        // Bounded final drain: process every READY slot that exists at
        // shutdown, not a fixed four.  Failing records stay owned.
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(100);
        while (std::chrono::steady_clock::now() < deadline &&
               (has_ready() || has_retained_tokens())) {
            try {
                process_one_ready(true);
                process_retained_token(true);
                Traits::retry_orphans();
            } catch (...) {
                std::fprintf(stderr, "TP5_CLEANUP_OWNER shutdown process exception\n");
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // Context members are gone by process shutdown; release joint-retained
        // backends now, outside any provider callback and outside the lock.
        for (;;) {
            std::size_t index = SIZE_MAX;
            slot retained;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                for (std::size_t i = 0; i < slots_.size(); ++i) {
                    if (slots_[i].state == slot::RETAINED) {
                        index = i;
                        retained = std::move(slots_[i]);
                        slots_[i] = slot{};
                        break;
                    }
                }
                if (index == SIZE_MAX) {
                    break;
                }
            }
            if (!retained.release_backend()) {
                // Keep the owner in the table; the process-lifetime owner
                // (or the terminal slot destructor) remains responsible.
                std::lock_guard<std::mutex> lock(mutex_);
                slots_[index] = std::move(retained);
                slots_[index].state = slot::RETAINED;
                break;
            }
        }
    }

private:
    static std::mutex & retained_raw_mutex() {
        static std::mutex * mutex = new std::mutex();
        return *mutex;
    }
    static retained_raw_node *& retained_raw_head() {
        static retained_raw_node * head = nullptr;
        return head;
    }
    static bool retain_raw_node(std::unique_ptr<retained_raw_node> node) {
        if (!node) {
            return false;
        }
        std::lock_guard<std::mutex> lock(retained_raw_mutex());
        node->next = retained_raw_head();
        retained_raw_head() = node.release();
        return true;
    }

    struct slot {
        enum state_t { FREE, RESERVED, READY, BUSY, RETAINED };
        slot() = default;
        slot(const slot &) = delete;
        slot & operator=(const slot &) = delete;
        slot(slot &&) noexcept = default;
        slot & operator=(slot &&) noexcept = default;
        bool retain_raw_owner(const char * why) {
            if (!backend) {
                return true;
            }
            if (!raw_node) {
                raw_node.reset(new (std::nothrow) retained_raw_node{});
            }
            if (!raw_node) {
                return false;
            }
            raw_node->backend = std::move(backend);
            raw_node->module = std::move(module);
            raw_node->why = why;
            return cleanup_owner::retain_raw_node(std::move(raw_node));
        }
        ~slot() {
            if (!backend) {
                return;
            }
            if (module_unloaded) {
                if (!retain_raw_owner("module unloaded")) {
                    std::fprintf(stderr,
                        "TP5_CLEANUP_OWNER cannot allocate terminal owner; leaking raw backend\n");
                    (void) backend.release();
                }
                return;
            }
            bool released = false;
            try {
                released = Traits::release(backend);
            } catch (...) {
                released = false;
            }
            if (!released) {
                if (!retain_raw_owner("release unproved")) {
                    std::fprintf(stderr,
                        "TP5_CLEANUP_OWNER cannot allocate terminal owner; leaking raw backend\n");
                    (void) backend.release();
                }
            }
        }
        // Declared before backend: module must be destroyed after backend.
        std::shared_ptr<void> module;
        backend_ptr backend{};
        std::uint64_t owner = 0;
        std::uint64_t generation = 0;
        state_t state = FREE;
        bool has_backend = false;
        bool has_token = false;
        bool retain_only = false;
        bool cleared = false;
        bool module_unloaded = false;
        std::shared_ptr<std::atomic<bool>> release_gate;
        std::unique_ptr<retained_raw_node> raw_node;
        std::unique_ptr<retained_token_node> token_node;
        std::chrono::steady_clock::time_point next_retry{};
        std::uint32_t release_attempts = 0;

        // Release the owned backend. On false the slot still owns it and the
        // worker must retry; Traits::release must leave backend non-null on
        // failure. On true the unique_ptr is empty.
        bool release_backend() {
            if (!backend) {
                return true;
            }
            if (module_unloaded) {
                if (!retain_raw_owner("module unloaded")) {
                    ++release_attempts;
                    const std::uint32_t shift =
                        std::min<std::uint32_t>(release_attempts, 6);
                    next_retry = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(
                            std::min<std::uint64_t>(5000, std::uint64_t(100) << shift));
                    return false;
                }
                has_backend = false;
                cleared = false;
                return true;
            }
            bool released = false;
            try {
                released = Traits::release(backend);
            } catch (...) {
                released = false;
            }
            if (!released) {
                ++release_attempts;
                const std::uint32_t shift =
                    std::min<std::uint32_t>(release_attempts, 6);
                const auto delay = std::chrono::milliseconds(
                    std::min<std::uint64_t>(5000, std::uint64_t(100) << shift));
                next_retry = std::chrono::steady_clock::now() + delay;
                return false;
            }
            has_backend = false;
            cleared = false;
            return true;
        }

        // Only legal when no backend is owned. Retiring an owned backend is
        // release_backend()'s job; reset must not silently drop it.
        void reset() {
            if (backend) {
                std::fprintf(stderr,
                    "TP5_CLEANUP_OWNER refusing to reset an owned backend\n");
                return;
            }
            module.reset();
            release_gate.reset();
            raw_node.reset();
            token_node.reset();
            owner = 0;
            generation = 0;
            state = FREE;
            has_backend = false;
            has_token = false;
            retain_only = false;
            cleared = false;
            module_unloaded = false;
            next_retry = {};
            release_attempts = 0;
        }
    };

    bool has_ready() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const slot & s : slots_) {
            if (s.state == slot::READY) {
                return true;
            }
        }
        return false;
    }

    // Token slots are ordinary READY records and are processed by
    // process_one_ready(); there is no separate token queue in this owner.
    bool has_retained_tokens() const { return false; }
    bool process_retained_token(bool) { return false; }

    std::size_t free_slots_locked() const {
        std::size_t n = 0;
        for (const slot & s : slots_) {
            n += s.state == slot::FREE ? 1 : 0;
        }
        return n;
    }

    // Take one due READY record out of the table under the lock, release the
    // lock for provider calls, then return the record or retire it. A failed
    // clear/unbind/release leaves the record in READY with backoff; ownership
    // is never dropped on the failure path.
    void process_one_ready(bool force = false) {
        slot work;
        bool have = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto now = std::chrono::steady_clock::now();
            for (slot & s : slots_) {
                if (s.state == slot::READY && (force || now >= s.next_retry)) {
                    if (s.release_gate &&
                            !s.release_gate->load(std::memory_order_acquire)) {
                        // Skip gate-held records so one of them cannot starve
                        // the rest of the table at shutdown.
                        s.next_retry = now + std::chrono::milliseconds(100);
                        continue;
                    }
                    s.state = slot::BUSY;
                    work = std::move(s);
                    have = true;
                    break;
                }
            }
        }
        if (!have) {
            return;
        }
        bool done = false;
        bool gate_hold = false;
        if (work.has_backend && !work.cleared) {
            if (work.module_unloaded) {
                // Do not call clear into a module the registry has already
                // unloaded. reset() retains the backend allocation.
                done = true;
                work.cleared = true;
            } else if (work.release_gate &&
                !work.release_gate->load(std::memory_order_acquire)) {
                // Context members still own the backend. Do not call the
                // provider clear yet: it may mutate provider state that the
                // scheduler/prefab destructors still use.
                gate_hold = true;
            } else {
                try {
                    done = Traits::clear(work.backend.get());
                } catch (...) {
                    done = false;
                }
                work.cleared = done;
            }
        } else if (work.has_backend) {
            done = true;
        } else if (work.has_token) {
            try {
                done = Traits::unbind(work.owner, work.generation);
            } catch (...) {
                done = false;
            }
        } else {
            // A READY record with neither a backend nor a token is empty;
            // retire it instead of spinning.
            done = true;
        }
        if (!gate_hold && done && work.has_backend && !work.retain_only &&
                work.release_gate &&
                !work.release_gate->load(std::memory_order_acquire)) {
            // The gate closed after clear; release only after the last
            // context member is gone.
            gate_hold = true;
        }
        bool released = true;
        if (!gate_hold && done && work.has_backend && !work.retain_only) {
            // Provider release runs outside every cleanup lock.
            released = work.release_backend();
        }
        std::shared_ptr<void> retired_module;
        bool handled = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto now = std::chrono::steady_clock::now();
            for (slot & s : slots_) {
                if (s.state == slot::BUSY) {
                    if (gate_hold) {
                        s = std::move(work);
                        s.state = slot::READY;
                        s.next_retry = now + std::chrono::milliseconds(100);
                    } else if (!done) {
                        s = std::move(work);
                        s.state = slot::READY;
                        ++s.release_attempts;
                        const std::uint32_t shift =
                            std::min<std::uint32_t>(s.release_attempts, 6);
                        s.next_retry = now + std::chrono::milliseconds(
                            std::min<std::uint64_t>(5000, std::uint64_t(100) << shift));
                    } else if (!released) {
                        // release_backend() kept the backend and set next_retry.
                        s = std::move(work);
                        s.state = slot::READY;
                    } else if (done && work.retain_only && work.has_backend) {
                        // Joint retention: clear succeeded, but the failed
                        // context's scheduler/prefab members may still reference
                        // the backend until unwind completes. Keep it until
                        // shutdown instead of freeing under them.
                        s = std::move(work);
                        s.state = slot::RETAINED;
                    } else if (done) {
                        retired_module = std::move(work.module);
                        s.reset();
                    }
                    cv_.notify_all();
                    handled = true;
                    break;
                }
            }
        }
        // Drop a completed slot's module outside the cleanup lock.
        retired_module.reset();
        if (handled) {
            return;
        }
        // The table cannot lose the BUSY slot: slots_ never shrinks.
        std::fprintf(stderr, "TP5_CLEANUP_OWNER lost a busy record\n");
    }

    void worker_loop() {
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                    if (stopping_) {
                        return true;
                    }
                    const auto now = std::chrono::steady_clock::now();
                    for (const slot & s : slots_) {
                        if (s.state == slot::READY && now >= s.next_retry) {
                            return true;
                        }
                    }
                    return false;
                });
                if (stopping_) {
                    return;
                }
            }
            try {
                process_one_ready(false);
                process_retained_token(false);
                Traits::retry_orphans();
            } catch (...) {
                std::fprintf(stderr, "TP5_CLEANUP_OWNER worker exception\n");
                // A provider callback must not kill the only worker. Keep
                // the loop alive; records remain owned by their slots.
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }

#ifdef TP5_TEST_HOOKS
    std::atomic<int> test_arm_fail_{0};
    std::atomic<int> test_token_fail_{0};
#endif
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<slot> slots_;
    std::thread worker_;
    bool stopping_ = false;
};

} // namespace tq::tp5
