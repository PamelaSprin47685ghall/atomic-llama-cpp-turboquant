#pragma once

// Per-run retained staging owners. A run whose bounded drain did not prove
// completion keeps its own contexts/backends/buffers and tokens until a later
// retry drains and releases them. The records are fixed-size, so retaining
// cannot fail after an allocation. The retainer instance must be
// process-lifetime (the runtime leaks it); a background thread and an atexit
// retry re-probe delayed completions.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <thread>
#include <utility>

#include "llama.h"
#include <vector>

#include <ggml-backend.h>

namespace tq::tp5 {

constexpr std::size_t kMaxRetainedRuns   = 4;
constexpr std::size_t kMaxRetainedTokens = 4;
constexpr std::size_t kMaxRetainedModels = 3;
// A shutdown retain blocks the exiting thread for at most this long while it
// probes a real completion; after the bound the owner stays process-owned.
constexpr uint32_t kShutdownDrainDeadlineMs = 1000;

class retainer_base {
public:
    virtual ~retainer_base() = default;
    virtual void retry() = 0;
    virtual void stop() = 0;
    virtual void shutdown() = 0;
    virtual bool has_retained() const = 0;
};

inline std::vector<std::shared_ptr<retainer_base>> & owner_core_registry() {
    static auto * registry = new std::vector<std::shared_ptr<retainer_base>>();
    return *registry;
}

inline std::mutex & owner_core_mutex() {
    static auto * mutex = new std::mutex();
    return *mutex;
}

inline bool register_owner_core(const std::shared_ptr<retainer_base> & core) {
    static std::once_flag once;
    std::call_once(once, [] {
        std::atexit([] {
            std::vector<std::shared_ptr<retainer_base>> cores;
            {
                std::lock_guard<std::mutex> lock(owner_core_mutex());
                cores.swap(owner_core_registry());
            }
            for (const auto & core : cores) {
                if (core) {
                    core->retry();
                    core->shutdown();
                }
            }
        });
    });
    std::lock_guard<std::mutex> lock(owner_core_mutex());
    auto & registry = owner_core_registry();
    registry.erase(std::remove_if(registry.begin(), registry.end(),
                                  [](const std::shared_ptr<retainer_base> & slot) {
                                      return !slot || !slot->has_retained();
                                  }),
                   registry.end());
    try {
        registry.push_back(core);
    } catch (...) {
        return false;
    }
    return true;
}

inline void retry_owner_cores() {
    std::vector<std::shared_ptr<retainer_base>> cores;
    {
        std::lock_guard<std::mutex> lock(owner_core_mutex());
        cores = owner_core_registry();
    }
    for (const auto & core : cores) {
        if (core) {
            core->retry();
        }
    }
}

struct owner_record {
    std::uint64_t owner = 0;
    std::uint64_t generation = 0;
    std::uint32_t window = 0;
};

template <typename ContextPtr, typename ModelPtr>
class staging_owner_retainer;

// Shared owner core handle used by run_lease. The retainer derives from this;
// a lease keeps the retainer alive for the lease lifetime.
class lease_core {
public:
    virtual ~lease_core() = default;
    virtual void cancel_slot(std::size_t slot, std::uint64_t generation) = 0;
};

// Owner-issued, move-only lease. The slot is reserved before owner
// acquisition; only the retainer can construct a valid lease. Copying and
// forging are deleted. Destruction, move assignment and reset cancel the
// owned slot through the shared core; commit/cancel consume it once.
class run_lease {
public:
    run_lease() = default;
    ~run_lease() { release_owned(); }
    run_lease(const run_lease &) = delete;
    run_lease & operator=(const run_lease &) = delete;
    run_lease(run_lease && other) noexcept { take_from(other); }
    run_lease & operator=(run_lease && other) noexcept {
        if (this != &other) {
            release_owned();
            take_from(other);
        }
        return *this;
    }
    bool valid() const {
        return core_ != nullptr && slot_ != SIZE_MAX && generation_ != 0;
    }
    std::size_t slot() const { return slot_; }
    std::uint64_t generation() const { return generation_; }
    lease_core * core() const { return core_.get(); }
private:
    template <typename, typename> friend class staging_owner_retainer;
    run_lease(std::shared_ptr<lease_core> core, std::size_t slot, std::uint64_t generation)
        : core_(std::move(core)), slot_(slot), generation_(generation) {}

    struct taken { std::shared_ptr<lease_core> core; std::size_t slot; std::uint64_t generation; };
    taken take() noexcept {
        taken t{core_, slot_, generation_};
        clear();
        return t;
    }
    void take_from(run_lease & other) noexcept {
        taken t = other.take();
        core_ = std::move(t.core); slot_ = t.slot; generation_ = t.generation;
    }
    void clear() noexcept { core_.reset(); slot_ = SIZE_MAX; generation_ = 0; }
    void release_owned() noexcept {
        if (valid()) {
            core_->cancel_slot(slot_, generation_);
        }
        clear();
    }

    std::shared_ptr<lease_core> core_;
    std::size_t slot_ = SIZE_MAX;
    std::uint64_t generation_ = 0;
};

template <typename ContextPtr, typename ModelPtr>
class staging_owner_retainer : public retainer_base,
                               public lease_core,
                               public std::enable_shared_from_this<staging_owner_retainer<ContextPtr, ModelPtr>> {
    struct run {
        // Dynamic per actual provider/token set. Allocated before contexts
        // and native handles are moved into the record.
        std::vector<owner_record> tokens;
        bool reserved = false;
        std::uint64_t reservation_generation = 0;
        // Declared before the contexts so it is destroyed after them.
        std::array<std::optional<ModelPtr>, kMaxRetainedModels> models;
        std::vector<ContextPtr>     contexts;
        std::vector<ContextPtr>     mtp;
        std::vector<ggml_backend_t> native;
    };

    std::mutex mutex_;
    std::mutex retry_mutex_;
    std::mutex thread_mutex_;
    std::array<std::optional<run>, kMaxRetainedRuns> runs_;
    std::uint64_t next_reservation_generation_ = 1;
    std::condition_variable slot_cv_;
    std::atomic<bool> retry_running_{false};
    std::atomic<bool> stop_{false};
    std::atomic<bool> drain_stop_{false};
    std::atomic<bool> drain_running_{false};
    std::atomic<bool> shutdown_{false};
    std::thread retry_thread_;
    std::thread drain_thread_;

public:
    staging_owner_retainer() = default;
    ~staging_owner_retainer() override {
        drain_stop_.store(true, std::memory_order_release);
        stop_.store(true, std::memory_order_release);
        slot_cv_.notify_all();
        {
            std::lock_guard<std::mutex> lock(thread_mutex_);
            if (retry_thread_.joinable()) {
                retry_thread_.join();
            }
            if (drain_thread_.joinable()) {
                drain_thread_.join();
            }
        }
        try {
            // Bounded final attempt before runs_ destruction. Failing owners
            // stay process-owned only after this retry.
            retry_until(std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(100));
        } catch (...) {
        }
    }
    static std::shared_ptr<staging_owner_retainer> create() {
        std::shared_ptr<staging_owner_retainer> self(new staging_owner_retainer());
        // The registry is bounded; a full registry must not turn a legal
        // create() into a null return. The caller still owns this core.
        (void) register_owner_core(std::static_pointer_cast<retainer_base>(self));
        return self;
    }
    staging_owner_retainer(const staging_owner_retainer &) = delete;
    staging_owner_retainer & operator=(const staging_owner_retainer &) = delete;

    void stop() override {
        drain_stop_.store(true, std::memory_order_release);
        stop_.store(true, std::memory_order_release);
        slot_cv_.notify_all();
        std::lock_guard<std::mutex> lock(thread_mutex_);
        if (retry_thread_.joinable()) {
            retry_thread_.join();
        }
        if (drain_thread_.joinable()) {
            drain_thread_.join();
        }
    }

    void shutdown() override {
        shutdown_.store(true, std::memory_order_release);
        drain_stop_.store(true, std::memory_order_release);
        stop_.store(true, std::memory_order_release);
        slot_cv_.notify_all();
        {
            std::lock_guard<std::mutex> lock(thread_mutex_);
            if (retry_thread_.joinable()) {
                retry_thread_.join();
            }
            if (drain_thread_.joinable()) {
                drain_thread_.join();
            }
        }
        // The worker is gone; the exiting thread owns a bounded final probe.
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(kShutdownDrainDeadlineMs);
        while (std::chrono::steady_clock::now() < deadline) {
            retry_until(deadline);
            if (retained_tokens() == 0) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Reserve an exclusive run slot and all token/context/native capacity
    // before the caller acquires owners or creates native submissions.
    run_lease reserve_run(std::size_t max_tokens, std::size_t max_contexts,
                          std::size_t max_mtp) {
        run_lease out;
        if (shutdown_.load(std::memory_order_acquire)) {
            return out;
        }
        const char * fail = std::getenv("TP5_RETAINER_FAIL_RESERVE");
        if (fail != nullptr && std::strcmp(fail, "1") == 0) {
            return out;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::optional<run> * slot = nullptr;
            for (auto & r : runs_) {
                if (!r) {
                    slot = &r;
                    break;
                }
            }
            if (slot == nullptr) {
                return out;
            }
            slot->emplace();
            run & r = **slot;
            r.reserved = true;
            r.reservation_generation = next_reservation_generation_++;
            try {
                r.tokens.reserve(max_tokens);
                r.contexts.reserve(max_contexts);
                r.mtp.reserve(max_mtp);
                r.native.reserve(max_contexts);
            } catch (...) {
                slot->reset();
                slot_cv_.notify_all();
                return out;
            }
            try {
                auto core = std::static_pointer_cast<lease_core>(this->shared_from_this());
                out = run_lease(std::move(core), std::size_t(slot - runs_.data()),
                                r.reservation_generation);
            } catch (...) {
                slot->reset();
                slot_cv_.notify_all();
                return run_lease{};
            }
        }
        // The slot exists before the worker starts, so an idle worker cannot
        // miss the reservation and exit before it is visible.
        if (stop_.exchange(false, std::memory_order_acq_rel) || !retry_running_.load(std::memory_order_acquire)) {
            bool worker_ok = false;
            try {
                worker_ok = ensure_retry_thread();
            } catch (...) {
                worker_ok = false;
            }
            if (!worker_ok) {
                cancel_run(std::move(out));
                return {};
            }
        }
        if (shutdown_.load(std::memory_order_acquire)) {
            cancel_run(std::move(out));
            return {};
        }
        return out;
    }

    void cancel_run(run_lease && res) {
        if (!res.valid()) {
            return;
        }
        if (res.core() != this) {
            std::fprintf(stderr, "TP5_STAGING_OWNER rejected a foreign lease\n");
            return;
        }
        const std::size_t slot = res.slot();
        const std::uint64_t generation = res.generation();
        cancel_slot(slot, generation); // may throw on lock; lease stays valid
        res.take();
    }

    void cancel_slot(std::size_t slot, std::uint64_t generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (slot < runs_.size() && runs_[slot] && runs_[slot]->reserved &&
                runs_[slot]->reservation_generation == generation) {
            runs_[slot].reset();
            slot_cv_.notify_all();
        }
    }

    void commit_run(run_lease && res,
                    std::vector<ContextPtr> * contexts,
                    std::vector<ContextPtr> * mtp,
                    std::vector<ggml_backend_t> * native,
                    const std::array<ModelPtr *, kMaxRetainedModels> & models,
                    std::vector<owner_record> * tokens) {
        if (!res.valid() || tokens == nullptr) {
            std::fprintf(stderr, "TP5_STAGING_OWNER rejected an invalid lease\n");
            return;
        }
        if (res.core() != this) {
            std::fprintf(stderr, "TP5_STAGING_OWNER rejected a foreign lease\n");
            return;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        const std::size_t slot = res.slot();
        const std::uint64_t generation = res.generation();
        if (!runs_[slot] || !runs_[slot]->reserved ||
                runs_[slot]->reservation_generation != generation) {
            std::fprintf(stderr, "TP5_STAGING_OWNER rejected a stale lease\n");
            return;
        }
        res.take();
        run & r = *runs_[slot];
        // The caller's vectors are the movable objects: move assignment is
        // allocation-free and cannot fail. There is no capacity copy and no
        // second allocation to lose the owner on.
        r.tokens = std::move(*tokens);
        if (contexts != nullptr && !contexts->empty()) {
            r.contexts = std::move(*contexts);
        }
        if (mtp != nullptr && !mtp->empty()) {
            r.mtp = std::move(*mtp);
        }
        if (native != nullptr && !native->empty()) {
            r.native = std::move(*native);
        }
        for (std::size_t i = 0; i < kMaxRetainedModels; ++i) {
            if (models[i] != nullptr && *models[i]) {
                r.models[i].emplace(std::move(*models[i]));
            }
        }
        r.reserved = false;
        const bool shutting_down = shutdown_.load(std::memory_order_acquire);
        slot_cv_.notify_all();
        lock.unlock();
        if (shutting_down) {
            std::fprintf(stderr,
                "TP5_STAGING_OWNER retained after shutdown; bounded synchronous retry\n");
            retry_until(std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(kShutdownDrainDeadlineMs));
            if (retained_tokens() != 0) {
                start_shutdown_drain();
            }
        }
    }

    void start_shutdown_drain() {
        if (drain_running_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        std::lock_guard<std::mutex> thread_lock(thread_mutex_);
        if (drain_thread_.joinable()) {
            drain_thread_.join();
        }
        if (shutdown_.load(std::memory_order_acquire)) {
            drain_running_.store(false, std::memory_order_release);
            return;
        }
        drain_stop_.store(false, std::memory_order_release);
        try {
            drain_thread_ = std::thread([this] {
                while (!drain_stop_.load(std::memory_order_acquire) &&
                       retained_tokens() != 0) {
                    retry();
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                drain_running_.store(false, std::memory_order_release);
            });
        } catch (...) {
            drain_running_.store(false, std::memory_order_release);
            std::fprintf(stderr, "TP5_STAGING_OWNER shutdown drain thread failed\n");
        }
    }

    // Moves one run's contexts/backends and its busy tokens into a new record.
    // With wait_for_slot the caller applies backpressure until a retry frees a
    // record; it never releases the owner. Without it, false means no record
    // was taken and the caller still owns the resources.
    bool retain_run(std::vector<ContextPtr> * contexts,
                    std::vector<ContextPtr> * mtp,
                    std::vector<ggml_backend_t> * native,
                    const std::array<ModelPtr *, kMaxRetainedModels> & models,
                    const std::uint64_t * owners,
                    const std::uint64_t * generations,
                    std::size_t count,
                    bool wait_for_slot = true,
                    const std::uint32_t * windows = nullptr) {
        if (count == 0) {
            return true;
        }
        const bool shutting_down = shutdown_.load(std::memory_order_acquire);
        // A retain after stop restarts the worker; a stopped retainer must not
        // silently enqueue an owner that no thread will ever re-probe. After
        // shutdown the record is still moved (process-owned); atexit retry may
        // drain it and no owner is destroyed.
        if (!shutting_down) {
            if (stop_.exchange(false, std::memory_order_acq_rel)) {
                if (!ensure_retry_thread()) {
                    return false;
                }
            }
            // A worker must exist before any run/owner is moved into a slot.
            if (!ensure_retry_thread()) {
                return false;
            }
        }
        {
            std::unique_lock<std::mutex> lock(mutex_);
            auto find_slot = [&]() -> std::optional<run> * {
                for (auto & r : runs_) {
                    if (!r) {
                        return &r;
                    }
                }
                return nullptr;
            };
            std::optional<run> * slot = nullptr;
            if (wait_for_slot) {
                for (;;) {
                    if (find_slot() != nullptr) {
                        break;
                    }
                    if (shutdown_.load(std::memory_order_acquire)) {
                        break;
                    }
                    if (stop_.load(std::memory_order_acquire)) {
                        lock.unlock();
                        stop_.store(false, std::memory_order_release);
                        if (!ensure_retry_thread()) {
                            return false;
                        }
                        lock.lock();
                        continue;
                    }
                    slot_cv_.wait(lock, [&] {
                        return find_slot() != nullptr ||
                               stop_.load(std::memory_order_acquire) ||
                               shutdown_.load(std::memory_order_acquire);
                    });
                }
            }
            slot = find_slot();
            if (slot == nullptr) {
                return false;
            }
            slot->emplace();
            run & r = **slot;
            try {
                r.tokens.resize(count);
                for (std::size_t i = 0; i < count; ++i) {
                    r.tokens[i].owner = owners[i];
                    r.tokens[i].generation = generations[i];
                    r.tokens[i].window = windows != nullptr ? windows[i] : uint32_t(i);
                }
            } catch (...) {
                slot->reset();
                return false;
            }
            if (contexts != nullptr && !contexts->empty()) {
                r.contexts = std::move(*contexts);
            }
            if (mtp != nullptr && !mtp->empty()) {
                r.mtp = std::move(*mtp);
            }
            if (native != nullptr && !native->empty()) {
                r.native = std::move(*native);
            }
            for (std::size_t i = 0; i < kMaxRetainedModels; ++i) {
                if (models[i] != nullptr && *models[i]) {
                    r.models[i].emplace(std::move(*models[i]));
                }
            }
        }
        if (shutting_down) {
            // No worker will be started after shutdown; make one bounded
            // attempt now and leave the record in the process-owned core for
            // a later retry_owner_cores()/atexit pass.
            retry_until(std::chrono::steady_clock::now() + std::chrono::milliseconds(100));
        }
        return true;
    }

    void retry() override {
        retry_until(std::chrono::steady_clock::now() + std::chrono::milliseconds(100));
    }

    void retry_until(std::chrono::steady_clock::time_point deadline) {
        // A provider callback may re-enter retry()/retry_until() on the same
        // thread. Use a thread-local marker so another thread (including a
        // shutdown probe) can still take the try_lock when the pass is free.
        static thread_local const void * t_retrying = nullptr;
        if (t_retrying == this) {
            return;
        }
        struct retry_guard {
            const void *& slot;
            ~retry_guard() { slot = nullptr; }
        } active{t_retrying};
        t_retrying = this;
        // Orphan and retained-pin passes take provider/pin locks. Run them
        // before retry_mutex_, so the per-retainer pass lock stays a leaf in
        // the lock order and a re-entrant callback returns above.
        ggml_backend_meta_staging_owner_retry_orphans();
        ggml_backend_retry_retained();
        std::unique_lock<std::mutex> retry_lock(retry_mutex_, std::try_to_lock);
        if (!retry_lock.owns_lock()) {
            return;
        }
        bool expired = false;
        for (auto & run_opt : runs_) {
            std::size_t n = 0;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!run_opt || run_opt->reserved) {
                    continue;
                }
                n = run_opt->tokens.size();
            }
            for (std::size_t i = 0; i < n; ++i) {
                std::uint64_t owner = 0;
                std::uint64_t generation = 0;
                std::uint32_t window = 0;
                llama_context * context = nullptr;
                llama_context * mtp_context = nullptr;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (!run_opt || i >= run_opt->tokens.size()) {
                        break;
                    }
                    owner = run_opt->tokens[i].owner;
                    generation = run_opt->tokens[i].generation;
                    window = run_opt->tokens[i].window;
                    if (window < run_opt->contexts.size()) {
                        context = run_opt->contexts[window].get();
                    }
                    if (window < run_opt->mtp.size()) {
                        mtp_context = run_opt->mtp[window].get();
                    }
                }
                if (owner == 0) {
                    continue;
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    expired = true;
                    break;
                }
                int drained = ggml_backend_meta_staging_owner_drain(owner, generation);
                if (drained != 0) {
                    std::fprintf(stderr,
                        "TP5_STAGING_OWNER retained drain result=%d; owner retained\n", drained);
                    continue;
                }
                // Context/provider calls run without mutex_. The run keeps
                // the contexts alive; only retry_until resets an empty run,
                // after this loop.
                if (context) {
                    (void) llama_tp5_context_staging_unbind(context);
                }
                if (mtp_context) {
                    (void) llama_tp5_context_staging_unbind(mtp_context);
                }
                if (!ggml_backend_meta_staging_owner_release(owner, generation)) {
                    std::fprintf(stderr,
                        "TP5_STAGING_OWNER release failed; owner retained\n");
                    continue;
                }
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (run_opt && i < run_opt->tokens.size() &&
                            run_opt->tokens[i].owner == owner) {
                        run_opt->tokens[i].owner = 0;
                    }
                }
            }
            std::optional<run> retired;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!run_opt) {
                    continue;
                }
                bool any = false;
                for (const owner_record & token : run_opt->tokens) {
                    any = any || token.owner != 0;
                }
                if (!any) {
                    // Move the run out and destroy its contexts/models after
                    // unlocking. Destruction can call llama_free/provider code.
                    retired = std::move(*run_opt);
                    run_opt.reset();
                    slot_cv_.notify_all();
                }
            }
            if (expired) {
                break;
            }
        }
    }

    bool has_retained() const override {
        return retained_runs() != 0 || retained_tokens() != 0;
    }

    bool is_shutdown() const {
        return shutdown_.load(std::memory_order_acquire);
    }

    std::size_t retained_runs() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t n = 0;
        for (const auto & r : runs_) {
            n += r.has_value() ? 1 : 0;
        }
        return n;
    }

    std::size_t retained_tokens() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t n = 0;
        for (const auto & r : runs_) {
            if (!r) {
                continue;
            }
            for (const owner_record & token : r->tokens) {
                n += token.owner != 0 ? 1 : 0;
            }
        }
        return n;
    }

private:
    bool ensure_retry_thread() {
        const char * fail = std::getenv("TP5_RETAINER_FAIL_THREAD");
        if (fail != nullptr && std::strcmp(fail, "1") == 0) {
            return false;
        }
        std::lock_guard<std::mutex> thread_lock(thread_mutex_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (shutdown_.load(std::memory_order_acquire)) {
                return false;
            }
            if (retry_running_) {
                return true;
            }
            if (retry_thread_.joinable()) {
                retry_thread_.join();
            }
            retry_running_ = true;
        }
        try {
            retry_thread_ = std::thread([this] {
                // One bounded worker, 100 ms poll. It runs while retained
                // tokens exist and exits when they drain or stop/shutdown is
                // requested; it does not depend on a future run.
                while (!stop_.load(std::memory_order_acquire)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    if (stop_.load(std::memory_order_acquire)) {
                        break;
                    }
                    retry();
                    std::lock_guard<std::mutex> lock(mutex_);
                    bool any = false;
                    for (const auto & r : runs_) {
                        if (!r) {
                            continue;
                        }
                        if (r->reserved) {
                            any = true;
                            break;
                        }
                        for (const owner_record & token : r->tokens) {
                            any = any || token.owner != 0;
                        }
                    }
                    if (!any) {
                        retry_running_ = false;
                        return;
                    }
                }
                std::lock_guard<std::mutex> lock(mutex_);
                retry_running_ = false;
            });
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            retry_running_ = false;
            return false;
        }
        return true;
    }
};

} // namespace tq::tp5
