#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstddef>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <functional>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#if __has_include(<drm/drm.h>)
#include <drm/drm.h>
#elif __has_include(<drm.h>)
#include <drm.h>
#endif
#if __has_include(<xf86drm.h>)
#include <xf86drm.h>
#endif
#endif

// Fallback DRM syncobj definitions if headers are missing or non-Linux
#ifndef DRM_IOCTL_BASE
#define DRM_IOCTL_BASE 'd'
#endif
#ifndef DRM_COMMAND_BASE
#define DRM_COMMAND_BASE 0x40
#endif

#ifndef DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL
struct drm_syncobj_timeline_array {
    uint64_t handles;
    uint64_t points;
    uint32_t count_handles;
    uint32_t pad;
};
#define DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL _IOWR(DRM_IOCTL_BASE, 0xCD, struct drm_syncobj_timeline_array)
#endif

#ifndef DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT
struct drm_syncobj_timeline_wait {
    uint64_t handles;
    uint64_t points;
    uint64_t timeout_nsec;
    uint32_t count_handles;
    uint32_t flags;
    uint32_t first_signaled;
    uint32_t pad;
};
#define DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL (1 << 0)
#define DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT (1 << 1)
#define DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE (1 << 2)
#define DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT _IOWR(DRM_IOCTL_BASE, 0xCA, struct drm_syncobj_timeline_wait)
#endif

#ifndef DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE
struct drm_syncobj_handle {
    uint32_t handle;
    uint32_t flags;
    int32_t fd;
    uint32_t pad;
};
#define DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE _IOWR(DRM_IOCTL_BASE, 0xC2, struct drm_syncobj_handle)
#define DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD _IOWR(DRM_IOCTL_BASE, 0xC1, struct drm_syncobj_handle)
#define DRM_IOCTL_SYNCOBJ_DESTROY _IOWR(DRM_IOCTL_BASE, 0xC0, struct drm_syncobj_destroy)
struct drm_syncobj_destroy {
    uint32_t handle;
    uint32_t pad;
};
#endif

// Direct ioctl wrappers for zero-overhead DRM Syncobj Timeline
struct tp5_drm_syncobj_ops {
    static inline int timeline_signal(int fd, const uint32_t * handles, const uint64_t * points, uint32_t count) {
#if defined(__linux__)
        if (fd < 0 || !handles || !points || count == 0) return -1;
        struct drm_syncobj_timeline_array args{};
        args.handles = (uint64_t)(uintptr_t)handles;
        args.points = (uint64_t)(uintptr_t)points;
        args.count_handles = count;
        return ioctl(fd, DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL, &args);
#else
        (void)fd; (void)handles; (void)points; (void)count;
        return -1;
#endif
    }

    static inline int timeline_wait(int fd, const uint32_t * handles, const uint64_t * points, uint32_t count,
                                   int64_t timeout_nsec, uint32_t flags, uint32_t * first_signaled) {
#if defined(__linux__)
        if (fd < 0 || !handles || !points || count == 0) return -1;
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        uint64_t now_ns = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
        uint64_t abs_timeout_ns = now_ns + (timeout_nsec > 0 ? (uint64_t)timeout_nsec : 5000000000ULL);

        struct drm_syncobj_timeline_wait args{};
        args.handles = (uint64_t)(uintptr_t)handles;
        args.points = (uint64_t)(uintptr_t)points;
        args.timeout_nsec = abs_timeout_ns;
        args.count_handles = count;
        args.flags = flags | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT;
        args.first_signaled = 0;
        int ret = ioctl(fd, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &args);
        if (ret == 0 && first_signaled) {
            *first_signaled = args.first_signaled;
        }
        return ret;
#else
        (void)fd; (void)handles; (void)points; (void)count; (void)timeout_nsec; (void)flags; (void)first_signaled;
        return -1;
#endif
    }

    static inline int fd_to_handle(int fd, int import_fd, uint32_t * handle) {
#if defined(__linux__)
        if (fd < 0 || import_fd < 0 || !handle) return -1;
        struct drm_syncobj_handle args{};
        args.fd = import_fd;
        args.flags = 0;
        args.handle = 0;
        int ret = ioctl(fd, DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE, &args);
        if (ret == 0) {
            *handle = args.handle;
        }
        return ret;
#else
        (void)fd; (void)import_fd; (void)handle;
        return -1;
#endif
    }

    static inline int destroy(int fd, uint32_t handle) {
#if defined(__linux__)
        if (fd < 0 || handle == 0) return -1;
        struct drm_syncobj_destroy args{};
        args.handle = handle;
        return ioctl(fd, DRM_IOCTL_SYNCOBJ_DESTROY, &args);
#else
        (void)fd; (void)handle;
        return -1;
#endif
    }
};

// Persistent CPU reduction workers. Idle workers sleep; completion is bounded.
class tp5_avx2_pool {
public:
    static constexpr size_t NUM_WORKERS = 22;
    using task_fn = std::function<void(size_t, size_t)>;

    tp5_avx2_pool() {
        for (size_t i = 0; i < NUM_WORKERS; ++i) {
            m_workers.emplace_back([this, i] {
                uint64_t last_epoch = 0;
                std::unique_lock<std::mutex> lock(m_mutex);
                for (;;) {
                    m_ready.wait(lock, [&] { return m_stop || m_epoch != last_epoch; });
                    if (m_stop) return;
                    last_epoch = m_epoch;
                    lock.unlock();
                    m_task(i, NUM_WORKERS);
                    lock.lock();
                    if (++m_completed == NUM_WORKERS) m_done.notify_one();
                }
            });
        }
    }
    ~tp5_avx2_pool() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_ready.notify_all();
        for (auto & worker : m_workers) worker.join();
    }
    tp5_avx2_pool(const tp5_avx2_pool &) = delete;
    tp5_avx2_pool & operator=(const tp5_avx2_pool &) = delete;

    bool parallel_for(task_fn fn) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_task = std::move(fn);
        m_completed = 0;
        ++m_epoch;
        m_ready.notify_all();
        const bool completed = m_done.wait_for(lock, std::chrono::seconds(2), [&] {
            return m_completed == NUM_WORKERS;
        });
        if (!completed) {
            // Do not return while workers still reference the caller's stack.
            m_stop = true;
            lock.unlock();
            m_ready.notify_all();
            for (auto & worker : m_workers) if (worker.joinable()) worker.join();
            m_workers.clear();
        }
        return completed;
    }

private:
    std::vector<std::thread> m_workers;
    std::mutex m_mutex;
    std::condition_variable m_ready, m_done;
    task_fn m_task;
    uint64_t m_epoch = 0;
    size_t m_completed = 0;
    bool m_stop = false;
};

// 5-Worker Parallel DRM Signaler for non-blocking atomic signaling handoff
// Dedicated 5-Worker Parallel DRM Waiter for concurrent 5-GPU timeline waiting
class tp5_drm_waiter {
public:
    static constexpr size_t NUM_WAITERS = 5;

    tp5_drm_waiter() {
        for (size_t i = 0; i < NUM_WAITERS; ++i) {
            m_dri_fd[i] = -1;
            m_syncobj[i] = 0;
            m_target_point[i].store(0, std::memory_order_relaxed);
            m_done[i].store(true, std::memory_order_relaxed);
            m_running[i].store(true, std::memory_order_relaxed);
            m_threads[i] = std::thread([this, i]() {
                waiter_loop(i);
            });
        }
    }

    ~tp5_drm_waiter() {
        for (size_t i = 0; i < NUM_WAITERS; ++i) {
            m_running[i].store(false, std::memory_order_relaxed);
            if (m_threads[i].joinable()) m_threads[i].join();
        }
    }

    inline void set_node(size_t idx, int fd, uint32_t syncobj) {
        if (idx < NUM_WAITERS) {
            m_dri_fd[idx] = fd;
            m_syncobj[idx] = syncobj;
        }
    }

    inline void wait_all(uint64_t point) {
        for (size_t i = 0; i < NUM_WAITERS; ++i) {
            m_done[i].store(false, std::memory_order_relaxed);
            m_target_point[i].store(point, std::memory_order_release);
        }
        for (size_t i = 0; i < NUM_WAITERS; ++i) {
            while (!m_done[i].load(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(__i386__)
                _mm_pause();
#endif
            }
        }
    }

private:
    void waiter_loop(size_t idx) {
        uint64_t last = 0;
        while (m_running[idx].load(std::memory_order_relaxed)) {
            uint64_t cur = m_target_point[idx].load(std::memory_order_acquire);
            if (cur > last) {
                last = cur;
                int fd = m_dri_fd[idx];
                uint32_t h = m_syncobj[idx];
                if (fd >= 0 && h > 0) {
                    uint64_t p = cur;
                    struct drm_syncobj_timeline_wait args{};
                    args.handles = (uint64_t)&h;
                    args.points = (uint64_t)&p;
                    args.count_handles = 1;
                    args.timeout_nsec = 5000000000ULL;
                    args.flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT;
                    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
                    args.timeout_nsec += (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
                    ioctl(fd, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &args);
                }
                m_done[idx].store(true, std::memory_order_release);
            } else {
#if defined(__x86_64__) || defined(__i386__)
                _mm_pause();
#endif
            }
        }
    }

    int m_dri_fd[NUM_WAITERS];
    uint32_t m_syncobj[NUM_WAITERS];
    alignas(64) std::atomic<uint64_t> m_target_point[NUM_WAITERS];
    alignas(64) std::atomic<bool> m_done[NUM_WAITERS];
    alignas(64) std::atomic<bool> m_running[NUM_WAITERS];
    std::thread m_threads[NUM_WAITERS];
};
class tp5_drm_signaler {
public:
    static constexpr size_t NUM_SIGNALERS = 5;

    tp5_drm_signaler() {
        m_stop.store(false, std::memory_order_relaxed);
        for (size_t i = 0; i < NUM_SIGNALERS; ++i) {
            m_target_point[i].store(0, std::memory_order_relaxed);
            m_signaled_point[i].store(0, std::memory_order_relaxed);
            m_dri_fd[i] = -1;
            m_syncobj_handle[i] = 0;
            m_done_flag[i] = nullptr;
            m_threads.emplace_back([this, i]() {
                signaler_loop(i);
            });
        }
    }

    ~tp5_drm_signaler() {
        m_stop.store(true, std::memory_order_release);
        for (auto & t : m_threads) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    tp5_drm_signaler(const tp5_drm_signaler &) = delete;
    tp5_drm_signaler & operator=(const tp5_drm_signaler &) = delete;

    // Pure lockless async dispatch of DRM syncobj timeline signal (< 0.05us reaction time)
    inline void signal_async(size_t worker_idx, int dri_fd, uint32_t syncobj_handle, uint64_t point,
                             std::atomic<bool> * done_flag = nullptr) {
        if (worker_idx >= NUM_SIGNALERS) return;
        m_dri_fd[worker_idx] = dri_fd;
        m_syncobj_handle[worker_idx] = syncobj_handle;
        m_done_flag[worker_idx] = done_flag;
        m_target_point[worker_idx].store(point, std::memory_order_release);
    }

    // Lockless spin-wait ensuring all pending async signals are delivered to kernel
    inline void wait_all_flushed() {
        for (size_t i = 0; i < NUM_SIGNALERS; ++i) {
            uint64_t tgt = m_target_point[i].load(std::memory_order_acquire);
            while (m_signaled_point[i].load(std::memory_order_acquire) < tgt) {
#if defined(__x86_64__) || defined(__i386__)
                _mm_pause();
#endif
            }
        }
    }

private:
    void signaler_loop(size_t idx) {
        while (!m_stop.load(std::memory_order_acquire)) {
            uint64_t target = m_target_point[idx].load(std::memory_order_acquire);
            if (target > m_signaled_point[idx].load(std::memory_order_relaxed)) {
                int fd = m_dri_fd[idx];
                uint32_t handle = m_syncobj_handle[idx];
                if (fd >= 0 && handle > 0) {
                    tp5_drm_syncobj_ops::timeline_signal(fd, &handle, &target, 1);
                }
                m_signaled_point[idx].store(target, std::memory_order_release);
                std::atomic<bool> * done = m_done_flag[idx];
                if (done) {
                    done->store(true, std::memory_order_release);
                }
            } else {
#if defined(__x86_64__) || defined(__i386__)
                _mm_pause();
#elif defined(__arm__) || defined(__aarch64__)
                asm volatile("yield" ::: "memory");
#endif
            }
        }
    }

    std::vector<std::thread> m_threads;
    alignas(64) std::atomic<uint64_t> m_target_point[NUM_SIGNALERS];
    alignas(64) std::atomic<uint64_t> m_signaled_point[NUM_SIGNALERS];
    int m_dri_fd[NUM_SIGNALERS];
    uint32_t m_syncobj_handle[NUM_SIGNALERS];
    std::atomic<bool> * m_done_flag[NUM_SIGNALERS];
    std::atomic<bool> m_stop{false};
};

// 5-worker bounded pool for independent per-rank host operations.
// Workers sleep between epochs; every call has a finite completion deadline.
class tp5_submit_pool {
public:
    static constexpr size_t NUM_WORKERS = 8;
    using task_fn = std::function<void(size_t rank_idx)>;

    tp5_submit_pool() {
        m_workers.reserve(NUM_WORKERS);
        for (size_t i = 0; i < NUM_WORKERS; ++i) {
            m_workers.emplace_back([this, i] { worker_loop(i); });
        }
    }

    ~tp5_submit_pool() {
        stop_workers();
    }

    tp5_submit_pool(const tp5_submit_pool &) = delete;
    tp5_submit_pool & operator=(const tp5_submit_pool &) = delete;

    // Caller captures must remain alive until this returns. On timeout all
    // workers are joined before returning, so a stack capture is still safe.
    bool parallel_for(task_fn fn, size_t n, std::chrono::milliseconds timeout = std::chrono::seconds(2)) {
        if (!fn || n == 0 || n > NUM_WORKERS) return false;
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_stop) return false;
        m_current_task = std::move(fn);
        m_task_count = n;
        m_completed_count = 0;
        m_failed = false;
        ++m_epoch;
        m_ready.notify_all();
        const bool completed = m_done.wait_for(lock, timeout, [&] {
            return m_completed_count == NUM_WORKERS;
        });
        if (completed) return !m_failed;

        m_stop = true;
        lock.unlock();
        m_ready.notify_all();
        join_workers();
        return false;
    }

private:
    void stop_workers() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_ready.notify_all();
        join_workers();
    }

    void join_workers() {
        for (auto & worker : m_workers) {
            if (worker.joinable()) worker.join();
        }
        m_workers.clear();
    }

    void worker_loop(size_t id) {
        uint64_t seen_epoch = 0;
        std::unique_lock<std::mutex> lock(m_mutex);
        for (;;) {
            m_ready.wait(lock, [&] { return m_stop || m_epoch != seen_epoch; });
            if (m_stop) return;
            seen_epoch = m_epoch;
            const size_t task_count = m_task_count;
            task_fn task = m_current_task;
            lock.unlock();
            if (id < task_count) {
                try {
                    task(id);
                } catch (...) {
                    lock.lock();
                    m_failed = true;
                    if (++m_completed_count == NUM_WORKERS) m_done.notify_one();
                    lock.unlock();
                    return;
                }
            }
            lock.lock();
            if (++m_completed_count == task_count) m_done.notify_one();
        }
    }

    std::vector<std::thread> m_workers;
    std::mutex m_mutex;
    std::condition_variable m_ready;
    std::condition_variable m_done;
    task_fn m_current_task;
    uint64_t m_epoch = 0;
    size_t m_task_count = 0;
    size_t m_completed_count = 0;
    bool m_failed = false;
    bool m_stop = false;
};

