#pragma once

// TP5 WS4: persistent pinned host-to-device staging pool.
//
// A lease is one pinned slot. The caller copies host data into the slot once;
// every rank then reads the same pinned address. The pool holds no GPU types,
// so a CPU test can exercise it with injected primitives.

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "ggml-backend.h"

namespace ggml_cuda_staging {

constexpr size_t   kMaxPoolBytes = 4u << 20;
constexpr uint32_t kRanks        = 16;  // == GGML_CUDA_MAX_DEVICES
constexpr uint32_t kWindows      = 4;
constexpr uint32_t kSlotsPerKey  = 2;   // double buffer

enum class kind : uint32_t {
    tokens  = 0,
    pos     = 1,
    out_ids = 2,
    mtp_h   = 3,
    kq_mask = 4,
    generic = 5,
    count   = 6,
};

constexpr size_t kKindCount = size_t(kind::count);

// Maps a graph-input tensor name to a slot kind. Names that do not match keep
// kind::generic; the meta layer still stages them when they carry the input
// flag, so an unnamed input (for example the MTP token tensor) is not lost.
// Model weights never match these names.
inline kind classify_name(const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return kind::generic;
    }
    if (std::strstr(name, "out_ids") != nullptr) {
        return kind::out_ids;
    }
    if (std::strstr(name, "kq_mask") != nullptr || std::strstr(name, "KQ_mask") != nullptr) {
        return kind::kq_mask;
    }
    if (std::strstr(name, "inp_tokens") != nullptr) {
        return kind::tokens;
    }
    if (std::strstr(name, "inp_pos") != nullptr) {
        return kind::pos;
    }
    if (std::strcmp(name, "inp_h") == 0 || std::strstr(name, "mtp_h_input") != nullptr) {
        return kind::mtp_h;
    }
    return kind::generic;
}

constexpr size_t kGenericMaxBytes = 64u << 10;

// A staging candidate is a graph input (the flag excludes weights and other
// one-shot uploads) within the per-request size bound. Generic kinds are
// allowed: unnamed inputs still rotate through their own slots, but only when
// small, so a large unnamed block cannot crowd out the dedicated kinds.
inline bool staging_candidate(bool has_input_flag, const char * name, size_t size,
                              size_t max_bytes, kind * out) {
    if (out == nullptr || !has_input_flag || size == 0 || size > max_bytes) {
        return false;
    }
    const kind k = classify_name(name);
    if (k == kind::generic && size > kGenericMaxBytes) {
        return false;
    }
    *out = k;
    return true;
}

struct lease {
    uint8_t * host   = nullptr;
    size_t    cap    = 0;
    kind      which  = kind::generic;
    uint32_t  window = 0;
    uint32_t  slot   = 0;
    uint64_t  epoch  = 0;
};

// Primitives supplied by the GPU backend. The pool calls them with its own
// lock held; they must not call back into the pool.
struct backend_ops {
    bool (*host_alloc)(size_t bytes, void ** out)                   = nullptr;
    void (*host_free)(void * ptr)                                   = nullptr;
    bool (*event_create)(void ** out)                               = nullptr;
    bool (*event_record)(void * event, void * stream)               = nullptr;
    bool (*event_sync)(void * event)                                = nullptr;
    void (*event_destroy)(void * event)                             = nullptr;
    bool (*copy_h2d)(void * dst, const void * src, size_t bytes, void * stream) = nullptr;
    bool (*device_ptr)(void * host, void ** out)                    = nullptr;
    // Device selection for event waits. Events belong to the device that
    // recorded them; HIP requires the matching current device to synchronize.
    int  (*current_device)()                                       = nullptr;
    bool (*select_device)(int rank)                                = nullptr;
    bool (*restore_device)(int physical)                           = nullptr;
    // Reads the runtime's first sticky error without clearing it.
    int  (*peek_error)()                                           = nullptr;
};

class pool {
public:
    struct stats {
        uint64_t leases                    = 0;
        uint64_t uploads                   = 0;
        uint64_t upload_bytes              = 0;
        uint64_t upload_bytes_by_kind[kKindCount] = {};
        uint64_t waits                     = 0;
        uint64_t wait_ns                   = 0;
        uint64_t enqueues                  = 0;
        uint64_t releases                  = 0;
        uint64_t hits                      = 0;
        uint64_t misses                    = 0;
        uint64_t oversize                  = 0;
        uint64_t alloc_fail                = 0;
        uint64_t stale_release             = 0;
        uint64_t wait_fail                 = 0;
        uint64_t no_slot                   = 0;
        uint64_t quarantine                = 0;
        int      first_error               = 0;
        uint64_t invalid                   = 0;
    };

    pool() = default;
    explicit pool(backend_ops ops, size_t max_bytes = kMaxPoolBytes)
        : ops_(ops), max_bytes_(max_bytes) {}
    ~pool() {
        for (auto & win : slots_) {
            for (auto & key : win) {
                for (auto & s : key) {
                    // A quarantined slot may still be read by a copy whose
                    // completion was never proven. Keep it until process exit.
                    if (s.quarantined) {
                        continue;
                    }
                    for (auto & e : s.events) {
                        if (e != nullptr && ops_.event_destroy != nullptr) {
                            ops_.event_destroy(e);
                        }
                    }
                    if (s.host != nullptr && ops_.host_free != nullptr) {
                        ops_.host_free(s.host);
                    }
                }
            }
        }
    }
    pool(const pool &) = delete;
    pool & operator=(const pool &) = delete;

    bool usable() const { return !disabled_.load(std::memory_order_relaxed); }

    size_t pool_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return bytes_used_;
    }

    // Reuses a slot for (window, kind). Grows a free slot within the pool cap.
    // A leased slot is never reused. A pending but unleased slot is waited
    // only when wait_pending is true; the async lane passes false so it never
    // blocks a pipeline on an in-flight event and falls back instead.
    bool acquire(uint32_t window, kind k, size_t bytes, lease * out, bool wait_pending = true) {
        if (out == nullptr || bytes == 0 || window >= kWindows || size_t(k) >= kKindCount) {
            std::lock_guard<std::mutex> lock(mutex_);
            stats_.invalid++;
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (disabled_.load(std::memory_order_relaxed)) {
            return false;
        }
        auto & key = slots_[window][size_t(k)];
        slot * chosen = nullptr;
        // 1) Free and retired, large enough.
        for (auto & s : key) {
            if (!s.leased && !s.pending && !s.quarantined && s.cap >= bytes) {
                chosen = &s;
                break;
            }
        }
        // 2) Free and retired, grow.
        if (chosen == nullptr) {
            for (auto & s : key) {
                if (!s.leased && !s.pending && !s.quarantined) {
                    chosen = &s;
                    break;
                }
            }
        }
        // 3) Free but pending, large enough: wait, then reuse.
        if (chosen == nullptr && wait_pending) {
            for (auto & s : key) {
                if (!s.leased && !s.quarantined && s.cap >= bytes) {
                    chosen = &s;
                    break;
                }
            }
        }
        // 4) Free but pending, grow after the wait.
        if (chosen == nullptr && wait_pending) {
            for (auto & s : key) {
                if (!s.leased && !s.quarantined) {
                    chosen = &s;
                    break;
                }
            }
        }
        // 5) No usable slot. Never steal a live lease and never wait unless
        //    the caller asked for it.
        if (chosen == nullptr) {
            stats_.no_slot++;
            return false;
        }
        if (!wait_slot_locked(*chosen)) {
            return false;
        }
        if (!grow_locked(*chosen, bytes)) {
            return false;
        }
        chosen->leased = true;
        chosen->epoch = next_epoch_++;
        out->host     = chosen->host;
        out->cap      = chosen->cap;
        out->which    = k;
        out->window   = window;
        out->slot     = uint32_t(chosen - key.data());
        out->epoch    = chosen->epoch;
        stats_.leases++;
        return true;
    }

    bool upload(const lease & l, const void * src, size_t bytes) {
        if (src == nullptr || bytes == 0) {
            return false;
        }
        uint8_t * host = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (disabled_.load(std::memory_order_relaxed)) {
                stats_.invalid++;
                return false;
            }
            slot * s = find_locked(l);
            if (s == nullptr || s->quarantined || bytes > s->cap) {
                stats_.invalid++;
                return false;
            }
            host = s->host;
            stats_.uploads++;
            stats_.upload_bytes += bytes;
            stats_.upload_bytes_by_kind[size_t(l.which)] += bytes;
        }
        // The slot is leased, so the pointer cannot be reused while copying.
        std::memcpy(host, src, bytes);
        return true;
    }

    // Copies rows into the slot. src_stride and dst_stride are independent so
    // a packed destination is possible; the meta 2D path passes equal strides
    // because its consumer reads with the source stride.
    bool upload_rows(const lease & l, const void * src, size_t src_stride,
                     size_t dst_stride, size_t row_bytes, size_t rows) {
        if (src == nullptr || row_bytes == 0 || rows == 0) {
            return false;
        }
        if (src_stride < row_bytes || dst_stride < row_bytes) {
            std::lock_guard<std::mutex> lock(mutex_);
            stats_.invalid++;
            return false;
        }
        const size_t span = (rows - 1) * dst_stride + row_bytes;
        uint8_t * host = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            slot * s = find_locked(l);
            if (s == nullptr || span > s->cap) {
                stats_.invalid++;
                return false;
            }
            host = s->host;
            stats_.uploads++;
            stats_.upload_bytes += span;
            stats_.upload_bytes_by_kind[size_t(l.which)] += span;
        }
        const auto * p = static_cast<const uint8_t *>(src);
        for (size_t r = 0; r < rows; ++r) {
            std::memcpy(host + r * dst_stride, p + r * src_stride, row_bytes);
        }
        return true;
    }

    bool enqueue(const lease & l, int rank, void * dst, size_t bytes, void * stream) {
        if (dst == nullptr || bytes == 0 || rank < 0 || rank >= int(kRanks)) {
            std::lock_guard<std::mutex> lock(mutex_);
            stats_.invalid++;
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (disabled_.load(std::memory_order_relaxed)) {
            stats_.invalid++;
            return false;
        }
        slot * s = find_locked(l);
        if (s == nullptr || s->quarantined || bytes > s->cap || ops_.copy_h2d == nullptr ||
                ops_.event_create == nullptr || ops_.event_record == nullptr) {
            stats_.invalid++;
            return false;
        }
        if (!ops_.copy_h2d(dst, s->host, bytes, stream)) {
            // A failed async copy is treated as unknown: the runtime cannot
            // prove that nothing was queued, so the buffer must not be reused.
            s->pending = true;
            quarantine_locked(*s);
            disabled_.store(true, std::memory_order_relaxed);
            stats_.alloc_fail++;
            return false;
        }
        if (s->events[rank] == nullptr && !ops_.event_create(&s->events[rank])) {
            // The copy is already queued. Without an event its completion is
            // unprovable, so the slot is kept and never reused or freed.
            s->pending = true;
            s->pending_ranks |= uint32_t(1) << uint32_t(rank);
            quarantine_locked(*s);
            disabled_.store(true, std::memory_order_relaxed);
            stats_.alloc_fail++;
            return false;
        }
        if (!ops_.event_record(s->events[rank], stream)) {
            s->pending = true;
            s->pending_ranks |= uint32_t(1) << uint32_t(rank);
            quarantine_locked(*s);
            disabled_.store(true, std::memory_order_relaxed);
            stats_.alloc_fail++;
            return false;
        }
        s->pending = true;
        s->pending_ranks |= uint32_t(1) << uint32_t(rank);
        stats_.enqueues++;
        return true;
    }

    bool wait(const lease & l) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (disabled_.load(std::memory_order_relaxed)) {
            stats_.invalid++;
            return false;
        }
        slot * s = find_locked(l);
        if (s == nullptr || s->quarantined) {
            stats_.invalid++;
            return false;
        }
        return wait_slot_locked(*s);
    }

    void release(const lease & l) {
        std::lock_guard<std::mutex> lock(mutex_);
        slot * s = find_locked(l);
        if (s == nullptr) {
            stats_.stale_release++;
            return;
        }
        s->leased = false;
        stats_.releases++;
    }

    bool owns(const void * p, size_t bytes) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return owns_locked(p, bytes);
    }

    void * device_ptr(const lease & l, int rank) {
        if (rank < 0 || rank >= int(kRanks)) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (disabled_.load(std::memory_order_relaxed)) {
            return nullptr;
        }
        slot * s = find_locked(l);
        if (s == nullptr || s->quarantined || ops_.device_ptr == nullptr) {
            return nullptr;
        }
        if (s->dev_ptrs[rank] == nullptr && !ops_.device_ptr(s->host, &s->dev_ptrs[rank])) {
            return nullptr;
        }
        return s->dev_ptrs[rank];
    }

    // Diagnostics only: counts whether a copy source belongs to the pool.
    void note_source(const void * p, size_t bytes) {
        // No slot has ever been allocated: this is the common case while the
        // weights load. Skip the lock entirely.
        if (!has_host_.load(std::memory_order_relaxed)) {
            misses_fast_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (owns_locked(p, bytes)) {
            stats_.hits++;
        } else {
            stats_.misses++;
        }
    }

    stats snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        stats copy = stats_;
        copy.misses += misses_fast_.load(std::memory_order_relaxed);
        return copy;
    }

    void dump(std::FILE * f) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::fprintf(f, "TP5_STAGING pool_bytes=%zu max_bytes=%zu disabled=%d\n",
                     bytes_used_, max_bytes_, int(disabled_.load(std::memory_order_relaxed)));
        static const char * names[kKindCount] = {
            "tokens", "pos", "out_ids", "mtp_h", "kq_mask", "generic",
        };
        for (size_t k = 0; k < kKindCount; ++k) {
            if (stats_.upload_bytes_by_kind[k] == 0) {
                continue;
            }
            std::fprintf(f, "TP5_STAGING kind=%s upload_bytes=%llu\n", names[k],
                         (unsigned long long) stats_.upload_bytes_by_kind[k]);
        }
        const uint64_t misses = stats_.misses + misses_fast_.load(std::memory_order_relaxed);
        std::fprintf(f,
            "TP5_STAGING leases=%llu uploads=%llu upload_bytes=%llu waits=%llu wait_ms=%.3f "
            "enqueues=%llu releases=%llu hits=%llu misses=%llu oversize=%llu alloc_fail=%llu "
            "stale_release=%llu wait_fail=%llu no_slot=%llu quarantine=%llu invalid=%llu "
            "first_error=%d\n",
            (unsigned long long) stats_.leases,
            (unsigned long long) stats_.uploads,
            (unsigned long long) stats_.upload_bytes,
            (unsigned long long) stats_.waits,
            double(stats_.wait_ns) / 1e6,
            (unsigned long long) stats_.enqueues,
            (unsigned long long) stats_.releases,
            (unsigned long long) stats_.hits,
            (unsigned long long) misses,
            (unsigned long long) stats_.oversize,
            (unsigned long long) stats_.alloc_fail,
            (unsigned long long) stats_.stale_release,
            (unsigned long long) stats_.wait_fail,
            (unsigned long long) stats_.no_slot,
            (unsigned long long) stats_.quarantine,
            (unsigned long long) stats_.invalid,
            stats_.first_error);
    }

private:
    struct slot {
        uint8_t * host = nullptr;
        size_t    cap = 0;
        uint64_t  epoch = 0;
        bool      leased = false;
        bool      pending = false;
        bool      quarantined = false;  // a copy may still read host: never reuse or free
        uint32_t  pending_ranks = 0;
        void *    events[kRanks] = {};
        void *    dev_ptrs[kRanks] = {};
    };

    using rotation = std::array<slot, kSlotsPerKey>;

    void note_fault_locked() {
        if (stats_.first_error != 0 || ops_.peek_error == nullptr) {
            return;
        }
        const int err = ops_.peek_error();
        if (err != 0) {
            stats_.first_error = err;
        }
    }

    void quarantine_locked(slot & s) {
        if (!s.quarantined) {
            s.quarantined = true;
            stats_.quarantine++;
        }
        note_fault_locked();
    }

    slot * find_locked(const lease & l) {
        if (l.window >= kWindows || size_t(l.which) >= kKindCount || l.slot >= kSlotsPerKey) {
            return nullptr;
        }
        slot & s = slots_[l.window][size_t(l.which)][l.slot];
        if (!s.leased || s.epoch != l.epoch || s.host != l.host) {
            return nullptr;
        }
        return &s;
    }

    bool owns_locked(const void * p, size_t bytes) const {
        if (p == nullptr || bytes == 0) {
            return false;
        }
        const auto * b = static_cast<const uint8_t *>(p);
        for (const auto & win : slots_) {
            for (const auto & key : win) {
                for (const auto & s : key) {
                    if (s.host != nullptr && b >= s.host && b + bytes <= s.host + s.cap) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    bool wait_slot_locked(slot & s) {
        if (!s.pending) {
            return true;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const int saved = ops_.current_device != nullptr ? ops_.current_device() : -1;
        bool ok = true;
        for (uint32_t r = 0; r < kRanks; ++r) {
            if ((s.pending_ranks & (uint32_t(1) << r)) == 0) {
                continue;
            }
            if (ops_.select_device != nullptr && !ops_.select_device(int(r))) {
                ok = false;
                break;
            }
            if (s.events[r] == nullptr || ops_.event_sync == nullptr ||
                    !ops_.event_sync(s.events[r])) {
                ok = false;
                break;
            }
        }
        if (ops_.restore_device != nullptr && !ops_.restore_device(saved)) {
            ok = false;
        }
        if (!ok) {
            quarantine_locked(s);
            disabled_.store(true, std::memory_order_relaxed);
            stats_.wait_fail++;
            return false;
        }
        const auto t1 = std::chrono::steady_clock::now();
        stats_.wait_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        stats_.waits++;
        s.pending = false;
        s.pending_ranks = 0;
        return true;
    }

    bool grow_locked(slot & s, size_t bytes) {
        if (s.quarantined) {
            return false;
        }
        if (bytes <= s.cap) {
            return true;
        }
        const size_t used = bytes_used_ - s.cap;
        if (used + bytes > max_bytes_) {
            stats_.oversize++;
            return false;
        }
        if (ops_.host_alloc == nullptr) {
            note_fault_locked();
            disabled_.store(true, std::memory_order_relaxed);
            stats_.alloc_fail++;
            return false;
        }
        void * host = nullptr;
        if (!ops_.host_alloc(bytes, &host)) {
            note_fault_locked();
            disabled_.store(true, std::memory_order_relaxed);
            stats_.alloc_fail++;
            return false;
        }
        if (s.host != nullptr && ops_.host_free != nullptr) {
            ops_.host_free(s.host);
        }
        s.host = static_cast<uint8_t *>(host);
        s.cap = bytes;
        for (auto & d : s.dev_ptrs) {
            d = nullptr;
        }
        bytes_used_ = used + bytes;
        has_host_.store(true, std::memory_order_relaxed);
        return true;
    }

    mutable std::mutex mutex_;
    backend_ops ops_ = {};
    size_t max_bytes_ = kMaxPoolBytes;
    size_t bytes_used_ = 0;
    uint64_t next_epoch_ = 1;
    std::atomic<bool> disabled_{false};
    std::atomic<bool> has_host_{false};
    mutable std::atomic<uint64_t> misses_fast_{0};
    stats stats_;
    std::array<std::array<rotation, kKindCount>, kWindows> slots_ = {};
};

// Process-wide pool. Defined by the GPU backend; never linked into ggml-base.
pool & global_pool();
bool enabled();
bool merge_wait_enabled();
void init();
bool zc_enabled();
void dump();

// Host-only vtable registered with the meta backend. The meta layer stays
// backend-agnostic: it calls this table and never links GPU symbols.
struct api {
    uint32_t abi_version;
    bool (*enabled)();
    // GGML_TP5_STAGING_MERGE_WAIT: the sync MIRRORED path enqueues every rank
    // first and waits all events once instead of syncing each rank in turn.
    bool (*merge_wait)();
    bool (*acquire)(uint32_t window, kind k, size_t bytes, lease * out);
    // Never waits a pending slot; the async lane uses it so an in-flight
    // event cannot block the pipeline.
    bool (*acquire_nowait)(uint32_t window, kind k, size_t bytes, lease * out);
    bool (*upload)(const lease * l, const void * src, size_t bytes);
    bool (*upload_rows)(const lease * l, const void * src, size_t src_stride,
                        size_t dst_stride, size_t row_bytes, size_t rows);
    bool (*wait)(const lease * l);
    void (*release)(const lease * l);
    bool (*owns)(const void * p, size_t bytes);
    void * (*device_ptr)(const lease * l, int rank);
    void (*dump)();
    // Async lane: the backend copies from the lease and records one event.
    bool (*can_enqueue_async)(ggml_backend_t backend, ggml_tensor * tensor,
                              size_t offset, size_t size);
    bool (*enqueue_async)(const lease * l, ggml_backend_t backend,
                          ggml_tensor * tensor, size_t offset, size_t size);
    // Merge-wait lane: same-device copies on cudaStreamPerThread, the stream
    // the synchronous set_tensor uses.
    bool (*can_enqueue_sync)(ggml_tensor * tensor, size_t offset, size_t size);
    bool (*enqueue_sync)(const lease * l, ggml_tensor * tensor,
                         size_t offset, size_t size);
    bool (*zc_enabled)();
};

// Defined by the CUDA/HIP backend.
bool can_enqueue_async_impl(ggml_backend_t backend, ggml_tensor * tensor,
                            size_t offset, size_t size);
bool enqueue_async_impl(const lease & l, ggml_backend_t backend,
                        ggml_tensor * tensor, size_t offset, size_t size);
bool can_enqueue_sync_impl(ggml_tensor * tensor, size_t offset, size_t size);
bool enqueue_sync_impl(const lease & l, ggml_tensor * tensor,
                       size_t offset, size_t size);

}  // namespace ggml_cuda_staging

extern "C" {
// Implemented in ggml-backend-meta.cpp (ggml-base). The GPU backend calls it
// once from its registry entry point. No-op when the meta layer is absent.
GGML_BACKEND_API bool ggml_backend_meta_staging_register(const void * impl);
GGML_BACKEND_API const void * ggml_backend_meta_staging_get();
GGML_BACKEND_API void ggml_backend_meta_staging_diag_dump();
}
