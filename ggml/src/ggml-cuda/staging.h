#pragma once

// TP5 WS4: persistent pinned host-to-device staging pool.
//
// A lease is one pinned slot. The caller copies host data into the slot once;
// every rank then reads the same pinned address. The pool holds no GPU types,
// so a CPU test can exercise it with injected primitives.

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>

#include "ggml-backend.h"

namespace ggml_cuda_staging {

constexpr size_t   kMaxPoolBytes = 4u << 20;
constexpr uint32_t kRanks        = 16;  // == GGML_CUDA_MAX_DEVICES
// One key per staging thread. The compose runtime has one worker thread per
// service window, so the two active workers get separate keys; a process with
// more staging threads than this wraps and shares a key.
constexpr uint32_t kWindows      = 4;
constexpr uint32_t kSlotsPerKey  = 2;   // double buffer
// Bump when the host-only api table grows. New fields are appended after the
// previous version's last field, but the meta layer accepts only the current
// version and checks the version before reading any later field. v1 cannot
// publish a partial submission; v2 has mark_partial but no kind domain; v3
// adds kind_count; v4 adds owner-issued credit and explicit drain. A mixed
// version install is explicitly unsupported and the registration is rejected;
// an existing registration is unchanged, and a first rejection leaves staging
// unregistered (the original copy path is used). This is not a claim of
// equivalent performance.
constexpr uint32_t kApiVersion   = 4;

enum class kind : uint32_t {
    tokens  = 0,
    pos     = 1,
    out_ids = 2,
    mtp_h   = 3,
    kq_mask = 4,
    generic = 5,
    // The two union index inputs are distinct graph inputs on the same graph.
    // Dedicated kinds give each its own double buffer instead of making both
    // copies share the generic key. The generic 64 KiB cap stays in force for
    // every other unnamed input. The split adds two double buffers; the 4 MiB
    // pool cap still bounds total pinned bytes.
    union_k = 6,
    union_v = 7,
    count   = 8,
};

constexpr size_t kKindCount = size_t(kind::count);

// Outcome of one enqueue. The caller may replay the destination only after
// rejected: ok is complete, fatal means a copy may already be in flight and
// nothing else may be submitted to that destination.
enum class submit_result : int {
    ok       = 0,
    rejected = 1,  // validated out before any runtime copy
    fatal    = 2,  // submission is unknown or an event failed; stop
};

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
    // These exact names are reachable from the async lane's source tensor.
    // The sync lane classifies the scheduler's destination copy, named
    // "%s#%s#%d"; n_copies==1 copies are not INPUT. Do not parse the
    // decorated name and do not relax the INPUT requirement: the sync
    // fallback stays generic or unstaged.
    if (std::strcmp(name, "union_inp_k_idxs") == 0) {
        return kind::union_k;
    }
    if (std::strcmp(name, "union_inp_v_idxs") == 0) {
        return kind::union_v;
    }
    return kind::generic;
}

constexpr size_t kGenericMaxBytes = 64u << 10;

// Why a tensor is not a staging candidate. The eligibility predicate and the
// diagnostics must read the same decision, so both go through
// staging_candidate_decision; a reason added here reaches the dump without a
// second copy of the rules.
enum class candidate_reject : uint32_t {
    none          = 0,
    no_input_flag = 1,
    zero_size     = 2,
    size          = 3,  // above the per-request bound
    slot_envelope = 4,  // generic/union_k/union_v above kGenericMaxBytes
};

// One decision carries the rejection reason and the classified kind, so a
// caller that needs both looks the name up once. Rejection priority is fixed
// here: input flag, zero size, per-request bound, then the per-kind slot
// envelope.
struct candidate_decision {
    candidate_reject reason = candidate_reject::none;
    kind             which  = kind::generic;
};

// A staging candidate is a graph input (the flag excludes weights and other
// one-shot uploads) within the per-request size bound. Generic kinds are
// allowed: unnamed inputs still rotate through their own slots, but only when
// small, so a large unnamed block cannot crowd out the dedicated kinds. The
// two union index kinds were generic before the split and keep the same
// 64 KiB per-slot envelope, so eligibility is unchanged. The split is not
// free: it adds two double buffers. In the active window 0 that is up to
// 4 x 64 KiB = 256 KiB of additional pinned capacity (up to 16 slot records
// and 1 MiB across all four window arrays), plus up to one lazily created
// event per rank per new slot. The 4 MiB pool cap still bounds pinned bytes;
// the extra event handles and metadata are outside that byte cap.
inline candidate_decision staging_candidate_decision(bool has_input_flag, const char * name,
                                                     size_t size, size_t max_bytes) {
    if (!has_input_flag) {
        return {candidate_reject::no_input_flag, kind::generic};
    }
    if (size == 0) {
        return {candidate_reject::zero_size, kind::generic};
    }
    if (size > max_bytes) {
        return {candidate_reject::size, kind::generic};
    }
    const kind k = classify_name(name);
    if ((k == kind::generic || k == kind::union_k || k == kind::union_v) &&
            size > kGenericMaxBytes) {
        return {candidate_reject::slot_envelope, k};
    }
    return {candidate_reject::none, k};
}

inline candidate_reject staging_candidate_reason(bool has_input_flag, const char * name, size_t size,
                                                 size_t max_bytes) {
    return staging_candidate_decision(has_input_flag, name, size, max_bytes).reason;
}

inline bool staging_candidate(bool has_input_flag, const char * name, size_t size,
                              size_t max_bytes, kind * out) {
    if (out == nullptr) {
        return false;
    }
    const candidate_decision decision =
        staging_candidate_decision(has_input_flag, name, size, max_bytes);
    if (decision.reason != candidate_reject::none) {
        return false;
    }
    *out = decision.which;
    return true;
}

struct lease {
    uint8_t * host   = nullptr;
    size_t    cap    = 0;
    kind      which  = kind::generic;
    uint32_t  window = 0;
    uint32_t  slot   = 0;
    uint64_t  epoch  = 0;
    // v4 owner credit. Zero for the legacy window-only acquire path.
    uint64_t  owner      = 0;
    uint64_t  generation = 0;
};

// Result of an explicit owner drain. Only drained permits release; busy and
// failed leave the owner and its credit/window resources in place.
enum class owner_drain_result : int {
    drained = 0,
    busy    = 1,  // a lease is still active; no new leases are admitted
    failed  = 2,  // a pending event could not be proven complete; retained
    invalid = 3,  // unknown owner/generation or not active/draining
};

enum class owner_phase {
    free,
    active,
    draining,
    drained,
    failed,
};

// Primitives supplied by the GPU backend. The pool calls them with its own
// lock held; they must not call back into the pool.
struct backend_ops {
    // All runtime primitives return 0 on success and the native error code on
    // failure. The pool records the code actually returned by the call.
    int  (*host_alloc)(size_t bytes, void ** out)                   = nullptr;
    void (*host_free)(void * ptr)                                   = nullptr;
    int  (*event_create)(void ** out)                               = nullptr;
    int  (*event_record)(void * event, void * stream)               = nullptr;
    int  (*event_sync)(void * event)                                = nullptr;
    // Non-blocking completion probe: returns 0 when the query ran and set
    // *complete, or a native error code when completion is unprovable.
    int  (*event_query)(void * event, bool * complete)              = nullptr;
    void (*event_destroy)(void * event)                             = nullptr;
    int  (*copy_h2d)(void * dst, const void * src, size_t bytes, void * stream) = nullptr;
    int  (*device_ptr)(void * host, void ** out)                    = nullptr;
    // Device selection for event waits. Events belong to the device that
    // recorded them; HIP requires the matching current device to synchronize.
    int  (*current_device)()                                       = nullptr;
    bool (*select_device)(int rank)                                = nullptr;
    bool (*restore_device)(int physical)                           = nullptr;
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
        uint64_t query_fail                = 0;
        uint64_t no_slot                   = 0;
        uint64_t no_slot_leased            = 0;
        uint64_t no_slot_pending           = 0;
        uint64_t no_slot_by_kind[kKindCount]  = {};
        uint64_t leases_by_kind[kKindCount]   = {};
        uint64_t reaps_by_kind[kKindCount]    = {};
        uint64_t queries_by_kind[kKindCount]  = {};
        // Per-window attribution. The compose runtime has one worker thread
        // per service window; the host assigns each staging thread its own
        // window. These counters prove that the split is used and that the
        // remaining misses are not one window starving another.
        uint64_t leases_by_window[kWindows]   = {};
        uint64_t no_slot_by_window[kWindows]  = {};
        uint64_t reaps                     = 0;
        uint64_t queries                   = 0;
        uint64_t quarantine                = 0;
        uint64_t partial                   = 0;
        int      first_error               = 0;
        uint64_t invalid                   = 0;
    };

    pool() = default;
    explicit pool(backend_ops ops, size_t max_bytes = kMaxPoolBytes)
        : ops_(ops), max_bytes_(max_bytes) {}
    ~pool() {
        std::unique_lock<std::mutex> lock(mutex_);
        shutting_down_ = true;
        drain_done_.wait(lock, [this] { return active_drains_ == 0; });
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
        std::lock_guard<std::mutex> lock(mutex_);
        return acquire_locked(window, k, bytes, out, wait_pending);
    }

    // Caller holds mutex_. The public acquire wrapper and the v4 owner path
    // share this body.
    bool acquire_locked(uint32_t window, kind k, size_t bytes, lease * out, bool wait_pending) {
        if (out == nullptr || bytes == 0 || window >= kWindows || size_t(k) >= kKindCount) {
            stats_.invalid++;
            return false;
        }
        if (wait_pending) {
            std::unique_lock<std::mutex> lock(mutex_);
            for (;;) {
                if (disabled_.load(std::memory_order_relaxed)) {
                    return false;
                }
                auto & key = slots_[window][size_t(k)];
                slot * chosen = pick_free_locked(key, bytes);
                // Free but pending, large enough: wait, then reuse.
                if (chosen == nullptr) {
                    for (auto & s : key) {
                        if (!s.leased && !s.quarantined && !s.reaping && s.cap >= bytes) {
                            chosen = &s;
                            break;
                        }
                    }
                }
                // Free but pending, grow after the wait.
                if (chosen == nullptr) {
                    for (auto & s : key) {
                        if (!s.leased && !s.quarantined && !s.reaping) {
                            chosen = &s;
                            break;
                        }
                    }
                }
                if (chosen != nullptr) {
                    return lease_slot_locked(*chosen, size_t(chosen - key.data()), window, k, bytes, out);
                }
                // A non-blocking acquire is querying every otherwise usable
                // pending slot. Its query is bounded and it only needs the
                // lock to clear reaping, so drop the lock and retry instead of
                // waiting on a slot it owns. This keeps the blocking wait
                // under mutex_ (the note_source cycle proof depends on it).
                bool any_reaping = false;
                for (const auto & s : key) {
                    if (!s.leased && !s.quarantined && s.reaping) {
                        any_reaping = true;
                        break;
                    }
                }
                if (!any_reaping) {
                    count_no_slot_locked(key, window, k);
                    return false;
                }
                lock.unlock();
                std::this_thread::yield();
                lock.lock();
            }
        }

        // Async lane: never wait a pending slot. A completed in-flight copy is
        // reclaimed without blocking the pipeline. The backend event queries
        // and their device switches run outside mutex_: the old path held the
        // process-global pool lock across up to kRanks HIP calls per slot, so
        // one worker's retire probe stalled the other worker's acquire.
        std::array<bool, kSlotsPerKey> examined{};
        for (;;) {
            size_t   reap_index  = kSlotsPerKey;
            uint32_t reap_ranks  = 0;
            uint64_t reap_epoch  = 0;
            void *   reap_events[kRanks] = {};
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (disabled_.load(std::memory_order_relaxed)) {
                    return false;
                }
                auto & key = slots_[window][size_t(k)];
                if (slot * chosen = pick_free_locked(key, bytes)) {
                    return lease_slot_locked(*chosen, size_t(chosen - key.data()), window, k, bytes, out);
                }
                for (size_t i = 0; i < key.size(); ++i) {
                    if (examined[i]) {
                        continue;
                    }
                    slot & s = key[i];
                    if (s.leased || s.quarantined || !s.pending || s.reaping) {
                        continue;
                    }
                    s.reaping    = true;
                    reap_index   = i;
                    reap_ranks   = s.pending_ranks;
                    reap_epoch   = s.epoch;
                    std::memcpy(reap_events, s.events, sizeof(reap_events));
                    break;
                }
                if (reap_index == kSlotsPerKey) {
                    count_no_slot_locked(key, window, k);
                    return false;
                }
            }
            examined[reap_index] = true;

            bool complete = true;
            int  fail_err = 0;
            uint64_t queries = 0;
            if (ops_.event_query == nullptr) {
                // No completion probe: leave the slot pending instead of
                // turning a missing optional hook into a pool fault.
                complete = false;
            } else {
                int saved = -1;
                reap_query_guard guard(*this, window, size_t(k), reap_index, reap_epoch, saved);
                saved = ops_.current_device != nullptr ? ops_.current_device() : -1;
                for (uint32_t r = 0; r < kRanks; ++r) {
                    if ((reap_ranks & (uint32_t(1) << r)) == 0) {
                        continue;
                    }
                    if (ops_.select_device != nullptr && !ops_.select_device(int(r))) {
                        fail_err = -1;
                        break;
                    }
                    if (reap_events[r] == nullptr) {
                        fail_err = -1;
                        break;
                    }
                    // Queries counts backend calls only: one per pending rank
                    // event examined, stopping at the first not-ready or error.
                    ++queries;
                    bool rank_complete = false;
                    const int query_err = ops_.event_query(reap_events[r], &rank_complete);
                    if (query_err != 0) {
                        fail_err = query_err;
                        break;
                    }
                    if (!rank_complete) {
                        complete = false;
                        break;
                    }
                }
                if (!guard.restore()) {
                    if (fail_err == 0) {
                        fail_err = -1;
                    }
                }
                guard.commit();
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto & key = slots_[window][size_t(k)];
                slot & s = key[reap_index];
                // A call count, not a latency measurement: count the backend
                // queries even when another thread faulted the pool while
                // they ran.
                stats_.queries += queries;
                stats_.queries_by_kind[size_t(k)] += queries;
                const bool still_ours = s.reaping && s.epoch == reap_epoch &&
                                        s.pending && !s.leased && !s.quarantined;
                if (s.reaping) {
                    s.reaping = false;
                }
                if (!still_ours) {
                    // Only this acquire could clear reaping, and a pending
                    // unleased slot cannot be leased elsewhere. Keep the
                    // failure path defensive: retry the remaining slots.
                    if (disabled_.load(std::memory_order_relaxed)) {
                        return false;
                    }
                    continue;
                }
                if (fail_err != 0) {
                    quarantine_locked(s, fail_err);
                    disabled_.store(true, std::memory_order_relaxed);
                    publish_fatal_locked();
                    stats_.query_fail++;
                    return false;
                }
                if (disabled_.load(std::memory_order_relaxed)) {
                    return false;
                }
                if (!complete) {
                    continue;
                }
                s.pending = false;
                s.pending_ranks = 0;
                stats_.reaps++;
                stats_.reaps_by_kind[size_t(k)]++;
                slot * chosen = pick_free_locked(key, bytes);
                if (chosen == nullptr) {
                    // The reaped slot is free; this is defense against a
                    // future state that frees a slot without growing it.
                    count_no_slot_locked(key, window, k);
                    return false;
                }
                return lease_slot_locked(*chosen, size_t(chosen - key.data()), window, k, bytes, out);
            }
        }
    }

    // v4: the pool issues a monotonic owner token and one of the four finite
    // credit windows. run_id/window_id are caller hints for duplicate
    // detection only; the identity and the credit come from the pool.
    bool owner_acquire(uint64_t run_id, uint32_t window_id, uint64_t * owner, uint64_t * generation) {
        if (owner == nullptr || generation == nullptr || run_id == 0 || window_id >= kWindows) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (disabled_.load(std::memory_order_relaxed)) {
            return false;
        }
        for (auto & o : owners_) {
            if (o.phase != owner_phase::free) {
                continue;
            }
            o.run_id = run_id;
            o.window_id = window_id;
            o.owner = next_owner_id_++;
            o.generation = next_generation_++;
            o.window = uint32_t(&o - owners_.data());
            o.phase = owner_phase::active;
            *owner = o.owner;
            *generation = o.generation;
            return true;
        }
        return false; // finite credit exhausted
    }

    bool acquire_owner(uint64_t owner, uint64_t generation, kind k, size_t bytes, lease * out, bool wait_pending) {
        std::lock_guard<std::mutex> lock(mutex_);
        owner_credit * oc = find_owner_locked(owner, generation);
        if (oc == nullptr || oc->phase != owner_phase::active) {
            return false;
        }
        if (!acquire_locked(oc->window, k, bytes, out, wait_pending)) {
            return false;
        }
        out->owner = owner;
        out->generation = generation;
        return true;
    }

    // Explicit drain. It admits no new lease after the owner enters draining.
    // A still-leased slot is busy, not success. A failed wait retains the
    // owner and its credit/window; only drained permits release.
    owner_drain_result owner_drain(uint64_t owner, uint64_t generation) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (shutting_down_) {
            return owner_drain_result::invalid;
        }
        owner_credit * oc = find_owner_locked(owner, generation);
        if (oc == nullptr) {
            return owner_drain_result::invalid;
        }
        if (oc->phase == owner_phase::drained) {
            return owner_drain_result::drained;
        }
        if (oc->phase == owner_phase::failed) {
            return owner_drain_result::failed;
        }
        if (oc->phase != owner_phase::active && oc->phase != owner_phase::draining) {
            return owner_drain_result::invalid;
        }
        oc->phase = owner_phase::draining;
        if (disabled_.load(std::memory_order_relaxed) || unknown_count_ > 0) {
            oc->phase = owner_phase::failed;
            return owner_drain_result::failed;
        }
        auto & window = slots_[oc->window];
        struct pending_snapshot {
            slot *   s;
            uint32_t ranks;
            void *   events[kRanks];
        };
        std::array<pending_snapshot, kKindCount * kSlotsPerKey> pending = {};
        size_t n_pending = 0;
        for (auto & by_kind : window) {
            for (auto & s : by_kind) {
                if (s.leased) {
                    return owner_drain_result::busy;
                }
                if (s.quarantined || s.partial_marked) {
                    oc->phase = owner_phase::failed;
                    return owner_drain_result::failed;
                }
                if (s.pending) {
                    pending[n_pending++] = { &s, s.pending_ranks, {} };
                    for (uint32_t r = 0; r < kRanks; ++r) {
                        pending[n_pending - 1].events[r] = s.events[r];
                    }
                }
            }
        }
        if (n_pending == 0) {
            oc->phase = owner_phase::drained;
            return owner_drain_result::drained;
        }
        // Never hold the process-wide pool mutex while waiting for a slow
        // owner's events: another owner must remain able to acquire/release.
        const int saved = ops_.current_device != nullptr ? ops_.current_device() : -1;
        int fail_err = 0;
        slot * fail_slot = nullptr;
        active_drains_++;
        lock.unlock();
        for (size_t i = 0; i < n_pending && fail_err == 0; ++i) {
            for (uint32_t r = 0; r < kRanks; ++r) {
                if ((pending[i].ranks & (uint32_t(1) << r)) == 0) {
                    continue;
                }
                if (ops_.select_device != nullptr && !ops_.select_device(int(r))) {
                    fail_err = -1;
                    fail_slot = pending[i].s;
                    break;
                }
                if (pending[i].events[r] == nullptr || ops_.event_sync == nullptr) {
                    fail_err = -1;
                    fail_slot = pending[i].s;
                    break;
                }
                const int sync_err = ops_.event_sync(pending[i].events[r]);
                if (sync_err != 0) {
                    fail_err = sync_err;
                    fail_slot = pending[i].s;
                    break;
                }
            }
        }
        if (ops_.restore_device != nullptr && !ops_.restore_device(saved) && fail_err == 0) {
            fail_err = -1;
            fail_slot = pending[0].s;
        }
        lock.lock();
        auto finish = [this] {
            if (active_drains_ > 0) {
                --active_drains_;
            }
            drain_done_.notify_all();
        };
        oc = find_owner_locked(owner, generation);
        if (oc == nullptr || oc->phase != owner_phase::draining) {
            finish();
            return owner_drain_result::invalid;
        }
        if (fail_err != 0) {
            if (fail_slot != nullptr) {
                quarantine_locked(*fail_slot, fail_err);
            }
            disabled_.store(true, std::memory_order_relaxed);
            publish_fatal_locked();
            stats_.wait_fail++;
            oc->phase = owner_phase::failed;
            finish();
            return owner_drain_result::failed;
        }
        if (disabled_.load(std::memory_order_relaxed) || unknown_count_ > 0) {
            oc->phase = owner_phase::failed;
            finish();
            return owner_drain_result::failed;
        }
        for (size_t i = 0; i < n_pending; ++i) {
            slot * s = pending[i].s;
            if (s->leased) {
                finish();
                return owner_drain_result::busy;
            }
            if (s->quarantined || s->partial_marked) {
                oc->phase = owner_phase::failed;
                finish();
                return owner_drain_result::failed;
            }
            if (s->pending && s->pending_ranks != pending[i].ranks) {
                finish();
                return owner_drain_result::busy;
            }
        }
        for (size_t i = 0; i < n_pending; ++i) {
            pending[i].s->pending = false;
            pending[i].s->pending_ranks = 0;
        }
        oc->phase = owner_phase::drained;
        finish();
        return owner_drain_result::drained;
    }

    bool owner_release(uint64_t owner, uint64_t generation) {
        if (owner == 0 || generation == 0) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        owner_credit * oc = find_owner_locked(owner, generation);
        if (oc != nullptr) {
            if (oc->phase != owner_phase::drained) {
                return false; // busy/failed/active: retain the credit
            }
            oc->phase = owner_phase::free;
            released_[released_next_] = {owner, generation};
            released_next_ = (released_next_ + 1) % released_.size();
            return true;
        }
        for (const auto & r : released_) {
            if (r.owner == owner && r.generation == generation) {
                return true; // idempotent repeat, no second free
            }
        }
        return false;
    }

    submit_result upload(const lease & l, const void * src, size_t bytes) {
        if (src == nullptr || bytes == 0) {
            return submit_result::rejected;
        }
        uint8_t * host = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (disabled_.load(std::memory_order_relaxed)) {
                stats_.invalid++;
                return submit_result::fatal;
            }
            slot * s = find_locked(l);
            if (s == nullptr || bytes > s->cap) {
                stats_.invalid++;
                return submit_result::rejected;
            }
            if (s->quarantined) {
                stats_.invalid++;
                return submit_result::fatal;
            }
            host = s->host;
            stats_.uploads++;
            stats_.upload_bytes += bytes;
            stats_.upload_bytes_by_kind[size_t(l.which)] += bytes;
        }
        // The slot is leased, so the pointer cannot be reused while copying.
        std::memcpy(host, src, bytes);
        return submit_result::ok;
    }

    // Copies rows into the slot. src_stride and dst_stride are independent so
    // a packed destination is possible; the meta 2D path passes equal strides
    // because its consumer reads with the source stride.
    submit_result upload_rows(const lease & l, const void * src, size_t src_stride,
                              size_t dst_stride, size_t row_bytes, size_t rows) {
        if (src == nullptr || row_bytes == 0 || rows == 0) {
            return submit_result::rejected;
        }
        if (src_stride < row_bytes || dst_stride < row_bytes) {
            std::lock_guard<std::mutex> lock(mutex_);
            stats_.invalid++;
            return submit_result::rejected;
        }
        const size_t span = (rows - 1) * dst_stride + row_bytes;
        uint8_t * host = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (disabled_.load(std::memory_order_relaxed)) {
                stats_.invalid++;
                return submit_result::fatal;
            }
            slot * s = find_locked(l);
            if (s == nullptr || span > s->cap) {
                stats_.invalid++;
                return submit_result::rejected;
            }
            if (s->quarantined) {
                stats_.invalid++;
                return submit_result::fatal;
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
        return submit_result::ok;
    }

    submit_result enqueue(const lease & l, int rank, void * dst, size_t bytes, void * stream) {
        if (dst == nullptr || bytes == 0 || rank < 0 || rank >= int(kRanks)) {
            std::lock_guard<std::mutex> lock(mutex_);
            stats_.invalid++;
            return submit_result::rejected;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (disabled_.load(std::memory_order_relaxed)) {
            stats_.invalid++;
            // A disabled pool is a global fault, not a removable capacity miss:
            // every caller must stop instead of replaying the destination.
            return submit_result::fatal;
        }
        slot * s = find_locked(l);
        if (s == nullptr) {
            stats_.invalid++;
            return submit_result::rejected;
        }
        if (s->quarantined) {
            stats_.invalid++;
            return submit_result::fatal;
        }
        if (bytes > s->cap || ops_.copy_h2d == nullptr ||
                ops_.event_create == nullptr || ops_.event_record == nullptr) {
            stats_.invalid++;
            return submit_result::rejected;
        }
        const int copy_err = ops_.copy_h2d(dst, s->host, bytes, stream);
        if (copy_err != 0) {
            // A failed async copy is treated as unknown: the runtime cannot
            // prove that nothing was queued, so the buffer must not be reused.
            s->pending = true;
            quarantine_locked(*s, copy_err);
            record_unknown_locked(dst, rank, copy_err);
            disabled_.store(true, std::memory_order_relaxed);
            publish_fatal_locked();
            stats_.alloc_fail++;
            return submit_result::fatal;
        }
        if (s->events[rank] == nullptr) {
            const int event_err = ops_.event_create(&s->events[rank]);
            if (event_err != 0) {
                // The copy is already queued. Without an event its completion
                // is unprovable, so the slot is kept and never reused or freed.
                s->pending = true;
                s->pending_ranks |= uint32_t(1) << uint32_t(rank);
                quarantine_locked(*s, event_err);
                record_unknown_locked(dst, rank, event_err);
                disabled_.store(true, std::memory_order_relaxed);
                publish_fatal_locked();
                stats_.alloc_fail++;
                return submit_result::fatal;
            }
        }
        const int record_err = ops_.event_record(s->events[rank], stream);
        if (record_err != 0) {
            s->pending = true;
            s->pending_ranks |= uint32_t(1) << uint32_t(rank);
            quarantine_locked(*s, record_err);
            record_unknown_locked(dst, rank, record_err);
            disabled_.store(true, std::memory_order_relaxed);
            publish_fatal_locked();
            stats_.alloc_fail++;
            return submit_result::fatal;
        }
        s->pending = true;
        s->pending_ranks |= uint32_t(1) << uint32_t(rank);
        stats_.enqueues++;
        return submit_result::ok;
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
        if (l.owner != 0) {
            owner_credit * oc = find_owner_locked(l.owner, l.generation);
            if (oc == nullptr || oc->window != l.window) {
                stats_.stale_release++;
                return;
            }
        }
        s->leased = false;
        stats_.releases++;
    }

    // A partial multi-rank submission has no clean rejection to return: at
    // least one rank copy was queued and a later rank was rejected. The
    // destination may be partially written and its completion is not
    // provable. Stop the pool, keep the staging slot quarantined, and publish
    // the same fatal gate as a native error. There may be no native error to
    // record, so first_error stays 0 and submission_pending() is the signal.
    // Idempotence is lease-scoped: the lease must still be valid. A call after
    // release cannot find the slot and records another unknown destination.
    void mark_partial(const lease & l, const void * dst) {
        std::lock_guard<std::mutex> lock(mutex_);
        slot * s = find_locked(l);
        if (s != nullptr && s->partial_marked) {
            // A concurrent or duplicate publication for the same slot keeps
            // the gate closed but must not count the destination twice.
            disabled_.store(true, std::memory_order_relaxed);
            publish_fatal_locked();
            return;
        }
        stats_.partial++;
        disabled_.store(true, std::memory_order_relaxed);
        record_unknown_locked(dst, -1, 0);
        if (s != nullptr) {
            quarantine_locked(*s, 0);
            s->partial_marked = true;
        }
        publish_fatal_locked();
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
        if (s->dev_ptrs[rank] == nullptr) {
            const int err = ops_.device_ptr(s->host, &s->dev_ptrs[rank]);
            if (err != 0) {
                return nullptr;
            }
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
        // The submit gate may call the backend set path on this same thread.
        // enter_submit() publishes submit_gate_held_ before that path and
        // leave_submit() clears it. This flag is a fast path, not a lock: a
        // gate can start between this load and the pool lock below.
        if (submit_gate_held_.load(std::memory_order_acquire)) {
            misses_fast_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        // Never block on mutex_ here. publish_fatal_locked() holds mutex_ and
        // waits for the submit gate; a blocking note_source can close the cycle
        // note_source -> mutex_ -> submit gate -> note_source when the gate
        // holder waits for the note. A contended pool is a diagnostic fast
        // miss, not a reason to sleep.
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
            misses_fast_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        // A gate may have started while this thread was taking mutex_. Recheck
        // before touching the stats: the flag keeps the documented "skip while
        // the gate is held" accounting. This is an accuracy check, not the
        // deadlock fix (try_lock above already makes it wait-free).
        if (submit_gate_held_.load(std::memory_order_acquire)) {
            misses_fast_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
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

    int first_error() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_.first_error;
    }

    // Submission gate. The fallback copy and the fatal publication share this
    // lock: a fatal cannot be published between the check and the submit.
    // note_source reads submit_gate_held_ as a fast path and takes mutex_ only
    // by try_lock, so it never waits while this gate is held (the fatal
    // publication holds mutex_ and waits for this gate). The lock is released
    // on a false return.
    bool enter_submit() {
        submit_mutex_.lock();
        if (submit_fatal_) {
            submit_mutex_.unlock();
            return false;
        }
        submit_gate_held_.store(true, std::memory_order_release);
        return true;
    }

    void leave_submit() {
        submit_gate_held_.store(false, std::memory_order_release);
        submit_mutex_.unlock();
    }

    // Number of destinations whose completion this process cannot prove.
    // They are owned by the process until native completion; the pool only
    // keeps its staging resource quarantined.
    size_t unknown_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return unknown_count_;
    }

    // True while an asynchronous submission may still be in flight. The
    // scheduler/context owner keeps the destination until this clears; it
    // never clears for an unprovable submission.
    bool submission_pending() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return disabled_.load(std::memory_order_relaxed) || unknown_count_ > 0;
    }

    void dump(std::FILE * f) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::fprintf(f, "TP5_STAGING pool_bytes=%zu max_bytes=%zu disabled=%d\n",
                     bytes_used_, max_bytes_, int(disabled_.load(std::memory_order_relaxed)));
        static const char * names[kKindCount] = {
            "tokens", "pos", "out_ids", "mtp_h", "kq_mask", "generic",
            "union_k", "union_v",
        };
        for (size_t k = 0; k < kKindCount; ++k) {
            if (stats_.upload_bytes_by_kind[k] == 0) {
                continue;
            }
            std::fprintf(f, "TP5_STAGING kind=%s upload_bytes=%llu\n", names[k],
                         (unsigned long long) stats_.upload_bytes_by_kind[k]);
        }
        // Per-kind attribution: a no_slot can be a real lease collision or an
        // unleased slot whose rank events have not all completed. The global
        // no_slot line alone cannot tell those apart.
        for (size_t k = 0; k < kKindCount; ++k) {
            if (stats_.leases_by_kind[k] == 0 && stats_.no_slot_by_kind[k] == 0 &&
                    stats_.reaps_by_kind[k] == 0 && stats_.queries_by_kind[k] == 0) {
                continue;
            }
            std::fprintf(f,
                "TP5_STAGING_KIND kind=%s leases=%llu no_slot=%llu reaps=%llu queries=%llu\n",
                names[k],
                (unsigned long long) stats_.leases_by_kind[k],
                (unsigned long long) stats_.no_slot_by_kind[k],
                (unsigned long long) stats_.reaps_by_kind[k],
                (unsigned long long) stats_.queries_by_kind[k]);
        }
        // Per-window attribution: a run with two compose workers should show
        // leases on two window keys. One hot window with every miss is the
        // signature of the old shared window 0.
        for (size_t w = 0; w < kWindows; ++w) {
            if (stats_.leases_by_window[w] == 0 && stats_.no_slot_by_window[w] == 0) {
                continue;
            }
            std::fprintf(f, "TP5_STAGING_WINDOW window=%zu leases=%llu no_slot=%llu\n",
                         w,
                         (unsigned long long) stats_.leases_by_window[w],
                         (unsigned long long) stats_.no_slot_by_window[w]);
        }
        const uint64_t misses = stats_.misses + misses_fast_.load(std::memory_order_relaxed);
        std::fprintf(f,
            "TP5_STAGING leases=%llu uploads=%llu upload_bytes=%llu waits=%llu wait_ms=%.3f "
            "enqueues=%llu releases=%llu hits=%llu misses=%llu oversize=%llu alloc_fail=%llu "
            "stale_release=%llu wait_fail=%llu query_fail=%llu no_slot=%llu "
            "no_slot_leased=%llu no_slot_pending=%llu reaps=%llu "
            "queries=%llu quarantine=%llu invalid=%llu "
            "partial=%llu first_error=%d\n",
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
            (unsigned long long) stats_.query_fail,
            (unsigned long long) stats_.no_slot,
            (unsigned long long) stats_.no_slot_leased,
            (unsigned long long) stats_.no_slot_pending,
            (unsigned long long) stats_.reaps,
            (unsigned long long) stats_.queries,
            (unsigned long long) stats_.quarantine,
            (unsigned long long) stats_.invalid,
            (unsigned long long) stats_.partial,
            stats_.first_error);
        std::fprintf(f, "TP5_STAGING_UNKNOWN unknown=%zu\n", unknown_count_);
        for (size_t i = 0; i < unknown_count_; ++i) {
            std::fprintf(f, "TP5_STAGING_UNKNOWN dst=%p rank=%d error=%d\n",
                         unknown_[i].dst, unknown_[i].rank, unknown_[i].err);
        }
    }

private:
    struct slot {
        uint8_t * host = nullptr;
        size_t    cap = 0;
        uint64_t  epoch = 0;
        bool      leased = false;
        bool      pending = false;
        bool      quarantined = false;  // a copy may still read host: never reuse or free
        bool      partial_marked = false; // mark_partial already recorded this slot
        // One non-blocking acquire is querying this slot's events without
        // holding mutex_. Other acquires skip it instead of duplicating the
        // device switches and queries.
        bool      reaping = false;
        uint32_t  pending_ranks = 0;
        void *    events[kRanks] = {};
        void *    dev_ptrs[kRanks] = {};
    };

    using rotation = std::array<slot, kSlotsPerKey>;

    // Restores the caller's device when a backend hook exits by exception.
    // saved < 0 means current_device was unavailable or failed; -1 is not a
    // valid device and must not reach restore_device.
    struct device_restore_guard {
        const backend_ops & ops;
        int &               saved;
        bool                done = false;

        device_restore_guard(const backend_ops & ops_in, int & saved_in)
            : ops(ops_in), saved(saved_in) {}

        ~device_restore_guard() {
            if (!done && saved >= 0 && ops.restore_device != nullptr) {
                (void) ops.restore_device(saved);
            }
        }

        bool restore() {
            done = true;
            if (saved < 0 || ops.restore_device == nullptr) {
                return true;
            }
            return ops.restore_device(saved);
        }
    };

    // Out-of-lock retire query guard. The normal path calls restore() and
    // commit(); on an exception the destructor restores the device and clears
    // the reaping claim for the matching epoch, so the slot cannot stay out of
    // rotation. Clearing reaping needs the pool lock; the query block itself
    // does not hold it.
    struct reap_query_guard {
        pool &   p;
        uint32_t window;
        size_t   kind_index;
        size_t   slot_index;
        uint64_t epoch;
        int &    saved;
        bool     restore_done = false;
        bool     restore_ok   = true;
        bool     committed    = false;

        reap_query_guard(pool & pool_in, uint32_t window_in, size_t kind_in, size_t slot_in,
                         uint64_t epoch_in, int & saved_in)
            : p(pool_in), window(window_in), kind_index(kind_in), slot_index(slot_in),
              epoch(epoch_in), saved(saved_in) {}

        ~reap_query_guard() {
            if (!restore_done && saved >= 0 && p.ops_.restore_device != nullptr) {
                (void) p.ops_.restore_device(saved);
            }
            if (!committed) {
                std::lock_guard<std::mutex> lock(p.mutex_);
                slot & s = p.slots_[window][kind_index][slot_index];
                if (s.reaping && s.epoch == epoch) {
                    s.reaping = false;
                }
            }
        }

        bool restore() {
            restore_done = true;
            if (saved >= 0 && p.ops_.restore_device != nullptr) {
                restore_ok = p.ops_.restore_device(saved);
            }
            return restore_ok;
        }

        void commit() { committed = true; }
    };

    struct unknown_work {
        const void * dst = nullptr;
        int rank = -1;
        int err = 0;
    };

    struct owner_credit {
        uint64_t    run_id     = 0;
        uint32_t    window_id  = 0;
        uint64_t    owner      = 0;
        uint64_t    generation = 0;
        uint32_t    window     = 0;
        owner_phase phase      = owner_phase::free;
    };
    struct released_token {
        uint64_t owner      = 0;
        uint64_t generation = 0;
    };

    owner_credit * find_owner_locked(uint64_t owner, uint64_t generation) {
        for (auto & o : owners_) {
            if (o.owner == owner && o.generation == generation && o.phase != owner_phase::free) {
                return &o;
            }
        }
        return nullptr;
    }

    void record_unknown_locked(const void * dst, int rank, int err) {
        if (unknown_count_ < unknown_.size()) {
            unknown_[unknown_count_++] = { dst, rank, err };
        }
    }

    void note_error_locked(int err) {
        if (stats_.first_error == 0 && err != 0) {
            stats_.first_error = err;
        }
    }

    void quarantine_locked(slot & s, int err) {
        if (!s.quarantined) {
            s.quarantined = true;
            stats_.quarantine++;
        }
        note_error_locked(err);
    }

    // Caller holds mutex_. A fatal publication waits for an in-flight submit
    // gate, so a submit that started before the fault completes first and no
    // new submit can start after it.
    void publish_fatal_locked() {
        submit_mutex_.lock();
        submit_fatal_ = true;
        submit_mutex_.unlock();
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

    // Caller holds mutex_. A free slot that fits is preferred; otherwise any
    // free slot is returned for growth.
    slot * pick_free_locked(rotation & key, size_t bytes) {
        for (auto & s : key) {
            if (!s.leased && !s.pending && !s.quarantined && !s.reaping && s.cap >= bytes) {
                return &s;
            }
        }
        for (auto & s : key) {
            if (!s.leased && !s.pending && !s.quarantined && !s.reaping) {
                return &s;
            }
        }
        return nullptr;
    }

    // Caller holds mutex_. Takes the chosen slot for this lease. A pending
    // slot is waited first (blocking lane only); the async lane only reaches
    // this with a free or just-reaped slot, so the wait is a no-op there.
    bool lease_slot_locked(slot & s, size_t slot_index, uint32_t window, kind k, size_t bytes, lease * out) {
        GGML_ASSERT(slot_index < kSlotsPerKey);
        if (!wait_slot_locked(s)) {
            return false;
        }
        if (!grow_locked(s, bytes)) {
            return false;
        }
        s.leased = true;
        s.epoch  = next_epoch_++;
        out->host   = s.host;
        out->cap    = s.cap;
        out->which  = k;
        out->window = window;
        out->slot   = uint32_t(slot_index);
        out->epoch  = s.epoch;
        stats_.leases++;
        stats_.leases_by_kind[size_t(k)]++;
        stats_.leases_by_window[window]++;
        return true;
    }

    // Caller holds mutex_. Never steal a live lease and never wait unless the
    // caller asked for it. A disabled pool returned before this point; a
    // usable free slot would have been chosen by pick_free_locked; a too-small
    // free slot fails in grow_locked as oversize. The only reachable no_slot
    // states are therefore an unleased pending slot (retire lag) or all usable
    // slots leased.
    void count_no_slot_locked(const rotation & key, uint32_t window, kind k) {
        stats_.no_slot++;
        stats_.no_slot_by_kind[size_t(k)]++;
        stats_.no_slot_by_window[window]++;
        bool any_unleased_pending = false;
        for (const auto & s : key) {
            if (!s.leased && s.pending) {
                any_unleased_pending = true;
                break;
            }
        }
        if (any_unleased_pending) {
            stats_.no_slot_pending++;
        } else {
            stats_.no_slot_leased++;
        }
    }

    bool wait_slot_locked(slot & s) {
        if (!s.pending) {
            return true;
        }
        const auto t0 = std::chrono::steady_clock::now();
        int saved = -1;
        device_restore_guard restore_guard(ops_, saved);
        saved = ops_.current_device != nullptr ? ops_.current_device() : -1;
        bool ok = true;
        int fail_err = 0;
        for (uint32_t r = 0; r < kRanks; ++r) {
            if ((s.pending_ranks & (uint32_t(1) << r)) == 0) {
                continue;
            }
            if (ops_.select_device != nullptr && !ops_.select_device(int(r))) {
                ok = false;
                fail_err = -1;
                break;
            }
            if (s.events[r] == nullptr || ops_.event_sync == nullptr) {
                ok = false;
                fail_err = -1;
                break;
            }
            const int sync_err = ops_.event_sync(s.events[r]);
            if (sync_err != 0) {
                ok = false;
                fail_err = sync_err;
                break;
            }
        }
        if (!restore_guard.restore()) {
            ok = false;
            if (fail_err == 0) {
                fail_err = -1;
            }
        }
        if (!ok) {
            quarantine_locked(s, fail_err);
            disabled_.store(true, std::memory_order_relaxed);
            publish_fatal_locked();
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
            disabled_.store(true, std::memory_order_relaxed);
            stats_.alloc_fail++;
            return false;
        }
        void * host = nullptr;
        const int alloc_err = ops_.host_alloc(bytes, &host);
        if (alloc_err != 0) {
            note_error_locked(alloc_err);
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
    bool shutting_down_ = false;
    uint32_t active_drains_ = 0;
    std::condition_variable drain_done_;
    std::mutex submit_mutex_;
    // True while enter_submit owns the gate. Published for note_source, which
    // must not touch submit_mutex_ on the owning thread.
    std::atomic<bool> submit_gate_held_{false};
    bool submit_fatal_ = false;
    backend_ops ops_ = {};
    size_t max_bytes_ = kMaxPoolBytes;
    size_t bytes_used_ = 0;
    uint64_t next_epoch_ = 1;
    std::atomic<bool> disabled_{false};
    std::atomic<bool> has_host_{false};
    mutable std::atomic<uint64_t> misses_fast_{0};
    stats stats_;
    std::array<unknown_work, 16> unknown_ = {};
    size_t unknown_count_ = 0;
    std::array<owner_credit, kWindows> owners_ = {};
    uint64_t next_owner_id_ = 1;
    uint64_t next_generation_ = 1;
    std::array<released_token, 8> released_ = {};
    size_t released_next_ = 0;
    std::array<std::array<rotation, kKindCount>, kWindows> slots_ = {};
};

// Process-wide pool. Defined by the GPU backend; never linked into ggml-base.
pool & global_pool();
bool enabled();
bool configured();
bool usable();
bool merge_wait_enabled();
void init(ggml_backend_reg_t reg);
ggml_backend_reg_t provider_reg();
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
    // Returns submit_result: 0 ok, 1 rejected (no side effect), 2 fatal.
    int (*upload)(const lease * l, const void * src, size_t bytes);
    int (*upload_rows)(const lease * l, const void * src, size_t src_stride,
                       size_t dst_stride, size_t row_bytes, size_t rows);
    bool (*wait)(const lease * l);
    void (*release)(const lease * l);
    bool (*owns)(const void * p, size_t bytes);
    void * (*device_ptr)(const lease * l, int rank);
    void (*dump)();
    // Async lane: the backend copies from the lease and records one event.
    bool (*can_enqueue_async)(ggml_backend_t backend, ggml_tensor * tensor,
                              size_t offset, size_t size);
    // Returns submit_result: 0 ok, 1 rejected (no runtime copy), 2 fatal.
    int (*enqueue_async)(const lease * l, ggml_backend_t backend,
                         ggml_tensor * tensor, size_t offset, size_t size);
    // Merge-wait lane: same-device copies on cudaStreamPerThread, the stream
    // the synchronous set_tensor uses.
    bool (*can_enqueue_sync)(ggml_tensor * tensor, size_t offset, size_t size);
    int (*enqueue_sync)(const lease * l, ggml_tensor * tensor,
                        size_t offset, size_t size);
    // First sticky runtime error observed by the pool; never cleared.
    int (*first_error)();
    // Submission gate shared with the fatal publication. The caller submits
    // the fallback copy between a true enter and leave.
    bool (*enter_submit)();
    void (*leave_submit)();
    size_t (*unknown_count)();
    bool (*submission_pending)();
    // configured: GGML_TP5_STAGING requests the pool. usable: the pool has not
    // stopped. A configured but unusable pool is a global fault.
    bool (*configured)();
    bool (*usable)();
    bool (*zc_enabled)();
    // v2 only, appended after the v1 prefix. Mark a multi-rank submission that
    // queued some ranks and then hit a clean rejection. The pool stops, records
    // the destination as unknown and makes submission_pending() true.
    // first_error is unchanged because a rejection is not a native error.
    void (*mark_partial)(const lease * l, const void * dst);
    // v3 only, appended after mark_partial. Number of kind values the
    // provider's pool understands. The host rejects a provider whose
    // kind_count is smaller than its own kKindCount instead of sending kind
    // values the provider would treat as invalid.
    uint32_t kind_count;
    // v4 only, appended after kind_count. The pool issues a monotonic owner
    // token and a finite credit window. run_id/window_id are caller hints for
    // duplicate detection; the identity and credit are pool-issued.
    bool (*owner_acquire)(uint64_t run_id, uint32_t window_id, uint64_t * owner, uint64_t * generation);
    bool (*acquire_owner)(uint64_t owner, uint64_t generation, kind k, size_t bytes,
                          lease * out, bool wait_pending);
    int  (*owner_drain)(uint64_t owner, uint64_t generation);
    bool (*owner_release)(uint64_t owner, uint64_t generation);
};

// Defined by the CUDA/HIP backend.
bool can_enqueue_async_impl(ggml_backend_t backend, ggml_tensor * tensor,
                            size_t offset, size_t size);
submit_result enqueue_async_impl(const lease & l, ggml_backend_t backend,
                                 ggml_tensor * tensor, size_t offset, size_t size);
bool can_enqueue_sync_impl(ggml_tensor * tensor, size_t offset, size_t size);
submit_result enqueue_sync_impl(const lease & l, ggml_tensor * tensor,
                                size_t offset, size_t size);

}  // namespace ggml_cuda_staging

extern "C" {
// Implemented in ggml-backend-meta.cpp (ggml-base). The GPU backend calls it
// from its registry entry point, including after a quiescent unload/reload;
// a reload gets a fresh provider tag. No-op when the meta layer is absent.
GGML_BASE_API bool ggml_backend_meta_staging_register(const void * impl);
GGML_BASE_API bool ggml_backend_meta_staging_register_for_reg(
        ggml_backend_reg_t reg, const void * impl);
GGML_BASE_API bool ggml_backend_meta_staging_unregister_reg(
        ggml_backend_reg_t reg, bool * had_provider);
GGML_BASE_API const void * ggml_backend_meta_staging_get();
GGML_BASE_API void ggml_backend_meta_staging_diag_dump();
}
