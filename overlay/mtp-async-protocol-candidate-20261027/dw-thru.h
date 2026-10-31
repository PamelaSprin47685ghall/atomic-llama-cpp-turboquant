#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>

// DUALWIN-THRU (GGML_TP5_DW_THRU=1) host-interval counter.
//
// Every number here is a HOST in-flight interval, never a device execution
// timestamp. overlap_ns proves that both windows had host work in flight at the
// same time; it never proves that two GPU streams executed concurrently.
//
// Clock discipline: transition() takes the mutex FIRST and samples the clock
// while holding it, so the order in which two windows' transitions are
// recorded is exactly the order in which they acquired the counter. Sampling
// outside the lock would let two workers' timestamps invert inside the
// critical section and rewind last_ (charging one interval twice and leaving
// another uncharged). A non-monotonic sample (steady_clock is monotonic, so
// this only fires on a clock anomaly) is dropped and counted, never rewound.
//
// The mutex serializes transitions and timestamps together. The algorithm is
// the interval accumulation of overlap-counter.h: on every transition at
// `now`, the elapsed interval since the previous transition is charged to
// whichever windows were active during it, and to the overlap when both were.
//
// What this counter can and cannot prove:
//   - overlap_ns is HOST in-flight intersection: both windows had host work
//     between two consecutive transitions. It is neither a device timestamp
//     nor a proof of GPU concurrency.
//   - overlap_ns == 0 does NOT prove the absence of cross-window
//     serialization. It is also what a lockstep schedule produces (the windows
//     alternate, so their host intervals never intersect), and what a
//     submission path that is serialized before the counter's begin() point
//     produces. It only means no host interval of one window straddled a host
//     interval of the other.
//   - overlap_ns > 0 proves only that host concurrency was not eliminated. It
//     is a necessary, not sufficient, condition for device concurrency.
//   - Cross-window serialization must be argued from the lock inventory
//     (per-window submit mutex, per-window courier, distinct relay priority)
//     and from submit_wait*_ns staying ~0, not from overlap_ns alone.
//   - order_violations == 0 proves the recorded event order is a single
//     monotonic sequence, so the intervals above are not artifacts of lock
//     unfairness. A nonzero value invalidates the overlap numbers.
class dw_thru_counter {
public:
    // Window enters host-in-flight (its decode submission was observed).
    void begin(unsigned window) { transition(window, true); }

    // Window leaves host-in-flight (its completion wait returned).
    void end(unsigned window) { transition(window, false); }

    // Host time blocked acquiring this window's OWN submit mutex. With one
    // submission thread per window this stays ~0; a nonzero value means
    // something else still sits on this window's submission path.
    void submit_wait(unsigned window, std::uint64_t ns) {
        std::lock_guard<std::mutex> lock(mutex_);
        submit_wait_[slot(window)] += ns;
    }

    // Host time blocked acquiring the device-global cold-capture mutex
    // (HIP graph capture/instantiate is a device-wide operation, so it cannot
    // be made private to a window; it is only taken on a shape change).
    void cold_wait(unsigned window, std::uint64_t ns) {
        std::lock_guard<std::mutex> lock(mutex_);
        cold_wait_[slot(window)] += ns;
    }

    // Host time holding that mutex across one shape-change decode.
    void cold_hold(std::uint64_t ns) {
        std::lock_guard<std::mutex> lock(mutex_);
        cold_hold_ += ns;
        ++cold_holds_;
    }

    // Busy host-in-flight time of one window; used only to compute idle
    // fractions against wall time after the workers have been joined.
    std::uint64_t busy_ns(unsigned window) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return busy_[slot(window)];
    }

    // One line, name=value, analyzers parse by name. Called after every worker
    // has been joined, but still takes the lock so a late submit_wait/cold_*
    // can never tear the line.
    void dump() const {
        std::lock_guard<std::mutex> lock(mutex_);
        incomplete_ = unsigned(active_[0] || active_[1]);
        const std::uint64_t wall = last_ > first_ ? last_ - first_ : 0;
        std::fprintf(stderr,
            "TP5_DW_THRU_CTR wall_ns=%llu intervals0=%llu intervals1=%llu busy0_ns=%llu busy1_ns=%llu overlap_ns=%llu both_entries=%llu max_concurrent=%u overlap_wall=%.9f overlap_busy0=%.9f overlap_busy1=%.9f submit_wait0_ns=%llu submit_wait1_ns=%llu cold_wait0_ns=%llu cold_wait1_ns=%llu cold_hold_ns=%llu cold_holds=%llu order_violations=%llu discarded_reversed=%llu incomplete=%u\n",
            (unsigned long long)wall,
            (unsigned long long)intervals_[0], (unsigned long long)intervals_[1],
            (unsigned long long)busy_[0], (unsigned long long)busy_[1],
            (unsigned long long)overlap_, (unsigned long long)both_entries_, max_concurrent_,
            wall ? double(overlap_) / double(wall) : 0.0,
            busy_[0] ? double(overlap_) / double(busy_[0]) : 0.0,
            busy_[1] ? double(overlap_) / double(busy_[1]) : 0.0,
            (unsigned long long)submit_wait_[0], (unsigned long long)submit_wait_[1],
            (unsigned long long)cold_wait_[0], (unsigned long long)cold_wait_[1],
            (unsigned long long)cold_hold_, (unsigned long long)cold_holds_,
            (unsigned long long)order_violations_, (unsigned long long)discarded_reversed_,
            incomplete_);
        // Event-order receipt: the per-window sequence ranges and the global
        // monotonic-order violations. A run whose overlap numbers matter must
        // show order_violations=0.
        std::fprintf(stderr,
            "TP5_DW_THRU_ORDER seq0_first=%llu seq0_last=%llu seq1_first=%llu seq1_last=%llu events=%llu order_violations=%llu\n",
            (unsigned long long)seq_first_[0], (unsigned long long)seq_last_[0],
            (unsigned long long)seq_first_[1], (unsigned long long)seq_last_[1],
            (unsigned long long)events_, (unsigned long long)order_violations_);
    }

private:
    static unsigned slot(unsigned window) { return window & 1u; }

    void transition(unsigned window, bool begin) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Sampled under the lock, AFTER acquiring it: the recorded order is
        // the order in which the windows actually acquired the counter, not
        // the order in which they sampled a clock outside it. Sampling before
        // the lock would let two workers' timestamps invert inside the
        // critical section and rewind last_.
        const std::uint64_t now = std::uint64_t(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        const unsigned w = slot(window);
        if (last_ && now < last_) {
            // Never rewind: a rewound last_ would charge one interval twice and
            // leave another uncharged. Drop the event and make it visible.
            ++discarded_reversed_;
            return;
        }
        if (first_ == 0 || now < first_) first_ = now;
        const std::uint64_t elapsed = last_ ? now - last_ : 0;
        if (last_) {
            if (active_[0]) busy_[0] += elapsed;
            if (active_[1]) busy_[1] += elapsed;
            if (active_[0] && active_[1]) overlap_ += elapsed;
        }
        last_ = now;
        active_[w] = begin;
        ++events_;
        ++seq_;
        if (seq_first_[w] == 0) seq_first_[w] = seq_;
        seq_last_[w] = seq_;
        if (begin) {
            ++intervals_[w];
            const unsigned concurrent = unsigned(active_[0]) + unsigned(active_[1]);
            if (concurrent > max_concurrent_) max_concurrent_ = concurrent;
            if (concurrent == 2) ++both_entries_;
        }
        // Per-window sequences must interleave monotonically: window w's
        // events are a subsequence of the global sequence, so its last recorded
        // sequence number is always the newest event of that window.
        if (seq_last_[w] > seq_) ++order_violations_;
    }

    mutable std::mutex mutex_;
    bool active_[2] = {};
    std::uint64_t busy_[2] = {};
    std::uint64_t intervals_[2] = {};
    std::uint64_t submit_wait_[2] = {};
    std::uint64_t cold_wait_[2] = {};
    std::uint64_t seq_first_[2] = {};
    std::uint64_t seq_last_[2] = {};
    std::uint64_t first_ = 0, last_ = 0, overlap_ = 0;
    std::uint64_t both_entries_ = 0, cold_hold_ = 0, cold_holds_ = 0;
    std::uint64_t events_ = 0, seq_ = 0, order_violations_ = 0, discarded_reversed_ = 0;
    unsigned max_concurrent_ = 0;
    mutable unsigned incomplete_ = 0;
};
