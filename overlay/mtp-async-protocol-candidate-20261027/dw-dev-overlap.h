#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

// Caller owns the bounded storage and supplies timestamps from one device and
// one clock domain. Overflow and rejected intervals invalidate completeness.
struct dw_dev_overlap_event {
    std::int64_t t_ns;
    std::int32_t delta;
    std::uint32_t window;
};

struct dw_dev_overlap_summary {
    std::int64_t union_ns = 0;
    std::int64_t overlap_ns = 0;
    std::int64_t covered_once_ns = 0;
    std::uint64_t events = 0;
    std::uint64_t intervals = 0;
    std::uint64_t skipped_bad_interval = 0;
    std::uint64_t skipped_bad_window = 0;
    std::uint64_t dropped_capacity = 0;
    std::uint32_t windows = 0;
    bool overflow = false;
};

class dw_dev_overlap_acc {
public:
    dw_dev_overlap_acc(dw_dev_overlap_event * storage, std::size_t capacity_events)
        : storage_(storage), capacity_(storage ? capacity_events : 0) {}

    bool add(std::int64_t begin_ns, std::int64_t end_ns, std::uint32_t window) {
        if (end_ns <= begin_ns) { ++bad_interval_; return false; }
        if (window >= 64) { ++bad_window_; return false; }
        if (capacity_ - used_ < 2) { ++dropped_; return false; }
        storage_[used_++] = {begin_ns, 1, window};
        storage_[used_++] = {end_ns, -1, window};
        windows_ |= std::uint64_t(1) << window;
        return true;
    }

    std::size_t size() const { return used_ / 2; }

    // Sorting happens only at drain, after all intervals in this epoch arrive.
    // For long-lived services, drain and reset bounded epochs; epochs must not
    // split intervals whose cross-boundary intersections are needed.
    void summarize(dw_dev_overlap_summary * out) {
        dw_dev_overlap_summary result{};
        result.events = used_;
        result.intervals = size();
        result.skipped_bad_interval = bad_interval_;
        result.skipped_bad_window = bad_window_;
        result.dropped_capacity = dropped_;
        result.overflow = dropped_ != 0;
        for (std::uint64_t bits = windows_; bits; bits &= bits - 1) ++result.windows;
        if (used_) {
            std::sort(storage_, storage_ + used_, [](const dw_dev_overlap_event & a, const dw_dev_overlap_event & b) {
                return a.t_ns != b.t_ns ? a.t_ns < b.t_ns : a.delta < b.delta;
            });
            std::size_t active[64] = {};
            unsigned distinct = 0;
            std::int64_t previous = storage_[0].t_ns;
            for (std::size_t i = 0; i < used_; ++i) {
                const auto & e = storage_[i];
                // Unsigned subtraction is defined even across the signed range.
                const std::uint64_t elapsed = std::uint64_t(e.t_ns) - std::uint64_t(previous);
                if (distinct) add_duration(result.union_ns, elapsed, result.overflow);
                if (distinct >= 2) add_duration(result.overlap_ns, elapsed, result.overflow);
                if (e.delta < 0) { if (--active[e.window] == 0) --distinct; }
                else { if (active[e.window]++ == 0) ++distinct; }
                previous = e.t_ns;
            }
        }
        result.covered_once_ns = result.union_ns - result.overlap_ns;
        *out = result;
    }

    void reset() { used_ = 0; windows_ = 0; bad_interval_ = bad_window_ = dropped_ = 0; }

private:
    static void add_duration(std::int64_t & total, std::uint64_t n, bool & overflow) {
        const auto available = std::uint64_t(std::numeric_limits<std::int64_t>::max() - total);
        if (n > available) { total = std::numeric_limits<std::int64_t>::max(); overflow = true; }
        else total += static_cast<std::int64_t>(n);
    }
    dw_dev_overlap_event * storage_;
    std::size_t capacity_, used_ = 0;
    std::uint64_t windows_ = 0, bad_interval_ = 0, bad_window_ = 0, dropped_ = 0;
};
