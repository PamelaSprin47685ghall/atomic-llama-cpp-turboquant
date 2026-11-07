// DUALWIN r8 event-overlap trace (GGML_TP5_DW_EVENT_TRACE=1) — self-contained
// host-correlation trace buffers shared verbatim by the fork tp5-compose and
// overlay mtp-async runtimes (single source of truth: the two copies are kept
// byte-identical by the same sync discipline as the dual-wait method blocks).
//
// HONEST SCOPE: every timestamp here is a HOST steady-clock sample of a
// host-visible observation (the relay snapshot's finished flag, the courier
// retirement equality, the proven dual-conjunction pass). Host correlation
// metadata NEVER proves device-side kernel overlap or device event timing;
// device evidence remains the rocprof route from the r7 GPU-event RUNBOOK.
//
// GATING: the gate is read once at static initialization; unset, "0", "2",
// "1x" or any other value keeps every trace call site a no-op (no timestamps,
// no output, no records). Records are pushed ONLY for epochs consumed through
// the proven dual wait (a fallback/timeout/latched epoch emits nothing).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace dw_event_trace {

static constexpr std::size_t kRingCapacity = 4096;
static constexpr std::size_t kDetailLines  = 8;

// Exact "1" gate, read once (same convention as every gate in this tree).
inline bool gate() {
    static const bool on = [] {
        const char * v = std::getenv("GGML_TP5_DW_EVENT_TRACE");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}

// One line, once per process, at the first trace summary emission.
inline void banner() {
    static const bool printed = [] {
        std::fprintf(stderr,
                     "TP5_DW_EVENT_TRACE enabled=1 flag=GGML_TP5_DW_EVENT_TRACE scope=host_correlation clock=steady_monotonic ring=%zu note=host_timestamps_never_prove_device_overlap\n",
                     kRingCapacity);
        return true;
    }();
    (void) printed;
}

// Event-point trace fields shared with the overlay runtime: host timestamps
// around real ggml_backend_event record/synchronize/wait call sites, per
// window. Still HOST correlation metadata — never device interval evidence.
// wait_calls/wait_first_ns/wait_last_ns are populated by the overlay product
// only (its GGML_TP5_DW_DEVICE_WAIT peer waits); the fork compose runtime
// removed its peer-wait layer, so those three stay 0 there and are not a
// zero-valued measurement. Its record_*/sync_* stamps are host call-site
// correlation for the peer events that no longer exist in the fork either.
struct evpt {
    std::uint64_t record_first_ns = 0;  // first record() call-site stamp
    std::uint64_t record_last_ns  = 0;  // latest record() call-site stamp
    std::uint64_t sync_first_ns   = 0;  // first drain synchronize() entry stamp
    std::uint64_t sync_last_ns    = 0;  // latest synchronize() entry stamp
    std::uint64_t sync_ns         = 0;  // accumulated host time inside synchronize()
    std::uint64_t sync_max_ns     = 0;  // longest single synchronize() host wait
    std::uint64_t wait_calls      = 0;  // device-side waits issued (GGML_TP5_DW_DEVICE_WAIT=1)
    std::uint64_t wait_first_ns   = 0;  // first device-side wait() call-site stamp
    std::uint64_t wait_last_ns    = 0;  // latest device-side wait() call-site stamp
};


struct rec {
    std::uint64_t seq         = 0;  // per-window traced-epoch sequence, 0-based
    std::uint64_t relay_epoch = 0;
    std::uint64_t entry_ns    = 0;  // dual wait entry (post-latch check)
    std::uint64_t dev_ns      = 0;  // first host observation of status.finished
    std::uint64_t cour_ns     = 0;  // first host observation of retirement equality
    std::uint64_t consumed_ns = 0;  // the proven dual-conjunction pass
    int           order       = 0;  // 1 device-first, 2 courier-first, 0 same-pass
};

// Single-writer ring: only the window's own dual-wait worker pushes; the exit
// dump reads after the service stopped and every lane joined. No atomics.
// The fixed ring is allocated lazily on the first push, so a gate-OFF window
// never allocates or initializes it (OFF adds one unallocated empty buf per
// window, nothing more). dev_ns/cour_ns of a pushed record are always
// initialized: the push only happens on a proven conjunction pass, where
// both sightings were stamped (with a defensive fallback to consumed_ns at
// the push, so a record can never carry an unset 0 stamp).
class buf {
  public:
    void push(const rec & r) {
        if (ring_ == nullptr) {
            // Lazy allocation: first traced epoch in this window pays one
            // allocation; gate-OFF windows never reach this line.
            ring_.reset(new rec[kRingCapacity]);
        }
        rec & slot      = ring_[head_];
        slot            = r;
        slot.seq        = pushed_;
        head_           = (head_ + 1) % kRingCapacity;
        if (count_ < kRingCapacity) {
            ++count_;
        } else {
            ++dropped_;
        }
        ++pushed_;
        if (r.order == 1) {
            ++device_first_;
        } else if (r.order == 2) {
            ++courier_first_;
        } else {
            ++same_pass_;
        }
    }

    std::uint64_t pushed() const noexcept { return pushed_; }
    std::uint64_t dropped() const noexcept { return dropped_; }
    std::uint64_t device_first() const noexcept { return device_first_; }
    std::uint64_t courier_first() const noexcept { return courier_first_; }
    std::uint64_t same_pass() const noexcept { return same_pass_; }

    // Held record with per-window sequence s, or nullptr when evicted/never
    // pushed. The ring holds the last min(pushed_, kRingCapacity) records.
    // Layout: while the ring is not full the seq-s record sits at ring_[s];
    // once full, head_ points at the OLDEST record. Both cases are covered
    // by (head_ + kRingCapacity - (pushed_ - s)) % kRingCapacity: the
    // unsigned subtraction never underflows because pushed_ - s <= count_ <=
    // kRingCapacity.
    const rec * by_seq(std::uint64_t s) const noexcept {
        if (ring_ == nullptr || s >= pushed_) return nullptr;
        const std::uint64_t first_seq = pushed_ - count_;
        if (s < first_seq) return nullptr;
        return &ring_[(head_ + kRingCapacity - (pushed_ - s)) % kRingCapacity];
    }

    // Per-window summary line plus the first kDetailLines chronological
    // detail lines (held records only; evicted epochs are counted, not
    // printed).
    void dump_win(std::size_t window) const {
        std::fprintf(stderr,
                     "TP5_DW_EVENT_TRACE_WIN window=%zu epochs=%llu recorded=%llu dropped=%llu "
                     "device_first=%llu courier_first=%llu same_tick=%llu\n",
                     window, (unsigned long long) pushed_, (unsigned long long) (pushed_ - dropped_),
                     (unsigned long long) dropped_, (unsigned long long) device_first_,
                     (unsigned long long) courier_first_, (unsigned long long) same_pass_);
        std::uint64_t printed = 0;
        for (std::uint64_t s = (pushed_ > count_ ? pushed_ - count_ : 0); s < pushed_ && printed < kDetailLines;
             ++s) {
            const rec * r = by_seq(s);
            if (r == nullptr) continue;
            std::fprintf(stderr,
                         "TP5_DW_EVENT_TRACE window=%zu epoch=%llu entry_ns=%llu dev_ns=%llu "
                         "cour_ns=%llu order=%s\n",
                         window, (unsigned long long) r->relay_epoch, (unsigned long long) r->entry_ns,
                         (unsigned long long) r->dev_ns, (unsigned long long) r->cour_ns,
                         r->order == 1 ? "device_first" : (r->order == 2 ? "courier_first" : "same_tick"));
            ++printed;
        }
    }

    // Cross-window host-interval overlap for one window pair. Records are
    // paired by EQUAL PER-WINDOW TRACED-EPOCH SEQUENCE — an INDEX pairing
    // assumption (pairing=traced_seq_index), NOT a proven same-request or
    // same-relay-epoch correlation; the line carries the pairing name so
    // no reader can mistake it for a verified identity join. Both
    // directions are interval intersections of HOST observation spans;
    // they are correlation metadata and never device evidence.
    static std::uint64_t intersect_ns(std::uint64_t a0, std::uint64_t a1, std::uint64_t b0,
                                      std::uint64_t b1) noexcept {
        const std::uint64_t lo = a0 > b0 ? a0 : b0;
        const std::uint64_t hi = a1 < b1 ? a1 : b1;
        return hi > lo ? hi - lo : 0;
    }

    void dump_overlap(std::size_t a_index, std::size_t b_index, const buf & other) const {
        const std::uint64_t first_seq =
            (pushed_ > count_ ? pushed_ - count_ : 0) > (other.pushed_ > other.count_ ? other.pushed_ - other.count_ : 0)
                ? (pushed_ > count_ ? pushed_ - count_ : 0)
                : (other.pushed_ > other.count_ ? other.pushed_ - other.count_ : 0);
        std::uint64_t pairs = 0, a_dev_b_cour = 0, b_dev_a_cour = 0, max_overlap = 0;
        for (std::uint64_t s = first_seq; s < pushed_ && s < other.pushed_; ++s) {
            const rec * a = by_seq(s);
            const rec * b = other.by_seq(s);
            if (a == nullptr || b == nullptr) continue;
            ++pairs;
            const std::uint64_t o1 = intersect_ns(a->dev_ns, a->consumed_ns, b->cour_ns, b->consumed_ns);
            const std::uint64_t o2 = intersect_ns(b->dev_ns, b->consumed_ns, a->cour_ns, a->consumed_ns);
            a_dev_b_cour += o1;
            b_dev_a_cour += o2;
            if (o1 > max_overlap) max_overlap = o1;
            if (o2 > max_overlap) max_overlap = o2;
        }
        std::fprintf(stderr,
                     "TP5_DW_EVENT_TRACE_OVERLAP a=%zu b=%zu pairing=traced_seq_index "
                     "seq_pairs=%llu a_dev_b_courier_ns=%llu "
                     "b_dev_a_courier_ns=%llu max_overlap_ns=%llu\n",
                     a_index, b_index, (unsigned long long) pairs, (unsigned long long) a_dev_b_cour,
                     (unsigned long long) b_dev_a_cour, (unsigned long long) max_overlap);
    }

  private:
    std::unique_ptr<rec[]> ring_;          // lazily allocated on first push
    std::size_t   head_                = 0;  // next write position
    std::size_t   count_               = 0;  // currently held records
    std::uint64_t pushed_              = 0;  // total records ever pushed (== seq space)
    std::uint64_t dropped_             = 0;  // records evicted by the ring
    std::uint64_t device_first_        = 0;
    std::uint64_t courier_first_       = 0;
    std::uint64_t same_pass_           = 0;
};

}  // namespace dw_event_trace
