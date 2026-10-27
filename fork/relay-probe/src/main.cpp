// TP5 mapped-relay platform probe (guide 31, 39.5 H-cases).
//
// Proves, on every present device, the capabilities the strict pre-committed
// relay needs from mapped host memory: running-visibility handshakes in both
// directions, hot reuse of the same slot across many generations with exact
// ticket matching, producer delay ordering, multi-workgroup completion, and
// bounded abort. Host-memory system RMW is probed separately because the
// design must not require it (guide 31.3). Five-rank cross-device cases run
// only when exactly five devices are present; otherwise they stay NOT_RUN.
//
// Control words are plain uint64_t; host and device access them with the
// same __atomic builtins, never through std::atomic reinterpretation
// (guide 31.6). This program reports platform facts: it does not qualify a
// model, a scheduler, or an end-to-end relay (guide 31.7).
//
// HIP failures never erase collected evidence: a per-case HIP error marks
// that case failed and moves on, a device that fails setup still appears in
// the report, and the five-rank case attributes errors by rank. Only a
// failure before any device can be enumerated aborts without a report.

#include "evidence.h"

#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <optional>
#include <ratio>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using tq::relay::probe::case_id;
using tq::relay::probe::device_record;
using tq::relay::probe::five_rank_case_records;
using tq::relay::probe::fold_partials;
using tq::relay::probe::outcome;
using tq::relay::probe::pattern_word;
using tq::relay::probe::probe_report;
using tq::relay::probe::rank_fold_weight;
using tq::relay::probe::to_json;
using tq::relay::probe::write_file_exclusive;
using tq::relay::probe::write_result;

// Geometry shared with the evidence layer so the config tests (no HIP) can
// pin the fold gate on the same constants the kernels use.
constexpr std::uint32_t PAYLOAD_WORDS  = tq::relay::probe::relay_payload_words;
constexpr std::uint32_t RANKS          = tq::relay::probe::relay_ranks;
constexpr std::uint64_t GUARD_VALUE    = 0x1F1E1D1C1B1A1918ULL;
constexpr std::uint64_t TICKET_STALE   = 0;  // below any issued ticket
constexpr std::uint32_t DEFAULT_ROUNDS = 2000;

// Kernel exit codes, written to host memory and read after completion.
enum device_exit : std::uint32_t {
    EXIT_OK               = 0,
    EXIT_MISMATCH         = 1,  // observed an impossible future ticket
    EXIT_TIMEOUT_EXPECTED = 2,  // spin budget ran out while waiting, as designed
    EXIT_ABORT_OBSERVED   = 3,  // stop word was set; left before consuming
};

struct control_block {
    std::uint64_t ticket;      // CPU -> GPU generation word
    std::uint64_t ready;       // GPU -> CPU generation word
    std::uint64_t ack;         // CPU -> GPU round acknowledgment
    std::uint64_t abort;       // stop word, owner CPU
    std::uint64_t half_done;   // delayed producer first-half marker
    std::uint64_t first_done;  // ordering trap marker: max-index group finished
    std::uint64_t exit_code;   // kernel exit code
};

struct slot {
    std::uint64_t guard_lo[2];
    std::uint64_t payload[PAYLOAD_WORDS];
    std::uint64_t guard_hi[2];
};

std::string format_error(const char * what, hipError_t error) {
    return std::string(what) + ": " + hipGetErrorString(error);
}

class hip_error : public std::exception {
  public:
    explicit hip_error(std::string message) : message_(std::move(message)) {}

    const char * what() const noexcept override { return message_.c_str(); }

  private:
    std::string message_;
};

void check_hip(hipError_t error, const char * what) {
    if (error != hipSuccess) {
        throw hip_error(format_error(what, error));
    }
}

// -------- shared atomics (host and device, same representation) ------------

__host__ __device__ inline std::uint64_t acquire_load(const std::uint64_t * word) {
    return __atomic_load_n(word, __ATOMIC_ACQUIRE);
}

__host__ __device__ inline void release_store(std::uint64_t * word, std::uint64_t value) {
    __atomic_store_n(word, value, __ATOMIC_RELEASE);
}

__host__ inline void host_pause() {
    __builtin_ia32_pause();
}

__host__ inline void host_fence_release() {
    __atomic_thread_fence(__ATOMIC_RELEASE);
}

__device__ inline void device_fence_system() {
    __threadfence_system();
}

__device__ inline void device_pause() {
    __builtin_amdgcn_s_sleep(2);
}

__device__ inline void spin_budget(std::uint64_t clock_ticks) {
    const std::uint64_t begin = clock64();
    while (clock64() - begin < clock_ticks) {
        device_pause();
    }
}

// Cumulative device wait budget. clock64() on gfx1030 ticks at the shader
// clock, measured at ~2.3 GHz on this machine (2026-10-19); re-measure on
// the target machine with the tq-clock-rate tool before relying on the
// margins. 24e9 ticks spans ~8.3 s at 2.9 GHz and more when the clock runs
// slower, so the device always outlasts the longest host-side deadline (5 s)
// for these waits: a genuine stall fails on the host, never as a
// device-side false timeout. The budget only bounds real stalls; hot rounds
// re-check every pause and stay fast.
constexpr std::uint64_t WAIT_BUDGET_TICKS = 24000000000ULL;

enum class wait_outcome : std::uint32_t { matched, mismatch, aborted, timed_out };

// Wait for the exact ticket. Stale values keep waiting; a future value is an
// identity violation. Re-checks every device_pause and only times out after
// the cumulative budget has elapsed.
__device__ wait_outcome wait_for_ticket(control_block * cb, std::uint64_t expected) {
    const std::uint64_t begin = clock64();
    while (true) {
        const std::uint64_t observed = acquire_load(&cb->ticket);
        if (observed == expected) {
            return wait_outcome::matched;
        }
        if (observed > expected) {
            release_store(&cb->exit_code, EXIT_MISMATCH);
            return wait_outcome::mismatch;
        }
        if (acquire_load(&cb->abort) != 0) {
            release_store(&cb->exit_code, EXIT_ABORT_OBSERVED);
            return wait_outcome::aborted;
        }
        device_pause();
        if (clock64() - begin >= WAIT_BUDGET_TICKS) {
            release_store(&cb->exit_code, EXIT_TIMEOUT_EXPECTED);
            return wait_outcome::timed_out;
        }
    }
}

// Wait for the host's per-round acknowledgment, same cumulative contract.
__device__ wait_outcome wait_for_ack(control_block * cb, std::uint64_t generation) {
    const std::uint64_t begin = clock64();
    while (true) {
        if (acquire_load(&cb->ack) == generation) {
            return wait_outcome::matched;
        }
        if (acquire_load(&cb->abort) != 0) {
            release_store(&cb->exit_code, EXIT_ABORT_OBSERVED);
            return wait_outcome::aborted;
        }
        device_pause();
        if (clock64() - begin >= WAIT_BUDGET_TICKS) {
            release_store(&cb->exit_code, EXIT_TIMEOUT_EXPECTED);
            return wait_outcome::timed_out;
        }
    }
}

// -------- kernels ----------------------------------------------------------

__global__ void write_pattern_kernel(slot * s, std::uint64_t generation, std::uint32_t seed) {
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        s->payload[i] = pattern_word(generation, i, seed);
    }
}

// CPU published a generation; the device acquires the exact ticket, verifies
// payload and guards, and publishes a derived word back to the CPU.
__global__ void consume_kernel(control_block * cb,
                               slot *          s,
                               std::uint64_t * out_word,
                               std::uint64_t   expected,
                               std::uint32_t   seed) {
    switch (wait_for_ticket(cb, expected)) {
        case wait_outcome::matched:
            break;
        default:
            return;
    }
    bool valid = s->guard_lo[0] == GUARD_VALUE && s->guard_lo[1] == GUARD_VALUE && s->guard_hi[0] == GUARD_VALUE &&
                 s->guard_hi[1] == GUARD_VALUE;
    std::uint64_t acc = 0;
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        valid = valid && (s->payload[i] == pattern_word(expected, i, seed));
        acc ^= s->payload[i];
    }
    *out_word = valid ? acc : 0xDEADBEEFDEADBEEFULL;
    device_fence_system();
    release_store(&cb->ack, expected);
}

// The device produces many generations into the same slot, waiting for the
// CPU acknowledgment of each round. Bounded by the ack spin budget.
__global__ void produce_loop_kernel(control_block * cb, slot * s, std::uint32_t rounds, std::uint32_t seed) {
    for (std::uint64_t generation = 1; generation <= rounds; ++generation) {
        for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
            s->payload[i] = pattern_word(generation, i, seed);
        }
        device_fence_system();
        release_store(&cb->ready, generation);
        switch (wait_for_ack(cb, generation)) {
            case wait_outcome::matched:
                break;
            default:
                return;
        }
    }
    release_store(&cb->exit_code, EXIT_OK);
}

// Writes the first half, publishes half_done, delays on the device, then
// writes the second half and publishes ready (guide 39.5 H07).
__global__ void delayed_producer_kernel(control_block * cb, slot * s, std::uint32_t seed) {
    const std::uint64_t generation = 1;
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS / 2; ++i) {
        s->payload[i] = pattern_word(generation, i, seed);
    }
    device_fence_system();
    release_store(&cb->half_done, 1);
    spin_budget(400000000ULL);  // ~174 ms at the measured 2.3 GHz clock64 (tq-clock-rate); margin for the 40 ms gate
    for (std::uint32_t i = PAYLOAD_WORDS / 2; i < PAYLOAD_WORDS; ++i) {
        s->payload[i] = pattern_word(generation, i, seed);
    }
    device_fence_system();
    release_store(&cb->ready, 1);
}

// Waits for an exact ticket. Stale values keep waiting; a future value is an
// identity violation; the abort word forces a bounded exit.
__global__ void ticket_wait_kernel(control_block * cb, std::uint64_t expected) {
    for (int budget = 8; budget > 0; --budget) {
        const std::uint64_t observed = acquire_load(&cb->ticket);
        if (observed == expected) {
            release_store(&cb->exit_code, EXIT_OK);
            return;
        }
        if (observed > expected) {
            release_store(&cb->exit_code, EXIT_MISMATCH);
            return;
        }
        if (acquire_load(&cb->abort) != 0) {
            release_store(&cb->exit_code, EXIT_ABORT_OBSERVED);
            return;
        }
        spin_budget(20000000ULL);
    }
    release_store(&cb->exit_code, EXIT_TIMEOUT_EXPECTED);
}

// Deliberately stalls the queue far longer than the host's short sync
// deadline, so the bounded-sync timeout path runs on the live device instead
// of shipping unexecuted to the operator. Exits only on the stop word or its
// own long budget; the waiter must recover so the following drain completes.
__global__ void stall_kernel(control_block * cb) {
    const std::uint64_t begin = clock64();
    while (acquire_load(&cb->abort) == 0) {
        device_pause();
        if (clock64() - begin >= WAIT_BUDGET_TICKS) {
            release_store(&cb->exit_code, EXIT_TIMEOUT_EXPECTED);
            return;
        }
    }
    release_store(&cb->exit_code, EXIT_ABORT_OBSERVED);
}

// Resident waiter with no payload access: it only proves that wait waves of
// one window hold occupancy while another window's work must still advance
// (guide 39.6 P05). The cumulative budget keeps the block honest without the
// short fixed budget of ticket_wait_kernel; every block writes the same
// EXIT_OK word on a match, so any non-match path is visible to the host.
__global__ void bare_waiter_kernel(control_block * cb, std::uint64_t expected) {
    switch (wait_for_ticket(cb, expected)) {
        case wait_outcome::matched:
            release_store(&cb->exit_code, EXIT_OK);
            return;
        default:
            return;
    }
}

// Finishes the whole verified read, publishes the per-consumer result word,
// then withholds the acknowledgment until the stop word or its cumulative
// budget (guide 39.7 F07): the host must rely on its bounded wait, and the
// slot must not be reused while this reader is un-acknowledged.
__global__ void read_hold_kernel(control_block * cb,
                                 slot *          s,
                                 std::uint64_t * result,
                                 std::uint32_t   consumer,
                                 std::uint64_t   expected,
                                 std::uint32_t   seed) {
    switch (wait_for_ticket(cb, expected)) {
        case wait_outcome::matched:
            break;
        default:
            return;
    }
    bool valid = s->guard_lo[0] == GUARD_VALUE && s->guard_lo[1] == GUARD_VALUE && s->guard_hi[0] == GUARD_VALUE &&
                 s->guard_hi[1] == GUARD_VALUE;
    std::uint64_t acc = 0;
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        valid = valid && (s->payload[i] == pattern_word(expected, i, seed));
        acc ^= s->payload[i];
    }
    result[consumer] = valid ? acc : 0xDEADBEEFDEADBEEFULL;
    device_fence_system();
    const std::uint64_t begin = clock64();
    while (acquire_load(&cb->abort) == 0) {
        device_pause();
        if (clock64() - begin >= WAIT_BUDGET_TICKS) {
            release_store(&cb->exit_code, EXIT_TIMEOUT_EXPECTED);
            return;
        }
    }
    release_store(&cb->exit_code, EXIT_ABORT_OBSERVED);
}

// Each participating workgroup writes its own payload word, orders it with a
// system fence, then bumps the device counter; the last participant publishes
// ready. live_groups may be smaller than the launched grid (guide 39.5 P03).
__global__ void group_writer_kernel(std::uint64_t * payload,
                                    std::uint32_t * counter,
                                    control_block * cb,
                                    std::uint32_t   live_groups,
                                    std::uint32_t   seed) {
    if (blockIdx.x >= live_groups) {
        return;
    }
    payload[blockIdx.x] = pattern_word(live_groups, blockIdx.x, seed);
    device_fence_system();
    const std::uint32_t old = atomicAdd(counter, 1u);
    if (old == live_groups - 1) {
        release_store(&cb->ready, live_groups);
    }
}

// Forced-out-of-order completion (guide 39.6 P01, P02): every workgroup
// spins a deterministic delay that shrinks with its index, so the
// maximum-index group provably finishes first and the minimum-index group
// last. A publisher that fires from the maximum index instead of the true
// last arrival is caught deterministically, not by scheduling luck. The
// first finisher publishes first_done, which gives the host a proper
// release/acquire pair for observing the trap window.
__global__ void ordered_group_writer_kernel(std::uint64_t * payload,
                                            std::uint32_t * counter,
                                            control_block * cb,
                                            std::uint32_t   seed,
                                            std::uint64_t   delay_step) {
    if (blockIdx.x >= 8) {
        return;
    }
    spin_budget((7 - blockIdx.x) * delay_step);
    payload[blockIdx.x] = pattern_word(8, blockIdx.x, seed);
    device_fence_system();
    const std::uint32_t old = atomicAdd(counter, 1u);
    if (blockIdx.x == 7) {
        release_store(&cb->first_done, 8);
    }
    if (old == 7) {
        release_store(&cb->ready, 8);
    }
}

// Single-shot system-scope RMW attempts on host memory. If the platform does
// not execute them, the words stay unchanged and the host records unsupported.
__global__ void rmw_probe_kernel(std::uint64_t * fetch_add_word, std::uint64_t * cas_word) {
    __atomic_fetch_add(fetch_add_word, 7, __ATOMIC_RELAXED);
    std::uint64_t expected = 0;
    __atomic_compare_exchange_n(cas_word, &expected, 9, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
}

// Five-rank case: each device writes one distinct word of the shared payload
// for the current generation and releases its own ready word. The value
// varies per round, so a stale write from an earlier round mismatches.
__global__ void rank_write_kernel(std::uint64_t * payload,
                                  std::uint64_t * ready,
                                  std::uint32_t   rank,
                                  std::uint32_t   seed,
                                  std::uint64_t   generation) {
    payload[rank] = pattern_word(generation, rank, seed);
    device_fence_system();
    // Every rank owns its own ready word (guide 31.3 single-writer rule).
    // Writing the base pointer would make all five ranks stomp ready[0] and
    // leave ready[1..4] unpublished: the host would wait out the whole 5 s
    // deadline every round and attribute a phantom stall to rank 1. The
    // five_rank_indexing case runs this exact kernel on one device, so the
    // indexing stays proven on every dev machine.
    release_store(&ready[rank], generation);
}

// Acquire the exact ticket, read the whole slot, publish a per-consumer
// verification word, then acknowledge. The fence orders the completed read
// before the acknowledgment, which is the device-side output-credit contract
// (guide 32.2, 33.5).
__global__ void consume_read_kernel(control_block * cb,
                                    slot *          s,
                                    std::uint64_t * ack,
                                    std::uint64_t * result,
                                    std::uint32_t   consumer,
                                    std::uint64_t   expected,
                                    std::uint32_t   seed) {
    switch (wait_for_ticket(cb, expected)) {
        case wait_outcome::matched:
            break;
        default:
            return;
    }
    bool valid = s->guard_lo[0] == GUARD_VALUE && s->guard_lo[1] == GUARD_VALUE && s->guard_hi[0] == GUARD_VALUE &&
                 s->guard_hi[1] == GUARD_VALUE;
    std::uint64_t acc = 0;
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        valid = valid && (s->payload[i] == pattern_word(expected, i, seed));
        acc ^= s->payload[i];
    }
    result[consumer] = valid ? acc : 0xDEADBEEFDEADBEEFULL;
    device_fence_system();
    release_store(&ack[consumer], expected);
}

// Delayed last reader (guide 39.5 H08): reads the first half, publishes
// progress, pauses on the device, then finishes the read and only then
// acknowledges. The acknowledgment must be withheld for the whole read.
__global__ void delayed_reader_kernel(control_block * cb,
                                      slot *          s,
                                      std::uint64_t * ack,
                                      std::uint64_t * result,
                                      std::uint32_t   consumer,
                                      std::uint64_t   expected,
                                      std::uint32_t   seed) {
    switch (wait_for_ticket(cb, expected)) {
        case wait_outcome::matched:
            break;
        default:
            return;
    }
    bool valid = s->guard_lo[0] == GUARD_VALUE && s->guard_lo[1] == GUARD_VALUE && s->guard_hi[0] == GUARD_VALUE &&
                 s->guard_hi[1] == GUARD_VALUE;
    std::uint64_t acc = 0;
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS / 2; ++i) {
        valid = valid && (s->payload[i] == pattern_word(expected, i, seed));
        acc ^= s->payload[i];
    }
    device_fence_system();
    release_store(&cb->half_done, 1);
    spin_budget(400000000ULL);  // ~174 ms at the measured 2.3 GHz clock64 (tq-clock-rate); margin for the 40 ms gate
    for (std::uint32_t i = PAYLOAD_WORDS / 2; i < PAYLOAD_WORDS; ++i) {
        valid = valid && (s->payload[i] == pattern_word(expected, i, seed));
        acc ^= s->payload[i];
    }
    result[consumer] = valid ? acc : 0xDEADBEEFDEADBEEFULL;
    device_fence_system();
    release_store(&ack[consumer], expected);
}

// Alternating two-bank producer: the ticket-to-bank derivation must keep the
// written bank correct and never touch the other bank (guide 32.7).
__global__ void produce_bank_kernel(control_block * cb,
                                    slot *          s0,
                                    slot *          s1,
                                    std::uint32_t   rounds,
                                    std::uint32_t   seed) {
    for (std::uint64_t generation = 1; generation <= rounds; ++generation) {
        slot * const target = ((generation - 1) & 1) == 0 ? s0 : s1;
        for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
            target->payload[i] = pattern_word(generation, i, seed);
        }
        device_fence_system();
        release_store(&cb->ready, generation);
        switch (wait_for_ack(cb, generation)) {
            case wait_outcome::matched:
                break;
            default:
                return;
        }
    }
    release_store(&cb->exit_code, EXIT_OK);
}

// Varying live extent (guide 39.5 H05): only the live prefix is written; the
// tail keeps its previous value and must be neither written nor read.
__global__ void produce_variable_kernel(control_block * cb, slot * s, std::uint32_t rounds, std::uint32_t seed) {
    for (std::uint64_t generation = 1; generation <= rounds; ++generation) {
        const std::uint32_t live = 2 + static_cast<std::uint32_t>((generation - 1) % 6);
        for (std::uint32_t i = 0; i < live; ++i) {
            s->payload[i] = pattern_word(generation, i, seed);
        }
        device_fence_system();
        release_store(&cb->ready, generation);
        switch (wait_for_ack(cb, generation)) {
            case wait_outcome::matched:
                break;
            default:
                return;
        }
    }
    release_store(&cb->exit_code, EXIT_OK);
}

// One rank's partial contribution for the emulated relay cycle (guide 31.3:
// each rank writes its own row and publishes its own ready word; no
// host-memory RMW is used anywhere). The weight identity lives in the
// evidence layer so the no-HIP config tests pin the same constants.
__global__ void partial_producer_kernel(std::uint64_t * partials,
                                        std::uint64_t * ready,
                                        std::uint32_t   rank,
                                        std::uint64_t   generation,
                                        std::uint32_t   seed) {
    std::uint64_t * const row    = partials + rank * PAYLOAD_WORDS;
    const std::uint64_t   weight = rank_fold_weight(rank);
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        row[i] = pattern_word(generation, i, seed) * weight;
    }
    device_fence_system();
    release_store(&ready[rank], generation);
}

// Single-shot producer used by the latency bench: no acknowledgment loop.
__global__ void produce_once_kernel(control_block * cb, slot * s, std::uint64_t generation, std::uint32_t seed) {
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        s->payload[i] = pattern_word(generation, i, seed);
    }
    device_fence_system();
    release_store(&cb->ready, generation);
}

// -------- host helpers -----------------------------------------------------

class stopwatch {
  public:
    using clock              = std::chrono::steady_clock;
    clock::time_point begin_ = clock::now();

    double milliseconds() const { return std::chrono::duration<double, std::milli>(clock::now() - begin_).count(); }
};

template <class Ready> bool spin_until(Ready ready, double deadline_ms, double & elapsed) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double, std::milli>(deadline_ms);
    while (true) {
        if (ready()) {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            elapsed = -1;
            return false;
        }
        host_pause();
    }
}

// -------- bounded native synchronization ----------------------------------

// Bare hipDeviceSynchronize / hipStreamSynchronize never return when a
// device stops completing work, and a run that never returns is the worst
// failure an operator can hit: no report, no attribution (SPEC 7). All
// native syncs therefore poll an event recorded after the submitted work,
// bounded by a host deadline far beyond any healthy sync; a healthy queue
// completes in microseconds, so the deadline only bounds stalls on an
// unhealthy device (guide 35.8 drain deadline). The timeout is raised as a
// regular hip_error so the existing per-case handling records it as that
// case's failure and moves on. The event is destroyed on every path and is
// never reused after expiry.
constexpr double SYNC_DEADLINE_MS        = 60000.0;
constexpr double STALL_PROBE_DEADLINE_MS = 2000.0;

void sync_with_deadline(hipStream_t stream, double deadline_ms, const char * what) {
    hipEvent_t event = nullptr;
    check_hip(hipEventCreate(&event), what);
    const hipError_t recorded = hipEventRecord(event, stream);
    if (recorded != hipSuccess) {
        (void) hipEventDestroy(event);
        throw hip_error(format_error(what, recorded));
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double, std::milli>(deadline_ms);
    while (true) {
        const hipError_t queried = hipEventQuery(event);
        if (queried == hipSuccess) {
            (void) hipEventDestroy(event);
            return;
        }
        if (queried != hipErrorNotReady) {
            (void) hipEventDestroy(event);
            throw hip_error(format_error(what, queried));
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            (void) hipEventDestroy(event);
            throw hip_error(std::string(what) + ": native sync deadline exceeded after " + std::to_string(deadline_ms) +
                            " ms; the device did not complete the submitted work");
        }
        host_pause();
    }
}

std::uint64_t derived_word(std::uint64_t generation, std::uint32_t seed) {
    std::uint64_t acc = 0;
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        acc ^= pattern_word(generation, i, seed);
    }
    return acc;
}

bool guards_intact(const slot & s) {
    return s.guard_lo[0] == GUARD_VALUE && s.guard_lo[1] == GUARD_VALUE && s.guard_hi[0] == GUARD_VALUE &&
           s.guard_hi[1] == GUARD_VALUE;
}

void init_guards(slot & s) {
    s.guard_lo[0] = s.guard_lo[1] = s.guard_hi[0] = s.guard_hi[1] = GUARD_VALUE;
}

// -------- cases ------------------------------------------------------------

struct device_context {
    device_record   record;
    control_block * cb            = nullptr;
    control_block * cb2           = nullptr;  // second independent channel (dual-window case)
    slot *          slot_ptr      = nullptr;
    slot *          slot_b        = nullptr;  // second channel payload
    slot *          slot2_ptr     = nullptr;  // second bank for two_bank_rotation
    std::uint64_t * out_word      = nullptr;
    std::uint32_t * counter       = nullptr;
    std::uint64_t * rmw_add       = nullptr;
    std::uint64_t * rmw_cas       = nullptr;
    std::uint64_t * rank_ack      = nullptr;  // RANKS words: consumer acknowledgments
    std::uint64_t * rank_result   = nullptr;  // RANKS words: per-consumer verification
    std::uint64_t * rank_ready    = nullptr;  // RANKS words: producer ready
    std::uint64_t * rank_partials = nullptr;  // RANKS * PAYLOAD_WORDS: producer partials
    std::uint32_t   rounds        = DEFAULT_ROUNDS;
    std::uint32_t   seed          = 0;
    bool            mapped        = false;
    std::string     mapped_flags;
};

// Handshake cases that require mapped host memory, minus mapped_host_access
// itself, which carries the availability detail in its own record.
constexpr case_id handshake_cases[] = {
    case_id::cpu_to_gpu_publish,
    case_id::gpu_to_cpu_publish,
    case_id::producer_delay_ordering,
    case_id::ticket_semantics,
    case_id::multi_group_completion,
    case_id::abort_stop,
    case_id::host_system_rmw_u64_add,
    case_id::host_system_rmw_u64_cas,
    case_id::multi_consumer_output_credit,
    case_id::delayed_consumer_ordering,
    case_id::two_bank_rotation,
    case_id::varying_live_bytes,
    case_id::five_stream_relay_cycle,
    case_id::offset_addressing,
    case_id::multi_group_ordering,
    case_id::native_sync_deadline,
    case_id::same_stream_inverse_dependency,
    case_id::dual_window_independent_progress,
    case_id::service_loop_independent_progress,
    case_id::nonfinite_reduction_gate,
    case_id::host_wait_deadline_no_reuse,
    case_id::five_rank_indexing,
};

outcome result_of(const device_context & ctx, case_id id) {
    for (const auto & item : ctx.record.cases) {
        if (item.id == id) {
            return item.result;
        }
    }
    return outcome::not_run;
}

void set_case(device_context & ctx, case_id id, outcome result, std::string detail) {
    for (auto & item : ctx.record.cases) {
        if (item.id == id) {
            item.result = result;
            item.detail = std::move(detail);
            return;
        }
    }
    ctx.record.cases.push_back({ id, result, std::move(detail) });
}

// Run one case. A HIP failure inside the case must not erase the evidence
// collected so far: it is recorded as this case's failure, the stop word is
// set for any still-waiting device work, and a best-effort drain keeps the
// next case starting from a quiescent device (guide 47.2, 53.5).
void run_case(device_context & ctx, case_id id, void (*run)(device_context &)) {
    try {
        run(ctx);
    } catch (const std::exception & error) {
        set_case(ctx, id, outcome::fail, std::string("unexpected failure: ") + error.what());
        if (ctx.cb != nullptr) {
            release_store(&ctx.cb->abort, 1);
        }
        if (ctx.record.ordinal >= 0) {
            (void) hipSetDevice(ctx.record.ordinal);
            try {
                sync_with_deadline(0, SYNC_DEADLINE_MS, "drain after case failure");
            } catch (const std::exception &) {
                // The failure is already recorded; a stalled drain must not
                // prevent the next case from being attempted.
            }
        }
    }
}

// Record a failure, abort still-waiting device work and drain it before the
// case returns. Device wait budgets deliberately outlast the host deadlines,
// so a kernel stranded by a stall must be stopped and drained here; without
// this its timeout store could land in the shared control block after the
// next case has reset it. The drain never throws: the failure itself is the
// evidence, and a dead device must not prevent the report from being written.
void fail_case(device_context & ctx, case_id id, outcome result, std::string detail) {
    release_store(&ctx.cb->abort, 1);
    try {
        sync_with_deadline(0, SYNC_DEADLINE_MS, "drain after failure");
    } catch (const std::exception & error) {
        detail += " (drain failed: ";
        detail += error.what();
        detail += ")";
    }
    set_case(ctx, id, result, std::move(detail));
}

void reset_control(control_block & cb) {
    release_store(&cb.ticket, 0);
    release_store(&cb.ready, 0);
    release_store(&cb.ack, 0);
    release_store(&cb.abort, 0);
    release_store(&cb.half_done, 0);
    release_store(&cb.first_done, 0);
    release_store(&cb.exit_code, 0);
}

void case_mapped_access(device_context & ctx) {
    reset_control(*ctx.cb);
    init_guards(*ctx.slot_ptr);
    std::memset(ctx.slot_ptr->payload, 0, sizeof(ctx.slot_ptr->payload));
    write_pattern_kernel<<<1, 64>>>(ctx.slot_ptr, 42, ctx.seed);
    sync_with_deadline(0, SYNC_DEADLINE_MS, "mapped access sync");
    bool ok = true;
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        ok = ok && ctx.slot_ptr->payload[i] == pattern_word(42, i, ctx.seed);
    }
    if (!ok || !guards_intact(*ctx.slot_ptr)) {
        set_case(ctx, case_id::mapped_host_access, outcome::fail,
                 "payload or guards wrong after a write-then-sync round trip");
        return;
    }
    set_case(ctx, case_id::mapped_host_access, outcome::pass,
             "device writes to the mapped host slot are CPU-visible after sync");
}

void case_cpu_to_gpu(device_context & ctx) {
    reset_control(*ctx.cb);
    init_guards(*ctx.slot_ptr);
    for (std::uint32_t generation = 1; generation <= ctx.rounds; ++generation) {
        for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
            ctx.slot_ptr->payload[i] = pattern_word(generation, i, ctx.seed);
        }
        release_store(&ctx.cb->exit_code, EXIT_OK);
        host_fence_release();
        release_store(&ctx.cb->ticket, generation);
        consume_kernel<<<1, 64>>>(ctx.cb, ctx.slot_ptr, ctx.out_word, generation, ctx.seed);
        double     elapsed = 0;
        const bool acked   = spin_until([&] { return acquire_load(&ctx.cb->ack) == generation; }, 2000.0, elapsed);
        if (!acked || *ctx.out_word != derived_word(generation, ctx.seed) ||
            acquire_load(&ctx.cb->exit_code) != EXIT_OK) {
            fail_case(ctx, case_id::cpu_to_gpu_publish, outcome::fail,
                      "round " + std::to_string(generation) + " of " + std::to_string(ctx.rounds) + " failed");
            return;
        }
    }
    if (!guards_intact(*ctx.slot_ptr)) {
        fail_case(ctx, case_id::cpu_to_gpu_publish, outcome::fail, "guards clobbered");
        return;
    }
    set_case(ctx, case_id::cpu_to_gpu_publish, outcome::pass,
             "hot reuse across " + std::to_string(ctx.rounds) + " generations without stale or torn reads");
}

void case_gpu_to_cpu(device_context & ctx) {
    reset_control(*ctx.cb);
    init_guards(*ctx.slot_ptr);
    produce_loop_kernel<<<1, 64>>>(ctx.cb, ctx.slot_ptr, ctx.rounds, ctx.seed);
    for (std::uint32_t generation = 1; generation <= ctx.rounds; ++generation) {
        double     elapsed  = 0;
        const bool produced = spin_until([&] { return acquire_load(&ctx.cb->ready) == generation; }, 2000.0, elapsed);
        if (!produced) {
            fail_case(ctx, case_id::gpu_to_cpu_publish, outcome::fail,
                      "generation " + std::to_string(generation) + " was not published");
            return;
        }
        for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
            if (ctx.slot_ptr->payload[i] != pattern_word(generation, i, ctx.seed)) {
                fail_case(ctx, case_id::gpu_to_cpu_publish, outcome::fail,
                          "torn or stale payload at generation " + std::to_string(generation));
                return;
            }
        }
        host_fence_release();
        release_store(&ctx.cb->ack, generation);
    }
    sync_with_deadline(0, SYNC_DEADLINE_MS, "produce loop sync");
    if (acquire_load(&ctx.cb->exit_code) != EXIT_OK || !guards_intact(*ctx.slot_ptr)) {
        fail_case(ctx, case_id::gpu_to_cpu_publish, outcome::fail, "producer exit or guards");
        return;
    }
    set_case(ctx, case_id::gpu_to_cpu_publish, outcome::pass,
             "hot reuse across " + std::to_string(ctx.rounds) + " generations without stale or torn reads");
}

void case_producer_delay(device_context & ctx) {
    reset_control(*ctx.cb);
    init_guards(*ctx.slot_ptr);
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        ctx.slot_ptr->payload[i] = 0;
    }
    stopwatch timer;
    delayed_producer_kernel<<<1, 64>>>(ctx.cb, ctx.slot_ptr, ctx.seed);
    double     unused = 0;
    const bool half   = spin_until([&] { return acquire_load(&ctx.cb->half_done) == 1; }, 5000.0, unused);
    if (!half) {
        fail_case(ctx, case_id::producer_delay_ordering, outcome::fail, "half_done never seen");
        return;
    }
    const double half_ms  = timer.milliseconds();
    const bool   full     = spin_until([&] { return acquire_load(&ctx.cb->ready) == 1; }, 5000.0, unused);
    const double ready_ms = timer.milliseconds();
    if (!full) {
        fail_case(ctx, case_id::producer_delay_ordering, outcome::fail, "ready never seen");
        return;
    }
    bool payload_ok = true;
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        payload_ok = payload_ok && ctx.slot_ptr->payload[i] == pattern_word(1, i, ctx.seed);
    }
    if (!payload_ok || !guards_intact(*ctx.slot_ptr)) {
        fail_case(ctx, case_id::producer_delay_ordering, outcome::fail, "payload incomplete when ready fired");
        return;
    }
    if (ready_ms - half_ms < 40.0) {
        fail_case(
            ctx, case_id::producer_delay_ordering, outcome::fail,
            "ready fired before the second half could be written (" + std::to_string(ready_ms - half_ms) + " ms)");
        return;
    }
    set_case(ctx, case_id::producer_delay_ordering, outcome::pass,
             "ready observed " + std::to_string(ready_ms - half_ms) + " ms after half_done; payload complete");
}

void case_ticket_semantics(device_context & ctx) {
    // (a) stale ticket: the waiter must keep waiting, then exit on budget.
    reset_control(*ctx.cb);
    release_store(&ctx.cb->ticket, TICKET_STALE);
    stopwatch timer;
    ticket_wait_kernel<<<1, 64>>>(ctx.cb, 1);
    double unused = 0;
    spin_until([&] { return acquire_load(&ctx.cb->exit_code) != 0; }, 5000.0, unused);
    sync_with_deadline(0, SYNC_DEADLINE_MS, "stale ticket sync");
    const double waited_ms = timer.milliseconds();
    if (acquire_load(&ctx.cb->exit_code) != EXIT_TIMEOUT_EXPECTED || waited_ms < 10.0) {
        set_case(ctx, case_id::ticket_semantics, outcome::fail, "stale ticket was consumed or the wait did not hold");
        return;
    }
    // (b) future ticket: an identity violation, diagnosed, never consumed.
    reset_control(*ctx.cb);
    release_store(&ctx.cb->ticket, 5);
    ticket_wait_kernel<<<1, 64>>>(ctx.cb, 1);
    spin_until([&] { return acquire_load(&ctx.cb->exit_code) != 0; }, 5000.0, unused);
    sync_with_deadline(0, SYNC_DEADLINE_MS, "future ticket sync");
    if (acquire_load(&ctx.cb->exit_code) != EXIT_MISMATCH) {
        set_case(ctx, case_id::ticket_semantics, outcome::fail,
                 "future ticket was not diagnosed as an identity violation");
        return;
    }
    set_case(ctx, case_id::ticket_semantics, outcome::pass,
             "stale ticket waited " + std::to_string(waited_ms) + " ms; future ticket reported mismatch");
}

void case_multi_group(device_context & ctx) {
    for (const std::uint32_t live : { 8u, 2u }) {
        reset_control(*ctx.cb);
        init_guards(*ctx.slot_ptr);
        for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
            ctx.slot_ptr->payload[i] = 0;
        }
        check_hip(hipMemset(ctx.counter, 0, sizeof(std::uint32_t)), "counter memset");
        group_writer_kernel<<<8, 64>>>(ctx.slot_ptr->payload, ctx.counter, ctx.cb, live, ctx.seed);
        double     unused = 0;
        const bool done   = spin_until([&] { return acquire_load(&ctx.cb->ready) == live; }, 5000.0, unused);
        sync_with_deadline(0, SYNC_DEADLINE_MS, "group writer sync");
        if (!done) {
            set_case(ctx, case_id::multi_group_completion, outcome::fail,
                     "ready did not fire for " + std::to_string(live) + " live groups");
            return;
        }
        bool ok = true;
        for (std::uint32_t i = 0; i < live; ++i) {
            ok = ok && ctx.slot_ptr->payload[i] == pattern_word(live, i, ctx.seed);
        }
        for (std::uint32_t i = live; i < PAYLOAD_WORDS; ++i) {
            ok = ok && ctx.slot_ptr->payload[i] == 0;
        }
        if (!ok || !guards_intact(*ctx.slot_ptr)) {
            set_case(ctx, case_id::multi_group_completion, outcome::fail,
                     "incomplete or overrun payload for " + std::to_string(live) + " live groups");
            return;
        }
    }
    set_case(ctx, case_id::multi_group_completion, outcome::pass,
             "last-arrival publishing works with 8 and with 2 live groups of an 8-group grid");
}

// Guide 39.6 P01/P02: the maximum-index workgroup provably finishes first
// and the minimum-index one last. The trap observation requires the ready
// word to stay unpublished while the max-index group is done; a publisher
// keyed on the largest block index would fire early and is caught every
// round instead of depending on incidental scheduling order.
void case_multi_group_ordering(device_context & ctx) {
    constexpr std::uint32_t ORDER_ROUNDS  = 8;
    // ~22 ms per step at the measured 2.3 GHz clock64 (tq-clock-rate): block 0 spins about
    // 154 ms, leaving a large deterministic trap window for the host.
    constexpr std::uint64_t DELAY_STEP    = 50000000ULL;
    bool                    trap_observed = false;
    for (std::uint32_t round = 0; round < ORDER_ROUNDS; ++round) {
        reset_control(*ctx.cb);
        init_guards(*ctx.slot_ptr);
        std::memset(ctx.slot_ptr->payload, 0, sizeof(ctx.slot_ptr->payload));
        check_hip(hipMemset(ctx.counter, 0, sizeof(std::uint32_t)), "ordering counter memset");
        ordered_group_writer_kernel<<<8, 64>>>(ctx.slot_ptr->payload, ctx.counter, ctx.cb, ctx.seed, DELAY_STEP);
        double     unused     = 0;
        const bool first_done = spin_until([&] { return acquire_load(&ctx.cb->first_done) == 8; }, 5000.0, unused);
        if (!first_done) {
            fail_case(ctx, case_id::multi_group_ordering, outcome::fail, "max-index group never finished");
            return;
        }
        if (acquire_load(&ctx.cb->ready) == 0) {
            trap_observed = true;
        }
        const bool all_done = spin_until([&] { return acquire_load(&ctx.cb->ready) == 8; }, 5000.0, unused);
        sync_with_deadline(0, SYNC_DEADLINE_MS, "ordering sync");
        if (!all_done) {
            fail_case(ctx, case_id::multi_group_ordering, outcome::fail,
                      "ready did not fire after the delayed last arrival");
            return;
        }
        bool ok = true;
        for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
            ok = ok && ctx.slot_ptr->payload[i] == pattern_word(8, i, ctx.seed);
        }
        if (!ok || !guards_intact(*ctx.slot_ptr)) {
            fail_case(ctx, case_id::multi_group_ordering, outcome::fail,
                      "payload incomplete when ready fired (premature publish)");
            return;
        }
    }
    if (!trap_observed) {
        fail_case(ctx, case_id::multi_group_ordering, outcome::fail,
                  "the max-index-done-before-ready window was never observed");
        return;
    }
    set_case(ctx, case_id::multi_group_ordering, outcome::pass,
             "max-index group finished first while ready stayed unpublished in " + std::to_string(ORDER_ROUNDS) +
                 " rounds; payload complete when ready fired");
}

void case_abort(device_context & ctx) {
    reset_control(*ctx.cb);
    ticket_wait_kernel<<<1, 64>>>(ctx.cb, 1);
    usleep(5000);
    release_store(&ctx.cb->abort, 1);
    stopwatch  timer;
    double     unused = 0;
    const bool exited = spin_until([&] { return acquire_load(&ctx.cb->exit_code) != 0; }, 5000.0, unused);
    sync_with_deadline(0, SYNC_DEADLINE_MS, "abort sync");
    if (!exited || acquire_load(&ctx.cb->exit_code) != EXIT_ABORT_OBSERVED) {
        set_case(ctx, case_id::abort_stop, outcome::fail, "waiter did not honor the stop word");
        return;
    }
    set_case(ctx, case_id::abort_stop, outcome::pass,
             "waiter exited on the stop word in " + std::to_string(timer.milliseconds()) + " ms");
}

// The operator's worst outcome is a run that never returns, so the bounded
// sync itself must be proven on the live device: a stalled queue raises the
// deadline error while the waiter is still spinning, and the stop word then
// recovers the waiter so the drain completes (guide 35.8; SPEC 7).
void case_native_sync_deadline(device_context & ctx) {
    reset_control(*ctx.cb);
    stall_kernel<<<1, 64>>>(ctx.cb);
    bool timed_out = false;
    try {
        sync_with_deadline(0, STALL_PROBE_DEADLINE_MS, "stalled-queue observation");
    } catch (const hip_error & error) {
        timed_out = std::string(error.what()).find("deadline exceeded") != std::string::npos;
    }
    if (!timed_out) {
        // The queue completed or failed inside the window, so the timeout
        // branch cannot be observed on this run; fail_case still aborts and
        // drains, so the case itself stays bounded.
        fail_case(ctx, case_id::native_sync_deadline, outcome::fail,
                  "stalled queue finished before the " + std::to_string(STALL_PROBE_DEADLINE_MS) +
                      " ms deadline, so the bounded-sync timeout path was not observed");
        return;
    }
    release_store(&ctx.cb->abort, 1);
    sync_with_deadline(0, SYNC_DEADLINE_MS, "stalled-queue drain");
    if (acquire_load(&ctx.cb->exit_code) != EXIT_ABORT_OBSERVED) {
        set_case(ctx, case_id::native_sync_deadline, outcome::fail,
                 "waiter did not honor the stop word after the sync deadline");
        return;
    }
    set_case(ctx, case_id::native_sync_deadline, outcome::pass,
             "stalled queue raised the bounded-sync deadline at " + std::to_string(STALL_PROBE_DEADLINE_MS) +
                 " ms; the stop word recovered the waiter and the drain completed");
}

void case_host_rmw(device_context & ctx) {
    release_store(ctx.rmw_add, 0);
    release_store(ctx.rmw_cas, 0);
    rmw_probe_kernel<<<1, 64>>>(ctx.rmw_add, ctx.rmw_cas);
    try {
        sync_with_deadline(0, SYNC_DEADLINE_MS, "rmw sync");
    } catch (const std::exception & error) {
        // One launch feeds both capability bits: a HIP failure fails both
        // of them, so neither record silently stays not_run.
        const std::string detail = error.what();
        set_case(ctx, case_id::host_system_rmw_u64_add, outcome::fail, detail);
        set_case(ctx, case_id::host_system_rmw_u64_cas, outcome::fail, detail);
        return;
    }
    const std::uint64_t add_value = acquire_load(ctx.rmw_add);
    const std::uint64_t cas_value = acquire_load(ctx.rmw_cas);
    // Per-operation capability bits (guide 31.3): a wrong value is worse
    // than a no-op, so garbage results are failures, not unsupported.
    if (add_value == 7) {
        set_case(ctx, case_id::host_system_rmw_u64_add, outcome::pass,
                 "system-scope fetch_add on host memory executed");
    } else if (add_value == 0) {
        set_case(ctx, case_id::host_system_rmw_u64_add, outcome::unsupported,
                 "host-memory fetch_add did not execute; the relay design must not require "
                 "it (guide 31.3)");
    } else {
        set_case(ctx, case_id::host_system_rmw_u64_add, outcome::fail,
                 "host-memory fetch_add produced a wrong value (" + std::to_string(add_value) + " instead of 7)");
    }
    if (cas_value == 9) {
        set_case(ctx, case_id::host_system_rmw_u64_cas, outcome::pass, "system-scope CAS on host memory executed");
    } else if (cas_value == 0) {
        set_case(ctx, case_id::host_system_rmw_u64_cas, outcome::unsupported,
                 "host-memory CAS did not execute; the relay design must not require it "
                 "(guide 31.3)");
    } else {
        set_case(ctx, case_id::host_system_rmw_u64_cas, outcome::fail,
                 "host-memory CAS produced a wrong value (" + std::to_string(cas_value) + " instead of 9)");
    }
}

// -------- multi-stream cases (single device, five concurrent streams) ------

class stream_set {
  public:
    explicit stream_set(std::uint32_t count) {
        streams.resize(count);
        for (auto & stream : streams) {
            check_hip(hipStreamCreate(&stream), "stream create");
        }
    }

    ~stream_set() {
        for (auto & stream : streams) {
            if (stream != nullptr) {
                (void) hipStreamDestroy(stream);
            }
        }
    }

    stream_set(const stream_set &)             = delete;
    stream_set & operator=(const stream_set &) = delete;

    std::vector<hipStream_t> streams;
};

// One emulated relay round: five producer streams write disjoint partials,
// the host folds them into the output slot, publishes the ticket, and five
// consumer streams read the result back and acknowledge. Returns nullopt on
// success and a diagnostic otherwise. No host-memory RMW anywhere.
std::optional<std::string> run_relay_round(device_context &                 ctx,
                                           const std::vector<hipStream_t> & streams,
                                           std::uint64_t                    generation) {
    for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
        release_store(&ctx.rank_ack[rank], 0);
        partial_producer_kernel<<<1, 64, 0, streams[rank]>>>(ctx.rank_partials, ctx.rank_ready, rank, generation,
                                                             ctx.seed);
    }
    double unused = 0;
    for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
        const bool produced =
            spin_until([&] { return acquire_load(&ctx.rank_ready[rank]) == generation; }, 5000.0, unused);
        if (!produced) {
            return "producer rank " + std::to_string(rank) + " did not publish generation " +
                   std::to_string(generation);
        }
    }
    std::uint64_t fold[PAYLOAD_WORDS] = {};
    const auto    fold_problem        = fold_partials(ctx.rank_partials, generation, ctx.seed, fold);
    if (fold_problem.has_value()) {
        return fold_problem;
    }
    std::uint64_t expected = 0;
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        ctx.slot_ptr->payload[i] = fold[i];
        expected ^= fold[i];
    }
    host_fence_release();
    release_store(&ctx.cb->ticket, generation);
    for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
        consume_read_kernel<<<1, 64, 0, streams[rank]>>>(ctx.cb, ctx.slot_ptr, ctx.rank_ack, ctx.rank_result, rank,
                                                         generation, ctx.seed);
    }
    for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
        const bool consumed =
            spin_until([&] { return acquire_load(&ctx.rank_ack[rank]) == generation; }, 5000.0, unused);
        if (!consumed) {
            return "consumer rank " + std::to_string(rank) + " did not acknowledge generation " +
                   std::to_string(generation);
        }
        if (ctx.rank_result[rank] != expected) {
            return "consumer rank " + std::to_string(rank) + " derived the wrong result word";
        }
    }
    return std::nullopt;
}

void case_multi_consumer(device_context & ctx) {
    reset_control(*ctx.cb);
    init_guards(*ctx.slot_ptr);
    stream_set streams(RANKS);
    for (std::uint64_t generation = 1; generation <= ctx.rounds; ++generation) {
        for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
            release_store(&ctx.rank_ack[rank], 0);
        }
        for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
            ctx.slot_ptr->payload[i] = pattern_word(generation, i, ctx.seed);
        }
        host_fence_release();
        release_store(&ctx.cb->ticket, generation);
        for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
            consume_read_kernel<<<1, 64, 0, streams.streams[rank]>>>(ctx.cb, ctx.slot_ptr, ctx.rank_ack,
                                                                     ctx.rank_result, rank, generation, ctx.seed);
        }
        double unused = 0;
        for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
            const bool acked =
                spin_until([&] { return acquire_load(&ctx.rank_ack[rank]) == generation; }, 5000.0, unused);
            if (!acked || ctx.rank_result[rank] != derived_word(generation, ctx.seed)) {
                fail_case(ctx, case_id::multi_consumer_output_credit, outcome::fail,
                          "generation " + std::to_string(generation) + ": consumer rank " + std::to_string(rank) +
                              " did not complete its read");
                return;
            }
        }
    }
    if (!guards_intact(*ctx.slot_ptr)) {
        fail_case(ctx, case_id::multi_consumer_output_credit, outcome::fail, "guards clobbered");
        return;
    }
    set_case(ctx, case_id::multi_consumer_output_credit, outcome::pass,
             "five concurrent consumers acknowledged " + std::to_string(ctx.rounds) +
                 " generations, each only after its complete verified read");
}

void case_delayed_consumer(device_context & ctx) {
    reset_control(*ctx.cb);
    init_guards(*ctx.slot_ptr);
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        ctx.slot_ptr->payload[i] = pattern_word(1, i, ctx.seed);
    }
    release_store(&ctx.rank_ack[0], 0);
    release_store(&ctx.rank_ack[1], 0);
    release_store(&ctx.cb->exit_code, EXIT_OK);
    host_fence_release();
    release_store(&ctx.cb->ticket, 1);
    stream_set streams(2);
    consume_read_kernel<<<1, 64, 0, streams.streams[0]>>>(ctx.cb, ctx.slot_ptr, ctx.rank_ack, ctx.rank_result, 0, 1,
                                                          ctx.seed);
    delayed_reader_kernel<<<1, 64, 0, streams.streams[1]>>>(ctx.cb, ctx.slot_ptr, ctx.rank_ack, ctx.rank_result, 1, 1,
                                                            ctx.seed);
    double     unused = 0;
    stopwatch  timer;
    const bool fast = spin_until([&] { return acquire_load(&ctx.rank_ack[0]) == 1; }, 5000.0, unused);
    if (!fast) {
        fail_case(ctx, case_id::delayed_consumer_ordering, outcome::fail, "fast consumer never acknowledged");
        return;
    }
    const bool progress = spin_until([&] { return acquire_load(&ctx.cb->half_done) == 1; }, 5000.0, unused);
    if (!progress) {
        fail_case(ctx, case_id::delayed_consumer_ordering, outcome::fail, "delayed reader never reported progress");
        return;
    }
    const double half_ms = timer.milliseconds();
    if (acquire_load(&ctx.rank_ack[1]) != 0) {
        fail_case(ctx, case_id::delayed_consumer_ordering, outcome::fail,
                  "delayed reader acknowledged before its read completed");
        return;
    }
    const bool   slow   = spin_until([&] { return acquire_load(&ctx.rank_ack[1]) == 1; }, 5000.0, unused);
    const double ack_ms = timer.milliseconds();
    if (!slow) {
        fail_case(ctx, case_id::delayed_consumer_ordering, outcome::fail, "delayed reader never acknowledged");
        return;
    }
    if (ack_ms - half_ms < 40.0) {
        fail_case(ctx, case_id::delayed_consumer_ordering, outcome::fail,
                  "ack fired too soon after progress (" + std::to_string(ack_ms - half_ms) + " ms)");
        return;
    }
    const std::uint64_t expect = derived_word(1, ctx.seed);
    if (ctx.rank_result[0] != expect || ctx.rank_result[1] != expect || !guards_intact(*ctx.slot_ptr)) {
        fail_case(ctx, case_id::delayed_consumer_ordering, outcome::fail, "consumer verification failed");
        return;
    }
    set_case(ctx, case_id::delayed_consumer_ordering, outcome::pass,
             "delayed reader's acknowledgment was withheld " + std::to_string(ack_ms - half_ms) +
                 " ms after progress; both consumers verified the payload");
}

void case_two_bank(device_context & ctx) {
    reset_control(*ctx.cb);
    init_guards(*ctx.slot_ptr);
    init_guards(*ctx.slot2_ptr);
    std::memset(ctx.slot_ptr->payload, 0, sizeof(ctx.slot_ptr->payload));
    std::memset(ctx.slot2_ptr->payload, 0, sizeof(ctx.slot2_ptr->payload));
    std::uint64_t last0[PAYLOAD_WORDS] = {};
    std::uint64_t last1[PAYLOAD_WORDS] = {};
    produce_bank_kernel<<<1, 64>>>(ctx.cb, ctx.slot_ptr, ctx.slot2_ptr, ctx.rounds, ctx.seed);
    for (std::uint64_t generation = 1; generation <= ctx.rounds; ++generation) {
        double     unused = 0;
        const bool ready  = spin_until([&] { return acquire_load(&ctx.cb->ready) == generation; }, 5000.0, unused);
        if (!ready) {
            fail_case(ctx, case_id::two_bank_rotation, outcome::fail,
                      "generation " + std::to_string(generation) + " was not published");
            return;
        }
        slot * const          written    = ((generation - 1) & 1) == 0 ? ctx.slot_ptr : ctx.slot2_ptr;
        slot * const          other      = ((generation - 1) & 1) == 0 ? ctx.slot2_ptr : ctx.slot_ptr;
        std::uint64_t * const last_other = ((generation - 1) & 1) == 0 ? last1 : last0;
        for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
            if (written->payload[i] != pattern_word(generation, i, ctx.seed)) {
                fail_case(ctx, case_id::two_bank_rotation, outcome::fail,
                          "generation " + std::to_string(generation) + ": torn or stale payload in the written bank");
                return;
            }
            if (other->payload[i] != last_other[i]) {
                fail_case(ctx, case_id::two_bank_rotation, outcome::fail,
                          "generation " + std::to_string(generation) + ": the untouched bank was clobbered");
                return;
            }
        }
        std::memcpy(last0, ctx.slot_ptr->payload, sizeof(last0));
        std::memcpy(last1, ctx.slot2_ptr->payload, sizeof(last1));
        host_fence_release();
        release_store(&ctx.cb->ack, generation);
    }
    sync_with_deadline(0, SYNC_DEADLINE_MS, "bank rotation sync");
    if (acquire_load(&ctx.cb->exit_code) != EXIT_OK || !guards_intact(*ctx.slot_ptr) ||
        !guards_intact(*ctx.slot2_ptr)) {
        fail_case(ctx, case_id::two_bank_rotation, outcome::fail, "producer exit or guards");
        return;
    }
    set_case(
        ctx, case_id::two_bank_rotation, outcome::pass,
        "two banks alternated across " + std::to_string(ctx.rounds) + " generations; the idle bank was never written");
}

void case_varying_live(device_context & ctx) {
    reset_control(*ctx.cb);
    init_guards(*ctx.slot_ptr);
    std::memset(ctx.slot_ptr->payload, 0, sizeof(ctx.slot_ptr->payload));
    std::uint64_t last[PAYLOAD_WORDS] = {};
    produce_variable_kernel<<<1, 64>>>(ctx.cb, ctx.slot_ptr, ctx.rounds, ctx.seed);
    for (std::uint64_t generation = 1; generation <= ctx.rounds; ++generation) {
        const std::uint32_t live   = 2 + static_cast<std::uint32_t>((generation - 1) % 6);
        double              unused = 0;
        const bool ready = spin_until([&] { return acquire_load(&ctx.cb->ready) == generation; }, 5000.0, unused);
        if (!ready) {
            fail_case(ctx, case_id::varying_live_bytes, outcome::fail,
                      "generation " + std::to_string(generation) + " was not published");
            return;
        }
        for (std::uint32_t i = 0; i < live; ++i) {
            if (ctx.slot_ptr->payload[i] != pattern_word(generation, i, ctx.seed)) {
                fail_case(ctx, case_id::varying_live_bytes, outcome::fail,
                          "generation " + std::to_string(generation) + ": live word " + std::to_string(i) +
                              " is torn or stale");
                return;
            }
        }
        for (std::uint32_t i = live; i < PAYLOAD_WORDS; ++i) {
            if (ctx.slot_ptr->payload[i] != last[i]) {
                fail_case(ctx, case_id::varying_live_bytes, outcome::fail,
                          "generation " + std::to_string(generation) + ": word " + std::to_string(i) +
                              " outside the live extent was overwritten");
                return;
            }
        }
        std::memcpy(last, ctx.slot_ptr->payload, sizeof(last));
        host_fence_release();
        release_store(&ctx.cb->ack, generation);
    }
    sync_with_deadline(0, SYNC_DEADLINE_MS, "varying live sync");
    if (acquire_load(&ctx.cb->exit_code) != EXIT_OK || !guards_intact(*ctx.slot_ptr)) {
        fail_case(ctx, case_id::varying_live_bytes, outcome::fail, "producer exit or guards");
        return;
    }
    set_case(ctx, case_id::varying_live_bytes, outcome::pass,
             "live extents from 2 to 7 words across " + std::to_string(ctx.rounds) +
                 " generations; the tail was never written");
}

// Guide 39.5 H06: the relay design places payloads at non-zero offsets
// inside shared arenas (guide 51.7), so address arithmetic is a first-class
// failure mode. The arena below places a full slot at byte 72: 8-byte
// aligned but not 64-byte line aligned, straddling two cache lines. Guard
// words around the arena catch a write aimed at the base or past the end;
// the slot's own guards catch payload overruns.
constexpr std::uint32_t OFFSET_PREFIX_WORDS = 9;  // 72 bytes
constexpr std::uint32_t OFFSET_SUFFIX_WORDS = 9;

struct offset_arena {
    std::uint64_t prefix[OFFSET_PREFIX_WORDS];
    slot          interior;
    std::uint64_t suffix[OFFSET_SUFFIX_WORDS];
};

static_assert(offsetof(offset_arena, interior) % sizeof(std::uint64_t) == 0,
              "the interior slot must stay 8-byte aligned");
static_assert(offsetof(offset_arena, interior) % 64 != 0, "the interior slot must straddle cache lines");

void case_offset_addressing(device_context & ctx) {
    constexpr std::uint32_t OFFSET_ROUNDS = 8;
    check_hip(hipSetDevice(ctx.record.ordinal), "offset set device");
    offset_arena * arena = nullptr;
    check_hip(hipHostMalloc(&arena, sizeof(offset_arena), hipHostMallocMapped | hipHostMallocPortable),
              "offset arena alloc");
    // Free the arena on every exit path, including thrown HIP errors.
    const auto cleanup = [&] {
        (void) hipSetDevice(ctx.record.ordinal);
        (void) hipHostFree(arena);
    };

    std::memset(arena, 0, sizeof(offset_arena));
    for (auto & word : arena->prefix) {
        word = GUARD_VALUE;
    }
    for (auto & word : arena->suffix) {
        word = GUARD_VALUE;
    }
    init_guards(arena->interior);
    const auto guards_ok = [&] {
        bool ok = guards_intact(arena->interior);
        for (const std::uint64_t word : arena->prefix) {
            ok = ok && word == GUARD_VALUE;
        }
        for (const std::uint64_t word : arena->suffix) {
            ok = ok && word == GUARD_VALUE;
        }
        return ok;
    };

    // Phase A: the host publishes at the interior offset and the device
    // consumes through the same interior address.
    reset_control(*ctx.cb);
    for (std::uint64_t generation = 1; generation <= OFFSET_ROUNDS; ++generation) {
        for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
            arena->interior.payload[i] = pattern_word(generation, i, ctx.seed);
        }
        release_store(&ctx.cb->exit_code, EXIT_OK);
        host_fence_release();
        release_store(&ctx.cb->ticket, generation);
        consume_kernel<<<1, 64>>>(ctx.cb, &arena->interior, ctx.out_word, generation, ctx.seed);
        double     unused = 0;
        const bool acked  = spin_until([&] { return acquire_load(&ctx.cb->ack) == generation; }, 2000.0, unused);
        if (!acked || *ctx.out_word != derived_word(generation, ctx.seed) ||
            acquire_load(&ctx.cb->exit_code) != EXIT_OK || !guards_ok()) {
            cleanup();
            fail_case(ctx, case_id::offset_addressing, outcome::fail,
                      "interior consume round " + std::to_string(generation) + " failed or guards damaged");
            return;
        }
    }

    // Phase B: the device produces at the interior offset and the host reads
    // it back, one generation at a time.
    reset_control(*ctx.cb);
    produce_loop_kernel<<<1, 64>>>(ctx.cb, &arena->interior, OFFSET_ROUNDS, ctx.seed);
    for (std::uint64_t generation = 1; generation <= OFFSET_ROUNDS; ++generation) {
        double     unused   = 0;
        const bool produced = spin_until([&] { return acquire_load(&ctx.cb->ready) == generation; }, 2000.0, unused);
        if (!produced) {
            cleanup();
            fail_case(ctx, case_id::offset_addressing, outcome::fail,
                      "interior produce generation " + std::to_string(generation) + " was not published");
            return;
        }
        for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
            if (arena->interior.payload[i] != pattern_word(generation, i, ctx.seed)) {
                cleanup();
                fail_case(ctx, case_id::offset_addressing, outcome::fail,
                          "interior produce generation " + std::to_string(generation) + " is torn or stale");
                return;
            }
        }
        if (!guards_ok()) {
            cleanup();
            fail_case(ctx, case_id::offset_addressing, outcome::fail,
                      "arena guards damaged at generation " + std::to_string(generation));
            return;
        }
        host_fence_release();
        release_store(&ctx.cb->ack, generation);
    }
    std::string sync_failure;
    try {
        sync_with_deadline(0, SYNC_DEADLINE_MS, "offset sync");
    } catch (const std::exception & error) {
        sync_failure = error.what();
    }
    cleanup();
    if (!sync_failure.empty()) {
        set_case(ctx, case_id::offset_addressing, outcome::fail, std::move(sync_failure));
        return;
    }
    if (acquire_load(&ctx.cb->exit_code) != EXIT_OK) {
        fail_case(ctx, case_id::offset_addressing, outcome::fail, "interior producer exit");
        return;
    }
    set_case(ctx, case_id::offset_addressing, outcome::pass,
             "interior slot at byte " + std::to_string(offsetof(offset_arena, interior)) +
                 " (8-byte aligned, cache-line straddling) round-tripped in both directions; arena guards intact");
}

void case_five_stream(device_context & ctx) {
    reset_control(*ctx.cb);
    init_guards(*ctx.slot_ptr);
    std::memset(ctx.slot_ptr->payload, 0, sizeof(ctx.slot_ptr->payload));
    std::memset(ctx.rank_ready, 0, RANKS * sizeof(std::uint64_t));
    stream_set streams(RANKS);
    for (std::uint64_t generation = 1; generation <= ctx.rounds; ++generation) {
        const auto problem = run_relay_round(ctx, streams.streams, generation);
        if (problem.has_value()) {
            fail_case(ctx, case_id::five_stream_relay_cycle, outcome::fail, *problem);
            return;
        }
    }
    if (!guards_intact(*ctx.slot_ptr)) {
        fail_case(ctx, case_id::five_stream_relay_cycle, outcome::fail, "guards clobbered");
        return;
    }
    set_case(ctx, case_id::five_stream_relay_cycle, outcome::pass,
             std::to_string(ctx.rounds) +
                 " rounds of five-producer reduce, publish and five-consumer read on one device "
                 "(stream emulation, not five-rank qualification)");
}

// Guide 39.6 P04: a consumer queued ahead of its payload producer on the
// same stream reads the pre-write payload. The inverse ordering is the
// device-side hazard class of guide 33.7 for this architecture (the ticket
// comes from the host, so the producer that can be ordered behind a waiter
// is the payload writer, not the ticket publisher). The consumer's
// acquire-then-verify read must catch the premature read deterministically
// instead of silently consuming garbage, and the run must stay bounded. A
// platform that reordered the two kernels (or a consumer that consumed the
// stale payload without noticing) would leave the detection path
// unobservable, and the case fails exactly like the sync-deadline case
// fails when its branch cannot be exercised.
void case_same_stream_inverse(device_context & ctx) {
    if (derived_word(1, ctx.seed) == 0xDEADBEEFDEADBEEFULL) {
        fail_case(ctx, case_id::same_stream_inverse_dependency, outcome::fail,
                  "the payload pattern collides with the verification marker for this seed");
        return;
    }
    reset_control(*ctx.cb);
    init_guards(*ctx.slot_ptr);
    std::memset(ctx.slot_ptr->payload, 0, sizeof(ctx.slot_ptr->payload));
    stream_set streams(1);
    // Consumer first, producer second, on the same stream: stream order
    // guarantees the consumer executes before the payload write exists.
    consume_kernel<<<1, 64, 0, streams.streams[0]>>>(ctx.cb, ctx.slot_ptr, ctx.out_word, 1, ctx.seed);
    produce_once_kernel<<<1, 64, 0, streams.streams[0]>>>(ctx.cb, ctx.slot_ptr, 1, ctx.seed);
    host_fence_release();
    release_store(&ctx.cb->ticket, 1);
    double     unused = 0;
    const bool acked  = spin_until([&] { return acquire_load(&ctx.cb->ack) == 1; }, 5000.0, unused);
    if (!acked) {
        fail_case(ctx, case_id::same_stream_inverse_dependency, outcome::fail,
                  "the consumer never acknowledged the prematurely published ticket");
        return;
    }
    sync_with_deadline(streams.streams[0], SYNC_DEADLINE_MS, "inverse dependency sync");
    if (*ctx.out_word != 0xDEADBEEFDEADBEEFULL) {
        fail_case(ctx, case_id::same_stream_inverse_dependency, outcome::fail,
                  "the consumer read a payload whose producer is queued behind it and the verification did not "
                  "catch the premature read");
        return;
    }
    if (acquire_load(&ctx.cb->ready) != 1 || !guards_intact(*ctx.slot_ptr)) {
        fail_case(ctx, case_id::same_stream_inverse_dependency, outcome::fail,
                  "the producer queued behind the consumer did not complete correctly");
        return;
    }
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        if (ctx.slot_ptr->payload[i] != pattern_word(1, i, ctx.seed)) {
            fail_case(ctx, case_id::same_stream_inverse_dependency, outcome::fail,
                      "the producer's eventual write is torn or wrong");
            return;
        }
    }
    set_case(ctx, case_id::same_stream_inverse_dependency, outcome::pass,
             "a consumer queued ahead of its payload producer read the pre-write payload and the verification "
             "caught it deterministically; the producer's later write was exact, so the corruption was ordering "
             "alone and the run stayed bounded");
}

// Guide 31.1 (two-window device timelines) and 39.6 P05/P06/P08 at the
// single-device software layer (probe LEMMA 4): two independent channels on
// one device, each with its own control block, payload slot and seed.
// Window B runs a host-driven consume loop the whole time. Window A is
// stalled three ways: a withheld ticket (missing producer), a delayed last
// reader, and 64 resident waiter workgroups that hold occupancy while B
// keeps advancing. The case proves B advances independently, A's slot is
// never touched by B's traffic, A completes exactly once released, and A's
// channel is reusable after the stall. Cross-device qualification still
// belongs to the five-rank case.
void case_dual_window(device_context & ctx) {
    constexpr std::uint32_t B_FIRST_ROUNDS  = 256;  // while A waits on a withheld ticket
    constexpr std::uint32_t B_SECOND_ROUNDS = 256;  // while A's last reader is mid-read
    constexpr std::uint32_t B_FINAL_ROUNDS  = 16;   // after A reused its channel
    constexpr std::uint32_t WAITER_BLOCKS   = 64;
    const std::uint32_t     seed_a          = ctx.seed;
    const std::uint32_t     seed_b          = ctx.seed ^ 0x55AAu;

    reset_control(*ctx.cb);
    reset_control(*ctx.cb2);
    init_guards(*ctx.slot_ptr);
    init_guards(*ctx.slot_b);
    std::memset(ctx.slot_ptr->payload, 0, sizeof(ctx.slot_ptr->payload));
    std::memset(ctx.slot_b->payload, 0, sizeof(ctx.slot_b->payload));
    std::memset(ctx.rank_ack, 0, RANKS * sizeof(std::uint64_t));

    stream_set        streams(2);
    const hipStream_t stream_a     = streams.streams[0];
    const hipStream_t stream_b     = streams.streams[1];
    std::uint64_t     b_generation = 0;

    // Window B: host-driven consume loop with an optional per-generation
    // cross-channel canary. Returns a diagnostic or nullopt.
    const auto run_b_gens = [&](std::uint32_t count, const char * phase,
                                const std::function<std::optional<std::string>(std::uint64_t)> & per_gen) {
        for (std::uint32_t step = 0; step < count; ++step) {
            ++b_generation;
            for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
                ctx.slot_b->payload[i] = pattern_word(b_generation, i, seed_b);
            }
            host_fence_release();
            release_store(&ctx.cb2->ticket, b_generation);
            consume_read_kernel<<<1, 64, 0, stream_b>>>(ctx.cb2, ctx.slot_b, ctx.rank_ack, ctx.rank_result, 1,
                                                        b_generation, seed_b);
            double     unused = 0;
            const bool acked =
                spin_until([&] { return acquire_load(&ctx.rank_ack[1]) == b_generation; }, 5000.0, unused);
            if (!acked) {
                return std::optional<std::string>(std::string(phase) + ": window B generation " +
                                                  std::to_string(b_generation) +
                                                  " did not complete while window A was stalled");
            }
            if (ctx.rank_result[1] != derived_word(b_generation, seed_b)) {
                return std::optional<std::string>(std::string(phase) + ": window B generation " +
                                                  std::to_string(b_generation) + " derived the wrong result");
            }
            if (per_gen) {
                const auto problem = per_gen(b_generation);
                if (problem.has_value()) {
                    return problem;
                }
            }
        }
        return std::optional<std::string>{};
    };
    const auto fail_dual = [&](const std::string & detail) {
        fail_case(ctx, case_id::dual_window_independent_progress, outcome::fail, detail);
    };

    // Phase 1: A's ticket is withheld (missing producer); 64 waiter
    // workgroups plus A's consumer sit resident on stream A while B runs a
    // full batch. Afterwards A must still be untouched.
    bare_waiter_kernel<<<WAITER_BLOCKS, 64, 0, stream_a>>>(ctx.cb, 1);
    consume_read_kernel<<<1, 64, 0, stream_a>>>(ctx.cb, ctx.slot_ptr, ctx.rank_ack, ctx.rank_result, 0, 1, seed_a);
    {
        const auto problem = run_b_gens(B_FIRST_ROUNDS, "phase 1", {});
        if (problem.has_value()) {
            fail_dual(*problem);
            return;
        }
    }
    {
        bool stalled = acquire_load(&ctx.cb->ticket) == 0 && acquire_load(&ctx.rank_ack[0]) == 0 &&
                       acquire_load(&ctx.cb->exit_code) == 0;
        for (std::uint32_t i = 0; stalled && i < PAYLOAD_WORDS; ++i) {
            stalled = ctx.slot_ptr->payload[i] == 0;
        }
        if (!stalled) {
            fail_dual("window A advanced or was written while its producer was missing");
            return;
        }
    }

    // Phase 2: release window A; its waiters and consumer must complete
    // with exact values, proving B's traffic never aliased A's channel.
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        ctx.slot_ptr->payload[i] = pattern_word(1, i, seed_a);
    }
    host_fence_release();
    release_store(&ctx.cb->ticket, 1);
    double unused = 0;
    if (!spin_until([&] { return acquire_load(&ctx.rank_ack[0]) == 1; }, 5000.0, unused)) {
        fail_dual("window A never acknowledged after its producer was released");
        return;
    }
    sync_with_deadline(stream_a, SYNC_DEADLINE_MS, "dual window A release sync");
    if (ctx.rank_result[0] != derived_word(1, seed_a) || acquire_load(&ctx.cb->exit_code) != EXIT_OK) {
        fail_dual("window A read a corrupted payload or a resident waiter exited on the wrong path");
        return;
    }

    // Phase 3: window A's last reader for generation 2 is delayed mid-read
    // while window B runs another full batch; the acknowledgment must be
    // withheld and A's slot must stay byte-identical throughout.
    reset_control(*ctx.cb);
    release_store(&ctx.rank_ack[0], 0);  // phase 2's acknowledgment must not read as a mid-read finish
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        ctx.slot_ptr->payload[i] = pattern_word(2, i, seed_a);
    }
    host_fence_release();
    release_store(&ctx.cb->ticket, 2);
    delayed_reader_kernel<<<1, 64, 0, stream_a>>>(ctx.cb, ctx.slot_ptr, ctx.rank_ack, ctx.rank_result, 0, 2, seed_a);
    stopwatch timer;
    if (!spin_until([&] { return acquire_load(&ctx.cb->half_done) == 1; }, 5000.0, unused)) {
        fail_dual("window A's delayed reader never reported progress");
        return;
    }
    const double half_ms = timer.milliseconds();
    if (acquire_load(&ctx.rank_ack[0]) != 0) {
        fail_dual("window A's delayed reader acknowledged before its read completed");
        return;
    }
    {
        const auto problem =
            run_b_gens(B_SECOND_ROUNDS, "phase 3", [&](std::uint64_t generation) -> std::optional<std::string> {
                for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
                    if (ctx.slot_ptr->payload[i] != pattern_word(2, i, seed_a)) {
                        return std::optional<std::string>("phase 3: window B generation " + std::to_string(generation) +
                                                          " wrote window A's slot");
                    }
                }
                if (acquire_load(&ctx.rank_ack[0]) != 0) {
                    return std::optional<std::string>(
                        "phase 3: window A's reader acknowledged mid-read at B "
                        "generation " +
                        std::to_string(generation));
                }
                return std::optional<std::string>{};
            });
        if (problem.has_value()) {
            fail_dual(*problem);
            return;
        }
    }
    if (!spin_until([&] { return acquire_load(&ctx.rank_ack[0]) == 2; }, 5000.0, unused)) {
        fail_dual("window A's delayed reader never finished");
        return;
    }
    const double ack_ms = timer.milliseconds();
    if (ack_ms - half_ms < 40.0) {
        fail_dual("window A's delayed reader finished too soon after progress (" + std::to_string(ack_ms - half_ms) +
                  " ms)");
        return;
    }
    if (ctx.rank_result[0] != derived_word(2, seed_a) || !guards_intact(*ctx.slot_ptr)) {
        fail_dual("window A's delayed reader verification failed");
        return;
    }

    // Phase 4: window A's channel is reusable after the stall; window B
    // keeps advancing, so both windows interleave through the whole case.
    reset_control(*ctx.cb);
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        ctx.slot_ptr->payload[i] = pattern_word(3, i, seed_a);
    }
    host_fence_release();
    release_store(&ctx.cb->ticket, 3);
    consume_read_kernel<<<1, 64, 0, stream_a>>>(ctx.cb, ctx.slot_ptr, ctx.rank_ack, ctx.rank_result, 0, 3, seed_a);
    if (!spin_until([&] { return acquire_load(&ctx.rank_ack[0]) == 3; }, 5000.0, unused)) {
        fail_dual("window A never acknowledged after its post-stall reuse");
        return;
    }
    if (ctx.rank_result[0] != derived_word(3, seed_a)) {
        fail_dual("window A's post-stall reuse read a corrupted payload");
        return;
    }
    {
        const auto problem = run_b_gens(B_FINAL_ROUNDS, "phase 4", {});
        if (problem.has_value()) {
            fail_dual(*problem);
            return;
        }
    }
    sync_with_deadline(stream_a, SYNC_DEADLINE_MS, "dual window A final sync");
    sync_with_deadline(stream_b, SYNC_DEADLINE_MS, "dual window B final sync");
    if (!guards_intact(*ctx.slot_ptr) || !guards_intact(*ctx.slot_b)) {
        fail_dual("guards clobbered");
        return;
    }
    set_case(ctx, case_id::dual_window_independent_progress, outcome::pass,
             "window B completed " + std::to_string(b_generation) +
                 " generations while window A was stalled by a withheld ticket, " + std::to_string(WAITER_BLOCKS) +
                 " resident waiter workgroups and a delayed last reader; A's slot was never touched by B's "
                 "traffic, A completed exactly on release and reused its channel after the stall");
}

// Guide 34.1/34.2 and 39.6 P07/P09/P10 at the single-device software layer
// (probe LEMMA 4): the CPU service-loop contract. P07: a detached worker
// thread drives channel B's whole generation loop while the main thread only
// observes a progress flag — progress must not depend on the observer's
// participation (the begin/ready/finish deadlock of guide 34.1). P09: two
// channels are ready simultaneously, one reader deterministically slow; the
// bounded service publishes both tickets back to back and the fast channel
// completes its verified read while the slow one is still mid-read, in both
// orientations. P10: shutdown arrives while a waiter is resident on channel
// A and the worker is mid-loop on B; the worker publishes the stop words and
// drains both streams inside itself before it exits, so the observer's join
// never strands live device work. Cross-device qualification still belongs
// to five_rank_cross_device.
void case_service_loop(device_context & ctx) {
    constexpr std::uint32_t P07_GENERATIONS  = 256;
    constexpr std::uint32_t P09_ORIENTATIONS = 2;
    constexpr std::uint32_t P10_PROGRESS     = 64;
    const std::uint32_t     seed_a           = ctx.seed;
    const std::uint32_t     seed_b           = ctx.seed ^ 0x55AAu;

    reset_control(*ctx.cb);
    reset_control(*ctx.cb2);
    init_guards(*ctx.slot_ptr);
    init_guards(*ctx.slot_b);
    std::memset(ctx.slot_ptr->payload, 0, sizeof(ctx.slot_ptr->payload));
    std::memset(ctx.slot_b->payload, 0, sizeof(ctx.slot_b->payload));
    std::memset(ctx.rank_ack, 0, RANKS * sizeof(std::uint64_t));
    std::memset(ctx.rank_result, 0, RANKS * sizeof(std::uint64_t));

    stream_set        streams(2);
    const hipStream_t stream_a = streams.streams[0];
    const hipStream_t stream_b = streams.streams[1];

    // One bounded service step shared by the worker phases: publish the
    // generation, wait for the consumer's complete verified read. The ack
    // wait is itself bounded, so a stuck consumer fails the worker instead
    // of hanging the run.
    const auto run_generation = [&](control_block * cb, slot * s, std::uint32_t consumer, std::uint32_t seed,
                                    hipStream_t stream, std::uint64_t generation) -> std::optional<std::string> {
        for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
            s->payload[i] = pattern_word(generation, i, seed);
        }
        host_fence_release();
        release_store(&cb->ticket, generation);
        consume_read_kernel<<<1, 64, 0, stream>>>(cb, s, ctx.rank_ack, ctx.rank_result, consumer, generation, seed);
        double     unused = 0;
        const bool acked =
            spin_until([&] { return acquire_load(&ctx.rank_ack[consumer]) == generation; }, 5000.0, unused);
        if (!acked) {
            return std::optional<std::string>("generation " + std::to_string(generation) + " did not complete");
        }
        if (ctx.rank_result[consumer] != derived_word(generation, seed)) {
            return std::optional<std::string>("generation " + std::to_string(generation) +
                                              " derived the wrong result word");
        }
        return std::optional<std::string>{};
    };
    const auto fail_service = [&](const std::string & detail) {
        fail_case(ctx, case_id::service_loop_independent_progress, outcome::fail, detail);
    };

    struct worker_result {
        std::atomic<bool>          done{ false };
        std::atomic<bool>          drained{ false };
        std::atomic<std::uint64_t> generations{ 0 };
        std::string                detail;
    };

    // Phase 1 (P07): the worker owns every publish; the observer makes no
    // HIP call and no store to either channel until the worker is done.
    {
        worker_result phase1;
        std::thread   worker([&] {
            try {
                check_hip(hipSetDevice(ctx.record.ordinal), "service worker set device");
                for (std::uint64_t generation = 1; generation <= P07_GENERATIONS; ++generation) {
                    const auto problem = run_generation(ctx.cb2, ctx.slot_b, 1, seed_b, stream_b, generation);
                    if (problem.has_value()) {
                        phase1.detail = *problem;
                        break;
                    }
                    phase1.generations.store(generation, std::memory_order_relaxed);
                }
            } catch (const std::exception & error) {
                phase1.detail = error.what();
            }
            phase1.done.store(true, std::memory_order_release);
        });
        double        unused = 0;
        const bool finished  = spin_until([&] { return phase1.done.load(std::memory_order_acquire); }, 15000.0, unused);
        if (!finished) {
            release_store(&ctx.cb2->abort, 1);
            worker.join();
            fail_service("the worker did not finish its generation loop within the observer's deadline");
            return;
        }
        worker.join();
        if (!phase1.detail.empty()) {
            fail_service("phase 1 worker: " + phase1.detail);
            return;
        }
        sync_with_deadline(stream_b, SYNC_DEADLINE_MS, "phase 1 worker sync");
        if (phase1.generations.load(std::memory_order_acquire) != P07_GENERATIONS ||
            ctx.rank_result[1] != derived_word(P07_GENERATIONS, seed_b) || !guards_intact(*ctx.slot_b)) {
            fail_service("the worker-driven channel did not finish with exact values");
            return;
        }
    }

    // Phase 2 (P09): both channels ready at once; one reader deterministically
    // slow (delayed_reader withholds its acknowledgment for its whole read),
    // the other fast. The service publishes the two tickets back to back;
    // the slow channel's unfinished round must never gate the fast channel,
    // in either orientation.
    for (std::uint32_t orientation = 0; orientation < P09_ORIENTATIONS; ++orientation) {
        reset_control(*ctx.cb);
        reset_control(*ctx.cb2);
        release_store(&ctx.rank_ack[0], 0);
        release_store(&ctx.rank_ack[1], 0);
        release_store(&ctx.rank_result[0], 0);
        release_store(&ctx.rank_result[1], 0);
        for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
            ctx.slot_ptr->payload[i] = pattern_word(1, i, seed_a);
            ctx.slot_b->payload[i]   = pattern_word(1, i, seed_b);
        }
        const std::uint32_t slow_channel = orientation == 0 ? 0u : 1u;
        const std::uint32_t fast_channel = orientation == 0 ? 1u : 0u;
        control_block *     slow_cb      = slow_channel == 0 ? ctx.cb : ctx.cb2;
        control_block *     fast_cb      = fast_channel == 0 ? ctx.cb : ctx.cb2;
        slot *              slow_slot    = slow_channel == 0 ? ctx.slot_ptr : ctx.slot_b;
        slot *              fast_slot    = fast_channel == 0 ? ctx.slot_ptr : ctx.slot_b;
        const std::uint32_t slow_seed    = slow_channel == 0 ? seed_a : seed_b;
        const std::uint32_t fast_seed    = fast_channel == 0 ? seed_a : seed_b;
        const hipStream_t   slow_stream  = slow_channel == 0 ? stream_a : stream_b;
        const hipStream_t   fast_stream  = fast_channel == 0 ? stream_a : stream_b;
        delayed_reader_kernel<<<1, 64, 0, slow_stream>>>(slow_cb, slow_slot, ctx.rank_ack, ctx.rank_result,
                                                         slow_channel, 1, slow_seed);
        consume_read_kernel<<<1, 64, 0, fast_stream>>>(fast_cb, fast_slot, ctx.rank_ack, ctx.rank_result, fast_channel,
                                                       1, fast_seed);
        host_fence_release();
        // The bounded service: publish the slow channel's ticket, then the
        // fast channel's, without waiting for the slow round to complete.
        release_store(&slow_cb->ticket, 1);
        release_store(&fast_cb->ticket, 1);
        double     unused = 0;
        const bool fast_acked =
            spin_until([&] { return acquire_load(&ctx.rank_ack[fast_channel]) == 1; }, 5000.0, unused);
        if (!fast_acked) {
            fail_service("orientation " + std::to_string(orientation) + ": the fast channel never completed");
            return;
        }
        if (acquire_load(&ctx.rank_ack[slow_channel]) != 0) {
            fail_service("orientation " + std::to_string(orientation) +
                         ": the slow reader finished before the fast channel, so head-of-line serialization "
                         "cannot be falsified");
            return;
        }
        // The fast channel finished its full verified read while the slow
        // reader was still mid-read: serving it did not wait for the slow
        // round. Now wait out the slow reader.
        const bool slow_acked =
            spin_until([&] { return acquire_load(&ctx.rank_ack[slow_channel]) == 1; }, 5000.0, unused);
        if (!slow_acked) {
            fail_service("orientation " + std::to_string(orientation) + ": the slow reader never completed");
            return;
        }
        sync_with_deadline(slow_stream, SYNC_DEADLINE_MS, "phase 2 slow sync");
        sync_with_deadline(fast_stream, SYNC_DEADLINE_MS, "phase 2 fast sync");
        if (ctx.rank_result[slow_channel] != derived_word(1, slow_seed) ||
            ctx.rank_result[fast_channel] != derived_word(1, fast_seed)) {
            fail_service("orientation " + std::to_string(orientation) + ": a reader verified a wrong payload");
            return;
        }
    }
    if (!guards_intact(*ctx.slot_ptr) || !guards_intact(*ctx.slot_b)) {
        fail_service("guards clobbered");
        return;
    }

    // Phase 3 (P10): shutdown while a waiter is resident on channel A and
    // the worker is mid-loop on channel B. The worker — the only progress
    // owner — publishes the stop words and drains both streams before it
    // exits; the observer only joins afterwards, so the join can never
    // strand live device work.
    {
        reset_control(*ctx.cb);
        reset_control(*ctx.cb2);
        release_store(&ctx.rank_ack[0], 0);
        release_store(&ctx.rank_ack[1], 0);
        release_store(&ctx.rank_result[1], 0);
        bare_waiter_kernel<<<1, 64, 0, stream_a>>>(ctx.cb, 1);
        worker_result     phase3;
        std::atomic<bool> shutdown{ false };
        std::thread       worker([&] {
            try {
                check_hip(hipSetDevice(ctx.record.ordinal), "shutdown worker set device");
                while (!shutdown.load(std::memory_order_acquire)) {
                    const std::uint64_t generation = phase3.generations.load(std::memory_order_relaxed) + 1;
                    const auto          problem = run_generation(ctx.cb2, ctx.slot_b, 1, seed_b, stream_b, generation);
                    if (problem.has_value()) {
                        phase3.detail = *problem;
                        break;
                    }
                    phase3.generations.store(generation, std::memory_order_relaxed);
                }
                if (phase3.detail.empty()) {
                    release_store(&ctx.cb->abort, 1);
                    release_store(&ctx.cb2->abort, 1);
                    sync_with_deadline(stream_a, SYNC_DEADLINE_MS, "shutdown drain A");
                    sync_with_deadline(stream_b, SYNC_DEADLINE_MS, "shutdown drain B");
                    if (acquire_load(&ctx.cb->exit_code) != EXIT_ABORT_OBSERVED) {
                        phase3.detail = "the resident waiter did not honor the stop word during shutdown";
                    }
                    phase3.drained.store(true, std::memory_order_release);
                }
            } catch (const std::exception & error) {
                phase3.detail = error.what();
            }
            phase3.done.store(true, std::memory_order_release);
        });
        double            unused     = 0;
        const bool        progressed = spin_until(
            [&] {
                return phase3.generations.load(std::memory_order_acquire) >= P10_PROGRESS ||
                       phase3.done.load(std::memory_order_acquire);
            },
            10000.0, unused);
        if (!progressed) {
            shutdown.store(true, std::memory_order_release);
            release_store(&ctx.cb->abort, 1);
            release_store(&ctx.cb2->abort, 1);
            worker.join();
            fail_service("the worker made no observable progress before shutdown");
            return;
        }
        if (phase3.done.load(std::memory_order_acquire)) {
            worker.join();
            fail_service("phase 3 worker: " + phase3.detail);
            return;
        }
        shutdown.store(true, std::memory_order_release);
        double     unused2  = 0;
        const bool finished = spin_until([&] { return phase3.done.load(std::memory_order_acquire); }, 15000.0, unused2);
        if (!finished) {
            release_store(&ctx.cb->abort, 1);
            release_store(&ctx.cb2->abort, 1);
            worker.join();
            fail_service("the worker did not finish its shutdown drain within the observer's deadline");
            return;
        }
        worker.join();
        if (!phase3.detail.empty()) {
            fail_service("phase 3 worker: " + phase3.detail);
            return;
        }
        if (!phase3.drained.load(std::memory_order_acquire) ||
            acquire_load(&ctx.cb->exit_code) != EXIT_ABORT_OBSERVED) {
            fail_service("the worker exited without draining the resident waiter");
            return;
        }
        if (ctx.rank_result[1] != derived_word(phase3.generations.load(std::memory_order_acquire), seed_b)) {
            fail_service("the worker's last generation did not finish with the exact result");
            return;
        }
        sync_with_deadline(stream_a, SYNC_DEADLINE_MS, "phase 3 final sync A");
        sync_with_deadline(stream_b, SYNC_DEADLINE_MS, "phase 3 final sync B");
        if (!guards_intact(*ctx.slot_ptr) || !guards_intact(*ctx.slot_b)) {
            fail_service("guards clobbered");
            return;
        }
    }

    set_case(ctx, case_id::service_loop_independent_progress, outcome::pass,
             "a worker thread drove channel B through " + std::to_string(P07_GENERATIONS) +
                 " generations while the observer only read status; two simultaneously-ready channels were both "
                 "served with the fast one finishing mid-read of the slow one in both orientations; and shutdown "
                 "with a resident waiter drained inside the worker before it exited");
}

// Guide 39.7 F04 / 35.5 / 41.6: the CPU reduction must refuse to publish a
// corrupted fold instead of handing it to the consumers. The probe's
// synthetic words carry no float semantics, so the gate is the exact fold
// identity (the rank weights sum to 1 mod 2^64): any corrupted word breaks
// it deterministically, and a quiet-NaN bit pattern is one such corruption.
// The clean fold passes; a flipped bit and a NaN pattern are both rejected
// before publication and nothing is written to the output slot.
void case_fold_gate(device_context & ctx) {
    constexpr std::uint64_t GENERATION  = 1;
    constexpr std::uint32_t INJECT_RANK = 2;
    constexpr std::uint32_t INJECT_WORD = 3;
    reset_control(*ctx.cb);
    init_guards(*ctx.slot_ptr);
    std::memset(ctx.slot_ptr->payload, 0, sizeof(ctx.slot_ptr->payload));
    std::memset(ctx.rank_ready, 0, RANKS * sizeof(std::uint64_t));
    std::memset(ctx.rank_partials, 0, RANKS * PAYLOAD_WORDS * sizeof(std::uint64_t));
    stream_set streams(RANKS);
    for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
        partial_producer_kernel<<<1, 64, 0, streams.streams[rank]>>>(ctx.rank_partials, ctx.rank_ready, rank,
                                                                     GENERATION, ctx.seed);
    }
    double unused = 0;
    for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
        const bool produced =
            spin_until([&] { return acquire_load(&ctx.rank_ready[rank]) == GENERATION; }, 5000.0, unused);
        if (!produced) {
            fail_case(ctx, case_id::nonfinite_reduction_gate, outcome::fail,
                      "producer rank " + std::to_string(rank) + " did not publish its partial");
            return;
        }
    }
    std::uint64_t fold[PAYLOAD_WORDS] = {};
    if (fold_partials(ctx.rank_partials, GENERATION, ctx.seed, fold).has_value()) {
        fail_case(ctx, case_id::nonfinite_reduction_gate, outcome::fail, "the clean fold was rejected");
        return;
    }
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        if (fold[i] != pattern_word(GENERATION, i, ctx.seed)) {
            fail_case(ctx, case_id::nonfinite_reduction_gate, outcome::fail,
                      "the clean fold does not reproduce the shared pattern at word " + std::to_string(i));
            return;
        }
    }
    const std::uint64_t saved           = ctx.rank_partials[INJECT_RANK * PAYLOAD_WORDS + INJECT_WORD];
    // The high-bit flip is the one a power-of-two weight would hide: the odd
    // weights are a bijection, so the gate must see it like any other bit.
    const std::uint64_t corruptions[]   = { saved ^ (1ULL << 63), 0x7FF8000000000000ULL };
    const auto          assert_rejected = [&]() {
        if (acquire_load(&ctx.cb->ticket) != 0) {
            fail_case(ctx, case_id::nonfinite_reduction_gate, outcome::fail,
                      "a generation was published despite the rejected fold");
            return false;
        }
        for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
            if (ctx.slot_ptr->payload[i] != 0) {
                fail_case(ctx, case_id::nonfinite_reduction_gate, outcome::fail,
                          "the output slot was written despite the rejected fold");
                return false;
            }
        }
        return true;
    };
    for (const std::uint64_t corruption : corruptions) {
        ctx.rank_partials[INJECT_RANK * PAYLOAD_WORDS + INJECT_WORD] = corruption;
        const auto problem = fold_partials(ctx.rank_partials, GENERATION, ctx.seed, fold);
        if (!problem.has_value()) {
            fail_case(ctx, case_id::nonfinite_reduction_gate, outcome::fail,
                      "a corrupted partial was folded and accepted instead of being rejected");
            return;
        }
        if (!assert_rejected()) {
            return;
        }
    }
    ctx.rank_partials[INJECT_RANK * PAYLOAD_WORDS + INJECT_WORD] = saved;
    for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
        sync_with_deadline(streams.streams[rank], SYNC_DEADLINE_MS, "fold gate producer sync");
    }
    set_case(ctx, case_id::nonfinite_reduction_gate, outcome::pass,
             "the clean five-way fold reproduces the shared pattern exactly; a high-bit flip and a quiet-NaN bit "
             "pattern in one producer partial were both rejected before publication with nothing written to the "
             "output slot");
}

// Guide 39.7 F07 / 35.8 handoff deadline: a consumer finishes its whole
// verified read but withholds the acknowledgment. The host's bounded wait
// must fire instead of waiting forever, the slot must stay untouched (no
// reuse while a reader is un-acknowledged), and the stop word then recovers
// the waiter for the drain. This exercises the spin_until timeout branch,
// which a healthy device never reaches in the other cases.
void case_host_wait_deadline(device_context & ctx) {
    constexpr double SHORT_DEADLINE_MS = 400.0;
    reset_control(*ctx.cb);
    init_guards(*ctx.slot_ptr);
    release_store(&ctx.rank_ack[0], 0);
    release_store(&ctx.rank_result[0], 0);
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        ctx.slot_ptr->payload[i] = pattern_word(1, i, ctx.seed);
    }
    host_fence_release();
    release_store(&ctx.cb->ticket, 1);
    read_hold_kernel<<<1, 64>>>(ctx.cb, ctx.slot_ptr, ctx.rank_result, 0, 1, ctx.seed);
    double     elapsed = 0;
    stopwatch  timer;
    const bool acked     = spin_until([&] { return acquire_load(&ctx.rank_ack[0]) == 1; }, SHORT_DEADLINE_MS, elapsed);
    const bool read_done = spin_until([&] { return acquire_load(&ctx.rank_result[0]) != 0; }, 2000.0, elapsed);
    if (!read_done) {
        fail_case(ctx, case_id::host_wait_deadline_no_reuse, outcome::fail,
                  "the consumer never completed its read before the deadline");
        return;
    }
    if (ctx.rank_result[0] != derived_word(1, ctx.seed)) {
        fail_case(ctx, case_id::host_wait_deadline_no_reuse, outcome::fail,
                  "the consumer verified a wrong payload while withholding its acknowledgment");
        return;
    }
    if (acked) {
        fail_case(ctx, case_id::host_wait_deadline_no_reuse, outcome::fail,
                  "an acknowledgment that the consumer never publishes appeared");
        return;
    }
    if (elapsed >= 0 || timer.milliseconds() < SHORT_DEADLINE_MS * 0.9) {
        fail_case(ctx, case_id::host_wait_deadline_no_reuse, outcome::fail,
                  "the bounded wait returned before its deadline without the acknowledgment");
        return;
    }
    // No reuse while the reader is un-acknowledged: the published payload
    // must still be byte-identical and the guards untouched.
    for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
        if (ctx.slot_ptr->payload[i] != pattern_word(1, i, ctx.seed)) {
            fail_case(ctx, case_id::host_wait_deadline_no_reuse, outcome::fail,
                      "the slot was rewritten while the consumer's read was un-acknowledged");
            return;
        }
    }
    if (!guards_intact(*ctx.slot_ptr)) {
        fail_case(ctx, case_id::host_wait_deadline_no_reuse, outcome::fail, "guards clobbered");
        return;
    }
    release_store(&ctx.cb->abort, 1);
    sync_with_deadline(0, SYNC_DEADLINE_MS, "read-hold drain");
    if (acquire_load(&ctx.cb->exit_code) != EXIT_ABORT_OBSERVED) {
        fail_case(ctx, case_id::host_wait_deadline_no_reuse, outcome::fail,
                  "the waiter did not honor the stop word after the deadline");
        return;
    }
    set_case(ctx, case_id::host_wait_deadline_no_reuse, outcome::pass,
             "a consumer that finished its verified read but withheld the acknowledgment fired the host's bounded "
             "wait at " +
                 std::to_string(SHORT_DEADLINE_MS) +
                 " ms; the slot stayed byte-identical (no reuse) and the stop word recovered the waiter");
}

void allocate_device_context(device_context & ctx, std::int32_t ordinal, std::uint32_t rounds) {
    check_hip(hipSetDevice(ordinal), "set device");
    hipDeviceProp_t properties{};
    check_hip(hipGetDeviceProperties(&properties, ordinal), "device properties");
    ctx.record.ordinal             = ordinal;
    ctx.record.name                = properties.name;
    ctx.record.gcn_arch            = properties.gcnArchName;
    ctx.record.total_vram_bytes    = properties.totalGlobalMem;
    ctx.record.can_map_host_memory = properties.canMapHostMemory;
    char bus_id[64]                = {};
    check_hip(hipDeviceGetPCIBusId(bus_id, sizeof(bus_id), ordinal), "pci bus id");
    ctx.record.pci_bus_id = bus_id;
    ctx.rounds            = rounds;
    ctx.seed              = static_cast<std::uint32_t>(0x7C0D + ordinal * 131);

    void * control_raw = nullptr;
    void * slot_raw    = nullptr;
    void * out_raw     = nullptr;
    check_hip(hipHostMalloc(&control_raw, sizeof(control_block), hipHostMallocMapped | hipHostMallocPortable),
              "control host alloc");
    check_hip(hipHostMalloc(&slot_raw, sizeof(slot), hipHostMallocMapped | hipHostMallocPortable), "slot host alloc");
    check_hip(hipHostMalloc(&out_raw, sizeof(std::uint64_t), hipHostMallocMapped | hipHostMallocPortable),
              "out host alloc");
    unsigned int flags = 0;
    check_hip(hipHostGetFlags(&flags, control_raw), "host flags");
    ctx.mapped          = (flags & hipHostMallocMapped) != 0;
    ctx.mapped_flags    = "0x" + std::to_string(flags);
    ctx.cb              = static_cast<control_block *>(control_raw);
    ctx.slot_ptr        = static_cast<slot *>(slot_raw);
    ctx.out_word        = static_cast<std::uint64_t *>(out_raw);
    void * control2_raw = nullptr;
    void * slot_b_raw   = nullptr;
    check_hip(hipHostMalloc(&control2_raw, sizeof(control_block), hipHostMallocMapped | hipHostMallocPortable),
              "second control host alloc");
    check_hip(hipHostMalloc(&slot_b_raw, sizeof(slot), hipHostMallocMapped | hipHostMallocPortable),
              "second slot host alloc");
    ctx.cb2    = static_cast<control_block *>(control2_raw);
    ctx.slot_b = static_cast<slot *>(slot_b_raw);
    check_hip(hipMalloc(&ctx.counter, sizeof(std::uint32_t)), "counter device alloc");
    void * rmw_add_raw = nullptr;
    void * rmw_cas_raw = nullptr;
    check_hip(hipHostMalloc(&rmw_add_raw, sizeof(std::uint64_t), hipHostMallocMapped | hipHostMallocPortable),
              "rmw add alloc");
    check_hip(hipHostMalloc(&rmw_cas_raw, sizeof(std::uint64_t), hipHostMallocMapped | hipHostMallocPortable),
              "rmw cas alloc");
    ctx.rmw_add         = static_cast<std::uint64_t *>(rmw_add_raw);
    ctx.rmw_cas         = static_cast<std::uint64_t *>(rmw_cas_raw);
    void * slot2_raw    = nullptr;
    void * ack_raw      = nullptr;
    void * result_raw   = nullptr;
    void * ready_raw    = nullptr;
    void * partials_raw = nullptr;
    check_hip(hipHostMalloc(&slot2_raw, sizeof(slot), hipHostMallocMapped | hipHostMallocPortable), "slot2 host alloc");
    check_hip(hipHostMalloc(&ack_raw, RANKS * sizeof(std::uint64_t), hipHostMallocMapped | hipHostMallocPortable),
              "rank ack alloc");
    check_hip(hipHostMalloc(&result_raw, RANKS * sizeof(std::uint64_t), hipHostMallocMapped | hipHostMallocPortable),
              "rank result alloc");
    check_hip(hipHostMalloc(&ready_raw, RANKS * sizeof(std::uint64_t), hipHostMallocMapped | hipHostMallocPortable),
              "rank ready alloc");
    check_hip(hipHostMalloc(&partials_raw, RANKS * PAYLOAD_WORDS * sizeof(std::uint64_t),
                            hipHostMallocMapped | hipHostMallocPortable),
              "rank partials alloc");
    ctx.slot2_ptr     = static_cast<slot *>(slot2_raw);
    ctx.rank_ack      = static_cast<std::uint64_t *>(ack_raw);
    ctx.rank_result   = static_cast<std::uint64_t *>(result_raw);
    ctx.rank_ready    = static_cast<std::uint64_t *>(ready_raw);
    ctx.rank_partials = static_cast<std::uint64_t *>(partials_raw);
    init_guards(*ctx.slot2_ptr);
    init_guards(*ctx.slot_b);
    std::memset(ctx.cb, 0, sizeof(control_block));
    std::memset(ctx.cb2, 0, sizeof(control_block));
}

void free_device_context(device_context & ctx) {
    // A context whose setup threw partway owns only some of these pointers;
    // release what exists so the report can still be written.
    if (ctx.record.ordinal < 0) {
        return;
    }
    // hipFree acts on the current device; after the five-rank case the
    // current device is whatever rank ran last, so pin the owner here.
    check_hip(hipSetDevice(ctx.record.ordinal), "free set device");
    if (ctx.cb != nullptr) {
        check_hip(hipHostFree(ctx.cb), "control host free");
    }
    if (ctx.cb2 != nullptr) {
        check_hip(hipHostFree(ctx.cb2), "second control host free");
    }
    if (ctx.slot_ptr != nullptr) {
        check_hip(hipHostFree(ctx.slot_ptr), "slot host free");
    }
    if (ctx.slot_b != nullptr) {
        check_hip(hipHostFree(ctx.slot_b), "second slot host free");
    }
    if (ctx.out_word != nullptr) {
        check_hip(hipHostFree(ctx.out_word), "out host free");
    }
    if (ctx.rmw_add != nullptr) {
        check_hip(hipHostFree(ctx.rmw_add), "rmw add free");
    }
    if (ctx.rmw_cas != nullptr) {
        check_hip(hipHostFree(ctx.rmw_cas), "rmw cas free");
    }
    if (ctx.slot2_ptr != nullptr) {
        check_hip(hipHostFree(ctx.slot2_ptr), "slot2 host free");
    }
    if (ctx.rank_ack != nullptr) {
        check_hip(hipHostFree(ctx.rank_ack), "rank ack free");
    }
    if (ctx.rank_result != nullptr) {
        check_hip(hipHostFree(ctx.rank_result), "rank result free");
    }
    if (ctx.rank_ready != nullptr) {
        check_hip(hipHostFree(ctx.rank_ready), "rank ready free");
    }
    if (ctx.rank_partials != nullptr) {
        check_hip(hipHostFree(ctx.rank_partials), "rank partials free");
    }
    if (ctx.counter != nullptr) {
        check_hip(hipFree(ctx.counter), "counter free");
    }
}

// One guarded host allocation for the five-rank case (guide 39.5 H01): two
// guard words on each side of the five disjoint rank words, then one ready
// word per rank. A rank writing past its word hits a guard instead of
// silently corrupting a sibling's region.
struct five_rank_mem {
    std::uint64_t guard_lo[2];
    std::uint64_t payload[RANKS];
    std::uint64_t guard_hi[2];
    std::uint64_t ready[RANKS];
};

// Per-rank execution binding for the shared five-rank rounds protocol.
struct rank_stream {
    hipStream_t   stream;   // owned by the caller
    std::int32_t  ordinal;  // device ordinal for this rank (equal for all ranks in the single-device case)
    std::uint32_t seed;     // per-rank payload seed
};

// Shared five-rank rounds protocol (guide 39.5 H01): every generation, each
// rank writes its disjoint payload word and its OWN ready word, the host
// observes all five ready words through acquire loads (running visibility),
// drains every stream, verifies fragment provenance and guards, then resets
// for the next round. The five-device case runs it with one device per
// rank; the per-device five_rank_indexing case runs it with five streams on
// one device, so the exact kernel, layout and per-rank ready indexing are
// exercised on every dev machine instead of shipping untested to the
// operator. A dead or misbehaving rank is attributed by rank and round;
// streams are always drained before the allocation is touched again or
// freed.
std::pair<std::string, std::uint32_t> run_rank_rounds(five_rank_mem & mem,
                                                      rank_stream (&ranks)[RANKS],
                                                      std::uint32_t rounds) {
    std::string   failure;
    std::uint32_t failed_rank  = RANKS;  // sentinel: not a specific rank
    std::uint32_t current_rank = RANKS;
    try {
        for (std::uint64_t generation = 1; generation <= rounds && failure.empty(); ++generation) {
            for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
                current_rank = rank;
                check_hip(hipSetDevice(ranks[rank].ordinal), "rank set device");
                rank_write_kernel<<<1, 64, 0, ranks[rank].stream>>>(mem.payload, mem.ready, rank, ranks[rank].seed,
                                                                    generation);
            }
            double     elapsed   = 0;
            const bool all_ready = spin_until(
                [&] {
                    for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
                        if (acquire_load(&mem.ready[rank]) != generation) {
                            return false;
                        }
                    }
                    return true;
                },
                5000.0, elapsed);
            if (!all_ready) {
                for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
                    if (acquire_load(&mem.ready[rank]) != generation) {
                        failed_rank = rank;
                        failure     = "rank " + std::to_string(rank) + " did not publish ready in round " +
                                      std::to_string(generation) + " within 5 s";
                        break;
                    }
                }
            }
            // Drain every stream before reading payload or resetting the
            // allocation; this also stops a late store after the next reset.
            for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
                current_rank = rank;
                check_hip(hipSetDevice(ranks[rank].ordinal), "rank sync device");
                try {
                    sync_with_deadline(ranks[rank].stream, SYNC_DEADLINE_MS, "rank drain");
                } catch (const std::exception & error) {
                    if (failure.empty()) {
                        failed_rank = rank;
                        failure     = "rank " + std::to_string(rank) + " drain failed: " + error.what();
                    }
                }
            }
            if (failure.empty()) {
                for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
                    const std::uint64_t expected = pattern_word(generation, rank, ranks[rank].seed);
                    if (mem.payload[rank] != expected) {
                        failed_rank = rank;
                        failure = "rank " + std::to_string(rank) + " wrote " + std::to_string(mem.payload[rank]) +
                                  " instead of " + std::to_string(expected) + " in round " + std::to_string(generation);
                        break;
                    }
                }
            }
            if (failure.empty() && (mem.guard_lo[0] != GUARD_VALUE || mem.guard_lo[1] != GUARD_VALUE ||
                                    mem.guard_hi[0] != GUARD_VALUE || mem.guard_hi[1] != GUARD_VALUE)) {
                failure = "guard word changed in round " + std::to_string(generation);
            }
            std::memset(mem.payload, 0, sizeof(mem.payload));
            std::memset(mem.ready, 0, sizeof(mem.ready));
        }
    } catch (const std::exception & error) {
        if (failure.empty()) {
            failure = std::string("hip failure: ") + error.what();
            if (current_rank < RANKS) {
                failed_rank = current_rank;
                failure += " (rank " + std::to_string(current_rank) + ")";
            }
        }
    }
    return { failure, failed_rank };
}

// Guide 39.5 H01's per-rank half on one device: five streams stand in for
// five ranks and run the exact cross-device kernel against one guarded
// allocation; the host verifies every rank's payload word AND its own ready
// word for the whole round count. This pins the five-rank kernel indexing
// on the live device of every machine that runs the probe — it is the
// regression net that caught the all-ranks-stomp-ready[0] bug before it
// could reach the five-card run — but it is NOT cross-device qualification:
// multi-device visibility stays with five_rank_cross_device.
void case_five_rank_indexing(device_context & ctx) {
    five_rank_mem * mem = nullptr;
    try {
        check_hip(hipHostMalloc(&mem, sizeof(five_rank_mem), hipHostMallocMapped | hipHostMallocPortable),
                  "five-rank indexing host alloc");
        std::memset(mem, 0, sizeof(five_rank_mem));
        mem->guard_lo[0] = mem->guard_lo[1] = mem->guard_hi[0] = mem->guard_hi[1] = GUARD_VALUE;
        stream_set  streams(RANKS);
        rank_stream ranks[RANKS];
        for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
            ranks[rank] = { streams.streams[rank], ctx.record.ordinal, ctx.seed + 131u * (rank + 1u) };
        }
        const auto result = run_rank_rounds(*mem, ranks, ctx.rounds);
        (void) hipHostFree(mem);
        if (!result.first.empty()) {
            fail_case(ctx, case_id::five_rank_indexing, outcome::fail, result.first);
            return;
        }
    } catch (const std::exception & error) {
        if (mem != nullptr) {
            (void) hipHostFree(mem);
        }
        fail_case(ctx, case_id::five_rank_indexing, outcome::fail, std::string("hip failure: ") + error.what());
        return;
    }
    set_case(ctx, case_id::five_rank_indexing, outcome::pass,
             "one device stood in for five ranks: every rank wrote its disjoint word and its own ready word for " +
                 std::to_string(ctx.rounds) +
                 " rounds with guards intact (proves per-rank kernel indexing, not "
                 "cross-device visibility)");
}

// Caller guarantees exactly five devices. Runs the shared rounds protocol
// with one device per rank. Before the first launch every rank must have
// passed setup AND mapped host memory: a rank that failed setup carries
// device_identity=fail on its record (its ordinal is not a setup fact), and
// a rank whose host allocation came back without the mapped flag cannot
// touch the shared payload. Both are attributed precisely instead of
// launching kernels on a broken context and misreporting the result as a
// stall (guide 47.2).
outcome run_five_rank(std::vector<device_context> & contexts, probe_report & report, std::uint32_t rounds) {
    std::string     failure;
    std::uint32_t   failed_rank  = RANKS;  // sentinel: not a specific rank
    std::uint32_t   current_rank = RANKS;
    five_rank_mem * mem          = nullptr;
    rank_stream     ranks[RANKS];
    std::uint32_t   streams_created = 0;

    // A HIP failure anywhere in this case is attributed and reported; it
    // never aborts the process, because the five-rank evidence is exactly
    // what the operator came to collect.
    try {
        for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
            if (result_of(contexts[rank], case_id::device_identity) != outcome::pass) {
                failed_rank = rank;
                failure     = "rank " + std::to_string(rank) + " was not set up (see its device_identity case)";
                break;
            }
            if (!contexts[rank].mapped) {
                failed_rank = rank;
                failure = "rank " + std::to_string(rank) + " cannot map host memory (" + contexts[rank].mapped_flags +
                          "); the five-rank case needs mapped host memory on "
                          "every rank";
                break;
            }
        }
        if (failure.empty()) {
            check_hip(hipHostMalloc(&mem, sizeof(five_rank_mem), hipHostMallocMapped | hipHostMallocPortable),
                      "five-rank host alloc");
            std::memset(mem, 0, sizeof(five_rank_mem));
            mem->guard_lo[0] = mem->guard_lo[1] = mem->guard_hi[0] = mem->guard_hi[1] = GUARD_VALUE;
            for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
                current_rank = rank;
                ranks[rank]  = { nullptr, contexts[rank].record.ordinal, contexts[rank].seed };
                check_hip(hipSetDevice(contexts[rank].record.ordinal), "five-rank set device");
                check_hip(hipStreamCreate(&ranks[rank].stream), "stream create");
                streams_created = rank + 1;
            }
            const auto result = run_rank_rounds(*mem, ranks, rounds);
            if (!result.first.empty()) {
                failure     = result.first;
                failed_rank = result.second;
            }
        }
    } catch (const std::exception & error) {
        if (failure.empty()) {
            failure = std::string("hip failure: ") + error.what();
            if (current_rank < RANKS) {
                failed_rank = current_rank;
                failure += " (rank " + std::to_string(current_rank) + ")";
            }
        }
    }

    // Best-effort drain and teardown: evidence collection is already over,
    // so a broken rank must not prevent freeing what can be freed.
    for (std::uint32_t rank = 0; rank < streams_created; ++rank) {
        (void) hipSetDevice(ranks[rank].ordinal);
        try {
            sync_with_deadline(ranks[rank].stream, SYNC_DEADLINE_MS, "teardown drain");
        } catch (const std::exception &) {
            // Evidence collection is over; a stalled rank must not block
            // freeing what can still be freed.
        }
        (void) hipStreamDestroy(ranks[rank].stream);
    }
    if (mem != nullptr) {
        (void) hipHostFree(mem);
    }

    report.five_rank_detail = failure.empty() ? "five devices each wrote a disjoint word of one guarded host "
                                                "allocation for " +
                                                    std::to_string(rounds) + " rounds" :
                                                failure;
    const auto records      = five_rank_case_records(failure, failed_rank);
    for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
        report.devices[rank].cases.push_back(records[rank]);
    }
    return failure.empty() ? outcome::pass : outcome::fail;
}

// -------- measurements for the cost model (guide 14), never gates ----------

struct timing_stats {
    std::uint64_t samples = 0;
    std::uint64_t min     = 0;
    std::uint64_t median  = 0;
    std::uint64_t p90     = 0;
    std::uint64_t max     = 0;
};

timing_stats summarize(std::vector<std::uint64_t> values) {
    std::sort(values.begin(), values.end());
    timing_stats out;
    out.samples       = static_cast<std::uint64_t>(values.size());
    out.min           = values.front();
    out.max           = values.back();
    out.median        = values[values.size() / 2];  // upper middle for even counts
    std::size_t index = (values.size() * 90) / 100;
    if (index >= values.size()) {
        index = values.size() - 1;
    }
    out.p90 = values[index];
    return out;
}

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

void add_measurement(tq::relay::probe::bench_report & report, std::string name, std::string unit, timing_stats stats) {
    report.measurements.push_back(
        { std::move(name), std::move(unit), stats.samples, stats.min, stats.median, stats.p90, stats.max });
}

int run_bench(device_context &    ctx,
              const std::string & path,
              std::uint32_t       rounds,
              const std::string & runtime_version) {
    // The bench runs after the five-rank case leaves the current device on
    // the last rank; every HIP entry point pins its own device (guide 52
    // single-owner discipline).
    check_hip(hipSetDevice(ctx.record.ordinal), "bench set device");
    tq::relay::probe::bench_report report;
    report.build_id                   = "tq-relay-probe-bench";
    report.hip_runtime_version_string = runtime_version;
    report.device_ordinal             = ctx.record.ordinal;
    report.device_name                = ctx.record.name;
    report.gcn_arch_name              = ctx.record.gcn_arch;
    report.rounds                     = rounds;
    report.limits                     = {
        "Measurements are observations for the cost model (guide 14); they are not qualification gates.",
        "Latencies use the host steady clock and include kernel-launch time on this machine.",
        "Single device, single stream unless noted; five-rank numbers stay not measured here.",
        "median is the upper middle of the sorted samples; p90 uses index floor(samples*90/100).",
    };

    // 1. host publish -> device consumption acknowledged (one consumer).
    reset_control(*ctx.cb);
    init_guards(*ctx.slot_ptr);
    std::vector<std::uint64_t> publish;
    publish.reserve(rounds);
    for (std::uint64_t generation = 1; generation <= rounds; ++generation) {
        release_store(&ctx.rank_ack[0], 0);
        for (std::uint32_t i = 0; i < PAYLOAD_WORDS; ++i) {
            ctx.slot_ptr->payload[i] = pattern_word(generation, i, ctx.seed);
        }
        host_fence_release();
        const std::uint64_t begin = now_ns();
        release_store(&ctx.cb->ticket, generation);
        consume_read_kernel<<<1, 64>>>(ctx.cb, ctx.slot_ptr, ctx.rank_ack, ctx.rank_result, 0, generation, ctx.seed);
        double     unused = 0;
        const bool acked  = spin_until([&] { return acquire_load(&ctx.rank_ack[0]) == generation; }, 5000.0, unused);
        if (!acked || ctx.rank_result[0] != derived_word(generation, ctx.seed)) {
            std::fprintf(stderr, "bench failure: publish round %llu\n", static_cast<unsigned long long>(generation));
            release_store(&ctx.cb->abort, 1);
            sync_with_deadline(0, SYNC_DEADLINE_MS, "bench drain after failure");
            return 1;
        }
        publish.push_back(now_ns() - begin);
    }
    add_measurement(report, "host_publish_to_device_ack", "ns", summarize(std::move(publish)));

    // 2. device produce -> host observes ready (no acknowledgment loop).
    std::vector<std::uint64_t> produce;
    produce.reserve(rounds);
    for (std::uint64_t generation = 1; generation <= rounds; ++generation) {
        const std::uint64_t begin = now_ns();
        produce_once_kernel<<<1, 64>>>(ctx.cb, ctx.slot_ptr, generation, ctx.seed);
        double     unused = 0;
        const bool ready  = spin_until([&] { return acquire_load(&ctx.cb->ready) == generation; }, 5000.0, unused);
        if (!ready) {
            std::fprintf(stderr, "bench failure: produce round %llu\n", static_cast<unsigned long long>(generation));
            release_store(&ctx.cb->abort, 1);
            sync_with_deadline(0, SYNC_DEADLINE_MS, "bench drain after failure");
            return 1;
        }
        produce.push_back(now_ns() - begin);
    }
    add_measurement(report, "device_produce_to_host_ready", "ns", summarize(std::move(produce)));

    // 3. full emulated relay cycle: five producers, host fold, publish, five consumers.
    reset_control(*ctx.cb);
    init_guards(*ctx.slot_ptr);
    std::memset(ctx.rank_ready, 0, RANKS * sizeof(std::uint64_t));
    stream_set                 streams(RANKS);
    std::vector<std::uint64_t> cycle;
    cycle.reserve(rounds);
    for (std::uint64_t generation = 1; generation <= rounds; ++generation) {
        const std::uint64_t begin   = now_ns();
        const auto          problem = run_relay_round(ctx, streams.streams, generation);
        if (problem.has_value()) {
            std::fprintf(stderr, "bench failure: relay cycle %llu: %s\n", static_cast<unsigned long long>(generation),
                         problem->c_str());
            release_store(&ctx.cb->abort, 1);
            sync_with_deadline(0, SYNC_DEADLINE_MS, "bench drain after failure");
            return 1;
        }
        cycle.push_back(now_ns() - begin);
    }
    add_measurement(report, "five_stream_relay_cycle", "ns", summarize(std::move(cycle)));

    // 4. host five-way fold bandwidth over a large plain host buffer.
    constexpr std::uint32_t    FOLD_WORDS = 1u << 20;  // 8 MiB per rank
    std::vector<std::uint64_t> src(static_cast<std::size_t>(RANKS) * FOLD_WORDS);
    std::vector<std::uint64_t> dst(FOLD_WORDS);
    for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
        for (std::uint32_t i = 0; i < FOLD_WORDS; ++i) {
            src[static_cast<std::size_t>(rank) * FOLD_WORDS + i] = pattern_word(1, i, ctx.seed + rank);
        }
    }
    std::uint64_t fold_checksum = 0;
    auto          fold_pass     = [&]() -> std::uint64_t {
        const std::uint64_t begin = now_ns();
        for (std::uint32_t i = 0; i < FOLD_WORDS; ++i) {
            std::uint64_t sum = 0;
            for (std::uint32_t rank = 0; rank < RANKS; ++rank) {
                sum += src[static_cast<std::size_t>(rank) * FOLD_WORDS + i];
            }
            dst[i] = sum;
            fold_checksum += sum;  // additive: stays observable across both passes
        }
        return now_ns() - begin;
    };
    const std::uint64_t        bytes       = static_cast<std::uint64_t>(RANKS) * FOLD_WORDS * sizeof(std::uint64_t);
    // One pass reads ~40 MiB in a few milliseconds: a single sample (or a
    // best-of-two) is dominated by scheduling noise. Summarize a batch of
    // passes like every other measurement instead, so min/median/p90/max
    // mean what the bench schema says they mean.
    constexpr std::uint32_t    FOLD_PASSES = 32;
    std::vector<std::uint64_t> fold_bandwidth;
    fold_bandwidth.reserve(FOLD_PASSES);
    for (std::uint32_t pass = 0; pass < FOLD_PASSES; ++pass) {
        const std::uint64_t pass_ns = fold_pass();
        fold_bandwidth.push_back(pass_ns == 0 ? 0 : (bytes * 1000000000ULL) / pass_ns);
    }
    report.limits.push_back("host_five_way_fold checksum " + std::to_string(fold_checksum) +
                            " anchors the fold against dead-code elimination");
    add_measurement(report, "host_five_way_fold", "bytes_per_second", summarize(std::move(fold_bandwidth)));

    const write_result result = write_file_exclusive(path, to_json(report) + "\n");
    if (result.code != 0) {
        std::fprintf(stderr, "probe failure: %s\n", result.message.c_str());
    }
    return result.code;
}

void print_usage(const char * name) {
    std::printf("usage: %s [--output FILE] [--rounds N] [--bench FILE] [--help]\n", name);
    std::printf("  --output FILE  exclusive-create evidence JSON (default tq-relay-probe-report.json)\n");
    std::printf("  --rounds N     hot-reuse rounds per direction (default %u)\n", DEFAULT_ROUNDS);
    std::printf("  --bench FILE   also run latency/bandwidth measurements into this exclusive-create JSON\n");
}

int write_report(const std::string & path, const probe_report & report) {
    const write_result result = write_file_exclusive(path, to_json(report) + "\n");
    if (result.code != 0) {
        std::fprintf(stderr, "probe failure: %s\n", result.message.c_str());
    }
    return result.code;
}

}  // namespace

static int run_probe(const std::string & output_path, const std::string & bench_path, std::uint32_t rounds);

int main(int argc, char ** argv) {
    // The run stdout is part of the operator's evidence path: if the process
    // is killed mid-run, everything already printed must survive. A
    // redirected stream is fully buffered by default and a kill would drop
    // whatever is still in the buffer. _IONBF gives the C-standard "as soon
    // as possible" guarantee for every stream, redirected or not, and the
    // probe prints a handful of lines, so the syscall cost is irrelevant.
    // The guarantee itself is pinned by the GPU-free config test
    // check_stdout_kill_survival.
    (void) setvbuf(stdout, nullptr, _IONBF, 0);
    std::string   output_path = "tq-relay-probe-report.json";
    std::string   bench_path;
    std::uint32_t rounds = DEFAULT_ROUNDS;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
            if (output_path.empty()) {
                std::fprintf(stderr, "--output must name a nonempty file\n");
                return 2;
            }
        } else if (arg == "--bench" && i + 1 < argc) {
            bench_path = argv[++i];
            if (bench_path.empty()) {
                std::fprintf(stderr, "--bench must name a nonempty file\n");
                return 2;
            }
        } else if (arg == "--rounds" && i + 1 < argc) {
            const char * text = argv[++i];
            // The README contract is "pure decimal integer": no sign, no
            // leading zero, no surrounding whitespace. strtoul alone would
            // silently accept "+2000", " 2000" and "02000", turning a typo
            // into a run of a different shape than the operator asked for.
            if (text[0] < '1' || text[0] > '9') {
                std::fprintf(stderr, "--rounds must be a positive integer up to %u, got: %s\n",
                             std::numeric_limits<std::uint32_t>::max(), text);
                return 2;
            }
            char * end                 = nullptr;
            errno                      = 0;
            const unsigned long parsed = std::strtoul(text, &end, 10);
            if (errno != 0 || *end != '\0' || parsed == 0 || parsed > std::numeric_limits<std::uint32_t>::max()) {
                std::fprintf(stderr, "--rounds must be a positive integer up to %u, got: %s\n",
                             std::numeric_limits<std::uint32_t>::max(), text);
                return 2;
            }
            rounds = static_cast<std::uint32_t>(parsed);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }
    if (rounds == 0) {
        std::fprintf(stderr, "--rounds must be positive\n");
        return 2;
    }
    if (!bench_path.empty() && output_path == bench_path) {
        std::fprintf(stderr, "--output and --bench must name different files\n");
        return 2;
    }
    try {
        return run_probe(output_path, bench_path, rounds);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "probe failure: %s\n", error.what());
        return 1;
    }
}

int run_probe(const std::string & output_path, const std::string & bench_path, std::uint32_t rounds) {
    probe_report report;
    report.build_id = "tq-relay-probe";
    check_hip(hipRuntimeGetVersion(&report.hip_runtime_version), "runtime version");
    report.hip_runtime_version_string = std::to_string(HIP_VERSION_MAJOR) + "." + std::to_string(HIP_VERSION_MINOR) +
                                        "." + std::to_string(HIP_VERSION_PATCH);
#ifdef __clang__
    report.compiler_version = "amd clang " __clang_version__;
#elif __GNUC__
    report.compiler_version = "gcc " __VERSION__;
#else
    report.compiler_version = "unknown";
#endif
#ifdef __x86_64__
    report.host_arch = "x86_64";
#else
    report.host_arch = "unknown";
#endif
    report.memory_flags_requested =
        "hipHostMallocMapped|hipHostMallocPortable for control and payload; effective flags "
        "recorded per run";
    report.hot_reuse_rounds = rounds;
    const std::string multi_stream_note =
        "multi_consumer, delayed_consumer, two_bank, varying_live, five_stream, same_stream_inverse_dependency, "
        "dual_window_independent_progress, service_loop_independent_progress, nonfinite_reduction_gate, "
        "and five_rank_indexing use one device with concurrent streams; they verify protocol software and five-rank "
        "kernel indexing, not five-device qualification.";
    report.limits = {
        "This is a platform probe, not a model or relay qualification (guide 31.7).",
        "Single-device results do not qualify five-rank cross-device behavior.",
        "Wall-clock ordering observations use the host steady clock.",
        "host_system_rmw is an independent capability; the relay design must not require it.",
        multi_stream_note,
        "delayed_consumer_ordering gates on a >=40 ms device pause, like producer_delay_ordering.",
        "native synchronizations are bounded at 60 s each; native_sync_deadline exercises the timeout path at 2 s on "
        "the live device.",
    };

    int device_count = 0;
    try {
        check_hip(hipGetDeviceCount(&device_count), "device count");
    } catch (const std::exception & error) {
        report.qualification        = outcome::fail;
        report.qualification_detail = std::string("cannot enumerate devices: ") + error.what();
        std::fprintf(stderr, "%s\n", report.qualification_detail.c_str());
        const int written = write_report(output_path, report);
        return written != 0 ? written : 1;
    }
    if (device_count == 0) {
        report.qualification        = outcome::fail;
        report.qualification_detail = "no HIP devices present";
        std::fprintf(stderr, "%s\n", report.qualification_detail.c_str());
        const int written = write_report(output_path, report);
        // Zero devices is a failed qualification, never exit 0, matching
        // the enumeration-failure branch above. (ROCm 7.2.4 errors inside
        // hipGetDeviceCount when nothing is visible, so this branch is
        // unreachable on this machine; it stays consistent for runtimes
        // that report zero instead of erroring.)
        return written != 0 ? written : 1;
    }

    std::vector<device_context> contexts(static_cast<std::size_t>(device_count));
    for (int ordinal = 0; ordinal < device_count; ++ordinal) {
        device_context & ctx = contexts[static_cast<std::size_t>(ordinal)];
        try {
            allocate_device_context(ctx, ordinal, rounds);
        } catch (const std::exception & error) {
            // A device that fails setup still gets a record: the operator
            // needs to see which rank broke and why, not lose the run.
            std::fprintf(stderr, "device %d setup failed: %s\n", ordinal, error.what());
            if (ctx.record.ordinal < 0) {
                ctx.record.ordinal = ordinal;
            }
            if (ctx.record.name.empty()) {
                ctx.record.name = "unavailable";
            }
            set_case(ctx, case_id::device_identity, outcome::fail, std::string("setup failed: ") + error.what());
            set_case(ctx, case_id::mapped_host_access, outcome::not_run, "device setup failed before this case ran");
            for (const case_id id : handshake_cases) {
                set_case(ctx, id, outcome::not_run, "device setup failed before this case ran");
            }
            try {
                free_device_context(ctx);
            } catch (const std::exception & teardown_error) {
                std::fprintf(stderr, "warning: device %d teardown failed: %s\n", ordinal, teardown_error.what());
            }
            continue;
        }
        set_case(ctx, case_id::device_identity, outcome::pass,
                 "mapped_host_memory_supported=" + std::to_string(ctx.record.can_map_host_memory));
        if (!ctx.mapped) {
            set_case(ctx, case_id::mapped_host_access, outcome::unsupported,
                     "hipHostMallocDefault did not return mapped memory (" + ctx.mapped_flags + ")");
            for (const case_id id : handshake_cases) {
                set_case(ctx, id, outcome::unsupported,
                         "handshake requires mapped host memory (" + ctx.mapped_flags + ")");
            }
            std::printf("rank %d: mapped host memory unavailable (%s)\n", ordinal, ctx.mapped_flags.c_str());
            continue;
        }
        run_case(ctx, case_id::mapped_host_access, case_mapped_access);
        if (result_of(ctx, case_id::mapped_host_access) == outcome::pass) {
            run_case(ctx, case_id::cpu_to_gpu_publish, case_cpu_to_gpu);
            run_case(ctx, case_id::gpu_to_cpu_publish, case_gpu_to_cpu);
            run_case(ctx, case_id::producer_delay_ordering, case_producer_delay);
            run_case(ctx, case_id::ticket_semantics, case_ticket_semantics);
            run_case(ctx, case_id::multi_group_completion, case_multi_group);
            run_case(ctx, case_id::multi_group_ordering, case_multi_group_ordering);
            run_case(ctx, case_id::abort_stop, case_abort);
            // One launch produces both capability bits (guide 31.3).
            run_case(ctx, case_id::host_system_rmw_u64_add, case_host_rmw);
            run_case(ctx, case_id::multi_consumer_output_credit, case_multi_consumer);
            run_case(ctx, case_id::delayed_consumer_ordering, case_delayed_consumer);
            run_case(ctx, case_id::two_bank_rotation, case_two_bank);
            run_case(ctx, case_id::varying_live_bytes, case_varying_live);
            run_case(ctx, case_id::offset_addressing, case_offset_addressing);
            run_case(ctx, case_id::five_stream_relay_cycle, case_five_stream);
            run_case(ctx, case_id::native_sync_deadline, case_native_sync_deadline);
            run_case(ctx, case_id::same_stream_inverse_dependency, case_same_stream_inverse);
            run_case(ctx, case_id::dual_window_independent_progress, case_dual_window);
            run_case(ctx, case_id::service_loop_independent_progress, case_service_loop);
            run_case(ctx, case_id::nonfinite_reduction_gate, case_fold_gate);
            run_case(ctx, case_id::host_wait_deadline_no_reuse, case_host_wait_deadline);
            run_case(ctx, case_id::five_rank_indexing, case_five_rank_indexing);
        }
        std::printf("rank %d (%s, %s): ", ordinal, ctx.record.gcn_arch.c_str(), ctx.record.name.c_str());
        for (const auto & item : ctx.record.cases) {
            std::printf("%s=%s ", tq::relay::probe::case_name(item.id), tq::relay::probe::to_string(item.result));
        }
        std::printf("\n");
    }

    bool all_pass = true;
    for (auto & ctx : contexts) {
        report.devices.push_back(ctx.record);
        for (const auto & item : ctx.record.cases) {
            if (item.id != case_id::host_system_rmw_u64_add && item.id != case_id::host_system_rmw_u64_cas &&
                item.result != outcome::pass) {
                all_pass = false;
            }
        }
    }
    if (device_count == 5) {
        report.five_rank_status = run_five_rank(contexts, report, rounds);
        if (report.five_rank_status != outcome::pass) {
            all_pass = false;
        }
    } else {
        report.five_rank_status = outcome::not_run;
        report.five_rank_detail = "requires exactly five devices; " + std::to_string(device_count) + " present";
    }
    int bench_result = 0;
    if (!bench_path.empty()) {
        if (contexts[0].cb == nullptr) {
            std::fprintf(stderr, "bench skipped: device 0 was not set up\n");
            bench_result = 1;
        } else if (!contexts[0].mapped) {
            std::fprintf(stderr, "bench skipped: device 0 cannot map host memory\n");
            bench_result = 1;
        } else {
            try {
                bench_result = run_bench(contexts[0], bench_path, rounds, report.hip_runtime_version_string);
            } catch (const std::exception & error) {
                std::fprintf(stderr, "bench failure: %s\n", error.what());
                bench_result = 1;
            }
        }
    }
    for (auto & ctx : contexts) {
        try {
            free_device_context(ctx);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "warning: device %d teardown failed: %s\n", ctx.record.ordinal, error.what());
        }
    }
    report.qualification        = all_pass ? outcome::pass : outcome::fail;
    report.qualification_detail = all_pass ? "all mandatory single-device cases passed on every present device" :
                                             "at least one mandatory case failed or was unavailable";

    const int write_result = write_report(output_path, report);
    std::printf("qualification=%s report=%s\n", tq::relay::probe::to_string(report.qualification), output_path.c_str());
    if (write_result != 0) {
        return write_result;
    }
    if (bench_result != 0) {
        return bench_result;
    }
    return report.qualification == outcome::pass ? 0 : 1;
}
