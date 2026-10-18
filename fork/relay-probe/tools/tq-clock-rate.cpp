// tq-clock-rate: measure the clock64() tick rate of every visible HIP
// device. The relay probe's device-side spin budgets (WAIT_BUDGET_TICKS and
// the per-case spin amounts) are calibrated on a 2026-10-19 RX 6800 at
// ~2.3 GHz; this tool re-derives that number on any machine so the operator
// can check the calibration before trusting the bounded-wait margins. It is
// a diagnostic: it writes no evidence files, has no rounds parameter, and
// its output is only a per-device rate table.
//
// Method: one kernel busy-spins a fixed tick budget between two clock64()
// reads and writes the exact tick count it spent to pinned host memory; the
// host times the wall duration with the steady clock and divides. Each pass
// is a separate launch, and the host waits with an event poll plus a wall
// deadline (SPEC 7: a hung queue must fail boundedly, never hang the tool),
// matching the probe's bounded-sync discipline.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t TICKS_PER_PASS = 1000000000ULL;  // ~0.43 s at 2.3 GHz
constexpr int           PASSES         = 5;
constexpr double        DEADLINE_MS    = 10000.0;        // far above the spin, far below forever

__global__ void spin_then_write(std::uint64_t ticks, std::uint64_t * out) {
    const std::uint64_t begin = clock64();
    while (clock64() - begin < ticks) {
        __builtin_amdgcn_s_sleep(2);
    }
    *out = clock64() - begin;
}

// Waits for the stream with a wall deadline instead of a bare synchronize:
// on a hung device this reports a bounded failure for the device and moves
// on. Returns false on timeout or HIP error.
bool wait_bounded(hipEvent_t event) {
    const auto       begin  = std::chrono::steady_clock::now();
    const hipError_t record = hipEventRecord(event, nullptr);
    if (record != hipSuccess) {
        std::fprintf(stderr, "  event record failed: %s\n", hipGetErrorString(record));
        return false;
    }
    for (;;) {
        const hipError_t status = hipEventQuery(event);
        if (status == hipSuccess) {
            return true;
        }
        if (status != hipErrorNotReady) {
            std::fprintf(stderr, "  event query failed: %s\n", hipGetErrorString(status));
            return false;
        }
        const auto elapsed =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        if (elapsed > DEADLINE_MS) {
            std::fprintf(stderr, "  bounded wait expired after %.0f ms\n", DEADLINE_MS);
            return false;
        }
    }
}

// The kernel publishes its spent tick count through this word; the host
// reads it only after the stream event completes, and the acquire keeps the
// read ordered exactly like the probe's own control words (LEMMA 1).
std::uint64_t acquire_load(const std::uint64_t * word) {
    return __atomic_load_n(word, __ATOMIC_ACQUIRE);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) {
        std::printf("usage: %s\n", argv[0]);
        return 0;
    }
    if (argc > 1) {
        std::fprintf(stderr, "usage: %s\n", argv[0]);
        return 2;
    }
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count <= 0) {
        std::fprintf(stderr, "no HIP device visible\n");
        return 2;
    }
    std::printf("clock64 tick rate per device (%d passes of %llu ticks each)\n", PASSES,
                static_cast<unsigned long long>(TICKS_PER_PASS));
    int failures = 0;
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        if (hipSetDevice(ordinal) != hipSuccess) {
            std::fprintf(stderr, "device %d: hipSetDevice failed\n", ordinal);
            ++failures;
            continue;
        }
        hipDeviceProp_t props{};
        if (hipGetDeviceProperties(&props, ordinal) != hipSuccess) {
            std::fprintf(stderr, "device %d: properties failed\n", ordinal);
            ++failures;
            continue;
        }
        std::uint64_t * out = nullptr;
        // The device writes this word, so it must be mapped host memory by
        // contract. hipHostMallocDefault is pinned but not device-mappable;
        // this machine happens to let the write land anyway (flags=0), which
        // is exactly the platform-accident class the probe exists to refuse
        // (AXIOM 1). Request mapped memory and verify the flag per device.
        if (hipHostMalloc(&out, sizeof(std::uint64_t), hipHostMallocMapped | hipHostMallocPortable) != hipSuccess) {
            std::fprintf(stderr, "device %d (%s): host allocation failed\n", ordinal, props.name);
            ++failures;
            continue;
        }
        unsigned int flags = 0;
        if (hipHostGetFlags(&flags, out) != hipSuccess || (flags & hipHostMallocMapped) == 0) {
            std::fprintf(stderr, "device %d (%s): host allocation is not device-mapped (flags 0x%x)\n", ordinal,
                         props.name, flags);
            (void) hipHostFree(out);
            ++failures;
            continue;
        }
        hipEvent_t event = nullptr;
        if (hipEventCreate(&event) != hipSuccess) {
            std::fprintf(stderr, "device %d (%s): event creation failed\n", ordinal, props.name);
            (void) hipHostFree(out);
            ++failures;
            continue;
        }
        std::vector<double> rates;
        rates.reserve(PASSES);
        for (int pass = 0; pass < PASSES; ++pass) {
            __atomic_store_n(out, 0, __ATOMIC_RELAXED);
            const auto begin = std::chrono::steady_clock::now();
            spin_then_write<<<1, 64>>>(TICKS_PER_PASS, out);
            const bool   done = wait_bounded(event);
            const double wall_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            if (!done) {
                std::fprintf(stderr, "device %d (%s): pass %d did not finish in time\n", ordinal, props.name, pass);
                ++failures;
                rates.clear();
                break;
            }
            const std::uint64_t ticks = acquire_load(out);
            if (ticks < TICKS_PER_PASS) {
                std::fprintf(stderr, "device %d (%s): kernel reported fewer ticks than requested\n", ordinal,
                             props.name);
                ++failures;
                rates.clear();
                break;
            }
            rates.push_back(wall_ms > 0.0 ? static_cast<double>(ticks) / wall_ms / 1000.0 : 0.0);
        }
        (void) hipEventDestroy(event);
        (void) hipHostFree(out);
        if (!rates.empty()) {
            std::vector<double> sorted = rates;
            std::sort(sorted.begin(), sorted.end());
            std::printf("device %d: %s (%s): min %.1f MHz, median %.1f MHz, max %.1f MHz\n", ordinal, props.name,
                        props.gcnArchName, sorted.front(), sorted[sorted.size() / 2], sorted.back());
        }
    }
    return failures == 0 ? 0 : 1;
}
