#include "common.cuh"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cuda.h"
#include "relay.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <mutex>
#include <new>
#include <system_error>
#include <type_traits>
#include <vector>

namespace {
constexpr uint32_t ranks = GGML_RELAY_RANKS;
// Strict graph submission, relay kernels and replay use the owned stream zero.
// Keep native-tail evidence for that actual lane only; HIP hardware queue
// assignment remains a runtime property, not a promise made by this count.
constexpr uint32_t lanes_per_rank = 1;
static_assert(std::atomic<int32_t>::is_always_lock_free, "relay abort must never acquire a hidden host lock");
constexpr uint32_t all_ranks        = (1u << ranks) - 1;
// Swift needs 36 recurrent-output, 48 expert-output and 97 HC joins. Reserve a
// fixed bound before submission; unused slots never become active transfers.
constexpr size_t   channel_capacity = 256;
// Isolated real MTP verifier T4 capacity: the original worst HC residual
// channel is 10240 F32 per token, so T4 needs 40960 scalars. Keep the
// existing strict bounds, five-rank host-mapped stride and producer/consumer
// wire checks, now with enough reserved capacity for 4 genuine target rows.
// Original selected source/runtime remain byte-identical elsewhere.
constexpr size_t   element_capacity = 16 * 4096;
// The gfx1030 shader cycle counter can move backwards during this wait.
// Use the constant-rate steady counter; the host deadline aborts first.
constexpr uint64_t wait_seconds = 600;

// Optional two-provider experiment. A lease names a simultaneously live
// provider, not a monotonically increasing creation ordinal. Retained unsafe
// owners keep their lease because only provider deletion releases it.
std::atomic<uint32_t> priority_pair_slots{ 0 };
static_assert(std::atomic<uint32_t>::is_always_lock_free, "relay priority leases require a lock-free host atomic");

int priority_pair_mode() {
    // Freeze once, before the first provider. Changing the process environment
    // later cannot bypass the two-provider bound or relabel existing streams.
    static const int mode = [] {
        const char * value = std::getenv("GGML_TP5_RELAY_PRIORITY_PAIR");
        if (!value || !value[0]) {
            return 0;
        }
        return std::strcmp(value, "0,-1") == 0 ? 1 : -1;
    }();
    return mode;
}

int acquire_priority_pair_slot() noexcept {
    uint32_t used = priority_pair_slots.load(std::memory_order_relaxed);
    for (;;) {
        if ((used & 3u) == 3u) {
            return -1;
        }
        const int slot = (used & 1u) == 0 ? 0 : 1;
        const uint32_t desired = used | (1u << slot);
        if (priority_pair_slots.compare_exchange_weak(used, desired, std::memory_order_acq_rel,
                                                       std::memory_order_relaxed)) {
            return slot;
        }
    }
}

// Single-writer words only. These are the probe's qualified acquire/release
// operations; no host-memory fetch_add/CAS is needed on this platform.
struct alignas(64) word {
    uint64_t value;
};

struct control {
    word generation;
    word ready[ranks];
    word ack[ranks];
    word error[ranks];
};

static_assert(element_capacity <= (SIZE_MAX - sizeof(control)) / (sizeof(float) * (ranks + 1)),
              "relay channel byte capacity overflows size_t");
constexpr size_t channel_stride = sizeof(control) + element_capacity * sizeof(float) * (ranks + 1);
static_assert(channel_stride % alignof(control) == 0, "relay controls must retain cache-line alignment");
static_assert(channel_capacity <= SIZE_MAX / channel_stride, "relay pool byte capacity overflows size_t");
constexpr size_t pool_bytes = channel_capacity * channel_stride;

__host__ __device__ uint64_t load(const word * w) {
    return __atomic_load_n(&w->value, __ATOMIC_ACQUIRE);
}

__device__ uint64_t poll_word(const word * w) {
    return __atomic_load_n(&w->value, __ATOMIC_RELAXED);
}

__host__ __device__ void store(word * w, uint64_t value) {
    __atomic_store_n(&w->value, value, __ATOMIC_RELEASE);
}

// ASD AXIOM: the wire representation changes payload bytes only. Every
// producer fences its complete payload before its real ready; every consumer
// acquires generation and finishes all payload reads before its real ack.
static_assert(sizeof(half) == 2 && sizeof(half2) == 4, "relay F16 wire requires IEEE binary16 storage");

template<typename Wire>
__device__ __forceinline__ void store_wire4(Wire * dst, const float4 value) {
    if constexpr (std::is_same<Wire, float>::value) {
        __builtin_memcpy(dst, &value, sizeof(value));
    } else {
        static_assert(std::is_same<Wire, half>::value, "unsupported relay wire");
        const half2 lo = __floats2half2_rn(value.x, value.y);
        const half2 hi = __floats2half2_rn(value.z, value.w);
        uint2 packed;
        __builtin_memcpy(&packed.x, &lo, sizeof(lo));
        __builtin_memcpy(&packed.y, &hi, sizeof(hi));
        __builtin_memcpy(dst, &packed, sizeof(packed));
    }
}

template<typename Wire>
__device__ __forceinline__ float4 load_wire4(const Wire * src) {
    if constexpr (std::is_same<Wire, float>::value) {
        float4 value;
        __builtin_memcpy(&value, src, sizeof(value));
        return value;
    } else {
        static_assert(std::is_same<Wire, half>::value, "unsupported relay wire");
        uint2 packed;
        __builtin_memcpy(&packed, src, sizeof(packed));
        half2 lo, hi;
        __builtin_memcpy(&lo, &packed.x, sizeof(lo));
        __builtin_memcpy(&hi, &packed.y, sizeof(hi));
        const float2 a = __half22float2(lo);
        const float2 b = __half22float2(hi);
        return make_float4(a.x, a.y, b.x, b.y);
    }
}

template<typename Wire>
__device__ __forceinline__ Wire store_wire1(float value) {
    if constexpr (std::is_same<Wire, float>::value) {
        return value;
    } else {
        return __float2half_rn(value);
    }
}

template<typename Wire>
__device__ __forceinline__ float load_wire1(Wire value) {
    if constexpr (std::is_same<Wire, float>::value) {
        return value;
    } else {
        return __half2float(value);
    }
}

template<typename Wire>
__global__ void publish_input(Wire * inbox, const float * terminal, size_t n,
                              control * c, uint32_t rank, const word * activation) {
    // One bounded workgroup owns the whole segment. Every writer completes
    // its system fence before the group permits thread zero to publish.
    const size_t vectors = n / 4;
    for (size_t i = threadIdx.x; i < vectors; i += blockDim.x) {
        float4 value;
        __builtin_memcpy(&value, terminal + 4 * i, sizeof(value));
        store_wire4(inbox + 4 * i, value);
    }
    for (size_t i = 4 * vectors + threadIdx.x; i < n; i += blockDim.x) {
        inbox[i] = store_wire1<Wire>(terminal[i]);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        store(&c->ready[rank], load(activation));
    }
}

template<typename Wire>
__global__ void consume_output(float *       terminal,
                            const Wire * output,
                            size_t        n,
                            control *     c,
                            word *        abort_word,
                            uint32_t      rank,
                            const word *  activation,
                            uint64_t      wait_ticks,
                            bool          publish_zero_input) {
    __shared__ uint64_t epoch;
    __shared__ bool available;
    // prepare() owns the zero payload. This workgroup publishes this rank's
    // real stream arrival before it waits for the CPU output.
    if (publish_zero_input) {
        __threadfence_system();
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        epoch = load(activation);
        if (publish_zero_input) {
            store(&c->ready[rank], epoch);
        }
        available = false;
        const uint64_t start = wall_clock64();
        for (;;) {
            if (poll_word(abort_word)) {
                store(&c->error[rank], 1);
                break;
            }
            // Polling grants no payload read permission. Every participant
            // still performs the system acquire below before reading output.
            const uint64_t generation = poll_word(&c->generation);
            if (generation == epoch) {
                available = true;
                break;
            }
            if (generation > epoch) {
                store(&c->error[rank], 2);
                break;
            }
            const uint64_t now = wall_clock64();
            if (now - start >= wait_ticks) {
                store(&c->error[rank], 3);
                break;
            }
            // ASYNC-PROTOCOL: pure busy-wait, no wave sleep.
        }
    }
    __syncthreads();
    if (!available) {
        return;
    }
    // Each payload reader performs its own system acquire. A group vote keeps
    // failure uniform, so no thread exits ahead of the final reader barrier.
    const bool valid = load(&c->generation) == epoch && !load(abort_word) && !load(&c->error[rank]);
    if (!__syncthreads_and(valid)) {
        if (threadIdx.x == 0 && !load(&c->error[rank])) {
            store(&c->error[rank], load(abort_word) ? 1 : 2);
        }
        return;
    }
    // memcpy permits wide transfers without inventing 16-byte alignment for
    // terminal views. Scalar tails still cover every valid element exactly once.
    const size_t vectors = n / 4;
    for (size_t i = threadIdx.x; i < vectors; i += blockDim.x) {
        const float4 value = load_wire4(output + 4 * i);
        __builtin_memcpy(terminal + 4 * i, &value, sizeof(value));
    }
    for (size_t i = 4 * vectors + threadIdx.x; i < n; i += blockDim.x) {
        terminal[i] = load_wire1(output[i]);
    }
    __threadfence_system();
    __syncthreads();
    // All readers have finished. An abort is not a normal consumption ack.
    if (threadIdx.x == 0 && !load(abort_word) && !load(&c->error[rank])) {
        if (load(&c->generation) == epoch) {
            store(&c->ack[rank], epoch);
        } else {
            store(&c->error[rank], 2);
        }
    }
}

struct channel {
    control *     host_control                   = nullptr;
    control *     device_control[ranks]          = {};
    void *        inbox[ranks]                   = {};
    void *        device_inbox[ranks]            = {};
    void *        output                         = nullptr;
    void *        device_output[ranks]           = {};
    ggml_tensor * terminal[ranks]                = {};
    void *        bound_data[ranks]              = {};
    int64_t       bound_ne[ranks][GGML_MAX_DIMS] = {};
    size_t        bound_nb[ranks][GGML_MAX_DIMS] = {};
    size_t        elements                       = 0;
    size_t        bytes                          = 0; // Effective bytes of one wire payload.
    uint32_t      computed_mask                  = 0; // Prepared COMPUTE flags, never ready/ack.
    bool          enqueued                       = false;
    bool          offered                        = false;
    bool          published                      = false;
    bool          released                       = false;
    uint32_t      ready_seen                     = 0;
    uint32_t      ack_seen                       = 0;
    uint32_t      zero_inbox_mask                = 0; // Provider storage, not a frame/terminal property.
};

struct lane {
    hipStream_t stream   = nullptr;
    hipEvent_t  tail     = nullptr;
    bool        recorded = false;
    bool        complete = false;
};

// The ordinary legacy pool can hipMalloc/hipDeviceSynchronize in a later
// layer, after a previously queued relay wait. Strict mode instead reserves
// bounded scratch before begin and reports exhaustion without native waits.
struct relay_pool final : ggml_cuda_pool {
    char * base;
    size_t capacity;
    size_t used = 0;

    relay_pool(void * ptr, size_t size) : base(static_cast<char *>(ptr)), capacity(size) {}

    void * alloc(size_t size, size_t * actual) override {
        if (size > SIZE_MAX - 255) {
            throw std::bad_alloc();
        }
        size = (size + 255) & ~size_t(255);
        if (size > capacity - used) {
            throw std::bad_alloc();
        }
        void * ptr = base + used;
        used += size;
        *actual = size;
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        GGML_ASSERT(size <= used && ptr == base + used - size);
        used -= size;
    }
};

struct rank_state {
    ggml_backend_t              backend  = nullptr;
    ggml_backend_cuda_context * context  = nullptr;
    int                         physical = -1;
    uint64_t                    wait_ticks = 0;
    void *                      scratch  = nullptr;
    lane                        lanes[lanes_per_rank];
    bool                        graph_pending   = false;
    uint64_t                    pending_nodes   = 0;
    uint64_t                    accepted_graphs = 0;
    bool                        capturing       = false;
};

// Two private activation arenas alternate in the caller. Own at most their
// two exact bindings; no backend-global cache or generic graph machinery.
struct frame {
    std::vector<unsigned char> signature;
    hipGraph_t                graph[ranks] = {};
    hipGraphExec_t            exec[ranks]  = {};
    uint64_t                  compute_nodes[ranks] = {};
    uint64_t                  relay_nodes[ranks] = {};
    uint64_t                  last_used = 0;
    bool                      warmed = false;
};

enum class submission { direct, capture, replay };

struct provider {
    int                                   priority_pair_slot = -1;

    ~provider() noexcept {
        if (priority_pair_slot >= 0) {
            priority_pair_slots.fetch_and(~(1u << priority_pair_slot), std::memory_order_release);
        }
    }

    std::mutex                            mutex;
    // Ordinary CPU memory, never mapped to a device or used for GPU RMW.
    std::atomic<int32_t>                  pending_abort{ 0 };
    // Idle event workers must not contend with the next submit snapshot.
    // 0: no pending events; 1: sealed epoch; -1: failure.
    std::atomic<int>                      event_status{ 0 };
    std::atomic<ggml_relay_wire_type>      wire_type{ GGML_RELAY_WIRE_F32 };
    bool                                  wire_configured = false;
    bool                                  wire_frozen = false;
    size_t                                outputs_released = 0;
    // Lower bounds only: advance over actual contiguous event facts, never
    // over observed ready/ack bits or the possibly noncontiguous event count.
    size_t                                release_scan_begin = 0;
    size_t                                offer_scan_begin = 0;
    size_t                                ready_scan_end = 0;
    size_t                                release_scan_end = 0;
    provider *                            retained_next = nullptr;
    rank_state                            rank[ranks];
    std::array<channel, channel_capacity> channels;
    void *                                channel_pool        = nullptr;
    word *                                abort_host          = nullptr;
    word *                                abort_device[ranks] = {};
    word *                                activation_host = nullptr;
    word *                                activation_device[ranks] = {};
    std::array<frame, 2>                   frames;
    std::vector<unsigned char>            signature;
    std::vector<const ggml_tensor *>       tensors;
    std::vector<ggml_tensor *>             tensor_keys;
    std::vector<ggml_bitset_t>             tensor_used;
    std::vector<uint64_t>                  stage_nodes;
    frame *                               current = nullptr;
    bool                                  definition_changed = true;
    submission                            mode = submission::direct;
    ggml_relay_status                     status              = {};
    bool                                  active              = false;
    bool                                  tails_attempted     = false;
    size_t                                next_channel        = 0;
    size_t                                borrowed            = std::numeric_limits<size_t>::max();
};

// Upper bounds cover only candidates admitted by the original poll predicates.
// Update both observed facts and the flag transitions that can expose them;
// an earlier unpublished hole need not be revisited by hot observation.
void include_ready_candidate(provider & p, size_t index) noexcept {
    const auto & c = p.channels[index];
    if (c.enqueued && !c.offered && c.ready_seen == all_ranks) {
        p.ready_scan_end = std::max(p.ready_scan_end, index + 1);
    }
}

void include_release_candidate(provider & p, size_t index) noexcept {
    const auto & c = p.channels[index];
    if (c.published && !c.released && c.ack_seen == all_ranks) {
        p.release_scan_end = std::max(p.release_scan_end, index + 1);
    }
}

// A failed initialization/free must not lose the final native owner. There is
// deliberately no static destructor: unsafe owners survive until process exit.
std::mutex owners_mutex;
provider * retained = nullptr;

void retain(provider * p) {
    std::lock_guard<std::mutex> lock(owners_mutex);
    p->retained_next = retained;
    retained         = p;
}

void fail(provider & p, int code, const char * message) {
    p.event_status.store(-1, std::memory_order_release);
    if (!p.status.failed) {
        const int32_t aborted = p.pending_abort.load(std::memory_order_acquire);
        p.status.failed       = true;
        p.status.error_code   = aborted ? aborted : (code ? code : EIO);
        snprintf(p.status.error, sizeof(p.status.error), "%s", aborted ? "relay aborted by host" : message);
    }
    if (p.abort_host) {
        store(p.abort_host, 1);
    }
}

void consume_abort(provider & p) {
    const int32_t error = p.pending_abort.load(std::memory_order_acquire);
    if (error) {
        fail(p, error, "relay aborted by host");
    }
}

bool checked(provider & p, hipError_t error, const char * operation) {
    if (error == hipSuccess) {
        return true;
    }
    char text[256];
    snprintf(text, sizeof(text), "%s: %s", operation, hipGetErrorString(error));
    fail(p, int(error), text);
    return false;
}

bool priority_stream_matches(provider & p, hipStream_t stream, int expected_priority) {
    unsigned int flags = 0;
    int priority = 0;
    if (!checked(p, hipStreamGetFlags(stream, &flags), "hipStreamGetFlags strict priority") ||
        !checked(p, hipStreamGetPriority(stream, &priority), "hipStreamGetPriority strict priority")) {
        return false;
    }
    if (flags != hipStreamNonBlocking || priority != expected_priority) {
        char detail[256];
        snprintf(detail, sizeof(detail),
                 "strict priority stream mismatch: expected priority=%d flags=%u, actual priority=%d flags=%u",
                 expected_priority, unsigned(hipStreamNonBlocking), priority, flags);
        fail(p, EINVAL, detail);
        return false;
    }
    return true;
}

bool select(provider & p, uint32_t rank) {
    return checked(p, hipSetDevice(p.rank[rank].physical), "hipSetDevice");
}

bool clear_frame(provider & p, frame & f) {
    bool ok = true;
    for (uint32_t r = 0; r < ranks; ++r) {
        if (!f.exec[r] && !f.graph[r]) {
            continue;
        }
        if (!select(p, r)) {
            ok = false;
            continue;
        }
        if (f.exec[r]) {
            if (!checked(p, hipGraphExecDestroy(f.exec[r]), "hipGraphExecDestroy relay")) {
                ok = false;
                continue;
            }
            f.exec[r] = nullptr;
        }
        if (f.graph[r]) {
            if (!checked(p, hipGraphDestroy(f.graph[r]), "hipGraphDestroy relay")) {
                ok = false;
                continue;
            }
            f.graph[r] = nullptr;
        }
    }
    if (ok) {
        f.warmed = false;
        f.signature.clear();
        std::fill(f.relay_nodes, f.relay_nodes + ranks, 0);
    }
    return ok;
}

bool free_channels(provider & p) {
    if (p.channel_pool && !checked(p, hipHostFree(p.channel_pool), "hipHostFree channel pool")) {
        return false;
    }
    p.channel_pool = nullptr;
    for (auto & c : p.channels) {
        c.host_control = nullptr;
    }
    return true;
}

bool release_owners(provider & p) {
    // Executables retain kernel arguments into every provider allocation.
    // Destroy them only after native/CPU drain and before freeing any storage.
    for (const auto & r : p.rank) {
        if (r.capturing) {
            fail(p, EBUSY, "relay stream capture termination is unproved; retaining owners");
            return false;
        }
    }
    for (auto & f : p.frames) {
        if (!clear_frame(p, f)) {
            return false;
        }
    }
    if (!free_channels(p)) {
        return false;
    }
    if (p.activation_host) {
        if (!checked(p, hipHostFree(p.activation_host), "hipHostFree activation")) {
            return false;
        }
        p.activation_host = nullptr;
    }
    if (p.abort_host) {
        if (!checked(p, hipHostFree(p.abort_host), "hipHostFree abort")) {
            return false;
        }
        p.abort_host = nullptr;
    }
    bool ok = true;
    for (uint32_t r = 0; r < ranks; ++r) {
        if (p.rank[r].physical < 0) {
            continue;
        }
        if (!select(p, r)) {
            ok = false;
            continue;
        }
        if (p.rank[r].scratch) {
            auto * ctx = p.rank[r].context;
            ctx->pools[ctx->device][0].reset();
            if (checked(p, hipFree(p.rank[r].scratch), "hipFree strict scratch")) {
                p.rank[r].scratch = nullptr;
            } else {
                ok = false;
            }
        }
        for (auto & l : p.rank[r].lanes) {
            if (l.tail && checked(p, hipEventDestroy(l.tail), "hipEventDestroy tail")) {
                l.tail = nullptr;
            } else if (l.tail) {
                ok = false;
            }
        }
    }
    if (ok) {
        for (auto & r : p.rank) {
            if (r.context && r.context->tp5_relay_owner == &p) {
                r.context->tp5_relay_owner = nullptr;
            }
        }
    }
    return ok;
}

// Only native nonblocking queries; never wait for GPU completion while holding
// the state mutex. Every initialized child lane gets a tail, including partial
// graph submissions whose return status was unknown.
void observe(provider & p, bool query_native = true) {
    consume_abort(p);
    if (p.status.epoch == 0) {
        return;
    }
    // Snapshots and failure retain the complete scan. Every skipped hot-path
    // channel is already released, so the original loop would skip it too.
    const size_t first = !query_native && !p.status.failed ? p.release_scan_begin : 0;
    for (size_t i = first; i < p.status.channels; ++i) {
        auto & c = p.channels[i];
        if (!c.host_control || c.released) {
            continue;
        }
        for (uint32_t r = 0; r < ranks; ++r) {
            const uint32_t bit = 1u << r;
            // The ack is the consumer's final write for this epoch. begin()
            // resets these observed facts only after every native tail ends.
            if (c.ack_seen & bit) {
                continue;
            }
            const uint64_t error = load(&c.host_control->error[r]);
            if (error) {
                char detail[sizeof(p.status.error)];
                std::snprintf(detail, sizeof(detail), "HIP relay wait failed channel=%zu rank=%u epoch=%llu device_code=%llu",
                              i, r, static_cast<unsigned long long>(p.status.epoch),
                              static_cast<unsigned long long>(error));
                fail(p, ETIMEDOUT, detail);
            }
            const uint64_t ready = (c.ready_seen & bit) ? p.status.epoch : load(&c.host_control->ready[r]);
            const uint64_t ack   = load(&c.host_control->ack[r]);
            if (ready > p.status.epoch || ack > p.status.epoch) {
                fail(p, EPROTO, "HIP relay observed a future ready or last-reader epoch");
            }
            if (!(c.ready_seen & bit) && ready == p.status.epoch) {
                c.ready_seen |= bit;
                p.status.rank_published_bytes[r] += c.bytes;
            }
            if (!(c.ack_seen & bit) && ack == p.status.epoch) {
                c.ack_seen |= bit;
                p.status.rank_consumed_bytes[r] += c.bytes;
            }
        }
        include_ready_candidate(p, i);
        include_release_candidate(p, i);
        // A rank cannot reach the next producer before consuming this output
        // on the same owned stream. Hot polling needs only this first
        // unpublished channel plus earlier readers; snapshots/failure scan all.
        if (!query_native && !p.status.failed && !c.published) {
            break;
        }
    }
    if (!query_native || !p.tails_attempted) {
        return;
    }
    uint32_t complete_mask = 0;
    for (uint32_t r = 0; r < ranks; ++r) {
        if (!(p.status.attempted_mask & (1u << r))) {
            continue;
        }
        if (p.status.completed_mask & (1u << r)) {
            complete_mask |= 1u << r;
            continue;
        }
        bool all = true;
        if (!select(p, r)) {
            continue;
        }
        for (auto & l : p.rank[r].lanes) {
            if (!l.recorded) {
                all = false;
                continue;
            }
            if (!l.complete) {
                const hipError_t result = hipEventQuery(l.tail);
                if (result == hipSuccess) {
                    l.complete = true;
                } else if (result != hipErrorNotReady) {
                    checked(p, result, "hipEventQuery tail");
                }
            }
            all = all && l.complete;
        }
        if (all) {
            complete_mask |= 1u << r;
        }
    }
    p.status.completed_mask = complete_mask;
    p.status.unknown_mask &= ~complete_mask;
    p.status.finished = (complete_mask & p.status.attempted_mask) == p.status.attempted_mask &&
                        std::none_of(std::begin(p.rank), std::end(p.rank),
                                     [](const rank_state & r) { return r.capturing; });
}

bool idle(provider & p) {
    observe(p);
    if (p.borrowed != std::numeric_limits<size_t>::max() || !p.status.finished) {
        return false;
    }
    if (p.status.epoch && !p.status.failed) {
        for (size_t i = 0; i < p.status.channels; ++i) {
            const auto & c = p.channels[i];
            if (!c.released) {
                return false;
            }
        }
    }
    return true;
}

bool same_binding(const channel & c, uint32_t rank, const ggml_tensor * tensor) {
    return c.terminal[rank] == tensor && c.bound_data[rank] == tensor->data &&
           memcmp(c.bound_ne[rank], tensor->ne, sizeof(tensor->ne)) == 0 &&
           memcmp(c.bound_nb[rank], tensor->nb, sizeof(tensor->nb)) == 0;
}

void bind(channel & c, uint32_t rank, ggml_tensor * tensor) {
    c.terminal[rank]   = tensor;
    c.bound_data[rank] = tensor->data;
    memcpy(c.bound_ne[rank], tensor->ne, sizeof(tensor->ne));
    memcpy(c.bound_nb[rank], tensor->nb, sizeof(tensor->nb));
}

bool reserve_channels(provider & p) {
    if (!select(p, 0) ||
        !checked(p, hipHostMalloc(&p.channel_pool, pool_bytes, hipHostMallocMapped | hipHostMallocPortable),
                 "hipHostMalloc channel pool")) {
        return false;
    }
    unsigned flags = 0;
    if (!checked(p, hipHostGetFlags(&flags, p.channel_pool), "hipHostGetFlags channel pool") ||
        !(flags & hipHostMallocMapped)) {
        fail(p, ENOTSUP, "relay channel pool is not mapped");
        return false;
    }
    for (size_t i = 0; i < channel_capacity; ++i) {
        auto & c       = p.channels[i];
        auto * host    = static_cast<char *>(p.channel_pool) + i * channel_stride;
        c.host_control = reinterpret_cast<control *>(host);
        memset(c.host_control, 0, sizeof(control));
        auto * payload = host + sizeof(control);
        for (uint32_t r = 0; r < ranks; ++r) {
            c.inbox[r] = payload + element_capacity * sizeof(float) * r;
        }
        c.output   = payload + element_capacity * sizeof(float) * ranks;
        c.released = true;
    }
    for (uint32_t r = 0; r < ranks; ++r) {
        void * mapped = nullptr;
        if (!select(p, r) ||
            !checked(p, hipHostGetDevicePointer(&mapped, p.channel_pool, 0), "hipHostGetDevicePointer channel pool")) {
            return false;
        }
        for (size_t i = 0; i < channel_capacity; ++i) {
            auto & c            = p.channels[i];
            auto * device       = static_cast<char *>(mapped) + i * channel_stride;
            c.device_control[r] = reinterpret_cast<control *>(device);
            auto * payload      = device + sizeof(control);
            c.device_inbox[r]   = payload + element_capacity * sizeof(float) * r;
            c.device_output[r]  = payload + element_capacity * sizeof(float) * ranks;
        }
    }
    return true;
}

void * create(ggml_backend_t * backends, size_t count) {
#if defined(GGML_HIP_GRAPHS)
    GGML_UNUSED(backends);
    GGML_UNUSED(count);
    return nullptr;
#else
    if (!backends || count != ranks) {
        return nullptr;
    }
    const int priority_mode = priority_pair_mode();
    if (priority_mode < 0) {
        GGML_LOG_ERROR("HIP relay priority pair requires exact GGML_TP5_RELAY_PRIORITY_PAIR=0,-1, or unset/empty to disable\n");
        return nullptr;
    }
    auto * p = new (std::nothrow) provider;
    if (!p) {
        return nullptr;
    }
    p->status.abi_version         = GGML_RELAY_ABI_VERSION;
    p->status.ranks               = ranks;
    p->status.finished            = true;
    if (priority_mode != 0) {
        p->priority_pair_slot = acquire_priority_pair_slot();
        if (p->priority_pair_slot < 0) {
            GGML_LOG_ERROR("HIP relay priority pair already has two live, initializing, or retained providers\n");
            delete p;
            return nullptr;
        }
    }
    const int requested_priority = p->priority_pair_slot == 1 ? -1 : 0;
    bool ok                       = true;
    int  version                  = 0;
    ok                            = checked(*p, hipRuntimeGetVersion(&version), "hipRuntimeGetVersion");
    p->status.hip_runtime_version = uint32_t(version);
    for (uint32_t r = 0; ok && r < ranks; ++r) {
        if (!ggml_backend_is_cuda(backends[r])) {
            ok = false;
            break;
        }
        auto *          ctx      = static_cast<ggml_backend_cuda_context *>(backends[r]->context);
        const int       physical = ggml_cuda_info().devices[ctx->device].physical_device;
        hipDeviceProp_t prop;
        ok = checked(*p, hipGetDeviceProperties(&prop, physical), "hipGetDeviceProperties");
        if (!ok || strncmp(prop.gcnArchName, "gfx1030", 7) != 0 || strcmp(prop.name, "AMD Radeon RX 6800") != 0 ||
            !prop.canMapHostMemory || ctx->tp5_relay_owner) {
            ok = false;
            break;
        }
        // Never discard a pre-existing side lane or concurrent plan and then
        // claim its work was drained by stream zero. The strict optimizer does
        // not create such plans after this provider takes ownership.
        bool single_lane = ctx->curr_stream_no == 0 &&
            ctx->concurrent_stream_context.concurrent_events.empty();
        for (int device = 0; device < GGML_CUDA_MAX_DEVICES; ++device) {
            for (uint32_t s = 0; s < GGML_CUDA_MAX_STREAMS; ++s) {
                if ((device != ctx->device || s != 0) && ctx->streams[device][s]) {
                    single_lane = false;
                }
            }
        }
        if (!single_lane) {
            fail(*p, EINVAL, "strict relay requires stream zero without existing side lanes or concurrent plans");
            ok = false;
            break;
        }
        for (uint32_t prior = 0; prior < r; ++prior) {
            if (p->rank[prior].physical == physical || p->rank[prior].context == ctx) {
                ok = false;
            }
        }
        if (!ok) {
            break;
        }
        if (p->priority_pair_slot >= 0) {
            int least_priority = 0;
            int greatest_priority = 0;
            if (!checked(*p, hipSetDevice(physical), "hipSetDevice strict priority admission") ||
                !checked(*p, hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority),
                         "hipDeviceGetStreamPriorityRange strict priority")) {
                ok = false;
                break;
            }
            if (greatest_priority > least_priority || requested_priority < greatest_priority ||
                requested_priority > least_priority) {
                fail(*p, ENOTSUP, "strict priority pair requires the requested priority without clamping");
                ok = false;
                break;
            }
            // This check precedes claiming this caller context. Never destroy,
            // replace, or clear an existing stream to obtain another priority.
            auto existing = ctx->streams[ctx->device][0];
            if (existing && !priority_stream_matches(*p, existing, requested_priority)) {
                ok = false;
                break;
            }
        }
        p->rank[r].physical  = physical;
        p->rank[r].context   = ctx;
        p->rank[r].backend   = backends[r];
        int wall_clock_khz = 0;
        if (!checked(*p, hipDeviceGetAttribute(&wall_clock_khz, hipDeviceAttributeWallClockRate, physical),
                     "hipDeviceGetAttribute steady clock rate") || wall_clock_khz <= 0) {
            fail(*p, ENOTSUP, "strict relay requires a positive steady clock rate");
            ok = false;
            break;
        }
        p->rank[r].wait_ticks = uint64_t(wall_clock_khz) * 1000 * wait_seconds;
        ctx->tp5_relay       = true;
        ctx->tp5_relay_owner = p;
        ok = select(*p, r) &&
             checked(*p, hipDeviceGetPCIBusId(p->status.pci_bdf[r], sizeof(p->status.pci_bdf[r]), physical),
                     "hipDeviceGetPCIBusId");
        constexpr size_t scratch_bytes = 128u * 1024u * 1024u;
        constexpr size_t cache_bytes   = 1u * 1024u * 1024u;
        if (ok && ctx->pools[ctx->device][0]) {
            fail(*p, EINVAL, "strict relay requires an unused child scratch pool");
            ok = false;
        }
        if (ok) {
            ok = checked(*p, hipMalloc(&p->rank[r].scratch, scratch_bytes), "hipMalloc strict scratch");
        }
        if (ok) {
            auto * pool = new (std::nothrow) relay_pool(p->rank[r].scratch, scratch_bytes);
            if (!pool) {
                fail(*p, ENOMEM, "allocating strict scratch metadata failed");
                ok = false;
            } else {
                ctx->pools[ctx->device][0].reset(pool);
            }
        }
        if (ok && !ctx->q8_cache.ptr) {
            ok = checked(*p, hipMalloc(reinterpret_cast<void **>(&ctx->q8_cache.ptr), cache_bytes),
                         "hipMalloc strict q8 cache");
            if (ok) {
                ctx->q8_cache.cap = cache_bytes;
                ctx->q8_cache.dev = ctx->device;
            }
        }
        if (ok && !ctx->cublas_handles[ctx->device]) {
            const auto result = hipblasCreate(&ctx->cublas_handles[ctx->device]);
            if (result != HIPBLAS_STATUS_SUCCESS) {
                fail(*p, EIO, "hipblasCreate strict child handle failed");
                ok = false;
            }
        }
        for (uint32_t s = 0; ok && s < lanes_per_rank; ++s) {
            // Never initialize through the legacy/default/per-thread stream:
            // another context can already be waiting on its CPU reducer there.
            auto & l = p->rank[r].lanes[s];
            if (!ctx->streams[ctx->device][s]) {
                if (p->priority_pair_slot >= 0) {
                    ok = checked(*p,
                        hipStreamCreateWithPriority(&ctx->streams[ctx->device][s], hipStreamNonBlocking,
                                                    requested_priority),
                        "hipStreamCreateWithPriority strict priority");
                    if (ok) {
                        ok = priority_stream_matches(*p, ctx->streams[ctx->device][s], requested_priority);
                    }
                } else {
                    ok = checked(*p, hipStreamCreateWithFlags(&ctx->streams[ctx->device][s], hipStreamNonBlocking),
                                 "hipStreamCreateWithFlags");
                }
            }
            l.stream = ctx->streams[ctx->device][s];
            if (ok) {
                ok = checked(*p, hipEventCreateWithFlags(&l.tail, hipEventDisableTiming), "hipEventCreateWithFlags");
            }
        }
    }
    if (ok) {
        void * allocation = nullptr;
        ok = checked(*p, hipHostMalloc(&allocation, sizeof(word), hipHostMallocMapped | hipHostMallocPortable),
                     "hipHostMalloc abort");
        p->abort_host = static_cast<word *>(allocation);
        if (ok) {
            memset(p->abort_host, 0, sizeof(word));
            for (uint32_t r = 0; ok && r < ranks; ++r) {
                ok = select(*p, r) &&
                     checked(*p,
                             hipHostGetDevicePointer(reinterpret_cast<void **>(&p->abort_device[r]), p->abort_host, 0),
                             "hipHostGetDevicePointer abort");
            }
        }
    }
    if (ok) {
        void * allocation = nullptr;
        ok = checked(*p, hipHostMalloc(&allocation, sizeof(word), hipHostMallocMapped | hipHostMallocPortable),
                     "hipHostMalloc activation");
        p->activation_host = static_cast<word *>(allocation);
        if (ok) {
            store(p->activation_host, 0);
            for (uint32_t r = 0; ok && r < ranks; ++r) {
                ok = select(*p, r) && checked(*p,
                    hipHostGetDevicePointer(reinterpret_cast<void **>(&p->activation_device[r]), allocation, 0),
                    "hipHostGetDevicePointer activation");
            }
        }
    }
    if (ok) {
        ok = reserve_channels(*p);
    }
    if (!ok) {
        if (p->priority_pair_slot >= 0) {
            GGML_LOG_ERROR("HIP relay priority pair slot=%d create failed: %s\n",
                           p->priority_pair_slot, p->status.error);
        }
        if (release_owners(*p)) {
            delete p;
        } else {
            retain(p);
        }
        return nullptr;
    }
    if (p->priority_pair_slot >= 0) {
        GGML_LOG_INFO("HIP relay priority pair provider=%p slot=%d priority=%d flags=%u ranks=%u\n",
                      static_cast<void *>(p), p->priority_pair_slot, requested_priority,
                      unsigned(hipStreamNonBlocking), ranks);
    }
    return p;
#endif
}

size_t wire_element_size(const provider & p) {
    return p.wire_type.load(std::memory_order_relaxed) == GGML_RELAY_WIRE_F16 ? sizeof(half) : sizeof(float);
}

bool set_wire_type(void * opaque, ggml_relay_wire_type wire_type) {
    if (!opaque || (wire_type != GGML_RELAY_WIRE_F32 && wire_type != GGML_RELAY_WIRE_F16)) {
        return false;
    }
    auto & p = *static_cast<provider *>(opaque);
    std::lock_guard<std::mutex> lock(p.mutex);
    if (p.wire_frozen || p.active || p.status.failed || p.status.epoch != 0 ||
            p.status.plan_generation != 0 || p.borrowed != std::numeric_limits<size_t>::max()) {
        return false;
    }
    if (p.wire_configured) {
        return p.wire_type.load(std::memory_order_relaxed) == wire_type;
    }
    const uint16_t endian = 1;
    if (wire_type == GGML_RELAY_WIRE_F16 && *reinterpret_cast<const unsigned char *>(&endian) != 1) {
        return false;
    }
    // No live payloads or frames exist here. The F32-sized pool remains the
    // physical capacity; only the first wire-sized prefix of each slot is used.
    for (auto & c : p.channels) {
        c.zero_inbox_mask = 0;
    }
    p.wire_type.store(wire_type, std::memory_order_release);
    p.wire_configured = true;
    return true;
}

bool prepare(void * opaque, ggml_tensor * const * terminals, size_t count) {
    auto &                      p = *static_cast<provider *>(opaque);
    std::lock_guard<std::mutex> lock(p.mutex);
    if (!idle(p) || p.status.failed) {
        return false;
    }
    p.wire_frozen = true; // Format is immutable before any captured graph can exist.
    // prepare announces a new owned definition even if its terminal bindings
    // are unchanged: intermediate topology, layouts or fusion may differ.
    p.definition_changed = true;
    if (!terminals || count == 0 || count > channel_capacity) {
        fail(p, ENOTSUP, "relay plan exceeds reserved 256-channel capacity or is empty");
        return false;
    }
    bool same_plan = p.status.channels == count;
    // Validate the entire next binding before changing any active plan. All
    // mapped storage was reserved at create, before either context decodes.
    for (size_t c = 0; c < count; ++c) {
        const auto * first = terminals[c * ranks];
        if (!first || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first) || ggml_nelements(first) <= 0) {
            fail(p, ENOTSUP, "relay requires nonempty contiguous F32 terminals");
            return false;
        }
        if (uint64_t(ggml_nelements(first)) > element_capacity) {
            char detail[sizeof(p.status.error)];
            std::snprintf(detail, sizeof(detail), "relay channel=%zu tensor=%s shape=%lld,%lld,%lld,%lld exceeds %zu-F32 capacity",
                          c, first->name, static_cast<long long>(first->ne[0]), static_cast<long long>(first->ne[1]),
                          static_cast<long long>(first->ne[2]), static_cast<long long>(first->ne[3]), element_capacity);
            fail(p, ENOTSUP, detail);
            return false;
        }
        for (uint32_t r = 0; r < ranks; ++r) {
            const auto * t = terminals[c * ranks + r];
            if (!t || !t->data || t->type != first->type || !ggml_is_contiguous(t) ||
                memcmp(t->ne, first->ne, sizeof(t->ne)) || memcmp(t->nb, first->nb, sizeof(t->nb))) {
                fail(p, EINVAL, "relay rank terminal shapes/types/strides differ");
                return false;
            }
            const bool unchanged = p.status.channels == count && same_binding(p.channels[c], r, t);
            same_plan            = same_plan && unchanged;
            if (unchanged) {
                continue;
            }
            hipPointerAttribute_t attributes;
            if (!select(p, r) ||
                !checked(p, hipPointerGetAttributes(&attributes, t->data), "hipPointerGetAttributes terminal") ||
                attributes.device != p.rank[r].physical || attributes.type != hipMemoryTypeDevice) {
                fail(p, EINVAL, "relay terminal is not rank-local native device memory");
                return false;
            }
        }
    }
    // idle() proved native tails, CPU borrows and old output releases ended.
    // Inputs are read-only to the CPU service. Only a computed producer can
    // dirty a slot; mark it before any direct/captured/replayed submission.
    // Clear the entire physical slot so later larger bindings remain safe.
    // Do this even for identical terminal bindings: COMPUTE flags may change.
    for (size_t i = 0; i < count; ++i) {
        auto & c = p.channels[i];
        c.computed_mask = 0;
        for (uint32_t r = 0; r < ranks; ++r) {
            const uint32_t bit = 1u << r;
            if (terminals[i * ranks + r]->flags & GGML_TENSOR_FLAG_COMPUTE) {
                c.computed_mask |= bit;
                c.zero_inbox_mask &= ~bit;
            } else if (!(c.zero_inbox_mask & bit)) {
                memset(c.inbox[r], 0, element_capacity * wire_element_size(p));
                c.zero_inbox_mask |= bit;
            }
        }
    }
    if (same_plan) {
        return true;
    }
    if (p.status.plan_generation == UINT64_MAX) {
        fail(p, EOVERFLOW, "relay plan generation overflow");
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        auto & c   = p.channels[i];
        c.elements = size_t(ggml_nelements(terminals[i * ranks]));
        c.bytes    = c.elements * wire_element_size(p);
        for (uint32_t r = 0; r < ranks; ++r) {
            bind(c, r, terminals[i * ranks + r]);
        }
        c.released = true;
    }
    ++p.status.plan_generation;
    p.status.channels = count;
    return true;
}

bool begin(provider & p) {
    if (!idle(p) || p.status.failed || p.status.channels == 0) {
        return false;
    }
    if (p.status.epoch == UINT64_MAX) {
        fail(p, EOVERFLOW, "relay epoch overflow");
        return false;
    }
    ++p.status.epoch;
    // idle proved prior native tails and CPU readers ended. Captured kernels
    // dereference this immutable activation, never a captured scalar epoch.
    store(p.activation_host, p.status.epoch);
    p.status.attempted_mask = p.status.accepted_mask = p.status.unknown_mask = p.status.completed_mask = 0;
    p.status.sealed                                                                                    = false;
    p.status.finished                                                                                  = false;
    p.active                                                                                           = true;
    p.tails_attempted                                                                                  = false;
    p.next_channel                                                                                     = 0;
    p.outputs_released = 0;
    p.release_scan_begin = p.offer_scan_begin = p.ready_scan_end = p.release_scan_end = 0;
    for (auto & r : p.rank) {
        r.graph_pending   = false;
        r.pending_nodes   = 0;
        r.accepted_graphs = 0;
        for (auto & l : r.lanes) {
            l.recorded = l.complete = false;
        }
    }
    for (size_t i = 0; i < p.status.channels; ++i) {
        auto & c   = p.channels[i];
        c.enqueued = c.offered = c.published = c.released = false;
        c.ready_seen = c.ack_seen = 0;
        // No native work or CPU borrow remains. GPU error words are only reset
        // here, never while their single-writer GPU could still access them.
        for (auto & error : c.host_control->error) {
            store(&error, 0);
        }
    }
    return true;
}

void graph_begin(provider & p, uint32_t r, uint64_t compute_nodes) {
    consume_abort(p);
    if (!p.active || p.tails_attempted || p.status.failed || r >= ranks || p.rank[r].graph_pending) {
        fail(p, EINVAL, "invalid relay graph_begin ordering");
        return;
    }
    if (p.mode == submission::direct) {
        p.status.attempted_mask |= 1u << r;
        p.status.unknown_mask |= 1u << r;
    }
    p.rank[r].graph_pending = true;
    p.rank[r].pending_nodes = compute_nodes;
}

void graph_accepted(provider & p, uint32_t r) {
    consume_abort(p);
    if (!p.active || p.tails_attempted || r >= ranks || !p.rank[r].graph_pending) {
        fail(p, EINVAL, "invalid relay graph_accepted ordering");
        return;
    }
    p.rank[r].graph_pending = false;
    ++p.rank[r].accepted_graphs;
    if (p.mode != submission::direct) {
        return; // Capture is construction, not native launch acceptance.
    }
    p.status.accepted_mask |= 1u << r;
    p.status.unknown_mask &= ~(1u << r);
    ++p.status.rank_graphs[r];
    p.status.rank_compute_nodes[r] += p.rank[r].pending_nodes;
}

bool enqueue(provider & p, uint32_t index) {
    consume_abort(p);
    if (!p.active || p.tails_attempted || p.status.failed || index != p.next_channel || index >= p.status.channels) {
        fail(p, EINVAL, "invalid relay channel enqueue ordering");
        return false;
    }
    auto &         c      = p.channels[index];
    for (uint32_t r = 0; r < ranks; ++r) {
        consume_abort(p);
        if (p.status.failed) {
            return false;
        }
        if (p.rank[r].graph_pending || p.rank[r].accepted_graphs != p.next_channel + 1 || !select(p, r)) {
            fail(p, EINVAL, "relay enqueue before accepted rank graph");
            return false;
        }
        auto * ctx = p.rank[r].context;
        if (ctx->curr_stream_no != 0) {
            fail(p, EINVAL, "relay rank must use its owned stream zero");
            return false;
        }
        const auto stream   = p.rank[r].lanes[0].stream;
        auto *     dc       = c.device_control[r];
        auto *     stop     = p.abort_device[r];
        auto *     terminal = static_cast<float *>(c.terminal[r]->data);
        // Every operation is submitted before finish/seal allows CPU service.
        // No host continuation, stream sync, allocation, or P2P transfer here.
        if (p.mode == submission::direct) {
            p.status.attempted_mask |= 1u << r;
            p.status.unknown_mask |= 1u << r;
        }
        const bool computed = (c.computed_mask & (1u << r)) != 0;
        const bool f16_wire = p.wire_type.load(std::memory_order_relaxed) == GGML_RELAY_WIRE_F16;
        if (computed) {
            if (f16_wire) {
                publish_input<half><<<1, 256, 0, stream>>>(static_cast<half *>(c.device_inbox[r]), terminal, c.elements,
                    dc, r, p.activation_device[r]);
            } else {
                publish_input<float><<<1, 256, 0, stream>>>(static_cast<float *>(c.device_inbox[r]), terminal, c.elements,
                    dc, r, p.activation_device[r]);
            }
            if (!checked(p, hipGetLastError(), "relay producer publication launch")) {
                return false;
            }
            if (p.mode == submission::direct) {
                ++p.status.rank_relay_nodes[r];
            } else if (p.mode == submission::capture) {
                ++p.current->relay_nodes[r];
            }
        }
        if (f16_wire) {
            consume_output<half><<<1, 256, 0, stream>>>(terminal, static_cast<const half *>(c.device_output[r]), c.elements,
                dc, stop, r, p.activation_device[r], p.rank[r].wait_ticks, !computed);
        } else {
            consume_output<float><<<1, 256, 0, stream>>>(terminal, static_cast<const float *>(c.device_output[r]), c.elements,
                dc, stop, r, p.activation_device[r], p.rank[r].wait_ticks, !computed);
        }
        if (!checked(p, hipGetLastError(), "relay wait consumption and last reader launch")) {
            return false;
        }
        if (p.mode == submission::direct) {
            ++p.status.rank_relay_nodes[r];
            p.status.unknown_mask &= ~(1u << r);
        } else if (p.mode == submission::capture) {
            ++p.current->relay_nodes[r];
        }
    }
    c.enqueued = true;
    include_ready_candidate(p, index);
    ++p.next_channel;
    return true;
}

bool finish(provider & p) {
    consume_abort(p);
    if (!p.active || p.tails_attempted) {
        fail(p, EINVAL, "relay finish without an active unsealed epoch");
        return false;
    }
    if (p.mode == submission::capture) {
        // Always end every capture, even if a later rank/stage failed. Until
        // EndCapture succeeds (or explicitly invalidates), keep its owners.
        for (uint32_t r = 0; r < ranks; ++r) {
            if (!p.rank[r].capturing || !select(p, r)) {
                continue;
            }
            const hipError_t result = hipStreamEndCapture(p.rank[r].lanes[0].stream, &p.current->graph[r]);
            if (result == hipSuccess || result == hipErrorStreamCaptureInvalidated) {
                p.rank[r].capturing = false;
            }
            checked(p, result, "hipStreamEndCapture relay epoch");
        }
        for (uint32_t r = 0; r < ranks; ++r) {
            if (p.rank[r].accepted_graphs != p.status.channels + 1 || p.rank[r].graph_pending) {
                fail(p, EIO, "relay capture is missing stages");
            }
        }
        // No launch until all five complete executables exist. Failure leaves
        // partially created graph handles owned for checked destruction.
        if (!p.status.failed) {
            // Each job owns one rank. All jobs finish before any graph launch.
            // Async futures also wait if worker creation throws.
            std::array<std::future<hipError_t>, ranks> jobs;
            uint32_t started = 0;
            try {
                for (; started < ranks; ++started) {
                    const uint32_t r = started;
                    jobs[r] = std::async(std::launch::async, [&p, r]() noexcept {
                        const hipError_t selected = hipSetDevice(p.rank[r].physical);
                        return selected == hipSuccess ?
                            hipGraphInstantiate(&p.current->exec[r], p.current->graph[r], nullptr, nullptr, 0) :
                            selected;
                    });
                }
            } catch (const std::bad_alloc &) {
                fail(p, ENOMEM, "allocating relay graph workers failed");
            } catch (const std::system_error &) {
                fail(p, EAGAIN, "starting relay graph workers failed");
            }
            for (uint32_t r = 0; r < started; ++r) {
                checked(p, jobs[r].get(), "select/instantiate relay epoch worker");
            }
        }
    }
    if (p.mode != submission::direct && !p.status.failed) {
        for (uint32_t r = 0; r < ranks; ++r) {
            consume_abort(p);
            if (p.status.failed || !select(p, r)) {
                break;
            }
            const uint32_t bit = 1u << r;
            p.status.attempted_mask |= bit;
            p.status.unknown_mask |= bit;
            if (!checked(p, hipGraphLaunch(p.current->exec[r], p.rank[r].lanes[0].stream),
                         "hipGraphLaunch relay epoch")) {
                break; // Never fall back after a possibly accepted launch.
            }
            p.status.accepted_mask |= bit;
            p.status.unknown_mask &= ~bit;
            p.rank[r].accepted_graphs = p.status.channels + 1;
            p.status.rank_graphs[r] += p.status.channels + 1;
            p.status.rank_compute_nodes[r] += p.current->compute_nodes[r];
            p.status.rank_relay_nodes[r] += p.current->relay_nodes[r];
        }
    }
    p.tails_attempted = true;
    for (uint32_t r = 0; r < ranks; ++r) {
        if (p.rank[r].accepted_graphs != p.status.channels + 1 || p.rank[r].graph_pending) {
            fail(p, EIO, "relay frame is missing accepted stage or final rank graphs");
        }
        if (!(p.status.attempted_mask & (1u << r))) {
            continue;
        }
        if (!select(p, r)) {
            continue;
        }
        for (auto & l : p.rank[r].lanes) {
            // Even after an earlier HIP error, attempt every actual lane tail.
            l.recorded = checked(p, hipEventRecord(l.tail, l.stream), "hipEventRecord native tail");
        }
    }
    if (p.next_channel != p.status.channels || p.status.accepted_mask != all_ranks || p.status.unknown_mask) {
        fail(p, EIO, "relay frame has incomplete or unknown submissions");
    }
    p.active = false;
    consume_abort(p);
    p.status.sealed = !p.status.failed;
    if (p.status.sealed) {
        p.event_status.store(1, std::memory_order_release);
    }
    if (p.status.sealed && p.mode == submission::direct) {
        p.current->warmed = true;
    }
    observe(p);
    return p.status.sealed;
}

template<typename... T> void describe(provider & p, const T &... values) {
    constexpr size_t bytes = (sizeof(T) + ...);
    const size_t offset = p.signature.size();
    if (bytes > SIZE_MAX - offset) {
        throw std::bad_alloc();
    }
    p.signature.resize(offset + bytes);
    auto * output = p.signature.data() + offset;
    const auto append = [&](const auto & value) {
        memcpy(output, &value, sizeof(value));
        output += sizeof(value);
    };
    (append(values), ...);
}

void describe_tensor(provider & p, const ggml_tensor * t) {
    describe(p, t, t->type, t->buffer, t->ne, t->nb, t->op, t->op_params,
             t->flags, t->src, t->view_src, t->view_offs, t->data, t->extra);
}

bool describe_frame(provider & p, ggml_cgraph * const * graphs, size_t stages, uint64_t * compute_nodes) {
    p.signature.clear();
    p.tensors.clear();
    p.stage_nodes.assign(stages * ranks, 0);
    describe(p, stages, p.wire_type.load(std::memory_order_relaxed));
    // Captured relay kernels bind channels in this order. A different terminal
    // can already belong to the same unchanged graph/source closure.
    describe(p, p.status.channels);
    for (size_t i = 0; i < p.status.channels; ++i) {
        const auto & c = p.channels[i];
        describe(p, c.elements, c.bytes, c.computed_mask);
        for (uint32_t r = 0; r < ranks; ++r) {
            const auto * terminal = c.terminal[r];
            describe(p, terminal, terminal->data, terminal->ne, terminal->nb, terminal->flags);
        }
    }
    // Keep the complete key, but collect its source/view closure in linear
    // time with GGML's pointer set rather than repeated prefix sorts/searches.
    size_t slots = 64;
    for (size_t i = 0; i < stages * ranks; ++i) {
        if (graphs[i] && graphs[i]->n_nodes > 0) {
            const size_t count = size_t(graphs[i]->n_nodes);
            if (count > (SIZE_MAX - slots) / 4) {
                throw std::bad_alloc();
            }
            slots += 4 * count;
        }
    }
    if (p.tensor_keys.size() < slots) {
        p.tensor_keys.resize(ggml_hash_size(slots));
    }
    p.tensor_used.assign(ggml_bitset_size(p.tensor_keys.size()), 0);
    ggml_hash_set seen{p.tensor_keys.size(), p.tensor_used.data(), p.tensor_keys.data()};
    const auto add_tensor = [&](const ggml_tensor * tensor) {
        if (!tensor) {
            return;
        }
        if (p.tensors.size() >= seen.size / 2) {
            if (seen.size > (SIZE_MAX - BITSET_MASK) / 2) {
                throw std::bad_alloc();
            }
            p.tensor_keys.resize(ggml_hash_size(2 * seen.size));
            p.tensor_used.assign(ggml_bitset_size(p.tensor_keys.size()), 0);
            seen = {p.tensor_keys.size(), p.tensor_used.data(), p.tensor_keys.data()};
            for (const auto * previous : p.tensors) {
                ggml_hash_insert(&seen, const_cast<ggml_tensor *>(previous));
            }
        }
        if (ggml_hash_insert(&seen, const_cast<ggml_tensor *>(tensor)) != GGML_HASHSET_ALREADY_EXISTS) {
            p.tensors.push_back(tensor);
        }
    };
    for (uint32_t r = 0; r < ranks; ++r) {
        const auto * ctx = p.rank[r].context;
        if (ctx->curr_stream_no != 0 || !ctx->tp5_relay || ctx->tp5_relay_owner != &p) {
            fail(p, EINVAL, "relay replay requires its owned stream zero");
            return false;
        }
        describe(p, p.rank[r].physical, p.rank[r].scratch, p.rank[r].lanes[0].stream, ctx->quantized_matmul_f32);
    }
    for (size_t stage = 0; stage < stages; ++stage) {
        for (uint32_t r = 0; r < ranks; ++r) {
            const auto * g = graphs[stage * ranks + r];
            if (!g || g->n_nodes < 0 || g->n_leafs < 0 || g->grads || g->grad_accs) {
                fail(p, EINVAL, "relay replay requires complete inference graphs");
                return false;
            }
            describe(p, g->n_nodes, g->n_leafs, g->order);
            for (int i = 0; i < g->n_nodes; ++i) {
                const auto * t = g->nodes[i];
                if (!t) {
                    fail(p, EINVAL, "relay graph contains a null node");
                    return false;
                }
                const int32_t uses = g->use_counts ? g->use_counts[ggml_hash_find(&g->visited_hash_set, t)] : 0;
                describe(p, t, uses); // Ordered topology and fusion use counts, not just terminal bindings.
                add_tensor(t);
                if (!ggml_is_empty(t) && (t->flags & GGML_TENSOR_FLAG_COMPUTE) && t->op != GGML_OP_NONE &&
                    t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE &&
                    t->op != GGML_OP_PERMUTE && t->op != GGML_OP_TRANSPOSE) {
                    ++compute_nodes[r];
                    ++p.stage_nodes[stage * ranks + r];
                }
            }
            for (int i = 0; i < g->n_leafs; ++i) {
                describe(p, g->leafs[i]);
                if (!g->leafs[i]) {
                    fail(p, EINVAL, "relay graph contains a null leaf");
                    return false;
                }
                add_tensor(g->leafs[i]);
            }
        }
    }
    for (size_t i = 0; i < p.tensors.size(); ++i) {
        const auto * tensor = p.tensors[i];
        for (const auto * source : tensor->src) {
            add_tensor(source);
        }
        add_tensor(tensor->view_src);
    }
    // Root order is already part of the key. First-visit order through src[]
    // and view_src is deterministic too; no pointer sort is needed.
    describe(p, p.tensors.size());
    for (const auto * t : p.tensors) {
        describe_tensor(p, t);
    }
    return true;
}

bool submit(void * opaque, ggml_cgraph * const * graphs, size_t stages) {
    auto & p = *static_cast<provider *>(opaque);
    const int64_t d0=ggml_time_us();
    std::lock_guard<std::mutex> lock(p.mutex);
    const int64_t dlock=ggml_time_us();
    const bool changed=p.definition_changed || !p.current;
    bool hit=!changed;
    if (!idle(p) || p.status.failed || !graphs || stages != p.status.channels + 1) {
        return false;
    }
    uint64_t compute_nodes[ranks] = {};
    try {
        if (p.definition_changed || !p.current) {
        if (!describe_frame(p, graphs, stages, compute_nodes)) {
            return false;
        }
        p.current = nullptr;
        for (auto & f : p.frames) {
            if (f.signature == p.signature) {
                hit=true;
                p.current = &f;
                break;
            }
        }
        if (!p.current) {
            auto & f = p.frames[p.frames[0].last_used <= p.frames[1].last_used ? 0 : 1];
            // idle above proves this old executable has no native or CPU users.
            if (!clear_frame(p, f)) {
                return false;
            }
            f.signature.swap(p.signature);
            std::copy(compute_nodes, compute_nodes + ranks, f.compute_nodes);
            p.current = &f;
        }
        p.definition_changed = false;
        }
    } catch (const std::bad_alloc &) {
        fail(p, ENOMEM, "allocating relay frame signature failed");
        return false;
    }
    p.mode = p.current->exec[0] ? submission::replay :
        (p.current->warmed ? submission::capture : submission::direct);
    const int64_t dkey=ggml_time_us();
    if (!begin(p)) {
        return false;
    }
    const int64_t dbegin=ggml_time_us();
    auto diag_finish=[&]() {
        const int64_t dfinish=ggml_time_us();
        const bool ok=finish(p);
        if (std::getenv("TP5_DIAG_GRAPH")) GGML_LOG_INFO("TP5_DIAG_RELAY ctx=%p epoch=%llu mode=%d changed=%d hit=%d stages=%zu lock_us=%lld key_us=%lld begin_us=%lld dispatch_us=%lld finish_us=%lld total_us=%lld\n", (void*)&p,(unsigned long long)p.status.epoch,int(p.mode),int(changed),int(hit),stages,(long long)(dlock-d0),(long long)(dkey-dlock),(long long)(dbegin-dkey),(long long)(dfinish-dbegin),(long long)(ggml_time_us()-dfinish),(long long)(ggml_time_us()-d0));
        return ok;
    };
    p.current->last_used = p.status.epoch;
    if (p.mode == submission::replay) {
        p.next_channel = p.status.channels;
        for (size_t i = 0; i < p.status.channels; ++i) {
            p.channels[i].enqueued = true;
            include_ready_candidate(p, i);
        }
        return diag_finish();
    }
    if (p.mode == submission::capture) {
        for (uint32_t r = 0; r < ranks && !p.status.failed; ++r) {
            if (select(p, r) && checked(p,
                hipStreamBeginCapture(p.rank[r].lanes[0].stream, hipStreamCaptureModeThreadLocal),
                "hipStreamBeginCapture relay epoch")) {
                p.rank[r].capturing = true;
            }
        }
    }
    for (size_t stage = 0; stage < stages && !p.status.failed; ++stage) {
        for (uint32_t r = 0; r < ranks && !p.status.failed; ++r) {
            auto * g = graphs[stage * ranks + r];
            graph_begin(p, r, p.stage_nodes[stage * ranks + r]);
            if (p.status.failed) {
                break;
            }
            if (ggml_backend_graph_compute_async(p.rank[r].backend, g) != GGML_STATUS_SUCCESS) {
                fail(p, EIO, "relay child graph submission failed");
                break;
            }
            graph_accepted(p, r);
        }
        if (stage < p.status.channels && !p.status.failed) {
            enqueue(p, uint32_t(stage));
        }
    }
    return diag_finish(); // Also terminates partial captures and records failed tails.
}

void fill_event(provider & p, size_t index, ggml_relay_event_kind kind, ggml_relay_event & event) {
    auto & c              = p.channels[index];
    event                 = {};
    event.abi_version     = GGML_RELAY_ABI_VERSION;
    event.kind            = kind;
    event.epoch           = p.status.epoch;
    event.plan_generation = p.status.plan_generation;
    event.channel         = uint32_t(index);
    event.ranks           = ranks;
    event.elements        = c.elements;
    if (kind == GGML_RELAY_INPUT_READY) {
        for (uint32_t r = 0; r < ranks; ++r) {
            event.inputs[r] = static_cast<const float *>(c.inbox[r]);
        }
        event.output = static_cast<float *>(c.output);
    }
}

void fill_event(provider & p, size_t index, ggml_relay_event_kind kind, ggml_relay_event_v2 & event) {
    auto & c              = p.channels[index];
    event                 = {};
    event.abi_version     = GGML_RELAY_EVENT_V2_ABI_VERSION;
    event.kind            = kind;
    event.epoch           = p.status.epoch;
    event.plan_generation = p.status.plan_generation;
    event.channel         = uint32_t(index);
    event.ranks           = ranks;
    event.elements        = c.elements;
    event.wire_type       = p.wire_type.load(std::memory_order_relaxed);
    event.computed_mask   = c.computed_mask;
    event.wire_bytes      = c.bytes;
    if (kind == GGML_RELAY_INPUT_READY) {
        for (uint32_t r = 0; r < ranks; ++r) {
            event.inputs[r] = c.inbox[r];
        }
        event.output = c.output;
    }
}

template<typename Event>
int poll_event(void * opaque, Event * event) {
    if (!opaque || !event) {
        return -1;
    }
    auto &                       p = *static_cast<provider *>(opaque);
    if constexpr (std::is_same<Event, ggml_relay_event>::value) {
        if (p.wire_type.load(std::memory_order_acquire) != GGML_RELAY_WIRE_F32) {
            return -1; // Never expose packed bytes as float pointers.
        }
    }
    if (p.pending_abort.load(std::memory_order_acquire)) {
        return -1;
    }
    const int event_status = p.event_status.load(std::memory_order_acquire);
    if (event_status <= 0) {
        return event_status;
    }
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        return 0;
    }
    consume_abort(p);
    if (!p.status.sealed || p.borrowed != std::numeric_limits<size_t>::max()) {
        return p.status.failed ? -1 : 0;
    }
    // Completion queries belong to snapshot/close. Calling into the HIP
    // runtime here switches five device contexts on every busy-wait iteration.
    observe(p, false);
    if (p.status.failed) {
        return -1;
    }
    for (size_t i = p.release_scan_begin; i < p.release_scan_end; ++i) {
        auto & c = p.channels[i];
        if (c.published && !c.released && c.ack_seen == all_ranks) {
            c.released = true;
            while (p.release_scan_begin < p.status.channels && p.channels[p.release_scan_begin].released) {
                ++p.release_scan_begin;
            }
            if (++p.outputs_released == p.status.channels) {
                p.event_status.store(0, std::memory_order_release);
            }
            fill_event(p, i, GGML_RELAY_OUTPUT_RELEASED, *event);
            return 1;
        }
    }
    for (size_t i = p.offer_scan_begin; i < p.ready_scan_end; ++i) {
        auto & c = p.channels[i];
        if (c.enqueued && !c.offered && c.ready_seen == all_ranks) {
            c.offered  = true;
            while (p.offer_scan_begin < p.status.channels && p.channels[p.offer_scan_begin].offered) {
                ++p.offer_scan_begin;
            }
            p.borrowed = i;
            fill_event(p, i, GGML_RELAY_INPUT_READY, *event);
            return 1;
        }
    }
    return 0;
}

int poll(void * opaque, ggml_relay_event * event) {
    return poll_event(opaque, event);
}

int poll_v2(void * opaque, ggml_relay_event_v2 * event, size_t event_size) {
    if (event_size < sizeof(ggml_relay_event_v2)) {
        return -1;
    }
    return poll_event(opaque, event);
}

bool matches_wire(const provider & p, const channel &, const ggml_relay_event & event) {
    return p.wire_type.load(std::memory_order_relaxed) == GGML_RELAY_WIRE_F32 &&
           event.abi_version == GGML_RELAY_ABI_VERSION;
}

bool matches_wire(const provider & p, const channel & c, const ggml_relay_event_v2 & event) {
    return event.abi_version == GGML_RELAY_EVENT_V2_ABI_VERSION &&
           event.wire_type == p.wire_type.load(std::memory_order_relaxed) &&
           event.wire_bytes == c.bytes && event.computed_mask == c.computed_mask;
}

template<typename Event>
bool publish_event(void * opaque, const Event * event) {
    if (!opaque || !event) {
        return false;
    }
    auto &                      p = *static_cast<provider *>(opaque);
    std::lock_guard<std::mutex> lock(p.mutex);
    if constexpr (std::is_same<Event, ggml_relay_event>::value) {
        if (p.wire_type.load(std::memory_order_relaxed) != GGML_RELAY_WIRE_F32) {
            return false;
        }
    }
    consume_abort(p);
    if (p.borrowed == std::numeric_limits<size_t>::max()) {
        return false;
    }
    const size_t index   = p.borrowed;
    auto &       c       = p.channels[index];
    bool         matches = matches_wire(p, c, *event) && event->kind == GGML_RELAY_INPUT_READY &&
                           event->epoch == p.status.epoch && event->plan_generation == p.status.plan_generation &&
                           event->channel == index && event->ranks == ranks && event->elements == c.elements &&
                           event->output == c.output;
    for (uint32_t r = 0; r < ranks; ++r) {
        matches = matches && event->inputs[r] == c.inbox[r];
    }
    if (!matches) {
        fail(p, EINVAL, "relay publish view identity mismatch");
        return false;
    }
    // Matching publish always returns the CPU borrow, including after abort.
    // A generic abort alone cannot revoke a concurrently executing CPU reader.
    p.borrowed = std::numeric_limits<size_t>::max();
    observe(p, false);
    if (p.status.failed || !p.status.sealed || !c.offered || c.published) {
        return false;
    }
    if (load(p.abort_host)) {
        fail(p, ECANCELED, "relay publication canceled by host abort");
        return false;
    }
    c.published = true;
    include_release_candidate(p, index);
    ++p.status.cpu_reductions;
    store(&c.host_control->generation, p.status.epoch);
    return true;
}

bool publish(void * opaque, const ggml_relay_event * event) {
    return publish_event(opaque, event);
}

bool publish_v2(void * opaque, const ggml_relay_event_v2 * event, size_t event_size) {
    if (event_size < sizeof(ggml_relay_event_v2)) {
        return false;
    }
    return publish_event(opaque, event);
}

bool snapshot(void * opaque, ggml_relay_status * status) {
    if (!opaque || !status) {
        return false;
    }
    auto &                       p = *static_cast<provider *>(opaque);
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        return false;
    }
    observe(p);
    *status = p.status;
    return true;
}

void abort(void * opaque, int32_t error) {
    if (!opaque) {
        return;
    }
    auto & p = *static_cast<provider *>(opaque);
    // Owner lifetime is externally serialized against destroy. This pointer
    // remains immutable while native work can exist; never wait for a HIP
    // launch/JIT/EventRecord holder before releasing device-side waits.
    if (p.abort_host) {
        store(p.abort_host, 1);
    }
    int32_t expected = 0;
    p.pending_abort.compare_exchange_strong(expected, error ? error : ECANCELED, std::memory_order_release,
                                            std::memory_order_relaxed);
    p.event_status.store(-1, std::memory_order_release);
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    if (lock.owns_lock()) {
        consume_abort(p);
    }
}

bool destroy(void * opaque) {
    if (!opaque) {
        return true;
    }
    auto *                       p = static_cast<provider *>(opaque);
    std::unique_lock<std::mutex> lock(p->mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        return false;
    }
    if (p->active || !idle(*p)) {
        fail(*p, EBUSY, "relay destroy cannot prove CPU and native completion; retaining owners");
        return false;
    }
    if (!release_owners(*p)) {
        return false;
    }
    lock.unlock();
    delete p;
    return true;
}

const ggml_relay_provider_i interface = {
    GGML_RELAY_PROVIDER_ABI_VERSION,
    create,
    destroy,
    prepare,
    submit,
    poll,
    publish,
    snapshot,
    abort,
};

const ggml_relay_wire_provider_i wire_interface = {
    GGML_RELAY_WIRE_PROVIDER_ABI_VERSION,
    set_wire_type,
    poll_v2,
    publish_v2,
};
}  // namespace

const ggml_relay_provider_i * ggml_backend_hip_relay_get_interface() {
    return &interface;
}

const ggml_relay_wire_provider_i * ggml_backend_hip_relay_get_wire_interface() {
    return &wire_interface;
}
