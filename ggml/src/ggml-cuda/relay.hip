#include "common.cuh"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cuda.h"
#include "ggml-relay-plan.h"
#include "ggml-relay-verify.h"
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
#include <unordered_map>
#include <vector>

namespace {
constexpr uint32_t ranks = GGML_RELAY_RANKS;
// Strict graph submission, relay kernels and replay use the owned stream zero.
// Keep native-tail evidence for that actual lane only; HIP hardware queue
// assignment remains a runtime property, not a promise made by this count.
constexpr uint32_t lanes_per_rank = 1;
static_assert(std::atomic<int32_t>::is_always_lock_free,
              "relay abort's status atomic must stay lock-free");
static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "relay abort's claim atomic must stay lock-free");
// Bit 63 is the sticky abort claim; bits 0..62 are the host-success counter.
// Both abort() and the host success paths use compare-exchange on this one
// atomic, so its modification order is the abort/success linearization order.
constexpr uint64_t abort_claim_bit = uint64_t(1) << 63;
constexpr uint32_t all_ranks        = (1u << ranks) - 1;
// Swift needs 36 recurrent-output, 48 expert-output and 97 HC joins. Reserve a
// fixed bound before submission; unused slots never become active transfers.
// Pillar 5 keeps the same 256-slot bound but lays the payload out per plan.
constexpr size_t   channel_capacity = ggml_relay_plan::channel_capacity;
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
static_assert(GGML_RELAY_TOKEN_SUCCESS == 0 && GGML_RELAY_TOKEN_FAILED == 1 && GGML_RELAY_TOKEN_UNKNOWN == 2,
              "relay token results must match the generic completion result values");
static_assert(std::atomic<uint32_t>::is_always_lock_free, "relay priority leases require a lock-free host atomic");
static_assert(std::atomic<size_t>::is_always_lock_free, "relay retirement queue requires a lock-free host atomic");

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
// Legacy fixed layout. Pillar 5 replaces it with a per-plan exact layout and a
// 68 MiB budget; the legacy value remains the GGML_TP5_POOL_EXACT=0 behavior.
constexpr size_t pool_bytes_legacy = ggml_relay_plan::pool_bytes_legacy;
static_assert(sizeof(control) == ggml_relay_plan::control_bytes,
              "relay control size must match the plan-policy control bytes");
static_assert(ggml_relay_plan::control_bytes % ggml_relay_plan::pool_alignment == 0,
              "relay control size must preserve the exact-pool alignment");
static_assert(alignof(control) <= ggml_relay_plan::pool_alignment &&
                  ggml_relay_plan::pool_alignment % alignof(control) == 0,
              "exact-pool alignment must preserve control alignment");

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
    size_t        capacity                       = 0; // F32 elements per rank in this slot.
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
    uint64_t                  uid = 0;             // graph uid this frame was captured for
    uint64_t                  plan_generation = 0; // plan that owned the bindings
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
    // Single lock-free abort/success claim. abort() sets the sticky bit with a
    // CAS; finish()/publish() increment the success counter with a CAS only
    // while the bit is clear. The CAS modification order decides whether an
    // abort is before or after a host success publication. snapshot() uses a
    // no-op CAS to linearize its abort read. No thread waits on a lock here.
    std::atomic<uint64_t>                  abort_claim{ 0 };
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
    size_t                                pool_bytes          = 0;
    size_t                                pool_budget         = pool_bytes_legacy;
    bool                                  pool_exact          = false;
    ggml_relay_plan::pool_layout          pool_layout;
    word *                                abort_host          = nullptr;
    word *                                abort_device[ranks] = {};
    word *                                activation_host = nullptr;
    word *                                activation_device[ranks] = {};
    std::array<frame, ggml_relay_plan::frame_capacity_max> frames;
    ggml_relay_plan::frame_lru             frame_lru{ 2 };
    ggml_relay_plan::plan_cache            plan_cache;
    // Exact per-slot terminal binding snapshot. The key's digest selects the
    // slot; this snapshot proves a hit before its generation is reused.
    std::array<ggml_relay_plan::binding_set, ggml_relay_plan::plan_capacity> plan_bindings;
    bool                                   plan_cache_enabled = false;
    bool                                   plan_verify_uid = false;
    bool                                   plan_stats_enabled = false;
    bool                                   plan_flush_each_prepare = false;
    bool                                   plan_pinned = false;
    bool                                   plan_cache_generation = false;
    bool                                   plan_key_absent_logged = false;
    uint64_t                               plan_epoch = 0;
    // A normal prepare retires at most one plan (the single LRU victim); the
    // diagnostic PLAN_FLUSH path can retire up to eight. The 16-entry queue is
    // burst headroom for that diagnostic path plus pending retirements.
    // Overflow is a hard failure: the provider must not drop a retirement
    // fact, because the runtime's PLAN_RETIRED handler is the only place that
    // proves no live reader still owns that generation.
    std::array<uint64_t, 2 * ggml_relay_plan::plan_capacity> retired_plans{};
    std::atomic<size_t>                    retired_count{ 0 };
    uint64_t                               pool_reallocations = 0;
    uint64_t                               frames_cleared_for_pool = 0;
    // Frame-cache evidence for the create/destroy gate. A hit is a frame
    // reuse: the current frame still serves an unchanged definition, or a
    // signature/uid match selected a cached frame. A miss installs a new
    // frame; an instantiate is one rank's hipGraphInstantiate; a destroy is
    // one frame teardown.
    uint64_t                               frame_hits = 0;
    uint64_t                               frame_misses = 0;
    uint64_t                               frame_evictions = 0;
    uint64_t                               frame_instantiates = 0;
    uint64_t                               frame_destroys = 0;
    // Verification evidence for the same gate. Requested counts submissions
    // that asked for verification, resolved those whose uid fast path found
    // the frame to verify, executed the describes that actually ran.
    ggml_relay_verify::ledger              verify;
    std::vector<unsigned char>            signature;
    std::vector<const ggml_tensor *>       tensors;
    std::vector<ggml_tensor *>             tensor_keys;
    std::vector<ggml_bitset_t>             tensor_used;
    std::vector<uint64_t>                  stage_nodes;
    frame *                               current = nullptr;
    bool                                  definition_changed = true;
    submission                            mode = submission::direct;
    ggml_relay_status                     status              = {};
    // The current error text is only the host abort. The first real error
    // after it may append its operation text without changing the code.
    bool                                  abort_attributed    = false;
    // Per-submission completion tokens. begin() creates one record when
    // submit_token asked for it; finish() attaches the native tails; only
    // release_token retires the record.
    uint64_t                              next_token          = 0;
    uint64_t                              active_token        = 0;
    bool                                  request_token       = false;
    // token -> {state: 0 active, 1 success, 2 failed, error}
    std::unordered_map<uint64_t, std::pair<int, int>> token_state;
    std::unordered_map<uint64_t, std::array<hipEvent_t, ranks * lanes_per_rank>> token_tails;
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

// Read the abort/success claim with a no-op read-modify-write. A plain load
// could read an older value without a happens-before edge; the RMW reads the
// modification order and is the linearization point for snapshot().
uint64_t read_abort_claim(provider & p) noexcept {
    uint64_t state = p.abort_claim.load(std::memory_order_relaxed);
    while (!p.abort_claim.compare_exchange_weak(state, state,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
    }
    return state;
}

// Read the pending abort error with a no-op RMW. Used before prepare/submit
// mutations and before begin() pins a plan, so an abort request that is
// ordered before the check fails closed before any provider mutation.
int32_t read_pending_abort(provider & p) noexcept {
    int32_t state = p.pending_abort.load(std::memory_order_relaxed);
    while (!p.pending_abort.compare_exchange_weak(state, state,
            std::memory_order_acq_rel, std::memory_order_acquire)) {
    }
    return state;
}

void record_failure(provider & p, int code, const char * message) {
    p.status.failed     = true;
    p.status.error_code = code ? code : EIO;
    snprintf(p.status.error, sizeof(p.status.error), "%s", message);
}

// A host abort is already the primary cause. Do not let its own recording
// append a second "native" detail to itself.
void fail_abort(provider & p, int32_t error) {
    p.event_status.store(-1, std::memory_order_release);
    if (!p.status.failed) {
        record_failure(p, error ? error : ECANCELED, "relay aborted by host");
        p.abort_attributed = true;
    }
    if (p.abort_host) {
        store(p.abort_host, 1);
    }
}

void fail(provider & p, int code, const char * message) {
    p.event_status.store(-1, std::memory_order_release);
    if (!p.status.failed) {
        if ((read_abort_claim(p) & abort_claim_bit) != 0) {
            // The host abort keeps the code, but the operation and HIP error
            // string still reach the caller. A real failure after an abort is
            // evidence, not noise.
            const int32_t error = p.pending_abort.load(std::memory_order_acquire);
            p.status.failed     = true;
            p.status.error_code = error ? error : (code ? code : ECANCELED);
            if (code != 0 && message != nullptr) {
                snprintf(p.status.error, sizeof(p.status.error), "relay aborted by host; first failure: %s", message);
            } else {
                snprintf(p.status.error, sizeof(p.status.error), "%s", "relay aborted by host");
            }
        } else {
            // A pending error that has not claimed the sticky bit is still an
            // in-progress abort; a concurrent non-abort failure keeps its own
            // code/message.
            record_failure(p, code, message);
        }
        p.abort_attributed = false;
    } else if (p.abort_attributed && code != 0 && message != nullptr) {
        // The host abort won the first-error race; keep its code and append
        // the first real operation text.
        snprintf(p.status.error, sizeof(p.status.error), "relay aborted by host; first failure: %s", message);
        p.abort_attributed = false;
    }
    if (p.abort_host) {
        store(p.abort_host, 1);
    }
}

// Local ordering/lifecycle failures are real, but they are not the first
// native operation error. Record one only when it is first; otherwise keep
// the existing code/text and log the secondary reason.
void fail_local(provider & p, int code, const char * message) {
    p.event_status.store(-1, std::memory_order_release);
    if (!p.status.failed) {
        const bool claimed = (read_abort_claim(p) & abort_claim_bit) != 0;
        const int32_t pending = claimed ? p.pending_abort.load(std::memory_order_acquire)
                                        : read_pending_abort(p);
        if (claimed || pending) {
            // The host abort owns the first-failure slot even when a local
            // ordering error raced into this window.
            fail_abort(p, pending);
            GGML_LOG_ERROR("HIP relay secondary failure: %s\n", message);
        } else {
            record_failure(p, code, message);
            p.abort_attributed = false;
        }
    } else {
        GGML_LOG_ERROR("HIP relay secondary failure: %s\n", message);
    }
    if (p.abort_host) {
        store(p.abort_host, 1);
    }
}

// A retryable local rejection. No provider state changed, so do not poison
// the epoch or consume the first-failure slot. The message is diagnostic.
void reject_local(const char * message) {
    GGML_LOG_ERROR("HIP relay rejected: %s\n", message);
}

void consume_abort(provider & p) {
    if ((p.abort_claim.load(std::memory_order_acquire) & abort_claim_bit) == 0) {
        return;
    }
    const int32_t error = p.pending_abort.load(std::memory_order_acquire);
    fail_abort(p, error);
}

// True when an abort is already claimed or its pending error is published.
// The two no-op RMW reads give the check a modification-order point. This is
// called before prepare/submit mutations and before begin() pins a plan.
bool abort_requested(provider & p) {
    if (read_abort_claim(p) & abort_claim_bit) {
        consume_abort(p);
        return true;
    }
    const int32_t pending = read_pending_abort(p);
    if (pending) {
        fail_abort(p, pending);
        return true;
    }
    return false;
}

// Cheap fail-closed check for hot submission and polling paths. A plain
// acquire load can miss an abort that becomes visible immediately after it;
// the success claim CAS still decides that race. Cold entry points use
// abort_requested(), whose no-op RMW reads the latest modification order.
bool abort_visible(provider & p) {
    if (p.abort_claim.load(std::memory_order_acquire) & abort_claim_bit) {
        consume_abort(p);
        return true;
    }
    const int32_t pending = p.pending_abort.load(std::memory_order_acquire);
    if (pending) {
        fail_abort(p, pending);
        return true;
    }
    return false;
}

// Claim one host success (finish or publish). The CAS on abort_claim is the
// linearization point: if abort set the sticky bit first, the CAS reads it and
// fails; if this CAS increments first, abort's later CAS sets the bit after
// this success. The 63-bit counter cannot wrap before 2^63 successes; fail
// closed if it would.
bool claim_host_success(provider & p) {
    uint64_t state = p.abort_claim.load(std::memory_order_relaxed);
    for (;;) {
        if (state & abort_claim_bit) {
            return false;
        }
        if ((state & ~abort_claim_bit) == ~abort_claim_bit) {
            fail(p, EOVERFLOW, "relay abort claim counter exhausted");
            return false;
        }
        if (p.abort_claim.compare_exchange_weak(state, state + 1,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            return true;
        }
    }
}

// Claim the sticky abort bit with one fetch_or RMW. The pending-error CAS is
// a read-modify-write: the first abort stores its code, later aborts read it
// with acquire, and the fetch_or release publishes that first error to any
// reader that sees the bit. fetch_or is a single RMW with no retry loop, so
// the claim does not wait for host lock holders or other aborts.
void claim_host_abort(provider & p, int32_t error) {
    int32_t expected = 0;
    p.pending_abort.compare_exchange_strong(expected, error ? error : ECANCELED,
                                            std::memory_order_release,
                                            std::memory_order_acquire);
    p.abort_claim.fetch_or(abort_claim_bit, std::memory_order_acq_rel);
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
    bool destroyed = false;
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
            destroyed = true;
        }
        if (f.graph[r]) {
            if (!checked(p, hipGraphDestroy(f.graph[r]), "hipGraphDestroy relay")) {
                ok = false;
                continue;
            }
            f.graph[r] = nullptr;
            destroyed = true;
        }
    }
    if (destroyed) {
        ++p.frame_destroys;
    }
    if (ok) {
        f.warmed = false;
        f.signature.clear();
        f.uid = 0;
        f.plan_generation = 0;
        std::fill(f.relay_nodes, f.relay_nodes + ranks, 0);
    }
    return ok;
}

bool free_channels(provider & p) {
    if (p.channel_pool && !checked(p, hipHostFree(p.channel_pool), "hipHostFree channel pool")) {
        return false;
    }
    p.channel_pool = nullptr;
    p.pool_bytes   = 0;
    p.pool_layout.clear();
    for (auto & c : p.channels) {
        c.host_control = nullptr;
        for (auto & inbox : c.inbox) {
            inbox = nullptr;
        }
        c.output   = nullptr;
        c.capacity = 0;
        for (uint32_t r = 0; r < ranks; ++r) {
            c.device_control[r] = nullptr;
            c.device_inbox[r]   = nullptr;
            c.device_output[r]  = nullptr;
        }
    }
    return true;
}

bool release_owners(provider & p) {
    // Executables retain kernel arguments into every provider allocation.
    // Destroy them only after native/CPU drain and before freeing any storage.
    for (const auto & r : p.rank) {
        if (r.capturing) {
            fail_local(p, EBUSY, "relay stream capture termination is unproved; retaining owners");
            return false;
        }
    }
    for (auto & f : p.frames) {
        if (!clear_frame(p, f)) {
            return false;
        }
    }
    p.frame_lru = ggml_relay_plan::frame_lru(p.frame_lru.capacity());
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
                const char * kind = "unknown device error";
                int          code = EIO;
                if (error == 1) {
                    kind = "abort";
                    code = ECANCELED;
                } else if (error == 2) {
                    kind = "generation mismatch";
                    code = EPROTO;
                } else if (error == 3) {
                    kind = "wait timeout";
                    code = ETIMEDOUT;
                }
                std::snprintf(detail, sizeof(detail),
                              "HIP relay %s channel=%zu rank=%u epoch=%llu device_code=%llu",
                              kind, i, r, static_cast<unsigned long long>(p.status.epoch),
                              static_cast<unsigned long long>(error));
                fail(p, code, detail);
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

// Digest of every fact same_binding() compares. Two prepares with the same
// coarse llama key but different terminal storage or layout therefore get
// different plan slots instead of failing the key-consistency check. FNV-1a
// is enough here: this is a cache key, not a security boundary, and a
// collision still falls back to the fail-closed same_binding check.
uint64_t binding_fingerprint(const ggml_tensor * const * terminals, size_t count) {
    uint64_t hash = 1469598103934665603ull;
    const auto mix = [&hash](uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ull;
    };
    for (size_t c = 0; c < count; ++c) {
        for (uint32_t r = 0; r < ranks; ++r) {
            const auto * t = terminals[c * ranks + r];
            mix(uint64_t(reinterpret_cast<uintptr_t>(t)));
            mix(uint64_t(reinterpret_cast<uintptr_t>(t->data)));
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                mix(uint64_t(t->ne[d]));
                mix(uint64_t(t->nb[d]));
            }
        }
    }
    mix(uint64_t(count));
    return hash != 0 ? hash : 1;
}

bool allocate_channel_pool(provider & p, size_t bytes) {
    if (!select(p, 0) ||
        !checked(p, hipHostMalloc(&p.channel_pool, bytes, hipHostMallocMapped | hipHostMallocPortable),
                 "hipHostMalloc channel pool")) {
        return false;
    }
    // Record the live allocation before capability checks: a failed check
    // still leaves this block owned by the provider until free_channels().
    p.pool_bytes = bytes;
    // The exact layout aligns every control block relative to this base, so
    // the base itself must satisfy the same bound. hipHostMalloc normally
    // returns page-aligned storage; fail closed instead of assuming it.
    if (reinterpret_cast<uintptr_t>(p.channel_pool) % ggml_relay_plan::pool_alignment != 0) {
        fail(p, EINVAL, "relay channel pool is not 64-byte aligned");
        return false;
    }
    unsigned flags = 0;
    if (!checked(p, hipHostGetFlags(&flags, p.channel_pool), "hipHostGetFlags channel pool") ||
        !(flags & hipHostMallocMapped)) {
        fail(p, ENOTSUP, "relay channel pool is not mapped");
        return false;
    }
    return true;
}

// Bind every used channel to its offset in the current layout. The layout is
// frozen while a frame exists, so these pointers stay valid for every
// captured graph.
bool bind_channels(provider & p, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        auto & c       = p.channels[i];
        auto * host    = static_cast<char *>(p.channel_pool) + p.pool_layout.offset[i];
        c.host_control = reinterpret_cast<control *>(host);
        if (reinterpret_cast<uintptr_t>(c.host_control) % ggml_relay_plan::pool_alignment != 0) {
            fail(p, EINVAL, "relay host control is not 64-byte aligned");
            return false;
        }
        memset(c.host_control, 0, sizeof(control));
        auto * payload = host + sizeof(control);
        const size_t capacity = p.pool_layout.capacity[i];
        for (uint32_t r = 0; r < ranks; ++r) {
            c.inbox[r] = payload + capacity * sizeof(float) * r;
        }
        c.output   = payload + capacity * sizeof(float) * ranks;
        c.capacity = capacity;
        // A grown layout hands out fresh host memory. The zero-inbox claim
        // belongs to the old payload and must be rebuilt by prepare.
        c.zero_inbox_mask = 0;
        c.released = true;
    }
    for (uint32_t r = 0; r < ranks; ++r) {
        void * mapped = nullptr;
        if (!select(p, r) ||
            !checked(p, hipHostGetDevicePointer(&mapped, p.channel_pool, 0), "hipHostGetDevicePointer channel pool")) {
            return false;
        }
        for (size_t i = 0; i < count; ++i) {
            auto & c            = p.channels[i];
            auto * device       = static_cast<char *>(mapped) + p.pool_layout.offset[i];
            c.device_control[r] = reinterpret_cast<control *>(device);
            if (reinterpret_cast<uintptr_t>(c.device_control[r]) % ggml_relay_plan::pool_alignment != 0) {
                fail(p, EINVAL, "relay device control is not 64-byte aligned");
                return false;
            }
            auto * payload      = device + sizeof(control);
            const size_t capacity = p.pool_layout.capacity[i];
            c.device_inbox[r]   = payload + capacity * sizeof(float) * r;
            c.device_output[r]  = payload + capacity * sizeof(float) * ranks;
        }
    }
    return true;
}

bool reserve_channels_legacy(provider & p) {
    if (!allocate_channel_pool(p, pool_bytes_legacy)) {
        return false;
    }
    p.pool_layout.clear();
    for (size_t i = 0; i < channel_capacity; ++i) {
        p.pool_layout.offset[i]   = i * channel_stride;
        p.pool_layout.capacity[i] = element_capacity;
    }
    p.pool_layout.bytes = pool_bytes_legacy;
    p.pool_layout.count = channel_capacity;
    return bind_channels(p, channel_capacity);
}

// Exact layout: create() allocated the 8 MiB baseline. Each prepare computes
// the real per-channel need and grows within the 68 MiB budget when the
// current layout cannot cover the plan. A growth clears every captured frame,
// because a captured graph holds the old device pointers; prepare is idle, so
// those frames have no native or CPU user. Everything else fails closed.
bool reserve_channels_exact(provider & p, const size_t * elements, size_t count) {
    size_t need = 0;
    const ggml_relay_plan::pool_growth decision =
        ggml_relay_plan::pool_growth_decision(p.pool_layout, elements, count, ranks, p.pool_budget, need);
    if (decision == ggml_relay_plan::pool_growth::fail_budget) {
        char detail[sizeof(p.status.error)];
        std::snprintf(detail, sizeof(detail),
                      "relay exact pool needs %zu bytes for %zu channels, budget is %zu; set GGML_TP5_POOL_EXACT=0",
                      need, count, p.pool_budget);
        fail(p, ENOSPC, detail);
        return false;
    }
    if (decision == ggml_relay_plan::pool_growth::grow) {
        // The layout is about to change, so every captured frame's channel
        // pointers are stale even when the allocation itself does not move.
        // prepare() is idle, so every captured frame is idle too. Clear them;
        // the next submit recaptures the shapes it needs.
        for (size_t i = 0; i < p.frame_lru.capacity(); ++i) {
            const bool was_valid = p.frame_lru.valid(static_cast<int>(i));
            if (!clear_frame(p, p.frames[i])) {
                return false;
            }
            if (was_valid) {
                ++p.frames_cleared_for_pool;
            }
            p.frame_lru.clear(static_cast<int>(i));
            if (p.current == &p.frames[i]) {
                p.current = nullptr;
            }
        }
        if (need > p.pool_bytes) {
            if (p.channel_pool && !free_channels(p)) {
                return false;
            }
            if (!allocate_channel_pool(p, need)) {
                return false;
            }
            ++p.pool_reallocations;
            GGML_LOG_INFO("HIP relay exact pool: %zu bytes (%.2f MiB) for %zu channels, %llu reallocation(s)\n",
                          p.pool_bytes, double(p.pool_bytes) / (1024.0 * 1024.0), count,
                          static_cast<unsigned long long>(p.pool_reallocations));
        }
        if (!ggml_relay_plan::compute_pool_layout(elements, count, ranks, p.pool_bytes, p.pool_layout)) {
            fail(p, ENOSPC, "relay exact pool layout does not fit the allocated pool");
            return false;
        }
    }
    if (decision == ggml_relay_plan::pool_growth::keep) {
        return true; // The existing layout and pointers still cover the plan.
    }
    return bind_channels(p, count);
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
    // Pillar 5 switches. Every switch defaults to the legacy behavior; an
    // illegal value fails the whole provider instead of clamping.
    ggml_relay_plan::config cfg;
    char                    cfg_error[160];
    if (!ggml_relay_plan::parse_config(
            std::getenv("GGML_TP5_PLAN_UNION"), std::getenv("GGML_TP5_RELAY_FRAMES"),
            std::getenv("GGML_TP5_POOL_EXACT"), std::getenv("GGML_TP5_RELAY_SLOT_CHANNELS"),
            std::getenv("GGML_TP5_RELAY_VERIFY_UID"), std::getenv("GGML_TP5_RELAY_PLAN_STATS"),
            std::getenv("GGML_TP5_GRAPH_UID"), std::getenv("GGML_TP5_RELAY_PLAN_FLUSH"),
            cfg, cfg_error, sizeof(cfg_error))) {
        GGML_LOG_ERROR("HIP relay switch matrix rejected: %s\n", cfg_error);
        delete p;
        return nullptr;
    }
    p->plan_cache_enabled       = cfg.plan_union;
    p->plan_verify_uid          = cfg.verify_uid;
    p->plan_stats_enabled       = cfg.plan_stats;
    p->plan_flush_each_prepare  = cfg.plan_flush;
    p->pool_exact               = cfg.pool_exact;
    p->pool_budget              = cfg.pool_exact ? ggml_relay_plan::pool_budget_default : pool_bytes_legacy;
    p->frame_lru                = ggml_relay_plan::frame_lru(cfg.frames);
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
        ok = p->pool_exact ? allocate_channel_pool(*p, ggml_relay_plan::pool_baseline)
                           : reserve_channels_legacy(*p);
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
    GGML_LOG_INFO("HIP relay pillar5 provider=%p plan_cache=%d frames=%zu pool=%s bytes=%zu budget=%zu "
                  "graph_uid=%d slot_channels=%d key_digest=1\n",
                  static_cast<void *>(p), int(p->plan_cache_enabled), p->frame_lru.capacity(),
                  p->pool_exact ? "exact" : "legacy", p->pool_bytes, p->pool_budget, int(cfg.graph_uid),
                  int(cfg.slot_channels));
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

void log_plan_stats(provider & p, const char * where) {
    if (!p.plan_stats_enabled) {
        return;
    }
    const auto & s = p.plan_cache.stats();
    GGML_LOG_INFO("TP5_PLAN_STATS %s hits=%llu misses=%llu inserts=%llu evictions=%llu pinned=%llu "
                  "flushes=%llu stale=%llu live=%zu pool=%zu reallocs=%llu frames_cleared=%llu frames=%zu "
                  "frame_hits=%llu frame_misses=%llu frame_evict=%llu frame_inst=%llu frame_destroy=%llu "
                  "verify_req=%llu verify_res=%llu verify_exec=%llu verify_ok=%d\n",
                  where,
                  static_cast<unsigned long long>(s.hits), static_cast<unsigned long long>(s.misses),
                  static_cast<unsigned long long>(s.inserts), static_cast<unsigned long long>(s.evictions),
                  static_cast<unsigned long long>(s.pinned_refusals),
                  static_cast<unsigned long long>(s.flushes),
                  static_cast<unsigned long long>(s.stale_generation),
                  p.plan_cache.live(), p.pool_bytes,
                  static_cast<unsigned long long>(p.pool_reallocations),
                  static_cast<unsigned long long>(p.frames_cleared_for_pool),
                  p.frame_lru.used(),
                  static_cast<unsigned long long>(p.frame_hits),
                  static_cast<unsigned long long>(p.frame_misses),
                  static_cast<unsigned long long>(p.frame_evictions),
                  static_cast<unsigned long long>(p.frame_instantiates),
                  static_cast<unsigned long long>(p.frame_destroys),
                  static_cast<unsigned long long>(p.verify.total.requested),
                  static_cast<unsigned long long>(p.verify.total.resolved),
                  static_cast<unsigned long long>(p.verify.total.executed),
                  int(p.verify.consistent()));
}

// Queue one retirement fact. Called with p.mutex held. A normal prepare
// queues at most one fact; the diagnostic PLAN_FLUSH path can queue up to
// eight. The 16-entry queue is burst headroom for that path plus pending
// retirements. If it is still full, the provider cannot prove the runtime
// observed the fact, so it fails closed instead of dropping the fact: a
// silent drop would lose the runtime's immediate "retirement has no live
// reader" assertion and could mask a reader leak until slot exhaustion. The
// failure is visible regardless of the stats switch. This queue-full branch is
// not exercised by a test; normal runtime operation cannot reach it because
// the runtime drains every epoch and prepare retires at most one plan.
bool queue_retirement(provider & p, uint64_t generation) {
    const size_t count = p.retired_count.load(std::memory_order_relaxed);
    if (count == p.retired_plans.size()) {
        fail_local(p, ENOSPC, "relay retirement queue overflow: the runtime did not drain PLAN_RETIRED");
        return false;
    }
    p.retired_plans[count] = generation;
    p.retired_count.store(count + 1, std::memory_order_relaxed);
    return true;
}

// Drop every frame captured for one plan generation and queue its retirement
// event. prepare() is idle, so no frame here has a native or CPU user.
bool retire_plan(provider & p, uint64_t generation, bool emit_event) {
    if (generation == 0) {
        return true;
    }
    for (size_t i = 0; i < p.frame_lru.capacity(); ++i) {
        const int index = static_cast<int>(i);
        if (!p.frame_lru.valid(index) || p.frame_lru.plan_generation(index) != generation) {
            continue;
        }
        if (!clear_frame(p, p.frames[i])) {
            return false;
        }
        p.frame_lru.clear(index);
        if (p.current == &p.frames[i]) {
            p.current = nullptr;
        }
    }
    if (emit_event && !queue_retirement(p, generation)) {
        return false;
    }
    return true;
}

// Flush every plan. Refused while any plan is pinned, because native work may
// still reference its bindings. Returns false after failing the provider.
bool flush_all_plans(provider & p, bool emit_events) {
    std::array<uint64_t, ggml_relay_plan::plan_capacity> generations{};
    size_t count = 0;
    for (size_t i = 0; i < ggml_relay_plan::plan_capacity; ++i) {
        if (const auto * slot = p.plan_cache.at(i)) {
            generations[count++] = slot->generation;
        }
    }
    if (!p.plan_cache.flush_all()) {
        fail_local(p, EBUSY, "relay plan flush refused: a plan is pinned");
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!retire_plan(p, generations[i], emit_events)) {
            return false;
        }
    }
    return true;
}

// Bind the plan identity to a generation. `same_plan` is the full binding
// validation result; a key hit with different bindings is a key bug and fails
// closed instead of silently sharing a generation.
bool plan_generation_for(provider & p, const ggml_relay_plan_key * key, size_t count, bool same_plan,
                         const ggml_tensor * const * terminals, uint64_t binding, uint64_t & generation) {
    if (!p.plan_cache_enabled || key == nullptr) {
        p.plan_cache_generation = false;
        if (p.plan_cache_enabled && !p.plan_key_absent_logged) {
            // The cache is on but this caller cannot describe plan identity.
            // Keep the legacy monotonic generation rather than guessing, and
            // say so once: otherwise the cache silently never fills.
            GGML_LOG_WARN("HIP relay plan cache is enabled but prepare has no plan key; "
                          "using the legacy generation rule (reuse an unchanged binding, "
                          "otherwise increment)\n");
            p.plan_key_absent_logged = true;
        }
        if (same_plan) {
            generation = p.status.plan_generation;
            return true;
        }
        if (p.status.plan_generation == UINT64_MAX) {
            fail(p, EOVERFLOW, "relay plan generation overflow");
            return false;
        }
        generation = ++p.status.plan_generation;
        return true;
    }
    ggml_relay_plan::plan_key pk = ggml_relay_plan::plan_key::from_c(*key);
    pk.channels = uint32_t(count);
    pk.binding  = binding;
    // A key hit is a plan-identity candidate: the key carries the terminal
    // binding digest, and the slot's exact binding snapshot is the authority.
    // same_plan only says whether the provider's CURRENT channels already
    // match this prepare; after a tail switch (180 <-> 181 channels) they
    // describe the other plan, so same_plan=false is not a key bug. A digest
    // collision with a different real binding is caught here and fails
    // closed; a genuinely different binding normally has a different digest
    // and is a miss.
    const auto * existing = p.plan_cache.find(pk);
    if (existing != nullptr &&
        !ggml_relay_plan::binding_matches(p.plan_bindings[existing->index], terminals, count)) {
        fail(p, EPROTO, "relay plan key collided with a different terminal binding");
        return false;
    }
    uint64_t evicted = 0;
    ggml_relay_plan::plan_slot * slot = p.plan_cache.acquire(pk, p.plan_epoch, &evicted);
    if (slot == nullptr) {
        fail(p, ENOSPC, "relay plan cache exhausted: every slot is pinned");
        return false;
    }
    if (existing == nullptr) {
        // Write the new binding before retiring the evicted generation: if a
        // native teardown fails, the slot's key and snapshot still describe
        // the same plan and a later diagnostic cannot report a false hit.
        ggml_relay_plan::snapshot_binding(p.plan_bindings[slot->index], terminals, count);
    }
    if (evicted != 0 && !retire_plan(p, evicted, true)) {
        return false;
    }
    p.plan_cache_generation = true;
    generation = slot->generation;
    return true;
}

bool prepare_impl(void * opaque, ggml_tensor * const * terminals, size_t count, const ggml_relay_plan_key * key) {
    auto &                      p = *static_cast<provider *>(opaque);
    std::lock_guard<std::mutex> lock(p.mutex);
    if (abort_requested(p)) {
        return false;
    }
    if (p.status.failed) {
        return false;
    }
    if (!idle(p)) {
        return false; // Busy is transient; the caller may retry after draining.
    }
    if (!p.token_tails.empty()) {
        reject_local("relay prepare requires every completion token released");
        return false;
    }
    if (!terminals || count == 0 || count > channel_capacity) {
        fail_local(p, ENOTSUP, "relay plan exceeds reserved 256-channel channel capacity or is empty");
        return false;
    }
    // Reject a contradictory key before touching the exact pool, the computed
    // masks, the plan cache or the wire format. Zero means "use the prepared
    // count".
    if (key != nullptr && key->channels != 0 && key->channels != count) {
        char detail[sizeof(p.status.error)];
        std::snprintf(detail, sizeof(detail), "relay plan key channel count %u differs from prepared channels %zu",
                      key->channels, count);
        fail_local(p, EPROTO, detail);
        return false;
    }
    p.wire_frozen = true; // Format is immutable before any captured graph can exist.
    // prepare announces a new owned definition even if its terminal bindings
    // are unchanged: intermediate topology, layouts or fusion may differ.
    p.definition_changed = true;
    ++p.plan_epoch;
    if (p.plan_flush_each_prepare && p.plan_cache_enabled && !flush_all_plans(p, true)) {
        return false;
    }
    bool same_plan = p.status.channels == count;
    std::array<size_t, channel_capacity> elements{};
    // Validate the entire next binding before changing any channel state or
    // inserting a plan. All mapped storage is either the legacy pool or the
    // exact layout below.
    for (size_t c = 0; c < count; ++c) {
        const auto * first = terminals[c * ranks];
        if (!first || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first) || ggml_nelements(first) <= 0) {
            fail_local(p, ENOTSUP, "relay requires nonempty contiguous F32 terminals");
            return false;
        }
        if (uint64_t(ggml_nelements(first)) > element_capacity) {
            char detail[sizeof(p.status.error)];
            std::snprintf(detail, sizeof(detail), "relay channel=%zu tensor=%s shape=%lld,%lld,%lld,%lld exceeds %zu-F32 capacity",
                          c, first->name, static_cast<long long>(first->ne[0]), static_cast<long long>(first->ne[1]),
                          static_cast<long long>(first->ne[2]), static_cast<long long>(first->ne[3]), element_capacity);
            fail_local(p, ENOTSUP, detail);
            return false;
        }
        elements[c] = size_t(ggml_nelements(first));
        for (uint32_t r = 0; r < ranks; ++r) {
            const auto * t = terminals[c * ranks + r];
            if (!t || !t->data || t->type != first->type || !ggml_is_contiguous(t) ||
                memcmp(t->ne, first->ne, sizeof(t->ne)) || memcmp(t->nb, first->nb, sizeof(t->nb))) {
                fail_local(p, EINVAL, "relay rank terminal shapes/types/strides differ");
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
                fail_local(p, EINVAL, "relay terminal is not rank-local native device memory");
                return false;
            }
        }
    }
    if (p.pool_exact) {
        // The exact layout must cover this plan before any inbox is cleared.
        if (!reserve_channels_exact(p, elements.data(), count)) {
            return false;
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
                memset(c.inbox[r], 0, c.capacity * wire_element_size(p));
                c.zero_inbox_mask |= bit;
            }
        }
    }
    uint64_t generation = 0;
    const uint64_t binding = (p.plan_cache_enabled && key != nullptr)
        ? binding_fingerprint(terminals, count) : 0;
    if (!plan_generation_for(p, key, count, same_plan, terminals, binding, generation)) {
        return false;
    }
    if (!same_plan) {
        for (size_t i = 0; i < count; ++i) {
            auto & c   = p.channels[i];
            c.elements = elements[i];
            c.bytes    = c.elements * wire_element_size(p);
            for (uint32_t r = 0; r < ranks; ++r) {
                bind(c, r, terminals[i * ranks + r]);
            }
            c.released = true;
        }
        p.status.channels = count;
    }
    // prepare leaves every channel released. Keep the CPU drain count in step
    // with the new plan so a snapshot or free in the prepare window sees a
    // settled provider, not an old count against new channels.
    p.outputs_released = p.status.channels;
    p.status.plan_generation = generation;
    if (p.status.epoch != 0 && (p.status.epoch % 64) == 0) {
        log_plan_stats(p, "prepare");
    }
    return true;
}

bool prepare(void * opaque, ggml_tensor * const * terminals, size_t count) {
    return prepare_impl(opaque, terminals, count, nullptr);
}

bool prepare_ex(void * opaque, ggml_tensor * const * terminals, size_t count, const ggml_relay_plan_key * key) {
    return prepare_impl(opaque, terminals, count, key);
}

bool begin(provider & p) {
    const bool want_token = p.request_token;
    p.request_token = false;
    if (abort_requested(p)) {
        return false;
    }
    if (p.status.failed) {
        return false;
    }
    if (!idle(p)) {
        return false; // Busy is transient; the caller may retry after draining.
    }
    if (p.status.channels == 0) {
        fail_local(p, EINVAL, "relay begin has no prepared channels");
        return false;
    }
    // A token record owns the lane tail event handles. Re-recording those
    // events for a new epoch would silently retarget wait_token, so every
    // token must be released before the next begin.
    if (!p.token_tails.empty()) {
        reject_local("relay begin requires every completion token released");
        return false;
    }
    if (p.status.epoch == UINT64_MAX) {
        fail(p, EOVERFLOW, "relay epoch overflow");
        return false;
    }
    if (want_token && p.next_token == UINT64_MAX) {
        fail(p, EOVERFLOW, "relay completion token counter exhausted");
        return false;
    }
    if (p.plan_cache_enabled && p.plan_cache_generation && p.status.plan_generation != 0) {
        if (!p.plan_cache.pin(p.status.plan_generation)) {
            fail_local(p, EPROTO, "relay begin could not pin its plan generation");
            return false;
        }
        p.plan_pinned = true;
    }
    // Create the token record before mutating the epoch. A failed allocation
    // must not leave a half-begun epoch or a pinned plan behind.
    uint64_t token = 0;
    if (want_token) {
        token = ++p.next_token;
        try {
            p.token_state[token] = {0, 0};
        } catch (const std::bad_alloc &) {
            if (p.plan_pinned) {
                p.plan_pinned = false;
                if (!p.plan_cache.release(p.status.plan_generation)) {
                    fail_local(p, EPROTO, "relay begin could not release its plan pin after token allocation failed");
                }
            }
            fail(p, ENOMEM, "relay begin could not record its completion token");
            return false;
        }
    }
    ++p.status.epoch;
    p.active_token = token;
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
    if (abort_visible(p)) {
        return;
    }
    if (!p.active || p.tails_attempted || p.status.failed || r >= ranks || p.rank[r].graph_pending) {
        fail_local(p, EINVAL, "invalid relay graph_begin ordering");
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
    if (abort_visible(p)) {
        return;
    }
    if (!p.active || p.tails_attempted || r >= ranks || !p.rank[r].graph_pending) {
        fail_local(p, EINVAL, "invalid relay graph_accepted ordering");
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
    if (abort_visible(p)) {
        return false;
    }
    if (!p.active || p.tails_attempted || p.status.failed || index != p.next_channel || index >= p.status.channels) {
        fail_local(p, EINVAL, "invalid relay channel enqueue ordering");
        return false;
    }
    auto &         c      = p.channels[index];
    for (uint32_t r = 0; r < ranks; ++r) {
        if (abort_visible(p)) {
            return false;
        }
        if (p.rank[r].graph_pending || p.rank[r].accepted_graphs != p.next_channel + 1 || !select(p, r)) {
            fail_local(p, EINVAL, "relay enqueue before accepted rank graph");
            return false;
        }
        auto * ctx = p.rank[r].context;
        if (ctx->curr_stream_no != 0) {
            fail_local(p, EINVAL, "relay rank must use its owned stream zero");
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
    // A pending or claimed abort denies new instantiate/launch work below.
    // Captures and tails must still be drained.
    abort_requested(p);
    if (!p.active || p.tails_attempted) {
        fail_local(p, EINVAL, "relay finish without an active unsealed epoch");
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
        // Observe an abort between the EndCapture and instantiate phases. The
        // EndCapture loop above must still end every capture, but no new
        // instantiate/launch work starts after an abort is visible here.
        abort_requested(p);
        for (uint32_t r = 0; r < ranks; ++r) {
            if (p.rank[r].accepted_graphs != p.status.channels + 1 || p.rank[r].graph_pending) {
                fail_local(p, EIO, "relay capture is missing stages");
            }
        }
        // No launch until all five complete executables exist. Failure leaves
        // partially created graph handles owned for checked destruction.
        abort_requested(p);
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
            uint32_t instantiated = 0;
            for (uint32_t r = 0; r < started; ++r) {
                const hipError_t result = jobs[r].get();
                if (result == hipSuccess) {
                    ++instantiated;
                }
                checked(p, result, "select/instantiate relay epoch worker");
            }
            // Count the instantiate calls that actually succeeded, even if an
            // abort arrives before the launch phase. This is a fact counter,
            // not a permission to launch.
            p.frame_instantiates += instantiated;
            // An abort during the blocking instantiate jobs is observed here;
            // the launch loop below must not start after it.
            abort_requested(p);
        }
    }
    if (p.mode != submission::direct && !p.status.failed) {
        for (uint32_t r = 0; r < ranks; ++r) {
            if (abort_requested(p) || p.status.failed || !select(p, r)) {
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
            fail_local(p, EIO, "relay frame is missing accepted stage or final rank graphs");
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
        fail_local(p, EIO, "relay frame has incomplete or unknown submissions");
    }
    p.active = false;
    abort_requested(p);
    if (p.plan_pinned) {
        p.plan_pinned = false;
        if (!p.plan_cache.release(p.status.plan_generation)) {
            fail_local(p, EPROTO, "relay finish could not release its plan pin");
        }
    }
    observe(p);
    // Attach this epoch's native tails to its completion token. A failed
    // token may still have native work in flight; wait_token synchronizes
    // these events before it reports FAILED.
    if (p.active_token != 0) {
        std::array<hipEvent_t, ranks * lanes_per_rank> tails = {};
        for (uint32_t r = 0; r < ranks; ++r) {
            for (uint32_t l = 0; l < lanes_per_rank; ++l) {
                tails[r * lanes_per_rank + l] = p.rank[r].lanes[l].tail;
            }
        }
        try {
            p.token_tails[p.active_token] = tails;
        } catch (const std::bad_alloc &) {
            fail(p, ENOMEM, "recording relay token tails failed");
        }
    }
    // The abort claim CAS is the host success linearization point. If abort
    // set the sticky bit first, this CAS reads it and fails; if this CAS
    // increments first, a later abort sets the bit after this success. A true
    // return is a point-in-time success, not immunity to a later abort.
    bool sealed = false;
    if (!p.status.failed) {
        p.status.sealed = true;
        p.event_status.store(1, std::memory_order_release);
        if (claim_host_success(p)) {
            sealed = true;
            if (p.mode == submission::direct) {
                p.current->warmed = true;
            }
        } else {
            p.status.sealed = false;
            p.event_status.store(-1, std::memory_order_release);
            consume_abort(p);
        }
    } else {
        p.status.sealed = false;
    }
    return sealed;
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
            fail_local(p, EINVAL, "relay replay requires its owned stream zero");
            return false;
        }
        describe(p, p.rank[r].physical, p.rank[r].scratch, p.rank[r].lanes[0].stream, ctx->quantized_matmul_f32);
    }
    for (size_t stage = 0; stage < stages; ++stage) {
        for (uint32_t r = 0; r < ranks; ++r) {
            const auto * g = graphs[stage * ranks + r];
            if (!g || g->n_nodes < 0 || g->n_leafs < 0 || g->grads || g->grad_accs) {
                fail_local(p, EINVAL, "relay replay requires complete inference graphs");
                return false;
            }
            describe(p, g->n_nodes, g->n_leafs, g->order);
            for (int i = 0; i < g->n_nodes; ++i) {
                const auto * t = g->nodes[i];
                if (!t) {
                    fail_local(p, EINVAL, "relay graph contains a null node");
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
                    fail_local(p, EINVAL, "relay graph contains a null leaf");
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

bool submit_internal(void * opaque, ggml_cgraph * const * graphs, size_t stages, bool request_token,
                     uint64_t * created_token = nullptr) {
    auto & p = *static_cast<provider *>(opaque);
    if (created_token != nullptr) {
        *created_token = 0;
    }
    const int64_t d0=ggml_time_us();
    std::lock_guard<std::mutex> lock(p.mutex);
    const int64_t dlock=ggml_time_us();
    if (abort_requested(p)) {
        return false;
    }
    if (p.status.failed) {
        return false;
    }
    if (!idle(p)) {
        return false; // Busy is transient; the caller may retry after draining.
    }
    if (!graphs || stages != p.status.channels + 1) {
        fail_local(p, EINVAL, "relay submit stage count does not match the prepared plan");
        return false;
    }
    const uint64_t uid = graphs[0] ? graphs[0]->uid : 0;
    const bool changed = p.definition_changed || !p.current;
    bool hit = !changed;
    uint64_t compute_nodes[ranks] = {};
    try {
        // uid fast path: the same graph object, plan generation and nonempty
        // signature means the captured executables are already current. The
        // optional verification still pays the O(nodes) describe and asserts
        // equality, so a wrong uid is caught instead of replayed.
        const bool fast = uid != 0 && p.current != nullptr && p.current->uid == uid &&
                          p.current->plan_generation == p.status.plan_generation &&
                          !p.current->signature.empty();
        if (fast) {
            hit = true;
        }
        // The shared helper owns the request -> resolve -> describe -> execute
        // order and the fast/switch gate. did_describe keeps the signature
        // check on the same branch as the verification describe.
        bool did_describe = false;
        if (!ggml_relay_verify::verify_submission(p.verify, p.plan_verify_uid, fast,
                [&]() {
                    return describe_frame(p, graphs, stages, compute_nodes);
                }, &did_describe)) {
            return false;
        }
        if (did_describe && p.current->signature != p.signature) {
            fail(p, EPROTO, "relay uid fast path signature mismatch");
            return false;
        }
        if (!fast && changed) {
            if (!describe_frame(p, graphs, stages, compute_nodes)) {
                return false;
            }
            // The previous current frame is not a reuse candidate for a new
            // signature, but it is still the most recent cache entry. Skip it
            // as an eviction victim like the design requires.
            const int previous_index = p.current != nullptr
                ? static_cast<int>(p.current - p.frames.data()) : -1;
            p.current = nullptr;
            int current_index = -1;
            for (size_t i = 0; i < p.frame_lru.capacity(); ++i) {
                if (p.frames[i].signature == p.signature) {
                    p.current = &p.frames[i];
                    current_index = static_cast<int>(i);
                    hit = true;
                    break;
                }
            }
            if (!p.current) {
                const int index = p.frame_lru.victim(previous_index);
                if (index < 0) {
                    fail(p, ENOSPC, "relay frame cache has no evictable slot");
                    return false;
                }
                auto & f = p.frames[static_cast<size_t>(index)];
                if (p.frame_lru.valid(index)) {
                    ++p.frame_evictions;
                }
                // idle above proves this old executable has no native or CPU users.
                if (!clear_frame(p, f)) {
                    return false;
                }
                p.frame_lru.clear(index);
                f.signature.swap(p.signature);
                std::copy(compute_nodes, compute_nodes + ranks, f.compute_nodes);
                p.current = &f;
                current_index = index;
                hit = false;
            }
            p.frame_lru.assign(current_index, uid, p.status.plan_generation, p.status.epoch);
        }
        p.definition_changed = false;
        if (hit) {
            ++p.frame_hits;
        } else {
            ++p.frame_misses;
        }
    } catch (const std::bad_alloc &) {
        fail(p, ENOMEM, "allocating relay frame signature failed");
        return false;
    }
    p.current->uid = uid;
    p.current->plan_generation = p.status.plan_generation;
    {
        const int index = static_cast<int>(p.current - p.frames.data());
        if (!p.frame_lru.valid(index) || p.frame_lru.uid(index) != uid ||
            p.frame_lru.plan_generation(index) != p.status.plan_generation) {
            p.frame_lru.assign(index, uid, p.status.plan_generation, p.status.epoch);
        } else {
            p.frame_lru.touch(index, p.status.epoch);
        }
    }
    p.mode = p.current->exec[0] ? submission::replay :
        (p.current->warmed ? submission::capture : submission::direct);
    const int64_t dkey=ggml_time_us();
    p.request_token = request_token;
    if (!begin(p)) {
        return false;
    }
    if (created_token != nullptr) {
        *created_token = p.active_token;
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

bool submit(void * opaque, ggml_cgraph * const * graphs, size_t stages) {
    return submit_internal(opaque, graphs, stages, false);
}

void settle_token(provider & p);

bool submit_token(void * opaque, ggml_cgraph * const * graphs, size_t stages, uint64_t * token) {
    if (opaque == nullptr || token == nullptr) {
        return false;
    }
    *token = 0;
    auto & p = *static_cast<provider *>(opaque);
    auto cleanup = [&p](uint64_t created) {
        if (created != 0) {
            p.token_state.erase(created);
            p.token_tails.erase(created);
            if (p.active_token == created) {
                p.active_token = 0;
            }
        }
    };
    uint64_t created = 0;
    bool ok = false;
    try {
        ok = submit_internal(opaque, graphs, stages, true, &created);
    } catch (...) {
        // The provider mutex is released by unwinding. Erase only the record
        // this call created; a previous epoch's token must stay.
        std::lock_guard<std::mutex> lock(p.mutex);
        cleanup(created);
        throw;
    }
    std::lock_guard<std::mutex> lock(p.mutex);
    if (!ok) {
        // begin() may have created a record before a later stage failed. Erase
        // only that record; a previous epoch's unreleased token must stay.
        cleanup(created);
        return false;
    }
    *token = created;
    return *token != 0;
}

int wait_token(void * opaque, uint64_t token, int * error) {
    if (opaque == nullptr || token == 0) {
        return GGML_RELAY_TOKEN_UNKNOWN;
    }
    auto & p = *static_cast<provider *>(opaque);
    std::unique_lock<std::mutex> lock(p.mutex);
    auto it = p.token_state.find(token);
    if (it == p.token_state.end()) {
        return GGML_RELAY_TOKEN_UNKNOWN;
    }
    if (it->second.first == 1) {
        if (error != nullptr) *error = 0;
        return GGML_RELAY_TOKEN_SUCCESS;
    }
    const bool was_failed = it->second.first == 2;
    const int prior_error = it->second.second;
    if (p.active_token != token && !was_failed) {
        return GGML_RELAY_TOKEN_UNKNOWN;
    }
    // Block on the exact native tails. A failed token may still have native
    // work in flight; only report FAILED after these tails synchronize.
    auto tails_it = p.token_tails.find(token);
    if (tails_it == p.token_tails.end()) {
        return GGML_RELAY_TOKEN_UNKNOWN;
    }
    const auto tails = tails_it->second;
    lock.unlock();
    int native_error = 0;
    bool have_tail = false;
    for (uint32_t r = 0; r < ranks; ++r) {
        if (hipSetDevice(p.rank[r].physical) != hipSuccess) {
            native_error = EIO;
            break;
        }
        for (uint32_t l = 0; l < lanes_per_rank; ++l) {
            const hipEvent_t tail = tails[r * lanes_per_rank + l];
            if (tail == nullptr) continue;
            have_tail = true;
            if (hipEventSynchronize(tail) != hipSuccess) {
                native_error = EIO;
                break;
            }
        }
        if (native_error != 0) break;
    }
    lock.lock();
    if (!have_tail || native_error != 0) {
        if (error != nullptr) *error = native_error != 0 ? native_error : -1;
        return GGML_RELAY_TOKEN_UNKNOWN;
    }
    observe(p);
    // This token's own tails have drained. Do not let a newer active token's
    // global status settle this token; record this token's result explicitly.
    // Success also needs the full CPU drain: every output released and no
    // outstanding work borrow.
    const bool plan_complete = p.status.finished &&
        p.event_status.load(std::memory_order_acquire) == 0 &&
        p.outputs_released == p.status.channels &&
        p.borrowed == std::numeric_limits<size_t>::max();
    it = p.token_state.find(token);
    if (it == p.token_state.end()) {
        return GGML_RELAY_TOKEN_UNKNOWN;
    }
    if (was_failed || p.status.failed) {
        it->second = {2, was_failed ? prior_error : p.status.error_code};
    } else if (plan_complete) {
        it->second = {1, 0};
    } else {
        if (error != nullptr) *error = -1;
        return GGML_RELAY_TOKEN_UNKNOWN;
    }
    if (p.active_token == token) {
        p.active_token = 0;
    }
    settle_token(p);
    it = p.token_state.find(token);
    if (it == p.token_state.end()) {
        return GGML_RELAY_TOKEN_UNKNOWN;
    }
    if (it->second.first == 2 || was_failed) {
        if (error != nullptr) *error = was_failed ? prior_error : it->second.second;
        return GGML_RELAY_TOKEN_FAILED;
    }
    if (it->second.first == 1) {
        if (error != nullptr) *error = 0;
        return GGML_RELAY_TOKEN_SUCCESS;
    }
    if (error != nullptr) *error = -1;
    return GGML_RELAY_TOKEN_UNKNOWN;
}

void release_token(void * opaque, uint64_t token) {
    if (opaque == nullptr || token == 0) {
        return;
    }
    auto & p = *static_cast<provider *>(opaque);
    std::lock_guard<std::mutex> lock(p.mutex);
    p.token_state.erase(token);
    p.token_tails.erase(token);
    if (p.active_token == token) {
        p.active_token = 0;
    }
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
void fill_retired_event(provider & p, uint64_t generation, Event & event) {
    event = {};
    if constexpr (std::is_same<Event, ggml_relay_event_v2>::value) {
        event.abi_version = GGML_RELAY_EVENT_V2_ABI_VERSION;
    } else {
        event.abi_version = GGML_RELAY_ABI_VERSION;
    }
    event.kind            = GGML_RELAY_PLAN_RETIRED;
    // The runtime rejects a zero epoch as an identity error. Retirement is a
    // provider fact, not an epoch fact; use the last observed epoch when the
    // provider has not submitted yet.
    event.epoch           = p.status.epoch != 0 ? p.status.epoch : 1;
    event.plan_generation = generation;
    event.ranks           = ranks;
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
    if ((p.abort_claim.load(std::memory_order_acquire) & abort_claim_bit) ||
        p.pending_abort.load(std::memory_order_acquire)) {
        return -1;
    }
    // The count is a lock-free hint; the queue itself is read under p.mutex.
    const bool have_retired = p.retired_count.load(std::memory_order_relaxed) != 0;
    const int event_status = p.event_status.load(std::memory_order_acquire);
    if (event_status <= 0 && !have_retired) {
        return event_status;
    }
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        return 0;
    }
    if (abort_visible(p) || p.status.failed) {
        return -1;
    }
    if (p.borrowed != std::numeric_limits<size_t>::max()) {
        return 0;
    }
    const size_t retired = p.retired_count.load(std::memory_order_relaxed);
    if (retired != 0) {
        const uint64_t generation = p.retired_plans[0];
        for (size_t i = 1; i < retired; ++i) {
            p.retired_plans[i - 1] = p.retired_plans[i];
        }
        p.retired_count.store(retired - 1, std::memory_order_relaxed);
        fill_retired_event(p, generation, *event);
        return 1;
    }
    if (!p.status.sealed) {
        return 0;
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
        reject_local("relay publish view identity mismatch");
        return false;
    }
    // Matching publish always returns the CPU borrow, including after abort.
    // A generic abort alone cannot revoke a concurrently executing CPU reader.
    p.borrowed = std::numeric_limits<size_t>::max();
    observe(p, false);
    // A pending abort that has not claimed the sticky bit is still an abort
    // request. Fail closed after the borrow is returned, so no success
    // generation follows it. A success CAS that races past this check is
    // still ordered by the claim word.
    if (abort_visible(p) || p.status.failed || !p.status.sealed || !c.offered || c.published) {
        return false;
    }
    // Claim this publication on the abort/success word. The CAS is the
    // publication linearization: an abort that claimed first makes this fail;
    // an abort that claims after is a later failure. The generation store is
    // the publication effect and follows the claim.
    if (!claim_host_success(p)) {
        consume_abort(p);
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

void settle_token(provider & p) {
    if (p.active_token == 0) {
        return;
    }
    auto it = p.token_state.find(p.active_token);
    if (it == p.token_state.end()) {
        p.active_token = 0;
        return;
    }
    if (p.status.failed) {
        it->second = {2, p.status.error_code};
        p.active_token = 0;
    } else if (p.status.finished &&
               p.event_status.load(std::memory_order_acquire) == 0 &&
               p.outputs_released == p.status.channels &&
               p.borrowed == std::numeric_limits<size_t>::max()) {
        it->second = {1, 0};
        p.active_token = 0;
    }
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
    // Linearize the abort read with a no-op CAS on abort_claim. An abort that
    // claimed before this point is visible through the claim word's
    // modification order; an abort that claims after is a later failure. A
    // settled snapshot has finished native tails, every output of the current
    // epoch released, and event_status==0. Epoch 0 is settled by definition;
    // after prepare, status.channels may change before begin without touching
    // outputs_released, so the explicit count comparison is required.
    if (abort_requested(p)) {
        // Failure is sticky, but physical drain is a separate fact. Keep
        // querying native tails and completion even in the failed state; do
        // not publish a success. This lets finished/borrow advance so a later
        // destroy can prove completion instead of retaining forever.
        observe(p);
        settle_token(p);
        *status = p.status;
        return true;
    }
    if (p.status.failed) {
        // Same rule for a non-abort failure: observe the tails, do not publish.
        observe(p);
        settle_token(p);
        *status = p.status;
        return true;
    }
    if (p.status.finished && p.event_status.load(std::memory_order_acquire) == 0 &&
        p.outputs_released == p.status.channels &&
        p.borrowed == std::numeric_limits<size_t>::max()) {
        settle_token(p);
        *status = p.status;
        return true;
    }
    observe(p);
    // A pending abort published during observe() must not leave this snapshot
    // looking healthy while poll() already returns -1.
    abort_requested(p);
    settle_token(p);
    *status = p.status;
    return true;
}

bool plan_snapshot(void * opaque, ggml_relay_plan_snapshot * snapshot) {
    if (!opaque || !snapshot) {
        return false;
    }
    auto &                       p = *static_cast<provider *>(opaque);
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        return false;
    }
    *snapshot = {};
    for (size_t i = 0; i < ggml_relay_plan::plan_capacity; ++i) {
        const auto * slot = p.plan_cache.at(i);
        if (slot == nullptr) {
            continue;
        }
        snapshot->valid_mask |= 1u << i;
        snapshot->generation[i] = slot->generation;
        snapshot->channels[i]   = slot->channels;
    }
    const auto & stats = p.plan_cache.stats();
    snapshot->hits             = stats.hits;
    snapshot->misses           = stats.misses;
    snapshot->inserts          = stats.inserts;
    snapshot->evictions        = stats.evictions;
    snapshot->pinned_refusals  = stats.pinned_refusals;
    snapshot->flushes          = stats.flushes;
    snapshot->stale_generation = stats.stale_generation;
    snapshot->pool_bytes       = p.pool_bytes;
    return true;
}

bool cpu_snapshot(void * opaque, ggml_relay_cpu_status * status) {
    if (!opaque || !status) {
        return false;
    }
    auto &                       p = *static_cast<provider *>(opaque);
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        return false;
    }
    status->epoch             = p.status.epoch;
    status->channels          = uint64_t(p.status.channels);
    status->outputs_released  = uint64_t(p.outputs_released);
    status->active_token      = p.active_token;
    status->tokens_outstanding = uint64_t(p.token_state.size());
    status->borrowed_channel  = p.borrowed == std::numeric_limits<size_t>::max()
        ? UINT32_MAX : uint32_t(p.borrowed);
    return true;
}

void abort(void * opaque, int32_t error) {
    if (!opaque) {
        return;
    }
    auto & p = *static_cast<provider *>(opaque);
    // Owner lifetime is externally serialized against destroy; the claim word
    // does not protect lifetime. This path is non-blocking: the pending-error
    // RMW stores the first error, then one fetch_or sets the sticky abort bit.
    // The device abort word is published only after that claim, so a host
    // success claim that wins the RMW order is before this abort, and a host
    // claim that loses sees the bit. Abort never blocks on the provider mutex;
    // it may try_lock it after publishing to consume the abort into status.
    // It takes no other host lock and never waits for another thread or a HIP
    // holder. A return guarantees the claim/device word, not status.failed.
    claim_host_abort(p, error);
    if (p.abort_host) {
        store(p.abort_host, 1);
    }
    p.event_status.store(-1, std::memory_order_release);
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    if (lock.owns_lock()) {
        consume_abort(p);
        settle_token(p);
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
    if (p->active) {
        reject_local("relay destroy during an active epoch; retaining owners");
        return false;
    }
    if (!idle(*p)) {
        char detail[sizeof(p->status.error)];
        if (p->borrowed != std::numeric_limits<size_t>::max()) {
            std::snprintf(detail, sizeof(detail),
                          "relay destroy has an outstanding CPU borrow channel=%zu; publish the saved view or retain owners",
                          p->borrowed);
        } else if (!p->status.finished) {
            std::snprintf(detail, sizeof(detail),
                          "relay destroy cannot prove native completion; retaining owners");
        } else {
            std::snprintf(detail, sizeof(detail),
                          "relay destroy has unreleased outputs; retaining owners");
        }
        reject_local(detail);
        return false;
    }
    if (!p->token_state.empty()) {
        char detail[sizeof(p->status.error)];
        std::snprintf(detail, sizeof(detail),
                      "relay destroy has %zu unreleased completion token(s); wait and release them or retain owners",
                      p->token_state.size());
        reject_local(detail);
        return false;
    }
    // Snapshot before the teardown flush: live plans and frames are the
    // steady-state evidence, not the post-flush zeros.
    log_plan_stats(*p, "destroy");
    if (p->plan_cache_enabled && !flush_all_plans(*p, false)) {
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
    prepare_ex,
    plan_snapshot,
    submit_token,
    wait_token,
    release_token,
    cpu_snapshot,
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
