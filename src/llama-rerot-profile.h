// llama-rerot-profile.h — Request-level observation ledger for RERoT
//
// Architecture Decision Record (ADR):
// - Context & Constraints:
//     RERoT (Recursive Elastic Ring-of-Thought) introduces dynamic DAG scheduling,
//     probe execution, lane-local recurrent state, DDVR KV sharing, and multi-lane
//     synthesis. Observing these runtime dynamics is essential for performance
//     profiling and correctness debugging.
//     However, profiling must satisfy strict engineering invariants:
//     1) Zero unbounded memory allocation: no dynamic resizing or unbounded vectors
//        during the decode/execution hot path;
//     2) Low overhead: hot-path operations must be cheap counter increments or bounded
//        ring-buffer writes, guarded by a compile-time or runtime environment check;
//     3) Zero log pollution: no verbose per-token logging; single summary output on finish;
//     4) High concurrency hygiene: explicitly defined concurrency model.
// - Concurrency Choice & Rationale:
//     A dual-tier concurrency approach is chosen:
//     1) Per-request single-threaded fast path: a server task or decode worker owns its
//        profile instance or operates on thread-local / request-local scope without lock
//        contention;
//     2) Thread-safe atomic counters (std::atomic<uint64_t>) and a global mutex-protected
//        active profile registry (mirroring ggml_tp5_profile) allow cross-thread callbacks
//        and multi-lane worker tasks to safely update metrics without races;
//     3) Summary output and ring-buffer draining happen exactly once at request termination,
//        completely isolated from active computation.
// - Chosen Path:
//     A self-contained struct llama_rerot_profile with cumulative 64-bit atomic counters,
//     a fixed-size compile-time ring buffer (capacity 256) for detailed event traces,
//     monotonic phase clocks (std::chrono::steady_clock), route rejection reason tracking,
//     and structured key-value summary formatting parseable by downstream analysis scripts.
// - Rejected Alternatives:
//     - Dynamic std::vector traces: rejected due to unpredictable heap allocations on hot path.
//     - Per-token stderr emission: rejected to prevent severe IO slowdown and log bloating.
//     - Tightly coupling with server_task: rejected to keep core llama library clean and
//       allow reuse in standalone benchmarks and tests.
// - Governing Contract:
//     RERoT Discussion Document §4.3, §9.1 and AGENTS.md.

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

// --- (1) Phase Timing ---

enum class llama_rerot_phase : uint8_t {
    normal_prefill = 0,
    probe,
    dag_prefix_rebuild,
    formal_p,
    workers_start,
    workers_frontier,
    workers_finish,
    synthesis,
    result_send,
    count,
};

const char * llama_rerot_phase_name(llama_rerot_phase phase);

// --- (2) Route Hit & Rejection Reasons (§9.1) ---

enum class llama_rerot_route : uint8_t {
    indexed_rerot = 0,
    flashprefill,
    xkv,
    cpu_fallback,
    count,
};

const char * llama_rerot_route_name(llama_rerot_route route);

enum class llama_rerot_rejection_reason : uint8_t {
    none = 0,
    shape,
    type,
    layout,
    alias,
    side_effect,
    unsupported_backend,
    count,
};

const char * llama_rerot_rejection_reason_name(llama_rerot_rejection_reason reason);

// Compile-time bounded ring buffer capacity for detailed traces
static constexpr size_t LLAMA_REROT_PROFILE_RING_CAPACITY = 256;

// Ring buffer entry for fine-grained trajectory observation
struct llama_rerot_profile_event {
    uint64_t timestamp_ns = 0;
    uint8_t  category     = 0; // 0=phase, 1=route, 2=sync, 3=gemm, 4=custom
    uint8_t  id           = 0; // phase id, route id, etc.
    uint16_t sub_id       = 0; // rejection reason, worker id, etc.
    uint32_t val32        = 0; // e.g. rows, keys, code
    uint64_t val64        = 0; // e.g. duration_us, bytes, generation
};

// Fixed-size, thread-safe bounded ring buffer
struct llama_rerot_bounded_ring_buffer {
    llama_rerot_profile_event entries[LLAMA_REROT_PROFILE_RING_CAPACITY] = {};
    std::atomic<uint64_t>     write_idx{0};

    void push(uint8_t category, uint8_t id, uint16_t sub_id, uint32_t val32, uint64_t val64);
    void reset();
    size_t size() const;
    bool get(size_t index, llama_rerot_profile_event & out) const;
};

// Main self-contained observation ledger
struct llama_rerot_profile {
    uint64_t request_id = 0;
    bool     is_active  = false;

    // (1) Phase timing cumulative ledger (microseconds and execution counts)
    struct phase_stat {
        std::atomic<uint64_t> total_us{0};
        std::atomic<uint64_t> max_us{0};
        std::atomic<uint64_t> count{0};
        std::atomic<uint64_t> first_us{0};    // §2.4 cold-transition billing: duration of the
        std::atomic<uint64_t> first_valid{0}; // first recorded occurrence (graph definition,
        std::atomic<int64_t>  open_start_ns{0}; // pipeline compilation, buffer allocs included).
    }; // first_valid gates first_us (0 duration is a legal first occurrence).
    phase_stat phases[(size_t) llama_rerot_phase::count];

    // (1b) Parallelism / phase-shape ledger (§4.3 阶段与并行): W = logical
    // DAG workers (episode nodes minus planner root), P = physical pen
    // capacity, logical frontiers opened, and the W>P time-slices that each
    // split one logical step into several physical slices. Waiting
    // successors are recorded per snapshot as the cohort's pending members.
    std::atomic<uint64_t> parallel_snapshots{0};
    std::atomic<uint64_t> sum_w{0};
    std::atomic<uint64_t> max_w{0};
    std::atomic<uint64_t> sum_p{0};
    std::atomic<uint64_t> sum_cohort{0};
    std::atomic<uint64_t> max_cohort{0};
    std::atomic<uint64_t> logical_frontiers{0};
    std::atomic<uint64_t> pen_yields{0};

    // (2) Route hits and rejections
    struct route_stat {
        std::atomic<uint64_t> hits{0};
        std::atomic<uint64_t> rejections[(size_t) llama_rerot_rejection_reason::count];
    };
    route_stat routes[(size_t) llama_rerot_route::count];

    // (3) Layout counts
    std::atomic<uint64_t> live_groups{0};
    std::atomic<uint64_t> cap_groups{0};
    std::atomic<uint64_t> live_entries{0};
    std::atomic<uint64_t> cap_entries{0};
    std::atomic<uint64_t> actual_query_rows{0};
    std::atomic<uint64_t> reader_visible_keys_total{0};
    std::atomic<uint64_t> reader_visible_keys_max{0};
    std::atomic<uint64_t> reader_visible_keys_samples{0};
    std::atomic<uint64_t> run_count{0};
    std::atomic<uint64_t> continuous_span_count{0};
    std::atomic<uint64_t> high_watermark_n_kv{0};

    // (4) Host and command — P0 evidence split: layout_* accumulates ONLY in
    // rerot_build_attn_layout (pure layout build); upload_* accumulates ONLY
    // in fill_spans (CPU staging + tensor set). The old single
    // layout_view_build_us mixed both clocks and double-counted.
    std::atomic<uint64_t> layout_view_build_us{0};
    std::atomic<uint64_t> layout_view_build_count{0};
    std::atomic<uint64_t> upload_bytes{0};
    std::atomic<uint64_t> upload_staging_us{0};
    std::atomic<uint64_t> upload_set_us{0};
    std::atomic<uint64_t> upload_count{0};
    std::atomic<uint64_t> graph_def_count{0};
    std::atomic<uint64_t> cb_allocations{0};
    std::atomic<uint64_t> cb_alloc_bytes{0};
    std::atomic<uint64_t> submit_count{0};
    std::atomic<uint64_t> sync_us_total{0};
    std::atomic<uint64_t> sync_max_us{0};
    std::atomic<uint64_t> sync_count{0};
    std::atomic<uint64_t> completed_generation{0};

    // (5) State and memory
    std::atomic<uint64_t> h2d_bytes{0};
    std::atomic<uint64_t> d2h_bytes{0};
    std::atomic<uint64_t> d2d_bytes{0};
    std::atomic<uint64_t> hand_seed_count{0};
    std::atomic<uint64_t> hand_seed_bytes{0};
    std::atomic<uint64_t> clear_count{0};
    std::atomic<uint64_t> scratch_peak_bytes{0};
    std::atomic<uint64_t> in_flight_slots{0};
    std::atomic<uint64_t> max_in_flight_slots{0};

    // (6) Computation
    std::atomic<uint64_t> q_prep_rows{0};
    std::atomic<uint64_t> gemv_count{0};
    std::atomic<uint64_t> gemm_count{0};
    std::atomic<uint64_t> multi_row_gdn_hits{0};
    std::atomic<uint64_t> attention_split_count{0};
    // §9.3 shared activation-rotation memoization (build_lora_mm): hits mean
    // sibling projections reused the rotated activation; misses are first
    // materializations. hit/(hit+miss) is the P7 recipe-expansion evidence.
    std::atomic<uint64_t> hadamard_memo_hits{0};
    std::atomic<uint64_t> hadamard_memo_misses{0};

    // Detailed trace bounded ring buffer
    llama_rerot_bounded_ring_buffer ring;

    // Reset all metrics for a new request
    void reset(uint64_t new_request_id = 0);

    // Phase management: enter, exit, and accumulate
    void phase_enter(llama_rerot_phase phase);
    void phase_exit(llama_rerot_phase phase);
    void phase_accumulate(llama_rerot_phase phase, uint64_t duration_us);

    // Route recording
    void route_hit(llama_rerot_route route);
    void route_rejection(llama_rerot_route route, llama_rerot_rejection_reason reason);

    // Synchronize recording
    void record_sync(uint64_t duration_us, uint64_t generation);

    // Memory / scratch updating with atomic high-watermark
    void update_scratch_peak(uint64_t bytes);
    void update_in_flight_slots(uint64_t current);
    void record_reader_visible_keys(uint64_t count);

    // GEMM / GEMV shape tracking
    void record_gemm(int64_t m, int64_t n, int64_t k);
    void record_gemv(int64_t m, int64_t k);

    // Summary formatting and script-parsable dump
    void print_summary(FILE * stream = stderr) const;
    std::string format_tsv() const;
};

// RAII Phase scope helper
class llama_rerot_phase_scope {
public:
    llama_rerot_phase_scope(llama_rerot_profile * prof, llama_rerot_phase phase)
        : prof_(prof), phase_(phase) {
        if (prof_) {
            prof_->phase_enter(phase_);
        }
    }
    ~llama_rerot_phase_scope() {
        if (prof_) {
            prof_->phase_exit(phase_);
        }
    }
private:
    llama_rerot_profile * prof_;
    llama_rerot_phase     phase_;
};

// Global lifecycle & singleton hooks (active only when LLAMA_REROT_PROFILE=1)
bool                  llama_rerot_profile_enabled();
llama_rerot_profile * llama_rerot_profile_active();
void                  llama_rerot_profile_begin(uint64_t request_id);
void                  llama_rerot_profile_end();
