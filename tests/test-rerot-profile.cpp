// test-rerot-profile.cpp — Hermetic unit tests for RERoT observation ledger
//
// Verifies:
// 1. Environment gating (LLAMA_REROT_PROFILE) and no-op safety when disabled.
// 2. Global singleton lifecycle and active profile registration.
// 3. Phase timing enter/exit/accumulate and cross-segment accumulation.
// 4. Route hits and rejection reasons per enum.
// 5. Layout counts, CAS high-watermark, and visible keys statistics.
// 6. Host & state byte metrics (upload_bytes, h2d_bytes, etc.).
// 7. Synchronize duration and generation recording (no_work vs with_work).
// 8. Bounded ring buffer FIFO overwrite (capacity 256) and value-initialization reset semantics.
// 9. Script-parsable format_tsv stability and print_summary execution.

#include "llama-rerot-profile.h"

#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/wait.h>
#include <unistd.h>
#define REROT_PROFILE_HAS_FORK 1
#else
#define REROT_PROFILE_HAS_FORK 0
#endif

static int g_failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        ++g_failures; \
    } \
} while (0)

#define CHECK_EQ(a, b) do { \
    if ((a) != (b)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s == %s (actual %" PRIu64 " != %" PRIu64 ")\n", \
                     __FILE__, __LINE__, #a, #b, (uint64_t)(a), (uint64_t)(b)); \
        ++g_failures; \
    } \
} while (0)

// Helper RAII for preserving environment variables hermetically
struct env_guard {
    std::string name;
    bool was_set = false;
    std::string original_value;

    explicit env_guard(const char * var_name) : name(var_name) {
        const char * val = std::getenv(var_name);
        if (val) {
            was_set = true;
            original_value = val;
        } else {
            was_set = false;
        }
    }

    ~env_guard() {
#if defined(_WIN32)
        if (was_set) {
            _putenv_s(name.c_str(), original_value.c_str());
        } else {
            _putenv_s(name.c_str(), "");
        }
#else
        if (was_set) {
            setenv(name.c_str(), original_value.c_str(), 1);
        } else {
            unsetenv(name.c_str());
        }
#endif
    }
};

// ----------------------------------------------------------------------------
// Test (1): Gating Disabled / Null Safety & Side Effects
// ----------------------------------------------------------------------------

static void test_gating_disabled_safety() {
    // When no active profile exists or prof is nullptr:
    CHECK(llama_rerot_profile_active() == nullptr);

    // RAII phase scope with nullptr must be completely safe and cause no segfaults
    {
        llama_rerot_phase_scope scope(nullptr, llama_rerot_phase::normal_prefill);
    }

    // llama_rerot_profile_end with nullptr active must be safe
    llama_rerot_profile_end();
    CHECK(llama_rerot_profile_active() == nullptr);
}

// ----------------------------------------------------------------------------
// Test (2): Phase Timing & Cross-Segment Accumulation
// ----------------------------------------------------------------------------

static void test_phase_timing_accumulation() {
    llama_rerot_profile prof;
    prof.reset(1001);

    // Initial state check
    for (size_t i = 0; i < (size_t) llama_rerot_phase::count; ++i) {
        CHECK_EQ(prof.phases[i].total_us.load(), 0);
        CHECK_EQ(prof.phases[i].max_us.load(), 0);
        CHECK_EQ(prof.phases[i].count.load(), 0);
        CHECK_EQ(prof.phases[i].open_start_ns.load(), 0);
    }

    // Cross-segment accumulation for formal_p
    prof.phase_accumulate(llama_rerot_phase::formal_p, 120);
    prof.phase_accumulate(llama_rerot_phase::formal_p, 300);
    prof.phase_accumulate(llama_rerot_phase::formal_p, 80);

    const auto & formal = prof.phases[(size_t) llama_rerot_phase::formal_p];
    CHECK_EQ(formal.total_us.load(), 500); // 120 + 300 + 80
    CHECK_EQ(formal.max_us.load(), 300);   // max(120, 300, 80)
    CHECK_EQ(formal.count.load(), 3);

    // Normal prefill
    prof.phase_accumulate(llama_rerot_phase::normal_prefill, 250);
    CHECK_EQ(prof.phases[(size_t) llama_rerot_phase::normal_prefill].total_us.load(), 250);
    CHECK_EQ(prof.phases[(size_t) llama_rerot_phase::normal_prefill].count.load(), 1);

    // Phase enter & exit cycle
    prof.phase_enter(llama_rerot_phase::synthesis);
    CHECK(prof.phases[(size_t) llama_rerot_phase::synthesis].open_start_ns.load() > 0);
    prof.phase_exit(llama_rerot_phase::synthesis);
    CHECK_EQ(prof.phases[(size_t) llama_rerot_phase::synthesis].open_start_ns.load(), 0);
    CHECK_EQ(prof.phases[(size_t) llama_rerot_phase::synthesis].count.load(), 1);

    // Exit without prior enter must be a safe no-op
    const uint64_t before_count = prof.phases[(size_t) llama_rerot_phase::probe].count.load();
    prof.phase_exit(llama_rerot_phase::probe);
    CHECK_EQ(prof.phases[(size_t) llama_rerot_phase::probe].count.load(), before_count);

    // Out-of-bounds phase enum safety
    prof.phase_enter((llama_rerot_phase) 99);
    prof.phase_exit((llama_rerot_phase) 99);
    prof.phase_accumulate((llama_rerot_phase) 99, 100);

    // Phase name lookup
    CHECK(std::string(llama_rerot_phase_name(llama_rerot_phase::normal_prefill)) == "normal_prefill");
    CHECK(std::string(llama_rerot_phase_name(llama_rerot_phase::dag_prefix_rebuild)) == "dag_prefix_rebuild");
    CHECK(std::string(llama_rerot_phase_name(llama_rerot_phase::result_send)) == "result_send");
    CHECK(std::string(llama_rerot_phase_name((llama_rerot_phase) 250)) == "unknown");
}

// ----------------------------------------------------------------------------
// Test (3): Route Hits and Rejection Reasons
// ----------------------------------------------------------------------------

static void test_route_hits_and_rejections() {
    llama_rerot_profile prof;
    prof.reset(1002);

    // Route hit recording
    prof.route_hit(llama_rerot_route::indexed_rerot);
    prof.route_hit(llama_rerot_route::indexed_rerot);
    prof.route_hit(llama_rerot_route::flashprefill);
    prof.route_hit(llama_rerot_route::xkv);
    prof.route_hit(llama_rerot_route::cpu_fallback);

    CHECK_EQ(prof.routes[(size_t) llama_rerot_route::indexed_rerot].hits.load(), 2);
    CHECK_EQ(prof.routes[(size_t) llama_rerot_route::flashprefill].hits.load(), 1);
    CHECK_EQ(prof.routes[(size_t) llama_rerot_route::xkv].hits.load(), 1);
    CHECK_EQ(prof.routes[(size_t) llama_rerot_route::cpu_fallback].hits.load(), 1);

    // Rejection reasons
    prof.route_rejection(llama_rerot_route::indexed_rerot, llama_rerot_rejection_reason::shape);
    prof.route_rejection(llama_rerot_route::indexed_rerot, llama_rerot_rejection_reason::shape);
    prof.route_rejection(llama_rerot_route::indexed_rerot, llama_rerot_rejection_reason::layout);
    prof.route_rejection(llama_rerot_route::flashprefill, llama_rerot_rejection_reason::unsupported_backend);
    prof.route_rejection(llama_rerot_route::xkv, llama_rerot_rejection_reason::side_effect);

    CHECK_EQ(prof.routes[(size_t) llama_rerot_route::indexed_rerot].rejections[(size_t) llama_rerot_rejection_reason::shape].load(), 2);
    CHECK_EQ(prof.routes[(size_t) llama_rerot_route::indexed_rerot].rejections[(size_t) llama_rerot_rejection_reason::layout].load(), 1);
    CHECK_EQ(prof.routes[(size_t) llama_rerot_route::indexed_rerot].rejections[(size_t) llama_rerot_rejection_reason::type].load(), 0);
    CHECK_EQ(prof.routes[(size_t) llama_rerot_route::flashprefill].rejections[(size_t) llama_rerot_rejection_reason::unsupported_backend].load(), 1);
    CHECK_EQ(prof.routes[(size_t) llama_rerot_route::xkv].rejections[(size_t) llama_rerot_rejection_reason::side_effect].load(), 1);

    // Out-of-bounds safety
    prof.route_hit((llama_rerot_route) 99);
    prof.route_rejection((llama_rerot_route) 99, llama_rerot_rejection_reason::shape);
    prof.route_rejection(llama_rerot_route::indexed_rerot, (llama_rerot_rejection_reason) 99);

    // Names
    CHECK(std::string(llama_rerot_route_name(llama_rerot_route::indexed_rerot)) == "indexed_rerot");
    CHECK(std::string(llama_rerot_route_name((llama_rerot_route) 100)) == "unknown");
    CHECK(std::string(llama_rerot_rejection_reason_name(llama_rerot_rejection_reason::shape)) == "shape");
    CHECK(std::string(llama_rerot_rejection_reason_name(llama_rerot_rejection_reason::side_effect)) == "side_effect");
    CHECK(std::string(llama_rerot_rejection_reason_name((llama_rerot_rejection_reason) 100)) == "unknown");
}

// ----------------------------------------------------------------------------
// Test (4): Layout Counts & CAS High-Watermark Semantics
// ----------------------------------------------------------------------------

static void test_layout_counts_and_cas_hwm() {
    llama_rerot_profile prof;
    prof.reset(1003);

    // Layout direct counters
    prof.live_groups.fetch_add(4, std::memory_order_relaxed);
    prof.cap_groups.store(8, std::memory_order_relaxed);
    prof.live_entries.fetch_add(16, std::memory_order_relaxed);
    prof.cap_entries.store(32, std::memory_order_relaxed);
    prof.actual_query_rows.fetch_add(64, std::memory_order_relaxed);
    prof.run_count.fetch_add(5, std::memory_order_relaxed);
    prof.continuous_span_count.fetch_add(3, std::memory_order_relaxed);

    CHECK_EQ(prof.live_groups.load(), 4);
    CHECK_EQ(prof.cap_groups.load(), 8);
    CHECK_EQ(prof.live_entries.load(), 16);
    CHECK_EQ(prof.cap_entries.load(), 32);
    CHECK_EQ(prof.actual_query_rows.load(), 64);
    CHECK_EQ(prof.run_count.load(), 5);
    CHECK_EQ(prof.continuous_span_count.load(), 3);

    // Reader visible keys tracking
    prof.record_reader_visible_keys(128);
    prof.record_reader_visible_keys(256);
    prof.record_reader_visible_keys(64);
    CHECK_EQ(prof.reader_visible_keys_total.load(), 448); // 128 + 256 + 64
    CHECK_EQ(prof.reader_visible_keys_samples.load(), 3);
    CHECK_EQ(prof.reader_visible_keys_max.load(), 256);

    // High watermark CAS logic (mirroring llama-kv-cache.cpp:6341)
    const uint64_t inputs[] = { 100, 50, 200, 150, 350, 300 };
    for (uint64_t n_kv : inputs) {
        uint64_t prev_hwm = prof.high_watermark_n_kv.load(std::memory_order_relaxed);
        while (n_kv > prev_hwm && !prof.high_watermark_n_kv.compare_exchange_weak(prev_hwm, n_kv, std::memory_order_relaxed)) {}
    }
    CHECK_EQ(prof.high_watermark_n_kv.load(), 350);

    // In-flight slots and scratch peak updates
    prof.update_in_flight_slots(10);
    prof.update_in_flight_slots(25);
    prof.update_in_flight_slots(15);
    CHECK_EQ(prof.in_flight_slots.load(), 15);
    CHECK_EQ(prof.max_in_flight_slots.load(), 25);

    prof.update_scratch_peak(1024);
    prof.update_scratch_peak(4096);
    prof.update_scratch_peak(2048);
    CHECK_EQ(prof.scratch_peak_bytes.load(), 4096);
}

// ----------------------------------------------------------------------------
// Test (5): Upload Bytes & H2D Bytes Accumulation
// ----------------------------------------------------------------------------

static void test_bytes_and_host_metrics() {
    llama_rerot_profile prof;
    prof.reset(1004);

    // Mirroring llama-graph.cpp:901 upload and h2d accumulation
    prof.upload_bytes.fetch_add(4096, std::memory_order_relaxed);
    prof.upload_bytes.fetch_add(2048, std::memory_order_relaxed);
    prof.h2d_bytes.fetch_add(4096, std::memory_order_relaxed);
    prof.h2d_bytes.fetch_add(2048, std::memory_order_relaxed);
    prof.d2h_bytes.fetch_add(512, std::memory_order_relaxed);
    prof.d2d_bytes.fetch_add(256, std::memory_order_relaxed);

    CHECK_EQ(prof.upload_bytes.load(), 6144);
    CHECK_EQ(prof.h2d_bytes.load(), 6144);
    CHECK_EQ(prof.d2h_bytes.load(), 512);
    CHECK_EQ(prof.d2d_bytes.load(), 256);

    // GEMM / GEMV records
    prof.record_gemm(16, 256, 128);
    prof.record_gemv(32, 64);
    CHECK_EQ(prof.gemm_count.load(), 1);
    CHECK_EQ(prof.gemv_count.load(), 1);
}

// ----------------------------------------------------------------------------
// Test (6): Sync Recording (no_work vs with_work)
// ----------------------------------------------------------------------------

static void test_sync_recording() {
    llama_rerot_profile prof;
    prof.reset(1005);

    // 1) with_work: positive duration and generation
    prof.record_sync(150 /* duration_us */, 10 /* generation */);
    CHECK_EQ(prof.sync_us_total.load(), 150);
    CHECK_EQ(prof.sync_count.load(), 1);
    CHECK_EQ(prof.sync_max_us.load(), 150);
    CHECK_EQ(prof.completed_generation.load(), 10);

    // 2) no_work: 0 duration (immediate return / no work pending)
    prof.record_sync(0 /* duration_us */, 12 /* generation */);
    CHECK_EQ(prof.sync_us_total.load(), 150); // total unchanged (0 added)
    CHECK_EQ(prof.sync_count.load(), 2);      // count incremented
    CHECK_EQ(prof.sync_max_us.load(), 150);   // max unchanged
    CHECK_EQ(prof.completed_generation.load(), 12); // generation progressed

    // 3) additional with_work with smaller generation and larger duration
    prof.record_sync(400 /* duration_us */, 11 /* generation */);
    CHECK_EQ(prof.sync_us_total.load(), 550); // 150 + 400
    CHECK_EQ(prof.sync_count.load(), 3);
    CHECK_EQ(prof.sync_max_us.load(), 400);   // max updated
    CHECK_EQ(prof.completed_generation.load(), 12); // generation kept at 12 via atomic_max
}

// ----------------------------------------------------------------------------
// Test (7): Bounded Ring Buffer FIFO Overwrite & Value-Init Reset Semantics
// ----------------------------------------------------------------------------

static void test_bounded_ring_buffer_and_reset() {
    llama_rerot_bounded_ring_buffer ring;

    // Check compile-time capacity constant
    static_assert(LLAMA_REROT_PROFILE_RING_CAPACITY == 256, "Capacity must be 256");
    CHECK_EQ(ring.size(), 0);

    // Push 10 entries
    for (uint32_t i = 0; i < 10; ++i) {
        ring.push(1 /* category */, (uint8_t) i, 0, i * 10, i * 100);
    }
    CHECK_EQ(ring.size(), 10);

    llama_rerot_profile_event ev;
    CHECK(ring.get(0, ev));
    CHECK_EQ(ev.category, 1);
    CHECK_EQ(ev.id, 0);
    CHECK_EQ(ev.val32, 0);
    CHECK_EQ(ev.val64, 0);

    CHECK(ring.get(9, ev));
    CHECK_EQ(ev.id, 9);
    CHECK_EQ(ev.val32, 90);
    CHECK_EQ(ev.val64, 900);

    CHECK(!ring.get(10, ev)); // Out of bounds

    // Overfill beyond capacity: push up to 300 entries total
    for (uint32_t i = 10; i < 300; ++i) {
        ring.push(2 /* category */, (uint8_t) (i % 256), 0, i, i * 2);
    }

    // Capacity must be strictly bounded at 256
    CHECK_EQ(ring.size(), 256);
    CHECK(!ring.get(256, ev));

    // Oldest surviving entry: total pushed is 300, so items [0..43] were overwritten.
    // Index 0 in logical ring must be entry 44.
    CHECK(ring.get(0, ev));
    CHECK_EQ(ev.val32, 44);
    CHECK_EQ(ev.val64, 88);

    // Latest entry: index 255 must be entry 299.
    CHECK(ring.get(255, ev));
    CHECK_EQ(ev.val32, 299);
    CHECK_EQ(ev.val64, 598);

    // Test reset behavior:
    // This locks in the value-initialization semantics (entries[pos] = {})
    // preventing regression to raw memset or partial reset.
    ring.reset();
    CHECK_EQ(ring.size(), 0);
    CHECK_EQ(ring.write_idx.load(), 0);
    CHECK(!ring.get(0, ev));

    // Inspect underlying storage to verify all 256 slots are value-initialized
    for (size_t i = 0; i < LLAMA_REROT_PROFILE_RING_CAPACITY; ++i) {
        CHECK_EQ(ring.entries[i].timestamp_ns, 0);
        CHECK_EQ(ring.entries[i].category, 0);
        CHECK_EQ(ring.entries[i].id, 0);
        CHECK_EQ(ring.entries[i].sub_id, 0);
        CHECK_EQ(ring.entries[i].val32, 0);
        CHECK_EQ(ring.entries[i].val64, 0);
    }
}

// ----------------------------------------------------------------------------
// Test (8): Full Profile Reset Semantics
// ----------------------------------------------------------------------------

static void test_profile_full_reset() {
    llama_rerot_profile prof;
    prof.reset(5555);

    // Populate various fields
    prof.phase_accumulate(llama_rerot_phase::synthesis, 999);
    prof.route_hit(llama_rerot_route::indexed_rerot);
    prof.live_groups.store(42);
    prof.high_watermark_n_kv.store(1000);
    prof.upload_bytes.store(8192);
    prof.record_sync(50, 5);
    prof.ring.push(1, 2, 3, 4, 5);

    CHECK_EQ(prof.phases[(size_t) llama_rerot_phase::synthesis].total_us.load(), 999);
    CHECK_EQ(prof.routes[(size_t) llama_rerot_route::indexed_rerot].hits.load(), 1);
    CHECK_EQ(prof.live_groups.load(), 42);
    CHECK_EQ(prof.high_watermark_n_kv.load(), 1000);
    CHECK_EQ(prof.upload_bytes.load(), 8192);
    CHECK_EQ(prof.ring.size(), 3); // 1 route + 1 sync + 1 custom push

    // Now reset with a new request ID
    prof.reset(6666);
    CHECK_EQ(prof.request_id, 6666);
    CHECK(prof.is_active);

    // Verify all phases are reset
    for (size_t i = 0; i < (size_t) llama_rerot_phase::count; ++i) {
        CHECK_EQ(prof.phases[i].total_us.load(), 0);
        CHECK_EQ(prof.phases[i].max_us.load(), 0);
        CHECK_EQ(prof.phases[i].count.load(), 0);
        CHECK_EQ(prof.phases[i].open_start_ns.load(), 0);
    }

    // Verify routes are reset
    for (size_t r = 0; r < (size_t) llama_rerot_route::count; ++r) {
        CHECK_EQ(prof.routes[r].hits.load(), 0);
        for (size_t j = 0; j < (size_t) llama_rerot_rejection_reason::count; ++j) {
            CHECK_EQ(prof.routes[r].rejections[j].load(), 0);
        }
    }

    // Verify layout and state counters are reset
    CHECK_EQ(prof.live_groups.load(), 0);
    CHECK_EQ(prof.cap_groups.load(), 0);
    CHECK_EQ(prof.live_entries.load(), 0);
    CHECK_EQ(prof.cap_entries.load(), 0);
    CHECK_EQ(prof.actual_query_rows.load(), 0);
    CHECK_EQ(prof.run_count.load(), 0);
    CHECK_EQ(prof.high_watermark_n_kv.load(), 0);
    CHECK_EQ(prof.upload_bytes.load(), 0);
    CHECK_EQ(prof.h2d_bytes.load(), 0);
    CHECK_EQ(prof.sync_us_total.load(), 0);
    CHECK_EQ(prof.sync_count.load(), 0);
    CHECK_EQ(prof.ring.size(), 0);
}

// ----------------------------------------------------------------------------
// Test (9): format_tsv Field Count Stability & Script Parsability
// ----------------------------------------------------------------------------

static void test_format_tsv_stability() {
    llama_rerot_profile prof;
    prof.reset(8888);

    // Set known values
    prof.phase_accumulate(llama_rerot_phase::formal_p, 420);
    prof.route_hit(llama_rerot_route::indexed_rerot);
    prof.route_rejection(llama_rerot_route::indexed_rerot, llama_rerot_rejection_reason::layout);
    prof.live_groups.store(7);
    prof.high_watermark_n_kv.store(2048);
    prof.upload_bytes.store(16384);
    prof.h2d_bytes.store(16384);
    prof.record_sync(88, 3);

    const std::string tsv = prof.format_tsv();
    CHECK(!tsv.empty());

    // Parse lines and ensure every line has exact key<TAB>value syntax
    std::istringstream iss(tsv);
    std::string line;
    size_t line_count = 0;
    bool found_req = false;
    bool found_formal_p = false;
    bool found_indexed_hit = false;
    bool found_hwm = false;
    bool found_upload = false;

    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        ++line_count;

        const size_t tab_pos = line.find('\t');
        CHECK(tab_pos != std::string::npos);
        CHECK(line.find('\t', tab_pos + 1) == std::string::npos); // Exactly one tab per line

        const std::string key = line.substr(0, tab_pos);
        const std::string val = line.substr(tab_pos + 1);
        CHECK(!key.empty());
        CHECK(!val.empty());

        if (key == "request_id" && val == "8888") found_req = true;
        if (key == "phase_formal_p_us" && val == "420") found_formal_p = true;
        if (key == "route_indexed_rerot_hits" && val == "1") found_indexed_hit = true;
        if (key == "high_watermark_n_kv" && val == "2048") found_hwm = true;
        if (key == "upload_bytes" && val == "16384") found_upload = true;
    }

    // Expected line count:
    // 1 (request_id)
    // + 9 * 3 (phases: us, max_us, count) = 27
    // + 4 * (1 hit + 6 rejections) = 28
    // + 11 (layout)
    // + 11 (host & command)
    // + 9 (state & memory)
    // + 5 (computation)
    // Total = 1 + 27 + 28 + 11 + 11 + 9 + 5 = 92
    CHECK_EQ(line_count, 92);
    CHECK(found_req);
    CHECK(found_formal_p);
    CHECK(found_indexed_hit);
    CHECK(found_hwm);
    CHECK(found_upload);

    // Verify print_summary does not crash
    FILE * tmp = std::tmpfile();
    if (tmp) {
        prof.print_summary(tmp);
        std::fflush(tmp);
        std::rewind(tmp);
        char buf[1024];
        size_t n = std::fread(buf, 1, sizeof(buf) - 1, tmp);
        buf[n] = '\0';
        CHECK(std::strstr(buf, "[rerot-profile]") != nullptr);
        CHECK(std::strstr(buf, "[rerot-profile-routes]") != nullptr);
        CHECK(std::strstr(buf, "[rerot-profile-layout]") != nullptr);
        std::fclose(tmp);
    }
}

// ----------------------------------------------------------------------------
// Test (10): Subprocess Hermetic Gating Tests (Enabled vs Disabled)
// ----------------------------------------------------------------------------

#if REROT_PROFILE_HAS_FORK

// Test gating when LLAMA_REROT_PROFILE is unset or 0
static void run_forked_test_disabled() {
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        // Child: enforce disabled environment before any static initialization
        unsetenv("LLAMA_REROT_PROFILE");
        if (llama_rerot_profile_enabled()) {
            std::fprintf(stderr, "FAIL: profile should be disabled when unset\n");
            std::_Exit(1);
        }

        // Begin must not activate profile when disabled
        llama_rerot_profile_begin(999);
        if (llama_rerot_profile_active() != nullptr) {
            std::fprintf(stderr, "FAIL: active profile should be null when disabled\n");
            std::_Exit(1);
        }

        // Scope with active profile (null) must not crash
        {
            llama_rerot_phase_scope scope(llama_rerot_profile_active(), llama_rerot_phase::probe);
        }

        llama_rerot_profile_end();
        std::_Exit(0);
    }

    int status = 0;
    waitpid(pid, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

// Test gating when LLAMA_REROT_PROFILE is set to 1
static void run_forked_test_enabled() {
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        // Child: set profile enabled before static initialization
        setenv("LLAMA_REROT_PROFILE", "1", 1);
        if (!llama_rerot_profile_enabled()) {
            std::fprintf(stderr, "FAIL: profile should be enabled when LLAMA_REROT_PROFILE=1\n");
            std::_Exit(1);
        }

        // Begin must activate profile
        llama_rerot_profile_begin(777);
        llama_rerot_profile * prof = llama_rerot_profile_active();
        if (!prof) {
            std::fprintf(stderr, "FAIL: active profile must not be null when enabled\n");
            std::_Exit(1);
        }
        if (prof->request_id != 777 || !prof->is_active) {
            std::fprintf(stderr, "FAIL: active profile fields not initialized correctly\n");
            std::_Exit(1);
        }

        // Phase scope updates active profile
        {
            llama_rerot_phase_scope scope(prof, llama_rerot_phase::workers_frontier);
        }
        if (prof->phases[(size_t) llama_rerot_phase::workers_frontier].count.load() != 1) {
            std::fprintf(stderr, "FAIL: phase scope did not increment count\n");
            std::_Exit(1);
        }

        // End drains profile
        llama_rerot_profile_end();
        if (llama_rerot_profile_active() != nullptr) {
            std::fprintf(stderr, "FAIL: profile active must be null after end\n");
            std::_Exit(1);
        }

        std::_Exit(0);
    }

    int status = 0;
    waitpid(pid, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

#endif // REROT_PROFILE_HAS_FORK

// ----------------------------------------------------------------------------
// Main Runner
// ----------------------------------------------------------------------------

int main() {
    // Preserve original environment value to keep test completely hermetic
    env_guard guard("LLAMA_REROT_PROFILE");

    std::printf("test-rerot-profile: starting unit test suite...\n");

    test_gating_disabled_safety();
    test_phase_timing_accumulation();
    test_route_hits_and_rejections();
    test_layout_counts_and_cas_hwm();
    test_bytes_and_host_metrics();
    test_sync_recording();
    test_bounded_ring_buffer_and_reset();
    test_profile_full_reset();
    test_format_tsv_stability();

#if REROT_PROFILE_HAS_FORK
    std::printf("test-rerot-profile: running isolated gating checks in sub-processes...\n");
    run_forked_test_disabled();
    run_forked_test_enabled();
#endif

    if (g_failures > 0) {
        std::fprintf(stderr, "test-rerot-profile: %d CHECKS FAILED\n", g_failures);
        return 1;
    }

    std::printf("test-rerot-profile: all checks passed successfully.\n");
    return 0;
}
