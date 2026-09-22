// llama-rerot-profile.cpp — Implementation of RERoT observation ledger
//
// Self-contained profiling and instrumentation implementation.
// Adheres strictly to bounded memory, zero per-token log, and script-parsable output.

#include "llama-rerot-profile.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace {

inline uint64_t get_monotonic_ns() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline void atomic_max(std::atomic<uint64_t> & target, uint64_t val) {
    uint64_t prev = target.load(std::memory_order_relaxed);
    while (prev < val && !target.compare_exchange_weak(prev, val, std::memory_order_relaxed)) {}
}

} // namespace

const char * llama_rerot_phase_name(llama_rerot_phase phase) {
    switch (phase) {
        case llama_rerot_phase::normal_prefill:     return "normal_prefill";
        case llama_rerot_phase::probe:              return "probe";
        case llama_rerot_phase::dag_prefix_rebuild: return "dag_prefix_rebuild";
        case llama_rerot_phase::formal_p:           return "formal_p";
        case llama_rerot_phase::workers_start:      return "workers_start";
        case llama_rerot_phase::workers_frontier:   return "workers_frontier";
        case llama_rerot_phase::workers_finish:     return "workers_finish";
        case llama_rerot_phase::synthesis:          return "synthesis";
        case llama_rerot_phase::result_send:        return "result_send";
        default:                                    return "unknown";
    }
}

const char * llama_rerot_route_name(llama_rerot_route route) {
    switch (route) {
        case llama_rerot_route::indexed_rerot: return "indexed_rerot";
        case llama_rerot_route::flashprefill:  return "flashprefill";
        case llama_rerot_route::xkv:           return "xkv";
        case llama_rerot_route::cpu_fallback:  return "cpu_fallback";
        default:                               return "unknown";
    }
}

const char * llama_rerot_rejection_reason_name(llama_rerot_rejection_reason reason) {
    switch (reason) {
        case llama_rerot_rejection_reason::none:                return "none";
        case llama_rerot_rejection_reason::shape:               return "shape";
        case llama_rerot_rejection_reason::type:                return "type";
        case llama_rerot_rejection_reason::layout:              return "layout";
        case llama_rerot_rejection_reason::alias:               return "alias";
        case llama_rerot_rejection_reason::side_effect:         return "side_effect";
        case llama_rerot_rejection_reason::unsupported_backend: return "unsupported_backend";
        default:                                                return "unknown";
    }
}

void llama_rerot_bounded_ring_buffer::push(uint8_t category, uint8_t id, uint16_t sub_id, uint32_t val32, uint64_t val64) {
    const uint64_t idx = write_idx.fetch_add(1, std::memory_order_relaxed);
    const size_t pos = (size_t) (idx % LLAMA_REROT_PROFILE_RING_CAPACITY);
    entries[pos] = llama_rerot_profile_event{
        get_monotonic_ns(),
        category,
        id,
        sub_id,
        val32,
        val64,
    };
}

void llama_rerot_bounded_ring_buffer::reset() {
    write_idx.store(0, std::memory_order_relaxed);
    for (auto & entry : entries) {
        entry = {};
    }
}

size_t llama_rerot_bounded_ring_buffer::size() const {
    const uint64_t total = write_idx.load(std::memory_order_relaxed);
    return (size_t) std::min<uint64_t>(total, LLAMA_REROT_PROFILE_RING_CAPACITY);
}

bool llama_rerot_bounded_ring_buffer::get(size_t index, llama_rerot_profile_event & out) const {
    const uint64_t total = write_idx.load(std::memory_order_relaxed);
    if (index >= total || index >= LLAMA_REROT_PROFILE_RING_CAPACITY) {
        return false;
    }
    uint64_t start_pos = 0;
    if (total > LLAMA_REROT_PROFILE_RING_CAPACITY) {
        start_pos = total - LLAMA_REROT_PROFILE_RING_CAPACITY;
    }
    const size_t actual_idx = (size_t) ((start_pos + index) % LLAMA_REROT_PROFILE_RING_CAPACITY);
    out = entries[actual_idx];
    return true;
}

void llama_rerot_profile::reset(uint64_t new_request_id) {
    request_id = new_request_id;
    is_active  = true;

    for (size_t i = 0; i < (size_t) llama_rerot_phase::count; ++i) {
        phases[i].total_us.store(0, std::memory_order_relaxed);
        phases[i].max_us.store(0, std::memory_order_relaxed);
        phases[i].count.store(0, std::memory_order_relaxed);
        phases[i].first_us.store(0, std::memory_order_relaxed);
        phases[i].first_valid.store(0, std::memory_order_relaxed);
        phases[i].open_start_ns.store(0, std::memory_order_relaxed);
    }

    for (size_t r = 0; r < (size_t) llama_rerot_route::count; ++r) {
        routes[r].hits.store(0, std::memory_order_relaxed);
        for (size_t j = 0; j < (size_t) llama_rerot_rejection_reason::count; ++j) {
            routes[r].rejections[j].store(0, std::memory_order_relaxed);
        }
    }

    parallel_snapshots.store(0, std::memory_order_relaxed);
    sum_w.store(0, std::memory_order_relaxed);
    max_w.store(0, std::memory_order_relaxed);
    sum_p.store(0, std::memory_order_relaxed);
    sum_cohort.store(0, std::memory_order_relaxed);
    max_cohort.store(0, std::memory_order_relaxed);
    logical_frontiers.store(0, std::memory_order_relaxed);
    pen_yields.store(0, std::memory_order_relaxed);

    live_groups.store(0, std::memory_order_relaxed);
    cap_groups.store(0, std::memory_order_relaxed);
    live_entries.store(0, std::memory_order_relaxed);
    cap_entries.store(0, std::memory_order_relaxed);
    actual_query_rows.store(0, std::memory_order_relaxed);
    reader_visible_keys_total.store(0, std::memory_order_relaxed);
    reader_visible_keys_max.store(0, std::memory_order_relaxed);
    reader_visible_keys_samples.store(0, std::memory_order_relaxed);
    run_count.store(0, std::memory_order_relaxed);
    continuous_span_count.store(0, std::memory_order_relaxed);
    high_watermark_n_kv.store(0, std::memory_order_relaxed);

    layout_view_build_us.store(0, std::memory_order_relaxed);
    layout_view_build_count.store(0, std::memory_order_relaxed);
    upload_bytes.store(0, std::memory_order_relaxed);
    upload_staging_us.store(0, std::memory_order_relaxed);
    upload_set_us.store(0, std::memory_order_relaxed);
    upload_count.store(0, std::memory_order_relaxed);
    graph_def_count.store(0, std::memory_order_relaxed);
    cb_allocations.store(0, std::memory_order_relaxed);
    cb_alloc_bytes.store(0, std::memory_order_relaxed);
    submit_count.store(0, std::memory_order_relaxed);
    sync_us_total.store(0, std::memory_order_relaxed);
    sync_max_us.store(0, std::memory_order_relaxed);
    sync_count.store(0, std::memory_order_relaxed);
    completed_generation.store(0, std::memory_order_relaxed);

    h2d_bytes.store(0, std::memory_order_relaxed);
    d2h_bytes.store(0, std::memory_order_relaxed);
    d2d_bytes.store(0, std::memory_order_relaxed);
    hand_seed_count.store(0, std::memory_order_relaxed);
    hand_seed_bytes.store(0, std::memory_order_relaxed);
    clear_count.store(0, std::memory_order_relaxed);
    scratch_peak_bytes.store(0, std::memory_order_relaxed);
    in_flight_slots.store(0, std::memory_order_relaxed);
    max_in_flight_slots.store(0, std::memory_order_relaxed);

    q_prep_rows.store(0, std::memory_order_relaxed);
    gemv_count.store(0, std::memory_order_relaxed);
    gemm_count.store(0, std::memory_order_relaxed);
    multi_row_gdn_hits.store(0, std::memory_order_relaxed);
    attention_split_count.store(0, std::memory_order_relaxed);

    ring.reset();
}

void llama_rerot_profile::phase_enter(llama_rerot_phase phase) {
    const size_t p = (size_t) phase;
    if (p >= (size_t) llama_rerot_phase::count) return;
    const uint64_t now_ns = get_monotonic_ns();
    phases[p].open_start_ns.store((int64_t) now_ns, std::memory_order_relaxed);
    ring.push(0, (uint8_t) phase, 0 /* enter */, 0, now_ns);
}

void llama_rerot_profile::phase_exit(llama_rerot_phase phase) {
    const size_t p = (size_t) phase;
    if (p >= (size_t) llama_rerot_phase::count) return;
    const int64_t start = phases[p].open_start_ns.exchange(0, std::memory_order_relaxed);
    if (start <= 0) return;
    const uint64_t now_ns = get_monotonic_ns();
    const uint64_t dur_us = (now_ns > (uint64_t) start) ? (now_ns - (uint64_t) start) / 1000ULL : 0;
    phase_accumulate(phase, dur_us);
    ring.push(0, (uint8_t) phase, 1 /* exit */, 0, dur_us);
}

void llama_rerot_profile::phase_accumulate(llama_rerot_phase phase, uint64_t duration_us) {
    const size_t p = (size_t) phase;
    if (p >= (size_t) llama_rerot_phase::count) return;
    // §2.4 cold-transition billing: the first occurrence of a phase carries
    // the one-time definition/compile/alloc cost that steady state amortizes.
    // CAS-gated so only the true first accumulate wins (concurrent firsts
    // keep whichever lands first; steady-state calls never overwrite).
    uint64_t seen = 0;
    if (phases[p].first_valid.compare_exchange_strong(seen, 1, std::memory_order_relaxed)) {
        phases[p].first_us.store(duration_us, std::memory_order_relaxed);
    }
    phases[p].total_us.fetch_add(duration_us, std::memory_order_relaxed);
    phases[p].count.fetch_add(1, std::memory_order_relaxed);
    atomic_max(phases[p].max_us, duration_us);
}

void llama_rerot_profile::route_hit(llama_rerot_route route) {
    const size_t r = (size_t) route;
    if (r >= (size_t) llama_rerot_route::count) return;
    routes[r].hits.fetch_add(1, std::memory_order_relaxed);
    ring.push(1, (uint8_t) route, 0 /* hit */, 0, 1);
}

void llama_rerot_profile::route_rejection(llama_rerot_route route, llama_rerot_rejection_reason reason) {
    const size_t r = (size_t) route;
    const size_t j = (size_t) reason;
    if (r >= (size_t) llama_rerot_route::count || j >= (size_t) llama_rerot_rejection_reason::count) return;
    routes[r].rejections[j].fetch_add(1, std::memory_order_relaxed);
    ring.push(1, (uint8_t) route, (uint16_t) reason, 0, 0);
}

void llama_rerot_profile::record_sync(uint64_t duration_us, uint64_t generation) {
    sync_us_total.fetch_add(duration_us, std::memory_order_relaxed);
    sync_count.fetch_add(1, std::memory_order_relaxed);
    atomic_max(sync_max_us, duration_us);
    atomic_max(completed_generation, generation);
    ring.push(2, 0, 0, 0, duration_us);
}

void llama_rerot_profile::update_scratch_peak(uint64_t bytes) {
    atomic_max(scratch_peak_bytes, bytes);
}

void llama_rerot_profile::update_in_flight_slots(uint64_t current) {
    in_flight_slots.store(current, std::memory_order_relaxed);
    atomic_max(max_in_flight_slots, current);
}

void llama_rerot_profile::record_reader_visible_keys(uint64_t count) {
    reader_visible_keys_total.fetch_add(count, std::memory_order_relaxed);
    reader_visible_keys_samples.fetch_add(1, std::memory_order_relaxed);
    atomic_max(reader_visible_keys_max, count);
}

void llama_rerot_profile::record_gemm(int64_t m, int64_t n, int64_t k) {
    gemm_count.fetch_add(1, std::memory_order_relaxed);
    ring.push(3, 1 /* GEMM */, (uint16_t) std::min<int64_t>(k, 65535), (uint32_t) m, (uint64_t) n);
}

void llama_rerot_profile::record_gemv(int64_t m, int64_t k) {
    gemv_count.fetch_add(1, std::memory_order_relaxed);
    ring.push(3, 0 /* GEMV */, (uint16_t) std::min<int64_t>(k, 65535), (uint32_t) m, 1);
}

std::string llama_rerot_profile::format_tsv() const {
    std::ostringstream ss;
    ss << "request_id\t" << request_id << "\n";

    // Phases
    for (size_t i = 0; i < (size_t) llama_rerot_phase::count; ++i) {
        const auto phase = (llama_rerot_phase) i;
        ss << "phase_" << llama_rerot_phase_name(phase) << "_us\t" << phases[i].total_us.load(std::memory_order_relaxed) << "\n";
        ss << "phase_" << llama_rerot_phase_name(phase) << "_max_us\t" << phases[i].max_us.load(std::memory_order_relaxed) << "\n";
        ss << "phase_" << llama_rerot_phase_name(phase) << "_count\t" << phases[i].count.load(std::memory_order_relaxed) << "\n";
        ss << "phase_" << llama_rerot_phase_name(phase) << "_first_us\t" << phases[i].first_us.load(std::memory_order_relaxed) << "\n";
    }

    // Routes
    for (size_t r = 0; r < (size_t) llama_rerot_route::count; ++r) {
        const auto route = (llama_rerot_route) r;
        ss << "route_" << llama_rerot_route_name(route) << "_hits\t" << routes[r].hits.load(std::memory_order_relaxed) << "\n";
        for (size_t j = 1; j < (size_t) llama_rerot_rejection_reason::count; ++j) {
            const auto reason = (llama_rerot_rejection_reason) j;
            ss << "route_" << llama_rerot_route_name(route) << "_reject_" << llama_rerot_rejection_reason_name(reason)
               << "\t" << routes[r].rejections[j].load(std::memory_order_relaxed) << "\n";
        }
    }

    // Layout
    ss << "live_groups\t" << live_groups.load(std::memory_order_relaxed) << "\n";
    ss << "cap_groups\t" << cap_groups.load(std::memory_order_relaxed) << "\n";
    ss << "live_entries\t" << live_entries.load(std::memory_order_relaxed) << "\n";
    ss << "cap_entries\t" << cap_entries.load(std::memory_order_relaxed) << "\n";
    ss << "parallel_snapshots\t" << parallel_snapshots.load(std::memory_order_relaxed) << "\n";
    ss << "sum_w\t" << sum_w.load(std::memory_order_relaxed) << "\n";
    ss << "max_w\t" << max_w.load(std::memory_order_relaxed) << "\n";
    ss << "avg_p\t" << (parallel_snapshots.load(std::memory_order_relaxed) > 0 ? sum_p.load(std::memory_order_relaxed) / parallel_snapshots.load(std::memory_order_relaxed) : 0) << "\n";
    ss << "avg_cohort\t" << (parallel_snapshots.load(std::memory_order_relaxed) > 0 ? sum_cohort.load(std::memory_order_relaxed) / parallel_snapshots.load(std::memory_order_relaxed) : 0) << "\n";
    ss << "max_cohort\t" << max_cohort.load(std::memory_order_relaxed) << "\n";
    ss << "logical_frontiers\t" << logical_frontiers.load(std::memory_order_relaxed) << "\n";
    ss << "pen_yields\t" << pen_yields.load(std::memory_order_relaxed) << "\n";
    ss << "actual_query_rows\t" << actual_query_rows.load(std::memory_order_relaxed) << "\n";
    ss << "reader_visible_keys_total\t" << reader_visible_keys_total.load(std::memory_order_relaxed) << "\n";
    ss << "reader_visible_keys_max\t" << reader_visible_keys_max.load(std::memory_order_relaxed) << "\n";
    ss << "reader_visible_keys_samples\t" << reader_visible_keys_samples.load(std::memory_order_relaxed) << "\n";
    ss << "run_count\t" << run_count.load(std::memory_order_relaxed) << "\n";
    ss << "continuous_span_count\t" << continuous_span_count.load(std::memory_order_relaxed) << "\n";
    ss << "high_watermark_n_kv\t" << high_watermark_n_kv.load(std::memory_order_relaxed) << "\n";

    // Host & command
    ss << "layout_view_build_us\t" << layout_view_build_us.load(std::memory_order_relaxed) << "\n";
    ss << "layout_view_build_count\t" << layout_view_build_count.load(std::memory_order_relaxed) << "\n";
    ss << "upload_bytes\t" << upload_bytes.load(std::memory_order_relaxed) << "\n";
    ss << "upload_staging_us\t" << upload_staging_us.load(std::memory_order_relaxed) << "\n";
    ss << "upload_set_us\t" << upload_set_us.load(std::memory_order_relaxed) << "\n";
    ss << "upload_count\t" << upload_count.load(std::memory_order_relaxed) << "\n";
    ss << "graph_def_count\t" << graph_def_count.load(std::memory_order_relaxed) << "\n";
    ss << "cb_allocations\t" << cb_allocations.load(std::memory_order_relaxed) << "\n";
    ss << "cb_alloc_bytes\t" << cb_alloc_bytes.load(std::memory_order_relaxed) << "\n";
    ss << "submit_count\t" << submit_count.load(std::memory_order_relaxed) << "\n";
    ss << "sync_us_total\t" << sync_us_total.load(std::memory_order_relaxed) << "\n";
    ss << "sync_max_us\t" << sync_max_us.load(std::memory_order_relaxed) << "\n";
    ss << "sync_count\t" << sync_count.load(std::memory_order_relaxed) << "\n";
    ss << "completed_generation\t" << completed_generation.load(std::memory_order_relaxed) << "\n";

    // State & memory
    ss << "h2d_bytes\t" << h2d_bytes.load(std::memory_order_relaxed) << "\n";
    ss << "d2h_bytes\t" << d2h_bytes.load(std::memory_order_relaxed) << "\n";
    ss << "d2d_bytes\t" << d2d_bytes.load(std::memory_order_relaxed) << "\n";
    ss << "hand_seed_count\t" << hand_seed_count.load(std::memory_order_relaxed) << "\n";
    ss << "hand_seed_bytes\t" << hand_seed_bytes.load(std::memory_order_relaxed) << "\n";
    ss << "clear_count\t" << clear_count.load(std::memory_order_relaxed) << "\n";
    ss << "scratch_peak_bytes\t" << scratch_peak_bytes.load(std::memory_order_relaxed) << "\n";
    ss << "in_flight_slots\t" << in_flight_slots.load(std::memory_order_relaxed) << "\n";
    ss << "max_in_flight_slots\t" << max_in_flight_slots.load(std::memory_order_relaxed) << "\n";

    // Computation
    ss << "q_prep_rows\t" << q_prep_rows.load(std::memory_order_relaxed) << "\n";
    ss << "gemv_count\t" << gemv_count.load(std::memory_order_relaxed) << "\n";
    ss << "gemm_count\t" << gemm_count.load(std::memory_order_relaxed) << "\n";
    ss << "multi_row_gdn_hits\t" << multi_row_gdn_hits.load(std::memory_order_relaxed) << "\n";
    ss << "attention_split_count\t" << attention_split_count.load(std::memory_order_relaxed) << "\n";

    return ss.str();
}

void llama_rerot_profile::print_summary(FILE * stream) const {
    if (!stream) stream = stderr;

    std::fprintf(stream,
        "[rerot-profile] req=%" PRIu64
        " prefill_us=%" PRIu64 " probe_us=%" PRIu64 " rebuild_us=%" PRIu64 " formal_p_us=%" PRIu64
        " workers_start_us=%" PRIu64 " workers_frontier_us=%" PRIu64 " workers_finish_us=%" PRIu64
        " synth_us=%" PRIu64 " send_us=%" PRIu64 "\n",
        request_id,
        phases[(size_t) llama_rerot_phase::normal_prefill].total_us.load(std::memory_order_relaxed),
        phases[(size_t) llama_rerot_phase::probe].total_us.load(std::memory_order_relaxed),
        phases[(size_t) llama_rerot_phase::dag_prefix_rebuild].total_us.load(std::memory_order_relaxed),
        phases[(size_t) llama_rerot_phase::formal_p].total_us.load(std::memory_order_relaxed),
        phases[(size_t) llama_rerot_phase::workers_start].total_us.load(std::memory_order_relaxed),
        phases[(size_t) llama_rerot_phase::workers_frontier].total_us.load(std::memory_order_relaxed),
        phases[(size_t) llama_rerot_phase::workers_finish].total_us.load(std::memory_order_relaxed),
        phases[(size_t) llama_rerot_phase::synthesis].total_us.load(std::memory_order_relaxed),
        phases[(size_t) llama_rerot_phase::result_send].total_us.load(std::memory_order_relaxed));

    // §2.4 cold-transition billing: first occurrence per phase (graph
    // definition, pipeline compile, buffer allocs). steady =
    // (total - first) / (count - 1) once count > 1.
    std::fprintf(stream,
        "[rerot-profile-cold] req=%" PRIu64
        " prefill_first_us=%" PRIu64 " probe_first_us=%" PRIu64
        " rebuild_first_us=%" PRIu64 " formal_p_first_us=%" PRIu64
        " wstart_first_us=%" PRIu64 " wfront_first_us=%" PRIu64
        " wfin_first_us=%" PRIu64 " synth_first_us=%" PRIu64 " send_first_us=%" PRIu64 "\n",
        request_id,
        phases[(size_t) llama_rerot_phase::normal_prefill].first_us.load(std::memory_order_relaxed),
        phases[(size_t) llama_rerot_phase::probe].first_us.load(std::memory_order_relaxed),
        phases[(size_t) llama_rerot_phase::dag_prefix_rebuild].first_us.load(std::memory_order_relaxed),
        phases[(size_t) llama_rerot_phase::formal_p].first_us.load(std::memory_order_relaxed),
        phases[(size_t) llama_rerot_phase::workers_start].first_us.load(std::memory_order_relaxed),
        phases[(size_t) llama_rerot_phase::workers_frontier].first_us.load(std::memory_order_relaxed),
        phases[(size_t) llama_rerot_phase::workers_finish].first_us.load(std::memory_order_relaxed),
        phases[(size_t) llama_rerot_phase::synthesis].first_us.load(std::memory_order_relaxed),
        phases[(size_t) llama_rerot_phase::result_send].first_us.load(std::memory_order_relaxed));

    std::fprintf(stream,
        "[rerot-profile-parallel] req=%" PRIu64
        " snaps=%" PRIu64 " W_sum=%" PRIu64 " W_max=%" PRIu64
        " P_avg=%" PRIu64 " cohort_avg=%" PRIu64 " cohort_max=%" PRIu64
        " frontiers=%" PRIu64 " pen_yields=%" PRIu64 "\n",
        request_id,
        parallel_snapshots.load(std::memory_order_relaxed),
        sum_w.load(std::memory_order_relaxed),
        max_w.load(std::memory_order_relaxed),
        parallel_snapshots.load(std::memory_order_relaxed) > 0
            ? sum_p.load(std::memory_order_relaxed) / parallel_snapshots.load(std::memory_order_relaxed) : 0,
        parallel_snapshots.load(std::memory_order_relaxed) > 0
            ? sum_cohort.load(std::memory_order_relaxed) / parallel_snapshots.load(std::memory_order_relaxed) : 0,
        max_cohort.load(std::memory_order_relaxed),
        logical_frontiers.load(std::memory_order_relaxed),
        pen_yields.load(std::memory_order_relaxed));

    std::fprintf(stream,
        "[rerot-profile-routes] req=%" PRIu64
        " indexed_hits=%" PRIu64 " flashprefill_hits=%" PRIu64 " xkv_hits=%" PRIu64 " cpu_fallback_hits=%" PRIu64 "\n",
        request_id,
        routes[(size_t) llama_rerot_route::indexed_rerot].hits.load(std::memory_order_relaxed),
        routes[(size_t) llama_rerot_route::flashprefill].hits.load(std::memory_order_relaxed),
        routes[(size_t) llama_rerot_route::xkv].hits.load(std::memory_order_relaxed),
        routes[(size_t) llama_rerot_route::cpu_fallback].hits.load(std::memory_order_relaxed));

    std::fprintf(stream,
        "[rerot-profile-layout] req=%" PRIu64
        " groups=%" PRIu64 "/%" PRIu64 " entries=%" PRIu64 "/%" PRIu64 " q_rows=%" PRIu64
        " runs=%" PRIu64 " spans=%" PRIu64 " hwm_kv=%" PRIu64 "\n",
        request_id,
        live_groups.load(std::memory_order_relaxed), cap_groups.load(std::memory_order_relaxed),
        live_entries.load(std::memory_order_relaxed), cap_entries.load(std::memory_order_relaxed),
        actual_query_rows.load(std::memory_order_relaxed),
        run_count.load(std::memory_order_relaxed),
        continuous_span_count.load(std::memory_order_relaxed),
        high_watermark_n_kv.load(std::memory_order_relaxed));

    std::fprintf(stream,
        "[rerot-profile-host] req=%" PRIu64
        " view_build_us=%" PRIu64 " up_bytes=%" PRIu64 " up_stg_us=%" PRIu64
        " up_set_us=%" PRIu64 " up_n=%" PRIu64 " gdefs=%" PRIu64
        " cb_allocs=%" PRIu64 " cb_bytes=%" PRIu64 " submits=%" PRIu64
        " sync_us=%" PRIu64 " sync_n=%" PRIu64 " gen=%" PRIu64 "\n",
        request_id,
        layout_view_build_us.load(std::memory_order_relaxed),
        upload_bytes.load(std::memory_order_relaxed),
        upload_staging_us.load(std::memory_order_relaxed),
        upload_set_us.load(std::memory_order_relaxed),
        upload_count.load(std::memory_order_relaxed),
        graph_def_count.load(std::memory_order_relaxed),
        cb_allocations.load(std::memory_order_relaxed),
        cb_alloc_bytes.load(std::memory_order_relaxed),
        submit_count.load(std::memory_order_relaxed),
        sync_us_total.load(std::memory_order_relaxed),
        sync_count.load(std::memory_order_relaxed),
        completed_generation.load(std::memory_order_relaxed));

    std::fprintf(stream,
        "[rerot-profile-state] req=%" PRIu64
        " h2d=%" PRIu64 " d2h=%" PRIu64 " d2d=%" PRIu64
        " hand_seed_n=%" PRIu64 " hand_seed_bytes=%" PRIu64 " clears=%" PRIu64
        " scratch_peak=%" PRIu64 " in_flight_slots=%" PRIu64 "/%" PRIu64 "\n",
        request_id,
        h2d_bytes.load(std::memory_order_relaxed),
        d2h_bytes.load(std::memory_order_relaxed),
        d2d_bytes.load(std::memory_order_relaxed),
        hand_seed_count.load(std::memory_order_relaxed),
        hand_seed_bytes.load(std::memory_order_relaxed),
        clear_count.load(std::memory_order_relaxed),
        scratch_peak_bytes.load(std::memory_order_relaxed),
        in_flight_slots.load(std::memory_order_relaxed),
        max_in_flight_slots.load(std::memory_order_relaxed));

    std::fprintf(stream,
        "[rerot-profile-compute] req=%" PRIu64
        " q_prep=%" PRIu64 " gemv=%" PRIu64 " gemm=%" PRIu64
        " multi_row_gdn=%" PRIu64 " attn_splits=%" PRIu64 "\n",
        request_id,
        q_prep_rows.load(std::memory_order_relaxed),
        gemv_count.load(std::memory_order_relaxed),
        gemm_count.load(std::memory_order_relaxed),
        multi_row_gdn_hits.load(std::memory_order_relaxed),
        attention_split_count.load(std::memory_order_relaxed));
}

// Global active profiling instance & lock
static std::mutex          g_rerot_prof_mu;
static llama_rerot_profile g_rerot_prof_storage;
static llama_rerot_profile * g_rerot_prof = nullptr;

bool llama_rerot_profile_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("LLAMA_REROT_PROFILE");
        return env != nullptr && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

llama_rerot_profile * llama_rerot_profile_active() {
    return g_rerot_prof;
}

void llama_rerot_profile_begin(uint64_t request_id) {
    if (!llama_rerot_profile_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_rerot_prof_mu);
    g_rerot_prof_storage.reset(request_id);
    g_rerot_prof = &g_rerot_prof_storage;
}

void llama_rerot_profile_end() {
    std::lock_guard<std::mutex> lock(g_rerot_prof_mu);
    if (g_rerot_prof && llama_rerot_profile_enabled()) {
        g_rerot_prof->print_summary(stderr);
    }
    g_rerot_prof = nullptr;
}
