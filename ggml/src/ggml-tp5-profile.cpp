#include "ggml-tp5-profile.h"

#include <cstdlib>

static ggml_tp5_profile * g_prof = nullptr;
static std::mutex         g_prof_mu;
static ggml_tp5_profile   g_prof_storage;

void ggml_tp5_profile::reset(uint64_t new_id, bool decode) {
    graph_exec_id = new_id;
    is_decode     = decode;
    queue_submits = 0;
    submit_batches = 0;
    host_wait_us = 0;
    host_wait_count = 0;
    backpressure_us = 0;
    collective_calls = 0;
    fd_exports = 0;
    fd_imports = 0;
    compute_replay_hits = 0;
    compute_replay_misses = 0;
    collective_plan_hits = 0;
    collective_plan_misses = 0;
    drm_wait_hits = 0;
}

void ggml_tp5_profile::print_summary() const {
    fprintf(stderr,
            "[tp5-profile] exec=%" PRIu64 " decode=%d "
            "submits=%" PRIu64 " batches=%" PRIu64 " "
            "wait_us=%" PRIu64 " wait_n=%" PRIu64 " bp_us=%" PRIu64 " "
            "collective=%" PRIu64 " fd_exp=%" PRIu64 " fd_imp=%" PRIu64 " "
            "drm_wait=%" PRIu64 " "
            "creplay=%" PRIu64 "/%" PRIu64 " plan=%" PRIu64 "/%" PRIu64 "\n",
            graph_exec_id, (int) is_decode,
            queue_submits.load(), submit_batches.load(),
            host_wait_us.load(), host_wait_count.load(), backpressure_us.load(),
            collective_calls.load(), fd_exports.load(), fd_imports.load(),
            drm_wait_hits.load(),
            compute_replay_hits.load(), compute_replay_misses.load(),
            collective_plan_hits.load(), collective_plan_misses.load());
}

ggml_tp5_profile * ggml_tp5_profile_active() {
    return g_prof;
}

void ggml_tp5_profile_begin(uint64_t exec_id, bool decode) {
    if (!getenv("GGML_TP5_PROFILE")) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_prof_mu);
    g_prof_storage.reset(exec_id, decode);
    g_prof = &g_prof_storage;
}

void ggml_tp5_profile_end() {
    std::lock_guard<std::mutex> lock(g_prof_mu);
    if (g_prof && getenv("GGML_TP5_PROFILE")) {
        g_prof->print_summary();
    }
    g_prof = nullptr;
}
