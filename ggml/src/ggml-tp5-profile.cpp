#include "ggml-tp5-profile.h"

#include <cstdlib>
#include <cstring>

const char * ggml_tp5_rerot_reject_reason_name(ggml_tp5_rerot_reject_reason reason) {
    switch (reason) {
        case ggml_tp5_rerot_reject_reason::none:                    return "none";
        case ggml_tp5_rerot_reject_reason::shmem_tune:              return "shmem_tune";
        case ggml_tp5_rerot_reject_reason::unsupported_shape:       return "unsupported_shape";
        case ggml_tp5_rerot_reject_reason::unsupported_type:        return "unsupported_type";
        case ggml_tp5_rerot_reject_reason::subgroup:                return "subgroup";
        case ggml_tp5_rerot_reject_reason::unsupported_quant_pq2_0: return "unsupported_quant_pq2_0";
        case ggml_tp5_rerot_reject_reason::layout:                   return "layout";
        default:                                                     return "unknown";
    }
}

static ggml_tp5_profile * g_prof = nullptr;
static std::mutex         g_prof_mu;
static ggml_tp5_profile   g_prof_storage;

void ggml_tp5_profile::reset(uint64_t new_id, bool decode) {
    graph_exec_id = new_id;
    is_decode     = decode;
    queue_submits = 0;
    submit_batches = 0;
    cb_allocations = 0;
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
    relay_chains = 0;
    relay_direct_stages = 0;
    relay_fallback_stages = 0;
    relay_route_patch_us = 0;
    relay_submit_us = 0;
    relay_ready_wait_us = 0;
    relay_ready_wait_max_us = 0;
    relay_ready_skew_us = 0;
    relay_ready_skew_max_us = 0;
    relay_arm_us = 0;
    relay_cpu_data_us = 0;
    relay_sidecar_wait_us = 0;
    relay_sidecar_data_us = 0;
    relay_y_publish_us = 0;
    relay_q_publish_us = 0;
    relay_q_spin_iters = 0;
    relay_q_spin_samples = 0;
    relay_q_spin_max = 0;
    relay_generation_us = 0;
    relay_handoff_total_us = 0;
    relay_poll_iters = 0;
    relay_gpu_spin_iters = 0;
    relay_gpu_spin_samples = 0;
    relay_gpu_spin_max = 0;
    vk_rerot_dispatch = 0;
    vk_rerot_split_k_total = 0;
    vk_rerot_split_k_max = 0;
    vk_rerot_queries = 0;
    vk_rerot_entries = 0;
    vk_rerot_live_entries = 0;
    vk_rerot_cap_entries = 0;
    vk_rerot_shmem_reject = 0;
    for (size_t i = 0; i < (size_t) ggml_tp5_rerot_reject_reason::count; ++i) {
        vk_rerot_reject_reasons[i] = 0;
    }
    vk_gdn_dispatch = 0;
    for (size_t i = 0; i < (size_t) ggml_tp5_rerot_reject_reason::count; ++i) {
        vk_gdn_reject_reasons[i] = 0;
    }
}

void ggml_tp5_profile::print_summary() const {
    const uint64_t spin_n = relay_gpu_spin_samples.load();
    const uint64_t spin_total = relay_gpu_spin_iters.load();
    fprintf(stderr,
            "[tp5-profile] exec=%" PRIu64 " decode=%d "
            "submits=%" PRIu64 " batches=%" PRIu64 " cb_alloc=%" PRIu64 " "
            "wait_us=%" PRIu64 " wait_n=%" PRIu64 " bp_us=%" PRIu64 " "
            "collective=%" PRIu64 " fd_exp=%" PRIu64 " fd_imp=%" PRIu64 " "
            "drm_wait=%" PRIu64 " "
            "creplay=%" PRIu64 "/%" PRIu64 " plan=%" PRIu64 "/%" PRIu64 "\n",
            graph_exec_id, (int) is_decode,
            queue_submits.load(), submit_batches.load(), cb_allocations.load(),
            host_wait_us.load(), host_wait_count.load(), backpressure_us.load(),
            collective_calls.load(), fd_exports.load(), fd_imports.load(),
            drm_wait_hits.load(),
            compute_replay_hits.load(), compute_replay_misses.load(),
            collective_plan_hits.load(), collective_plan_misses.load());
    if (relay_chains.load() != 0) {
        fprintf(stderr,
                "[tp5-relay-profile] exec=%" PRIu64 " chains=%" PRIu64
                " stages=%" PRIu64 " direct=%" PRIu64 " fallback=%" PRIu64
                " route_us=%" PRIu64 " submit_us=%" PRIu64
                " ready_us=%" PRIu64 " ready_max_us=%" PRIu64
                " skew_us=%" PRIu64 " skew_max_us=%" PRIu64
                " arm_us=%" PRIu64 " cpu_data_us=%" PRIu64
                " generation_us=%" PRIu64 " handoff_us=%" PRIu64
                " polls=%" PRIu64 " gpu_spin_avg=%.1f gpu_spin_max=%" PRIu64
                " gpu_spin_n=%" PRIu64 "\n",
                graph_exec_id, relay_chains.load(),
                relay_direct_stages.load() + relay_fallback_stages.load(),
                relay_direct_stages.load(), relay_fallback_stages.load(),
                relay_route_patch_us.load(), relay_submit_us.load(),
                relay_ready_wait_us.load(), relay_ready_wait_max_us.load(),
                relay_ready_skew_us.load(), relay_ready_skew_max_us.load(),
                relay_arm_us.load(), relay_cpu_data_us.load(),
                relay_generation_us.load(), relay_handoff_total_us.load(),
                relay_poll_iters.load(),
                spin_n ? double(spin_total) / double(spin_n) : 0.0,
                relay_gpu_spin_max.load(), spin_n);
        fprintf(stderr,
                "[tp5-latebind-profile] exec=%" PRIu64 " sidecar_wait_us=%" PRIu64
                " sidecar_data_us=%" PRIu64 " y_publish_us=%" PRIu64 " q_publish_us=%" PRIu64
                " q_spin_avg=%.1f q_spin_max=%" PRIu64 " q_spin_n=%" PRIu64
                " cpu_data_excludes_wait=1\n",
                graph_exec_id, relay_sidecar_wait_us.load(), relay_sidecar_data_us.load(),
                relay_y_publish_us.load(), relay_q_publish_us.load(),
                relay_q_spin_samples.load() ? double(relay_q_spin_iters.load()) / double(relay_q_spin_samples.load()) : -1.0,
                relay_q_spin_max.load(), relay_q_spin_samples.load());
    }
    const uint64_t r_rej_type = vk_rerot_reject_reasons[(size_t) ggml_tp5_rerot_reject_reason::unsupported_type].load();
    const uint64_t r_rej_pq2_0 = vk_rerot_reject_reasons[(size_t) ggml_tp5_rerot_reject_reason::unsupported_quant_pq2_0].load();
    const uint64_t r_rej_shape = vk_rerot_reject_reasons[(size_t) ggml_tp5_rerot_reject_reason::unsupported_shape].load();
    const uint64_t r_rej_subgrp = vk_rerot_reject_reasons[(size_t) ggml_tp5_rerot_reject_reason::subgroup].load();
    const uint64_t r_rej_layout = vk_rerot_reject_reasons[(size_t) ggml_tp5_rerot_reject_reason::layout].load();
    if (vk_rerot_dispatch.load() != 0 || vk_rerot_shmem_reject.load() != 0 || r_rej_type != 0 || r_rej_pq2_0 != 0 ||
        r_rej_shape != 0 || r_rej_subgrp != 0 || r_rej_layout != 0) {
        const uint64_t r_disp = vk_rerot_dispatch.load();
        const uint64_t r_sk_tot = vk_rerot_split_k_total.load();
        fprintf(stderr,
                "[tp5-rerot-profile] exec=%" PRIu64 " dispatch=%" PRIu64
                " split_k_avg=%.2f split_k_max=%" PRIu64
                " queries=%" PRIu64 " entries=%" PRIu64
                " live_entries=%" PRIu64 " cap_entries=%" PRIu64
                " shmem_reject=%" PRIu64 " reject_tune=%" PRIu64
                " reject_type=%" PRIu64 " reject_pq2_0=%" PRIu64
                " reject_shape=%" PRIu64 " reject_subgroup=%" PRIu64
                " reject_layout=%" PRIu64 "\n",
                graph_exec_id, r_disp,
                r_disp ? double(r_sk_tot) / double(r_disp) : 0.0,
                vk_rerot_split_k_max.load(),
                vk_rerot_queries.load(), vk_rerot_entries.load(),
                vk_rerot_live_entries.load(), vk_rerot_cap_entries.load(),
                vk_rerot_shmem_reject.load(),
                vk_rerot_reject_reasons[(size_t) ggml_tp5_rerot_reject_reason::shmem_tune].load(),
                r_rej_type,
                r_rej_pq2_0,
                r_rej_shape,
                r_rej_subgrp,
                r_rej_layout);
    }
    const uint64_t g_disp = vk_gdn_dispatch.load();
    const uint64_t g_rej_shape = vk_gdn_reject_reasons[(size_t) ggml_tp5_rerot_reject_reason::unsupported_shape].load();
    const uint64_t g_rej_type = vk_gdn_reject_reasons[(size_t) ggml_tp5_rerot_reject_reason::unsupported_type].load();
    const uint64_t g_rej_layout = vk_gdn_reject_reasons[(size_t) ggml_tp5_rerot_reject_reason::layout].load();
    if (g_disp != 0 || g_rej_shape != 0 || g_rej_type != 0 || g_rej_layout != 0) {
        fprintf(stderr,
                "[tp5-gdn-profile] exec=%" PRIu64 " dispatch=%" PRIu64
                " reject_shape=%" PRIu64 " reject_type=%" PRIu64
                " reject_layout=%" PRIu64 "\n",
                graph_exec_id, g_disp, g_rej_shape, g_rej_type, g_rej_layout);
    }
}

ggml_tp5_profile * ggml_tp5_profile_active() {
    return g_prof;
}

void ggml_tp5_profile_begin(uint64_t exec_id, bool decode) {
    const char * env = getenv("GGML_TP5_PROFILE");
    if (!env || strcmp(env, "0") == 0 || strcmp(env, "off") == 0 || strcmp(env, "false") == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_prof_mu);
    g_prof_storage.reset(exec_id, decode);
    g_prof = &g_prof_storage;
}

void ggml_tp5_profile_end() {
    std::lock_guard<std::mutex> lock(g_prof_mu);
    const char * env = getenv("GGML_TP5_PROFILE");
    const bool env_active = env && strcmp(env, "0") != 0 && strcmp(env, "off") != 0 && strcmp(env, "false") != 0;
    if (g_prof && env_active) {
        g_prof->print_summary();
    }
    g_prof = nullptr;
}
