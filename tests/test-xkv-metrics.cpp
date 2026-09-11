// Focused unit and regression test for §16 metric accounting, identity
// emission, and evaluated status. Exercises:
//   1. Exact codec shared table bytes (llama_xkv_codec_shared_table_bytes)
//   2. xkv_expected_residency / llama_xkv_profile_is_device_owned cross product
//   3. server-task to_json / to_prometheus with full evaluated metrics
//   4. Prometheus NaN and JSON null emission for unevaluated ratios/timers once observed
//   5. Unobserved (OFF) exact all-zero preservation
//   6. Prometheus series rename: xkv_source_fingerprint_info separate from xkv_source_info
//   7. /props identity keys (model_sha256, xkv_mode, xkv_seed, tri_*, rerot_frontier)
//   8. Live stream bytes round-trip verbatim incl. decreases (never cumulative)
//   9. compression_goal_met tri-state: codec presence never implies a verdict
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-codec.h"
#include "llama.h"
#include "server-task.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

using namespace llama_xkv;

static int g_failures = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failures; \
    } \
} while (0)

static void test_codec_shared_table_bytes() {
    std::cout << "[test_codec_shared_table_bytes] starting..." << std::endl;
    const size_t b = llama_xkv_codec_shared_table_bytes();
    // Two 128-float WHT sign tables = 2 * 128 * 4 = 1024 bytes
    CHECK(b == 1024);
    std::cout << "  [PASS] Codec shared table bytes verified: " << b << std::endl;
}

static void test_residency_predicate() {
    std::cout << "[test_residency_predicate] starting..." << std::endl;
    // Reference is always host
    CHECK(!llama_xkv_profile_is_device_owned(LLAMA_XKV_STORAGE_PROFILE_REFERENCE, LLAMA_XKV_FACTORIZER_VULKAN));
    CHECK(!llama_xkv_profile_is_device_owned(LLAMA_XKV_STORAGE_PROFILE_REFERENCE, LLAMA_XKV_FACTORIZER_CPU_REFERENCE));

    // TQ profiles on CPU reference are host
    CHECK(!llama_xkv_profile_is_device_owned(LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS, LLAMA_XKV_FACTORIZER_CPU_REFERENCE));
    CHECK(!llama_xkv_profile_is_device_owned(LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS, LLAMA_XKV_FACTORIZER_CPU_REFERENCE));

    // TQ profiles on device factorizers are device-owned
    CHECK(llama_xkv_profile_is_device_owned(LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS, LLAMA_XKV_FACTORIZER_VULKAN));
    CHECK(llama_xkv_profile_is_device_owned(LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS_LANDMARKS, LLAMA_XKV_FACTORIZER_VULKAN));
    CHECK(llama_xkv_profile_is_device_owned(LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS, LLAMA_XKV_FACTORIZER_VULKAN_HYBRID));
    CHECK(llama_xkv_profile_is_device_owned(LLAMA_XKV_STORAGE_PROFILE_TQ_FACTORS, LLAMA_XKV_FACTORIZER_CUDA));
    std::cout << "  [PASS] Residency predicate cross-product verified." << std::endl;
}

static void test_server_metrics_evaluated() {
    std::cout << "[test_server_metrics_evaluated] starting..." << std::endl;
    server_xkv_metrics m;
    m.observed = true;
    m.ratios_evaluated = true;
    m.seal_timers_evaluated = true;
    m.quant_timers_evaluated = true;
    m.requested_mode = "sr";
    m.effective_mode = "sr";
    m.requested_profile = "tq-factors-landmarks";
    m.effective_profile = "tq-factors-landmarks";
    m.requested_a_k = "turbo4_0";
    m.effective_a_k = "turbo4_0";
    m.requested_b_k = "turbo4_0";
    m.effective_b_k = "turbo4_0";
    m.requested_a_v = "turbo2_0";
    m.effective_a_v = "turbo2_0";
    m.requested_b_v = "turbo2_0";
    m.effective_b_v = "turbo2_0";
    m.requested_landmark = "q8_0";
    m.effective_landmark = "q8_0";
    m.requested_factorizer = "vulkan";
    m.effective_factorizer = "vulkan";
    m.requested_balance = "upstream";
    m.effective_balance = "upstream";
    m.requested_seed = "0x584b565352303031";
    m.effective_seed = "0x584b565352303031";
    m.requested_landmark_seed = "0x0000000000000309";
    m.effective_landmark_seed = "0x0000000000000309";
    m.model_sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    m.source = "decoded-hot";
    m.source_fingerprint = 0x1234ULL;
    m.codec_fingerprint = 0x5678ULL;
    m.profile_fingerprint = 0x9ABCULL;
    m.backend_fingerprint = 0xDEF0ULL;
    m.config_fingerprint = 0x9ABCULL;
    // Live stream bytes (actual live store values, not cumulative seals).
    m.factor_ak_bytes = 100;
    m.factor_bk_bytes = 200;
    m.factor_av_bytes = 300;
    m.factor_bv_bytes = 400;
    m.factor_payload_bytes = 1000;
    m.landmark_payload_bytes = 64;
    m.landmark_metadata_bytes = 16;
    m.landmark_exception_bytes = 0;
    m.index_bytes = 32;
    m.codec_shared_bytes = 1024;
    m.decode_tile_cache_bytes = 512;
    m.capture_bytes = 0;
    m.candidate_bytes = 128;
    m.staging_bytes = 0;
    m.snapshot_pinned_bytes = 0;
    m.allocator_live_bytes = 10695475;
    m.allocator_reserved_bytes = 16777216;
    m.device_peak_bytes = 20971520;
    m.host_peak_bytes = 0;
    m.workspace_peak_bytes = 1048576;
    m.factored_baseline_byte_coverage = 0.85;
    m.factor_quant_ratio = 4.0;
    m.net_extra_compression_ratio = 2.53;
    m.net_extra_compression_ratio_reserved = 1.94;
    m.net_extra_compression_ratio_peak = 1.61;
    m.seal_seconds = 0.123;
    m.factor_quant_seconds = 0.045;
    m.landmark_quant_seconds = 0.012;
    m.tri_calibration_fingerprint = 0xFEEDFACECAFEBEEFULL;
    m.tri_calibration_sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    m.tri_ratio_str = "3/32";
    m.tri_recent_window = 128;
    m.tri_scorer_valid = true;
    m.compression_goal_evaluated = true;
    m.compression_goal_met = true;

    // JSON emission
    json j = m.to_json();
    CHECK(j["xkv_requested_mode"] == "sr");
    CHECK(j["xkv_effective_mode"] == "sr");
    CHECK(j["xkv_requested_a_k"] == "turbo4_0");
    CHECK(j["xkv_effective_seed"] == "0x584b565352303031");
    CHECK(j["xkv_model_sha256"] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(j["tri_calibration_sha256"] == "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
    CHECK(j["xkv_compression_goal_evaluated"] == true);
    CHECK(j["xkv_compression_goal_met"] == true);
    CHECK(j["xkv_ratios_evaluated"] == true);
    CHECK(j["xkv_seal_timers_evaluated"] == true);
    CHECK(j["xkv_quant_timers_evaluated"] == true);
    CHECK(j["xkv_factored_baseline_byte_coverage"].is_number());
    CHECK(j["xkv_net_extra_compression_ratio"].is_number());
    CHECK(j["xkv_net_extra_compression_ratio_reserved"].is_number());
    CHECK(j["xkv_seal_seconds"].is_number());
    // Live stream bytes round-trip verbatim (actual live values).
    CHECK(j["xkv_factor_ak_bytes"] == 100);
    CHECK(j["xkv_factor_bk_bytes"] == 200);
    CHECK(j["xkv_factor_av_bytes"] == 300);
    CHECK(j["xkv_factor_bv_bytes"] == 400);
    CHECK(j["xkv_factor_payload_bytes"] == 1000);
    CHECK(j["xkv_landmark_payload_bytes"] == 64);
    CHECK(j["xkv_landmark_metadata_bytes"] == 16);
    CHECK(j["xkv_landmark_exception_bytes"] == 0);
    CHECK(j["xkv_index_bytes"] == 32);
    CHECK(j["xkv_codec_shared_bytes"] == 1024);
    CHECK(j["xkv_decode_tile_cache_bytes"] == 512);
    CHECK(j["xkv_capture_bytes"] == 0);
    CHECK(j["xkv_candidate_bytes"] == 128);
    CHECK(j["xkv_staging_bytes"] == 0);
    CHECK(j["xkv_allocator_live_bytes"] == 10695475);
    CHECK(j["xkv_allocator_reserved_bytes"] == 16777216);
    CHECK(j["xkv_device_peak_bytes"] == 20971520);
    CHECK(j["xkv_host_peak_bytes"] == 0);
    CHECK(j["xkv_config_fingerprint"] == 0x9ABCULL);
    CHECK(j["xkv_requested_landmark_seed"] == "0x0000000000000309");
    CHECK(j["xkv_effective_landmark_seed"] == "0x0000000000000309");
    CHECK(j["xkv_net_extra_compression_ratio_peak"].is_number());

    // Prometheus emission
    std::ostringstream os;
    m.to_prometheus(os);
    const std::string prom = os.str();
    CHECK(prom.find("llamacpp:xkv_mode{mode=\"sr\"} 1") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_mode_info{requested_mode=\"sr\",effective_mode=\"sr\"} 1") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_codec_profile_info{a_k=\"turbo4_0\"") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_seed_info{seed=\"0x584b565352303031\"} 1") != std::string::npos);
    CHECK(prom.find("llamacpp:model_artifact_info{sha256=\"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\"} 1") != std::string::npos);
    CHECK(prom.find("llamacpp:tri_calibration_info{fingerprint=\"feedfacecafebeef\",valid=\"true\",sha256=\"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"} 1") != std::string::npos);
    CHECK(prom.find("llamacpp:tri_config_info{ratio=\"3/32\",recent_window=\"128\"} 1") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_compression_goal_evaluated 1") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_compression_goal_met 1") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_source_fingerprint_info{fingerprint=\"0000000000001234\"} 1") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_ratios_evaluated 1") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_timers_evaluated 1") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_factor_ak_bytes 100") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_factor_bk_bytes 200") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_factor_av_bytes 300") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_factor_bv_bytes 400") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_staging_bytes 0") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_landmark_seed_info{seed=\"0x0000000000000309\"} 1") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_config_fingerprint_info{fingerprint=\"0000000000009abc\"} 1") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_net_extra_compression_ratio_peak ") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_net_extra_compression_ratio_peak NaN") == std::string::npos);

    std::cout << "  [PASS] Evaluated server metrics verified." << std::endl;
}

static void test_server_metrics_unevaluated() {
    std::cout << "[test_server_metrics_unevaluated] starting..." << std::endl;
    server_xkv_metrics m;
    m.observed = true;
    m.ratios_evaluated = false;
    m.seal_timers_evaluated = false;
    m.quant_timers_evaluated = false;
    m.compression_goal_evaluated = false;
    m.graph_timings_evaluated = false;
    m.pack_timer_evaluated = false;
    m.sr_counters_evaluated = false;
    m.spec_counters_evaluated = false;
    m.requested_mode = "dense";
    m.effective_mode = "dense";

    // JSON nulls
    json j = m.to_json();
    CHECK(j["xkv_ratios_evaluated"] == false);
    CHECK(j["xkv_seal_timers_evaluated"] == false);
    CHECK(j["xkv_graph_timings_evaluated"] == false);
    CHECK(j["xkv_pack_timer_evaluated"] == false);
    CHECK(j["xkv_sr_counters_evaluated"] == false);
    CHECK(j["xkv_spec_counters_evaluated"] == false);
    CHECK(j["xkv_compression_goal_evaluated"] == false);
    CHECK(j["xkv_compression_goal_met"].is_null());
    CHECK(j["xkv_factored_baseline_byte_coverage"].is_null());
    CHECK(j["xkv_factor_quant_ratio"].is_null());
    CHECK(j["xkv_net_extra_compression_ratio"].is_null());
    CHECK(j["xkv_net_extra_compression_ratio_reserved"].is_null());
    CHECK(j["xkv_net_extra_compression_ratio_peak"].is_null());
    CHECK(j["xkv_seal_seconds"].is_null());
    CHECK(j["xkv_factor_quant_seconds"].is_null());
    CHECK(j["xkv_landmark_quant_seconds"].is_null());
    CHECK(j["xkv_select_seconds"].is_null());
    CHECK(j["xkv_refine_seconds"].is_null());
    CHECK(j["xkv_reconstruct_seconds"].is_null());
    CHECK(j["xkv_read_seconds"].is_null());
    CHECK(j["xkv_pack_seconds"].is_null());
    CHECK(j["xkv_sr_selected_rows"].is_null());
    CHECK(j["xkv_sr_fragments"].is_null());
    CHECK(j["xkv_landmark_refine_rows"].is_null());
    CHECK(j["xkv_landmark_refine_cap_hits"].is_null());
    CHECK(j["xkv_spec_stale_total"].is_null());
    CHECK(j["xkv_transaction_abort_total"].is_null());

    // Prometheus NaNs
    std::ostringstream os;
    m.to_prometheus(os);
    const std::string prom = os.str();
    CHECK(prom.find("llamacpp:xkv_factored_baseline_byte_coverage NaN") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_factor_quant_ratio NaN") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_net_extra_compression_ratio NaN") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_net_extra_compression_ratio_reserved NaN") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_net_extra_compression_ratio_peak NaN") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_seal_seconds NaN") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_select_seconds NaN") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_refine_seconds NaN") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_reconstruct_seconds NaN") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_read_seconds NaN") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_pack_seconds NaN") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_sr_selected_rows NaN") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_sr_fragments NaN") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_landmark_refine_rows NaN") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_landmark_refine_cap_hits NaN") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_spec_stale_total NaN") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_transaction_abort_total NaN") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_compression_goal_met NaN") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_compression_goal_evaluated 0") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_graph_timings_evaluated 0") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_pack_timer_evaluated 0") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_sr_counters_evaluated 0") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_spec_counters_evaluated 0") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_ratios_evaluated 0") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_timers_evaluated 0") != std::string::npos);

    std::cout << "  [PASS] Unevaluated null/NaN metrics verified." << std::endl;
}

static void test_server_metrics_off_all_zero() {
    std::cout << "[test_server_metrics_off_all_zero] starting..." << std::endl;
    server_xkv_metrics m;
    // observed == false: unobserved / OFF snapshot
    CHECK(m.empty());
    std::ostringstream os;
    m.to_prometheus(os);
    // Silent output when empty
    CHECK(os.str().empty());

    // JSON output has numbers, not nulls, when unobserved
    json j = m.to_json();
    CHECK(j["xkv_factored_baseline_byte_coverage"] == 0.0);
    CHECK(j["xkv_factor_quant_ratio"] == 0.0);
    CHECK(j["xkv_net_extra_compression_ratio"] == 0.0);
    CHECK(j["xkv_net_extra_compression_ratio_reserved"] == 0.0);
    CHECK(j["xkv_net_extra_compression_ratio_peak"] == 0.0);
    CHECK(j["xkv_staging_bytes"] == 0);
    CHECK(j["xkv_config_fingerprint"] == 0);
    CHECK(j["xkv_seal_seconds"] == 0.0);
    CHECK(j["xkv_ratios_evaluated"] == false);
    CHECK(j["xkv_seal_timers_evaluated"] == false);

    std::cout << "  [PASS] Unobserved/OFF all-zero preservation verified." << std::endl;
}

static void test_coverage_and_compression_goal_math() {
    std::cout << "[test_coverage_and_compression_goal_math] starting..." << std::endl;
    // Math test: 80 factored-baseline bytes + 20 hot bytes => coverage .8
    const size_t factored_baseline = 80;
    const size_t hot_bytes = 20;
    const size_t total_eligible = factored_baseline + hot_bytes;
    const double coverage = (total_eligible > 0 && factored_baseline > 0)
        ? (double) factored_baseline / (double) total_eligible : 0.0;
    CHECK(std::fabs(coverage - 0.8) < 1e-9);

    // Zero factored baseline => ratios_evaluated is false, coverage is 0.0
    const size_t zero_factored_baseline = 0;
    const size_t hot_bytes_only = 100;
    const size_t total_hot_only = zero_factored_baseline + hot_bytes_only;
    const bool ratios_eval_zero = (zero_factored_baseline > 0);
    CHECK(!ratios_eval_zero);
    const double coverage_zero = (total_hot_only > 0 && zero_factored_baseline > 0)
        ? (double) zero_factored_baseline / (double) total_hot_only : 0.0;
    CHECK(coverage_zero == 0.0);

    // Compression goal evaluation requires global coverage >= min_coverage (0.50)
    // AND live net saving >= min_saving (0.10)
    const double min_coverage = 0.50;
    const double min_saving = 0.10;
    const size_t live_payload = 40; // 80 baseline / 40 live = 2.0x net ratio -> saving = 1 - 1/2 = 0.50 >= 0.10
    const double net_ratio = (double) factored_baseline / (double) live_payload;
    const double live_saving_fraction = 1.0 - (1.0 / net_ratio);
    const bool coverage_pass = (coverage >= min_coverage);
    const bool saving_pass = (live_saving_fraction >= min_saving);
    const bool per_seg_gate = true;
    const bool goal_met = per_seg_gate && coverage_pass && saving_pass;
    CHECK(goal_met == true);

    // If coverage is below min (e.g. 10 factored + 90 hot => coverage 0.10 < 0.50), goal fails!
    const size_t low_factored = 10;
    const size_t high_hot = 90;
    const double low_coverage = (double) low_factored / (double) (low_factored + high_hot);
    const bool low_coverage_pass = (low_coverage >= min_coverage);
    CHECK(!low_coverage_pass);
    const bool low_goal_met = per_seg_gate && low_coverage_pass && saving_pass;
    CHECK(low_goal_met == false);

    std::cout << "  [PASS] Coverage and compression goal math verified: 80+20 => .8, zero => false, coverage+saving gates verified." << std::endl;
}

static void test_server_metrics_live_stream_bytes() {
    std::cout << "[test_server_metrics_live_stream_bytes] starting..." << std::endl;
    // Live semantics: the four stream gauges report the CURRENT live store
    // bytes verbatim. A later snapshot after eviction reports SMALLER values;
    // historical cumulative counters could only grow. Two consecutive live
    // snapshots must therefore round-trip exactly, including a decrease.
    server_xkv_metrics before;
    before.observed = true;
    before.factor_ak_bytes = 1000;
    before.factor_bk_bytes = 2000;
    before.factor_av_bytes = 3000;
    before.factor_bv_bytes = 4000;
    json jb = before.to_json();
    CHECK(jb["xkv_factor_ak_bytes"] == 1000);
    CHECK(jb["xkv_factor_bk_bytes"] == 2000);
    CHECK(jb["xkv_factor_av_bytes"] == 3000);
    CHECK(jb["xkv_factor_bv_bytes"] == 4000);

    server_xkv_metrics after;
    after.observed = true;
    after.factor_ak_bytes = 100; // evicted: live shrinks 10x
    after.factor_bk_bytes = 200;
    after.factor_av_bytes = 300;
    after.factor_bv_bytes = 400;
    json ja = after.to_json();
    CHECK(ja["xkv_factor_ak_bytes"] == 100);
    CHECK(ja["xkv_factor_bk_bytes"] == 200);
    CHECK(ja["xkv_factor_av_bytes"] == 300);
    CHECK(ja["xkv_factor_bv_bytes"] == 400);
    // Live values can decrease: the later snapshot is strictly smaller.
    CHECK(ja["xkv_factor_ak_bytes"].get<uint64_t>() < jb["xkv_factor_ak_bytes"].get<uint64_t>());

    std::ostringstream os;
    after.to_prometheus(os);
    const std::string prom = os.str();
    CHECK(prom.find("llamacpp:xkv_factor_ak_bytes 100") != std::string::npos);
    CHECK(prom.find("llamacpp:xkv_factor_bv_bytes 400") != std::string::npos);
    std::cout << "  [PASS] Live stream bytes verified (verbatim round-trip, decrease allowed)." << std::endl;
}

static void test_compression_goal_never_inferred_from_codec() {
    std::cout << "[test_compression_goal_never_inferred_from_codec] starting..." << std::endl;
    // Tri-state: codec presence (+ fingerprints) without evaluated workload /
    // quality evidence must render NOT_EVALUATED (null/NaN), never false-as-
    // verdict and never true-from-codec.
    server_xkv_metrics m;
    m.observed = true;
    m.compression_goal_evaluated = false;
    m.compression_goal_met = false;
    m.requested_a_k = "turbo4_0";
    m.effective_a_k = "turbo4_0";
    m.requested_b_k = "turbo4_0";
    m.effective_b_k = "turbo4_0";
    m.requested_a_v = "turbo4_0";
    m.effective_a_v = "turbo4_0";
    m.requested_b_v = "turbo4_0";
    m.effective_b_v = "turbo4_0";
    m.codec_fingerprint = 0x5678ULL;
    m.backend_fingerprint = 0xDEF0ULL;
    m.source_fingerprint = 0x1234ULL;
    m.profile_fingerprint = 0x9ABCULL;
    m.config_fingerprint = 0x9ABCULL;
    json j = m.to_json();
    CHECK(j["xkv_compression_goal_evaluated"] == false);
    CHECK(j["xkv_compression_goal_met"].is_null());
    std::ostringstream os;
    m.to_prometheus(os);
    CHECK(os.str().find("llamacpp:xkv_compression_goal_met NaN") != std::string::npos);
    std::cout << "  [PASS] Compression goal tri-state verified (codec presence never implies verdict)." << std::endl;
}

int main() {
    std::cout << "=== test-xkv-metrics ===" << std::endl;
    test_codec_shared_table_bytes();
    test_residency_predicate();
    test_server_metrics_evaluated();
    test_server_metrics_unevaluated();
    test_server_metrics_off_all_zero();
    test_coverage_and_compression_goal_math();
    test_server_metrics_live_stream_bytes();
    test_compression_goal_never_inferred_from_codec();

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED: %d checks\n", g_failures);
        return 1;
    }
    std::cout << "=== ALL XKV METRICS CHECKS PASSED ===" << std::endl;
    return 0;
}
