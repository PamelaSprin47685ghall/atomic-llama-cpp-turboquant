#include "../ggml/src/ggml-vulkan/ggml-vulkan-tp5-coverage.h"

#include <cstdio>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) throw std::runtime_error("failed: " #x); } while (0)

// Pure CPU coverage-record tests. No Vulkan loader/device, shader compiler,
// model load, or GPU execution is involved. These tests exercise the
// definition-time record + fail-open diagnostic only.

static void valid_definition_has_no_gap() {
    // 2 dynamic computes (1 tokens-driven, 1 outputs-driven), 0 copies:
    // captured == classified, no data-movement blind spot.
    vk_tp5_predefined_coverage cov{};
    vk_tp5_coverage_note_dynamic_tokens(cov);
    vk_tp5_coverage_note_dynamic_outputs(cov);
    cov.copy_total = 0;
    cov.barrier_total = 1; // dependency boundary preserved verbatim
    cov.state_total = 4;   // bind/push companions are expected, not gaps
    cov.uncovered_dispatch = 0;
    cov.would_reject = vk_tp5_coverage_would_reject(cov);
    CHECK(!cov.would_reject);
    CHECK(cov.first_gap[0] == '\0');
    // Report returns false (fail-open, no diagnostic needed).
    CHECK(!vk_tp5_coverage_report(cov, 32, 2, 2));
}

static void fixed_copy_range_is_flagged_fail_open() {
    // Negative example: dispatches are fully classified (2 == 2), but the
    // definition also captured 1 copy/fill/update with a definition-time
    // fixed range (src/dst offset + size baked at record time). There is no
    // effective-range proof for row-count changes, so the record must flag
    // it with an explicit gap and would_reject == true, while execution
    // itself continues (fail-open: report() never rejects).
    vk_tp5_predefined_coverage cov{};
    vk_tp5_coverage_note_dynamic_tokens(cov);
    vk_tp5_coverage_note_fixed(cov);
    cov.copy_total = 1;
    cov.barrier_total = 1;
    cov.state_total = 4;
    cov.uncovered_dispatch = 0;
    vk_tp5_coverage_set_gap(cov,
        "1 copy/fill/update op(s) use definition-time fixed ranges (no effective-range proof; view/KV/output-index/push-constant capacity params likewise unproven)");
    cov.would_reject = vk_tp5_coverage_would_reject(cov);
    CHECK(cov.would_reject); // future fail-closed WOULD reject here
    CHECK(cov.first_gap[0] != '\0');
    // Fail-open proof: the diagnostic prints but does not reject.
    CHECK(!vk_tp5_coverage_report(cov, 32, 2, 2));
    CHECK(cov.would_reject); // still only advisory in phase 1
}

static void unclassified_dispatch_is_flagged() {
    // 3 captured, 2 classified -> 1 uncovered dynamic/fixed dispatch.
    vk_tp5_predefined_coverage cov{};
    vk_tp5_coverage_note_dynamic_tokens(cov);
    vk_tp5_coverage_note_fixed(cov);
    cov.copy_total = 0;
    cov.uncovered_dispatch = 1;
    vk_tp5_coverage_set_gap(cov, "1 dispatch(es) captured without static/dynamic classification");
    CHECK(vk_tp5_coverage_would_reject(cov));
    CHECK(!vk_tp5_coverage_report(cov, 32, 3, 2)); // fail-open
}

int main() {
    try {
        valid_definition_has_no_gap();
        fixed_copy_range_is_flagged_fail_open();
        unclassified_dispatch_is_flagged();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    std::puts("predefined coverage record checks passed (fail-open diagnostics verified)");
    return 0;
}
