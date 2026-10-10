#include "../ggml/src/ggml-vulkan/ggml-vulkan-tp5-coverage.h"

#include <cstdio>
#include <stdexcept>

#define CHECK(x)                                     \
    do {                                             \
        if (!(x))                                    \
            throw std::runtime_error("failed: " #x); \
    } while (0)

// Pure CPU coverage-record tests. No Vulkan loader/device, shader compiler,
// model load, or GPU execution is involved. These tests exercise the
// production definition-time coverage record and completeness decisions.

static void classified_definition_is_complete() {
    // 2 dynamic computes (1 tokens-driven, 1 outputs-driven), 0 copies:
    // captured == classified (2 == 2), no unproven copy/view/state writes.
    vk_tp5_predefined_coverage cov{};
    vk_tp5_coverage_note_dynamic_tokens(cov);
    vk_tp5_coverage_note_dynamic_outputs(cov);

    const size_t captured = 2;
    const size_t classified = 2;
    const size_t copies = 0;
    const size_t barriers = 1;
    const size_t states = 4;

    vk_tp5_coverage_evaluate(cov, 32, captured, classified, copies, barriers, states);

    // All five categories must resolve to safe (zero unresolved):
    CHECK(cov.fixed_compute_state == vk_tp5_coverage_state::fixed_safe);
    CHECK(cov.dynamic_compute_state == vk_tp5_coverage_state::dynamic_safe);
    CHECK(cov.data_movement_state == vk_tp5_coverage_state::fixed_safe);
    CHECK(cov.state_writes_state == vk_tp5_coverage_state::dynamic_safe);
    CHECK(cov.dependency_boundaries_state == vk_tp5_coverage_state::fixed_safe);

    CHECK(cov.uncovered_dispatch == 0);
    CHECK(cov.first_gap[0] == '\0');
    CHECK(!cov.would_reject);
    CHECK(vk_tp5_coverage_is_complete(cov));

    // Diagnostic reporting returns false (no unhandled gap to warn)
    CHECK(!vk_tp5_coverage_report(cov, 32, captured, classified));
}

static void unresolved_data_movement_blocks_completeness() {
    // Dispatches are classified (2 == 2), but definition captured 1 raw copy/fill/update op.
    // Fixed range copy has no effective-range proof -> data_movement_state must be unresolved.
    vk_tp5_predefined_coverage cov{};
    vk_tp5_coverage_note_dynamic_tokens(cov);
    vk_tp5_coverage_note_fixed(cov);

    const size_t captured = 2;
    const size_t classified = 2;
    const size_t copies = 1;
    const size_t barriers = 1;
    const size_t states = 4;

    vk_tp5_coverage_evaluate(cov, 32, captured, classified, copies, barriers, states);

    CHECK(cov.data_movement_state == vk_tp5_coverage_state::unresolved);
    CHECK(cov.would_reject);
    CHECK(!vk_tp5_coverage_is_complete(cov));
}

static void unclassified_dispatch_blocks_completeness() {
    // 3 captured, 2 classified -> 1 uncovered dispatch.
    // Must set compute and state writes to unresolved and block dynamic execution.
    vk_tp5_predefined_coverage cov{};
    vk_tp5_coverage_note_dynamic_tokens(cov);
    vk_tp5_coverage_note_fixed(cov);

    const size_t captured = 3;
    const size_t classified = 2;
    const size_t copies = 0;
    const size_t barriers = 1;
    const size_t states = 2;

    vk_tp5_coverage_evaluate(cov, 32, captured, classified, copies, barriers, states);

    CHECK(cov.uncovered_dispatch == 1);
    CHECK(cov.fixed_compute_state == vk_tp5_coverage_state::unresolved);
    CHECK(cov.dynamic_compute_state == vk_tp5_coverage_state::unresolved);
    CHECK(cov.state_writes_state == vk_tp5_coverage_state::unresolved);
    CHECK(cov.dependency_boundaries_state == vk_tp5_coverage_state::unresolved);
    CHECK(!vk_tp5_coverage_is_complete(cov));
}

static void all_fixed_compute_definition_is_complete() {
    // Graph with only statically classified dispatches (independent of active rows).
    vk_tp5_predefined_coverage cov{};
    vk_tp5_coverage_note_fixed(cov);
    vk_tp5_coverage_note_fixed(cov);

    const size_t captured = 2;
    const size_t classified = 2;
    const size_t copies = 0;
    const size_t barriers = 0;
    const size_t states = 2;

    vk_tp5_coverage_evaluate(cov, 32, captured, classified, copies, barriers, states);

    CHECK(cov.fixed_compute_state == vk_tp5_coverage_state::fixed_safe);
    CHECK(cov.dynamic_compute_state == vk_tp5_coverage_state::fixed_safe);
    CHECK(cov.data_movement_state == vk_tp5_coverage_state::fixed_safe);
    CHECK(cov.state_writes_state == vk_tp5_coverage_state::fixed_safe);
    CHECK(cov.dependency_boundaries_state == vk_tp5_coverage_state::fixed_safe);
    CHECK(vk_tp5_coverage_is_complete(cov));
}

static void unresolved_state_blocks_completeness() {
    vk_tp5_predefined_coverage cov{};
    vk_tp5_coverage_note_fixed(cov);
    vk_tp5_coverage_evaluate(cov, 32, 1, 1, 0, 0, 1);
    CHECK(vk_tp5_coverage_is_complete(cov));
    const auto            complete                                         = cov;
    vk_tp5_coverage_state vk_tp5_predefined_coverage::* const categories[] = {
        &vk_tp5_predefined_coverage::fixed_compute_state,         &vk_tp5_predefined_coverage::dynamic_compute_state,
        &vk_tp5_predefined_coverage::data_movement_state,         &vk_tp5_predefined_coverage::state_writes_state,
        &vk_tp5_predefined_coverage::dependency_boundaries_state,
    };
    for (auto category : categories) {
        cov           = complete;
        cov.*category = vk_tp5_coverage_state::unresolved;
        CHECK(!vk_tp5_coverage_is_complete(cov));
    }
}

int main() {
    try {
        classified_definition_is_complete();
        unresolved_data_movement_blocks_completeness();
        unclassified_dispatch_blocks_completeness();
        all_fixed_compute_definition_is_complete();
        unresolved_state_blocks_completeness();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    std::puts("predefined semantic coverage record checks passed");
    return 0;
}
