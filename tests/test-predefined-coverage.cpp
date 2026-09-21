#include "../ggml/src/ggml-vulkan/ggml-vulkan-tp5-coverage.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) throw std::runtime_error("failed: " #x); } while (0)

// Pure CPU coverage-record tests. No Vulkan loader/device, shader compiler,
// model load, or GPU execution is involved. These tests exercise the
// definition-time semantic coverage record, tri-state evaluation, completeness,
// and the dynamic execution gate contract (ggml-vulkan.cpp:27647).

static bool simulate_dynamic_execution_gate(bool predefined_complete,
                                            uint32_t active_rows,
                                            uint32_t capacity_rows) {
    // Exact logic from ggml-vulkan.cpp:27647:
    // (!program.predefined_complete && active_rows != capacity_rows) -> reject
    if (!predefined_complete && active_rows != capacity_rows) {
        return false; // dynamic execution rejected (fail-closed)
    }
    return true; // permitted
}

static void valid_definition_is_complete_and_permits_dynamic_execution() {
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

    // Dynamic execution qualification: permitted for active_rows < capacity_rows
    CHECK(simulate_dynamic_execution_gate(vk_tp5_coverage_is_complete(cov), 1, 32));
    CHECK(simulate_dynamic_execution_gate(vk_tp5_coverage_is_complete(cov), 4, 32));
    CHECK(simulate_dynamic_execution_gate(vk_tp5_coverage_is_complete(cov), 32, 32));

    // Diagnostic reporting returns false (no unhandled gap to warn)
    CHECK(!vk_tp5_coverage_report(cov, 32, captured, classified));
}

static void unresolved_data_movement_blocks_dynamic_execution() {
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
    CHECK(std::strstr(cov.first_gap, "data_movement") != nullptr);
    CHECK(cov.would_reject);
    CHECK(!vk_tp5_coverage_is_complete(cov));

    // Enforcement: Dynamic row execution is REJECTED by gate
    CHECK(!simulate_dynamic_execution_gate(vk_tp5_coverage_is_complete(cov), 1, 32));
    CHECK(!simulate_dynamic_execution_gate(vk_tp5_coverage_is_complete(cov), 4, 32));
    // Exact capacity execution is still permitted
    CHECK(simulate_dynamic_execution_gate(vk_tp5_coverage_is_complete(cov), 32, 32));
}

static void unclassified_dispatch_sets_unresolved_and_blocks_dynamic() {
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
    CHECK(std::strstr(cov.first_gap, "compute") != nullptr);
    CHECK(!vk_tp5_coverage_is_complete(cov));

    // Dynamic execution qualification: REJECTED
    CHECK(!simulate_dynamic_execution_gate(vk_tp5_coverage_is_complete(cov), 1, 32));
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
    CHECK(simulate_dynamic_execution_gate(vk_tp5_coverage_is_complete(cov), 1, 32));
}

static void tri_state_and_first_gap_contract() {
    // Verify string names
    CHECK(std::strcmp(vk_tp5_coverage_state_name(vk_tp5_coverage_state::unresolved), "unresolved") == 0);
    CHECK(std::strcmp(vk_tp5_coverage_state_name(vk_tp5_coverage_state::fixed_safe), "fixed_safe") == 0);
    CHECK(std::strcmp(vk_tp5_coverage_state_name(vk_tp5_coverage_state::dynamic_safe), "dynamic_safe") == 0);

    // Verify first_gap retains only the first reason
    vk_tp5_predefined_coverage cov{};
    vk_tp5_coverage_set_gap(cov, "first reason");
    vk_tp5_coverage_set_gap(cov, "second reason");
    CHECK(std::strcmp(cov.first_gap, "first reason") == 0);

    // Any individual unresolved state causes is_complete() == false
    cov.state_writes_state = vk_tp5_coverage_state::unresolved;
    CHECK(!vk_tp5_coverage_is_complete(cov));
}

int main() {
    try {
        valid_definition_is_complete_and_permits_dynamic_execution();
        unresolved_data_movement_blocks_dynamic_execution();
        unclassified_dispatch_sets_unresolved_and_blocks_dynamic();
        all_fixed_compute_definition_is_complete();
        tri_state_and_first_gap_contract();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    std::puts("predefined semantic coverage checks passed (tri-state & gate qualification verified)");
    return 0;
}
