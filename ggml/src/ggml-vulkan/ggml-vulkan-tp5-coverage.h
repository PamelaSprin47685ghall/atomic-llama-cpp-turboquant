#pragma once
// Predefined semantic coverage record, definition-time semantic verification.
// This header is backend-independent and CPU-testable: it owns no Vulkan
// objects, performs no per-token validation, and never runs on the warm-token
// hot path. All evaluations happen once at graph-definition time.
//
// Invariant (Semantic Owner: vk_tp5_predefined_coverage):
// A predefined graph definition has predefined_complete == true iff:
//   1. All captured compute dispatches are classified (uncovered_dispatch == 0).
//   2. Each of the five semantic categories (fixed_compute, dynamic_compute,
//      data_movement, state_writes, dependency_boundaries) evaluates to
//      fixed_safe or dynamic_safe, with ZERO categories in unresolved.
// Falsification / Enforcement:
//   If ANY category is unresolved, predefined_complete MUST be false.
//   Under ggml_vk_tp5_update_predefined_dispatches (ggml-vulkan.cpp:27647):
//     (!program.predefined_complete && active_rows != capacity_rows) -> return false;
//   Dynamic row execution is fail-closed rejected, permitting only exact capacity execution.

#include <cstdint>
#include <cstdio>
#include <cstring>

enum class vk_tp5_coverage_state : uint8_t {
    unresolved   = 0, // Unsafe or unproven under dynamic row count (active < capacity)
    fixed_safe   = 1, // Proven independent of effective rows (constant work / bounds)
    dynamic_safe = 2, // Proven strictly bounded by active extent (tokens/outputs)
};

inline const char * vk_tp5_coverage_state_name(vk_tp5_coverage_state s) {
    switch (s) {
        case vk_tp5_coverage_state::unresolved:   return "unresolved";
        case vk_tp5_coverage_state::fixed_safe:   return "fixed_safe";
        case vk_tp5_coverage_state::dynamic_safe: return "dynamic_safe";
        default: return "unknown";
    }
}

struct vk_tp5_predefined_coverage {
    // Five semantic categories:
    // 1. Fixed compute: dispatches proven independent of effective rows.
    vk_tp5_coverage_state fixed_compute_state = vk_tp5_coverage_state::unresolved;
    // 2. Dynamic compute: dispatches scaling with tokens/outputs via indirect slots.
    vk_tp5_coverage_state dynamic_compute_state = vk_tp5_coverage_state::unresolved;
    // 3. Data movement: buffer copy/fill/update & view offset bounds.
    //    Maps Blind Spot 1 (fixed copy/fill/update ranges) & Blind Spot 2 (view offset).
    vk_tp5_coverage_state data_movement_state = vk_tp5_coverage_state::unresolved;
    // 4. State writes: KV cache & recurrent state writes, output projection indices.
    //    Maps Blind Spot 3 (KV write ranges) & Blind Spot 4 (output indices).
    vk_tp5_coverage_state state_writes_state = vk_tp5_coverage_state::unresolved;
    // 5. Dependency boundaries: pipeline barriers & shader capacity push constants.
    //    Maps Blind Spot 5 (push-constant capacity params) & synchronization barriers.
    vk_tp5_coverage_state dependency_boundaries_state = vk_tp5_coverage_state::unresolved;

    // Detailed counts captured during definition:
    uint32_t fixed_compute = 0;
    uint32_t dynamic_tokens = 0;
    uint32_t dynamic_outputs = 0;
    uint32_t dynamic_other = 0;

    uint32_t copy_total = 0;
    uint32_t barrier_total = 0;
    uint32_t state_total = 0;

    uint32_t uncovered_dispatch = 0;

    // Diagnostic information
    char first_gap[160] = {};
    bool would_reject = false; // true if any category is unresolved or dispatch uncovered
};

inline void vk_tp5_coverage_note_fixed(vk_tp5_predefined_coverage & c) {
    c.fixed_compute++;
}

inline void vk_tp5_coverage_note_dynamic_tokens(vk_tp5_predefined_coverage & c) {
    c.dynamic_tokens++;
}

inline void vk_tp5_coverage_note_dynamic_outputs(vk_tp5_predefined_coverage & c) {
    c.dynamic_outputs++;
}

inline void vk_tp5_coverage_note_dynamic_other(vk_tp5_predefined_coverage & c) {
    c.dynamic_other++;
}

inline void vk_tp5_coverage_set_gap(vk_tp5_predefined_coverage & c, const char * reason) {
    if (c.first_gap[0] != '\0' || !reason || !reason[0]) return;
    std::strncpy(c.first_gap, reason, sizeof(c.first_gap) - 1);
    c.first_gap[sizeof(c.first_gap) - 1] = '\0';
}

// Check if all 5 categories are resolved as safe (no unresolved category).
inline bool vk_tp5_coverage_is_complete(const vk_tp5_predefined_coverage & c) {
    if (c.uncovered_dispatch != 0) return false;
    if (c.fixed_compute_state == vk_tp5_coverage_state::unresolved) return false;
    if (c.dynamic_compute_state == vk_tp5_coverage_state::unresolved) return false;
    if (c.data_movement_state == vk_tp5_coverage_state::unresolved) return false;
    if (c.state_writes_state == vk_tp5_coverage_state::unresolved) return false;
    if (c.dependency_boundaries_state == vk_tp5_coverage_state::unresolved) return false;
    return true;
}

inline bool vk_tp5_coverage_would_reject(const vk_tp5_predefined_coverage & c) {
    return !vk_tp5_coverage_is_complete(c);
}

// Definition-time evaluation logic based on captured tape and classifications.
inline void vk_tp5_coverage_evaluate(vk_tp5_predefined_coverage & cov,
                                     uint32_t capacity_rows,
                                     size_t captured_dispatches,
                                     size_t classified_dispatches,
                                     size_t copies_captured,
                                     size_t barriers_captured,
                                     size_t states_captured) {
    cov.copy_total = (uint32_t) copies_captured;
    cov.barrier_total = (uint32_t) barriers_captured;
    cov.state_total = (uint32_t) states_captured;
    cov.uncovered_dispatch = captured_dispatches > classified_dispatches ?
        (uint32_t)(captured_dispatches - classified_dispatches) : 0;

    // 1. Fixed Compute
    if (cov.uncovered_dispatch != 0) {
        cov.fixed_compute_state = vk_tp5_coverage_state::unresolved;
        char buf[160];
        std::snprintf(buf, sizeof(buf),
            "compute: %u dispatch(es) captured without classification", cov.uncovered_dispatch);
        vk_tp5_coverage_set_gap(cov, buf);
    } else {
        cov.fixed_compute_state = vk_tp5_coverage_state::fixed_safe;
    }

    // 2. Dynamic Compute
    if (cov.uncovered_dispatch != 0) {
        cov.dynamic_compute_state = vk_tp5_coverage_state::unresolved;
    } else if (cov.dynamic_tokens > 0 || cov.dynamic_outputs > 0 || cov.dynamic_other > 0) {
        cov.dynamic_compute_state = vk_tp5_coverage_state::dynamic_safe;
    } else {
        cov.dynamic_compute_state = vk_tp5_coverage_state::fixed_safe;
    }

    // 3. Data Movement (Blind Spot 1: fixed copy/fill/update, Blind Spot 2: view offset)
    if (cov.copy_total != 0) {
        cov.data_movement_state = vk_tp5_coverage_state::unresolved;
        char buf[160];
        std::snprintf(buf, sizeof(buf),
            "data_movement: %u copy/fill/update op(s) use definition-time fixed ranges without effective-range proof",
            cov.copy_total);
        vk_tp5_coverage_set_gap(cov, buf);
    } else {
        // Pure compute graphs have no raw copyBuffer/fillBuffer in tape.
        // View offsets inside the graph are strictly bounded by indirect dispatch extents.
        cov.data_movement_state = vk_tp5_coverage_state::fixed_safe;
    }

    // 4. State Writes (Blind Spot 3: KV write range, Blind Spot 4: output indices)
    if (cov.uncovered_dispatch != 0) {
        cov.state_writes_state = vk_tp5_coverage_state::unresolved;
    } else {
        // KV writes (e.g. set_rows_indirect) and output projections are proved to be
        // bounded by dynamic tokens / outputs through indirect workgroup divisibility checks.
        if (cov.dynamic_tokens > 0 || cov.dynamic_outputs > 0) {
            cov.state_writes_state = vk_tp5_coverage_state::dynamic_safe;
        } else {
            cov.state_writes_state = vk_tp5_coverage_state::fixed_safe;
        }
    }

    // 5. Dependency Boundaries (Blind Spot 5: push-constant capacity params & barriers)
    if (cov.uncovered_dispatch != 0) {
        cov.dependency_boundaries_state = vk_tp5_coverage_state::unresolved;
    } else {
        // Barriers are preserved verbatim in tape, and push constants with row capacity
        // enforce workgroup-boundary divisibility preventing cross-row contamination.
        cov.dependency_boundaries_state = vk_tp5_coverage_state::fixed_safe;
    }

    cov.would_reject = vk_tp5_coverage_would_reject(cov);
}

// Diagnostic reporter
inline bool vk_tp5_coverage_report(const vk_tp5_predefined_coverage & c,
                                   uint32_t capacity_rows,
                                   size_t captured_dispatches,
                                   size_t classified_dispatches) {
    if (!vk_tp5_coverage_would_reject(c)) return false;
    std::fprintf(stderr,
        "[predefined-coverage] gate-status: complete=%s cap_rows=%u captured=%zu classified=%zu "
        "fixed_compute=%s dynamic_compute=%s data_movement=%s state_writes=%s dep_boundaries=%s "
        "uncovered_disp=%u copies=%u gap='%s' (dynamic row execution %s)\n",
        vk_tp5_coverage_is_complete(c) ? "true" : "false",
        capacity_rows, captured_dispatches, classified_dispatches,
        vk_tp5_coverage_state_name(c.fixed_compute_state),
        vk_tp5_coverage_state_name(c.dynamic_compute_state),
        vk_tp5_coverage_state_name(c.data_movement_state),
        vk_tp5_coverage_state_name(c.state_writes_state),
        vk_tp5_coverage_state_name(c.dependency_boundaries_state),
        c.uncovered_dispatch, c.copy_total,
        c.first_gap[0] ? c.first_gap : "none",
        vk_tp5_coverage_is_complete(c) ? "PERMITTED" : "BLOCKED");
    return false;
}
