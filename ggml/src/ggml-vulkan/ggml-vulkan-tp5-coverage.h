#pragma once
// Predefined semantic coverage record, phase 1: record + fail-open diagnostics.
// This header is intentionally backend-independent and CPU-testable: it owns
// no Vulkan objects, performs no per-token validation, and never runs on the
// warm-token hot path. All notes happen once at graph-definition time.
//
// Background: predefined_complete today only compares dispatch counts
// (classified == captured). Known blind spots with no semantic proof when the
// active row count changes: fixed vkCmdCopyBuffer ranges, view offsets, KV
// write ranges, output indices, and capacity parameters baked into push
// constants. This record makes those classes visible without changing
// execution semantics.
//
// Phase 1 policy is fail-open: vk_tp5_coverage_report() only prints a
// diagnostic. vk_tp5_coverage_would_reject() is the clean predicate reserved
// for a future fail-closed gate; it is NOT wired to reject/return-false yet.
#include <cstdint>
#include <cstdio>
#include <cstring>

struct vk_tp5_predefined_coverage {
    // Fixed compute: classified via ggml_vk_predefined_classify_static, i.e.
    // the author asserts this dispatch does not depend on effective rows.
    uint32_t fixed_compute = 0;
    // Dynamic compute, split by which frame extent drives the indirect args.
    // TOKENS=1, OUTPUTS=2 per ggml_predefined_extent; OTHER covers CONTEXT /
    // SEQUENCES / PAYLOAD or mixed axes. Counts come from
    // ggml_vk_predefined_indirect_slot success paths only.
    uint32_t dynamic_tokens = 0;
    uint32_t dynamic_outputs = 0;
    uint32_t dynamic_other = 0;
    // Observed tape populations at finalize time (definition-time counts).
    // Copies include vkCmdCopyBuffer/FillBuffer/UpdateBuffer: all currently
    // carry definition-time fixed offsets/sizes, hence no effective-range
    // proof yet. Barriers are dependency boundaries preserved verbatim.
    // States are bind/push companions (pipeline/descriptor/push-constant).
    uint32_t copy_total = 0;
    uint32_t barrier_total = 0;
    uint32_t state_total = 0;
    // Derived at finalize: captured dispatches minus classified dispatches.
    uint32_t uncovered_dispatch = 0;
    // First uncovered class, human-readable. Only the first gap is kept so
    // the record stays small; see the stderr diagnostic for the full line.
    char first_gap[160] = {};
    // Cached result of vk_tp5_coverage_would_reject() at finalize time.
    // Informational only in phase 1; execution ignores it (fail-open).
    bool would_reject = false;
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

// Clean predicate reserved for a future fail-closed gate. Phase 1 callers
// must only use it for diagnostics/tests, never to reject a definition.
// Rejects (in the future sense) when any dispatch lacks classification or
// any data-movement op lacks an effective-range proof. Dependency barriers
// and bind/push companions alone do not trip it.
inline bool vk_tp5_coverage_would_reject(const vk_tp5_predefined_coverage & c) {
    if (c.uncovered_dispatch != 0) return true;
    if (c.copy_total != 0) return true;
    if (c.first_gap[0] != '\0') return true;
    return false;
}

// Fail-open diagnostic: prints one line when something lacks coverage and
// always returns false (never a rejection). Call once at definition finalize,
// never on the warm-token path. Hot-path updaters
// (ggml_vk_tp5_update_predefined_dispatches) must not call this.
inline bool vk_tp5_coverage_report(const vk_tp5_predefined_coverage & c,
                                   uint32_t capacity_rows,
                                   size_t captured_dispatches,
                                   size_t classified_dispatches) {
    if (!vk_tp5_coverage_would_reject(c)) return false;
    std::fprintf(stderr,
        "[predefined-coverage] fail-open: cap_rows=%u captured=%zu classified=%zu "
        "fixed=%u dyn_tokens=%u dyn_outputs=%u dyn_other=%u copies=%u barriers=%u states=%u "
        "uncovered_disp=%u gap='%s' (execution continues; future fail-closed would reject)\n",
        capacity_rows, captured_dispatches, classified_dispatches,
        c.fixed_compute, c.dynamic_tokens, c.dynamic_outputs, c.dynamic_other,
        c.copy_total, c.barrier_total, c.state_total,
        c.uncovered_dispatch, c.first_gap[0] ? c.first_gap : "unspecified");
    return false;
}
