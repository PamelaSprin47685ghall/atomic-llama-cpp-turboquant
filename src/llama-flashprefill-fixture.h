// llama-flashprefill-fixture.h — model-free fixture entry for state tests.
//
// Thin adapter over the canonical owner-shared planner
// (llama_flashprefill_build_rerot_plan, declared in
// llama-flashprefill-layout.h): identical inputs, bool outcome (true = an
// eligible, self-validated layout in out). The tri-state collapses
// losslessly here: the RERoT core never reports INELIGIBLE (the indexed path
// has no stock-path content class), so false always means hard error with
// *error set.
//
// Fixture inputs are exactly (cells, views, ubatch, params, out, error):
//   cells  - synthetic llama_kv_cells (pos_set/seq_add/rerot_set); borrowed.
//   views  - reader states indexed by seq id (views[seq] must be active for
//            every seq appearing in ubatch rows); borrowed.
//   ubatch - borrowed rows; only n_tokens, n_seq_id, seq_id, pos are read
//            (n_pos/ext are ignored by the indexed path, exactly like the
//            old builder; token/embd/data may stay null).
//   params - planning params; fixtures use want_exact_rows=true to cross-check
//            exact_rows against the use expansion.
//   out    - cleared first, then filled; eligible + validated on success.
//   error  - human reason on false (may be null).
// Owner path keeps the canonical name and full tri-state; tests
// auto-activate this header via __has_include and report PENDING without it.

#pragma once

#include "llama-flashprefill-layout.h"

#include <algorithm>
#include <string>
#include <vector>

inline bool llama_flashprefill_fixture_build_rerot(
    const llama_kv_cells & cells,
    const std::vector<llama_rerot_reader_state> & views,
    const llama_ubatch & ubatch,
    const llama_flashprefill_layout_params & params,
    llama_flashprefill_layout & out,
    std::string * error = nullptr) {
    out.clear();
    const llama_flashprefill_build_status st =
        llama_flashprefill_build_rerot_plan(cells, views, ubatch, params, out, error);
    if (st != llama_flashprefill_build_status::OK) {
        return false;
    }
    out.block_k = params.block_k;
    out.is_rerot = 1;
    out.causal = 1;
    out.swa_window = 0;
    out.swa_type = 0;
    out.cells_epoch = 0; // no owner epoch fixture-side; freshness is N/A here
    out.cell_stamps.clear();
    out.cell_stamps.push_back({ cells.get_generation_stamp(), cells.get_generation() });
    out.n_kv_at_build = cells.size();
    out.eligible = true;
    std::string vmsg;
    if (!out.validate(cells.size(), &vmsg)) {
        out.eligible = false;
        out.error = "fixture plan failed validation: " + vmsg;
        if (error) {
            *error = out.error;
        }
        return false;
    }
    return true;
}

// Ordinary-path counterpart: same forwarding discipline over the canonical
// owner-shared llama_flashprefill_build_ordinary_plan (no mock, no duplicate
// planner). Fixed causal 1-D text policy (n_swa=0, SWA NONE, causal=true,
// require_text_cells=true): fixture ubatches are synthetic 1-D/text-pattern
// rows and synthetic cells carry default (zero) extents, which satisfy text
// compatibility. want_exact_rows is honored from params for oracle
// cross-checks. Multi-stream aware: stamps cover every stream, scan width is
// the widest stream.
inline bool llama_flashprefill_fixture_build_ordinary(
    const llama_kv_cells_vec & v_cells,
    const std::vector<uint32_t> & seq_to_stream,
    const llama_ubatch & ubatch,
    const llama_flashprefill_layout_params & params,
    llama_flashprefill_layout & out,
    std::string * error = nullptr) {
    out.clear();
    const llama_flashprefill_build_status st = llama_flashprefill_build_ordinary_plan(
        v_cells, seq_to_stream, 0, LLAMA_SWA_TYPE_NONE, true, true,
        ubatch, params, out, error);
    if (st != llama_flashprefill_build_status::OK) {
        return false;
    }
    out.block_k = params.block_k;
    out.is_rerot = 0;
    out.causal = 1;
    out.swa_window = 0;
    out.swa_type = 0;
    out.cells_epoch = 0; // no owner epoch fixture-side; freshness is N/A here
    out.cell_stamps.clear();
    uint32_t width = 0;
    for (const auto & cells : v_cells) {
        out.cell_stamps.push_back({ cells.get_generation_stamp(), cells.get_generation() });
        width = std::max(width, cells.size());
    }
    out.n_kv_at_build = width;
    out.eligible = true;
    std::string vmsg;
    if (!out.validate(width, &vmsg)) {
        out.eligible = false;
        out.error = "fixture plan failed validation: " + vmsg;
        if (error) {
            *error = out.error;
        }
        return false;
    }
    return true;
}
