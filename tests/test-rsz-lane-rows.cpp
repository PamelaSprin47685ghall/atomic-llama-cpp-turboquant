#include "../src/llama-nout-state.h"

#include <cstdint>
#include <cstdio>

//
// test-rsz-lane-rows
//
// The two decision predicates of the rs_z runtime lane, as pure functions.
// No model, no context, no backend: this test runs in every CPU build, so the
// arming rule and the decode-time row check are pinned by a test and not only
// by prose.
//
// Contract (llama-nout-state.h):
//   llama_rsz_rt_armable(gate_ok, has_recr, n_rs_seq, rs_z)
//       true iff the umbrella path can carry the lane, the graph has a
//       recurrent sub-cache, the cache keeps no rollback snapshots
//       (n_rs_seq == 0), and the live selection names a cell (rs_z >= 0).
//       n_rs_seq > 0 is the _n_rs_seq refusal: the state rows are widened to
//       mem_size*(1+n_rs_seq) and a row index is no longer a cell index
//       (build_rs(), llama-graph.cpp).
//   llama_rsz_rt_check_row(rs_z, zero_view_rows)
//       ok      iff 0 <= rs_z < zero_view_rows
//       cell    iff rs_z < 0            (no free zero-state cell)
//       envelope iff rs_z >= zero_view_rows (outside the cleared row range)
//
// Every failing case prints its own inputs, so a broken predicate names the
// argument that broke it.

namespace {

int g_fail = 0;
int g_checks_run = 0;

void check_arm(bool gate_ok, bool has_recr, uint32_t n_rs_seq, int32_t rs_z, bool expected) {
    ++g_checks_run;
    const bool got = llama_rsz_rt_armable(gate_ok, has_recr, n_rs_seq, rs_z);
    if (got != expected) {
        std::fprintf(stderr,
                     "FAIL llama_rsz_rt_armable(gate_ok=%d has_recr=%d n_rs_seq=%u rs_z=%d) "
                     "= %d, expected %d\n",
                     gate_ok ? 1 : 0, has_recr ? 1 : 0, n_rs_seq, rs_z,
                     got ? 1 : 0, expected ? 1 : 0);
        ++g_fail;
    }
}

void check_row(int32_t rs_z, uint32_t rows, llama_rsz_rt_row_check expected) {
    ++g_checks_run;
    const llama_rsz_rt_row_check got = llama_rsz_rt_check_row(rs_z, rows);
    if (got != expected) {
        std::fprintf(stderr,
                     "FAIL llama_rsz_rt_check_row(rs_z=%d rows=%u) = %d, expected %d\n",
                     rs_z, rows, (int) got, (int) expected);
        ++g_fail;
    }
}

} // namespace

int main() {
    // ---- arming: the n_rs_seq refusal is a property of the predicate ------
    //
    // The reference point is the one configuration that arms: the umbrella
    // path on, a recurrent sub-cache, no rollback snapshots, a named cell.
    check_arm(true, true, 0, 0, true);
    check_arm(true, true, 0, 7, true);

    // every other n_rs_seq value refuses, at every legal rs_z: the row set is
    // no longer the cell set, so the lane must not arm
    for (uint32_t n_rs_seq = 1; n_rs_seq <= 8; ++n_rs_seq) {
        check_arm(true, true, n_rs_seq,  0, false);
        check_arm(true, true, n_rs_seq,  3, false);
        check_arm(true, true, n_rs_seq, 31, false);
    }

    // rs_z < 0 is the _cell refusal: no free zero-state cell to name
    check_arm(true, true, 0, -1, false);
    check_arm(true, true, 0, -2, false);

    // no recurrent sub-cache: nothing to select (_arch)
    check_arm(true, false, 0, 0, false);

    // the umbrella path cannot carry the lane (_gate)
    check_arm(false, true, 0, 0, false);

    // the refusals compose: a refused configuration stays refused whatever
    // the other arguments say
    check_arm(false, false, 4, -1, false);

    // ---- row check: the decode-time envelope ------------------------------
    //
    // 8 state cells is the fixture shape the live tests use.
    const uint32_t rows = 8;
    for (int32_t rs_z = 0; rs_z < (int32_t) rows; ++rs_z) {
        check_row(rs_z, rows, llama_rsz_rt_row_check::ok);
    }
    // the first row outside the view is the _envelope refusal, and it stays
    // one for every larger index
    check_row((int32_t) rows,     rows, llama_rsz_rt_row_check::envelope);
    check_row((int32_t) rows + 1, rows, llama_rsz_rt_row_check::envelope);
    check_row(1000,               rows, llama_rsz_rt_row_check::envelope);
    // rs_z < 0 is _cell, never _envelope: the two refusals are distinct
    check_row(-1, rows, llama_rsz_rt_row_check::cell);
    check_row(-2, rows, llama_rsz_rt_row_check::cell);
    // a zero-row view admits no row at all, so every rs_z >= 0 is _envelope
    check_row(0, 0, llama_rsz_rt_row_check::envelope);
    check_row(-1, 0, llama_rsz_rt_row_check::cell);

    if (g_checks_run == 0) {
        std::fprintf(stderr, "RSZ_LANE_ROWS_TEST UNPROVEN reason=no_check_ran checks=0\n");
        return 2;
    }
    if (g_fail != 0) {
        std::printf("RSZ_LANE_ROWS_TEST FAIL %d\n", g_fail);
        return 1;
    }
    std::printf("RSZ_LANE_ROWS_TEST PASS\n");
    return 0;
}
