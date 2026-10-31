// verify/host_frontier_boundary.cpp — CPU boundary check for the production
// per-sequence used-cell frontier (the source of the FA true KV length).
//
// NOT EXECUTED in this session (user discipline: code + static review only).
// Exact build/run commands: ../RUNBOOK.md.
//
// Includes the production translation unit, so llama_kv_cells::used_max_p1 and
// llama_kv_cells::seq_used_max_p1 (baseline/new code) are the code under test.
// Boundaries 1/255/256/257/511/512, multi-sequence, removal, absent sequence,
// holes, and a deterministic pseudo-random brute-force cross-check.

#include "llama-kv-cache.cpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>

namespace {

int g_fail = 0;


void expect(bool ok, const std::string & what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) {
        ++g_fail;
    }
}

} // namespace

int main() {
    // empty cells: both frontiers are 0
    {
        llama_kv_cells cells;
        cells.resize(512);
        expect(cells.used_max_p1() == 0, "empty used_max_p1 == 0");
#ifdef WITH_NKV_TRUE_LEN
        expect(cells.seq_used_max_p1(0) == 0, "empty seq_used_max_p1(0) == 0");
        expect(cells.seq_used_max_p1(7) == 0, "empty seq_used_max_p1(7) == 0");
#endif
    }

    // single sequence, highest used cell exactly at width-1
    for (uint32_t w : { 1u, 255u, 256u, 257u, 511u, 512u }) {
        llama_kv_cells cells;
        cells.resize(w);
        cells.pos_set(w - 1, (llama_pos) w - 1);
        cells.seq_add(w - 1, 0);
        if (w > 1) {
            cells.pos_set(0, 0);
            cells.seq_add(0, 0);
        }

        expect(cells.used_max_p1() == w, "used_max_p1 == " + std::to_string(w));
#ifdef WITH_NKV_TRUE_LEN
        expect(cells.seq_used_max_p1(0) == w,
               "seq_used_max_p1 == " + std::to_string(w) + " (highest cell at width-1)");
        expect(cells.seq_used_max_p1(1) == 0,
               "seq_used_max_p1(absent seq 1) == 0 at width " + std::to_string(w));
#endif
    }

    // holes below the frontier do not change it; a sequence's frontier ignores
    // cells of other sequences
    {
        llama_kv_cells cells;
        cells.resize(512);
        // seq 0: cells 10 and 300; seq 1: cells 0 and 511
        cells.pos_set(10, 1);
        cells.seq_add(10, 0);
        cells.pos_set(300, 2);
        cells.seq_add(300, 0);
        cells.pos_set(0, 0);
        cells.seq_add(0, 1);
        cells.pos_set(511, 3);
        cells.seq_add(511, 1);
        // cell 400 carries both sequences
        cells.pos_set(400, 4);
        cells.seq_add(400, 0);
        cells.seq_add(400, 1);

#ifdef WITH_NKV_TRUE_LEN
        expect(cells.seq_used_max_p1(0) == 401, "multi-seq seq0 frontier = 401 (shared cell 400)");
        expect(cells.seq_used_max_p1(1) == 512, "multi-seq seq1 frontier = 512");
        expect(cells.seq_used_max_p1(2) == 0, "multi-seq absent seq2 frontier = 0");

        // removal of the highest cell drops the frontier to the next one
        expect(!cells.seq_rm(400, 0), "shared cell remains occupied after removing seq 0");
        expect(cells.seq_used_max_p1(0) == 301, "seq0 frontier after removing cell 400 = 301");
        expect(cells.used_max_p1() == 512, "used_max_p1 unaffected by sequence removal = 512");
#endif
    }

#ifdef WITH_NKV_TRUE_LEN
    // deterministic pseudo-random cross-check against brute force, at the
    // padding boundaries that motivated the true-length lane
    for (uint32_t cap : { 256u, 257u, 511u, 512u }) {
        llama_kv_cells cells;
        cells.resize(cap);

        uint32_t state = 0x1234567u ^ cap;
        auto next = [&state]() {
            state = state * 1664525u + 1013904223u;
            return state;
        };

        std::vector<uint32_t> best(4, 0);
        for (uint32_t i = 0; i < cap; ++i) {
            // ~35% of cells used, spread over sequences 0..3
            if (next() % 100 >= 35) {
                continue;
            }
            cells.pos_set(i, (llama_pos) i);
            const uint32_t nseq = 1 + next() % 3;
            for (uint32_t k = 0; k < nseq; ++k) {
                const llama_seq_id s = (llama_seq_id) (next() % 4);
                if (!cells.seq_has(i, s)) {
                    cells.seq_add(i, s);
                }
                best[s] = std::max(best[s], i + 1);
            }
        }

        for (llama_seq_id s = 0; s < 4; ++s) {
            expect(cells.seq_used_max_p1(s) == best[s],
                   "brute-force frontier cap=" + std::to_string(cap) + " seq=" + std::to_string(s));
        }
        expect(cells.used_max_p1() <= cap, "frontier bounded by capacity cap=" + std::to_string(cap));
    }
#endif

    std::printf("%s host_frontier_boundary failures=%d\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
