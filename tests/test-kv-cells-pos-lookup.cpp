// Behavioural boundary tests for llama_kv_cells::find_cell_by_pos.
//
// The lookup has a conservative fast path: a query above the monotone
// position upper bound returns -1 without walking `used`. The tests compare
// every result with a brute-force scan of the used set, so a bound that is
// too low (a false miss) fails here. Removals and downward shifts may leave
// the bound high; those queries must fall back to the scan and still return
// the same cell (or -1) as the reference.
#include "llama-kv-cells.h"

#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

static int failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

static void check_lookup(const llama_kv_cells & cells, llama_pos p, int line) {
    int64_t want = -1;

    for (const uint32_t i : cells.get_used_set()) {
        if (cells.pos_get(i) == p) {
            want = i;
            break;
        }
    }

    const int64_t got = cells.find_cell_by_pos(p);

    if (got != want) {
        std::fprintf(stderr,
                "FAIL %s:%d: find_cell_by_pos(%d) = %lld, want %lld\n",
                __FILE__, line, (int) p, (long long) got, (long long) want);
        ++failures;
    }
}

#define CHECK_LOOKUP(cells, p) check_lookup((cells), (p), __LINE__)

static void seed(llama_kv_cells & cells, uint32_t n) {
    cells.resize(n);
}

static void add_cell(llama_kv_cells & cells, uint32_t i, llama_pos p, llama_seq_id s) {
    cells.pos_set(i, p);
    cells.seq_add(i, s);
}

static void test_empty() {
    llama_kv_cells cells;
    seed(cells, 8);

    CHECK(cells.find_cell_by_pos(-1) == -1);
    CHECK(cells.find_cell_by_pos(0) == -1);
    CHECK(cells.find_cell_by_pos(1) == -1);
    CHECK(cells.find_cell_by_pos(std::numeric_limits<llama_pos>::max()) == -1);
}

static void test_hits_gaps_and_duplicates() {
    llama_kv_cells cells;
    seed(cells, 10);

    add_cell(cells, 0, 0, 0);
    add_cell(cells, 1, 2, 0);
    cells.seq_add(1, 1); // one cell visible to two sequences
    add_cell(cells, 2, 3, 1);
    add_cell(cells, 4, 7, 2);
    add_cell(cells, 5, 7, 3); // a second cell with the same position is legal

    for (llama_pos p = -1; p <= 10; ++p) {
        CHECK_LOOKUP(cells, p);
    }

    // The reference order is the ascending cell index of `used`.
    CHECK(cells.find_cell_by_pos(2) == 1);
    CHECK(cells.find_cell_by_pos(7) == 4);
    CHECK(cells.find_cell_by_pos(8) == -1);
}

static void test_used_cell_without_sequence() {
    llama_kv_cells cells;
    seed(cells, 4);

    cells.pos_set(0, 42);

    CHECK(cells.find_cell_by_pos(42) == 0);
    CHECK(cells.find_cell_by_pos(43) == -1);

    cells.rm(0);

    CHECK(cells.find_cell_by_pos(42) == -1);
    CHECK(cells.find_cell_by_pos(43) == -1);
}

static void test_remove_max_keeps_bound_conservative() {
    llama_kv_cells cells;
    seed(cells, 8);

    add_cell(cells, 0, 4, 0);
    add_cell(cells, 1, 9, 0);

    CHECK(cells.find_cell_by_pos(9) == 1);

    cells.rm(1);

    // The bound stays at 9. The query below it must fall back to the scan and
    // must not report the removed cell.
    CHECK(cells.find_cell_by_pos(9) == -1);
    CHECK(cells.find_cell_by_pos(10) == -1);
    CHECK(cells.find_cell_by_pos(4) == 0);

    add_cell(cells, 2, 20, 0);
    CHECK(cells.find_cell_by_pos(20) == 2);
    CHECK(cells.seq_rm(2, 0));
    CHECK(cells.find_cell_by_pos(20) == -1);
    CHECK(cells.find_cell_by_pos(21) == -1);

    add_cell(cells, 3, 30, 0);
    CHECK(cells.find_cell_by_pos(30) == 3);
    CHECK(cells.seq_keep(3, 1)); // the cell holds only seq 0
    CHECK(cells.find_cell_by_pos(30) == -1);
    CHECK(cells.find_cell_by_pos(31) == -1);
}

static void test_shift_up_and_down() {
    llama_kv_cells cells;
    seed(cells, 8);

    add_cell(cells, 0, 5, 0);
    add_cell(cells, 1, 6, 0);

    cells.pos_add(0, 100);
    cells.pos_add(1, 100);

    CHECK(cells.find_cell_by_pos(105) == 0);
    CHECK(cells.find_cell_by_pos(106) == 1);
    CHECK(cells.find_cell_by_pos(5) == -1);
    CHECK(cells.find_cell_by_pos(107) == -1);

    cells.pos_add(0, -3); // 102
    cells.pos_add(1, -3); // 103

    CHECK(cells.find_cell_by_pos(102) == 0);
    CHECK(cells.find_cell_by_pos(103) == 1);
    CHECK(cells.find_cell_by_pos(104) == -1);
    CHECK(cells.find_cell_by_pos(105) == -1); // in range of the stale bound
    CHECK(cells.find_cell_by_pos(106) == -1);

    CHECK(cells.pos_add(0, -200)); // the cell becomes empty
    CHECK(cells.find_cell_by_pos(102) == -1);
    CHECK(cells.find_cell_by_pos(103) == 1);
    CHECK(cells.find_cell_by_pos(104) == -1);
}

static void test_pos_div() {
    llama_kv_cells cells;
    seed(cells, 8);

    add_cell(cells, 0, 10, 0);
    add_cell(cells, 1, 20, 0);

    cells.pos_div(0, 2);
    cells.pos_div(1, 2);

    CHECK(cells.find_cell_by_pos(5) == 0);
    CHECK(cells.find_cell_by_pos(10) == 1);
    CHECK(cells.find_cell_by_pos(4) == -1);
    CHECK(cells.find_cell_by_pos(11) == -1);
}

static void test_copy_set_and_swap() {
    llama_kv_cells src;
    seed(src, 8);
    add_cell(src, 0, 100, 0);
    add_cell(src, 1, 200, 0);

    llama_kv_cells copy(src);
    CHECK(copy.find_cell_by_pos(100) == 0);
    CHECK(copy.find_cell_by_pos(200) == 1);
    CHECK(copy.find_cell_by_pos(201) == -1);

    llama_kv_cells assigned;
    seed(assigned, 4);
    add_cell(assigned, 0, 5, 0);
    assigned = src;
    CHECK(assigned.find_cell_by_pos(100) == 0);
    CHECK(assigned.find_cell_by_pos(200) == 1);
    CHECK(assigned.find_cell_by_pos(5) == -1);
    CHECK(assigned.find_cell_by_pos(201) == -1);

    llama_kv_cells moved(std::move(src));
    CHECK(moved.find_cell_by_pos(100) == 0);
    CHECK(moved.find_cell_by_pos(200) == 1);
    CHECK(moved.find_cell_by_pos(201) == -1);

    llama_kv_cells moved_assign;
    seed(moved_assign, 4);
    moved_assign = std::move(moved);
    CHECK(moved_assign.find_cell_by_pos(100) == 0);
    CHECK(moved_assign.find_cell_by_pos(200) == 1);
    CHECK(moved_assign.find_cell_by_pos(201) == -1);

    llama_kv_cells part = copy.cp(0, 2);
    CHECK(part.get_used() == 0); // snapshots carry pos/seq, not the used set
    CHECK(part.pos_get(0) == 100);
    CHECK(part.pos_get(1) == 200);
    CHECK(part.find_cell_by_pos(100) == -1);

    llama_kv_cells part2 = copy.cp(std::vector<uint32_t>{1});
    CHECK(part2.get_used() == 0);
    CHECK(part2.pos_get(0) == 200);
    CHECK(part2.find_cell_by_pos(200) == -1);

    llama_kv_cells out;
    seed(out, 8);
    add_cell(out, 0, 999, 0);
    copy.cp_into(std::vector<uint32_t>{0, 1}, out);
    CHECK(out.get_used() == 0); // cp_into keeps the snapshot contract
    CHECK(out.pos_get(0) == 100);
    CHECK(out.pos_get(1) == 200);
    CHECK(out.find_cell_by_pos(100) == -1);
    CHECK(out.find_cell_by_pos(999) == -1);

    llama_kv_cells dst;
    seed(dst, 16);
    add_cell(dst, 0, 7, 0);
    dst.set(2, copy);
    CHECK(dst.find_cell_by_pos(7) == 0);
    CHECK(dst.find_cell_by_pos(100) == 2);
    CHECK(dst.find_cell_by_pos(200) == 3);
    CHECK(dst.find_cell_by_pos(201) == -1);

    llama_kv_cells small;
    seed(small, 2);
    add_cell(small, 0, 300, 0);
    add_cell(small, 1, 400, 0);

    llama_kv_cells dst2;
    seed(dst2, 4);
    add_cell(dst2, 1, 7, 0);
    dst2.set(std::vector<uint32_t>{0, 2}, small);
    CHECK(dst2.find_cell_by_pos(300) == 0);
    CHECK(dst2.find_cell_by_pos(400) == 2);
    CHECK(dst2.find_cell_by_pos(7) == 1);
    CHECK(dst2.find_cell_by_pos(401) == -1);

    llama_kv_cells a;
    seed(a, 4);
    add_cell(a, 0, 10, 0);

    llama_kv_cells b;
    seed(b, 4);
    add_cell(b, 0, 1000, 0);

    a.swap_contents(b);
    CHECK(a.find_cell_by_pos(1000) == 0);
    CHECK(a.find_cell_by_pos(10) == -1);
    CHECK(a.find_cell_by_pos(1001) == -1);
    CHECK(b.find_cell_by_pos(10) == 0);
    CHECK(b.find_cell_by_pos(1000) == -1);
    CHECK(b.find_cell_by_pos(11) == -1);
}

static void test_reset_and_resize() {
    llama_kv_cells cells;
    seed(cells, 8);
    add_cell(cells, 0, 50, 0);

    CHECK(cells.find_cell_by_pos(50) == 0);

    cells.reset();

    CHECK(cells.find_cell_by_pos(50) == -1);
    CHECK(cells.find_cell_by_pos(51) == -1);
    CHECK(cells.find_cell_by_pos(0) == -1);

    add_cell(cells, 1, 5, 0);
    CHECK(cells.find_cell_by_pos(5) == 1);
    CHECK(cells.find_cell_by_pos(50) == -1);

    cells.resize(2);

    CHECK(cells.find_cell_by_pos(5) == -1);
    CHECK(cells.find_cell_by_pos(6) == -1);

    add_cell(cells, 0, 3, 0);
    CHECK(cells.find_cell_by_pos(3) == 0);
    CHECK(cells.find_cell_by_pos(4) == -1);
}

static void test_restore_cell_writer() {
    llama_kv_cells cells;
    seed(cells, 4);

    llama_kv_cell_ext ext {};
    std::bitset<LLAMA_MAX_SEQ> seq;
    seq.set(0);

    cells.restore_cell(0, 500, ext, 0, seq);
    CHECK(cells.find_cell_by_pos(500) == 0);
    CHECK(cells.find_cell_by_pos(501) == -1);

    cells.restore_cell(1, 600, ext, 0, seq);
    CHECK(cells.find_cell_by_pos(600) == 1);
    CHECK(cells.find_cell_by_pos(601) == -1);

    cells.restore_cell(0, -1, ext, 0, {});
    CHECK(cells.find_cell_by_pos(500) == -1);
    CHECK(cells.find_cell_by_pos(501) == -1);
}

static void test_journal_replay() {
    llama_kv_cells cells;
    seed(cells, 4);
    add_cell(cells, 0, 7, 2);
    add_cell(cells, 1, 11, 2);

    cell_journal j;
    j.begin(4);
    cells.set_journal(&j);

    cells.rm(0);
    cells.seq_rm(1, 2);

    CHECK(cells.find_cell_by_pos(7) == -1);
    CHECK(cells.find_cell_by_pos(11) == -1);
    CHECK(cells.find_cell_by_pos(12) == -1);

    cells.set_journal(nullptr);
    j.replay([&](cell_journal::entry & e) {
        CHECK(cells.restore_from_journal(j, e));
    });

    CHECK(cells.find_cell_by_pos(7) == 0);
    CHECK(cells.find_cell_by_pos(11) == 1);
    CHECK(cells.find_cell_by_pos(12) == -1);

    // A trial that adds a higher position raises the bound; the replay removes
    // that cell but leaves the bound high. The removed position must fall back
    // to the scan and return -1.
    llama_kv_cells cells2;
    seed(cells2, 4);
    add_cell(cells2, 0, 5, 0);

    cell_journal j2;
    j2.begin(4);
    cells2.set_journal(&j2);

    add_cell(cells2, 1, 50, 0);
    CHECK(cells2.find_cell_by_pos(50) == 1);
    CHECK(cells2.find_cell_by_pos(51) == -1);

    cells2.set_journal(nullptr);
    j2.replay([&](cell_journal::entry & e) {
        CHECK(cells2.restore_from_journal(j2, e));
    });

    CHECK(cells2.find_cell_by_pos(5) == 0);
    CHECK(cells2.find_cell_by_pos(50) == -1);
    CHECK(cells2.find_cell_by_pos(51) == -1);
}

static uint32_t rng_state = 0x9e3779b9u;

static uint32_t next_rand() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static void test_randomized_differential() {
    llama_kv_cells cells;
    seed(cells, 32);

    for (int step = 0; step < 400; ++step) {
        const uint32_t op = next_rand() % 8;
        const uint32_t i  = next_rand() % 32;

        switch (op) {
            case 0:
                if (cells.is_empty(i)) {
                    const llama_pos p = (llama_pos) (next_rand() % 200);
                    cells.pos_set(i, p);
                    cells.seq_add(i, (llama_seq_id) (next_rand() % 4));
                }
                break;
            case 1:
                if (!cells.is_empty(i)) {
                    cells.rm(i);
                }
                break;
            case 2:
                if (!cells.is_empty(i)) {
                    const llama_seq_id s = (llama_seq_id) (next_rand() % 4);
                    if (!cells.seq_has(i, s)) {
                        cells.seq_add(i, s);
                    }
                }
                break;
            case 3:
                if (!cells.is_empty(i)) {
                    const llama_seq_id s = (llama_seq_id) (next_rand() % 4);
                    if (cells.seq_has(i, s)) {
                        cells.seq_rm(i, s);
                    }
                }
                break;
            case 4:
                if (!cells.is_empty(i)) {
                    cells.pos_add(i, (llama_pos) ((int) (next_rand() % 41) - 20));
                }
                break;
            case 5:
                if (!cells.is_empty(i)) {
                    cells.pos_div(i, 2);
                }
                break;
            case 6:
                if ((next_rand() % 16) == 0) {
                    cells.reset();
                }
                break;
            case 7:
                if (!cells.is_empty(i)) {
                    cells.seq_keep(i, (llama_seq_id) (next_rand() % 4));
                }
                break;
        }

        // A too-low bound would turn an existing position into a miss. Check
        // every position that is currently present, plus the empty edges.
        llama_pos maxp = -1;
        for (const uint32_t idx : cells.get_used_set()) {
            const llama_pos p = cells.pos_get(idx);
            check_lookup(cells, p, __LINE__);
            if (p > maxp) {
                maxp = p;
            }
        }

        check_lookup(cells, -1, __LINE__);
        check_lookup(cells, maxp + 1, __LINE__);
        check_lookup(cells, (llama_pos) (next_rand() % (maxp + 2)), __LINE__);
    }
}

int main() {
    test_empty();
    test_hits_gaps_and_duplicates();
    test_used_cell_without_sequence();
    test_remove_max_keeps_bound_conservative();
    test_shift_up_and_down();
    test_pos_div();
    test_copy_set_and_swap();
    test_reset_and_resize();
    test_restore_cell_writer();
    test_journal_replay();
    test_randomized_differential();

    if (failures != 0) {
        std::fprintf(stderr, "KV_CELLS_POS_LOOKUP_TEST FAIL %d\n", failures);
        return 1;
    }

    std::printf("KV_CELLS_POS_LOOKUP_TEST PASS\n");
    return 0;
}
