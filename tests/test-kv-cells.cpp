#include "../src/llama-kv-cells.h"

#include <cassert>

int main() {
    llama_kv_cells cells;
    cells.resize(4);

    // Duplicate positions still occupy distinct resident cells.
    cells.pos_set(0, 7);
    cells.seq_add(0, 0);
    cells.pos_set(1, 7);
    cells.seq_add(1, 0);
    assert(cells.seq_get_used(0) == 2);
    assert(cells.seq_pos_min(0) == 7);
    assert(cells.seq_pos_max(0) == 7);

    // A shared cell counts as resident for every sequence that references it.
    cells.seq_add(0, 1);
    assert(cells.seq_get_used(0) == 2);
    assert(cells.seq_get_used(1) == 1);

    cells.seq_rm(0, 0);
    assert(cells.seq_get_used(0) == 1);
    assert(cells.seq_get_used(1) == 1);

    // State restore must rebuild the per-sequence resident counters.
    const auto saved = cells.cp(0, 2);
    cells.rm(0);
    cells.rm(1);
    assert(cells.seq_get_used(0) == 0);
    assert(cells.seq_get_used(1) == 0);

    cells.set(0, saved);
    assert(cells.seq_get_used(0) == 1);
    assert(cells.seq_get_used(1) == 1);

    // Position changes and sequence filtering must preserve exact counts.
    cells.seq_keep(0, 1);
    assert(cells.seq_get_used(0) == 1);
    assert(cells.seq_get_used(1) == 1);

    cells.pos_add(0, -8);
    assert(cells.seq_get_used(1) == 0);
    assert(cells.seq_get_used(0) == 1);

    cells.rm(1);
    assert(cells.seq_get_used(0) == 0);

    // === Compaction tests ===

    // Test 1: Basic packing — cells at 0,1,2, 5,6, 10,11
    {
        llama_kv_cells cells;
        cells.resize(16);

        cells.pos_set(0, 0);
        cells.seq_add(0, 0);
        cells.pos_set(1, 1);
        cells.seq_add(1, 0);
        cells.pos_set(2, 2);
        cells.seq_add(2, 0);

        cells.pos_set(5, 5);
        cells.seq_add(5, 0);
        cells.pos_set(6, 6);
        cells.seq_add(6, 0);

        cells.pos_set(10, 10);
        cells.seq_add(10, 0);
        cells.pos_set(11, 11);
        cells.seq_add(11, 0);

        assert(cells.get_used() == 7);
        assert(cells.used_max_p1() == 12);

        auto plan = cells.make_pack_plan();

        assert(plan.retained_count == 7);

        // All moves have dst <= src
        for (const auto & move : plan.moves) {
            assert(move.dst_begin <= move.src_begin);
        }

        // Moves cover all used cells
        uint32_t total_covered = 0;
        for (const auto & move : plan.moves) {
            total_covered += move.length;
        }
        assert(total_covered == 7);

        cells.apply_pack(plan);

        // Used set is [0, 7)
        assert(cells.get_used() == 7);
        assert(cells.used_min() == 0);
        assert(cells.used_max_p1() == 7);

        // Positions preserved (original values, now at dense indices)
        const llama_pos expected_pos[7] = {0, 1, 2, 5, 6, 10, 11};
        for (uint32_t i = 0; i < 7; ++i) {
            assert(!cells.is_empty(i));
            assert(cells.pos_get(i) == expected_pos[i]);
            assert(cells.seq_has(i, 0));
        }

        assert(cells.seq_pos_min(0) == 0);
        assert(cells.seq_pos_max(0) == 11);
        assert(cells.seq_get_used(0) == 7);
    }

    // Test 2: Shared cells (multiple seq refs on one cell)
    {
        llama_kv_cells cells;
        cells.resize(16);

        cells.pos_set(0, 0);
        cells.seq_add(0, 0);
        cells.seq_add(0, 1);  // shared cell

        cells.pos_set(3, 3);
        cells.seq_add(3, 0);

        cells.pos_set(7, 7);
        cells.seq_add(7, 1);

        assert(cells.get_used() == 3);
        assert(cells.seq_get_used(0) == 2);
        assert(cells.seq_get_used(1) == 2);

        auto plan = cells.make_pack_plan();
        assert(plan.retained_count == 3);

        cells.apply_pack(plan);

        assert(cells.get_used() == 3);
        assert(cells.used_max_p1() == 3);

        // Cell 0: shared by seq 0 and 1, pos 0
        assert(cells.pos_get(0) == 0);
        assert(cells.seq_has(0, 0));
        assert(cells.seq_has(0, 1));
        assert(cells.seq_count(0) == 2);

        // Cell 1: seq 0, pos 3
        assert(cells.pos_get(1) == 3);
        assert(cells.seq_has(1, 0));
        assert(!cells.seq_has(1, 1));

        // Cell 2: seq 1, pos 7
        assert(cells.pos_get(2) == 7);
        assert(!cells.seq_has(2, 0));
        assert(cells.seq_has(2, 1));

        assert(cells.seq_pos_min(0) == 0);
        assert(cells.seq_pos_max(0) == 3);
        assert(cells.seq_pos_min(1) == 0);
        assert(cells.seq_pos_max(1) == 7);
        assert(cells.seq_get_used(0) == 2);
        assert(cells.seq_get_used(1) == 2);
    }

    // Test 3: Duplicate positions (same position in different cells)
    {
        llama_kv_cells cells;
        cells.resize(16);

        cells.pos_set(0, 5);
        cells.seq_add(0, 0);
        cells.pos_set(3, 5);  // duplicate position
        cells.seq_add(3, 0);

        cells.pos_set(7, 10);
        cells.seq_add(7, 0);

        assert(cells.get_used() == 3);
        assert(cells.seq_get_used(0) == 3);
        assert(cells.seq_pos_min(0) == 5);
        assert(cells.seq_pos_max(0) == 10);

        auto plan = cells.make_pack_plan();
        assert(plan.retained_count == 3);

        cells.apply_pack(plan);

        assert(cells.get_used() == 3);
        assert(cells.used_max_p1() == 3);

        assert(cells.pos_get(0) == 5);
        assert(cells.pos_get(1) == 5);
        assert(cells.pos_get(2) == 10);

        // seq_pos with duplicate: count at pos 5 should be 2
        assert(cells.seq_pos_min(0) == 5);
        assert(cells.seq_pos_max(0) == 10);
        assert(cells.seq_get_used(0) == 3);
    }

    // Test 4: Already dense (no compaction needed)
    {
        llama_kv_cells cells;
        cells.resize(16);

        cells.pos_set(0, 0);
        cells.seq_add(0, 0);
        cells.pos_set(1, 1);
        cells.seq_add(1, 0);
        cells.pos_set(2, 2);
        cells.seq_add(2, 0);

        auto plan = cells.make_pack_plan();
        assert(plan.retained_count == 3);
        for (const auto & move : plan.moves) {
            assert(move.dst_begin == move.src_begin);
        }

        cells.apply_pack(plan);

        assert(cells.get_used() == 3);
        assert(cells.used_max_p1() == 3);
        assert(cells.pos_get(0) == 0);
        assert(cells.pos_get(1) == 1);
        assert(cells.pos_get(2) == 2);
    }

    // Test 5: Empty cache
    {
        llama_kv_cells cells;
        cells.resize(16);

        auto plan = cells.make_pack_plan();
        assert(plan.retained_count == 0);
        assert(plan.moves.empty());

        cells.apply_pack(plan);
        assert(cells.get_used() == 0);
    }

    // Test 6: All cells used except one gap — verify ext field preservation
    {
        llama_kv_cells cells;
        cells.resize(8);

        cells.pos_set(0, 0);
        cells.seq_add(0, 0);
        cells.ext_set(0, {1, 2});

        cells.pos_set(1, 1);
        cells.seq_add(1, 0);

        // gap at index 2

        cells.pos_set(3, 3);
        cells.seq_add(3, 0);
        cells.ext_set(3, {7, 8});

        auto plan = cells.make_pack_plan();
        assert(plan.retained_count == 3);

        cells.apply_pack(plan);

        assert(cells.get_used() == 3);
        assert(cells.used_max_p1() == 3);

        assert(cells.pos_get(0) == 0);
        assert(cells.ext_get(0).x == 1);
        assert(cells.ext_get(0).y == 2);

        assert(cells.pos_get(1) == 1);

        assert(cells.pos_get(2) == 3);
        assert(cells.ext_get(2).x == 7);
        assert(cells.ext_get(2).y == 8);
    }

    return 0;
}
