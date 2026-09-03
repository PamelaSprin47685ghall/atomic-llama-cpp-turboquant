#include "../src/llama-kv-cells.h"

#include <cassert>

static llama_kv_rerot_meta make_rerot_meta(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        llama_rerot_run_id run_id,
        llama_rerot_visibility visibility,
        uint64_t publish_epoch,
        uint64_t frontier) {
    llama_kv_rerot_meta result;
    result.episode_id = episode_id;
    result.node_id = node_id;
    result.run_id = run_id;
    result.visibility = visibility;
    result.publish_epoch = publish_epoch;
    result.frontier = frontier;
    return result;
}

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

    // Test 7: RERoT metadata follows the physical cell through copy/restore,
    // publication, compaction, and final reference removal.
    {
        llama_kv_cells cells;
        cells.resize(12);

        cells.pos_set(2, 17);
        cells.seq_add(2, 3);
        const auto pending = make_rerot_meta(
            91, 4, 7, llama_rerot_visibility::pending_record, 0, 13);
        cells.rerot_set(2, pending);
        assert(cells.rerot_get(2) == pending);

        const auto saved = cells.cp(2, 1);
        cells.rm(2);
        assert(!cells.rerot_get(2).active());
        cells.set(8, saved);
        assert(cells.pos_get(8) == 17);
        assert(cells.rerot_get(8) == pending);

        assert(!cells.rerot_publish(8, 91, 8, 14));
        assert(cells.rerot_publish(8, 91, 7, 14));
        const auto published = cells.rerot_get(8);
        assert(published.visibility == llama_rerot_visibility::public_live);
        assert(published.publish_epoch == 14);

        cells.pos_set(10, 23);
        cells.seq_add(10, 4);
        const auto private_meta = make_rerot_meta(
            91, 6, 9, llama_rerot_visibility::private_control, 0, 14);
        cells.rerot_set(10, private_meta);

        const auto plan = cells.make_pack_plan();
        assert(plan.retained_count == 2);
        cells.apply_pack(plan);

        assert(cells.pos_get(0) == 17);
        assert(cells.rerot_get(0) == published);
        assert(cells.pos_get(1) == 23);
        assert(cells.rerot_get(1) == private_meta);

        assert(cells.seq_rm(0, 3));
        assert(cells.is_empty(0));
        assert(!cells.rerot_get(0).active());

        assert(cells.pos_add(1, -24));
        assert(cells.is_empty(1));
        assert(!cells.rerot_get(1).active());
    }

    return 0;
}
