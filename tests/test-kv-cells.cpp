// This test uses assert() as its failure mechanism. Keep those checks active
// when the surrounding project is configured with CMAKE_BUILD_TYPE=Release.
#ifdef NDEBUG
#undef NDEBUG
#endif

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
        assert(!cells.rerot_reclassify(
            8, 91, 7,
            llama_rerot_visibility::pending_record,
            llama_rerot_visibility::private_control,
            14));
        assert(cells.rerot_get(8) == pending);
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

    // === Generation (FlashPrefill invalidation stamp) tests ===
    // Contract: OFF by default with stamp 0 / generation 0, no bumps and no
    // atomic work; opt-in via set_generation_enabled() which lazily assigns a
    // unique stamp and bumps once; identity is (stamp, generation); consumers
    // treat generation == MAX or stamp == MAX as always-invalid (saturating
    // overflow needs 2^64 steps and is documented but not exercised here).
    {
        // Disabled by default: lazy stamp 0, mutations do not bump or assign.
        llama_kv_cells cells;
        cells.resize(4);
        assert(!cells.get_generation_enabled());
        assert(cells.get_generation() == 0);
        assert(cells.get_generation_stamp() == 0);

        cells.pos_set(0, 7);
        cells.seq_add(0, 0);
        cells.ext_set(0, {1, 2});
        assert(cells.get_generation() == 0);
        assert(cells.get_generation_stamp() == 0);

        // Disabled copies stay untracked with no atomic work.
        llama_kv_cells disabled_copy(cells);
        assert(!disabled_copy.get_generation_enabled());
        assert(disabled_copy.get_generation() == 0);
        assert(disabled_copy.get_generation_stamp() == 0);
        const auto disabled_snap = cells.cp(0, 2);
        assert(!disabled_snap.get_generation_enabled());
        assert(disabled_snap.get_generation_stamp() == 0);

        // Enabling lazily assigns a unique stamp and bumps to a non-zero baseline.
        cells.set_generation_enabled(true);
        assert(cells.get_generation_enabled());
        assert(cells.get_generation() != 0);
        assert(cells.get_generation_stamp() != 0);
        // Enabling twice is idempotent (no second bump, same stamp).
        const uint64_t base = cells.get_generation();
        const uint64_t base_stamp = cells.get_generation_stamp();
        cells.set_generation_enabled(true);
        assert(cells.get_generation() == base);
        assert(cells.get_generation_stamp() == base_stamp);
    }

    {
        // Enabled membership/ext mutations bump; read-only queries do not.
        llama_kv_cells cells;
        cells.resize(4);
        cells.set_generation_enabled(true);
        const uint64_t base = cells.get_generation();

        cells.pos_set(0, 10);
        assert(cells.get_generation() != base);
        uint64_t g = cells.get_generation();

        cells.seq_add(0, 0);
        assert(cells.get_generation() != g);
        g = cells.get_generation();

        cells.ext_set(0, {3, 4});
        assert(cells.get_generation() != g);
        g = cells.get_generation();

        // Read-only queries never bump.
        (void) cells.is_empty(0);
        (void) cells.get_used();
        (void) cells.used_min();
        (void) cells.used_max_p1();
        (void) cells.get_has_shift();
        (void) cells.seq_has(0, 0);
        (void) cells.seq_count(0);
        (void) cells.seq_get(0);
        (void) cells.seq_pos_min(0);
        (void) cells.seq_pos_max(0);
        (void) cells.seq_get_used(0);
        (void) cells.pos_get(0);
        (void) cells.ext_get(0);
        (void) cells.rerot_get(0);
        (void) cells.get_shift(0);
        (void) cells.pos_in(0, 0, 20);
        (void) cells.rerot_has_active();
        (void) cells.rerot_has_active_seq(0);
        (void) cells.make_pack_plan();
        std::vector<uint32_t> out;
        (void) cells.rerot_collect_run(1, 1, out);
        assert(cells.get_generation() == g);
    }

    {
        // Shift/position mutations bump; no-op reset_shift does not.
        llama_kv_cells cells;
        cells.resize(4);
        cells.set_generation_enabled(true);
        cells.pos_set(0, 10);
        cells.seq_add(0, 0);
        uint64_t g = cells.get_generation();

        // No pending shift: no-op reset does not bump.
        cells.reset_shift();
        assert(cells.get_generation() == g);

        cells.pos_add(0, 5);
        assert(cells.get_generation() != g);
        g = cells.get_generation();

        cells.pos_div(0, 2);
        assert(cells.get_generation() != g);
        g = cells.get_generation();

        // Consuming the pending shift bumps once.
        assert(cells.get_has_shift());
        cells.reset_shift();
        assert(cells.get_generation() != g);
        g = cells.get_generation();
        // Second reset with no shift is a no-op again.
        cells.reset_shift();
        assert(cells.get_generation() == g);
    }

    {
        // Sequence membership: real changes bump, single-keeper/empty no-ops do not.
        llama_kv_cells cells;
        cells.resize(4);
        cells.set_generation_enabled(true);
        cells.pos_set(0, 1);
        cells.seq_add(0, 0);
        uint64_t g = cells.get_generation();

        // Keeping the sole ref is a no-op.
        assert(!cells.seq_keep(0, 0));
        assert(cells.get_generation() == g);

        // Adding a second ref bumps.
        cells.seq_add(0, 1);
        assert(cells.get_generation() != g);
        g = cells.get_generation();

        // Keeping one of two refs removes the other: bumps.
        assert(!cells.seq_keep(0, 0));
        assert(cells.get_generation() != g);
        g = cells.get_generation();

        // Removing the last ref via seq_rm bumps and frees.
        assert(cells.seq_rm(0, 0));
        assert(cells.is_empty(0));
        assert(cells.get_generation() != g);
        g = cells.get_generation();

        // Filtering an already-empty cell is a no-op.
        assert(!cells.seq_keep(1, 0));
        assert(cells.get_generation() == g);
    }

    {
        // Failed rerot guards do not bump; success bumps.
        llama_kv_cells cells;
        cells.resize(4);
        cells.set_generation_enabled(true);
        cells.pos_set(0, 5);
        cells.seq_add(0, 0);
        const auto pending = make_rerot_meta(11, 2, 3, llama_rerot_visibility::pending_record, 0, 5);
        cells.rerot_set(0, pending);
        uint64_t g = cells.get_generation();

        // Wrong run: fail, no bump.
        assert(!cells.rerot_publish(0, 11, 4, 7));
        assert(!cells.rerot_reclassify(
            0, 11, 3,
            llama_rerot_visibility::pending_record,
            llama_rerot_visibility::private_control,
            7));
        assert(cells.get_generation() == g);
        assert(cells.rerot_get(0) == pending);

        // Success bumps.
        assert(cells.rerot_publish(0, 11, 3, 7));
        assert(cells.get_generation() != g);
        g = cells.get_generation();

        // Resetting an active tag bumps; resetting inactive does not.
        cells.rerot_reset(0);
        assert(cells.get_generation() != g);
        g = cells.get_generation();
        cells.rerot_reset(0);
        assert(cells.get_generation() == g);

        // Re-tag then reclassify success bumps.
        cells.rerot_set(0, pending);
        g = cells.get_generation();
        assert(cells.rerot_reclassify(
            0, 11, 3,
            llama_rerot_visibility::pending_record,
            llama_rerot_visibility::private_control,
            0));
        assert(cells.get_generation() != g);
    }

    {
        // Clearing preserves enabled+stamp and bumps (never resets to 0).
        llama_kv_cells cells;
        cells.resize(4);
        cells.set_generation_enabled(true);
        cells.pos_set(0, 1);
        cells.seq_add(0, 0);
        const uint64_t stamp = cells.get_generation_stamp();
        const uint64_t g = cells.get_generation();

        cells.reset();
        assert(cells.get_generation_enabled());
        assert(cells.get_generation_stamp() == stamp);
        assert(cells.get_generation() != g);
        assert(cells.get_generation() != 0);
        assert(cells.get_used() == 0);

        // resize (which clears) also preserves and bumps.
        const uint64_t g2 = cells.get_generation();
        cells.resize(4);
        assert(cells.get_generation_enabled());
        assert(cells.get_generation_stamp() == stamp);
        assert(cells.get_generation() != g2);

        // Disabling preserves the version; mutations stop bumping; re-enable bumps.
        cells.set_generation_enabled(false);
        assert(!cells.get_generation_enabled());
        const uint64_t g3 = cells.get_generation();
        const uint64_t stamp3 = cells.get_generation_stamp();
        cells.pos_set(1, 9);
        assert(cells.get_generation() == g3);
        cells.set_generation_enabled(true);
        assert(cells.get_generation_stamp() == stamp3);
        assert(cells.get_generation() != g3);
    }

    {
        // Copy/cp/assign never collide on (stamp, generation).
        llama_kv_cells src;
        src.resize(4);
        src.set_generation_enabled(true);
        src.pos_set(0, 3);
        src.seq_add(0, 0);
        const uint64_t src_stamp = src.get_generation_stamp();
        const uint64_t src_gen = src.get_generation();

        // cp() does not bump the source; snapshot inherits baseline with fresh stamp.
        const auto snap = src.cp(0, 2);
        assert(src.get_generation() == src_gen);
        assert(snap.get_generation_enabled());
        assert(snap.get_generation() == src_gen);
        assert(snap.get_generation_stamp() != src_stamp);

        // Copy construction: same baseline, different stamp.
        llama_kv_cells copy(src);
        assert(copy.get_generation() == src_gen);
        assert(copy.get_generation_stamp() != src_stamp);

        // Copy assignment: fresh stamp, inherited baseline.
        llama_kv_cells dst;
        dst.resize(4);
        dst.set_generation_enabled(true);
        const uint64_t dst_stamp_before = dst.get_generation_stamp();
        dst = src;
        assert(dst.get_generation() == src_gen);
        assert(dst.get_generation_stamp() != src_stamp);
        assert(dst.get_generation_stamp() != dst_stamp_before);

        // Mutating the copy diverges without colliding with the source.
        const uint64_t copy_gen = copy.get_generation();
        copy.pos_set(1, 8);
        assert(copy.get_generation() != copy_gen);
        assert(copy.get_generation_stamp() != src_stamp);
        assert(src.get_generation() == src_gen);

        // Moves transfer identity with ordinary data-move semantics and clear
        // the source to disabled 0/0/false with no atomic work.
        llama_kv_cells moved_src;
        moved_src.resize(4);
        moved_src.set_generation_enabled(true);
        moved_src.pos_set(0, 5);
        const uint64_t moved_stamp = moved_src.get_generation_stamp();
        const uint64_t moved_gen = moved_src.get_generation();
        assert(moved_stamp != 0);
        llama_kv_cells moved(std::move(moved_src));
        assert(moved.get_generation_stamp() == moved_stamp);
        assert(moved.get_generation() == moved_gen);
        assert(moved.get_generation_enabled());
        assert(moved_src.get_generation_stamp() == 0);
        assert(moved_src.get_generation() == 0);
        assert(!moved_src.get_generation_enabled());
    }

    {
        // set() restore bumps; empty restore does not.
        llama_kv_cells cells;
        cells.resize(4);
        cells.set_generation_enabled(true);
        cells.pos_set(0, 3);
        cells.seq_add(0, 0);
        const auto saved = cells.cp(0, 2);
        uint64_t g = cells.get_generation();

        cells.rm(0);
        assert(cells.get_generation() != g);
        g = cells.get_generation();

        cells.set(0, saved);
        assert(cells.get_generation() != g);
        g = cells.get_generation();

        llama_kv_cells empty;
        empty.resize(0);
        cells.set(0, empty);
        assert(cells.get_generation() == g);

        std::vector<uint32_t> no_idxs;
        cells.set(no_idxs, empty);
        assert(cells.get_generation() == g);
    }

    {
        // Pack: empty plan is a no-op; non-empty plan bumps.
        llama_kv_cells cells;
        cells.resize(8);
        cells.set_generation_enabled(true);
        uint64_t g = cells.get_generation();

        auto empty_plan = cells.make_pack_plan();
        assert(empty_plan.moves.empty());
        cells.apply_pack(empty_plan);
        assert(cells.get_generation() == g);

        cells.pos_set(0, 0);
        cells.seq_add(0, 0);
        cells.pos_set(5, 5);
        cells.seq_add(5, 0);
        g = cells.get_generation();

        // Planning itself never bumps.
        auto plan = cells.make_pack_plan();
        assert(cells.get_generation() == g);
        assert(!plan.moves.empty());

        cells.apply_pack(plan);
        assert(cells.get_generation() != g);
        assert(cells.get_used() == 2);
        assert(cells.used_max_p1() == 2);
    }

    {
        // Freeze: empty exec set is a no-op; otherwise bumps.
        llama_kv_cells cells;
        cells.resize(4);
        cells.set_generation_enabled(true);
        uint64_t g = cells.get_generation();
        // No cell references exec seq 0: no-op.
        assert(cells.rerot_freeze_to_archive(99, 0, 1) == 0);
        assert(cells.get_generation() == g);

        cells.pos_set(0, 1);
        cells.seq_add(0, 0);
        const auto pending = make_rerot_meta(99, 2, 3, llama_rerot_visibility::pending_record, 0, 5);
        cells.rerot_set(0, pending);
        g = cells.get_generation();
        // Pending cells are released (kept == 0) but exec-ref removal still bumps.
        assert(cells.rerot_freeze_to_archive(99, 0, 1) == 0);
        assert(cells.get_generation() != g);
        assert(cells.is_empty(0));
    }

    return 0;
}
