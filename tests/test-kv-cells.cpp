// This test uses assert() as its failure mechanism. Keep those checks active
// when the surrounding project is configured with CMAKE_BUILD_TYPE=Release.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../src/llama-kv-cells.h"
#include <cassert>
#include <cstdint>
#include <vector>

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

    // Test 8: Stable payload ID and storage generation lifecycle
    {
        llama_kv_cells cells;
        cells.resize(10);

        // Empty cells must have 0 payload_id and 0 generation
        for (uint32_t i = 0; i < 10; ++i) {
            assert(cells.is_empty(i));
            assert(cells.payload_id_get(i) == 0);
            assert(cells.storage_generation_get(i) == 0);
        }

        // Nonzero unique ID allocated on occupancy, generation initialized to 1
        cells.pos_set(1, 100);
        cells.seq_add(1, 0);
        const uint64_t pid1 = cells.payload_id_get(1);
        const uint64_t gen1 = cells.storage_generation_get(1);
        assert(pid1 != 0);
        assert(gen1 == 1);

        cells.pos_set(3, 101);
        cells.seq_add(3, 0);
        const uint64_t pid3 = cells.payload_id_get(3);
        const uint64_t gen3 = cells.storage_generation_get(3);
        assert(pid3 != 0);
        assert(gen3 == 1);
        assert(pid1 != pid3);

        // Same-cell seq_cp sharing: adding another sequence to an occupied cell must not change payload_id or generation
        cells.seq_add(1, 1);
        assert(cells.payload_id_get(1) == pid1);
        assert(cells.storage_generation_get(1) == gen1);

        // Storage generation increment preserved
        cells.storage_generation_inc(1);
        assert(cells.storage_generation_get(1) == gen1 + 1);
        const uint64_t gen1_inc = cells.storage_generation_get(1);

        // Exact preservation through cp / set
        const auto cp1 = cells.cp(1, 1);
        assert(cp1.payload_id_get(0) == pid1);
        assert(cp1.storage_generation_get(0) == gen1_inc);

        // Exact preservation through compaction (make_pack_plan / apply_pack)
        cells.pos_set(7, 102);
        cells.seq_add(7, 2);
        const uint64_t pid7 = cells.payload_id_get(7);
        const uint64_t gen7 = cells.storage_generation_get(7);
        assert(pid7 != 0 && pid7 != pid1 && pid7 != pid3);

        auto plan = cells.make_pack_plan();
        assert(plan.retained_count == 3);
        cells.apply_pack(plan);

        // Cells [1, 3, 7] packed to [0, 1, 2]
        assert(cells.get_used() == 3);
        assert(cells.payload_id_get(0) == pid1);
        assert(cells.storage_generation_get(0) == gen1_inc);
        assert(cells.payload_id_get(1) == pid3);
        assert(cells.storage_generation_get(1) == gen3);
        assert(cells.payload_id_get(2) == pid7);
        assert(cells.storage_generation_get(2) == gen7);
        for (uint32_t i = 3; i < 10; ++i) {
            assert(cells.is_empty(i));
            assert(cells.payload_id_get(i) == 0);
            assert(cells.storage_generation_get(i) == 0);
        }

        // Empty-only reset: removing sequence without making cell empty keeps payload_id and generation
        assert(!cells.seq_rm(0, 0)); // seq 1 remains on cell 0
        assert(cells.payload_id_get(0) == pid1);
        assert(cells.storage_generation_get(0) == gen1_inc);

        // Removing last sequence empties cell: resets payload_id and generation to 0
        assert(cells.seq_rm(0, 1));
        assert(cells.is_empty(0));
        assert(cells.payload_id_get(0) == 0);
        assert(cells.storage_generation_get(0) == 0);

        // seq_keep that retains sequence preserves payload_id and generation
        assert(!cells.seq_keep(1, 0));
        assert(cells.payload_id_get(1) == pid3);
        assert(cells.storage_generation_get(1) == gen3);

        // seq_keep that empties cell resets payload_id and generation to 0
        assert(cells.seq_keep(2, 5)); // seq 5 not present in cell 2 (which had seq 2)
        assert(cells.is_empty(2));
        assert(cells.payload_id_get(2) == 0);
        assert(cells.storage_generation_get(2) == 0);

        // Explicit rm resets to 0
        cells.rm(1);
        assert(cells.is_empty(1));
        assert(cells.payload_id_get(1) == 0);
        assert(cells.storage_generation_get(1) == 0);

        // New content allocated in previously cleared cell gets a fresh new ID
        cells.pos_set(0, 200);
        cells.seq_add(0, 0);
        const uint64_t pid_new = cells.payload_id_get(0);
        assert(pid_new != 0);
        assert(pid_new != pid1);
        assert(cells.storage_generation_get(0) == 1);

        // Full reset resets all payload_ids and generations
        cells.reset();
        for (uint32_t i = 0; i < 10; ++i) {
            assert(cells.is_empty(i));
            assert(cells.payload_id_get(i) == 0);
            assert(cells.storage_generation_get(i) == 0);
        }
    }

    // Test 9: Same-stream vs distinct cross-stream copy identity semantics
    {
        llama_kv_cells src_cells;
        src_cells.resize(4);
        src_cells.pos_set(0, 10);
        src_cells.seq_add(0, 0);
        const uint64_t pid0 = src_cells.payload_id_get(0);
        const uint64_t gen0 = src_cells.storage_generation_get(0);
        assert(pid0 != 0);

        // Same-stream reference addition: payload_id and storage_generation unchanged
        src_cells.seq_add(0, 1);
        assert(src_cells.payload_id_get(0) == pid0);
        assert(src_cells.storage_generation_get(0) == gen0);

        // Cross-stream distinct physical copy: fresh allocation via pos_set gives distinct payload_id
        llama_kv_cells dst_cells;
        dst_cells.resize(4);
        dst_cells.pos_set(0, src_cells.pos_get(0));
        dst_cells.seq_add(0, 2);
        const uint64_t pid_dst = dst_cells.payload_id_get(0);
        assert(pid_dst != 0);
        assert(pid_dst != pid0); // Fresh identity for distinct physical copy
    }

// =========================================================================
    // Phase 7: Shared-prefix physical cell union stress test (§A.24, Gate 24 of DoD A.30)
    // 8K common prefix ├ A ├ B └ C
    // Verifies 3 layers of sharing:
    //   1. chat/prompt shared prefix across multiple outer sequences
    //   2. outer n_cmpl sharing
    //   3. inner RERoT fork ancestry & child lanes
    // Invariants:
    //   - references_removed and physical_freed are strictly separated metrics
    //   - Removing any reference NEVER prematurely frees a physical cell still referenced by others
    //   - Compaction (pack) preserves multi-sequence sharing and exact positions
    // =========================================================================
    {
        llama_kv_cells cells;
        const uint32_t prefix_tokens = 8192; // 8K common prefix
        const uint32_t total_capacity = prefix_tokens + 1024;
        cells.resize(total_capacity);

        // Sequence IDs:
        // seq 0: common base prompt
        // seq 1: outer completion A
        // seq 2: outer completion B
        // seq 3: outer completion C
        // seq 4: inner RERoT child lane 1 (forked from A)
        // seq 5: inner RERoT child lane 2 (forked from A)
        // seq 6: inner RERoT archive handle for A
        const llama_seq_id seq_base = 0;
        const llama_seq_id seq_cmpl_a = 1;
        const llama_seq_id seq_cmpl_b = 2;
        const llama_seq_id seq_cmpl_c = 3;
        const llama_seq_id seq_child_1 = 4;
        const llama_seq_id seq_child_2 = 5;
        const llama_seq_id seq_archive = 6;

        // 1. Populate 8K common prefix shared across base, A, B, C, child1, child2, archive
        for (uint32_t i = 0; i < prefix_tokens; ++i) {
            cells.pos_set(i, (llama_pos)i);
            cells.seq_add(i, seq_base);
            cells.seq_add(i, seq_cmpl_a);
            cells.seq_add(i, seq_cmpl_b);
            cells.seq_add(i, seq_cmpl_c);
            cells.seq_add(i, seq_child_1);
            cells.seq_add(i, seq_child_2);
            cells.seq_add(i, seq_archive);
        }

        assert(cells.get_used() == prefix_tokens);
        assert(cells.seq_get_used(seq_base) == prefix_tokens);
        assert(cells.seq_get_used(seq_cmpl_a) == prefix_tokens);
        assert(cells.seq_get_used(seq_cmpl_b) == prefix_tokens);
        assert(cells.seq_get_used(seq_cmpl_c) == prefix_tokens);
        assert(cells.seq_get_used(seq_child_1) == prefix_tokens);
        assert(cells.seq_get_used(seq_child_2) == prefix_tokens);
        assert(cells.seq_get_used(seq_archive) == prefix_tokens);

        // 2. Populate distinct suffixes:
        // A has tokens at 8192..8223 (32 tokens)
        for (uint32_t i = prefix_tokens; i < prefix_tokens + 32; ++i) {
            cells.pos_set(i, (llama_pos)i);
            cells.seq_add(i, seq_cmpl_a);
            cells.seq_add(i, seq_child_1); // Child 1 shares A's early suffix
        }
        // B has tokens at 8224..8255 (32 tokens)
        for (uint32_t i = prefix_tokens + 32; i < prefix_tokens + 64; ++i) {
            cells.pos_set(i, (llama_pos)i);
            cells.seq_add(i, seq_cmpl_b);
        }
        // C has tokens at 8256..8287 (32 tokens)
        for (uint32_t i = prefix_tokens + 64; i < prefix_tokens + 96; ++i) {
            cells.pos_set(i, (llama_pos)i);
            cells.seq_add(i, seq_cmpl_c);
        }

        const uint32_t total_used = prefix_tokens + 96;
        assert(cells.get_used() == total_used);

        // 3. Remove references from seq_child_1 on the 8K shared prefix.
        // Invariant (§A.24):
        // references_removed = 8192, but physical_freed = 0 because all 8192 cells
        // are still referenced by seq_base, seq_cmpl_a, seq_cmpl_b, seq_cmpl_c, etc.!
        uint32_t references_removed = 0;
        uint32_t physical_before = cells.get_used();
        for (uint32_t i = 0; i < prefix_tokens; ++i) {
            if (cells.seq_has(i, seq_child_1)) {
                cells.seq_rm(i, seq_child_1);
                ++references_removed;
            }
        }
        uint32_t physical_after = cells.get_used();
        uint32_t physical_freed = physical_before - physical_after;
        assert(references_removed == prefix_tokens);
        assert(physical_freed == 0); // ZERO physical cells freed!
        assert(cells.get_used() == total_used);
        assert(cells.seq_get_used(seq_child_1) == 32); // Still has its 32 suffix tokens

        // 4. Remove seq_child_2, seq_cmpl_a, seq_cmpl_b, seq_archive references on the prefix
        for (uint32_t i = 0; i < prefix_tokens; ++i) {
            cells.seq_rm(i, seq_child_2);
            cells.seq_rm(i, seq_cmpl_a);
            cells.seq_rm(i, seq_cmpl_b);
            cells.seq_rm(i, seq_archive);
        }
        // Still referenced by seq_base and seq_cmpl_c:
        assert(cells.get_used() == total_used);

        // 5. Remove seq_cmpl_c from prefix: seq_base remains the sole owner. Physical freed still 0!
        for (uint32_t i = 0; i < prefix_tokens; ++i) {
            cells.seq_rm(i, seq_cmpl_c);
        }
        assert(cells.get_used() == total_used);

        // 6. Finally remove seq_base from first 1000 cells of the prefix.
        // Now no sequence references cells 0..999.
        // Invariant: exactly 1000 cells are physically freed!
        physical_before = cells.get_used();
        references_removed = 0;
        for (uint32_t i = 0; i < 1000; ++i) {
            if (cells.seq_has(i, seq_base)) {
                cells.seq_rm(i, seq_base);
                ++references_removed;
            }
        }
        physical_after = cells.get_used();
        physical_freed = physical_before - physical_after;
        assert(references_removed == 1000);
        assert(physical_freed == 1000); // Now physically freed!
        assert(cells.get_used() == total_used - 1000);

        // 7. Compaction of sparse physical cells under multi-branch sharing
        auto pack_plan = cells.make_pack_plan();
        assert(pack_plan.retained_count == total_used - 1000);
        cells.apply_pack(pack_plan);

        assert(cells.get_used() == total_used - 1000);
        assert(cells.used_min() == 0);
        assert(cells.used_max_p1() == total_used - 1000);

        // Suffix tokens of B and C are still completely intact and correct
        assert(cells.seq_get_used(seq_cmpl_b) == 32);
        assert(cells.seq_get_used(seq_cmpl_c) == 32);
    }

    // Test 10: reserve_payload_ids_through preflight monotonicity and fail-closed saturation
    {
        llama_kv_cells c;
        c.resize(4);
        c.pos_set(0, 1);
        const uint64_t pid_initial = c.payload_id_get(0);
        assert(pid_initial > 0);

        // Advancing past a higher target: next allocation must strictly exceed target
        const uint64_t target_pid = pid_initial + 1000;
        assert(llama_kv_cells::reserve_payload_ids_through(target_pid));
        c.pos_set(1, 2);
        const uint64_t pid_after_reserve = c.payload_id_get(1);
        assert(pid_after_reserve > target_pid);

        // Monotonicity: reserving a smaller value than current counter succeeds without decrementing
        assert(llama_kv_cells::reserve_payload_ids_through(pid_initial));
        c.pos_set(2, 3);
        const uint64_t pid_after_lower = c.payload_id_get(2);
        assert(pid_after_lower > pid_after_reserve);

        // Fail-closed at UINT64_MAX: cannot wrap, returns false
        assert(!llama_kv_cells::reserve_payload_ids_through(UINT64_MAX));
    }

    // Test 11: RERoT freeze to archive and shared reference union without double importance
    {
        llama_kv_cells c;
        c.resize(8);

        const uint64_t ep = 42;
        const llama_seq_id exec_seq = 0;
        const llama_seq_id archive_seq = 1;
        const llama_seq_id foreign_seq = 2;

        // Cell 0: PUBLIC_LIVE in exec_seq -> should be archived (gains archive_seq, loses exec_seq)
        c.pos_set(0, 10);
        c.seq_add(0, exec_seq);
        c.rerot_set(0, make_rerot_meta(ep, 1, 101, llama_rerot_visibility::pending_record, 0, 10));
        assert(c.rerot_publish(0, ep, 101, 5));

        // Cell 1: PRIVATE_CONTROL in exec_seq -> should NOT be archived; exec_seq removed -> becomes empty
        c.pos_set(1, 11);
        c.seq_add(1, exec_seq);
        c.rerot_set(1, make_rerot_meta(ep, 2, 102, llama_rerot_visibility::private_control, 0, 11));

        // Cell 2: PENDING_RECORD in exec_seq -> should NOT be archived; exec_seq removed -> becomes empty
        c.pos_set(2, 12);
        c.seq_add(2, exec_seq);
        c.rerot_set(2, make_rerot_meta(ep, 3, 103, llama_rerot_visibility::pending_record, 0, 12));

        // Cell 3: PUBLIC_LIVE already shared with archive_seq -> gains no duplicate reference, loses exec_seq, keeps archive_seq
        c.pos_set(3, 13);
        c.seq_add(3, exec_seq);
        c.seq_add(3, archive_seq);
        c.rerot_set(3, make_rerot_meta(ep, 4, 104, llama_rerot_visibility::pending_record, 0, 13));
        assert(c.rerot_publish(3, ep, 104, 5));

        // Cell 4: foreign sequence cell -> completely untouched
        c.pos_set(4, 14);
        c.seq_add(4, foreign_seq);

        assert(c.seq_get_used(exec_seq) == 4);
        assert(c.seq_get_used(archive_seq) == 1);
        assert(c.seq_get_used(foreign_seq) == 1);

        // Physical run lookup before freeze
        std::vector<uint32_t> run_cells;
        assert(c.rerot_collect_run(ep, 101, run_cells) == 1);
        assert(run_cells[0] == 0);
        run_cells.clear();
        assert(c.rerot_collect_run(0, 101, run_cells) == 0); // ep=0 returns 0

        // Freeze exec_seq to archive_seq
        size_t kept = c.rerot_freeze_to_archive(ep, exec_seq, archive_seq);
        assert(kept == 2); // Cell 0 and Cell 3

        // Exec seq has 0 cells left
        assert(c.seq_get_used(exec_seq) == 0);

        // Cell 0 survived under archive_seq
        assert(!c.is_empty(0));
        assert(c.seq_has(0, archive_seq));
        assert(!c.seq_has(0, exec_seq));
        assert(c.seq_count(0) == 1);
        assert(c.rerot_get(0).visibility == llama_rerot_visibility::public_live);

        // Cells 1 and 2 emptied (private and pending dropped)
        assert(c.is_empty(1));
        assert(c.payload_id_get(1) == 0);
        assert(c.is_empty(2));
        assert(c.payload_id_get(2) == 0);

        // Cell 3 survived under archive_seq without double importance
        assert(!c.is_empty(3));
        assert(c.seq_has(3, archive_seq));
        assert(!c.seq_has(3, exec_seq));
        assert(c.seq_count(3) == 1); // No double count for archive_seq

        // Cell 4 unaffected
        assert(!c.is_empty(4));
        assert(c.seq_has(4, foreign_seq));

        assert(c.seq_get_used(archive_seq) == 2);
        assert(c.seq_get_used(foreign_seq) == 1);
    }

    // Test 12: Exact multi-ref sequence snapshot/restore for overwrite-victim lifecycle
    {
        llama_kv_cells c;
        c.resize(6);

        c.pos_set(1, 50);
        c.seq_add(1, 0);
        c.seq_add(1, 2);
        c.seq_add(1, 5);

        assert(c.seq_get_used(0) == 1);
        assert(c.seq_get_used(2) == 1);
        assert(c.seq_get_used(5) == 1);

        const auto snap = c.seq_snapshot(1);
        assert(snap.count() == 3);
        assert(snap.test(0) && snap.test(2) && snap.test(5));

        // Remove cell 1
        c.rm(1);
        assert(c.is_empty(1));
        assert(c.seq_get_used(0) == 0);
        assert(c.seq_get_used(2) == 0);
        assert(c.seq_get_used(5) == 0);

        // Restore onto cell 3 with pos_set then seq_restore
        c.pos_set(3, 50);
        c.seq_restore(3, snap);

        assert(!c.is_empty(3));
        assert(c.seq_count(3) == 3);
        assert(c.seq_has(3, 0) && c.seq_has(3, 2) && c.seq_has(3, 5));
        assert(c.seq_get_used(0) == 1);
        assert(c.seq_get_used(2) == 1);
        assert(c.seq_get_used(5) == 1);
        assert(c.seq_pos_min(0) == 50 && c.seq_pos_max(0) == 50);
        assert(c.seq_pos_min(2) == 50 && c.seq_pos_max(2) == 50);
        assert(c.seq_pos_min(5) == 50 && c.seq_pos_max(5) == 50);
    }

    // Test 13: One-row boundary and non-divisible compaction preservation
    {
        llama_kv_cells c;
        c.resize(17); // prime size / non-divisible

        // Single isolated row at the very end
        c.pos_set(16, 999);
        c.seq_add(16, 3);
        const uint64_t single_pid = c.payload_id_get(16);
        const uint64_t single_gen = c.storage_generation_get(16);

        auto plan = c.make_pack_plan();
        assert(plan.retained_count == 1);
        assert(plan.moves.size() == 1);
        assert(plan.moves[0].src_begin == 16);
        assert(plan.moves[0].dst_begin == 0);
        assert(plan.moves[0].length == 1);

        c.apply_pack(plan);
        assert(c.get_used() == 1);
        assert(c.used_min() == 0);
        assert(c.used_max_p1() == 1);
        assert(!c.is_empty(0));
        assert(c.pos_get(0) == 999);
        assert(c.seq_has(0, 3));
        assert(c.payload_id_get(0) == single_pid);
        assert(c.storage_generation_get(0) == single_gen);
        for (uint32_t i = 1; i < 17; ++i) {
            assert(c.is_empty(i));
            assert(c.payload_id_get(i) == 0);
        }
    }

    return 0;
}
