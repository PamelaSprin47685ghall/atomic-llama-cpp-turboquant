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

    return 0;
}
