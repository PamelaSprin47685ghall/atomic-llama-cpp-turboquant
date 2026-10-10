#include "ggml-vulkan-tp5-sparse-timing.h"
#include <cassert>
#include <cstdio>

int main() {
    const auto dense = vk_tp5_sparse_timing::parse(nullptr);
    const auto span = vk_tp5_sparse_timing::parse("span");
    unsigned count = 0;
    for (unsigned q = 0; q < 483; ++q) { assert(dense.query(q,96)); count += span.query(q,96); }
    assert(count == 2 && !span.query(483,96));
    for (const char * bad : {"", "0:0", "8:8", "-1:0", "8:0junk", "8:", "98:0", "1:0:0"})
        assert(!vk_tp5_sparse_timing::parse(bad).valid);
    for (unsigned stage = 0; stage <= 96; ++stage) {
        unsigned cover = 0;
        for (unsigned residue = 0; residue < 8; ++residue)
            cover += vk_tp5_sparse_timing::parse(("8:" + std::to_string(residue)).c_str()).stage(stage);
        assert(cover == 1);
    }
    std::puts("Sparse queries partition 97 stages; missing queries excluded; invalid syntax rejected.");
}
