#include "server-triattention.h"

#include <cstdio>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main() {
    CHECK(server_triattention_compressed_after_load(true, true, false, 0, 192));
    CHECK(server_triattention_compressed_after_load(true, false, false, 0, 192));
    CHECK(server_triattention_compressed_after_load(false, true, true, 48, 192));
    CHECK(!server_triattention_compressed_after_load(true, true, true, 192, 192));
    int target_calls = 0, draft_calls = 0;
    auto target = [&]() { ++target_calls; llama_memory_kv_reclaim_result r{}; r.supported = true; return r; };
    auto draft = [&]() { ++draft_calls; llama_memory_kv_reclaim_result r{}; r.supported = r.changed = true; r.physical_freed = 144; return r; };
    auto result = server_triattention_reclaim_pair(true, target, draft);
    CHECK(target_calls == 1 && draft_calls == 1);
    CHECK(!result.first.changed && result.second.changed && result.second.physical_freed == 144);
    server_triattention_reclaim_pair(false, target, draft);
    CHECK(target_calls == 2 && draft_calls == 1);
    std::fprintf(stderr, "PASS: server TriAttention state transitions\n");
    return 0;
}
