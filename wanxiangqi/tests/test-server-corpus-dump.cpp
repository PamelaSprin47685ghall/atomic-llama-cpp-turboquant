#include "server-corpus-dump.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#define REQUIRE(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "requirement failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        return 1; \
    } \
} while (0)

int main() {
    const auto unique = std::to_string(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const auto dir = std::filesystem::temp_directory_path() / ("wanxiangshu-corpus-dump-" + unique);
    std::filesystem::remove_all(dir);

    {
        server_token_dump dump(dir.string(), 2);
        std::string error;

        REQUIRE(dump.append({1, 2, 3}, server_token_dump_kind::completion, &error));
        REQUIRE(dump.append({1, 2, 4}, server_token_dump_kind::completion, &error));
        REQUIRE(dump.append({1, 2, 3}, server_token_dump_kind::infill, &error));

        // Four unique trie nodes: 1 -> 2 -> {3,4}; duplicate request is only
        // another leaf record.
        REQUIRE(dump.node_count() == 4);
        REQUIRE(dump.request_count() == 3);

        // records_per_shard=2 forces both node and request multi-file storage.
        REQUIRE(std::filesystem::exists(dir / "nodes-000000.bin"));
        REQUIRE(std::filesystem::exists(dir / "nodes-000001.bin"));
        REQUIRE(std::filesystem::exists(dir / "requests-000000.bin"));
        REQUIRE(std::filesystem::exists(dir / "requests-000001.bin"));
        REQUIRE(std::filesystem::exists(dir / "format.json"));
    }

    // Simulate a process dying in the middle of the final fixed-size record.
    // Only the trailing shard is repairable; complete earlier shards remain
    // strict corruption boundaries.
    {
        std::ofstream nodes(dir / "nodes-000001.bin", std::ios::binary | std::ios::app);
        nodes.write("bad", 3);
        std::ofstream requests(dir / "requests-000001.bin", std::ios::binary | std::ios::app);
        requests.write("bad", 3);
    }

    // Reopen and rebuild the in-memory edge index from parent records.
    {
        server_token_dump dump(dir.string(), 2);
        std::string error;
        REQUIRE(dump.node_count() == 4);
        REQUIRE(dump.request_count() == 3);

        REQUIRE(dump.append({1, 5}, server_token_dump_kind::embedding, &error));
        REQUIRE(dump.node_count() == 5);
        REQUIRE(dump.request_count() == 4);
    }

    std::filesystem::remove_all(dir);
    return 0;
}
