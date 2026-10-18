// Compile against the UNMODIFIED archived server-corpus-dump.{h,cpp}, the
// installed upstream llama headers, and the independent TQCorpus library.
// There are no mocked loader functions or private-state access tricks.
#include "server-corpus-dump.h"
#include <tq/corpus/trie.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::vector<char> read(const fs::path & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot read fixture " + path.string());
    }
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

static void check(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static void verify_legacy_read_only(const fs::path & directory,
                                    std::uint64_t nodes, std::uint64_t requests) {
    // The production constructor writes a missing format.json and repairs torn
    // tails. These are complete test files with an existing production manifest.
    // Compare every fixture file before/after to prove this verification is read-only.
    const auto before_nodes = read(directory / "nodes-000000.bin");
    const auto before_requests = read(directory / "requests-000000.bin");
    const auto before_manifest = read(directory / "format.json");
    {
        server_token_dump production_reader(directory.string(), 1000000);
        check(production_reader.node_count() == nodes, "production reader node mismatch");
        check(production_reader.request_count() == requests, "production reader request mismatch");
    }
    check(read(directory / "nodes-000000.bin") == before_nodes &&
          read(directory / "requests-000000.bin") == before_requests &&
          read(directory / "format.json") == before_manifest,
          "production reader modified the corpus during verification");
}

int main(int argc, char ** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("usage: corpus-production-interop NEW_FIXTURE_DIRECTORY");
        }
        const fs::path root(argv[1]);
        check(fs::create_directory(root), "fixture target must not already exist");
        const auto original = root / "production";
        {
            server_token_dump writer(original.string(), 1000000);
            std::string error;
            check(writer.append({0x1234, 0x2345}, server_token_dump_kind::completion, &error),
                  "production completion append failed");
            check(writer.append({0x1234, 0x2345}, server_token_dump_kind::infill, &error),
                  "production infill append failed");
            check(writer.append({0x1234, 0x2345}, server_token_dump_kind::embedding, &error),
                  "production embedding append failed");
            check(writer.append({0x1234}, server_token_dump_kind::rerank, &error),
                  "production rerank append failed");
            check(writer.append({}, server_token_dump_kind::embedding, &error),
                  "production empty request append failed");
        }

        auto corpus = tq::corpus::token_trie::load(original, 10000);
        check(corpus.node_count() == 2 && corpus.request_count() == 5 &&
              corpus.leaf_positions() == 7 && corpus.request_leaves().back() == 0,
              "independent corpus changed production paths/counts");
        const std::uint32_t kinds[] = {1, 2, 3, 4, 3};
        for (std::uint64_t i = 0; i < corpus.request_count(); ++i) {
            const auto metadata = corpus.request_metadata_at(i);
            check(static_cast<std::uint32_t>(metadata.kind) == kinds[i] && metadata.timestamp_ms > 0,
                  "independent corpus changed captured metadata");
        }
        const auto copy = root / "resaved";
        corpus.save(copy);
        fs::copy_file(original / "format.json", copy / "format.json");
        check(read(original / "nodes-000000.bin") == read(copy / "nodes-000000.bin") &&
              read(original / "requests-000000.bin") == read(copy / "requests-000000.bin"),
              "production corpus did not round trip byte for byte");
        verify_legacy_read_only(copy, 2, 5);

        corpus.append({0x2345});
        corpus.append({});
        const auto appended = root / "appended";
        corpus.save(appended);
        fs::copy_file(original / "format.json", appended / "format.json");
        verify_legacy_read_only(appended, 3, 7);
        check(corpus.request_metadata_at(5).kind == tq::corpus::request_kind::completion &&
              corpus.request_metadata_at(5).timestamp_ms == 0,
              "default append metadata is not explicit completion/unknown-time");
        const auto empty_only = root / "production-empty-only";
        {
            server_token_dump writer(empty_only.string(), 1000000);
            check(writer.append({}, server_token_dump_kind::rerank),
                  "production empty-only append failed");
        }
        check(!fs::exists(empty_only / "nodes-000000.bin"),
              "production writer unexpectedly created an empty nodes stream");
        const auto empty_corpus = tq::corpus::token_trie::load(empty_only, 10000);
        check(empty_corpus.request_count() == 1 && empty_corpus.node_count() == 0 &&
              empty_corpus.request_leaves() == std::vector<std::uint64_t>{0} &&
              empty_corpus.request_metadata_at(0).kind == tq::corpus::request_kind::rerank,
              "production empty-only corpus was not preserved");
        const auto empty_copy = root / "empty-only-resaved";
        empty_corpus.save(empty_copy);
        fs::copy_file(empty_only / "format.json", empty_copy / "format.json");
        check(read(empty_only / "requests-000000.bin") == read(empty_copy / "requests-000000.bin"),
              "production empty-only request metadata changed");
        verify_legacy_read_only(empty_copy, 0, 1);
        std::cout << "actual production writer -> TQCorpus -> actual production reader passed; "
                     "all kinds, timestamps, duplicates, empty/prefix requests; reader unchanged files\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
