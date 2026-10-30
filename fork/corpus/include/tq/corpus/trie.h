#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace tq::corpus {

struct trie_node {
    std::uint64_t parent = 0;
    std::int32_t token = 0;
};

// Values match the original request-token corpus version 1 format.
enum class request_kind : std::uint32_t {
    completion = 1,
    infill = 2,
    embedding = 3,
    rerank = 4,
};

struct request_metadata {
    request_kind kind = request_kind::completion;
    // Milliseconds since the Unix epoch; zero means capture time was not recorded.
    std::uint64_t timestamp_ms = 0;
};

// Node 0 is the root. It has no token.
// All returned values own their data.
class token_trie {
public:
    explicit token_trie(std::int32_t vocabulary_size);

    static token_trie load(const std::filesystem::path & directory,
                           std::int32_t vocabulary_size);

    // The one-argument overload records a completion with an unknown timestamp.
    std::uint64_t append(const std::vector<std::int32_t> & tokens);
    std::uint64_t append(const std::vector<std::int32_t> & tokens,
                         request_metadata metadata);

    // Request ids are zero-based. The returned metadata is an owned value.
    request_metadata request_metadata_at(std::uint64_t request_id) const;

    // The target directory must not exist.
    void save(const std::filesystem::path & directory) const;

    trie_node node(std::uint64_t id) const;
    std::vector<std::uint64_t> children(std::uint64_t id) const;
    std::vector<std::uint64_t> request_leaves() const;
    std::vector<std::int32_t> path_to(std::uint64_t id) const;

    std::uint64_t node_count() const noexcept;
    std::uint64_t request_count() const noexcept;
    std::uint64_t leaf_positions() const noexcept;

private:
    std::int32_t vocabulary_size_;
    std::vector<trie_node> nodes_;
    std::vector<std::vector<std::uint64_t>> children_;
    std::vector<std::uint64_t> depths_;
    std::vector<std::uint64_t> leaves_;
    std::vector<request_metadata> request_metadata_;
    std::uint64_t leaf_positions_ = 0;

    void check_id(std::uint64_t id, bool allow_root) const;
};

} // namespace tq::corpus
