#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace tq::corpus {

struct trie_node {
    std::uint64_t parent = 0;
    std::int32_t token = 0;
};

// Node 0 is the root. It has no token.
// All returned values own their data.
class token_trie {
public:
    explicit token_trie(std::int32_t vocabulary_size);

    static token_trie load(const std::filesystem::path & directory,
                           std::int32_t vocabulary_size);

    std::uint64_t append(const std::vector<std::int32_t> & tokens);

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
    std::uint64_t leaf_positions_ = 0;

    void check_id(std::uint64_t id, bool allow_root) const;
};

} // namespace tq::corpus
