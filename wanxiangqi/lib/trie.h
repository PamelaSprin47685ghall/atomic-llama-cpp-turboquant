// Request-token prefix trie (format: cachyllama-request-token-trie v1).
//
// Written by `llama-server --run-dump`, see wanxiangqi/docs/request-corpus-dump.md.
// The committed population lives in wanxiangqi/calibration/.
//
// Node ids are 1-based; id 0 is the implicit root and stores no token. Parents
// are strictly smaller than their children, so every derived pass (depth,
// children, subtree sizes) runs in one forward sweep.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace wxq {

struct trie_node {
    uint64_t parent = 0;
    int32_t  token  = 0;
};

struct trie {
    // nodes[i] describes node id i+1
    std::vector<trie_node>             nodes;
    std::vector<std::vector<uint64_t>> children;  // indexed by node id, 0 = root
    std::vector<uint64_t>              leaves;    // one per request record
    uint64_t                           n_leaf_positions = 0;

    uint64_t n_nodes() const { return nodes.size(); }

    // root -> node token path. cap 0 means no limit.
    std::vector<int32_t> path_to(uint64_t node, size_t cap = 0) const;

    // depth[id], depth[0] == 0
    std::vector<uint64_t> depths() const;
};

// Throws std::runtime_error on a malformed or truncated dump. n_vocab bounds the
// token ids: a trie captured with a different tokenizer must not silently load.
trie load_trie(const std::string & dir, int32_t n_vocab);

// Whole-file read, throws std::runtime_error if the path cannot be opened.
std::vector<uint8_t> read_whole_file(const std::string & path);

} // namespace wxq
