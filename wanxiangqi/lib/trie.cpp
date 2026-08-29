#include "trie.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace wxq {

std::vector<uint8_t> read_whole_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// 24 byte header: 8 byte magic, u32 version, u32 record size, 8 reserved
static void check_header(const std::vector<uint8_t> & bytes, const char * magic, uint32_t rec_size,
                         const std::string & path) {
    if (bytes.size() < 24 || std::memcmp(bytes.data(), magic, 8) != 0) {
        throw std::runtime_error("bad magic in " + path);
    }
    uint32_t version = 0;
    uint32_t stride  = 0;
    std::memcpy(&version, bytes.data() + 8,  sizeof(version));
    std::memcpy(&stride,  bytes.data() + 12, sizeof(stride));
    if (version != 1 || stride != rec_size) {
        throw std::runtime_error("unsupported version/record size in " + path);
    }
    if ((bytes.size() - 24) % rec_size != 0) {
        throw std::runtime_error("truncated records in " + path);
    }
}

std::vector<int32_t> trie::path_to(uint64_t node, size_t cap) const {
    std::vector<int32_t> path;
    for (uint64_t c = node; c != 0; c = nodes[c - 1].parent) {
        path.push_back(nodes[c - 1].token);
    }
    std::reverse(path.begin(), path.end());
    if (cap && path.size() > cap) {
        path.resize(cap);
    }
    return path;
}

std::vector<uint64_t> trie::depths() const {
    std::vector<uint64_t> depth(nodes.size() + 1, 0);
    for (size_t i = 0; i < nodes.size(); ++i) {
        depth[i + 1] = depth[nodes[i].parent] + 1;
    }
    return depth;
}

trie load_trie(const std::string & dir_str, int32_t n_vocab) {
    const std::filesystem::path dir(dir_str);
    const std::string nodes_path = (dir / "nodes-000000.bin").string();
    const std::string reqs_path  = (dir / "requests-000000.bin").string();

    const auto nodes_bytes = read_whole_file(nodes_path);
    const auto reqs_bytes  = read_whole_file(reqs_path);

    check_header(nodes_bytes, "CLTNOD01", 16, nodes_path);
    check_header(reqs_bytes,  "CLTREQ01", 24, reqs_path);

    trie t;

    const size_t n_nodes = (nodes_bytes.size() - 24) / 16;
    t.nodes.resize(n_nodes);
    for (size_t i = 0; i < n_nodes; ++i) {
        const uint8_t * rec = nodes_bytes.data() + 24 + i*16;
        std::memcpy(&t.nodes[i].parent, rec + 0, sizeof(uint64_t));
        std::memcpy(&t.nodes[i].token,  rec + 8, sizeof(int32_t));

        // node ids are 1-based, so a parent must be strictly smaller: this both
        // rejects cycles and lets every derived pass run in one forward sweep
        const uint64_t id = i + 1;
        if (t.nodes[i].parent >= id) {
            throw std::runtime_error("trie node " + std::to_string(id) + " has non-decreasing parent");
        }
        if (t.nodes[i].token < 0 || t.nodes[i].token >= n_vocab) {
            throw std::runtime_error("trie token out of vocabulary: " + std::to_string(t.nodes[i].token));
        }
    }

    t.children.resize(n_nodes + 1);
    for (size_t i = 0; i < n_nodes; ++i) {
        t.children[t.nodes[i].parent].push_back(i + 1);
    }

    const std::vector<uint64_t> depth = t.depths();

    const size_t n_req = (reqs_bytes.size() - 24) / 24;
    for (size_t i = 0; i < n_req; ++i) {
        const uint8_t * rec = reqs_bytes.data() + 24 + i*24;
        uint64_t leaf     = 0;
        uint32_t n_tokens = 0;
        std::memcpy(&leaf,     rec + 0, sizeof(leaf));
        std::memcpy(&n_tokens, rec + 8, sizeof(n_tokens));
        if (leaf == 0 || leaf > n_nodes) {
            throw std::runtime_error("trie request references missing leaf");
        }
        if (depth[leaf] != n_tokens) {
            throw std::runtime_error("trie leaf depth does not match recorded token count");
        }
        t.leaves.push_back(leaf);
        t.n_leaf_positions += n_tokens;
    }

    return t;
}

} // namespace wxq
