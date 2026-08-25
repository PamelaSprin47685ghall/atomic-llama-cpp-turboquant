#pragma once

#include "llama.h"

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

enum class server_token_dump_kind : uint32_t {
    completion = 1,
    infill     = 2,
    embedding  = 3,
    rerank     = 4,
};

// Persistent request-token corpus stored as a prefix tree.
//
// Node 0 is an implicit root. Every persisted node stores only
// (parent_node_id, token_id), i.e. the parent representation of a trie.
// Every accepted request stores its terminal node id separately, preserving
// duplicate requests without writing duplicate token prefixes.
class server_token_dump {
public:
    server_token_dump(const std::string & path, uint64_t records_per_shard);
    ~server_token_dump();

    server_token_dump(const server_token_dump &) = delete;
    server_token_dump & operator=(const server_token_dump &) = delete;

    bool append(
            const std::vector<llama_token> & tokens,
            server_token_dump_kind kind,
            std::string * error = nullptr);

    // Persisted trie nodes, excluding the implicit root node 0.
    uint64_t node_count() const;

    // Number of request/sample records persisted.
    uint64_t request_count() const;

private:
    struct node {
        uint64_t parent;
        llama_token token;
    };

    struct edge_key {
        uint64_t parent;
        llama_token token;

        bool operator==(const edge_key & other) const {
            return parent == other.parent && token == other.token;
        }
    };

    struct edge_hash {
        size_t operator()(const edge_key & key) const;
    };

    const std::string path_;
    const uint64_t records_per_shard_;

    mutable std::mutex mutex_;

    // nodes_[0] is the implicit root.
    std::vector<node> nodes_;
    std::unordered_map<edge_key, uint64_t, edge_hash> edge_index_;

    uint64_t request_count_ = 0;

    uint64_t node_shard_index_ = 0;
    uint64_t node_shard_records_ = 0;
    uint64_t request_shard_index_ = 0;
    uint64_t request_shard_records_ = 0;

    std::ofstream node_stream_;
    std::ofstream request_stream_;

    bool failed_ = false;
    std::string failure_message_;

    void load_existing();
    void write_manifest_if_missing() const;

    void ensure_node_stream();
    void ensure_request_stream();

    void write_node_record(uint64_t parent, llama_token token);
    void write_request_record(uint64_t leaf, uint32_t n_tokens, server_token_dump_kind kind);

    std::string node_shard_path(uint64_t index) const;
    std::string request_shard_path(uint64_t index) const;
};
