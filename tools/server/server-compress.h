#pragma once

#include "llama.h"
#include "server-compress-json.h"

#include <cstddef>
#include <string>
#include <vector>

// A server-private deletion plan. It owns the original HTTP bytes. Model work
// stays in the server queue; this class only maps tokens, ranks, and deletes.
class server_compress_plan {
public:
    struct text {
        std::vector<llama_token> tokens;
        std::vector<size_t> token_blocks;
        size_t message = 0;
    };

    server_compress_plan(std::string raw, const llama_vocab * vocab);

    const std::vector<text> & texts() const { return texts_; }
    bool needs_scores(size_t index) const;
    void add_scores(size_t text_index, size_t begin, const std::vector<float> & nll);

    // The minimum keeps instructions, the latest user and final group shells, and
    // each remaining text field's first and last blocks.
    std::string minimum() const;
    void rank();
    bool remove_tokens(size_t estimate);
    std::string render() const;
    size_t removed_messages() const;

private:
    struct block {
        server_compress_json::range source;
        size_t message = 0;
        size_t tokens = 0;
        double loss_sum = 0;
        size_t loss_count = 0;
        bool pinned = false;
        bool removed = false;
    };
    struct group {
        std::vector<size_t> messages;
        size_t estimated_tokens = 0;
        bool pinned = false;
        bool removed = false;
    };
    struct action {
        size_t index = 0;
        size_t tokens = 0;
        size_t order = 0;
        double score = 0;
        bool whole_group = false;
    };

    std::string raw_;
    server_compress_json::node root_;
    std::vector<size_t> message_groups_;
    std::vector<group> groups_;
    std::vector<text> texts_;
    std::vector<block> blocks_;
    std::vector<action> actions_;
    size_t next_action_ = 0;

    void add_text(const server_compress_json::node & value, size_t message,
                  const llama_vocab * vocab, const server_compress_json::node * outer = nullptr);
    void add_arguments(const server_compress_json::node & value, size_t message,
                       const llama_vocab * vocab, const server_compress_json::node & outer);
    std::string render(bool minimum) const;
};
