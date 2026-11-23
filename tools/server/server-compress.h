#pragma once

#include "llama.h"
#include "server-compress-json.h"

#include <cstddef>
#include <string>
#include <vector>

// A server-private edit plan. It owns the original HTTP bytes. Model work stays
// in the server queue; this class maps scores to source spans and marks deletions.
class server_compress_plan {
public:
    // One native content item, reasoning field, or complete tool call. These are
    // the units of scoring order, never fixed-size token chunks.
    struct text {
        std::vector<llama_token> tokens;
        std::vector<size_t> token_units;
        std::vector<float> forward_nll;
        std::vector<float> reverse_nll;
        size_t message = 0;
    };

    server_compress_plan(std::string raw, const llama_vocab * vocab);

    const std::vector<text> & texts() const { return texts_; }
    void add_scores(size_t text_index, size_t begin, const std::vector<float> & nll, bool reverse);

    // The minimum keeps instructions, the latest user and final group shells, and
    // each remaining text field's protected prefix and suffix.
    server_compress_json::deletion_result minimum() const;
    void rank();
    bool remove_tokens(size_t estimate);
    server_compress_json::deletion_result render() const;
    size_t removed_messages() const;

private:
    // The smallest source span bounded by both tokenizer and Unicode/escape
    // boundaries. Adjacent removed units render as a single deletion marker.
    struct unit {
        server_compress_json::range source;
        size_t message = 0;
        size_t tokens = 0;
        bool pinned = false;
        bool removed = false;
    };
    struct group {
        std::vector<size_t> messages;
        size_t estimated_tokens = 0;
        size_t erased_tokens = 0;
        size_t markers = 0;
        bool pinned = false;
        bool removed = false;
    };
    struct action {
        size_t index = 0;
        size_t order = 0;
        double score = 0;
        double forward = 0;
        bool whole_group = false;
    };

    std::string raw_;
    server_compress_json::node root_;
    std::vector<size_t> message_groups_;
    std::vector<group> groups_;
    std::vector<text> texts_;
    std::vector<unit> units_;
    std::vector<action> actions_;
    size_t next_action_ = 0;
    size_t marker_tokens_ = 0;

    struct mapped_unit {
        server_compress_json::range text;
        server_compress_json::range source;
    };
    using region = std::vector<mapped_unit>;
    void add_text(const server_compress_json::node & value, size_t message,
                  const llama_vocab * vocab, bool pinned);
    void add_function(const server_compress_json::node & fn, size_t message, const llama_vocab * vocab);
    void add_content(const std::string & input, const std::vector<region> & regions,
                     size_t message, const llama_vocab * vocab, bool pinned);
    server_compress_json::deletion_result render(bool minimum) const;
};
