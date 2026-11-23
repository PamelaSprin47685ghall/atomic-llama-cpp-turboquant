#pragma once

#include "server-compress.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

struct server_compress_window {
    size_t text_index = 0;
    size_t begin = 0;
    std::vector<llama_token> tokens;
    // Scores from this offset to the end belong to the text at `begin`.
    // Earlier tokens supply context and must not be counted a second time.
    size_t score_offset = 0;
};

// Order natural JSON text blocks before applying the inference capacity limit.
// A reversed stream reverses block order only. Tokens inside each block retain
// their original order, including when one block needs several inference windows.
class server_compress_stream {
public:
    server_compress_stream(
            const std::vector<server_compress_plan::text> & texts,
            const std::vector<llama_token> & separator,
            size_t capacity,
            size_t overlap,
            llama_token bos,
            bool reverse) : bos_(bos) {
        const size_t bos_count = bos != LLAMA_TOKEN_NULL ? 1 : 0;
        if (capacity <= bos_count || overlap >= capacity - bos_count) {
            throw std::invalid_argument("compression scoring capacity must exceed its prefix and overlap");
        }
        capacity_ = capacity - bos_count;
        step_ = capacity_ - overlap;

        for (size_t i = 0; i < texts.size(); ++i) {
            const size_t index = reverse ? texts.size() - i - 1 : i;
            const auto & tokens = texts[index].tokens;
            if (tokens.empty()) {
                continue;
            }
            if (!stream_.empty()) {
                stream_.insert(stream_.end(), separator.begin(), separator.end());
            }
            const size_t begin = stream_.size();
            stream_.insert(stream_.end(), tokens.begin(), tokens.end());
            spans_.push_back({index, begin, stream_.size()});
        }
    }

    bool next(server_compress_window & window) {
        if (span_index_ == spans_.size()) {
            return false;
        }
        const auto & span = spans_[span_index_];
        const size_t target_begin = span.begin + local_begin_;
        const size_t target_size = std::min(step_, span.end - target_begin);
        const size_t prefix_size = std::min(target_begin, capacity_ - target_size);
        const size_t prefix_begin = target_begin - prefix_size;
        const size_t target_end = target_begin + target_size;

        window.text_index = span.text_index;
        window.begin = local_begin_;
        window.tokens.clear();
        if (bos_ != LLAMA_TOKEN_NULL) {
            window.tokens.push_back(bos_);
        }
        window.score_offset = window.tokens.size() + prefix_size;
        window.tokens.insert(window.tokens.end(), stream_.begin() + prefix_begin, stream_.begin() + target_end);

        // Match the local target ranges in both directions. Otherwise different
        // global window boundaries could create a score difference by changing
        // how much of the same block prefix was visible to the model.
        local_begin_ += target_size;
        if (target_end == span.end) {
            ++span_index_;
            local_begin_ = 0;
        }
        return true;
    }

private:
    struct span {
        size_t text_index;
        size_t begin;
        size_t end;
    };

    llama_token bos_;
    size_t capacity_ = 0;
    size_t step_ = 0;
    std::vector<llama_token> stream_;
    std::vector<span> spans_;
    size_t span_index_ = 0;
    size_t local_begin_ = 0;
};
