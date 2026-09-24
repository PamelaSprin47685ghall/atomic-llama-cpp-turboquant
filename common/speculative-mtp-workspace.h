#pragma once

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

// The MTP CPU draft sampler uses top-k + dist, then discards the random draw
// and chooses candidate[0]. With no confidence threshold, a unique finite
// maximum gives the same token without materializing candidates. Leave ties,
// non-finite logits and a suppressed winner to the original sampler: its
// partial-sort tie ordering and suppression semantics remain authoritative.
inline llama_token common_mtp_confidence_free_token(const float *       logits,
                                                    int32_t             n_vocab,
                                                    const llama_token * suppressed,
                                                    int32_t             n_suppressed) {
    if (!logits || n_vocab <= 0 || n_suppressed < 0 || (n_suppressed > 0 && !suppressed)) {
        return LLAMA_TOKEN_NULL;
    }
    // Independent lanes avoid a vocabulary-length scalar dependency chain.
    // The second, branch-free pass checks uniqueness and exceptional values;
    // no floating-point arithmetic or changed tie order is introduced.
    float   maxima[4] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
    int32_t i         = 0;
    for (; i <= n_vocab - 4; i += 4) {
        for (int lane = 0; lane < 4; ++lane) {
            maxima[lane] = std::max(maxima[lane], logits[i + lane]);
        }
    }
    float maximum = std::max(std::max(maxima[0], maxima[1]), std::max(maxima[2], maxima[3]));
    for (; i < n_vocab; ++i) {
        maximum = std::max(maximum, logits[i]);
    }
    int32_t     matches = 0, invalid = 0;
    llama_token best = 0;
    for (i = 0; i < n_vocab; ++i) {
        const bool match = logits[i] == maximum;
        matches += match;
        invalid += !std::isfinite(logits[i]);
        best |= match ? i : 0;
    }
    if (matches != 1 || invalid != 0) {
        return LLAMA_TOKEN_NULL;
    }
    for (int32_t i = 0; i < n_suppressed; ++i) {
        if (suppressed[i] == best) {
            return LLAMA_TOKEN_NULL;
        }
    }
    return best;
}

// One maximum-sized host handoff arena for an MTP driver. A verification batch
// is shared by all its sequences, so allocate rows*width once, NOT
// sequences*rows*width. Effective row counts and accepted prefixes are metadata.
// host_hidden=false retains only row/token/position metadata. The device path
// uses exactly these accepted ranges without allocating a host hidden matrix.
class common_mtp_workspace {
public:
    common_mtp_workspace(uint32_t sequences, uint32_t rows, uint32_t width, bool host_hidden = true)
        : sequences_(sequences), capacity_(rows), width_(width),
          host_hidden_(host_hidden),
          verified_(host_hidden ? elements(rows, width) : 0), pending_(host_hidden ? elements(sequences, width) : 0),
          seed_(host_hidden ? elements(sequences, width) : 0), tokens_(rows), positions_(rows), ranges_(sequences) {
        if (sequences == 0 || rows == 0 || width == 0) {
            throw std::invalid_argument("MTP workspace capacity must be nonzero");
        }
    }

    uint32_t capacity() const { return capacity_; }
    uint32_t width() const { return width_; }
    uint32_t first_row(uint32_t sequence) const { return sequence < sequences_ ? ranges_[sequence].first : 0; }
    uint32_t verified_rows(uint32_t sequence) const { return sequence < sequences_ ? ranges_[sequence].rows : 0; }

    float * pending(uint32_t sequence) {
        return host_hidden_ && sequence < sequences_ ? pending_.data() + size_t(sequence) * width_ : nullptr;
    }
    const float * pending(uint32_t sequence) const {
        return host_hidden_ && sequence < sequences_ ? pending_.data() + size_t(sequence) * width_ : nullptr;
    }

    bool has_staged(uint32_t sequence) const {
        return sequence < sequences_ && ranges_[sequence].staged;
    }

    // Preflight all ranges before the first write. Call only after target output
    // has completed; this copies actual rows, never the padding to capacity.
    bool begin(uint32_t rows, const float * hidden, const llama_token * tokens, const llama_pos * positions) {
        if (rows == 0 || rows > capacity_ || (host_hidden_ && !hidden) || !tokens || !positions ||
            std::any_of(ranges_.begin(), ranges_.end(), [](const range & r) { return r.staged; })) {
            return false;
        }
        if (host_hidden_) {
            std::copy_n(hidden, size_t(rows) * width_, verified_.data());
        }
        std::copy_n(tokens, rows, tokens_.data());
        std::copy_n(positions, rows, positions_.data());
        active_rows_ = rows;
        std::fill(ranges_.begin(), ranges_.end(), range{});
        return true;
    }

    bool sequence(uint32_t seq, uint32_t first, uint32_t rows, bool staged, bool deferred) {
        if (seq >= sequences_ || rows == 0 || first > active_rows_ || rows > active_rows_ - first ||
            ranges_[seq].rows != 0) {
            return false;
        }
        for (const auto & existing : ranges_) {
            if (existing.rows != 0 && first < uint64_t(existing.first) + existing.rows &&
                existing.first < uint64_t(first) + rows) {
                return false;
            }
        }
        auto & r = ranges_[seq];
        r.first = first;
        r.rows = rows;
        r.staged = staged;
        r.commit = staged ? (deferred ? -1 : int64_t(rows)) : 0;
        if (staged && host_hidden_) {
            // Only the cross-batch seed needs its own copy. Remaining catch-up
            // inputs are verified_[first+i-1], not a second hidden-row matrix.
            std::copy_n(pending(seq), width_, seed_.data() + size_t(seq) * width_);
        }
        if (host_hidden_) {
            std::copy_n(verified_.data() + size_t(first + rows - 1) * width_, width_, pending(seq));
        }
        return true;
    }

    bool accept(uint32_t seq, uint32_t accepted) {
        if (seq >= sequences_ || ranges_[seq].rows == 0) {
            return false;
        }
        auto & r = ranges_[seq];
        // Match the existing driver's indexing: row 0 is the sampled token,
        // row N is the Nth accepted candidate. A count is not a hidden state.
        const uint32_t row = std::min(accepted, r.rows - 1);
        if (host_hidden_) {
            std::copy_n(verified_.data() + size_t(r.first + row) * width_, width_, pending(seq));
        }
        if (r.staged) {
            r.commit = std::min<uint64_t>(uint64_t(accepted) + 1, r.rows);
        }
        return true;
    }

    uint32_t commit_rows(uint32_t seq) const {
        if (seq >= sequences_ || ranges_[seq].commit <= 0) {
            return 0;
        }
        return (uint32_t) ranges_[seq].commit;
    }

    uint32_t total_commit_rows() const {
        uint64_t total = 0;
        for (uint32_t seq = 0; seq < sequences_; ++seq) {
            total += commit_rows(seq);
        }
        // Disjoint sequence ranges are supplied by the driver. The explicit
        // cap also protects callers from accidentally staging overlapping ones.
        return total <= capacity_ ? (uint32_t) total : UINT32_MAX;
    }

    bool commit_row(uint32_t seq, uint32_t row, llama_token & token, llama_pos & position,
                    const float * & hidden) const {
        if (seq >= sequences_ || row >= commit_rows(seq)) {
            return false;
        }
        const auto & r = ranges_[seq];
        token = tokens_[r.first + row];
        position = positions_[r.first + row];
        hidden = !host_hidden_ ? nullptr : (row == 0 ? seed_.data() + size_t(seq) * width_
                          : verified_.data() + size_t(r.first + row - 1) * width_);
        return true;
    }

    void clear_staged() {
        for (auto & r : ranges_) {
            r.staged = false;
            r.commit = 0;
        }
    }

    void reset_sequence(uint32_t seq) {
        if (seq < sequences_) {
            if (host_hidden_) std::fill_n(pending(seq), width_, 0.0f);
            ranges_[seq] = {};
        }
    }

    void reset() {
        active_rows_ = 0;
        std::fill(pending_.begin(), pending_.end(), 0.0f);
        std::fill(ranges_.begin(), ranges_.end(), range{});
        // No clear/shrink of the owning vectors. Old inactive rows are never
        // addressable through commit_row or the active sequence ranges.
    }

    size_t hidden_storage_bytes() const {
        return (verified_.size() + pending_.size() + seed_.size()) * sizeof(float);
    }

private:
    struct range {
        uint32_t first = 0;
        uint32_t rows = 0;
        int64_t commit = 0;
        bool staged = false;
    };

    static size_t elements(uint32_t rows, uint32_t width) {
        if (rows == 0 || width == 0 || uint64_t(rows) * width > SIZE_MAX / sizeof(float)) {
            throw std::length_error("MTP workspace size overflow");
        }
        return size_t(rows) * width;
    }

    uint32_t sequences_;
    uint32_t capacity_;
    uint32_t width_;
    uint32_t active_rows_ = 0;
    bool host_hidden_ = true;
    std::vector<float> verified_;
    std::vector<float> pending_;
    std::vector<float> seed_;
    std::vector<llama_token> tokens_;
    std::vector<llama_pos> positions_;
    std::vector<range> ranges_;
};

// llama_batch_init chooses token OR embedding storage. MTP needs both; own
// both in one fixed-capacity allocation group rather than adding an unguarded
// malloc and growing it during a verification cycle. No llama_batch_free may
// be used on the returned borrowed view.
class common_mtp_batch_storage {
public:
    common_mtp_batch_storage(uint32_t rows, uint32_t width, bool host_embeddings = true)
        : tokens_(checked_rows(rows)), embd_(host_embeddings ? checked_elements(rows, width) : 0), positions_(rows),
          sequence_counts_(rows), sequence_ids_(rows), sequence_ptrs_(size_t(rows) + 1), logits_(rows) {
        for (uint32_t i = 0; i < rows; ++i) {
            sequence_ptrs_[i] = &sequence_ids_[i];
        }
        sequence_ptrs_[rows] = nullptr;
    }

    common_mtp_batch_storage(const common_mtp_batch_storage &) = delete;
    common_mtp_batch_storage & operator=(const common_mtp_batch_storage &) = delete;

    llama_batch view() {
        llama_batch result{};
        result.token = tokens_.data();
        result.embd = embd_.empty() ? nullptr : embd_.data();
        result.pos = positions_.data();
        result.n_seq_id = sequence_counts_.data();
        result.seq_id = sequence_ptrs_.data();
        result.logits = logits_.data();
        return result;
    }

private:
    static size_t checked_rows(uint32_t rows) {
        if (rows == 0 || rows > INT32_MAX) {
            throw std::length_error("MTP batch row capacity is outside the llama_batch ABI");
        }
        return rows;
    }
    static size_t checked_elements(uint32_t rows, uint32_t width) {
        if (width == 0 || uint64_t(rows) * width > SIZE_MAX / sizeof(float)) {
            throw std::length_error("MTP batch embedding capacity overflow");
        }
        return size_t(rows) * width;
    }

    std::vector<llama_token> tokens_;
    std::vector<float> embd_;
    std::vector<llama_pos> positions_;
    std::vector<int32_t> sequence_counts_;
    std::vector<llama_seq_id> sequence_ids_;
    std::vector<llama_seq_id *> sequence_ptrs_;
    std::vector<int8_t> logits_;
};
