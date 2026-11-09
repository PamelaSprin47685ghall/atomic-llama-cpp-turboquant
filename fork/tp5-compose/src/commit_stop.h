#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace tq::tp5 {

// One committed row's stop verdict. The order of the checks matches the
// runtime's historical precedence: cancel, EOG, then the predict budget.
enum class commit_stop {
    none,
    cancelled,
    eos,
    budget,
};

inline commit_stop classify_commit_stop(bool is_eog, bool ignore_eos, bool cancel_match,
                                        std::size_t token_count, std::size_t cancel_after,
                                        std::size_t predict) {
    if (cancel_match && token_count >= cancel_after) {
        return commit_stop::cancelled;
    }
    if (!ignore_eos && is_eog) {
        return commit_stop::eos;
    }
    if (token_count == predict) {
        return commit_stop::budget;
    }
    return commit_stop::none;
}

inline std::string_view commit_stop_reason(commit_stop stop) noexcept {
    switch (stop) {
        case commit_stop::cancelled:
            return "cancelled";
        case commit_stop::eos:
            return "eos";
        case commit_stop::budget:
            return "budget";
        case commit_stop::none:
            break;
    }
    return {};
}

// Production-used per-window stop/trim state. The commit loop calls note_stop
// after committing a row; the first stop for a request records the target
// trim position when that request has a later speculative row. reset() runs
// at the start of each epoch.
class commit_stop_state {
public:
    explicit commit_stop_state(std::size_t request_count) : trims_(request_count) {}

    void reset() noexcept {
        for (auto & trim : trims_) {
            trim.reset();
        }
    }

    // Returns the stop reason and records the target trim position. The
    // caller must remove the suffix, not only suppress the text.
    std::string_view note_stop(std::size_t local, std::int64_t input_position,
                               bool later_row, commit_stop stop) noexcept {
        if (stop == commit_stop::none) {
            return {};
        }
        if (later_row) {
            trims_[local] = input_position + 1;
        }
        return commit_stop_reason(stop);
    }

    const std::vector<std::optional<std::int64_t>> & trims() const noexcept { return trims_; }

private:
    std::vector<std::optional<std::int64_t>> trims_;
};

// True when the same request has another row after `row` in this speculative
// batch. The commit loop uses it to decide whether the target state contains
// an invalid suffix after the stop row.
inline bool has_later_row(const std::vector<std::size_t> & local_rows, std::size_t row,
                          std::size_t local) noexcept {
    for (std::size_t later = row + 1; later < local_rows.size(); ++later) {
        if (local_rows[later] == local) {
            return true;
        }
    }
    return false;
}

// Position of the first row owned by `local` at or before `row`. A draft
// context is per window; when a batch carries several requests, batch row 0
// is not necessarily the stopped request's source token. The caller supplies
// the row position accessor. The stopped row itself must belong to `local`.
template <typename PositionFn>
inline std::optional<std::int64_t> stopped_source_position(
        const std::vector<std::size_t> & local_rows, std::size_t row, std::size_t local,
        PositionFn && position) noexcept {
    if (row >= local_rows.size() || local_rows[row] != local) {
        return std::nullopt;
    }
    for (std::size_t source = 0; source <= row; ++source) {
        if (local_rows[source] == local) {
            return position(source);
        }
    }
    return std::nullopt;
}

// The original replay contract: accepted draft rows must match exactly; the
// correction row is allowed to differ only inside the KL/rel/overlap gate.
// A stop row must match because its token decides whether the request ends.
// A full accept does not replay the extra row, so row == accepted there is
// not a replay row.
inline bool replay_token_must_match(std::size_t row, std::size_t accepted, bool stop_row) noexcept {
    return row < accepted || (stop_row && row == accepted);
}

// Apply the per-request trim positions. The caller supplies the actual
// sequence-state removal; the test supplies a recorder. A nullopt entry means
// that request committed its last row and has no speculative suffix.
template <typename TrimFn>
void apply_commit_trims(const std::vector<std::optional<std::int64_t>> & trims, TrimFn && trim) {
    for (std::size_t local = 0; local < trims.size(); ++local) {
        if (trims[local].has_value()) {
            trim(local, *trims[local]);
        }
    }
}

} // namespace tq::tp5
