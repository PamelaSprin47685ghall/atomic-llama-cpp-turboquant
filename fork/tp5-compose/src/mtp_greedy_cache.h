#pragma once

#include <array>
#include <cstddef>

#include "tq/sampling/shortlist.h"

namespace tq::tp5 {

// Per-epoch cache for the MTP target logits rows.
//
// The verify loop, the rejected-prefix replay diagnostic and the commit loop
// used to scan the same target logits rows with select_finite_greedy. This
// cache is local to one verify epoch: a later target decode can overwrite the
// logits buffer, so it must not outlive the verify/replay/commit sequence.
//
// The commit may use a cached row only when the current logits row is
// guaranteed to have the same argmax:
//   - row < accepted: the replay code requires the replay argmax to equal the
//     original verifier argmax for every accepted-prefix row;
//   - row == accepted in the native-trim branch: there is no replay, so the
//     verify row is still the current row;
//   - row == accepted in the replay branch: only after the replay scan has
//     refreshed the row. The replay argmax may differ there; the code only
//     requires the replay/original distributions to be close, not the top-1;
//   - the full-accept new row n is never verified and must be scanned;
//   - row 0 may be seeded from the LateBind prefetch g0. It follows the same
//     rules: accepted row or native-trim mismatch may use it directly, while
//     an accepted==0 replay must refresh it from the replay scan.
//
// Preconditions: the MTP runtime calls this only after its own
// require(n >= 1 && n <= 3 && local_rows.size() == n + 1), so row and
// accepted are in [0, n] with n <= 3. full_accept is (accepted == n).
// store_* silently ignore row >= max_rows; get() requires has(row).
class mtp_greedy_cache {
public:
    static constexpr std::size_t max_rows = 4;

    // Store the verify result for a target row. The LateBind g0 uses the same
    // entry: it is the original row 0 read before the verify loop.
    void store_verify(std::size_t row, const tq::sampling::greedy_result & result) {
        if (row < max_rows) {
            rows_[row] = result;
            state_[row] = state::verified;
        }
    }

    // Refresh a row with the post-replay result. The replay buffer is a new
    // decode of the verified prefix; its argmax is not guaranteed to equal the
    // original for the mismatch row, so the commit must use this value.
    void store_replay(std::size_t row, const tq::sampling::greedy_result & result) {
        if (row < max_rows) {
            rows_[row] = result;
            state_[row] = state::replayed;
        }
    }

    bool has(std::size_t row) const {
        return row < max_rows && state_[row] != state::empty;
    }

    bool replayed(std::size_t row) const {
        return row < max_rows && state_[row] == state::replayed;
    }

    const tq::sampling::greedy_result & get(std::size_t row) const {
        return rows_[row];
    }

    bool can_commit(std::size_t row, std::size_t accepted, bool native_trim, bool full_accept) const {
        if (row >= max_rows || !has(row)) {
            return false;
        }
        if (full_accept && row == accepted) {
            // The full-accept new row n was never verified; it must be scanned.
            return false;
        }
        if (row < accepted) {
            return true;
        }
        if (row == accepted) {
            return native_trim || replayed(row);
        }
        return false;
    }

private:
    enum class state : unsigned char {
        empty,
        verified,
        replayed,
    };

    std::array<tq::sampling::greedy_result, max_rows> rows_{};
    std::array<state, max_rows> state_{};
};

} // namespace tq::tp5
