#pragma once

#include <cstddef>

// Relay completion-token ordering policy, shared by the HIP provider and its
// CPU contract test.
//
// prepare and begin both reject while any token record exists. The record
// owns the lane tail event handles, so a new epoch must not re-record them
// before release_token retires both maps. A settled-but-unreleased token still
// blocks the next prepare/begin.
inline bool relay_token_blocks_new_epoch(size_t tail_records, size_t state_records) {
    return tail_records != 0 || state_records != 0;
}

// release_token is idempotent: an absent record is already retired.
inline bool relay_token_release_is_noop(size_t state_records) {
    return state_records == 0;
}

// wait_token must never recreate a record after release. An absent record is
// GGML_RELAY_TOKEN_UNKNOWN, not a resurrected token.
inline bool relay_token_wait_after_release_is_unknown(size_t state_records) {
    return state_records == 0;
}
