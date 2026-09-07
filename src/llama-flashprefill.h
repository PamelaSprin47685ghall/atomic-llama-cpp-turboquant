// llama-flashprefill.h — internal FlashPrefill V2 reference policy (v1).
//
// Unit-testable, allocation-free C++ policy functions over the public C types
// in ../include/llama-flashprefill.h. No GGML/llama.h dependency; no wire
// conversion declared here. All functions are pure and deterministic; OFF
// short-circuits first.

#pragma once

#include "../include/llama-flashprefill.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C++" {
#include <cstddef>
#include <cstdint>
namespace llama_flashprefill {

// ---------------------------------------------------------------------------
// GQA packing (exact; BM may not divide G; tiles may start/end mid-group)
// ---------------------------------------------------------------------------

// Prevalidated small helpers: the caller must have established gqa > 0 and
// token * gqa + sub_head <= UINT32_MAX via packed_layout_checked() first;
// unpack_* additionally require the packed index to come from pack_index().
// These stay unchecked inline so hot planning code pays no repeat checks;
// checked entry points are packed_layout_checked() and the route_* functions.
inline uint32_t pack_index(uint32_t token, uint32_t sub_head, uint32_t gqa) {
    return token * gqa + sub_head;
}

inline uint32_t unpack_token(uint32_t packed, uint32_t gqa) {
    return packed / gqa;
}

inline uint32_t unpack_subhead(uint32_t packed, uint32_t gqa) {
    return packed % gqa;
}

// Derive total packed rows (= n_tokens * gqa) and packed-Q tile count
// (= ceil(total / block_q)). Checked; OVERFLOW when the product or the tile
// count exceeds uint32_t, COUNT when block_q/gqa is zero.
llama_flashprefill_error packed_layout_checked(
    uint32_t n_tokens,
    uint32_t gqa,
    uint32_t block_q,
    uint32_t * out_total_packed,
    uint32_t * out_tiles);

// True iff `packed_row` falls in the trailing dense tail: tile granularity,
// tile_id = packed_row / block_q, tiles = ceil(total_packed / block_q),
// dense iff tile_id >= tiles - min(tail_tiles, tiles).
// total_packed == 0 or tail_tiles == 0 -> false. Out-of-range packed_row
// (>= total) -> false (never sparse on doubt; caller treats as dense).
bool packed_row_in_dense_tail(
    uint32_t packed_row,
    uint32_t total_packed,
    uint32_t block_q,
    uint32_t tail_tiles);

// ---------------------------------------------------------------------------
// Routing (per-row; pure; no allocation; invalid config never routes sparse)
// ---------------------------------------------------------------------------

// Role/length/capability gate shared by both tail scopes. Does NOT evaluate
// the dense tail (no position info). Precedence:
//   OFF > ROLE > SHORT_CONTEXT > UNSUPPORTED > SPARSE,
// except exact_all: OFF > ROLE > UNSUPPORTED > EXACT_ALL (debug bypasses the
// length gate but never a missing backend). Ineligible rows keep their own
// reason (role/short) even when the backend is unsupported; the capability
// error is reported only for otherwise-eligible rows.
// resident_visible_tokens counts resident AND visible AND legal tokens only
// (never physical capacity, never Tri-deleted history).
llama_flashprefill_route route_for_role(
    const llama_flashprefill_config * cfg,
    int32_t role,
    uint32_t resident_visible_tokens,
    bool backend_supported);

// Call scope: dense tail measured against this call's query range.
// total_packed = packed rows in this call (see packed_layout_checked);
// packed_row = this row's packed index within the call.
// Full precedence: OFF > ROLE > SHORT > TAIL > UNSUPPORTED > SPARSE,
// except exact_all: OFF > ROLE > UNSUPPORTED > EXACT_ALL.
llama_flashprefill_route route_row_call(
    const llama_flashprefill_config * cfg,
    const llama_flashprefill_row * row,
    uint32_t resident_visible_tokens,
    bool backend_supported,
    uint32_t packed_row,
    uint32_t total_packed);

// Logical-prompt scope: dense tail measured against the row's frozen logical
// prefill interval. Requires row->prefill_known, else DENSE_UNKNOWN_BOUNDARY.
// packed_row_in_prefill = row's packed index inside the frozen interval;
// total_prefill_packed = interval packed size (interval_tokens * gqa).
// Full precedence: OFF > ROLE > UNKNOWN_BOUNDARY > SHORT > TAIL >
//   UNSUPPORTED > SPARSE, except exact_all: OFF > ROLE > UNKNOWN_BOUNDARY >
//   UNSUPPORTED > EXACT_ALL.
llama_flashprefill_route route_row_logical(
    const llama_flashprefill_config * cfg,
    const llama_flashprefill_row * row,
    uint32_t resident_visible_tokens,
    bool backend_supported,
    uint32_t packed_row_in_prefill,
    uint32_t total_prefill_packed);

// ---------------------------------------------------------------------------
// Scratch sizing (generic capacities; no hardcoded VRAM reserve constant)
// ---------------------------------------------------------------------------

struct scratch_inputs {
    uint32_t n_fragments;   // actual fragment count F (never assumed ceil(K/BN))
    uint32_t n_kv_heads;    // Hkv
    uint32_t d_k;           // K head dim (unpadded logical dim)
    uint32_t d_v;           // V head dim (unpadded logical dim)
    uint32_t n_packed_rows; // packed Q rows in this planning call
    uint32_t n_splits;      // split-K segments (0/1 == no split path)
    uint32_t mean_bytes;    // pooled-mean element width: 2 (F16) or 4 (F32)
};

// Peak scratch bytes over live-range overlap (pool + worst-case bounded index
// + descriptors + split (m,l,o) triples), each section 64B-aligned, all math
// checked 64-bit. OFF (or NULL/invalid cfg) -> *out_bytes = 0 with OK... but
// invalid cfg reports its validation error instead (never silently 0 for a
// broken config; explicit OFF reports OK + 0). Overflow -> ERR_OVERFLOW.
// Capacity guards (fragments <= 1<<24, heads <= 256, dims <= 4096,
// packed <= 1<<28, splits <= 1024, mean_bytes in {2,4}) -> ERR_COUNT /
// ERR_FLAG before any arithmetic.
llama_flashprefill_error scratch_bytes_checked(
    const llama_flashprefill_config * cfg,
    const scratch_inputs * in,
    uint64_t * out_bytes);

} // namespace llama_flashprefill
}
#endif
