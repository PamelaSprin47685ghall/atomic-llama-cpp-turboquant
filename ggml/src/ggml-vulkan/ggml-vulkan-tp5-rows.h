#pragma once

#include <cstdint>
#include <limits>
#include <cstddef>

// One column-tile program, not one shader/graph per token count. Keep the
// single-token contraction unchanged; batched direct output uses fixed tiles
// and masks the last tile before *any* activation/bias load.
static constexpr uint32_t VK_TP5_DIRECT_COLUMN_TILE = 4;

enum class vk_tp5_row_program : uint32_t { none, dense_columns, add };
struct vk_tp5_row_dispatch {
    vk_tp5_row_program program = vk_tp5_row_program::none;
    uint32_t element_capacity = 0; // bound tensor/descriptor, not bank padding
    uint32_t width = 0;
    uint32_t matrix_rows_per_group = 0;
};
struct vk_tp5_row_arguments { uint32_t x = 0, y = 1, z = 1; };
static_assert(sizeof(vk_tp5_row_arguments) == 12);

struct vk_tp5_rows_geometry {
    uint32_t width = 0;
    uint32_t rows = 0;
    uint32_t elements = 0;
    uint32_t groups_x = 0;
    uint32_t groups_y = 0;
};

inline bool vk_tp5_make_rows_geometry(uint64_t width, uint64_t elements, uint32_t matrix_rows_per_group,
                                       uint32_t max_groups_x, uint32_t max_groups_y,
                                       vk_tp5_rows_geometry & out) {
    if (!width || !elements || elements % width || width > UINT32_MAX || elements > UINT32_MAX ||
        !matrix_rows_per_group || !max_groups_x || !max_groups_y) {
        return false;
    }
    const uint64_t rows = elements / width;
    const uint64_t x = width / matrix_rows_per_group + (width % matrix_rows_per_group != 0);
    const uint64_t y = rows / VK_TP5_DIRECT_COLUMN_TILE + (rows % VK_TP5_DIRECT_COLUMN_TILE != 0);
    if (x > max_groups_x || y > max_groups_y) return false;
    out = {uint32_t(width), uint32_t(rows), uint32_t(elements), uint32_t(x), uint32_t(y)};
    return true;
}

inline bool vk_tp5_payload_elements_fit(uint64_t active_elements, uint64_t bank0_bytes, uint64_t bank1_bytes,
                                        uint32_t wire_bytes, uint32_t & capacity_elements) {
    // The last cache line belongs to the bounded relay status protocol, never
    // to the producer payload. Aligned spare bytes are capacity, not useful rows.
    if (!active_elements || active_elements > UINT32_MAX || (wire_bytes != 2 && wire_bytes != 4) ||
        bank0_bytes < 64 || bank1_bytes < 64) return false;
    const uint64_t smaller = bank0_bytes < bank1_bytes ? bank0_bytes : bank1_bytes;
    const uint64_t capacity = (smaller - 64) / wire_bytes;
    if (active_elements > capacity) return false;
    capacity_elements = uint32_t(capacity > UINT32_MAX ? UINT32_MAX : capacity);
    return true;
}

inline bool vk_tp5_make_row_arguments(const vk_tp5_row_dispatch & definition, uint64_t active_elements,
                                      uint32_t max_x, uint32_t max_y, vk_tp5_row_arguments & out) {
    if (definition.program == vk_tp5_row_program::none) {
        out = {};
        return true;
    }
    if (!active_elements || active_elements > definition.element_capacity || !max_x || !max_y) return false;
    vk_tp5_row_arguments args;
    if (definition.program == vk_tp5_row_program::dense_columns) {
        vk_tp5_rows_geometry shape;
        if (!vk_tp5_make_rows_geometry(definition.width, active_elements, definition.matrix_rows_per_group,
                                        max_x, max_y, shape)) return false;
        args = {shape.groups_x, shape.groups_y, 1};
    } else if (definition.program == vk_tp5_row_program::add) {
        const uint64_t groups = (active_elements + 1023) / 1024;
        if (groups > max_x) return false;
        args = {uint32_t(groups), 1, 1};
    } else {
        return false;
    }
    out = args;
    return true;
}

// Unified LateBind token dimension: scatter[token][stream], rho[token][stream],
// Q[token][stream][rank_dim], Y[token][width].
// Fixed small column tile (VK_TP5_DIRECT_COLUMN_TILE = 4) handles effective rows.
struct vk_tp5_latebind_layout {
    uint32_t width = 0;
    uint32_t streams = 4;
    uint32_t rank_dim = 320;
    uint32_t capacity_rows = VK_TP5_DIRECT_COLUMN_TILE; // 4 rows

    uint64_t scatter_elements() const { return uint64_t(capacity_rows) * streams; }
    uint64_t rho_elements()     const { return uint64_t(capacity_rows) * streams; }
    uint64_t q_elements()       const { return uint64_t(capacity_rows) * streams * rank_dim; }
    uint64_t y_elements()       const { return uint64_t(capacity_rows) * width; }
    uint64_t act_q8_blocks()    const { return uint64_t(capacity_rows) * streams * (width / 32); }
    uint64_t lo_elements()      const { return uint64_t(capacity_rows) * rank_dim; }

    uint64_t scatter_offset(uint32_t token, uint32_t stream) const {
        return uint64_t(token) * streams + stream;
    }
    uint64_t rho_offset(uint32_t token, uint32_t stream) const {
        return uint64_t(token) * streams + stream;
    }
    uint64_t q_offset(uint32_t token, uint32_t stream, uint32_t row) const {
        return (uint64_t(token) * streams + stream) * rank_dim + row;
    }
    uint64_t y_offset(uint32_t token, uint32_t col) const {
        return uint64_t(token) * width + col;
    }
    uint64_t lo_offset(uint32_t token, uint32_t row) const {
        return uint64_t(token) * rank_dim + row;
    }
};

inline bool vk_tp5_latebind_token_is_active(uint32_t token, uint32_t active_rows) {
    return token < active_rows;
}

inline uint64_t vk_tp5_latebind_active_q_elements(uint32_t active_rows, uint32_t streams, uint32_t rank_dim) {
    return uint64_t(active_rows) * streams * rank_dim;
}

// ===== LateBind Q-sidecar mailbox ABI (single source of truth) =====
//
// M0 closure: this layout was previously defined as statics in
// ggml-vulkan-collective.cpp and mirrored by hand in three test-tp5-plan
// fixtures. CPU reset, CPU polling, CPU reduce, shader push constants and
// the tests must all derive from these definitions — never a second copy.
// In the aggressive Q8 fast path the 64-byte control area lives at
// B + 64 + L and the payload at B + 128 + L; in the exact fallback the
// control uses status[6] (host-imported RAM) and the payload is B + 64 + L.
constexpr size_t TP5_RELAY_HEADER_BYTES   = 64;
constexpr size_t TP5_LATE_Q_CONTROL_BYTES = 64;
constexpr size_t TP5_LATE_Q_READY_WORD    = 0; // qctrl[0]: sidecar ready
constexpr size_t TP5_LATE_Q_COUNTER_WORD  = 1; // qctrl[1]: WG completion counter
// qctrl[2]: active rows for the Q8DOT producer. Under Host Push the aperture base
// is the host-import rank slot, whose first bytes are the P1 Y payload: the old
// qmail[5] row word was overwritten by every P1 copy, so Q8DOT always ran the
// full capacity. The control line is never touched by P1.
constexpr size_t TP5_LATE_Q_ROWS_WORD     = 2;
// Capacity ceiling: VK_TP5_DIRECT_COLUMN_TILE rows * 4 streams * rank <= 512.
constexpr size_t TP5_LATE_MAX_FLOATS      = 8192;

constexpr size_t tp5_late_q_control_offset(size_t late_host_offset) {
    return TP5_RELAY_HEADER_BYTES + late_host_offset;
}

inline volatile uint32_t * tp5_late_q_control_ptr(void * bcast_host, size_t late_host_offset) {
    if (!bcast_host || late_host_offset == 0) return nullptr;
    return (volatile uint32_t *) ((char *) bcast_host + tp5_late_q_control_offset(late_host_offset));
}

inline const volatile uint32_t * tp5_late_q_control_cptr(const void * bcast_host, size_t late_host_offset) {
    if (!bcast_host || late_host_offset == 0) return nullptr;
    return (const volatile uint32_t *) ((const char *) bcast_host + tp5_late_q_control_offset(late_host_offset));
}

constexpr size_t tp5_late_q_bcast_payload_offset(size_t late_host_offset, bool late_q8_fast) {
    return TP5_RELAY_HEADER_BYTES + late_host_offset + (late_q8_fast ? TP5_LATE_Q_CONTROL_BYTES : 0);
}

constexpr uint32_t tp5_late_q_payload_word_offset(size_t late_host_offset, bool late_q8_fast) {
    return (uint32_t) (tp5_late_q_bcast_payload_offset(late_host_offset, late_q8_fast) / 4);
}

// P1-B packed Q8 activation/weight layout: one block per 32 elements, each
// block { float d; uint qs_words[8]; } = 36 bytes. The activation buffers
// carry no done-flag block (per-token transients, not immutable weights).
constexpr uint64_t tp5_late_q8_packed_bytes(size_t elems) {
    return (elems / 32u) * (sizeof(float) + 8u * sizeof(uint32_t));
}

// ===== L2 (tip) statistics record and relay-formed LO blocks =====
//
// Single source for every party of the L2 sufficient-statistic protocol:
//   * the Q8DOT producer (workgroup 0) writes, per token, 32 words into the SAME
//     aperture it writes the F16 Q sidecar to (host-import under Host Push):
//       [4b+0] a_b = |R_b|^2  [4b+1] d_b = <R_b, z_p> (this rank's partial)
//       [4b+2] s_b (scatter)  [4b+3] tag = tp5_late_l2_tag(z_p[t][0], z_p[t][1])
//       (b = 0..3, each entry ONE 16-byte store)
//       [16..19] sum R_b  [20..23] sum z_p  [24..31] reserved
//     (the two plain sums are identity checks, not inputs to rho).
//     The ready word does NOT order these stores for the host: measured, the
//     relay occasionally read the entry left by the previous use of the bank
//     after seeing ready. Each entry therefore carries a tag derived from this
//     epoch's z_p, which the relay recomputes from its own P1 copy of z_p and
//     spins on until it matches;
//   * the relay folds a (rank 0), d (sum over ranks), s (rank 0) and forms
//       rho_b = 1 / sqrt((a_b + 2 s_b d_b + s_b^2 |y|^2) / width + eps)
//     -- exactly the norm kernel's inversesqrt(fold / width + eps), because
//     fold = |R_b + s_b y|^2 is the same quantity expanded;
//   * the relay publishes LO = SiLU(0.25 * sum_b rho_b Q_b) as Q8_0 blocks into
//     the bank's Q payload region, where the UP consumer reads them.
// The record lives right after the Q payload capacity of its aperture.
constexpr size_t TP5_LATE_STATS_WORDS_PER_TOKEN = 32;
constexpr size_t TP5_LATE_STATS_BYTES =
    size_t(VK_TP5_DIRECT_COLUMN_TILE) * TP5_LATE_STATS_WORDS_PER_TOKEN * sizeof(uint32_t);
constexpr size_t tp5_late_q_payload_capacity_bytes(bool late_q8_fast) {
    return TP5_LATE_MAX_FLOATS * (late_q8_fast ? 2u : 4u);
}
constexpr size_t tp5_late_stats_offset(size_t late_host_offset, bool late_q8_fast) {
    return tp5_late_q_bcast_payload_offset(late_host_offset, late_q8_fast) +
           tp5_late_q_payload_capacity_bytes(late_q8_fast);
}
constexpr uint32_t tp5_late_stats_word_offset(size_t late_host_offset, bool late_q8_fast) {
    return (uint32_t) (tp5_late_stats_offset(late_host_offset, late_q8_fast) / 4);
}
// Relay-formed LO blocks share the bank's Q payload offset (the F16 Q fanout they
// replace is no longer written in L2 mode). Same 36-byte block as P1-B.
constexpr size_t TP5_LATE_LO_Q8_BLOCK_WORDS = 9;
constexpr size_t tp5_late_lo_q8_offset(size_t late_host_offset) {
    return tp5_late_q_bcast_payload_offset(late_host_offset, true);
}
constexpr size_t tp5_late_lo_q8_bytes(uint32_t rank_dim) {
    return size_t(VK_TP5_DIRECT_COLUMN_TILE) * (rank_dim / 32u) * TP5_LATE_LO_Q8_BLOCK_WORDS * sizeof(uint32_t);
}
static_assert(TP5_LATE_STATS_BYTES % 64 == 0, "L2 statistics record must keep 64-byte alignment");
// ACT_Q8 -> Q8DOT per-block partials: [capacity tokens][streams][width/32] x vec4
// (a_part, d_part, sumR_part, sumZ_part), then one vec4 per capacity token whose
// .x carries the z_p tag (formed in ACT_Q8, the only L2 kernel that reads z_p
// before norm may overwrite it).
constexpr uint64_t tp5_late_l2_partials_bytes(uint32_t width, uint32_t streams) {
    return (uint64_t(VK_TP5_DIRECT_COLUMN_TILE) * streams * (width / 32u) + VK_TP5_DIRECT_COLUMN_TILE) *
           4u * sizeof(float);
}
// Freshness tag of a statistics entry: a 32-bit mix of the bit patterns of this
// epoch's z_p[t][0], z_p[t][1] and the armed epoch (RELAY header word 6). The GPU
// forms it in ACT_Q8; the relay forms it from its own P1 copy of the same z_p.
// Mirrored bit-for-bit in tp5_hc_latebind.comp.
constexpr size_t TP5_RELAY_EPOCH_WORD = 6;   // RELAY header word 6: armed epoch (low 32 bits)
constexpr uint32_t tp5_late_l2_tag(uint32_t z0_bits, uint32_t z1_bits, uint32_t epoch) {
    uint32_t h = ((z0_bits ^ (epoch * 0x27D4EB2Fu)) * 0x9E3779B1u) ^ ((z1_bits + 0x7F4A7C15u) * 0x85EBCA77u);
    h ^= h >> 15;
    return h | 1u;   // never 0: a zeroed entry can never validate
}
// Per-Q8DOT-workgroup Q checksum entries, right after the statistics record:
// uvec4(xor of the F16 payload words the workgroup wrote, nonce = this epoch's
// token-0 statistics tag, 0, 0), one 16-byte store before the workgroup's
// completion increment. The relay waits for the nonce, recomputes the xor from
// the payload it is about to reduce, and re-reads until they agree -- the Q
// ready word alone did not guarantee the host saw the payload (measured: the
// reduced Q differed between runs while the same payload re-read later was
// identical).
constexpr size_t TP5_LATE_QSUM_MAX_WGS = 32;   // rank_dim <= 512, 16 rows per WG
constexpr size_t TP5_LATE_QSUM_BYTES   = TP5_LATE_QSUM_MAX_WGS * 16;
constexpr uint32_t tp5_late_qsum_word_offset(size_t late_host_offset) {
    return tp5_late_stats_word_offset(late_host_offset, true) +
           uint32_t(VK_TP5_DIRECT_COLUMN_TILE * TP5_LATE_STATS_WORDS_PER_TOKEN);
}
static_assert(tp5_late_lo_q8_bytes(320) <= tp5_late_q_payload_capacity_bytes(true),
              "relay-formed LO blocks must fit in the Q payload region they replace");
