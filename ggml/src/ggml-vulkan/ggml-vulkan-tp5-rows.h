#pragma once

#include <cstdint>
#include <limits>

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
