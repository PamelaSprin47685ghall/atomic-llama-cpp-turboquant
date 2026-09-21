#include "../ggml/src/ggml-vulkan/ggml-vulkan-tp5-rows.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)

// Pure CPU contract tests; no Vulkan loader/device, shader compiler, model,
// environment variable or writable system setting is involved.
static void same_definition_handles_variable_rows() {
    const uint32_t width = 2560, capacity_rows = 32;
    const vk_tp5_row_dispatch program{vk_tp5_row_program::dense_columns, width * capacity_rows, width, 4};
    const std::array<uint32_t, 12> sequence{32, 1, 4, 2, 3, 5, 7, 8, 17, 18, 31, 32};
    for (uint32_t rows : sequence) {
        vk_tp5_row_arguments args;
        CHECK(vk_tp5_make_row_arguments(program, uint64_t(width) * rows, 65535, 65535, args));
        CHECK(args.x == 640 && args.y == (rows + 3) / 4 && args.z == 1);
        std::vector<uint32_t> visits(capacity_rows, 0);
        for (uint32_t group = 0; group < args.y; ++group) {
            for (uint32_t column = 0; column < VK_TP5_DIRECT_COLUMN_TILE; ++column) {
                const uint32_t token = group * VK_TP5_DIRECT_COLUMN_TILE + column;
                // This is the uniform guard before B/bias reads in both native
                // floating-point and MMVQ direct producer variants.
                if (token < rows) ++visits.at(token);
            }
        }
        for (uint32_t token = 0; token < capacity_rows; ++token) {
            CHECK(visits[token] == (token < rows ? 1u : 0u));
        }
        CHECK(program.element_capacity == width * capacity_rows);
    }
}

static void invalid_geometry_is_transactional() {
    const vk_tp5_rows_geometry sentinel{11, 12, 13, 14, 15};
    auto out = sentinel;
    auto unchanged = [&] {
        CHECK(out.width == sentinel.width && out.rows == sentinel.rows && out.elements == sentinel.elements &&
              out.groups_x == sentinel.groups_x && out.groups_y == sentinel.groups_y);
    };
    CHECK(!vk_tp5_make_rows_geometry(0, 16, 4, 65535, 65535, out)); unchanged();
    CHECK(!vk_tp5_make_rows_geometry(16, 0, 4, 65535, 65535, out)); unchanged();
    CHECK(!vk_tp5_make_rows_geometry(16, 17, 4, 65535, 65535, out)); unchanged();
    CHECK(!vk_tp5_make_rows_geometry(16, 32, 0, 65535, 65535, out)); unchanged();
    CHECK(!vk_tp5_make_rows_geometry(16, 32, 4, 3, 65535, out)); unchanged();
    CHECK(!vk_tp5_make_rows_geometry(16, 80, 4, 65535, 1, out)); unchanged();
    CHECK(!vk_tp5_make_rows_geometry(uint64_t(UINT32_MAX) + 1, 32, 4, UINT32_MAX, UINT32_MAX, out)); unchanged();
    CHECK(!vk_tp5_make_rows_geometry(16, uint64_t(UINT32_MAX) + 1, 4, UINT32_MAX, UINT32_MAX, out)); unchanged();
    CHECK(vk_tp5_make_rows_geometry(7, 35, 4, 65535, 65535, out));
    CHECK(out.width == 7 && out.rows == 5 && out.groups_x == 2 && out.groups_y == 2);
}

static void arguments_never_exceed_bound_descriptors() {
    const vk_tp5_row_dispatch dense{vk_tp5_row_program::dense_columns, 2560 * 8, 2560, 4};
    const vk_tp5_row_dispatch add{vk_tp5_row_program::add, 2560 * 8, 2560, 0};
    vk_tp5_row_arguments args{7, 8, 9};
    for (const auto & program : {dense, add}) {
        CHECK(!vk_tp5_make_row_arguments(program, uint64_t(program.element_capacity) + 1, 65535, 65535, args));
        CHECK(args.x == 7 && args.y == 8 && args.z == 9);
        CHECK(!vk_tp5_make_row_arguments(program, 0, 65535, 65535, args));
        CHECK(args.x == 7 && args.y == 8 && args.z == 9);
    }
    CHECK(!vk_tp5_make_row_arguments(dense, 2561, 65535, 65535, args));
    CHECK(!vk_tp5_make_row_arguments(dense, 2560, 639, 65535, args));
    CHECK(!vk_tp5_make_row_arguments(add, 2048, 1, 65535, args));
    CHECK(args.x == 7 && args.y == 8 && args.z == 9);
    CHECK(vk_tp5_make_row_arguments({}, 2560, 65535, 65535, args));
    CHECK(args.x == 0 && args.y == 1 && args.z == 1); // legacy single-row CB never reads these args
    CHECK(!vk_tp5_make_row_arguments({static_cast<vk_tp5_row_program>(99), 8, 1, 1}, 8, 8, 8, args));
}

static void payload_capacity_excludes_status_cache_line() {
    for (uint32_t bytes : {2u, 4u}) {
        const uint64_t active = 2560 * 4;
        const uint64_t bank = active * bytes + 64;
        uint32_t capacity = 123;
        CHECK(vk_tp5_payload_elements_fit(active, bank, bank, bytes, capacity));
        CHECK(capacity == active);
        CHECK(vk_tp5_payload_elements_fit(2560, bank, bank, bytes, capacity));
        CHECK(capacity == active); // useful rows changed, physical capacity did not
        capacity = 123;
        CHECK(!vk_tp5_payload_elements_fit(active + 1, bank, bank, bytes, capacity));
        CHECK(capacity == 123);
        CHECK(!vk_tp5_payload_elements_fit(active, bank, bank - bytes, bytes, capacity));
        CHECK(capacity == 123);
        CHECK(!vk_tp5_payload_elements_fit(1, 64, 64, bytes, capacity));
        CHECK(!vk_tp5_payload_elements_fit(1, 63, bank, bytes, capacity));
        CHECK(!vk_tp5_payload_elements_fit(0, bank, bank, bytes, capacity));
        CHECK(!vk_tp5_payload_elements_fit(uint64_t(UINT32_MAX) + 1, UINT64_MAX, UINT64_MAX, bytes, capacity));
        CHECK(capacity == 123);
    }
    uint32_t capacity = 456;
    CHECK(!vk_tp5_payload_elements_fit(1, 1024, 1024, 1, capacity));
    CHECK(capacity == 456);
}

static void add_vector_tail_preserves_inactive_storage() {
    const vk_tp5_row_dispatch program{vk_tp5_row_program::add, 4096, 1, 0};
    for (uint32_t active : {1u, 2u, 3u, 4u, 255u, 256u, 1023u, 1024u, 1025u, 4095u, 4096u}) {
        vk_tp5_row_arguments args;
        CHECK(vk_tp5_make_row_arguments(program, active, 65535, 65535, args));
        CHECK(args.x == (active + 1023) / 1024 && args.y == 1 && args.z == 1);
        std::vector<uint32_t> output(4096, UINT32_MAX);
        for (uint32_t thread = 0; thread < args.x * 256; ++thread) {
            const uint32_t base = thread * 4;
            if (base >= active || base >= program.element_capacity) continue;
            for (uint32_t i = 0; i < 4; ++i) {
                const uint32_t element = base + i;
                if (element >= active || element >= program.element_capacity) break;
                CHECK(output[element] == UINT32_MAX);
                output[element] = element;
            }
        }
        for (uint32_t i = 0; i < output.size(); ++i) CHECK(output[i] == (i < active ? i : UINT32_MAX));
    }
    vk_tp5_row_arguments args;
    CHECK(vk_tp5_make_row_arguments({vk_tp5_row_program::add, UINT32_MAX, 1, 0},
                                    UINT32_MAX, UINT32_MAX, 1, args));
    CHECK(args.x == uint64_t(UINT32_MAX) / 1024 + 1);
}

int main() {
    try {
        same_definition_handles_variable_rows();
        invalid_geometry_is_transactional();
        arguments_never_exceed_bound_descriptors();
        payload_capacity_excludes_status_cache_line();
        add_vector_tail_preserves_inactive_storage();
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
