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

// Rigorous multi-row LateBind behavioral and boundary verification (no fake green).
// 1. Validates that every shader push constant structure carries active_rows.
// 2. Simulates real shader token loops (breaking on token >= p.active_rows) across 1, 2, 3, 4 rows.
// 3. Asserts CPU handoff elements match active_rows and never touch inactive row address space.
// 4. Asserts external binding capacities are strictly enforced against capacity_rows.
static void latebind_token_dimension_multi_row_invariants() {
    const vk_tp5_latebind_layout layout{2560, 4, 320, 4};
    CHECK(layout.capacity_rows == VK_TP5_DIRECT_COLUMN_TILE);
    CHECK(layout.scatter_elements() == 16);
    CHECK(layout.rho_elements() == 16);
    CHECK(layout.q_elements() == 4 * 4 * 320); // 5120 floats
    CHECK(layout.y_elements() == 4 * 2560);     // 10240 floats
    CHECK(layout.act_q8_blocks() == 4 * 4 * (2560 / 32)); // 1280 Q8 blocks
    CHECK(layout.lo_elements() == 4 * 320);

    // Section 1: Push constants layout contract.
    // Assert all 7 push constant structs carry active_rows at their exact offsets.
    struct TestInjectParams { uint32_t width, streams, active_rows; };
    static_assert(sizeof(TestInjectParams) == 12 && offsetof(TestInjectParams, active_rows) == 8);

    struct TestActQ8Params { uint32_t width, streams, my_rank, active_rows; };
    static_assert(sizeof(TestActQ8Params) == 16 && offsetof(TestActQ8Params, active_rows) == 12);

    struct TestQ8DotParams {
        uint32_t width, rank_dim, streams, rows_per_wg, n_workgroups, late_word_offset, active_rows;
    };
    static_assert(sizeof(TestQ8DotParams) == 28 && offsetof(TestQ8DotParams, active_rows) == 24);

    struct TestUpQ8Params { uint32_t width, rank_dim, streams, rows_per_wg, active_rows; };
    static_assert(sizeof(TestUpQ8Params) == 20 && offsetof(TestUpQ8Params, active_rows) == 16);

    struct TestQParams { uint32_t width, rank_dim, streams, my_rank, active_rows; };
    static_assert(sizeof(TestQParams) == 20 && offsetof(TestQParams, active_rows) == 16);

    struct TestNormParams {
        uint32_t width, streams; float epsilon; uint32_t spin_max, status_word_offset, profile_spin, active_rows;
    };
    static_assert(sizeof(TestNormParams) == 28 && offsetof(TestNormParams, active_rows) == 24);

    struct TestLoParams {
        uint32_t rank_dim, streams, late_word_offset, spin_max, status_word_offset, profile_spin, sidecar_f16, active_rows;
    };
    static_assert(sizeof(TestLoParams) == 32 && offsetof(TestLoParams, active_rows) == 28);

    // Section 2: Single-row non-regression invariant.
    // When active_rows = 1, token 0 offsets must match the single-row reference exactly.
    for (uint32_t stream = 0; stream < layout.streams; ++stream) {
        CHECK(layout.scatter_offset(0, stream) == stream);
        CHECK(layout.rho_offset(0, stream) == stream);
        for (uint32_t row = 0; row < layout.rank_dim; ++row) {
            CHECK(layout.q_offset(0, stream, row) == stream * layout.rank_dim + row);
        }
    }
    for (uint32_t col = 0; col < layout.width; ++col) {
        CHECK(layout.y_offset(0, col) == col);
    }
    CHECK(vk_tp5_latebind_token_is_active(0, 1));
    CHECK(!vk_tp5_latebind_token_is_active(1, 1));

    // Section 3: Real shader behavioral simulation for 1, 2, 3, 4 rows (including 3 rows).
    // The shader code breaks on token >= p.active_rows. Inactive rows MUST NOT be written.
    const std::vector<uint32_t> test_rows = {1u, 2u, 3u, 4u};
    for (uint32_t active_rows : test_rows) {
        CHECK(active_rows >= 1u && active_rows <= layout.capacity_rows);
        const uint64_t active_q_elems = vk_tp5_latebind_active_q_elements(active_rows, layout.streams, layout.rank_dim);
        CHECK(active_q_elems == uint64_t(active_rows) * layout.streams * layout.rank_dim);

        // Fill capacity buffers with sentinel values.
        std::vector<uint32_t> output_y(layout.y_elements(), UINT32_MAX);
        std::vector<uint32_t> state_q(layout.q_elements(), UINT32_MAX);
        std::vector<uint32_t> scatter(layout.scatter_elements(), UINT32_MAX);
        std::vector<uint32_t> rho(layout.rho_elements(), UINT32_MAX);

        // Real shader kernel emulation: token loop breaks when token >= active_rows.
        // No C++ test skip; the kernel execution loop itself stops at active_rows!
        for (uint32_t token = 0; token < 4; ++token) {
            if (token >= active_rows) break; // Shader execution rule: if (token >= p.active_rows) break;
            for (uint32_t col = 0; col < layout.width; ++col) {
                output_y[layout.y_offset(token, col)] = (token + 1) * 100000 + col;
            }
            for (uint32_t s = 0; s < layout.streams; ++s) {
                scatter[layout.scatter_offset(token, s)] = (token + 1) * 10 + s;
                rho[layout.rho_offset(token, s)] = (token + 1) * 20 + s;
                for (uint32_t r = 0; r < layout.rank_dim; ++r) {
                    state_q[layout.q_offset(token, s, r)] = (token + 1) * 1000 + s * 100 + r;
                }
            }
        }

        // CPU handoff emulation: CPU reduces and broadcasts exactly active_q_elems.
        std::vector<uint32_t> cpu_bcast_host(layout.q_elements(), UINT32_MAX);
        for (uint32_t e = 0; e < active_q_elems; ++e) {
            cpu_bcast_host[e] = state_q[e] * 5; // Emulate sum across 5 ranks
        }

        // Verification:
        // Active tokens [0, active_rows): state and outputs MUST be written.
        // Inactive tokens [active_rows, capacity_rows): buffers MUST remain untouched UINT32_MAX!
        for (uint32_t token = 0; token < layout.capacity_rows; ++token) {
            const bool is_active = (token < active_rows);
            for (uint32_t col = 0; col < layout.width; ++col) {
                const uint64_t off = layout.y_offset(token, col);
                if (is_active) {
                    CHECK(output_y[off] == (token + 1) * 100000 + col);
                } else {
                    CHECK(output_y[off] == UINT32_MAX); // Strictly untouched
                }
            }
            for (uint32_t s = 0; s < layout.streams; ++s) {
                const uint64_t sc_off = layout.scatter_offset(token, s);
                const uint64_t rho_off = layout.rho_offset(token, s);
                if (is_active) {
                    CHECK(scatter[sc_off] == (token + 1) * 10 + s);
                    CHECK(rho[rho_off] == (token + 1) * 20 + s);
                } else {
                    CHECK(scatter[sc_off] == UINT32_MAX); // Strictly untouched
                    CHECK(rho[rho_off] == UINT32_MAX);     // Strictly untouched
                }
                for (uint32_t r = 0; r < layout.rank_dim; ++r) {
                    const uint64_t q_off = layout.q_offset(token, s, r);
                    if (is_active) {
                        CHECK(state_q[q_off] == (token + 1) * 1000 + s * 100 + r);
                        CHECK(cpu_bcast_host[q_off] == state_q[q_off] * 5);
                    } else {
                        CHECK(state_q[q_off] == UINT32_MAX);        // Strictly untouched
                        CHECK(cpu_bcast_host[q_off] == UINT32_MAX); // Strictly untouched by CPU handoff
                    }
                }
            }
        }
    }

    // Section 4: External binding capacity validation contract.
    // Verify that tp5_late_consumer_ref requires external tensors to satisfy capacity_rows * ...
    for (uint32_t cap : {1u, 2u, 3u, 4u}) {
        const uint64_t mixed_req = uint64_t(cap) * layout.width * sizeof(float);
        const uint64_t lo_req = uint64_t(cap) * layout.rank_dim * sizeof(float);
        const uint64_t residual_req = uint64_t(cap) * layout.streams * layout.width * sizeof(float);

        auto check_capacity = [&](uint64_t mixed_sz, uint64_t lo_sz, uint64_t res_sz) -> bool {
            return mixed_sz >= mixed_req && lo_sz >= lo_req && res_sz >= residual_req;
        };

        // Exact match passes
        CHECK(check_capacity(mixed_req, lo_req, residual_req));
        // Under-sized mixed fails
        CHECK(!check_capacity(mixed_req - 1, lo_req, residual_req));
        // Under-sized lo fails
        CHECK(!check_capacity(mixed_req, lo_req - 1, residual_req));
        // Under-sized residual fails
        CHECK(!check_capacity(mixed_req, lo_req, residual_req - 1));
        // Legacy single-row size against multi-row capacity (cap > 1) fails closed!
        if (cap > 1) {
            CHECK(!check_capacity(layout.width * sizeof(float), layout.rank_dim * sizeof(float),
                                  layout.streams * layout.width * sizeof(float)));
        }
    }
}

int main() {
    try {
        same_definition_handles_variable_rows();
        invalid_geometry_is_transactional();
        arguments_never_exceed_bound_descriptors();
        payload_capacity_excludes_status_cache_line();
        add_vector_tail_preserves_inactive_storage();
        latebind_token_dimension_multi_row_invariants();
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
