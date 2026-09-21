#include "ggml-vulkan-tp5-liveness.hpp"

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <vector>

// CPU-only graph/alias policy fixture. Does not initialize a Vulkan instance,
// load a model, submit commands, or require any GPU to be present.
int main() {
    alignas(64) unsigned char storage[4096] = {};
    const auto tensor = [&](ggml_op op, size_t offset, int64_t length = 16) {
        ggml_tensor result{};
        result.type = GGML_TYPE_F32;
        result.op = op;
        result.data = storage + offset;
        result.ne[0] = length;
        result.nb[0] = sizeof(float);
        for (int d = 1; d < GGML_MAX_DIMS; ++d) {
            result.ne[d] = 1;
            result.nb[d] = result.nb[d - 1] * result.ne[d - 1];
        }
        return result;
    };
    const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (!a->data || !b->data) {
            return true; // match the backend's fail-closed unresolved range
        }
        const auto begin_a = reinterpret_cast<uintptr_t>(a->data);
        const auto begin_b = reinterpret_cast<uintptr_t>(b->data);
        const auto bytes_a = static_cast<uintptr_t>(a->ne[0]) * sizeof(float);
        const auto bytes_b = static_cast<uintptr_t>(b->ne[0]) * sizeof(float);
        return begin_a < begin_b + bytes_b && begin_b < begin_a + bytes_a;
    };

    int cases = 0;
    int failures = 0;
    const auto check = [&](const char * name, bool expected,
                           std::initializer_list<const ggml_tensor *> graph,
                           const ggml_tensor * producer, const ggml_tensor * target) {
        const std::vector<const ggml_tensor *> nodes(graph);
        const bool result = ggml_vk_tp5_output_can_elide_local(
            nodes.data(), static_cast<int>(nodes.size()), producer, target, overlap);
        ++cases;
        if (result != expected) {
            std::fprintf(stderr, "FAIL %s: elide=%d expected=%d\n", name, int(result), int(expected));
            ++failures;
        }
    };

    ggml_tensor producer = tensor(GGML_OP_MUL_MAT, 0);
    ggml_tensor view = tensor(GGML_OP_VIEW, 0);
    view.src[0] = &producer;
    view.view_src = &producer;
    ggml_tensor target = tensor(GGML_OP_RESHAPE, 0);
    target.src[0] = &view;
    target.view_src = &producer;
    check("terminal contraction", true, {&producer}, &producer, &producer);
    check("metadata-only tail", true, {&producer, &view, &target}, &producer, &target);

    ggml_tensor independent = tensor(GGML_OP_ADD, 256);
    check("independent trailing compute", true,
          {&producer, &independent, &view, &target}, &producer, &target);

    independent.src[0] = &producer;
    check("direct local reader", false, {&producer, &independent, &view, &target}, &producer, &target);
    independent.src[0] = &view;
    check("view local reader", false, {&producer, &view, &independent, &target}, &producer, &target);

    // A different tensor identity is not evidence of different storage.
    ggml_tensor alias = tensor(GGML_OP_VIEW, 8, 2);
    alias.src[0] = &producer;
    alias.view_src = &producer;
    alias.view_offs = 8;
    independent.src[0] = &alias;
    check("nonzero-offset alias reader", false,
          {&producer, &alias, &independent, &view, &target}, &producer, &target);

    ggml_tensor adjacent = tensor(GGML_OP_NONE, 64);
    independent.src[0] = &adjacent;
    check("adjacent range is not overlap", true,
          {&producer, &independent, &view, &target}, &producer, &target);
    independent.src[0] = nullptr;

    ggml_tensor overwriter = tensor(GGML_OP_ADD, 16);
    check("another writer aliases output", false,
          {&producer, &overwriter, &view, &target}, &producer, &target);

    // Earlier readers consume the previous contents, not this producer's z_p.
    ggml_tensor old_value = tensor(GGML_OP_NONE, 0);
    independent.src[0] = &old_value;
    check("earlier reader", true,
          {&independent, &producer, &view, &target}, &producer, &target);
    independent.src[0] = nullptr;

    ggml_tensor unresolved = tensor(GGML_OP_NONE, 512);
    unresolved.data = nullptr;
    independent.src[0] = &unresolved;
    check("unresolved trailing input retains local", false,
          {&producer, &independent, &view, &target}, &producer, &target);
    independent.src[0] = nullptr;

    check("missing producer", false, {&view, &target}, &producer, &target);
    check("duplicate producer", false, {&producer, &producer, &view, &target}, &producer, &target);
    check("wrong terminal", false, {&producer, &view, &target, &independent}, &producer, &target);
    check("null graph node", false, {&producer, nullptr, &target}, &producer, &target);
    check("empty graph", false, {}, &producer, &target);
    check("null producer", false, {&producer, &view, &target}, nullptr, &target);

    ggml_tensor cyclic_view = tensor(GGML_OP_VIEW, 0);
    cyclic_view.src[0] = &cyclic_view;
    check("cyclic metadata fails closed", false, {&producer, &cyclic_view}, &producer, &cyclic_view);
    ggml_tensor non_alias = tensor(GGML_OP_ADD, 0);
    non_alias.src[0] = &producer;
    check("terminal arithmetic is not a metadata alias", false,
          {&producer, &non_alias}, &producer, &non_alias);
    ggml_tensor foreign_view = tensor(GGML_OP_VIEW, 0);
    foreign_view.src[0] = &old_value;
    check("unrelated terminal alias", false,
          {&producer, &foreign_view}, &producer, &foreign_view);

    const auto scalar_check = [&](const char * name, bool ok) {
        ++cases;
        if (!ok) {
            std::fprintf(stderr, "FAIL %s\n", name);
            ++failures;
        }
    };
    scalar_check("whole bound range", ggml_vk_tp5_range_fits(1024, 0, 0, 1024));
    scalar_check("nonzero bound view", ggml_vk_tp5_range_fits(1024, 512, 256, 256));
    scalar_check("view beyond allocation", !ggml_vk_tp5_range_fits(1024, 512, 513, 0));
    scalar_check("payload beyond allocation", !ggml_vk_tp5_range_fits(1024, 512, 256, 257));
    scalar_check("base beyond allocation", !ggml_vk_tp5_range_fits(1024, 1025, 0, 0));
    scalar_check("offset addition must not wrap", !ggml_vk_tp5_range_fits(UINT64_MAX, UINT64_MAX, 1, 0));
    scalar_check("payload addition must not wrap", !ggml_vk_tp5_range_fits(UINT64_MAX, 1, 0, UINT64_MAX));
    scalar_check("null BDA plus nonzero view stays null", ggml_vk_tp5_checked_address(0, 256) == 0);
    scalar_check("BDA overflow fails closed", ggml_vk_tp5_checked_address(UINT64_MAX - 7, 8) == 0);
    scalar_check("valid BDA plus view", ggml_vk_tp5_checked_address(4096, 256) == 4352);

    std::printf("TP5 output liveness: %d cases, %d failures\n", cases, failures);
    return failures == 0 ? 0 : 1;
}
