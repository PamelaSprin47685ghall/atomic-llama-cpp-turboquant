#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static bool fail_next_allocation = false;
static ggml_backend_buffer_t (*cpu_alloc)(ggml_backend_buffer_type_t, size_t);

static ggml_backend_buffer_t failing_alloc(ggml_backend_buffer_type_t type, size_t size) {
    if (fail_next_allocation) {
        fail_next_allocation = false;
        return nullptr;
    }
    return cpu_alloc(type, size);
}

static bool test_static_root() {
    ggml_init_params p = { 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(p);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!ctx || !cpu) return false;
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, 128, false, false);
    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * constant = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(input);
    ggml_tensor * output = ggml_add(ctx, input, constant);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, output);
    bool ok = sched && ggml_backend_sched_alloc_graph(sched, graph);
    if (ok) {
        const float a[] = { 1, 2, 3, 4 }, c[] = { 4, 3, 2, 1 };
        float got[4] = {};
        ggml_backend_tensor_set(input, a, 0, sizeof(a));
        ggml_backend_tensor_set(constant, c, 0, sizeof(c));
        ggml_tensor * tensors[] = { input, constant, output };
        const uint8_t fixed[] = { 0, 0, 0 };
        const ggml_tensor before = *constant;
        ok = !ggml_backend_sched_replan_graph(sched, tensors, fixed, 3, nullptr, nullptr, 0) &&
             std::memcmp(&before, constant, sizeof(before)) == 0 &&
             ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
        if (ok) {
            ggml_backend_tensor_get(output, got, 0, sizeof(got));
            for (float value : got) ok = ok && value == 5.0f;
        }
    }
    if (sched) ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return ok;
}

static bool test_multi_split() {
    ggml_init_params p = { 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(p);
    ggml_backend_t first = ggml_backend_cpu_init();
    ggml_backend_t second = ggml_backend_cpu_init();
    if (!ctx || !first || !second) return false;
    // Distinct host buffer types prevent the scheduler from upgrading both
    // explicitly assigned nodes to the same backend.
    ggml_backend_buffer_type first_type = *ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type second_type = *ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_t bufts[] = { &first_type, &second_type };
    ggml_backend_t backends[] = { first, second };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 2, 128, false, false);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(a);
    ggml_set_input(b);
    ggml_tensor * sum = ggml_add(ctx, a, b);
    ggml_tensor * output = ggml_mul(ctx, sum, b);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, output);
    bool ok = sched != nullptr;
    if (ok) {
        ggml_backend_sched_set_tensor_backend(sched, sum, first);
        ggml_backend_sched_set_tensor_backend(sched, output, second);
        ok = ggml_backend_sched_alloc_graph(sched, graph) &&
             ggml_backend_sched_get_n_splits(sched) == 2;
    }
    if (ok) {
        const float av[] = { 1, 2, 3, 4 }, bv[] = { 2, 3, 4, 5 };
        float got[4] = {};
        ggml_backend_tensor_set(a, av, 0, sizeof(av));
        ggml_backend_tensor_set(b, bv, 0, sizeof(bv));
        ggml_tensor * tensors[] = { a, b, sum, output };
        const uint8_t fixed[] = { 0, 0, 0, 0 };
        const ggml_tensor before_a = *a, before_b = *b, before_sum = *sum, before_out = *output;
        ok = !ggml_backend_sched_replan_graph(sched, tensors, fixed, 4, nullptr, nullptr, 0) &&
             std::memcmp(a, &before_a, sizeof(*a)) == 0 && std::memcmp(b, &before_b, sizeof(*b)) == 0 &&
             std::memcmp(sum, &before_sum, sizeof(*sum)) == 0 &&
             std::memcmp(output, &before_out, sizeof(*output)) == 0 &&
             ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
        if (ok) {
            ggml_backend_tensor_get(output, got, 0, sizeof(got));
            for (int i = 0; i < 4; ++i) ok = ok && got[i] == (av[i] + bv[i]) * bv[i];
        }
    }
    if (sched) ggml_backend_sched_free(sched);
    ggml_backend_free(second);
    ggml_backend_free(first);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_init_params gp = { 4 * 1024 * 1024, nullptr, true };
    ggml_init_params fp = { 4096, nullptr, false };
    ggml_context * ctx = ggml_init(gp);
    ggml_context * fixed_ctx = ggml_init(fp);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!ctx || !fixed_ctx || !cpu) return 1;
    ggml_backend_t backends[] = { cpu };
    ggml_backend_buffer_type failing_type = *ggml_backend_cpu_buffer_type();
    cpu_alloc = failing_type.iface.alloc_buffer;
    failing_type.iface.alloc_buffer = failing_alloc;
    ggml_backend_buffer_type_t bufts[] = { &failing_type };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 1, 1024, false, false);
    if (!sched) return 1;

    ggml_tensor * weights = ggml_new_tensor_2d(fixed_ctx, GGML_TYPE_F32, 8, 1);
    for (int i = 0; i < 8; ++i) static_cast<float *>(weights->data)[i] = float(i);
    void * fixed_data = weights->data;
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 1);
    ggml_set_input(input);
    ggml_tensor * view = ggml_view_2d(ctx, input, 8, 1, input->nb[1], 0);
    ggml_tensor * output = ggml_add(ctx, view, weights);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1024, false);
    ggml_build_forward_expand(graph, output);
    if (!ggml_backend_sched_alloc_graph(sched, graph)) return 1;

    ggml_tensor * tensors[] = { weights, input, view, output };
    const uint8_t fixed[] = { 1, 0, 0, 0 };
    const int widths[] = { 1, 3, 2, 4 };
    void * initial_input = input->data;
    ggml_backend_buffer_t initial_buffer = input->buffer;
    const uint8_t wrong_fixed[] = { 1, 1, 0, 0 };
    ggml_tensor * duplicate[] = { weights, input, view, input };
    const ggml_tensor input_before = *input;
    const ggml_tensor view_before = *view;
    const ggml_tensor output_before = *output;
    bool ok = !ggml_backend_sched_replan_graph(sched, tensors + 1, fixed + 1, 3, nullptr, nullptr, 0) &&
              !ggml_backend_sched_replan_graph(sched, tensors, wrong_fixed, 4, nullptr, nullptr, 0) &&
              !ggml_backend_sched_replan_graph(sched, duplicate, fixed, 4, nullptr, nullptr, 0) &&
              input->data == initial_input && input->buffer == initial_buffer && weights->data == fixed_data &&
              std::memcmp(input, &input_before, sizeof(*input)) == 0 &&
              std::memcmp(view, &view_before, sizeof(*view)) == 0 &&
              std::memcmp(output, &output_before, sizeof(*output)) == 0;
    // An arena-owned, non-input leaf is a static value and must not be lost.
    input->flags &= ~GGML_TENSOR_FLAG_INPUT;
    const ggml_tensor static_before = *input;
    ok = ok && !ggml_backend_sched_replan_graph(sched, tensors, fixed, 4, nullptr, nullptr, 0) &&
         std::memcmp(input, &static_before, sizeof(*input)) == 0;
    input->flags = input_before.flags;
    // A proposed view beyond its root must leave the old graph untouched.
    ggml_backend_sched_tensor_layout bad = {};
    bad.tensor = view;
    std::memcpy(bad.ne, view->ne, sizeof(view->ne));
    std::memcpy(bad.nb, view->nb, sizeof(view->nb));
    const size_t bad_offset = ggml_nbytes(input);
    ok = ok && !ggml_backend_sched_replan_graph(sched, tensors, fixed, 4, &bad, &bad_offset, 1) &&
         std::memcmp(view, &view_before, sizeof(*view)) == 0;
    bad.ne[0] = 0;
    bad.ne[1] = -1;
    const size_t zero_offset = 0;
    ok = ok && !ggml_backend_sched_replan_graph(sched, tensors, fixed, 4, &bad, &zero_offset, 1) &&
         std::memcmp(view, &view_before, sizeof(*view)) == 0;

    // Fail the replacement arena allocation and execute the original graph.
    float original[8], restored[8];
    for (int i = 0; i < 8; ++i) original[i] = float(i * 2);
    ggml_backend_tensor_set(input, original, 0, sizeof(original));
    size_t plan_before = 0, plan_after = 0;
    ok = ok && ggml_backend_sched_get_tensor_plan_size(sched, input, &plan_before);
    ggml_backend_sched_tensor_layout proposed[3] = {};
    size_t proposed_offsets[3] = {};
    ggml_tensor * dynamic[] = { input, view, output };
    for (int i = 0; i < 3; ++i) {
        proposed[i].tensor = dynamic[i];
        std::memcpy(proposed[i].ne, dynamic[i]->ne, sizeof(dynamic[i]->ne));
        std::memcpy(proposed[i].nb, dynamic[i]->nb, sizeof(dynamic[i]->nb));
        proposed[i].ne[1] = 3;
        proposed[i].nb[2] = proposed[i].nb[1] * 3;
        proposed[i].nb[3] = proposed[i].nb[2];
        proposed_offsets[i] = dynamic[i]->view_offs;
    }
    fail_next_allocation = true;
    const bool injected = ggml_backend_sched_replan_graph(sched, tensors, fixed, 4, proposed, proposed_offsets, 3);
    ok = ok && !injected && !fail_next_allocation &&
         std::memcmp(input, &input_before, sizeof(*input)) == 0 &&
         std::memcmp(view, &view_before, sizeof(*view)) == 0 &&
         std::memcmp(output, &output_before, sizeof(*output)) == 0 &&
         ggml_backend_sched_get_tensor_plan_size(sched, input, &plan_after) && plan_before == plan_after;
    if (ok) {
        ok = ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
        if (ok) {
            ggml_backend_tensor_get(output, restored, 0, sizeof(restored));
            for (int i = 0; i < 8; ++i) ok = ok && restored[i] == original[i] + float(i);
        }
    }
    for (int step = 0; ok && step < 800; ++step) {
        const int width = widths[step % 4];
        ggml_backend_sched_tensor_layout layouts[3] = {};
        size_t offsets[3] = {};
        ggml_tensor * dynamic[] = { input, view, output };
        for (int i = 0; i < 3; ++i) {
            ggml_tensor * t = dynamic[i];
            layouts[i].tensor = t;
            std::memcpy(layouts[i].ne, t->ne, sizeof(t->ne));
            std::memcpy(layouts[i].nb, t->nb, sizeof(t->nb));
            layouts[i].ne[1] = width;
            layouts[i].nb[2] = layouts[i].nb[1] * size_t(width);
            layouts[i].nb[3] = layouts[i].nb[2];
            offsets[i] = t->view_offs;
        }
        if (!ggml_backend_sched_replan_graph(sched, tensors, fixed, 4, layouts, offsets, 3)) {
            std::fprintf(stderr, "replan failed at %d\n", step);
            ok = false;
            break;
        }
        size_t input_plan = 0, output_plan = 0;
        const size_t bytes = size_t(width) * 8 * sizeof(float);
        if (!ggml_backend_sched_get_tensor_plan_size(sched, input, &input_plan) ||
                !ggml_backend_sched_get_tensor_plan_size(sched, output, &output_plan) ||
                input_plan != bytes || output_plan != bytes ||
                weights->data != fixed_data || view->data != input->data || view->buffer != input->buffer) {
            std::fprintf(stderr, "layout mismatch at %d, plan=%zu/%zu expected=%zu\n",
                    step, input_plan, output_plan, bytes);
            ok = false;
            break;
        }
        // Machine-readable plan-size lines for the no-headroom gate: the
        // allocator request must equal the width this step asked for, not a
        // wider envelope. The equality above already fails on a difference;
        // these lines let the gate script re-check the same values.
        std::fprintf(stderr, "PLAN_SIZE step=%d kind=token active_tokens=%d planned_bytes=%zu requested_bytes=%zu\n",
                step, width, input_plan, bytes);
        std::fprintf(stderr, "PLAN_SIZE step=%d kind=output active_outputs=%d planned_bytes=%zu requested_bytes=%zu\n",
                step, width, output_plan, bytes);
        std::vector<float> values(size_t(width) * 8), got(values.size());
        for (size_t i = 0; i < values.size(); ++i) values[i] = float(step + int(i));
        ggml_backend_tensor_set(input, values.data(), 0, bytes);
        if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) { ok = false; break; }
        ggml_backend_tensor_get(output, got.data(), 0, bytes);
        for (size_t i = 0; i < got.size(); ++i) {
            const float expected = values[i] + float(i % 8);
            if (std::memcmp(&got[i], &expected, sizeof(float)) != 0) { ok = false; break; }
        }
    }
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    ggml_free(fixed_ctx);
    const bool static_ok = test_static_root();
    const bool split_ok = test_multi_split();
    if (!static_ok) std::fprintf(stderr, "static root rejection failed\n");
    if (!split_ok) std::fprintf(stderr, "multi-split rejection failed\n");
    return ok && static_ok && split_ok ? 0 : 1;
}
