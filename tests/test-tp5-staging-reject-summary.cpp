// TP5 staging rejection-summary CPU test (no GPU, no HIP runtime).
//
// The exit dump keeps the first rejected candidate of each (lane, reason)
// class. That example must be a self-owned copy: the source tensor is freed
// before the dump runs. The test records a rejection, mutates the tensor name,
// releases the tensor and the Meta owners, then dumps and checks that the
// original name is still printed and the mutated name is not.
//
// It also records an oversized union index input. The consumer path must name
// that class slot_size, not other: the diagnostic and the eligibility
// predicate read one decision.
//
// Build and run (CPU only, CTest target in tests/CMakeLists.txt):
//   cmake --build build-ws4-sched-ctest -j --target test-tp5-staging-reject-summary
//   ctest --test-dir build-ws4-sched-ctest -R test-tp5-staging-reject-summary --output-on-failure

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "../ggml/src/ggml-cuda/staging.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

using namespace ggml_cuda_staging;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            ++g_failures;                                                                \
        }                                                                                \
    } while (0)

bool fake_enabled() {
    return true;
}

bool fake_configured() {
    return true;
}

bool fake_usable() {
    return true;
}

void fake_mark_partial(const lease *, const void *) {
}

bool fake_enter_submit() {
    return true;
}

void fake_leave_submit() {
}

// Registration needs abi_version, mark_partial and kind_count: a zero
// kind_count is smaller than the host's domain and is rejected before the
// rejection path can run. The rejection path itself only needs
// enabled/configured/usable. No staging candidate reaches acquire, upload,
// wait, release or the enqueue hooks here, so they stay null.
const api g_fake_api = {
    /* .abi_version       = */ kApiVersion,
    /* .enabled           = */ fake_enabled,
    /* .merge_wait        = */ nullptr,
    /* .acquire           = */ nullptr,
    /* .acquire_nowait    = */ nullptr,
    /* .upload            = */ nullptr,
    /* .upload_rows       = */ nullptr,
    /* .wait              = */ nullptr,
    /* .release           = */ nullptr,
    /* .owns              = */ nullptr,
    /* .device_ptr        = */ nullptr,
    /* .dump              = */ nullptr,
    /* .can_enqueue_async = */ nullptr,
    /* .enqueue_async     = */ nullptr,
    /* .can_enqueue_sync  = */ nullptr,
    /* .enqueue_sync      = */ nullptr,
    /* .first_error       = */ nullptr,
    /* .enter_submit      = */ fake_enter_submit,
    /* .leave_submit      = */ fake_leave_submit,
    /* .unknown_count     = */ nullptr,
    /* .submission_pending= */ nullptr,
    /* .configured        = */ fake_configured,
    /* .usable            = */ fake_usable,
    /* .zc_enabled        = */ nullptr,
    /* .mark_partial      = */ fake_mark_partial,
    /* .kind_count        = */ kKindCount,
};

ggml_backend_meta_split_state split_mirrored(const ggml_tensor *, void *) {
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

struct fixture {
    ggml_backend_t        backend = nullptr;
    ggml_context *        ctx     = nullptr;
    ggml_backend_buffer_t buffer  = nullptr;
    ggml_tensor *         tensor  = nullptr;
    ggml_tensor *         union_tensor = nullptr;
};

bool make_fixture(ggml_backend_dev_t cpu, fixture & f) {
    ggml_backend_dev_t children[] = { cpu, cpu };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(children, 2, split_mirrored, nullptr);
    CHECK(meta_dev != nullptr);
    if (meta_dev == nullptr) {
        return false;
    }
    f.backend = ggml_backend_dev_init(meta_dev, nullptr);
    CHECK(f.backend != nullptr);
    if (f.backend == nullptr) {
        return false;
    }

    ggml_init_params params = { /*.mem_size=*/ 64 * 1024, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    f.ctx = ggml_init(params);
    CHECK(f.ctx != nullptr);
    if (f.ctx == nullptr) {
        return false;
    }

    f.tensor = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, 4, 4);
    CHECK(f.tensor != nullptr);
    if (f.tensor == nullptr) {
        return false;
    }
    ggml_set_name(f.tensor, "reject-probe");
    // No GGML_TENSOR_FLAG_INPUT: staging_eligible() rejects it as
    // no_input_flag, which is the class recorded by the summary.

    // An oversized union index input: above the 64 KiB slot envelope but still
    // inside the per-request bound. staging_eligible() rejects it as
    // slot_envelope, which the consumer must report as slot_size.
    f.union_tensor = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32,
                                        int64_t(kGenericMaxBytes / 4 + 1), 1);
    CHECK(f.union_tensor != nullptr);
    if (f.union_tensor == nullptr) {
        return false;
    }
    ggml_set_name(f.union_tensor, "union_inp_k_idxs");
    f.union_tensor->flags |= GGML_TENSOR_FLAG_INPUT;

    f.buffer = ggml_backend_alloc_ctx_tensors_from_buft(
            f.ctx, ggml_backend_dev_buffer_type(meta_dev));
    CHECK(f.buffer != nullptr);
    CHECK(f.tensor->data != nullptr);
    CHECK(f.union_tensor->data != nullptr);
    return f.buffer != nullptr && f.tensor->data != nullptr && f.union_tensor->data != nullptr;
}

}  // namespace

int main() {
    // The aggregation only runs with the diagnostic flag on. Set it before the
    // first staging call caches the flag.
    setenv("GGML_TP5_STAGING_DIAG", "1", 1);

    CHECK(ggml_backend_meta_staging_register(&g_fake_api));

    ggml_backend_reg_t cpu_reg = ggml_backend_cpu_reg();
    CHECK(cpu_reg != nullptr);
    ggml_backend_dev_t cpu = cpu_reg != nullptr ? ggml_backend_reg_dev_get(cpu_reg, 0) : nullptr;
    CHECK(cpu != nullptr);
    if (cpu == nullptr) {
        return 1;
    }

    fixture f;
    if (!make_fixture(cpu, f)) {
        return 1;
    }

    std::vector<uint8_t> source(ggml_nbytes(f.tensor), 0x5a);
    ggml_backend_tensor_set(f.tensor, source.data(), 0, source.size());
    std::vector<uint8_t> union_source(ggml_nbytes(f.union_tensor), 0x6b);
    ggml_backend_tensor_set(f.union_tensor, union_source.data(), 0, union_source.size());

    // The summary must own its example. Mutate the source name, then release
    // the tensor, the buffer and the backend before the dump.
    std::snprintf(f.tensor->name, GGML_MAX_NAME, "%s", "mutated-after-log");
    std::snprintf(f.union_tensor->name, GGML_MAX_NAME, "%s", "mutated-union-after-log");
    ggml_backend_buffer_free(f.buffer);
    ggml_free(f.ctx);
    ggml_backend_free(f.backend);

    FILE * capture = std::tmpfile();
    CHECK(capture != nullptr);
    if (capture == nullptr) {
        return 1;
    }
    const int saved = dup(fileno(stderr));
    CHECK(saved >= 0);
    dup2(fileno(capture), fileno(stderr));
    ggml_backend_meta_staging_diag_dump();
    std::fflush(stderr);
    dup2(saved, fileno(stderr));
    close(saved);

    std::rewind(capture);
    char out[8192];
    const size_t n = std::fread(out, 1, sizeof(out) - 1, capture);
    out[n] = '\0';
    std::fclose(capture);

    CHECK(std::strstr(out, "reason=no_input_flag") != nullptr);
    CHECK(std::strstr(out, "reason=no_input_flag count=1") != nullptr);
    CHECK(std::strstr(out, "example_name=reject-probe") != nullptr);
    // The oversized union input reaches the consumer as slot_size, never as
    // other, and its example name is the name captured at rejection time.
    CHECK(std::strstr(out, "reason=slot_size count=1") != nullptr);
    CHECK(std::strstr(out, "example_name=union_inp_k_idxs") != nullptr);
    CHECK(std::strstr(out, "reason=other") == nullptr);
    CHECK(std::strstr(out, "mutated-after-log") == nullptr);
    CHECK(std::strstr(out, "mutated-union-after-log") == nullptr);

    if (g_failures != 0) {
        std::fprintf(stderr, "TP5_STAGING_REJECT_TEST FAIL (%d)\n", g_failures);
        return 1;
    }
    std::printf("TP5_STAGING_REJECT_TEST PASS\n");
    return 0;
}
