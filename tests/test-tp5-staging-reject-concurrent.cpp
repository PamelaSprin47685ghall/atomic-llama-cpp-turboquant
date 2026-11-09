// TP5 staging rejection-summary concurrency CPU test (no GPU, no HIP runtime).
//
// Several threads record rejections through independent Meta backends while a
// dump thread reads the summary. The hot path is a relaxed counter plus a
// one-time compare_exchange example; the dump must not block it, must not hold
// a lock across fprintf, and must not crash. The final dump must report the
// exact total and a published example.
//
// Build (CPU only):
//   cmake --build build-ws4-polish-cpu -j --target test-tp5-staging-reject-concurrent
//   ctest --test-dir build-ws4-polish-cpu -R test-tp5-staging-reject-concurrent

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "../ggml/src/ggml-cuda/staging.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
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

bool fake_enter_submit() {
    return true;
}

void fake_leave_submit() {
}

void fake_mark_partial(const lease *, const void *) {
}

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
};

ggml_backend_meta_split_state split_mirrored(const ggml_tensor *, void *) {
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

struct fixture {
    ggml_backend_t        backend = nullptr;
    ggml_context *        ctx     = nullptr;
    ggml_backend_buffer_t buffer  = nullptr;
    ggml_tensor *         tensor  = nullptr;
};

bool make_fixture(ggml_backend_dev_t cpu, int index, fixture & f) {
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
    char name[GGML_MAX_NAME];
    std::snprintf(name, sizeof(name), "reject-concurrent-%d", index);
    ggml_set_name(f.tensor, name);
    // No input flag: every set takes the no_input_flag rejection path.

    f.buffer = ggml_backend_alloc_ctx_tensors_from_buft(
            f.ctx, ggml_backend_dev_buffer_type(meta_dev));
    CHECK(f.buffer != nullptr);
    CHECK(f.tensor->data != nullptr);
    return f.buffer != nullptr && f.tensor->data != nullptr;
}

void free_fixture(fixture & f) {
    if (f.buffer != nullptr) {
        ggml_backend_buffer_free(f.buffer);
        f.buffer = nullptr;
    }
    if (f.ctx != nullptr) {
        ggml_free(f.ctx);
        f.ctx = nullptr;
    }
    if (f.backend != nullptr) {
        ggml_backend_free(f.backend);
        f.backend = nullptr;
    }
}

}  // namespace

int main() {
    setenv("GGML_TP5_STAGING_DIAG", "1", 1);

    CHECK(ggml_backend_meta_staging_register(&g_fake_api));

    ggml_backend_reg_t cpu_reg = ggml_backend_cpu_reg();
    CHECK(cpu_reg != nullptr);
    ggml_backend_dev_t cpu = cpu_reg != nullptr ? ggml_backend_reg_dev_get(cpu_reg, 0) : nullptr;
    CHECK(cpu != nullptr);
    if (cpu == nullptr) {
        return 1;
    }

    constexpr int kThreads = 4;
    constexpr int kIterations = 100;
    std::vector<fixture> fixtures(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        if (!make_fixture(cpu, i, fixtures[i])) {
            return 1;
        }
    }

    // Capture the concurrent diagnostic output; the dump thread and the
    // rejection threads share stderr, which the C library serializes. These
    // intermediate lines may show (none)/(pending) and are not asserted.
    FILE * concurrent = std::tmpfile();
    CHECK(concurrent != nullptr);
    if (concurrent == nullptr) {
        return 1;
    }
    const int saved = dup(fileno(stderr));
    CHECK(saved >= 0);
    dup2(fileno(concurrent), fileno(stderr));

    std::atomic<bool> start{false};
    std::atomic<bool> stop_dump{false};
    std::vector<std::thread> workers;
    workers.reserve(kThreads + 1);

    for (int i = 0; i < kThreads; ++i) {
        workers.emplace_back([&, i] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            std::vector<uint8_t> source(ggml_nbytes(fixtures[i].tensor), 0x5a);
            for (int n = 0; n < kIterations; ++n) {
                ggml_backend_tensor_set(fixtures[i].tensor, source.data(), 0, source.size());
            }
        });
    }
    workers.emplace_back([&] {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        // Dump while the rejection threads are running: no lock may be held
        // across the print and the readers must tolerate a not-yet-published
        // example.
        while (!stop_dump.load(std::memory_order_acquire)) {
            ggml_backend_meta_staging_diag_dump();
            std::this_thread::yield();
        }
    });

    start.store(true, std::memory_order_release);
    for (int i = 0; i < kThreads; ++i) {
        workers[i].join();
    }
    stop_dump.store(true, std::memory_order_release);
    workers[kThreads].join();

    // Stop the concurrent capture, then dump once with no writers: the count
    // is exact and the example must be READY.
    dup2(saved, fileno(stderr));
    std::fclose(concurrent);

    FILE * final_capture = std::tmpfile();
    CHECK(final_capture != nullptr);
    if (final_capture == nullptr) {
        return 1;
    }
    dup2(fileno(final_capture), fileno(stderr));
    ggml_backend_meta_staging_diag_dump();
    std::fflush(stderr);
    dup2(saved, fileno(stderr));
    close(saved);

    std::rewind(final_capture);
    std::vector<char> out(1u << 20);
    const size_t n = std::fread(out.data(), 1, out.size() - 1, final_capture);
    out[n] = '\0';
    std::fclose(final_capture);

    char expected_count[128];
    std::snprintf(expected_count, sizeof(expected_count),
                  "reason=no_input_flag count=%d", kThreads * kIterations);
    CHECK(std::strstr(out.data(), expected_count) != nullptr);
    CHECK(std::strstr(out.data(), "example_name=reject-concurrent-") != nullptr);
    CHECK(std::strstr(out.data(), "example_name=(none)") == nullptr);
    CHECK(std::strstr(out.data(), "example_name=(pending)") == nullptr);

    for (fixture & f : fixtures) {
        free_fixture(f);
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "TP5_STAGING_REJECT_CONCURRENT_TEST FAIL (%d)\n", g_failures);
        return 1;
    }
    std::printf("TP5_STAGING_REJECT_CONCURRENT_TEST PASS\n");
    return 0;
}
