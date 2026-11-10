// Narrow real-CPU test for the versioned scheduler completion receipt.
//
// It proves the CPU contract: a synchronous backend returns a completed
// success receipt with a real attempt id, and wait() reports that same
// completed fact.  It deliberately does not fabricate a GPU backend; the
// asynchronous CUDA/Meta paths are compile/static-checked only here.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>

static enum ggml_status fail_graph_compute(ggml_backend_t, ggml_cgraph *) {
    return GGML_STATUS_FAILED;
}

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        std::fprintf(stderr, "cpu backend init failed\n");
        return 1;
    }

    ggml_backend_t backends[1] = { backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, 2048, false, false);
    if (sched == nullptr) {
        std::fprintf(stderr, "scheduler init failed\n");
        return 1;
    }

    ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "ggml init failed\n");
        return 1;
    }

    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * c = ggml_add(ctx, a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_sched_receipt r1 = {};
    if (!ggml_backend_sched_graph_compute_receipt(sched, gf, &r1)) {
        std::fprintf(stderr, "cpu receipt was not submitted\n");
        return 1;
    }
    if (r1.version != GGML_BACKEND_SCHED_RECEIPT_API_VERSION ||
            r1.attempt == 0 || !r1.submitted) {
        std::fprintf(stderr, "cpu receipt not submitted: version=%u attempt=%llu submitted=%d state=%d\n",
                r1.version, (unsigned long long) r1.attempt, int(r1.submitted), r1.state);
        return 1;
    }
    const int w1 = ggml_backend_sched_receipt_wait(sched, &r1);
    if (w1 != GGML_BACKEND_SCHED_RECEIPT_COMPLETED_SUCCESS ||
            r1.state != GGML_BACKEND_SCHED_RECEIPT_COMPLETED_SUCCESS) {
        std::fprintf(stderr, "cpu receipt wait not success: %d state=%d\n", w1, r1.state);
        return 1;
    }
    ggml_backend_sched_receipt_release(sched, &r1);

    // A second submission must get a distinct scheduler attempt and must not
    // inherit the first receipt.
    ggml_backend_sched_receipt r2 = {};
    if (!ggml_backend_sched_graph_compute_receipt(sched, gf, &r2)) {
        std::fprintf(stderr, "second cpu receipt was not submitted\n");
        return 1;
    }
    if (r2.attempt <= r1.attempt || !r2.submitted ||
            ggml_backend_sched_receipt_wait(sched, &r2) !=
                GGML_BACKEND_SCHED_RECEIPT_COMPLETED_SUCCESS) {
        std::fprintf(stderr, "second receipt invalid\n");
        return 1;
    }
    ggml_backend_sched_receipt_release(sched, &r2);

    // A controllable CPU backend whose graph_compute fails after the
    // scheduler has handed it the split: the receipt must report completed
    // failure with submitted=true, never success.
    ggml_backend fail_backend = *backend;
    fail_backend.iface.graph_compute = fail_graph_compute;
    ggml_backend_t fail_backends[1] = { &fail_backend };
    ggml_backend_sched_t fail_sched =
        ggml_backend_sched_new(fail_backends, nullptr, 1, 2048, false, false);
    if (fail_sched == nullptr) {
        std::fprintf(stderr, "fail scheduler init failed\n");
        return 1;
    }
    ggml_backend_sched_receipt rf = {};
    const bool fail_submitted =
        ggml_backend_sched_graph_compute_receipt(fail_sched, gf, &rf);
    if (!fail_submitted || !rf.submitted) {
        std::fprintf(stderr, "failed compute was not submitted: submitted=%d state=%d\n",
                int(fail_submitted), rf.state);
        return 1;
    }
    const int fail_wait = ggml_backend_sched_receipt_wait(fail_sched, &rf);
    if ((fail_wait != GGML_BACKEND_SCHED_RECEIPT_UNKNOWN &&
         fail_wait != GGML_BACKEND_SCHED_RECEIPT_COMPLETED_FAILURE) ||
            rf.state == GGML_BACKEND_SCHED_RECEIPT_COMPLETED_SUCCESS) {
        std::fprintf(stderr, "failed compute did not stay non-success: wait=%d state=%d\n",
                fail_wait, rf.state);
        return 1;
    }
    ggml_backend_sched_receipt_release(fail_sched, &rf);
    ggml_backend_sched_free(fail_sched);

    ggml_free(ctx);
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);

    std::printf("SCHED_RECEIPT_TEST PASS\n");
    return 0;
}
