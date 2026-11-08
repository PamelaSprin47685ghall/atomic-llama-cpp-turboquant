// TP5 WS4 scheduler-boundary CPU test: success, rejected and fatal async
// submissions, no replay after a fatal, and copy-slot ownership.
//
// The test forces the graph operation onto a fake backend (index 0) and the
// graph inputs onto the CPU backend (last index). The scheduler must split
// at the operation and copy each input through the fake backend's
// cpy_tensor_async; this is the path the worker's earlier fixture missed when
// the ADD was assigned to the CPU and no async copy happened.
//
// Backends:
//   * fake-staging: device type GPU, supports ADD/NONE and the CPU buffer
//     type, implements cpy_tensor_async and the optional submission-error /
//     submission-pending capability.
//   * CPU: the real ggml CPU backend; the inputs live in its buffer.
//
// Build (CPU only, no GPU; build the ggml libs first):
/*
   g++ -std=c++17 -O2 -I ggml/include -I ggml/src \
       tests/test-tp5-staging-scheduler.cpp \
       -L build-ws4-sched-test/bin -Wl,-rpath,"$PWD/build-ws4-sched-test/bin" \
       -lggml -lggml-cpu -lggml-base -o /tmp/test-tp5-staging-scheduler
   /tmp/test-tp5-staging-scheduler
*/
//
// The capability is provided in both forms so the test compiles against the
// committed iface-field version and the registry-proc version:
//   * ggml_backend_i::submission_error, if that field exists in the header;
//   * "ggml_backend_submission_error"/"ggml_backend_submission_pending" via
//     the backend device registry proc-address hook.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "../ggml/src/ggml-backend-impl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            ++g_failures;                                                                \
        }                                                                                \
    } while (0)

// Detect the committed iface-field capability without requiring it.
template <typename T, typename = void>
struct has_submission_error : std::false_type {};
template <typename T>
struct has_submission_error<T, std::void_t<decltype(std::declval<T &>().submission_error)>>
    : std::true_type {};

constexpr bool iface_has_submission_error = has_submission_error<ggml_backend_i>::value;

// Assign the committed iface-field capability only when the header has it.
// The field access stays in a dependent context so the missing-field variant
// is discarded before instantiation.
template <typename BackendIface>
void set_iface_submission_error(BackendIface & iface, int (*fn)(ggml_backend_t)) {
    if constexpr (has_submission_error<BackendIface>::value) {
        iface.submission_error = fn;
    }
}

constexpr int kFatalError = 0x4b; // nonzero, stable

struct fake_state {
    int  submission_error = 0;
    bool submission_pending = false;
    bool reject_copy = false;

    int cpy_calls = 0;
    int error_calls = 0;
    int pending_calls = 0;
    int compute_calls = 0;
    int sync_calls = 0;

    const ggml_tensor * last_src = nullptr;
    ggml_tensor *       last_dst = nullptr;
};

fake_state g_fake;

ggml_backend_buffer_type g_fake_buft;

// Fake backend (stream) interface.
const char * fake_get_name(ggml_backend_t) { return "fake-staging"; }

void fake_free(ggml_backend_t) {}

void fake_synchronize(ggml_backend_t) { ++g_fake.sync_calls; }

enum ggml_status fake_graph_compute(ggml_backend_t, ggml_cgraph *) {
    ++g_fake.compute_calls;
    return GGML_STATUS_SUCCESS;
}

bool fake_cpy_tensor_async(ggml_backend_t, ggml_backend_t, const ggml_tensor * src, ggml_tensor * dst) {
    ++g_fake.cpy_calls;
    g_fake.last_src = src;
    g_fake.last_dst = dst;
    if (g_fake.reject_copy) {
        return false; // rejected before any copy: the scheduler may fall back
    }
    if (src->data != nullptr && dst->data != nullptr) {
        const size_t nbytes = ggml_nbytes(src) < ggml_nbytes(dst) ? ggml_nbytes(src) : ggml_nbytes(dst);
        if (g_fake.submission_error != 0) {
            // An accepted but unprovable submission: leave a recognizable
            // sentinel that no replay may change.
            std::memset(dst->data, 0xA5, nbytes);
        } else {
            std::memcpy(dst->data, src->data, nbytes);
        }
    }
    return true;
}

int fake_submission_error(ggml_backend_t) {
    ++g_fake.error_calls;
    return g_fake.submission_error;
}

bool fake_submission_pending(ggml_backend_t) {
    ++g_fake.pending_calls;
    return g_fake.submission_pending;
}

// Fake device interface.
const char * fake_dev_get_name(ggml_backend_dev_t) { return "fake-staging-dev"; }
const char * fake_dev_get_description(ggml_backend_dev_t) { return "fake staging scheduler device"; }

void fake_dev_get_memory(ggml_backend_dev_t, size_t * free_bytes, size_t * total_bytes) {
    if (free_bytes)  *free_bytes  = 0;
    if (total_bytes) *total_bytes = 0;
}

enum ggml_backend_dev_type fake_dev_get_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }

void fake_dev_get_props(ggml_backend_dev_t, ggml_backend_dev_props * props) {
    std::memset(props, 0, sizeof(*props));
    props->name        = "fake-staging-dev";
    props->description = "fake staging scheduler device";
    props->type        = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->caps.async  = true;
}

bool fake_dev_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    return op->op == GGML_OP_ADD || op->op == GGML_OP_NONE;
}

bool fake_dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) {
    // The fake device supports only its own buffer type, never the CPU buffer
    // type. If it claims the CPU buffer, the scheduler uses the CPU input in
    // place and never creates an input copy; cpy_tensor_async is then never
    // called. This is the trap in the worker's earlier fixture: forcing the op
    // backend is not enough, the input buffer must be recognized as foreign.
    return buft == &g_fake_buft;
}

bool fake_dev_offload_op(ggml_backend_dev_t, const ggml_tensor *) { return true; }

ggml_backend_buffer_type_t fake_dev_get_buffer_type(ggml_backend_dev_t) {
    return &g_fake_buft;
}

// A host buffer type owned by the fake device. It is deliberately distinct
// from the CPU buffer type: the scheduler must classify a CPU input as foreign
// to this backend and create a copy on this buffer. If the fake device claims
// the CPU buffer type, the scheduler uses the input in place and never calls
// cpy_tensor_async -- the trap in the earlier fixture.
const char * fake_buft_get_name(ggml_backend_buffer_type_t) { return "fake-staging-buft"; }

size_t fake_buft_get_alignment(ggml_backend_buffer_type_t) { return 64; }
size_t fake_buft_get_max_size(ggml_backend_buffer_type_t) { return SIZE_MAX; }
size_t fake_buft_get_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * tensor) { return ggml_nbytes(tensor); }
bool fake_buft_is_host(ggml_backend_buffer_type_t) { return true; }

void fake_buffer_free(ggml_backend_buffer_t buffer) { std::free(buffer->context); }
void * fake_buffer_get_base(ggml_backend_buffer_t buffer) { return buffer->context; }

void fake_buffer_memset(ggml_backend_buffer_t, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    std::memset((char *) tensor->data + offset, value, size);
}

void fake_buffer_set(ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    std::memcpy((char *) tensor->data + offset, data, size);
}

void fake_buffer_get(ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    std::memcpy(data, (const char *) tensor->data + offset, size);
}

bool fake_buffer_cpy(ggml_backend_buffer_t, const ggml_tensor * src, ggml_tensor * dst) {
    std::memcpy(dst->data, src->data, ggml_nbytes(src));
    return true;
}

void fake_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    std::memset(buffer->context, value, buffer->size);
}

ggml_backend_buffer_i fake_buffer_iface() {
    ggml_backend_buffer_i iface = {};
    iface.free_buffer   = fake_buffer_free;
    iface.get_base      = fake_buffer_get_base;
    iface.memset_tensor = fake_buffer_memset;
    iface.set_tensor    = fake_buffer_set;
    iface.get_tensor    = fake_buffer_get;
    iface.cpy_tensor    = fake_buffer_cpy;
    iface.clear         = fake_buffer_clear;
    return iface;
}

ggml_backend_buffer_t fake_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = std::malloc(size != 0 ? size : 1);
    if (ptr == nullptr) {
        return nullptr;
    }
    return ggml_backend_buffer_init(buft, fake_buffer_iface(), ptr, size);
}

// Fake registry interface: the scheduler looks up the optional submission
// capability through the device registry's proc-address hook.
const char * fake_reg_get_name(ggml_backend_reg_t) { return "fake-staging-reg"; }

size_t fake_reg_get_device_count(ggml_backend_reg_t) { return 1; }

ggml_backend_dev_t fake_reg_get_device(ggml_backend_reg_t, size_t index);

void * fake_reg_get_proc_address(ggml_backend_reg_t, const char * name) {
    if (name == nullptr) {
        return nullptr;
    }
    if (std::strcmp(name, "ggml_backend_submission_error") == 0) {
        return reinterpret_cast<void *>(+fake_submission_error);
    }
    if (std::strcmp(name, "ggml_backend_submission_pending") == 0) {
        return reinterpret_cast<void *>(+fake_submission_pending);
    }
    return nullptr;
}

ggml_backend_reg    g_fake_reg;
ggml_backend_device g_fake_device;
ggml_backend        g_fake_backend;

ggml_backend_dev_t fake_reg_get_device(ggml_backend_reg_t, size_t index) {
    return index == 0 ? &g_fake_device : nullptr;
}

void init_fake_backend() {
    g_fake_reg = {};
    g_fake_reg.api_version = GGML_BACKEND_API_VERSION;
    g_fake_reg.iface.get_name         = fake_reg_get_name;
    g_fake_reg.iface.get_device_count = fake_reg_get_device_count;
    g_fake_reg.iface.get_device       = fake_reg_get_device;
    g_fake_reg.iface.get_proc_address = fake_reg_get_proc_address;

    g_fake_device = {};
    g_fake_device.iface.get_name         = fake_dev_get_name;
    g_fake_device.iface.get_description  = fake_dev_get_description;
    g_fake_device.iface.get_memory       = fake_dev_get_memory;
    g_fake_device.iface.get_type         = fake_dev_get_type;
    g_fake_device.iface.get_props        = fake_dev_get_props;
    g_fake_device.iface.get_buffer_type  = fake_dev_get_buffer_type;
    g_fake_device.iface.supports_op      = fake_dev_supports_op;
    g_fake_device.iface.supports_buft    = fake_dev_supports_buft;
    g_fake_device.iface.offload_op       = fake_dev_offload_op;
    g_fake_device.reg                    = &g_fake_reg;

    g_fake_buft = {};
    g_fake_buft.iface.get_name       = fake_buft_get_name;
    g_fake_buft.iface.alloc_buffer   = fake_buft_alloc_buffer;
    g_fake_buft.iface.get_alignment  = fake_buft_get_alignment;
    g_fake_buft.iface.get_max_size   = fake_buft_get_max_size;
    g_fake_buft.iface.get_alloc_size = fake_buft_get_alloc_size;
    g_fake_buft.iface.is_host        = fake_buft_is_host;
    g_fake_buft.device               = &g_fake_device;

    g_fake_backend = {};
    g_fake_backend.guid = nullptr;
    g_fake_backend.iface.get_name          = fake_get_name;
    g_fake_backend.iface.free              = fake_free;
    g_fake_backend.iface.cpy_tensor_async  = fake_cpy_tensor_async;
    g_fake_backend.iface.synchronize       = fake_synchronize;
    g_fake_backend.iface.graph_compute     = fake_graph_compute;
    g_fake_backend.device  = &g_fake_device;
    g_fake_backend.context = &g_fake;
    set_iface_submission_error(g_fake_backend.iface, fake_submission_error);
}

// One scheduler fixture: a CPU input pair, an ADD on the fake backend, and
// the scheduler's own input copies.
struct fixture {
    ggml_backend_t        cpu   = nullptr;
    ggml_backend_sched_t  sched = nullptr;
    ggml_context *        ctx   = nullptr;
    ggml_cgraph *         graph = nullptr;
    ggml_tensor *         a     = nullptr;
    ggml_tensor *         b     = nullptr;
    ggml_tensor *         out   = nullptr;
    ggml_backend_buffer_t input_buf = nullptr;
};

bool fixture_init(fixture & f) {
    f.cpu = ggml_backend_cpu_init();
    if (f.cpu == nullptr) {
        return false;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 1u << 20,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    f.ctx = ggml_init(params);
    if (f.ctx == nullptr) {
        return false;
    }

    f.a   = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, 8);
    f.b   = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, 8);
    f.out = ggml_add(f.ctx, f.a, f.b);
    if (f.a == nullptr || f.b == nullptr || f.out == nullptr) {
        return false;
    }
    ggml_set_input(f.a);
    ggml_set_input(f.b);

    f.graph = ggml_new_graph(f.ctx);
    ggml_build_forward_expand(f.graph, f.out);

    f.input_buf = ggml_backend_alloc_ctx_tensors(f.ctx, f.cpu);
    if (f.input_buf == nullptr) {
        return false;
    }

    ggml_backend_t backends[2] = { &g_fake_backend, f.cpu };
    ggml_backend_buffer_type_t bufts[2] = {
        &g_fake_buft,
        ggml_backend_cpu_buffer_type(),
    };
    f.sched = ggml_backend_sched_new(backends, bufts, 2, ggml_graph_size(f.graph), false, false);
    if (f.sched == nullptr) {
        return false;
    }

    // Force the split: inputs on the CPU, ADD on the fake backend. Without
    // these assignments the ADD's sources make the scheduler choose the CPU
    // backend, and cpy_tensor_async is never called.
    ggml_backend_sched_set_tensor_backend(f.sched, f.a,   f.cpu);
    ggml_backend_sched_set_tensor_backend(f.sched, f.b,   f.cpu);
    ggml_backend_sched_set_tensor_backend(f.sched, f.out, &g_fake_backend);

    if (!ggml_backend_sched_alloc_graph(f.sched, f.graph)) {
        return false;
    }

    // The assignments must survive the split.
    CHECK(ggml_backend_sched_get_tensor_backend(f.sched, f.out) == &g_fake_backend);
    CHECK(ggml_backend_sched_get_tensor_backend(f.sched, f.a) == f.cpu);
    CHECK(ggml_backend_sched_get_tensor_backend(f.sched, f.b) == f.cpu);

    std::vector<float> av(8), bv(8);
    for (size_t i = 0; i < av.size(); ++i) {
        av[i] = float(i) + 1.0f;
        bv[i] = float(i) + 101.0f;
    }
    ggml_backend_tensor_set(f.a, av.data(), 0, av.size() * sizeof(float));
    ggml_backend_tensor_set(f.b, bv.data(), 0, bv.size() * sizeof(float));

    return true;
}

void fixture_free(fixture & f, bool retain_scheduler) {
    if (f.sched) {
        ggml_backend_sched_free(f.sched); // retained when a fatal submission is pending
        f.sched = nullptr;
    }
    if (!retain_scheduler) {
        if (f.input_buf) {
            ggml_backend_buffer_free(f.input_buf);
            f.input_buf = nullptr;
        }
        if (f.ctx) {
            ggml_free(f.ctx);
            f.ctx = nullptr;
        }
    }
    if (f.cpu) {
        ggml_backend_free(f.cpu);
        f.cpu = nullptr;
    }
}

void reset_fake() {
    g_fake = {};
}

// Success: cpy_tensor_async accepts every input, submission_error is 0, the
// scheduler completes and the copies carry the source bytes.
void test_success() {
    reset_fake();
    fixture f;
    CHECK(fixture_init(f));

    const int cpy_before = g_fake.cpy_calls;
    const enum ggml_status st = ggml_backend_sched_graph_compute(f.sched, f.graph);
    CHECK(st == GGML_STATUS_SUCCESS);
    CHECK(g_fake.cpy_calls == cpy_before + 2); // both ADD inputs
    CHECK(g_fake.error_calls == 2);
    CHECK(g_fake.compute_calls == 1);
    CHECK(g_fake.submission_error == 0);

    // The fake copied the first source into the input copy.
    CHECK(g_fake.last_dst != nullptr);
    if (g_fake.last_dst && g_fake.last_dst->data && g_fake.last_src && g_fake.last_src->data) {
        CHECK(std::memcmp(g_fake.last_dst->data, g_fake.last_src->data, 8 * sizeof(float)) == 0);
    }

    // A second compute is a normal reuse, not a retained fatal state.
    CHECK(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS);
    CHECK(g_fake.cpy_calls == cpy_before + 4);

    fixture_free(f, false);
}

// Rejected: cpy_tensor_async returns false without a runtime copy; the
// scheduler must fall back to the synchronous copy and must not ask for a
// submission error. The destination still receives the source bytes.
void test_rejected() {
    reset_fake();
    g_fake.reject_copy = true;
    fixture f;
    CHECK(fixture_init(f));

    const int cpy_before = g_fake.cpy_calls;
    const enum ggml_status st = ggml_backend_sched_graph_compute(f.sched, f.graph);
    CHECK(st == GGML_STATUS_SUCCESS);
    CHECK(g_fake.cpy_calls >= cpy_before + 1);
    CHECK(g_fake.error_calls == 0); // only asked after an accepted copy
    CHECK(g_fake.compute_calls == 1);

    // The rejected call left last_dst untouched; the scheduler's fallback
    // filled the same destination with the same source bytes.
    CHECK(g_fake.last_dst != nullptr);
    if (g_fake.last_dst && g_fake.last_dst->data && g_fake.last_src && g_fake.last_src->data) {
        CHECK(std::memcmp(g_fake.last_dst->data, g_fake.last_src->data, 8 * sizeof(float)) == 0);
    }

    fixture_free(f, false);
}

// Fatal: cpy_tensor_async accepts but submission_error is nonzero. The
// scheduler must return FAILED, keep the destination copy slot, and never
// replay or overwrite it on reset/reserve/next compute.
void test_fatal() {
    reset_fake();
    g_fake.submission_error = kFatalError;
    g_fake.submission_pending = true;
    fixture f;
    CHECK(fixture_init(f));

    const enum ggml_status st = ggml_backend_sched_graph_compute(f.sched, f.graph);
    CHECK(st == GGML_STATUS_FAILED);
    CHECK(g_fake.cpy_calls == 1);   // stopped at the first input
    CHECK(g_fake.error_calls == 1);
    CHECK(g_fake.compute_calls == 0);
    CHECK(g_fake.last_dst != nullptr);

    // The fake wrote the sentinel into the accepted destination.
    std::vector<unsigned char> sentinel;
    if (g_fake.last_dst && g_fake.last_dst->data) {
        const unsigned char * p = static_cast<const unsigned char *>(g_fake.last_dst->data);
        sentinel.assign(p, p + 8 * sizeof(float));
    }

    // Reset and reserve are refused while the fault is pending.
    ggml_backend_sched_reset(f.sched);
    CHECK(!ggml_backend_sched_reserve(f.sched, f.graph));

    // No replay: the next compute fails before any copy or graph submission.
    const int cpy_after = g_fake.cpy_calls;
    const int err_after = g_fake.error_calls;
    const int comp_after = g_fake.compute_calls;
    CHECK(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_FAILED);
    CHECK(g_fake.cpy_calls == cpy_after);
    CHECK(g_fake.error_calls == err_after);
    CHECK(g_fake.compute_calls == comp_after);

    // The destination still holds the sentinel: the scheduler did not replay,
    // flush or free the copy slot while the submission is unproven.
    if (g_fake.last_dst && g_fake.last_dst->data) {
        CHECK(std::memcmp(g_fake.last_dst->data, sentinel.data(), sentinel.size()) == 0);
    }

    // The current registry-proc scheduler also retains the copy slots in
    // ggml_backend_sched_free when submission_pending is true. The committed
    // iface-field version has no pending hook, so it may free them; do not
    // read the destination after free in that variant.
    const bool retains_on_free = !iface_has_submission_error;
    fixture_free(f, retains_on_free);
    if (retains_on_free) {
        CHECK(g_fake.pending_calls >= 1);
        if (g_fake.last_dst && g_fake.last_dst->data) {
            CHECK(std::memcmp(g_fake.last_dst->data, sentinel.data(), sentinel.size()) == 0);
        }
    }
}

} // namespace

int main() {
    init_fake_backend();

    test_success();
    test_rejected();
    test_fatal();

    if (g_failures != 0) {
        std::fprintf(stderr, "TP5_STAGING_SCHED_TEST FAIL failures=%d\n", g_failures);
        return 1;
    }
    std::printf("TP5_STAGING_SCHED_TEST PASS\n");
    return 0;
}
