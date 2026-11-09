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
#include "../ggml/src/ggml-backend-moe-cache.h"

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
    // Publish an unproven submission after an accepted copy, with no native
    // error. This is a real Meta state: a staging pool can be disabled before
    // it records first_error.
    bool pending_on_accept = false;
    // Publish a native error after an accepted copy. The scheduler's entry
    // gate therefore sees a clean backend; only the copy observation does.
    bool error_on_accept = false;
    bool compute_fail = false;
    // Publish pending during graph_compute. With compute_fail=false this
    // models a successful compute whose result is still unproven.
    bool compute_pending = false;

    int cpy_calls = 0;
    int error_calls = 0;
    int pending_calls = 0;
    int compute_calls = 0;
    int sync_calls = 0;

    const ggml_tensor * last_src = nullptr;
    ggml_tensor *       last_dst = nullptr;

    bool backend_freed = false;
    int  pending_after_free = 0; // submission_pending called after iface.free
};

fake_state g_fake;

ggml_backend_buffer_type g_fake_buft;

// Fake backend (stream) interface.
const char * fake_get_name(ggml_backend_t) { return "fake-staging"; }

void fake_free(ggml_backend_t) { g_fake.backend_freed = true; }

void fake_synchronize(ggml_backend_t) { ++g_fake.sync_calls; }

enum ggml_status fake_graph_compute(ggml_backend_t, ggml_cgraph *) {
    ++g_fake.compute_calls;
    if (g_fake.compute_pending) {
        g_fake.submission_pending = true;
    }
    if (g_fake.compute_fail) {
        return GGML_STATUS_FAILED;
    }
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
    if (g_fake.pending_on_accept) {
        g_fake.submission_pending = true;
    }
    if (g_fake.error_on_accept) {
        g_fake.submission_error = kFatalError;
    }
    return true;
}

int fake_submission_error(ggml_backend_t) {
    ++g_fake.error_calls;
    return g_fake.submission_error;
}

bool fake_submission_pending(ggml_backend_t) {
    ++g_fake.pending_calls;
    if (g_fake.backend_freed) {
        // The scheduler must never query a backend the owner already freed.
        ++g_fake.pending_after_free;
    }
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

size_t fake_buft_get_alignment(ggml_backend_buffer_type_t) { return 16; }
size_t fake_buft_get_max_size(ggml_backend_buffer_type_t) { return SIZE_MAX; }
size_t fake_buft_get_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * tensor) { return GGML_PAD(ggml_nbytes(tensor), 16); }
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

// Fake MoE cache provider. The scheduler probes providers with a null config
// at construction; only an explicit ggml_backend_sched_set_moe_cache creates
// a session. It records creation/destruction so the pending gate is visible
// from outside the scheduler.
int g_moe_creates = 0;
int g_moe_destroys = 0;
int g_moe_enters = 0;
int g_moe_leaves = 0;
int g_moe_session_marker = 0;

int fake_moe_query_config(int, size_t budget_mib, ggml_moe_cache_config * config) {
    if (config == nullptr) {
        return 0;
    }
    std::memset(config, 0, sizeof(*config));
    config->budget_bytes = budget_mib * 1024 * 1024;
    config->min_devices  = 1;
    return 1;
}

void * fake_moe_session_create(void * const *, int, const ggml_moe_cache_config * config) {
    if (config == nullptr) {
        return nullptr; // construction probe: leave the scheduler without a session
    }
    ++g_moe_creates;
    return &g_moe_session_marker;
}

void fake_moe_session_destroy(void *) { ++g_moe_destroys; }
void fake_moe_session_enter(void *) { ++g_moe_enters; }
void fake_moe_session_leave(void *) { ++g_moe_leaves; }

void register_fake_moe_provider() {
    ggml_moe_cache_api api = {};
    api.owner            = &g_fake_reg;
    api.query_config     = fake_moe_query_config;
    api.session_create   = fake_moe_session_create;
    api.session_destroy  = fake_moe_session_destroy;
    api.session_enter    = fake_moe_session_enter;
    api.session_leave    = fake_moe_session_leave;
    ggml_moe_cache_register(&api);
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

bool fixture_init(fixture & f, bool track_copies = false) {
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
    if (track_copies && !ggml_backend_sched_set_split_copy_tracking(f.sched, true)) {
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
    const int err_before = g_fake.error_calls;
    const enum ggml_status st = ggml_backend_sched_graph_compute(f.sched, f.graph);
    CHECK(st == GGML_STATUS_SUCCESS);
    CHECK(g_fake.cpy_calls == cpy_before + 2); // both ADD inputs
    // Two submission-state gates (async entry, sync-wrapper exit) plus one
    // query after each accepted copy.
    CHECK(g_fake.error_calls == err_before + 4);
    CHECK(g_fake.compute_calls == 1);
    CHECK(g_fake.submission_error == 0);

    // The fake copied the first source into the input copy.
    CHECK(g_fake.last_dst != nullptr);
    if (g_fake.last_dst && g_fake.last_dst->data && g_fake.last_src && g_fake.last_src->data) {
        CHECK(std::memcmp(g_fake.last_dst->data, g_fake.last_src->data, 8 * sizeof(float)) == 0);
    }

    // A second compute is a normal reuse, not a retained fatal state.
    const int cpy_second = g_fake.cpy_calls;
    const int err_second = g_fake.error_calls;
    CHECK(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS);
    CHECK(g_fake.cpy_calls == cpy_second + 2);
    CHECK(g_fake.error_calls == err_second + 4);

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
    const int err_before = g_fake.error_calls;
    const enum ggml_status st = ggml_backend_sched_graph_compute(f.sched, f.graph);
    CHECK(st == GGML_STATUS_SUCCESS);
    CHECK(g_fake.cpy_calls >= cpy_before + 1);
    // No accepted copy, so only the entry and sync-wrapper gates asked.
    CHECK(g_fake.error_calls == err_before + 2);
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
    fixture f;
    CHECK(fixture_init(f));
    // Publish the fault after allocation: an error observed before the graph
    // is allocated makes alloc_graph refuse (the entry gate covers it).
    g_fake.error_on_accept = true;

    const int err_before = g_fake.error_calls;
    const enum ggml_status st = ggml_backend_sched_graph_compute(f.sched, f.graph);
    CHECK(st == GGML_STATUS_FAILED);
    CHECK(g_fake.cpy_calls == 1);   // stopped at the first input
    // The entry gate and the accepted copy observed the native error.
    CHECK(g_fake.error_calls == err_before + 2);
    CHECK(g_fake.compute_calls == 0);
    CHECK(g_fake.last_dst != nullptr);

    // The accepted destination bytes, which no later path may change.
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
    g_fake.submission_pending = true; // observation after the fault, not before alloc
    fixture_free(f, retains_on_free);
    if (retains_on_free) {
        if (g_fake.last_dst && g_fake.last_dst->data) {
            CHECK(std::memcmp(g_fake.last_dst->data, sentinel.data(), sentinel.size()) == 0);
        }
    }
}

// A pending submission may exist even when this scheduler did not observe a
// failure (e.g. a Meta reduce-path copy or a process-global staging fault).
// ggml_backend_sched_free must retain the copy slots in that case too.
void test_pending_without_failure() {
    if (iface_has_submission_error) {
        // The older iface-field scheduler predates the any-pending free path.
        return;
    }
    reset_fake();
    g_fake.submission_error  = 0;
    fixture f;
    CHECK(fixture_init(f));

    const int err_before = g_fake.error_calls;
    CHECK(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS);
    CHECK(g_fake.error_calls == err_before + 4);
    CHECK(g_fake.last_dst != nullptr);
    g_fake.submission_pending = true; // generic pending observed after a clean compute
    CHECK(ggml_backend_sched_has_unproven(f.sched));

    std::vector<unsigned char> copied;
    if (g_fake.last_dst && g_fake.last_dst->data) {
        const unsigned char * p = static_cast<const unsigned char *>(g_fake.last_dst->data);
        copied.assign(p, p + 8 * sizeof(float));
    }

    // has_unproven() cached the pending state; sched_free must retain using
    // that cache without querying the backends again.
    fixture_free(f, true);
    if (g_fake.last_dst && g_fake.last_dst->data && !copied.empty()) {
        CHECK(std::memcmp(g_fake.last_dst->data, copied.data(), copied.size()) == 0);
    }
}

// Public helper contract: true for ordered/synchronous copies (including a
// rejected async callback completed by the blocking fallback), false ONLY for
// the accepted-but-unprovable case.
void test_public_copy_async_return() {
    reset_fake();

    ggml_init_params params = { /*.mem_size=*/ 1u << 16, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);
    ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    CHECK(src != nullptr && dst != nullptr);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, &g_fake_buft);
    CHECK(buf != nullptr);

    const float values[4] = { 3.0f, 5.0f, 7.0f, 11.0f };
    ggml_backend_tensor_set(src, values, 0, sizeof(values));

    // Ordered/accepted -> true.
    g_fake.submission_error = 0;
    g_fake.reject_copy = false;
    CHECK(ggml_backend_tensor_copy_async(&g_fake_backend, &g_fake_backend, src, dst));
    CHECK(std::memcmp(dst->data, values, sizeof(values)) == 0);

    // A no-op copy performs no callback work and remains true while clean.
    const int self_cpy_before = g_fake.cpy_calls;
    CHECK(ggml_backend_tensor_copy_async(&g_fake_backend, &g_fake_backend, dst, dst));
    CHECK(g_fake.cpy_calls == self_cpy_before);

    // Accepted but unprovable -> false; the destination must not be replayed.
    g_fake.submission_error = kFatalError;
    const int errors_before = g_fake.error_calls;
    CHECK(!ggml_backend_tensor_copy_async(&g_fake_backend, &g_fake_backend, src, dst));
    CHECK(g_fake.error_calls == errors_before + 1);
    std::vector<unsigned char> unproven;
    if (dst->data) {
        const unsigned char * p = static_cast<const unsigned char *>(dst->data);
        unproven.assign(p, p + sizeof(values));
    }

    // The pre-check runs before the self-copy shortcut: an unproven backend
    // returns false even when no copy would be queued. True is not a completion
    // proof for the earlier submission.
    CHECK(!ggml_backend_tensor_copy_async(&g_fake_backend, &g_fake_backend, dst, dst));

    // The state is sticky: a second call must refuse before invoking the
    // callback, so the unproven destination bytes are not overwritten by a
    // blocking fallback and no new copy is queued.
    const int unproven_cpy_before = g_fake.cpy_calls;
    g_fake.reject_copy = true;
    CHECK(!ggml_backend_tensor_copy_async(&g_fake_backend, &g_fake_backend, src, dst));
    CHECK(g_fake.cpy_calls == unproven_cpy_before);
    CHECK(std::memcmp(dst->data, unproven.data(), unproven.size()) == 0);
    g_fake.reject_copy = false;

    // Accepted but pending with no native error -> also false. This is the
    // state a disabled staging pool can report before it records an error.
    g_fake.submission_error = 0;
    g_fake.pending_on_accept = true;
    CHECK(!ggml_backend_tensor_copy_async(&g_fake_backend, &g_fake_backend, src, dst));
    g_fake.pending_on_accept = false;
    g_fake.submission_pending = false;

    // Rejected async callback -> the helper performs the blocking copy and
    // returns true. The false return above is not a rejected copy.
    g_fake.submission_error = 0;
    g_fake.reject_copy = true;
    CHECK(ggml_backend_tensor_copy_async(&g_fake_backend, &g_fake_backend, src, dst));
    CHECK(std::memcmp(dst->data, values, sizeof(values)) == 0);

    // No async callback: the helper synchronizes and performs its own blocking
    // copy. The pre/post capability checks must still return true when clean.
    g_fake.reject_copy = false;
    g_fake_backend.iface.cpy_tensor_async = nullptr;
    const int fallback_cpy_before = g_fake.cpy_calls;
    CHECK(ggml_backend_tensor_copy_async(&g_fake_backend, &g_fake_backend, src, dst));
    CHECK(g_fake.cpy_calls == fallback_cpy_before);
    CHECK(std::memcmp(dst->data, values, sizeof(values)) == 0);
    g_fake_backend.iface.cpy_tensor_async = fake_cpy_tensor_async;

    // The later arms clear the fake capability only to exercise the remaining
    // return paths. That clear is not a production release operation: while
    // the capability stays set, the helper refuses before any callback.

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// An accepted copy that publishes pending without a native error must stop the
// split at once: the destination is unprovable, so no later input may overwrite
// it and no compute may consume it.
void test_pending_accept_stops_split() {
    reset_fake();
    fixture f;
    CHECK(fixture_init(f));
    g_fake.pending_on_accept = true;

    CHECK(ggml_backend_sched_graph_compute_async(f.sched, f.graph) == GGML_STATUS_FAILED);
    CHECK(g_fake.cpy_calls == 1);   // stopped at the first accepted copy
    CHECK(g_fake.compute_calls == 0);
    CHECK(g_fake.submission_error == 0);
    CHECK(ggml_backend_sched_has_unproven(f.sched));

    fixture_free(f, true);
}

static bool fake_eval_callback(ggml_tensor *, bool ask, void *) {
    return ask;
}

// A failed split compute on the callback path may be the first observation of
// a staging fault. It must publish the fail-stop state just like the normal
// split path.
void test_callback_failed_split_pending() {
    reset_fake();
    fixture f;
    CHECK(fixture_init(f));
    ggml_backend_sched_set_eval_callback(f.sched, fake_eval_callback, nullptr);
    g_fake.compute_fail = true;
    g_fake.compute_pending = true;

    CHECK(ggml_backend_sched_graph_compute_async(f.sched, f.graph) == GGML_STATUS_FAILED);
    CHECK(g_fake.compute_calls == 1);
    CHECK(g_fake.submission_error == 0);
    CHECK(ggml_backend_sched_has_unproven(f.sched));

    fixture_free(f, true);
}

// A native error with no pending flag is still unproven for this scheduler.
// any_pending must query the error capability and cache it, not only pending.
void test_error_only_gate() {
    reset_fake();
    fixture f;
    CHECK(fixture_init(f));
    g_fake.submission_error = kFatalError;
    g_fake.submission_pending = false;

    CHECK(ggml_backend_sched_has_unproven(f.sched));
    CHECK(!ggml_backend_sched_alloc_graph(f.sched, f.graph));
    CHECK(!ggml_backend_sched_reserve(f.sched, f.graph));

    fixture_free(f, true);
}

// The synchronous wrapper must not turn an unproven submission into
// completion. It skips the synchronize call and never returns SUCCESS.
void test_sync_wrapper_no_silent_success() {
    {
        reset_fake();
        fixture f;
        CHECK(fixture_init(f));
        g_fake.compute_fail = true;
        g_fake.compute_pending = true;
        const int sync_before = g_fake.sync_calls;

        CHECK(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_FAILED);
        CHECK(g_fake.compute_calls == 1);
        // Slot wait + input-copy flush, and no third synchronize from a
        // wrapper that claims completion.
        CHECK(g_fake.sync_calls == sync_before + 2);
        CHECK(ggml_backend_sched_has_unproven(f.sched));
        fixture_free(f, true);
    }
    {
        reset_fake();
        fixture f;
        CHECK(fixture_init(f));
        // The graph compute itself succeeds but leaves a pending submission.
        // The async call reports success; the synchronous wrapper must not.
        g_fake.compute_pending = true;
        const int sync_before = g_fake.sync_calls;

        CHECK(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_FAILED);
        CHECK(g_fake.compute_calls == 1);
        CHECK(g_fake.sync_calls == sync_before + 2);
        CHECK(ggml_backend_sched_has_unproven(f.sched));
        fixture_free(f, true);
    }
    {
        // Retained inputs defer the copy flush until a successful submission.
        // A failed split that reveals an unknown submission must not drain in
        // the scope destructor either.
        reset_fake();
        fixture f;
        CHECK(fixture_init(f));
        g_fake.compute_fail = true;
        g_fake.compute_pending = true;
        const int sync_before = g_fake.sync_calls;

        CHECK(ggml_backend_sched_graph_compute_async_retained_inputs(f.sched, f.graph) == GGML_STATUS_FAILED);
        CHECK(g_fake.compute_calls == 1);
        // Only the pre-copy slot wait. The failed compute turned the state
        // unproven, so the scope kept the upload ordered and did not flush.
        CHECK(g_fake.sync_calls == sync_before + 1);
        CHECK(ggml_backend_sched_has_unproven(f.sched));
        fixture_free(f, true);
    }
}

// update_split_copies and refresh_split_copy_views mutate live copy metadata.
// Both must refuse while a destination is unproven; refresh is observable
// through the dependency view returned by the diagnostic accessor.
void test_update_and_refresh_pending_gate() {
    reset_fake();
    fixture f;
    CHECK(fixture_init(f, true));
    CHECK(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_sched_tensor_has_split_copy(f.sched, f.a));

    ggml_backend_sched_tensor_layout layout = {};
    layout.tensor = f.a;
    std::memcpy(layout.ne, f.a->ne, sizeof(layout.ne));
    std::memcpy(layout.nb, f.a->nb, sizeof(layout.nb));
    CHECK(ggml_backend_sched_update_split_copies(f.sched, &layout, 1, false));
    CHECK(ggml_backend_sched_update_split_copies(f.sched, &layout, 1, true));

    ggml_tensor * dep = ggml_backend_sched_debug_copy_dependency(f.sched, f.a);
    CHECK(dep != nullptr);
    if (dep != nullptr) {
        // The clean path really refreshes: mutate the source width, then check
        // the dependency view followed it.
        f.a->ne[0] = 4;
        ggml_backend_sched_refresh_split_copy_views(f.sched);
        CHECK(dep->ne[0] == 4);
        CHECK(dep->data == f.a->data);
        CHECK(dep->view_src == (f.a->view_src ? f.a->view_src : f.a));
        f.a->ne[0] = 8;
        ggml_backend_sched_refresh_split_copy_views(f.sched);
        CHECK(dep->ne[0] == 8);

        const ggml_tensor dep_before = *dep;
        g_fake.submission_pending = true;
        CHECK(ggml_backend_sched_has_unproven(f.sched));

        f.a->ne[0] = 2;
        CHECK(!ggml_backend_sched_update_split_copies(f.sched, &layout, 1, false));
        CHECK(!ggml_backend_sched_update_split_copies(f.sched, &layout, 1, true));
        ggml_backend_sched_refresh_split_copy_views(f.sched);
        CHECK(dep->ne[0] == 8);
        CHECK(std::memcmp(dep, &dep_before, sizeof(*dep)) == 0);
        CHECK(f.a->ne[0] == 2); // the source itself was not touched
    }

    fixture_free(f, true);
}

// A MoE cache session may own provider resources that an unproven submission
// still uses. set_moe_cache must neither destroy nor replace it.
void test_set_moe_cache_pending_gate() {
    reset_fake();
    g_moe_creates = g_moe_destroys = g_moe_enters = g_moe_leaves = 0;
    fixture f;
    CHECK(fixture_init(f));

    ggml_backend_sched_set_moe_cache(f.sched, GGML_MOE_CACHE_MODE_ON, 16);
    CHECK(g_moe_creates == 1 && g_moe_destroys == 0);
    CHECK(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS);
    CHECK(g_moe_enters == 1 && g_moe_leaves == 1);

    // Clean switch-off works: the session is destroyed.
    ggml_backend_sched_set_moe_cache(f.sched, GGML_MOE_CACHE_MODE_OFF, 0);
    CHECK(g_moe_destroys == 1);
    ggml_backend_sched_set_moe_cache(f.sched, GGML_MOE_CACHE_MODE_ON, 16);
    CHECK(g_moe_creates == 2 && g_moe_destroys == 1);

    g_fake.submission_pending = true;
    CHECK(ggml_backend_sched_has_unproven(f.sched));
    ggml_backend_sched_set_moe_cache(f.sched, GGML_MOE_CACHE_MODE_OFF, 0);
    CHECK(g_moe_destroys == 1);              // refused: no destroy
    ggml_backend_sched_set_moe_cache(f.sched, GGML_MOE_CACHE_MODE_AUTO, 16);
    CHECK(g_moe_creates == 2);               // refused: no replacement

    fixture_free(f, true);
}

// replan_graph replaces the graph arena. It must refuse while unproven and
// leave every tensor address, layout and plan size untouched. The first
// replan proves the same call succeeds on a clean scheduler.
void test_replan_pending_gate() {
    ggml_init_params gp = { /*.mem_size=*/ 4u << 20, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    ggml_init_params fp = { /*.mem_size=*/ 4096,     /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ false };
    ggml_context * ctx = ggml_init(gp);
    ggml_context * fixed_ctx = ggml_init(fp);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    CHECK(ctx != nullptr && fixed_ctx != nullptr && cpu != nullptr);
    if (ctx == nullptr || fixed_ctx == nullptr || cpu == nullptr) {
        return;
    }

    ggml_tensor * weights = ggml_new_tensor_2d(fixed_ctx, GGML_TYPE_F32, 8, 1);
    CHECK(weights != nullptr);
    if (weights == nullptr || weights->data == nullptr) {
        return;
    }
    for (int i = 0; i < 8; ++i) {
        static_cast<float *>(weights->data)[i] = float(i);
    }
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 1);
    ggml_set_input(input);
    ggml_tensor * view = ggml_view_2d(ctx, input, 8, 1, input->nb[1], 0);
    ggml_tensor * output = ggml_add(ctx, view, weights);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1024, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, 1024, false, false);
    CHECK(sched != nullptr);
    if (sched == nullptr) {
        return;
    }
    CHECK(ggml_backend_sched_alloc_graph(sched, graph));

    ggml_tensor * tensors[] = { weights, input, view, output };
    const uint8_t fixed[] = { 1, 0, 0, 0 };
    auto make_layouts = [&](int width, ggml_backend_sched_tensor_layout * layouts, size_t * offsets) {
        ggml_tensor * dynamic[] = { input, view, output };
        for (int i = 0; i < 3; ++i) {
            layouts[i] = {};
            layouts[i].tensor = dynamic[i];
            std::memcpy(layouts[i].ne, dynamic[i]->ne, sizeof(layouts[i].ne));
            std::memcpy(layouts[i].nb, dynamic[i]->nb, sizeof(layouts[i].nb));
            layouts[i].ne[1] = width;
            layouts[i].nb[2] = layouts[i].nb[1] * size_t(width);
            layouts[i].nb[3] = layouts[i].nb[2];
            offsets[i] = dynamic[i]->view_offs;
        }
    };
    ggml_backend_sched_tensor_layout layouts[3];
    size_t offsets[3];
    make_layouts(3, layouts, offsets);
    CHECK(ggml_backend_sched_replan_graph(sched, tensors, fixed, 4, layouts, offsets, 3));
    size_t plan = 0;
    CHECK(ggml_backend_sched_get_tensor_plan_size(sched, input, &plan));
    CHECK(plan == size_t(3) * 8 * sizeof(float));

    std::vector<ggml_tensor> snapshot;
    for (ggml_tensor * tensor : tensors) {
        snapshot.push_back(*tensor);
    }
    void * fixed_data = weights->data;

    ggml_backend_sched_mark_unproven(sched);
    CHECK(ggml_backend_sched_has_unproven(sched));
    make_layouts(2, layouts, offsets);
    CHECK(!ggml_backend_sched_replan_graph(sched, tensors, fixed, 4, layouts, offsets, 3));
    for (size_t i = 0; i < snapshot.size(); ++i) {
        CHECK(std::memcmp(tensors[i], &snapshot[i], sizeof(ggml_tensor)) == 0);
    }
    size_t plan_after = 0;
    CHECK(ggml_backend_sched_get_tensor_plan_size(sched, input, &plan_after));
    CHECK(plan_after == plan);
    CHECK(weights->data == fixed_data);

    // The scheduler and its backends are retained for the process: the owner
    // contract forbids freeing them while unproven.
}

// The production llama_context destroys its backends before the scheduler
// (members destroy in reverse declaration order). The scheduler free must not
// query them then: it uses its cached unproven state only. This test frees the
// backends first and asserts the capability hook was not called afterwards.
void test_backends_first_shutdown() {
    reset_fake();
    fixture f;
    CHECK(fixture_init(f));
    CHECK(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS);
    CHECK(g_fake.pending_after_free == 0);

    // Production order: backends are destroyed before the scheduler.
    ggml_backend_free(f.cpu);
    f.cpu = nullptr;
    ggml_backend_free(&g_fake_backend); // fake iface.free sets backend_freed
    ggml_backend_sched_free(f.sched);
    f.sched = nullptr;

    CHECK(g_fake.pending_after_free == 0);

    ggml_backend_buffer_free(f.input_buf);
    ggml_free(f.ctx);
}

// Pending state must stop every entry that can reuse or free the copy slots:
// alloc_graph, reserve, reserve_size, release_buffers, reset and compute.
void test_pending_entry_points() {
    if (iface_has_submission_error) {
        return; // the older iface-field scheduler predates these gates
    }
    reset_fake();
    g_fake.submission_error = 0;
    fixture f;
    CHECK(fixture_init(f));
    CHECK(ggml_backend_sched_graph_compute(f.sched, f.graph) == GGML_STATUS_SUCCESS);
    g_fake.submission_pending = true;
    CHECK(ggml_backend_sched_has_unproven(f.sched));

    std::vector<unsigned char> copy_bytes;
    if (g_fake.last_dst && g_fake.last_dst->data) {
        const unsigned char * p = static_cast<const unsigned char *>(g_fake.last_dst->data);
        copy_bytes.assign(p, p + 8 * sizeof(float));
    }

    CHECK(!ggml_backend_sched_alloc_graph(f.sched, f.graph));
    CHECK(!ggml_backend_sched_reserve(f.sched, f.graph));
    size_t sizes[2] = { 0, 0 };
    ggml_backend_sched_reserve_size(f.sched, f.graph, sizes); // refused, no mutation

    // release_buffers must not free the gallocr that owns the copy slots.
    ggml_backend_sched_release_buffers(f.sched);
    if (g_fake.last_dst && g_fake.last_dst->data && !copy_bytes.empty()) {
        CHECK(std::memcmp(g_fake.last_dst->data, copy_bytes.data(), copy_bytes.size()) == 0);
    }

    ggml_backend_sched_reset(f.sched);
    CHECK(ggml_backend_sched_graph_compute_async(f.sched, f.graph) == GGML_STATUS_FAILED);
    CHECK(g_fake.cpy_calls == 2);

    // Free retains on the cached unproven state; do not read the scheduler
    // after this call.
    fixture_free(f, true);
    if (g_fake.last_dst && g_fake.last_dst->data && !copy_bytes.empty()) {
        CHECK(std::memcmp(g_fake.last_dst->data, copy_bytes.data(), copy_bytes.size()) == 0);
    }
}

// On the unknown return path the input copy scope must not flush. The async
// entry point isolates that flush from any wrapper-level synchronization.
void test_fatal_no_flush() {
    reset_fake();
    fixture f;
    CHECK(fixture_init(f));
    g_fake.error_on_accept = true;
    const int sync_before = g_fake.sync_calls; // alloc/split may synchronize
    const int err_before = g_fake.error_calls;

    CHECK(ggml_backend_sched_graph_compute_async(f.sched, f.graph) == GGML_STATUS_FAILED);
    CHECK(g_fake.cpy_calls == 1);
    CHECK(g_fake.error_calls == err_before + 2); // entry gate + accepted copy
    // One sync is the scheduler's pre-copy slot wait. The input_copy_scope
    // must add no second flush on the unknown return; before the fix it did.
    CHECK(g_fake.sync_calls == sync_before + 1);
    CHECK(g_fake.compute_calls == 0);
    CHECK(ggml_backend_sched_has_unproven(f.sched));

    fixture_free(f, true);
}

} // namespace

int main() {
    init_fake_backend();
    register_fake_moe_provider();

    test_success();
    test_rejected();
    test_pending_accept_stops_split();
    test_fatal();
    test_callback_failed_split_pending();
    test_pending_without_failure();
    test_error_only_gate();
    test_public_copy_async_return();
    test_sync_wrapper_no_silent_success();
    test_update_and_refresh_pending_gate();
    test_set_moe_cache_pending_gate();
    test_replan_pending_gate();
    test_pending_entry_points();
    test_fatal_no_flush();
    test_backends_first_shutdown();

    ggml_moe_cache_unregister(&g_fake_reg);

    if (g_failures != 0) {
        std::fprintf(stderr, "TP5_STAGING_SCHED_TEST FAIL failures=%d\n", g_failures);
        return 1;
    }
    std::printf("TP5_STAGING_SCHED_TEST PASS\n");
    return 0;
}
