// TP5 staging synchronous source identity CPU test (no GPU, no HIP runtime).
//
// The scheduler's blocking input fallback reaches the Meta buffer through
// ggml_backend_tensor_set, which only sees the internal copy destination: with
// n_copies == 1 that destination has no input flag and its name is
// "%s#%s#%d". This test drives a real Meta backend over two CPU child devices
// and injects a v3 staging table. It checks that the buffer cpy_tensor entry
// classifies the original source (union_inp_k_idxs), uploads its bytes through
// the staging lane (merge-wait and per-child set), and still writes every
// child. It also checks that a non-input source never leases a slot, that a
// direct set of the decorated destination stays on the old path, that a
// destination view takes the preflight rejection and falls back at its
// offset, and that a partially submitted mirrored copy publishes
// mark_partial, is visible through the public submission capability and the
// next graph compute, and is never replayed by the blocking fallback.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include "../ggml/src/ggml-cuda/staging.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace ggml_cuda_staging;

namespace {

int g_failures = 0;
uint64_t g_next_run_id = 1;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            ++g_failures;                                                                \
        }                                                                                \
    } while (0)

struct fake_staging_state {
    bool pending            = false;
    bool pending_on_enter   = false;
    bool pending_on_leave   = false;
    int  partial_after      = 0; // 0: never reject; N: reject enqueue N+1
    bool merge_wait_mode    = true;
    int  acquire_calls      = 0;
    int  acquire_nowait_calls = 0;
    int  owner_acquire_calls = 0;
    int  acquire_owner_calls = 0;
    int  owner_drain_calls = 0;
    int  owner_release_calls = 0;
    uint64_t last_owner = 0;
    uint64_t last_generation = 0;
    int  upload_calls       = 0;
    int  enqueue_sync_calls = 0;
    int  can_enqueue_async_calls = 0;
    int  enqueue_async_calls = 0;
    int  mark_partial_calls = 0;
    int  release_calls      = 0;
    int  enter_submit_calls = 0;
    kind last_kind          = kind::count;
    size_t last_bytes       = 0;
    std::vector<uint8_t> slot;
    ggml_tensor * enqueue_tensors[8] = {};
    ggml_tensor * enqueue_async_tensors[8] = {};
    ggml_backend_t enqueue_async_backends[8] = {};
};

fake_staging_state g_fake;
uint64_t g_fake_next_owner = 0x0abc;
uint64_t g_fake_next_generation = 0x0def;

bool fake_enabled()    { return !g_fake.pending; }
bool fake_usable()     { return !g_fake.pending; }
bool fake_configured() { return true; }
bool fake_merge_wait() { return g_fake.merge_wait_mode; }

bool fake_acquire(uint32_t window, kind k, size_t bytes, lease * out) {
    if (g_fake.pending || out == nullptr || bytes == 0) {
        return false;
    }
    ++g_fake.acquire_calls;
    g_fake.last_kind  = k;
    g_fake.last_bytes = bytes;
    g_fake.slot.assign(bytes, 0);
    *out = {};
    out->host   = g_fake.slot.data();
    out->cap    = bytes;
    out->which  = k;
    out->window = window;
    out->slot   = 0;
    out->epoch  = 1;
    return true;
}

bool fake_acquire_nowait(uint32_t window, kind k, size_t bytes, lease * out) {
    ++g_fake.acquire_nowait_calls;
    return fake_acquire(window, k, bytes, out);
}

bool fake_owner_acquire(uint64_t, uint32_t, uint64_t * owner, uint64_t * generation) {
    ++g_fake.owner_acquire_calls;
    if (owner == nullptr || generation == nullptr) {
        return false;
    }
    *owner = g_fake_next_owner++;
    *generation = g_fake_next_generation++;
    return true;
}

bool fake_acquire_owner(uint64_t owner, uint64_t generation, kind k, size_t bytes,
                        lease * out, bool wait_pending) {
    ++g_fake.acquire_owner_calls;
    g_fake.last_owner = owner;
    g_fake.last_generation = generation;
    if (!fake_acquire(0, k, bytes, out)) {
        return false;
    }
    if (out != nullptr) {
        out->owner = owner;
    }
    (void) wait_pending;
    return true;
}

int fake_owner_drain(uint64_t, uint64_t) {
    ++g_fake.owner_drain_calls;
    return int(owner_drain_result::drained);
}

bool fake_owner_release(uint64_t, uint64_t) {
    ++g_fake.owner_release_calls;
    return true;
}

int fake_upload(const lease * l, const void * src, size_t bytes) {
    if (l == nullptr || src == nullptr || bytes == 0 || bytes > g_fake.slot.size()) {
        return int(submit_result::rejected);
    }
    ++g_fake.upload_calls;
    std::memcpy(g_fake.slot.data(), src, bytes);
    return int(submit_result::ok);
}

int fake_upload_rows(const lease *, const void *, size_t, size_t, size_t, size_t) {
    return int(submit_result::rejected);
}

bool fake_wait(const lease *) {
    return !g_fake.pending;
}

void fake_release(const lease *) {
    ++g_fake.release_calls;
}

bool fake_can_enqueue_sync(ggml_tensor * tensor, size_t offset, size_t size) {
    return tensor != nullptr && tensor->data != nullptr && size != 0 &&
        offset + size <= ggml_nbytes(tensor);
}

int fake_enqueue_sync(const lease * l, ggml_tensor * tensor, size_t offset, size_t size) {
    ++g_fake.enqueue_sync_calls;
    if (g_fake.enqueue_sync_calls <= int(sizeof(g_fake.enqueue_tensors)/sizeof(g_fake.enqueue_tensors[0]))) {
        g_fake.enqueue_tensors[g_fake.enqueue_sync_calls - 1] = tensor;
    }
    if (g_fake.partial_after > 0 && g_fake.enqueue_sync_calls > g_fake.partial_after) {
        return int(submit_result::rejected);
    }
    if (l == nullptr || l->host == nullptr || tensor == nullptr || tensor->data == nullptr) {
        return int(submit_result::rejected);
    }
    std::memcpy((char *) tensor->data + offset, l->host, size);
    return int(submit_result::ok);
}

bool fake_can_enqueue_async(ggml_backend_t backend, ggml_tensor * tensor, size_t offset, size_t size) {
    ++g_fake.can_enqueue_async_calls;
    return backend != nullptr && tensor != nullptr && tensor->data != nullptr && size != 0 &&
        offset + size <= ggml_nbytes(tensor);
}

int fake_enqueue_async(const lease * l, ggml_backend_t backend, ggml_tensor * tensor,
        size_t offset, size_t size) {
    ++g_fake.enqueue_async_calls;
    if (g_fake.enqueue_async_calls <= int(sizeof(g_fake.enqueue_async_tensors)/sizeof(g_fake.enqueue_async_tensors[0]))) {
        g_fake.enqueue_async_tensors[g_fake.enqueue_async_calls - 1] = tensor;
        g_fake.enqueue_async_backends[g_fake.enqueue_async_calls - 1] = backend;
    }
    if (g_fake.partial_after > 0 && g_fake.enqueue_async_calls > g_fake.partial_after) {
        return int(submit_result::rejected);
    }
    if (l == nullptr || l->host == nullptr || tensor == nullptr || tensor->data == nullptr) {
        return int(submit_result::rejected);
    }
    std::memcpy((char *) tensor->data + offset, l->host, size);
    return int(submit_result::ok);
}

int fake_first_error() {
    return 0;
}

bool fake_enter_submit() {
    ++g_fake.enter_submit_calls;
    if (g_fake.pending_on_enter) {
        g_fake.pending = true;
        return false;
    }
    return !g_fake.pending;
}

void fake_leave_submit() {
    if (g_fake.pending_on_leave) {
        g_fake.pending = true;
    }
}

size_t fake_unknown_count() {
    return g_fake.pending ? 1 : 0;
}

bool fake_submission_pending() {
    return g_fake.pending;
}

// A source backend that reports a native submission error. Used to prove a
// direct capability caller cannot bypass a pending source.
int fake_source_error(ggml_backend_t) {
    return 1;
}

bool fake_source_pending(ggml_backend_t) {
    return false;
}

void fake_mark_partial(const lease * l, const void * dst) {
    CHECK(l != nullptr);
    CHECK(dst != nullptr);
    ++g_fake.mark_partial_calls;
    g_fake.pending = true;
}

// This is the v3 staging table of this branch. The host rejects a table
// whose kind_count is smaller than its own kKindCount, so keep this
// initializer in sync with the v3 api layout.
const api g_fake_api = {
    /* .abi_version       = */ kApiVersion,
    /* .enabled           = */ fake_enabled,
    /* .merge_wait        = */ fake_merge_wait,
    /* .acquire           = */ fake_acquire,
    /* .acquire_nowait    = */ fake_acquire_nowait,
    /* .upload            = */ fake_upload,
    /* .upload_rows       = */ fake_upload_rows,
    /* .wait              = */ fake_wait,
    /* .release           = */ fake_release,
    /* .owns              = */ nullptr,
    /* .device_ptr        = */ nullptr,
    /* .dump              = */ nullptr,
    /* .can_enqueue_async = */ fake_can_enqueue_async,
    /* .enqueue_async     = */ fake_enqueue_async,
    /* .can_enqueue_sync  = */ fake_can_enqueue_sync,
    /* .enqueue_sync      = */ fake_enqueue_sync,
    /* .first_error       = */ fake_first_error,
    /* .enter_submit      = */ fake_enter_submit,
    /* .leave_submit      = */ fake_leave_submit,
    /* .unknown_count     = */ fake_unknown_count,
    /* .submission_pending= */ fake_submission_pending,
    /* .configured        = */ fake_configured,
    /* .usable            = */ fake_usable,
    /* .zc_enabled        = */ nullptr,
    /* .mark_partial      = */ fake_mark_partial,
    /* .kind_count        = */ kKindCount,
    /* .owner_acquire     = */ fake_owner_acquire,
    /* .acquire_owner     = */ fake_acquire_owner,
    /* .owner_drain       = */ fake_owner_drain,
    /* .owner_release     = */ fake_owner_release,
};

// A second, independent provider used only to prove that a tagged owner routes
// back to the provider that issued it after another provider becomes active.
fake_staging_state g_fake_b;

bool fake_b_owner_acquire(uint64_t, uint32_t, uint64_t * owner, uint64_t * generation) {
    ++g_fake_b.owner_acquire_calls;
    if (owner == nullptr || generation == nullptr) {
        return false;
    }
    *owner = 0x0bcd;
    *generation = 0x0bce;
    return true;
}

bool fake_b_acquire_owner(uint64_t, uint64_t, kind, size_t, lease *, bool) {
    ++g_fake_b.acquire_owner_calls;
    return false;
}

int fake_b_owner_drain(uint64_t, uint64_t) {
    ++g_fake_b.owner_drain_calls;
    return int(owner_drain_result::drained);
}

bool fake_b_owner_release(uint64_t, uint64_t) {
    ++g_fake_b.owner_release_calls;
    return true;
}

int fake_b_first_error() { return 0; }
bool fake_b_submission_pending() { return false; }
bool fake_b_configured() { return true; }
bool fake_b_usable() { return true; }

api make_fake_api_b() {
    api a = g_fake_api;
    a.owner_acquire = fake_b_owner_acquire;
    a.acquire_owner = fake_b_acquire_owner;
    a.owner_drain = fake_b_owner_drain;
    a.owner_release = fake_b_owner_release;
    a.first_error = fake_b_first_error;
    a.submission_pending = fake_b_submission_pending;
    a.configured = fake_b_configured;
    a.usable = fake_b_usable;
    return a;
}

const api g_fake_api_b = make_fake_api_b();
ggml_backend_reg g_fake_reg_b = {};

ggml_backend_meta_split_state split_mirrored(const ggml_tensor *, void *) {
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

struct fixture {
    ggml_backend_t        meta     = nullptr;
    ggml_context *        src_ctx  = nullptr;
    ggml_context *        dst_ctx  = nullptr;
    ggml_backend_buffer_t src_buf  = nullptr;
    ggml_backend_buffer_t dst_buf  = nullptr;
    ggml_tensor *         src      = nullptr;
    ggml_tensor *         dst_root = nullptr;
    ggml_tensor *         dst      = nullptr;
};

bool bind_fake_owner(ggml_backend_t backend) {
    uint64_t owner = 0;
    uint64_t generation = 0;
    if (!ggml_backend_meta_staging_owner_acquire_for_backend(
                backend, g_next_run_id++, 0, &owner, &generation)) {
        return false;
    }
    return ggml_backend_meta_set_staging_owner(backend, owner, generation);
}

bool make_fixture(ggml_backend_dev_t cpu, fixture & f, bool src_input,
                  bool dst_view = false, bool bind_owner = true) {
    ggml_backend_dev_t children[] = { cpu, cpu };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(children, 2, split_mirrored, nullptr);
    CHECK(meta_dev != nullptr);
    if (meta_dev == nullptr) {
        return false;
    }
    f.meta = ggml_backend_dev_init(meta_dev, nullptr);
    CHECK(f.meta != nullptr);
    if (f.meta == nullptr) {
        return false;
    }
    if (bind_owner) {
        CHECK(bind_fake_owner(f.meta));
    }

    ggml_init_params params = { /*.mem_size=*/ 1u << 20, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };

    f.src_ctx = ggml_init(params);
    CHECK(f.src_ctx != nullptr);
    if (f.src_ctx == nullptr) {
        return false;
    }
    f.src = ggml_new_tensor_1d(f.src_ctx, GGML_TYPE_I64, 16);
    CHECK(f.src != nullptr);
    if (f.src == nullptr) {
        return false;
    }
    ggml_set_name(f.src, "union_inp_k_idxs");
    if (src_input) {
        ggml_set_input(f.src);
    }
    f.src_buf = ggml_backend_alloc_ctx_tensors_from_buft(f.src_ctx, ggml_backend_cpu_buffer_type());
    CHECK(f.src_buf != nullptr);
    if (f.src_buf == nullptr) {
        return false;
    }

    f.dst_ctx = ggml_init(params);
    CHECK(f.dst_ctx != nullptr);
    if (f.dst_ctx == nullptr) {
        return false;
    }
    if (dst_view) {
        f.dst_root = ggml_new_tensor_1d(f.dst_ctx, GGML_TYPE_I64, 32);
        CHECK(f.dst_root != nullptr);
        if (f.dst_root == nullptr) {
            return false;
        }
        ggml_set_name(f.dst_root, "dst_root");
        f.dst_buf = ggml_backend_alloc_ctx_tensors_from_buft(
                f.dst_ctx, ggml_backend_dev_buffer_type(meta_dev));
        CHECK(f.dst_buf != nullptr);
        if (f.dst_buf == nullptr) {
            return false;
        }
        f.dst = ggml_view_1d(f.dst_ctx, f.dst_root, 16, 2 * sizeof(int64_t));
        CHECK(f.dst != nullptr);
        if (f.dst == nullptr) {
            return false;
        }
        // The decorated name and the absent input flag are the scheduler's
        // copy shape. The view must still fall back through set_tensor.
        ggml_set_name(f.dst, "Meta#union_inp_k_idxs#0");
        CHECK(ggml_backend_view_init(f.dst) == GGML_STATUS_SUCCESS);
    } else {
        f.dst = ggml_new_tensor_1d(f.dst_ctx, GGML_TYPE_I64, 16);
        CHECK(f.dst != nullptr);
        if (f.dst == nullptr) {
            return false;
        }
        // The scheduler decorates the internal copy and, with n_copies == 1,
        // does not set the input flag. The exact union name must not be parsed.
        ggml_set_name(f.dst, "Meta#union_inp_k_idxs#0");
        f.dst_buf = ggml_backend_alloc_ctx_tensors_from_buft(
                f.dst_ctx, ggml_backend_dev_buffer_type(meta_dev));
        CHECK(f.dst_buf != nullptr);
        CHECK(f.dst->data != nullptr);
    }
    return f.dst_buf != nullptr && f.dst->data != nullptr;
}

void free_fixture(fixture & f) {
    if (f.dst_buf) {
        ggml_backend_buffer_free(f.dst_buf);
        f.dst_buf = nullptr;
    }
    if (f.src_buf) {
        ggml_backend_buffer_free(f.src_buf);
        f.src_buf = nullptr;
    }
    if (f.dst_ctx) {
        ggml_free(f.dst_ctx);
        f.dst_ctx = nullptr;
    }
    if (f.src_ctx) {
        ggml_free(f.src_ctx);
        f.src_ctx = nullptr;
    }
    if (f.meta) {
        ggml_backend_free(f.meta);
        f.meta = nullptr;
    }
}

struct meta_pair {
    ggml_backend_t        meta = nullptr;
    ggml_context *        ctx  = nullptr;
    ggml_backend_buffer_t buf  = nullptr;
    ggml_tensor *         src  = nullptr;
    ggml_tensor *         dst  = nullptr;
};

bool make_meta_pair(ggml_backend_dev_t cpu, meta_pair & p) {
    ggml_backend_dev_t children[] = { cpu, cpu };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(children, 2, split_mirrored, nullptr);
    CHECK(meta_dev != nullptr);
    if (meta_dev == nullptr) {
        return false;
    }
    p.meta = ggml_backend_dev_init(meta_dev, nullptr);
    CHECK(p.meta != nullptr);
    if (p.meta == nullptr) {
        return false;
    }
    CHECK(bind_fake_owner(p.meta));
    ggml_init_params params = { /*.mem_size=*/ 1u << 20, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    p.ctx = ggml_init(params);
    CHECK(p.ctx != nullptr);
    if (p.ctx == nullptr) {
        return false;
    }
    p.src = ggml_new_tensor_1d(p.ctx, GGML_TYPE_I64, 16);
    p.dst = ggml_new_tensor_1d(p.ctx, GGML_TYPE_I64, 16);
    CHECK(p.src != nullptr && p.dst != nullptr);
    if (p.src == nullptr || p.dst == nullptr) {
        return false;
    }
    ggml_set_name(p.src, "meta_src");
    ggml_set_name(p.dst, "meta_dst");
    p.buf = ggml_backend_alloc_ctx_tensors_from_buft(p.ctx, ggml_backend_dev_buffer_type(meta_dev));
    CHECK(p.buf != nullptr);
    CHECK(p.src->data != nullptr && p.dst->data != nullptr);
    return p.buf != nullptr && p.src->data != nullptr && p.dst->data != nullptr;
}

void free_meta_pair(meta_pair & p) {
    if (p.buf) {
        ggml_backend_buffer_free(p.buf);
        p.buf = nullptr;
    }
    if (p.ctx) {
        ggml_free(p.ctx);
        p.ctx = nullptr;
    }
    if (p.meta) {
        ggml_backend_free(p.meta);
        p.meta = nullptr;
    }
}

void reset_fake() {
    g_fake = {};
}

std::vector<int64_t> make_data() {
    std::vector<int64_t> data(16);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = int64_t(1000 + 7 * i);
    }
    return data;
}

void check_dst(const fixture & f, const std::vector<int64_t> & data) {
    std::vector<int64_t> got(data.size(), 0);
    ggml_backend_tensor_get(f.dst, got.data(), 0, data.size() * sizeof(int64_t));
    CHECK(std::memcmp(got.data(), data.data(), data.size() * sizeof(int64_t)) == 0);
}

using sync_fn_t = int (*)(ggml_backend_t, ggml_backend_t,
                           const ggml_tensor *, ggml_tensor *);

sync_fn_t meta_sync_fn(ggml_backend_t backend) {
    ggml_backend_dev_t dev = backend != nullptr ? ggml_backend_get_device(backend) : nullptr;
    ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg == nullptr) {
        return nullptr;
    }
    auto api = (const ggml_backend_copy_tensor_sync_api *)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_copy_tensor_sync_v2");
    if (api == nullptr || api->abi_version != GGML_BACKEND_COPY_TENSOR_SYNC_API_VERSION ||
            api->copy == nullptr) {
        return nullptr;
    }
    return api->copy;
}

// The named capability carries the original source identity and the real
// destination backend. The async setter uses the destination owner's child
// backends, so this also covers the clean blocking path.
void test_capability_source_identity(ggml_backend_dev_t cpu) {
    reset_fake();
    fixture f;
    CHECK(make_fixture(cpu, f, /*src_input=*/ true));
    if (f.meta == nullptr) {
        return;
    }
    sync_fn_t fn = meta_sync_fn(f.meta);
    CHECK(fn != nullptr);
    const std::vector<int64_t> data = make_data();
    ggml_backend_tensor_set(f.src, data.data(), 0, data.size() * sizeof(int64_t));

    CHECK(fn != nullptr && fn(nullptr, f.meta, f.src, f.dst) ==
          GGML_BACKEND_COPY_TENSOR_SYNC_OK);

    CHECK(g_fake.acquire_owner_calls == 1);
    CHECK(g_fake.last_owner != 0);
    CHECK(g_fake.last_generation != 0);
    CHECK(g_fake.last_kind == kind::union_k);
    CHECK(g_fake.last_bytes == data.size() * sizeof(int64_t));
    CHECK(g_fake.upload_calls == 1);
    CHECK(g_fake.can_enqueue_async_calls == 2);
    CHECK(g_fake.enqueue_async_calls == 2);
    CHECK(g_fake.mark_partial_calls == 0);
    CHECK(g_fake.release_calls == 1);
    CHECK(!g_fake.pending);
    check_dst(f, data);

    free_fixture(f);
}

// Two live Meta backends share the same device and buffer type but own
// different child backends. The capability must use backend_dst, not
// backend_src.
void test_capability_two_backend_selection(ggml_backend_dev_t cpu) {
    reset_fake();
    fixture f;
    CHECK(make_fixture(cpu, f, /*src_input=*/ true));
    if (f.meta == nullptr) {
        return;
    }
    ggml_backend_t other = ggml_backend_dev_init(ggml_backend_get_device(f.meta), nullptr);
    CHECK(other != nullptr);
    if (other == nullptr) {
        free_fixture(f);
        return;
    }
    sync_fn_t fn = meta_sync_fn(f.meta);
    CHECK(fn != nullptr);
    const std::vector<int64_t> data = make_data();
    ggml_backend_tensor_set(f.src, data.data(), 0, data.size() * sizeof(int64_t));

    CHECK(fn != nullptr && fn(other, f.meta, f.src, f.dst) ==
          GGML_BACKEND_COPY_TENSOR_SYNC_OK);

    CHECK(g_fake.enqueue_async_calls == 2);
    for (size_t j = 0; j < 2; ++j) {
        CHECK(g_fake.enqueue_async_backends[j] == ggml_backend_meta_simple_backend(f.meta, j));
        CHECK(g_fake.enqueue_async_backends[j] != ggml_backend_meta_simple_backend(other, j));
    }
    check_dst(f, data);

    ggml_backend_free(other);
    free_fixture(f);
}

// Zero bytes need no destination write or staging lease.
void test_capability_zero(ggml_backend_dev_t cpu) {
    reset_fake();
    ggml_backend_dev_t children[] = { cpu, cpu };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(children, 2, split_mirrored, nullptr);
    CHECK(meta_dev != nullptr);
    if (meta_dev == nullptr) {
        return;
    }
    ggml_backend_t meta = ggml_backend_dev_init(meta_dev, nullptr);
    CHECK(meta != nullptr);
    if (meta == nullptr) {
        return;
    }

    ggml_init_params params = { /*.mem_size=*/ 1u << 20, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    ggml_context * src_ctx = ggml_init(params);
    ggml_context * dst_ctx = ggml_init(params);
    CHECK(src_ctx != nullptr && dst_ctx != nullptr);
    if (src_ctx == nullptr || dst_ctx == nullptr) {
        return;
    }

    // A host zero view keeps a real host buffer behind the zero-size source.
    ggml_tensor * src_root = ggml_new_tensor_1d(src_ctx, GGML_TYPE_I64, 1);
    ggml_tensor * src = ggml_view_1d(src_ctx, src_root, 0, 0);
    ggml_tensor * dst = ggml_new_tensor_1d(dst_ctx, GGML_TYPE_I64, 0);
    CHECK(src_root != nullptr && src != nullptr && dst != nullptr);
    if (src == nullptr || dst == nullptr) {
        return;
    }
    ggml_set_name(src, "union_inp_k_idxs");
    ggml_set_input(src);
    ggml_set_name(dst, "Meta#union_inp_k_idxs#0");

    ggml_backend_buffer_t src_buf = ggml_backend_alloc_ctx_tensors_from_buft(
            src_ctx, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_t dst_buf = ggml_backend_alloc_ctx_tensors_from_buft(
            dst_ctx, ggml_backend_dev_buffer_type(meta_dev));
    CHECK(src_buf != nullptr && dst_buf != nullptr);
    CHECK(ggml_nbytes(src) == 0 && ggml_nbytes(dst) == 0);

    sync_fn_t fn = meta_sync_fn(meta);
    CHECK(fn != nullptr);
    CHECK(fn != nullptr && fn(nullptr, meta, src, dst) ==
          GGML_BACKEND_COPY_TENSOR_SYNC_OK);
    CHECK(g_fake.acquire_nowait_calls == 0);
    CHECK(g_fake.upload_calls == 0);
    CHECK(g_fake.enqueue_async_calls == 0);
    CHECK(!g_fake.pending);

    if (dst_buf) {
        ggml_backend_buffer_free(dst_buf);
    }
    if (src_buf) {
        ggml_backend_buffer_free(src_buf);
    }
    ggml_free(dst_ctx);
    ggml_free(src_ctx);
    ggml_backend_free(meta);
}

// A partial mirrored submission must publish mark_partial, report FAIL_STOP
// without draining, and never replay either child on a later call.
void test_capability_partial(ggml_backend_dev_t cpu) {
    reset_fake();
    g_fake.partial_after = 1;
    fixture f;
    CHECK(make_fixture(cpu, f, /*src_input=*/ true));
    if (f.meta == nullptr) {
        return;
    }
    sync_fn_t fn = meta_sync_fn(f.meta);
    CHECK(fn != nullptr);
    const std::vector<int64_t> data = make_data();
    ggml_backend_tensor_set(f.src, data.data(), 0, data.size() * sizeof(int64_t));

    CHECK(fn != nullptr && fn(nullptr, f.meta, f.src, f.dst) ==
          GGML_BACKEND_COPY_TENSOR_SYNC_FAIL_STOP);

    CHECK(g_fake.acquire_owner_calls == 1);
    CHECK(g_fake.upload_calls == 1);
    CHECK(g_fake.enqueue_async_calls == 2);
    CHECK(g_fake.mark_partial_calls == 1);
    CHECK(g_fake.release_calls == 1);
    CHECK(g_fake.pending);

    ggml_tensor * rank0 = g_fake.enqueue_async_tensors[0];
    ggml_tensor * rank1 = g_fake.enqueue_async_tensors[1];
    CHECK(rank0 != nullptr);
    CHECK(rank1 != nullptr);
    std::vector<uint8_t> rank0_before;
    std::vector<uint8_t> rank1_before;
    if (rank0 != nullptr && rank1 != nullptr) {
        rank0_before.resize(ggml_nbytes(rank0));
        rank1_before.resize(ggml_nbytes(rank1));
        ggml_backend_tensor_get(rank0, rank0_before.data(), 0, rank0_before.size());
        ggml_backend_tensor_get(rank1, rank1_before.data(), 0, rank1_before.size());
        CHECK(std::memcmp(rank0_before.data(), data.data(), data.size() * sizeof(int64_t)) == 0);
    }

    const int enqueues = g_fake.enqueue_async_calls;
    const int marks     = g_fake.mark_partial_calls;
    CHECK(fn != nullptr && fn(nullptr, f.meta, f.src, f.dst) ==
          GGML_BACKEND_COPY_TENSOR_SYNC_FAIL_STOP);
    CHECK(g_fake.enqueue_async_calls == enqueues);
    CHECK(g_fake.mark_partial_calls == marks);
    CHECK(g_fake.pending);

    if (rank0 != nullptr && rank1 != nullptr) {
        std::vector<uint8_t> rank0_after(rank0_before.size());
        std::vector<uint8_t> rank1_after(rank1_before.size());
        ggml_backend_tensor_get(rank0, rank0_after.data(), 0, rank0_after.size());
        ggml_backend_tensor_get(rank1, rank1_after.data(), 0, rank1_after.size());
        CHECK(std::memcmp(rank0_after.data(), rank0_before.data(), rank0_after.size()) == 0);
        CHECK(std::memcmp(rank1_after.data(), rank1_before.data(), rank1_after.size()) == 0);
    }

    // Retained by the unknown-submission contract.
}

// The source identity comes from src, not from the decorated destination:
// union_k is leased, the bytes go through upload + one enqueue per child, and
// the real Meta read path returns them.
// The Meta drain receipt is DRAINED only when the staging pool is clean and
// every child has a completion fact (CPU children have no async work). A
// pending pool is BUSY, never DRAINED.
void test_meta_drain_receipt(ggml_backend_dev_t cpu) {
    reset_fake();
    fixture f;
    CHECK(make_fixture(cpu, f, /*src_input=*/ true));
    if (f.meta == nullptr) {
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(f.meta);
    ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
    CHECK(reg != nullptr);
    auto api = reg != nullptr
        ? (const ggml_backend_drain_api *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_drain_v1")
        : nullptr;
    CHECK(api != nullptr);
    if (api == nullptr || api->drain == nullptr) {
        free_fixture(f);
        return;
    }
    CHECK(api->abi_version == GGML_BACKEND_DRAIN_API_VERSION);
    CHECK(api->drain(f.meta) == GGML_BACKEND_DRAIN_DRAINED);
    g_fake.pending = true;
    CHECK(api->drain(f.meta) == GGML_BACKEND_DRAIN_BUSY);
    g_fake.pending = false;
    CHECK(api->drain(f.meta) == GGML_BACKEND_DRAIN_DRAINED);
    free_fixture(f);
}

void test_source_identity_and_content(ggml_backend_dev_t cpu) {
    reset_fake();
    fixture f;
    CHECK(make_fixture(cpu, f, /*src_input=*/ true));
    if (f.meta == nullptr) {
        return;
    }
    const std::vector<int64_t> data = make_data();
    ggml_backend_tensor_set(f.src, data.data(), 0, data.size() * sizeof(int64_t));

    ggml_backend_buffer_copy_tensor(f.src, f.dst);

    CHECK(g_fake.acquire_calls == 0);
    CHECK(g_fake.upload_calls == 0);
    CHECK(g_fake.enqueue_sync_calls == 0);
    CHECK(g_fake.mark_partial_calls == 0);
    CHECK(!g_fake.pending);
    check_dst(f, data);

    free_fixture(f);
}

// The non-merge synchronous lane writes each child from the pinned slot
// instead of enqueueing an event. Source identity must survive that path too.
void test_source_identity_without_merge_wait(ggml_backend_dev_t cpu) {
    reset_fake();
    g_fake.merge_wait_mode = false;
    fixture f;
    CHECK(make_fixture(cpu, f, /*src_input=*/ true));
    if (f.meta == nullptr) {
        return;
    }
    const std::vector<int64_t> data = make_data();
    ggml_backend_tensor_set(f.src, data.data(), 0, data.size() * sizeof(int64_t));

    ggml_backend_buffer_copy_tensor(f.src, f.dst);

    CHECK(g_fake.acquire_calls == 0);
    CHECK(g_fake.upload_calls == 0);
    CHECK(g_fake.enqueue_sync_calls == 0);
    CHECK(g_fake.enter_submit_calls == 1);
    CHECK(!g_fake.pending);
    check_dst(f, data);

    free_fixture(f);
}

// A view destination is explicitly excluded from the source-identity fast
// path. The preflight returns false and the blocking set path writes the view
// at its offset, with no staging lease or upload.
void test_dst_view_falls_back(ggml_backend_dev_t cpu) {
    reset_fake();
    fixture f;
    CHECK(make_fixture(cpu, f, /*src_input=*/ true, /*dst_view=*/ true));
    if (f.meta == nullptr) {
        return;
    }
    std::vector<int64_t> zeros(32, 0);
    ggml_backend_tensor_set(f.dst_root, zeros.data(), 0, zeros.size() * sizeof(int64_t));
    const std::vector<int64_t> data = make_data();
    ggml_backend_tensor_set(f.src, data.data(), 0, data.size() * sizeof(int64_t));

    // A view destination cannot use the owner staging lane. The capability
    // writes it without a staging lease instead of returning false to an
    // ownerless generic copy.
    sync_fn_t fn = meta_sync_fn(f.meta);
    CHECK(fn != nullptr);
    CHECK(fn != nullptr && fn(nullptr, f.meta, f.src, f.dst) ==
          GGML_BACKEND_COPY_TENSOR_SYNC_OK);
    CHECK(g_fake.acquire_calls == 0);
    CHECK(g_fake.enqueue_async_calls == 0);

    CHECK(g_fake.acquire_calls == 0);
    CHECK(g_fake.upload_calls == 0);
    CHECK(g_fake.enqueue_sync_calls == 0);
    CHECK(!g_fake.pending);

    constexpr size_t view_offset = 2;
    std::vector<int64_t> got(32, -1);
    ggml_backend_tensor_get(f.dst_root, got.data(), 0, got.size() * sizeof(int64_t));
    for (size_t i = 0; i < view_offset; ++i) {
        CHECK(got[i] == 0);
    }
    CHECK(std::memcmp(got.data() + view_offset, data.data(), data.size() * sizeof(int64_t)) == 0);
    for (size_t i = view_offset + data.size(); i < got.size(); ++i) {
        CHECK(got[i] == 0);
    }

    free_fixture(f);
}

// The owner-aware helper must not fall back to the ownerless Meta buffer
// staging path when the destination backend has no compatible capability.
// The explicit ownerless ggml_backend_tensor_copy still stages.
void test_owner_path_no_ownerless_meta_staging(ggml_backend_dev_t cpu) {
    reset_fake();
    fixture f;
    CHECK(make_fixture(cpu, f, /*src_input=*/ true));
    if (f.meta == nullptr) {
        return;
    }

    ggml_backend_reg fake_reg = {};
    fake_reg.api_version = GGML_BACKEND_API_VERSION;
    fake_reg.iface.get_name = [](ggml_backend_reg_t) { return "fake-no-sync"; };
    fake_reg.iface.get_proc_address = [](ggml_backend_reg_t, const char *) -> void * {
        return nullptr;
    };
    ggml_backend_device fake_dev = {};
    fake_dev.reg = &fake_reg;
    ggml_backend fake_backend = {};
    fake_backend.iface.get_name = [](ggml_backend_t) { return "fake-no-sync"; };
    fake_backend.iface.synchronize = [](ggml_backend_t) {};
    fake_backend.device = &fake_dev;

    std::vector<int64_t> zeros(16, 0);
    ggml_backend_tensor_set(f.dst, zeros.data(), 0, zeros.size() * sizeof(int64_t));
    reset_fake();
    const std::vector<int64_t> data = make_data();
    ggml_backend_tensor_set(f.src, data.data(), 0, data.size() * sizeof(int64_t));

    CHECK(!ggml_backend_tensor_copy_async(&fake_backend, &fake_backend, f.src, f.dst));
    CHECK(g_fake.acquire_calls == 0);
    CHECK(g_fake.upload_calls == 0);
    CHECK(g_fake.enqueue_async_calls == 0);

    std::vector<int64_t> got(16, -1);
    ggml_backend_tensor_get(f.dst, got.data(), 0, got.size() * sizeof(int64_t));
    CHECK(std::memcmp(got.data(), zeros.data(), zeros.size() * sizeof(int64_t)) == 0);

    // The public sync copy has no backend owner. It uses the ownerless
    // no-staging path and must not lease a staging slot.
    ggml_backend_tensor_copy(f.src, f.dst);
    CHECK(g_fake.acquire_calls == 0);
    CHECK(g_fake.upload_calls == 0);
    CHECK(g_fake.enqueue_async_calls == 0);
    check_dst(f, data);

    free_fixture(f);
}

// The public sync copy has no owner. Even a destination that carries the
// input flag must use the no-staging path, not the old destination-based
// staging decision.
void test_public_sync_copy_no_staging(ggml_backend_dev_t cpu) {
    reset_fake();
    fixture f;
    CHECK(make_fixture(cpu, f, /*src_input=*/ true));
    if (f.meta == nullptr) {
        return;
    }
    ggml_set_input(f.dst);
    const std::vector<int64_t> data = make_data();
    ggml_backend_tensor_set(f.src, data.data(), 0, data.size() * sizeof(int64_t));
    reset_fake();

    ggml_backend_tensor_copy(f.src, f.dst);

    CHECK(g_fake.acquire_calls == 0);
    CHECK(g_fake.upload_calls == 0);
    CHECK(g_fake.enqueue_async_calls == 0);
    check_dst(f, data);

    free_fixture(f);
}

// A direct capability caller must not bypass a pending source backend.
void test_capability_source_pending(ggml_backend_dev_t cpu) {
    reset_fake();
    fixture f;
    CHECK(make_fixture(cpu, f, /*src_input=*/ true));
    if (f.meta == nullptr) {
        return;
    }

    ggml_backend_reg fake_reg = {};
    fake_reg.api_version = GGML_BACKEND_API_VERSION;
    fake_reg.iface.get_name = [](ggml_backend_reg_t) { return "fake-pending-src"; };
    fake_reg.iface.get_proc_address = [](ggml_backend_reg_t, const char * name) -> void * {
        if (name == nullptr) {
            return nullptr;
        }
        if (std::strcmp(name, "ggml_backend_submission_error") == 0) {
            return reinterpret_cast<void *>(+fake_source_error);
        }
        if (std::strcmp(name, "ggml_backend_submission_pending") == 0) {
            return reinterpret_cast<void *>(+fake_source_pending);
        }
        return nullptr;
    };
    ggml_backend_device fake_dev = {};
    fake_dev.reg = &fake_reg;
    ggml_backend fake_backend = {};
    fake_backend.iface.get_name = [](ggml_backend_t) { return "fake-pending-src"; };
    fake_backend.iface.synchronize = [](ggml_backend_t) {};
    fake_backend.device = &fake_dev;

    std::vector<int64_t> zeros(16, 0);
    ggml_backend_tensor_set(f.dst, zeros.data(), 0, zeros.size() * sizeof(int64_t));
    reset_fake();
    const std::vector<int64_t> data = make_data();
    ggml_backend_tensor_set(f.src, data.data(), 0, data.size() * sizeof(int64_t));

    sync_fn_t fn = meta_sync_fn(f.meta);
    CHECK(fn != nullptr);
    CHECK(fn != nullptr && fn(&fake_backend, f.meta, f.src, f.dst) ==
          GGML_BACKEND_COPY_TENSOR_SYNC_FAIL_STOP);
    CHECK(g_fake.acquire_calls == 0);
    CHECK(g_fake.upload_calls == 0);
    CHECK(g_fake.enqueue_async_calls == 0);

    std::vector<int64_t> got(16, -1);
    ggml_backend_tensor_get(f.dst, got.data(), 0, got.size() * sizeof(int64_t));
    CHECK(std::memcmp(got.data(), zeros.data(), zeros.size() * sizeof(int64_t)) == 0);

    free_fixture(f);
}

// Meta-to-Meta copy_async is a supported ordinary path. It must keep the
// generic get/set result and never lease a staging slot.
void test_meta_to_meta_copy_async(ggml_backend_dev_t cpu) {
    reset_fake();
    meta_pair p;
    CHECK(make_meta_pair(cpu, p));
    if (p.meta == nullptr) {
        return;
    }
    const std::vector<int64_t> data = make_data();
    ggml_backend_tensor_set(p.src, data.data(), 0, data.size() * sizeof(int64_t));
    reset_fake();

    CHECK(ggml_backend_tensor_copy_async(p.meta, p.meta, p.src, p.dst));
    CHECK(g_fake.acquire_calls == 0);
    CHECK(g_fake.upload_calls == 0);
    CHECK(g_fake.enqueue_async_calls == 0);
    CHECK(g_fake.enter_submit_calls >= 2); // gated synchronize of source and destination

    std::vector<int64_t> got(16, -1);
    ggml_backend_tensor_get(p.dst, got.data(), 0, got.size() * sizeof(int64_t));
    CHECK(std::memcmp(got.data(), data.data(), data.size() * sizeof(int64_t)) == 0);

    free_meta_pair(p);
}

// A non-contiguous host source keeps the ordinary byte copy. It must not be
// treated as unsupported, and it must not lease a staging slot.
void test_non_contiguous_host_source(ggml_backend_dev_t cpu) {
    reset_fake();
    ggml_backend_dev_t children[] = { cpu, cpu };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(children, 2, split_mirrored, nullptr);
    CHECK(meta_dev != nullptr);
    if (meta_dev == nullptr) {
        return;
    }
    ggml_backend_t meta = ggml_backend_dev_init(meta_dev, nullptr);
    CHECK(meta != nullptr);
    if (meta == nullptr) {
        return;
    }

    ggml_init_params params = { /*.mem_size=*/ 1u << 20, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    ggml_context * src_ctx = ggml_init(params);
    ggml_context * dst_ctx = ggml_init(params);
    CHECK(src_ctx != nullptr && dst_ctx != nullptr);
    if (src_ctx == nullptr || dst_ctx == nullptr) {
        return;
    }
    ggml_tensor * src = ggml_new_tensor_2d(src_ctx, GGML_TYPE_F32, 4, 4);
    ggml_tensor * dst = ggml_new_tensor_2d(dst_ctx, GGML_TYPE_F32, 4, 4);
    CHECK(src != nullptr && dst != nullptr);
    if (src == nullptr || dst == nullptr) {
        return;
    }
    // Row padding makes the tensor non-contiguous while keeping both layouts
    // identical.
    src->nb[1] = 32;
    src->nb[2] = 32 * 4;
    src->nb[3] = 32 * 4;
    dst->nb[1] = 32;
    dst->nb[2] = 32 * 4;
    dst->nb[3] = 32 * 4;
    CHECK(!ggml_is_contiguous(src));

    ggml_backend_buffer_t src_buf = ggml_backend_alloc_ctx_tensors_from_buft(
            src_ctx, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_t dst_buf = ggml_backend_alloc_ctx_tensors_from_buft(
            dst_ctx, ggml_backend_dev_buffer_type(meta_dev));
    CHECK(src_buf != nullptr && dst_buf != nullptr);
    if (src_buf == nullptr || dst_buf == nullptr) {
        return;
    }

    const size_t nbytes = ggml_nbytes(src);
    std::vector<uint8_t> bytes(nbytes);
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = uint8_t((i * 37 + 11) & 0xff);
    }
    ggml_backend_tensor_set(src, bytes.data(), 0, bytes.size());
    reset_fake();

    sync_fn_t fn = meta_sync_fn(meta);
    CHECK(fn != nullptr);
    CHECK(fn != nullptr && fn(nullptr, meta, src, dst) ==
          GGML_BACKEND_COPY_TENSOR_SYNC_OK);
    CHECK(g_fake.acquire_calls == 0);
    CHECK(g_fake.upload_calls == 0);
    CHECK(g_fake.enqueue_async_calls == 0);

    std::vector<uint8_t> got(nbytes);
    ggml_backend_tensor_get(dst, got.data(), 0, got.size());
    CHECK(std::memcmp(got.data(), bytes.data(), bytes.size()) == 0);

    ggml_backend_buffer_free(dst_buf);
    ggml_backend_buffer_free(src_buf);
    ggml_free(dst_ctx);
    ggml_free(src_ctx);
    ggml_backend_free(meta);
}

ggml_backend_meta_split_state split_axis0(const ggml_tensor *, void *) {
    return {GGML_BACKEND_SPLIT_AXIS_0, {8, 8}, {1}, 1};
}

// 67c71b1da's Meta set path asserts contiguous-or-mirrored. A non-contiguous
// non-mirrored destination aborted there, so it is not a supported copy. The
// capability now reports FAIL_STOP without staging or a destination write.
void test_non_contiguous_non_mirrored_dst_fail_stop(ggml_backend_dev_t cpu) {
    reset_fake();
    ggml_backend_dev_t children[] = { cpu, cpu };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(children, 2, split_axis0, nullptr);
    CHECK(meta_dev != nullptr);
    if (meta_dev == nullptr) {
        return;
    }
    ggml_backend_t meta = ggml_backend_dev_init(meta_dev, nullptr);
    CHECK(meta != nullptr);
    if (meta == nullptr) {
        return;
    }
    ggml_init_params params = { /*.mem_size=*/ 1u << 20, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    ggml_context * src_ctx = ggml_init(params);
    ggml_context * dst_ctx = ggml_init(params);
    CHECK(src_ctx != nullptr && dst_ctx != nullptr);
    if (src_ctx == nullptr || dst_ctx == nullptr) {
        return;
    }
    ggml_tensor * src = ggml_new_tensor_2d(src_ctx, GGML_TYPE_F32, 16, 4);
    ggml_tensor * dst = ggml_new_tensor_2d(dst_ctx, GGML_TYPE_F32, 16, 4);
    CHECK(src != nullptr && dst != nullptr);
    if (src == nullptr || dst == nullptr) {
        return;
    }
    src->nb[1] = 128;
    src->nb[2] = 128 * 4;
    src->nb[3] = 128 * 4;
    dst->nb[1] = 128;
    dst->nb[2] = 128 * 4;
    dst->nb[3] = 128 * 4;
    CHECK(!ggml_is_contiguous(dst));

    ggml_backend_buffer_t src_buf = ggml_backend_alloc_ctx_tensors_from_buft(
            src_ctx, ggml_backend_cpu_buffer_type());
    ggml_backend_buffer_t dst_buf = ggml_backend_alloc_ctx_tensors_from_buft(
            dst_ctx, ggml_backend_dev_buffer_type(meta_dev));
    CHECK(src_buf != nullptr && dst_buf != nullptr);
    if (src_buf == nullptr || dst_buf == nullptr) {
        return;
    }

    sync_fn_t fn = meta_sync_fn(meta);
    CHECK(fn != nullptr);
    CHECK(fn != nullptr && fn(nullptr, meta, src, dst) ==
          GGML_BACKEND_COPY_TENSOR_SYNC_FAIL_STOP);
    CHECK(g_fake.acquire_calls == 0);
    CHECK(g_fake.upload_calls == 0);
    CHECK(g_fake.enqueue_async_calls == 0);

    ggml_backend_buffer_free(dst_buf);
    ggml_backend_buffer_free(src_buf);
    ggml_free(dst_ctx);
    ggml_free(src_ctx);
    ggml_backend_free(meta);
}

// A Meta source is read under the staging submit gate. A fault published at
// gate entry must stop before the read and before any destination write.
void test_capability_source_get_gated(ggml_backend_dev_t cpu) {
    reset_fake();
    meta_pair p;
    CHECK(make_meta_pair(cpu, p));
    if (p.meta == nullptr) {
        return;
    }
    std::vector<int64_t> zeros(16, 0);
    ggml_backend_tensor_set(p.dst, zeros.data(), 0, zeros.size() * sizeof(int64_t));
    const std::vector<int64_t> data = make_data();
    ggml_backend_tensor_set(p.src, data.data(), 0, data.size() * sizeof(int64_t));
    reset_fake();
    g_fake.pending_on_enter = true;

    sync_fn_t fn = meta_sync_fn(p.meta);
    CHECK(fn != nullptr);
    CHECK(fn != nullptr && fn(p.meta, p.meta, p.src, p.dst) ==
          GGML_BACKEND_COPY_TENSOR_SYNC_FAIL_STOP);
    CHECK(g_fake.enter_submit_calls >= 1);
    CHECK(g_fake.acquire_calls == 0);
    CHECK(g_fake.upload_calls == 0);
    CHECK(g_fake.enqueue_async_calls == 0);
    CHECK(g_fake.pending);

    std::vector<int64_t> got(16, -1);
    ggml_backend_tensor_get(p.dst, got.data(), 0, got.size() * sizeof(int64_t));
    CHECK(std::memcmp(got.data(), zeros.data(), zeros.size() * sizeof(int64_t)) == 0);

    // Retained by the unknown-submission contract.
}

// A destination buffer that does not belong to backend_dst's device is an
// owner mismatch. The capability must fail-stop before any staging or write.
void test_capability_owner_mismatch(ggml_backend_dev_t cpu) {
    reset_fake();
    fixture f;
    CHECK(make_fixture(cpu, f, /*src_input=*/ true));
    if (f.meta == nullptr) {
        return;
    }
    std::vector<int64_t> zeros(16, 0);
    ggml_backend_tensor_set(f.dst, zeros.data(), 0, zeros.size() * sizeof(int64_t));
    const std::vector<int64_t> data = make_data();
    ggml_backend_tensor_set(f.src, data.data(), 0, data.size() * sizeof(int64_t));
    reset_fake();

    ggml_backend_device fake_dev = {};
    ggml_backend fake = *f.meta;
    fake.device = &fake_dev;
    sync_fn_t fn = meta_sync_fn(f.meta);
    CHECK(fn != nullptr);
    CHECK(fn != nullptr && fn(&fake, &fake, f.src, f.dst) ==
          GGML_BACKEND_COPY_TENSOR_SYNC_FAIL_STOP);
    CHECK(g_fake.acquire_calls == 0);
    CHECK(g_fake.upload_calls == 0);
    CHECK(g_fake.enqueue_async_calls == 0);

    std::vector<int64_t> got(16, -1);
    ggml_backend_tensor_get(f.dst, got.data(), 0, got.size() * sizeof(int64_t));
    CHECK(std::memcmp(got.data(), zeros.data(), zeros.size() * sizeof(int64_t)) == 0);

    free_fixture(f);
}

// A fault published when the fallback releases the submit gate is a
// fail-stop after the write; the destination is not replayed.
void test_capability_post_gate_fail_stop(ggml_backend_dev_t cpu) {
    reset_fake();
    fixture f;
    CHECK(make_fixture(cpu, f, /*src_input=*/ true));
    if (f.meta == nullptr) {
        return;
    }
    const std::vector<int64_t> data = make_data();
    ggml_backend_tensor_set(f.src, data.data(), 0, data.size() * sizeof(int64_t));
    reset_fake();
    g_fake.pending_on_leave = true;

    sync_fn_t fn = meta_sync_fn(f.meta);
    CHECK(fn != nullptr);
    CHECK(fn != nullptr && fn(nullptr, f.meta, f.src, f.dst) ==
          GGML_BACKEND_COPY_TENSOR_SYNC_FAIL_STOP);
    CHECK(g_fake.pending);
    CHECK(g_fake.enqueue_async_calls == 2);
    check_dst(f, data);

    const int enqueues = g_fake.enqueue_async_calls;
    CHECK(fn != nullptr && fn(nullptr, f.meta, f.src, f.dst) ==
          GGML_BACKEND_COPY_TENSOR_SYNC_FAIL_STOP);
    CHECK(g_fake.enqueue_async_calls == enqueues);

    // Retained by the unknown-submission contract.
}

// A fault published at the gated synchronize boundary must stop before the
// drain and before any copy. The pair is retained by the unknown-submission
// contract.
void test_gated_sync_blocks_unproven(ggml_backend_dev_t cpu) {
    reset_fake();
    meta_pair p;
    CHECK(make_meta_pair(cpu, p));
    if (p.meta == nullptr) {
        return;
    }
    std::vector<int64_t> zeros(16, 0);
    ggml_backend_tensor_set(p.dst, zeros.data(), 0, zeros.size() * sizeof(int64_t));
    const std::vector<int64_t> data = make_data();
    ggml_backend_tensor_set(p.src, data.data(), 0, data.size() * sizeof(int64_t));
    reset_fake();
    g_fake.pending_on_enter = true;

    CHECK(!ggml_backend_tensor_copy_async(p.meta, p.meta, p.src, p.dst));
    CHECK(g_fake.enter_submit_calls >= 1);
    CHECK(g_fake.acquire_calls == 0);
    CHECK(g_fake.upload_calls == 0);
    CHECK(g_fake.enqueue_async_calls == 0);
    CHECK(g_fake.pending);

    std::vector<int64_t> got(16, -1);
    ggml_backend_tensor_get(p.dst, got.data(), 0, got.size() * sizeof(int64_t));
    CHECK(std::memcmp(got.data(), zeros.data(), zeros.size() * sizeof(int64_t)) == 0);

    // Retained by the unknown-submission contract.
}

// A non-input source must not lease a slot even though its name matches a
// dedicated kind. The blocking path still writes every child.
void test_non_input_source(ggml_backend_dev_t cpu) {
    reset_fake();
    fixture f;
    CHECK(make_fixture(cpu, f, /*src_input=*/ false));
    if (f.meta == nullptr) {
        return;
    }
    const std::vector<int64_t> data = make_data();
    ggml_backend_tensor_set(f.src, data.data(), 0, data.size() * sizeof(int64_t));

    ggml_backend_buffer_copy_tensor(f.src, f.dst);

    CHECK(g_fake.acquire_calls == 0);
    CHECK(g_fake.upload_calls == 0);
    CHECK(g_fake.enqueue_sync_calls == 0);
    CHECK(g_fake.enter_submit_calls == 1);
    CHECK(!g_fake.pending);
    check_dst(f, data);

    free_fixture(f);
}

// A direct set of the decorated destination has no source identity. It must
// stay on the destination-based path: the exact union name is not parsed.
void test_direct_set_has_no_source(ggml_backend_dev_t cpu) {
    reset_fake();
    fixture f;
    CHECK(make_fixture(cpu, f, /*src_input=*/ true));
    if (f.meta == nullptr) {
        return;
    }
    const std::vector<int64_t> data = make_data();

    ggml_backend_tensor_set(f.dst, data.data(), 0, data.size() * sizeof(int64_t));

    CHECK(g_fake.acquire_calls == 0);
    CHECK(g_fake.upload_calls == 0);
    CHECK(g_fake.enqueue_sync_calls == 0);
    CHECK(!g_fake.pending);
    check_dst(f, data);

    free_fixture(f);
}

// The v4 buffer-only path has no owner and must not stage. The owner-aware
// capability covers the partial submission and no-replay path separately.
void test_partial_submission_not_replayed(ggml_backend_dev_t cpu) {
    reset_fake();
    g_fake.partial_after = 1;
    fixture f;
    CHECK(make_fixture(cpu, f, /*src_input=*/ true));
    if (f.meta == nullptr) {
        return;
    }
    const std::vector<int64_t> data = make_data();
    ggml_backend_tensor_set(f.src, data.data(), 0, data.size() * sizeof(int64_t));

    ggml_backend_buffer_copy_tensor(f.src, f.dst);

    CHECK(g_fake.acquire_calls == 0);
    CHECK(g_fake.upload_calls == 0);
    CHECK(g_fake.enqueue_sync_calls == 0);
    CHECK(g_fake.mark_partial_calls == 0);
    CHECK(!g_fake.pending);
    check_dst(f, data);

    free_fixture(f);
}

// A token issued by provider A must drain/release through A even after B has
// registered and become the active provider. B's own token must route to B.
void test_owner_provider_binding(ggml_backend_dev_t cpu) {
    reset_fake();
    g_fake_b = fake_staging_state{};

    fixture f;
    CHECK(make_fixture(cpu, f, /*src_input=*/ true, /*dst_view=*/ false, /*bind_owner=*/ false));
    if (f.meta == nullptr) {
        return;
    }

    CHECK(ggml_backend_meta_staging_provider_reg_for_device(f.meta->device) ==
          ggml_backend_dev_backend_reg(cpu));
    uint64_t owner_a = 0;
    uint64_t generation_a = 0;
    CHECK(ggml_backend_meta_staging_owner_acquire_for_device(
            f.meta->device, g_next_run_id++, 0, &owner_a, &generation_a));
    CHECK((owner_a >> 48) != 0);

    uint64_t owner_copy = 0;
    uint64_t generation_copy = 0;
    CHECK(ggml_backend_meta_staging_owner_acquire_for_backend(
            f.meta, g_next_run_id++, 0, &owner_copy, &generation_copy));

    CHECK(ggml_backend_meta_staging_register_for_reg(&g_fake_reg_b, &g_fake_api_b));
    CHECK(ggml_backend_meta_staging_get() == &g_fake_api_b);

    CHECK(ggml_backend_meta_staging_owner_drain(owner_a, generation_a) ==
          int(owner_drain_result::drained));
    CHECK(g_fake.owner_drain_calls == 1);
    CHECK(g_fake_b.owner_drain_calls == 0);
    CHECK(ggml_backend_meta_staging_owner_release(owner_a, generation_a));
    CHECK(g_fake.owner_release_calls == 1);
    CHECK(g_fake_b.owner_release_calls == 0);

    uint64_t owner_b = 0;
    uint64_t generation_b = 0;
    CHECK(ggml_backend_meta_staging_owner_acquire_for_reg(
            &g_fake_reg_b, g_next_run_id++, 0, &owner_b, &generation_b));
    CHECK((owner_b >> 48) != (owner_a >> 48));
    const uint64_t old_b_tag = owner_b >> 48;
    CHECK(g_fake_b.owner_acquire_calls == 1);

    // The legacy selector has no backend and must not guess the active
    // provider once two providers exist.
    uint64_t legacy_owner = 0;
    uint64_t legacy_generation = 0;
    CHECK(!ggml_backend_meta_staging_owner_acquire(
            g_next_run_id++, 0, &legacy_owner, &legacy_generation));

    CHECK(ggml_backend_meta_staging_owner_drain(owner_b, generation_b) ==
          int(owner_drain_result::drained));
    CHECK(g_fake_b.owner_drain_calls == 1);
    CHECK(g_fake.owner_drain_calls == 1);
    CHECK(ggml_backend_meta_staging_owner_release(owner_b, generation_b));
    CHECK(g_fake_b.owner_release_calls == 1);
    CHECK(g_fake.owner_release_calls == 1);

    // Ownerless submission state must use the backend's child provider, not
    // the active provider. A reports pending; active B reports clean.
    {
        fixture ownerless;
        CHECK(make_fixture(cpu, ownerless, /*src_input=*/ true,
                           /*dst_view=*/ false, /*bind_owner=*/ false));
        if (ownerless.meta != nullptr) {
            g_fake.pending = true;
            auto pending_fn = (bool (*)(ggml_backend_t))
                ggml_backend_reg_get_proc_address(ownerless.meta->device->reg,
                                                  "ggml_backend_submission_pending");
            CHECK(pending_fn != nullptr && pending_fn(ownerless.meta));
            g_fake.pending = false;
            free_fixture(ownerless);
        }
    }

    // The owner-aware copy path must route upload/enqueue/submission through
    // A even though B is active. Bind A's tagged token to a real Meta backend,
    // then copy through the v2 capability; B's acquire_owner must stay unused.
    g_fake.acquire_owner_calls = 0;
    g_fake_b.acquire_owner_calls = 0;
    if (f.meta != nullptr) {
        // B was issued by a different DSO than this backend's CPU children.
        CHECK(!ggml_backend_meta_set_staging_owner(f.meta, owner_b, generation_b));
        // Ambiguous legacy and unknown-tag tokens must not be accepted while
        // two providers exist.
        CHECK(!ggml_backend_meta_set_staging_owner(f.meta, 1, 1));
        CHECK(!ggml_backend_meta_set_staging_owner(
                f.meta, (uint64_t(99) << 48) | 1, 1));
        CHECK(ggml_backend_meta_set_staging_owner(f.meta, owner_copy, generation_copy));
        const std::vector<int64_t> data = make_data();
        ggml_backend_tensor_set(f.src, data.data(), 0, data.size() * sizeof(int64_t));
        sync_fn_t fn = meta_sync_fn(f.meta);
        CHECK(fn != nullptr);
        if (fn != nullptr) {
            CHECK(fn(nullptr, f.meta, f.src, f.dst) ==
                  GGML_BACKEND_COPY_TENSOR_SYNC_OK);
        }
        CHECK(g_fake.acquire_owner_calls == 1);
        CHECK(g_fake_b.acquire_owner_calls == 0);
        check_dst(f, data);
        free_fixture(f);
    }
    CHECK(ggml_backend_meta_staging_owner_drain(owner_copy, generation_copy) ==
          int(owner_drain_result::drained));
    CHECK(ggml_backend_meta_staging_owner_release(owner_copy, generation_copy));

    // Provider lifetime: an active owner blocks unload; a quiescent unload
    // removes the provider; reloading the same DSO gets a fresh tag and the
    // old tag can never route again.
    g_fake_reg_b.api_version = GGML_BACKEND_API_VERSION;
    g_fake_reg_b.iface.get_name = [](ggml_backend_reg_t) { return "fake-b"; };
    g_fake_reg_b.iface.get_device_count = [](ggml_backend_reg_t) { return size_t(0); };
    ggml_backend_register(&g_fake_reg_b);
    const size_t count_before = ggml_backend_reg_count();

    uint64_t live_owner = 0;
    uint64_t live_generation = 0;
    CHECK(ggml_backend_meta_staging_owner_acquire_for_reg(
            &g_fake_reg_b, g_next_run_id++, 0, &live_owner, &live_generation));
    ggml_backend_unload(&g_fake_reg_b);
    CHECK(ggml_backend_reg_count() == count_before);

    CHECK(ggml_backend_meta_staging_owner_drain(live_owner, live_generation) ==
          int(owner_drain_result::drained));
    CHECK(ggml_backend_meta_staging_owner_release(live_owner, live_generation));
    ggml_backend_unload(&g_fake_reg_b);
    CHECK(ggml_backend_reg_count() + 1 == count_before);

    CHECK(ggml_backend_meta_staging_register_for_reg(&g_fake_reg_b, &g_fake_api_b));
    uint64_t reloaded_owner = 0;
    uint64_t reloaded_generation = 0;
    CHECK(ggml_backend_meta_staging_owner_acquire_for_reg(
            &g_fake_reg_b, g_next_run_id++, 0, &reloaded_owner, &reloaded_generation));
    CHECK((reloaded_owner >> 48) != old_b_tag);
    CHECK(ggml_backend_meta_staging_owner_drain(reloaded_owner, reloaded_generation) ==
          int(owner_drain_result::drained));
    CHECK(ggml_backend_meta_staging_owner_release(reloaded_owner, reloaded_generation));
}

} // namespace

int main() {
    ggml_backend_reg_t cpu_reg = ggml_backend_cpu_reg();
    CHECK(cpu_reg != nullptr);
    if (cpu_reg != nullptr) {
        // Bind provider A to the real CPU child registry so the Meta backend
        // mapping is exercised, not a global active provider.
        CHECK(ggml_backend_meta_staging_register_for_reg(cpu_reg, &g_fake_api));
        CHECK(ggml_backend_meta_staging_get() == &g_fake_api);
    }
    ggml_backend_dev_t cpu = cpu_reg != nullptr ? ggml_backend_reg_dev_get(cpu_reg, 0) : nullptr;
    CHECK(cpu != nullptr);
    if (cpu == nullptr) {
        return 1;
    }

    test_meta_drain_receipt(cpu);
    test_source_identity_and_content(cpu);
    test_source_identity_without_merge_wait(cpu);
    test_capability_source_identity(cpu);
    test_capability_two_backend_selection(cpu);
    test_capability_zero(cpu);
    test_dst_view_falls_back(cpu);
    test_owner_path_no_ownerless_meta_staging(cpu);
    test_public_sync_copy_no_staging(cpu);
    test_capability_source_pending(cpu);
    test_meta_to_meta_copy_async(cpu);
    test_non_contiguous_host_source(cpu);
    test_non_contiguous_non_mirrored_dst_fail_stop(cpu);
    test_non_input_source(cpu);
    test_direct_set_has_no_source(cpu);
    test_partial_submission_not_replayed(cpu);
    test_capability_partial(cpu);
    test_capability_source_get_gated(cpu);
    test_capability_owner_mismatch(cpu);
    test_capability_post_gate_fail_stop(cpu);
    test_gated_sync_blocks_unproven(cpu);
    test_owner_provider_binding(cpu);

    if (g_failures != 0) {
        std::fprintf(stderr, "TP5_STAGING_SYNC_SRC_TEST FAIL (%d)\n", g_failures);
        return 1;
    }
    std::printf("TP5_STAGING_SYNC_SRC_TEST PASS\n");
    return 0;
}
