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

#include <cstdint>
#include <cstdio>
#include <cstring>
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

struct fake_staging_state {
    bool pending            = false;
    bool configured         = true;
    int  partial_after      = 0; // 0: never reject; N: reject enqueue N+1
    bool merge_wait_mode    = true;
    int  acquire_calls      = 0;
    int  upload_calls       = 0;
    int  enqueue_sync_calls = 0;
    int  mark_partial_calls = 0;
    int  release_calls      = 0;
    int  enter_submit_calls = 0;
    kind last_kind          = kind::count;
    size_t last_bytes       = 0;
    std::vector<uint8_t> slot;
    ggml_tensor * enqueue_tensors[8] = {};
};

fake_staging_state g_fake;

bool fake_enabled()    { return g_fake.configured && !g_fake.pending; }
bool fake_usable()     { return !g_fake.pending; }
bool fake_configured() { return g_fake.configured; }
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

bool fake_acquire_nowait(uint32_t, kind, size_t, lease *) {
    return false;
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

int fake_first_error() {
    return 0;
}

bool fake_enter_submit() {
    ++g_fake.enter_submit_calls;
    return !g_fake.pending;
}

void fake_leave_submit() {
}

size_t fake_unknown_count() {
    return g_fake.pending ? 1 : 0;
}

bool fake_submission_pending() {
    return g_fake.pending;
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
    /* .can_enqueue_async = */ nullptr,
    /* .enqueue_async     = */ nullptr,
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
};

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

bool make_fixture(ggml_backend_dev_t cpu, fixture & f, bool src_input, bool dst_view = false) {
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

// The source identity comes from src, not from the decorated destination:
// union_k is leased, the bytes go through upload + one enqueue per child, and
// the real Meta read path returns them.
void test_source_identity_and_content(ggml_backend_dev_t cpu) {
    reset_fake();
    fixture f;
    CHECK(make_fixture(cpu, f, /*src_input=*/ true));
    if (f.meta == nullptr) {
        return;
    }
    const std::vector<int64_t> data = make_data();
    ggml_backend_tensor_set(f.src, data.data(), 0, data.size() * sizeof(int64_t));

    ggml_backend_tensor_copy(f.src, f.dst);

    CHECK(g_fake.acquire_calls == 1);
    CHECK(g_fake.last_kind == kind::union_k);
    CHECK(g_fake.last_bytes == data.size() * sizeof(int64_t));
    CHECK(g_fake.upload_calls == 1);
    CHECK(g_fake.enqueue_sync_calls == 2);
    CHECK(g_fake.mark_partial_calls == 0);
    CHECK(g_fake.release_calls == 1);
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

    ggml_backend_tensor_copy(f.src, f.dst);

    CHECK(g_fake.acquire_calls == 1);
    CHECK(g_fake.last_kind == kind::union_k);
    CHECK(g_fake.upload_calls == 1);
    CHECK(g_fake.enqueue_sync_calls == 0);
    CHECK(g_fake.enter_submit_calls == 1);
    CHECK(g_fake.release_calls == 1);
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

    ggml_backend_tensor_copy(f.src, f.dst);

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

    ggml_backend_tensor_copy(f.src, f.dst);

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

// Rank 0 is queued and rank 1 is cleanly rejected. mark_partial must publish
// the fail-stop state; the blocking fallback must not replay the destination.
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

    ggml_backend_tensor_copy(f.src, f.dst);

    CHECK(g_fake.acquire_calls == 1);
    CHECK(g_fake.upload_calls == 1);
    CHECK(g_fake.enqueue_sync_calls == 2);
    CHECK(g_fake.mark_partial_calls == 1);
    CHECK(g_fake.release_calls == 1);
    CHECK(g_fake.enter_submit_calls == 0);
    CHECK(g_fake.pending);

    ggml_tensor * rank0 = g_fake.enqueue_tensors[0];
    ggml_tensor * rank1 = g_fake.enqueue_tensors[1];
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

    // The public capability sees the partial submission: no native error, but
    // pending. A later graph compute is refused before touching the graph.
    ggml_backend_dev_t meta_dev = ggml_backend_get_device(f.meta);
    CHECK(meta_dev != nullptr);
    ggml_backend_reg_t reg = meta_dev != nullptr ? ggml_backend_dev_backend_reg(meta_dev) : nullptr;
    CHECK(reg != nullptr);
    auto error_fn = reg != nullptr
        ? (int (*)(ggml_backend_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_submission_error")
        : nullptr;
    auto pending_fn = reg != nullptr
        ? (bool (*)(ggml_backend_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_submission_pending")
        : nullptr;
    CHECK(error_fn != nullptr);
    CHECK(pending_fn != nullptr);
    if (error_fn != nullptr && pending_fn != nullptr) {
        CHECK(error_fn(f.meta) == 0);
        CHECK(pending_fn(f.meta));
    }
    ggml_cgraph * graph = ggml_new_graph(f.dst_ctx);
    CHECK(graph != nullptr);
    if (graph != nullptr) {
        CHECK(ggml_backend_graph_compute(f.meta, graph) == GGML_STATUS_FAILED);
    }

    const int enqueues = g_fake.enqueue_sync_calls;
    const int marks     = g_fake.mark_partial_calls;
    const int enters    = g_fake.enter_submit_calls;
    ggml_backend_tensor_copy(f.src, f.dst);
    CHECK(g_fake.enqueue_sync_calls == enqueues);
    CHECK(g_fake.mark_partial_calls == marks);
    CHECK(g_fake.enter_submit_calls == enters + 1); // the gate is the only effect
    CHECK(g_fake.pending);

    // Consumer-visible no-replay proof: both child destinations are byte-for-
    // byte unchanged, including the rank that was rejected before its copy.
    if (rank0 != nullptr && rank1 != nullptr) {
        std::vector<uint8_t> rank0_after(rank0_before.size());
        std::vector<uint8_t> rank1_after(rank1_before.size());
        ggml_backend_tensor_get(rank0, rank0_after.data(), 0, rank0_after.size());
        ggml_backend_tensor_get(rank1, rank1_after.data(), 0, rank1_after.size());
        CHECK(std::memcmp(rank0_after.data(), rank0_before.data(), rank0_after.size()) == 0);
        CHECK(std::memcmp(rank1_after.data(), rank1_before.data(), rank1_after.size()) == 0);
    }

    // The destination may be partially written and is retained by the
    // unknown-submission contract. The process owns it until exit.
}

} // namespace

int main() {
    CHECK(ggml_backend_meta_staging_register(&g_fake_api));
    CHECK(ggml_backend_meta_staging_get() == &g_fake_api);

    ggml_backend_reg_t cpu_reg = ggml_backend_cpu_reg();
    CHECK(cpu_reg != nullptr);
    ggml_backend_dev_t cpu = cpu_reg != nullptr ? ggml_backend_reg_dev_get(cpu_reg, 0) : nullptr;
    CHECK(cpu != nullptr);
    if (cpu == nullptr) {
        return 1;
    }

    test_source_identity_and_content(cpu);
    test_source_identity_without_merge_wait(cpu);
    test_dst_view_falls_back(cpu);
    test_non_input_source(cpu);
    test_direct_set_has_no_source(cpu);
    test_partial_submission_not_replayed(cpu);

    if (g_failures != 0) {
        std::fprintf(stderr, "TP5_STAGING_SYNC_SRC_TEST FAIL (%d)\n", g_failures);
        return 1;
    }
    std::printf("TP5_STAGING_SYNC_SRC_TEST PASS\n");
    return 0;
}
