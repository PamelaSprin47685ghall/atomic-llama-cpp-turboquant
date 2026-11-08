// TP5 WS4 CPU test for the persistent pinned staging pool.
//
// Build and run without a GPU or HIP:
//   g++ -std=c++17 -O2 -I ggml/include tests/test-tp5-staging-pool.cpp -o /tmp/test-tp5-staging
//   /tmp/test-tp5-staging

#include "../ggml/src/ggml-cuda/staging.h"
#include "../ggml/src/ggml-impl.h"  // ggml_are_same_layout (static inline)

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace ggml_cuda_staging;

namespace {

int g_allocs = 0;
int g_frees = 0;
int g_events = 0;
int g_records = 0;
int g_syncs = 0;
int g_copies = 0;
bool g_fail_alloc = false;
bool g_fail_sync = false;
int  g_fail_copy_remaining = 0;
bool g_fail_event_create = false;
bool g_fail_event_record = false;
int  g_peek_error = 0;

bool fake_alloc(size_t bytes, void ** out) {
    if (g_fail_alloc) {
        return false;
    }
    void * p = std::malloc(bytes);
    if (p == nullptr) {
        return false;
    }
    std::memset(p, 0, bytes);
    *out = p;
    ++g_allocs;
    return true;
}

void fake_free(void * p) {
    if (p != nullptr) {
        std::free(p);
        ++g_frees;
    }
}

bool fake_event_create(void ** out) {
    if (g_fail_event_create) {
        return false;
    }
    *out = std::malloc(1);
    ++g_events;
    return *out != nullptr;
}

bool fake_event_record(void *, void *) {
    if (g_fail_event_record) {
        return false;
    }
    ++g_records;
    return true;
}

bool fake_event_sync(void *) {
    ++g_syncs;
    return !g_fail_sync;
}

void fake_event_destroy(void * p) {
    std::free(p);
}

bool fake_copy(void * dst, const void * src, size_t bytes, void *) {
    if (g_fail_copy_remaining > 0) {
        --g_fail_copy_remaining;
        return false;
    }
    std::memcpy(dst, src, bytes);
    ++g_copies;
    return true;
}

bool fake_device_ptr(void *, void ** out) {
    *out = reinterpret_cast<void *>(uintptr_t(0x1000));
    return true;
}

int g_current_device = 0;
int g_select_count = 0;
int g_restore_count = 0;
int g_last_selected = -1;
int g_last_restored = -1;

int fake_current_device() {
    return g_current_device;
}

bool fake_select_device(int rank) {
    g_last_selected = rank;
    ++g_select_count;
    return true;
}

bool fake_restore_device(int physical) {
    g_last_restored = physical;
    ++g_restore_count;
    return true;
}

int fake_peek_error() {
    return g_peek_error;
}

backend_ops fake_ops() {
    backend_ops ops;
    ops.host_alloc    = fake_alloc;
    ops.host_free     = fake_free;
    ops.event_create  = fake_event_create;
    ops.event_record  = fake_event_record;
    ops.event_sync    = fake_event_sync;
    ops.event_destroy = fake_event_destroy;
    ops.copy_h2d      = fake_copy;
    ops.device_ptr    = fake_device_ptr;
    ops.peek_error    = fake_peek_error;
    return ops;
}

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

int test_lease_release() {
    pool p(fake_ops());
    uint8_t src[1024];
    std::memset(src, 0x5a, sizeof(src));

    CHECK(p.usable());
    CHECK(p.pool_bytes() == 0);

    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(a.host != nullptr);
    CHECK(a.cap >= sizeof(src));
    CHECK(a.epoch == 1);
    CHECK(a.slot == 0);
    CHECK(a.which == kind::tokens);
    CHECK(p.pool_bytes() >= sizeof(src));

    CHECK(p.upload(a, src, sizeof(src)));
    CHECK(std::memcmp(a.host, src, sizeof(src)) == 0);

    p.release(a);
    lease b;
    CHECK(p.acquire(0, kind::tokens, 16, &b));
    CHECK(b.epoch == 2);
    CHECK(b.host == a.host);  // reuse, no reallocation
    CHECK(g_allocs == 1);
    p.release(b);
    return 0;
}

int test_double_buffer_and_stale_release() {
    pool p(fake_ops());
    uint8_t src[4096] = {};

    lease a;
    lease b;
    CHECK(p.acquire(0, kind::mtp_h, sizeof(src), &a));
    CHECK(p.acquire(0, kind::mtp_h, sizeof(src), &b));
    CHECK(a.slot != b.slot);
    CHECK(a.host != b.host);

    // Both slots are leased. A third acquire must fail, never steal a lease.
    lease c;
    CHECK(!p.acquire(0, kind::mtp_h, sizeof(src), &c));
    CHECK(p.snapshot().no_slot == 1);
    CHECK(p.usable());
    CHECK(p.upload(a, src, sizeof(src)));  // both leases stay valid
    CHECK(p.upload(b, src, sizeof(src)));

    // Releasing one slot makes it reusable.
    p.release(a);
    CHECK(p.acquire(0, kind::mtp_h, sizeof(src), &c));
    CHECK(c.slot == a.slot);
    CHECK(c.epoch > b.epoch);

    // The old lease no longer names a live slot.
    CHECK(!p.upload(a, src, 16));
    p.release(a);  // stale
    CHECK(p.snapshot().stale_release == 1);

    CHECK(p.upload(c, src, sizeof(src)));
    p.release(b);
    p.release(c);
    return 0;
}

int test_pending_slot_reuse() {
    pool p(fake_ops());
    uint8_t src[64] = {};
    uint8_t dst[64] = {};

    // Both slots become pending but unleased: enqueue, then release without
    // waiting.
    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)));
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr));
    p.release(a);

    lease b;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &b));
    CHECK(b.slot != a.slot);
    CHECK(p.upload(b, src, sizeof(src)));
    CHECK(p.enqueue(b, 1, dst, sizeof(src), nullptr));
    p.release(b);

    // The next acquire waits one pending event and reuses that slot.
    const int syncs_before = g_syncs;
    lease c;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &c));
    CHECK(c.slot == a.slot);
    CHECK(g_syncs == syncs_before + 1);
    p.release(c);
    return 0;
}

int test_acquire_nowait() {
    pool p(fake_ops());
    uint8_t src[64] = {};
    uint8_t dst[64] = {};

    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)));
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr));
    p.release(a);

    lease b;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &b));
    CHECK(b.slot != a.slot);
    CHECK(p.upload(b, src, sizeof(src)));
    CHECK(p.enqueue(b, 1, dst, sizeof(src), nullptr));
    p.release(b);

    // Both slots are pending. The async lane must not wait: it fails and
    // falls back to the pageable path.
    const int syncs_before = g_syncs;
    lease c;
    CHECK(!p.acquire(0, kind::tokens, sizeof(src), &c, /*wait_pending =*/ false));
    CHECK(g_syncs == syncs_before);
    CHECK(p.snapshot().no_slot >= 1);

    // The synchronous lane still waits one event and reuses a slot.
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &c));
    CHECK(g_syncs == syncs_before + 1);
    p.release(c);
    return 0;
}

int test_note_source_fast_path() {
    pool p(fake_ops());
    // No slot has been allocated: note_source must not take the lock.
    p.note_source(reinterpret_cast<const void *>(uintptr_t(0x1000)), 16);
    p.note_source(nullptr, 1);
    const auto s = p.snapshot();
    CHECK(s.misses == 2);
    CHECK(s.hits == 0);
    return 0;
}

int test_window_isolation() {
    pool p(fake_ops());
    lease a;
    lease b;
    CHECK(p.acquire(0, kind::pos, 64, &a));
    CHECK(p.acquire(1, kind::pos, 64, &b));
    CHECK(a.host != b.host);
    CHECK(a.slot == 0);
    CHECK(b.slot == 0);
    CHECK(a.window == 0);
    CHECK(b.window == 1);
    CHECK(p.acquire(9, kind::pos, 64, &b) == false);
    p.release(a);
    p.release(b);
    return 0;
}

int test_pool_cap() {
    pool p(fake_ops(), 4u << 20);
    lease a;

    // A single request above the cap is a miss, not a permanent disable.
    CHECK(!p.acquire(0, kind::generic, (4u << 20) + 1, &a));
    CHECK(p.usable());

    CHECK(p.acquire(0, kind::generic, 3u << 20, &a));
    lease b;
    CHECK(!p.acquire(0, kind::tokens, 2u << 20, &b));
    CHECK(p.usable());

    lease c;
    CHECK(p.acquire(0, kind::pos, 1u << 20, &c));
    CHECK(p.pool_bytes() == (4u << 20));
    CHECK(!p.acquire(0, kind::out_ids, 1, &b));
    p.release(a);
    p.release(c);
    return 0;
}

int test_alloc_failure_fail_closed() {
    pool p(fake_ops());
    lease a;
    g_fail_alloc = true;
    CHECK(!p.acquire(0, kind::tokens, 1024, &a));
    CHECK(!p.usable());
    g_fail_alloc = false;
    CHECK(!p.acquire(0, kind::tokens, 1024, &a));
    return 0;
}

int test_upload_bounds() {
    pool p(fake_ops());
    lease a;
    uint8_t src[128] = {};
    CHECK(p.acquire(0, kind::generic, 64, &a));
    CHECK(!p.upload(a, src, 65));
    CHECK(p.upload(a, src, 64));
    CHECK(!p.upload(a, nullptr, 64));
    CHECK(!p.upload(a, src, 0));
    p.release(a);
    return 0;
}

int test_upload_rows() {
    pool p(fake_ops());
    uint8_t src[3 * 16];
    for (size_t i = 0; i < sizeof(src); ++i) {
        src[i] = uint8_t(i);
    }
    lease a;
    CHECK(p.acquire(0, kind::kq_mask, 3 * 16, &a));

    // Strided destination keeps the source stride (the meta 2D consumer reads
    // with that stride).
    CHECK(p.upload_rows(a, src, 16, 16, 8, 3));
    for (size_t r = 0; r < 3; ++r) {
        CHECK(std::memcmp(a.host + r * 16, src + r * 16, 8) == 0);
    }

    // Packed destination removes the holes.
    CHECK(p.upload_rows(a, src, 16, 8, 8, 3));
    for (size_t r = 0; r < 3; ++r) {
        CHECK(std::memcmp(a.host + r * 8, src + r * 16, 8) == 0);
    }

    CHECK(!p.upload_rows(a, src, 16, 16, 8, 4));  // span exceeds the slot
    CHECK(!p.upload_rows(a, src, 4, 8, 8, 3));    // src_stride < row_bytes
    CHECK(!p.upload_rows(a, src, 8, 4, 8, 3));    // dst_stride < row_bytes
    p.release(a);
    return 0;
}

int test_enqueue_wait() {
    pool p(fake_ops());
    uint8_t src[64];
    std::memset(src, 0x33, sizeof(src));
    uint8_t dst[64] = {};

    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)));

    const int syncs_before = g_syncs;
    const int copies_before = g_copies;
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr));
    CHECK(p.enqueue(a, 4, dst, sizeof(src), nullptr));
    CHECK(g_copies == copies_before + 2);
    CHECK(p.wait(a));
    CHECK(g_syncs == syncs_before + 2);
    CHECK(std::memcmp(dst, src, sizeof(src)) == 0);

    CHECK(!p.enqueue(a, 99, dst, sizeof(src), nullptr));
    CHECK(!p.enqueue(a, -1, dst, sizeof(src), nullptr));
    CHECK(!p.enqueue(a, 0, dst, sizeof(src) + 1, nullptr));
    p.release(a);
    return 0;
}

int test_merged_wait() {
    pool p(fake_ops());
    uint8_t src[64];
    std::memset(src, 0x44, sizeof(src));
    uint8_t dst[5][64] = {};

    lease a;
    CHECK(p.acquire(0, kind::mtp_h, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)));

    const int copies_before = g_copies;
    const int syncs_before = g_syncs;
    const uint64_t enqueues_before = p.snapshot().enqueues;
    for (int rank = 0; rank < 5; ++rank) {
        CHECK(p.enqueue(a, rank, dst[rank], sizeof(src), nullptr));
    }
    CHECK(g_copies == copies_before + 5);
    CHECK(p.snapshot().enqueues == enqueues_before + 5);

    // One wait covers all five rank events.
    const uint64_t waits_before = p.snapshot().waits;
    CHECK(p.wait(a));
    CHECK(g_syncs == syncs_before + 5);
    CHECK(p.snapshot().waits == waits_before + 1);
    for (int rank = 0; rank < 5; ++rank) {
        CHECK(std::memcmp(dst[rank], src, sizeof(src)) == 0);
    }
    p.release(a);
    return 0;
}

int test_wait_failure_disables() {
    pool p(fake_ops());
    uint8_t src[64] = {};
    uint8_t dst[64] = {};
    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)));
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr));

    g_fail_sync = true;
    g_peek_error = 43;
    CHECK(!p.wait(a));
    CHECK(!p.usable());
    CHECK(p.snapshot().quarantine == 1);
    CHECK(p.snapshot().first_error == 43);
    g_fail_sync = false;
    g_peek_error = 0;
    return 0;
}

int test_fault_event_create_quarantine() {
    pool p(fake_ops());
    uint8_t src[64] = {};
    uint8_t dst[64] = {};
    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)));

    // The copy succeeds, the event cannot be created: completion is
    // unprovable, so the slot is quarantined and the pool stops.
    g_fail_event_create = true;
    g_peek_error = 11;
    CHECK(!p.enqueue(a, 0, dst, sizeof(src), nullptr));
    g_fail_event_create = false;
    CHECK(!p.usable());
    CHECK(p.snapshot().quarantine == 1);
    CHECK(p.snapshot().first_error == 11);
    CHECK(p.pool_bytes() > 0);  // the pinned buffer is retained

    p.release(a);
    CHECK(!p.upload(a, src, sizeof(src)));  // released lease is stale
    CHECK(p.snapshot().quarantine == 1);
    g_peek_error = 0;
    return 0;
}

int test_fault_event_record_quarantine() {
    pool p(fake_ops());
    uint8_t src[64] = {};
    uint8_t dst[64] = {};
    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)));

    g_fail_event_record = true;
    g_peek_error = 12;
    CHECK(!p.enqueue(a, 0, dst, sizeof(src), nullptr));
    g_fail_event_record = false;
    CHECK(!p.usable());
    CHECK(p.snapshot().quarantine == 1);
    CHECK(p.snapshot().first_error == 12);
    p.release(a);
    CHECK(p.snapshot().quarantine == 1);
    g_peek_error = 0;
    return 0;
}

int test_owns_and_diagnostics() {
    pool p(fake_ops());
    lease a;
    CHECK(p.acquire(0, kind::generic, 256, &a));
    CHECK(p.owns(a.host, 256));
    CHECK(p.owns(a.host + 16, 32));
    CHECK(!p.owns(a.host + 200, 100));
    CHECK(!p.owns(nullptr, 1));
    CHECK(!p.owns(a.host, 0));

    p.note_source(a.host, 128);
    p.note_source(a.host + 200, 100);
    p.note_source(nullptr, 1);
    const auto s = p.snapshot();
    CHECK(s.hits == 1);
    CHECK(s.misses == 2);
    CHECK(s.leases == 1);
    p.dump(stderr);
    p.release(a);
    return 0;
}

int test_device_ptr() {
    pool p(fake_ops());
    lease a;
    CHECK(p.acquire(0, kind::out_ids, 32, &a));
    CHECK(p.device_ptr(a, 0) != nullptr);
    CHECK(p.device_ptr(a, 99) == nullptr);
    p.release(a);
    return 0;
}

int test_staging_candidate() {
    const size_t max_bytes = 2u << 20;

    CHECK(classify_name("inp_tokens") == kind::tokens);
    CHECK(classify_name("union_inp_tokens") == kind::tokens);
    CHECK(classify_name("inp_pos") == kind::pos);
    CHECK(classify_name("inp_out_ids") == kind::out_ids);
    CHECK(classify_name("attn_inp_kq_mask") == kind::kq_mask);
    CHECK(classify_name("mtp_h_input") == kind::mtp_h);
    CHECK(classify_name("inp_h") == kind::mtp_h);
    CHECK(classify_name("") == kind::generic);
    CHECK(classify_name(nullptr) == kind::generic);
    CHECK(classify_name("blk.0.attn_q.weight") == kind::generic);
    CHECK(classify_name("position_embd.weight") == kind::generic);
    CHECK(classify_name("inp_hca") == kind::generic);

    kind k = kind::count;
    // The MTP token tensor is unnamed but carries the input flag: it stages
    // through the generic rotation.
    CHECK(staging_candidate(true, "", 128, max_bytes, &k));
    CHECK(k == kind::generic);
    // Large unnamed blocks stay out of the generic rotation.
    CHECK(!staging_candidate(true, "", kGenericMaxBytes + 1, max_bytes, &k));
    CHECK(staging_candidate(true, "", kGenericMaxBytes, max_bytes, &k));
    // A weight without the input flag never stages, even with a matching name.
    CHECK(!staging_candidate(false, "inp_tokens", 128, max_bytes, &k));
    // Size bound covers the 32-row MTP hidden input and rejects larger blocks.
    CHECK(staging_candidate(true, "mtp_h_input", max_bytes, max_bytes, &k));
    CHECK(k == kind::mtp_h);
    CHECK(!staging_candidate(true, "mtp_h_input", max_bytes + 1, max_bytes, &k));
    CHECK(!staging_candidate(true, "inp_tokens", 0, max_bytes, &k));
    CHECK(!staging_candidate(true, "inp_tokens", 128, max_bytes, nullptr));
    return 0;
}

int test_source_mutation_isolation() {
    pool p(fake_ops());
    uint8_t src[64];
    uint8_t dst[64] = {};
    std::memset(src, 0x11, sizeof(src));

    lease a;
    CHECK(p.acquire(0, kind::mtp_h, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)));
    // The caller may reuse the source as soon as upload returns: the pinned
    // copy is independent, unlike the original path which holds the pointer
    // until the H2D copy drains.
    std::memset(src, 0x22, sizeof(src));
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr));
    CHECK(p.wait(a));
    for (size_t i = 0; i < sizeof(src); ++i) {
        CHECK(dst[i] == 0x11);
    }
    p.release(a);
    return 0;
}

int test_fallback_reupload() {
    pool p(fake_ops());
    uint8_t src[64];
    uint8_t dst0[64] = {};
    uint8_t dst1[64] = {};
    std::memset(src, 0x33, sizeof(src));

    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)));

    // Rank 0 is queued, rank 1 fails: the pool disables and the caller falls
    // back to the pageable path.
    CHECK(p.enqueue(a, 0, dst0, sizeof(src), nullptr));
    g_fail_copy_remaining = 1;
    CHECK(!p.enqueue(a, 1, dst1, sizeof(src), nullptr));
    CHECK(!p.usable());
    CHECK(g_fail_copy_remaining == 0);

    // The fallback repeats the copies from the same source. A fresh upload and
    // a full enqueue must still deliver the source bytes.
    CHECK(p.upload(a, src, sizeof(src)));
    CHECK(p.enqueue(a, 0, dst0, sizeof(src), nullptr));
    CHECK(p.enqueue(a, 1, dst1, sizeof(src), nullptr));
    CHECK(p.wait(a));
    CHECK(std::memcmp(dst0, src, sizeof(src)) == 0);
    CHECK(std::memcmp(dst1, src, sizeof(src)) == 0);
    p.release(a);
    return 0;
}

int test_layout_predicate() {
    // The async staging lane relies on ggml_are_same_layout + contiguous to
    // make nbytes(src) == nbytes(dst). Pin the predicate down here.
    ggml_tensor a = {};
    ggml_tensor b = {};
    a.type = b.type = GGML_TYPE_F32;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        a.ne[i] = b.ne[i] = (i == 0 ? 32 : 1);
        a.nb[i] = b.nb[i] = (i == 0 ? int64_t(sizeof(float)) : int64_t(32 * sizeof(float)));
    }
    CHECK(ggml_are_same_layout(&a, &b));
    const size_t bytes_a = size_t(a.ne[0]) * a.ne[1] * a.ne[2] * a.ne[3] * sizeof(float);
    const size_t bytes_b = size_t(b.ne[0]) * b.ne[1] * b.ne[2] * b.ne[3] * sizeof(float);
    CHECK(bytes_a == bytes_b);

    b.nb[1] = 256;
    CHECK(!ggml_are_same_layout(&a, &b));
    b.nb[1] = a.nb[1];
    b.type = GGML_TYPE_F16;
    CHECK(!ggml_are_same_layout(&a, &b));
    b.type = GGML_TYPE_F32;
    b.ne[1] = 2;
    CHECK(!ggml_are_same_layout(&a, &b));
    return 0;
}

int test_wait_device_switch() {
    backend_ops ops = fake_ops();
    ops.current_device = fake_current_device;
    ops.select_device  = fake_select_device;
    ops.restore_device = fake_restore_device;
    g_current_device = 5;
    g_select_count = 0;
    g_restore_count = 0;
    g_last_selected = -1;
    g_last_restored = -1;

    pool p(ops);
    uint8_t src[32];
    uint8_t dst[32];
    std::memset(src, 0x55, sizeof(src));

    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)));
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr));
    CHECK(p.enqueue(a, 3, dst, sizeof(src), nullptr));

    // Events live on their recording devices. The wait selects each pending
    // rank, then restores the caller's physical device.
    CHECK(p.wait(a));
    CHECK(g_select_count == 2);
    CHECK(g_restore_count == 1);
    CHECK(g_last_restored == 5);
    p.release(a);
    return 0;
}

}  // namespace

int main() {
    const struct {
        const char * name;
        int (*fn)();
    } tests[] = {
        {"lease_release",              test_lease_release},
        {"double_buffer_stale_release", test_double_buffer_and_stale_release},
        {"pending_slot_reuse",         test_pending_slot_reuse},
        {"acquire_nowait",             test_acquire_nowait},
        {"note_source_fast_path",      test_note_source_fast_path},
        {"window_isolation",           test_window_isolation},
        {"pool_cap",                   test_pool_cap},
        {"alloc_failure_fail_closed",  test_alloc_failure_fail_closed},
        {"upload_bounds",              test_upload_bounds},
        {"upload_rows",                test_upload_rows},
        {"enqueue_wait",               test_enqueue_wait},
        {"merged_wait",                test_merged_wait},
        {"wait_failure_disables",      test_wait_failure_disables},
        {"fault_event_create_quarantine", test_fault_event_create_quarantine},
        {"fault_event_record_quarantine", test_fault_event_record_quarantine},
        {"owns_and_diagnostics",       test_owns_and_diagnostics},
        {"device_ptr",                 test_device_ptr},
        {"staging_candidate",          test_staging_candidate},
        {"source_mutation_isolation",  test_source_mutation_isolation},
        {"fallback_reupload",          test_fallback_reupload},
        {"layout_predicate",           test_layout_predicate},
        {"wait_device_switch",         test_wait_device_switch},
    };

    int failed = 0;
    for (const auto & t : tests) {
        g_fail_alloc = false;
        g_fail_sync = false;
        g_fail_copy_remaining = 0;
        g_fail_event_create = false;
        g_fail_event_record = false;
        g_peek_error = 0;
        g_current_device = 0;
        g_select_count = 0;
        g_restore_count = 0;
        const int rc = t.fn();
        std::printf("%s %s\n", rc == 0 ? "PASS" : "FAIL", t.name);
        failed += rc != 0;
    }
    std::printf("%d/%zu passed\n", int(sizeof(tests) / sizeof(tests[0])) - failed,
                sizeof(tests) / sizeof(tests[0]));
    return failed == 0 ? 0 : 1;
}
