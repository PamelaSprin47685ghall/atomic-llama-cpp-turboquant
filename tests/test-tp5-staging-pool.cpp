// TP5 WS4 CPU test for the persistent pinned staging pool.
//
// Build and run without a GPU or HIP:
//   g++ -std=c++17 -O2 -I ggml/include tests/test-tp5-staging-pool.cpp -o /tmp/test-tp5-staging
//   /tmp/test-tp5-staging

#include "../ggml/src/ggml-cuda/staging.h"
#include "../ggml/src/ggml-impl.h"  // ggml_are_same_layout (static inline)

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using namespace ggml_cuda_staging;

namespace {

// Frozen v1 host-only api prefix. The meta layer rejects a v1 table because it
// cannot publish a partial submission, but the prefix is still the ABI
// baseline: v2 must append its new field after zc_enabled and move nothing.
// Keep this mirror in sync with the header's prefix; the assertions below fail
// the CPU test build if a field is inserted.
struct api_v1 {
    uint32_t abi_version;
    bool (*enabled)();
    bool (*merge_wait)();
    bool (*acquire)(uint32_t window, kind k, size_t bytes, lease * out);
    bool (*acquire_nowait)(uint32_t window, kind k, size_t bytes, lease * out);
    int (*upload)(const lease * l, const void * src, size_t bytes);
    int (*upload_rows)(const lease * l, const void * src, size_t src_stride,
                       size_t dst_stride, size_t row_bytes, size_t rows);
    bool (*wait)(const lease * l);
    void (*release)(const lease * l);
    bool (*owns)(const void * p, size_t bytes);
    void * (*device_ptr)(const lease * l, int rank);
    void (*dump)();
    bool (*can_enqueue_async)(ggml_backend_t backend, ggml_tensor * tensor,
                              size_t offset, size_t size);
    int (*enqueue_async)(const lease * l, ggml_backend_t backend,
                         ggml_tensor * tensor, size_t offset, size_t size);
    bool (*can_enqueue_sync)(ggml_tensor * tensor, size_t offset, size_t size);
    int (*enqueue_sync)(const lease * l, ggml_tensor * tensor,
                        size_t offset, size_t size);
    int (*first_error)();
    bool (*enter_submit)();
    void (*leave_submit)();
    size_t (*unknown_count)();
    bool (*submission_pending)();
    bool (*configured)();
    bool (*usable)();
    bool (*zc_enabled)();
};

static_assert(offsetof(api, zc_enabled) == offsetof(api_v1, zc_enabled),
              "the v1 api prefix moved");
static_assert(offsetof(api, mark_partial) >= sizeof(api_v1),
              "the v2 mark_partial field overlaps the v1 prefix");
static_assert(offsetof(api, mark_partial) > offsetof(api, zc_enabled),
              "new api fields must be appended after zc_enabled");

int g_allocs = 0;
int g_frees = 0;
int g_destroys = 0;
int g_events = 0;
int g_records = 0;
int g_syncs = 0;
int g_queries = 0;
int g_copies = 0;
bool g_fail_alloc = false;
bool g_fail_sync = false;
bool g_fail_query = false;
int  g_fail_copy_remaining = 0;
bool g_fail_event_create = false;
bool g_fail_event_record = false;

struct fake_event {
    int  id       = 0;
    int  records  = 0;
    bool complete = false;
};

std::vector<fake_event *> g_record_log;

fake_event * fake_record_at(int index) {
    return index >= 0 && index < int(g_record_log.size())
        ? g_record_log[size_t(index)] : nullptr;
}

void fake_complete_record(int index) {
    fake_event * event = fake_record_at(index);
    if (event != nullptr) {
        event->complete = true;
    }
}

void fake_complete_all_records() {
    for (fake_event * event : g_record_log) {
        event->complete = true;
    }
}
int fake_alloc(size_t bytes, void ** out) {
    if (g_fail_alloc) {
        return 1;
    }
    void * p = std::malloc(bytes);
    if (p == nullptr) {
        return 1;
    }
    std::memset(p, 0, bytes);
    *out = p;
    ++g_allocs;
    return 0;
}

void fake_free(void * p) {
    if (p != nullptr) {
        std::free(p);
        ++g_frees;
    }
}

int fake_event_create(void ** out) {
    if (g_fail_event_create) {
        return 2;
    }
    fake_event * event = new fake_event;
    event->id = g_events++;
    *out = event;
    return 0;
}

int fake_event_record(void * opaque, void *) {
    if (g_fail_event_record) {
        return 3;
    }
    fake_event * event = static_cast<fake_event *>(opaque);
    event->complete = false;
    event->records++;
    g_record_log.push_back(event);
    ++g_records;
    return 0;
}

int fake_event_sync(void * opaque) {
    ++g_syncs;
    if (g_fail_sync) {
        return 4;
    }
    static_cast<fake_event *>(opaque)->complete = true;
    return 0;
}

int fake_event_query(void * opaque, bool * complete) {
    ++g_queries;
    if (g_fail_query) {
        return 6;
    }
    *complete = static_cast<fake_event *>(opaque)->complete;
    return 0;
}

void fake_event_destroy(void * opaque) {
    delete static_cast<fake_event *>(opaque);
    ++g_destroys;
}

int fake_copy(void * dst, const void * src, size_t bytes, void *) {
    if (g_fail_copy_remaining > 0) {
        --g_fail_copy_remaining;
        return 5;
    }
    std::memcpy(dst, src, bytes);
    ++g_copies;
    return 0;
}

int fake_device_ptr(void *, void ** out) {
    *out = reinterpret_cast<void *>(uintptr_t(0x1000));
    return 0;
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

backend_ops fake_ops() {
    backend_ops ops;
    ops.host_alloc    = fake_alloc;
    ops.host_free     = fake_free;
    ops.event_create  = fake_event_create;
    ops.event_record  = fake_event_record;
    ops.event_sync    = fake_event_sync;
    ops.event_query   = fake_event_query;
    ops.event_destroy = fake_event_destroy;
    ops.copy_h2d      = fake_copy;
    ops.device_ptr    = fake_device_ptr;
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

    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
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
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);  // both leases stay valid
    CHECK(p.upload(b, src, sizeof(src)) == submit_result::ok);

    // Releasing one slot makes it reusable.
    p.release(a);
    CHECK(p.acquire(0, kind::mtp_h, sizeof(src), &c));
    CHECK(c.slot == a.slot);
    CHECK(c.epoch > b.epoch);

    // The old lease no longer names a live slot.
    CHECK(p.upload(a, src, 16) == submit_result::rejected);
    p.release(a);  // stale
    CHECK(p.snapshot().stale_release == 1);

    CHECK(p.upload(c, src, sizeof(src)) == submit_result::ok);
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
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::ok);
    p.release(a);

    lease b;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &b));
    CHECK(b.slot != a.slot);
    CHECK(p.upload(b, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(b, 1, dst, sizeof(src), nullptr) == submit_result::ok);
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
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::ok);
    p.release(a);

    lease b;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &b));
    CHECK(b.slot != a.slot);
    CHECK(p.upload(b, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(b, 1, dst, sizeof(src), nullptr) == submit_result::ok);
    p.release(b);

    // Both slots are pending. The async lane must not wait: it fails and
    // falls back to the pageable path.
    const int syncs_before = g_syncs;
    lease c;
    CHECK(!p.acquire(0, kind::tokens, sizeof(src), &c, /*wait_pending =*/ false));
    CHECK(g_syncs == syncs_before);
    const auto s = p.snapshot();
    CHECK(s.no_slot == 1);
    CHECK(s.reaps == 0);
    CHECK(s.queries == 2);
    CHECK(g_queries == 2);

    // The synchronous lane still waits one event and reuses a slot.
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &c));
    CHECK(g_syncs == syncs_before + 1);
    p.release(c);
    return 0;
}

int test_acquire_nowait_reaps_completed() {
    pool p(fake_ops());
    uint8_t src[64] = {};
    uint8_t dst[64] = {};

    // Two async copies fill both slots. Neither event has completed.
    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::ok);
    p.release(a);

    lease b;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &b));
    CHECK(b.slot != a.slot);
    CHECK(p.upload(b, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(b, 0, dst, sizeof(src), nullptr) == submit_result::ok);
    p.release(b);

    const int syncs_before = g_syncs;
    lease c;
    CHECK(!p.acquire(0, kind::tokens, sizeof(src), &c, false));
    auto s = p.snapshot();
    CHECK(s.no_slot == 1);
    CHECK(s.reaps == 0);
    CHECK(s.queries == 2);  // one query per pending slot, both not ready
    CHECK(s.waits == 0);
    CHECK(g_queries == 2);
    CHECK(g_syncs == syncs_before);

    // A completed first slot is reclaimed; the second slot is not examined
    // because the first one is enough.
    fake_complete_record(0);
    fake_complete_record(1);
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &c, false));
    CHECK(c.slot == a.slot);
    s = p.snapshot();
    CHECK(s.no_slot == 1);
    CHECK(s.reaps == 1);
    CHECK(s.queries == 3);
    CHECK(s.waits == 0);

    // The reused slot is in flight again. The next acquire skips it and
    // reclaims the still-completed second slot.
    CHECK(p.upload(c, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(c, 0, dst, sizeof(src), nullptr) == submit_result::ok);
    p.release(c);

    lease d;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &d, false));
    CHECK(d.slot == b.slot);
    s = p.snapshot();
    CHECK(s.reaps == 2);
    CHECK(s.queries == 5);
    CHECK(s.waits == 0);
    CHECK(p.upload(d, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(d, 0, dst, sizeof(src), nullptr) == submit_result::ok);
    fake_complete_record(g_records - 1);
    p.release(d);

    // Async-only traffic keeps rotating while each new copy completes. The
    // first slot stays in flight for the whole loop and is never reclaimed.
    const int records_before_loop = g_records;
    for (int i = 0; i < 64; ++i) {
        lease e;
        CHECK(p.acquire(0, kind::tokens, sizeof(src), &e, false));
        CHECK(p.upload(e, src, sizeof(src)) == submit_result::ok);
        CHECK(p.enqueue(e, 0, dst, sizeof(src), nullptr) == submit_result::ok);
        fake_complete_record(g_records - 1);
        p.release(e);
    }
    s = p.snapshot();
    CHECK(s.no_slot == 1);
    CHECK(s.waits == 0);
    CHECK(s.reaps == 2 + 64);
    CHECK(s.queries == 5 + 2 * 64);
    CHECK(g_queries == int(s.queries));
    CHECK(g_records == records_before_loop + 64);
    CHECK(g_syncs == syncs_before);
    return 0;
}

int test_reap_partial_rank() {
    pool p(fake_ops());
    uint8_t src[32] = {};
    uint8_t dst[32] = {};

    // Slot A has two recorded rank events; slot B has one.
    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::ok);  // 0
    CHECK(p.enqueue(a, 1, dst, sizeof(src), nullptr) == submit_result::ok);  // 1
    p.release(a);

    lease b;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &b));
    CHECK(p.upload(b, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(b, 0, dst, sizeof(src), nullptr) == submit_result::ok);  // 2
    p.release(b);

    // A/rank0 is complete; A/rank1 and B/rank0 are still in flight.
    fake_complete_record(0);
    const int syncs_before = g_syncs;
    lease c;
    CHECK(!p.acquire(0, kind::tokens, sizeof(src), &c, false));
    auto s = p.snapshot();
    CHECK(s.no_slot == 1);
    CHECK(s.reaps == 0);
    CHECK(s.queries == 3);  // A/r0 ready, A/r1 not ready, B/r0 not ready
    CHECK(s.waits == 0);
    CHECK(g_syncs == syncs_before);
    CHECK(p.usable());

    // Completing the last rank of A reclaims exactly A, not B.
    fake_complete_record(1);
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &c, false));
    CHECK(c.slot == a.slot);
    s = p.snapshot();
    CHECK(s.no_slot == 1);
    CHECK(s.reaps == 1);
    CHECK(s.queries == 5);  // A/r0, A/r1; stops before B
    CHECK(s.waits == 0);
    p.release(c);
    return 0;
}

int test_reap_skips_active_lease() {
    pool p(fake_ops());
    uint8_t src[32] = {};
    uint8_t dst[32] = {};

    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::ok);  // 0
    // a stays leased on purpose.

    lease b;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &b));
    CHECK(p.upload(b, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(b, 0, dst, sizeof(src), nullptr) == submit_result::ok);  // 1
    p.release(b);

    // A's event completed while its lease is still active. A must not be
    // queried or reclaimed.
    fake_complete_record(0);
    lease c;
    CHECK(!p.acquire(0, kind::tokens, sizeof(src), &c, false));
    auto s = p.snapshot();
    CHECK(s.no_slot == 1);
    CHECK(s.reaps == 0);
    CHECK(s.queries == 1);  // only slot B is pending and unleased
    CHECK(s.waits == 0);
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);

    // Once released, the completed slot joins the rotation.
    p.release(a);
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &c, false));
    CHECK(c.slot == a.slot);
    s = p.snapshot();
    CHECK(s.reaps == 1);
    CHECK(s.queries == 2);
    CHECK(s.no_slot == 1);
    CHECK(s.waits == 0);
    p.release(c);
    return 0;
}

int test_reap_event_rerecord() {
    pool p(fake_ops());
    uint8_t src[32] = {};
    uint8_t dst[32] = {};

    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::ok);  // 0
    p.release(a);

    lease b;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &b));
    CHECK(p.upload(b, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(b, 0, dst, sizeof(src), nullptr) == submit_result::ok);  // 1
    p.release(b);

    fake_complete_record(0);
    fake_complete_record(1);
    const int creates_before = g_events;
    const int destroys_before = g_destroys;

    lease c;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &c, false));
    CHECK(c.slot == a.slot);
    CHECK(p.upload(c, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(c, 0, dst, sizeof(src), nullptr) == submit_result::ok);  // 2

    // The reused slot keeps its event object; record resets completion.
    CHECK(fake_record_at(2) == fake_record_at(0));
    CHECK(fake_record_at(2) != nullptr);
    CHECK(fake_record_at(2)->records == 2);
    CHECK(fake_record_at(2)->complete == false);
    CHECK(g_events == creates_before);
    CHECK(g_destroys == destroys_before);
    CHECK(g_records == 3);

    // The re-recorded event is in flight, so the next acquire takes slot B.
    lease d;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &d, false));
    CHECK(d.slot == b.slot);

    // Once the re-recorded event completes, slot A is reclaimable again.
    // Keep d leased so the free-slot fast path cannot choose B.
    fake_complete_record(2);
    p.release(c);
    lease e;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &e, false));
    CHECK(e.slot == a.slot);
    auto s = p.snapshot();
    CHECK(s.reaps == 3);
    CHECK(s.queries == 3);
    CHECK(s.waits == 0);
    p.release(d);
    p.release(e);
    return 0;
}

int test_reap_preserves_other_slot() {
    pool p(fake_ops());
    uint8_t src[32] = {};
    uint8_t dst[32] = {};

    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::ok);  // 0
    CHECK(p.enqueue(a, 2, dst, sizeof(src), nullptr) == submit_result::ok);  // 1
    p.release(a);

    lease b;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &b));
    CHECK(p.upload(b, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(b, 0, dst, sizeof(src), nullptr) == submit_result::ok);  // 2
    p.release(b);

    // A is partially complete, so the first acquire reclaims B.
    fake_complete_record(0);
    fake_complete_record(2);
    lease c;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &c, false));
    CHECK(c.slot == b.slot);
    auto s = p.snapshot();
    CHECK(s.no_slot == 0);
    CHECK(s.reaps == 1);
    CHECK(s.queries == 3);  // A/r0 ready, A/r2 not ready, B/r0 ready
    CHECK(s.waits == 0);

    // A's rank2 bit survived B's reap; A becomes reclaimable when it clears.
    // Keep c leased so the free-slot fast path cannot choose B.
    fake_complete_record(1);
    lease d;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &d, false));
    CHECK(d.slot == a.slot);
    s = p.snapshot();
    CHECK(s.no_slot == 0);
    CHECK(s.reaps == 2);
    CHECK(s.queries == 5);  // A/r0, A/r2
    p.release(c);
    p.release(d);
    return 0;
}

int test_reap_query_failure_disables() {
    pool p(fake_ops());
    uint8_t src[8] = {};
    uint8_t dst[8] = {};

    lease a;
    lease b;
    CHECK(p.acquire(0, kind::pos, sizeof(src), &a));
    CHECK(p.acquire(0, kind::pos, sizeof(src), &b));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
    CHECK(p.upload(b, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::ok);
    CHECK(p.enqueue(b, 1, dst, sizeof(src), nullptr) == submit_result::ok);
    p.release(a);
    p.release(b);

    g_fail_query = true;
    lease c;
    CHECK(!p.acquire(0, kind::pos, sizeof(src), &c, false));
    CHECK(!p.usable());
    const auto s = p.snapshot();
    CHECK(s.query_fail == 1);
    CHECK(s.quarantine == 1);
    CHECK(s.queries == 1);
    CHECK(s.first_error == 6);
    CHECK(s.no_slot == 0);
    return 0;
}

int test_no_slot_attribution() {
    pool p(fake_ops());
    uint8_t src[32] = {};
    uint8_t dst[32] = {};

    // Both token slots are leased: a real lease collision, not a pending
    // event. No query is issued because there is no unleased slot.
    lease a;
    lease b;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &b));
    lease c;
    CHECK(!p.acquire(0, kind::tokens, sizeof(src), &c, false));
    auto s = p.snapshot();
    CHECK(s.no_slot == 1);
    CHECK(s.no_slot_leased == 1);
    CHECK(s.no_slot_pending == 0);
    CHECK(s.no_slot_quarantined == 0);
    CHECK(s.no_slot_by_kind[size_t(kind::tokens)] == 1);
    CHECK(s.no_slot_by_kind[size_t(kind::generic)] == 0);
    CHECK(s.leases_by_kind[size_t(kind::tokens)] == 2);
    CHECK(s.queries == 0);
    p.release(a);
    p.release(b);

    // Both token slots are now pending with incomplete events: retire lag.
    lease d;
    lease e;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &d));
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &e));
    CHECK(p.upload(d, src, sizeof(src)) == submit_result::ok);
    CHECK(p.upload(e, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(d, 0, dst, sizeof(src), nullptr) == submit_result::ok);  // 0
    CHECK(p.enqueue(e, 0, dst, sizeof(src), nullptr) == submit_result::ok);  // 1
    p.release(d);
    p.release(e);
    CHECK(!p.acquire(0, kind::tokens, sizeof(src), &c, false));
    s = p.snapshot();
    CHECK(s.no_slot == 2);
    CHECK(s.no_slot_leased == 1);
    CHECK(s.no_slot_pending == 1);
    CHECK(s.no_slot_by_kind[size_t(kind::tokens)] == 2);
    CHECK(s.queries == 2);  // one not-ready rank in each pending slot
    CHECK(s.reaps == 0);

    // Completing one slot turns the next acquire into a reap.
    fake_complete_record(0);
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &c, false));
    s = p.snapshot();
    CHECK(s.no_slot == 2);
    CHECK(s.reaps == 1);
    CHECK(s.reaps_by_kind[size_t(kind::tokens)] == 1);
    CHECK(s.queries == 3);
    CHECK(s.queries_by_kind[size_t(kind::tokens)] == 3);
    CHECK(s.leases_by_kind[size_t(kind::tokens)] == 5);
    p.release(c);

    // A second kind is attributed separately.
    lease g1;
    lease g2;
    CHECK(p.acquire(0, kind::generic, sizeof(src), &g1));
    CHECK(p.acquire(0, kind::generic, sizeof(src), &g2));
    CHECK(p.upload(g1, src, sizeof(src)) == submit_result::ok);
    CHECK(p.upload(g2, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(g1, 0, dst, sizeof(src), nullptr) == submit_result::ok);  // 2
    CHECK(p.enqueue(g2, 0, dst, sizeof(src), nullptr) == submit_result::ok);  // 3
    p.release(g1);
    p.release(g2);
    lease g3;
    CHECK(!p.acquire(0, kind::generic, sizeof(src), &g3, false));
    s = p.snapshot();
    CHECK(s.no_slot == 3);
    CHECK(s.no_slot_pending == 2);
    CHECK(s.no_slot_by_kind[size_t(kind::generic)] == 1);
    CHECK(s.no_slot_by_kind[size_t(kind::tokens)] == 2);

    // The global counters equal the per-kind sums, and the no_slot split
    // covers every miss.
    uint64_t leases_sum = 0;
    uint64_t no_slot_sum = 0;
    uint64_t reaps_sum = 0;
    uint64_t queries_sum = 0;
    for (size_t k = 0; k < kKindCount; ++k) {
        leases_sum += s.leases_by_kind[k];
        no_slot_sum += s.no_slot_by_kind[k];
        reaps_sum += s.reaps_by_kind[k];
        queries_sum += s.queries_by_kind[k];
    }
    CHECK(leases_sum == s.leases);
    CHECK(no_slot_sum == s.no_slot);
    CHECK(reaps_sum == s.reaps);
    CHECK(queries_sum == s.queries);
    CHECK(s.no_slot_other == 0);
    CHECK(s.no_slot_leased + s.no_slot_pending + s.no_slot_quarantined +
          s.no_slot_other == s.no_slot);
    return 0;
}

int test_reap_device_switch() {
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
    uint8_t src[16] = {};
    uint8_t dst[16] = {};

    // Both slots carry events on ranks 0 and 3.
    lease a;
    lease b;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &b));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
    CHECK(p.upload(b, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::ok);
    CHECK(p.enqueue(a, 3, dst, sizeof(src), nullptr) == submit_result::ok);
    CHECK(p.enqueue(b, 0, dst, sizeof(src), nullptr) == submit_result::ok);
    CHECK(p.enqueue(b, 3, dst, sizeof(src), nullptr) == submit_result::ok);
    p.release(a);
    p.release(b);

    // Reaping two pending ranks selects each recording device, then restores
    // the caller's physical device exactly as a blocking wait does.
    fake_complete_all_records();
    lease c;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &c, false));
    CHECK(g_select_count == 2);
    CHECK(g_last_selected == 3);
    CHECK(g_restore_count == 1);
    CHECK(g_last_restored == 5);
    CHECK(p.snapshot().queries == 2);
    CHECK(g_queries == 2);
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
    CHECK(p.upload(a, src, 65) == submit_result::rejected);
    CHECK(p.upload(a, src, 64) == submit_result::ok);
    CHECK(p.upload(a, nullptr, 64) == submit_result::rejected);
    CHECK(p.upload(a, src, 0) == submit_result::rejected);
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
    CHECK(p.upload_rows(a, src, 16, 16, 8, 3) == submit_result::ok);
    for (size_t r = 0; r < 3; ++r) {
        CHECK(std::memcmp(a.host + r * 16, src + r * 16, 8) == 0);
    }

    // Packed destination removes the holes.
    CHECK(p.upload_rows(a, src, 16, 8, 8, 3) == submit_result::ok);
    for (size_t r = 0; r < 3; ++r) {
        CHECK(std::memcmp(a.host + r * 8, src + r * 16, 8) == 0);
    }

    CHECK(p.upload_rows(a, src, 16, 16, 8, 4) == submit_result::rejected);  // span exceeds the slot
    CHECK(p.upload_rows(a, src, 4, 8, 8, 3) == submit_result::rejected);    // src_stride < row_bytes
    CHECK(p.upload_rows(a, src, 8, 4, 8, 3) == submit_result::rejected);    // dst_stride < row_bytes
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
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);

    const int syncs_before = g_syncs;
    const int copies_before = g_copies;
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::ok);
    CHECK(p.enqueue(a, 4, dst, sizeof(src), nullptr) == submit_result::ok);
    CHECK(g_copies == copies_before + 2);
    CHECK(p.wait(a));
    CHECK(g_syncs == syncs_before + 2);
    CHECK(std::memcmp(dst, src, sizeof(src)) == 0);

    CHECK(p.enqueue(a, 99, dst, sizeof(src), nullptr) == submit_result::rejected);
    CHECK(p.enqueue(a, -1, dst, sizeof(src), nullptr) == submit_result::rejected);
    CHECK(p.enqueue(a, 0, dst, sizeof(src) + 1, nullptr) == submit_result::rejected);
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
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);

    const int copies_before = g_copies;
    const int syncs_before = g_syncs;
    const uint64_t enqueues_before = p.snapshot().enqueues;
    for (int rank = 0; rank < 5; ++rank) {
        CHECK(p.enqueue(a, rank, dst[rank], sizeof(src), nullptr) == submit_result::ok);
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
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::ok);

    g_fail_sync = true;
    CHECK(!p.wait(a));
    CHECK(!p.usable());
    CHECK(p.snapshot().quarantine == 1);
    CHECK(p.snapshot().first_error == 4);  // fake_event_sync error code
    g_fail_sync = false;
    return 0;
}

int test_fault_event_create_quarantine() {
    pool p(fake_ops());
    uint8_t src[64] = {};
    uint8_t dst[64] = {};
    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);

    // The copy succeeds, the event cannot be created: completion is
    // unprovable, so the slot is quarantined and the pool stops.
    g_fail_event_create = true;
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::fatal);
    g_fail_event_create = false;
    CHECK(!p.usable());
    CHECK(p.snapshot().quarantine == 1);
    CHECK(p.snapshot().first_error == 2);  // fake_event_create error code
    CHECK(p.pool_bytes() > 0);  // the pinned buffer is retained

    p.release(a);
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::fatal);  // pool stopped
    CHECK(p.snapshot().quarantine == 1);
    return 0;
}

int test_fault_event_record_quarantine() {
    pool p(fake_ops());
    uint8_t src[64] = {};
    uint8_t dst[64] = {};
    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);

    g_fail_event_record = true;
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::fatal);
    g_fail_event_record = false;
    CHECK(!p.usable());
    CHECK(p.snapshot().quarantine == 1);
    CHECK(p.snapshot().first_error == 3);  // fake_event_record error code
    p.release(a);
    CHECK(p.snapshot().quarantine == 1);
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

int test_union_kinds_interleaved_and_slow() {
    pool p(fake_ops());
    uint8_t src[32] = {};
    uint8_t dst[32] = {};

    // Interleave both union kinds. Each kind has its own two slots, so a
    // single graph's k/v copies no longer compete for the generic key.
    lease k0;
    lease k1;
    lease v0;
    lease v1;
    CHECK(p.acquire(0, kind::union_k, sizeof(src), &k0));
    CHECK(p.acquire(0, kind::union_v, sizeof(src), &v0));
    CHECK(p.acquire(0, kind::union_k, sizeof(src), &k1));
    CHECK(p.acquire(0, kind::union_v, sizeof(src), &v1));
    CHECK(k0.slot != k1.slot);
    CHECK(v0.slot != v1.slot);
    CHECK(p.upload(k0, src, sizeof(src)) == submit_result::ok);
    CHECK(p.upload(k1, src, sizeof(src)) == submit_result::ok);
    CHECK(p.upload(v0, src, sizeof(src)) == submit_result::ok);
    CHECK(p.upload(v1, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(k0, 0, dst, sizeof(src), nullptr) == submit_result::ok);  // 0
    CHECK(p.enqueue(k1, 0, dst, sizeof(src), nullptr) == submit_result::ok);  // 1
    CHECK(p.enqueue(v0, 0, dst, sizeof(src), nullptr) == submit_result::ok);  // 2
    CHECK(p.enqueue(v1, 0, dst, sizeof(src), nullptr) == submit_result::ok);  // 3
    p.release(k0);
    p.release(k1);
    p.release(v0);
    p.release(v1);

    // The generic key is independent of both union kinds.
    lease g;
    CHECK(p.acquire(0, kind::generic, sizeof(src), &g));
    p.release(g);

    // A third union_k is a pending miss on its own kind; union_v is untouched.
    lease k2;
    CHECK(!p.acquire(0, kind::union_k, sizeof(src), &k2, false));
    auto s = p.snapshot();
    CHECK(s.no_slot == 1);
    CHECK(s.no_slot_pending == 1);
    CHECK(s.no_slot_by_kind[size_t(kind::union_k)] == 1);
    CHECK(s.no_slot_by_kind[size_t(kind::union_v)] == 0);
    CHECK(s.queries_by_kind[size_t(kind::union_k)] == 2);
    CHECK(s.queries_by_kind[size_t(kind::union_v)] == 0);

    // Slow consumer: only v0's event completed. union_v rotates; union_k
    // remains blocked on its own incomplete rank event.
    fake_complete_record(2);
    lease v2;
    CHECK(p.acquire(0, kind::union_v, sizeof(src), &v2, false));
    CHECK(v2.slot == v0.slot || v2.slot == v1.slot);
    s = p.snapshot();
    CHECK(s.reaps_by_kind[size_t(kind::union_v)] == 1);
    CHECK(s.reaps_by_kind[size_t(kind::union_k)] == 0);
    CHECK(s.no_slot_by_kind[size_t(kind::union_k)] == 1);

    // Completing k0 lets union_k rotate without changing union_v's counts.
    fake_complete_record(0);
    CHECK(p.acquire(0, kind::union_k, sizeof(src), &k2, false));
    CHECK(k2.slot == k0.slot || k2.slot == k1.slot);
    s = p.snapshot();
    CHECK(s.reaps_by_kind[size_t(kind::union_k)] == 1);
    CHECK(s.reaps_by_kind[size_t(kind::union_v)] == 1);
    CHECK(s.no_slot_by_kind[size_t(kind::union_k)] == 1);
    CHECK(s.no_slot_by_kind[size_t(kind::union_v)] == 0);
    CHECK(s.queries_by_kind[size_t(kind::union_k)] == 3);
    p.release(v2);
    p.release(k2);
    return 0;
}

int test_union_kind_failure_disables() {
    pool p(fake_ops());
    uint8_t src[32] = {};
    uint8_t dst[32] = {};

    lease k;
    CHECK(p.acquire(0, kind::union_k, sizeof(src), &k));
    CHECK(p.upload(k, src, sizeof(src)) == submit_result::ok);
    g_fail_event_record = true;
    CHECK(p.enqueue(k, 0, dst, sizeof(src), nullptr) == submit_result::fatal);
    g_fail_event_record = false;
    CHECK(!p.usable());
    CHECK(p.snapshot().quarantine == 1);
    CHECK(p.snapshot().first_error == 3);
    p.release(k);

    // The fail-stop state is global: a different kind also refuses a lease.
    lease v;
    CHECK(!p.acquire(0, kind::union_v, sizeof(src), &v, false));
    CHECK(!p.usable());
    return 0;
}

int test_rank_count_reap_queries() {
    const uint32_t counts[3] = {1, 5, 16};
    for (uint32_t n : counts) {
        g_record_log.clear();
        pool p(fake_ops());
        uint8_t src[32] = {};
        uint8_t dst[32] = {};

        lease a;
        lease b;
        CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
        CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
        for (uint32_t r = 0; r < n; ++r) {
            CHECK(p.enqueue(a, int(r), dst, sizeof(src), nullptr) == submit_result::ok);
        }
        CHECK(p.acquire(0, kind::tokens, sizeof(src), &b));
        CHECK(p.upload(b, src, sizeof(src)) == submit_result::ok);
        CHECK(p.enqueue(b, 0, dst, sizeof(src), nullptr) == submit_result::ok);
        p.release(a);
        p.release(b);

        // Complete A's n rank events; leave B's single event not ready. The
        // reap of A must query exactly popcount(A.pending_ranks) = n, not a
        // fixed five. kRanks is 16, so n=16 is the mask boundary.
        for (uint32_t r = 0; r < n; ++r) {
            fake_complete_record(int(r));
        }
        const int q0 = g_queries;
        lease c;
        CHECK(p.acquire(0, kind::tokens, sizeof(src), &c, false));
        const auto s = p.snapshot();
        CHECK(s.reaps == 1);
        CHECK(s.reaps_by_kind[size_t(kind::tokens)] == 1);
        CHECK(s.queries_by_kind[size_t(kind::tokens)] == n);
        CHECK(g_queries - q0 == int(n));
        p.release(c);
    }
    return 0;
}

int test_sync_all_leased_no_slot() {
    pool p(fake_ops());
    uint8_t src[16] = {};
    uint8_t dst[16] = {};

    lease a;
    lease b;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &b));

    // Synchronous acquire with both slots leased and no pending event: this
    // is a lease collision, not a retire miss, and it issues no queries.
    lease c;
    CHECK(!p.acquire(0, kind::tokens, sizeof(src), &c));
    auto s = p.snapshot();
    CHECK(s.no_slot == 1);
    CHECK(s.no_slot_leased == 1);
    CHECK(s.no_slot_pending == 0);
    CHECK(s.queries == 0);
    p.release(a);
    p.release(b);

    // The asynchronous pending miss is a distinct state and does query.
    lease d;
    lease e;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &d));
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &e));
    CHECK(p.upload(d, src, sizeof(src)) == submit_result::ok);
    CHECK(p.upload(e, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(d, 0, dst, sizeof(src), nullptr) == submit_result::ok);
    CHECK(p.enqueue(e, 0, dst, sizeof(src), nullptr) == submit_result::ok);
    p.release(d);
    p.release(e);
    lease f;
    CHECK(!p.acquire(0, kind::tokens, sizeof(src), &f, false));
    s = p.snapshot();
    CHECK(s.no_slot == 2);
    CHECK(s.no_slot_leased == 1);
    CHECK(s.no_slot_pending == 1);
    CHECK(s.queries == 2);
    return 0;
}

int test_four_window_union_budget() {
    // Four windows x two union kinds x two slots x 64 KiB = 1 MiB. The
    // production pool is 4 MiB; this smaller cap makes the boundary exact.
    const size_t max_bytes = size_t(kWindows) * 2 * 2 * kGenericMaxBytes;
    pool p(fake_ops(), max_bytes);
    uint8_t src[kGenericMaxBytes] = {};
    uint8_t dst[kGenericMaxBytes] = {};
    const kind kinds[2] = {kind::union_k, kind::union_v};

    for (uint32_t w = 0; w < kWindows; ++w) {
        for (kind k : kinds) {
            lease l0;
            lease l1;
            CHECK(p.acquire(w, k, kGenericMaxBytes, &l0));
            CHECK(p.upload(l0, src, kGenericMaxBytes) == submit_result::ok);
            CHECK(p.enqueue(l0, 0, dst, kGenericMaxBytes, nullptr) == submit_result::ok);
            p.release(l0);
            CHECK(p.acquire(w, k, kGenericMaxBytes, &l1));
            CHECK(p.upload(l1, src, kGenericMaxBytes) == submit_result::ok);
            CHECK(p.enqueue(l1, 0, dst, kGenericMaxBytes, nullptr) == submit_result::ok);
            p.release(l1);
        }
    }

    auto s = p.snapshot();
    CHECK(s.leases == uint64_t(kWindows) * 2 * 2);
    CHECK(p.pool_bytes() == max_bytes);
    CHECK(s.oversize == 0);
    CHECK(s.no_slot == 0);

    // The byte cap is hard: a new key cannot grow even by one byte.
    lease t;
    CHECK(!p.acquire(0, kind::tokens, 1, &t));
    s = p.snapshot();
    CHECK(s.oversize == 1);
    CHECK(s.no_slot == 0);
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
    CHECK(classify_name("union_inp_k_idxs") == kind::union_k);
    CHECK(classify_name("union_inp_v_idxs") == kind::union_v);
    CHECK(classify_name("union_inp_k_idxs_extra") == kind::generic);
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
    // The union index inputs keep the 64 KiB envelope they had as generic
    // inputs; the split changes only the key, not eligibility.
    CHECK(staging_candidate(true, "union_inp_k_idxs", kGenericMaxBytes, max_bytes, &k));
    CHECK(k == kind::union_k);
    CHECK(!staging_candidate(true, "union_inp_k_idxs", kGenericMaxBytes + 1, max_bytes, &k));
    CHECK(staging_candidate(true, "union_inp_v_idxs", kGenericMaxBytes, max_bytes, &k));
    CHECK(k == kind::union_v);
    CHECK(!staging_candidate(true, "union_inp_v_idxs", kGenericMaxBytes + 1, max_bytes, &k));
    CHECK(!staging_candidate(true, "union_inp_k_idxs", max_bytes + 1, max_bytes, &k));
    CHECK(!staging_candidate(true, "union_inp_v_idxs", 0, max_bytes, &k));
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
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
    // The caller may reuse the source as soon as upload returns: the pinned
    // copy is independent, unlike the original path which holds the pointer
    // until the H2D copy drains.
    std::memset(src, 0x22, sizeof(src));
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::ok);
    CHECK(p.wait(a));
    for (size_t i = 0; i < sizeof(src); ++i) {
        CHECK(dst[i] == 0x11);
    }
    p.release(a);
    return 0;
}

int test_partial_enqueue_stops_pool() {
    pool p(fake_ops());
    uint8_t src[64];
    uint8_t dst0[64] = {};
    uint8_t dst1[64] = {};
    std::memset(src, 0x33, sizeof(src));

    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);

    // Rank 0 is queued; rank 1's copy fails. Whether anything was queued is
    // unprovable, so the slot is quarantined and the pool stops.
    CHECK(p.enqueue(a, 0, dst0, sizeof(src), nullptr) == submit_result::ok);
    g_fail_copy_remaining = 1;
    CHECK(p.enqueue(a, 1, dst1, sizeof(src), nullptr) == submit_result::fatal);
    CHECK(!p.usable());
    CHECK(p.snapshot().quarantine == 1);
    CHECK(g_fail_copy_remaining == 0);

    // No lease operation may continue after the stop. The caller falls back
    // outside the pool with the original pageable path.
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::fatal);
    CHECK(p.enqueue(a, 0, dst0, sizeof(src), nullptr) == submit_result::fatal);
    CHECK(!p.wait(a));
    p.release(a);
    return 0;
}

int test_partial_rejection_marks_pool() {
    pool p(fake_ops());
    uint8_t src[64];
    uint8_t dst[64] = {};
    std::memset(src, 0x44, sizeof(src));

    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::ok);
    const uint64_t enqueues_before = p.snapshot().enqueues;

    // The meta layer observed that rank 0 was queued and rank 1 was rejected
    // cleanly. That is a partially written destination, not a clean rejection:
    // the pool must stop and publish an unknown submission with no native
    // error code.
    CHECK(p.usable());
    CHECK(!p.submission_pending());
    p.mark_partial(a, dst);
    CHECK(!p.usable());
    CHECK(p.submission_pending());
    CHECK(p.unknown_count() == 1);
    CHECK(p.snapshot().partial == 1);
    CHECK(p.snapshot().quarantine == 1);
    CHECK(p.snapshot().first_error == 0);
    CHECK(p.snapshot().enqueues == enqueues_before);

    // Nothing may be submitted after the partial publication.
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::fatal);
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::fatal);
    CHECK(!p.wait(a));
    CHECK(!p.enter_submit());
    p.release(a);
    CHECK(p.submission_pending());
    lease c;
    CHECK(!p.acquire(0, kind::tokens, sizeof(src), &c));
    return 0;
}

int test_partial_gate_race() {
    pool p(fake_ops());
    uint8_t src[64];
    uint8_t dst[64] = {};
    std::memset(src, 0x55, sizeof(src));

    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);

    // Hold the submit gate. mark_partial takes mutex_ and then waits for
    // submit_mutex_ inside publish_fatal_locked; it must not complete until
    // the gate is released, and then it must publish exactly once.
    CHECK(p.enter_submit());
    std::atomic<bool> started{false};
    std::atomic<bool> done{false};
    std::thread t([&] {
        started.store(true, std::memory_order_release);
        p.mark_partial(a, dst);
        done.store(true, std::memory_order_release);
    });
    while (!started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK(!done.load(std::memory_order_acquire));

    p.leave_submit();
    t.join();
    CHECK(done.load(std::memory_order_acquire));
    CHECK(!p.usable());
    CHECK(p.unknown_count() == 1);
    CHECK(p.snapshot().partial == 1);
    p.release(a);
    return 0;
}

int test_partial_duplicate_marks() {
    pool p(fake_ops());
    uint8_t src[64];
    uint8_t dst[64] = {};
    std::memset(src, 0x66, sizeof(src));

    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);

    std::atomic<int> ready{0};
    auto publish = [&] {
        ready.fetch_add(1, std::memory_order_release);
        while (ready.load(std::memory_order_acquire) < 2) {
            std::this_thread::yield();
        }
        p.mark_partial(a, dst);
    };
    std::thread t1(publish);
    std::thread t2(publish);
    t1.join();
    t2.join();

    // Both callers observed the same lease. The first records it; the second
    // must not count the same destination again.
    CHECK(!p.usable());
    CHECK(p.unknown_count() == 1);
    CHECK(p.snapshot().partial == 1);
    CHECK(p.submission_pending());
    p.release(a);
    return 0;
}

int test_copy_failure_quarantine() {
    pool p(fake_ops());
    uint8_t src[64] = {};
    uint8_t dst[64] = {};
    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);

    g_fail_copy_remaining = 1;
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::fatal);
    CHECK(!p.usable());
    CHECK(p.snapshot().quarantine == 1);
    CHECK(p.snapshot().first_error == 5);  // fake_copy error code
    CHECK(p.pool_bytes() > 0);  // retained: a copy may still read it
    p.release(a);
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::fatal);
    return 0;
}

int test_fatal_stops_submits() {
    pool p(fake_ops());
    uint8_t src[32] = {};
    uint8_t dst[32] = {};
    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);

    g_fail_copy_remaining = 1;
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::fatal);

    // After fatal, nothing else may be submitted: no copy call is made and the
    // destination is never replayed.
    const int copies_after_fatal = g_copies;
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::fatal);
    CHECK(p.enqueue(a, 1, dst, sizeof(src), nullptr) == submit_result::fatal);
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::fatal);
    CHECK(g_copies == copies_after_fatal);
    p.release(a);
    return 0;
}

int test_disabled_blocks_lease_ops() {
    pool p(fake_ops());
    uint8_t src[64] = {};
    uint8_t dst[64] = {};
    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);

    g_fail_event_create = true;
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::fatal);
    g_fail_event_create = false;
    CHECK(!p.usable());

    // An existing lease cannot bypass the stop.
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::fatal);
    CHECK(p.enqueue(a, 1, dst, sizeof(src), nullptr) == submit_result::fatal);
    CHECK(!p.wait(a));
    CHECK(p.device_ptr(a, 0) == nullptr);

    const uint64_t stale_before = p.snapshot().stale_release;
    p.release(a);  // releasing the handle is still allowed
    CHECK(p.snapshot().stale_release == stale_before);
    return 0;
}

int test_two_lease_race_fatal() {
    pool p(fake_ops());
    uint8_t src[64] = {};
    uint8_t dst[64] = {};

    // Two live leases in different windows.
    lease a;
    lease b;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.acquire(1, kind::mtp_h, sizeof(src), &b));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
    CHECK(p.upload(b, src, sizeof(src)) == submit_result::ok);

    // Lease a trips the global fault while b is still live.
    g_fail_copy_remaining = 1;
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::fatal);
    CHECK(!p.usable());

    // The other window's live lease must see fatal, never a replayable
    // rejection: no further copy may be submitted for either destination.
    const int copies_after_fatal = g_copies;
    CHECK(p.enqueue(b, 0, dst, sizeof(src), nullptr) == submit_result::fatal);
    CHECK(p.upload(b, src, sizeof(src)) == submit_result::fatal);
    CHECK(p.enqueue(b, 0, dst, sizeof(src), nullptr) == submit_result::fatal);
    CHECK(g_copies == copies_after_fatal);
    p.release(a);
    p.release(b);
    return 0;
}

int test_capacity_reject_keeps_other_lease() {
    pool p(fake_ops());
    uint8_t src[32] = {};
    uint8_t dst[64] = {};

    lease a;
    lease b;
    CHECK(p.acquire(0, kind::tokens, 32, &a));
    CHECK(p.acquire(0, kind::generic, 32, &b));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
    CHECK(p.upload(b, src, sizeof(src)) == submit_result::ok);

    // Capacity rejection has no side effect and does not stop the pool.
    CHECK(p.enqueue(a, 0, dst, sizeof(dst), nullptr) == submit_result::rejected);
    CHECK(p.usable());

    // The other live lease still submits normally.
    CHECK(p.enqueue(b, 0, dst, sizeof(src), nullptr) == submit_result::ok);
    p.release(a);
    p.release(b);
    return 0;
}

int test_submit_gate_race() {
    pool p(fake_ops());
    uint8_t src[32] = {};
    uint8_t dst[32] = {};

    lease a;
    lease b;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.acquire(0, kind::generic, sizeof(src), &b));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
    CHECK(p.upload(b, src, sizeof(src)) == submit_result::ok);

    // A fallback submit holds the gate.
    CHECK(p.enter_submit());

    g_fail_copy_remaining = 1;
    std::atomic<bool> started{false};
    std::atomic<bool> published{false};
    std::thread worker([&] {
        started.store(true);
        p.enqueue(b, 0, dst, sizeof(src), nullptr);  // trips the fatal
        published.store(true);
    });

    while (!started.load()) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    // The fatal publication cannot pass the held submit gate.
    CHECK(!published.load());

    // Completing the submit lets the fatal publish; after that no new submit
    // may start.
    p.leave_submit();
    worker.join();
    CHECK(published.load());
    CHECK(!p.usable());
    CHECK(!p.enter_submit());
    CHECK(p.unknown_count() == 1);  // the failed copy is recorded as unknown
    p.release(a);
    p.release(b);
    return 0;
}

int test_quarantine_destructor_retains() {
    const int frees_before = g_frees;
    const int destroys_before = g_destroys;
    {
        pool p(fake_ops());
        uint8_t src[64] = {};
        uint8_t dst[64] = {};
        lease a;
        CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
        CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
        g_fail_event_create = true;
        CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::fatal);
        g_fail_event_create = false;
        p.release(a);
    }
    // The destructor must not free or destroy a quarantined slot.
    CHECK(g_frees == frees_before);
    CHECK(g_destroys == destroys_before);
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
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);
    CHECK(p.enqueue(a, 0, dst, sizeof(src), nullptr) == submit_result::ok);
    CHECK(p.enqueue(a, 3, dst, sizeof(src), nullptr) == submit_result::ok);

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
        {"acquire_nowait_reaps_completed", test_acquire_nowait_reaps_completed},
        {"reap_partial_rank",          test_reap_partial_rank},
        {"reap_skips_active_lease",    test_reap_skips_active_lease},
        {"reap_event_rerecord",        test_reap_event_rerecord},
        {"reap_preserves_other_slot",  test_reap_preserves_other_slot},
        {"reap_query_failure_disables", test_reap_query_failure_disables},
        {"no_slot_attribution",        test_no_slot_attribution},
        {"reap_device_switch",         test_reap_device_switch},
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
        {"copy_failure_quarantine",    test_copy_failure_quarantine},
        {"fatal_stops_submits",        test_fatal_stops_submits},
        {"disabled_blocks_lease_ops",  test_disabled_blocks_lease_ops},
        {"two_lease_race_fatal",       test_two_lease_race_fatal},
        {"capacity_reject_keeps_other_lease", test_capacity_reject_keeps_other_lease},
        {"submit_gate_race",           test_submit_gate_race},
        {"quarantine_destructor_retains", test_quarantine_destructor_retains},
        {"owns_and_diagnostics",       test_owns_and_diagnostics},
        {"device_ptr",                 test_device_ptr},
        {"union_kinds_interleaved_and_slow", test_union_kinds_interleaved_and_slow},
        {"union_kind_failure_disables", test_union_kind_failure_disables},
        {"rank_count_reap_queries",    test_rank_count_reap_queries},
        {"sync_all_leased_no_slot",    test_sync_all_leased_no_slot},
        {"four_window_union_budget",   test_four_window_union_budget},
        {"staging_candidate",          test_staging_candidate},
        {"source_mutation_isolation",  test_source_mutation_isolation},
        {"partial_enqueue_stops_pool", test_partial_enqueue_stops_pool},
        {"partial_rejection_marks_pool", test_partial_rejection_marks_pool},
        {"partial_gate_race",          test_partial_gate_race},
        {"partial_duplicate_marks",    test_partial_duplicate_marks},
        {"layout_predicate",           test_layout_predicate},
        {"wait_device_switch",         test_wait_device_switch},
    };

    int failed = 0;
    for (const auto & t : tests) {
        g_fail_alloc = false;
        g_fail_sync = false;
        g_fail_query = false;
        g_queries = 0;
        g_records = 0;
        g_record_log.clear();
        g_fail_copy_remaining = 0;
        g_fail_event_create = false;
        g_fail_event_record = false;
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
