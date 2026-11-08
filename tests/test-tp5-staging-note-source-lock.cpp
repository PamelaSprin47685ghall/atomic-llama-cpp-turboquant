// TP5 staging submit-gate / note_source lock ownership (CPU only, no GPU).
//
// The meta fallback holds the submit gate (submit_mutex_) while it calls the
// backend set path. A CUDA simple tensor on that path calls
// pool::note_source(), so the same thread can enter note_source while it owns
// submit_mutex_. note_source must not try_lock or take that mutex: try_lock on
// an owned std::mutex is undefined, and a successful try_lock would drop the
// gate. It must also never take mutex_ while the gate is held, because the
// fatal publication holds mutex_ and waits for submit_mutex_.
//
// Build and run:
//   cmake --build build-ws4-sched-ctest -j --target test-tp5-staging-note-source-lock
//   ./build-ws4-sched-ctest/bin/test-tp5-staging-note-source-lock
//
// The test uses the header-only pool with injected host/event/copy primitives.

#include "../ggml/src/ggml-cuda/staging.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

using namespace ggml_cuda_staging;

namespace {

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

int fake_alloc(size_t bytes, void ** out) {
    void * p = std::malloc(bytes);
    if (p == nullptr) {
        return 1;
    }
    std::memset(p, 0, bytes);
    *out = p;
    return 0;
}

void fake_free(void * p) {
    std::free(p);
}

int fake_event_create(void ** out) {
    *out = std::malloc(1);
    return *out != nullptr ? 0 : 2;
}

int fake_event_record(void *, void *) {
    return 0;
}

int fake_event_sync(void *) {
    return 0;
}

void fake_event_destroy(void * p) {
    std::free(p);
}

bool g_fail_copy = false;

int fake_copy(void * dst, const void * src, size_t bytes, void *) {
    if (g_fail_copy) {
        return 5;
    }
    std::memcpy(dst, src, bytes);
    return 0;
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
    return ops;
}

void wait_flag(const std::atomic<bool> & flag) {
    while (!flag.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

bool wait_pool_unusable(const pool & p, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(timeout_ms);
    while (p.usable() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return !p.usable();
}

// Same-thread note_source inside the gate must neither try_lock the owned
// submit_mutex_ nor drop the gate. A concurrent enter_submit must stay
// blocked until leave_submit, and the diagnostic must count a fast miss.
int test_same_thread_note_source_keeps_gate() {
    pool p(fake_ops());
    uint8_t src[64] = {};
    std::memset(src, 0x5a, sizeof(src));

    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);

    p.note_source(a.host, sizeof(src));
    CHECK(p.snapshot().hits == 1);
    CHECK(p.snapshot().misses == 0);

    CHECK(p.enter_submit());
    p.note_source(a.host, sizeof(src)); // the fallback -> set_tensor_2d path

    std::atomic<bool> attempted{false};
    std::atomic<bool> entered{false};
    std::thread other([&] {
        attempted.store(true, std::memory_order_release);
        const bool ok = p.enter_submit();
        entered.store(ok, std::memory_order_release);
        if (ok) {
            p.leave_submit();
        }
    });
    wait_flag(attempted);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK(!entered.load());
    p.leave_submit();
    other.join();
    CHECK(entered.load());

    const auto s = p.snapshot();
    CHECK(s.hits == 1);
    CHECK(s.misses == 1); // the inside-gate note was counted as a fast miss
    p.release(a);
    return 0;
}

// A fatal publication holds mutex_ and waits for the gate. While the gate is
// held, note_source (same thread and another thread) must not take mutex_:
// that would deadlock against the publisher. The gate must remain held until
// leave_submit, then the publisher completes.
int test_note_source_does_not_deadlock_fatal_publication() {
    pool p(fake_ops());
    uint8_t src[32] = {};
    uint8_t dst[32] = {};
    std::memset(src, 0x33, sizeof(src));

    lease a;
    CHECK(p.acquire(0, kind::tokens, sizeof(src), &a));
    CHECK(p.upload(a, src, sizeof(src)) == submit_result::ok);

    g_fail_copy = true;
    CHECK(p.enter_submit());

    std::atomic<bool> started{false};
    std::atomic<bool> published{false};
    std::thread worker([&] {
        started.store(true, std::memory_order_release);
        p.enqueue(a, 0, dst, sizeof(src), nullptr); // copy fails -> fatal path
        published.store(true, std::memory_order_release);
    });
    wait_flag(started);
    CHECK(wait_pool_unusable(p, 2000));
    CHECK(!published.load()); // the fatal publication waits for the gate

    p.note_source(a.host, sizeof(src));
    std::thread observer([&] {
        p.note_source(a.host, sizeof(src));
    });
    observer.join();
    CHECK(!published.load());

    p.leave_submit();
    worker.join();
    g_fail_copy = false;

    CHECK(published.load());
    CHECK(!p.usable());
    CHECK(p.unknown_count() == 1);

    const auto s = p.snapshot();
    CHECK(s.hits == 0);
    CHECK(s.misses == 2); // both inside-gate notes are fast misses

    p.release(a);
    return 0;
}

} // namespace

int main() {
    int failed = 0;
    const struct {
        const char * name;
        int (*fn)();
    } tests[] = {
        { "same_thread_note_source_keeps_gate", test_same_thread_note_source_keeps_gate },
        { "note_source_fatal_publication_order", test_note_source_does_not_deadlock_fatal_publication },
    };

    for (const auto & t : tests) {
        g_fail_copy = false;
        const int rc = t.fn();
        std::printf("%s %s\n", rc == 0 ? "PASS" : "FAIL", t.name);
        failed += rc != 0;
    }

    if (failed != 0) {
        std::fprintf(stderr, "TP5_STAGING_NOTE_SOURCE_LOCK FAIL failures=%d\n", failed);
        return 1;
    }
    std::printf("TP5_STAGING_NOTE_SOURCE_LOCK PASS\n");
    return 0;
}
