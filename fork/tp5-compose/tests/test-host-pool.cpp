// TP5 MTP batch pool and piece scratch: lease reuse, fallback batches,
// inline token storage, and in-memory LeaseEvent accounting.

#include "host_pool.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static int g_failures = 0;

static void check(bool ok, const char * name) {
    if (!ok) {
        ++g_failures;
    }
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
}

static void set_flag(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

static void test_pooled_lease() {
    set_flag("TP5_MTP_BATCH_POOL", "1");

    constexpr int wide = 8;

    tq::tp5::mtp_batch_pool pool(4, wide);

    float embd[wide];
    for (int i = 0; i < wide; ++i) {
        embd[i] = static_cast<float>(i) + 0.5f;
    }

    for (int round = 0; round < 3; ++round) {
        auto lease = pool.acquire();

        llama_batch & batch = lease.batch();
        check(batch.token != nullptr, "pooled: token storage is set");
        check(batch.embd != nullptr, "pooled: embd storage is set");

        lease.set(100 + round, 7 + round, 1, embd, wide);

        check(batch.n_tokens == 1, "pooled: one token");
        check(batch.token[0] == 100 + round, "pooled: token");
        check(batch.pos[0] == 7 + round, "pooled: position");
        check(batch.n_seq_id[0] == 1, "pooled: n_seq_id");
        check(batch.seq_id[0][0] == 0, "pooled: seq id");
        check(batch.logits[0] == 1, "pooled: logits");
        check(std::memcmp(batch.embd, embd, sizeof(embd)) == 0, "pooled: embedding copy");

        // The next acquire inside this scope would take another slot. The
        // lease returns this one when the scope ends.
    }

    check(true, "pooled: three sequential leases completed");
}

static void test_fallback_lease() {
    set_flag("TP5_MTP_BATCH_POOL", nullptr);

    constexpr int wide = 4;

    tq::tp5::mtp_batch_pool pool(4, wide);

    float embd[wide] = { 1.0f, 2.0f, 3.0f, 4.0f };

    {
        auto lease = pool.acquire();
        lease.set(5, 6, 0, embd, wide);

        llama_batch & batch = lease.batch();
        check(batch.token != nullptr, "fallback: token storage is set");
        check(batch.token[0] == 5, "fallback: token");
        check(batch.pos[0] == 6, "fallback: position");
        check(batch.logits[0] == 0, "fallback: logits");
        check(std::memcmp(batch.embd, embd, sizeof(embd)) == 0, "fallback: embedding copy");
    }

    // A second lease must own a fresh heap batch, not a freed pointer.
    {
        auto lease = pool.acquire();
        lease.set(9, 10, 1, nullptr, wide);
        check(lease.batch().token[0] == 9, "fallback: second lease is valid");
    }
}

static void test_exhaustion_falls_back() {
    set_flag("TP5_MTP_BATCH_POOL", "1");

    constexpr int wide = 4;

    tq::tp5::mtp_batch_pool pool(1, wide);

    auto first = pool.acquire();
    first.set(1, 1, 1, nullptr, wide);

    {
        auto second = pool.acquire();
        second.set(2, 2, 1, nullptr, wide);
        check(second.batch().token[0] == 2, "exhaustion: fallback lease works");
    }

    check(first.batch().token[0] == 1, "exhaustion: first lease untouched");
}

static void test_events() {
    set_flag("TP5_MTP_BATCH_POOL", "1");
    set_flag("TP5_ALLOC_PROBE", "1");

    constexpr int wide = 4;

    {
        tq::tp5::mtp_batch_pool pool(2, wide);

        {
            auto lease = pool.acquire();
            lease.set(1, 0, 1, nullptr, wide);
        }

        const auto & events = pool.events();
        check(events.size() == 2, "events: lease and release recorded");
        if (events.size() == 2) {
            check(events[0].action == tq::alloc::LeaseAction::lease, "events: first action is lease");
            check(events[1].action == tq::alloc::LeaseAction::release, "events: second action is release");
            check(events[0].site == tq::tp5::mtp_batch_pool::event_site, "events: site id");
            check(events[0].slot == events[1].slot, "events: same slot");
            check(events[1].slot_gen == events[0].slot_gen, "events: same generation");
            check(events[0].to == tq::alloc::State::leased, "events: lease state");
            check(events[1].to == tq::alloc::State::free, "events: release state");
        }
    }

    set_flag("TP5_ALLOC_PROBE", nullptr);
    set_flag("TP5_MTP_BATCH_POOL", nullptr);
}

int main() {
    test_pooled_lease();
    test_fallback_lease();
    test_exhaustion_falls_back();
    test_events();

    std::printf("%s failures=%d\n", g_failures ? "FAILED" : "OK", g_failures);

    return g_failures ? 1 : 0;
}
