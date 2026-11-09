// TP5 MTP batch pool and piece scratch: lease reuse, fallback batches,
// inline token storage, the full-width embedding contract, and in-memory
// LeaseEvent accounting.

#include "host_pool.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <new>
#include <stdexcept>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

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

enum class partial_batch { none, embd_only, no_embd, seq_id_only, seq_id_full };

static int          g_batch_init_failures  = 0;
static int          g_token_alloc_failures = 0;
static partial_batch g_partial_batch       = partial_batch::none;

static llama_batch partial_batch_init(int wide) {
    llama_batch batch{};
    switch (g_partial_batch) {
        case partial_batch::embd_only:
            batch.embd = static_cast<float *>(std::malloc(sizeof(float) * static_cast<std::size_t>(wide)));
            break;
        case partial_batch::no_embd:
            batch.pos      = static_cast<llama_pos *>(std::malloc(sizeof(llama_pos)));
            batch.n_seq_id = static_cast<int32_t *>(std::malloc(sizeof(int32_t)));
            batch.seq_id   = static_cast<llama_seq_id **>(std::malloc(sizeof(llama_seq_id *) * 2));
            if (batch.seq_id != nullptr) {
                batch.seq_id[0] = static_cast<llama_seq_id *>(std::malloc(sizeof(llama_seq_id)));
                batch.seq_id[1] = nullptr;
            }
            batch.logits = static_cast<int8_t *>(std::malloc(sizeof(int8_t)));
            break;
        case partial_batch::seq_id_only:
            batch.seq_id = static_cast<llama_seq_id **>(std::malloc(sizeof(llama_seq_id *) * 2));
            if (batch.seq_id != nullptr) {
                batch.seq_id[0] = nullptr;
                batch.seq_id[1] = nullptr;
            }
            break;
        case partial_batch::seq_id_full:
            batch.seq_id = static_cast<llama_seq_id **>(std::malloc(sizeof(llama_seq_id *) * 2));
            if (batch.seq_id != nullptr) {
                batch.seq_id[0] = static_cast<llama_seq_id *>(std::malloc(sizeof(llama_seq_id)));
                batch.seq_id[1] = nullptr;
            }
            break;
        case partial_batch::none:
            break;
    }
    return batch;
}

static llama_batch fail_once_batch_init(int wide) {
    if (g_batch_init_failures > 0) {
        --g_batch_init_failures;
        return partial_batch_init(wide);
    }
    return llama_batch_init(1, wide, 1);
}

static llama_token * fail_once_token_alloc() {
    if (g_token_alloc_failures > 0) {
        --g_token_alloc_failures;
        return nullptr;
    }
    return static_cast<llama_token *>(std::malloc(sizeof(llama_token)));
}

static llama_batch complete_with_token_batch_init(int wide) {
    llama_batch batch = llama_batch_init(1, wide, 1);
    batch.token = static_cast<llama_token *>(std::malloc(sizeof(llama_token)));
    return batch;
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
        float embd2[wide] = { 5.0f, 6.0f, 7.0f, 8.0f };
        auto lease = pool.acquire();
        lease.set(9, 10, 1, embd2, wide);
        check(lease.batch().token[0] == 9, "fallback: second lease is valid");
        check(std::memcmp(lease.batch().embd, embd2, sizeof(embd2)) == 0,
              "fallback: second lease copies its own embedding");
    }
}

static void test_exhaustion_falls_back() {
    set_flag("TP5_MTP_BATCH_POOL", "1");

    constexpr int wide = 4;

    tq::tp5::mtp_batch_pool pool(1, wide);

    float embd[wide] = { 1.0f, 2.0f, 3.0f, 4.0f };

    auto first = pool.acquire();
    first.set(1, 1, 1, embd, wide);

    {
        auto second = pool.acquire();
        second.set(2, 2, 1, embd, wide);
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
            float embd[wide] = { 0.5f, 0.25f, 0.125f, 0.0625f };
            auto lease = pool.acquire();
            lease.set(1, 0, 1, embd, wide);
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

static void test_cancel_releases_lease() {
    set_flag("TP5_MTP_BATCH_POOL", "1");
    set_flag("TP5_ALLOC_PROBE", "1");

    tq::tp5::mtp_batch_pool pool(1, 4);

    float embd[4] = { 1.0f, 2.0f, 3.0f, 4.0f };

    try {
        auto lease = pool.acquire();
        lease.set(1, 0, 1, embd, 4);
        throw std::runtime_error("cancelled");
    } catch (const std::runtime_error &) {
    }

    auto lease = pool.acquire();
    lease.set(2, 1, 1, embd, 4);

    const auto & events = pool.events();
    check(events.size() == 3, "cancel: acquire, release on unwind, acquire");
    if (events.size() == 3) {
        check(events[1].action == tq::alloc::LeaseAction::release, "cancel: unwind released the slot");
        check(events[2].slot == events[0].slot, "cancel: the slot is reused");
        check(events[2].slot_gen == events[0].slot_gen + 1, "cancel: generation advances");
    }

    set_flag("TP5_ALLOC_PROBE", nullptr);
    set_flag("TP5_MTP_BATCH_POOL", nullptr);
}

static void test_steady_state() {
#if defined(__GLIBC__)
    set_flag("TP5_MTP_BATCH_POOL", "1");

    constexpr int wide = 4;
    tq::tp5::mtp_batch_pool pool(4, wide);

    float embd[wide] = { 1.0f, 2.0f, 3.0f, 4.0f };

    // Warm the first slot. The pool scans from slot 0, so a sequential
    // lease stream always reuses it after the first acquire.
    for (int i = 0; i < 1; ++i) {
        auto lease = pool.acquire();
        lease.set(i, i, 1, embd, wide);
    }

    const std::size_t before = mallinfo2().uordblks;

    for (int i = 0; i < 10000; ++i) {
        auto lease = pool.acquire();
        lease.set(i, i, 1, embd, wide);
    }

    const std::size_t after = mallinfo2().uordblks;
    check(before == after, "steady: 10k MTP leases, no heap growth");

    set_flag("TP5_MTP_BATCH_POOL", nullptr);
#else
    check(true, "steady: skipped without mallinfo2");
#endif
}

static void test_event_cap() {
    set_flag("TP5_MTP_BATCH_POOL", "1");
    set_flag("TP5_ALLOC_PROBE", "1");

    constexpr int wide = 4;
    tq::tp5::mtp_batch_pool pool(1, wide);

    // Two events per lease. The log must stop at its fixed cap instead of
    // growing without bound.
    float embd[wide] = { 1.0f, 2.0f, 3.0f, 4.0f };
    for (int i = 0; i < 3000; ++i) {
        auto lease = pool.acquire();
        lease.set(i, i, 1, embd, wide);
    }

    const auto & events = pool.events();
    check(events.size() == tq::tp5::mtp_batch_pool::max_events, "events: fixed cap is reached");
    check(events.size() <= tq::tp5::mtp_batch_pool::max_events, "events: no growth past the cap");

    set_flag("TP5_ALLOC_PROBE", nullptr);
    set_flag("TP5_MTP_BATCH_POOL", nullptr);
}

// Independent reference path: a fresh 256-byte buffer with the same retry.
static std::string reference_piece(const llama_vocab * vocab, llama_token token) {
    std::string buffer(256, '\0');
    auto size = llama_token_to_piece(vocab, token, buffer.data(), 256, 0, false);
    if (size < 0) {
        if (size == INT32_MIN) {
            return {};
        }
        buffer.resize(static_cast<std::size_t>(-size));
        size = llama_token_to_piece(vocab, token, buffer.data(),
                                    static_cast<std::int32_t>(buffer.size()), 0, false);
    }
    if (size < 0 || static_cast<std::size_t>(size) > buffer.size()) {
        return {};
    }
    buffer.resize(static_cast<std::size_t>(size));
    return buffer;
}

static void test_piece_scratch_buffer() {
    const char * env = std::getenv("TP5_TEST_VOCAB");
#ifdef TP5_TEST_VOCAB_DEFAULT
    const char * fallback = TP5_TEST_VOCAB_DEFAULT;
#else
    const char * fallback = nullptr;
#endif
    const char * path = env != nullptr && env[0] != '\0' ? env : fallback;
    if (path == nullptr || path[0] == '\0') {
        std::printf("SKIP piece scratch: no vocabulary fixture (set TP5_TEST_VOCAB)\n");
        return;
    }

    FILE * probe = std::fopen(path, "rb");
    if (probe == nullptr) {
        std::printf("SKIP piece scratch: fixture not readable: %s\n", path);
        return;
    }
    std::fclose(probe);

    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(path, mp);
    check(model != nullptr, "piece scratch: vocabulary loads");
    if (model == nullptr) {
        return;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    check(n_vocab > 0, "piece scratch: vocabulary is not empty");

    tq::tp5::piece_scratch scratch;
    std::size_t mismatches = 0;
    std::size_t max_piece = 0;
    llama_token max_token = -1;

    for (int token = 0; token < n_vocab; ++token) {
        const std::string want = reference_piece(vocab, token);
        const auto view = scratch.view(vocab, token);
        if (std::string(view) != want) {
            ++mismatches;
            if (mismatches <= 4) {
                std::printf("FAIL piece scratch: token %d differs (want %zu, got %zu)\n",
                            token, want.size(), view.size());
            }
        }
        if (want.size() > max_piece) {
            max_piece = want.size();
            max_token = token;
        }
        if (scratch.buffer_size() < 256) {
            std::printf("FAIL piece scratch: buffer fell below 256 at token %d\n", token);
            ++mismatches;
            break;
        }
    }

    check(mismatches == 0, "piece scratch: every token matches the reference");
    check(max_token >= 0, "piece scratch: longest token is known");

    if (max_piece > 256) {
        const auto view = scratch.view(vocab, max_token);
        const std::string want = reference_piece(vocab, max_token);
        check(std::string(view) == want, "piece scratch: the long piece matches after the retry");
        check(scratch.buffer_size() >= max_piece, "piece scratch: the retry grew the buffer");
    } else {
        std::printf("SKIP piece scratch retry: fixture max piece %zu <= 256\n", max_piece);
    }

    check(scratch.buffer_size() >= 256, "piece scratch: buffer keeps its 256-byte floor");

    // Lifetime: a copied view survives the next view call; the next view is
    // the current piece, not the previous one.
    {
        const auto first = scratch.view(vocab, 0);
        const std::string first_copy(first);
        const auto second = scratch.view(vocab, 1);
        check(std::string(second) == reference_piece(vocab, 1),
              "piece scratch: the next view is the next token");
        check(first_copy == reference_piece(vocab, 0),
              "piece scratch: a copied view keeps its bytes");
    }

    llama_model_free(model);
}
static void test_set_contract() {
    set_flag("TP5_MTP_BATCH_POOL", "1");

    constexpr int wide = 4;
    tq::tp5::mtp_batch_pool pool(1, wide);

    float embd[wide]      = { 1.0f, 2.0f, 3.0f, 4.0f };
    float wider[wide + 1] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

    {
        auto lease = pool.acquire();

        bool threw = false;
        try {
            lease.set(1, 0, 1, nullptr, wide);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        check(threw, "set: a missing embedding row is refused");

        threw = false;
        try {
            lease.set(1, 0, 1, wider, wide + 1);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        check(threw, "set: a wider row is refused before it overruns the slot");

        threw = false;
        try {
            lease.set(1, 0, 1, embd, wide - 1);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        check(threw, "set: a narrower row is refused instead of leaving stale bytes");

        lease.set(7, 3, 1, embd, wide);
        check(lease.batch().token[0] == 7, "set: a full-width row is accepted");
        check(std::memcmp(lease.batch().embd, embd, sizeof(embd)) == 0,
              "set: the accepted row is copied");
    }

    // A refused set must not consume the slot.
    auto again = pool.acquire();
    again.set(8, 4, 1, embd, wide);
    check(again.batch().token[0] == 8, "set: the slot survives a refused set");

    set_flag("TP5_MTP_BATCH_POOL", nullptr);
}

static void test_fallback_set_contract() {
    set_flag("TP5_MTP_BATCH_POOL", nullptr);

    constexpr int wide = 4;
    tq::tp5::mtp_batch_pool pool(1, wide);

    float embd[wide]      = { 1.0f, 2.0f, 3.0f, 4.0f };
    float wider[wide + 1] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

    auto lease = pool.acquire();
    bool threw = false;
    try {
        lease.set(1, 0, 1, nullptr, wide);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    check(threw, "fallback set: a missing embedding row is refused");

    threw = false;
    try {
        lease.set(2, 1, 1, wider, wide + 1);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    check(threw, "fallback set: a wider row is refused");

    threw = false;
    try {
        lease.set(2, 1, 1, embd, wide - 1);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    check(threw, "fallback set: a narrower row is refused");

    lease.set(2, 1, 1, embd, wide);
    check(lease.batch().token[0] == 2, "fallback set: a full-width row is accepted");
}

static void test_zero_width_refused() {
    bool threw = false;
    try {
        tq::tp5::mtp_batch_pool pool(1, 0);
        (void) pool;
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    check(threw, "pool: a zero width is refused");
}

static void test_pool_lifetime() {
#if defined(__GLIBC__)
    set_flag("TP5_MTP_BATCH_POOL", "1");
    set_flag("TP5_ALLOC_PROBE", nullptr);

    constexpr int wide = 4;
    float embd[wide] = { 1.0f, 2.0f, 3.0f, 4.0f };

    // Warm the allocator bins once so the measured round is steady.
    {
        tq::tp5::mtp_batch_pool pool(4, wide);
        auto lease0 = pool.acquire();
        auto lease1 = pool.acquire();
        auto lease2 = pool.acquire();
        auto lease3 = pool.acquire();
        lease0.set(0, 0, 1, embd, wide);
        lease1.set(1, 1, 1, embd, wide);
        lease2.set(2, 2, 1, embd, wide);
        lease3.set(3, 3, 1, embd, wide);
    }

    const std::size_t before = mallinfo2().uordblks;
    {
        tq::tp5::mtp_batch_pool pool(4, wide);
        auto lease0 = pool.acquire();
        auto lease1 = pool.acquire();
        auto lease2 = pool.acquire();
        auto lease3 = pool.acquire();
        lease0.set(0, 0, 1, embd, wide);
        lease1.set(1, 1, 1, embd, wide);
        lease2.set(2, 2, 1, embd, wide);
        lease3.set(3, 3, 1, embd, wide);
    }
    const std::size_t after = mallinfo2().uordblks;
    check(before == after, "pool lifetime: destruction frees every initialized slot");

    set_flag("TP5_MTP_BATCH_POOL", nullptr);
#else
    check(true, "pool lifetime: skipped without mallinfo2");
#endif
}

static void test_partial_batch_free_shapes() {
    set_flag("TP5_MTP_BATCH_POOL", "1");

    constexpr int wide = 4;
    const partial_batch shapes[] = { partial_batch::none,      partial_batch::embd_only,
                                     partial_batch::no_embd,   partial_batch::seq_id_only,
                                     partial_batch::seq_id_full };

    for (const auto shape : shapes) {
        g_partial_batch = shape;
        tq::tp5::mtp_batch_pool pool(1, wide, { partial_batch_init, nullptr });

        bool threw = false;
        try {
            auto lease = pool.acquire();
            (void) lease;
        } catch (const std::bad_alloc &) {
            threw = true;
        }
        check(threw, "alloc: an incomplete first slot throws bad_alloc");
    }

    g_partial_batch = partial_batch::none;
    set_flag("TP5_MTP_BATCH_POOL", nullptr);
}

static void test_first_acquire_failure_retry() {
    set_flag("TP5_MTP_BATCH_POOL", "1");

    constexpr int wide = 4;
    float embd[wide] = { 1.0f, 2.0f, 3.0f, 4.0f };

    g_batch_init_failures = 1;
    g_partial_batch       = partial_batch::embd_only;
    tq::tp5::mtp_batch_pool pool(1, wide, { fail_once_batch_init, nullptr });

    bool threw = false;
    try {
        auto lease = pool.acquire();
        (void) lease;
    } catch (const std::bad_alloc &) {
        threw = true;
    }
    check(threw, "alloc: the first acquire failure is reported");

    auto lease = pool.acquire();
    lease.set(3, 0, 1, embd, wide);
    check(lease.batch().token[0] == 3, "alloc: the slot is retried after a failed first acquire");

    g_partial_batch = partial_batch::none;
    set_flag("TP5_MTP_BATCH_POOL", nullptr);
}

static void test_fallback_alloc_failure_retry() {
    set_flag("TP5_MTP_BATCH_POOL", nullptr);

    constexpr int wide = 4;
    float embd[wide] = { 1.0f, 2.0f, 3.0f, 4.0f };

    g_batch_init_failures = 1;
    g_partial_batch       = partial_batch::embd_only;
    tq::tp5::mtp_batch_pool pool(1, wide, { fail_once_batch_init, nullptr });

    bool threw = false;
    try {
        auto lease = pool.acquire();
        (void) lease;
    } catch (const std::bad_alloc &) {
        threw = true;
    }
    check(threw, "alloc: a partial fallback batch is reported");

    auto lease = pool.acquire();
    lease.set(4, 1, 1, embd, wide);
    check(lease.batch().token[0] == 4, "alloc: the fallback path is retried after a failed acquire");

    g_token_alloc_failures = 1;
    tq::tp5::mtp_batch_pool token_pool(1, wide, { nullptr, fail_once_token_alloc });
    threw = false;
    try {
        auto failed = token_pool.acquire();
        (void) failed;
    } catch (const std::bad_alloc &) {
        threw = true;
    }
    check(threw, "alloc: a null fallback token is reported");

    auto retried = token_pool.acquire();
    retried.set(5, 2, 1, embd, wide);
    check(retried.batch().token[0] == 5, "alloc: the fallback token path is retried");

    g_partial_batch = partial_batch::none;
    set_flag("TP5_MTP_BATCH_POOL", nullptr);
}

static void test_allocator_token_is_replaced() {
    constexpr int wide = 4;
    float embd[wide] = { 1.0f, 2.0f, 3.0f, 4.0f };

    set_flag("TP5_MTP_BATCH_POOL", "1");
    {
        tq::tp5::mtp_batch_pool pool(1, wide, { complete_with_token_batch_init, nullptr });
        auto lease = pool.acquire();
        lease.set(6, 0, 1, embd, wide);
        check(lease.batch().token[0] == 6,
              "alloc: the pooled slot replaces an allocator-provided token");
    }
    set_flag("TP5_MTP_BATCH_POOL", nullptr);

    {
        tq::tp5::mtp_batch_pool pool(1, wide, { complete_with_token_batch_init, nullptr });
        auto lease = pool.acquire();
        lease.set(7, 1, 1, embd, wide);
        check(lease.batch().token[0] == 7,
              "alloc: the fallback lease replaces an allocator-provided token");
    }
}

int main() {
    test_pooled_lease();
    test_fallback_lease();
    test_exhaustion_falls_back();
    test_events();
    test_cancel_releases_lease();
    test_steady_state();
    test_event_cap();
    test_piece_scratch_buffer();
    test_set_contract();
    test_fallback_set_contract();
    test_zero_width_refused();
    test_pool_lifetime();
    test_partial_batch_free_shapes();
    test_first_acquire_failure_retry();
    test_fallback_alloc_failure_retry();
    test_allocator_token_is_replaced();

    std::printf("%s failures=%d\n", g_failures ? "FAILED" : "OK", g_failures);

    return g_failures ? 1 : 0;
}
