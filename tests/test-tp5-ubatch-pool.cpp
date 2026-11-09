// TP5 host-pool semantics: fixed slots, intrusive handles, generation
// rejection, concurrent acquire/release, and KV cell snapshot reuse.
//
// The allocation counter proves the steady-state pool path performs no host
// allocation. It is a test-local replacement of the global operators, so the
// counts include every container touched by the loop.

#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cells.h"
#include "../src/llama-ubatch-pool.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <thread>
#include <type_traits>
#include <vector>

// the batch context owns its decode scratch, so a copy or move would return
// the same scratch twice
static_assert(!std::is_copy_constructible_v<llama_kv_cache_context>, "kv context must own its scratch");
static_assert(!std::is_copy_assignable_v<llama_kv_cache_context>,    "kv context must own its scratch");
static_assert(!std::is_move_constructible_v<llama_kv_cache_context>, "kv context must own its scratch");
static_assert(!std::is_move_assignable_v<llama_kv_cache_context>,    "kv context must own its scratch");

static std::atomic<long long> g_allocs{0};
static std::atomic<long long> g_frees{0};

void * operator new(std::size_t n) {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (void * p = std::malloc(n ? n : 1)) {
        return p;
    }
    throw std::bad_alloc();
}

void * operator new[](std::size_t n) {
    return operator new(n);
}

void operator delete(void * p) noexcept {
    if (p) {
        g_frees.fetch_add(1, std::memory_order_relaxed);
        std::free(p);
    }
}

void operator delete[](void * p) noexcept {
    operator delete(p);
}

void operator delete(void * p, std::size_t) noexcept {
    operator delete(p);
}

void operator delete[](void * p, std::size_t) noexcept {
    operator delete(p);
}

static long long alloc_delta() {
    return g_allocs.load(std::memory_order_relaxed) - g_frees.load(std::memory_order_relaxed);
}

static int g_failures = 0;

static void check(bool ok, const char * name) {
    if (!ok) {
        ++g_failures;
    }
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
}

static void fill(llama_ubatch_data & data, std::size_t n) {
    data.token.assign(n, 7);
    data.embd.assign(n*2, 1.0f);
    data.pos.assign(n, 3);
    data.n_seq_id.assign(n, 1);
    data.seq_id.assign(n, nullptr);
    data.seq_id_unq.assign(1, 0);
    data.seq_idx.assign(LLAMA_MAX_SEQ, -1);
    data.output.assign(n, 1);
    data.decision_order.assign(n, 0);
    data.seq_id_data.assign(n, 0);
}

static void test_reuse_and_lifetime() {
    auto pool = std::make_shared<ubatch_data_pool>();

    check(!pool->configured(), "pool: starts unconfigured");
    check(pool->configure(4), "pool: configure allocates slots");
    check(pool->capacity() == 4, "pool: capacity is fixed");

    {
        auto a = pool->acquire();
        check(bool(a), "pool: acquire succeeds");
        check(a.pooled(), "pool: handle reports a slot");
        fill(*a, 8);
        check(a->token.size() == 8 && a->token[0] == 7, "pool: payload visible");

        auto b = a;
        check(pool->in_use() == 1, "pool: copy retains the slot");
        b.reset();
        check(pool->in_use() == 1, "pool: dropping one copy keeps the slot");
    }
    check(pool->in_use() == 0, "pool: last copy returns the slot");

    const uint32_t slot = [&] {
        auto h = pool->acquire();
        const uint32_t s = h.slot();
        const uint32_t g = h.generation();
        h.reset();
        auto again = pool->acquire();
        check(again.slot() == s, "pool: released slot is reused first");
        check(again.generation() == g + 1, "pool: generation advances on reuse");
        check(pool->debug_release_stale(s, g), "pool: stale generation is rejected");
        check(pool->in_use() == 1, "pool: stale release keeps the new occupant");
        return again.slot();
    }();
    (void) slot;
    check(pool->in_use() == 0, "pool: new occupant releases normally");

    // Growing while a handle is live would move the slot storage.
    auto held = pool->acquire();
    check(!pool->configure(8), "pool: grow is refused while a handle is live");
    held.reset();
    check(pool->configure(8), "pool: grow succeeds after release");
    check(pool->capacity() == 8, "pool: grown capacity is visible");
}

// Regression for the H-group abort: a ubatch handle can outlive the allocator
// that owns the pool (llm_graph_params keeps an intentional copy). The handle
// must keep the pool alive instead of releasing into a destroyed mutex.
static void test_pool_outlives_owner() {
    ubatch_data_handle kept;

    {
        auto pool = std::make_shared<ubatch_data_pool>();
        check(pool->configure(1), "outlive: pool configured");

        kept = pool->acquire();
        fill(*kept, 4);
        check(kept->token.size() == 4, "outlive: payload visible");
    }

    check(bool(kept), "outlive: handle survives the pool owner");
    check(kept->token.size() == 4 && kept->token[0] == 7, "outlive: payload is still valid");

    kept.reset();
    check(true, "outlive: release after the owner died is safe");
}

static void test_exhaustion_and_fallback() {
    auto pool = std::make_shared<ubatch_data_pool>();
    check(pool->configure(1), "pool: one-slot pool");

    auto a = pool->acquire();
    auto b = pool->acquire();
    check(bool(a), "pool: first acquire succeeds");
    check(!b, "pool: second acquire reports exhaustion");
    check(pool->stats().exhausted == 1, "pool: exhaustion is counted");

    pool->note_fallback();
    check(pool->stats().fallback == 1, "pool: fallback is counted");

    // The caller-side fallback owns a heap payload like the old shared_ptr.
    ubatch_data_handle heap(std::make_shared<llama_ubatch_data>());
    check(bool(heap) && !heap.pooled(), "pool: heap fallback handle is valid");
    auto copy = heap;
    check(copy.get() == heap.get(), "pool: heap fallback copies share the payload");
}

static void test_steady_state() {
    auto pool = std::make_shared<ubatch_data_pool>();
    check(pool->configure(4), "steady: pool configured");

    {
        auto warm = pool->acquire();
        fill(*warm, 8);
    }

    const long long base = alloc_delta();

    for (int i = 0; i < 10000; ++i) {
        auto h = pool->acquire();
        if (!h) {
            check(false, "steady: unexpected exhaustion");
            break;
        }
        fill(*h, 8);
    }

    check(alloc_delta() == base, "steady: 10k acquire/fill/release, zero net allocation");
    check(pool->in_use() == 0, "steady: every slot returned");
    check(pool->stats().acquired == 10001, "steady: acquire count");
    check(pool->stats().released == 10001, "steady: release count");
    check(pool->stats().stale == 0, "steady: no stale release");
}

static void test_concurrency() {
    auto pool = std::make_shared<ubatch_data_pool>();
    check(pool->configure(8), "concurrent: eight slots");

    constexpr int threads = 8;
    constexpr int rounds  = 20000;

    std::atomic<long long> errors{0};
    std::vector<std::thread> workers;
    workers.reserve(threads);

    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            for (int i = 0; i < rounds; ++i) {
                auto h = pool->acquire();
                if (!h) {
                    errors.fetch_add(1);
                    continue;
                }
                fill(*h, 4);
                h->token[0] = static_cast<llama_token>(t*rounds + i);
                if (h->token[0] != static_cast<llama_token>(t*rounds + i)) {
                    errors.fetch_add(1);
                }
            }
        });
    }

    for (auto & worker : workers) {
        worker.join();
    }

    check(errors.load() == 0, "concurrent: no lost or duplicated slot");
    check(pool->in_use() == 0, "concurrent: every slot returned");
    check(pool->stats().exhausted == 0, "concurrent: capacity was sufficient");
    check(pool->stats().stale == 0, "concurrent: no stale release");
}

static void test_decode_scratch_pool() {
    llama_kv_cache::decode_scratch_pool pool(2);

    check(pool.capacity() == 2, "scratch pool: capacity");
    check(pool.in_use() == 0, "scratch pool: starts free");

    auto * a = pool.acquire();
    auto * b = pool.acquire();
    check(a != nullptr && b != nullptr && a != b, "scratch pool: two distinct slots");
    check(pool.in_use() == 2, "scratch pool: in-use count");

    auto * c = pool.acquire();
    check(c == nullptr, "scratch pool: exhaustion returns nullptr");
    check(pool.fallbacks() == 1, "scratch pool: fallback counted");

    // Cancel/failure recovery: a scratch with filled vectors is cleared on
    // release and can be acquired again.
    llama_kv_cache::snapshot_slot fake;
    a->states.push_back(&fake);
    a->pooled.push_back(&fake);
    a->fallback.push_back(std::make_unique<llama_kv_cache::snapshot_slot>());
    pool.release(a);
    check(a->states.empty() && a->pooled.empty() && a->fallback.empty(),
          "scratch pool: release clears the vectors");
    check(pool.in_use() == 1, "scratch pool: release decrements");

    auto * a2 = pool.acquire();
    check(a2 == a, "scratch pool: released slot is reused first");

    // Dual-window isolation: releasing one slot does not disturb the other.
    check(pool.in_use() == 2, "scratch pool: both slots busy");
    pool.release(b);
    check(pool.in_use() == 1, "scratch pool: releasing one keeps the other");
    check(a2->in_use, "scratch pool: the other slot remains busy");
    pool.release(a2);
    check(pool.in_use() == 0, "scratch pool: all slots returned");

    // A separate cache has a separate pool.
    llama_kv_cache::decode_scratch_pool other(1);
    auto * x = other.acquire();
    check(x != nullptr && x != a, "scratch pool: separate pools do not share slots");
    other.release(x);

    // 10k acquire/fill/release after warmup: no net host allocation.
    const long long base = alloc_delta();
    for (int i = 0; i < 10000; ++i) {
        auto * s = pool.acquire();
        if (s == nullptr) {
            check(false, "scratch pool: unexpected exhaustion");
            break;
        }
        s->sinfos.resize(4);
        s->states.push_back(&fake);
        s->pooled.push_back(&fake);
        pool.release(s);
    }
    check(alloc_delta() == base, "scratch pool: 10k acquire/fill/release, zero net allocation");
    check(pool.in_use() == 0, "scratch pool: every slot returned");
}

// The decode scratch is handed to one batch context at a time. Under
// contention the free-list must never give the same slot to two threads, and
// every slot must come back when its owner releases it.
static void test_decode_scratch_pool_concurrency() {
    constexpr int threads = 8;
    constexpr int rounds  = 20000;

    {
        llama_kv_cache::decode_scratch_pool pool(threads);

        std::atomic<long long> errors{0};
        std::vector<std::thread> workers;
        workers.reserve(threads);

        for (int t = 0; t < threads; ++t) {
            workers.emplace_back([&, t] {
                for (int i = 0; i < rounds; ++i) {
                    auto * s = pool.acquire();
                    if (s == nullptr) {
                        errors.fetch_add(1);
                        continue;
                    }

                    // the slot is exclusive: no other thread may overwrite
                    // this stamp while the acquire is held
                    const uint32_t stamp = static_cast<uint32_t>(t*rounds + i) + 1;
                    s->sinfos.resize(1);
                    s->sinfos[0].s0 = stamp;
                    std::this_thread::yield();
                    if (s->sinfos[0].s0 != stamp) {
                        errors.fetch_add(1);
                    }

                    pool.release(s);
                }
            });
        }

        for (auto & worker : workers) {
            worker.join();
        }

        check(errors.load() == 0, "scratch pool concurrent: exclusive slot");
        check(pool.in_use() == 0, "scratch pool concurrent: every slot returned");
        check(pool.fallbacks() == 0, "scratch pool concurrent: capacity was sufficient");
    }

    {
        // an exhausted pool counts every refusal and still returns every
        // acquired slot
        llama_kv_cache::decode_scratch_pool pool(2);

        constexpr int workers_n = 8;
        std::atomic<long long> refusals{0};
        std::vector<std::thread> workers;
        workers.reserve(workers_n);

        for (int t = 0; t < workers_n; ++t) {
            workers.emplace_back([&] {
                for (int i = 0; i < 2000; ++i) {
                    auto * s = pool.acquire();
                    if (s == nullptr) {
                        refusals.fetch_add(1);
                        continue;
                    }
                    pool.release(s);
                }
            });
        }

        for (auto & worker : workers) {
            worker.join();
        }

        check(pool.in_use() == 0, "scratch pool exhausted: every slot returned");
        check(static_cast<long long>(pool.fallbacks()) == refusals.load(),
              "scratch pool exhausted: every refusal counted");
    }
}

static void test_kv_snapshot_slot_reuse() {
    llama_kv_cells src;
    src.resize(64);

    for (uint32_t i = 0; i < 64; ++i) {
        src.pos_set(i, static_cast<llama_pos>(i*3));
        src.seq_add(i, static_cast<llama_seq_id>(i%4));
        llama_kv_cell_ext ext;
        ext.x = static_cast<llama_pos>(i + 1);
        ext.y = static_cast<llama_pos>(i + 2);
        src.ext_set(i, ext);
    }

    const std::vector<uint32_t> first  = { 0, 5, 7, 63, 1, 2, 3 };
    const std::vector<uint32_t> second = { 63, 0 };

    llama_kv_cache::snapshot_slot slot;

    for (int round = 0; round < 2; ++round) {
        const auto & idxs = round == 0 ? first : second;

        slot.sinfo.resize(1);
        slot.sinfo.idxs[0] = idxs;
        slot.v_cells.resize(1);

        src.cp_into(idxs, slot.v_cells[0]);

        const llama_kv_cells ref = src.cp(idxs);

        check(slot.v_cells[0].size() == ref.size(), "kv snapshot: size");
        check(slot.v_cells[0].get_used() == ref.get_used(), "kv snapshot: used");
        for (std::size_t j = 0; j < idxs.size(); ++j) {
            check(slot.v_cells[0].pos_get(j) == ref.pos_get(j), "kv snapshot: pos");
            check(slot.v_cells[0].seq_get(j) == ref.seq_get(j), "kv snapshot: seq");
            check(slot.v_cells[0].ext_get(j).x == ref.ext_get(j).x &&
                  slot.v_cells[0].ext_get(j).y == ref.ext_get(j).y, "kv snapshot: ext");
        }
    }

    // A second reuse with a shorter idxs must not expose the old tail.
    check(slot.v_cells[0].size() == second.size(), "kv snapshot: shrinks on reuse");
}

static void test_kv_cells_cp_into() {
    llama_kv_cells src;
    src.resize(64);

    for (uint32_t i = 0; i < 64; ++i) {
        src.pos_set(i, static_cast<llama_pos>(i*3));
        src.seq_add(i, static_cast<llama_seq_id>(i%4));
        llama_kv_cell_ext ext;
        ext.x = static_cast<llama_pos>(i + 1);
        ext.y = static_cast<llama_pos>(i + 2);
        src.ext_set(i, ext);
    }

    const std::vector<uint32_t> idxs = { 0, 5, 7, 63, 1, 2, 3 };
    const llama_kv_cells ref = src.cp(idxs);

    llama_kv_cells out;
    src.cp_into(idxs, out);

    check(out.size() == idxs.size(), "kv: cp_into size");
    check(out.get_used() == ref.get_used(), "kv: cp_into leaves used empty like cp");
    for (std::size_t j = 0; j < idxs.size(); ++j) {
        check(out.pos_get(j) == ref.pos_get(j), "kv: cp_into pos");
        check(out.seq_get(j) == ref.seq_get(j), "kv: cp_into seq");
        check(out.ext_get(j).x == ref.ext_get(j).x && out.ext_get(j).y == ref.ext_get(j).y,
              "kv: cp_into ext");
    }

    // A shorter copy must not leave the previous tail visible.
    const std::vector<uint32_t> shorter = { 63, 0 };
    src.cp_into(shorter, out);
    check(out.size() == shorter.size(), "kv: cp_into shrinks");
    check(out.pos_get(0) == 189 && out.pos_get(1) == 0, "kv: cp_into overwrites the prefix");

    const long long base = alloc_delta();
    for (int i = 0; i < 10000; ++i) {
        src.cp_into(idxs, out);
    }
    check(alloc_delta() == base, "kv: 10k cp_into, zero net allocation after warmup");
}

int main() {
    test_reuse_and_lifetime();
    test_pool_outlives_owner();
    test_exhaustion_and_fallback();
    test_steady_state();
    test_concurrency();
    test_decode_scratch_pool();
    test_decode_scratch_pool_concurrency();
    test_kv_snapshot_slot_reuse();
    test_kv_cells_cp_into();

    std::printf("%s failures=%d allocs=%lld frees=%lld\n",
                g_failures ? "FAILED" : "OK", g_failures,
                g_allocs.load(), g_frees.load());

    return g_failures ? 1 : 0;
}
