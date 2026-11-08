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
#include <vector>

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
    test_kv_snapshot_slot_reuse();
    test_kv_cells_cp_into();

    std::printf("%s failures=%d allocs=%lld frees=%lld\n",
                g_failures ? "FAILED" : "OK", g_failures,
                g_allocs.load(), g_frees.load());

    return g_failures ? 1 : 0;
}
