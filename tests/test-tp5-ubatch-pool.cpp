// TP5 host-pool semantics: fixed slots, intrusive handles, generation
// rejection, concurrent acquire/release, and KV cell snapshot reuse.
//
// The allocation counter proves the steady-state pool path performs no host
// allocation. It is a test-local replacement of the global operators, so the
// counts include every container touched by the loop.

#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cells.h"
#include "../src/llama-ubatch-pool.h"
#include "../src/llama-batch.h"
#include "../src/llama-hparams.h"
#include "../src/llama-model.h"
#include "ggml-backend.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <thread>
#include <vector>

static std::atomic<long long> g_allocs{0};
static std::atomic<long long> g_frees{0};
static std::atomic<bool>      g_fault_enabled{false};
static std::atomic<bool>      g_fault_repeat{false};
static std::atomic<long long> g_fault_countdown{-1};

struct mutation_probe;
static mutation_probe * g_probe = nullptr;
static void probe_alloc();

static void fault_check() {
    probe_alloc();
    if (g_fault_enabled.load(std::memory_order_relaxed)) {
        if (g_fault_countdown.fetch_sub(1, std::memory_order_relaxed) == 0) {
            if (!g_fault_repeat.load(std::memory_order_relaxed)) {
                g_fault_enabled.store(false, std::memory_order_relaxed);
            }
            throw std::bad_alloc();
        }
    }
}

void * operator new(std::size_t n) {
    fault_check();
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (void * p = std::malloc(n ? n : 1)) {
        return p;
    }
    throw std::bad_alloc();
}

void * operator new[](std::size_t n) {
    return operator new(n);
}

void * operator new(std::size_t n, std::align_val_t a) {
    fault_check();
    g_allocs.fetch_add(1, std::memory_order_relaxed);

    const std::size_t align = static_cast<std::size_t>(a);
    const std::size_t size  = (n + align - 1) / align * align;

    if (void * p = std::aligned_alloc(align, size ? size : align)) {
        return p;
    }
    throw std::bad_alloc();
}

void * operator new[](std::size_t n, std::align_val_t a) {
    return operator new(n, a);
}

void * operator new(std::size_t n, const std::nothrow_t &) noexcept {
    try {
        return operator new(n);
    } catch (...) {
        return nullptr;
    }
}

void * operator new[](std::size_t n, const std::nothrow_t &) noexcept {
    try {
        return operator new[](n);
    } catch (...) {
        return nullptr;
    }
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

void operator delete(void * p, std::align_val_t) noexcept {
    operator delete(p);
}

void operator delete[](void * p, std::align_val_t) noexcept {
    operator delete(p);
}

void operator delete(void * p, std::size_t, std::align_val_t) noexcept {
    operator delete(p);
}

void operator delete[](void * p, std::size_t, std::align_val_t) noexcept {
    operator delete(p);
}

void operator delete(void * p, const std::nothrow_t &) noexcept {
    operator delete(p);
}

void operator delete[](void * p, const std::nothrow_t &) noexcept {
    operator delete(p);
}

static long long alloc_delta() {
    return g_allocs.load(std::memory_order_relaxed) - g_frees.load(std::memory_order_relaxed);
}

static void reset_fault() {
    g_fault_enabled.store(false, std::memory_order_relaxed);
    g_fault_repeat.store(false, std::memory_order_relaxed);
    g_fault_countdown.store(-1, std::memory_order_relaxed);
}

static void arm_fault_at(long long index, bool repeat) {
    g_fault_repeat.store(repeat, std::memory_order_relaxed);
    g_fault_countdown.store(index, std::memory_order_relaxed);
    g_fault_enabled.store(true, std::memory_order_relaxed);
}

// Resets the global allocator fault on every exit path, so a fault cannot
// leak an armed fault into the rest of the suite.
struct fault_guard {
    ~fault_guard() { reset_fault(); }
};

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

// shared_from_this() throws when the pool is not owned by a shared_ptr. The
// slot must not be marked before that call, or a refused acquire would leave
// it in_use with no handle left to release it.
static void test_acquire_requires_shared_owner() {
    ubatch_data_pool pool;
    check(pool.configure(2), "owner: pool configured");

    bool threw = false;
    try {
        auto handle = pool.acquire();
        (void) handle;
    } catch (const std::bad_weak_ptr &) {
        threw = true;
    }

    check(threw, "owner: acquire refuses a pool without a shared owner");
    check(pool.in_use() == 0, "owner: the refused acquire leaves no slot marked");
    check(pool.stats().acquired == 0, "owner: the refused acquire is not counted");
    check(pool.stats().exhausted == 0, "owner: the refusal is not an exhaustion");
}

// Each split builds one index list before ubatch_add() consumes it. The
// allocator must keep that list, or a steady one-token decode allocates and
// frees a vector on every decode call.
static void test_split_scratch_reuse() {
#ifdef _WIN32
    _putenv_s("TP5_UBATCH_POOL", "1");
#else
    setenv("TP5_UBATCH_POOL", "1", 1);
#endif

    llama_vocab vocab;

    constexpr uint32_t n_tokens = 8;
    constexpr uint32_t n_embd   = 2;

    std::vector<float>          embd(n_tokens * n_embd, 0.0f);
    std::vector<llama_pos>      pos(n_tokens);
    std::vector<int32_t>        n_seq_id(n_tokens, 1);
    std::vector<llama_seq_id>   seq_ids(n_tokens, 0);
    std::vector<llama_seq_id *> seq_id(n_tokens);
    std::vector<int8_t>         logits(n_tokens, 0);

    for (uint32_t i = 0; i < n_tokens; ++i) {
        pos[i]    = (llama_pos) i;
        seq_id[i] = &seq_ids[i];
    }
    logits[n_tokens - 1] = 1;

    llama_batch batch = {};
    batch.n_tokens = (int32_t) n_tokens;
    batch.embd     = embd.data();
    batch.pos      = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id   = seq_id.data();
    batch.logits   = logits.data();

    llama_batch_allocr ba(1);
    check(ba.init(batch, vocab, nullptr, n_embd, 4, false), "split scratch: init");

    // Warm the index list, the out_ids list and one pool slot.
    while (ba.split_simple(1).n_tokens > 0) {
    }

    const long long base = alloc_delta();

    for (int round = 0; round < 1000; ++round) {
        ba.split_reset();
        while (ba.split_simple(1).n_tokens > 0) {
        }
    }

    check(alloc_delta() == base, "split scratch: 1000 resplit rounds, zero net allocation");

#ifdef _WIN32
    _putenv_s("TP5_UBATCH_POOL", "");
#else
    unsetenv("TP5_UBATCH_POOL");
#endif
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

// A multi-stream ubatch can name its streams out of order. The reverse restore
// must return each stream's head to that stream's own snapshot, and it must run
// on the success path as well as on the find_slot failure path.
static void test_restore_cells_multi_stream() {
    const auto set_cell = [](llama_kv_cells & cells, uint32_t idx, llama_pos pos, llama_seq_id seq) {
        cells.pos_set(idx, pos);
        cells.seq_add(idx, seq);
    };

    const auto make_cells = [&]() {
        llama_kv_cells_vec cells(4);
        for (auto & cells_i : cells) {
            cells_i.resize(8);
        }
        set_cell(cells[1], 2, 10, 1);
        set_cell(cells[3], 0, 30, 3);
        set_cell(cells[3], 1, 31, 3);
        return cells;
    };

    const auto snapshot = [](const llama_kv_cells_vec & cells, const llama_kv_cache::slot_info & sinfo) {
        std::vector<llama_kv_cells> out(sinfo.n_stream());
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            cells[sinfo.strm[s]].cp_into(sinfo.idxs[s], out[s]);
        }
        return out;
    };

    const auto place = [](llama_kv_cells_vec & cells, std::vector<uint32_t> & heads,
                          const llama_kv_cache::slot_info & sinfo,
                          llama_pos base, llama_seq_id seq) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells_s = cells[sinfo.strm[s]];
            for (uint32_t i = 0; i < sinfo.idxs[s].size(); ++i) {
                const uint32_t idx = sinfo.idxs[s][i];
                if (!cells_s.is_empty(idx)) {
                    cells_s.rm(idx);
                }
                cells_s.pos_set(idx, base + static_cast<llama_pos>(i));
                cells_s.seq_add(idx, seq);
            }
            heads[sinfo.strm[s]] = sinfo.idxs[s].back() + 1;
        }
    };

    {
        auto cells = make_cells();
        std::vector<uint32_t> heads = { 0, 3, 0, 2 };

        llama_kv_cache::slot_info sinfo = {};
        sinfo.s0 = 1;
        sinfo.s1 = 3;
        sinfo.strm = { 3, 1 };
        sinfo.idxs = { { 0, 1 }, { 2 } };

        const auto old_cells = snapshot(cells, sinfo);
        const auto old_heads = heads;

        place(cells, heads, sinfo, 90, 7);
        llama_kv_cache::restore_cells(cells, heads, sinfo, old_heads, old_cells);

        check(heads == old_heads, "restore multi-stream: heads (3,1)");
        check(cells[3].pos_get(0) == 30 && cells[3].seq_get(0) == 3,
              "restore multi-stream: stream 3 cell 0");
        check(cells[3].pos_get(1) == 31 && cells[3].seq_get(1) == 3,
              "restore multi-stream: stream 3 cell 1");
        check(cells[1].pos_get(2) == 10 && cells[1].seq_get(2) == 1,
              "restore multi-stream: stream 1 cell 2");
        check(cells[0].get_used() == 0 && cells[2].get_used() == 0,
              "restore multi-stream: untouched streams");
    }

    {
        auto cells = make_cells();
        std::vector<uint32_t> heads = { 0, 5, 0, 6 };

        llama_kv_cache::slot_info sinfo = {};
        sinfo.s0 = 1;
        sinfo.s1 = 3;
        sinfo.strm = { 1, 3 };
        sinfo.idxs = { { 2 }, { 0, 1 } };

        const auto old_cells = snapshot(cells, sinfo);
        const auto old_heads = heads;

        place(cells, heads, sinfo, 90, 7);
        llama_kv_cache::restore_cells(cells, heads, sinfo, old_heads, old_cells);

        check(heads == old_heads, "restore multi-stream: heads (1,3)");
        check(cells[1].pos_get(2) == 10 && cells[1].seq_get(2) == 1,
              "restore multi-stream: stream 1 cell 2");
        check(cells[3].pos_get(0) == 30 && cells[3].seq_get(0) == 3,
              "restore multi-stream: stream 3 cell 0");
        check(cells[3].pos_get(1) == 31 && cells[3].seq_get(1) == 3,
              "restore multi-stream: stream 3 cell 1");
    }

    {
        // two successful trial placements restore in reverse order
        auto cells = make_cells();
        std::vector<uint32_t> heads = { 0, 3, 0, 2 };

        llama_kv_cache::slot_info first = {};
        first.s0 = 3;
        first.s1 = 3;
        first.strm = { 3 };
        first.idxs = { { 0 } };

        llama_kv_cache::slot_info second = {};
        second.s0 = 1;
        second.s1 = 3;
        second.strm = { 1, 3 };
        second.idxs = { { 2 }, { 1 } };

        const auto old_heads = heads;
        const auto first_cells = snapshot(cells, first);
        place(cells, heads, first, 70, 7);

        const auto second_heads = heads;
        const auto second_cells = snapshot(cells, second);
        place(cells, heads, second, 80, 8);

        llama_kv_cache::restore_cells(cells, heads, second, second_heads, second_cells);
        llama_kv_cache::restore_cells(cells, heads, first, old_heads, first_cells);

        check(heads == old_heads, "restore trial placement: heads");
        check(cells[3].pos_get(0) == 30 && cells[3].seq_get(0) == 3,
              "restore trial placement: stream 3 cell 0");
        check(cells[3].pos_get(1) == 31 && cells[3].seq_get(1) == 3,
              "restore trial placement: stream 3 cell 1");
        check(cells[1].pos_get(2) == 10 && cells[1].seq_get(2) == 1,
              "restore trial placement: stream 1 cell 2");
    }

    {
        // find_slot fails for the next ubatch: only the recorded trial
        // placement is undone
        auto cells = make_cells();
        std::vector<uint32_t> heads = { 0, 3, 0, 2 };

        llama_kv_cache::slot_info applied = {};
        applied.s0 = 3;
        applied.s1 = 3;
        applied.strm = { 3 };
        applied.idxs = { { 0, 1 } };

        const auto old_heads = heads;
        const auto applied_cells = snapshot(cells, applied);
        place(cells, heads, applied, 70, 7);

        llama_kv_cache::restore_cells(cells, heads, applied, old_heads, applied_cells);

        check(heads == old_heads, "restore find_slot failure: heads");
        check(cells[3].pos_get(0) == 30 && cells[3].seq_get(0) == 3,
              "restore find_slot failure: stream 3 cell 0");
        check(cells[3].pos_get(1) == 31 && cells[3].seq_get(1) == 3,
              "restore find_slot failure: stream 3 cell 1");
        check(cells[1].pos_get(2) == 10 && cells[1].seq_get(2) == 1,
              "restore find_slot failure: untouched stream 1");
    }
}

// The CPU test seam for the real-cache SWA counterexample. It calls the
// production prepare paths and reads the cells/heads they mutate; it adds no
// runtime behavior.
struct llama_kv_cache_prepare_test_access {
    static llama_kv_cache::slot_info_vec_t prepare_reference(
            llama_kv_cache & kv, const std::vector<llama_ubatch> & ubatches) {
        return kv.prepare_reference(ubatches);
    }

    static bool prepare_into(
            llama_kv_cache & kv, const std::vector<llama_ubatch> & ubatches,
            llama_kv_cache::slot_info_vec_t & res) {
        return kv.prepare_into(ubatches, res, nullptr);
    }

    static llama_kv_cells & cells(llama_kv_cache & kv, uint32_t stream) {
        return kv.v_cells[stream];
    }

    static std::vector<uint32_t> & heads(llama_kv_cache & kv) {
        return kv.v_heads;
    }

    static bool journal_begin(llama_kv_cache & kv) {
        return kv.prepare_journal_begin();
    }

    static void journal_end(llama_kv_cache & kv) {
        kv.prepare_journal_end(true);
    }

    static std::vector<uint32_t> & heads_before(llama_kv_cache & kv) {
        return kv.heads_before;
    }

    static llama_kv_cache::decode_scratch * scratch_acquire(llama_kv_cache & kv) {
        return kv.scratch_acquire();
    }

    static void scratch_return(llama_kv_cache & kv, llama_kv_cache::decode_scratch * scratch) {
        kv.scratch_return(scratch->sinfos, scratch->ubatches, scratch);
    }

    static bool prepare_into_scratch(
            llama_kv_cache & kv, const std::vector<llama_ubatch> & ubatches,
            llama_kv_cache::decode_scratch * scratch) {
        return kv.prepare_into(ubatches, scratch->sinfos, scratch);
    }

};

// A model that exists only to carry hparams. The loaders are never called;
// the cache constructor only reads hparams and the CPU buffer type.
struct test_stub_model : llama_model {
    explicit test_stub_model(const llama_model_params & params) : llama_model(params) {}

    void load_stats        (struct llama_model_loader &) override {}
    void load_hparams      (struct llama_model_loader &) override {}
    void load_vocab        (struct llama_model_loader &) override {}
    bool load_tensors      (struct llama_model_loader &) override { return false; }
    void load_arch_hparams (struct llama_model_loader &) override {}
    void load_arch_tensors (struct llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override {
        return nullptr;
    }
};

struct test_kv_fixture {
    ggml_backend_dev_t devices[1] = { nullptr };
    test_stub_model *  model      = nullptr;

    test_kv_fixture(uint32_t n_swa, llama_swa_type swa_type) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.vocab_only   = true;
        mp.devices      = devices;

        model = new test_stub_model(mp);

        model->hparams.n_layer_all          = 1;
        model->hparams.n_layer_nextn        = 0;
        model->hparams.n_embd               = 8;
        model->hparams.n_ctx_train          = 1024;
        model->hparams.n_head_arr[0]        = 1;
        model->hparams.n_head_kv_arr[0]     = 1;
        model->hparams.n_embd_head_k_full   = 8;
        model->hparams.n_embd_head_v_full   = 8;
        model->hparams.n_embd_head_k_swa    = 8;
        model->hparams.n_embd_head_v_swa    = 8;
        model->hparams.is_swa_impl[0]       = swa_type != LLAMA_SWA_TYPE_NONE ? 1 : 0;
        model->hparams.n_swa                = n_swa;
        model->hparams.swa_type             = swa_type;
        model->hparams.n_layer_kv_from_start = -1;
    }

    ~test_kv_fixture() {
        delete model;
    }
};

static llama_kv_cache * make_test_cache(
        test_kv_fixture & f, uint32_t n_swa, llama_swa_type swa_type, uint32_t kv_size,
        bool unified = true, uint32_t n_seq_max = 1) {
    return new llama_kv_cache(
            *f.model, f.model->hparams,
            GGML_TYPE_F32, GGML_TYPE_F32,
            /*v_trans=*/false,
            /*offload=*/false,
            /*unified=*/unified,
            kv_size,
            /*n_seq_max=*/n_seq_max,
            /*n_pad=*/1,
            n_swa,
            swa_type,
            /*mem_other=*/nullptr,
            /*filter=*/{},
            /*reuse=*/{},
            /*share=*/{});
}

static void seed_test_cache(llama_kv_cache & kv, uint32_t n_cells, uint32_t gap0, uint32_t gap1, uint32_t head) {
    auto & cells = llama_kv_cache_prepare_test_access::cells(kv, 0);

    for (uint32_t i = 0; i < n_cells; ++i) {
        if (i == gap0 || i == gap1) {
            continue;
        }

        cells.pos_set(i, static_cast<llama_pos>(i));
        cells.seq_add(i, 0);
    }

    llama_kv_cache_prepare_test_access::heads(kv)[0] = head;
}

static void seed_stream(llama_kv_cache & kv, uint32_t stream, uint32_t n_cells, uint32_t head) {
    auto & cells = llama_kv_cache_prepare_test_access::cells(kv, stream);

    for (uint32_t i = 0; i < n_cells; ++i) {
        cells.pos_set(i, static_cast<llama_pos>(i));
        cells.seq_add(i, static_cast<llama_seq_id>(stream));
    }

    llama_kv_cache_prepare_test_access::heads(kv)[stream] = head;
}

struct test_ubatch {
    llama_token  token     = 1;
    llama_pos    pos;
    int32_t      n_seq_id  = 1;
    llama_seq_id seq       = 0;
    llama_seq_id * seq_ptr = &seq;
    int8_t       output    = 0;
    int32_t      seq_idx[LLAMA_MAX_SEQ];
    llama_ubatch ub        = {};

    explicit test_ubatch(llama_pos p, llama_seq_id seq_id = 0) : pos(p), seq(seq_id) {
        for (int i = 0; i < LLAMA_MAX_SEQ; ++i) {
            seq_idx[i] = -1;
        }
        seq_idx[seq_id] = 0;

        ub.n_tokens     = 1;
        ub.n_seq_tokens = 1;
        ub.n_seqs       = 1;
        ub.n_seqs_unq   = 1;
        ub.n_pos        = 1;
        ub.token        = &token;
        ub.pos          = &pos;
        ub.n_seq_id     = &n_seq_id;
        ub.seq_id       = &seq_ptr;
        ub.seq_id_unq   = &seq;
        ub.seq_idx      = seq_idx;
        ub.output       = &output;
    }
};

// Two tokens in two sequences, one token per sequence. Used with a
// non-unified cache whose two streams each hold one sequence.
struct test_ubatch_multi {
    llama_token  tokens[2] = { 1, 2 };
    llama_pos    pos[2];
    int32_t      n_seq_id[2] = { 1, 1 };
    llama_seq_id seq_ids[2][1] = { { 0 }, { 1 } };
    llama_seq_id * seq_ptr[2] = { seq_ids[0], seq_ids[1] };
    int8_t       output[2] = { 0, 0 };
    int32_t      seq_idx[LLAMA_MAX_SEQ];
    llama_seq_id seq_unq[2] = { 0, 1 };
    llama_ubatch ub = {};

    test_ubatch_multi(llama_pos p0, llama_pos p1) {
        pos[0] = p0;
        pos[1] = p1;

        for (int i = 0; i < LLAMA_MAX_SEQ; ++i) {
            seq_idx[i] = -1;
        }
        seq_idx[0] = 0;
        seq_idx[1] = 1;

        ub.n_tokens     = 2;
        ub.n_seq_tokens = 1;
        ub.n_seqs       = 2;
        ub.n_seqs_unq   = 2;
        ub.n_pos        = 1;
        ub.token        = tokens;
        ub.pos          = pos;
        ub.n_seq_id     = n_seq_id;
        ub.seq_id       = seq_ptr;
        ub.seq_id_unq   = seq_unq;
        ub.seq_idx      = seq_idx;
        ub.output       = output;
    }
};

struct test_prepare_result {
    bool ok = false;
    llama_kv_cache::slot_info_vec_t res;
};

static test_prepare_result run_test_prepare(
        llama_kv_cache & kv, const std::vector<llama_ubatch> & ubatches, bool pooled) {
    test_prepare_result out;

    if (pooled) {
        out.res.resize(ubatches.size());
        out.ok = llama_kv_cache_prepare_test_access::prepare_into(kv, ubatches, out.res);
    } else {
        out.res = llama_kv_cache_prepare_test_access::prepare_reference(kv, ubatches);
        out.ok  = !out.res.empty();
    }

    return out;
}

// SWA counterexample: the last trial overwrites a masked old cell and its
// seq_rm purge removes positions outside the trial slot. The reference path
// purges during prepare; a pooled path that skips the last trial would defer
// the purge until after the earlier ubatches' graphs read the cache.
static void test_swa_last_trial_purge() {
    test_kv_fixture f(512, LLAMA_SWA_TYPE_STANDARD);

    test_ubatch a(900);
    test_ubatch b(1024);
    const std::vector<llama_ubatch> ubatches = { a.ub, b.ub };

    for (int pooled = 0; pooled < 2; ++pooled) {
        llama_kv_cache * kv = make_test_cache(f, 512, LLAMA_SWA_TYPE_STANDARD, 1024);
        seed_test_cache(*kv, 1024, 399, 900, 399);

        const auto r = run_test_prepare(*kv, ubatches, pooled != 0);

        check(r.ok, pooled ? "swa pooled: prepare succeeds" : "swa reference: prepare succeeds");
        check(r.res.size() == 2, pooled ? "swa pooled: two slots" : "swa reference: two slots");

        if (r.res.size() == 2) {
            check(r.res[0].idxs[0].size() == 1 && r.res[0].idxs[0][0] == 399,
                  pooled ? "swa pooled: first slot" : "swa reference: first slot");
            check(r.res[1].idxs[0].size() == 1 && r.res[1].idxs[0][0] == 400,
                  pooled ? "swa pooled: last slot" : "swa reference: last slot");
        }

        auto & cells = llama_kv_cache_prepare_test_access::cells(*kv, 0);
        check(cells.is_empty(390),
              pooled ? "swa pooled: earlier-window cell purged" : "swa reference: earlier-window cell purged");
        check(cells.pos_get(400) == 400,
              pooled ? "swa pooled: overwritten cell restored" : "swa reference: overwritten cell restored");
        check(cells.pos_get(401) == 401,
              pooled ? "swa pooled: later cell retained" : "swa reference: later cell retained");
        check(llama_kv_cache_prepare_test_access::heads(*kv)[0] == 399,
              pooled ? "swa pooled: head restored" : "swa reference: head restored");

        delete kv;
    }
}

// Non-SWA control: the last trial is still skipped, but apply_ubatch has no
// purge. Both paths must leave the same restored cache and output slots.
static void test_non_swa_last_trial_skip() {
    test_kv_fixture f(0, LLAMA_SWA_TYPE_NONE);

    test_ubatch a(6);
    test_ubatch b(7);
    const std::vector<llama_ubatch> ubatches = { a.ub, b.ub };

    for (int pooled = 0; pooled < 2; ++pooled) {
        llama_kv_cache * kv = make_test_cache(f, 0, LLAMA_SWA_TYPE_NONE, 8);
        seed_test_cache(*kv, 6, ~0u, ~0u, 6);

        const auto r = run_test_prepare(*kv, ubatches, pooled != 0);

        check(r.ok, pooled ? "non-swa pooled: prepare succeeds" : "non-swa reference: prepare succeeds");
        check(r.res.size() == 2, pooled ? "non-swa pooled: two slots" : "non-swa reference: two slots");

        if (r.res.size() == 2) {
            check(r.res[0].idxs[0].size() == 1 && r.res[0].idxs[0][0] == 6,
                  pooled ? "non-swa pooled: first slot" : "non-swa reference: first slot");
            check(r.res[1].idxs[0].size() == 1 && r.res[1].idxs[0][0] == 7,
                  pooled ? "non-swa pooled: last slot" : "non-swa reference: last slot");
        }

        auto & cells = llama_kv_cache_prepare_test_access::cells(*kv, 0);
        check(cells.get_used() == 6,
              pooled ? "non-swa pooled: cache restored" : "non-swa reference: cache restored");
        check(cells.is_empty(6) && cells.is_empty(7),
              pooled ? "non-swa pooled: trial cells empty" : "non-swa reference: trial cells empty");
        check(llama_kv_cache_prepare_test_access::heads(*kv)[0] == 6,
              pooled ? "non-swa pooled: head restored" : "non-swa reference: head restored");

        delete kv;
    }
}

// The allocator-index sweep arms the real global allocator fault at each
// allocation of a prepare. A caught fault must leave the cache in the initial
// state (the SWA journal rolls back the full write domain, including the purge
// cells outside sinfo.idxs) or, when the rollback itself fails, mark the cache
// failed so every public path refuses. It must never report a partial state
// as restored.
static bool same_cells(const llama_kv_cells & a, const llama_kv_cells & b, uint32_t n) {
    if (a.size() < n || b.size() < n) {
        return false;
    }
    if (a.get_has_shift() != b.get_has_shift()) {
        return false;
    }
    if (a.get_used_set() != b.get_used_set()) {
        return false;
    }
    for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (a.get_seq_pos(s) != b.get_seq_pos(s)) {
            return false;
        }
    }
    for (uint32_t i = 0; i < n; ++i) {
        const bool ea = a.is_empty(i);
        if (ea != b.is_empty(i)) {
            return false;
        }
        if (!ea && (a.pos_get(i) != b.pos_get(i) ||
                    a.seq_get_all(i) != b.seq_get_all(i) ||
                    a.get_shift(i) != b.get_shift(i) ||
                    a.ext_get(i).x != b.ext_get(i).x ||
                    a.ext_get(i).y != b.ext_get(i).y)) {
            return false;
        }
    }
    return true;
}

struct mutation_probe {
    llama_kv_cache * kv;
    std::vector<llama_kv_cells> init;
    uint32_t target_mask;
    uint32_t seen_mask = 0;
    long long alloc_index = -1;
    bool fired = false;

    mutation_probe(llama_kv_cache * kv_, std::vector<llama_kv_cells> init_, uint32_t mask)
        : kv(kv_), init(std::move(init_)), target_mask(mask) {}

    void on_alloc(long long index) noexcept {
        if (fired || kv == nullptr) {
            return;
        }
        for (size_t s = 0; s < init.size(); ++s) {
            if (!same_cells(llama_kv_cache_prepare_test_access::cells(*kv, s), init[s], init[s].size())) {
                seen_mask |= 1u << s;
            }
        }
        if ((seen_mask & target_mask) != target_mask) {
            return;
        }
        fired = true;
        alloc_index = index;
        g_fault_repeat.store(true, std::memory_order_relaxed);
        g_fault_countdown.store(0, std::memory_order_relaxed);
        g_fault_enabled.store(true, std::memory_order_relaxed);
    }
};

static void probe_alloc() {
    if (g_probe != nullptr) {
        g_probe->on_alloc(g_allocs.load(std::memory_order_relaxed));
    }
}

struct probe_guard {
    explicit probe_guard(mutation_probe * probe) { g_probe = probe; }
    ~probe_guard() { g_probe = nullptr; }
};

struct mutation_fault_result {
    bool threw = false;
    bool fired = false;
    uint32_t seen_mask = 0;
    long long alloc_index = -1;
};

static mutation_fault_result run_mutation_fault(
        llama_kv_cache & kv,
        std::vector<llama_kv_cells> init,
        uint32_t target_mask,
        const std::vector<llama_ubatch> & ubatches,
        bool pooled,
        llama_kv_cache::slot_info_vec_t & res) {
    mutation_probe probe(&kv, std::move(init), target_mask);
    mutation_fault_result out;
    {
        fault_guard guard;
        probe_guard pg(&probe);
        try {
            if (pooled) {
                (void) llama_kv_cache_prepare_test_access::prepare_into(kv, ubatches, res);
            } else {
                (void) llama_kv_cache_prepare_test_access::prepare_reference(kv, ubatches);
            }
        } catch (const std::bad_alloc &) {
            out.threw = true;
        }
    }
    out.fired = probe.fired;
    out.seen_mask = probe.seen_mask;
    out.alloc_index = probe.alloc_index;
    return out;
}


// A failed shift_backup_save() must not commit a partially copied backup.
// The retry must start from a complete snapshot and restore it exactly.
static void test_shift_backup_strong_oom() {
    test_kv_fixture f(0, LLAMA_SWA_TYPE_NONE);
    llama_kv_cache * kv = make_test_cache(f, 0, LLAMA_SWA_TYPE_NONE, 8);
    seed_test_cache(*kv, 4, ~0u, ~0u, 4);

    auto & cells = llama_kv_cache_prepare_test_access::cells(*kv, 0);
    bool threw = false;
    {
        fault_guard guard;
        arm_fault_at(0, false);
        try {
            kv->shift_backup_save();
        } catch (const std::bad_alloc &) {
            threw = true;
        }
    }
    check(threw, "shift backup: save allocation failure propagated");

    cells.pos_set(0, 99);
    kv->shift_backup_restore();
    check(cells.pos_get(0) == 99, "shift backup: failed save leaves no partial backup");
    kv->shift_backup_clear();

    llama_kv_cells retry_before = cells;
    const auto heads_before = llama_kv_cache_prepare_test_access::heads(*kv);
    kv->shift_backup_save();
    const auto host = kv->host_memory_breakdown();
    check(host.snapshots > 0 && host.total() >= host.snapshots,
          "host breakdown counts the shift backup");
    cells.pos_set(0, 77);
    kv->shift_backup_restore();
    auto & restored = llama_kv_cache_prepare_test_access::cells(*kv, 0);
    check(&restored == &cells, "shift backup: restore preserves the cells object address");
    check(same_cells(restored, retry_before, restored.size()) &&
          llama_kv_cache_prepare_test_access::heads(*kv) == heads_before,
          "shift backup: retry saves and restores complete state");

    delete kv;
}

// clear() is a synchronous C API: a retained memory-context object is not
// native work and must not block a legal clear. The caller owns the ordering.
static void test_clear_with_context() {
    test_kv_fixture f(0, LLAMA_SWA_TYPE_NONE);
    llama_kv_cache * kv = make_test_cache(f, 0, LLAMA_SWA_TYPE_NONE, 8);
    seed_test_cache(*kv, 8, 7, ~0u, 8);

    {
        auto mctx = kv->init_full();
        check(bool(mctx), "clear context: dummy context created");
        check(llama_memory_clear(kv, false), "clear context: the dummy context can clear the cache");
        check(llama_kv_cache_prepare_test_access::cells(*kv, 0).get_used() == 0,
              "clear context: clear works with a dummy context");
    }

    delete kv;
}

// Three ubatches keep a second, non-last trial after the first stream's
// mutation, so the allocator probe can arm the fault after the mutation.
static void test_non_swa_persistent_oom() {
    test_kv_fixture f(0, LLAMA_SWA_TYPE_NONE);
    test_ubatch_multi a(0, 0);
    test_ubatch_multi b(1, 1);
    test_ubatch_multi c(2, 1);
    const std::vector<llama_ubatch> ubatches = { a.ub, b.ub, c.ub };

    for (int pooled = 0; pooled < 2; ++pooled) {
        llama_kv_cache * kv = make_test_cache(f, 0, LLAMA_SWA_TYPE_NONE, 8, false, 2);
        seed_stream(*kv, 0, 4, 4);
        seed_stream(*kv, 1, 4, 4);
        llama_kv_cache_prepare_test_access::cells(*kv, 0).ext_set(0, llama_kv_cell_ext{1, 2});
        llama_kv_cache_prepare_test_access::cells(*kv, 1).ext_set(1, llama_kv_cell_ext{3, 4});

        const llama_kv_cells init0 = llama_kv_cache_prepare_test_access::cells(*kv, 0);
        const llama_kv_cells init1 = llama_kv_cache_prepare_test_access::cells(*kv, 1);
        const uint32_t head0 = llama_kv_cache_prepare_test_access::heads(*kv)[0];
        const uint32_t head1 = llama_kv_cache_prepare_test_access::heads(*kv)[1];

        llama_kv_cache::slot_info_vec_t res;
        if (pooled) res.resize(ubatches.size());

        const auto r = run_mutation_fault(*kv, {init0, init1}, 1u, ubatches, pooled != 0, res);

        check(r.fired && r.alloc_index >= 0,
              pooled ? "non-swa persistent pooled: fault armed after a real mutation"
                     : "non-swa persistent reference: fault armed after a real mutation");
        check((r.seen_mask & 1u) != 0,
              pooled ? "non-swa persistent pooled: stream 0 mutated before the fault"
                     : "non-swa persistent reference: stream 0 mutated before the fault");
        check(r.threw, pooled ? "non-swa persistent pooled: prepare throws"
                              : "non-swa persistent reference: prepare throws");

        auto & c0 = llama_kv_cache_prepare_test_access::cells(*kv, 0);
        auto & c1 = llama_kv_cache_prepare_test_access::cells(*kv, 1);
        const auto & heads = llama_kv_cache_prepare_test_access::heads(*kv);
        check(same_cells(c0, init0, c0.size()) && same_cells(c1, init1, c1.size()) &&
              heads[0] == head0 && heads[1] == head1,
              pooled ? "non-swa persistent pooled: full initial state restored"
                     : "non-swa persistent reference: full initial state restored");
        delete kv;
    }
}


// The fourth ubatch gives an allocation point after both streams have been
// mutated, so the fault is armed from the observed cache state, not a fixed
// allocation index.
static void test_multi_swa_oom_rollback() {
    test_kv_fixture f(0, LLAMA_SWA_TYPE_STANDARD);
    test_ubatch a(0, 0);
    test_ubatch b(1, 0);
    test_ubatch_multi c(2, 3);
    test_ubatch d(4, 0);
    const std::vector<llama_ubatch> ubatches = { a.ub, b.ub, c.ub, d.ub };

    for (int pooled = 0; pooled < 2; ++pooled) {
        llama_kv_cache * kv = make_test_cache(f, 0, LLAMA_SWA_TYPE_STANDARD, 8, false, 2);
        seed_stream(*kv, 0, 4, 4);
        seed_stream(*kv, 1, 4, 4);

        const llama_kv_cells init0 = llama_kv_cache_prepare_test_access::cells(*kv, 0);
        const llama_kv_cells init1 = llama_kv_cache_prepare_test_access::cells(*kv, 1);
        const uint32_t head0 = llama_kv_cache_prepare_test_access::heads(*kv)[0];
        const uint32_t head1 = llama_kv_cache_prepare_test_access::heads(*kv)[1];

        llama_kv_cache::slot_info_vec_t res;
        if (pooled) res.resize(ubatches.size());

        const auto r = run_mutation_fault(*kv, {init0, init1}, 3u, ubatches, pooled != 0, res);

        check(r.fired && r.alloc_index >= 0,
              pooled ? "multi-swa persistent pooled: fault armed after both streams mutated"
                     : "multi-swa persistent reference: fault armed after both streams mutated");
        check((r.seen_mask & 3u) == 3u,
              pooled ? "multi-swa persistent pooled: both streams mutated before the fault"
                     : "multi-swa persistent reference: both streams mutated before the fault");
        check(r.threw, pooled ? "multi-swa persistent pooled: prepare throws"
                              : "multi-swa persistent reference: prepare throws");

        auto & c0 = llama_kv_cache_prepare_test_access::cells(*kv, 0);
        auto & c1 = llama_kv_cache_prepare_test_access::cells(*kv, 1);
        const auto & heads = llama_kv_cache_prepare_test_access::heads(*kv);
        check(same_cells(c0, init0, c0.size()) && same_cells(c1, init1, c1.size()) &&
              heads[0] == head0 && heads[1] == head1,
              pooled ? "multi-swa persistent pooled: both streams restored"
                     : "multi-swa persistent reference: both streams restored");
        delete kv;
    }
}

static void test_cell_journal_same_cell_two_seqs() {
    llama_kv_cells cells;
    cells.resize(4);
    cells.pos_set(0, 7);
    cells.seq_add(0, 0);
    cells.seq_add(0, 1);

    cell_journal journal;
    journal.begin(4);
    cells.set_journal(&journal);

    cells.seq_rm(0, 0); // non-last: extracts the seq 0 node
    cells.seq_rm(0, 1); // last: extracts the seq 1 and used nodes
    cells.set_journal(nullptr);

    check(cells.is_empty(0), "journal same-cell: empty after both seq_rm");

    journal.replay([&](cell_journal::entry & e) {
        check(cells.restore_from_journal(journal, e.i), "journal same-cell: restore");
    });

    check(!cells.is_empty(0), "journal same-cell: cell restored");
    check(cells.pos_get(0) == 7, "journal same-cell: pos restored");
    check(cells.seq_has(0, 0) && cells.seq_has(0, 1), "journal same-cell: both seqs restored");
    check(cells.get_used_set().count(0) == 1, "journal same-cell: used restored");
    check(cells.seq_pos_min(0) == 7 && cells.seq_pos_max(0) == 7, "journal same-cell: seq 0 pos restored");
    check(cells.seq_pos_min(1) == 7 && cells.seq_pos_max(1) == 7, "journal same-cell: seq 1 pos restored");
}

// A cell visible to more sequences than the inline journal slots must spill
// into the heap vector, and the replay must still move every node back.
static void test_cell_journal_wide_seq_mask() {
    llama_kv_cells cells;
    cells.resize(4);
    cells.pos_set(0, 7);
    for (int s = 0; s < 4; ++s) {
        cells.seq_add(0, s);
    }

    const llama_kv_cells init = cells;

    cell_journal journal;
    journal.begin(4);
    cells.set_journal(&journal);
    cells.rm(0);
    cells.set_journal(nullptr);

    check(cells.is_empty(0), "journal wide seq: cell empty after rm");

    journal.replay([&](cell_journal::entry & e) {
        check(cells.restore_from_journal(journal, e), "journal wide seq: restore");
    });

    check(same_cells(cells, init, cells.size()), "journal wide seq: full state restored");
}

// The dedup branch must reserve the node storage before extracting; an OOM
// there must not lose the already extracted seq_pos node.
static void test_cell_journal_same_cell_oom() {
    llama_kv_cells cells;
    cells.resize(4);
    cells.pos_set(0, 7);
    cells.seq_add(0, 0);
    cells.seq_add(0, 1);

    const llama_kv_cells init = cells;

    cell_journal journal;
    journal.begin(4);
    cells.set_journal(&journal);

    cells.seq_rm(0, 0); // non-last: the entry has seq 0

    bool threw = false;
    {
        fault_guard guard;
        arm_fault_at(0, false); // the reserve before the seq 1 extract
        try {
            cells.seq_rm(0, 1);
        } catch (const std::bad_alloc &) {
            threw = true;
        }
    }
    check(threw, "journal same-cell oom: reserve failure propagated");
    cells.set_journal(nullptr);

    journal.replay([&](cell_journal::entry & e) {
        check(cells.restore_from_journal(journal, e.i), "journal same-cell oom: restore");
    });

    check(cells.pos_get(0) == 7, "journal same-cell oom: pos restored");
    check(cells.seq_has(0, 0) && cells.seq_has(0, 1), "journal same-cell oom: both seqs restored");
    check(cells.get_seq_pos(0).at(7) == 1, "journal same-cell oom: seq 0 map restored");
    check(cells.get_seq_pos(1).at(7) == 1, "journal same-cell oom: seq 1 map restored");
    check(same_cells(cells, init, cells.size()), "journal same-cell oom: full initial state restored");
}

// A normal restore returns the journal's set/map nodes to the containers. A
// later prepare must be able to extract and own those same nodes again, and
// its exception rollback must return them without allocating. A third prepare
// then proves that the second rollback did not leak or double-own a node.
static void test_reown_after_normal_restore() {
    llama_kv_cells cells;
    cells.resize(4);
    cells.pos_set(0, 7);
    cells.seq_add(0, 0);
    cells.seq_add(0, 1);

    const llama_kv_cells init = cells;

    // prepare #1: the normal restore moves the seq 0 node back into the map
    {
        cell_journal journal;
        journal.begin(4);
        journal.begin_trial();
        cells.set_journal(&journal);

        cells.seq_rm(0, 0); // non-last: extracts the seq 0 node
        cells.set_journal(nullptr);

        check(cells.restore_from_journal(journal, 0), "reown: normal restore");
        check(same_cells(cells, init, cells.size()), "reown: normal restore returns the initial state");
        journal.clear();
    }

    // prepare #2: extract the returned node again, then replay the exception
    {
        cell_journal journal;
        journal.begin(4);
        journal.begin_trial();
        cells.set_journal(&journal);

        cells.seq_rm(0, 0);
        cells.seq_rm(0, 1); // last: extracts the seq 1 and used nodes
        cells.set_journal(nullptr);

        check(cells.is_empty(0), "reown: both seqs removed");

        bool threw = false;
        size_t restored = 0;
        {
            fault_guard guard;
            arm_fault_at(0, false); // the replay must not allocate
            try {
                journal.replay([&](cell_journal::entry & e) {
                    if (cells.restore_from_journal(journal, e.i)) {
                        ++restored;
                    }
                });
            } catch (const std::bad_alloc &) {
                threw = true;
            }
        }
        check(!threw, "reown: exception replay performs no allocation");
        check(restored == 1, "reown: exception replay restores the recorded cell");
        check(same_cells(cells, init, cells.size()), "reown: exception replay restores the full state");
    }

    // prepare #3: the nodes are back in the containers and can be owned again
    {
        cell_journal journal;
        journal.begin(4);
        journal.begin_trial();
        cells.set_journal(&journal);

        cells.seq_rm(0, 0);
        cells.seq_rm(0, 1);
        cells.set_journal(nullptr);

        journal.replay([&](cell_journal::entry & e) {
            check(cells.restore_from_journal(journal, e.i), "reown: second rollback restores the cell");
        });
        check(same_cells(cells, init, cells.size()), "reown: second rollback restores the full state");
    }
}

// Two streams can hold the same cell index and position. Each journal owns
// only its own stream's nodes; a replay of one stream must not take the
// other stream's node.
static void test_journal_stream_isolation() {
    llama_kv_cells c0, c1;
    c0.resize(4);
    c1.resize(4);
    c0.pos_set(0, 7);
    c0.seq_add(0, 0);
    c1.pos_set(0, 7);
    c1.seq_add(0, 0);

    const llama_kv_cells init0 = c0;
    const llama_kv_cells init1 = c1;

    cell_journal j0, j1;
    j0.begin(4, 0);
    j0.begin_trial();
    j1.begin(4, 1);
    j1.begin_trial();
    c0.set_journal(&j0);
    c1.set_journal(&j1);

    c0.seq_rm(0, 0); // last: extracts stream 0's used and seq 0 nodes
    c1.seq_rm(0, 0); // the same cell/position in stream 1
    c0.set_journal(nullptr);
    c1.set_journal(nullptr);

    check(c0.is_empty(0) && c1.is_empty(0), "stream isolation: both cells empty");

    j0.replay([&](cell_journal::entry & e) {
        check(c0.restore_from_journal(j0, e.i), "stream isolation: stream 0 restore");
    });
    check(same_cells(c0, init0, c0.size()), "stream isolation: stream 0 state restored");
    check(c1.is_empty(0), "stream isolation: stream 1 untouched by stream 0 replay");

    j1.replay([&](cell_journal::entry & e) {
        check(c1.restore_from_journal(j1, e.i), "stream isolation: stream 1 restore");
    });
    check(same_cells(c1, init1, c1.size()), "stream isolation: stream 1 state restored");
}

// A seq_pos node is keyed by (seq, position), not by the cell. Two cells can
// share the key; the node is extracted by the last removal and the first
// restore inserts it, the second increments the count. A non-empty snapshot
// must restore the real trial state without allocating.
static void test_journal_shared_seq_pos_snapshot() {
    llama_kv_cells cells;
    cells.resize(4);
    cells.pos_set(0, 7);
    cells.seq_add(0, 0);
    cells.pos_set(1, 7);
    cells.seq_add(1, 0); // the same seq/position, count 2

    const llama_kv_cells init = cells;

    cell_journal journal;
    journal.begin(4, 0);
    journal.begin_trial();
    cells.set_journal(&journal);

    cells.rm(0); // count 2 -> 1: no seq node extracted, used node extracted
    cells.rm(1); // count 1 -> 0: seq and used nodes extracted
    cells.set_journal(nullptr);

    check(cells.is_empty(0) && cells.is_empty(1), "shared seq_pos: both cells empty");

    bool ok0 = false;
    bool threw = false;
    {
        fault_guard guard;
        arm_fault_at(0, false);
        try {
            ok0 = cells.restore_cell_from_snapshot(journal, 0, init, 0);
        } catch (const std::bad_alloc &) {
            threw = true;
        }
    }
    check(ok0, "shared seq_pos: first snapshot restore");
    check(!threw, "shared seq_pos: first restore allocates nothing");
    check(cells.pos_get(0) == 7 && cells.seq_has(0, 0), "shared seq_pos: first cell restored");
    check(cells.get_seq_pos(0).at(7) == 1, "shared seq_pos: count 1 after first restore");

    bool ok1 = false;
    threw = false;
    {
        fault_guard guard;
        arm_fault_at(0, false);
        try {
            ok1 = cells.restore_cell_from_snapshot(journal, 1, init, 1);
        } catch (const std::bad_alloc &) {
            threw = true;
        }
    }
    check(ok1, "shared seq_pos: second snapshot restore");
    check(!threw, "shared seq_pos: second restore allocates nothing");
    check(same_cells(cells, init, cells.size()), "shared seq_pos: full state restored");
}

// begin_trial must keep the earlier trial's nodes. The replay of both trials
// restores the cell; the first trial's vector is still populated.
static void test_journal_two_trials_retained() {
    llama_kv_cells cells;
    cells.resize(4);
    cells.pos_set(0, 7);
    cells.seq_add(0, 0);
    cells.seq_add(0, 1);

    const llama_kv_cells init = cells;

    cell_journal journal;
    journal.begin(4, 0);
    journal.begin_trial();
    cells.set_journal(&journal);

    cells.seq_rm(0, 0); // trial 1: the global entry owns seq 0
    cells.set_journal(nullptr);

    journal.begin_trial();
    cells.set_journal(&journal);
    cells.seq_rm(0, 1); // trial 2: the trial entry owns seq 1 and used
    cells.set_journal(nullptr);

    check(journal.trial_count() == 2, "two trials: both trial lists exist");
    check(journal.size() >= 1, "two trials: first trial entry retained");
    check(!journal.trial_at(1).empty(), "two trials: second trial nodes retained");

    journal.replay([&](cell_journal::entry & e) {
        check(cells.restore_from_journal(journal, e.i), "two trials: restore");
    });
    check(same_cells(cells, init, cells.size()), "two trials: full state restored");
}

// The snapshot copy must keep a non-zero shift and the has_shift flag.
static void test_cp_shift() {
    llama_kv_cells cells;
    cells.resize(4);
    cells.pos_set(0, 7);
    cells.seq_add(0, 0);
    cells.pos_add(0, 3);

    const std::vector<uint32_t> idxs = { 0 };

    const llama_kv_cells copy = cells.cp(idxs);
    check(copy.get_shift(0) == 3, "cp: non-zero shift copied");

    llama_kv_cells into;
    cells.cp_into(idxs, into);
    check(into.get_shift(0) == 3, "cp_into: non-zero shift copied");

    llama_kv_cells dst;
    dst.resize(4);
    dst.set(idxs, copy);
    check(dst.get_shift(0) == 3, "set: non-zero shift copied");
}

// Two normal multi-stream non-SWA prepares on the same cache reuse the
// journal storage and leave the initial state both times, reference and
// pooled.
static void test_non_swa_reprepare() {
    test_kv_fixture f(0, LLAMA_SWA_TYPE_NONE);
    test_ubatch_multi a(0, 0);
    test_ubatch_multi b(1, 1);
    const std::vector<llama_ubatch> ubatches = { a.ub, b.ub };

    for (int pooled = 0; pooled < 2; ++pooled) {
        llama_kv_cache * kv = make_test_cache(f, 0, LLAMA_SWA_TYPE_NONE, 8, false, 2);
        seed_stream(*kv, 0, 4, 4);
        seed_stream(*kv, 1, 4, 4);
        llama_kv_cache_prepare_test_access::cells(*kv, 0).ext_set(0, llama_kv_cell_ext{1, 2});
        llama_kv_cache_prepare_test_access::cells(*kv, 1).ext_set(1, llama_kv_cell_ext{3, 4});

        const llama_kv_cells init0 = llama_kv_cache_prepare_test_access::cells(*kv, 0);
        const llama_kv_cells init1 = llama_kv_cache_prepare_test_access::cells(*kv, 1);
        const uint32_t head0 = llama_kv_cache_prepare_test_access::heads(*kv)[0];
        const uint32_t head1 = llama_kv_cache_prepare_test_access::heads(*kv)[1];

        for (int k = 0; k < 2; ++k) {
            const auto r = run_test_prepare(*kv, ubatches, pooled != 0);
            check(r.ok, pooled ? "non-swa reprepare pooled: prepare succeeds"
                               : "non-swa reprepare reference: prepare succeeds");
            check(same_cells(llama_kv_cache_prepare_test_access::cells(*kv, 0), init0, init0.size()) &&
                  same_cells(llama_kv_cache_prepare_test_access::cells(*kv, 1), init1, init1.size()) &&
                  llama_kv_cache_prepare_test_access::heads(*kv)[0] == head0 &&
                  llama_kv_cache_prepare_test_access::heads(*kv)[1] == head1,
                  pooled ? "non-swa reprepare pooled: initial state after the prepare"
                         : "non-swa reprepare reference: initial state after the prepare");
        }

        delete kv;
    }
}

// A ubatch that touches only stream 0 advances only stream 0's journal. A
// later ubatch touches both streams; its stream 1 trial index differs from
// stream 0's. The restore must use each stream's own index, not stream 0's.
static void test_prepare_stream_trial_index() {
    test_kv_fixture f(0, LLAMA_SWA_TYPE_STANDARD);

    test_ubatch a(0, 0); // stream 0 only
    test_ubatch b(1, 0); // stream 0 only
    test_ubatch_multi c(2, 3); // both streams
    const std::vector<llama_ubatch> ubatches = { a.ub, b.ub, c.ub };

    for (int pooled = 0; pooled < 2; ++pooled) {
        llama_kv_cache * kv = make_test_cache(f, 0, LLAMA_SWA_TYPE_STANDARD, 8,
                                              /*unified=*/false, /*n_seq_max=*/2);
        seed_stream(*kv, 0, 4, 4);
        seed_stream(*kv, 1, 4, 4);

        const llama_kv_cells init0 = llama_kv_cache_prepare_test_access::cells(*kv, 0);
        const llama_kv_cells init1 = llama_kv_cache_prepare_test_access::cells(*kv, 1);
        const uint32_t head0 = llama_kv_cache_prepare_test_access::heads(*kv)[0];
        const uint32_t head1 = llama_kv_cache_prepare_test_access::heads(*kv)[1];

        const auto r = run_test_prepare(*kv, ubatches, pooled != 0);

        check(r.ok, pooled ? "stream trial index pooled: prepare succeeds"
                           : "stream trial index reference: prepare succeeds");
        check(same_cells(llama_kv_cache_prepare_test_access::cells(*kv, 0), init0,
                         init0.size()) &&
              same_cells(llama_kv_cache_prepare_test_access::cells(*kv, 1), init1,
                         init1.size()) &&
              llama_kv_cache_prepare_test_access::heads(*kv)[0] == head0 &&
              llama_kv_cache_prepare_test_access::heads(*kv)[1] == head1,
              pooled ? "stream trial index pooled: both streams restored"
                     : "stream trial index reference: both streams restored");

        delete kv;
    }
}

// begin_trial's only allocation is the range vector growth. Force the growth
// deterministically (begin() reserves 8 ranges), arm the fault at the next
// allocation, and verify the previous ranges, trial entries and cells are
// unchanged and a retry succeeds.
static void test_journal_begin_trial_oom() {
    llama_kv_cells cells;
    cells.resize(4);
    cells.pos_set(0, 7);
    cells.seq_add(0, 0);

    cell_journal journal;
    journal.begin(4, 0);

    // trial 0 creates the global entry; trial 1 creates a trial entry
    journal.begin_trial();
    cells.set_journal(&journal);
    cells.seq_rm(0, 0);
    journal.begin_trial();
    cells.pos_set(0, 9);
    cells.seq_add(0, 0);
    cells.set_journal(nullptr);

    // fill the reserved range capacity; the next begin_trial must grow it
    for (int k = 2; k < 8; ++k) {
        journal.begin_trial();
    }

    const size_t ranges_before = journal.trial_count();
    const size_t entries_before = journal.size();
    const size_t trial1_before = journal.trial_at(1).count();
    const llama_kv_cells cells_before = cells;

    bool threw = false;
    {
        fault_guard guard;
        arm_fault_at(0, false); // the range vector growth
        try {
            journal.begin_trial();
        } catch (const std::bad_alloc &) {
            threw = true;
        }
    }

    check(threw, "begin_trial oom: the range growth allocation failed");
    check(journal.trial_count() == ranges_before, "begin_trial oom: ranges unchanged");
    check(journal.size() == entries_before, "begin_trial oom: entries unchanged");
    check(trial1_before > 0 && journal.trial_at(1).count() == trial1_before,
          "begin_trial oom: earlier trial entry retained");
    check(same_cells(cells, cells_before, cells.size()), "begin_trial oom: cells unchanged");

    // the journal still works after the caught fault
    journal.begin_trial();
    check(journal.trial_count() == ranges_before + 1, "begin_trial oom: retry adds a range");
}

static void test_cell_journal_purge_then_reuse() {
    llama_kv_cells cells;
    cells.resize(4);
    cells.pos_set(0, 3);
    cells.seq_add(0, 0);

    cell_journal journal;
    journal.begin(4);
    cells.set_journal(&journal);

    cells.seq_rm(0, 0); // SWA purge removes the old cell
    check(cells.is_empty(0), "journal purge: cell purged");

    const llama_kv_cells snapshot = cells; // per-trial snapshot: post-purge empty

    cells.pos_set(0, 9); // the trial reuses the cell
    cells.seq_add(0, 0);
    cells.set_journal(nullptr);

    check(cells.restore_cell_from_snapshot(journal, 0, snapshot, 0),
          "journal purge: normal restore leaves the purge");
    check(cells.is_empty(0), "journal purge: normal restore leaves the purge");
    check(cells.seq_pos_min(0) == -1, "journal purge: normal seq_pos stays empty");

    check(cells.restore_from_journal(journal, 0), "journal purge: exception restore");
    check(cells.pos_get(0) == 3 && cells.seq_has(0, 0), "journal purge: pre-purge state restored");
    check(cells.seq_pos_min(0) == 3 && cells.seq_pos_max(0) == 3, "journal purge: pre-purge seq_pos restored");
}

// Persistent OOM: the fault is armed after the cache has observed a real
// mutation, and every later allocation also fails. The restore itself must
// not allocate, so it still completes.
static void test_persistent_oom() {
    test_kv_fixture f(512, LLAMA_SWA_TYPE_STANDARD);
    test_ubatch a(900);
    test_ubatch b(1024);
    const std::vector<llama_ubatch> ubatches = { a.ub, b.ub };

    for (int pooled = 0; pooled < 2; ++pooled) {
        llama_kv_cache * kv = make_test_cache(f, 512, LLAMA_SWA_TYPE_STANDARD, 1024);
        seed_test_cache(*kv, 1024, 399, 900, 399);
        llama_kv_cache_prepare_test_access::cells(*kv, 0).pos_add(100, 5);

        const llama_kv_cells init = llama_kv_cache_prepare_test_access::cells(*kv, 0);
        const uint32_t init_head = llama_kv_cache_prepare_test_access::heads(*kv)[0];

        llama_kv_cache::slot_info_vec_t res;
        if (pooled) res.resize(ubatches.size());

        const auto r = run_mutation_fault(*kv, {init}, 1u, ubatches, pooled != 0, res);

        check(r.fired && r.alloc_index >= 0,
              pooled ? "persistent oom pooled: fault armed after a real mutation"
                     : "persistent oom reference: fault armed after a real mutation");
        check((r.seen_mask & 1u) != 0,
              pooled ? "persistent oom pooled: cell mutated before the fault"
                     : "persistent oom reference: cell mutated before the fault");
        check(r.threw, pooled ? "persistent oom pooled: first error propagated"
                              : "persistent oom reference: first error propagated");

        auto & cells = llama_kv_cache_prepare_test_access::cells(*kv, 0);
        check(same_cells(cells, init, cells.size()) &&
              llama_kv_cache_prepare_test_access::heads(*kv)[0] == init_head,
              pooled ? "persistent oom pooled: full initial state restored"
                     : "persistent oom reference: full initial state restored");
        delete kv;
    }
}

// The host account counts payload capacity, not the number of live elements.
// Every container below reserves more than its current size, so a size() vs
// capacity() mistake is visible.
static void test_host_breakdown_scratch_payload() {
    llama_kv_cache::decode_scratch scratch;

    scratch.sinfos.reserve(4);
    scratch.sinfos.resize(2);
    for (auto & slot : scratch.sinfos) {
        slot.strm.reserve(4);
        slot.strm.resize(2);
        slot.idxs.reserve(3);
        slot.idxs.resize(2);
        slot.idxs[0].reserve(16);
        slot.idxs[0].assign(8, 0);
        slot.idxs[1].reserve(8);
        slot.idxs[1].assign(4, 0);
    }

    scratch.ubatches.reserve(3);
    scratch.trial_idx.reserve(5);
    scratch.states.reserve(3);
    scratch.pooled.reserve(2);

    auto fallback = std::make_unique<llama_kv_cache::snapshot_slot>();
    fallback->sinfo.resize(1);
    fallback->sinfo.strm.reserve(2);
    fallback->sinfo.idxs.reserve(2);
    fallback->sinfo.idxs[0].reserve(12);
    fallback->sinfo.idxs[0].assign(6, 0);
    fallback->v_heads_old.reserve(3);
    fallback->v_cells.reserve(2);
    fallback->v_cells.resize(1);
    fallback->v_cells[0].resize(2);
    scratch.fallback.push_back(std::move(fallback));

    const auto b = scratch.host_memory_breakdown();

    check(b.trial_index == scratch.trial_idx.capacity()*sizeof(size_t),
          "host account: trial index uses the reserved capacity");

    size_t sinfo_payload = scratch.sinfos.capacity()*sizeof(llama_kv_cache::slot_info);
    for (const auto & slot : scratch.sinfos) {
        sinfo_payload += slot.strm.capacity()*sizeof(llama_seq_id) +
                         slot.idxs.capacity()*sizeof(llama_kv_cache::slot_info::idx_vec_t);
        for (const auto & idx : slot.idxs) {
            sinfo_payload += idx.capacity()*sizeof(uint32_t);
        }
    }
    check(b.scratch >= sinfo_payload + scratch.ubatches.capacity()*sizeof(llama_ubatch),
          "host account: scratch counts reserved slot-info capacity");

    const auto & slot = *scratch.fallback[0];
    const size_t snapshot_payload =
        slot.sinfo.strm.capacity()*sizeof(llama_seq_id) +
        slot.sinfo.idxs.capacity()*sizeof(llama_kv_cache::slot_info::idx_vec_t) +
        slot.sinfo.idxs[0].capacity()*sizeof(uint32_t) +
        slot.v_heads_old.capacity()*sizeof(uint32_t) +
        slot.v_cells.capacity()*sizeof(llama_kv_cells) +
        slot.v_cells[0].host_bytes();
    check(b.snapshots >= snapshot_payload,
          "host account: non-empty fallback slot is counted");
}

// A journal entry owns the set/map nodes it extracted from a live cell. Each
// case isolates one node kind or one owner, so a wrong kind or a wrong owner
// cannot hide behind the same total.
static void test_host_breakdown_journal_nodes() {
    // Used-node only.
    {
        llama_kv_cells cells;
        cells.resize(4);
        cells.pos_set(0, 7);

        cell_journal j;
        j.begin(4);
        cells.set_journal(&j);

        const size_t live0 = cells.host_bytes();
        cells.rm(0);
        const size_t live1 = cells.host_bytes();
        check(live0 - live1 == sizeof(uint32_t),
              "host account: live used-node payload");

        cells.set_journal(nullptr);
        const size_t owned = j.bytes();
        j.clear();
        check(owned - j.bytes() == live0 - live1,
              "host account: journal owns the used node");
    }

    // Seq-node only: a non-last seq_rm extracts one map node and keeps the
    // used node.
    {
        llama_kv_cells cells;
        cells.resize(4);
        cells.pos_set(0, 7);
        cells.seq_add(0, 1);
        cells.seq_add(0, 2);

        cell_journal j;
        j.begin(4);
        cells.set_journal(&j);

        const size_t live0 = cells.host_bytes();
        cells.seq_rm(0, 1);
        const size_t live1 = cells.host_bytes();
        check(live0 - live1 == sizeof(llama_pos) + sizeof(int),
              "host account: live seq-node payload");

        cells.set_journal(nullptr);
        const size_t owned = j.bytes();
        j.clear();
        check(owned - j.bytes() == live0 - live1,
              "host account: journal owns the seq node");
    }

    // Trial entry: the first mutation of a recorded cell owns both nodes; a
    // replay moves them back into the live cell.
    {
        llama_kv_cells cells;
        cells.resize(4);
        cells.pos_set(0, 7);
        cells.seq_add(0, 2);

        cell_journal j;
        j.begin(4);
        cells.set_journal(&j);
        cells.ext_set(0, llama_kv_cell_ext{});

        j.begin_trial();
        const size_t live0 = cells.host_bytes();
        cells.rm(0);
        const size_t live1 = cells.host_bytes();
        check(live0 - live1 == sizeof(uint32_t) + sizeof(llama_pos) + sizeof(int),
              "host account: live trial node payload");

        cells.set_journal(nullptr);
        const size_t owned = j.bytes();
        j.replay([&](cell_journal::entry & e) {
            check(cells.restore_from_journal(j, e),
                  "host account: journal node replay");
        });
        check(cells.host_bytes() == live0,
              "host account: replay restores the live node payload");
        check(owned - j.bytes() == live0 - live1,
              "host account: replay releases the trial node payload");
    }

    // Discard seq node: a seq added after the trial snapshot is not in the
    // trial entry, so its removal goes to discard_seq_.
    {
        llama_kv_cells cells;
        cells.resize(4);
        cells.pos_set(0, 7);
        cells.seq_add(0, 1);

        cell_journal j;
        j.begin(4);
        cells.set_journal(&j);
        cells.ext_set(0, llama_kv_cell_ext{});
        j.begin_trial();
        cells.seq_add(0, 2);

        const size_t live0 = cells.host_bytes();
        cells.seq_rm(0, 2);
        const size_t live1 = cells.host_bytes();
        check(live0 - live1 == sizeof(llama_pos) + sizeof(int),
              "host account: live discard-seq payload");

        cells.set_journal(nullptr);
        const size_t owned = j.bytes();
        j.clear();
        check(owned - j.bytes() == live0 - live1,
              "host account: discard owns the seq node");
    }

    // Discard used node: pos_set records the empty pre-state, so the trial
    // entry has pos == -1 and cannot own the used node extracted by rm.
    {
        llama_kv_cells cells;
        cells.resize(4);

        cell_journal j;
        j.begin(4);
        cells.set_journal(&j);
        cells.pos_set(0, 7);

        const size_t live0 = cells.host_bytes();
        cells.rm(0);
        const size_t live1 = cells.host_bytes();
        check(live0 - live1 == sizeof(uint32_t),
              "host account: live discard-used payload");

        cells.set_journal(nullptr);
        const size_t owned = j.bytes();
        j.clear();
        check(owned - j.bytes() == live0 - live1,
              "host account: discard owns the used node");
    }
}

// A real prepare fills a snapshot slot and a cache-owned scratch. This is the
// only test that runs the journal, snapshot and scratch owners together.
static void test_host_breakdown_real_prepare() {
    test_kv_fixture f(4, LLAMA_SWA_TYPE_STANDARD);
    llama_kv_cache * kv = make_test_cache(f, 4, LLAMA_SWA_TYPE_STANDARD, 8);
    seed_test_cache(*kv, 4, ~0u, ~0u, 4);

    test_ubatch a(4);
    const std::vector<llama_ubatch> ubatches = { a.ub };

    const auto before = kv->host_memory_breakdown();

    llama_kv_cache::decode_scratch * scratch =
        llama_kv_cache_prepare_test_access::scratch_acquire(*kv);
    check(scratch != nullptr, "host account: real prepare acquires a scratch");
    if (scratch == nullptr) {
        delete kv;
        return;
    }

    check(llama_kv_cache_prepare_test_access::prepare_into_scratch(*kv, ubatches, scratch),
          "host account: real prepare succeeds");

    const auto during = kv->host_memory_breakdown();

    size_t sinfo_payload = scratch->sinfos.capacity()*sizeof(llama_kv_cache::slot_info);
    for (const auto & slot : scratch->sinfos) {
        sinfo_payload += slot.strm.capacity()*sizeof(llama_seq_id) +
                         slot.idxs.capacity()*sizeof(llama_kv_cache::slot_info::idx_vec_t);
        for (const auto & idx : slot.idxs) {
            sinfo_payload += idx.capacity()*sizeof(uint32_t);
        }
    }
    check(during.scratch >= sinfo_payload,
          "host account: real prepare leaves slot-info payload in scratch");
    check(during.snapshots >= before.snapshots +
                              sizeof(llama_kv_cells) +
                              sizeof(llama_seq_id) +
                              sizeof(uint32_t),
          "host account: real prepare fills a snapshot slot");
    check(during.journal > before.journal,
          "host account: real prepare leaves journal capacity");

    llama_kv_cache_prepare_test_access::scratch_return(*kv, scratch);

    delete kv;
}

// heads_before is the prepare snapshot of v_heads, not decode scratch.
static void test_host_breakdown_prepare_heads() {
    test_kv_fixture f(0, LLAMA_SWA_TYPE_NONE);
    llama_kv_cache * kv = make_test_cache(f, 0, LLAMA_SWA_TYPE_NONE, 8);

    const auto before = kv->host_memory_breakdown();
    check(llama_kv_cache_prepare_test_access::journal_begin(*kv),
          "host account: prepare journal begins");
    const auto & heads = llama_kv_cache_prepare_test_access::heads_before(*kv);
    const auto during = kv->host_memory_breakdown();
    check(during.scratch == before.scratch,
          "host account: prepare heads are not scratch");
    check(during.snapshots == before.snapshots + heads.capacity()*sizeof(uint32_t),
          "host account: prepare heads are snapshot payload");
    llama_kv_cache_prepare_test_access::journal_end(*kv);

    const auto after = kv->host_memory_breakdown();
    check(after.scratch == before.scratch,
          "host account: journal end keeps prepare heads out of scratch");
    check(after.snapshots == during.snapshots,
          "host account: journal end keeps the head capacity in snapshots");

    delete kv;
}

int main() {
    test_reuse_and_lifetime();
    test_pool_outlives_owner();
    test_exhaustion_and_fallback();
    test_acquire_requires_shared_owner();
    test_split_scratch_reuse();
    test_steady_state();
    test_concurrency();
    test_decode_scratch_pool();
    test_decode_scratch_pool_concurrency();
    test_kv_snapshot_slot_reuse();
    test_kv_cells_cp_into();
    test_restore_cells_multi_stream();
    test_swa_last_trial_purge();
    test_non_swa_last_trial_skip();
    test_persistent_oom();
    test_non_swa_persistent_oom();
    test_multi_swa_oom_rollback();
    test_cell_journal_same_cell_two_seqs();
    test_cell_journal_wide_seq_mask();
    test_cell_journal_same_cell_oom();
    test_reown_after_normal_restore();
    test_journal_stream_isolation();
    test_journal_shared_seq_pos_snapshot();
    test_journal_two_trials_retained();
    test_journal_begin_trial_oom();
    test_cp_shift();
    test_non_swa_reprepare();
    test_prepare_stream_trial_index();
    test_cell_journal_purge_then_reuse();
    test_shift_backup_strong_oom();
    test_clear_with_context();
    test_host_breakdown_scratch_payload();
    test_host_breakdown_journal_nodes();
    test_host_breakdown_real_prepare();
    test_host_breakdown_prepare_heads();

    std::printf("%s failures=%d allocs=%lld frees=%lld\n",
                g_failures ? "FAILED" : "OK", g_failures,
                g_allocs.load(), g_frees.load());

    return g_failures ? 1 : 0;
}
