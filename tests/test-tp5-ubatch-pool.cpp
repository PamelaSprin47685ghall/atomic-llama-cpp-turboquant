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
        test_kv_fixture & f, uint32_t n_swa, llama_swa_type swa_type, uint32_t kv_size) {
    return new llama_kv_cache(
            *f.model, f.model->hparams,
            GGML_TYPE_F32, GGML_TYPE_F32,
            /*v_trans=*/false,
            /*offload=*/false,
            /*unified=*/true,
            kv_size,
            /*n_seq_max=*/1,
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

struct test_ubatch {
    llama_token  token     = 1;
    llama_pos    pos;
    int32_t      n_seq_id  = 1;
    llama_seq_id seq       = 0;
    llama_seq_id * seq_ptr = &seq;
    int8_t       output    = 0;
    int32_t      seq_idx[LLAMA_MAX_SEQ];
    llama_ubatch ub        = {};

    explicit test_ubatch(llama_pos p) : pos(p) {
        for (int i = 0; i < LLAMA_MAX_SEQ; ++i) {
            seq_idx[i] = -1;
        }
        seq_idx[0] = 0;

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
    test_restore_cells_multi_stream();
    test_swa_last_trial_purge();
    test_non_swa_last_trial_skip();

    std::printf("%s failures=%d allocs=%lld frees=%lld\n",
                g_failures ? "FAILED" : "OK", g_failures,
                g_allocs.load(), g_frees.load());

    return g_failures ? 1 : 0;
}
