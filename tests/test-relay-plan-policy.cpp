// TP5 pillar-5 CPU policy tests. No HIP, no model, no GPU.
//
// These pin the decisions relay.hip and the compose runtime share:
//   * K=8 plan cache: hit/evict/pin/flush and channels-in-key,
//   * frame LRU: the five-shape working set needs capacity 8,
//   * exact pool layout: the T4 table fits 68 MiB, every control block stays
//     64-byte aligned, and the 65536 bound does not fit,
//   * per-window channel slots: binding, retirement, dual-window isolation,
//   * the switch matrix: every switch defaults to the legacy behavior.
//
// The same header is compiled into relay.hip and the runtime patch, so a
// failure here is a policy failure, not a test-fixture failure.

#include "ggml-relay-plan.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

int failures = 0;

bool expect(bool condition, const char * expression, int line) {
    if (!condition) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, expression);
        ++failures;
    }
    return condition;
}

#define CHECK(expr) expect((expr), #expr, __LINE__)

using ggml_relay_plan::config;
using ggml_relay_plan::frame_lru;
using ggml_relay_plan::plan_cache;
using ggml_relay_plan::plan_key;
using ggml_relay_plan::plan_slot;

plan_key make_key(uint32_t n_tokens, uint32_t n_outputs, uint32_t channels = 181) {
    plan_key key;
    key.model       = 7;
    key.window      = 0;
    key.n_tokens    = n_tokens;
    key.n_outputs   = n_outputs;
    key.rs_class    = 3;
    key.n_kv_bucket = 1024;
    key.alloc_gen   = 1;
    key.channels    = channels;
    key.binding     = 1;
    return key;
}

void test_plan_cache() {
    plan_cache cache;
    std::vector<plan_key> keys;
    for (uint32_t i = 0; i < plan_cache::capacity; ++i) {
        keys.push_back(make_key(1 + i, 2 + i));
        auto * slot = cache.acquire(keys.back(), i + 1);
        CHECK(slot != nullptr);
        CHECK(slot->generation == i + 1);
        CHECK(slot->pins == 0);
        CHECK(slot->channels == 181);
    }
    CHECK(cache.live() == plan_cache::capacity);
    CHECK(cache.stats().inserts == plan_cache::capacity);
    CHECK(cache.stats().evictions == 0);

    // Every key is a hit and refreshes the LRU position.
    for (uint32_t i = 0; i < plan_cache::capacity; ++i) {
        auto * slot = cache.acquire(keys[i], 100 + i);
        CHECK(slot != nullptr);
        CHECK(slot->generation == i + 1);
    }
    CHECK(cache.stats().hits == plan_cache::capacity);

    // Pin slot 2 and make it the oldest: it must survive the next insert.
    CHECK(cache.pin(2));
    CHECK(cache.pinned() == 1);
    for (uint32_t i = 0; i < plan_cache::capacity; ++i) {
        cache.acquire(keys[i], 200 + i); // slot 2 touched last -> oldest becomes slot 1
    }
    uint64_t evicted = 0;
    auto * fresh = cache.acquire(make_key(9, 9), 300, &evicted);
    CHECK(fresh != nullptr);
    CHECK(evicted == 1); // slot 1 was the oldest unpinned slot
    CHECK(cache.stats().evictions == 1);
    CHECK(cache.stats().stale_generation == 1);
    CHECK(cache.find(keys[0]) == nullptr); // evicted key is gone
    CHECK(cache.find(keys[1]) != nullptr); // pinned key survived

    // A pinned slot cannot be flushed.
    CHECK(!cache.flush(2));
    CHECK(cache.stats().pinned_refusals == 1);
    CHECK(cache.release(2));
    CHECK(cache.flush(2));
    CHECK(cache.stats().flushes == 1);

    // Channels are part of the key: same seven fields, different count.
    plan_key narrow = make_key(1, 2, 180);
    plan_key wide   = make_key(1, 2, 181);
    CHECK(narrow != wide);
    plan_cache split;
    CHECK(split.acquire(narrow, 1) != nullptr);
    CHECK(split.acquire(wide, 1) != nullptr);
    CHECK(split.live() == 2);

    // The terminal-binding digest is the second half of the identity. The
    // 512-row catch-up rebuilt the graph with the same coarse llama fields but
    // different terminal storage; it must be a miss, not a key collision.
    plan_key base_512 = make_key(12, 1);
    plan_key catch_up = base_512;
    catch_up.binding  = 0x9e3779b97f4a7c15ull; // different terminal digest
    CHECK(base_512 != catch_up);
    plan_cache rows;
    const plan_slot * first_512 = rows.acquire(base_512, 1);
    CHECK(first_512 != nullptr);
    const plan_slot * second_512 = rows.acquire(catch_up, 2);
    CHECK(second_512 != nullptr);
    CHECK(second_512->generation != first_512->generation);
    CHECK(rows.live() == 2);
    CHECK(rows.acquire(base_512, 3) != nullptr); // old plan still its own slot
    CHECK(rows.find(catch_up) != nullptr);
    // A true conflict is only a binding-digest collision; the cache itself
    // never merges two different keys.
    CHECK(rows.find(base_512)->generation == first_512->generation);
    CHECK(rows.find(catch_up)->generation == second_512->generation);

    // 512-row tail switch: plans A (180 channels) and B (181) alternate. A
    // key hit must return A's own slot even when the provider's current
    // channels describe B; the cache never consults a separate "current
    // binding" status. A different binding is a new plan, not a conflict.
    plan_cache tail;
    plan_key plan_a = make_key(12, 1, 180);
    plan_a.binding  = 0x1111;
    plan_key plan_b = make_key(12, 1, 181);
    plan_b.binding  = 0x2222;
    const plan_slot * slot_a = tail.acquire(plan_a, 1);
    const plan_slot * slot_b = tail.acquire(plan_b, 2);
    CHECK(slot_a != nullptr && slot_b != nullptr);
    CHECK(slot_a != slot_b);
    const plan_slot * again_a = tail.acquire(plan_a, 3);
    CHECK(again_a == slot_a); // same slot, same generation
    CHECK(again_a->generation == slot_a->generation);
    CHECK(tail.live() == 2);
    plan_key plan_a_other = plan_a;
    plan_a_other.binding  = 0x3333;
    const plan_slot * other = tail.acquire(plan_a_other, 4);
    CHECK(other != slot_a); // different binding -> its own plan
    CHECK(tail.find(plan_a)->generation == slot_a->generation); // old plan intact

    // Every slot pinned fails closed instead of dropping a plan.
    plan_cache pinned;
    for (uint32_t i = 0; i < plan_cache::capacity; ++i) {
        auto * slot = pinned.acquire(make_key(i + 1, 1), i + 1);
        CHECK(slot != nullptr);
        CHECK(pinned.pin(slot->generation));
    }
    uint64_t none = 12345;
    CHECK(pinned.acquire(make_key(99, 1), 99, &none) == nullptr);
    CHECK(none == 0);
    CHECK(pinned.stats().pinned_refusals == 1);
    CHECK(!pinned.flush_all());
    CHECK(pinned.live() == plan_cache::capacity);
    for (uint32_t i = 1; i <= plan_cache::capacity; ++i) {
        CHECK(pinned.release(i));
    }
    CHECK(pinned.flush_all());
    CHECK(pinned.live() == 0);

    // Generations are monotonic and never reused.
    plan_cache reuse;
    auto * a = reuse.acquire(make_key(1, 1), 1);
    CHECK(a != nullptr && a->generation == 1);
    CHECK(reuse.flush(1));
    auto * b = reuse.acquire(make_key(1, 1), 2);
    CHECK(b != nullptr && b->generation == 2);
}

// Exact LRU replay of the five-shape working set named by the M4 report.
size_t replay(const std::vector<int> & order, size_t capacity) {
    static const char * names[] = { "N2-out", "N1-out", "N3-out", "N1-noout", "N4-noout" };
    (void) names;
    std::array<uint64_t, 5> uid{};
    uint64_t next_uid = 1;
    frame_lru lru(capacity);
    size_t misses = 0;
    for (size_t step = 0; step < order.size(); ++step) {
        const int shape = order[step];
        int index = lru.find(uid[static_cast<size_t>(shape)], 1);
        if (index < 0) {
            ++misses;
            index = lru.victim(-1);
            CHECK(index >= 0);
            uid[static_cast<size_t>(shape)] = next_uid++;
            lru.assign(index, uid[static_cast<size_t>(shape)], 1, step);
        } else {
            lru.touch(index, step);
        }
    }
    return misses;
}

void test_frame_lru() {
    const std::vector<int> cycle = { 0, 1, 2, 3, 4 };
    std::vector<int> order;
    for (int round = 0; round < 20; ++round) {
        order.insert(order.end(), cycle.begin(), cycle.end());
    }
    CHECK(replay(order, 8) == 5); // cold start only
    CHECK(replay(order, 4) == order.size()); // reuse distance 5 > capacity 4
    CHECK(replay(order, 2) >= replay(order, 4));

    // The worst-case revisit: a shape revisited after four distinct others
    // always misses at capacity 4 (the design's 9/12 argument).
    const std::vector<int> worst = { 0, 1, 2, 3, 4, 2, 0, 1, 2, 3, 4, 2 };
    CHECK(replay(worst, 4) == 9);
    CHECK(replay(worst, 8) == 5);

    // uid fast path requires both uid and plan generation.
    frame_lru lru(2);
    CHECK(lru.find(0, 1) == -1); // uid 0 is never a fast-path candidate
    lru.assign(0, 42, 7, 1);
    CHECK(lru.find(42, 7) == 0);
    CHECK(lru.find(42, 8) == -1);
    CHECK(lru.find(43, 7) == -1);

    // victim skips the current frame and prefers an unused slot.
    frame_lru pick(2);
    CHECK(pick.victim(-1) == 0);
    pick.assign(0, 1, 1, 10);
    CHECK(pick.victim(-1) == 1);
    pick.assign(1, 2, 1, 20);
    CHECK(pick.victim(1) == 0);
    CHECK(pick.victim(0) == 1);
    CHECK(pick.victim(-1) == 0); // oldest wins

    // Capacity is clamped to the eight-frame bound.
    frame_lru big(64);
    CHECK(big.capacity() == ggml_relay_plan::frame_capacity_max);
}

void test_pool_layout() {
    // T4 channel table: 36 recurrent-output, 48 expert-output, 97 HC joins.
    std::vector<size_t> t4;
    for (int i = 0; i < 36; ++i) t4.push_back(40960);
    for (int i = 0; i < 48; ++i) t4.push_back(10240);
    for (int i = 0; i < 97; ++i) t4.push_back(10240);
    CHECK(t4.size() == 181);

    ggml_relay_plan::pool_layout layout;
    CHECK(ggml_relay_plan::compute_pool_layout(t4.data(), t4.size(), 5,
                                               ggml_relay_plan::pool_budget_default, layout));
    CHECK(layout.bytes == 71209984);
    CHECK(layout.bytes <= ggml_relay_plan::pool_budget_default);
    CHECK(layout.count == 181);
    CHECK(layout.offset[0] == 0);
    CHECK(layout.capacity[0] == 40960);
    CHECK(layout.capacity[36] == 10240);
    CHECK(layout.offset[1] == 1024 + 40960 * 4 * 6);

    // Every control block is a 64-byte-aligned object whose words are 64
    // bytes apart. A channel whose payload is not a multiple of 64 bytes must
    // not misalign the next control block. T4 is already aligned; odd element
    // counts exercise the padding.
    std::vector<size_t> odd = { 1, 3, 5, 7, 9, 17, 65521 };
    ggml_relay_plan::pool_layout odd_layout;
    CHECK(ggml_relay_plan::compute_pool_layout(odd.data(), odd.size(), 5,
                                               ggml_relay_plan::pool_budget_default, odd_layout));
    std::size_t odd_need = 0;
    CHECK(ggml_relay_plan::pool_need(odd.data(), odd.size(), 5, odd_need));
    CHECK(odd_need == odd_layout.bytes);
    for (std::size_t i = 0; i < odd.size(); ++i) {
        CHECK(odd_layout.offset[i] % ggml_relay_plan::pool_alignment == 0);
        CHECK(odd_layout.capacity[i] == odd[i]);
        if (i + 1 < odd.size()) {
            const std::size_t payload_end = odd_layout.offset[i] +
                ggml_relay_plan::control_bytes + odd_layout.capacity[i] * 4 * 6;
            CHECK(odd_layout.offset[i + 1] >= payload_end);
            CHECK(odd_layout.offset[i + 1] - payload_end < ggml_relay_plan::pool_alignment);
        }
    }
    // Padding is between slots, not extra capacity: one more element still
    // needs a new layout. The final slot's padding is part of the allocation.
    std::vector<size_t> odd_more = odd;
    ++odd_more.back();
    CHECK(!ggml_relay_plan::layout_covers(odd_layout, odd_more.data(), odd_more.size()));
    CHECK(odd_layout.bytes % ggml_relay_plan::pool_alignment == 0);
    // An impossible rank count is rejected before the stride can overflow.
    CHECK(!ggml_relay_plan::pool_need(odd.data(), odd.size(), SIZE_MAX, odd_need));

    // The old 65536 bound cannot fit the budget; the exact layout must fail
    // closed rather than silently overrun.
    std::vector<size_t> bound(181, 65536);
    std::size_t need = 0;
    CHECK(ggml_relay_plan::pool_need(bound.data(), bound.size(), 5, need));
    CHECK(need > ggml_relay_plan::pool_budget_default);
    ggml_relay_plan::pool_layout over;
    CHECK(!ggml_relay_plan::compute_pool_layout(bound.data(), bound.size(), 5,
                                                ggml_relay_plan::pool_budget_default, over));
    CHECK(over.bytes == 0);

    std::size_t growth_need = 0;
    CHECK(ggml_relay_plan::pool_growth_decision(layout, bound.data(), bound.size(), 5,
                                                ggml_relay_plan::pool_budget_default, growth_need) ==
          ggml_relay_plan::pool_growth::fail_budget);

    // A smaller plan fits the T4 layout unchanged.
    std::vector<size_t> small = t4;
    for (auto & value : small) value /= 2;
    CHECK(ggml_relay_plan::layout_covers(layout, small.data(), small.size()));
    CHECK(ggml_relay_plan::pool_growth_decision(layout, small.data(), small.size(), 5,
                                                ggml_relay_plan::pool_budget_default, growth_need) ==
          ggml_relay_plan::pool_growth::keep);

    // A larger channel count needs a new layout but stays inside the budget
    // only if the elements fit.
    std::vector<size_t> more = t4;
    more.push_back(1024);
    CHECK(!ggml_relay_plan::layout_covers(layout, more.data(), more.size()));
    CHECK(ggml_relay_plan::pool_growth_decision(layout, more.data(), more.size(), 5,
                                                ggml_relay_plan::pool_budget_default, growth_need) ==
          ggml_relay_plan::pool_growth::grow);

    // Legacy pool accounting is the fixed 384.25 MiB layout.
    CHECK(ggml_relay_plan::pool_bytes_legacy == 402915328);
    CHECK(ggml_relay_plan::pool_bytes_legacy ==
          256 * (ggml_relay_plan::control_bytes + 65536 * 4 * 6));
    CHECK(ggml_relay_plan::pool_baseline == (8ull << 20));
}

struct dummy_channel {
    bool held = false;
    int  id   = -1;
};

using slot_table = ggml_relay_plan::channel_slots<dummy_channel>;

bool slot_idle(const slot_table::slot & slot) {
    for (const auto & owner : slot.states) {
        if (owner && owner->held) {
            return false;
        }
    }
    return true;
}

void test_channel_slots() {
    // Bound: eight plan slots x 181 live channels = 1448 per window, 2896
    // across the two windows. The table reserves 256 states per slot.
    CHECK(slot_table::slot_capacity * 181 == 1448);
    CHECK(2 * slot_table::slot_capacity * 181 == 2896);
    CHECK(slot_table::channel_capacity_ == 256);

    slot_table table;
    CHECK(table.empty());

    // Bind eight generations; each owns one live channel state.
    for (uint64_t generation = 11; generation < 11 + 8; ++generation) {
        auto * slot = table.acquire(generation, generation, slot_idle);
        CHECK(slot != nullptr);
        CHECK(slot->generation == generation);
        slot->states[0] = std::make_unique<dummy_channel>();
        slot->states[0]->id = static_cast<int>(generation);
    }
    CHECK(table.live() == 8);
    CHECK(table.live_channels() == 8);

    // Rebinding the same generation returns the same slot.
    auto * same = table.acquire(12, 99, slot_idle);
    CHECK(same != nullptr && same->generation == 12);
    CHECK(same->states[0]->id == 12);

    // A ninth generation evicts the least recently used idle slot.
    auto * ninth = table.acquire(99, 100, slot_idle);
    CHECK(ninth != nullptr);
    CHECK(ninth->generation == 99);
    CHECK(table.live() == 8);
    CHECK(table.find(11) == nullptr);
    CHECK(table.find(12) != nullptr);

    // A held slot is not a victim. Hold every slot: acquire fails closed.
    for (size_t i = 0; i < slot_table::slot_capacity; ++i) {
        table.at(i).states[1] = std::make_unique<dummy_channel>();
        table.at(i).states[1]->held = true;
    }
    CHECK(table.acquire(123, 200, slot_idle) == nullptr);

    // Release one slot and prove it is now the only candidate.
    table.at(0).states[1]->held = false;
    auto * rebound = table.acquire(124, 201, slot_idle);
    CHECK(rebound == &table.at(0));
    CHECK(rebound->generation == 124);
    CHECK(rebound->states[0] == nullptr); // eviction cleared the old owner
    CHECK(table.live() == 8);

    // Retirement is by generation and clears the whole slot.
    CHECK(!table.retire(7777));
    CHECK(table.retire(124));
    CHECK(table.live() == 7);
    CHECK(table.find(124) == nullptr);

    // Dual-window isolation: the same generation in two tables owns
    // independent channel states.
    slot_table left;
    slot_table right;
    auto * l = left.acquire(5, 1, slot_idle);
    auto * r = right.acquire(5, 1, slot_idle);
    CHECK(l != nullptr && r != nullptr);
    CHECK(l != r);
    l->states[7] = std::make_unique<dummy_channel>();
    l->states[7]->id = 1;
    CHECK(r->states[7] == nullptr);
    CHECK(left.live_channels() == 1);
    CHECK(right.live_channels() == 0);

    // Non-monotonic plan generations are consumer-visible after a tail
    // switch: generation 2 must return to its own slot after generation 5
    // was used, with its state preserved and not shared with 5.
    slot_table tail;
    auto * gen2 = tail.acquire(2, 10, slot_idle);
    CHECK(gen2 != nullptr);
    gen2->states[3] = std::make_unique<dummy_channel>();
    gen2->states[3]->id = 2;
    auto * gen5 = tail.acquire(5, 11, slot_idle);
    CHECK(gen5 != nullptr);
    gen5->states[3] = std::make_unique<dummy_channel>();
    gen5->states[3]->id = 5;
    auto * gen2_again = tail.acquire(2, 12, slot_idle);
    CHECK(gen2_again == gen2);
    CHECK(gen2_again->states[3] != nullptr && gen2_again->states[3]->id == 2);
    CHECK(tail.find(5)->states[3]->id == 5);
    CHECK(tail.live() == 2);

    table.clear();
    CHECK(table.empty());
    CHECK(table.live_channels() == 0);
}

void test_binding_snapshot() {
    using ggml_relay_plan::binding_matches;
    using ggml_relay_plan::binding_set;
    using ggml_relay_plan::snapshot_binding;

    ggml_tensor t0 = {};
    ggml_tensor t1 = {};
    t0.data = &t0;
    t1.data = &t1;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        t0.ne[d] = 10 + d;
        t1.ne[d] = 20 + d;
        t0.nb[d] = 100 + d;
        t1.nb[d] = 200 + d;
    }
    std::array<const ggml_tensor *, GGML_RELAY_RANKS * 2> terms{};
    for (std::size_t r = 0; r < GGML_RELAY_RANKS; ++r) {
        terms[r]                    = &t0;
        terms[GGML_RELAY_RANKS + r] = &t1;
    }

    auto set = std::make_unique<binding_set>();
    snapshot_binding(*set, terms.data(), 2);
    CHECK(binding_matches(*set, terms.data(), 2));

    // The digest is only a filter. Each real binding fact must be compared:
    // a digest collision that reused a different binding is caught here and
    // fails closed.
    void * saved_data = t1.data;
    t1.data = &t0;
    CHECK(!binding_matches(*set, terms.data(), 2));
    t1.data = saved_data;

    t1.ne[2] += 1;
    CHECK(!binding_matches(*set, terms.data(), 2));
    t1.ne[2] -= 1;

    t1.nb[3] += 8;
    CHECK(!binding_matches(*set, terms.data(), 2));
    t1.nb[3] -= 8;

    terms[GGML_RELAY_RANKS] = &t0; // terminal pointer changed
    CHECK(!binding_matches(*set, terms.data(), 2));
    terms[GGML_RELAY_RANKS] = &t1;
    CHECK(binding_matches(*set, terms.data(), 2));

    // Same key (including the same digest) but a different real binding is
    // exactly the fail-closed case: the cache finds the slot, the full
    // snapshot comparison rejects it.
    plan_cache collided;
    plan_key key = make_key(12, 1);
    key.binding  = 0x5a5a5a5a;
    const plan_slot * slot = collided.acquire(key, 1);
    CHECK(slot != nullptr);
    auto stored = std::make_unique<binding_set>();
    snapshot_binding(*stored, terms.data(), 2);
    void * saved2 = t1.data;
    t1.data = &t0;
    CHECK(collided.find(key) != nullptr);              // digest alone still matches
    CHECK(!binding_matches(*stored, terms.data(), 2)); // full check rejects
    t1.data = saved2;
    CHECK(binding_matches(*stored, terms.data(), 2));
}

void test_config() {
    config cfg;
    char error[160] = {};
    CHECK(ggml_relay_plan::parse_config(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        cfg, error, sizeof(error)));
    // Every switch defaults to the legacy behavior.
    CHECK(!cfg.plan_union);
    CHECK(cfg.frames == 2);
    CHECK(!cfg.pool_exact);
    CHECK(!cfg.slot_channels);
    CHECK(!cfg.verify_uid);
    CHECK(!cfg.plan_stats);
    CHECK(!cfg.graph_uid); // OFF == stock: every graph uid stays zero
    CHECK(!cfg.plan_flush);

    CHECK(ggml_relay_plan::parse_config("1", "8", "1", "1", "1", "1", "0", "1",
                                        cfg, error, sizeof(error)));
    CHECK(cfg.plan_union);
    CHECK(cfg.frames == 8);
    CHECK(cfg.pool_exact);
    CHECK(cfg.slot_channels);
    CHECK(cfg.verify_uid);
    CHECK(cfg.plan_stats);
    CHECK(!cfg.graph_uid);
    CHECK(cfg.plan_flush);

    // The uid assignment is an explicit ON-leg switch, never a default.
    CHECK(ggml_relay_plan::parse_config(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "1", nullptr,
                                        cfg, error, sizeof(error)));
    CHECK(cfg.graph_uid);
    CHECK(ggml_relay_plan::parse_config(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "0", nullptr,
                                        cfg, error, sizeof(error)));
    CHECK(!cfg.graph_uid);

    // FRAMES=0 is the legacy two-frame tier; 4 is the structural tier.
    CHECK(ggml_relay_plan::parse_config(nullptr, "0", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        cfg, error, sizeof(error)));
    CHECK(cfg.frames == 2);
    CHECK(ggml_relay_plan::parse_config(nullptr, "4", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        cfg, error, sizeof(error)));
    CHECK(cfg.frames == 4);

    // Slot channels follow the plan switch unless overridden.
    CHECK(ggml_relay_plan::parse_config("1", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        cfg, error, sizeof(error)));
    CHECK(cfg.plan_union && cfg.slot_channels);
    CHECK(ggml_relay_plan::parse_config("1", nullptr, nullptr, "0", nullptr, nullptr, nullptr, nullptr,
                                        cfg, error, sizeof(error)));
    CHECK(cfg.plan_union && !cfg.slot_channels);

    // Illegal values fail closed with a message, not a clamp.
    CHECK(!ggml_relay_plan::parse_config(nullptr, "9", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                         cfg, error, sizeof(error)));
    CHECK(error[0] != '\0');
    CHECK(!ggml_relay_plan::parse_config("2", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                         cfg, error, sizeof(error)));
    CHECK(!ggml_relay_plan::parse_config(nullptr, "abc", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                         cfg, error, sizeof(error)));
    CHECK(!ggml_relay_plan::parse_config(nullptr, nullptr, "yes", nullptr, nullptr, nullptr, nullptr, nullptr,
                                         cfg, error, sizeof(error)));
}

} // namespace

int main() {
    test_plan_cache();
    test_frame_lru();
    test_pool_layout();
    test_channel_slots();
    test_binding_snapshot();
    test_config();
    if (failures != 0) {
        std::fprintf(stderr, "RELAY_PLAN_POLICY_TEST FAIL %d\n", failures);
        return 1;
    }
    std::printf("RELAY_PLAN_POLICY_TEST PASS\n");
    return 0;
}
