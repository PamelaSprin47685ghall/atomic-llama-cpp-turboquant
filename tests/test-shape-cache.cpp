// TP5 llama-side per-shape plan cache: pure CPU policy test.
//
// The context uses this header to select a (llm_graph_result, scheduler) pair
// by shape. These checks pin the decisions that selection depends on:
//   * the same shape hits the same slot on the second epoch,
//   * a different alloc_gen never reuses a slot from an older placement,
//   * flush drops every slot,
//   * LRU eviction is deterministic and reports the victim,
//   * rekey after a can_reuse failure keeps the slot and its new identity,
//   * two windows (two cache instances) never share a slot,
//   * the flag value is the capacity; 0 keeps the legacy path.

#include "llama-shape-cache.h"

#include <cstdint>
#include <cstdio>

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

using llama_shape_cache::cache;
using llama_shape_cache::key;

key make_key(uint32_t n_tokens, uint32_t n_outputs = 2, uint64_t alloc_gen = 1) {
    key k;
    k.n_tokens     = n_tokens;
    k.n_seq_tokens = n_tokens;
    k.n_outputs    = n_outputs;
    k.n_seqs       = 1;
    k.n_seqs_unq   = 1;
    k.equal_seqs   = 0;
    k.n_kv_target  = 1024;
    k.gtype        = 0;
    k.alloc_gen    = alloc_gen;
    return k;
}

void test_same_shape_hit() {
    cache c(4);
    const auto first = c.acquire(make_key(1), 1);
    CHECK(first.index == 0);
    CHECK(!first.hit);
    CHECK(first.evicted == -1);
    CHECK(c.live() == 1);

    const auto second = c.acquire(make_key(1), 2);
    CHECK(second.hit);
    CHECK(second.index == first.index);
    CHECK(c.get_stats().hits == 1);
    CHECK(c.get_stats().misses == 1);
    CHECK(c.get_stats().inserts == 1);
    CHECK(c.live() == 1);

    // Every field is part of the identity.
    const key base = make_key(1);
    CHECK(c.acquire(base, 3).hit);
    key other = base;
    other.n_outputs = 3;
    CHECK(!c.acquire(other, 4).hit);
    other = base;
    other.n_kv_target = 2048;
    CHECK(!c.acquire(other, 5).hit);
    other = base;
    other.gtype = 1;
    CHECK(!c.acquire(other, 6).hit);
    other = base;
    other.n_seqs_unq = 2;
    CHECK(!c.acquire(other, 7).hit);
}

void test_alloc_gen() {
    cache c(4);
    CHECK(!c.acquire(make_key(1, 2, 1), 1).hit);
    // A new placement generation must not reuse the old slot.
    const auto next = c.acquire(make_key(1, 2, 2), 2);
    CHECK(!next.hit);
    CHECK(next.index != 0);
    CHECK(c.live() == 2);
    CHECK(!c.acquire(make_key(1, 2, 3), 3).hit);
    CHECK(c.live() == 3);
}

void test_lru_eviction() {
    cache c(2);
    const auto a = c.acquire(make_key(1), 1);
    const auto b = c.acquire(make_key(2), 2);
    CHECK(a.index == 0 && b.index == 1);
    // Touch A so B is the oldest.
    CHECK(c.acquire(make_key(1), 3).hit);
    const auto victim = c.acquire(make_key(3), 4);
    CHECK(!victim.hit);
    CHECK(victim.evicted == b.index);
    CHECK(victim.index == b.index);
    CHECK(c.get_stats().evictions == 1);
    CHECK(c.live() == 2);
    // The evicted key is gone; the touched key remains.
    CHECK(c.acquire(make_key(1), 5).hit);
    CHECK(!c.acquire(make_key(2), 6).hit);
}

void test_flush_and_rekey() {
    cache c(2);
    CHECK(!c.acquire(make_key(1), 1).hit);
    CHECK(!c.acquire(make_key(2), 2).hit);
    CHECK(c.live() == 2);

    // rekey: can_reuse() rejected a key hit, so the entry now holds a new
    // shape. The slot keeps its identity.
    const auto hit = c.acquire(make_key(1), 3);
    CHECK(hit.hit);
    c.rekey(hit.index, make_key(9), 4);
    CHECK(c.get_stats().rekeys == 1);
    CHECK(c.at(hit.index) != nullptr);
    CHECK(*c.at(hit.index) == make_key(9));
    CHECK(c.acquire(make_key(9), 5).hit);
    CHECK(!c.acquire(make_key(1), 6).hit);

    c.flush();
    CHECK(c.live() == 0);
    CHECK(c.get_stats().flushes == 1);
    CHECK(!c.acquire(make_key(9), 7).hit);

    // invalidate drops one slot only.
    CHECK(!c.acquire(make_key(10), 8).hit);
    CHECK(!c.acquire(make_key(11), 9).hit);
    c.invalidate(0);
    CHECK(c.live() == 1);
    CHECK(c.at(0) == nullptr);
}

void test_dual_window_isolation() {
    cache left(4);
    cache right(4);
    const auto l = left.acquire(make_key(1), 1);
    const auto r = right.acquire(make_key(1), 1);
    CHECK(!l.hit && !r.hit);
    CHECK(l.index == r.index); // same slot number, different cache
    CHECK(left.get_stats().inserts == 1);
    CHECK(right.get_stats().inserts == 1);

    left.flush();
    CHECK(left.live() == 0);
    CHECK(right.live() == 1);
    CHECK(right.acquire(make_key(1), 2).hit);
}

void test_output_rows() {
    using llama_shape_cache::output_rows_bound;
    using llama_shape_cache::parse_output_rows;
    CHECK(parse_output_rows(nullptr) == 0);
    CHECK(parse_output_rows("") == 0);
    CHECK(parse_output_rows("0") == 0);
    CHECK(parse_output_rows("13") == 13);
    CHECK(parse_output_rows("16") == 16);
    CHECK(parse_output_rows("-1") == 0);
    CHECK(parse_output_rows("x") == 0);
    CHECK(parse_output_rows("4x") == 0);
    CHECK(parse_output_rows("1048577") == 0);

    // The constructor bound: max(n_outputs_max, n_seq_max, explicit rows).
    CHECK(output_rows_bound(4, 1, nullptr) == 4);
    CHECK(output_rows_bound(4, 1, "13") == 13);
    CHECK(output_rows_bound(16, 1, "13") == 16);
    CHECK(output_rows_bound(4, 2, "0") == 4);
    CHECK(output_rows_bound(4, 8, nullptr) == 8);
    CHECK(output_rows_bound(4, 1, "junk") == 4);
}

void test_flag_and_capacity() {
    using llama_shape_cache::parse_capacity;
    CHECK(parse_capacity(nullptr) == 0);
    CHECK(parse_capacity("") == 0);
    CHECK(parse_capacity("0") == 0);
    CHECK(parse_capacity("1") == 1);
    CHECK(parse_capacity("4") == 4);
    CHECK(parse_capacity("8") == llama_shape_cache::capacity_max);
    CHECK(parse_capacity("9") == 0);
    CHECK(parse_capacity("-1") == 0);
    CHECK(parse_capacity("x") == 0);
    CHECK(parse_capacity("4x") == 0);

    cache off;
    CHECK(!off.enabled());
    const auto none = off.acquire(make_key(1), 1);
    CHECK(none.index == -1);
    CHECK(!none.hit);

    cache clamped(64);
    CHECK(clamped.capacity() == llama_shape_cache::capacity_max);

    // The five-shape relay working set needs at least five slots to avoid the
    // LRU thrash the frame analysis measured.
    cache five(5);
    for (uint32_t shape = 1; shape <= 5; ++shape) {
        CHECK(!five.acquire(make_key(shape), shape).hit);
    }
    for (uint32_t shape = 1; shape <= 5; ++shape) {
        CHECK(five.acquire(make_key(shape), 10 + shape).hit);
    }
    CHECK(five.get_stats().misses == 5);
    CHECK(five.get_stats().evictions == 0);
}

} // namespace

int main() {
    test_same_shape_hit();
    test_alloc_gen();
    test_lru_eviction();
    test_flush_and_rekey();
    test_dual_window_isolation();
    test_output_rows();
    test_flag_and_capacity();
    if (failures != 0) {
        std::fprintf(stderr, "SHAPE_CACHE_TEST FAIL %d\n", failures);
        return 1;
    }
    std::printf("SHAPE_CACHE_TEST PASS\n");
    return 0;
}
