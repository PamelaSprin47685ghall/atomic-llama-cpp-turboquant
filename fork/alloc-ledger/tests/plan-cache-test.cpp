#include <cstdio>
#include <cstdint>

#include "tq/alloc/plan_cache.h"

using namespace tq::alloc;

namespace {

int failures = 0;

#define CHECK(expr)                                                            \
    do {                                                                       \
        if (!(expr)) {                                                         \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__,        \
                         __LINE__, #expr);                                     \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

PlanEntry make_entry(std::uint64_t model, std::uint64_t footprint) {
    PlanEntry entry;
    entry.key.model = model;
    entry.key.footprint = footprint;
    entry.footprint = footprint;
    entry.slots = 1;
    return entry;
}

PlanKey make_key(std::uint64_t model, std::uint64_t footprint) {
    return PlanKey{model, 0, footprint};
}

void test_hit_and_miss() {
    PlanCache cache;
    CHECK(cache.find(make_key(1, 10)) == nullptr);
    CHECK(cache.stats().misses == 1);

    CHECK(cache.insert(make_entry(1, 10)).ok());
    const PlanEntry * found = cache.find(make_key(1, 10));
    CHECK(found != nullptr);
    CHECK(found->key.model == 1);
    CHECK(found->footprint == 10);
    CHECK(cache.stats().hits == 1);
    CHECK(cache.stats().inserts == 1);
}

void test_same_key_does_not_duplicate() {
    PlanCache cache;
    PlanEntry first = make_entry(1, 100);
    CHECK(cache.insert(first).ok());

    PlanEntry second = first;
    second.footprint = 200;
    second.slots = 4;
    CHECK(cache.insert(second).ok());
    CHECK(cache.size() == 1);
    CHECK(cache.stats().inserts == 1);
    CHECK(cache.stats().replaces == 1);

    const PlanEntry * found = cache.find(make_key(1, 100));
    CHECK(found != nullptr);
    CHECK(found->footprint == 200);
    CHECK(found->slots == 4);
}

void test_lru_eviction() {
    PlanCache cache;
    for (std::uint64_t i = 1; i <= 8; ++i) {
        CHECK(cache.insert(make_entry(i, i * 10)).ok());
    }
    CHECK(cache.size() == 8);

    CHECK(cache.insert(make_entry(9, 90)).ok());
    CHECK(cache.size() == 8);
    CHECK(cache.stats().evictions == 1);
    CHECK(cache.find(make_key(1, 10)) == nullptr);
    CHECK(cache.find(make_key(9, 90)) != nullptr);
}

void test_find_refreshes_lru() {
    PlanCache cache;
    for (std::uint64_t i = 1; i <= 8; ++i) {
        CHECK(cache.insert(make_entry(i, i * 10)).ok());
    }
    CHECK(cache.find(make_key(1, 10)) != nullptr);
    CHECK(cache.insert(make_entry(9, 90)).ok());

    // Entry 1 was touched, so entry 2 is the oldest.
    CHECK(cache.find(make_key(2, 20)) == nullptr);
    CHECK(cache.find(make_key(1, 10)) != nullptr);
}

void test_pin_blocks_eviction() {
    PlanCache cache;
    for (std::uint64_t i = 1; i <= 8; ++i) {
        CHECK(cache.insert(make_entry(i, i * 10)).ok());
    }
    CHECK(cache.pin(make_key(1, 10), true).ok());
    CHECK(cache.insert(make_entry(9, 90)).ok());

    CHECK(cache.find(make_key(1, 10)) != nullptr);
    CHECK(cache.find(make_key(2, 20)) == nullptr);

    for (std::uint64_t i = 3; i <= 9; ++i) {
        CHECK(cache.pin(make_key(i, i * 10), true).ok());
    }
    CHECK(cache.insert(make_entry(10, 100)).code == Code::full);
    CHECK(cache.size() == 8);
    CHECK(cache.stats().pin_rejects == 1);

    CHECK(cache.pin(make_key(1, 10), false).ok());
    CHECK(cache.insert(make_entry(10, 100)).ok());
    CHECK(cache.find(make_key(1, 10)) == nullptr);
    CHECK(cache.find(make_key(10, 100)) != nullptr);
}

void test_flush_invalidates_old_keys() {
    PlanCache cache;
    CHECK(cache.insert(make_entry(1, 10)).ok());
    CHECK(cache.find(make_key(1, 10)) != nullptr);

    cache.flush_all();
    CHECK(cache.size() == 0);
    CHECK(cache.find(make_key(1, 10)) == nullptr);
    CHECK(cache.at(0) == nullptr);
    CHECK(cache.stats().flushes == 1);

    // A key inserted after a flush is a new insert, not a replacement.
    CHECK(cache.insert(make_entry(1, 10)).ok());
    CHECK(cache.size() == 1);
    CHECK(cache.stats().inserts == 2);
    CHECK(cache.stats().replaces == 0);
}

void test_erase_and_missing() {
    PlanCache cache;
    CHECK(cache.erase(make_key(5, 50)).code == Code::missing);
    CHECK(cache.pin(make_key(5, 50), true).code == Code::missing);

    CHECK(cache.insert(make_entry(5, 50)).ok());
    CHECK(cache.erase(make_key(5, 50)).ok());
    CHECK(cache.size() == 0);
    CHECK(cache.find(make_key(5, 50)) == nullptr);
}

void test_instances_are_independent() {
    PlanCache first;
    PlanCache second;
    CHECK(first.insert(make_entry(1, 10)).ok());
    CHECK(first.size() == 1);
    CHECK(second.size() == 0);
    CHECK(second.find(make_key(1, 10)) == nullptr);
    CHECK(first.find(make_key(1, 10)) != nullptr);
}

void test_external_values_cannot_mutate_state() {
    PlanCache cache;
    PlanEntry input = make_entry(1, 100);
    CHECK(cache.insert(input).ok());
    input.footprint = 999;
    CHECK(cache.find(make_key(1, 100))->footprint == 100);

    const PlanEntry * found = cache.find(make_key(1, 100));
    PlanEntry copy = *found;
    copy.footprint = 777;
    CHECK(copy.footprint == 777);
    CHECK(cache.find(make_key(1, 100))->footprint == 100);
}

} // namespace

int main() {
    test_hit_and_miss();
    test_same_key_does_not_duplicate();
    test_lru_eviction();
    test_find_refreshes_lru();
    test_pin_blocks_eviction();
    test_flush_invalidates_old_keys();
    test_erase_and_missing();
    test_instances_are_independent();
    test_external_values_cannot_mutate_state();

    if (failures != 0) {
        std::fprintf(stderr, "%d plan cache checks failed\n", failures);
        return 1;
    }
    std::printf("plan cache tests passed\n");
    return 0;
}
