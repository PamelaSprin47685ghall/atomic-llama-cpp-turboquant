#include <cstdio>
#include <cstring>
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

PlanKey base_key() {
    PlanKey key;
    key.model = 1;
    key.window = 2;
    key.n_tokens = 3;
    key.n_outputs = 4;
    key.rs_class = -1;
    key.n_kv_bucket = 5;
    key.alloc_gen = 6;
    return key;
}

PlanEntry make_entry(const PlanKey & key, std::uint64_t footprint) {
    PlanEntry entry;
    entry.key = key;
    entry.footprint = footprint;
    entry.slots = 1;
    return entry;
}

void test_key_fields_are_distinct() {
    const PlanKey base = base_key();
    CHECK(base.valid());
    CHECK(base == base_key());
    CHECK(!(base != base_key()));

    PlanKey changed = base;
    changed.model = 2;
    CHECK(base != changed);
    changed = base;
    changed.window = 3;
    CHECK(base != changed);
    changed = base;
    changed.n_tokens = 4;
    CHECK(base != changed);
    changed = base;
    changed.n_outputs = 5;
    CHECK(base != changed);
    changed = base;
    changed.rs_class = 0;
    CHECK(base != changed);
    changed = base;
    changed.n_kv_bucket = 6;
    CHECK(base != changed);
    changed = base;
    changed.alloc_gen = 7;
    CHECK(base != changed);
}

void test_key_validation() {
    PlanKey key = base_key();
    CHECK(key.valid());
    key.n_tokens = 0;
    CHECK(!key.valid());
    key.n_tokens = 5;
    CHECK(!key.valid());
    key = base_key();
    key.n_kv_bucket = 256;
    CHECK(!key.valid());

    PlanCache cache;
    CHECK(cache.insert(make_entry(key, 100)).code == Code::invalid_argument);
    CHECK(cache.size() == 0);
}

void test_key_description() {
    const PlanKey key = base_key();
    const std::array<char, 192> text = key.describe();
    CHECK(text[0] != '\0');
    CHECK(std::strstr(text.data(), "model=1") != nullptr);
    CHECK(std::strstr(text.data(), "window=2") != nullptr);
    CHECK(std::strstr(text.data(), "n_tokens=3") != nullptr);
    CHECK(std::strstr(text.data(), "n_outputs=4") != nullptr);
    CHECK(std::strstr(text.data(), "rs_class=-1") != nullptr);
    CHECK(std::strstr(text.data(), "n_kv_bucket=5") != nullptr);
    CHECK(std::strstr(text.data(), "alloc_gen=6") != nullptr);
}

void test_cache_separates_all_key_fields() {
    PlanCache cache;
    const PlanKey base = base_key();
    CHECK(cache.insert(make_entry(base, 10)).ok());

    PlanKey keys[7];
    for (std::size_t i = 0; i < 7; ++i) {
        keys[i] = base;
    }
    keys[0].model += 1;
    keys[1].window += 1;
    keys[2].n_tokens += 1;
    keys[3].n_outputs += 1;
    keys[4].rs_class += 1;
    keys[5].n_kv_bucket += 1;
    keys[6].alloc_gen += 1;
    for (std::size_t i = 0; i < 7; ++i) {
        CHECK(keys[i].valid());
        CHECK(cache.insert(make_entry(keys[i], 20 + i)).ok());
    }
    CHECK(cache.size() == 8);

    // A ninth key evicts the first one. A field collision would have kept
    // the size below eight.
    PlanKey ninth = base;
    ninth.n_outputs = 99;
    CHECK(cache.insert(make_entry(ninth, 99)).ok());
    CHECK(cache.size() == 8);
    CHECK(cache.stats().evictions == 1);
    CHECK(cache.find(base) == nullptr);
    for (std::size_t i = 0; i < 7; ++i) {
        CHECK(cache.find(keys[i]) != nullptr);
    }
}

void test_footprint_is_not_key() {
    PlanCache cache;
    const PlanKey key = base_key();
    CHECK(cache.insert(make_entry(key, 100)).ok());
    CHECK(cache.insert(make_entry(key, 200)).ok());
    CHECK(cache.size() == 1);
    CHECK(cache.stats().replaces == 1);

    const PlanEntry * found = cache.find(key);
    CHECK(found != nullptr);
    CHECK(found->footprint == 200);
}

void test_hit_and_miss() {
    PlanCache cache;
    CHECK(cache.find(base_key()) == nullptr);
    CHECK(cache.stats().misses == 1);

    CHECK(cache.insert(make_entry(base_key(), 10)).ok());
    const PlanEntry * found = cache.find(base_key());
    CHECK(found != nullptr);
    CHECK(found->key.model == 1);
    CHECK(found->footprint == 10);
    CHECK(cache.stats().hits == 1);
    CHECK(cache.stats().inserts == 1);
}

void test_entry_handles_are_values() {
    PlanCache cache;
    PlanEntry entry = make_entry(base_key(), 100);
    entry.graph_uid = 11;
    entry.sched_uid = 12;
    entry.meta_uid = 13;
    entry.relay_uid = 14;
    entry.arena_gen = 15;
    entry.generation = 16;
    CHECK(cache.insert(entry).ok());

    const PlanEntry * found = cache.find(base_key());
    CHECK(found != nullptr);
    CHECK(found->graph_uid == 11);
    CHECK(found->sched_uid == 12);
    CHECK(found->meta_uid == 13);
    CHECK(found->relay_uid == 14);
    CHECK(found->arena_gen == 15);
    CHECK(found->generation == 16);
}

void test_lru_eviction() {
    PlanCache cache;
    for (std::uint64_t i = 1; i <= 8; ++i) {
        PlanKey key = base_key();
        key.model = i;
        CHECK(cache.insert(make_entry(key, i * 10)).ok());
    }
    CHECK(cache.size() == 8);

    PlanKey ninth = base_key();
    ninth.model = 9;
    CHECK(cache.insert(make_entry(ninth, 90)).ok());
    CHECK(cache.size() == 8);
    CHECK(cache.stats().evictions == 1);

    PlanKey first = base_key();
    first.model = 1;
    CHECK(cache.find(first) == nullptr);
    CHECK(cache.find(ninth) != nullptr);
}

void test_find_refreshes_lru() {
    PlanCache cache;
    for (std::uint64_t i = 1; i <= 8; ++i) {
        PlanKey key = base_key();
        key.model = i;
        CHECK(cache.insert(make_entry(key, i * 10)).ok());
    }
    PlanKey first = base_key();
    first.model = 1;
    CHECK(cache.find(first) != nullptr);

    PlanKey ninth = base_key();
    ninth.model = 9;
    CHECK(cache.insert(make_entry(ninth, 90)).ok());

    // Entry 1 was touched, so entry 2 is the oldest.
    PlanKey second = base_key();
    second.model = 2;
    CHECK(cache.find(second) == nullptr);
    CHECK(cache.find(first) != nullptr);
}

void test_pin_blocks_eviction() {
    PlanCache cache;
    for (std::uint64_t i = 1; i <= 8; ++i) {
        PlanKey key = base_key();
        key.model = i;
        CHECK(cache.insert(make_entry(key, i * 10)).ok());
    }
    PlanKey first = base_key();
    first.model = 1;
    CHECK(cache.pin(first, true).ok());

    PlanKey ninth = base_key();
    ninth.model = 9;
    CHECK(cache.insert(make_entry(ninth, 90)).ok());
    CHECK(cache.find(first) != nullptr);
    PlanKey second = base_key();
    second.model = 2;
    CHECK(cache.find(second) == nullptr);

    for (std::uint64_t i = 3; i <= 9; ++i) {
        PlanKey key = base_key();
        key.model = i;
        CHECK(cache.pin(key, true).ok());
    }
    PlanKey tenth = base_key();
    tenth.model = 10;
    CHECK(cache.insert(make_entry(tenth, 100)).code == Code::full);
    CHECK(cache.size() == 8);
    CHECK(cache.stats().pin_rejects == 1);

    CHECK(cache.pin(first, false).ok());
    CHECK(cache.insert(make_entry(tenth, 100)).ok());
    CHECK(cache.find(first) == nullptr);
    CHECK(cache.find(tenth) != nullptr);
}

void test_flush_invalidates_old_keys() {
    PlanCache cache;
    CHECK(cache.insert(make_entry(base_key(), 10)).ok());
    CHECK(cache.find(base_key()) != nullptr);

    cache.flush_all();
    CHECK(cache.size() == 0);
    CHECK(cache.find(base_key()) == nullptr);
    CHECK(cache.at(0) == nullptr);
    CHECK(cache.stats().flushes == 1);

    // A key inserted after a flush is a new insert, not a replacement.
    CHECK(cache.insert(make_entry(base_key(), 10)).ok());
    CHECK(cache.size() == 1);
    CHECK(cache.stats().inserts == 2);
    CHECK(cache.stats().replaces == 0);
}

void test_erase_and_missing() {
    PlanCache cache;
    CHECK(cache.erase(base_key()).code == Code::missing);
    CHECK(cache.pin(base_key(), true).code == Code::missing);

    CHECK(cache.insert(make_entry(base_key(), 10)).ok());
    CHECK(cache.erase(base_key()).ok());
    CHECK(cache.size() == 0);
    CHECK(cache.find(base_key()) == nullptr);
}

void test_instances_are_independent() {
    PlanCache first;
    PlanCache second;
    CHECK(first.insert(make_entry(base_key(), 10)).ok());
    CHECK(first.size() == 1);
    CHECK(second.size() == 0);
    CHECK(second.find(base_key()) == nullptr);
    CHECK(first.find(base_key()) != nullptr);
}

void test_external_values_cannot_mutate_state() {
    PlanCache cache;
    PlanEntry input = make_entry(base_key(), 100);
    CHECK(cache.insert(input).ok());
    input.footprint = 999;
    input.graph_uid = 999;
    CHECK(cache.find(base_key())->footprint == 100);
    CHECK(cache.find(base_key())->graph_uid == 0);

    const PlanEntry * found = cache.find(base_key());
    PlanEntry copy = *found;
    copy.footprint = 777;
    CHECK(copy.footprint == 777);
    CHECK(cache.find(base_key())->footprint == 100);
}

} // namespace

int main() {
    test_key_fields_are_distinct();
    test_key_validation();
    test_key_description();
    test_cache_separates_all_key_fields();
    test_footprint_is_not_key();
    test_hit_and_miss();
    test_entry_handles_are_values();
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
