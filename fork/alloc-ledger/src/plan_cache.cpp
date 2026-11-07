#include "tq/alloc/plan_cache.h"

#include <cstdio>

namespace tq::alloc {

std::array<char, 192> PlanKey::describe() const noexcept {
    std::array<char, 192> text{};
    std::snprintf(
        text.data(), text.size(),
        "model=%llu window=%u n_tokens=%u n_outputs=%u rs_class=%d "
        "n_kv_bucket=%u alloc_gen=%llu",
        static_cast<unsigned long long>(model), static_cast<unsigned>(window),
        static_cast<unsigned>(n_tokens), static_cast<unsigned>(n_outputs),
        static_cast<int>(rs_class), static_cast<unsigned>(n_kv_bucket),
        static_cast<unsigned long long>(alloc_gen));
    return text;
}

std::uint64_t PlanKey::hash() const noexcept {
    // FNV-1a over the seven fields, low byte first. The byte order is fixed,
    // so the digest does not depend on host endianness or padding.
    std::uint64_t value = 1469598103934665603ULL;
    const auto mix = [&value](std::uint64_t part) {
        for (int i = 0; i < 8; ++i) {
            value ^= (part >> (8 * i)) & 0xffU;
            value *= 1099511628211ULL;
        }
    };
    mix(model);
    mix(window);
    mix(n_tokens);
    mix(n_outputs);
    mix(static_cast<std::uint8_t>(rs_class));
    mix(n_kv_bucket);
    mix(alloc_gen);
    return value;
}

const PlanEntry * PlanCache::find(const PlanKey & key) noexcept {
    for (std::size_t i = 0; i < capacity; ++i) {
        Node & node = nodes_[i];
        if (node.used && node.entry.key == key) {
            node.tick = ++clock_;
            ++stats_.hits;
            return &node.entry;
        }
    }
    ++stats_.misses;
    return nullptr;
}

Result PlanCache::insert(const PlanEntry & entry) noexcept {
    if (!entry.key.valid()) {
        return failure(Code::invalid_argument);
    }
    for (std::size_t i = 0; i < capacity; ++i) {
        Node & node = nodes_[i];
        if (node.used && node.entry.key == entry.key) {
            node.entry = entry;
            node.tick = ++clock_;
            ++stats_.replaces;
            return success();
        }
    }
    for (std::size_t i = 0; i < capacity; ++i) {
        Node & node = nodes_[i];
        if (!node.used) {
            node.entry = entry;
            node.tick = ++clock_;
            node.pinned = false;
            node.used = true;
            ++size_;
            ++stats_.inserts;
            return success();
        }
    }

    std::size_t victim = capacity;
    std::uint64_t oldest = 0;
    for (std::size_t i = 0; i < capacity; ++i) {
        const Node & node = nodes_[i];
        if (node.pinned) {
            continue;
        }
        if (victim == capacity || node.tick < oldest) {
            victim = i;
            oldest = node.tick;
        }
    }
    if (victim == capacity) {
        ++stats_.pin_rejects;
        return failure(Code::full);
    }
    nodes_[victim].entry = entry;
    nodes_[victim].tick = ++clock_;
    nodes_[victim].pinned = false;
    nodes_[victim].used = true;
    ++stats_.inserts;
    ++stats_.evictions;
    return success();
}

Result PlanCache::pin(const PlanKey & key, bool pinned) noexcept {
    for (std::size_t i = 0; i < capacity; ++i) {
        Node & node = nodes_[i];
        if (node.used && node.entry.key == key) {
            node.pinned = pinned;
            return success();
        }
    }
    return failure(Code::missing);
}

Result PlanCache::erase(const PlanKey & key) noexcept {
    for (std::size_t i = 0; i < capacity; ++i) {
        Node & node = nodes_[i];
        if (node.used && node.entry.key == key) {
            node = Node{};
            --size_;
            return success();
        }
    }
    return failure(Code::missing);
}

void PlanCache::flush_all() noexcept {
    for (std::size_t i = 0; i < capacity; ++i) {
        nodes_[i] = Node{};
    }
    size_ = 0;
    ++stats_.flushes;
}

} // namespace tq::alloc
