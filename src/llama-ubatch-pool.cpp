#include "llama-ubatch-pool.h"

#include <cassert>
#include <utility>

ubatch_data_handle::ubatch_data_handle(std::shared_ptr<llama_ubatch_data> data) : fallback_(std::move(data)) {}

ubatch_data_handle::ubatch_data_handle(ubatch_data_pool * pool, uint32_t slot, uint32_t gen)
    : pool_(pool), slot_(slot), gen_(gen) {}

ubatch_data_handle::ubatch_data_handle(const ubatch_data_handle & other)
    : pool_(other.pool_), slot_(other.slot_), gen_(other.gen_), fallback_(other.fallback_) {
    if (pool_ != nullptr) {
        pool_->retain(slot_);
    }
}

ubatch_data_handle & ubatch_data_handle::operator=(const ubatch_data_handle & other) {
    if (this != &other) {
        reset();
        pool_     = other.pool_;
        slot_     = other.slot_;
        gen_      = other.gen_;
        fallback_ = other.fallback_;
        if (pool_ != nullptr) {
            pool_->retain(slot_);
        }
    }

    return *this;
}

ubatch_data_handle::ubatch_data_handle(ubatch_data_handle && other) noexcept
    : pool_(other.pool_), slot_(other.slot_), gen_(other.gen_), fallback_(std::move(other.fallback_)) {
    other.pool_ = nullptr;
}

ubatch_data_handle & ubatch_data_handle::operator=(ubatch_data_handle && other) noexcept {
    if (this != &other) {
        reset();
        pool_     = other.pool_;
        slot_     = other.slot_;
        gen_      = other.gen_;
        fallback_ = std::move(other.fallback_);
        other.pool_ = nullptr;
    }

    return *this;
}

ubatch_data_handle::~ubatch_data_handle() {
    reset();
}

llama_ubatch_data * ubatch_data_handle::get() const {
    if (pool_ != nullptr) {
        return pool_->object(slot_, gen_);
    }

    return fallback_.get();
}

void ubatch_data_handle::reset() {
    if (pool_ != nullptr) {
        pool_->release(slot_, gen_);
        pool_ = nullptr;
    }

    fallback_.reset();
}

bool ubatch_data_pool::configure(uint32_t capacity) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (capacity == 0) {
        return false;
    }

    if (capacity <= slots_.size()) {
        return true;
    }

    if (in_use_ != 0) {
        // Growing would move live slots. The caller keeps using the current
        // capacity and falls back to the heap when it runs out.
        return false;
    }

    slots_.resize(capacity);

    return true;
}

bool ubatch_data_pool::configured() const {
    std::lock_guard<std::mutex> lock(mutex_);

    return !slots_.empty();
}

uint32_t ubatch_data_pool::capacity() const {
    std::lock_guard<std::mutex> lock(mutex_);

    return static_cast<uint32_t>(slots_.size());
}

ubatch_data_handle ubatch_data_pool::acquire() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (uint32_t i = 0; i < slots_.size(); ++i) {
        slot_t & slot = slots_[i];

        if (slot.in_use) {
            continue;
        }

        slot.in_use = true;
        slot.refs   = 1;
        ++slot.gen;
        ++in_use_;
        ++stats_.acquired;

        return ubatch_data_handle(this, i, slot.gen);
    }

    ++stats_.exhausted;

    return ubatch_data_handle();
}

void ubatch_data_pool::retain(uint32_t slot) {
    std::lock_guard<std::mutex> lock(mutex_);

    assert(slot < slots_.size());
    assert(slots_[slot].in_use);

    ++slots_[slot].refs;
}

void ubatch_data_pool::note_fallback() {
    std::lock_guard<std::mutex> lock(mutex_);

    ++stats_.fallback;
}

void ubatch_data_pool::release(uint32_t slot, uint32_t gen) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (slot >= slots_.size()) {
        ++stats_.stale;
        return;
    }

    slot_t & entry = slots_[slot];

    if (!entry.in_use || entry.gen != gen) {
        ++stats_.stale;
        return;
    }

    assert(entry.refs > 0);

    if (--entry.refs == 0) {
        entry.in_use = false;
        assert(in_use_ > 0);
        --in_use_;
        ++stats_.released;
    }
}

llama_ubatch_data * ubatch_data_pool::object(uint32_t slot, uint32_t gen) const {
    // The caller owns a reference, so this slot cannot be handed out or
    // regenerated while the handle lives.
    if (slot >= slots_.size()) {
        return nullptr;
    }

    const slot_t & entry = slots_[slot];

    return entry.in_use && entry.gen == gen ? const_cast<llama_ubatch_data *>(&entry.data) : nullptr;
}

ubatch_data_pool::stats_t ubatch_data_pool::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    return stats_;
}

uint32_t ubatch_data_pool::in_use() const {
    std::lock_guard<std::mutex> lock(mutex_);

    return in_use_;
}

bool ubatch_data_pool::debug_release_stale(uint32_t slot, uint32_t gen) {
    std::uint64_t stale_before = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stale_before = stats_.stale;
    }

    release(slot, gen);

    std::lock_guard<std::mutex> lock(mutex_);

    return stats_.stale != stale_before;
}

ubatch_data_pool::~ubatch_data_pool() {
    // A live handle would outlive its slot storage. The owner must drop every
    // copy before the pool dies.
    assert(in_use_ == 0);
}
