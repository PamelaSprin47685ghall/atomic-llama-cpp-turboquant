#include "llama-ubatch-pool.h"

#include <cassert>
#include <utility>

ubatch_data_handle::ubatch_data_handle(std::shared_ptr<llama_ubatch_data> data) : fallback_(std::move(data)) {}

ubatch_data_handle::ubatch_data_handle(std::shared_ptr<ubatch_data_pool> pool, uint32_t slot, uint32_t gen)
    : pool_(std::move(pool)), slot_(slot), gen_(gen) {}

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
    : pool_(std::move(other.pool_)), slot_(other.slot_), gen_(other.gen_), fallback_(std::move(other.fallback_)) {}

ubatch_data_handle & ubatch_data_handle::operator=(ubatch_data_handle && other) noexcept {
    if (this != &other) {
        reset();
        pool_     = std::move(other.pool_);
        slot_     = other.slot_;
        gen_      = other.gen_;
        fallback_ = std::move(other.fallback_);
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
        pool_.reset();
    }

    fallback_.reset();
}

bool ubatch_data_pool::configure(uint32_t capacity) {
    if (capacity == 0) {
        return false;
    }

    // A previous grow already provides this capacity. The release store after
    // slots_.resize() pairs with this acquire load; slots_ itself is only read
    // under the mutex by the callers that use it. This keeps the per-split
    // configure call out of the pool mutex in the steady state.
    if (capacity <= capacity_.load(std::memory_order_acquire)) {
        return true;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (capacity <= slots_.size()) {
        capacity_.store(static_cast<uint32_t>(slots_.size()), std::memory_order_release);
        return true;
    }

    if (in_use_ != 0) {
        // Growing would move live slots. The caller keeps using the current
        // capacity and falls back to the heap when it runs out.
        return false;
    }

    slots_.resize(capacity);
    capacity_.store(capacity, std::memory_order_release);

    return true;
}

bool ubatch_data_pool::configured() const {
    return capacity_.load(std::memory_order_acquire) != 0;
}

uint32_t ubatch_data_pool::capacity() const {
    return capacity_.load(std::memory_order_acquire);
}

ubatch_data_handle ubatch_data_pool::acquire() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (uint32_t i = 0; i < slots_.size(); ++i) {
        slot_t & slot = slots_[i];

        if (slot.in_use) {
            continue;
        }

        // Take the owner reference before marking the slot. shared_from_this()
        // throws when the pool is not owned by a shared_ptr; marking the slot
        // first would leave it in_use with no handle left to release it, so
        // the pool would silently lose one slot for the rest of its life.
        std::shared_ptr<ubatch_data_pool> self = shared_from_this();

        slot.in_use = true;
        slot.refs.store(1, std::memory_order_release);
        ++slot.gen;
        ++in_use_;
        ++stats_.acquired;

        return ubatch_data_handle(std::move(self), i, slot.gen);
    }

    ++stats_.exhausted;

    return ubatch_data_handle();
}

void ubatch_data_pool::retain(uint32_t slot) {
    // A handle copy already owns a reference. While that reference exists the
    // slot stays in_use, its generation is stable, and configure() cannot
    // resize slots_ (in_use_ != 0). Only the atomic refcount changes, so the
    // increment does not need the pool lock.
    assert(slot < slots_.size());
    assert(slots_[slot].in_use);

    slots_[slot].refs.fetch_add(1, std::memory_order_relaxed);
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

    assert(entry.refs.load(std::memory_order_relaxed) > 0);

    if (entry.refs.fetch_sub(1, std::memory_order_acq_rel) == 1) {
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
