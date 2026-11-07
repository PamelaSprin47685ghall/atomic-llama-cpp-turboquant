#include "tq/alloc/domain.h"

namespace tq::alloc {

Result Domain::lease(Class cls, Extent extent, Lease & out) noexcept {
    if (state_ == DomainState::draining) {
        const Result drained = drain();
        if (!drained) {
            return drained;
        }
    }
    const Result result = arena_->acquire(cls, extent, out);
    if (result) {
        state_ = DomainState::active;
    }
    return result;
}

Result Domain::try_lease(Class cls, Extent extent, Lease & out) noexcept {
    if (state_ == DomainState::draining) {
        return failure(Code::busy);
    }
    const Result result = arena_->acquire(cls, extent, out);
    if (result) {
        state_ = DomainState::active;
    }
    return result;
}

Result Domain::cancel(const Lease & lease) noexcept {
    if (lease.cls == Class::cold) {
        return failure(Code::invalid_argument);
    }
    if (pending_count_ == max_pending) {
        return failure(Code::full);
    }
    const Result result = arena_->release(lease);
    if (!result) {
        return result;
    }
    const Result queued = enqueue(lease.slot);
    if (!queued) {
        return queued;
    }
    state_ = DomainState::draining;
    return success();
}

Result Domain::release(const Lease & lease) noexcept {
    const bool deferred = lease.cls == Class::frame || lease.cls == Class::staging;
    if (deferred && pending_count_ == max_pending) {
        return failure(Code::full);
    }
    const Result result = arena_->release(lease);
    if (!result) {
        return result;
    }
    if (deferred) {
        const Result queued = enqueue(lease.slot);
        if (!queued) {
            return queued;
        }
        state_ = DomainState::draining;
    } else {
        settle();
    }
    return success();
}

Result Domain::drain() noexcept {
    Result first = success();
    for (std::size_t i = 0; i < pending_count_; ++i) {
        const Result result = arena_->drain(pending_[i]);
        if (!result && first) {
            first = result;
        }
    }
    pending_count_ = 0;
    settle();
    return first;
}

Result Domain::quarantine(const Lease & lease) noexcept {
    const Result result = arena_->quarantine(lease);
    if (!result) {
        return result;
    }
    state_ = DomainState::quarantined;
    return success();
}

Result Domain::enqueue(std::uint32_t slot) noexcept {
    if (pending_count_ == max_pending) {
        return failure(Code::full);
    }
    pending_[pending_count_] = slot;
    ++pending_count_;
    return success();
}

void Domain::settle() noexcept {
    if (pending_count_ != 0) {
        state_ = DomainState::draining;
        return;
    }
    state_ = arena_->live_count() != 0 ? DomainState::active
                                       : DomainState::idle;
}

} // namespace tq::alloc
