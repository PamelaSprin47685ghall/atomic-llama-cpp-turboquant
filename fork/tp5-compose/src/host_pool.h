#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <llama.h>
#include <tq/alloc/events.h>

namespace tq::tp5 {

namespace host_pool_detail {

// True only for the exact string "1".
inline bool exact_one(const char * name) noexcept {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

inline std::uint64_t monotonic_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

} // namespace host_pool_detail

// A worker-local pool of one-token MTP decode batches. A slot owns one
// llama_batch and one inline token. The pool removes the per-token malloc and
// free cost. The pool is not thread safe: one worker owns one pool. The pool
// must outlive every lease. An MTP decode reads the NextN hidden row, not the
// token, so every lease must carry one full-width embedding row. The width is
// fixed at construction; a mismatched set() is refused instead of overrunning
// the slot buffer. An allocation failure frees the partial batch and throws
// std::bad_alloc; the slot stays unready so a later acquire retries.
class mtp_batch_pool {
    struct slot {
        llama_batch batch{};
        llama_token token = 0;
        std::uint64_t gen = 0;
        bool ready = false;
        bool in_use = false;
    };

    // llama_batch_free checks every array and stops the seq_id walk at the
    // first null entry, so a partially allocated llama_batch_init result is
    // safe to free. The caller must clear an inline token pointer first.
    static void discard_partial(llama_batch & batch) noexcept {
        llama_batch_free(batch);
        batch = {};
    }

public:
    // Site id 2 identifies the MTP batch lease in the in-memory probe record.
    // Keep it stable: the C adapter maps the numeric id to a symbol.
    static constexpr std::uint32_t event_site = 2;
    // Bound the optional probe log. Steady state would otherwise append two
    // records per token.
    static constexpr std::size_t max_events = 4096;

    // A one-token batch. A pooled lease borrows a pool slot. A fallback lease
    // owns a heap batch. Both release their batch when the lease dies.
    class lease {
    public:
        lease(const lease &) = delete;
        lease & operator=(const lease &) = delete;
        lease(lease &&) = delete;
        lease & operator=(lease &&) = delete;

        ~lease() {
            if (batch_ == nullptr) {
                return;
            }
            batch_->n_tokens = 0;
            if (pooled_) {
                pool_->release(slot_, slot_gen_);
            } else {
                std::free(batch_->token);
                batch_->token = nullptr;
                llama_batch_free(*batch_);
                pool_->record(slot_, slot_gen_, tq::alloc::LeaseAction::release,
                              tq::alloc::State::leased, tq::alloc::State::free);
            }
        }

        llama_batch & batch() { return *batch_; }

        void set(llama_token t, llama_pos p, int8_t logits, const float * embd, int wide) {
            if (!storage_complete()) {
                throw std::bad_alloc();
            }
            if (embd == nullptr) {
                throw std::invalid_argument("MTP batch lease requires an embedding row");
            }
            if (wide != pool_->wide_) {
                throw std::invalid_argument("MTP batch lease width differs from the pool width");
            }
            batch_->n_tokens = 1;
            batch_->token[0] = t;
            batch_->pos[0] = p;
            batch_->n_seq_id[0] = 1;
            batch_->seq_id[0][0] = 0;
            batch_->logits[0] = logits;
            std::memcpy(batch_->embd, embd, static_cast<std::size_t>(wide) * sizeof(float));
        }

    private:
        friend class mtp_batch_pool;

        // The arrays are fixed when the lease is created, and acquire refuses
        // an incomplete batch, so this is a local fail-stop guard against a
        // future constructor path rather than a reachable state today.
        bool storage_complete() const noexcept {
            return batch_ != nullptr && batch_->token != nullptr && batch_->embd != nullptr &&
                   batch_->pos != nullptr && batch_->n_seq_id != nullptr &&
                   batch_->seq_id != nullptr && batch_->seq_id[0] != nullptr &&
                   batch_->logits != nullptr;
        }

        lease(mtp_batch_pool * pool, slot & s, std::uint32_t index)
            : batch_(&s.batch), pool_(pool), slot_(index), slot_gen_(s.gen), pooled_(true) {}

        explicit lease(mtp_batch_pool * pool) : pool_(pool) {
            // An out-of-range slot marks a heap fallback in the probe log.
            slot_ = static_cast<std::uint32_t>(pool->slots_.size());
            llama_batch batch = llama_batch_init(1, pool->wide_, 1);
            if (!llama_batch_is_complete(&batch, 1, true, false)) {
                discard_partial(batch);
                throw std::bad_alloc();
            }
            // llama_batch_init leaves token null when embd > 0. The lease owns
            // that one token separately; its destructor frees it and clears
            // the pointer before llama_batch_free runs. Free any token a
            // future init might return so the lease is the only owner.
            std::free(batch.token);
            batch.token = static_cast<llama_token *>(std::malloc(sizeof(llama_token)));
            if (batch.token == nullptr) {
                discard_partial(batch);
                throw std::bad_alloc();
            }
            owned_batch_ = batch;
            batch_ = &owned_batch_;
            pool_->record(slot_, slot_gen_, tq::alloc::LeaseAction::lease,
                          tq::alloc::State::free, tq::alloc::State::leased);
        }

        llama_batch owned_batch_{};
        llama_batch * batch_ = nullptr;
        mtp_batch_pool * pool_ = nullptr;
        std::uint32_t slot_ = 0;
        std::uint64_t slot_gen_ = 0;
        bool pooled_ = false;
    };

    // Each slot starts uninitialized. The first acquire of a slot allocates
    // its batch. A pool that is disabled or exhausted returns a fallback
    // lease that owns a heap batch.
    mtp_batch_pool(std::size_t n_slots, int wide)
        : slots_(n_slots), wide_(wide), probe_(host_pool_detail::exact_one("TP5_ALLOC_PROBE")),
          enabled_(host_pool_detail::exact_one("TP5_MTP_BATCH_POOL")) {
        if (wide <= 0) {
            // A non-positive width would make llama_batch_init allocate the
            // token array that the slot then replaces, and set() could not
            // carry an embedding row. Refuse it at construction.
            throw std::invalid_argument("MTP batch pool width must be positive");
        }
        if (probe_) {
            events_.reserve(max_events);
        }
    }

    ~mtp_batch_pool() {
        for (slot & s : slots_) {
            if (!s.ready) {
                continue;
            }
            // The token lives in the slot, not on the heap. Clear the pointer
            // before llama_batch_free frees the batch arrays.
            s.batch.token = nullptr;
            llama_batch_free(s.batch);
        }
    }

    mtp_batch_pool(const mtp_batch_pool &) = delete;
    mtp_batch_pool & operator=(const mtp_batch_pool &) = delete;

    lease acquire() {
        if (enabled_) {
            for (std::size_t i = 0; i < slots_.size(); ++i) {
                slot & s = slots_[i];
                if (s.in_use) {
                    continue;
                }
                if (!s.ready) {
                    llama_batch batch = llama_batch_init(1, wide_, 1);
                    if (!llama_batch_is_complete(&batch, 1, true, false)) {
                        // Free whatever llama_batch_init produced. The slot stays
                        // unready, so a later acquire retries instead of
                        // caching a partial batch.
                        discard_partial(batch);
                        throw std::bad_alloc();
                    }
                    // embd > 0 makes llama_batch_init leave token null; the
                    // slot supplies inline storage. Free any token a future
                    // init might return so the inline storage is the only
                    // owner.
                    std::free(batch.token);
                    batch.token = &s.token;
                    s.batch = batch;
                    s.ready = true;
                }
                s.in_use = true;
                ++s.gen;
                record(static_cast<std::uint32_t>(i), s.gen, tq::alloc::LeaseAction::lease,
                       tq::alloc::State::free, tq::alloc::State::leased);
                return lease(this, s, static_cast<std::uint32_t>(i));
            }
        }
        return lease(this);
    }

    const std::vector<tq::alloc::LeaseEvent> & events() const { return events_; }

private:
    void release(std::uint32_t index, std::uint64_t gen) {
        slot & s = slots_[index];
        s.in_use = false;
        record(index, gen, tq::alloc::LeaseAction::release, tq::alloc::State::leased,
               tq::alloc::State::free);
    }

    void record(std::uint32_t index, std::uint64_t gen, tq::alloc::LeaseAction action,
                tq::alloc::State from, tq::alloc::State to) {
        if (!probe_ || events_.size() >= max_events) {
            return;
        }
        tq::alloc::LeaseEvent event;
        event.ts_ns = host_pool_detail::monotonic_ns();
        event.seq = event_seq_++;
        event.lifetime = tq::alloc::Lifetime::step;
        event.reuse_class = tq::alloc::ReuseClass::staging;
        event.site = event_site;
        event.bytes = static_cast<std::uint64_t>(wide_) * sizeof(float) + sizeof(llama_token);
        event.slot = index;
        event.slot_gen = gen;
        event.action = action;
        event.from = from;
        event.to = to;
        events_.push_back(event);
    }

    std::vector<slot> slots_;
    int wide_ = 0;
    bool probe_ = false;
    bool enabled_ = false;
    std::uint64_t event_seq_ = 0;
    std::vector<tq::alloc::LeaseEvent> events_;
};

// A reusable buffer for one token piece. The view points into the buffer and
// stays valid until the next view call. The buffer keeps its capacity across
// tokens.
class piece_scratch {
public:
    std::string_view view(const llama_vocab * vocab, llama_token token) {
        // Keep at least 256 bytes, as the former piece() did. A retry may grow
        // the buffer; it never shrinks, so the next call does not zero-fill
        // the bytes between the last piece and the 256-byte floor.
        if (buffer_.size() < 256) {
            buffer_.resize(256);
        }
        auto size = llama_token_to_piece(vocab, token, buffer_.data(),
                                         static_cast<std::int32_t>(buffer_.size()), 0, false);
        if (size < 0) {
            if (size == std::numeric_limits<std::int32_t>::min()) {
                throw std::runtime_error("token piece size overflow");
            }
            buffer_.resize(static_cast<std::size_t>(-size));
            size = llama_token_to_piece(vocab, token, buffer_.data(),
                                        static_cast<std::int32_t>(buffer_.size()), 0, false);
        }
        if (!(size >= 0 && static_cast<std::size_t>(size) <= buffer_.size())) {
            throw std::runtime_error("cannot detokenize accepted token");
        }
        return std::string_view(buffer_.data(), static_cast<std::size_t>(size));
    }

    // Test and diagnostics: the retained buffer size, not the last piece size.
    std::size_t buffer_size() const { return buffer_.size(); }

private:
    std::string buffer_;
};

} // namespace tq::tp5
