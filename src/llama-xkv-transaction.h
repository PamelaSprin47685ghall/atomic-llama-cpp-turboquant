#pragma once

// Boring RAII multi-slot transaction/quiescence coordinator for XKV speculative verification.
//
// Owns NO payload bytes, NO semantic cells, NO graph state, and NO duplicate
// authoritative stamp. It delegates all snapshot stamp queries and epoch
// publications to an injected `xkv_stamp_provider` (typically backed by
// `llama_xkv_cache_store`):
//   - snapshot queries read directly from the authoritative provider,
//   - batch settlement performs a two-phase advance:
//     1. Prebuilds every container, node, and copy (zero-allocation tail).
//     2. Preflights one content advance reservation token with provider before
//        the removal callback runs. If preflight fails, aborts immediately
//        (zero removals, callback never called).
//     3. Invokes removal callback out of lock with per-tx identities and generations.
//        The callback returns the removal result stamp in feedback (or empty if staged-no-epoch).
//        If feedback reports a store stamp from actual store removal, coordinator accepts
//        it rather than falsely reporting stale stamp.
//     4. Infallibly applies the content advance via provider (no allocation).
//     5. Performs noexcept swaps/moves to publish committed state.
//   - single-slot commit routes through the batch settlement engine cleanly.
//   - rollback and cancel remove tentative private hot rows out of lock without
//     advancing RERoT public view.publish_epoch (which would spuriously invalidate all lanes).
//     view.publish_epoch is advanced ONLY by actual public frontier publication or final_fence.
//   - postcompute rebase permits private hot/content/live/binding epoch deltas
//     while requiring exact external RERoT view and codec epoch match.
//   - multi-slot: active records are keyed by tx_id + seq_id; exactly one active
//     tx per seq_id; multiple active seq_ids concurrently.
//   - no one slot commit stales valid peer transactions.
//   - maintenance, final fence, and context shift are excluded while ANY active
//     transaction, removal callback, or reader snapshot is in flight.
//     Publication (seal/pack/relocate/landmark_publish) must run only after
//     snapshot pins are released (§10.2: release pins before reclaim/pack/seal
//     and epoch/view publication; §10.3 COW/pin).
//   - all public mutators catch std::bad_alloc / exceptions and roll back safely,
//     never throwing into C ABI or leaving half-modified coordinator state.

#include "llama-xkv-cache.h"
#include "llama.h"

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llama_xkv {

enum class xkv_tx_status : uint8_t {
    ok = 0,
    bad_tx_id,
    already_active,
    tx_id_reused,
    tx_mismatch,
    no_active_tx,
    stale_stamp,
    maintenance_active,
    tx_active,
    busy,
    callback_failed,
    invalid_accept,
    cancelled,
    epoch_overflow,
};

const char * xkv_tx_status_to_str(xkv_tx_status status);

enum class xkv_maintenance_op : uint8_t {
    seal = 0,
    pack,
    relocate,
    landmark_publish,
};

const char * xkv_maintenance_op_to_str(xkv_maintenance_op op);

// Item describing a rejected payload in a batch settlement.
struct xkv_rejected_item {
    uint64_t tx_id = 0;
    llama_seq_id seq_id = 0;
    uint64_t payload_id = 0;
    uint64_t generation = 0;

    bool operator==(const xkv_rejected_item & o) const {
        return tx_id == o.tx_id && seq_id == o.seq_id &&
               payload_id == o.payload_id && generation == o.generation;
    }
};

// Batch rejected-suffix removal callback receiving items with per-tx identities and generations.
// The coordinator passes its live preflight reservation token: the store's
// token-carried remove path refuses while a reservation is held unless the
// caller presents this exact token, so the callback MUST forward it to the
// store removal call. A zero token means no reservation is held (accept-all
// path invokes no callback at all).
using xkv_batch_remove_rejected_fn = std::function<bool(
    const std::vector<xkv_rejected_item> & rejected_items,
    uint64_t reservation_token,
    xkv_removal_feedback * feedback,
    std::string * err
)>;

// Single-tx removal callback (for single-slot commit overload).
using xkv_remove_rejected_fn = std::function<bool(
    const std::vector<uint64_t> & rejected_ids,
    uint64_t reservation_token,
    xkv_removal_feedback * feedback,
    std::string * err
)>;

// Removes every transaction-created payload (rollback/cancel) — recorded order.
using xkv_rollback_fn = std::function<bool(
    const std::vector<uint64_t> & created_ids,
    std::string * err
)>;

struct xkv_quiesce_options {
    bool wait = false; // false => deterministic refusal (busy) when blocked
};

// Single slot decision for batch settlement.
struct xkv_slot_settlement {
    uint64_t tx_id = 0;
    std::vector<uint64_t> accepted_ids; // must be exact recorded prefix of tx order
};

// In-memory fake provider for standalone unit testing without a store.
class xkv_in_memory_stamp_provider : public xkv_stamp_provider {
public:
    explicit xkv_in_memory_stamp_provider(const xkv_snapshot_stamp & init = {}) : stamp_(init) {}

    xkv_snapshot_stamp current_stamp() const override;

    bool preflight_content_advance(xkv_stamp_reservation * out_res, std::string * err) override;
    xkv_snapshot_stamp apply_content_advance(const xkv_stamp_reservation & res,
                                            const xkv_removal_feedback & feedback) noexcept override;
    void abort_content_advance(const xkv_stamp_reservation & res) noexcept override;

    bool install_stable_view(const llama_rerot_view_stamp & stable_view,
                             xkv_snapshot_stamp * out_new_stamp,
                             std::string * err) override;
    bool advance_context_shift(const llama_rerot_view_stamp & shifted_view,
                               xkv_snapshot_stamp * out_new_stamp,
                               std::string * err) override;

    void set_stamp_for_testing(const xkv_snapshot_stamp & s);
    void set_fail_content_advance(bool fail) { fail_content_ = fail; }

    // Test-friendly advance_content_epoch helper (for fault-matrix tests).
    bool advance_content_epoch(const xkv_snapshot_stamp & /*cur*/, std::string * err) {
        xkv_stamp_reservation res;
        if (!preflight_content_advance(&res, err)) return false;
        xkv_removal_feedback fb;
        apply_content_advance(res, fb);
        return true;
    }

private:
    mutable std::mutex mtx_;
    xkv_snapshot_stamp stamp_ = {};
    uint64_t next_token_ = 1;
    bool fail_content_ = false;
};

class xkv_transaction_coordinator {
public:
    xkv_transaction_coordinator();
    explicit xkv_transaction_coordinator(xkv_stamp_provider * provider);
    xkv_transaction_coordinator(const xkv_transaction_coordinator &) = delete;
    xkv_transaction_coordinator & operator=(const xkv_transaction_coordinator &) = delete;

    // ---- stamp provider binding ----
    xkv_stamp_provider * stamp_provider() const;
    xkv_tx_status bind_stamp_provider(xkv_stamp_provider * provider, std::string * err = nullptr);

    // ---- stamp (delegates directly to authoritative provider) ----
    xkv_snapshot_stamp current_stamp() const;
    bool is_stale(const xkv_snapshot_stamp & expected, std::string * reason = nullptr) const;
    void observe_store_stamp(const xkv_snapshot_stamp & observed);
    void sync_from_store(const xkv_snapshot_stamp & store_stamp);

    // ---- transaction lifecycle (multi-slot: keyed by tx_id + seq_id) ----
    xkv_tx_status begin(uint64_t tx_id, llama_seq_id seq_id, const xkv_snapshot_stamp & snapshot, std::string * err = nullptr);
    xkv_tx_status begin(uint64_t tx_id, const xkv_snapshot_stamp & snapshot, std::string * err = nullptr); // seq_id = 0

    xkv_tx_status record_payload(uint64_t tx_id, uint64_t payload_id, uint64_t generation, std::string * err = nullptr);
    xkv_tx_status record_byte_complete(uint64_t tx_id, uint64_t payload_id, std::string * err = nullptr);

    // Postcompute rebase: verifies exact external RERoT view + codec epoch match,
    // permitting private hot/content/live/binding epoch deltas.
    xkv_tx_status rebase(uint64_t tx_id, std::string * err = nullptr);
    xkv_tx_status rebase(uint64_t tx_id, const xkv_snapshot_stamp & cur_stamp, std::string * err = nullptr);

    // Batch settlement: prebuilds every container upfront, preflights one content advance,
    // invokes removal callback out of lock with per-tx identities/generations, accepts
    // feedback store stamp if present, then infallibly advances once/finalizes all via noexcept swaps.
    xkv_tx_status commit_batch(const std::vector<xkv_slot_settlement> & settlements,
                               xkv_batch_remove_rejected_fn remove_rejected,
                               std::string * err = nullptr);

    // Single-slot commit overload (routes cleanly through commit_batch).
    xkv_tx_status commit(uint64_t tx_id,
                         const std::vector<uint64_t> & accepted_ids,
                         xkv_remove_rejected_fn remove_rejected,
                         const xkv_snapshot_stamp & expected_stamp,
                         std::string * err = nullptr);

    // Explicit overload for nullptr removal callback to disambiguate from simple callback.
    xkv_tx_status commit(uint64_t tx_id,
                         const std::vector<uint64_t> & accepted_ids,
                         std::nullptr_t,
                         const xkv_snapshot_stamp & expected_stamp,
                         std::string * err = nullptr) {
        return commit(tx_id, accepted_ids, xkv_remove_rejected_fn(nullptr), expected_stamp, err);
    }

    xkv_tx_status rollback(uint64_t tx_id, xkv_rollback_fn rollback_fn, std::string * err = nullptr);
    xkv_tx_status cancel(uint64_t tx_id, xkv_rollback_fn rollback_fn, std::string * err = nullptr);

    // ---- slot queries ----
    bool has_active_tx() const;
    bool has_active_tx(uint64_t tx_id) const;
    bool has_active_seq(llama_seq_id seq_id) const;
    size_t active_tx_count() const;
    uint64_t active_tx_id_for_seq(llama_seq_id seq_id) const;
    uint64_t active_tx_id() const;

    std::vector<uint64_t> created_payload_ids(uint64_t tx_id) const;
    std::vector<uint64_t> created_payload_ids() const;
    std::vector<uint64_t> accepted_payload_ids(uint64_t tx_id) const;
    std::vector<uint64_t> rejected_payload_ids(uint64_t tx_id) const;

    bool has_committed_publication() const;
    uint64_t committed_tx_id() const;
    std::vector<uint64_t> committed_accepted() const;
    std::vector<uint64_t> committed_rejected() const;
    xkv_snapshot_stamp committed_stamp() const;
    bool is_byte_complete(uint64_t tx_id, uint64_t payload_id) const;
    bool is_byte_complete(uint64_t payload_id) const;

    // ---- readers (immutable pins) ----
    class reader_lease {
    public:
        reader_lease() = default;
        reader_lease(const reader_lease &) = delete;
        reader_lease & operator=(const reader_lease &) = delete;
        reader_lease(reader_lease && o) noexcept;
        reader_lease & operator=(reader_lease && o) noexcept;
        ~reader_lease();
        void release();
        explicit operator bool() const { return coord_ != nullptr; }
        const xkv_snapshot_stamp & pinned_stamp() const { return pinned_; }
    private:
        friend class xkv_transaction_coordinator;
        reader_lease(xkv_transaction_coordinator * coord, const xkv_snapshot_stamp & pinned)
            : coord_(coord), pinned_(pinned) {}
        xkv_transaction_coordinator * coord_ = nullptr;
        xkv_snapshot_stamp pinned_ = {};
    };

    reader_lease acquire_reader();
    size_t active_reader_count() const;
    bool has_quiesce_pending() const;
    xkv_tx_status wait_for_readers_drained(const xkv_quiesce_options & opt, std::string * err = nullptr);

    // ---- maintenance exclusion ----
    xkv_tx_status begin_maintenance(xkv_maintenance_op op, uint64_t maint_id, std::string * err = nullptr);
    xkv_tx_status end_maintenance(uint64_t maint_id, std::string * err = nullptr);
    bool maintenance_allowed() const;
    bool has_active_maintenance() const;

    void add_maintenance_candidate(uint64_t payload_id);
    void remove_maintenance_candidate(uint64_t payload_id);
    bool is_maintenance_candidate(uint64_t payload_id) const;
    size_t maintenance_candidate_count() const;
    bool is_eligible_for_maintenance(uint64_t payload_id) const;
    void retire_payload(uint64_t payload_id);

    // ---- quiescence ----
    xkv_tx_status final_fence(const llama_rerot_view_stamp & stable_view,
                              const xkv_quiesce_options & opt,
                              std::string * err = nullptr);
    xkv_tx_status apply_context_shift(const llama_rerot_view_stamp & shifted_view,
                                      const xkv_quiesce_options & opt,
                                      std::string * err = nullptr);

    xkv_tx_status clear(std::string * err = nullptr);

private:
    struct payload_record {
        uint64_t generation = 0;
        bool byte_complete = false;
    };

    struct active_tx_state {
        uint64_t tx_id = 0;
        llama_seq_id seq_id = 0;
        xkv_snapshot_stamp start_stamp = {};
        xkv_snapshot_stamp rebased_stamp = {};
        bool is_rebased = false;
        std::vector<uint64_t> order;
        std::unordered_map<uint64_t, payload_record> payloads;
        std::vector<uint64_t> accepted;
        std::vector<uint64_t> rejected;
    };

    enum class callback_phase : uint8_t {
        none = 0,
        commit,
        rollback,
        cancel,
    };

    void release_reader();
    friend class reader_lease;

    mutable std::mutex mtx_;
    mutable std::condition_variable cv_;

    std::unique_ptr<xkv_in_memory_stamp_provider> owned_provider_;
    xkv_stamp_provider * provider_ = nullptr;

    std::unordered_map<uint64_t, active_tx_state> active_txs_;
    std::unordered_map<llama_seq_id, uint64_t> seq_to_tx_;

    uint64_t max_tx_id_ = 0;
    callback_phase phase_ = callback_phase::none;
    size_t quiesce_waiters_ = 0;

    bool has_committed_ = false;
    uint64_t committed_tx_id_ = 0;
    std::vector<uint64_t> committed_accepted_;
    std::vector<uint64_t> committed_rejected_;
    xkv_snapshot_stamp committed_stamp_ = {};
    std::unordered_map<uint64_t, bool> committed_byte_complete_;
    std::unordered_set<uint64_t> eligible_;

    size_t active_readers_ = 0;

    bool maintenance_active_ = false;
    xkv_maintenance_op maintenance_op_ = xkv_maintenance_op::seal;
    uint64_t maintenance_id_ = 0;
    std::unordered_set<uint64_t> maintenance_candidates_;
};

// RAII verification guard
class xkv_transaction_guard {
public:
    xkv_transaction_guard() = default;
    xkv_transaction_guard(xkv_transaction_coordinator * coord, uint64_t tx_id,
                          llama_seq_id seq_id,
                          const xkv_snapshot_stamp & snapshot,
                          xkv_rollback_fn rollback_fn = nullptr,
                          std::string * err = nullptr);
    xkv_transaction_guard(xkv_transaction_coordinator * coord, uint64_t tx_id,
                          const xkv_snapshot_stamp & snapshot,
                          xkv_rollback_fn rollback_fn = nullptr,
                          std::string * err = nullptr);
    xkv_transaction_guard(const xkv_transaction_guard &) = delete;
    xkv_transaction_guard & operator=(const xkv_transaction_guard &) = delete;
    xkv_transaction_guard(xkv_transaction_guard && o) noexcept;
    xkv_transaction_guard & operator=(xkv_transaction_guard && o) noexcept;
    ~xkv_transaction_guard();

    xkv_tx_status status() const { return status_; }
    bool begun() const { return status_ == xkv_tx_status::ok; }
    uint64_t tx_id() const { return tx_id_; }
    llama_seq_id seq_id() const { return seq_id_; }

    xkv_tx_status record(uint64_t payload_id, uint64_t generation, std::string * err = nullptr);
    xkv_tx_status byte_complete(uint64_t payload_id, std::string * err = nullptr);
    xkv_tx_status rebase(std::string * err = nullptr);
    xkv_tx_status commit(const std::vector<uint64_t> & accepted_ids,
                         xkv_remove_rejected_fn remove_rejected,
                         const xkv_snapshot_stamp & expected_stamp,
                         std::string * err = nullptr);
    xkv_tx_status commit(const std::vector<uint64_t> & accepted_ids,
                         std::nullptr_t,
                         const xkv_snapshot_stamp & expected_stamp,
                         std::string * err = nullptr) {
        return commit(accepted_ids, xkv_remove_rejected_fn(nullptr), expected_stamp, err);
    }
    xkv_tx_status rollback(std::string * err = nullptr);
    void dismiss() { dismissed_ = true; }

private:
    void settle_owned();
    xkv_transaction_coordinator * coord_ = nullptr;
    uint64_t tx_id_ = 0;
    llama_seq_id seq_id_ = 0;
    xkv_tx_status status_ = xkv_tx_status::no_active_tx;
    xkv_rollback_fn rollback_fn_;
    bool finished_ = false;
    bool dismissed_ = false;
};

// RAII maintenance holder: releases on destruction.
class xkv_maintenance_guard {
public:
    xkv_maintenance_guard() = default;
    xkv_maintenance_guard(xkv_transaction_coordinator * coord, xkv_maintenance_op op, uint64_t maint_id,
                          std::string * err = nullptr);
    xkv_maintenance_guard(const xkv_maintenance_guard &) = delete;
    xkv_maintenance_guard & operator=(const xkv_maintenance_guard &) = delete;
    xkv_maintenance_guard(xkv_maintenance_guard && o) noexcept;
    xkv_maintenance_guard & operator=(xkv_maintenance_guard && o) noexcept;
    ~xkv_maintenance_guard();

    xkv_tx_status status() const { return status_; }
    bool held() const { return held_; }
    void release();

private:
    xkv_transaction_coordinator * coord_ = nullptr;
    uint64_t maint_id_ = 0;
    xkv_tx_status status_ = xkv_tx_status::no_active_tx;
    bool held_ = false;
};

} // namespace llama_xkv
