#include "llama-xkv-transaction.h"

#include <algorithm>
#include <exception>
#include <limits>

namespace llama_xkv {

const char * xkv_tx_status_to_str(xkv_tx_status status) {
    switch (status) {
        case xkv_tx_status::ok:                 return "ok";
        case xkv_tx_status::bad_tx_id:          return "bad_tx_id";
        case xkv_tx_status::already_active:     return "already_active";
        case xkv_tx_status::tx_id_reused:       return "tx_id_reused";
        case xkv_tx_status::tx_mismatch:        return "tx_mismatch";
        case xkv_tx_status::no_active_tx:       return "no_active_tx";
        case xkv_tx_status::stale_stamp:        return "stale_stamp";
        case xkv_tx_status::maintenance_active: return "maintenance_active";
        case xkv_tx_status::tx_active:          return "tx_active";
        case xkv_tx_status::busy:               return "busy";
        case xkv_tx_status::callback_failed:    return "callback_failed";
        case xkv_tx_status::invalid_accept:     return "invalid_accept";
        case xkv_tx_status::cancelled:          return "cancelled";
        case xkv_tx_status::epoch_overflow:     return "epoch_overflow";
    }
    return "unknown";
}

const char * xkv_maintenance_op_to_str(xkv_maintenance_op op) {
    switch (op) {
        case xkv_maintenance_op::seal:             return "seal";
        case xkv_maintenance_op::pack:             return "pack";
        case xkv_maintenance_op::relocate:         return "relocate";
        case xkv_maintenance_op::landmark_publish: return "landmark_publish";
    }
    return "unknown";
}

static void set_err(std::string * err, const std::string & msg) {
    if (err != nullptr) {
        *err = msg;
    }
}

static constexpr uint64_t k_epoch_max = std::numeric_limits<uint64_t>::max();

static bool checked_inc(uint64_t & v, std::string * err, const char * name) {
    if (v == k_epoch_max) {
        set_err(err, std::string(name) + " overflow");
        return false;
    }
    ++v;
    return true;
}

// ============================================================================
// In-memory fake provider implementation (two-phase reservation contract)
// ============================================================================

xkv_snapshot_stamp xkv_in_memory_stamp_provider::current_stamp() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return stamp_;
}

bool xkv_in_memory_stamp_provider::preflight_content_advance(xkv_stamp_reservation * out_res,
                                                             std::string * err) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (fail_content_) {
        set_err(err, "advance_content_epoch: injected failure");
        return false;
    }
    if (stamp_.content_epoch == k_epoch_max) {
        set_err(err, "content_epoch overflow");
        return false;
    }
    if (out_res != nullptr) {
        out_res->token = next_token_++;
        out_res->base_stamp = stamp_;
        out_res->reserved_stamp = stamp_;
        out_res->reserved_stamp.content_epoch += 1;
        out_res->valid = true;
    }
    return true;
}

xkv_snapshot_stamp xkv_in_memory_stamp_provider::apply_content_advance(
    const xkv_stamp_reservation & res,
    const xkv_removal_feedback & feedback) noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    if (res.valid) {
        if (feedback.has_store_stamp) {
            // Incorporate actual store removal updates (e.g. live/binding/content epochs)
            stamp_ = feedback.store_stamp;
        } else {
            stamp_ = res.reserved_stamp;
        }
    }
    return stamp_;
}

void xkv_in_memory_stamp_provider::abort_content_advance(
    const xkv_stamp_reservation & /*res*/) noexcept {
}

bool xkv_in_memory_stamp_provider::install_stable_view(const llama_rerot_view_stamp & stable_view,
                                                      xkv_snapshot_stamp * out_new_stamp,
                                                      std::string * /*err*/) {
    std::lock_guard<std::mutex> lock(mtx_);
    stamp_.view = stable_view;
    if (out_new_stamp != nullptr) {
        *out_new_stamp = stamp_;
    }
    return true;
}

bool xkv_in_memory_stamp_provider::advance_context_shift(const llama_rerot_view_stamp & shifted_view,
                                                        xkv_snapshot_stamp * out_new_stamp,
                                                        std::string * err) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!checked_inc(stamp_.binding_epoch, err, "binding_epoch")) {
        return false;
    }
    stamp_.view = shifted_view;
    if (out_new_stamp != nullptr) {
        *out_new_stamp = stamp_;
    }
    return true;
}

void xkv_in_memory_stamp_provider::set_stamp_for_testing(const xkv_snapshot_stamp & s) {
    std::lock_guard<std::mutex> lock(mtx_);
    stamp_ = s;
}

// ============================================================================
// Multi-slot coordinator implementation
// ============================================================================

xkv_transaction_coordinator::xkv_transaction_coordinator() {
    owned_provider_ = std::make_unique<xkv_in_memory_stamp_provider>();
    provider_ = owned_provider_.get();
}

xkv_transaction_coordinator::xkv_transaction_coordinator(xkv_stamp_provider * provider) {
    if (provider == nullptr) {
        owned_provider_ = std::make_unique<xkv_in_memory_stamp_provider>();
        provider_ = owned_provider_.get();
    } else {
        provider_ = provider;
    }
}

xkv_stamp_provider * xkv_transaction_coordinator::stamp_provider() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return provider_;
}

xkv_tx_status xkv_transaction_coordinator::bind_stamp_provider(xkv_stamp_provider * provider,
                                                               std::string * err) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!active_txs_.empty()) {
        set_err(err, "bind_stamp_provider: verification transaction active");
        return xkv_tx_status::busy;
    }
    if (phase_ != callback_phase::none) {
        set_err(err, "bind_stamp_provider: removal callback in flight");
        return xkv_tx_status::busy;
    }
    if (active_readers_ > 0) {
        set_err(err, "bind_stamp_provider: readers still pinned");
        return xkv_tx_status::busy;
    }
    if (maintenance_active_) {
        set_err(err, "bind_stamp_provider: maintenance active");
        return xkv_tx_status::busy;
    }
    if (quiesce_waiters_ > 0) {
        set_err(err, "bind_stamp_provider: quiescence waiter outstanding");
        return xkv_tx_status::busy;
    }
    if (provider != nullptr) {
        provider_ = provider;
    } else {
        if (!owned_provider_) {
            owned_provider_ = std::make_unique<xkv_in_memory_stamp_provider>();
        }
        provider_ = owned_provider_.get();
    }
    cv_.notify_all();
    return xkv_tx_status::ok;
}

xkv_snapshot_stamp xkv_transaction_coordinator::current_stamp() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return provider_->current_stamp();
}

bool xkv_transaction_coordinator::is_stale(const xkv_snapshot_stamp & expected, std::string * reason) const {
    xkv_snapshot_stamp cur = current_stamp();
    if (expected != cur) {
        if (reason != nullptr) {
            if (expected.view.topology_epoch != cur.view.topology_epoch) {
                *reason = "stale view.topology_epoch";
            } else if (expected.view.publish_epoch != cur.view.publish_epoch) {
                *reason = "stale view.publish_epoch";
            } else if (expected.view.layout_epoch != cur.view.layout_epoch) {
                *reason = "stale view.layout_epoch";
            } else if (expected.live_epoch != cur.live_epoch) {
                *reason = "stale live_epoch";
            } else if (expected.content_epoch != cur.content_epoch) {
                *reason = "stale content_epoch";
            } else if (expected.codec_epoch != cur.codec_epoch) {
                *reason = "stale codec_epoch";
            } else if (expected.binding_epoch != cur.binding_epoch) {
                *reason = "stale binding_epoch";
            } else {
                *reason = "stale stamp";
            }
        }
        return true;
    }
    return false;
}

void xkv_transaction_coordinator::observe_store_stamp(const xkv_snapshot_stamp & /*observed*/) {
    std::lock_guard<std::mutex> lock(mtx_);
    cv_.notify_all();
}

void xkv_transaction_coordinator::sync_from_store(const xkv_snapshot_stamp & store_stamp) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (owned_provider_) {
        owned_provider_->set_stamp_for_testing(store_stamp);
    }
    cv_.notify_all();
}

// ---- multi-slot transaction lifecycle ----

xkv_tx_status xkv_transaction_coordinator::begin(uint64_t tx_id, llama_seq_id seq_id,
                                                 const xkv_snapshot_stamp & snapshot,
                                                 std::string * err) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (tx_id == 0) {
        set_err(err, "begin: tx_id 0 is reserved");
        return xkv_tx_status::bad_tx_id;
    }
    if (active_txs_.count(tx_id) != 0) {
        set_err(err, "begin: nested/overlapping transaction id rejected");
        return xkv_tx_status::already_active;
    }
    if (seq_to_tx_.count(seq_id) != 0) {
        set_err(err, "begin: sequence slot already has an active transaction");
        return xkv_tx_status::already_active;
    }
    if (phase_ != callback_phase::none) {
        set_err(err, "begin: removal callback in flight");
        return xkv_tx_status::busy;
    }
    if (quiesce_waiters_ > 0) {
        set_err(err, "begin: quiescence pending");
        return xkv_tx_status::busy;
    }
    if (tx_id <= max_tx_id_) {
        set_err(err, "begin: transaction id not monotonic");
        return xkv_tx_status::tx_id_reused;
    }
    if (maintenance_active_) {
        set_err(err, "begin: maintenance in progress");
        return xkv_tx_status::maintenance_active;
    }
    xkv_snapshot_stamp cur = provider_->current_stamp();
    if (snapshot != cur) {
        set_err(err, "begin: stale snapshot stamp");
        return xkv_tx_status::stale_stamp;
    }

    try {
        active_tx_state st;
        st.tx_id = tx_id;
        st.seq_id = seq_id;
        st.start_stamp = snapshot;
        st.rebased_stamp = snapshot;
        st.is_rebased = false;

        auto em_res = active_txs_.emplace(tx_id, std::move(st));
        try {
            seq_to_tx_.emplace(seq_id, tx_id);
        } catch (...) {
            active_txs_.erase(em_res.first);
            throw;
        }
        max_tx_id_ = tx_id;
    } catch (const std::exception & e) {
        set_err(err, std::string("begin allocation failed: ") + e.what());
        return xkv_tx_status::callback_failed;
    } catch (...) {
        set_err(err, "begin allocation failed: unknown exception");
        return xkv_tx_status::callback_failed;
    }

    return xkv_tx_status::ok;
}

xkv_tx_status xkv_transaction_coordinator::begin(uint64_t tx_id, const xkv_snapshot_stamp & snapshot,
                                                 std::string * err) {
    return begin(tx_id, 0, snapshot, err);
}

xkv_tx_status xkv_transaction_coordinator::record_payload(uint64_t tx_id, uint64_t payload_id,
                                                          uint64_t generation, std::string * err) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = active_txs_.find(tx_id);
    if (it == active_txs_.end()) {
        set_err(err, "record_payload: no active transaction for tx_id");
        return xkv_tx_status::no_active_tx;
    }
    if (phase_ != callback_phase::none) {
        set_err(err, "record_payload: removal callback in flight");
        return xkv_tx_status::busy;
    }
    if (it->second.payloads.count(payload_id) != 0) {
        set_err(err, "record_payload: duplicate payload id");
        return xkv_tx_status::invalid_accept;
    }

    try {
        payload_record rec;
        rec.generation = generation;
        rec.byte_complete = false;

        auto pit = it->second.payloads.emplace(payload_id, rec);
        try {
            it->second.order.push_back(payload_id);
        } catch (...) {
            it->second.payloads.erase(pit.first);
            throw;
        }
    } catch (const std::exception & e) {
        set_err(err, std::string("record_payload allocation failed: ") + e.what());
        return xkv_tx_status::callback_failed;
    } catch (...) {
        set_err(err, "record_payload allocation failed: unknown exception");
        return xkv_tx_status::callback_failed;
    }

    return xkv_tx_status::ok;
}

xkv_tx_status xkv_transaction_coordinator::record_byte_complete(uint64_t tx_id, uint64_t payload_id,
                                                                std::string * err) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = active_txs_.find(tx_id);
    if (it == active_txs_.end()) {
        set_err(err, "record_byte_complete: no active transaction for tx_id");
        return xkv_tx_status::no_active_tx;
    }
    if (phase_ != callback_phase::none) {
        set_err(err, "record_byte_complete: removal callback in flight");
        return xkv_tx_status::busy;
    }
    auto pit = it->second.payloads.find(payload_id);
    if (pit == it->second.payloads.end()) {
        set_err(err, "record_byte_complete: unknown payload id");
        return xkv_tx_status::invalid_accept;
    }
    pit->second.byte_complete = true;
    return xkv_tx_status::ok;
}

// Postcompute rebase: verifies exact external RERoT view + codec epoch match,
// permitting private hot/content/live/binding epoch deltas.
xkv_tx_status xkv_transaction_coordinator::rebase(uint64_t tx_id, const xkv_snapshot_stamp & cur_stamp,
                                                  std::string * err) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = active_txs_.find(tx_id);
    if (it == active_txs_.end()) {
        set_err(err, "rebase: no active transaction for tx_id");
        return xkv_tx_status::no_active_tx;
    }
    if (phase_ != callback_phase::none) {
        set_err(err, "rebase: removal callback in flight");
        return xkv_tx_status::busy;
    }
    const auto & s = it->second.start_stamp;
    if (s.view.topology_epoch != cur_stamp.view.topology_epoch ||
        s.view.publish_epoch != cur_stamp.view.publish_epoch ||
        s.view.layout_epoch != cur_stamp.view.layout_epoch) {
        set_err(err, "rebase failed: external RERoT view stamp mismatch");
        return xkv_tx_status::stale_stamp;
    }
    if (s.codec_epoch != cur_stamp.codec_epoch) {
        set_err(err, "rebase failed: codec epoch mismatch");
        return xkv_tx_status::stale_stamp;
    }
    it->second.rebased_stamp = cur_stamp;
    it->second.is_rebased = true;
    return xkv_tx_status::ok;
}

xkv_tx_status xkv_transaction_coordinator::rebase(uint64_t tx_id, std::string * err) {
    return rebase(tx_id, current_stamp(), err);
}

// Batch settlement engine: prebuilds every container upfront, preflights one content advance,
// invokes removal callback out of lock with per-tx identities/generations, accepts
// feedback store stamp if present, then infallibly advances once/finalizes all via noexcept swaps.
xkv_tx_status xkv_transaction_coordinator::commit_batch(
    const std::vector<xkv_slot_settlement> & settlements,
    xkv_batch_remove_rejected_fn remove_rejected,
    std::string * err) {
    if (settlements.empty()) {
        set_err(err, "commit_batch: empty settlement list");
        return xkv_tx_status::invalid_accept;
    }

    struct slot_preflight {
        uint64_t tx_id = 0;
        llama_seq_id seq_id = 0;
        std::vector<uint64_t> accepted;
        std::vector<xkv_rejected_item> rejected_items;
        std::vector<uint64_t> snap_order;
        std::vector<uint64_t> snap_generations;
    };

    std::vector<slot_preflight> preflighted_slots;
    std::vector<xkv_rejected_item> all_rejected_items;
    xkv_snapshot_stamp commit_stamp = {};
    xkv_stamp_reservation reservation = {};

    // PREBUILT zero-allocation tail containers: allocated BEFORE callback is invoked!
    std::vector<uint64_t> prebuilt_accepted;
    std::vector<uint64_t> prebuilt_rejected;
    std::unordered_set<uint64_t> prebuilt_eligible_delta;
    std::unordered_map<uint64_t, bool> prebuilt_byte_complete;

    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (phase_ != callback_phase::none) {
            set_err(err, "commit_batch: removal callback in flight");
            return xkv_tx_status::busy;
        }

        xkv_snapshot_stamp cur = provider_->current_stamp();
        std::unordered_set<uint64_t> seen_tx;

        try {
            preflighted_slots.reserve(settlements.size());

            for (const auto & s : settlements) {
                if (seen_tx.count(s.tx_id) != 0) {
                    set_err(err, "commit_batch: duplicate tx_id in settlement");
                    return xkv_tx_status::invalid_accept;
                }
                seen_tx.insert(s.tx_id);

                auto it = active_txs_.find(s.tx_id);
                if (it == active_txs_.end()) {
                    set_err(err, "commit_batch: unknown active tx_id: " + std::to_string(s.tx_id));
                    return xkv_tx_status::no_active_tx;
                }
                const auto & tx_st = it->second;

                if (tx_st.is_rebased) {
                    if (tx_st.start_stamp.view.topology_epoch != cur.view.topology_epoch ||
                        tx_st.start_stamp.view.publish_epoch != cur.view.publish_epoch ||
                        tx_st.start_stamp.view.layout_epoch != cur.view.layout_epoch ||
                        tx_st.start_stamp.codec_epoch != cur.codec_epoch) {
                        set_err(err, "commit_batch: stale external stamp for tx " + std::to_string(s.tx_id));
                        return xkv_tx_status::stale_stamp;
                    }
                } else {
                    if (tx_st.start_stamp != cur) {
                        set_err(err, "commit_batch: stale start stamp for tx " + std::to_string(s.tx_id));
                        return xkv_tx_status::stale_stamp;
                    }
                }

                if (s.accepted_ids.size() > tx_st.order.size()) {
                    set_err(err, "commit_batch: accepted set larger than recorded order for tx " + std::to_string(s.tx_id));
                    return xkv_tx_status::invalid_accept;
                }
                for (size_t i = 0; i < s.accepted_ids.size(); ++i) {
                    if (s.accepted_ids[i] != tx_st.order[i]) {
                        set_err(err, "commit_batch: accepted set is not the recorded prefix for tx " + std::to_string(s.tx_id));
                        return xkv_tx_status::invalid_accept;
                    }
                }
                for (uint64_t id : s.accepted_ids) {
                    auto pit = tx_st.payloads.find(id);
                    if (pit == tx_st.payloads.end()) {
                        set_err(err, "commit_batch: accepted id not recorded for tx " + std::to_string(s.tx_id));
                        return xkv_tx_status::invalid_accept;
                    }
                    if (!pit->second.byte_complete) {
                        set_err(err, "commit_batch: accepted payload not byte-complete for tx " + std::to_string(s.tx_id));
                        return xkv_tx_status::invalid_accept;
                    }
                }

                slot_preflight pf;
                pf.tx_id = s.tx_id;
                pf.seq_id = tx_st.seq_id;
                pf.accepted = s.accepted_ids;
                pf.snap_order = tx_st.order;
                pf.snap_generations.reserve(tx_st.order.size());
                for (uint64_t id : tx_st.order) {
                    pf.snap_generations.push_back(tx_st.payloads.at(id).generation);
                }

                for (size_t i = s.accepted_ids.size(); i < tx_st.order.size(); ++i) {
                    uint64_t pid = tx_st.order[i];
                    xkv_rejected_item item;
                    item.tx_id = s.tx_id;
                    item.seq_id = tx_st.seq_id;
                    item.payload_id = pid;
                    item.generation = tx_st.payloads.at(pid).generation;
                    pf.rejected_items.push_back(item);
                    all_rejected_items.push_back(item);
                }

                preflighted_slots.push_back(std::move(pf));
            }

            if (!all_rejected_items.empty() && !remove_rejected) {
                set_err(err, "commit_batch: non-empty rejected items require a removal callback");
                return xkv_tx_status::callback_failed;
            }

            // Prebuild all tail publication state now, so after the callback
            // succeeds NO heap allocation can fail!
            for (const auto & pf : preflighted_slots) {
                for (uint64_t id : pf.accepted) {
                    prebuilt_accepted.push_back(id);
                    prebuilt_eligible_delta.insert(id);
                    prebuilt_byte_complete[id] = true;
                }
                for (const auto & r : pf.rejected_items) {
                    prebuilt_rejected.push_back(r.payload_id);
                }
            }
        } catch (const std::exception & e) {
            set_err(err, std::string("commit_batch prebuild allocation failed: ") + e.what());
            return xkv_tx_status::callback_failed;
        } catch (...) {
            set_err(err, "commit_batch prebuild allocation failed: unknown exception");
            return xkv_tx_status::callback_failed;
        }

        // Preflight one content advance for the whole batch.
        // If preflight fails (e.g. overflow, refusal), commit immediately
        // aborts: callback is NEVER invoked, exactly zero removals take place.
        std::string pre_err;
        if (!provider_->preflight_content_advance(&reservation, &pre_err)) {
            set_err(err, pre_err.empty() ? "commit_batch: content advance preflight failed" : pre_err);
            if (pre_err.find("overflow") != std::string::npos) {
                return xkv_tx_status::epoch_overflow;
            }
            return xkv_tx_status::callback_failed;
        }

        commit_stamp = cur;
        phase_ = callback_phase::commit;
    }

    // Phase 2: Removal callback runs WITHOUT the coordinator mutex.
    bool cb_ok = true;
    std::string cb_err;
    xkv_removal_feedback feedback;

    if (!all_rejected_items.empty()) {
        try {
            cb_ok = remove_rejected(all_rejected_items, reservation.token, &feedback, &cb_err);
        } catch (const std::exception & ex) {
            cb_ok = false;
            cb_err = ex.what();
        } catch (...) {
            cb_ok = false;
            cb_err = "commit_batch: remove_rejected threw";
        }
    }

    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (phase_ != callback_phase::commit) {
            provider_->abort_content_advance(reservation);
            phase_ = callback_phase::none;
            cv_.notify_all();
            set_err(err, "commit_batch: phase cancelled during callback");
            return xkv_tx_status::cancelled;
        }

        bool slots_intact = true;
        for (const auto & pf : preflighted_slots) {
            auto it = active_txs_.find(pf.tx_id);
            if (it == active_txs_.end() || it->second.order != pf.snap_order) {
                slots_intact = false;
                break;
            }
            for (size_t i = 0; i < it->second.order.size(); ++i) {
                auto pit = it->second.payloads.find(it->second.order[i]);
                if (pit == it->second.payloads.end() || pit->second.generation != pf.snap_generations[i]) {
                    slots_intact = false;
                    break;
                }
            }
            if (!slots_intact) break;
        }

        if (!slots_intact) {
            provider_->abort_content_advance(reservation);
            phase_ = callback_phase::none;
            cv_.notify_all();
            set_err(err, "commit_batch: active slots mutated during removal callback");
            return xkv_tx_status::cancelled;
        }

        xkv_snapshot_stamp cur = provider_->current_stamp();
        // If removal callback gave a store stamp (from real store remove_payloads),
        // cur will match that store stamp rather than commit_stamp.
        if (feedback.has_store_stamp) {
            if (cur != feedback.store_stamp) {
                provider_->abort_content_advance(reservation);
                phase_ = callback_phase::none;
                cv_.notify_all();
                set_err(err, "commit_batch: external stamp changed concurrently with removal callback");
                return xkv_tx_status::stale_stamp;
            }
        } else {
            if (cur != commit_stamp) {
                provider_->abort_content_advance(reservation);
                phase_ = callback_phase::none;
                cv_.notify_all();
                set_err(err, "commit_batch: stamp changed during removal callback");
                return xkv_tx_status::stale_stamp;
            }
        }

        if (!cb_ok) {
            provider_->abort_content_advance(reservation);
            phase_ = callback_phase::none;
            cv_.notify_all();
            set_err(err, cb_err.empty() ? "commit_batch: removal callback failed" : cb_err);
            return xkv_tx_status::callback_failed;
        }

        // Phase 3: Infallible commit of content advance reservation token.
        xkv_snapshot_stamp new_stamp = provider_->apply_content_advance(reservation, feedback);

        // ZERO-ALLOCATION TAIL: all containers were prebuilt/prereserved upfront!
        committed_accepted_ = std::move(prebuilt_accepted);
        committed_rejected_ = std::move(prebuilt_rejected);
        committed_byte_complete_ = std::move(prebuilt_byte_complete);
        for (uint64_t id : prebuilt_eligible_delta) {
            eligible_.insert(id);
        }

        for (const auto & pf : preflighted_slots) {
            auto it = active_txs_.find(pf.tx_id);
            if (it != active_txs_.end()) {
                seq_to_tx_.erase(it->second.seq_id);
                active_txs_.erase(it);
            }
        }

        has_committed_ = true;
        committed_tx_id_ = preflighted_slots.front().tx_id;
        committed_stamp_ = new_stamp;
        phase_ = callback_phase::none;
        cv_.notify_all();
        return xkv_tx_status::ok;
    }
}

// Single-slot commit overload (routes cleanly through commit_batch).
xkv_tx_status xkv_transaction_coordinator::commit(uint64_t tx_id,
                                                  const std::vector<uint64_t> & accepted_ids,
                                                  xkv_remove_rejected_fn remove_rejected,
                                                  const xkv_snapshot_stamp & expected_stamp,
                                                  std::string * err) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = active_txs_.find(tx_id);
        if (it == active_txs_.end()) {
            set_err(err, "commit: no active transaction for tx_id");
            return xkv_tx_status::no_active_tx;
        }
        xkv_snapshot_stamp cur = provider_->current_stamp();
        if (it->second.is_rebased) {
            if (it->second.start_stamp.view.topology_epoch != cur.view.topology_epoch ||
                it->second.start_stamp.view.publish_epoch != cur.view.publish_epoch ||
                it->second.start_stamp.view.layout_epoch != cur.view.layout_epoch ||
                it->second.start_stamp.codec_epoch != cur.codec_epoch) {
                set_err(err, "commit: stale external stamp");
                return xkv_tx_status::stale_stamp;
            }
        } else {
            if (expected_stamp != cur || it->second.start_stamp != cur) {
                set_err(err, "commit: stale stamp");
                return xkv_tx_status::stale_stamp;
            }
        }
    }

    xkv_slot_settlement slot;
    slot.tx_id = tx_id;
    slot.accepted_ids = accepted_ids;

    xkv_batch_remove_rejected_fn batch_cb = nullptr;
    if (remove_rejected) {
        batch_cb = [remove_rejected](const std::vector<xkv_rejected_item> & items,
                                     uint64_t reservation_token,
                                     xkv_removal_feedback * fb,
                                     std::string * e) {
            std::vector<uint64_t> pids;
            pids.reserve(items.size());
            for (const auto & item : items) {
                pids.push_back(item.payload_id);
            }
            return remove_rejected(pids, reservation_token, fb, e);
        };
    }

    return commit_batch({slot}, batch_cb, err);
}

// Rollback removes tentative private hot rows without touching RERoT view.publish_epoch.
xkv_tx_status xkv_transaction_coordinator::rollback(uint64_t tx_id, xkv_rollback_fn rollback_fn,
                                                    std::string * err) {
    std::vector<uint64_t> created;
    std::vector<uint64_t> snap_generations;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = active_txs_.find(tx_id);
        if (it == active_txs_.end()) {
            set_err(err, "rollback: no active transaction for tx_id");
            return xkv_tx_status::no_active_tx;
        }
        if (phase_ != callback_phase::none) {
            set_err(err, "rollback: removal callback in flight");
            return xkv_tx_status::busy;
        }
        created = it->second.order;
        if (!created.empty() && !rollback_fn) {
            set_err(err, "rollback: created rows require a removal callback");
            return xkv_tx_status::callback_failed;
        }
        try {
            snap_generations.reserve(created.size());
            for (uint64_t id : created) {
                snap_generations.push_back(it->second.payloads.at(id).generation);
            }
        } catch (const std::exception & e) {
            set_err(err, std::string("rollback allocation failed: ") + e.what());
            return xkv_tx_status::callback_failed;
        }
        phase_ = callback_phase::rollback;
    }

    bool cb_ok = true;
    std::string cb_err;
    if (!created.empty()) {
        try {
            cb_ok = rollback_fn(created, &cb_err);
        } catch (const std::exception & ex) {
            cb_ok = false;
            cb_err = ex.what();
        } catch (...) {
            cb_ok = false;
            cb_err = "rollback callback threw";
        }
    }

    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (phase_ != callback_phase::rollback) {
            phase_ = callback_phase::none;
            cv_.notify_all();
            set_err(err, "rollback: phase cancelled during callback");
            return xkv_tx_status::cancelled;
        }
        auto it = active_txs_.find(tx_id);
        if (it == active_txs_.end()) {
            phase_ = callback_phase::none;
            cv_.notify_all();
            set_err(err, "rollback: transaction closed during callback");
            return xkv_tx_status::cancelled;
        }
        bool order_same = (it->second.order == created);
        if (order_same) {
            for (size_t i = 0; i < it->second.order.size(); ++i) {
                auto pit = it->second.payloads.find(it->second.order[i]);
                if (pit == it->second.payloads.end() || pit->second.generation != snap_generations[i]) {
                    order_same = false;
                    break;
                }
            }
        }
        if (!order_same) {
            phase_ = callback_phase::none;
            cv_.notify_all();
            set_err(err, "rollback: recorded order mutated during callback");
            return xkv_tx_status::cancelled;
        }
        if (!cb_ok) {
            phase_ = callback_phase::none;
            cv_.notify_all();
            set_err(err, cb_err.empty() ? "rollback: removal callback failed" : cb_err);
            return xkv_tx_status::callback_failed;
        }

        seq_to_tx_.erase(it->second.seq_id);
        active_txs_.erase(it);
        phase_ = callback_phase::none;
        cv_.notify_all();
        return xkv_tx_status::ok;
    }
}

// Cancel removes tentative private hot rows without advancing RERoT view.publish_epoch.
xkv_tx_status xkv_transaction_coordinator::cancel(uint64_t tx_id, xkv_rollback_fn rollback_fn,
                                                  std::string * err) {
    return rollback(tx_id, rollback_fn, err);
}

// ---- slot queries ----

bool xkv_transaction_coordinator::has_active_tx() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return !active_txs_.empty();
}

bool xkv_transaction_coordinator::has_active_tx(uint64_t tx_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    return active_txs_.count(tx_id) != 0;
}

bool xkv_transaction_coordinator::has_active_seq(llama_seq_id seq_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    return seq_to_tx_.count(seq_id) != 0;
}

size_t xkv_transaction_coordinator::active_tx_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return active_txs_.size();
}

uint64_t xkv_transaction_coordinator::active_tx_id_for_seq(llama_seq_id seq_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = seq_to_tx_.find(seq_id);
    return (it != seq_to_tx_.end()) ? it->second : 0;
}

uint64_t xkv_transaction_coordinator::active_tx_id() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return active_txs_.empty() ? 0 : active_txs_.begin()->first;
}

std::vector<uint64_t> xkv_transaction_coordinator::created_payload_ids(uint64_t tx_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = active_txs_.find(tx_id);
    return (it != active_txs_.end()) ? it->second.order : std::vector<uint64_t>{};
}

std::vector<uint64_t> xkv_transaction_coordinator::created_payload_ids() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return active_txs_.empty() ? std::vector<uint64_t>{} : active_txs_.begin()->second.order;
}

std::vector<uint64_t> xkv_transaction_coordinator::accepted_payload_ids(uint64_t tx_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = active_txs_.find(tx_id);
    return (it != active_txs_.end()) ? it->second.accepted : std::vector<uint64_t>{};
}

std::vector<uint64_t> xkv_transaction_coordinator::rejected_payload_ids(uint64_t tx_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = active_txs_.find(tx_id);
    return (it != active_txs_.end()) ? it->second.rejected : std::vector<uint64_t>{};
}

bool xkv_transaction_coordinator::has_committed_publication() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return has_committed_;
}

uint64_t xkv_transaction_coordinator::committed_tx_id() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return committed_tx_id_;
}

std::vector<uint64_t> xkv_transaction_coordinator::committed_accepted() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return committed_accepted_;
}

std::vector<uint64_t> xkv_transaction_coordinator::committed_rejected() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return committed_rejected_;
}

xkv_snapshot_stamp xkv_transaction_coordinator::committed_stamp() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return committed_stamp_;
}

bool xkv_transaction_coordinator::is_byte_complete(uint64_t tx_id, uint64_t payload_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = active_txs_.find(tx_id);
    if (it == active_txs_.end()) return false;
    auto pit = it->second.payloads.find(payload_id);
    return (pit != it->second.payloads.end()) && pit->second.byte_complete;
}

bool xkv_transaction_coordinator::is_byte_complete(uint64_t payload_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    for (const auto & kv : active_txs_) {
        auto pit = kv.second.payloads.find(payload_id);
        if (pit != kv.second.payloads.end() && pit->second.byte_complete) return true;
    }
    return false;
}

// ---- readers ----

xkv_transaction_coordinator::reader_lease::reader_lease(reader_lease && o) noexcept
    : coord_(o.coord_), pinned_(o.pinned_) {
    o.coord_ = nullptr;
}

xkv_transaction_coordinator::reader_lease &
xkv_transaction_coordinator::reader_lease::operator=(reader_lease && o) noexcept {
    if (this != &o) {
        release();
        coord_ = o.coord_;
        pinned_ = o.pinned_;
        o.coord_ = nullptr;
    }
    return *this;
}

xkv_transaction_coordinator::reader_lease::~reader_lease() {
    release();
}

void xkv_transaction_coordinator::reader_lease::release() {
    if (coord_ != nullptr) {
        coord_->release_reader();
        coord_ = nullptr;
    }
}

xkv_transaction_coordinator::reader_lease xkv_transaction_coordinator::acquire_reader() {
    std::unique_lock<std::mutex> lock(mtx_);
    if (phase_ == callback_phase::none) {
        cv_.wait(lock, [this] { return quiesce_waiters_ == 0; });
    }
    ++active_readers_;
    return reader_lease(this, provider_->current_stamp());
}

void xkv_transaction_coordinator::release_reader() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (active_readers_ > 0) {
        --active_readers_;
    }
    cv_.notify_all();
}

size_t xkv_transaction_coordinator::active_reader_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return active_readers_;
}

bool xkv_transaction_coordinator::has_quiesce_pending() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return quiesce_waiters_ > 0;
}

xkv_tx_status xkv_transaction_coordinator::wait_for_readers_drained(const xkv_quiesce_options & opt,
                                                                     std::string * err) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (!opt.wait) {
        if (active_readers_ > 0) {
            set_err(err, "readers still active");
            return xkv_tx_status::busy;
        }
        return xkv_tx_status::ok;
    }
    ++quiesce_waiters_;
    cv_.wait(lock, [this] { return active_readers_ == 0; });
    --quiesce_waiters_;
    cv_.notify_all();
    return xkv_tx_status::ok;
}

// ---- maintenance ----

xkv_tx_status xkv_transaction_coordinator::begin_maintenance(xkv_maintenance_op op, uint64_t maint_id,
                                                             std::string * err) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (maint_id == 0) {
        set_err(err, "begin_maintenance: id 0 is reserved");
        return xkv_tx_status::bad_tx_id;
    }
    if (!active_txs_.empty() || phase_ != callback_phase::none) {
        set_err(err, "begin_maintenance: verification/frontier commit active");
        return xkv_tx_status::tx_active;
    }
    if (maintenance_active_) {
        set_err(err, "begin_maintenance: maintenance already active");
        return xkv_tx_status::busy;
    }
    maintenance_active_ = true;
    maintenance_op_ = op;
    maintenance_id_ = maint_id;
    return xkv_tx_status::ok;
}

xkv_tx_status xkv_transaction_coordinator::end_maintenance(uint64_t maint_id, std::string * err) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!maintenance_active_) {
        set_err(err, "end_maintenance: no active maintenance");
        return xkv_tx_status::no_active_tx;
    }
    if (maint_id != maintenance_id_) {
        set_err(err, "end_maintenance: id mismatch");
        return xkv_tx_status::tx_mismatch;
    }
    maintenance_active_ = false;
    maintenance_id_ = 0;
    cv_.notify_all();
    return xkv_tx_status::ok;
}

bool xkv_transaction_coordinator::maintenance_allowed() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return active_txs_.empty() && !maintenance_active_;
}

bool xkv_transaction_coordinator::has_active_maintenance() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return maintenance_active_;
}

void xkv_transaction_coordinator::add_maintenance_candidate(uint64_t payload_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    maintenance_candidates_.insert(payload_id);
}

void xkv_transaction_coordinator::remove_maintenance_candidate(uint64_t payload_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    maintenance_candidates_.erase(payload_id);
}

bool xkv_transaction_coordinator::is_maintenance_candidate(uint64_t payload_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    return maintenance_candidates_.count(payload_id) != 0;
}

size_t xkv_transaction_coordinator::maintenance_candidate_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return maintenance_candidates_.size();
}

bool xkv_transaction_coordinator::is_eligible_for_maintenance(uint64_t payload_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    return eligible_.count(payload_id) != 0;
}

void xkv_transaction_coordinator::retire_payload(uint64_t payload_id) {
    std::lock_guard<std::mutex> lock(mtx_);
    eligible_.erase(payload_id);
    maintenance_candidates_.erase(payload_id);
    committed_byte_complete_.erase(payload_id);
    for (auto it = committed_accepted_.begin(); it != committed_accepted_.end();) {
        it = (*it == payload_id) ? committed_accepted_.erase(it) : (it + 1);
    }
    for (auto it = committed_rejected_.begin(); it != committed_rejected_.end();) {
        it = (*it == payload_id) ? committed_rejected_.erase(it) : (it + 1);
    }
    cv_.notify_all();
}

// ---- quiescence ----

xkv_tx_status xkv_transaction_coordinator::final_fence(const llama_rerot_view_stamp & stable_view,
                                                       const xkv_quiesce_options & opt,
                                                       std::string * err) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (phase_ != callback_phase::none) {
        set_err(err, "final_fence: removal callback in flight");
        return xkv_tx_status::busy;
    }
    if (!opt.wait) {
        if (!active_txs_.empty()) {
            set_err(err, "final_fence: verification transaction active");
            return xkv_tx_status::tx_active;
        }
        if (active_readers_ > 0 || maintenance_active_) {
            set_err(err, "final_fence: readers or factor candidates in flight");
            return xkv_tx_status::busy;
        }
        xkv_snapshot_stamp new_stamp = {};
        if (!provider_->install_stable_view(stable_view, &new_stamp, err)) {
            return xkv_tx_status::callback_failed;
        }
        cv_.notify_all();
        return xkv_tx_status::ok;
    }
    ++quiesce_waiters_;
    cv_.wait(lock, [this] {
        return active_readers_ == 0 && !maintenance_active_ && active_txs_.empty() &&
               phase_ == callback_phase::none;
    });
    --quiesce_waiters_;
    xkv_snapshot_stamp new_stamp = {};
    if (!provider_->install_stable_view(stable_view, &new_stamp, err)) {
        cv_.notify_all();
        return xkv_tx_status::callback_failed;
    }
    cv_.notify_all();
    return xkv_tx_status::ok;
}

xkv_tx_status xkv_transaction_coordinator::apply_context_shift(const llama_rerot_view_stamp & shifted_view,
                                                               const xkv_quiesce_options & opt,
                                                               std::string * err) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (phase_ != callback_phase::none) {
        set_err(err, "context_shift: removal callback in flight");
        return xkv_tx_status::busy;
    }
    if (!opt.wait) {
        if (!active_txs_.empty()) {
            set_err(err, "context_shift: verification transaction active");
            return xkv_tx_status::tx_active;
        }
        if (active_readers_ > 0 || maintenance_active_) {
            set_err(err, "context_shift: readers or maintenance in flight");
            return xkv_tx_status::busy;
        }
        xkv_snapshot_stamp new_stamp = {};
        std::string prov_err;
        if (!provider_->advance_context_shift(shifted_view, &new_stamp, &prov_err)) {
            set_err(err, prov_err);
            if (prov_err.find("overflow") != std::string::npos) {
                return xkv_tx_status::epoch_overflow;
            }
            return xkv_tx_status::callback_failed;
        }
        cv_.notify_all();
        return xkv_tx_status::ok;
    }
    ++quiesce_waiters_;
    cv_.wait(lock, [this] {
        return active_readers_ == 0 && !maintenance_active_ && active_txs_.empty() &&
               phase_ == callback_phase::none;
    });
    --quiesce_waiters_;
    xkv_snapshot_stamp new_stamp = {};
    std::string prov_err;
    if (!provider_->advance_context_shift(shifted_view, &new_stamp, &prov_err)) {
        cv_.notify_all();
        set_err(err, prov_err);
        if (prov_err.find("overflow") != std::string::npos) {
            return xkv_tx_status::epoch_overflow;
        }
        return xkv_tx_status::callback_failed;
    }
    cv_.notify_all();
    return xkv_tx_status::ok;
}

xkv_tx_status xkv_transaction_coordinator::clear(std::string * err) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (phase_ != callback_phase::none) {
        set_err(err, "clear: removal callback in flight");
        return xkv_tx_status::busy;
    }
    if (!active_txs_.empty()) {
        set_err(err, "clear: transactions still active");
        return xkv_tx_status::busy;
    }
    if (active_readers_ > 0) {
        set_err(err, "clear: readers still pinned");
        return xkv_tx_status::busy;
    }
    if (maintenance_active_) {
        set_err(err, "clear: maintenance still active");
        return xkv_tx_status::busy;
    }
    if (quiesce_waiters_ > 0) {
        set_err(err, "clear: quiescence waiter outstanding");
        return xkv_tx_status::busy;
    }
    active_txs_.clear();
    seq_to_tx_.clear();
    max_tx_id_ = 0;
    has_committed_ = false;
    committed_tx_id_ = 0;
    committed_accepted_.clear();
    committed_rejected_.clear();
    committed_stamp_ = {};
    committed_byte_complete_.clear();
    eligible_.clear();
    active_readers_ = 0;
    maintenance_active_ = false;
    maintenance_id_ = 0;
    maintenance_candidates_.clear();
    cv_.notify_all();
    return xkv_tx_status::ok;
}

// ---- guards ----

xkv_transaction_guard::xkv_transaction_guard(xkv_transaction_coordinator * coord, uint64_t tx_id,
                                             llama_seq_id seq_id,
                                             const xkv_snapshot_stamp & snapshot,
                                             xkv_rollback_fn rollback_fn, std::string * err)
    : coord_(coord), tx_id_(tx_id), seq_id_(seq_id), rollback_fn_(std::move(rollback_fn)) {
    if (coord_ == nullptr) {
        status_ = xkv_tx_status::bad_tx_id;
        return;
    }
    status_ = coord_->begin(tx_id, seq_id, snapshot, err);
}

xkv_transaction_guard::xkv_transaction_guard(xkv_transaction_coordinator * coord, uint64_t tx_id,
                                             const xkv_snapshot_stamp & snapshot,
                                             xkv_rollback_fn rollback_fn, std::string * err)
    : xkv_transaction_guard(coord, tx_id, 0, snapshot, std::move(rollback_fn), err) {}

xkv_transaction_guard::xkv_transaction_guard(xkv_transaction_guard && o) noexcept
    : coord_(o.coord_), tx_id_(o.tx_id_), seq_id_(o.seq_id_), status_(o.status_),
      rollback_fn_(std::move(o.rollback_fn_)), finished_(o.finished_), dismissed_(o.dismissed_) {
    o.coord_ = nullptr;
    o.finished_ = true;
    o.dismissed_ = true;
}

void xkv_transaction_guard::settle_owned() {
    if (coord_ != nullptr && !finished_ && !dismissed_ && status_ == xkv_tx_status::ok &&
        coord_->has_active_tx(tx_id_)) {
        std::string ignored;
        coord_->rollback(tx_id_, rollback_fn_, &ignored);
    }
}

xkv_transaction_guard & xkv_transaction_guard::operator=(xkv_transaction_guard && o) noexcept {
    if (this != &o) {
        settle_owned();
        coord_ = o.coord_;
        tx_id_ = o.tx_id_;
        seq_id_ = o.seq_id_;
        status_ = o.status_;
        rollback_fn_ = std::move(o.rollback_fn_);
        finished_ = o.finished_;
        dismissed_ = o.dismissed_;
        o.coord_ = nullptr;
        o.finished_ = true;
        o.dismissed_ = true;
    }
    return *this;
}

xkv_transaction_guard::~xkv_transaction_guard() {
    settle_owned();
}

xkv_tx_status xkv_transaction_guard::record(uint64_t payload_id, uint64_t generation, std::string * err) {
    if (status_ != xkv_tx_status::ok) {
        return status_;
    }
    return coord_->record_payload(tx_id_, payload_id, generation, err);
}

xkv_tx_status xkv_transaction_guard::byte_complete(uint64_t payload_id, std::string * err) {
    if (status_ != xkv_tx_status::ok) {
        return status_;
    }
    return coord_->record_byte_complete(tx_id_, payload_id, err);
}

xkv_tx_status xkv_transaction_guard::rebase(std::string * err) {
    if (status_ != xkv_tx_status::ok) {
        return status_;
    }
    return coord_->rebase(tx_id_, err);
}

xkv_tx_status xkv_transaction_guard::commit(const std::vector<uint64_t> & accepted_ids,
                                            xkv_remove_rejected_fn remove_rejected,
                                            const xkv_snapshot_stamp & expected_stamp,
                                            std::string * err) {
    if (status_ != xkv_tx_status::ok) {
        return status_;
    }
    xkv_tx_status st = coord_->commit(tx_id_, accepted_ids, std::move(remove_rejected), expected_stamp, err);
    if (st == xkv_tx_status::ok) {
        finished_ = true;
    }
    return st;
}

xkv_tx_status xkv_transaction_guard::rollback(std::string * err) {
    if (status_ != xkv_tx_status::ok) {
        return status_;
    }
    xkv_tx_status st = coord_->rollback(tx_id_, rollback_fn_, err);
    if (st == xkv_tx_status::ok) {
        finished_ = true;
    }
    return st;
}

xkv_maintenance_guard::xkv_maintenance_guard(xkv_transaction_coordinator * coord, xkv_maintenance_op op,
                                             uint64_t maint_id, std::string * err)
    : coord_(coord), maint_id_(maint_id) {
    if (coord_ == nullptr) {
        status_ = xkv_tx_status::bad_tx_id;
        held_ = false;
        return;
    }
    status_ = coord_->begin_maintenance(op, maint_id, err);
    held_ = (status_ == xkv_tx_status::ok);
    if (!held_) {
        coord_ = nullptr;
        maint_id_ = 0;
    }
}

xkv_maintenance_guard::xkv_maintenance_guard(xkv_maintenance_guard && o) noexcept
    : coord_(o.coord_), maint_id_(o.maint_id_), status_(o.status_), held_(o.held_) {
    o.coord_ = nullptr;
    o.maint_id_ = 0;
    o.held_ = false;
}

xkv_maintenance_guard & xkv_maintenance_guard::operator=(xkv_maintenance_guard && o) noexcept {
    if (this != &o) {
        release();
        coord_ = o.coord_;
        maint_id_ = o.maint_id_;
        status_ = o.status_;
        held_ = o.held_;
        o.coord_ = nullptr;
        o.maint_id_ = 0;
        o.held_ = false;
    }
    return *this;
}

xkv_maintenance_guard::~xkv_maintenance_guard() {
    release();
}

void xkv_maintenance_guard::release() {
    if (coord_ != nullptr && maint_id_ != 0 && held_) {
        std::string ignored;
        coord_->end_maintenance(maint_id_, &ignored);
    }
    coord_ = nullptr;
    maint_id_ = 0;
    held_ = false;
}

} // namespace llama_xkv
