// Deterministic tests for xkv_transaction_coordinator:
// multi-slot concurrent transactions (slots 2/6), rebase, batch settlement,
// accept 0/partial/all, prefix/suffix shape, null-callback requirements,
// remove-callback failure, compute-failure rollback, nested/overlapping and
// monotonic-id rejection, maintenance exclusion, reader drain/final fence with
// tx exclusion and quiesce gate, stale view/content/binding stamps,
// cancellation without advancing public publish_epoch, callback-phase refusals,
// reentry without deadlock, clear quiescence, guard move/held semantics,
// multi-commit eligibility, and checked epoch overflow via authoritative
// stamp provider. No sleeps for correctness.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama-xkv-transaction.h"

#include <atomic>
#include <cassert>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

using namespace llama_xkv;

static constexpr uint64_t k_max = std::numeric_limits<uint64_t>::max();

static xkv_snapshot_stamp fresh_stamp() {
    return xkv_snapshot_stamp{};
}

static xkv_in_memory_stamp_provider * get_fake(xkv_transaction_coordinator & c) {
    return dynamic_cast<xkv_in_memory_stamp_provider *>(c.stamp_provider());
}

static void record_all_complete(xkv_transaction_coordinator & c, uint64_t tx,
                                const std::vector<uint64_t> & ids) {
    std::string err;
    for (uint64_t id : ids) {
        assert(c.record_payload(tx, id, 1000 + id, &err) == xkv_tx_status::ok);
    }
    for (uint64_t id : ids) {
        assert(c.record_byte_complete(tx, id, &err) == xkv_tx_status::ok);
    }
}

static void test_accept_all() {
    std::cout << "[Test] accept all..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    auto s0 = c.current_stamp();
    assert(c.begin(1, s0, &err) == xkv_tx_status::ok);
    record_all_complete(c, 1, {11, 12, 13});
    std::vector<uint64_t> removed;
    auto st = c.commit(1, {11, 12, 13},
                       [&](const std::vector<uint64_t> & rej, uint64_t, xkv_removal_feedback *, std::string *) {
                           removed = rej;
                           return true;
                       },
                       c.current_stamp(), &err);
    assert(st == xkv_tx_status::ok);
    assert(removed.empty());
    assert(c.has_committed_publication());
    assert(c.committed_tx_id() == 1);
    assert(c.committed_accepted() == std::vector<uint64_t>({11, 12, 13}));
    assert(c.committed_rejected().empty());
    assert(c.committed_stamp().content_epoch == s0.content_epoch + 1);
    assert(c.current_stamp().content_epoch == s0.content_epoch + 1);
    assert(c.is_eligible_for_maintenance(11));
    assert(!c.has_active_tx());
}

static void test_accept_partial() {
    std::cout << "[Test] accept partial..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    assert(c.begin(7, fresh_stamp(), &err) == xkv_tx_status::ok);
    record_all_complete(c, 7, {21, 22, 23, 24});
    std::vector<uint64_t> removed;
    auto st = c.commit(7, {21, 22},
                       [&](const std::vector<uint64_t> & rej, uint64_t, xkv_removal_feedback *, std::string *) {
                           removed = rej;
                           return true;
                       },
                       c.current_stamp(), &err);
    assert(st == xkv_tx_status::ok);
    assert(removed == std::vector<uint64_t>({23, 24}));
    assert(c.committed_accepted() == std::vector<uint64_t>({21, 22}));
    assert(c.committed_rejected() == std::vector<uint64_t>({23, 24}));
    assert(c.is_eligible_for_maintenance(21));
    assert(c.is_eligible_for_maintenance(22));
    assert(!c.is_eligible_for_maintenance(23));
    assert(!c.is_eligible_for_maintenance(24));
}

static void test_accept_zero() {
    std::cout << "[Test] accept zero..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    assert(c.begin(9, fresh_stamp(), &err) == xkv_tx_status::ok);
    record_all_complete(c, 9, {31, 32});
    std::vector<uint64_t> removed;
    auto st = c.commit(9, {},
                       [&](const std::vector<uint64_t> & rej, uint64_t, xkv_removal_feedback *, std::string *) {
                           removed = rej;
                           return true;
                       },
                       c.current_stamp(), &err);
    assert(st == xkv_tx_status::ok);
    assert(removed == std::vector<uint64_t>({31, 32}));
    assert(c.committed_accepted().empty());
    assert(!c.is_eligible_for_maintenance(31));
    assert(!c.is_eligible_for_maintenance(32));
    assert(c.has_committed_publication());
}

static void test_prefix_enforcement() {
    std::cout << "[Test] accepted must equal recorded prefix..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    assert(c.begin(97, fresh_stamp(), &err) == xkv_tx_status::ok);
    record_all_complete(c, 97, {961, 962, 963, 964});
    auto ok_cb = [](const std::vector<uint64_t> &, uint64_t, xkv_removal_feedback *, std::string *) { return true; };
    assert(c.commit(97, {961, 963}, ok_cb, c.current_stamp(), &err) == xkv_tx_status::invalid_accept);
    assert(c.commit(97, {962}, ok_cb, c.current_stamp(), &err) == xkv_tx_status::invalid_accept);
    assert(c.commit(97, {962, 961}, ok_cb, c.current_stamp(), &err) == xkv_tx_status::invalid_accept);
    assert(c.commit(97, {961, 962, 963, 964, 999}, ok_cb, c.current_stamp(), &err) ==
           xkv_tx_status::invalid_accept);
    assert(!c.has_committed_publication());
    assert(c.has_active_tx());
    std::vector<uint64_t> removed;
    assert(c.commit(97, {961, 962},
                    [&](const std::vector<uint64_t> & rej, uint64_t, xkv_removal_feedback *, std::string *) {
                        removed = rej;
                        return true;
                    },
                    c.current_stamp(), &err) == xkv_tx_status::ok);
    assert(removed == std::vector<uint64_t>({963, 964}));
}

static void test_null_callback_required() {
    std::cout << "[Test] nonempty rejected/created sets require a callback..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    assert(c.begin(95, fresh_stamp(), &err) == xkv_tx_status::ok);
    record_all_complete(c, 95, {951, 952});
    assert(c.commit(95, {951}, nullptr, c.current_stamp(), &err) == xkv_tx_status::callback_failed);
    assert(!c.has_committed_publication());
    assert(c.has_active_tx());
    assert(c.rollback(95, nullptr, &err) == xkv_tx_status::callback_failed);
    assert(c.has_active_tx());
    assert(c.created_payload_ids(95) == std::vector<uint64_t>({951, 952}));
    assert(c.commit(95, {951, 952}, nullptr, c.current_stamp(), &err) == xkv_tx_status::ok);
    assert(c.has_committed_publication());

    xkv_transaction_coordinator c2;
    assert(c2.begin(96, fresh_stamp(), &err) == xkv_tx_status::ok);
    assert(c2.record_payload(96, 953, 1, &err) == xkv_tx_status::ok);
    assert(c2.rollback(96, nullptr, &err) == xkv_tx_status::callback_failed);
    assert(c2.has_active_tx());
    assert(c2.rollback(96, [](const std::vector<uint64_t> &, std::string *) { return true; }, &err) ==
           xkv_tx_status::ok);
}

static void test_remove_callback_failure() {
    std::cout << "[Test] remove callback failure leaves recoverable state, no publication..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    assert(c.begin(11, fresh_stamp(), &err) == xkv_tx_status::ok);
    record_all_complete(c, 11, {41, 42, 43});
    auto fail = [&](const std::vector<uint64_t> &, uint64_t, xkv_removal_feedback *, std::string * e) {
        if (e) {
            *e = "seq_rm io error";
        }
        return false;
    };
    auto st = c.commit(11, {41}, fail, c.current_stamp(), &err);
    assert(st == xkv_tx_status::callback_failed);
    assert(!c.has_committed_publication());
    assert(c.has_active_tx());
    assert(c.active_tx_id() == 11);
    std::vector<uint64_t> removed;
    st = c.commit(11, {41},
                  [&](const std::vector<uint64_t> & rej, uint64_t, xkv_removal_feedback *, std::string *) {
                      removed = rej;
                      return true;
                  },
                  c.current_stamp(), &err);
    assert(st == xkv_tx_status::ok);
    assert(removed == std::vector<uint64_t>({42, 43}));
    assert(c.has_committed_publication());
}

static void test_compute_failure_rollback() {
    std::cout << "[Test] compute failure rollback removes all created payloads..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    assert(c.begin(21, fresh_stamp(), &err) == xkv_tx_status::ok);
    assert(c.record_payload(21, 51, 1, &err) == xkv_tx_status::ok);
    assert(c.record_payload(21, 52, 2, &err) == xkv_tx_status::ok);
    assert(c.record_byte_complete(21, 51, &err) == xkv_tx_status::ok);
    std::vector<uint64_t> rolled;
    auto st = c.rollback(21,
                         [&](const std::vector<uint64_t> & created, std::string *) {
                             rolled = created;
                             return true;
                         },
                         &err);
    assert(st == xkv_tx_status::ok);
    assert(rolled == std::vector<uint64_t>({51, 52}));
    assert(!c.has_active_tx());
    assert(!c.has_committed_publication());
    assert(c.created_payload_ids(21).empty());
}

static void test_nested_overlapping_rejection() {
    std::cout << "[Test] nested/overlapping rejection..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    assert(c.begin(31, 1, fresh_stamp(), &err) == xkv_tx_status::ok);
    assert(c.begin(31, 2, fresh_stamp(), &err) == xkv_tx_status::already_active); // tx_id reused
    assert(c.begin(32, 1, fresh_stamp(), &err) == xkv_tx_status::already_active); // seq 1 active
    assert(c.begin(0, 3, fresh_stamp(), &err) == xkv_tx_status::bad_tx_id);
    assert(c.record_payload(32, 61, 1, &err) == xkv_tx_status::no_active_tx);
    assert(c.commit(32, {}, nullptr, c.current_stamp(), &err) == xkv_tx_status::no_active_tx);
    assert(c.rollback(32, nullptr, &err) == xkv_tx_status::no_active_tx);
    assert(c.rollback(31, nullptr, &err) == xkv_tx_status::ok);
    assert(c.begin(31, 1, fresh_stamp(), &err) == xkv_tx_status::tx_id_reused);
    assert(c.begin(33, 1, fresh_stamp(), &err) == xkv_tx_status::ok);
    assert(c.rollback(33, nullptr, &err) == xkv_tx_status::ok);
    {
        xkv_transaction_guard g(&c, 34, 1, c.current_stamp(),
                                [](const std::vector<uint64_t> &, std::string *) { return true; },
                                &err);
        assert(g.begun());
        assert(g.record(71, 1, &err) == xkv_tx_status::ok);
    }
    assert(!c.has_active_tx());
}

static void test_monotonic_ids() {
    std::cout << "[Test] strictly monotonic ids, no unbounded reuse set..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    assert(c.begin(5, fresh_stamp(), &err) == xkv_tx_status::ok);
    assert(c.rollback(5, nullptr, &err) == xkv_tx_status::ok);
    assert(c.begin(5, fresh_stamp(), &err) == xkv_tx_status::tx_id_reused);
    assert(c.begin(4, fresh_stamp(), &err) == xkv_tx_status::tx_id_reused);
    assert(c.begin(0, fresh_stamp(), &err) == xkv_tx_status::bad_tx_id);
    assert(c.begin(6, fresh_stamp(), &err) == xkv_tx_status::ok);
    assert(c.rollback(6, nullptr, &err) == xkv_tx_status::ok);
}

static void test_maintenance_exclusion() {
    std::cout << "[Test] maintenance exclusion..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    assert(c.maintenance_allowed());
    assert(c.begin(41, fresh_stamp(), &err) == xkv_tx_status::ok);
    assert(!c.maintenance_allowed());
    const xkv_maintenance_op ops[] = {xkv_maintenance_op::seal, xkv_maintenance_op::pack,
                                      xkv_maintenance_op::relocate, xkv_maintenance_op::landmark_publish};
    for (auto op : ops) {
        assert(c.begin_maintenance(op, 100 + (uint64_t) op, &err) == xkv_tx_status::tx_active);
    }
    assert(c.rollback(41, nullptr, &err) == xkv_tx_status::ok);
    assert(c.maintenance_allowed());
    {
        xkv_maintenance_guard mg(&c, xkv_maintenance_op::seal, 500, &err);
        assert(mg.held());
        assert(c.has_active_maintenance());
        assert(!c.maintenance_allowed());
        assert(c.begin(42, c.current_stamp(), &err) == xkv_tx_status::maintenance_active);
    }
    assert(!c.has_active_maintenance());
    assert(c.maintenance_allowed());
    assert(c.begin(42, c.current_stamp(), &err) == xkv_tx_status::ok);
    assert(c.rollback(42, nullptr, &err) == xkv_tx_status::ok);
    c.add_maintenance_candidate(81);
    c.add_maintenance_candidate(82);
    assert(c.is_maintenance_candidate(81));
    assert(c.maintenance_candidate_count() == 2);
    assert(!c.is_eligible_for_maintenance(81));
    c.remove_maintenance_candidate(82);
    assert(!c.is_maintenance_candidate(82));
}

static void test_reader_drain_final_fence() {
    std::cout << "[Test] reader drain / final fence..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    xkv_quiesce_options no_wait;
    no_wait.wait = false;
    xkv_quiesce_options do_wait;
    do_wait.wait = true;
    {
        auto lease = c.acquire_reader();
        assert(c.active_reader_count() == 1);
        llama_rerot_view_stamp v{1, 1, 1};
        assert(c.final_fence(v, no_wait, &err) == xkv_tx_status::busy);
        assert(c.wait_for_readers_drained(no_wait, &err) == xkv_tx_status::busy);
        assert(c.apply_context_shift(v, no_wait, &err) == xkv_tx_status::busy);
    }
    assert(c.active_reader_count() == 0);
    assert(c.wait_for_readers_drained(no_wait, &err) == xkv_tx_status::ok);
    llama_rerot_view_stamp stable{7, 8, 9};
    assert(c.final_fence(stable, no_wait, &err) == xkv_tx_status::ok);
    auto cur = c.current_stamp();
    assert(cur.view.topology_epoch == 7 && cur.view.publish_epoch == 8 && cur.view.layout_epoch == 9);

    {
        auto lease = c.acquire_reader();
        std::atomic<xkv_tx_status> fence_st{xkv_tx_status::busy};
        std::thread t([&] {
            std::string e2;
            fence_st.store(c.final_fence(llama_rerot_view_stamp{10, 11, 12}, do_wait, &e2));
        });
        lease.release();
        t.join();
        assert(fence_st.load() == xkv_tx_status::ok);
        assert(c.current_stamp().view.topology_epoch == 10);
    }
    assert(c.begin_maintenance(xkv_maintenance_op::pack, 900, &err) == xkv_tx_status::ok);
    assert(c.final_fence(llama_rerot_view_stamp{1, 2, 3}, no_wait, &err) == xkv_tx_status::busy);
    assert(c.end_maintenance(900, &err) == xkv_tx_status::ok);
    assert(c.final_fence(llama_rerot_view_stamp{1, 2, 3}, no_wait, &err) == xkv_tx_status::ok);
}

static void test_fence_excludes_tx() {
    std::cout << "[Test] fence/shift exclude active tx..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    xkv_quiesce_options no_wait;
    no_wait.wait = false;
    xkv_quiesce_options do_wait;
    do_wait.wait = true;
    assert(c.begin(201, c.current_stamp(), &err) == xkv_tx_status::ok);
    assert(c.record_payload(201, 1201, 1, &err) == xkv_tx_status::ok);
    assert(c.final_fence(llama_rerot_view_stamp{1, 1, 1}, no_wait, &err) == xkv_tx_status::tx_active);
    assert(c.apply_context_shift(llama_rerot_view_stamp{1, 1, 1}, no_wait, &err) == xkv_tx_status::tx_active);
    assert(c.has_active_tx());
    std::atomic<xkv_tx_status> fence_st{xkv_tx_status::busy};
    std::thread t([&] {
        std::string e2;
        fence_st.store(c.final_fence(llama_rerot_view_stamp{3, 3, 3}, do_wait, &e2));
    });
    assert(c.rollback(201,
                      [](const std::vector<uint64_t> &, std::string *) { return true; },
                      &err) == xkv_tx_status::ok);
    t.join();
    assert(fence_st.load() == xkv_tx_status::ok);
    assert(c.current_stamp().view.topology_epoch == 3);
}

static void test_quiesce_gate_parks_readers() {
    std::cout << "[Test] quiesce gate parks late readers instead of starving..." << std::endl;
    xkv_transaction_coordinator c;
    xkv_quiesce_options do_wait;
    do_wait.wait = true;
    auto r1 = c.acquire_reader();
    std::atomic<xkv_tx_status> fence_st{xkv_tx_status::busy};
    std::thread tf([&] {
        fence_st.store(c.final_fence(llama_rerot_view_stamp{4, 5, 6}, do_wait, nullptr));
    });
    while (!c.has_quiesce_pending()) {
    }
    std::atomic<bool> r2_got{false};
    std::thread tr([&] {
        auto r2 = c.acquire_reader();
        r2_got.store(static_cast<bool>(r2));
    });
    r1.release();
    tf.join();
    assert(fence_st.load() == xkv_tx_status::ok);
    tr.join();
    assert(r2_got.load());
    assert(c.active_reader_count() == 0);
    assert(c.current_stamp().view.topology_epoch == 4);
}

static void test_stale_stamps() {
    std::cout << "[Test] stale view/content/binding stamps from store..." << std::endl;
    xkv_transaction_coordinator c;
    auto * fake = get_fake(c);
    assert(fake != nullptr);
    std::string err, reason;

    auto s_view = c.current_stamp();
    s_view.view.topology_epoch = 2;
    fake->set_stamp_for_testing(s_view);
    assert(c.begin(51, fresh_stamp(), &err) == xkv_tx_status::stale_stamp);

    auto s0 = c.current_stamp();
    assert(c.begin(52, s0, &err) == xkv_tx_status::ok);
    record_all_complete(c, 52, {91});
    xkv_snapshot_stamp s_bump = s0;
    s_bump.content_epoch += 1;
    fake->set_stamp_for_testing(s_bump);
    assert(c.is_stale(s0, &reason));
    assert(reason == "stale content_epoch");
    assert(c.commit(52, {91}, nullptr, s0, &err) == xkv_tx_status::stale_stamp);
    assert(c.has_active_tx());
    assert(!c.has_committed_publication());
    assert(c.rollback(52, [](const std::vector<uint64_t> &, std::string *) { return true; }, &err) ==
           xkv_tx_status::ok);

    s0 = c.current_stamp();
    assert(c.begin(53, s0, &err) == xkv_tx_status::ok);
    record_all_complete(c, 53, {92});
    s_bump = s0;
    s_bump.binding_epoch += 1;
    fake->set_stamp_for_testing(s_bump);
    assert(c.is_stale(s0, &reason));
    assert(reason == "stale binding_epoch");
    assert(c.commit(53, {92}, nullptr, s0, &err) == xkv_tx_status::stale_stamp);
    assert(c.rollback(53, [](const std::vector<uint64_t> &, std::string *) { return true; }, &err) ==
           xkv_tx_status::ok);

    xkv_quiesce_options no_wait;
    no_wait.wait = false;
    uint64_t b0 = c.current_stamp().binding_epoch;
    assert(c.apply_context_shift(llama_rerot_view_stamp{5, 6, 7}, no_wait, &err) == xkv_tx_status::ok);
    assert(c.current_stamp().binding_epoch == b0 + 1);
    assert(c.current_stamp().view.topology_epoch == 5);
}

static void test_cancellation() {
    std::cout << "[Test] cancellation removes private rows without advancing public publish_epoch..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    uint64_t pub0 = c.current_stamp().view.publish_epoch;
    assert(c.begin(61, c.current_stamp(), &err) == xkv_tx_status::ok);
    record_all_complete(c, 61, {101, 102});
    std::vector<uint64_t> rolled;
    auto st = c.cancel(61,
                       [&](const std::vector<uint64_t> & created, std::string *) {
                           rolled = created;
                           return true;
                       },
                       &err);
    assert(st == xkv_tx_status::ok);
    assert(rolled == std::vector<uint64_t>({101, 102}));
    assert(!c.has_active_tx());
    assert(!c.has_committed_publication());
    // Invariant: tentative private rows rollback must NOT bump public view.publish_epoch
    assert(c.current_stamp().view.publish_epoch == pub0);
    assert(c.cancel(61, nullptr, &err) == xkv_tx_status::no_active_tx);
}

static void test_callback_reentry_no_lock() {
    std::cout << "[Test] callback reentry: observers/pins ok, mutators refused, no deadlock..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    xkv_quiesce_options no_wait;
    no_wait.wait = false;
    assert(c.begin(71, c.current_stamp(), &err) == xkv_tx_status::ok);
    record_all_complete(c, 71, {111, 112, 113});
    bool reentered = false;
    auto st = c.commit(71, {111},
                       [&](const std::vector<uint64_t> & rej, uint64_t, xkv_removal_feedback *, std::string * e) {
                           auto snap = c.current_stamp();
                           (void) snap;
                           auto n = c.active_reader_count();
                           auto lease = c.acquire_reader();
                           assert(c.active_reader_count() == n + 1);
                           assert(c.has_active_tx());
                           assert(rej == std::vector<uint64_t>({112, 113}));
                           assert(c.record_payload(71, 999, 1, e) == xkv_tx_status::busy);
                           assert(c.record_byte_complete(71, 111, e) == xkv_tx_status::busy);
                           assert(c.commit(71, {111}, nullptr, c.current_stamp(), e) == xkv_tx_status::busy);
                           assert(c.rollback(71, nullptr, e) == xkv_tx_status::busy);
                           assert(c.cancel(71, nullptr, e) == xkv_tx_status::busy);
                           assert(c.final_fence(llama_rerot_view_stamp{0, 0, 0}, no_wait, e) ==
                                  xkv_tx_status::busy);
                           assert(c.apply_context_shift(llama_rerot_view_stamp{0, 0, 0}, no_wait, e) ==
                                  xkv_tx_status::busy);
                           assert(c.begin(72, c.current_stamp(), e) == xkv_tx_status::already_active);
                           reentered = true;
                           return true;
                       },
                       c.current_stamp(), &err);
    assert(st == xkv_tx_status::ok);
    assert(reentered);
    assert(c.active_reader_count() == 0);
    assert(c.committed_accepted() == std::vector<uint64_t>({111}));
    assert(c.committed_rejected() == std::vector<uint64_t>({112, 113}));

    assert(c.begin(72, c.current_stamp(), &err) == xkv_tx_status::ok);
    assert(c.record_payload(72, 121, 1, &err) == xkv_tx_status::ok);
    bool rb_reentered = false;
    st = c.rollback(72,
                    [&](const std::vector<uint64_t> & created, std::string * e) {
                        assert(created == std::vector<uint64_t>({121}));
                        auto ids = c.created_payload_ids(72);
                        assert(ids == std::vector<uint64_t>({121}));
                        auto lease = c.acquire_reader();
                        (void) lease;
                        assert(c.record_payload(72, 122, 1, e) == xkv_tx_status::busy);
                        rb_reentered = true;
                        return true;
                    },
                    &err);
    assert(st == xkv_tx_status::ok);
    assert(rb_reentered);
    assert(c.active_reader_count() == 0);
}

static void test_commit_validation() {
    std::cout << "[Test] commit validation (unknown / not byte-complete)..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    assert(c.begin(81, c.current_stamp(), &err) == xkv_tx_status::ok);
    assert(c.record_payload(81, 131, 1, &err) == xkv_tx_status::ok);
    assert(c.commit(81, {131}, nullptr, c.current_stamp(), &err) == xkv_tx_status::invalid_accept);
    assert(!c.has_committed_publication());
    assert(c.commit(81, {999}, nullptr, c.current_stamp(), &err) == xkv_tx_status::invalid_accept);
    assert(c.rollback(81, nullptr, &err) == xkv_tx_status::callback_failed);
    assert(c.rollback(81, [](const std::vector<uint64_t> &, std::string *) { return true; }, &err) ==
           xkv_tx_status::ok);
}

static void test_clear_quiescence() {
    std::cout << "[Test] clear refuses live state, resets when quiescent..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    assert(c.begin(101, c.current_stamp(), &err) == xkv_tx_status::ok);
    assert(c.clear(&err) == xkv_tx_status::busy);
    assert(c.has_active_tx());
    assert(c.rollback(101, nullptr, &err) == xkv_tx_status::ok);
    {
        auto lease = c.acquire_reader();
        assert(c.clear(&err) == xkv_tx_status::busy);
        assert(c.active_reader_count() == 1);
    }
    assert(c.begin_maintenance(xkv_maintenance_op::seal, 700, &err) == xkv_tx_status::ok);
    assert(c.clear(&err) == xkv_tx_status::busy);
    assert(c.end_maintenance(700, &err) == xkv_tx_status::ok);
    assert(c.clear(&err) == xkv_tx_status::ok);
    assert(!c.has_committed_publication());
    assert(c.begin(1, c.current_stamp(), &err) == xkv_tx_status::ok);
    assert(c.rollback(1, nullptr, &err) == xkv_tx_status::ok);
}

static void test_held_semantics() {
    std::cout << "[Test] maintenance guard held/move/release..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    xkv_maintenance_guard mg(&c, xkv_maintenance_op::pack, 601, &err);
    assert(mg.held());
    mg.release();
    assert(!mg.held());
    mg.release();
    assert(!c.has_active_maintenance());

    xkv_maintenance_guard a(&c, xkv_maintenance_op::seal, 602, &err);
    assert(a.held());
    xkv_maintenance_guard b(std::move(a));
    assert(!a.held());
    assert(b.held());
    xkv_maintenance_guard dst;
    assert(!dst.held());
    dst = std::move(b);
    assert(!b.held());
    assert(dst.held());
    assert(c.has_active_maintenance());
}

static void test_guard_move_assign_settles() {
    std::cout << "[Test] transaction guard move-assign settles destination tx..." << std::endl;
    xkv_transaction_coordinator c1, c2;
    std::string err;
    std::vector<uint64_t> rolled;
    xkv_transaction_guard g1(&c1, 5, 1, c1.current_stamp(),
                             [&](const std::vector<uint64_t> & created, std::string *) {
                                 rolled = created;
                                 return true;
                             },
                             &err);
    assert(g1.begun());
    assert(g1.record(501, 1, &err) == xkv_tx_status::ok);
    xkv_transaction_guard g2(&c2, 6, 1, c2.current_stamp(), nullptr, &err);
    assert(g2.begun());
    g1 = std::move(g2);
    assert(rolled == std::vector<uint64_t>({501}));
    assert(!c1.has_active_tx());
    assert(g1.begun() && g1.tx_id() == 6);
    assert(c2.has_active_tx(6));
}

static void test_multi_commit_eligibility() {
    std::cout << "[Test] eligibility persists across commits until retirement..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    assert(c.begin(111, c.current_stamp(), &err) == xkv_tx_status::ok);
    record_all_complete(c, 111, {1001});
    assert(c.commit(111, {1001}, nullptr, c.current_stamp(), &err) == xkv_tx_status::ok);
    assert(c.begin(112, c.current_stamp(), &err) == xkv_tx_status::ok);
    record_all_complete(c, 112, {1002});
    assert(c.commit(112, {1002}, nullptr, c.current_stamp(), &err) == xkv_tx_status::ok);
    assert(c.committed_accepted() == std::vector<uint64_t>({1002}));
    assert(c.is_eligible_for_maintenance(1001));
    assert(c.is_eligible_for_maintenance(1002));
    c.retire_payload(1001);
    assert(!c.is_eligible_for_maintenance(1001));
    assert(c.is_eligible_for_maintenance(1002));
}

static void test_provider_advance_failures_and_overflow() {
    std::cout << "[Test] provider advance failures & overflow handling..." << std::endl;
    xkv_transaction_coordinator c;
    auto * fake = get_fake(c);
    assert(fake != nullptr);
    std::string err;

    fake->set_fail_content_advance(true);
    assert(c.begin(85, c.current_stamp(), &err) == xkv_tx_status::ok);
    record_all_complete(c, 85, {851});
    assert(c.commit(85, {851}, nullptr, c.current_stamp(), &err) == xkv_tx_status::callback_failed);
    assert(!c.has_committed_publication());
    assert(c.has_active_tx());
    fake->set_fail_content_advance(false);
    assert(c.commit(85, {851}, nullptr, c.current_stamp(), &err) == xkv_tx_status::ok);
    assert(c.has_committed_publication());

    xkv_snapshot_stamp s = c.current_stamp();
    s.content_epoch = k_max;
    fake->set_stamp_for_testing(s);
    assert(c.begin(90, c.current_stamp(), &err) == xkv_tx_status::ok);
    record_all_complete(c, 90, {901});
    assert(c.commit(90, {901}, nullptr, c.current_stamp(), &err) == xkv_tx_status::epoch_overflow);
    // has_committed_publication() already latched true from tx85 above; the
    // overflow commit must leave the tx active with no NEW publication.
    assert(c.committed_tx_id() == 85);
    assert(c.has_active_tx());
    assert(c.rollback(90, [](const std::vector<uint64_t> &, std::string *) { return true; }, &err) ==
           xkv_tx_status::ok);
    s.content_epoch = 5;
    fake->set_stamp_for_testing(s);
    assert(c.begin(92, c.current_stamp(), &err) == xkv_tx_status::ok);
    record_all_complete(c, 92, {903});
    assert(c.commit(92, {903}, nullptr, c.current_stamp(), &err) == xkv_tx_status::ok);
    assert(c.is_eligible_for_maintenance(903));

    s = c.current_stamp();
    s.binding_epoch = k_max;
    fake->set_stamp_for_testing(s);
    xkv_quiesce_options no_wait;
    no_wait.wait = false;
    auto before = c.current_stamp();
    assert(c.apply_context_shift(llama_rerot_view_stamp{8, 8, 8}, no_wait, &err) ==
           xkv_tx_status::epoch_overflow);
    assert(c.current_stamp().view.topology_epoch == before.view.topology_epoch);
}

static void test_preflight_reservation_guarantees_zero_removals() {
    std::cout << "[Test] preflight reservation: overflow/refusal aborts BEFORE callback (0 removals)..." << std::endl;
    xkv_transaction_coordinator c;
    auto * fake = get_fake(c);
    assert(fake != nullptr);
    std::string err;

    xkv_snapshot_stamp s = c.current_stamp();
    s.content_epoch = k_max;
    fake->set_stamp_for_testing(s);
    assert(c.begin(120, c.current_stamp(), &err) == xkv_tx_status::ok);
    record_all_complete(c, 120, {1201, 1202});

    bool callback_called = false;
    auto remover = [&](const std::vector<uint64_t> &, uint64_t, xkv_removal_feedback *, std::string *) {
        callback_called = true;
        return true;
    };
    auto st = c.commit(120, {1201}, remover, c.current_stamp(), &err);
    assert(st == xkv_tx_status::epoch_overflow);
    assert(!callback_called);
    assert(c.created_payload_ids(120) == std::vector<uint64_t>({1201, 1202}));
    assert(c.has_active_tx(120));
    assert(c.rollback(120, [](const std::vector<uint64_t> &, std::string *) { return true; }, &err) == xkv_tx_status::ok);
}

static void test_bind_stamp_provider_refusal() {
    std::cout << "[Test] bind_stamp_provider refused during tx, readers, maintenance..." << std::endl;
    xkv_transaction_coordinator c;
    xkv_in_memory_stamp_provider other;
    std::string err;

    assert(c.begin(130, c.current_stamp(), &err) == xkv_tx_status::ok);
    assert(c.bind_stamp_provider(&other, &err) == xkv_tx_status::busy);
    assert(c.rollback(130, nullptr, &err) == xkv_tx_status::ok);

    {
        auto lease = c.acquire_reader();
        assert(c.bind_stamp_provider(&other, &err) == xkv_tx_status::busy);
    }

    assert(c.begin_maintenance(xkv_maintenance_op::seal, 800, &err) == xkv_tx_status::ok);
    assert(c.bind_stamp_provider(&other, &err) == xkv_tx_status::busy);
    assert(c.end_maintenance(800, &err) == xkv_tx_status::ok);

    assert(c.bind_stamp_provider(&other, &err) == xkv_tx_status::ok);
    assert(c.stamp_provider() == &other);
}

static void test_guard_move_assign_dismissed_reset() {
    std::cout << "[Test] guard move-assignment resets dismissed_ from source..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    bool src_rolled_back = false;

    {
        xkv_transaction_guard g_dest(&c, 140, 1, c.current_stamp(), nullptr, &err);
        assert(g_dest.begun());
        g_dest.dismiss();

        xkv_transaction_guard g_src(&c, 141, 2, c.current_stamp(),
                                    [&](const std::vector<uint64_t> &, std::string *) {
                                        src_rolled_back = true;
                                        return true;
                                    },
                                    &err);
        assert(g_src.begun());

        // Record a payload so destination destruction invokes the callback.
        assert(g_src.record(1411, 1, &err) == xkv_tx_status::ok);

        g_dest = std::move(g_src);
    }

    assert(src_rolled_back);
    // tx140 was dismissed before the move, so the destination never owned it:
    // it stays active (user responsibility) while adopted tx141 was settled.
    assert(c.has_active_tx(140));
    assert(!c.has_active_tx(141));
    assert(c.rollback(140, nullptr, &err) == xkv_tx_status::ok);
    assert(!c.has_active_tx());
}

// ============================================================================
// Multi-slot tests: concurrent 2 and 6 slots, rebase, peer isolation, batch
// ============================================================================

static void test_concurrent_2_slots_batch_settle() {
    std::cout << "[Test] multi-slot: 2 concurrent slots batch settlement (accept zero + partial)..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    auto s0 = c.current_stamp();

    assert(c.begin(201, 1, s0, &err) == xkv_tx_status::ok);
    assert(c.begin(202, 2, s0, &err) == xkv_tx_status::ok);
    assert(c.active_tx_count() == 2);
    assert(c.has_active_seq(1));
    assert(c.has_active_seq(2));

    record_all_complete(c, 201, {2011, 2012});
    record_all_complete(c, 202, {2021, 2022, 2023});

    xkv_slot_settlement setA;
    setA.tx_id = 201;
    setA.accepted_ids = {};

    xkv_slot_settlement setB;
    setB.tx_id = 202;
    setB.accepted_ids = {2021, 2022};

    std::vector<xkv_rejected_item> removed_items;
    auto batch_cb = [&](const std::vector<xkv_rejected_item> & items, uint64_t, xkv_removal_feedback *, std::string *) {
        removed_items = items;
        return true;
    };

    auto st = c.commit_batch({setA, setB}, batch_cb, &err);
    assert(st == xkv_tx_status::ok);
    assert(c.active_tx_count() == 0);
    assert(!c.has_active_tx());

    assert(removed_items.size() == 3);
    assert(removed_items[0].tx_id == 201 && removed_items[0].seq_id == 1 && removed_items[0].payload_id == 2011);
    assert(removed_items[1].tx_id == 201 && removed_items[1].seq_id == 1 && removed_items[1].payload_id == 2012);
    assert(removed_items[2].tx_id == 202 && removed_items[2].seq_id == 2 && removed_items[2].payload_id == 2023);

    assert(!c.is_eligible_for_maintenance(2011));
    assert(!c.is_eligible_for_maintenance(2012));
    assert(c.is_eligible_for_maintenance(2021));
    assert(c.is_eligible_for_maintenance(2022));
    assert(!c.is_eligible_for_maintenance(2023));

    assert(c.current_stamp().content_epoch == s0.content_epoch + 1);
}

static void test_concurrent_6_slots_batch_settle() {
    std::cout << "[Test] multi-slot: 6 concurrent slots (all variations of accept)..." << std::endl;
    xkv_transaction_coordinator c;
    std::string err;
    auto s0 = c.current_stamp();

    std::vector<xkv_slot_settlement> settlements;
    for (int i = 1; i <= 6; ++i) {
        uint64_t tx = 300 + i;
        llama_seq_id seq = 10 + i;
        assert(c.begin(tx, seq, s0, &err) == xkv_tx_status::ok);

        uint64_t p1 = tx * 10 + 1;
        uint64_t p2 = tx * 10 + 2;
        record_all_complete(c, tx, {p1, p2});

        xkv_slot_settlement s;
        s.tx_id = tx;
        if (i == 1) {
            s.accepted_ids = {};
        } else if (i == 2) {
            s.accepted_ids = {p1};
        } else {
            s.accepted_ids = {p1, p2};
        }
        settlements.push_back(s);
    }
    assert(c.active_tx_count() == 6);

    std::vector<xkv_rejected_item> removed_items;
    auto batch_cb = [&](const std::vector<xkv_rejected_item> & items, uint64_t, xkv_removal_feedback *, std::string *) {
        removed_items = items;
        return true;
    };

    assert(c.commit_batch(settlements, batch_cb, &err) == xkv_tx_status::ok);
    assert(c.active_tx_count() == 0);

    assert(removed_items.size() == 3);
    assert(c.current_stamp().content_epoch == s0.content_epoch + 1);
}

static void test_postcompute_rebase_and_peer_isolation() {
    std::cout << "[Test] postcompute rebase: private epoch deltas allowed; external RERoT mismatch stales..." << std::endl;
    xkv_transaction_coordinator c;
    auto * fake = get_fake(c);
    assert(fake != nullptr);
    std::string err;
    auto s0 = c.current_stamp();

    assert(c.begin(401, 1, s0, &err) == xkv_tx_status::ok);
    assert(c.begin(402, 2, s0, &err) == xkv_tx_status::ok);
    record_all_complete(c, 401, {4011});
    record_all_complete(c, 402, {4021});

    // Slot 1 commits, advancing content_epoch
    assert(c.commit(401, {4011}, nullptr, s0, &err) == xkv_tx_status::ok);
    assert(!c.has_active_seq(1));
    assert(c.has_active_seq(2));

    assert(c.current_stamp().content_epoch == s0.content_epoch + 1);

    // Slot 2 rebases against updated private store content_epoch
    assert(c.rebase(402, &err) == xkv_tx_status::ok);

    // Slot 2 commits cleanly: peer commit did NOT stale slot 2!
    assert(c.commit(402, {4021}, nullptr, c.current_stamp(), &err) == xkv_tx_status::ok);
    assert(c.is_eligible_for_maintenance(4011));
    assert(c.is_eligible_for_maintenance(4021));

    // External RERoT mismatch stales
    assert(c.begin(403, 3, c.current_stamp(), &err) == xkv_tx_status::ok);
    record_all_complete(c, 403, {4031});

    xkv_snapshot_stamp s_rerot = c.current_stamp();
    s_rerot.view.topology_epoch += 1;
    fake->set_stamp_for_testing(s_rerot);

    assert(c.rebase(403, &err) == xkv_tx_status::stale_stamp);
    assert(err.find("external RERoT view") != std::string::npos);

    assert(c.commit(403, {4031}, nullptr, c.current_stamp(), &err) == xkv_tx_status::stale_stamp);
    assert(c.rollback(403, [](const std::vector<uint64_t> &, std::string *) { return true; }, &err) == xkv_tx_status::ok);
}

// Test removal callback feedback: store remove_payloads advancing live/binding/content
static void test_removal_feedback_store_stamp_accepted() {
    std::cout << "[Test] removal callback feedback: store remove_payloads stamp accepted without stale error..." << std::endl;
    xkv_transaction_coordinator c;
    auto * fake = get_fake(c);
    assert(fake != nullptr);
    std::string err;
    auto s0 = c.current_stamp();

    assert(c.begin(501, 1, s0, &err) == xkv_tx_status::ok);
    record_all_complete(c, 501, {5011, 5012});

    // Simulated callback that mimics real store remove_payloads by advancing
    // live, binding, and content epochs on the store during callback flight!
    auto store_remover = [&](const std::vector<uint64_t> &, uint64_t, xkv_removal_feedback * fb, std::string *) {
        xkv_snapshot_stamp updated = fake->current_stamp();
        updated.live_epoch += 1;
        updated.binding_epoch += 1;
        updated.content_epoch += 1;
        fake->set_stamp_for_testing(updated);

        // Feedback informs coordinator of the updated store stamp
        fb->has_store_stamp = true;
        fb->store_stamp = updated;
        return true;
    };

    // Coordinator commit MUST accept this feedback and NOT report stale stamp!
    auto st = c.commit(501, {5011}, store_remover, s0, &err);
    assert(st == xkv_tx_status::ok);
    assert(c.has_committed_publication());
    assert(c.current_stamp().live_epoch == s0.live_epoch + 1);
    assert(c.current_stamp().binding_epoch == s0.binding_epoch + 1);
    assert(c.current_stamp().content_epoch == s0.content_epoch + 1);
}

static llama_cparams make_store_cparams() {
    llama_cparams cparams = {};
    cparams.xkv_mode = LLAMA_XKV_MODE_SHADOW;
    cparams.xkv_storage_profile = LLAMA_XKV_STORAGE_PROFILE_REFERENCE;
    cparams.xkv_group_size = 4;
    cparams.xkv_rank_k = 16;
    cparams.xkv_rank_v = 16;
    cparams.xkv_segment_tokens = 64;
    cparams.xkv_chunk_tokens = 8;
    cparams.xkv_workspace_mib = 16;
    cparams.xkv_decode_cache_mib = 8;
    cparams.xkv_min_saving = 0.10;
    return cparams;
}

// Store-backed 0/partial/all settlement through the real contract: the removal
// callback forwards the coordinator's live reservation token to the store's
// token-carried remove path (bare removal refuses while reserved), returns the
// store stamp as feedback, and the coordinator applies exactly once.
// pressure-before-settlement property: rejected rows are unlocatable after
// settlement (never sealable later) while accepted rows stay hot_committed.
static void test_store_backed_settlement_0_partial_all() {
    std::cout << "[Test] store-backed settlement 0/partial/all with live token..." << std::endl;
    auto cparams = make_store_cparams();
    llama_xkv_cache_store store(cparams);
    xkv_transaction_coordinator c(&store);
    std::string err;

    const std::vector<uint64_t> all = {7001, 7002, 7003, 7004, 7005, 7006};
    for (size_t i = 0; i < all.size(); ++i) {
        assert(store.register_hot_payload(all[i], (uint32_t) i, all[i], xkv_state::hot_writing, &err));
    }
    assert(store.commit_hot_payloads(all));
    auto s0 = store.current_stamp();

    assert(c.begin(701, 1, s0, &err) == xkv_tx_status::ok);
    assert(c.begin(702, 2, s0, &err) == xkv_tx_status::ok);
    assert(c.begin(703, 3, s0, &err) == xkv_tx_status::ok);
    // Coordinator generations must equal store storage_generations (pid here)
    // so the token-carried removal validates exactly.
    for (uint64_t pid : {7001, 7002}) {
        assert(c.record_payload(701, pid, pid, &err) == xkv_tx_status::ok);
        assert(c.record_byte_complete(701, pid, &err) == xkv_tx_status::ok);
    }
    for (uint64_t pid : {7003, 7004, 7005}) {
        assert(c.record_payload(702, pid, pid, &err) == xkv_tx_status::ok);
        assert(c.record_byte_complete(702, pid, &err) == xkv_tx_status::ok);
    }
    assert(c.record_payload(703, 7006, 7006, &err) == xkv_tx_status::ok);
    assert(c.record_byte_complete(703, 7006, &err) == xkv_tx_status::ok);

    xkv_slot_settlement accept_all;
    accept_all.tx_id = 701;
    accept_all.accepted_ids = {7001, 7002};
    xkv_slot_settlement accept_partial;
    accept_partial.tx_id = 702;
    accept_partial.accepted_ids = {7003};
    xkv_slot_settlement accept_none;
    accept_none.tx_id = 703;
    accept_none.accepted_ids = {};

    bool saw_token = false;
    auto remover = [&](const std::vector<xkv_rejected_item> & items, uint64_t tok,
                       xkv_removal_feedback * fb, std::string * e) {
        assert(tok != 0); // live reservation token must reach the store call
        saw_token = true;
        std::vector<uint64_t> ids;
        std::vector<uint64_t> gens;
        for (const auto & it : items) {
            ids.push_back(it.payload_id);
            gens.push_back(it.generation);
        }
        xkv_payload_removal_result res;
        if (!store.remove_payloads(ids, gens, &res, e, nullptr, nullptr, nullptr, tok)) {
            return false;
        }
        fb->has_store_stamp = true;
        fb->store_stamp = res.stamp;
        return true;
    };

    assert(c.commit_batch({accept_all, accept_partial, accept_none}, remover, &err) == xkv_tx_status::ok);
    assert(saw_token);
    assert(c.active_tx_count() == 0);

    // Exactly one content advance for the whole batch (removal bumped once;
    // apply adopted the feedback stamp without a second bump).
    assert(store.current_stamp().content_epoch == s0.content_epoch + 1);

    // Rejected rows are unlocatable: no later maintain can seal them.
    xkv_location loc;
    assert(!store.find_location(7004, loc));
    assert(!store.find_location(7005, loc));
    assert(!store.find_location(7006, loc));
    // Accepted rows stay hot_committed and locatable (sealable later).
    assert(store.find_location(7001, loc));
    assert(loc.state == xkv_state::hot_committed);
    assert(store.find_location(7002, loc));
    assert(loc.state == xkv_state::hot_committed);
    assert(store.find_location(7003, loc));
    assert(loc.state == xkv_state::hot_committed);

    assert(c.is_eligible_for_maintenance(7001));
    assert(c.is_eligible_for_maintenance(7003));
    assert(!c.is_eligible_for_maintenance(7004));
    assert(!c.is_eligible_for_maintenance(7006));
}

int main() {
    std::cout << "=== Running XKV Transaction Tests ===" << std::endl;
    test_accept_all();
    test_accept_partial();
    test_accept_zero();
    test_prefix_enforcement();
    test_null_callback_required();
    test_remove_callback_failure();
    test_compute_failure_rollback();
    test_nested_overlapping_rejection();
    test_monotonic_ids();
    test_maintenance_exclusion();
    test_reader_drain_final_fence();
    test_fence_excludes_tx();
    test_quiesce_gate_parks_readers();
    test_stale_stamps();
    test_cancellation();
    test_callback_reentry_no_lock();
    test_commit_validation();
    test_clear_quiescence();
    test_held_semantics();
    test_guard_move_assign_settles();
    test_multi_commit_eligibility();
    test_provider_advance_failures_and_overflow();
    test_preflight_reservation_guarantees_zero_removals();
    test_bind_stamp_provider_refusal();
    test_guard_move_assign_dismissed_reset();
    test_concurrent_2_slots_batch_settle();
    test_concurrent_6_slots_batch_settle();
    test_postcompute_rebase_and_peer_isolation();
    test_removal_feedback_store_stamp_accepted();
    test_store_backed_settlement_0_partial_all();
    std::cout << "All XKV transaction tests passed." << std::endl;
    return 0;
}
