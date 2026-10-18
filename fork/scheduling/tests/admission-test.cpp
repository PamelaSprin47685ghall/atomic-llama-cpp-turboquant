#include <tq/scheduling/admission.h>

#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

using tq::scheduling::admission_budget;

static void check(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

template<class F> static void rejects(F call) {
    bool rejected = false;
    try { call(); } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "invalid demand was accepted");
}

static void test_capacity_and_slots() {
    rejects([] { admission_budget invalid(0, 1); });
    rejects([] { admission_budget invalid(1, 0); });
    admission_budget budget(100, 2);
    auto first = budget.try_acquire(8, 60);
    check(first && first->slot() == 0 && first->request() == 8 && first->tokens() == 60, "wrong first reservation");
    check(!budget.try_acquire(9, 41), "token capacity was exceeded");
    auto second = budget.try_acquire(9, 40);
    check(second && second->slot() == 1, "exact capacity did not fit");
    check(budget.reserved_tokens() == 100 && budget.available_tokens() == 0 && budget.active_count() == 2,
          "capacity counters disagree");
    check(budget.release(*first), "first release failed");
    auto third = budget.try_acquire(10, 1);
    check(third && third->slot() == 0, "free slot was not reused");
    check(!budget.try_acquire(11, 1), "slot capacity was exceeded");
    check(budget.reserved_tokens() == 41 && budget.available_tokens() == 59, "slot rejection changed token count");
    check(budget.release(*second) && budget.release(*third), "live reservation was rejected");
    check(budget.active_count() == 0 && budget.available_tokens() == 100, "budget did not return to empty");
}

static void test_stale_and_foreign_owners() {
    admission_budget first(10, 1);
    admission_budget second(10, 1);
    auto a = first.try_acquire(0, 10);
    auto b = second.try_acquire(0, 10);
    check(!first.release(*b) && !second.release(*a), "a foreign budget released matching values");
    const auto old = *a;
    check(first.release(*a), "release failed");
    auto next = first.try_acquire(0, 10);
    check(next->generation() > old.generation(), "slot reuse did not change its generation");
    check(!first.release(old), "an old reservation released a new request");
    check(first.reserved_tokens() == 10 && first.active_count() == 1, "stale release changed accounting");
    auto moved = std::move(*next);
    check(!first.release(*next), "moved-from reservation released state");
    check(first.release(moved) && !first.release(moved), "release is not protected against duplicate calls");
    check(second.release(*b), "foreign rejection damaged its owner");
}

static void test_invalid_calls_preserve_state() {
    admission_budget budget(12, 2);
    auto held = budget.try_acquire(5, 7);
    rejects([&] { budget.try_acquire(5, 1); });
    rejects([&] { budget.try_acquire(6, 0); });
    rejects([&] { budget.try_acquire(6, 13); });
    check(budget.reserved_tokens() == 7 && budget.active_count() == 1, "failed calls changed the budget");
    check(budget.release(*held), "failed calls invalidated a live reservation");
}

static void test_large_counts_and_independent_lifetimes() {
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    admission_budget budget(maximum, 2);
    auto first = budget.try_acquire(maximum, maximum - 1);
    check(!budget.try_acquire(1, 2), "addition overflow bypassed the budget");
    auto last = budget.try_acquire(1, 1);
    check(last && budget.reserved_tokens() == maximum, "UINT64_MAX capacity was not represented exactly");
    check(budget.release(*first) && budget.release(*last), "large reservation could not be released");
    auto orphan = [] {
        admission_budget temporary(1, 1);
        return *temporary.try_acquire(7, 1);
    }();
    admission_budget later(1, 1);
    auto fresh = later.try_acquire(7, 1);
    check(!later.release(orphan) && later.release(*fresh), "destroyed owner identity was reused");
}

int main() {
    try {
        test_capacity_and_slots();
        test_stale_and_foreign_owners();
        test_invalid_calls_preserve_state();
        test_large_counts_and_independent_lifetimes();
        std::cout << "admission budget tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
