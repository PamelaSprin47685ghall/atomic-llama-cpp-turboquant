// TP5 relay verification-counter tests. No HIP, no model, no GPU.
//
// relay.hip and this test call the same ggml_relay_verify::verify_submission()
// with a describe callback. The test therefore exercises the production
// request -> resolve -> describe -> execute order and the fast/switch gate with
// a fake describe, instead of reimplementing that sequence:
//   * switch off: a fast path and a describe result count nothing,
//   * requested but no fast path: requested only, describe is not called,
//   * resolved: describe is called once, all three counts stay nested,
//   * describe failure: executed is still counted and the helper returns false,
//   * cumulative counts over several submissions,
//   * consistent() detects a manually drifted count.
//
// It does not execute HIP or prove the relay call site; that seam is reviewed
// in relay.hip. The test pins the shared orchestration that call site uses.

#include "ggml-relay-verify.h"

#include <cstdint>
#include <cstdio>

namespace {

int failures = 0;

bool expect(bool condition, const char * expression, int line) {
    if (!condition) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, expression);
        ++failures;
    }
    return condition;
}

#define CHECK(expr) expect((expr), #expr, __LINE__)

using ggml_relay_verify::ledger;
using ggml_relay_verify::verify_submission;

struct fake_describe {
    int  calls  = 0;
    bool result = true;

    bool operator()() {
        ++calls;
        return result;
    }
};

void test_switch_off_counts_nothing() {
    ledger l;
    fake_describe d;
    bool did_describe = true;

    // The switch is off: a real fast path and a describe that would have run
    // are not verification work and must not appear in the counters.
    CHECK(verify_submission(l, false, true, d, &did_describe));
    CHECK(!did_describe);
    CHECK(d.calls == 0);
    CHECK(l.total.requested == 0);
    CHECK(l.total.resolved == 0);
    CHECK(l.total.executed == 0);
    CHECK(l.consistent());
}

void test_requested_without_fast_path() {
    ledger l;
    fake_describe d;
    bool did_describe = true;

    // Switch on, but the submission did not take the uid fast path: the
    // verification describe must not run.
    CHECK(verify_submission(l, true, false, d, &did_describe));
    CHECK(!did_describe);
    CHECK(d.calls == 0);
    CHECK(l.total.requested == 1);
    CHECK(l.total.resolved == 0);
    CHECK(l.total.executed == 0);
    CHECK(l.consistent());
}

void test_resolved_and_executed() {
    ledger l;
    fake_describe d;
    bool did_describe = false;

    CHECK(verify_submission(l, true, true, d, &did_describe));
    CHECK(did_describe);
    CHECK(d.calls == 1);
    CHECK(l.total.requested == 1);
    CHECK(l.total.resolved == 1);
    CHECK(l.total.executed == 1);
    CHECK(l.consistent());
}

void test_describe_failure_still_counts_executed() {
    ledger l;
    fake_describe d;
    d.result = false;
    bool did_describe = false;

    CHECK(!verify_submission(l, true, true, d, &did_describe));
    CHECK(did_describe);
    CHECK(d.calls == 1);
    CHECK(l.total.requested == 1);
    CHECK(l.total.resolved == 1);
    CHECK(l.total.executed == 1);
    CHECK(l.consistent());
}

void test_cumulative_across_submissions() {
    ledger l;
    int describe_calls = 0;

    // Five accepted submissions with the switch on. Only the fast ones call
    // the describe; the slow ones remain requested-only.
    const bool fasts[] = { true, false, true, false, true };
    for (bool fast : fasts) {
        bool did_describe = false;
        CHECK(verify_submission(l, true, fast,
                [&]() {
                    ++describe_calls;
                    return true;
                }, &did_describe));
        CHECK(did_describe == fast);
    }

    CHECK(describe_calls == 3);
    CHECK(l.total.requested == 5);
    CHECK(l.total.resolved == 3);
    CHECK(l.total.executed == 3);
    CHECK(l.consistent());
}

void test_consistency_detects_drift() {
    ledger ok;
    ok.total.requested = 1;
    ok.total.resolved  = 1;
    ok.total.executed  = 1;
    CHECK(ok.consistent());

    ledger bad;
    bad.total.requested = 1;
    bad.total.resolved  = 1;
    bad.total.executed  = 2;
    CHECK(!bad.consistent());

    ledger inverted;
    inverted.total.requested = 0;
    inverted.total.resolved  = 1;
    CHECK(!inverted.consistent());
}

} // namespace

int main() {
    test_switch_off_counts_nothing();
    test_requested_without_fast_path();
    test_resolved_and_executed();
    test_describe_failure_still_counts_executed();
    test_cumulative_across_submissions();
    test_consistency_detects_drift();
    if (failures != 0) {
        std::fprintf(stderr, "RELAY_VERIFY_COUNTERS_TEST FAIL %d\n", failures);
        return 1;
    }
    std::printf("RELAY_VERIFY_COUNTERS_TEST PASS\n");
    return 0;
}
