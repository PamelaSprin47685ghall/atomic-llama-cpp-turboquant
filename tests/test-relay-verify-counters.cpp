// TP5 relay verification-counter tests. No HIP, no model, no GPU.
//
// These pin the ledger semantics that relay.hip binds to the one describe
// call of GGML_TP5_RELAY_VERIFY_UID:
//   * a submission that did not request verification counts nothing,
//   * a requested submission that never reached the uid fast path counts
//     requested only,
//   * a resolved submission counts requested+resolved, and the describe
//     that ran adds executed,
//   * the three counts stay nested (executed <= resolved <= requested),
//   * counters are cumulative: begin/flush/destroy never clears them.
//
// The same header is compiled into relay.hip, so a failure here is a counting
// failure in the relay, not a test-fixture failure.

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

void test_disabled_switch_counts_nothing() {
    ledger l;
    // The switch is off: even a real fast path and a describe that ran are
    // not verification work and must not appear in the counters.
    l.request(false);
    l.resolve(true);
    l.execute();
    CHECK(l.total.requested == 0);
    CHECK(l.total.resolved == 0);
    CHECK(l.total.executed == 0);
    CHECK(l.consistent());
}

void test_requested_without_fast_path() {
    ledger l;
    // Switch on, but the submission took the slow path: the verification
    // describe never ran. This is the "no actual verify hit" case a run log
    // must be able to show.
    l.request(true);
    l.resolve(false);
    l.execute();
    CHECK(l.total.requested == 1);
    CHECK(l.total.resolved == 0);
    CHECK(l.total.executed == 0);
    CHECK(l.consistent());
}

void test_resolved_and_executed() {
    ledger l;
    l.request(true);
    l.resolve(true);
    l.execute();
    CHECK(l.total.requested == 1);
    CHECK(l.total.resolved == 1);
    CHECK(l.total.executed == 1);
    CHECK(l.consistent());
}

void test_execute_requires_resolve() {
    ledger l;
    l.request(true);
    // A describe that runs outside the resolved fast path is not a
    // verification describe. Order is enforced, not assumed.
    l.execute();
    CHECK(l.total.executed == 0);
    l.resolve(true);
    l.execute();
    CHECK(l.total.executed == 1);
}

void test_single_count_per_submission() {
    ledger l;
    l.request(true);
    l.resolve(true);
    l.resolve(true);
    l.execute();
    l.execute();
    CHECK(l.total.resolved == 1);
    CHECK(l.total.executed == 1);
}

void test_cumulative_across_submissions() {
    ledger l;
    l.request(true);
    l.resolve(true);
    l.execute();
    l.request(true);
    l.resolve(true);
    l.execute();
    CHECK(l.total.requested == 2);
    CHECK(l.total.resolved == 2);
    CHECK(l.total.executed == 2);

    // A submission that never reached the relay (refused submit) does not
    // call request(); it must not disturb the per-submission state either.
    CHECK(l.consistent());
}

void test_mixed_stream() {
    ledger l;
    // 10 submissions: 5 request verification, 3 resolve, 2 execute.
    for (int i = 0; i < 10; ++i) {
        const bool requested = i < 5;
        const bool fast      = i < 3;
        const bool described = i < 2;
        l.request(requested);
        l.resolve(fast);
        if (described) {
            l.execute();
        }
    }
    CHECK(l.total.requested == 5);
    CHECK(l.total.resolved == 3);
    CHECK(l.total.executed == 2);
    CHECK(l.consistent());
}

void test_consistency_detects_drift() {
    ledger ok;
    ok.request(true);
    ok.resolve(true);
    ok.execute();
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
    test_disabled_switch_counts_nothing();
    test_requested_without_fast_path();
    test_resolved_and_executed();
    test_execute_requires_resolve();
    test_single_count_per_submission();
    test_cumulative_across_submissions();
    test_mixed_stream();
    test_consistency_detects_drift();
    if (failures != 0) {
        std::fprintf(stderr, "RELAY_VERIFY_COUNTERS_TEST FAIL %d\n", failures);
        return 1;
    }
    std::printf("RELAY_VERIFY_COUNTERS_TEST PASS\n");
    return 0;
}
