// MTP target greedy cache boundary/semantics.
//
// These are deterministic tests of the cache class only. They do not compile
// or execute runtime.cpp; the runtime wiring, the cancellation/exception path,
// no_sync_logits_calls, capture_logits and the real end-to-end are not covered
// here.

#include "mtp_greedy_cache.h"

#include "tq/sampling/shortlist.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {

int g_failures = 0;

void check(bool ok, const char * name) {
    if (!ok) {
        ++g_failures;
    }
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
}

tq::sampling::greedy_result make_greedy(std::int32_t id, float logit = 0.0f) {
    return tq::sampling::greedy_result{true, tq::sampling::candidate{id, logit}};
}

void test_partial_mismatch_replay_refresh() {
    tq::tp5::mtp_greedy_cache cache;
    cache.store_verify(0, make_greedy(10, 1.0f));
    cache.store_verify(1, make_greedy(20, 2.0f)); // mismatch row

    check(cache.can_commit(0, 1, false, false), "accepted row uses the verify result");
    check(!cache.can_commit(1, 1, false, false),
          "replay mismatch row is not cached before the refresh");

    const auto verifier_copy = cache.get(1);
    check(verifier_copy.selected.id == 20, "verifier copy keeps the original id");
    check(verifier_copy.selected.logit == 2.0f, "verifier copy keeps the verify bits");

    cache.store_replay(0, make_greedy(10, 3.0f));
    cache.store_replay(1, make_greedy(21, 4.0f)); // replay argmax changed

    check(cache.can_commit(1, 1, false, false),
          "replay mismatch row is usable after the refresh");
    check(cache.get(1).selected.id == 21,
          "replay refresh replaces the mismatch id");
    check(verifier_copy.selected.id == 20,
          "verifier copy is not aliased by the refresh");
    check(verifier_copy.selected.logit == 2.0f,
          "verifier copy logit bits are not aliased by the refresh");
    check(cache.get(0).selected.id == 10, "accepted row id is unchanged by the replay");
    check(cache.get(0).selected.logit == 3.0f,
          "accepted row logit bits are the replay result");
}

void test_full_accept_new_row() {
    tq::tp5::mtp_greedy_cache cache;
    for (std::size_t row = 0; row < 3; ++row) {
        cache.store_verify(row, make_greedy(10 + static_cast<std::int32_t>(row)));
    }

    check(cache.can_commit(0, 3, false, true), "full accept row 0 uses the verify result");
    check(cache.can_commit(1, 3, false, true), "full accept row 1 uses the verify result");
    check(cache.can_commit(2, 3, false, true), "full accept row 2 uses the verify result");
    check(!cache.can_commit(3, 3, false, true),
          "full-accept new row n is never cached");

    // A future path must not bypass the guard by caching row n directly.
    cache.store_verify(3, make_greedy(99));
    check(!cache.can_commit(3, 3, true, true),
          "full-accept row n is rejected even if cached (native-trim)");
    check(!cache.can_commit(3, 3, false, true),
          "full-accept row n is rejected even if cached (replay)");
    cache.store_replay(3, make_greedy(98));
    check(!cache.can_commit(3, 3, true, true),
          "full-accept row n is rejected after a replay store (native-trim)");
    check(!cache.can_commit(3, 3, false, true),
          "full-accept row n is rejected after a replay store (replay)");
}

void test_latebind_g0() {
    tq::tp5::mtp_greedy_cache cache;
    cache.store_verify(0, make_greedy(7)); // LateBind g0

    check(cache.can_commit(0, 1, false, false), "g0 is usable for an accepted row 0");
    check(cache.can_commit(0, 0, true, false), "g0 is usable for a native-trim mismatch");
    check(!cache.can_commit(0, 0, false, false),
          "g0 is not usable for an accepted==0 replay before the refresh");

    cache.store_replay(0, make_greedy(8));
    check(cache.can_commit(0, 0, false, false),
          "g0 is usable for an accepted==0 replay after the refresh");
    check(cache.get(0).selected.id == 8, "g0 refresh replaces the id");
}

void test_signed_zero_tie_nan() {
    const float zeros[] = {-0.0f, 0.0f, -1.0f};
    const auto z = tq::sampling::select_finite_greedy(zeros, 3);
    check(z.usable && z.selected.id == 0, "signed-zero tie selects the smallest id");

    std::uint32_t bits = 0;
    std::memcpy(&bits, &z.selected.logit, sizeof(bits));
    check(bits == 0x80000000u, "signed-zero tie retains the first input bits");

    tq::tp5::mtp_greedy_cache cache;
    cache.store_verify(0, z);
    check(cache.get(0).usable && cache.get(0).selected.id == 0,
          "cache preserves the signed-zero tie result");

    const float nan_row[] = {std::numeric_limits<float>::quiet_NaN(), 1.0f};
    const auto n = tq::sampling::select_finite_greedy(nan_row, 2);
    check(!n.usable, "NaN row is unusable");
    cache.store_verify(1, n);
    check(!cache.get(1).usable, "cache preserves the unusable result");
}

void test_row_bounds() {
    tq::tp5::mtp_greedy_cache cache;
    cache.store_verify(4, make_greedy(40));
    check(!cache.has(4), "verify store beyond max_rows is ignored");
    check(!cache.can_commit(4, 4, false, false), "out-of-range row cannot commit");
    cache.store_replay(4, make_greedy(41));
    check(!cache.has(4), "replay store beyond max_rows is ignored");
}

} // namespace

int main() {
    test_partial_mismatch_replay_refresh();
    test_full_accept_new_row();
    test_latebind_g0();
    test_signed_zero_tie_nan();
    test_row_bounds();

    std::printf("%s failures=%d\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
