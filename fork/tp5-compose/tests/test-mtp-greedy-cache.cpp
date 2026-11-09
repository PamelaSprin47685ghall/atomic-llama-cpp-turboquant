// MTP target greedy row cache: verify/replay/commit row selection.
//
// The cache is per epoch. The tests exercise the branch decisions, not only
// cache forwarding:
//   - partial mismatch with a replay argmax change (the commit must use the
//     replay id for the mismatch row, and the verifier copy must not alias the
//     refreshed entry);
//   - full accept (the new row n is never verified and must be scanned);
//   - LateBind g0 (accepted row and native-trim mismatch may use it; an
//     accepted==0 replay must refresh it);
//   - signed zero/tie and NaN semantics of select_finite_greedy;
//   - the cancellation gate (a stopped request does not commit a cached id);
//   - scan counts (verify + replay only for the cached rows; the new full-accept
//     row is scanned once).

#include "mtp_greedy_cache.h"

#include "tq/sampling/shortlist.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>

namespace {

int g_failures = 0;

void check(bool ok, const char * name) {
    if (!ok) {
        ++g_failures;
    }
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", name);
}

tq::sampling::greedy_result make_greedy(std::int32_t id, bool usable = true) {
    return tq::sampling::greedy_result{usable, tq::sampling::candidate{id, 0.0f}};
}

// The runtime's commit path: a stopped request does not commit; a cached row
// that is unusable is rejected by the require(usable) check.
std::optional<std::int32_t> commit_visible(const tq::tp5::mtp_greedy_cache & cache,
                                           std::size_t row,
                                           std::size_t accepted,
                                           bool native_trim,
                                           bool full_accept,
                                           bool stopped) {
    if (stopped) {
        return std::nullopt;
    }
    if (!cache.can_commit(row, accepted, native_trim, full_accept)) {
        return std::nullopt; // the caller would scan the row
    }
    const auto & result = cache.get(row);
    if (!result.usable) {
        return std::nullopt; // the runtime's require(usable) rejects it
    }
    return result.selected.id;
}

std::int32_t commit_with_scan(const tq::tp5::mtp_greedy_cache & cache,
                              std::size_t row,
                              std::size_t accepted,
                              bool native_trim,
                              bool full_accept,
                              const std::function<tq::sampling::greedy_result(std::size_t)> & scan) {
    if (cache.can_commit(row, accepted, native_trim, full_accept)) {
        return cache.get(row).selected.id;
    }
    return scan(row).selected.id;
}

void test_partial_mismatch_replay_refresh() {
    tq::tp5::mtp_greedy_cache cache;
    cache.store_verify(0, make_greedy(10));
    cache.store_verify(1, make_greedy(20)); // mismatch row

    check(cache.can_commit(0, 1, false, false), "accepted row uses the verify result");
    check(!cache.can_commit(1, 1, false, false),
          "replay mismatch row is not cached before the refresh");

    const auto verifier_copy = cache.get(1);
    check(verifier_copy.selected.id == 20, "verifier copy keeps the original id");

    cache.store_replay(0, make_greedy(10));
    cache.store_replay(1, make_greedy(21)); // replay argmax changed

    check(cache.can_commit(1, 1, false, false),
          "replay mismatch row is usable after the refresh");
    check(cache.get(1).selected.id == 21,
          "commit uses the replay argmax, not the verify argmax");
    check(verifier_copy.selected.id == 20,
          "verifier copy is not aliased by the refresh");
    check(cache.get(0).selected.id == 10, "accepted row id is unchanged by the replay");
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

    int scans = 0;
    const auto id = commit_with_scan(cache, 3, 3, false, true,
                                     [&](std::size_t row) {
                                         ++scans;
                                         check(row == 3, "full-accept scan uses row n");
                                         return make_greedy(33);
                                     });
    check(scans == 1, "full-accept new row is scanned exactly once");
    check(id == 33, "full-accept commit uses the scanned row n id");
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
    check(!commit_visible(cache, 1, 2, false, false, false).has_value(),
          "unusable cached row is not committed");
}

void test_cancellation_gate() {
    tq::tp5::mtp_greedy_cache cache;
    cache.store_verify(0, make_greedy(10));

    check(commit_visible(cache, 0, 1, false, false, false).value() == 10,
          "cached value commits when the request is not stopped");
    check(!commit_visible(cache, 0, 1, false, false, true).has_value(),
          "stopped request does not commit a cached value");
}

void test_scan_counts() {
    {
        int scans = 0;
        const auto scan = [&](std::size_t row) {
            ++scans;
            return make_greedy(30 + static_cast<std::int32_t>(row));
        };

        tq::tp5::mtp_greedy_cache cache;
        cache.store_verify(0, scan(0));
        cache.store_verify(1, scan(1));
        cache.store_replay(0, scan(0));
        cache.store_replay(1, scan(1));

        for (std::size_t row = 0; row < 2; ++row) {
            const auto id = commit_with_scan(cache, row, 1, false, false, scan);
            check(id == 30 + static_cast<std::int32_t>(row),
                  "partial mismatch commit id");
        }
        check(scans == 4, "partial mismatch scans verify and replay only (4)");
    }

    {
        int scans = 0;
        const auto scan = [&](std::size_t row) {
            ++scans;
            return make_greedy(30 + static_cast<std::int32_t>(row));
        };

        tq::tp5::mtp_greedy_cache cache;
        for (std::size_t row = 0; row < 3; ++row) {
            cache.store_verify(row, scan(row));
        }

        for (std::size_t row = 0; row < 3; ++row) {
            check(commit_with_scan(cache, row, 3, false, true, scan) ==
                      30 + static_cast<std::int32_t>(row),
                  "full accept cached row id");
        }
        check(commit_with_scan(cache, 3, 3, false, true, scan) == 33,
              "full accept new row id");
        check(scans == 4, "full accept scans verify rows 0..2 and new row 3 (4)");
    }
}

} // namespace

int main() {
    test_partial_mismatch_replay_refresh();
    test_full_accept_new_row();
    test_latebind_g0();
    test_signed_zero_tie_nan();
    test_cancellation_gate();
    test_scan_counts();

    std::printf("%s failures=%d\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
