#include "tq/rerot/episode.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace tq::rerot;

namespace {

void check(bool condition, int line) {
    if (!condition) {
        throw std::runtime_error("check failed at line " + std::to_string(line));
    }
}
#define CHECK(condition) check((condition), __LINE__)

template<class Exception = std::exception, class Function>
void rejects(Function function) {
    try {
        function();
    } catch (const Exception &) {
        return;
    }
    throw std::runtime_error("an invalid operation was accepted");
}

Plan make_plan(std::size_t count, std::vector<Edge> edges = {}) {
    std::vector<Node> nodes;
    for (std::size_t i = 0; i < count; ++i) {
        nodes.push_back({std::to_string(i + 1), "work " + std::to_string(i + 1)});
    }
    return Plan(std::move(nodes), std::move(edges));
}

bool respects(const std::vector<StageId> & order, const std::vector<Edge> & edges) {
    for (const auto & edge : edges) {
        const auto a = std::find(order.begin(), order.end(), edge.before);
        const auto b = std::find(order.begin(), order.end(), edge.after);
        if (a != order.end() && b != order.end() && a >= b) {
            return false;
        }
    }
    return true;
}

std::vector<StageId> exhaustive_order(std::vector<StageId> started, const std::vector<Edge> & edges,
                                     StageId reader, std::size_t count) {
    std::sort(started.begin(), started.end());
    std::vector<StageId> best;
    std::vector<std::size_t> best_rank;
    do {
        if (!respects(started, edges)) {
            continue;
        }
        std::vector<std::size_t> rank;
        for (const auto stage : started) {
            const auto priority = reader == count + 1 ? stage - 1 :
                (std::size_t(stage) + count - reader - 1) % count;
            rank.push_back(priority);
        }
        if (best.empty() || rank < best_rank) {
            best = started;
            best_rank = std::move(rank);
        }
    } while (std::next_permutation(started.begin(), started.end()));
    return best;
}

void test_dag() {
    std::vector<Edge> possible;
    for (StageId a = 1; a <= 4; ++a) {
        for (StageId b = 1; b <= 4; ++b) {
            if (a != b) {
                possible.push_back({a, b});
            }
        }
    }
    std::size_t dag_count = 0;
    std::size_t state_count = 0;
    std::size_t view_count = 0;
    for (std::uint32_t mask = 0; mask < (1u << possible.size()); ++mask) {
        std::vector<Edge> edges;
        for (std::size_t bit = 0; bit < possible.size(); ++bit) {
            if (mask & (1u << bit)) {
                edges.push_back(possible[bit]);
            }
        }
        const auto truth = exhaustive_order({1, 2, 3, 4}, edges, 5, 4);
        if (truth.empty()) {
            rejects<std::invalid_argument>([&] { (void) make_plan(4, edges); });
            continue;
        }
        Plan plan = make_plan(4, edges);
        ++dag_count;
        CHECK(plan.order(5, {4, 3, 2, 1}) == truth);
        for (std::uint32_t sealed = 0; sealed < 16; ++sealed) {
            const auto done = [&](StageId id) { return (sealed & (1u << (id - 1))) != 0; };
            bool closed = true;
            for (const auto & edge : edges) {
                closed = closed && (!done(edge.after) || done(edge.before));
            }
            if (!closed) {
                continue;
            }
            std::vector<StageId> active;
            std::vector<StageId> started;
            for (StageId id = 1; id <= 4; ++id) {
                bool ready = !done(id);
                for (const auto & edge : edges) {
                    if (edge.after == id && !done(edge.before)) {
                        ready = false;
                    }
                }
                if (ready) {
                    active.push_back(id);
                }
                if (done(id) || ready) {
                    started.push_back(id);
                }
            }
            if (active.empty()) {
                continue;
            }
            ++state_count;
            for (const StageId reader : active) {
                const auto expected = exhaustive_order(started, edges, reader, 4);
                const auto actual = plan.order(reader, started);
                CHECK(actual == expected);
                CHECK(actual.back() == reader);
                auto shuffled = started;
                do {
                    CHECK(plan.order(reader, shuffled) == expected);
                } while (std::next_permutation(shuffled.begin(), shuffled.end()));
                ++view_count;
            }
        }
    }
    CHECK(dag_count == 543);
    CHECK(state_count == 3007);
    CHECK(view_count == 3904);
    rejects<std::invalid_argument>([] { (void) Plan({}); });
    rejects<std::invalid_argument>([] { (void) Plan(std::vector<Node>{{"x", "a"}, {"x", "b"}}); });
    rejects<std::invalid_argument>([] { (void) Plan(std::vector<Node>{{"0", "a"}}); });
    rejects<std::invalid_argument>([] { (void) Plan(std::vector<Node>{{"x", " \n"}}); });
    rejects<std::invalid_argument>([] { (void) make_plan(2, {{1, 1}}); });
    rejects<std::invalid_argument>([] { (void) make_plan(2, {{1, 3}}); });
    rejects<std::invalid_argument>([] { (void) make_plan(2, {{1, 2}, {1, 2}}); });
    rejects<std::invalid_argument>([] { (void) make_plan(2, {{1, 2}, {2, 1}}); });
    rejects<std::invalid_argument>([] { (void) make_plan(2, {{1, 2}}).order(2, {2}); });
    rejects<std::invalid_argument>([] { (void) make_plan(2).order(1, {1, 1}); });
    rejects<std::invalid_argument>([] { (void) make_plan(2).order(1, {2}); });
    rejects<std::invalid_argument>([] { (void) make_plan(2, {{1, 2}}).order(1, {1, 2}); });
    std::cout << "DAGs=" << dag_count << " states=" << state_count << " readers=" << view_count << '\n';
}

Append frame(RunId run, std::int64_t position, std::uint64_t count = 1) {
    return {run, SegmentKind::frame, Visibility::public_kv, position, count};
}

Append body(RunId run, std::int64_t position) {
    return {run, SegmentKind::body, Visibility::public_kv, position, 1};
}

Append source_end(RunId run, std::int64_t position) {
    return {run, SegmentKind::source_end, Visibility::private_kv, position, 1};
}

bool contains(const ReadView & view, RunId run, std::uint64_t offset) {
    for (const auto & span : view.spans()) {
        if (span.run == run && offset >= span.offset && offset - span.offset < span.count) {
            return true;
        }
    }
    return false;
}

void complete_all(Episode & episode, const Step & step, std::uint64_t rows = 1) {
    for (auto it = step.members().rbegin(); it != step.members().rend(); ++it) {
        episode.complete(step, it->stage, {rows, EndOrigin::none});
    }
}

void test_transaction() {
    Episode episode(make_plan(3, {{1, 3}}), {7, 2}, {{1, 0, 3}});
    auto first = episode.begin_step();
    CHECK(first.members().size() == 2);
    CHECK(first.members()[0].stage == 1 && first.members()[1].stage == 2);
    const auto a_pen = first.members()[0].pen;
    const auto b_pen = first.members()[1].pen;
    episode.prepare(first, 1, {frame(10, 3, 2)});
    episode.prepare(first, 2, {frame(20, 3)});
    const auto first_a = episode.view(first, 1, {10, 0});
    CHECK(first_a.key_count() == 4);
    CHECK(!contains(first_a, 10, 1));
    CHECK(!contains(first_a, 20, 0));
    auto frozen_a = episode.view(first, 1, {10, 1});
    CHECK(frozen_a.key_count() == 5);
    episode.complete(first, 1, {2, EndOrigin::none});
    rejects<std::logic_error>([&] { (void) episode.commit(first); });
    CHECK(episode.revision() == 1 && episode.run_length(10) == 0);
    CHECK(!contains(episode.view(first, 2, {20, 0}), 10, 0));
    episode.complete(first, 2, {1, EndOrigin::none});
    const auto committed = episode.commit(first);
    CHECK(committed.revision == 2 && committed.frontier == 1);
    rejects<std::logic_error>([&] { (void) episode.commit(first); });
    rejects<std::logic_error>([&] { (void) episode.view(first, 1, {10, 1}); });
    CHECK(frozen_a.key_count() == 5 && !contains(frozen_a, 20, 0));

    auto second = episode.begin_step();
    episode.prepare(second, 2, {body(21, 4)});
    episode.prepare(second, 1, {body(11, 5)});
    const auto second_a = episode.view(second, 1, {11, 0});
    const auto second_b = episode.view(second, 2, {21, 0});
    CHECK(contains(second_a, 20, 0) && !contains(second_a, 21, 0));
    CHECK(contains(second_b, 10, 1) && !contains(second_b, 11, 0));
    CHECK(second_a.query_virtual_pos() == 6);
    complete_all(episode, second);
    episode.commit(second);

    auto third = episode.begin_step();
    episode.prepare(third, 1, {body(11, 6)});
    episode.prepare(third, 2, {{22, SegmentKind::control, Visibility::private_kv, 5, 1}});
    auto third_b = episode.view(third, 2, {22, 0});
    CHECK(contains(third_b, 11, 0));
    CHECK(!contains(third_b, 11, 1));
    CHECK(contains(third_b, 22, 0));
    CHECK(!contains(episode.view(third, 1, {11, 1}), 22, 0));
    complete_all(episode, third);
    episode.commit(third);
    CHECK(episode.run_length(11) == 2);
    CHECK(!contains(third_b, 11, 1));

    auto fourth = episode.begin_step();
    episode.prepare(fourth, 1, {source_end(12, 7)});
    episode.prepare(fourth, 2, {body(23, 6)});
    CHECK(contains(episode.view(fourth, 2, {23, 0}), 11, 1));
    CHECK(!contains(episode.view(fourth, 2, {23, 0}), 12, 0));
    CHECK(!contains(episode.view(fourth, 1, {12, 0}), 22, 0));
    rejects<std::invalid_argument>([&] { episode.complete(fourth, 1, {1, EndOrigin::none}); });
    rejects<std::invalid_argument>([&] { episode.complete(fourth, 1, {1, EndOrigin::synthesis_source}); });
    episode.complete(fourth, 1, {1, EndOrigin::worker_source});
    CHECK(episode.stage_state(3) == StageState::queued);
    episode.complete(fourth, 2, {1, EndOrigin::none});
    const auto sealed = episode.commit(fourth);
    CHECK(sealed.sealed == std::vector<StageId>{1});

    auto fifth = episode.begin_step();
    CHECK(fifth.members().size() == 2);
    CHECK(fifth.members()[0].stage == 3 && fifth.members()[0].pen == a_pen);
    CHECK(fifth.members()[1].stage == 2 && fifth.members()[1].pen == b_pen);
    episode.prepare(fifth, 3, {frame(30, 3)});
    episode.prepare(fifth, 2, {body(23, 7)});
    const auto successor = episode.view(fifth, 3, {30, 0});
    CHECK(contains(successor, 10, 1) && contains(successor, 11, 1));
    CHECK(contains(successor, 20, 0) && contains(successor, 21, 0));
    CHECK(!contains(successor, 12, 0) && !contains(successor, 22, 0));
    CHECK(!contains(episode.view(fifth, 2, {23, 1}), 30, 0));
    complete_all(episode, fifth);
    episode.commit(fifth);

    auto sixth = episode.begin_step();
    episode.prepare(sixth, 2, {source_end(24, 8)});
    episode.prepare(sixth, 3, {source_end(31, 4)});
    for (const auto & member : sixth.members()) {
        episode.complete(sixth, member.stage, {1, EndOrigin::worker_source});
    }
    episode.commit(sixth);
    auto synthesis = episode.begin_step();
    CHECK(synthesis.members().size() == 1 && synthesis.members()[0].stage == 4);
    episode.prepare(synthesis, 4, {frame(40, 3)});
    const auto final_view = episode.view(synthesis, 4, {40, 0});
    CHECK(contains(final_view, 30, 0) && contains(final_view, 23, 1));
    CHECK(!contains(final_view, 31, 0) && !contains(final_view, 24, 0));
    complete_all(episode, synthesis);
    episode.commit(synthesis);
    auto end = episode.begin_step();
    episode.prepare(end, 4, {source_end(41, 4)});
    episode.complete(end, 4, {1, EndOrigin::synthesis_source});
    CHECK(episode.commit(end).finished);
    CHECK(episode.state() == EpisodeState::finished);
    rejects<std::logic_error>([&] { (void) episode.begin_step(); });

    // A ready third worker waits for a pen even without a dependency edge.
    Episode bounded(make_plan(3), {8, 2}, {});
    auto bound = bounded.begin_step();
    CHECK(bound.members().size() == 2 && bounded.stage_state(3) == StageState::queued);
    bounded.prepare(bound, 1, {frame(1, 0)});
    bounded.prepare(bound, 2, {frame(2, 0)});
    complete_all(bounded, bound);
    bounded.commit(bound);
    for (std::uint32_t round = 0; round < 3; ++round) {
        auto next = bounded.begin_step();
        CHECK(next.members()[0].stage == 1 && next.members()[1].stage == 2);
        CHECK(bounded.stage_state(3) == StageState::queued);
        bounded.prepare(next, 1, {body(10, round + 1)});
        bounded.prepare(next, 2, {body(20, round + 1)});
        complete_all(bounded, next);
        bounded.commit(next);
    }

    // An abort never releases a partial whiteboard, even after one member
    // reports success. An immutable view can outlive its Episode.
    auto failed = bounded.begin_step();
    bounded.prepare(failed, 1, {body(10, 4)});
    bounded.prepare(failed, 2, {body(20, 4)});
    auto saved_view = bounded.view(failed, 1, {10, 3});
    bounded.complete(failed, 1, {1, EndOrigin::none});
    const auto revision = bounded.revision();
    const auto tokens = bounded.committed_tokens();
    bounded.abort(failed);
    CHECK(bounded.state() == EpisodeState::draining);
    CHECK(bounded.revision() == revision && bounded.committed_tokens() == tokens);
    CHECK(bounded.run_length(10) == 3 && bounded.run_length(20) == 3);
    rejects<std::logic_error>([&] { (void) bounded.commit(failed); });
    rejects<std::logic_error>([&] { (void) bounded.view(failed, 2, {20, 3}); });
    rejects<std::logic_error>([&] { (void) bounded.begin_step(); });
    bounded.confirm_drained(failed);
    CHECK(bounded.state() == EpisodeState::aborted);
    rejects<std::logic_error>([&] { bounded.confirm_drained(failed); });
    CHECK(contains(saved_view, 10, 3) && !contains(saved_view, 20, 3));
    std::cout << "token publication, own causal, private, source end, P-bound, synthesis, abort: passed\n";
}

void test_bounds() {
    rejects<std::invalid_argument>([] { Episode e(make_plan(1), {0, 1}, {}); });
    rejects<std::invalid_argument>([] { Episode e(make_plan(1), {1, 0}, {}); });
    rejects<std::invalid_argument>([] { Episode e(make_plan(1), {1, 1}, {{0, 0, 1}}); });
    rejects<std::invalid_argument>([] { Episode e(make_plan(1), {1, 1}, {{1, 0, 2}, {2, 1, 1}}); });
    rejects<std::overflow_error>([] { Episode e(make_plan(1), {1, 1}, {{1, -1, 1}}); });
    rejects<std::overflow_error>([] {
        Episode e(make_plan(1), {1, 1}, {{1, std::numeric_limits<std::int64_t>::max(), 1}});
    });
    rejects<std::length_error>([] { Episode e(make_plan(1), {1, 1, 1, 10}, {{1, 0, 2}}); });

    Episode a(make_plan(2), {1, 2, 20, 4}, {});
    Episode b(make_plan(2), {1, 2, 20, 4}, {});
    auto ta = a.begin_step();
    auto tb = b.begin_step();
    rejects<std::logic_error>([&] { (void) a.prepare(tb, 1, {frame(1, 0)}); });
    rejects<std::invalid_argument>([&] { (void) a.prepare(ta, 3, {frame(1, 0)}); });
    rejects<std::logic_error>([&] { (void) a.view(ta, 1, {1, 0}); });
    rejects<std::invalid_argument>([&] { (void) a.prepare(ta, 1, {body(1, 0)}); });
    rejects<std::overflow_error>([&] { (void) a.prepare(ta, 1, {frame(1, -1)}); });
    rejects<std::overflow_error>([&] { (void) a.prepare(ta, 1, {frame(1, 0, 0)}); });
    rejects<std::invalid_argument>([&] {
        (void) a.prepare(ta, 1, {{1, static_cast<SegmentKind>(100), Visibility::public_kv, 0, 1}});
    });
    rejects<std::invalid_argument>([&] {
        (void) a.prepare(ta, 1, {{1, SegmentKind::frame, static_cast<Visibility>(100), 0, 1}});
    });
    a.prepare(ta, 1, {frame(1, 0, 2)});
    rejects<std::invalid_argument>([&] { (void) a.prepare(ta, 2, {frame(1, 0)}); });
    a.prepare(ta, 2, {frame(2, 0)});
    rejects<std::logic_error>([&] { (void) a.prepare(ta, 1, {frame(3, 2)}); });
    rejects<std::invalid_argument>([&] { (void) a.view(ta, 1, {2, 0}); });
    rejects<std::invalid_argument>([&] { (void) a.view(ta, 1, {1, 2}); });
    rejects<std::logic_error>([&] { a.complete(ta, 1, {1, EndOrigin::none}); });
    rejects<std::invalid_argument>([&] { a.complete(ta, 1, {2, EndOrigin::worker_source}); });
    rejects<std::invalid_argument>([&] { a.complete(ta, 1, {2, static_cast<EndOrigin>(100)}); });
    a.complete(ta, 1, {2, EndOrigin::none});
    rejects<std::logic_error>([&] { a.complete(ta, 1, {2, EndOrigin::none}); });
    a.complete(ta, 2, {1, EndOrigin::none});
    a.commit(ta);
    auto next = a.begin_step();
    rejects<std::invalid_argument>([&] { (void) a.prepare(next, 1, {frame(3, 2)}); });
    rejects<std::invalid_argument>([&] {
        (void) a.prepare(next, 1, {{3, SegmentKind::body, Visibility::public_kv, 2, 2}});
    });
    rejects<std::invalid_argument>([&] {
        (void) a.prepare(next, 1, {{3, SegmentKind::body, Visibility::private_kv, 2, 1}});
    });
    a.prepare(next, 1, {body(3, 2)});
    a.prepare(next, 2, {body(4, 1)});
    complete_all(a, next);
    a.commit(next);
    auto grow = a.begin_step();
    rejects<std::length_error>([&] { (void) a.prepare(grow, 1, {body(5, 3)}); });
    a.prepare(grow, 1, {body(3, 3)});
    a.prepare(grow, 2, {body(4, 2)});
    auto view = a.view(grow, 1, {3, 1});
    rejects<std::length_error>([&] { (void) expand(view, 1); });
    CHECK(expand(view, 20).size() == view.key_count());
    complete_all(a, grow);
    a.commit(grow);

    Episode moved(std::move(b));
    moved.prepare(tb, 1, {frame(1, 0)});
    moved.prepare(tb, 2, {frame(2, 0)});
    complete_all(moved, tb);
    moved.commit(tb);

    Episode limit(make_plan(2), {2, 2, 2, 10}, {});
    auto scarce = limit.begin_step();
    limit.prepare(scarce, 1, {frame(1, 0, 2)});
    rejects<std::length_error>([&] { (void) limit.prepare(scarce, 2, {frame(2, 0)}); });
    CHECK(limit.committed_tokens() == 0);
    rejects<std::logic_error>([&] { (void) limit.commit(scarce); });
    limit.abort(scarce);
    limit.confirm_drained(scarce);

    // Count distinct runs across members, including the base prefix. A failed
    // proposal must not keep a run ID or consume the remaining token budget.
    Episode exact(make_plan(2), {4, 2, 5, 3}, {{1, 0, 1}});
    auto reserved = exact.begin_step();
    exact.prepare(reserved, 1, {frame(2, 1), frame(2, 2)});
    rejects<std::length_error>([&] {
        (void) exact.prepare(reserved, 2, {frame(3, 1), frame(4, 2)});
    });
    rejects<std::length_error>([&] { (void) exact.prepare(reserved, 2, {frame(3, 1, 3)}); });
    CHECK(exact.revision() == 1 && exact.committed_tokens() == 1);
    exact.prepare(reserved, 2, {frame(4, 1, 2)});
    complete_all(exact, reserved, 2);
    exact.commit(reserved);
    CHECK(exact.committed_tokens() == 5 && exact.run_length(3) == 0);
    CHECK(exact.run_length(2) == 2 && exact.run_length(4) == 2);

    // Inputs and result values must not provide a second writer. Keep a view
    // after its episode is gone so a copied pointer cannot pass this check.
    ReadView retained;
    {
        Plan source_plan = make_plan(1);
        Config source_config{5, 1, 20, 8};
        std::vector<BaseRun> source_prefix{{1, 3, 2}};
        Episode owned(source_plan, source_config, source_prefix);
        source_plan = make_plan(2);
        source_config.max_tokens = 1;
        source_prefix[0].count = 10;
        CHECK(owned.plan().worker_count() == 1 && owned.committed_tokens() == 2);
        auto setup = owned.begin_step();
        std::vector<Append> source_appends{frame(2, 5, 2)};
        auto writes = owned.prepare(setup, 1, source_appends);
        source_appends[0].count = 100;
        writes[0].run = 99;
        writes[0].count = 100;
        retained = owned.view(setup, 1, {2, 1});
        CHECK(retained.key_count() == 4 && contains(retained, 2, 1));
        CHECK(!contains(retained, 99, 0));
        owned.complete(setup, 1, {2, EndOrigin::none});
        owned.commit(setup);
        auto end = owned.begin_step();
        owned.prepare(end, 1, {source_end(3, 7)});
        owned.complete(end, 1, {1, EndOrigin::worker_source});
        auto result = owned.commit(end);
        CHECK(result.sealed == std::vector<StageId>{1});
        result.sealed.clear();
        CHECK(owned.stage_state(1) == StageState::sealed);
        CHECK(owned.run_length(1) == 2 && owned.run_length(2) == 2);
    }
    auto copied_entries = expand(retained, 4);
    CHECK(copied_entries.size() == 4 && copied_entries.back().run == 2);
    copied_entries.back().run = 99;
    CHECK(expand(retained, 4).back().run == 2);

    // Large writer positions must not wrap when another reader puts the run
    // near the start of its virtual document.
    Episode high(make_plan(2), {3, 2}, {});
    auto large = high.begin_step();
    high.prepare(large, 1, {frame(1, std::numeric_limits<std::int64_t>::max() - 1)});
    high.prepare(large, 2, {frame(2, 0, 3)});
    high.complete(large, 1, {1, EndOrigin::none});
    high.complete(large, 2, {3, EndOrigin::none});
    high.commit(large);
    auto boundary = high.begin_step();
    high.prepare(boundary, 2, {body(3, 3)});
    rejects<std::overflow_error>([&] { (void) expand(high.view(boundary, 2, {3, 0}), 10); });
    high.abort(boundary);
    high.confirm_drained(boundary);
    std::cout << "identity, stale tickets, enum/range/row/resource limits, signed position overflow: passed\n";
}

using Vector = std::array<double, 8>;
using Value = std::array<double, 3>;
using Matrix = std::array<Vector, 8>;
using Key = std::pair<RunId, std::uint64_t>;

struct ExpectedKey {
    Key key;
    std::int64_t storage;
};

Vector rotate(Vector input, std::int64_t position, bool interleaved) {
    for (std::size_t pair = 0; pair < 2; ++pair) {
        const std::size_t a = interleaved ? 2 * pair : pair;
        const std::size_t b = interleaved ? 2 * pair + 1 : pair + 2;
        const double angle = static_cast<double>(position) * std::pow(10000.0, -double(pair) / 2.0);
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        const double x = input[a];
        const double y = input[b];
        input[a] = x * c - y * s;
        input[b] = x * s + y * c;
    }
    return input;
}

Matrix storage_matrix(bool transformed) {
    Matrix result{};
    for (std::size_t row = 0; row < 8; ++row) {
        for (std::size_t col = 0; col < 8; ++col) {
            if (!transformed) {
                result[row][col] = row == col ? 1.0 : 0.0;
                continue;
            }
            std::size_t bits = row & col;
            bool negative = (col & 1u) != 0;
            while (bits) {
                negative = !negative;
                bits &= bits - 1;
            }
            result[row][col] = (negative ? -1.0 : 1.0) / std::sqrt(8.0);
        }
    }
    return result;
}

Vector project(const Matrix & matrix, const Vector & input, bool transpose) {
    Vector out{};
    for (std::size_t row = 0; row < 8; ++row) {
        for (std::size_t col = 0; col < 8; ++col) {
            out[row] += (transpose ? matrix[col][row] : matrix[row][col]) * input[col];
        }
    }
    return out;
}

double dot(const Vector & a, const Vector & b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

struct Cell {
    std::array<Vector, 2> keys;
    std::array<Value, 2> values;
};

Value dense_softmax(const std::vector<double> & scores, const std::vector<Value> & values) {
    const double maximum = *std::max_element(scores.begin(), scores.end());
    double denominator = 0.0;
    Value result{};
    for (std::size_t i = 0; i < scores.size(); ++i) {
        const double weight = std::exp(scores[i] - maximum);
        denominator += weight;
        for (std::size_t d = 0; d < result.size(); ++d) {
            result[d] += weight * values[i][d];
        }
    }
    for (auto & item : result) {
        item /= denominator;
    }
    return result;
}

Value online_softmax(const std::vector<double> & scores, const std::vector<Value> & values) {
    double maximum = -std::numeric_limits<double>::infinity();
    double denominator = 0.0;
    Value result{};
    for (std::size_t i = 0; i < scores.size(); ++i) {
        const double next = std::max(maximum, scores[i]);
        const double old_weight = std::exp(maximum - next);
        const double weight = std::exp(scores[i] - next);
        denominator = denominator * old_weight + weight;
        for (std::size_t d = 0; d < result.size(); ++d) {
            result[d] = result[d] * old_weight + weight * values[i][d];
        }
        maximum = next;
    }
    for (auto & item : result) {
        item /= denominator;
    }
    return result;
}

void test_oracle() {
    Episode episode(make_plan(2), {91, 2}, {{1, 0, 3}});
    auto setup = episode.begin_step();
    episode.prepare(setup, 1, {frame(10, 3, 2)});
    episode.prepare(setup, 2, {frame(20, 3)});
    episode.complete(setup, 1, {2, EndOrigin::none});
    episode.complete(setup, 2, {1, EndOrigin::none});
    episode.commit(setup);
    for (std::uint64_t round = 0; round < 2; ++round) {
        auto step = episode.begin_step();
        episode.prepare(step, 1, {body(11, 5 + static_cast<std::int64_t>(round))});
        episode.prepare(step, 2, {body(21, 4 + static_cast<std::int64_t>(round))});
        complete_all(episode, step);
        episode.commit(step);
    }
    auto step = episode.begin_step();
    episode.prepare(step, 1, {body(11, 7)});
    episode.prepare(step, 2, {body(21, 6)});
    const std::array<ReadView, 2> views = {
        episode.view(step, 1, {11, 2}), episode.view(step, 2, {21, 2})
    };
    // This list is independent of the library's topology and span builders.
    const std::array<std::vector<ExpectedKey>, 2> expected = {{
        {{{1, 0}, 0}, {{1, 1}, 1}, {{1, 2}, 2}, {{20, 0}, 3}, {{21, 0}, 4}, {{21, 1}, 5},
         {{10, 0}, 3}, {{10, 1}, 4}, {{11, 0}, 5}, {{11, 1}, 6}, {{11, 2}, 7}},
        {{{1, 0}, 0}, {{1, 1}, 1}, {{1, 2}, 2}, {{10, 0}, 3}, {{10, 1}, 4}, {{11, 0}, 5},
         {{11, 1}, 6}, {{20, 0}, 3}, {{21, 0}, 4}, {{21, 1}, 5}, {{21, 2}, 6}}
    }};
    std::array<std::vector<Entry>, 2> entries = {expand(views[0], 20), expand(views[1], 20)};
    for (std::size_t reader = 0; reader < 2; ++reader) {
        CHECK(entries[reader].size() == expected[reader].size());
        CHECK(views[reader].query_virtual_pos() == 10);
        for (std::size_t i = 0; i < expected[reader].size(); ++i) {
            const auto & entry = entries[reader][i];
            CHECK((Key{entry.run, entry.offset} == expected[reader][i].key));
            CHECK(entry.storage_pos == expected[reader][i].storage);
            CHECK(entry.virtual_pos == static_cast<std::int64_t>(i));
        }
    }
    std::uint64_t random = 0x243f6a8885a308d3ULL;
    const auto draw = [&]() {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        return (double(random & 0xffffu) / 32768.0) - 1.0;
    };
    double max_error = 0.0;
    double wrong_phase_error = 0.0;
    std::size_t compared = 0;
    for (const bool interleaved : {false, true}) {
        for (const bool transformed : {false, true}) {
            const auto matrix = storage_matrix(transformed);
            for (std::size_t trial = 0; trial < 100; ++trial) {
                std::map<Key, Cell> cells;
                for (const auto & reader : expected) {
                    for (const auto & entry : reader) {
                        if (cells.count(entry.key)) {
                            continue;
                        }
                        Cell cell;
                        for (std::size_t head = 0; head < 2; ++head) {
                            Vector raw;
                            for (auto & item : raw) {
                                item = 3.0 * draw();
                            }
                            cell.keys[head] = project(matrix, rotate(raw, entry.storage, interleaved), false);
                            for (auto & item : cell.keys[head]) {
                                item = std::round(item * 5.0) / 5.0;
                            }
                            for (auto & item : cell.values[head]) {
                                item = draw();
                            }
                        }
                        cells.emplace(entry.key, cell);
                    }
                }
                for (std::size_t reader = 0; reader < 2; ++reader) {
                    for (std::size_t qhead = 0; qhead < 4; ++qhead) {
                        // Four Q heads share two KV heads. Tail dimensions
                        // 4..7 remain unrotated in both pairing layouts.
                        const std::size_t kvhead = qhead / 2;
                        Vector raw_q;
                        for (auto & item : raw_q) {
                            item = 2.0 * draw();
                        }
                        const Vector reference_q = rotate(raw_q, 10, interleaved);
                        std::vector<double> reference_scores;
                        std::vector<double> actual_scores;
                        std::vector<double> wrong_scores;
                        std::vector<Value> values;
                        for (std::size_t i = 0; i < expected[reader].size(); ++i) {
                            const auto & ref = expected[reader][i];
                            const auto & cell = cells.at(ref.key);
                            const auto decoded = project(matrix, cell.keys[kvhead], true);
                            const auto virtual_k = rotate(decoded, static_cast<std::int64_t>(i) - ref.storage, interleaved);
                            reference_scores.push_back(0.37 * dot(reference_q, virtual_k));
                            const auto & entry = entries[reader][i];
                            const auto effective_q = project(matrix, rotate(raw_q, entry.effective_pos, interleaved), false);
                            actual_scores.push_back(0.37 * dot(effective_q, cell.keys[kvhead]));
                            const auto wrong_q = project(matrix, rotate(raw_q, reader == 0 ? 7 : 6, interleaved), false);
                            wrong_scores.push_back(0.37 * dot(wrong_q, cell.keys[kvhead]));
                            values.push_back(cell.values[kvhead]);
                        }
                        const auto truth = dense_softmax(reference_scores, values);
                        const auto actual = online_softmax(actual_scores, values);
                        const auto wrong = online_softmax(wrong_scores, values);
                        for (std::size_t d = 0; d < truth.size(); ++d) {
                            max_error = std::max(max_error, std::abs(truth[d] - actual[d]));
                            wrong_phase_error = std::max(wrong_phase_error, std::abs(truth[d] - wrong[d]));
                            CHECK(std::abs(truth[d] - actual[d]) < 2e-12);
                        }
                        ++compared;
                    }
                }
            }
        }
    }
    CHECK(wrong_phase_error > 0.01);
    const Value zero{0.0, 0.0, 0.0};
    const Value one{1.0, 1.0, 1.0};
    const auto whole = dense_softmax({0.0, std::log(3.0)}, {zero, one});
    const auto separate_a = dense_softmax({0.0}, {zero});
    const auto separate_b = dense_softmax({std::log(3.0)}, {one});
    CHECK(std::abs(whole[0] - 0.75) < 1e-15);
    CHECK(std::abs(whole[0] - (separate_a[0] + separate_b[0])) > 0.2);
    CHECK(std::abs(whole[0] - (separate_a[0] + separate_b[0]) / 2.0) > 0.2);
    std::cout << "F64 reader/head comparisons=" << compared << " max_error=" << max_error
              << " wrong_phase_error=" << wrong_phase_error << '\n';
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 2) {
            throw std::invalid_argument("expected dag, transaction, bounds, or oracle");
        }
        const std::string name = argv[1];
        if (name == "dag") {
            test_dag();
        } else if (name == "transaction") {
            test_transaction();
        } else if (name == "bounds") {
            test_bounds();
        } else if (name == "oracle") {
            test_oracle();
        } else {
            throw std::invalid_argument("unknown test name");
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
