#include <tq/rerot/episode.h>
#include <tq/rerot/view.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {
using namespace tq::rerot;

void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

bool contains(const ReadView & view, RunId run, std::uint64_t offset) {
    const auto entries = expand(view, 64);
    return std::any_of(entries.begin(), entries.end(), [&](const Entry & entry) {
        return entry.run == run && entry.offset == offset;
    });
}

void publication() {
    Config config;
    config.pens = 2;
    config.max_tokens = 64;
    config.max_runs = 16;
    Episode episode(Plan({{"first", "Explore the first approach."},
                          {"second", "Explore the second approach."}}),
                    config, {{1, 0, 2}});

    const auto setup = episode.begin_step();
    require(setup.members().size() == 2, "expected two bound workers");
    for (const auto & member : setup.members()) {
        episode.prepare(setup, member.stage,
            {{10 + member.stage, SegmentKind::frame, Visibility::public_kv, 2, 2}});
        episode.complete(setup, member.stage, {2, EndOrigin::none});
    }
    episode.commit(setup);
    require(episode.committed_tokens() == 6, "FRAME publication count differs");

    const auto body = episode.begin_step();
    for (const auto & member : body.members()) {
        episode.prepare(body, member.stage,
            {{20 + member.stage, SegmentKind::body, Visibility::public_kv, 4, 1}});
    }
    const auto frozen = episode.view(body, 1, {21, 0});
    require(contains(frozen, 21, 0), "own current token is missing");
    require(!contains(frozen, 22, 0), "pending peer token became visible");
    episode.complete(body, 1, {1, EndOrigin::none});
    require(episode.run_length(21) == 0 && episode.run_length(22) == 0,
            "one member completion partially published the cohort");
    episode.complete(body, 2, {1, EndOrigin::none});
    const auto committed = episode.commit(body);
    require(committed.revision == body.read_revision() + 1 &&
            episode.run_length(21) == 1 && episode.run_length(22) == 1,
            "BODY cohort was not published together");

    const auto next = episode.begin_step();
    for (const auto & member : next.members()) {
        const RunId own = 20 + member.stage;
        const RunId peer = member.stage == 1 ? 22 : 21;
        episode.prepare(next, member.stage,
            {{own, SegmentKind::body, Visibility::public_kv, 5, 1}});
        const auto view = episode.view(next, member.stage, {own, 1});
        require(contains(view, peer, 0), "committed peer token is not immediately visible");
        require(!contains(view, peer, 1), "next pending peer token became visible");
    }
    require(!contains(frozen, 22, 0) && frozen.key_count() == 7,
            "a committed publication changed an older owned view");
    // This consumer submits no external work, so there is nothing to drain.
    episode.abort(next);
    episode.confirm_drained(next);
    require(episode.state() == EpisodeState::aborted, "abort did not finish");
}
} // namespace

int main() {
    try {
        publication();
        std::cout << "installed TQRerot consumer: two-worker token publication passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "installed TQRerot consumer: " << error.what() << '\n';
        return 1;
    }
}
