#pragma once

#include "tq/rerot/plan.h"
#include "tq/rerot/view.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tq::rerot {

enum class StageState : std::uint8_t { queued, starting, running, sealed };
enum class EpisodeState : std::uint8_t { ready, computing, draining, aborted, finished };
enum class EndOrigin : std::uint8_t { none, worker_source, synthesis_source };

struct Config {
    std::uint64_t episode = 1;
    std::uint32_t pens = 1;
    std::uint64_t max_tokens = 1048576;
    std::size_t max_runs = 65536;
};

struct BaseRun {
    RunId run = 0;
    std::int64_t storage_pos = 0;
    std::uint64_t count = 0;
};

struct Append {
    RunId run = 0;
    SegmentKind kind = SegmentKind::body;
    Visibility visibility = Visibility::public_kv;
    std::int64_t storage_pos = 0;
    std::uint64_t count = 0;
};

struct Member {
    StageId stage = 0;
    std::uint32_t pen = 0;
    StageState state = StageState::queued;
};

struct Completion {
    std::uint64_t computed_rows = 0;
    EndOrigin end = EndOrigin::none;
};

struct Commit {
    std::uint64_t frontier = 0;
    std::uint64_t revision = 0;
    std::vector<StageId> sealed;
    bool finished = false;
};

// A ticket is bound to one Episode instance. Copying it does not create work.
class Step {
public:
    std::uint64_t frontier() const noexcept;
    std::uint64_t read_revision() const noexcept;
    const std::vector<Member> & members() const noexcept;

private:
    std::shared_ptr<const std::uint8_t> identity_;
    std::uint64_t frontier_ = 0;
    std::uint64_t revision_ = 0;
    std::vector<Member> members_;
    friend class Episode;
};

// One writer calls this object. It does not execute model or device work.
// C owns actual KV, recurrent state, samplers, bindings, and completion events.
class Episode {
public:
    Episode(Plan plan, Config config, std::vector<BaseRun> prefix);
    ~Episode();
    Episode(const Episode &) = delete;
    Episode & operator=(const Episode &) = delete;
    Episode(Episode &&) noexcept;
    Episode & operator=(Episode &&) noexcept;

    const Plan & plan() const noexcept;
    EpisodeState state() const noexcept;
    StageState stage_state(StageId stage) const;
    std::uint64_t revision() const noexcept;
    std::uint64_t committed_tokens() const noexcept;
    std::uint64_t run_length(RunId run) const;

    // Keep bound stages on their pens until source end. Fill free pens in rank
    // order. A successor can start only after all its predecessors are sealed.
    Step begin_step();

    // A starting member supplies its complete FRAME. A running member supplies
    // exactly one input token. Peers see that token at the next frontier.
    // A run can grow only at the end of its owning stage's tape.
    std::vector<WriteSpan> prepare(const Step & step, StageId stage, std::vector<Append> appends);

    // Query must name an own key in this step's prepared write spans. Each view
    // contains the committed peer prefixes plus the reader's own causal rows.
    ReadView view(const Step & step, StageId stage, QueryPoint query) const;

    // C calls complete only after the whole member operation has succeeded.
    // Source-end provenance is explicit and cannot be supplied by a FRAME.
    void complete(const Step & step, StageId stage, Completion completion);
    Commit commit(const Step & step);

    // Failure stops the episode. No pending rows become public. C must drain
    // its real work before confirming resource release; this is not a fence.
    void abort(const Step & step);
    void confirm_drained(const Step & step);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tq::rerot
