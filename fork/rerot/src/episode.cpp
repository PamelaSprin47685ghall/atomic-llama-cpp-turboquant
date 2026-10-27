#include "tq/rerot/episode.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

namespace tq::rerot {
namespace {

constexpr std::uint32_t no_pen = std::numeric_limits<std::uint32_t>::max();

std::int64_t end_position(std::int64_t first, std::uint64_t count) {
    const auto limit = std::numeric_limits<std::int64_t>::max();
    if (first < 0 || count == 0 || count > static_cast<std::uint64_t>(limit - first)) {
        throw std::overflow_error("RERoT storage range is invalid or overflows");
    }
    return first + static_cast<std::int64_t>(count);
}

std::uint64_t add_count(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("RERoT token count overflows");
    }
    return a + b;
}

std::uint64_t next_counter(std::uint64_t value) {
    if (value == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("RERoT counter is exhausted");
    }
    return value + 1;
}

bool valid_visibility(Visibility visibility) {
    return visibility == Visibility::public_kv || visibility == Visibility::private_kv;
}

bool valid_kind(SegmentKind kind) {
    switch (kind) {
        case SegmentKind::prefix:
        case SegmentKind::frame:
        case SegmentKind::body:
        case SegmentKind::source_end:
        case SegmentKind::control: return true;
    }
    return false;
}

} // namespace

struct Episode::Impl {
    struct Stage {
        StageState state = StageState::queued;
        std::uint32_t pen = no_pen;
        std::int64_t next_storage = 0;
        std::vector<RunId> runs;
    };
    struct World {
        EpisodeState phase = EpisodeState::ready;
        std::uint64_t revision = 1;
        std::uint64_t frontier = 0;
        std::uint64_t tokens = 0;
        std::vector<Stage> stages;
        std::vector<RunId> prefix;
        std::map<RunId, WriteSpan> runs;
    };
    struct Pending {
        Member member;
        bool prepared = false;
        std::vector<WriteSpan> writes;
        std::uint64_t rows = 0;
        std::optional<Completion> completion;
    };
    struct Active {
        std::uint64_t frontier = 0;
        std::uint64_t revision = 0;
        std::uint64_t rows = 0;
        std::vector<Pending> members;
    };

    Plan plan;
    Config config;
    std::shared_ptr<const std::uint8_t> identity = std::make_shared<const std::uint8_t>(0);
    std::unique_ptr<World> world = std::make_unique<World>();
    std::unique_ptr<Active> active;

    Impl(Plan input, Config cfg) : plan(std::move(input)), config(cfg) {}

    void ticket(const std::shared_ptr<const std::uint8_t> & key, std::uint64_t frontier,
                std::uint64_t revision, EpisodeState phase = EpisodeState::computing) const {
        if (key != identity || world->phase != phase || !active ||
            active->frontier != frontier || active->revision != revision) {
            throw std::logic_error("RERoT step ticket is stale or belongs to another episode");
        }
    }

    Pending & pending(StageId stage) {
        for (auto & member : active->members) {
            if (member.member.stage == stage) {
                return member;
            }
        }
        throw std::invalid_argument("RERoT stage is not a step member");
    }

    const Pending & pending(StageId stage) const {
        for (const auto & member : active->members) {
            if (member.member.stage == stage) {
                return member;
            }
        }
        throw std::invalid_argument("RERoT stage is not a step member");
    }
};

std::uint64_t Step::frontier() const noexcept { return frontier_; }
std::uint64_t Step::read_revision() const noexcept { return revision_; }
const std::vector<Member> & Step::members() const noexcept { return members_; }

Episode::Episode(Plan plan, Config config, std::vector<BaseRun> prefix)
    : impl_(std::make_unique<Impl>(std::move(plan), config)) {
    if (config.episode == 0 || config.pens == 0 || config.max_tokens == 0 || config.max_runs == 0 ||
        prefix.size() > config.max_runs) {
        throw std::invalid_argument("RERoT episode configuration is invalid");
    }
    auto & world = *impl_->world;
    std::int64_t storage_end = 0;
    for (const auto & base : prefix) {
        const auto end = end_position(base.storage_pos, base.count);
        if (base.run == 0 || base.storage_pos < storage_end || world.runs.count(base.run)) {
            throw std::invalid_argument("RERoT prefix has an invalid run or overlapping storage");
        }
        world.tokens = add_count(world.tokens, base.count);
        if (world.tokens > config.max_tokens) {
            throw std::length_error("RERoT prefix exceeds the token limit");
        }
        world.prefix.push_back(base.run);
        world.runs.emplace(base.run, WriteSpan{base.run, planner_stage, 0, base.count,
            base.storage_pos, SegmentKind::prefix, Visibility::public_kv});
        storage_end = end;
    }
    world.stages.resize(impl_->plan.worker_count() + 2);
    for (auto & stage : world.stages) {
        stage.next_storage = storage_end;
    }
    world.stages[planner_stage].state = StageState::sealed;
}

Episode::~Episode() = default;
Episode::Episode(Episode &&) noexcept = default;
Episode & Episode::operator=(Episode &&) noexcept = default;
const Plan & Episode::plan() const noexcept { return impl_->plan; }
EpisodeState Episode::state() const noexcept { return impl_->world->phase; }
std::uint64_t Episode::revision() const noexcept { return impl_->world->revision; }
std::uint64_t Episode::committed_tokens() const noexcept { return impl_->world->tokens; }

StageState Episode::stage_state(StageId stage) const {
    if (stage >= impl_->world->stages.size()) {
        throw std::out_of_range("RERoT stage is out of range");
    }
    return impl_->world->stages[stage].state;
}

std::uint64_t Episode::run_length(RunId run) const {
    const auto found = impl_->world->runs.find(run);
    return found == impl_->world->runs.end() ? 0 : found->second.count;
}

Step Episode::begin_step() {
    if (state() != EpisodeState::ready || impl_->active) {
        throw std::logic_error("RERoT episode cannot start another step");
    }
    auto next = std::make_unique<Impl::World>(*impl_->world);
    auto active = std::make_unique<Impl::Active>();
    active->frontier = next_counter(next->frontier);
    active->revision = next->revision;
    const std::size_t pens = std::min<std::size_t>(impl_->config.pens, impl_->plan.worker_count());
    std::vector<bool> used(pens, false);
    for (StageId id = 1; id <= impl_->plan.synthesis_stage(); ++id) {
        const auto & stage = next->stages[id];
        if (stage.state == StageState::running || stage.state == StageState::starting) {
            if (stage.pen >= used.size() || used[stage.pen]) {
                throw std::logic_error("RERoT pen binding is inconsistent");
            }
            used[stage.pen] = true;
        }
    }
    const auto bind = [&](StageId id) {
        const auto free = std::find(used.begin(), used.end(), false);
        if (free == used.end()) {
            return false;
        }
        const auto pen = static_cast<std::uint32_t>(free - used.begin());
        *free = true;
        next->stages[id].pen = pen;
        next->stages[id].state = StageState::starting;
        return true;
    };
    bool all_sealed = true;
    for (StageId id = 1; id <= impl_->plan.worker_count(); ++id) {
        auto & stage = next->stages[id];
        all_sealed = all_sealed && stage.state == StageState::sealed;
        if (stage.state != StageState::queued) {
            continue;
        }
        bool ready = true;
        for (const StageId before : impl_->plan.predecessors(id)) {
            ready = ready && next->stages[before].state == StageState::sealed;
        }
        if (ready) {
            (void) bind(id);
        }
    }
    if (all_sealed && next->stages[impl_->plan.synthesis_stage()].state == StageState::queued) {
        if (!bind(impl_->plan.synthesis_stage())) {
            throw std::logic_error("RERoT synthesis has no free pen");
        }
    }
    Step result;
    result.identity_ = impl_->identity;
    result.frontier_ = active->frontier;
    result.revision_ = active->revision;
    for (StageId id = 1; id <= impl_->plan.synthesis_stage(); ++id) {
        const auto & stage = next->stages[id];
        if (stage.state == StageState::running || stage.state == StageState::starting) {
            result.members_.push_back({id, stage.pen, stage.state});
        }
    }
    std::sort(result.members_.begin(), result.members_.end(), [](const Member & a, const Member & b) {
        return a.pen < b.pen;
    });
    if (result.members_.empty()) {
        throw std::logic_error("RERoT episode has no eligible work");
    }
    active->members.reserve(result.members_.size());
    for (const auto & member : result.members_) {
        Impl::Pending pending;
        pending.member = member;
        active->members.push_back(std::move(pending));
    }
    next->phase = EpisodeState::computing;
    impl_->world.swap(next);
    impl_->active.swap(active);
    return result;
}

std::vector<WriteSpan> Episode::prepare(const Step & step, StageId stage, std::vector<Append> appends) {
    impl_->ticket(step.identity_, step.frontier_, step.revision_);
    auto & target = impl_->pending(stage);
    if (target.prepared || appends.empty()) {
        throw std::logic_error("RERoT member is already prepared or has no input rows");
    }
    const auto & owner = impl_->world->stages[stage];
    Impl::Pending candidate;
    candidate.member = target.member;
    candidate.prepared = true;
    std::map<RunId, WriteSpan> changed;
    std::vector<RunId> tape = owner.runs;
    std::int64_t next_storage = owner.next_storage;
    std::set<RunId> reserved;
    for (const auto & member : impl_->active->members) {
        for (const auto & write : member.writes) {
            if (!impl_->world->runs.count(write.run)) {
                reserved.insert(write.run);
            }
        }
    }
    std::size_t new_runs = 0;
    for (const auto & append : appends) {
        const auto end = end_position(append.storage_pos, append.count);
        if (append.run == 0 || !valid_kind(append.kind) || !valid_visibility(append.visibility) ||
            append.storage_pos < next_storage || append.kind == SegmentKind::prefix) {
            throw std::invalid_argument("RERoT append has an invalid identity, kind, or storage range");
        }
        if (owner.state == StageState::starting) {
            if (append.kind != SegmentKind::frame || append.visibility != Visibility::public_kv) {
                throw std::invalid_argument("RERoT starting stage requires a complete public FRAME");
            }
        } else if (append.kind == SegmentKind::frame) {
            throw std::invalid_argument("RERoT FRAME cannot be forwarded twice");
        }
        if (append.kind == SegmentKind::body && append.visibility != Visibility::public_kv) {
            throw std::invalid_argument("RERoT BODY must be public");
        }
        if ((append.kind == SegmentKind::control || append.kind == SegmentKind::source_end) &&
            append.visibility != Visibility::private_kv) {
            throw std::invalid_argument("RERoT control and source end must remain private");
        }
        auto found = changed.find(append.run);
        if (found == changed.end()) {
            const auto old = impl_->world->runs.find(append.run);
            if (old == impl_->world->runs.end()) {
                if (reserved.count(append.run)) {
                    throw std::invalid_argument("RERoT run is already reserved by another member");
                }
                ++new_runs;
                tape.push_back(append.run);
                found = changed.emplace(append.run, WriteSpan{append.run, stage, 0, 0,
                    append.storage_pos, append.kind, append.visibility}).first;
            } else {
                found = changed.emplace(append.run, old->second).first;
            }
        }
        auto & run = found->second;
        if (tape.empty() || tape.back() != append.run || run.stage != stage ||
            run.kind != append.kind || run.visibility != append.visibility ||
            (run.count != 0 && end_position(run.storage_pos, run.count) != append.storage_pos)) {
            throw std::invalid_argument("RERoT run can grow only at its owner's matching tape tail");
        }
        candidate.writes.push_back({append.run, stage, run.count, append.count,
            append.storage_pos, append.kind, append.visibility});
        run.count = add_count(run.count, append.count);
        candidate.rows = add_count(candidate.rows, append.count);
        next_storage = end;
    }
    if (owner.state == StageState::running && candidate.rows != 1) {
        throw std::invalid_argument("RERoT running stages advance one token per frontier");
    }
    const auto rows = add_count(impl_->active->rows, candidate.rows);
    if (add_count(impl_->world->tokens, rows) > impl_->config.max_tokens ||
        new_runs > impl_->config.max_runs - impl_->world->runs.size() ||
        reserved.size() > impl_->config.max_runs - impl_->world->runs.size() - new_runs) {
        throw std::length_error("RERoT prepared work exceeds the resource limit");
    }
    auto result = candidate.writes;
    target = std::move(candidate);
    impl_->active->rows = rows;
    return result;
}

ReadView Episode::view(const Step & step, StageId stage, QueryPoint query) const {
    impl_->ticket(step.identity_, step.frontier_, step.revision_);
    const auto & pending = impl_->pending(stage);
    if (!pending.prepared) {
        throw std::logic_error("RERoT member has no prepared input");
    }
    std::optional<std::int64_t> query_storage;
    for (const auto & write : pending.writes) {
        if (write.run == query.run && query.offset >= write.offset && query.offset - write.offset < write.count) {
            query_storage = write.storage_pos + static_cast<std::int64_t>(query.offset - write.offset);
        }
    }
    if (!query_storage) {
        throw std::invalid_argument("RERoT query is not one of this member's prepared rows");
    }
    ReadView result;
    result.stamp_ = {impl_->config.episode, step.frontier_, step.revision_, stage};
    bool query_found = false;
    std::int64_t virtual_end = 0;
    const auto append_visible = [&](const WriteSpan & write, bool own) {
        std::uint64_t count = write.count;
        if (own) {
            if (write.storage_pos > *query_storage) {
                return;
            }
            count = std::min(count, static_cast<std::uint64_t>(*query_storage - write.storage_pos) + 1);
        } else if (write.visibility != Visibility::public_kv || write.kind == SegmentKind::source_end ||
                   write.kind == SegmentKind::control) {
            return;
        }
        const auto next = end_position(virtual_end, count);
        Span span;
        static_cast<WriteSpan &>(span) = write;
        span.count = count;
        span.virtual_pos = virtual_end;
        if (own && write.run == query.run && query.offset >= write.offset && query.offset - write.offset < count) {
            if (query_found) {
                throw std::logic_error("RERoT query occurs twice in its view");
            }
            query_found = true;
            result.query_virtual_pos_ = virtual_end + static_cast<std::int64_t>(query.offset - write.offset);
        }
        if (!result.spans_.empty()) {
            auto & last = result.spans_.back();
            if (last.run == span.run && last.stage == span.stage && last.kind == span.kind &&
                last.visibility == span.visibility && last.offset + last.count == span.offset &&
                end_position(last.storage_pos, last.count) == span.storage_pos) {
                last.count = add_count(last.count, span.count);
                virtual_end = next;
                return;
            }
        }
        result.spans_.push_back(span);
        virtual_end = next;
    };
    for (const RunId run : impl_->world->prefix) {
        append_visible(impl_->world->runs.at(run), false);
    }
    std::vector<StageId> started;
    for (StageId id = 1; id <= impl_->plan.worker_count(); ++id) {
        if (impl_->world->stages[id].state != StageState::queued) {
            started.push_back(id);
        }
    }
    auto order = impl_->plan.order(stage, started);
    if (stage == impl_->plan.synthesis_stage()) {
        order.push_back(stage);
    }
    for (const StageId id : order) {
        const bool own = id == stage;
        for (const RunId run : impl_->world->stages[id].runs) {
            append_visible(impl_->world->runs.at(run), own);
        }
        if (own) {
            for (const auto & write : pending.writes) {
                append_visible(write, true);
            }
        }
    }
    if (!query_found) {
        throw std::logic_error("RERoT view omits its own query");
    }
    for (const auto & span : result.spans_) {
        const auto delta = span.storage_pos - span.virtual_pos;
        if (delta > 0 && result.query_virtual_pos_ > std::numeric_limits<std::int64_t>::max() - delta) {
            throw std::overflow_error("RERoT effective position overflows");
        }
    }
    result.key_count_ = static_cast<std::uint64_t>(virtual_end);
    return result;
}

void Episode::complete(const Step & step, StageId stage, Completion completion) {
    impl_->ticket(step.identity_, step.frontier_, step.revision_);
    auto & pending = impl_->pending(stage);
    if (!pending.prepared || pending.completion || completion.computed_rows != pending.rows) {
        throw std::logic_error("RERoT completion is duplicate, incomplete, or has a wrong row count");
    }
    const bool source_end = std::any_of(pending.writes.begin(), pending.writes.end(), [](const WriteSpan & write) {
        return write.kind == SegmentKind::source_end;
    });
    switch (completion.end) {
        case EndOrigin::none:
            if (source_end) {
                throw std::invalid_argument("RERoT source end lacks source provenance");
            }
            break;
        case EndOrigin::worker_source:
        case EndOrigin::synthesis_source:
            if (!source_end || pending.member.state != StageState::running ||
                ((stage == impl_->plan.synthesis_stage()) != (completion.end == EndOrigin::synthesis_source))) {
                throw std::invalid_argument("RERoT source end does not match the current stage");
            }
            break;
        default: throw std::invalid_argument("RERoT completion origin is invalid");
    }
    pending.completion = completion;
}

Commit Episode::commit(const Step & step) {
    impl_->ticket(step.identity_, step.frontier_, step.revision_);
    for (const auto & pending : impl_->active->members) {
        if (!pending.prepared || !pending.completion) {
            throw std::logic_error("RERoT cannot publish a partial cohort");
        }
    }
    auto next = std::make_unique<Impl::World>(*impl_->world);
    Commit result;
    result.frontier = step.frontier_;
    result.revision = next_counter(next->revision);
    for (const auto & pending : impl_->active->members) {
        auto & stage = next->stages[pending.member.stage];
        for (const auto & write : pending.writes) {
            auto found = next->runs.find(write.run);
            if (found == next->runs.end()) {
                if (write.offset != 0) {
                    throw std::logic_error("RERoT new run has a nonzero offset");
                }
                next->runs.emplace(write.run, write);
                stage.runs.push_back(write.run);
            } else {
                if (found->second.count != write.offset) {
                    throw std::logic_error("RERoT run publication offset is inconsistent");
                }
                found->second.count = add_count(found->second.count, write.count);
            }
            stage.next_storage = end_position(write.storage_pos, write.count);
        }
        stage.state = StageState::running;
        if (pending.completion->end != EndOrigin::none) {
            stage.state = StageState::sealed;
            stage.pen = no_pen;
            result.sealed.push_back(pending.member.stage);
        }
    }
    result.finished = next->stages[impl_->plan.synthesis_stage()].state == StageState::sealed;
    next->phase = result.finished ? EpisodeState::finished : EpisodeState::ready;
    next->tokens = add_count(next->tokens, impl_->active->rows);
    next->revision = result.revision;
    next->frontier = step.frontier_;
    impl_->world.swap(next);
    impl_->active.reset();
    return result;
}

void Episode::abort(const Step & step) {
    impl_->ticket(step.identity_, step.frontier_, step.revision_);
    impl_->world->phase = EpisodeState::draining;
}

void Episode::confirm_drained(const Step & step) {
    impl_->ticket(step.identity_, step.frontier_, step.revision_, EpisodeState::draining);
    impl_->world->phase = EpisodeState::aborted;
    impl_->active.reset();
}

} // namespace tq::rerot
