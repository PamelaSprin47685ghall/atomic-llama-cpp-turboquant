#include "tq/rerot/plan.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>

namespace tq::rerot {
namespace {

bool blank(const std::string & text) {
    return text.empty() || std::all_of(text.begin(), text.end(), [](unsigned char c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
    });
}

} // namespace

Plan::Plan(std::vector<Node> nodes, std::vector<Edge> edges, PlanLimits limits)
    : nodes_(std::move(nodes)), edges_(std::move(edges)) {
    if (nodes_.empty() || nodes_.size() > limits.nodes ||
        nodes_.size() > std::numeric_limits<StageId>::max() - 2u || edges_.size() > limits.edges) {
        throw std::invalid_argument("RERoT plan exceeds its node or edge limit");
    }
    std::set<std::string> names;
    std::size_t text_bytes = 0;
    for (const auto & item : nodes_) {
        if (blank(item.name) || blank(item.intent) || item.name == "0" ||
            item.name == "0.plan" || item.name == "0.synthesize" || !names.insert(item.name).second) {
            throw std::invalid_argument("RERoT plan has an invalid or duplicate node name");
        }
        for (const auto * value : { &item.name, &item.intent }) {
            if (value->size() > limits.text_bytes - text_bytes) {
                throw std::invalid_argument("RERoT plan exceeds its text limit");
            }
            text_bytes += value->size();
        }
    }
    before_.resize(nodes_.size() + 1);
    after_.resize(nodes_.size() + 1);
    std::set<std::pair<StageId, StageId>> seen;
    for (const auto & edge : edges_) {
        if (edge.before == 0 || edge.after == 0 || edge.before > nodes_.size() ||
            edge.after > nodes_.size() || edge.before == edge.after ||
            !seen.emplace(edge.before, edge.after).second) {
            throw std::invalid_argument("RERoT plan has an invalid or duplicate edge");
        }
        before_[edge.after].push_back(edge.before);
        after_[edge.before].push_back(edge.after);
    }
    std::vector<std::size_t> degrees(before_.size());
    std::queue<StageId> ready;
    for (StageId stage = 1; stage <= nodes_.size(); ++stage) {
        degrees[stage] = before_[stage].size();
        if (degrees[stage] == 0) {
            ready.push(stage);
        }
    }
    std::size_t visited = 0;
    while (!ready.empty()) {
        const StageId stage = ready.front();
        ready.pop();
        ++visited;
        for (const StageId next : after_[stage]) {
            if (--degrees[next] == 0) {
                ready.push(next);
            }
        }
    }
    if (visited != nodes_.size()) {
        throw std::invalid_argument("RERoT plan contains a cycle");
    }
}

std::size_t Plan::worker_count() const noexcept { return nodes_.size(); }
StageId Plan::synthesis_stage() const noexcept { return static_cast<StageId>(nodes_.size() + 1); }
const std::vector<Edge> & Plan::edges() const noexcept { return edges_; }

const Node & Plan::node(StageId stage) const {
    if (stage == 0 || stage > nodes_.size()) {
        throw std::out_of_range("RERoT worker stage is out of range");
    }
    return nodes_[stage - 1];
}

const std::vector<StageId> & Plan::predecessors(StageId worker) const {
    (void) node(worker);
    return before_[worker];
}

std::vector<StageId> Plan::order(StageId reader, const std::vector<StageId> & started) const {
    if (reader == 0 || reader > synthesis_stage()) {
        throw std::invalid_argument("RERoT reader stage is out of range");
    }
    std::vector<bool> present(nodes_.size() + 1, false);
    for (const StageId stage : started) {
        if (stage == 0 || stage > nodes_.size() || present[stage]) {
            throw std::invalid_argument("RERoT started set has an invalid or duplicate stage");
        }
        present[stage] = true;
    }
    const bool synthesis = reader == synthesis_stage();
    if (!synthesis && !present[reader]) {
        throw std::invalid_argument("RERoT reader has not started");
    }
    for (const StageId stage : started) {
        for (const StageId before : before_[stage]) {
            if (!present[before]) {
                throw std::invalid_argument("RERoT started set omits a predecessor");
            }
        }
    }
    if (!synthesis) {
        for (const StageId next : after_[reader]) {
            if (present[next]) {
                throw std::invalid_argument("RERoT active reader has a started successor");
            }
        }
    }
    using Choice = std::pair<std::size_t, StageId>;
    std::priority_queue<Choice, std::vector<Choice>, std::greater<Choice>> ready;
    std::vector<std::size_t> degrees(nodes_.size() + 1);
    const auto priority = [&](StageId stage) {
        return synthesis ? std::size_t(stage - 1) :
            (std::size_t(stage) + nodes_.size() - reader - 1) % nodes_.size();
    };
    for (const StageId stage : started) {
        degrees[stage] = before_[stage].size();
        if (degrees[stage] == 0) {
            ready.emplace(priority(stage), stage);
        }
    }
    std::vector<StageId> result;
    result.reserve(started.size());
    while (!ready.empty()) {
        const StageId stage = ready.top().second;
        ready.pop();
        result.push_back(stage);
        for (const StageId next : after_[stage]) {
            if (present[next] && --degrees[next] == 0) {
                ready.emplace(priority(next), next);
            }
        }
    }
    if (result.size() != started.size() || (!synthesis && result.back() != reader)) {
        throw std::logic_error("RERoT reader order is inconsistent");
    }
    return result;
}

} // namespace tq::rerot
