#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tq::rerot {

using StageId = std::uint32_t;
using RunId = std::uint64_t;
constexpr StageId planner_stage = 0;

struct Node {
    std::string name;
    std::string intent;
};

struct Edge {
    StageId before = 0;
    StageId after = 0;
};

struct PlanLimits {
    std::size_t nodes = 4096;
    std::size_t edges = 65536;
    std::size_t text_bytes = 1048576;
};

// Worker IDs are 1..worker_count(). Input order is the plan rank.
// The planner is 0. The synthesis stage is worker_count() + 1.
class Plan {
public:
    explicit Plan(std::vector<Node> nodes, std::vector<Edge> edges = {}, PlanLimits limits = {});

    std::size_t worker_count() const noexcept;
    StageId synthesis_stage() const noexcept;
    const Node & node(StageId stage) const;
    const std::vector<Edge> & edges() const noexcept;
    const std::vector<StageId> & predecessors(StageId worker) const;

    // All inputs use logical stage IDs. The result excludes planner/synthesis.
    // A worker reader must be started and cannot have a started successor.
    std::vector<StageId> order(StageId reader, const std::vector<StageId> & started) const;

private:
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
    std::vector<std::vector<StageId>> before_;
    std::vector<std::vector<StageId>> after_;
};

} // namespace tq::rerot
