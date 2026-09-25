#include "llama-rerot.h"

#include <cstdio>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>

namespace {

bool set_error(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
    return false;
}

bool is_public_visibility(llama_rerot_visibility visibility) {
    return visibility == llama_rerot_visibility::normal ||
           visibility == llama_rerot_visibility::public_live;
}

uint32_t rotary_pair_axis(const llama_rerot_rope_config & config, uint32_t pair) {
    const uint32_t n_pairs = config.rotary_dim / 2;
    const uint32_t configured = std::accumulate(
        config.axis_pair_count.begin(), config.axis_pair_count.end(), uint32_t(0));

    if (configured == 0) {
        return 0;
    }

    // Validation guarantees equality. Keep this branch defensive for callers
    // that bypass the public helper's error result.
    if (configured != n_pairs) {
        return 0;
    }

    uint32_t offset = 0;
    for (uint32_t axis = 0; axis < config.axis_pair_count.size(); ++axis) {
        const uint32_t next = offset + config.axis_pair_count[axis];
        if (pair < next) {
            return axis;
        }
        offset = next;
    }
    return 0;
}

bool validate_rope_config(
        size_t vector_size,
        const llama_rerot_rope_config & config,
        std::string * error) {
    if (config.head_dim == 0) {
        return set_error(error, "head_dim must be positive");
    }
    if (vector_size != config.head_dim) {
        return set_error(error, "vector size must equal head_dim");
    }
    if (config.rotary_dim == 0 || config.rotary_dim > config.head_dim || config.rotary_dim % 2 != 0) {
        return set_error(error, "rotary_dim must be positive, even, and no larger than head_dim");
    }
    if (!std::isfinite(config.theta) || config.theta <= 0.0) {
        return set_error(error, "theta must be finite and positive");
    }
    if (!std::isfinite(config.freq_scale) || config.freq_scale <= 0.0) {
        return set_error(error, "freq_scale must be finite and positive");
    }

    const uint32_t configured = std::accumulate(
        config.axis_pair_count.begin(), config.axis_pair_count.end(), uint32_t(0));
    if (configured != 0 && configured != config.rotary_dim / 2) {
        return set_error(error, "axis_pair_count must sum to rotary_dim / 2");
    }

    return true;
}

void validate_ddvr_problem(
        const std::vector<float> & raw_query,
        const std::vector<float> & raw_keys,
        const std::vector<float> & values,
        uint32_t value_dim,
        const std::vector<llama_rerot_ddvr_span> & spans,
        const llama_rerot_rope_config & config) {
    std::string error;
    if (!validate_rope_config(raw_query.size(), config, &error)) {
        throw std::invalid_argument(error);
    }
    if (value_dim == 0) {
        throw std::invalid_argument("value_dim must be positive");
    }
    if (raw_keys.size() % config.head_dim != 0) {
        throw std::invalid_argument("raw_keys size must be a multiple of head_dim");
    }

    const uint32_t n_keys = raw_keys.size() / config.head_dim;
    if (values.size() != size_t(n_keys) * value_dim) {
        throw std::invalid_argument("values size does not match key count and value_dim");
    }

    std::vector<uint8_t> covered(n_keys, 0);
    for (const auto & span : spans) {
        if (span.key_count == 0) {
            throw std::invalid_argument("DDVR spans must not be empty");
        }
        if (span.key_begin > n_keys || span.key_count > n_keys - span.key_begin) {
            throw std::invalid_argument("DDVR span is outside the key array");
        }
        for (uint32_t i = 0; i < span.key_count; ++i) {
            uint8_t & count = covered[span.key_begin + i];
            if (++count != 1) {
                throw std::invalid_argument("DDVR spans must cover each key exactly once");
            }
        }
    }
    if (std::find(covered.begin(), covered.end(), uint8_t(0)) != covered.end()) {
        throw std::invalid_argument("DDVR spans must cover every key");
    }
}

float dot_product(const float * a, const float * b, uint32_t n) {
    double sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        sum += double(a[i]) * double(b[i]);
    }
    return float(sum);
}

std::vector<float> softmax_weighted_values(
        const std::vector<float> & scores,
        const std::vector<float> & values,
        uint32_t value_dim) {
    if (scores.empty()) {
        return std::vector<float>(value_dim, 0.0f);
    }

    const float max_score = *std::max_element(scores.begin(), scores.end());
    std::vector<double> weights(scores.size());
    double normalizer = 0.0;
    for (size_t i = 0; i < scores.size(); ++i) {
        weights[i] = std::exp(double(scores[i] - max_score));
        normalizer += weights[i];
    }

    std::vector<float> output(value_dim, 0.0f);
    for (size_t key = 0; key < scores.size(); ++key) {
        const double weight = weights[key] / normalizer;
        for (uint32_t d = 0; d < value_dim; ++d) {
            output[d] += float(weight * values[key * value_dim + d]);
        }
    }
    return output;
}

} // namespace

llama_rerot_document::llama_rerot_document(uint64_t episode_id) {
    reset(episode_id);
}

void llama_rerot_document::reset(uint64_t episode_id) {
    episode_id_ = episode_id;
    is_dag_mode_ = false;
    nodes_.clear();
    runs_.clear();

    llama_rerot_node root_node;
    root_node.id = 0;
    root_node.parent = LLAMA_REROT_NODE_INVALID;
    root_node.depth = 0;
    root_node.child_index = 0;
    root_node.state = llama_rerot_node_state::planning;
    nodes_.push_back(std::move(root_node));
}

bool llama_rerot_document::is_dag_mode() const {
    return is_dag_mode_;
}

void llama_rerot_document::set_dag_mode(bool enabled) {
    is_dag_mode_ = enabled;
}

bool llama_rerot_document::set_plan_rank(llama_rerot_node_id node_id, uint32_t rank) {
    if (node_id >= nodes_.size()) {
        return false;
    }
    nodes_[node_id].plan_rank = rank;
    return true;
}

bool llama_rerot_document::set_stage_role(llama_rerot_node_id node_id, llama_rerot_stage_role role) {
    if (node_id >= nodes_.size()) {
        return false;
    }
    nodes_[node_id].stage_role = role;
    return true;
}

bool llama_rerot_document::add_edge(llama_rerot_node_id before, llama_rerot_node_id after, std::string * error) {
    if (before >= nodes_.size() || after >= nodes_.size()) {
        return set_error(error, "edge endpoint out of range");
    }
    if (before == after) {
        return set_error(error, "self loop not permitted in DAG");
    }
    auto & preds = nodes_[after].predecessors;
    if (std::find(preds.begin(), preds.end(), before) != preds.end()) {
        return set_error(error, "duplicate edge");
    }
    preds.push_back(before);
    nodes_[before].successors.push_back(after);
    return true;
}

std::vector<llama_rerot_node_id> llama_rerot_document::topo_sort_cycle_preferred(
        const std::vector<llama_rerot_node_id> & candidates,
        llama_rerot_node_id reader,
        std::string * error) const {
    if (candidates.empty()) {
        return {};
    }

    std::unordered_set<llama_rerot_node_id> candidate_set(candidates.begin(), candidates.end());
    if (candidate_set.size() != candidates.size()) {
        set_error(error, "duplicate candidate node");
        return {};
    }
    if (candidate_set.find(reader) == candidate_set.end()) {
        set_error(error, "reader not in candidates");
        return {};
    }

    // Sort candidates by plan_rank to establish initial plan_order
    std::vector<llama_rerot_node_id> plan_order = candidates;
    std::sort(plan_order.begin(), plan_order.end(), [&](llama_rerot_node_id a, llama_rerot_node_id b) {
        return nodes_[a].plan_rank < nodes_[b].plan_rank;
    });

    // Find position of reader in plan_order
    size_t reader_idx = 0;
    for (size_t i = 0; i < plan_order.size(); ++i) {
        if (plan_order[i] == reader) {
            reader_idx = i;
            break;
        }
    }

    // Build cycle priority list:
    // Elements after reader in plan_order, then before reader, and reader self last.
    std::vector<llama_rerot_node_id> cycle_priority;
    cycle_priority.reserve(plan_order.size());
    for (size_t i = reader_idx + 1; i < plan_order.size(); ++i) {
        cycle_priority.push_back(plan_order[i]);
    }
    for (size_t i = 0; i < reader_idx; ++i) {
        cycle_priority.push_back(plan_order[i]);
    }
    cycle_priority.push_back(reader);

    // Map each node to its priority rank (lower number = higher priority)
    std::unordered_map<llama_rerot_node_id, size_t> rank;
    for (size_t i = 0; i < cycle_priority.size(); ++i) {
        rank[cycle_priority[i]] = i;
    }

    // Build sub-graph in-degrees and adjacency within candidate_set
    std::unordered_map<llama_rerot_node_id, size_t> in_degree;
    std::unordered_map<llama_rerot_node_id, std::vector<llama_rerot_node_id>> following;
    for (const auto node : candidates) {
        in_degree[node] = 0;
        following[node] = {};
    }

    for (const auto node : candidates) {
        for (const auto succ : nodes_[node].successors) {
            if (candidate_set.count(succ)) {
                following[node].push_back(succ);
                in_degree[succ]++;
            }
        }
    }

    // Min-heap ordered by priority rank (rank[node])
    auto cmp = [&](llama_rerot_node_id a, llama_rerot_node_id b) {
        return rank[a] > rank[b]; // greater for min-heap
    };
    std::priority_queue<llama_rerot_node_id, std::vector<llama_rerot_node_id>, decltype(cmp)> heap(cmp);

    for (const auto node : candidates) {
        if (in_degree[node] == 0) {
            heap.push(node);
        }
    }

    std::vector<llama_rerot_node_id> result;
    result.reserve(candidates.size());

    while (!heap.empty()) {
        const auto node = heap.top();
        heap.pop();
        result.push_back(node);

        for (const auto succ : following[node]) {
            if (--in_degree[succ] == 0) {
                heap.push(succ);
            }
        }
    }

    if (result.size() != candidates.size()) {
        set_error(error, "cycle detected in candidate DAG");
        return {};
    }

    return result;
}

llama_rerot_reader_view llama_rerot_document::build_dag_view(
        llama_rerot_node_id reader,
        const std::vector<llama_rerot_node_id> & started_nodes,
        uint64_t frozen_read_publish_epoch) const {
    if (reader >= nodes_.size()) {
        throw std::out_of_range("RERoT reader node does not exist");
    }

    std::unordered_set<llama_rerot_node_id> started_set(started_nodes.begin(), started_nodes.end());
    started_set.insert(0);
    for (const auto nid : started_set) {
        if (nid >= nodes_.size()) {
            throw std::out_of_range("RERoT started node does not exist");
        }
        if (nid == 0) {
            continue;
        }
        for (const auto pred : nodes_[nid].predecessors) {
            if (started_set.find(pred) == started_set.end()) {
                throw std::runtime_error("DAG started set misses a predecessor");
            }
        }
    }

    llama_rerot_reader_view result;
    result.episode_id = episode_id_;
    result.reader = reader;

    llama_pos virtual_pos = 0;

    const auto append_visible_run = [&](const llama_rerot_run & current_run) {
        if (!run_visible_to(current_run, reader) || current_run.token_count == 0) {
            return;
        }
        // Frozen logical-step read version (AGENTS.md §06.3): omit every
        // foreign PUBLIC run published after the snapshot — BODY and FRAME
        // alike — so later slices cannot observe this step's cohort writes.
        // Only the reader's own runs (including its current FRAME) are
        // exempt. Root P predates any freeze snapshot, so it is unaffected.
        if (frozen_read_publish_epoch != 0 &&
            current_run.owner != reader &&
            current_run.visibility == llama_rerot_visibility::public_live &&
            current_run.publish_epoch > frozen_read_publish_epoch) {
            return;
        }
        result.runs.push_back({
            current_run.id,
            current_run.owner,
            current_run.storage_pos0,
            virtual_pos,
            current_run.token_count,
            current_run.publish_epoch,
        });
        virtual_pos += static_cast<llama_pos>(current_run.token_count);
    };

    // 1. Root node (0.plan / public prefix) always comes first
    for (const auto run_id : nodes_[0].runs) {
        append_visible_run(runs_[run_id]);
    }

    // 2. Compute cycle-preferred topological sort for started worker nodes (excluding 0)
    std::vector<llama_rerot_node_id> workers;
    workers.reserve(started_nodes.size());
    for (const auto nid : started_nodes) {
        if (nid != 0) {
            workers.push_back(nid);
        }
    }

    if (!workers.empty()) {
        std::string err;
        std::vector<llama_rerot_node_id> ordered_workers;
        const bool is_synthesis_reader = (reader == 0 || nodes_[reader].stage_role == llama_rerot_stage_role::synthesis);
        if (is_synthesis_reader) {
            // Synthesis reader: plan-rank order subject to DAG dependencies.
            // The reader (0) is not among the workers, so anchor the cycle
            // priority on the max-rank worker: its cycle list is exactly the
            // plan order, independent of the caller's started_nodes order.
            auto anchor = workers.front();
            for (const auto nid : workers) {
                if (nodes_[nid].plan_rank > nodes_[anchor].plan_rank) {
                    anchor = nid;
                }
            }
            ordered_workers = topo_sort_cycle_preferred(workers, anchor, &err);
        } else {
            ordered_workers = topo_sort_cycle_preferred(workers, reader, &err);
        }
        if (ordered_workers.empty()) {
            throw std::runtime_error("DAG topological sort failed: " + err);
        }

        for (const auto nid : ordered_workers) {
            const auto & node = nodes_[nid];
            for (const auto run_id : node.runs) {
                append_visible_run(runs_[run_id]);
            }
        }
    }

    result.query_virtual_pos = virtual_pos;
    return result;
}

uint64_t llama_rerot_document::episode_id() const {
    return episode_id_;
}

llama_rerot_node_id llama_rerot_document::root() const {
    return 0;
}

llama_rerot_node_id llama_rerot_document::create_child(
        llama_rerot_node_id parent_id,
        std::string title,
        llama_rerot_node_state state) {
    if (parent_id >= nodes_.size()) {
        throw std::out_of_range("RERoT parent node does not exist");
    }

    llama_rerot_node child;
    child.id = static_cast<llama_rerot_node_id>(nodes_.size());
    child.parent = parent_id;
    child.depth = nodes_[parent_id].depth + 1;
    child.child_index = static_cast<uint32_t>(nodes_[parent_id].children.size());
    child.title = std::move(title);
    child.state = state;

    nodes_[parent_id].children.push_back(child.id);
    nodes_.push_back(std::move(child));
    return static_cast<llama_rerot_node_id>(nodes_.size() - 1);
}

llama_rerot_run_id llama_rerot_document::append_run(
        llama_rerot_node_id owner,
        llama_rerot_visibility visibility,
        llama_pos storage_pos0,
        uint32_t token_count,
        uint64_t publish_epoch,
        llama_rerot_segment_kind kind) {
    if (owner >= nodes_.size()) {
        throw std::out_of_range("RERoT run owner does not exist");
    }
    if (storage_pos0 < 0) {
        throw std::invalid_argument("RERoT storage positions must be non-negative");
    }
    // Publication-epoch contract, mirroring reclassify_run and the KV cell
    // rules: only public_live carries a non-zero epoch; pending runs gain
    // theirs atomically at publish/reclassify time.
    const bool wants_epoch = visibility == llama_rerot_visibility::public_live;
    if (wants_epoch == (publish_epoch == 0)) {
        throw std::invalid_argument("RERoT run publish epoch does not match its visibility");
    }

    llama_rerot_run run;
    run.id = static_cast<llama_rerot_run_id>(runs_.size());
    run.owner = owner;
    run.visibility = visibility;
    run.kind = kind;
    run.storage_pos0 = storage_pos0;
    run.token_count = token_count;
    run.publish_epoch = publish_epoch;

    nodes_[owner].runs.push_back(run.id);
    runs_.push_back(std::move(run));
    return static_cast<llama_rerot_run_id>(runs_.size() - 1);
}

bool llama_rerot_document::set_run_segment_kind(llama_rerot_run_id run_id, llama_rerot_segment_kind kind) {
    if (run_id >= runs_.size()) {
        return false;
    }
    runs_[run_id].kind = kind;
    return true;
}

bool llama_rerot_document::set_run_token_count(llama_rerot_run_id run_id, uint32_t token_count) {
    if (run_id >= runs_.size()) {
        return false;
    }
    runs_[run_id].token_count = token_count;
    return true;
}

bool llama_rerot_document::publish_run(llama_rerot_run_id run_id, uint64_t publish_epoch) {
    if (run_id >= runs_.size() || publish_epoch == 0) {
        return false;
    }
    auto & current = runs_[run_id];
    if (current.visibility != llama_rerot_visibility::pending_record &&
        current.visibility != llama_rerot_visibility::public_live) {
        return false;
    }
    current.visibility = llama_rerot_visibility::public_live;
    current.publish_epoch = publish_epoch;
    return true;
}

bool llama_rerot_document::reclassify_run(
        llama_rerot_run_id run_id,
        llama_rerot_visibility expected,
        llama_rerot_visibility replacement,
        uint64_t publish_epoch) {
    if (run_id >= runs_.size() || expected == llama_rerot_visibility::normal ||
        replacement == llama_rerot_visibility::normal) {
        return false;
    }
    auto & current = runs_[run_id];
    if (current.visibility != expected) {
        return false;
    }
    if ((replacement == llama_rerot_visibility::public_live && publish_epoch == 0) ||
        (replacement != llama_rerot_visibility::public_live && publish_epoch != 0)) {
        return false;
    }
    current.visibility = replacement;
    current.publish_epoch = replacement == llama_rerot_visibility::public_live ? publish_epoch : 0;
    return true;
}

bool llama_rerot_document::set_node_state(llama_rerot_node_id node_id, llama_rerot_node_state state) {
    if (node_id >= nodes_.size()) {
        return false;
    }
    nodes_[node_id].state = state;
    return true;
}

const llama_rerot_node * llama_rerot_document::node(llama_rerot_node_id node_id) const {
    return node_id < nodes_.size() ? &nodes_[node_id] : nullptr;
}

const llama_rerot_run * llama_rerot_document::run(llama_rerot_run_id run_id) const {
    return run_id < runs_.size() ? &runs_[run_id] : nullptr;
}

size_t llama_rerot_document::node_count() const {
    return nodes_.size();
}

size_t llama_rerot_document::run_count() const {
    return runs_.size();
}

bool llama_rerot_document::is_ancestor(
        llama_rerot_node_id ancestor,
        llama_rerot_node_id descendant) const {
    if (ancestor >= nodes_.size() || descendant >= nodes_.size()) {
        return false;
    }
    llama_rerot_node_id current = descendant;
    while (current != LLAMA_REROT_NODE_INVALID) {
        if (current == ancestor) {
            return true;
        }
        current = nodes_[current].parent;
    }
    return false;
}

std::vector<uint32_t> llama_rerot_document::tree_path(llama_rerot_node_id node_id) const {
    if (node_id >= nodes_.size()) {
        return {};
    }

    std::vector<uint32_t> path;
    llama_rerot_node_id current = node_id;
    while (nodes_[current].parent != LLAMA_REROT_NODE_INVALID) {
        path.push_back(nodes_[current].child_index);
        current = nodes_[current].parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

bool llama_rerot_document::run_visible_to(
        const llama_rerot_run & run,
        llama_rerot_node_id reader) const {
    // AGENTS.md §04.7: source terminal tokens (source_end) are preserved in
    // owner's tape for accounting and state, but are NOT exported to foreign
    // views to prevent double-close in concatenated views.
    if (run.kind == llama_rerot_segment_kind::probe_control) {
        return false;
    }
    if (run.kind == llama_rerot_segment_kind::source_end && run.owner != reader) {
        return false;
    }
    if (is_public_visibility(run.visibility)) {
        return true;
    }
    // Private and pending lexical material is visible only to its exact owner,
    // not to descendants. Descendants inherit causal recurrent state but not
    // the parent's private token stream.
    return run.owner == reader;
}

void llama_rerot_document::render_node(
        llama_rerot_node_id node_id,
        llama_rerot_node_id reader,
        std::vector<llama_rerot_view_run> & out,
        llama_pos & virtual_pos) const {
    const auto & current = nodes_[node_id];

    for (const auto run_id : current.runs) {
        const auto & current_run = runs_[run_id];
        if (!run_visible_to(current_run, reader) || current_run.token_count == 0) {
            continue;
        }

        out.push_back({
            current_run.id,
            current_run.owner,
            current_run.storage_pos0,
            virtual_pos,
            current_run.token_count,
            current_run.publish_epoch,
        });
        virtual_pos += static_cast<llama_pos>(current_run.token_count);
    }

    if (current.children.empty()) {
        return;
    }

    size_t reader_child = current.children.size();
    for (size_t i = 0; i < current.children.size(); ++i) {
        if (is_ancestor(current.children[i], reader)) {
            reader_child = i;
            break;
        }
    }

    if (reader_child == current.children.size()) {
        // Off-path subtrees are stable and retain the original <ol> order.
        for (const auto child : current.children) {
            render_node(child, reader, out, virtual_pos);
        }
        return;
    }

    // Path-Anchored Cyclic DFS: siblings following the reader branch, then
    // preceding siblings, and finally the reader branch recursively.
    for (size_t i = reader_child + 1; i < current.children.size(); ++i) {
        render_node(current.children[i], reader, out, virtual_pos);
    }
    for (size_t i = 0; i < reader_child; ++i) {
        render_node(current.children[i], reader, out, virtual_pos);
    }
    render_node(current.children[reader_child], reader, out, virtual_pos);
}

llama_rerot_reader_view llama_rerot_document::build_view(llama_rerot_node_id reader) const {
    if (reader >= nodes_.size()) {
        throw std::out_of_range("RERoT reader node does not exist");
    }

    llama_rerot_reader_view result;
    result.episode_id = episode_id_;
    result.reader = reader;

    llama_pos virtual_pos = 0;
    render_node(root(), reader, result.runs, virtual_pos);
    result.query_virtual_pos = virtual_pos;
    return result;
}

bool llama_rerot_document::validate(std::string * error) const {
    if (episode_id_ == 0) {
        return set_error(error, "episode id must be non-zero");
    }
    if (nodes_.empty()) {
        return set_error(error, "document must contain a root node");
    }
    if (nodes_[0].id != 0 || nodes_[0].parent != LLAMA_REROT_NODE_INVALID || nodes_[0].depth != 0) {
        return set_error(error, "root node metadata is invalid");
    }

    std::vector<uint32_t> run_refs(runs_.size(), 0);
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto & current = nodes_[i];
        if (current.id != i) {
            return set_error(error, "node ids must be dense and stable");
        }
        if (static_cast<uint8_t>(current.state) > static_cast<uint8_t>(llama_rerot_node_state::ready_suspended)) {
            return set_error(error, "node state is out of range");
        }
        if (i != 0) {
            if (current.parent >= nodes_.size()) {
                return set_error(error, "node parent is out of range");
            }
            const auto & parent = nodes_[current.parent];
            if (current.depth != parent.depth + 1 || current.child_index >= parent.children.size() ||
                parent.children[current.child_index] != current.id) {
                return set_error(error, "node path metadata is inconsistent");
            }
        }

        std::unordered_set<llama_rerot_node_id> children_seen;
        for (size_t child_index = 0; child_index < current.children.size(); ++child_index) {
            const auto child = current.children[child_index];
            if (child >= nodes_.size() || !children_seen.insert(child).second) {
                return set_error(error, "node has an invalid or duplicate child");
            }
            if (nodes_[child].parent != current.id || nodes_[child].child_index != child_index) {
                return set_error(error, "child backlink is inconsistent");
            }
        }

        for (const auto run_id : current.runs) {
            if (run_id >= runs_.size()) {
                return set_error(error, "node references an unknown run");
            }
            if (runs_[run_id].owner != current.id) {
                return set_error(error, "run owner and node run list disagree");
            }
            if (++run_refs[run_id] != 1) {
                return set_error(error, "run appears more than once in the document");
            }
        }
    }

    for (size_t i = 0; i < runs_.size(); ++i) {
        const auto & current = runs_[i];
        if (current.id != i || current.owner >= nodes_.size() || run_refs[i] != 1) {
            return set_error(error, "run metadata is inconsistent");
        }
        if (current.storage_pos0 < 0) {
            return set_error(error, "run storage position must be non-negative");
        }
        // Publication-epoch contract: only public_live carries a non-zero
        // epoch; pending runs gain theirs atomically at publish time.
        const bool has_epoch = current.publish_epoch != 0;
        const bool wants_epoch = current.visibility == llama_rerot_visibility::public_live;
        if (has_epoch != wants_epoch) {
            return set_error(error, "run publish epoch does not match its visibility");
        }
    }

    // PAC-DFS verification with every node as reader: dense virtual positions
    // [0, L) with no overlap, each visible run exactly once, view/source
    // field fidelity, and sibling-block ordering with the reader branch last.
    // Scheduling state is deliberately not consulted: the render must be
    // queue-independent.
    for (const auto & reader_node : nodes_) {
        const auto reader = reader_node.id;
        const auto view = build_view(reader);
        if (view.episode_id != episode_id_ || view.reader != reader) {
            return set_error(error, "reader view carries the wrong episode or reader");
        }
        std::unordered_set<llama_rerot_run_id> expected_runs;
        for (const auto & candidate : runs_) {
            if (candidate.token_count != 0 && run_visible_to(candidate, reader)) {
                expected_runs.insert(candidate.id);
            }
        }
        llama_pos expected_pos = 0;
        std::unordered_set<llama_rerot_run_id> seen_runs;
        for (const auto & view_run : view.runs) {
            if (view_run.run_id >= runs_.size()) {
                return set_error(error, "reader view references an unknown run");
            }
            const auto & source = runs_[view_run.run_id];
            if (view_run.owner != source.owner || view_run.storage_pos0 != source.storage_pos0 ||
                view_run.token_count != source.token_count || view_run.publish_epoch != source.publish_epoch) {
                return set_error(error, "reader view run does not mirror its source run");
            }
            if (!run_visible_to(source, reader)) {
                return set_error(error, "reader view exposes a run that is not visible to the reader");
            }
            if (view_run.virtual_pos0 != expected_pos || view_run.token_count == 0) {
                return set_error(error, "reader view is not densely packed");
            }
            if (!seen_runs.insert(view_run.run_id).second) {
                return set_error(error, "reader view contains a run more than once");
            }
            expected_pos += static_cast<llama_pos>(view_run.token_count);
        }
        if (view.query_virtual_pos != expected_pos) {
            return set_error(error, "reader query position does not follow the final visible token");
        }
        if (seen_runs != expected_runs) {
            return set_error(error, "reader view omits a visible run");
        }
        for (const auto & parent : nodes_) {
            if (parent.children.empty()) {
                continue;
            }
            size_t reader_child = parent.children.size();
            for (size_t i = 0; i < parent.children.size(); ++i) {
                if (is_ancestor(parent.children[i], reader)) {
                    reader_child = i;
                    break;
                }
            }
            std::vector<size_t> expected_order;
            expected_order.reserve(parent.children.size());
            if (reader_child == parent.children.size()) {
                for (size_t i = 0; i < parent.children.size(); ++i) {
                    expected_order.push_back(i);
                }
            } else {
                for (size_t i = reader_child + 1; i < parent.children.size(); ++i) {
                    expected_order.push_back(i);
                }
                for (size_t i = 0; i < reader_child; ++i) {
                    expected_order.push_back(i);
                }
                expected_order.push_back(reader_child);
            }
            std::vector<size_t> observed_order;
            std::vector<bool> block_seen(parent.children.size(), false);
            bool block_started = false;
            for (const auto & view_run : view.runs) {
                if (view_run.owner == parent.id) {
                    if (block_started) {
                        return set_error(error, "node runs do not precede their subtree in the reader view");
                    }
                    continue;
                }
                if (!is_ancestor(parent.id, view_run.owner)) {
                    continue;
                }
                llama_rerot_node_id block = view_run.owner;
                while (nodes_[block].parent != parent.id) {
                    block = nodes_[block].parent;
                }
                const size_t tag = nodes_[block].child_index;
                if (!block_seen[tag]) {
                    block_seen[tag] = true;
                    observed_order.push_back(tag);
                } else if (observed_order.back() != tag) {
                    return set_error(error, "child subtree runs are not contiguous in the reader view");
                }
                block_started = true;
            }
            std::vector<size_t> nonempty_expected;
            for (const size_t tag : expected_order) {
                if (block_seen[tag]) {
                    nonempty_expected.push_back(tag);
                }
            }
            if (observed_order != nonempty_expected) {
                return set_error(error, "reader view sibling order breaks own-subtree-last PAC-DFS");
            }
        }
    }
    return true;
}

std::vector<llama_rerot_node_id> llama_rerot_document::leaf_order(
        llama_rerot_node_id reader,
        order_provider provider) const {
    if (reader >= nodes_.size()) {
        throw std::out_of_range("RERoT reader node does not exist");
    }
    std::vector<llama_rerot_node_id> order;
    if (provider == order_provider::kahn) {
        // Cycle-preferred topological sort over the DAG edges. The reader's own
        // node renders last; ties break by plan_rank so the order is stable.
        std::vector<llama_rerot_node_id> all;
        all.reserve(nodes_.size());
        for (const auto & n : nodes_) {
            if (n.children.empty()) {
                all.push_back(n.id);
            }
        }
        std::sort(all.begin(), all.end(), [&](llama_rerot_node_id a, llama_rerot_node_id b) {
            const bool a_last = (a == reader);
            const bool b_last = (b == reader);
            if (a_last != b_last) {
                return b_last; // reader last
            }
            return nodes_[a].plan_rank < nodes_[b].plan_rank;
        });
        return all;
    }

    // tree_dfs: emit leaves in Path-Anchored Cyclic DFS order. Emitting through
    // the same render walk guarantees the leaf order and the run-level view can
    // never disagree -- a divergence between the two would silently change the
    // attention layout, so they are computed from one traversal.
    std::vector<llama_rerot_node_id> stack;
    stack.push_back(root());
    while (!stack.empty()) {
        const llama_rerot_node_id u = stack.back();
        stack.pop_back();
        const auto & node = nodes_[u];
        if (node.children.empty()) {
            order.push_back(u);
            continue;
        }
        size_t reader_child = node.children.size();
        for (size_t i = 0; i < node.children.size(); ++i) {
            if (is_ancestor(node.children[i], reader)) {
                reader_child = i;
                break;
            }
        }
        // Push in reverse of emission order: following siblings, then preceding
        // siblings, then the reader branch.
        std::vector<llama_rerot_node_id> emit;
        emit.reserve(node.children.size());
        if (reader_child == node.children.size()) {
            for (const auto c : node.children) {
                emit.push_back(c);
            }
        } else {
            for (size_t i = reader_child + 1; i < node.children.size(); ++i) {
                emit.push_back(node.children[i]);
            }
            for (size_t i = 0; i < reader_child; ++i) {
                emit.push_back(node.children[i]);
            }
            emit.push_back(node.children[reader_child]);
        }
        for (size_t k = emit.size(); k-- > 0;) {
            stack.push_back(emit[k]);
        }
    }
    return order;
}

llama_rerot_reader_view llama_rerot_document::build_view_for(
        llama_rerot_node_id reader,
        order_provider provider,
        const std::vector<llama_rerot_node_id> & started_nodes,
        uint64_t frozen_read_publish_epoch) const {
    if (provider == order_provider::kahn) {
        return build_dag_view(reader, started_nodes, frozen_read_publish_epoch);
    }
    if (frozen_read_publish_epoch != 0 || !started_nodes.empty()) {
        // Missing parameters are a caller defect, not a permissive path: the
        // tree provider has no Kahn equivalents for them, so silently ignoring
        // the arguments would produce a view the caller did not ask for.
        throw std::invalid_argument(
            "RERoT tree_dfs order provider does not accept started_nodes or epoch freeze");
    }
    return build_view(reader);
}

bool llama_rerot_attn_layout::validate(uint32_t n_keys, std::string * error) const {
    // Predefine discipline (thirteenth round): the builder is the single
    // source of truth and its construction already guarantees the checked
    // invariants structurally (offsets are push_back-monotonic, group_index
    // is builder-assigned, per-query lists are disjoint-set subsequences —
    // every key lands in exactly one segment of one list, so per-query
    // duplicates are impossible by construction).
    // LLAMA_REROT_LAYOUT_VALIDATE (default 1):
    //   0 = O(groups + queries) structural checks only;
    //   1 = 0 + sequential per-entry range/group checks (no random access);
    //   2 = 1 + the per-query duplicate-key bitmap (O(entries) RANDOM
    //       access — at production shapes it costs more than the numeric
    //       pass itself; kept as the paranoid audit tier).
    static const int validate_mode = [] {
        const char * m = getenv("LLAMA_REROT_LAYOUT_VALIDATE");
        return m ? atoi(m) : 1;
    }();
    if (validate_mode <= 1) {
        if (n_queries == 0) {
            if (!groups.empty() || !entries.empty() || !query_offsets.empty() || !spans.empty()) {
                return set_error(error, "empty RERoT attention layout contains data");
            }
            return true;
        }
        if (query_offsets.size() != size_t(n_queries) + 1 || query_offsets.front() != 0 ||
            query_offsets.back() != entries.size()) {
            return set_error(error, "RERoT query offsets are inconsistent");
        }
        for (uint32_t query = 0; query < n_queries; ++query) {
            if (query_offsets[query] > query_offsets[query + 1]) {
                return set_error(error, "RERoT query offsets are not monotonic");
            }
        }
        const uint32_t n_groups = uint32_t(groups.size());
        for (const auto & group : groups) {
            if (group.query_index >= n_queries || group.effective_pos < 0) {
                return set_error(error, "RERoT attention query group metadata is invalid");
            }
        }
        if (validate_mode == 1) {
            // Sequential range audit (sixteenth round, question two):
            // check the span table's RANGE BOUNDS instead of walking every
            // entry — one bound check covers a whole const-effective
            // segment. Uncovered rows (scalar merge arms, general shapes
            // built without the span channel) fall back to the sequential
            // per-entry max walk. No random access in either branch.
            uint64_t covered = 0;
            for (const auto & sp : spans) {
                if (sp.group_index >= n_groups) {
                    return set_error(error, "RERoT attention entry is out of range");
                }
                if ((uint64_t) sp.entry_index + sp.count > entries.size()) {
                    return set_error(error, "RERoT span entry range is out of bounds");
                }
                if ((uint64_t) sp.key_start + sp.count > n_keys) {
                    return set_error(error, "RERoT attention entry is out of range");
                }
                covered += sp.count;
            }
            if (covered < entries.size()) {
                // Uncovered rows (scalar merge arms / base rows): walk
                // ONLY the gaps between spans (same cursor discipline as
                // the loader) so the tier-1 audit stays O(entries -
                // span_rows) instead of O(entries).
                uint32_t max_key = 0, max_group = 0;
                size_t cursor = 0;
                const auto scan = [&](size_t from, size_t to) {
                    for (size_t i = from; i < to; ++i) {
                        max_key = entries[i].key_index > max_key ? entries[i].key_index : max_key;
                        max_group = entries[i].group_index > max_group ? entries[i].group_index : max_group;
                    }
                };
                for (const auto & sp : spans) {
                    if (sp.entry_index < cursor) {
                        return set_error(error, "RERoT span table is not entry-ordered");
                    }
                    scan(cursor, sp.entry_index);
                    cursor = (size_t) sp.entry_index + sp.count;
                }
                scan(cursor, entries.size());
                if (max_key >= n_keys || max_group >= n_groups) {
                    return set_error(error, "RERoT attention entry is out of range");
                }
            }
        }
        return true;
    }
    // ---- validate_mode >= 2: paranoid audit tier (adds the duplicate-key
    // bitmap with random access). Falls through from the tier-1 ladder
    // above only when validate_mode >= 2.
    if (n_queries == 0) {
        if (!groups.empty() || !entries.empty() || !query_offsets.empty()) {
            return set_error(error, "empty RERoT attention layout contains data");
        }
        return true;
    }

    if (query_offsets.size() != size_t(n_queries) + 1 || query_offsets.front() != 0 ||
        query_offsets.back() != entries.size()) {
        return set_error(error, "RERoT query offsets are inconsistent");
    }
    for (uint32_t query = 0; query < n_queries; ++query) {
        if (query_offsets[query] > query_offsets[query + 1]) {
            return set_error(error, "RERoT query offsets are not monotonic");
        }
    }
    for (const auto & group : groups) {
        if (group.query_index >= n_queries || group.effective_pos < 0) {
            return set_error(error, "RERoT query group metadata is invalid");
        }
    }
    // Per-query duplicate-key check, zero-allocation discipline: the
    // previous byte bitmap was allocated and zeroed per validate() call
    // (O(n_keys) memset per frontier on top of O(entries)); the layout
    // already carries a persistent scratch owned by the cache-level
    // builder, so validate() reuses it when the caller provides one and
    // only falls back to a local vector otherwise.
    uint8_t * seen = nullptr;
    std::vector<uint8_t> local_scratch;
    if (validate_scratch_storage.size() >= size_t(n_keys)) {
        seen = validate_scratch_storage.data();
    } else {
        validate_scratch_storage.assign(n_keys, 0);
        seen = validate_scratch_storage.data();
    }
    for (uint32_t query = 0; query < n_queries; ++query) {
        for (uint32_t i = query_offsets[query]; i < query_offsets[query + 1]; ++i) {
            if (groups[entries[i].group_index].query_index != query) {
                return set_error(error, "RERoT attention entry references another query's group");
            }
        }
    }
    for (uint32_t query = 0; query < n_queries; ++query) {
        // The duplicate check was a per-entry unordered_set insert — the
        // dominant validation cost at production shapes (one hash insert
        // per entry, ~n_kv per reader per query). A byte bitmap keeps the
        // same fail-loud guarantee at O(entries) sequential writes.
        for (uint32_t i = query_offsets[query]; i < query_offsets[query + 1]; ++i) {
            const auto & entry = entries[i];
            if (entry.key_index >= n_keys || entry.group_index >= groups.size()) {
                return set_error(error, "RERoT attention entry is out of range");
            }
            if (groups[entry.group_index].query_index != query) {
                return set_error(error, "RERoT attention entry references another query's group");
            }
            if (seen[entry.key_index]++) {
                return set_error(error, "RERoT attention query contains a duplicate physical key");
            }
        }
        // Reset only the touched bytes: entries reference distinct keys
        // when valid, so clearing via the same walk is exact; on failure we
        // return immediately and the vector dies with the call.
        for (uint32_t i = query_offsets[query]; i < query_offsets[query + 1]; ++i) {
            seen[entries[i].key_index] = 0;
        }
    }
    return true;
}

// --- R13-A compact span ABI (C05) -----------------------------------------------

llama_rerot_span_coverage llama_rerot_spans_coverage(
        const llama_rerot_attn_layout & layout) {
    llama_rerot_span_coverage cov;
    cov.total_rows = layout.entries.size();
    uint64_t covered = 0;
    for (const auto & sp : layout.spans) {
        covered += sp.count;
    }
    cov.span_rows = covered;
    // Covered ranges are entry-disjoint by construction (non-decreasing
    // entry_index + count windows), so spill is total minus covered.
    cov.spill_rows = covered < cov.total_rows ? cov.total_rows - covered : 0;
    return cov;
}

std::vector<llama_rerot_attn_entry> llama_rerot_spans_expand(
        const llama_rerot_attn_layout & layout) {
    const size_t n_entries = layout.entries.size();
    std::vector<llama_rerot_attn_entry> out(n_entries);

    size_t cursor = 0;
    for (const auto & sp : layout.spans) {
        if (sp.count == 0) {
            throw std::runtime_error(
                "RERoT span expand: zero-count span is not expressible");
        }
        if ((uint64_t) sp.entry_index + sp.count > n_entries) {
            throw std::runtime_error(
                "RERoT span expand: span range beyond live entries");
        }
        if (sp.entry_index < cursor) {
            throw std::runtime_error(
                "RERoT span expand: span table is not entry-ordered");
        }
        // Scalar rows before this span come from the authoritative stream.
        for (; cursor < sp.entry_index; ++cursor) {
            out[cursor] = layout.entries[cursor];
        }
        // Span rows: ascending key range, one group.
        for (uint32_t k = 0; k < sp.count; ++k) {
            out[cursor].key_index   = sp.key_start + k;
            out[cursor].group_index = sp.group_index;
            ++cursor;
        }
    }
    // Tail rows after the last span.
    for (; cursor < n_entries; ++cursor) {
        out[cursor] = layout.entries[cursor];
    }
    return out;
}

// Batched shared-reader layout construction (2026-09-22 compute-organization
// round). Structure/numeric separation, mirroring llama_rerot_build_query_layout
// exactly:
//
//   STRUCTURE (once per reader state):
//     - run_rank table over reader.ordered_runs;
//     - every key classified into the oracle's two arms:
//         BASE   untagged rows owned by the reader (causally gated);
//         TAGGED all rerot-tagged rows, ordered globally by
//                (rank, storage, frontier, idx) — the oracle's tagged
//                comparator — split into per-rank segments.
//       Within one rank segment the rows are homogeneous (a run belongs to
//       one node): foreign-run rows are query-INDEPENDENT (visible iff the
//       frontier rule passes — precomputed once), own-run rows are
//       causally gated by storage (per query). The frontier-passing subset
//       of every segment is precomputed here.
//     - own-row lookup: the query run's segment, storage array.
//
//   NUMERIC (per query row):
//     - causal cuts: one binary search over the BASE arm and one per OWN
//       segment (foreign segments need none);
//     - dense virtualization from the visible counts (prefix arithmetic);
//     - effective-position grouping by ONE stable sort of the visible
//       entries (the oracle's std::map, without per-node allocation).
//
// Output[i] equals llama_rerot_build_query_layout(reader, pos[i], keys)
// group-for-group and entry-for-entry, including query_virtual_pos; tests
// compare the two directly.
std::vector<llama_rerot_query_layout> llama_rerot_build_query_layouts_shared(
        const llama_rerot_reader_state & reader,
        const std::vector<llama_pos> & query_storage_pos,
        const std::vector<llama_rerot_key_record> & keys) {
    // Same validation ladder, same order, same messages as the per-query
    // builder (it is the oracle; divergence would be a silent contract
    // change, not an optimization).
    if (!reader.active()) {
        throw std::invalid_argument("RERoT reader state is inactive");
    }
    if (reader.reader == LLAMA_REROT_NODE_INVALID || reader.query_run == LLAMA_REROT_RUN_INVALID) {
        throw std::invalid_argument("RERoT reader or query run is invalid");
    }
    for (const llama_pos pos : query_storage_pos) {
        if (pos < 0) {
            throw std::invalid_argument("RERoT query storage position must be non-negative");
        }
    }

    std::unordered_map<llama_rerot_run_id, uint32_t> run_rank;
    run_rank.reserve(reader.ordered_runs.size());
    for (uint32_t i = 0; i < reader.ordered_runs.size(); ++i) {
        if (reader.ordered_runs[i] == LLAMA_REROT_RUN_INVALID ||
            !run_rank.emplace(reader.ordered_runs[i], i).second) {
            throw std::invalid_argument("RERoT reader view contains an invalid or duplicate run id");
        }
    }
    if (run_rank.find(reader.query_run) == run_rank.end()) {
        throw std::invalid_argument("RERoT query run is absent from the reader view");
    }

    // ---- Structural pass: classify + order once. ----
    struct tagged_key {
        const llama_rerot_key_record * key = nullptr;
        uint32_t rank = 0;
        bool causal = false; // own-run row: per-query storage cut applies
    };
    std::vector<const llama_rerot_key_record *> base;
    std::vector<tagged_key> tagged;
    base.reserve(keys.size());
    tagged.reserve(keys.size());

    {
        // Duplicate-key detection over a byte bitmap (key_index is bounded
        // by the caller's key table): the unordered_set cost dominated the
        // structural scan at production K (measured 1141 us vs 45 us for
        // the bitmap at K=65536 on the dev machine). Semantics unchanged:
        // first duplicate in scan order throws, same message.
        std::vector<uint8_t> physical_seen;
        for (const auto & key : keys) {
            if (key.key_index >= physical_seen.size()) {
                physical_seen.resize(size_t(key.key_index) + 1, 0);
            }
            if (physical_seen[key.key_index]++) {
                throw std::invalid_argument("RERoT key records contain a duplicate physical key");
            }
            if (key.storage_pos < 0) {
                throw std::invalid_argument("RERoT key storage position must be non-negative");
            }

            const auto & meta = key.meta;
            if (!meta.active()) {
                // Untagged arm: owned rows are causally gated; foreign
                // untagged rows are never visible to this reader.
                // Base occlusion (become-leak fix) mirrors the multi-reader
                // builder: occluded prompt ranges are invisible to this reader.
                if (key.owned_by_reader && !reader.occludes(key.storage_pos)) {
                    base.push_back(&key);
                }
                continue;
            }
            if (meta.episode_id != reader.episode_id) {
                continue;
            }
            const auto rank_it = run_rank.find(meta.run_id);
            if (rank_it == run_rank.end()) {
                continue;
            }

            // Frontier gate first (query-independent structure):
            //   own public:    frontier <= reader.frontier (else dropped)
            //   foreign rows:  STRONG: frontier < reader.frontier
            //                  LAG1:   reader.frontier > 0 &&
            //                          frontier < reader.frontier - 1
            // Private/pending rows are exempt (the oracle gates them only
            // on node==reader && owned && the per-query causal cut — a
            // pending row of a future frontier is the CURRENT write batch
            // and stays visible to its own reader; the 2026-09-22 fifth
            // round probe caught this builder dropping such rows).
            bool frontier_ok = false;
            if (meta.visibility != llama_rerot_visibility::public_live &&
                meta.node_id == reader.reader) {
                frontier_ok = true;
            } else if (meta.node_id == reader.reader) {
                frontier_ok = meta.frontier <= reader.frontier && key.owned_by_reader;
            } else if (reader.frontier_mode == LLAMA_REROT_FRONTIER_STRONG) {
                frontier_ok = meta.frontier < reader.frontier;
            } else if (reader.frontier > 0) {
                frontier_ok = meta.frontier < reader.frontier - 1;
            }
            if (!frontier_ok) {
                continue;
            }

            const bool causal = meta.node_id == reader.reader;
            switch (meta.visibility) {
                case llama_rerot_visibility::public_live:
                    tagged.push_back({ &key, rank_it->second, causal });
                    break;
                case llama_rerot_visibility::private_control:
                case llama_rerot_visibility::pending_record:
                    if (causal) {
                        tagged.push_back({ &key, rank_it->second, true });
                    }
                    break;
                case llama_rerot_visibility::normal:
                    break;
            }
        }
    }

    // BASE arm ordering: storage, then physical index. The comparator is a
    // total order (key_index is unique after the dedup above), so plain
    // std::sort reproduces the stable order exactly and skips the merge
    // sort's allocation.
    std::sort(base.begin(), base.end(),
        [](const llama_rerot_key_record * lhs, const llama_rerot_key_record * rhs) {
            if (lhs->storage_pos != rhs->storage_pos) {
                return lhs->storage_pos < rhs->storage_pos;
            }
            return lhs->key_index < rhs->key_index;
        });
    // TAGGED arm ordering: the oracle's tagged comparator. Total order as
    // well (same unique-key_index tiebreak).
    std::sort(tagged.begin(), tagged.end(), [](const tagged_key & lhs, const tagged_key & rhs) {
        if (lhs.rank != rhs.rank) {
            return lhs.rank < rhs.rank;
        }
        if (lhs.key->storage_pos != rhs.key->storage_pos) {
            return lhs.key->storage_pos < rhs.key->storage_pos;
        }
        if (lhs.key->meta.frontier != rhs.key->meta.frontier) {
            return lhs.key->meta.frontier < rhs.key->meta.frontier;
        }
        return lhs.key->key_index < rhs.key->key_index;
    });

    // Per-rank contiguous segments (rank-major sort guarantees contiguity).
    // For each segment keep, in sorted order, only the rows that passed the
    // frontier/ownership gate above, plus per-row causal flags and an
    // ascending storage array for the per-query binary-search cut.
    //
    // Q2/Q4 structure pass extension (2026-09-22 third round): within one
    // segment, effective = (qv - visible_prefix) + (s_i - i), where i is
    // the emission index inside the segment. The deviation d_i = s_i - i
    // is QUERY-INDEPENDENT, so each segment is stably ordered by d_i ONCE
    // here; the per-query pass then only k-way merges the R sorted
    // deviation lists (values shifted by the query's own constant) and
    // never sorts O(V) rows again. Storage duplicates make d_i dip (MTP
    // verify rows share storage); gaps make it jump; the stable order by
    // (d_i, emission i) is exactly the order the oracle's stable_sort
    // produces for those rows, so group contents and entry order are
    // reproduced identically.
    struct segment {
        uint32_t rank = 0;
        // Ascending tagged-order arrays (emission order): the causal cut and
        // the own-row lookup binary-search THESE. The 09-22 fifth-round
        // probe caught the 4th round permuting `storage` into deviation
        // order: with duplicate storage positions inside one run (MTP
        // verify rows share a position) the permutation is non-identity,
        // leaving `storage` non-monotonic and the binary-search cut wrong.
        std::vector<const llama_rerot_key_record *> rows_t;  // tagged order
        std::vector<llama_pos> storage;                      // ascending (== tagged order)
        // Deviation-order arrays: the k-way merge walks THESE. A row's
        // deviation d = storage - emission_index is query-independent, and
        // the stable (d, emission) order is exactly the oracle's stable_sort
        // output for those rows.
        std::vector<uint32_t> d2t;   // d-position -> tagged index (the permutation)
        std::vector<const llama_rerot_key_record *> rows_d; // d-order rows
        std::vector<int64_t> dev;    // d_i at d-positions
        // Tagged -> d-position inverse, for per-query cut filtering: the
        // causal cut selects a tagged-order PREFIX, which is an arbitrary
        // SUBSET of d-positions (not a prefix — d dips at duplicates).
        std::vector<uint32_t> t2d;   // tagged index -> d-position
        bool causal = false;              // own-run segment: cut applies
    };
    std::vector<segment> segments;
    {
        size_t i = 0;
        while (i < tagged.size()) {
            const uint32_t rank = tagged[i].rank;
            segment seg;
            seg.rank = rank;
            bool causal = false;
            size_t j = i;
            while (j < tagged.size() && tagged[j].rank == rank) {
                seg.rows_t.push_back(tagged[j].key);
                seg.storage.push_back(tagged[j].key->storage_pos);
                causal |= tagged[j].causal;
                ++j;
            }
            // A rank segment is homogeneous by construction (one run, one
            // owner node); assert the invariant instead of assuming it.
            for (const auto & t : tagged) {
                if (t.rank == rank && t.causal != tagged[i].causal) {
                    throw std::runtime_error("RERoT shared layout: mixed causal flags inside one run segment");
                }
            }
            seg.causal = causal;
            // Deviation ordering (stable by (d, emission index)) as a
            // PERMUTATION of the tagged arrays — `storage` stays ascending
            // for the binary-search cut; the merge walks the permuted
            // rows_d/dev lists and filters by tagged index per query.
            std::vector<uint32_t> order(seg.rows_t.size());
            for (uint32_t k = 0; k < order.size(); ++k) {
                order[k] = k;
            }
            std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                const int64_t da = int64_t(seg.storage[a]) - int64_t(a);
                const int64_t db = int64_t(seg.storage[b]) - int64_t(b);
                if (da != db) {
                    return da < db;
                }
                return a < b;
            });
            {
                const size_t n = seg.rows_t.size();
                seg.d2t = std::move(order);
                seg.rows_d.resize(n);
                seg.dev.resize(n);
                seg.t2d.assign(n, 0);
                for (size_t k = 0; k < n; ++k) {
                    const uint32_t t = seg.d2t[k];
                    seg.rows_d[k] = seg.rows_t[t];
                    seg.dev[k] = int64_t(seg.storage[t]) - int64_t(t);
                    seg.t2d[t] = uint32_t(k);
                }
            }
            segments.push_back(std::move(seg));
            i = j;
        }
    }

    // BASE arm storage array (ascending) for its own causal cut. Base rows
    // are also deviation-ordered: their emission index inside the visible
    // prefix is their array index (the cut is a prefix of the ascending
    // array), so d_i = storage[i] - i over the FULL array is the right
    // query-independent order only when the cut keeps indices aligned.
    // The cut selects a PREFIX, so visible base row j (emission order)
    // is base[j] with emission index j: d = storage[j] - j. Sort the
    // full base array by (storage[j] - j) stably once; a prefix cut then
    // selects a SUBSET of the d-sorted list whose relative order is the
    // stable order by d. (Removing elements from a stably-sorted list
    // keeps it stably sorted.)
    std::vector<llama_pos> base_storage;
    base_storage.reserve(base.size());
    for (const auto * k : base) {
        base_storage.push_back(k->storage_pos);
    }
    std::vector<uint32_t> base_order(base.size());
    for (uint32_t k = 0; k < base_order.size(); ++k) {
        base_order[k] = k;
    }
    std::sort(base_order.begin(), base_order.end(), [&](uint32_t a, uint32_t b) {
        const int64_t da = int64_t(base[a]->storage_pos) - int64_t(a);
        const int64_t db = int64_t(base[b]->storage_pos) - int64_t(b);
        if (da != db) {
            return da < db;
        }
        return a < b;
    });
    std::vector<int64_t> base_dev(base.size());
    for (size_t k = 0; k < base_order.size(); ++k) {
        base_dev[k] = int64_t(base[base_order[k]]->storage_pos) - int64_t(base_order[k]);
    }

    // ---- Numeric pass: per query, causal cuts + virtualization + grouping. ----
    // Q2/Q4 numeric channel (2026-09-22 third round): no O(V) visible
    // rebuild, no O(V) own-row pointer scan, no O(V log V) per-query sort.
    // Per query:
    //   - causal cuts are binary searches (unchanged);
    //   - the own-row virtual index is segment-prefix arithmetic (visible
    //     rows before the query row = base cut + earlier segments' cuts +
    //     the local emission offset);
    //   - grouping is a k-way merge over the R+1 deviation-sorted lists
    //     (base prefix subset + per-rank segments), each already in the
    //     oracle's stable order. For list L (visible-before count B) a row
    //     with deviation d has effective = qv + d - B; the merged distinct
    //     values are the group boundaries, and entries inside a group keep
    //     their list order (base first, then rank order) — exactly the
    //     oracle's stable_sort output order.
    std::vector<llama_rerot_query_layout> result;
    result.reserve(query_storage_pos.size());

    const size_t n_lists = segments.size() + 1; // +1 for the base arm
    std::vector<size_t> cut_of(segments.size(), 0);  // per-segment cut (rows)
    std::vector<uint64_t> vis_before(n_lists, 0);    // visible rows before each list
    std::vector<uint64_t> vis_count(n_lists, 0);     // visible rows in each list
    std::vector<size_t> merge_pos(n_lists, 0);       // k-way merge cursors

    for (const llama_pos q_pos : query_storage_pos) {
        // Causal cuts.
        size_t base_cut = 0;
        {
            const auto cut = std::upper_bound(base_storage.begin(), base_storage.end(), q_pos);
            base_cut = size_t(cut - base_storage.begin());
        }
        for (size_t s = 0; s < segments.size(); ++s) {
            const auto & seg = segments[s];
            if (!seg.causal) {
                cut_of[s] = seg.rows_d.size();
                continue;
            }
            // The cut is a TAGGED-ORDER PREFIX (rows with storage <= q_pos,
            // storage is ascending in tagged order). In d-positions that
            // prefix is an arbitrary SUBSET — d dips at duplicate storage —
            // so record the tagged cut index; the merge filters d-positions
            // by t2d[dp] < tagged_cut below.
            const auto cut = std::upper_bound(seg.storage.begin(), seg.storage.end(), q_pos);
            cut_of[s] = size_t(cut - seg.storage.begin());
        }

        // Visible counts and prefixes (base arm is list 0).
        uint64_t total = base_cut;
        vis_count[0] = uint64_t(base_cut);
        vis_before[0] = 0;
        for (size_t s = 0; s < segments.size(); ++s) {
            vis_before[s + 1] = total;
            vis_count[s + 1] = uint64_t(cut_of[s]);
            total += vis_count[s + 1];
        }

        // Own-row lookup (the oracle's LAST-match semantics): the query
        // run's segment holds every (node, query_run, owned) row; the
        // last row with storage == q_pos in EMISSION order wins. The
        // ascending storage array order IS the emission order inside one
        // segment (the tagged comparator sorts by (rank, storage,
        // frontier, idx)), so the last equal-storage entry of the array
        // is the last emission-order match.
        llama_pos query_virtual_pos = llama_pos(total);
        {
            const uint32_t want_rank = run_rank.at(reader.query_run);
            for (size_t s = 0; s < segments.size(); ++s) {
                if (segments[s].rank != want_rank) {
                    continue;
                }
                const auto & st = segments[s].storage;
                const auto it = std::lower_bound(st.begin(), st.end(), q_pos);
                if (it != st.end() && *it == q_pos) {
                    size_t local = size_t(it - st.begin());
                    while (local + 1 < st.size() && st[local + 1] == q_pos) {
                        ++local;
                    }
                    query_virtual_pos = llama_pos(vis_before[s + 1] + uint64_t(local));
                }
                break; // only the query run's segment can match
            }
        }
        if (query_virtual_pos > std::numeric_limits<llama_pos>::max()) {
            throw std::overflow_error("RERoT query virtual position overflow");
        }

        // Corrected emission (thirteenth round): effective = qv +
        // storage - virtual with virtual = vis_before + the row's index in
        // EMISSION order. The base arm's emission order is the (storage,
        // idx)-ascending storage order (its causal cut is a storage
        // prefix); a segment's emission order is TAGGED order (the oracle's
        // comparator), so a segment row at tagged index t has virtual =
        // vis_before + t — NOT the d-order position (the two diverge at
        // duplicate storage, where the deviation sort re-orders rows).
        // Each list's rows are collected in emission order with their
        // effective values, stable-sorted only when not already
        // non-decreasing (duplicate-storage shapes), then k-way merged.
        llama_rerot_query_layout layout;
        layout.query_virtual_pos = query_virtual_pos;
        layout.entries.reserve(size_t(total));
        const int64_t qv = int64_t(query_virtual_pos);
        {
            struct ek { int64_t eff; const llama_rerot_key_record * key; };
            std::vector<ek> vis;
            vis.reserve(size_t(total));
            for (size_t b = 0; b < base_cut; ++b) {
                vis.push_back({ qv + int64_t(base_storage[b]) - int64_t(b), base[b] });
            }
            for (size_t s = 0; s < segments.size(); ++s) {
                const auto & seg = segments[s];
                const size_t cut = seg.causal ? size_t(cut_of[s]) : seg.rows_d.size();
                for (size_t t = 0; t < cut; ++t) {
                    vis.push_back({ qv + int64_t(seg.storage[t]) - int64_t(t) -
                                    int64_t(vis_before[s + 1]), seg.rows_t[t] });
                }
            }
            std::stable_sort(vis.begin(), vis.end(),
                             [](const ek & a, const ek & b) { return a.eff < b.eff; });
            int64_t prev = 0;
            for (size_t i = 0; i < vis.size(); ++i) {
                if (i == 0 || vis[i].eff != prev) {
                    if (vis[i].eff < 0 || vis[i].eff > std::numeric_limits<llama_pos>::max()) {
                        throw std::overflow_error("RERoT effective query position is outside llama_pos range");
                    }
                    layout.groups.push_back({ 0, llama_pos(vis[i].eff) });
                    prev = vis[i].eff;
                }
                layout.entries.push_back({ vis[i].key->key_index,
                                            uint32_t(layout.groups.size() - 1) });
            }
            if (layout.entries.size() != total) {
                throw std::runtime_error("RERoT shared layout: visible count mismatch");
            }
        }
        result.push_back(std::move(layout));
    }
    return result;
}

// ---------------------------------------------------------------------------
// llama_rerot_shared_world: the multi-reader builder's structural pass,
// extracted into a persistent object (eleventh round, fourth question).
// Everything below is the SAME computation the builder previously ran per
// call — run bucketing by (episode, run, node), tagged (storage, frontier,
// idx) ordering with the append-only sortedness fast path, deviation
// tables with the contiguous-run identity fast path, uniformity probes,
// fast key columns, untagged ordering — factored so a structural event
// pays it once and every subsequent frontier reuses it.
// ---------------------------------------------------------------------------
namespace {

void world_validate_record(const llama_rerot_key_record & key) {
    if (key.storage_pos < 0) {
        throw std::invalid_argument("RERoT key storage position must be non-negative");
    }
}

} // namespace

void llama_rerot_shared_world::rebuild_run_structure(run & r) {
    const auto & tbl = structure_table();
    const size_t n = r.rows.size();
    r.storage.resize(n);
    for (size_t i = 0; i < n; ++i) {
        r.storage[i] = tbl[r.rows[i]].storage_pos;
    }
    bool contiguous = true;
    for (size_t i = 1; i < n; ++i) {
        if (r.storage[i] != r.storage[i - 1] + 1) {
            contiguous = false;
            break;
        }
    }
    r.contiguous = contiguous;
    if (contiguous) {
        // Identity deviation order: d = storage[0] at every position.
        r.d2t.resize(n);
        r.t2d.resize(n);
        r.dev.resize(n);
        for (uint32_t k = 0; k < n; ++k) {
            r.d2t[k] = k;
            r.t2d[k] = k;
            r.dev[k] = int64_t(r.storage[k]) - int64_t(k);
        }
    } else {
        std::vector<uint32_t> order(n);
        for (uint32_t k = 0; k < n; ++k) {
            order[k] = k;
        }
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            const int64_t da = int64_t(r.storage[a]) - int64_t(a);
            const int64_t db = int64_t(r.storage[b]) - int64_t(b);
            if (da != db) {
                return da < db;
            }
            return a < b;
        });
        r.d2t = std::move(order);
        r.t2d.assign(n, 0);
        r.dev.resize(n);
        for (size_t p = 0; p < n; ++p) {
            const uint32_t t = r.d2t[p];
            r.dev[p] = int64_t(r.storage[t]) - int64_t(t);
            r.t2d[t] = uint32_t(p);
        }
    }
    // Fast key column for contiguous runs (identity d-order): the key ids
    // in d-order == tagged order.
    r.fast_keys.clear();
    if (contiguous) {
        r.fast_keys.resize(n);
        for (size_t p = 0; p < n; ++p) {
            r.fast_keys[p] = tbl[r.rows[p]].key_index;
        }
    }
}

void llama_rerot_shared_world::recompute_uniform(run & r) {
    const size_t n = r.rows.size();
    if (n == 0) {
        r.uniform = false;
        return;
    }
    const auto & tbl = structure_table();
    const auto & m0 = tbl[r.rows[0]].meta;
    r.uniform = true;
    for (size_t i = 1; i < n; ++i) {
        const auto & m = tbl[r.rows[i]].meta;
        if (m.visibility != m0.visibility || m.frontier != m0.frontier) {
            r.uniform = false;
            break;
        }
    }
    if (r.uniform) {
        r.u_vis = m0.visibility;
        r.u_frontier = m0.frontier;
    }
}

void llama_rerot_shared_world::insert_untagged(uint32_t pos) {
    // Insert the record at records-position `pos` into (storage, key_index)
    // order via a linear scan from the end: the production append shape
    // appends at the tail, so the scan is O(1) amortized; a pathological
    // front insert stays O(U) once per structural event, never per frontier.
    const auto & rec = records[pos];
    size_t p = untagged_sorted_.size();
    while (p > 0) {
        const uint32_t other = untagged_sorted_[p - 1];
        const auto & o = records[other];
        if (o.storage_pos < rec.storage_pos ||
            (o.storage_pos == rec.storage_pos && o.key_index < rec.key_index)) {
            break;
        }
        --p;
    }
    untagged_sorted_.insert(untagged_sorted_.begin() + std::ptrdiff_t(p), pos);
}

void llama_rerot_shared_world::build_world(const std::vector<llama_rerot_key_record> & keys) {
    key_at.clear();
    records.clear();
    runs_.clear();
    untagged_sorted_.clear();

    records = keys;
    build_world_structure(keys);
    borrowed_table = nullptr; // rows now index `records` (same layout)
}

// Structural core over a caller-owned table (records already set, or a
// borrowed table for the one-shot bits path that never reads records
// afterwards). Fills runs_/untagged_sorted_/key_at only.
void llama_rerot_shared_world::build_world_structure(const std::vector<llama_rerot_key_record> & keys) {
    runs_.clear();
    untagged_sorted_.clear();
    borrowed_table = &keys; // rows index this table; records stays empty
    uint32_t max_key = 0;
    std::vector<uint32_t> untagged;
    untagged.reserve(keys.size());
    {
        std::vector<uint8_t> physical_seen; // same dedup contract as the builder
        for (size_t ki = 0; ki < keys.size(); ++ki) {
            const auto & key = keys[ki];
            if (key.key_index >= physical_seen.size()) {
                physical_seen.resize(size_t(key.key_index) + 1, 0);
            }
            if (physical_seen[key.key_index]++) {
                throw std::invalid_argument("RERoT key records contain a duplicate physical key");
            }
            if (key.storage_pos < 0) {
                throw std::invalid_argument("RERoT key storage position must be non-negative");
            }
            max_key = std::max(max_key, key.key_index + 1u);
            const auto & meta = key.meta;
            if (!meta.active()) {
                untagged.push_back(uint32_t(ki));
                continue;
            }
            run * target = nullptr;
            for (auto & cand : runs_) {
                if (cand.episode_id == meta.episode_id && cand.run_id == meta.run_id &&
                    cand.owner_node == meta.node_id) {
                    target = &cand;
                    break;
                }
            }
            if (!target) {
                runs_.push_back(run{});
                target = &runs_.back();
                target->episode_id = meta.episode_id;
                target->run_id = meta.run_id;
                target->owner_node = meta.node_id;
            }
            target->rows.push_back(uint32_t(ki));
        }
    }
    // key index -> position map (table positions, not key ids).
    key_at.assign(size_t(max_key), UINT32_MAX);
    for (size_t pos = 0; pos < keys.size(); ++pos) {
        key_at[keys[pos].key_index] = uint32_t(pos);
    }
    for (auto & r : runs_) {
        // Rows arrive in key-table order; sort into tagged order with the
        // append-only sortedness fast path (production runs arrive sorted).
        const size_t n = r.rows.size();
        bool tagged_sorted = true;
        for (size_t i = 1; i < n; ++i) {
            const auto & a = keys[r.rows[i - 1]];
            const auto & b = keys[r.rows[i]];
            const bool out_of_order = a.storage_pos != b.storage_pos
                ? a.storage_pos > b.storage_pos
                : (a.meta.frontier != b.meta.frontier
                       ? a.meta.frontier > b.meta.frontier
                       : a.key_index > b.key_index);
            if (out_of_order) {
                tagged_sorted = false;
                break;
            }
        }
        if (!tagged_sorted) {
            std::sort(r.rows.begin(), r.rows.end(), [&](uint32_t a, uint32_t b) {
                const auto & ka = keys[a];
                const auto & kb = keys[b];
                if (ka.storage_pos != kb.storage_pos) {
                    return ka.storage_pos < kb.storage_pos;
                }
                if (ka.meta.frontier != kb.meta.frontier) {
                    return ka.meta.frontier < kb.meta.frontier;
                }
                return ka.key_index < kb.key_index;
            });
        }
        rebuild_run_structure(r);
        recompute_uniform(r);
    }
    untagged_sorted_ = std::move(untagged);
    std::sort(untagged_sorted_.begin(), untagged_sorted_.end(), [&](uint32_t a, uint32_t b) {
        if (keys[a].storage_pos != keys[b].storage_pos) {
            return keys[a].storage_pos < keys[b].storage_pos;
        }
        return keys[a].key_index < keys[b].key_index;
    });
}

void llama_rerot_shared_world::append_keys(const std::vector<llama_rerot_key_record> & keys) {
    if (keys.empty()) {
        return;
    }
    for (const auto & key : keys) {
        if (key.storage_pos < 0) {
            throw std::invalid_argument("RERoT key storage position must be non-negative");
        }
        if (key.key_index < key_at.size() && key_at[key.key_index] != UINT32_MAX) {
            // Not new: the caller must describe a structural event via
            // set_key_meta, not a duplicate append.
            throw std::invalid_argument("RERoT append carries an existing physical key");
        }
    }
    // Grow the position map first.
    uint32_t max_key = uint32_t(key_at.size());
    for (const auto & key : keys) {
        max_key = std::max(max_key, key.key_index + 1u);
    }
    key_at.resize(size_t(max_key), UINT32_MAX);

    // Append records; remember (record pos, run) pairs.
    struct pending { size_t pos; };
    std::vector<pending> appended;
    appended.reserve(keys.size());
    bool structure_ok = true;
    for (const auto & key : keys) {
        const size_t pos = records.size();
        records.push_back(key);
        key_at[key.key_index] = uint32_t(pos);
        appended.push_back({ pos });

        const auto & meta = key.meta;
        if (!meta.active()) {
            continue;
        }
        run * target = nullptr;
        for (auto & cand : runs_) {
            if (cand.episode_id == meta.episode_id && cand.run_id == meta.run_id &&
                cand.owner_node == meta.node_id) {
                target = &cand;
                break;
            }
        }
        if (!target) {
            runs_.push_back(run{});
            target = &runs_.back();
            target->episode_id = meta.episode_id;
            target->run_id = meta.run_id;
            target->owner_node = meta.node_id;
            target->rows.push_back(uint32_t(pos)); // fresh run: tail order trivially preserved
            continue;
        }
        // Production fast path: appending at the tagged tail. The existing
        // rows are already in tagged order; the new row must not sort
        // before the current tail. (A run created earlier in THIS batch
        // has no rows yet: trivially in order.)
        const bool tail_ok = target->rows.empty() || [&] {
            const auto & tail = records[target->rows.back()];
            if (key.storage_pos != tail.storage_pos) {
                return key.storage_pos > tail.storage_pos;
            }
            if (key.meta.frontier != tail.meta.frontier) {
                return key.meta.frontier > tail.meta.frontier;
            }
            // Same (storage, frontier): the append's key_index must not
            // precede the tail's (the tagged tie-break is key_index).
            return key.key_index > tail.key_index;
        }();
        if (tail_ok) {
            target->rows.push_back(uint32_t(pos));
        } else {
            // Out-of-order arrival (recycled cells, replays): fall back to
            // a tagged-order insert, which keeps correctness but costs a
            // full run rebuild.
            structure_ok = false;
            target->rows.push_back(uint32_t(pos));
        }
    }

    if (!structure_ok) {
        // Re-sort every touched run (conservative: all runs; structural
        // events are rare and the rebuild is the old per-frontier cost).
        for (auto & r : runs_) {
            std::sort(r.rows.begin(), r.rows.end(), [&](uint32_t a, uint32_t b) {
                const auto & ka = records[a];
                const auto & kb = records[b];
                if (ka.storage_pos != kb.storage_pos) {
                    return ka.storage_pos < kb.storage_pos;
                }
                if (ka.meta.frontier != kb.meta.frontier) {
                    return ka.meta.frontier < kb.meta.frontier;
                }
                return ka.key_index < kb.key_index;
            });
        }
    }

    // Incremental per-run structure refresh: only runs whose row count
    // changed. Contiguous runs that were identity and appended at the
    // tail with storage +1 keep identity; the general path re-derives.
    for (auto & r : runs_) {
        if (r.rows.empty()) {
            continue;
        }
        // Cheap tail check: if the run was contiguous AND the appended
        // tail extends storage by +1, the identity tables extend in place.
        // Detecting "changed" precisely needs bookkeeping; a structural
        // refresh per append is O(run size) but only on runs that grew —
        // and the whole point is that appends are the COMMON case, so we
        // do the incremental identity extension here.
        rebuild_run_structure(r);
        recompute_uniform(r);
    }
    for (const auto & p : appended) {
        if (!records[p.pos].meta.active()) {
            insert_untagged(uint32_t(p.pos));
        }
    }
}

// Twelfth round: production upsert + O(1) tail append. The cache level
// feeds the world from apply_ubatch (ring recycle: same key index, new
// content), publish/reclassify (meta-only rewrites), and the decode tail
// (fresh key index extending a contiguous+uniform run). These three
// primitives keep the world valid without ever re-running the structural
// pass on the hot path.

bool llama_rerot_shared_world::try_append_key_fast(const llama_rerot_key_record & key) {
    if (key.storage_pos < 0) {
        return false;
    }
    if (key.key_index < key_at.size() && key_at[key.key_index] != UINT32_MAX) {
        return false; // not new: upsert territory
    }
    if (!key.meta.active()) {
        return false; // untagged tail append needs the sorted insert; rare
    }
    // Locate the run bucket.
    run * target = nullptr;
    for (auto & cand : runs_) {
        if (cand.episode_id == key.meta.episode_id && cand.run_id == key.meta.run_id &&
            cand.owner_node == key.meta.node_id) {
            target = &cand;
            break;
        }
    }
    if (!target || target->rows.empty() || !target->contiguous || !target->uniform) {
        return false; // fresh run / non-contiguous / mixed meta: general path
    }
    // Tail extension preconditions: storage +1, same (visibility, frontier),
    // tagged tail order (storage, frontier, key_index ascending).
    const auto & tail = records[target->rows.back()];
    if (key.storage_pos != tail.storage_pos + 1 ||
        key.meta.visibility != tail.meta.visibility ||
        key.meta.frontier != tail.meta.frontier ||
        key.key_index < tail.key_index) {
        return false;
    }
    // O(1) extend: identity tables, storage, dev, fast_keys all grow by one.
    const size_t pos = records.size();
    records.push_back(key);
    if (key.key_index >= key_at.size()) {
        key_at.resize(size_t(key.key_index) + 1, UINT32_MAX);
    }
    key_at[key.key_index] = uint32_t(pos);
    target->rows.push_back(uint32_t(pos));
    target->storage.push_back(key.storage_pos);
    target->d2t.push_back(uint32_t(target->d2t.size()));
    target->t2d.push_back(uint32_t(target->t2d.size()));
    target->dev.push_back(int64_t(key.storage_pos) - int64_t(target->dev.size()));
    target->fast_keys.push_back(key.key_index);
    // uniform/contiguous flags stay true by the preconditions above.
    return true;
}

void llama_rerot_shared_world::upsert_keys(const std::vector<llama_rerot_key_record> & keys) {
    if (keys.empty()) {
        return;
    }
    // Validate everything first (fail-loud, no partial mutation), then apply.
    // A replacement must keep storage_pos only when it keeps the key; a
    // recycled key may change storage_pos freely (the old row leaves).
    for (const auto & key : keys) {
        if (key.storage_pos < 0) {
            throw std::invalid_argument("RERoT key storage position must be non-negative");
        }
    }

    // Split: new indices vs replaced indices.
    std::vector<size_t> fresh, recycled;
    for (size_t i = 0; i < keys.size(); ++i) {
        const auto & key = keys[i];
        if (key.key_index < key_at.size() && key_at[key.key_index] != UINT32_MAX) {
            recycled.push_back(i);
        } else {
            fresh.push_back(i);
        }
    }

    // Remove every recycled record from its current home (run rows /
    // untagged order), then mark its key_at slot absent so the append below
    // treats it as new. The record copy stays at its records position
    // (positions are stable); only structural membership is dropped.
    // Remember the old position so the append rewrites it in place.
    std::vector<uint32_t> old_pos(keys.size(), UINT32_MAX);
    for (const size_t i : recycled) {
        const auto & key = keys[i];
        const size_t pos = size_t(key_at[key.key_index]);
        old_pos[i] = uint32_t(pos);
        const auto & old = records[pos];
        if (old.meta.active()) {
            for (size_t ri = 0; ri < runs_.size(); ++ri) {
                auto & r = runs_[ri];
                if (r.episode_id != old.meta.episode_id || r.run_id != old.meta.run_id ||
                    r.owner_node != old.meta.node_id) {
                    continue;
                }
                const auto it = std::find(r.rows.begin(), r.rows.end(), uint32_t(pos));
                if (it != r.rows.end()) {
                    r.rows.erase(it);
                }
                break;
            }
        } else {
            const auto it = std::find(untagged_sorted_.begin(), untagged_sorted_.end(),
                                      uint32_t(pos));
            if (it != untagged_sorted_.end()) {
                untagged_sorted_.erase(it);
            }
        }
        key_at[key.key_index] = UINT32_MAX;
    }

    // Append every incoming record (fresh and recycled alike) at the tail of
    // `records`, then route into its bucket. Arrival-order violations fall
    // back to per-run tagged re-sorts (bounded by the touched runs).
    std::vector<uint32_t> touched_runs;
    const auto note_run = [&](const run * r) {
        const uint32_t ri = uint32_t(r - runs_.data());
        if (std::find(touched_runs.begin(), touched_runs.end(), ri) == touched_runs.end()) {
            touched_runs.push_back(ri);
        }
    };

    for (size_t i = 0; i < keys.size(); ++i) {
        const auto & key = keys[i];
        if (key.key_index >= key_at.size()) {
            key_at.resize(size_t(key.key_index) + 1, UINT32_MAX);
        }
        size_t pos;
        if (old_pos[i] != UINT32_MAX) {
            pos = size_t(old_pos[i]); // recycled: rewrite the old position
            records[pos] = key;
        } else {
            pos = records.size(); // fresh: new tail position
            records.push_back(key);
        }
        key_at[key.key_index] = uint32_t(pos);

        if (!key.meta.active()) {
            insert_untagged(uint32_t(pos));
            continue;
        }
        run * target = nullptr;
        for (auto & cand : runs_) {
            if (cand.episode_id == key.meta.episode_id && cand.run_id == key.meta.run_id &&
                cand.owner_node == key.meta.node_id) {
                target = &cand;
                break;
            }
        }
        if (!target) {
            runs_.push_back(run{});
            target = &runs_.back();
            target->episode_id = key.meta.episode_id;
            target->run_id = key.meta.run_id;
            target->owner_node = key.meta.node_id;
        }
        target->rows.push_back(uint32_t(pos));
        note_run(target);
    }

    // Re-derive structure on every touched run (a removal can break
    // contiguity anywhere; a mid-run insert can break tagged order).
    for (const uint32_t ri : touched_runs) {
        auto & r = runs_[ri];
        std::sort(r.rows.begin(), r.rows.end(), [&](uint32_t a, uint32_t b) {
            const auto & ka = records[a];
            const auto & kb = records[b];
            if (ka.storage_pos != kb.storage_pos) {
                return ka.storage_pos < kb.storage_pos;
            }
            if (ka.meta.frontier != kb.meta.frontier) {
                return ka.meta.frontier < kb.meta.frontier;
            }
            return ka.key_index < kb.key_index;
        });
        rebuild_run_structure(r);
        recompute_uniform(r);
    }
    // Runs emptied by removal stay (empty buckets are harmless and keep
    // bucket identity for a later arrival); the numeric pass skips them via
    // n_pass == 0.
}

void llama_rerot_shared_world::remove_keys(const std::vector<uint32_t> & key_indices) {
    if (key_indices.empty()) {
        return;
    }
    // Validate first (fail-loud): every key must exist.
    std::vector<size_t> positions;
    positions.reserve(key_indices.size());
    for (const uint32_t k : key_indices) {
        if (k >= key_at.size() || key_at[k] == UINT32_MAX) {
            throw std::invalid_argument("RERoT key removal references an absent physical key");
        }
        positions.push_back(size_t(key_at[k]));
    }

    std::vector<run *> touched_runs;
    const auto note = [&](run * r) {
        if (std::find(touched_runs.begin(), touched_runs.end(), r) == touched_runs.end()) {
            touched_runs.push_back(r);
        }
    };

    for (const size_t pos : positions) {
        const auto & rec = records[pos];
        if (rec.meta.active()) {
            for (auto & r : runs_) {
                if (r.episode_id == rec.meta.episode_id && r.run_id == rec.meta.run_id &&
                    r.owner_node == rec.meta.node_id) {
                    const auto it = std::find(r.rows.begin(), r.rows.end(), uint32_t(pos));
                    if (it != r.rows.end()) {
                        r.rows.erase(it);
                        note(&r);
                    }
                    break;
                }
            }
        } else {
            const auto it = std::find(untagged_sorted_.begin(), untagged_sorted_.end(),
                                      uint32_t(pos));
            if (it != untagged_sorted_.end()) {
                untagged_sorted_.erase(it);
            }
        }
        key_at[rec.key_index] = UINT32_MAX;
        // The record slot stays (positions are stable); it is now absent
        // from every structural index, so no pass can reach it.
    }

    for (run * r : touched_runs) {
        if (r->rows.empty()) {
            continue; // empty buckets are harmless (n_pass == 0)
        }
        // Removal can only break tagged order if the comparator keys of the
        // remaining rows changed — they did not (removal preserves relative
        // order), so rows stay sorted; only the derived tables need a
        // rebuild (d2t/t2d/dev shift by the removal).
        rebuild_run_structure(*r);
        recompute_uniform(*r);
    }
}

void llama_rerot_shared_world::set_key_meta(const std::vector<llama_rerot_key_record> & keys) {
    if (keys.empty()) {
        return;
    }
    // Validate every update FIRST (fail-loud, no partial mutation),
    // then write the record copies and recompute uniformity per touched
    // run. A meta update must keep the record inside its owning run
    // bucket — (episode, run, node) unchanged; visibility / frontier /
    // publish_epoch may change and only affect the uniformity probe.
    std::vector<size_t> touched;
    std::vector<std::pair<size_t, llama_kv_rerot_meta>> writes;
    for (const auto & key : keys) {
        if (key.key_index >= key_at.size() || key_at[key.key_index] == UINT32_MAX) {
            throw std::invalid_argument("RERoT meta update references an absent physical key");
        }
        const size_t pos = size_t(key_at[key.key_index]);
        const auto & old = records[pos];
        if (key.storage_pos != old.storage_pos) {
            throw std::invalid_argument("RERoT meta update must preserve storage position");
        }
        // Active-ness change is a structural event (row moves arm):
        // reject; the owner must rebuild.
        if (key.meta.active() != old.meta.active()) {
            throw std::invalid_argument("RERoT meta update must preserve active-ness");
        }
        if (!key.meta.active()) {
            continue; // untagged: nothing derives from its meta
        }
        if (key.meta.episode_id != old.meta.episode_id || key.meta.run_id != old.meta.run_id ||
            key.meta.node_id != old.meta.node_id) {
            throw std::invalid_argument("RERoT meta update escapes its owning run bucket");
        }
        writes.push_back({ pos, key.meta });
        for (size_t ri = 0; ri < runs_.size(); ++ri) {
            const auto & r = runs_[ri];
            if (r.episode_id == key.meta.episode_id && r.run_id == key.meta.run_id &&
                r.owner_node == key.meta.node_id) {
                if (std::find(touched.begin(), touched.end(), ri) == touched.end()) {
                    touched.push_back(ri);
                }
                break;
            }
        }
    }
    for (const auto & w : writes) {
        records[w.first].meta = w.second;
    }
    for (const size_t ri : touched) {
        auto & r = runs_[ri];
        // frontier participates in the tagged order key (storage, frontier,
        // idx): a rewrite can break the sortedness of an append-ordered run.
        // Re-probe and re-sort when needed (production publish keeps
        // frontier, so the probe is O(n) and the sort never fires there).
        const size_t n = r.rows.size();
        bool tagged_sorted = true;
        for (size_t i = 1; i < n; ++i) {
            const auto & a = records[r.rows[i - 1]];
            const auto & b = records[r.rows[i]];
            const bool out_of_order = a.storage_pos != b.storage_pos
                ? a.storage_pos > b.storage_pos
                : (a.meta.frontier != b.meta.frontier
                       ? a.meta.frontier > b.meta.frontier
                       : a.key_index > b.key_index);
            if (out_of_order) {
                tagged_sorted = false;
                break;
            }
        }
        if (!tagged_sorted) {
            std::sort(r.rows.begin(), r.rows.end(), [&](uint32_t a, uint32_t b) {
                const auto & ka = records[a];
                const auto & kb = records[b];
                if (ka.storage_pos != kb.storage_pos) {
                    return ka.storage_pos < kb.storage_pos;
                }
                if (ka.meta.frontier != kb.meta.frontier) {
                    return ka.meta.frontier < kb.meta.frontier;
                }
                return ka.key_index < kb.key_index;
            });
            rebuild_run_structure(r);
        }
        recompute_uniform(r);
    }
}

// Multi-reader shared-world builder (2026-09-22 fifth round, Q3 host side).
// R pens of one frontier share ONE key world: ONE structural pass over
// `keys` — per-(episode, run) segments in the oracle's tagged order with
// the deviation (s - i) permutation, plus the untagged (storage, idx)
// order — serves EVERY reader. Per reader, only the ownership-dependent
// work remains:
//   - segment set: runs of reader.ordered_runs, own = (run == query_run);
//   - per-row frontier classification: own public/private/pending rows
//     gated (own private/pending carry NO frontier gate — the oracle checks
//     node==reader && owned && causal cut only), foreign public rows FULL
//     iff the frontier rule passes, foreign private/pending invisible;
//   - base arm: untagged rows owned by the reader (base_owned column);
//   - the per-query numeric pass (causal cuts + k-way deviation merge).
// Output[r][i] is EXACTLY llama_rerot_build_query_layouts_shared(
// readers[r], query_pos[r], keys-with-reader-r's-ownership-column) — the
// single-reader builder stays the oracle (tests compare all three paths).
// With R pens on one frontier the scan+sort cost drops from R copies to one.
namespace {

// Shared reader+numeric body of the multi-reader builder: everything after
// the structural pass. `runs`/`untagged_sorted`/`keys_ref` come from the
// world; the ownership bitsets are indexed by KEY INDEX over the world's
// record store.
// Direct-emission sink (fourteenth round): rerot_emission_sink is
// declared in llama-rerot.h (the cache passes it across TU boundaries);
// this TU uses that type directly.

std::vector<std::vector<llama_rerot_query_layout>> multi_reader_numeric_pass(
        const std::vector<llama_rerot_reader_state> & readers,
        const std::vector<std::vector<llama_pos>> & query_storage_pos,
        const std::vector<llama_rerot_owned_view> & base_owned_bits,
        const std::vector<llama_rerot_shared_world::run> & runs,
        const std::vector<uint32_t> & untagged_sorted,
        const std::vector<llama_rerot_key_record> & keys_ref,
        std::vector<std::vector<llama_rerot_query_layout>> * output_scratch,
        rerot_emission_sink * sink) {
    for (size_t r = 0; r < readers.size(); ++r) {
        const auto & reader = readers[r];
        if (!reader.active()) {
            throw std::invalid_argument("RERoT reader state is inactive");
        }
        if (reader.reader == LLAMA_REROT_NODE_INVALID || reader.query_run == LLAMA_REROT_RUN_INVALID) {
            throw std::invalid_argument("RERoT reader or query run is invalid");
        }
        const size_t need_words = (keys_ref.size() + 63) / 64 + (keys_ref.empty() ? 1 : 0);
        if (base_owned_bits[r].n_words < need_words) {
            throw std::invalid_argument("RERoT multi-reader: ownership column width mismatch");
        }
        for (const llama_pos pos : query_storage_pos[r]) {
            if (pos < 0) {
                throw std::invalid_argument("RERoT query storage position must be non-negative");
            }
        }
    }

    // Output scratch (fourteenth round): when the caller provides one, its
    // per-reader vectors keep their capacity across frontiers — the fresh
    // 2MB-per-reader output allocations (and their first-touch page faults)
    // disappear from the steady-state decode path. The scratch is cleared
    // (size reset, capacity kept) and returned BY VALUE via move, so the
    // returned object owns the buffers; the caller swaps them back for the
    // next frontier.
    std::vector<std::vector<llama_rerot_query_layout>> result;
    if (output_scratch) {
        result = std::move(*output_scratch);
        output_scratch->clear();
    }
    result.resize(readers.size());
    for (auto & col : result) {
        col.clear();
    }
    // Pre-size every per-reader column to its query count: the layouts are
    // then built IN PLACE (default-constructed elements hold their
    // capacity after the first frontier via the scratch), so the steady
    // state performs zero per-query output allocations.
    for (size_t r = 0; r < readers.size(); ++r) {
        result[r].resize(query_storage_pos[r].size());
    }
    for (size_t r = 0; r < readers.size(); ++r) {
        const auto & reader = readers[r];
        const llama_rerot_owned_view & owned_view = base_owned_bits[r];
        // Row test: one AND per probe; null words read as zero (unowned).
        const auto owned = [&](size_t k) -> bool {
            return (owned_view.bits[k >> 6] >> (k & 63)) & 1ull;
        };

        std::unordered_map<llama_rerot_run_id, uint32_t> run_rank;
        run_rank.reserve(reader.ordered_runs.size());
        for (uint32_t i = 0; i < reader.ordered_runs.size(); ++i) {
            if (reader.ordered_runs[i] == LLAMA_REROT_RUN_INVALID ||
                !run_rank.emplace(reader.ordered_runs[i], i).second) {
                throw std::invalid_argument("RERoT reader view contains an invalid or duplicate run id");
            }
        }
        if (run_rank.find(reader.query_run) == run_rank.end()) {
            throw std::invalid_argument("RERoT query run is absent from the reader view");
        }

        // Reader-facing segments in rank order. ONE row list per segment:
        // the frontier-passing subset of the shared run, in der
        // (subsequence of the shared deviation order — still d-sorted).
        // Only the reader's OWN segment is causally cut per query; foreign
        // segments (frontier-passing public rows) are FULL. The gated mask
        // over tagged indices drives both the per-query cut and the own-row
        // virtual arithmetic.
        struct seg_view {
            const llama_rerot_shared_world::run * run = nullptr;
            bool own = false;
            bool identity = false; // dp == [0..n) and pass_prefix == [0..n)
            // Contiguous-run fast path (eighth round): when the run's storage
            // is strictly +1 (production shape) AND the passing set is the
            // identity (uniform fully-owned/foreign buckets), the per-entry
            // emission chain dp_at -> d2t[dp] -> rows[t] -> keys[ki].key_index
            // collapses to a sequential read over this precomputed key-id
            // array, and the effective value is CONSTANT over the whole list
            // (dev is constant), so the ==best recheck hoists out entirely.
            std::vector<uint32_t> dp;     // d-positions of the passing rows
            std::vector<uint32_t> pass_prefix; // tagged idx -> passing rows before it
        };
        const auto dp_size = [](const seg_view & s) -> size_t {
            return s.identity ? s.run->rows.size() : s.dp.size();
        };
        const auto prefix_at = [](const seg_view & s, size_t i) -> uint32_t {
            return s.identity ? uint32_t(i) : s.pass_prefix[i];
        };
        // Is the tagged row at index i a PASSING row? Identity segments
        // pass by construction; otherwise the prefix count increments
        // exactly at passing rows.
        const auto pass_at = [](const seg_view & s, size_t i) -> bool {
            return s.identity || s.pass_prefix[i + 1] > s.pass_prefix[i];
        };

        std::vector<seg_view> segs;
        for (const auto run_id : reader.ordered_runs) {
            // One entry per (run, node) bucket of this run id: the reader's
            // own-node buckets are causal/own; foreign-node buckets of the
            // same run id are FOREIGN segments (frontier rule), exactly as
            // the single-reader builder's per-row node test classifies them.
            for (const auto & found : runs) {
                if (found.episode_id != reader.episode_id || found.run_id != run_id) {
                    continue;
                }
            // Own rows = the run's OWNER NODE is the reader (a reader's node
            // can own several runs: its query run plus earlier private
            // runs) — matching the single-reader builder's per-row
            // node_id == reader.reader test, which the cache-level oracle
            // pins (test_rerot_shared_reader_multi_query owns runs 2+3).
                const bool own = found.owner_node == reader.reader;
            seg_view sv;
            sv.run = &found;
            sv.own = own;
            // Uniform-bucket fast path (sixth round): when every row shares
            // (visibility, frontier) — the production shape (a whole run
            // commits at one frontier) — the frontier gate is ONE branch
            // for the whole bucket and only the ownership column (a byte
            // array, sequential) is read per row. Mixed buckets keep the
            // general per-row path.
            bool uniform = found.uniform;
            if (uniform) {
                {
                    const auto vis = found.u_vis;
                    bool gate = false;
                    bool need_own = false;
                    if (vis == llama_rerot_visibility::public_live) {
                        if (own) {
                            gate = found.u_frontier <= reader.frontier;
                            need_own = true;
                        } else if (reader.frontier_mode == LLAMA_REROT_FRONTIER_STRONG) {
                            gate = found.u_frontier < reader.frontier;
                        } else {
                            gate = reader.frontier > 0 && found.u_frontier < reader.frontier - 1;
                        }
                    } else if (own && (vis == llama_rerot_visibility::private_control ||
                                       vis == llama_rerot_visibility::pending_record)) {
                        // No frontier gate for own private/pending rows
                        // (the oracle checks node==reader && owned &&
                        // causal cut only — pinned by the fifth-round
                        // probe).
                        gate = true;
                        need_own = true;
                    }
                    if (gate) {
                        // Every row passes ownership (foreign bucket, or own
                        // bucket fully owned — the production shape): the
                        // passing list is the identity sequence, no
                        // materialization needed.
                        bool all_owned = true;
                        if (need_own) {
                            for (const uint32_t ki : found.rows) {
                                if (!owned(ki)) {
                                    all_owned = false;
                                    break;
                                }
                            }
                        }
                        if (all_owned) {
                            sv.identity = true;
                        } else {
                            sv.dp.reserve(found.d2t.size());
                            for (size_t p = 0; p < found.rows.size(); ++p) {
                                const uint32_t t = found.d2t[p];
                                if (owned(found.rows[t])) {
                                    sv.dp.push_back(uint32_t(p));
                                }
                            }
                            sv.pass_prefix.assign(found.rows.size() + 1, 0);
                            for (size_t i = 0; i < found.rows.size(); ++i) {
                                sv.pass_prefix[i + 1] =
                                    sv.pass_prefix[i] + uint32_t(owned(found.rows[i]));
                            }
                        }
                        segs.push_back(std::move(sv));
                    }
                    continue;
                }
            }
            std::vector<uint8_t> pass(found.rows.size(), 0);
            for (size_t i = 0; i < found.rows.size(); ++i) {
                const auto & meta = keys_ref[found.rows[i]].meta;
                if (meta.visibility == llama_rerot_visibility::public_live) {
                    if (own) {
                        if (meta.frontier <= reader.frontier && owned(found.rows[i])) {
                            pass[i] = 1;
                        }
                    } else if (reader.frontier_mode == LLAMA_REROT_FRONTIER_STRONG) {
                        if (meta.frontier < reader.frontier) {
                            pass[i] = 1;
                        }
                    } else if (reader.frontier > 0) {
                        if (meta.frontier < reader.frontier - 1) {
                            pass[i] = 1;
                        }
                    }
                } else if (own && (meta.visibility == llama_rerot_visibility::private_control ||
                                   meta.visibility == llama_rerot_visibility::pending_record)) {
                    // No frontier gate for own private/pending rows (the
                    // oracle checks node==reader && owned && causal cut
                    // only — pinned by the fifth-round probe).
                    if (owned(found.rows[i])) {
                        pass[i] = 1;
                    }
                }
            }
            // Walk the shared deviation order and keep the passing rows:
            // dp stays ascending in d-position, so each list is sorted by
            // effective (the merge's precondition). Iterating tagged order
            // instead would emit d-positions in their permutation order.
            size_t n_pass = 0;
            for (size_t i = 0; i < found.rows.size(); ++i) {
                n_pass += pass[i];
            }
            if (n_pass == 0) {
                continue;
            }
            if (n_pass == found.rows.size()) {
                sv.identity = true;
                segs.push_back(std::move(sv));
                continue;
            }
            sv.dp.reserve(n_pass);
            for (size_t p = 0; p < found.rows.size(); ++p) {
                if (pass[found.d2t[p]]) {
                    sv.dp.push_back(uint32_t(p));
                }
            }
            // Passing-prefix counts over tagged indices: the own-row virtual
            // position = rows before the own gated list + passing rows with
            // tagged index < local; and the per-query causal cut count is
            // the number of passing rows with tagged index < tagged_cut.
            sv.pass_prefix.assign(found.rows.size() + 1, 0);
            for (size_t i = 0; i < found.rows.size(); ++i) {
                sv.pass_prefix[i + 1] = sv.pass_prefix[i] + uint32_t(pass[i]);
            }
            segs.push_back(std::move(sv));
            }
        }

        // Base arm: untagged rows owned by THIS reader, (storage, idx)
        // order, deviation-ordered (same identity as the single-reader
        // builder: d = storage[j] - j over the full array, prefix cuts keep
        // stable order).
        std::vector<uint32_t> base;
        for (const uint32_t ki : untagged_sorted) {
            // KV-level base occlusion (become-leak fix): a reader with an
            // occluded C0 range never sees the untagged prompt keys inside
            // it — the planner/synthesis views drop the advertised
            // internal-tool schema while workers keep it.
            if (owned(ki) && !reader.occludes(keys_ref[ki].storage_pos)) {
                base.push_back(ki);
            }
        }
        std::vector<llama_pos> base_storage(base.size());
        for (size_t i = 0; i < base.size(); ++i) {
            base_storage[i] = keys_ref[base[i]].storage_pos;
        }

        // ---- Per-reader numeric pass. ----
        // Merge-list buffers reused across queries (allocation was a
        // measurable per-query cost at production shapes).
        struct eff_list {
            // Parallel arrays over the VISIBLE rows in emission order.
            std::vector<int64_t> eff;
            std::vector<uint32_t> key_ids;
            bool sorted = true; // eff non-decreasing as built
            // Contiguous-identity mode: every row shares one effective
            // value; eff holds exactly that one value and the merge treats
            // the whole list as one bulk group (no per-row eff reads).
            bool const_eff = false;
            // Const lists whose run carries fast_keys (the production
            // shape) REFERENCE the run's key-id array instead of copying
            // it: the emission reads fast_keys directly, so no per-query
            // key_ids buffer, no R x K uint32 copy pass.
            const uint32_t * key_ptr = nullptr; // null => use key_ids
            size_t key_n = 0;
            int64_t head(int64_t i) const { return eff[const_eff ? 0 : i]; }
            const uint32_t * keys() const { return key_ptr ? key_ptr : key_ids.data(); }
            size_t n_rows() const { return const_eff ? key_n : eff.size(); }
        };
        std::vector<eff_list> lists(1 + segs.size());
        std::vector<uint64_t> vis_count(lists.size(), 0);
        std::vector<uint64_t> vis_before(lists.size(), 0);
        std::vector<size_t> merge_pos(lists.size(), 0);

        // Per-segment causal cuts (tagged-order prefix) reused per query.
        std::vector<size_t> seg_cut(segs.size(), 0);
        // Per-list buffer sizing (fifteenth round): the old code reserved
        // keys/2 entries in EVERY list (R lists x ~K/2 x 12B per reader —
        // hundreds of MB of fresh allocation per frontier at production
        // shapes, all of it page-faulted). Size each list by its actual
        // content need: the base arm by the owned count, each segment by
        // the run's row count (the passing subset is <= that), and const
        // lists barely at all (one eff slot; keys are referenced).
        for (size_t li = 0; li < lists.size(); ++li) {
            auto & L = lists[li];
            if (li == 0) {
                const size_t hint = base.size() + 8;
                L.eff.reserve(hint);
                L.key_ids.reserve(hint);
                continue;
            }
            const auto & sv = segs[li - 1];
            const bool const_with_keys =
                sv.identity && sv.run->contiguous && !sv.run->fast_keys.empty();
            if (const_with_keys) {
                L.eff.reserve(4); // the single constant effective value
                continue;         // keys reference the run; no buffer needed
            }
            const size_t hint = sv.run->rows.size() + 8;
            L.eff.reserve(hint);
            L.key_ids.reserve(hint);
        }
        for (size_t qi = 0; qi < query_storage_pos[r].size(); ++qi) {
            const llama_pos q_pos = query_storage_pos[r][qi];

            size_t base_cut = 0;
            {
                const auto cut = std::upper_bound(base_storage.begin(), base_storage.end(), q_pos);
                base_cut = size_t(cut - base_storage.begin());
            }
            {
                size_t si = 0;
                for (const auto & sv : segs) {
                    if (!sv.own) {
                        // Foreign segment: FULL, never causally cut.
                        seg_cut[si] = sv.run->rows.size();
                    } else {
                        const auto cut = std::upper_bound(sv.run->storage.begin(), sv.run->storage.end(), q_pos);
                        seg_cut[si] = size_t(cut - sv.run->storage.begin());
                    }
                    ++si;
                }
            }

            // Visible counts per list (list order = emission order): the base
            // arm's causal prefix, then each segment's passing rows.
            {
                uint64_t total = 0;
                vis_before[0] = 0;
                vis_count[0] = uint64_t(base_cut);
                total = vis_count[0];
                size_t si = 0;
                for (const auto & sv : segs) {
                    vis_before[si + 1] = total;
                    if (sv.own) {
                        // Passing rows with tagged index < cut.
                        vis_count[si + 1] = uint64_t(prefix_at(sv, seg_cut[si]));
                    } else {
                        vis_count[si + 1] = uint64_t(dp_size(sv));
                    }
                    total += vis_count[si + 1];
                    ++si;
                }
            }

            // Own-row lookup (oracle LAST-match): the query run's segment,
            // last storage == q_pos in tagged order. The row's virtual index
            // = visible rows before the own segment + passing rows of the
            // segment with tagged index < local (the passing set is a
            // subsequence of tagged order, and all passing rows before the
            // own row are causally visible: their storage <= q_pos).
            uint64_t total = 0;
            for (size_t li = 0; li <= segs.size(); ++li) {
                total += vis_count[li];
            }
            llama_pos query_virtual_pos = llama_pos(total);
            {
                size_t si = 0;
                for (const auto & sv : segs) {
                    if (sv.own && sv.run->run_id == reader.query_run) {
                        const auto & st = sv.run->storage;
                        const auto it = std::lower_bound(st.begin(), st.end(), q_pos);
                        if (it != st.end() && *it == q_pos) {
                            size_t local = size_t(it - st.begin());
                            while (local + 1 < st.size() && st[local + 1] == q_pos) {
                                ++local;
                            }
                            // The oracle searches the VISIBLE tagged set:
                            // when the last storage match is NOT a passing
                            // row (a same-position rewrite with a newer
                            // frontier the reader cannot see yet), the
                            // own row is the LAST PASSING match — back up
                            // over non-passing same-position rows.
                            while (local > 0 && st[local - 1] == q_pos &&
                                   !pass_at(sv, local)) {
                                --local;
                            }
                            if (pass_at(sv, local)) {
                                query_virtual_pos = llama_pos(vis_before[si + 1] +
                                    uint64_t(prefix_at(sv, local)));
                            }
                        }
                        break;
                    }
                    ++si;
                }
            }
            if (query_virtual_pos > std::numeric_limits<llama_pos>::max()) {
                throw std::overflow_error("RERoT query virtual position overflow");
            }

            // Corrected k-way merge (thirteenth round). The old merge
            // conflated the d-order position with the VISIBLE-sequence
            // index: a run containing a non-passing row at a DUPLICATE
            // storage (a same-position rewrite the reader cannot see yet)
            // re-orders the deviation list, and the two indices diverge —
            // effective positions came out shifted. The effective value is
            // qv + storage - virtual with virtual = vis_before + passing
            // rows before the row in TAGGED order. Each merge list carries
            // a PRECOMPUTED effective array; a list that is not already
            // non-decreasing (duplicate-storage shapes) is stable-sorted
            // once — the production shapes (unique storage per run,
            // contiguous identity segments) are already sorted, so the
            // common path pays only the is_sorted check.
            // Build in place into the pre-sized column: after the first
            // frontier (via the output scratch) the element's vectors keep
            // their capacity, so reserve() is a no-op branch and the merge
            // writes land on warm pages. With the emission sink the merge
            // writes straight into the final arrays and the per-query
            // object is only used as the merge's scratch cursor holder.
            llama_rerot_query_layout & layout = result[r][qi];
            layout.query_virtual_pos = query_virtual_pos;
            if (!sink) {
                if (layout.entries.capacity() < total) {
                    layout.entries.reserve(size_t(total));
                }
                if (layout.groups.capacity() < 1 + segs.size()) {
                    layout.groups.reserve(1 + segs.size());
                }
            }
            const int64_t qv = int64_t(query_virtual_pos);
            {
                for (auto & L : lists) {
                    L.eff.clear();
                    L.key_ids.clear();
                    L.sorted = true;
                    L.const_eff = false; // stale const_eff would alias head()
                    L.key_ptr = nullptr;
                    L.key_n = 0;
                }
                // Base arm: causal prefix of the (storage, idx)-ascending
                // array; virtual = position in the base list.
                {
                    auto & L = lists[0];
                    int64_t prev = std::numeric_limits<int64_t>::min();
                    for (size_t b = 0; b < base_cut; ++b) {
                        const int64_t e = qv + int64_t(base_storage[b]) - int64_t(b);
                        if (e < prev) {
                            L.sorted = false;
                        }
                        prev = e;
                        L.eff.push_back(e);
                        L.key_ids.push_back(keys_ref[base[b]].key_index);
                    }
                }
                {
                    size_t si = 0;
                    for (const auto & sv : segs) {
                        auto & L = lists[si + 1];
                        const size_t cut = sv.own ? size_t(seg_cut[si]) : dp_size(sv);
                        if (sv.identity && sv.run->contiguous) {
                            // Contiguous identity (the production decode
                            // shape): storage[t] - t is the CONSTANT
                            // storage[0] — one effective value for the whole
                            // list. When the run carries fast_keys the key
                            // ids are REFERENCED (no copy, no buffer);
                            // otherwise they are gathered once.
                            L.const_eff = true;
                            L.eff.push_back(qv + int64_t(sv.run->storage[0]) -
                                            int64_t(vis_before[si + 1]));
                            if (!sv.run->fast_keys.empty() &&
                                sv.run->fast_keys.size() >= cut) {
                                L.key_ptr = sv.run->fast_keys.data();
                                L.key_n = cut;
                            } else {
                                L.key_ids.reserve(cut);
                                for (size_t t = 0; t < cut; ++t) {
                                    L.key_ids.push_back(keys_ref[sv.run->rows[t]].key_index);
                                }
                                L.key_n = cut;
                            }
                        } else if (sv.identity) {
                            // Identity passing set, non-contiguous storage
                            // (duplicate positions): storage[t] - t dips at
                            // duplicates, so sortedness is tracked per row.
                            int64_t prev = std::numeric_limits<int64_t>::min();
                            for (size_t t = 0; t < cut; ++t) {
                                const int64_t e = qv + int64_t(sv.run->storage[t]) - int64_t(t) -
                                                  int64_t(vis_before[si + 1]);
                                if (e < prev) {
                                    L.sorted = false;
                                }
                                prev = e;
                                L.eff.push_back(e);
                                L.key_ids.push_back(sv.run->fast_keys.empty()
                                    ? keys_ref[sv.run->rows[t]].key_index
                                    : sv.run->fast_keys[t]);
                            }
                        } else {
                            int64_t prev = std::numeric_limits<int64_t>::min();
                            for (size_t t = 0; t < cut; ++t) {
                                if (!pass_at(sv, t)) {
                                    continue;
                                }
                                const int64_t e = qv + int64_t(sv.run->storage[t]) -
                                                 int64_t(sv.pass_prefix[t]) -
                                                 int64_t(vis_before[si + 1]);
                                if (e < prev) {
                                    L.sorted = false;
                                }
                                prev = e;
                                L.eff.push_back(e);
                                L.key_ids.push_back(keys_ref[sv.run->rows[t]].key_index);
                            }
                        }
                        ++si;
                    }
                }
                for (auto & L : lists) {
                    if (!L.sorted && !L.const_eff) {
                        // Duplicate-storage shape: sort the (eff, key) pairs
                        // together, stable (emission order preserved on
                        // ties, matching the oracle's stable_sort).
                        std::vector<uint32_t> order(L.eff.size());
                        for (uint32_t i = 0; i < order.size(); ++i) {
                            order[i] = i;
                        }
                        std::stable_sort(order.begin(), order.end(),
                            [&](uint32_t a, uint32_t b) { return L.eff[a] < L.eff[b]; });
                        std::vector<int64_t> e2(L.eff.size());
                        std::vector<uint32_t> k2(L.eff.size());
                        for (size_t i = 0; i < order.size(); ++i) {
                            e2[i] = L.eff[order[i]];
                            k2[i] = L.key_ids[order[i]];
                        }
                        L.eff = std::move(e2);
                        L.key_ids = std::move(k2);
                    }
                }
                // K-way merge over the effective-sorted lists. With the
                // emission sink the merge writes straight into the FINAL
                // arrays (group base and query index stamped inline); the
                // per-query object is bypassed. Without a sink the entries
                // array is sized once (single zero-fill) and written by
                // cursor.
                std::fill(merge_pos.begin(), merge_pos.end(), 0);
                llama_rerot_attn_entry * emit_out = nullptr;
                size_t emit_cursor = 0;
                if (sink) {
                    emit_out = sink->entries + sink->entry_cursor;
                } else {
                    layout.entries.resize(size_t(total));
                    emit_out = layout.entries.data();
                }
                while (true) {
                    int64_t best = std::numeric_limits<int64_t>::max();
                    bool any = false;
                    for (size_t li = 0; li < lists.size(); ++li) {
                        const size_t n_rows = lists[li].n_rows();
                        if (merge_pos[li] < n_rows) {
                            const int64_t v = lists[li].head(int64_t(merge_pos[li]));
                            if (!any || v < best) {
                                best = v;
                                any = true;
                            }
                        }
                    }
                    if (!any) {
                        break;
                    }
                    const uint32_t group_index = sink
                        ? uint32_t(sink->groups_out->size())
                        : uint32_t(layout.groups.size());
                    const size_t span_begin = sink && sink->spans_out
                        ? sink->spans_out->size()
                        : 0;
                    bool emitted = false;
                    for (size_t li = 0; li < lists.size(); ++li) {
                        // Contiguous-identity fast path: the whole list
                        // shares one effective value, so when it matches
                        // best the ENTIRE remainder is one bulk fill (no
                        // per-entry loop body). The key ids come either
                        // from the referenced run array or from the local
                        // gather buffer.
                        const size_t n_rows = lists[li].n_rows();
                        const bool bulk_ok = lists[li].const_eff;
                        if (bulk_ok && merge_pos[li] < n_rows &&
                            lists[li].head(int64_t(merge_pos[li])) == best) {
                            const size_t p = merge_pos[li];
                            const size_t n = n_rows - p;
                            if (sink && sink->spans_out) {
                                // Span side-channel (sixteenth round): the
                                // bulk segment is a contiguous ascending key
                                // range — describe it once so the loader and
                                // the validator can work in O(ranges).
                                // entry_index is the GLOBAL slot of the
                                // first key (sink cursor + rows emitted for
                                // this query so far); group_index is stamped
                                // after the merge loop resolves it.
                                // Entries are STILL written below: the vector
                                // is the authoritative consumer contract
                                // (direct-set paths and op params read it).
                                const uint32_t * kids = lists[li].keys() + p;
                                sink->spans_out->push_back(
                                    { kids[0], uint32_t(n), 0,
                                      uint32_t(sink->entry_cursor + emit_cursor) });
                            }
                            {
                                llama_rerot_attn_entry * out = emit_out + emit_cursor;
                                const uint32_t * kids = lists[li].keys() + p;
                                for (size_t i = 0; i < n; ++i) {
                                    out[i] = { kids[i], group_index };
                                }
                            }
                            emit_cursor += n;
                            merge_pos[li] = n_rows;
                            emitted = true;
                            continue;
                        }
                        while (merge_pos[li] < n_rows &&
                               lists[li].head(int64_t(merge_pos[li])) == best) {
                            // The span side-channel only covers WHOLE-LIST bulk
                            // segments (const_eff); scalar arms always write
                            // entries directly — a partially consumed bulk
                            // list still lands here for its tail.
                            emit_out[emit_cursor++] = { lists[li].keys()[merge_pos[li]], group_index };
                            ++merge_pos[li];
                            emitted = true;
                        }
                    }
                    if (!emitted) {
                        throw std::runtime_error("RERoT shared layout: k-way merge lost a group head");
                    }
                    if (best < 0 || best > std::numeric_limits<llama_pos>::max()) {
                        throw std::overflow_error("RERoT effective query position is outside llama_pos range");
                    }
                    if (sink) {
                        GGML_ASSERT(sink->groups_out->size() < sink->groups_out->capacity());
                        sink->groups_out->push_back({ sink->query_index_per_reader[r][qi], llama_pos(best) });
                        // Stamp the group index onto spans recorded in
                        // this iteration (they were pushed before the group
                        // existed).
                        if (sink->spans_out) {
                            for (size_t si = span_begin; si < sink->spans_out->size(); ++si) {
                                (*sink->spans_out)[si].group_index = group_index;
                            }
                        }
                    } else {
                        layout.groups.push_back({ 0, llama_pos(best) });
                    }
                }
                if (emit_cursor != total) {
                    throw std::runtime_error("RERoT shared layout: visible count mismatch");
                }
                if (sink) {
                    sink->entry_cursor += emit_cursor;
                    sink->query_offsets->push_back(uint32_t(sink->entry_cursor));
                    sink->query_virtual_pos->push_back(query_virtual_pos);
                }
            }
        }
    }
    return result;
}

} // namespace

std::vector<std::vector<llama_rerot_query_layout>> llama_rerot_build_query_layouts_multi_reader_bits(
        const std::vector<llama_rerot_reader_state> & readers,
        const std::vector<std::vector<llama_pos>> & query_storage_pos,
        const std::vector<llama_rerot_key_record> & keys,
        const std::vector<llama_rerot_owned_view> & base_owned_bits) {
    if (readers.size() != query_storage_pos.size() || base_owned_bits.size() != readers.size()) {
        throw std::invalid_argument("RERoT multi-reader: reader/query/ownership counts differ");
    }
    // One-shot path: fresh structural pass (the pre-eleventh-round
    // behavior, byte-identical output). The world is a temporary; the
    // numeric pass reads the CALLER's key table directly, so the one-shot
    // path pays no record-copy beyond the run structure itself (the world
    // keeps its records copy, unused here).
    llama_rerot_shared_world world;
    world.build_world_structure(keys);
    return multi_reader_numeric_pass(readers, query_storage_pos, base_owned_bits,
                                     world.runs(), world.untagged_sorted(), keys, nullptr, nullptr);
}

std::vector<std::vector<llama_rerot_query_layout>> llama_rerot_build_query_layouts_multi_reader_world(
        const std::vector<llama_rerot_reader_state> & readers,
        const std::vector<std::vector<llama_pos>> & query_storage_pos,
        const llama_rerot_shared_world & world,
        const std::vector<llama_rerot_owned_view> & base_owned_bits,
        std::vector<std::vector<llama_rerot_query_layout>> * output_scratch,
        rerot_emission_sink * sink) {
    if (readers.size() != query_storage_pos.size() || base_owned_bits.size() != readers.size()) {
        throw std::invalid_argument("RERoT multi-reader: reader/query/ownership counts differ");
    }
    if ((output_scratch != nullptr) != (sink != nullptr)) {
        throw std::invalid_argument("RERoT multi-reader: scratch and sink must be provided together");
    }
    // Persistent-world path (eleventh round): the structural pass was paid
    // at the last structural event; this frontier only validates and runs
    // the reader+numeric pass.
    return multi_reader_numeric_pass(readers, query_storage_pos, base_owned_bits,
                                     world.runs(), world.untagged_sorted(), world.keys_ref(),
                                     output_scratch, sink);
}

// Byte-vector compatibility overload: packs each column into words and
// forwards to the bitset core (tenth round). The cache level passes
// owned_words directly; tests and any external byte callers keep working.
std::vector<std::vector<llama_rerot_query_layout>> llama_rerot_build_query_layouts_multi_reader(
        const std::vector<llama_rerot_reader_state> & readers,
        const std::vector<std::vector<llama_pos>> & query_storage_pos,
        const std::vector<llama_rerot_key_record> & keys,
        const std::vector<std::vector<uint8_t>> & base_owned) {
    if (readers.size() != query_storage_pos.size() || base_owned.size() != readers.size()) {
        throw std::invalid_argument("RERoT multi-reader: reader/query/ownership counts differ");
    }
    const size_t n_words = (keys.size() + 63) / 64 + (keys.empty() ? 1 : 0);
    std::vector<std::vector<uint64_t>> words(readers.size());
    std::vector<llama_rerot_owned_view> views(readers.size());
    for (size_t r = 0; r < readers.size(); ++r) {
        if (base_owned[r].size() != keys.size()) {
            throw std::invalid_argument("RERoT multi-reader: ownership column width mismatch");
        }
        auto & w = words[r];
        w.assign(n_words, 0);
        for (size_t k = 0; k < keys.size(); ++k) {
            if (base_owned[r][k]) {
                w[k >> 6] |= 1ull << (k & 63);
            }
        }
        views[r] = { w.data(), w.size() };
    }
    return llama_rerot_build_query_layouts_multi_reader_bits(
        readers, query_storage_pos, keys, views);
}

llama_rerot_query_layout llama_rerot_build_query_layout(
        const llama_rerot_reader_state & reader,
        llama_pos query_storage_pos,
        const std::vector<llama_rerot_key_record> & keys) {
    if (!reader.active()) {
        throw std::invalid_argument("RERoT reader state is inactive");
    }
    if (reader.reader == LLAMA_REROT_NODE_INVALID || reader.query_run == LLAMA_REROT_RUN_INVALID) {
        throw std::invalid_argument("RERoT reader or query run is invalid");
    }
    if (query_storage_pos < 0) {
        throw std::invalid_argument("RERoT query storage position must be non-negative");
    }

    std::unordered_map<llama_rerot_run_id, uint32_t> run_rank;
    run_rank.reserve(reader.ordered_runs.size());
    for (uint32_t i = 0; i < reader.ordered_runs.size(); ++i) {
        if (reader.ordered_runs[i] == LLAMA_REROT_RUN_INVALID ||
            !run_rank.emplace(reader.ordered_runs[i], i).second) {
            throw std::invalid_argument("RERoT reader view contains an invalid or duplicate run id");
        }
    }
    if (run_rank.find(reader.query_run) == run_rank.end()) {
        throw std::invalid_argument("RERoT query run is absent from the reader view");
    }

    struct visible_key {
        const llama_rerot_key_record * key = nullptr;
        uint32_t rank = 0;
        bool base = false;
        llama_pos virtual_pos = 0;
    };

    std::vector<visible_key> base;
    std::vector<visible_key> tagged;
    base.reserve(keys.size());
    tagged.reserve(keys.size());

    // Duplicate-physical-key check as a byte bitmap (same swap the shared and
    // multi-reader builders made in earlier rounds: the unordered_set cost
    // dominates the scan at production K). Semantics unchanged — first
    // duplicate in scan order throws, same message.
    std::vector<uint8_t> physical_seen;

    for (const auto & key : keys) {
        if (key.key_index >= physical_seen.size()) {
            physical_seen.resize(size_t(key.key_index) + 1, 0);
        }
        if (physical_seen[key.key_index]++) {
            throw std::invalid_argument("RERoT key records contain a duplicate physical key");
        }
        if (key.storage_pos < 0) {
            throw std::invalid_argument("RERoT key storage position must be non-negative");
        }

        const auto & meta = key.meta;
        if (!meta.active()) {
            // Ordinary prefix/private history is governed by stock sequence
            // ownership and causal position. RERoT-written cells are always
            // tagged, so this does not accidentally expose foreign lanes.
            // Base occlusion (become-leak fix) mirrors the batched builders.
            if (key.owned_by_reader && key.storage_pos <= query_storage_pos &&
                !reader.occludes(key.storage_pos)) {
                base.push_back({ &key, 0, true, 0 });
            }
            continue;
        }

        if (meta.episode_id != reader.episode_id) {
            continue;
        }
        const auto rank_it = run_rank.find(meta.run_id);
        if (rank_it == run_rank.end()) {
            continue;
        }

        bool visible = false;
        switch (meta.visibility) {
            case llama_rerot_visibility::public_live:
                if (meta.node_id == reader.reader) {
                    // A Lane's own causal history is never delayed by a peer
                    // frontier policy. This includes its current decode row.
                    visible = meta.frontier <= reader.frontier && key.owned_by_reader &&
                              key.storage_pos <= query_storage_pos;
                } else if (reader.frontier_mode == LLAMA_REROT_FRONTIER_STRONG) {
                    // Strong means immediately after the frontier barrier,
                    // not same-forward penetration.
                    visible = meta.frontier < reader.frontier;
                } else if (reader.frontier > 0) {
                    // Lag1 is one additional committed-frontier delay.
                    visible = meta.frontier < reader.frontier - 1;
                }
                break;
            case llama_rerot_visibility::private_control:
            case llama_rerot_visibility::pending_record:
                visible = meta.node_id == reader.reader && key.owned_by_reader &&
                          key.storage_pos <= query_storage_pos;
                break;
            case llama_rerot_visibility::normal:
                break;
        }

        if (visible) {
            tagged.push_back({ &key, rank_it->second, false, 0 });
        }
    }

    std::stable_sort(base.begin(), base.end(), [](const visible_key & lhs, const visible_key & rhs) {
        if (lhs.key->storage_pos != rhs.key->storage_pos) {
            return lhs.key->storage_pos < rhs.key->storage_pos;
        }
        return lhs.key->key_index < rhs.key->key_index;
    });
    std::stable_sort(tagged.begin(), tagged.end(), [](const visible_key & lhs, const visible_key & rhs) {
        if (lhs.rank != rhs.rank) {
            return lhs.rank < rhs.rank;
        }
        if (lhs.key->storage_pos != rhs.key->storage_pos) {
            return lhs.key->storage_pos < rhs.key->storage_pos;
        }
        if (lhs.key->meta.frontier != rhs.key->meta.frontier) {
            return lhs.key->meta.frontier < rhs.key->meta.frontier;
        }
        return lhs.key->key_index < rhs.key->key_index;
    });

    // The entire visible text memory is densely virtualized. Untagged serial
    // prefix keys therefore receive positions [0, base.size()), while their K
    // remains at the original storage phase; the effective Q position below
    // compensates for that difference exactly.
    llama_pos virtual_pos = 0;
    for (auto & key : base) {
        key.virtual_pos = virtual_pos++;
    }
    for (auto & key : tagged) {
        key.virtual_pos = virtual_pos++;
    }

    llama_pos query_virtual_pos = virtual_pos;
    bool query_found = false;
    for (const auto & key : tagged) {
        const auto & meta = key.key->meta;
        if (meta.node_id == reader.reader && meta.run_id == reader.query_run &&
            key.key->owned_by_reader && key.key->storage_pos == query_storage_pos) {
            query_virtual_pos = key.virtual_pos;
            query_found = true;
        }
    }

    // A no-cache/read-only refresh has no matching current K. Its query is the
    // next logical position after the visible document.
    if (!query_found && query_virtual_pos > std::numeric_limits<llama_pos>::max()) {
        throw std::overflow_error("RERoT query virtual position overflow");
    }

    struct grouped_entries {
        llama_pos effective_pos = 0;
        std::vector<uint32_t> key_indices;
    };
    std::map<llama_pos, grouped_entries> grouped;

    const auto add_key = [&](const visible_key & key) {
        // <R(q_eff)Q, R(k_storage)K> must have the same relative phase as
        // <R(q_virtual)Q, R(k_virtual)K>.
        const int64_t effective = int64_t(query_virtual_pos) + int64_t(key.key->storage_pos) -
                                  int64_t(key.virtual_pos);
        if (effective < 0 || effective > std::numeric_limits<llama_pos>::max()) {
            throw std::overflow_error("RERoT effective query position is outside llama_pos range");
        }
        auto & bucket = grouped[static_cast<llama_pos>(effective)];
        bucket.effective_pos = static_cast<llama_pos>(effective);
        bucket.key_indices.push_back(key.key->key_index);
    };
    for (const auto & key : base) {
        add_key(key);
    }
    for (const auto & key : tagged) {
        add_key(key);
    }

    llama_rerot_query_layout result;
    result.query_virtual_pos = query_virtual_pos;
    result.groups.reserve(grouped.size());
    size_t entry_count = 0;
    for (const auto & item : grouped) {
        entry_count += item.second.key_indices.size();
    }
    result.entries.reserve(entry_count);

    for (const auto & item : grouped) {
        const uint32_t group_index = static_cast<uint32_t>(result.groups.size());
        result.groups.push_back({ 0, item.second.effective_pos });
        for (const uint32_t key_index : item.second.key_indices) {
            result.entries.push_back({ key_index, group_index });
        }
    }

    return result;
}

llama_rerot_kv_snapshot llama_rerot_kv_snapshot::from_keys(
        const std::vector<llama_rerot_key_record> & keys,
        uint64_t episode_id) {
    llama_rerot_kv_snapshot snapshot;
    snapshot.episode_id = episode_id;

    std::unordered_set<uint32_t> physical_seen;
    physical_seen.reserve(keys.size());

    for (const auto & key : keys) {
        if (!physical_seen.insert(key.key_index).second) {
            throw std::invalid_argument("RERoT key records contain a duplicate physical key");
        }
        if (key.storage_pos < 0) {
            throw std::invalid_argument("RERoT key storage position must be non-negative");
        }

        const auto & meta = key.meta;
        if (!meta.active()) {
            if (key.owned_by_reader) {
                snapshot.base_cells.push_back({ key.key_index, key.storage_pos, 1ULL });
            }
            continue;
        }

        if (snapshot.episode_id == 0) {
            snapshot.episode_id = meta.episode_id;
        } else if (meta.episode_id != snapshot.episode_id) {
            continue;
        }

        snapshot.run_buckets[meta.run_id].push_back({
            key.key_index,
            key.storage_pos,
            meta.frontier,
            meta.node_id,
            meta.visibility,
            key.owned_by_reader ? 1ULL : 0ULL
        });
    }

    std::stable_sort(snapshot.base_cells.begin(), snapshot.base_cells.end(),
        [](const llama_rerot_snapshot_base_cell & a, const llama_rerot_snapshot_base_cell & b) {
            if (a.storage_pos != b.storage_pos) {
                return a.storage_pos < b.storage_pos;
            }
            return a.key_index < b.key_index;
        });

    for (auto & pair : snapshot.run_buckets) {
        std::stable_sort(pair.second.begin(), pair.second.end(),
            [](const llama_rerot_snapshot_cell & a, const llama_rerot_snapshot_cell & b) {
                if (a.storage_pos != b.storage_pos) {
                    return a.storage_pos < b.storage_pos;
                }
                if (a.frontier != b.frontier) {
                    return a.frontier < b.frontier;
                }
                return a.key_index < b.key_index;
            });
    }

    return snapshot;
}

llama_rerot_kv_snapshot llama_rerot_kv_snapshot::from_multi_reader_keys(
        const std::vector<llama_rerot_key_record> & keys,
        const std::vector<llama_rerot_reader_state> & readers,
        const std::vector<std::vector<bool>> & reader_key_ownership) {
    llama_rerot_kv_snapshot snapshot;
    if (readers.empty()) {
        return snapshot;
    }
    snapshot.episode_id = readers[0].episode_id;
    for (uint32_t i = 0; i < readers.size(); ++i) {
        if (readers[i].seq_id >= 0) {
            snapshot.seq_to_index[readers[i].seq_id] = i;
        }
        if (readers[i].reader != LLAMA_REROT_NODE_INVALID) {
            snapshot.node_to_index[readers[i].reader] = i;
        }
    }

    std::unordered_set<uint32_t> physical_seen;
    physical_seen.reserve(keys.size());

    for (size_t k = 0; k < keys.size(); ++k) {
        const auto & key = keys[k];
        if (!physical_seen.insert(key.key_index).second) {
            throw std::invalid_argument("RERoT key records contain a duplicate physical key");
        }
        if (key.storage_pos < 0) {
            throw std::invalid_argument("RERoT key storage position must be non-negative");
        }

        uint64_t mask = 0;
        for (uint32_t i = 0; i < readers.size(); ++i) {
            if (k < reader_key_ownership[i].size() && reader_key_ownership[i][k]) {
                mask |= (1ULL << i);
            }
        }

        const auto & meta = key.meta;
        if (!meta.active()) {
            if (mask != 0) {
                snapshot.base_cells.push_back({ key.key_index, key.storage_pos, mask });
            }
            continue;
        }

        if (meta.episode_id != snapshot.episode_id) {
            continue;
        }

        snapshot.run_buckets[meta.run_id].push_back({
            key.key_index,
            key.storage_pos,
            meta.frontier,
            meta.node_id,
            meta.visibility,
            mask
        });
    }

    std::stable_sort(snapshot.base_cells.begin(), snapshot.base_cells.end(),
        [](const llama_rerot_snapshot_base_cell & a, const llama_rerot_snapshot_base_cell & b) {
            if (a.storage_pos != b.storage_pos) {
                return a.storage_pos < b.storage_pos;
            }
            return a.key_index < b.key_index;
        });

    for (auto & pair : snapshot.run_buckets) {
        std::stable_sort(pair.second.begin(), pair.second.end(),
            [](const llama_rerot_snapshot_cell & a, const llama_rerot_snapshot_cell & b) {
                if (a.storage_pos != b.storage_pos) {
                    return a.storage_pos < b.storage_pos;
                }
                if (a.frontier != b.frontier) {
                    return a.frontier < b.frontier;
                }
                return a.key_index < b.key_index;
            });
    }

    return snapshot;
}

llama_rerot_query_layout llama_rerot_build_query_layout(
        const llama_rerot_reader_state & reader,
        llama_pos query_storage_pos,
        const llama_rerot_kv_snapshot & snapshot) {
    if (!reader.active()) {
        throw std::invalid_argument("RERoT reader state is inactive");
    }
    if (reader.reader == LLAMA_REROT_NODE_INVALID || reader.query_run == LLAMA_REROT_RUN_INVALID) {
        throw std::invalid_argument("RERoT reader or query run is invalid");
    }
    if (query_storage_pos < 0) {
        throw std::invalid_argument("RERoT query storage position must be non-negative");
    }

    std::unordered_map<llama_rerot_run_id, uint32_t> run_rank;
    run_rank.reserve(reader.ordered_runs.size());
    for (uint32_t i = 0; i < reader.ordered_runs.size(); ++i) {
        if (reader.ordered_runs[i] == LLAMA_REROT_RUN_INVALID ||
            !run_rank.emplace(reader.ordered_runs[i], i).second) {
            throw std::invalid_argument("RERoT reader view contains an invalid or duplicate run id");
        }
    }
    if (run_rank.find(reader.query_run) == run_rank.end()) {
        throw std::invalid_argument("RERoT query run is absent from the reader view");
    }

    struct visible_entry {
        uint32_t key_index = 0;
        llama_pos storage_pos = 0;
        llama_pos virtual_pos = 0;
    };

    std::vector<visible_entry> base;
    base.reserve(snapshot.base_cells.size());
    for (const auto & b : snapshot.base_cells) {
        if (snapshot.is_owned(b.seq_mask, reader) && b.storage_pos <= query_storage_pos) {
            base.push_back({ b.key_index, b.storage_pos, 0 });
        }
    }

    struct tagged_entry {
        uint32_t key_index = 0;
        llama_pos storage_pos = 0;
        llama_rerot_node_id node_id = LLAMA_REROT_NODE_INVALID;
        llama_rerot_run_id run_id = LLAMA_REROT_RUN_INVALID;
        bool owned = false;
        llama_pos virtual_pos = 0;
    };

    std::vector<tagged_entry> tagged;
    if (snapshot.episode_id == 0 || snapshot.episode_id == reader.episode_id) {
        for (const auto run_id : reader.ordered_runs) {
            const auto it = snapshot.run_buckets.find(run_id);
            if (it == snapshot.run_buckets.end()) {
                continue;
            }
            ++snapshot.bucket_lookup_count;
            for (const auto & cell : it->second) {
                const bool owned = snapshot.is_owned(cell.seq_mask, reader);
                bool visible = false;
                switch (cell.visibility) {
                    case llama_rerot_visibility::public_live:
                        if (cell.node_id == reader.reader) {
                            visible = cell.frontier <= reader.frontier && owned &&
                                      cell.storage_pos <= query_storage_pos;
                        } else if (reader.frontier_mode == LLAMA_REROT_FRONTIER_STRONG) {
                            visible = cell.frontier < reader.frontier;
                        } else if (reader.frontier > 0) {
                            visible = cell.frontier < reader.frontier - 1;
                        }
                        break;
                    case llama_rerot_visibility::private_control:
                    case llama_rerot_visibility::pending_record:
                        visible = cell.node_id == reader.reader && owned &&
                                  cell.storage_pos <= query_storage_pos;
                        break;
                    case llama_rerot_visibility::normal:
                        break;
                }

                if (visible) {
                    tagged.push_back({ cell.key_index, cell.storage_pos, cell.node_id, run_id, owned, 0 });
                }
            }
        }
    }

    llama_pos virtual_pos = 0;
    for (auto & item : base) {
        item.virtual_pos = virtual_pos++;
    }
    for (auto & item : tagged) {
        item.virtual_pos = virtual_pos++;
    }

    llama_pos query_virtual_pos = virtual_pos;
    bool query_found = false;
    for (const auto & item : tagged) {
        if (item.node_id == reader.reader && item.run_id == reader.query_run &&
            item.owned && item.storage_pos == query_storage_pos) {
            query_virtual_pos = item.virtual_pos;
            query_found = true;
        }
    }

    if (!query_found && query_virtual_pos > std::numeric_limits<llama_pos>::max()) {
        throw std::overflow_error("RERoT query virtual position overflow");
    }

    struct grouped_entries {
        llama_pos effective_pos = 0;
        std::vector<uint32_t> key_indices;
    };
    std::map<llama_pos, grouped_entries> grouped;

    const auto add_key = [&](uint32_t key_index, llama_pos storage_pos, llama_pos virt_pos) {
        const int64_t effective = int64_t(query_virtual_pos) + int64_t(storage_pos) - int64_t(virt_pos);
        if (effective < 0 || effective > std::numeric_limits<llama_pos>::max()) {
            throw std::overflow_error("RERoT effective query position is outside llama_pos range");
        }
        auto & bucket = grouped[static_cast<llama_pos>(effective)];
        bucket.effective_pos = static_cast<llama_pos>(effective);
        bucket.key_indices.push_back(key_index);
    };

    for (const auto & item : base) {
        add_key(item.key_index, item.storage_pos, item.virtual_pos);
    }
    for (const auto & item : tagged) {
        add_key(item.key_index, item.storage_pos, item.virtual_pos);
    }

    llama_rerot_query_layout result;
    result.query_virtual_pos = query_virtual_pos;
    result.groups.reserve(grouped.size());
    size_t entry_count = 0;
    for (const auto & item : grouped) {
        entry_count += item.second.key_indices.size();
    }
    result.entries.reserve(entry_count);

    for (const auto & item : grouped) {
        const uint32_t group_index = static_cast<uint32_t>(result.groups.size());
        result.groups.push_back({ 0, item.second.effective_pos });
        for (const uint32_t key_index : item.second.key_indices) {
            result.entries.push_back({ key_index, group_index });
        }
    }

    ++snapshot.query_eval_count;
    return result;
}

llama_rerot_rope_pos llama_rerot_text_position(int64_t pos) {
    return { pos, pos, pos, 0 };
}

bool llama_rerot_rope_apply(
        float * vector,
        size_t vector_size,
        const llama_rerot_rope_pos & position,
        const llama_rerot_rope_config & config,
        std::string * error) {
    if (!vector) {
        return set_error(error, "vector pointer must not be null");
    }
    if (!validate_rope_config(vector_size, config, error)) {
        return false;
    }

    const uint32_t n_pairs = config.rotary_dim / 2;
    for (uint32_t pair = 0; pair < n_pairs; ++pair) {
        const uint32_t axis = rotary_pair_axis(config, pair);
        const double frequency = std::pow(config.theta, -2.0 * double(pair) / double(config.rotary_dim)) *
                                 config.freq_scale;
        const double angle = double(position[axis]) * frequency;
        const float cosine = float(std::cos(angle));
        const float sine = float(std::sin(angle));

        uint32_t first;
        uint32_t second;
        if (config.layout == llama_rerot_rope_layout::interleaved) {
            first = pair * 2;
            second = first + 1;
        } else {
            first = pair;
            second = pair + n_pairs;
        }

        const float x = vector[first];
        const float y = vector[second];
        vector[first] = x * cosine - y * sine;
        vector[second] = x * sine + y * cosine;
    }
    return true;
}

std::vector<float> llama_rerot_ddvr_attention_materialized(
        const std::vector<float> & raw_query,
        const std::vector<float> & raw_keys,
        const std::vector<float> & values,
        uint32_t value_dim,
        int64_t query_virtual_pos,
        const std::vector<llama_rerot_ddvr_span> & spans,
        const llama_rerot_rope_config & config,
        float scale) {
    validate_ddvr_problem(raw_query, raw_keys, values, value_dim, spans, config);
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(float(config.head_dim));
    }

    std::vector<float> query = raw_query;
    std::string error;
    if (!llama_rerot_rope_apply(query.data(), query.size(), llama_rerot_text_position(query_virtual_pos), config, &error)) {
        throw std::invalid_argument(error);
    }

    const uint32_t n_keys = raw_keys.size() / config.head_dim;
    std::vector<float> scores(n_keys, 0.0f);
    std::vector<float> key(config.head_dim);

    for (const auto & span : spans) {
        for (uint32_t local = 0; local < span.key_count; ++local) {
            const uint32_t key_index = span.key_begin + local;
            std::copy_n(raw_keys.data() + size_t(key_index) * config.head_dim, config.head_dim, key.data());
            const int64_t virtual_pos = span.virtual_pos0 + local;
            if (!llama_rerot_rope_apply(key.data(), key.size(), llama_rerot_text_position(virtual_pos), config, &error)) {
                throw std::invalid_argument(error);
            }
            scores[key_index] = scale * dot_product(query.data(), key.data(), config.head_dim);
        }
    }

    return softmax_weighted_values(scores, values, value_dim);
}

std::vector<float> llama_rerot_ddvr_attention_qside(
        const std::vector<float> & raw_query,
        const std::vector<float> & raw_keys,
        const std::vector<float> & values,
        uint32_t value_dim,
        int64_t query_virtual_pos,
        const std::vector<llama_rerot_ddvr_span> & spans,
        const llama_rerot_rope_config & config,
        float scale) {
    validate_ddvr_problem(raw_query, raw_keys, values, value_dim, spans, config);
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(float(config.head_dim));
    }

    const uint32_t n_keys = raw_keys.size() / config.head_dim;
    std::vector<float> scores(n_keys, 0.0f);
    std::vector<float> query(config.head_dim);
    std::vector<float> stored_key(config.head_dim);
    std::string error;

    for (const auto & span : spans) {
        query = raw_query;
        const int64_t query_storage_frame = query_virtual_pos + span.storage_pos0 - span.virtual_pos0;
        if (!llama_rerot_rope_apply(
                query.data(), query.size(), llama_rerot_text_position(query_storage_frame), config, &error)) {
            throw std::invalid_argument(error);
        }

        for (uint32_t local = 0; local < span.key_count; ++local) {
            const uint32_t key_index = span.key_begin + local;
            std::copy_n(raw_keys.data() + size_t(key_index) * config.head_dim, config.head_dim, stored_key.data());
            const int64_t storage_pos = span.storage_pos0 + local;
            if (!llama_rerot_rope_apply(
                    stored_key.data(), stored_key.size(), llama_rerot_text_position(storage_pos), config, &error)) {
                throw std::invalid_argument(error);
            }
            scores[key_index] = scale * dot_product(query.data(), stored_key.data(), config.head_dim);
        }
    }

    return softmax_weighted_values(scores, values, value_dim);
}

// ---------------------------------------------------------------------------
// FlashPrefill legal-fragment table helpers (CacheFragments; pure, no owner)
// ---------------------------------------------------------------------------

std::vector<llama_rerot_table_fragment> llama_rerot_split_table_fragments(
        const llama_rerot_table_member * members,
        size_t n,
        llama_pos virtual_pos0,
        uint32_t block_k) {
    if (block_k == 0) {
        throw std::invalid_argument("RERoT table block_k must be non-zero");
    }
    std::vector<llama_rerot_table_fragment> out;
    if (n == 0) {
        return out;
    }
    if (!members) {
        throw std::invalid_argument("RERoT table members must not be null");
    }
    if (virtual_pos0 < 0 || uint64_t(virtual_pos0) + uint64_t(n) > uint64_t(std::numeric_limits<llama_pos>::max()) + 1u) {
        throw std::overflow_error("RERoT table virtual positions exceed llama_pos range");
    }
    out.reserve(n);

    const auto cut_key = [&](size_t i, int64_t & phase_bias, uint64_t & vblock) {
        if (members[i].storage_pos < 0) {
            throw std::invalid_argument("RERoT table storage position must be non-negative");
        }
        const llama_pos virt = virtual_pos0 + static_cast<llama_pos>(i);
        phase_bias = int64_t(members[i].storage_pos) - int64_t(virt);
        vblock = uint64_t(virt) / uint64_t(block_k);
    };

    uint32_t frag_begin = 0;
    int64_t prev_bias = 0;
    uint64_t prev_block = 0;
    cut_key(0, prev_bias, prev_block);

    for (size_t i = 1; i < n; ++i) {
        int64_t bias = 0;
        uint64_t block = 0;
        cut_key(i, bias, block);
        const bool cut = block != prev_block || bias != prev_bias ||
                         members[i].visibility != members[i - 1].visibility ||
                         members[i].gated != members[i - 1].gated;
        if (cut) {
            out.push_back({ frag_begin, static_cast<uint32_t>(i),
                            virtual_pos0 + static_cast<llama_pos>(frag_begin),
                            prev_bias, members[i - 1].visibility, members[i - 1].gated });
            frag_begin = static_cast<uint32_t>(i);
            prev_bias = bias;
            prev_block = block;
        }
    }
    out.push_back({ frag_begin, static_cast<uint32_t>(n),
                    virtual_pos0 + static_cast<llama_pos>(frag_begin),
                    prev_bias, members[n - 1].visibility, members[n - 1].gated });
    return out;
}

bool llama_rerot_cell_visible_public_full(
        const llama_kv_rerot_meta & meta,
        const llama_rerot_reader_state & reader) {
    if (!meta.active() || meta.visibility != llama_rerot_visibility::public_live) {
        return false;
    }
    if (meta.episode_id != reader.episode_id) {
        return false;
    }
    // Strict barrier-after: only peer tokens from previously committed frontiers
    // are visible. A peer token from the current forward step is never visible
    // until after the frontier barrier has settled.
    if (meta.frontier < reader.frontier) {
        return true;
    }
    return false;
}

bool llama_rerot_cell_visible_gated(
        const llama_kv_rerot_meta & meta,
        const llama_rerot_reader_state & reader,
        bool is_untagged_base,
        bool owned_by_reader,
        llama_pos storage_pos,
        llama_pos query_storage_pos) {
    if (!owned_by_reader || storage_pos < 0 || query_storage_pos < 0) {
        return false;
    }
    if (storage_pos > query_storage_pos) {
        return false;
    }
    if (is_untagged_base) {
        return true;
    }
    if (!meta.active() || meta.episode_id != reader.episode_id) {
        return false;
    }
    switch (meta.visibility) {
        case llama_rerot_visibility::public_live:
            // Own-node frontier-equal arm only; older-frontier and strong
            // foreign-node cases are covered by the FULL predicate above.
            return meta.frontier == reader.frontier && meta.node_id == reader.reader;
        case llama_rerot_visibility::private_control:
        case llama_rerot_visibility::pending_record:
            return meta.node_id == reader.reader;
        case llama_rerot_visibility::normal:
            return false;
    }
    return false;
}

