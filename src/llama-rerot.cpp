#include "llama-rerot.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <queue>
#include <unordered_map>
#include <unordered_set>

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

bool llama_rerot_attn_layout::validate(uint32_t n_keys, std::string * error) const {
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
    std::vector<uint8_t> seen(n_keys, 0); // per-query duplicate-key bitmap
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
                if (key.owned_by_reader) {
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

        // K-way merge over deviation-sorted lists.
        llama_rerot_query_layout layout;
        layout.query_virtual_pos = query_virtual_pos;
        layout.entries.reserve(size_t(total));
        for (auto & p : merge_pos) {
            p = 0;
        }
        const int64_t qv = int64_t(query_virtual_pos);
        while (true) {
            // Advance the base cursor past cut rows (the cut selects a
            // subset of the d-sorted full array).
            while (merge_pos[0] < base_order.size() && base_order[merge_pos[0]] >= base_cut) {
                ++merge_pos[0];
            }
            int64_t best = std::numeric_limits<int64_t>::max();
            bool any = false;
            if (merge_pos[0] < base_order.size()) {
                best = qv + base_dev[merge_pos[0]] - int64_t(vis_before[0]);
                any = true;
            }
            for (size_t s = 0; s < segments.size(); ++s) {
                // Advance past d-positions whose tagged index failed the
                // causal cut (the cut is a tagged-order prefix; in d-order
                // the surviving rows are an arbitrary subset).
                while (merge_pos[s + 1] < segments[s].rows_d.size() &&
                       (segments[s].causal
                            ? segments[s].d2t[merge_pos[s + 1]] >= cut_of[s]
                            : false)) {
                    ++merge_pos[s + 1];
                }
                if (merge_pos[s + 1] >= segments[s].rows_d.size()) {
                    continue;
                }
                const int64_t v = qv + segments[s].dev[merge_pos[s + 1]] - int64_t(vis_before[s + 1]);
                if (!any || v < best) {
                    best = v;
                    any = true;
                }
            }
            if (!any) {
                break;
            }
            // Emit one group: every list head whose effective == best.
            const uint32_t group_index = uint32_t(layout.groups.size());
            bool emitted = false;
            while (merge_pos[0] < base_order.size() && base_order[merge_pos[0]] < base_cut &&
                   qv + base_dev[merge_pos[0]] - int64_t(vis_before[0]) == best) {
                const uint32_t emi = base_order[merge_pos[0]];
                layout.entries.push_back({ base[emi]->key_index, group_index });
                ++merge_pos[0];
                emitted = true;
            }
            for (size_t s = 0; s < segments.size(); ++s) {
                const auto & seg = segments[s];
                while (merge_pos[s + 1] < seg.rows_d.size() &&
                       (!seg.causal || seg.d2t[merge_pos[s + 1]] < cut_of[s]) &&
                       qv + seg.dev[merge_pos[s + 1]] - int64_t(vis_before[s + 1]) == best) {
                    layout.entries.push_back({ seg.rows_d[merge_pos[s + 1]]->key_index, group_index });
                    ++merge_pos[s + 1];
                    emitted = true;
                }
            }
            if (!emitted) {
                // Defensive: each list is sorted by effective, so a head
                // equal to the minimum must exist whenever best was taken
                // from a list head.
                throw std::runtime_error("RERoT shared layout: k-way merge lost a group head");
            }
            if (best < 0 || best > std::numeric_limits<llama_pos>::max()) {
                throw std::overflow_error("RERoT effective query position is outside llama_pos range");
            }
            layout.groups.push_back({ 0, llama_pos(best) });
        }
        result.push_back(std::move(layout));
    }
    return result;
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
std::vector<std::vector<llama_rerot_query_layout>> llama_rerot_build_query_layouts_multi_reader_bits(
        const std::vector<llama_rerot_reader_state> & readers,
        const std::vector<std::vector<llama_pos>> & query_storage_pos,
        const std::vector<llama_rerot_key_record> & keys,
        const std::vector<llama_rerot_owned_view> & base_owned_bits) {
    if (readers.size() != query_storage_pos.size() || base_owned_bits.size() != readers.size()) {
        throw std::invalid_argument("RERoT multi-reader: reader/query/ownership counts differ");
    }
    for (size_t r = 0; r < readers.size(); ++r) {
        const auto & reader = readers[r];
        if (!reader.active()) {
            throw std::invalid_argument("RERoT reader state is inactive");
        }
        if (reader.reader == LLAMA_REROT_NODE_INVALID || reader.query_run == LLAMA_REROT_RUN_INVALID) {
            throw std::invalid_argument("RERoT reader or query run is invalid");
        }
        const size_t need_words = (keys.size() + 63) / 64 + (keys.empty() ? 1 : 0);
        if (base_owned_bits[r].n_words < need_words) {
            throw std::invalid_argument("RERoT multi-reader: ownership column width mismatch");
        }
        for (const llama_pos pos : query_storage_pos[r]) {
            if (pos < 0) {
                throw std::invalid_argument("RERoT query storage position must be non-negative");
            }
        }
    }

    // ---- ONE structural pass over the shared key world. ----
    struct shared_run {
        uint64_t episode_id = 0;
        llama_rerot_run_id run_id = LLAMA_REROT_RUN_INVALID;
        // Owner node: rows are bucketed by (episode, run, node) so a run id
        // shared by two nodes (possible in the cell lifetime: a run id is
        // re-assigned per node after retirement) never merges owners. The
        // single-reader builder filters per row by node_id; this bucket key
        // reproduces that grouping exactly.
        llama_rerot_node_id owner_node = LLAMA_REROT_NODE_INVALID;
        std::vector<uint32_t> rows;      // key indices, tagged order
        std::vector<llama_pos> storage;  // ascending (== tagged order)
        std::vector<uint32_t> d2t;       // d-position -> tagged index
        std::vector<uint32_t> t2d;       // tagged index -> d-position
        std::vector<int64_t> dev;        // d = storage - tagged idx, at d-positions
        bool contiguous = false;       // storage strictly +1 (dev constant)
        // Uniform (visibility, frontier) over all rows — computed ONCE in
        // the structural pass (ninth round). The per-reader seg construction
        // previously re-probed every row's meta per reader (R x n reads);
        // for uniform buckets the probe result is a run property.
        bool uniform = false;
        llama_rerot_visibility u_vis = llama_rerot_visibility::normal;
        uint64_t u_frontier = 0;
        // Contiguous-run fast-path key column (eighth round, moved to the
        // shared run in the ninth): key ids in d-order. Reader-independent
        // content (keys[rows[p]].key_index), previously rebuilt per reader.
        std::vector<uint32_t> fast_keys;
    };
    std::vector<shared_run> runs;
    std::vector<uint32_t> untagged_sorted;
    {
        std::vector<uint8_t> physical_seen; // same dedup contract as above
        std::vector<uint32_t> untagged;
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
            const auto & meta = key.meta;
            if (!meta.active()) {
                untagged.push_back(uint32_t(ki));
                continue;
            }
            shared_run * run = nullptr;
            for (auto & cand : runs) {
                if (cand.episode_id == meta.episode_id && cand.run_id == meta.run_id &&
                    cand.owner_node == meta.node_id) {
                    run = &cand;
                    break;
                }
            }
            if (!run) {
                runs.push_back(shared_run{});
                run = &runs.back();
                run->episode_id = meta.episode_id;
                run->run_id = meta.run_id;
                run->owner_node = meta.node_id;
            }
            run->rows.push_back(uint32_t(ki));
        }
        for (auto & run : runs) {
            const size_t n = run.rows.size();
            // Production fast path (sixth round): append-only runs arrive
            // in write order, which IS the tagged (storage, frontier, idx)
            // order — an O(n) sortedness probe skips the sort. Contiguous
            // storage (s_i = s_0 + i) makes the deviation d = s_0 constant,
            // so the deviation order is the identity and both the deviation
            // sort and the permutation tables collapse to O(n) fills.
            // Runs with holes or reordering (reclaimed cells, MTP verify
            // duplicates) still take the general sort path below.
            bool tagged_sorted = true;
            for (size_t i = 1; i < n; ++i) {
                const auto & a = keys[run.rows[i - 1]];
                const auto & b = keys[run.rows[i]];
                if (a.storage_pos != b.storage_pos ? a.storage_pos > b.storage_pos
                                                    : a.meta.frontier > b.meta.frontier) {
                    tagged_sorted = false;
                    break;
                }
            }
            if (!tagged_sorted) {
                std::sort(run.rows.begin(), run.rows.end(), [&](uint32_t a, uint32_t b) {
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
            // tagged_sorted: rows arrive key_index-ascending (they were
            // pushed in keys order), so within (storage, frontier) ties the
            // arrival order already matches the key_index tie-break — the
            // array IS in tagged order, no sort needed.
            run.storage.resize(n);
            for (size_t i = 0; i < n; ++i) {
                run.storage[i] = keys[run.rows[i]].storage_pos;
            }
            bool contiguous = true;
            for (size_t i = 1; i < n; ++i) {
                if (run.storage[i] != run.storage[i - 1] + 1) {
                    contiguous = false;
                    break;
                }
            }
            if (contiguous) {
                run.contiguous = true;
                // d = storage[0] at every position: identity deviation order.
                run.d2t.resize(n);
                run.t2d.resize(n);
                run.dev.resize(n);
                for (uint32_t k = 0; k < n; ++k) {
                    run.d2t[k] = k;
                    run.t2d[k] = k;
                    run.dev[k] = int64_t(run.storage[k]) - int64_t(k);
                }
            } else {
                std::vector<uint32_t> order(n);
                for (uint32_t k = 0; k < n; ++k) {
                    order[k] = k;
                }
                std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                    const int64_t da = int64_t(run.storage[a]) - int64_t(a);
                    const int64_t db = int64_t(run.storage[b]) - int64_t(b);
                    if (da != db) {
                        return da < db;
                    }
                    return a < b;
                });
                run.d2t = std::move(order);
                run.t2d.assign(n, 0);
                run.dev.resize(n);
                for (size_t p = 0; p < n; ++p) {
                    const uint32_t t = run.d2t[p];
                    run.dev[p] = int64_t(run.storage[t]) - int64_t(t);
                    run.t2d[t] = uint32_t(p);
                }
            }
        }
        // Shared per-run properties (ninth round): uniformity of
        // (visibility, frontier) over the bucket's rows, and the
        // contiguous-run fast key column — both reader-independent, both
        // previously recomputed per reader.
        for (auto & run : runs) {
            const size_t n = run.rows.size();
            const auto & m0 = keys[run.rows[0]].meta;
            run.uniform = true;
            for (size_t i = 1; i < n; ++i) {
                const auto & m = keys[run.rows[i]].meta;
                if (m.visibility != m0.visibility || m.frontier != m0.frontier) {
                    run.uniform = false;
                    break;
                }
            }
            if (run.uniform) {
                run.u_vis = m0.visibility;
                run.u_frontier = m0.frontier;
            }
            if (run.contiguous) {
                // Identity deviation order: fast_keys[p] = key id at
                // d-position p == tagged index p.
                run.fast_keys.resize(n);
                for (size_t p = 0; p < n; ++p) {
                    run.fast_keys[p] = keys[run.rows[p]].key_index;
                }
            }
        }
        untagged_sorted = std::move(untagged);
        std::sort(untagged_sorted.begin(), untagged_sorted.end(), [&](uint32_t a, uint32_t b) {
            if (keys[a].storage_pos != keys[b].storage_pos) {
                return keys[a].storage_pos < keys[b].storage_pos;
            }
            return keys[a].key_index < keys[b].key_index;
        });
    }

    std::vector<std::vector<llama_rerot_query_layout>> result(readers.size());
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
            const shared_run * run = nullptr;
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
        const auto dp_at = [](const seg_view & s, size_t i) -> uint32_t {
            return s.identity ? uint32_t(i) : s.dp[i];
        };
        const auto prefix_at = [](const seg_view & s, size_t i) -> uint32_t {
            return s.identity ? uint32_t(i) : s.pass_prefix[i];
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
                const auto & meta = keys[found.rows[i]].meta;
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
            if (owned(ki)) {
                base.push_back(ki);
            }
        }
        std::vector<llama_pos> base_storage(base.size());
        for (size_t i = 0; i < base.size(); ++i) {
            base_storage[i] = keys[base[i]].storage_pos;
        }
        std::vector<uint32_t> base_order(base.size());
        for (uint32_t k = 0; k < base_order.size(); ++k) {
            base_order[k] = k;
        }
        if (!base.empty()) {
            std::sort(base_order.begin(), base_order.end(), [&](uint32_t a, uint32_t b) {
                const int64_t da = int64_t(base_storage[a]) - int64_t(a);
                const int64_t db = int64_t(base_storage[b]) - int64_t(b);
                if (da != db) {
                    return da < db;
                }
                return a < b;
            });
        }
        std::vector<int64_t> base_dev(base.size());
        for (size_t k = 0; k < base_order.size(); ++k) {
            base_dev[k] = int64_t(base_storage[base_order[k]]) - int64_t(base_order[k]);
        }

        // ---- Per-reader numeric pass. ----
        // Merge lists: list 0 = base arm (gated, prefix cut over the
        // ascending storage array); then per segment, in rank order, the
        // FULL d-positions and the gated d-positions as two lists. List
        // order = the oracle's emission order (base, then rank order,
        // FULL before gated inside one segment — the tagged comparator
        // interleaves them by (storage, frontier, idx), but the oracle's
        // stable_sort groups by EFFECTIVE value, not by list; the merge
        // below only needs each list individually sorted by effective,
        // which holds: both are filtered subsequences of the shared
        // deviation order).
        struct list_ref {
            const shared_run * run = nullptr; // nullptr = base arm
            const std::vector<uint32_t> * dp = nullptr; // d-positions (or base_order)
            const std::vector<int64_t> * dev = nullptr;
            bool gated = false;
        };
        std::vector<list_ref> lists;
        lists.reserve(1 + segs.size());
        lists.push_back(list_ref{ nullptr, &base_order, &base_dev, true });
        for (const auto & sv : segs) {
            lists.push_back(list_ref{ sv.run, &sv.dp, &sv.run->dev, sv.own });
        }

        std::vector<uint64_t> vis_count(lists.size(), 0);
        std::vector<uint64_t> vis_before(lists.size(), 0);
        std::vector<size_t> merge_pos(lists.size(), 0);

        // Per-segment causal cuts (tagged-order prefix) reused per query.
        std::vector<size_t> seg_cut(segs.size(), 0);

        for (size_t qi = 0; qi < query_storage_pos[r].size(); ++qi) {
            const llama_pos q_pos = query_storage_pos[r][qi];

            // Causal cuts: base prefix + per-segment tagged prefix.
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
                            query_virtual_pos = llama_pos(vis_before[si + 1] +
                                uint64_t(prefix_at(sv, local)));
                        }
                        break;
                    }
                    ++si;
                }
            }
            if (query_virtual_pos > std::numeric_limits<llama_pos>::max()) {
                throw std::overflow_error("RERoT query virtual position overflow");
            }

            // K-way merge over the deviation-sorted lists (same shape as
            // the single-reader builder: effective = qv + d - B).
            llama_rerot_query_layout layout;
            layout.query_virtual_pos = query_virtual_pos;
            layout.entries.reserve(size_t(total));
            for (auto & p : merge_pos) {
                p = 0;
            }
            const int64_t qv = int64_t(query_virtual_pos);
            while (true) {
                // Base cursor: advance past cut rows (prefix cut over the
                // ascending array; base_order is d-order).
                while (merge_pos[0] < base_order.size() && base_order[merge_pos[0]] >= base_cut) {
                    ++merge_pos[0];
                }
                int64_t best = std::numeric_limits<int64_t>::max();
                bool any = false;
                if (merge_pos[0] < base_order.size()) {
                    best = qv + base_dev[merge_pos[0]] - int64_t(vis_before[0]);
                    any = true;
                }
                {
                    size_t li = 1;
                    size_t si = 0;
                    for (const auto & sv : segs) {
                        if (sv.own) {
                            // Advance past d-positions that failed the causal
                            // cut (the cut is a tagged-order prefix; in d-order
                            // the surviving passing rows are an arbitrary
                            // subset — d dips at duplicate storage).
                            while (merge_pos[li] < dp_size(sv) &&
                                   sv.run->d2t[dp_at(sv, merge_pos[li])] >= seg_cut[si]) {
                                ++merge_pos[li];
                            }
                        }
                        if (merge_pos[li] < dp_size(sv)) {
                            const uint32_t dp = dp_at(sv, merge_pos[li]);
                            const int64_t v = qv + sv.run->dev[dp] - int64_t(vis_before[li]);
                            if (!any || v < best) {
                                best = v;
                                any = true;
                            }
                        }
                        ++li;
                        ++si;
                    }
                }
                if (!any) {
                    break;
                }
                const uint32_t group_index = uint32_t(layout.groups.size());
                bool emitted = false;
                while (merge_pos[0] < base_order.size() && base_order[merge_pos[0]] < base_cut &&
                       qv + base_dev[merge_pos[0]] - int64_t(vis_before[0]) == best) {
                    const uint32_t emi = base_order[merge_pos[0]];
                    layout.entries.push_back({ keys[base[emi]].key_index, group_index });
                    ++merge_pos[0];
                    emitted = true;
                }
                {
                    size_t li = 1;
                    size_t si = 0;
                    for (const auto & sv : segs) {
                        if (!sv.run->fast_keys.empty()) {
                            // Contiguous identity segment: dev is constant, so
                            // the ==best recheck hoists out and the entry
                            // emission is one sequential read over fast_keys
                            // (replacing the dp_at -> d2t -> rows -> keys
                            // chain of dependent random reads).
                            const int64_t v =
                                qv + sv.run->dev[0] - int64_t(vis_before[li]);
                            if (v == best) {
                                size_t p = merge_pos[li];
                                if (sv.own) {
                                    // Causal cut in tagged order == d-order here
                                    // (identity permutation), so the cut is a
                                    // simple prefix bound.
                                    while (p < dp_size(sv) && p < seg_cut[si]) {
                                        layout.entries.push_back({ sv.run->fast_keys[p], group_index });
                                        ++p;
                                        emitted = true;
                                    }
                                } else {
                                    while (p < dp_size(sv)) {
                                        layout.entries.push_back({ sv.run->fast_keys[p], group_index });
                                        ++p;
                                        emitted = true;
                                    }
                                }
                                merge_pos[li] = uint32_t(p);
                            }
                            ++li;
                            ++si;
                            continue;
                        }
                        while (merge_pos[li] < dp_size(sv) &&
                               (!sv.own || sv.run->d2t[dp_at(sv, merge_pos[li])] < seg_cut[si]) &&
                               qv + sv.run->dev[dp_at(sv, merge_pos[li])] - int64_t(vis_before[li]) == best) {
                            const uint32_t dp = dp_at(sv, merge_pos[li]);
                            layout.entries.push_back({ keys[sv.run->rows[sv.run->d2t[dp]]].key_index, group_index });
                            ++merge_pos[li];
                            emitted = true;
                        }
                        ++li;
                        ++si;
                    }
                }
                if (!emitted) {
                    throw std::runtime_error("RERoT shared layout: k-way merge lost a group head");
                }
                if (best < 0 || best > std::numeric_limits<llama_pos>::max()) {
                    throw std::overflow_error("RERoT effective query position is outside llama_pos range");
                }
                layout.groups.push_back({ 0, llama_pos(best) });
            }
            result[r].push_back(std::move(layout));
        }
    }
    return result;
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
            // Ordinary prefix/private history is governed by stock sequence
            // ownership and causal position. RERoT-written cells are always
            // tagged, so this does not accidentally expose foreign lanes.
            if (key.owned_by_reader && key.storage_pos <= query_storage_pos) {
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

