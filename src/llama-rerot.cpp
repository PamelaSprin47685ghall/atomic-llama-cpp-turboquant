#include "llama-rerot.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
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
        uint64_t publish_epoch) {
    if (owner >= nodes_.size()) {
        throw std::out_of_range("RERoT run owner does not exist");
    }
    if (storage_pos0 < 0) {
        throw std::invalid_argument("RERoT storage positions must be non-negative");
    }

    llama_rerot_run run;
    run.id = static_cast<llama_rerot_run_id>(runs_.size());
    run.owner = owner;
    run.visibility = visibility;
    run.storage_pos0 = storage_pos0;
    run.token_count = token_count;
    run.publish_epoch = publish_epoch;

    nodes_[owner].runs.push_back(run.id);
    runs_.push_back(std::move(run));
    return static_cast<llama_rerot_run_id>(runs_.size() - 1);
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
        if (current.visibility == llama_rerot_visibility::pending_record && current.publish_epoch != 0) {
            return set_error(error, "pending run cannot have a publish epoch");
        }
    }

    for (const auto & current : nodes_) {
        if (!current.children.empty()) {
            continue;
        }
        const auto view = build_view(current.id);
        llama_pos expected = 0;
        std::unordered_set<llama_rerot_run_id> visible_runs;
        for (const auto & view_run : view.runs) {
            if (view_run.virtual_pos0 != expected || view_run.token_count == 0) {
                return set_error(error, "reader view is not densely packed");
            }
            if (!visible_runs.insert(view_run.run_id).second) {
                return set_error(error, "reader view contains a run more than once");
            }
            expected += static_cast<llama_pos>(view_run.token_count);
        }
        if (view.query_virtual_pos != expected) {
            return set_error(error, "reader query position does not follow the final visible token");
        }
    }

    return true;
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

