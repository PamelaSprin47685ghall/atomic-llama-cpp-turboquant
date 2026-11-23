#include "server-compress.h"

#include "common.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace {

using server_compress_json::kind;
using server_compress_json::node;
using server_compress_json::range;

// Preserve the existing edge protection policy. This does not divide content
// into scoring blocks: every native content item is scored in its own order.
constexpr size_t protected_tokens = 32;
constexpr size_t no_unit = std::numeric_limits<size_t>::max();

bool is_null(const node * value) {
    return value && value->type == kind::primitive && value->text == "null";
}

const node & require(const node & parent, const char * key, kind type) {
    const node * value = parent.find(key);
    if (!value || value->type != type) {
        throw std::invalid_argument(std::string("compression requires ") + key + " with the expected JSON type");
    }
    return *value;
}

class message_sets {
public:
    explicit message_sets(size_t n) : parents(n) {
        std::iota(parents.begin(), parents.end(), size_t(0));
    }
    size_t find(size_t i) {
        while (parents[i] != i) {
            parents[i] = parents[parents[i]];
            i = parents[i];
        }
        return i;
    }
    void join(size_t a, size_t b) { parents[find(b)] = find(a); }
private:
    std::vector<size_t> parents;
};

} // namespace

server_compress_plan::server_compress_plan(std::string raw, const llama_vocab * vocab)
    : raw_(std::move(raw)), root_(server_compress_json::parse(raw_)) {
    if (root_.type != kind::object) {
        throw std::invalid_argument("compression requires a chat completions JSON object");
    }
    const auto & messages = require(root_, "messages", kind::array).children;
    if (messages.empty()) {
        throw std::invalid_argument("messages must not be empty");
    }
    marker_tokens_ = common_tokenize(vocab, "\xe2\x90\xa1", false, false).size();

    message_sets sets(messages.size());
    std::vector<bool> pinned(messages.size(), false);
    struct call { size_t message; bool answered = false; };
    std::unordered_map<std::string, call> calls;
    std::unordered_map<std::string, size_t> legacy_calls;
    size_t latest_user = messages.size();

    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & msg = messages[i];
        if (msg.type != kind::object) {
            throw std::invalid_argument("each message must be an object");
        }
        const std::string & role = require(msg, "role", kind::string).text;
        if (role == "system" || role == "developer") {
            pinned[i] = true;
            continue;
        }
        if (role != "user" && role != "assistant" && role != "tool" && role != "function") {
            throw std::invalid_argument("unsupported message role for compression");
        }
        if (role == "user") {
            latest_user = i;
        }
        const node * tool_calls = msg.find("tool_calls");
        const node * function_call = msg.find("function_call");
        if (role == "assistant" && i > 0 && messages[i - 1].find("role")->text == "user" &&
                (!tool_calls || is_null(tool_calls) || tool_calls->children.empty()) &&
                (!function_call || is_null(function_call))) {
            // Preserve ordinary question/answer pairs for alternating templates.
            // Tool exchanges remain independent of the user that started the loop.
            sets.join(i - 1, i);
        }
        if (tool_calls && !is_null(tool_calls)) {
            if (role != "assistant" || tool_calls->type != kind::array) {
                throw std::invalid_argument("tool_calls must be an assistant array");
            }
            for (const auto & tc : tool_calls->children) {
                const auto & id = require(tc, "id", kind::string).text;
                const auto & fn = require(tc, "function", kind::object);
                require(fn, "name", kind::string);
                require(fn, "arguments", kind::string);
                if (id.empty() || !calls.emplace(id, call{i, false}).second) {
                    throw std::invalid_argument("tool call IDs must be nonempty and unique");
                }
            }
        }
        if (role == "tool") {
            const auto & id = require(msg, "tool_call_id", kind::string).text;
            auto it = calls.find(id);
            if (it == calls.end() || it->second.answered) {
                throw std::invalid_argument("tool result must match one preceding tool call");
            }
            it->second.answered = true;
            sets.join(it->second.message, i);
        }
        if (const node * fc = function_call; fc && !is_null(fc)) {
            if (role != "assistant" || fc->type != kind::object) {
                throw std::invalid_argument("function_call must be an assistant object");
            }
            const auto & name = require(*fc, "name", kind::string).text;
            require(*fc, "arguments", kind::string);
            if (!legacy_calls.emplace(name, i).second) {
                throw std::invalid_argument("ambiguous pending legacy function call");
            }
        }
        if (role == "function") {
            const auto & name = require(msg, "name", kind::string).text;
            auto it = legacy_calls.find(name);
            if (it == legacy_calls.end()) {
                throw std::invalid_argument("function result must match a preceding function call");
            }
            sets.join(it->second, i);
            legacy_calls.erase(it);
        }
    }

    pinned.back() = true;
    if (latest_user != messages.size()) {
        pinned[latest_user] = true;
    }
    std::unordered_map<size_t, size_t> group_ids;
    for (size_t i = 0; i < messages.size(); ++i) {
        const size_t parent = sets.find(i);
        auto it = group_ids.emplace(parent, groups_.size()).first;
        if (it->second == groups_.size()) {
            groups_.push_back({});
        }
        message_groups_.push_back(it->second);
        auto & group = groups_[it->second];
        group.messages.push_back(i);
        group.pinned = group.pinned || pinned[i];
        const auto source = messages[i].source;
        // Tool names, IDs, and argument syntax also occupy prompt space. The
        // source JSON is a cheap estimate; the route checks the real template.
        group.estimated_tokens += common_tokenize(vocab,
            raw_.substr(source.begin, source.end - source.begin), false, false).size();
    }

    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & msg = messages[i];
        const auto & role = msg.find("role")->text;
        const bool instructions = role == "system" || role == "developer";
        // JSON key order need not match generation order. Reasoning precedes
        // visible content, which precedes tool calls in an assistant message.
        if (role == "assistant") {
            const node * reasoning = msg.find("reasoning_content");
            if (reasoning && reasoning->type == kind::string) {
                add_text(*reasoning, i, vocab, false);
            }
        }
        if (const node * content = msg.find("content")) {
            if (content->type == kind::string) {
                add_text(*content, i, vocab, instructions);
            } else if (content->type == kind::array) {
                for (const auto & part : content->children) {
                    const node * type = part.find("type");
                    const node * value = part.find("text");
                    if (type && type->type == kind::string && type->text == "text" &&
                        value && value->type == kind::string) {
                        add_text(*value, i, vocab, instructions);
                    }
                }
            }
        }
        if (instructions) {
            continue;
        }
        if (const node * tool_calls = msg.find("tool_calls"); tool_calls && !is_null(tool_calls)) {
            for (const auto & tc : tool_calls->children) {
                add_function(*tc.find("function"), i, vocab);
            }
        }
        if (const node * fc = msg.find("function_call"); fc && !is_null(fc)) {
            add_function(*fc, i, vocab);
        }
    }
}

void server_compress_plan::add_text(const node & value, size_t message,
        const llama_vocab * vocab, bool pinned) {
    region mapped;
    mapped.reserve(value.units.size());
    for (const auto & unit : value.units) {
        mapped.push_back({{unit.text_begin, unit.text_end}, unit.source});
    }
    add_content(value.text, {std::move(mapped)}, message, vocab, pinned);
}

void server_compress_plan::add_function(const node & fn, size_t message, const llama_vocab * vocab) {
    const auto & arguments = require(fn, "arguments", kind::string);
    const std::string prefix = require(fn, "name", kind::string).text + "\n";
    std::vector<region> regions;
    node inner;
    bool parsed = true;
    try {
        inner = server_compress_json::parse(arguments.text);
    } catch (const std::invalid_argument &) {
        // Unfinished or opaque arguments still provide scoring context. They
        // cannot be edited internally because their value boundaries are unknown.
        parsed = false;
    }
    if (parsed) {
        std::function<void(const node &)> visit = [&](const node & value) {
            if (value.type == kind::string) {
                region mapped;
                mapped.reserve(value.units.size());
                for (const auto & unit : value.units) {
                    // Score the complete argument JSON, including keys and
                    // syntax. Only complete string value units can be removed.
                    const range source = server_compress_json::map_string_range(
                        arguments, unit.source.begin, unit.source.end);
                    mapped.push_back({{prefix.size() + unit.source.begin,
                                       prefix.size() + unit.source.end}, source});
                }
                regions.push_back(std::move(mapped));
            } else {
                for (const auto & child : value.children) {
                    visit(child); // Object children contain values, never keys.
                }
            }
        };
        visit(inner);
    }
    // One tool call is one native block. Do not reverse its individual arguments.
    add_content(prefix + arguments.text, regions, message, vocab, false);
}

void server_compress_plan::add_content(const std::string & input, const std::vector<region> & regions,
        size_t message, const llama_vocab * vocab, bool pinned) {
    text field;
    field.message = message;
    field.tokens = common_tokenize(vocab, input, false, false);
    if (field.tokens.empty()) {
        return;
    }
    field.token_units.assign(field.tokens.size(), no_unit);
    field.forward_nll.assign(field.tokens.size(), std::numeric_limits<float>::quiet_NaN());
    field.reverse_nll = field.forward_nll;

    std::string pieces;
    std::vector<size_t> ends = {0};
    for (llama_token token : field.tokens) {
        pieces += common_token_to_piece(vocab, token, true);
        ends.push_back(pieces.size());
    }
    const bool leading_space = pieces.size() == input.size() + 1 && pieces[0] == ' ' &&
        pieces.compare(1, input.size(), input) == 0;
    const bool exact = pieces == input || leading_space;
    if (leading_space) {
        for (auto & end : ends) {
            end -= end != 0;
        }
    }

    // A normalizing tokenizer may not have an invertible source map. Preserve
    // that field internally rather than inventing fixed code-point chunks. Its
    // native block still participates in scoring and whole-group deletion.
    if (exact && !pinned) {
        for (const auto & region : regions) {
            if (region.empty()) {
                continue;
            }
            struct boundary { size_t token; size_t source; };
            std::vector<boundary> boundaries;
            size_t token = std::lower_bound(ends.begin(), ends.end(), region.front().text.begin) - ends.begin();
            auto match = [&](size_t byte, size_t source) {
                while (token < ends.size() && ends[token] < byte) {
                    ++token;
                }
                if (token < ends.size() && ends[token] == byte) {
                    // An artificial leading space can have zero source bytes.
                    while (token + 1 < ends.size() && ends[token + 1] == byte) {
                        ++token;
                    }
                    boundaries.push_back({token, source});
                }
            };
            match(region.front().text.begin, region.front().source.begin);
            for (const auto & mapped : region) {
                match(mapped.text.end, mapped.source.end);
            }
            if (boundaries.size() < 2) {
                continue;
            }
            for (size_t i = 1; i < boundaries.size(); ++i) {
                const auto & begin = boundaries[i - 1];
                const auto & end = boundaries[i];
                if (begin.token == end.token || begin.source == end.source) {
                    continue;
                }
                const bool edge = begin.token - boundaries.front().token < protected_tokens ||
                    boundaries.back().token - end.token < protected_tokens;
                const size_t id = units_.size();
                units_.push_back({{begin.source, end.source}, message, end.token - begin.token, edge, false});
                std::fill(field.token_units.begin() + begin.token, field.token_units.begin() + end.token, id);
            }
        }
    }
    texts_.push_back(std::move(field));
}

void server_compress_plan::add_scores(size_t text_index, size_t begin, const std::vector<float> & nll, bool reverse) {
    auto & field = texts_.at(text_index);
    if (begin > field.tokens.size() || nll.size() > field.tokens.size() - begin) {
        throw std::logic_error("compression score range is out of bounds");
    }
    auto & target = reverse ? field.reverse_nll : field.forward_nll;
    std::copy(nll.begin(), nll.end(), target.begin() + begin);
}

void server_compress_plan::rank() {
    actions_.clear();
    next_action_ = 0;
    struct losses {
        double delta = 0;
        double forward = 0;
        size_t pairs = 0;
        size_t forwards = 0;
        size_t tokens = 0;
        void add(float f, float r) {
            ++tokens;
            if (std::isfinite(f)) {
                forward += f;
                ++forwards;
                if (std::isfinite(r)) {
                    delta += double(f) - double(r);
                    ++pairs;
                }
            }
        }
        double score() const {
            return pairs ? delta / pairs : tokens ? std::numeric_limits<double>::infinity() : 0;
        }
        double tie_break() const {
            return forwards ? forward / forwards : tokens ? std::numeric_limits<double>::infinity() : 0;
        }
    };
    std::vector<losses> unit_losses(units_.size());
    std::vector<losses> group_losses(groups_.size());
    for (const auto & field : texts_) {
        auto & group = group_losses[message_groups_[field.message]];
        for (size_t i = 0; i < field.tokens.size(); ++i) {
            // One native block has identical conditioning in both directions.
            // Do not let backend rounding noise replace its forward-NLL tie break.
            const float reverse = texts_.size() == 1 ? field.forward_nll[i] : field.reverse_nll[i];
            group.add(field.forward_nll[i], reverse);
            if (field.token_units[i] != no_unit) {
                unit_losses[field.token_units[i]].add(field.forward_nll[i], reverse);
            }
        }
    }
    for (size_t i = 0; i < units_.size(); ++i) {
        const auto & unit = units_[i];
        if (!unit.pinned) {
            actions_.push_back({i, unit.source.begin,
                               unit_losses[i].score(), unit_losses[i].tie_break(), false});
        }
    }
    const auto & messages = root_.find("messages")->children;
    for (size_t i = 0; i < groups_.size(); ++i) {
        const auto & group = groups_[i];
        if (!group.pinned) {
            // Metadata-only historical turns can be removed without model work.
            actions_.push_back({i, messages[group.messages.front()].source.begin,
                                group_losses[i].score(), group_losses[i].tie_break(), true});
        }
    }
    std::stable_sort(actions_.begin(), actions_.end(), [](const action & a, const action & b) {
        if (a.score != b.score) {
            return a.score < b.score;
        }
        // A single content block has identical passes. Forward surprisal gives
        // it a useful ranking without an extra weight or a synthetic partition.
        if (a.forward != b.forward) {
            return a.forward < b.forward;
        }
        return a.order < b.order;
    });
}

bool server_compress_plan::remove_tokens(size_t estimate) {
    bool changed = false;
    double saved = 0;
    while (next_action_ < actions_.size() && saved < double(estimate)) {
        const auto & action = actions_[next_action_++];
        if (action.whole_group) {
            auto & group = groups_[action.index];
            if (group.removed) {
                continue;
            }
            saved += std::max(1.0, double(group.estimated_tokens) - double(group.erased_tokens) +
                                  double(group.markers) * double(marker_tokens_));
            group.removed = true;
        } else {
            auto & unit = units_[action.index];
            if (unit.removed || groups_[message_groups_[unit.message]].removed) {
                continue;
            }
            const size_t index = action.index;
            const bool left = index > 0 && units_[index - 1].removed &&
                units_[index - 1].source.end == unit.source.begin;
            const bool right = index + 1 < units_.size() && units_[index + 1].removed &&
                unit.source.end == units_[index + 1].source.begin;
            const int added_markers = 1 - int(left) - int(right);
            saved += double(unit.tokens) - double(added_markers) * double(marker_tokens_);
            auto & group = groups_[message_groups_[unit.message]];
            group.erased_tokens += unit.tokens;
            if (added_markers < 0) {
                --group.markers;
            } else {
                group.markers += size_t(added_markers);
            }
            unit.removed = true;
        }
        changed = true;
    }
    return changed;
}

server_compress_json::deletion_result server_compress_plan::minimum() const { return render(true); }
server_compress_json::deletion_result server_compress_plan::render() const { return render(false); }

server_compress_json::deletion_result server_compress_plan::render(bool minimum) const {
    const auto & messages = root_.find("messages")->children;
    std::vector<bool> removed(messages.size(), false);
    for (const auto & group : groups_) {
        if (group.removed || (minimum && !group.pinned)) {
            for (size_t message : group.messages) {
                removed[message] = true;
            }
        }
    }
    std::vector<range> strings;
    for (const auto & unit : units_) {
        if (!removed[unit.message] && (unit.removed || (minimum && !unit.pinned))) {
            strings.push_back(unit.source);
        }
    }
    std::vector<range> structural;
    for (size_t first = 0; first < messages.size();) {
        if (!removed[first]) {
            ++first;
            continue;
        }
        size_t end = first + 1;
        while (end < messages.size() && removed[end]) {
            ++end;
        }
        if (end < messages.size()) {
            structural.push_back({messages[first].source.begin, messages[end].source.begin});
        } else if (first > 0) {
            structural.push_back({messages[first - 1].source.end, messages.back().source.end});
        } else {
            throw std::logic_error("compression must retain a message");
        }
        first = end;
    }
    return server_compress_json::apply_deletions(raw_, root_, std::move(structural), std::move(strings));
}

size_t server_compress_plan::removed_messages() const {
    size_t result = 0;
    for (const auto & group : groups_) {
        if (group.removed) {
            result += group.messages.size();
        }
    }
    return result;
}
