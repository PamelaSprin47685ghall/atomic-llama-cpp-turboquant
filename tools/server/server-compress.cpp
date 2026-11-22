#include "server-compress.h"

#include "common.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace {

using server_compress_json::kind;
using server_compress_json::node;
using server_compress_json::range;

constexpr size_t block_tokens = 32;

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
        if (role == "system" || role == "developer") {
            continue;
        }
        if (const node * content = msg.find("content")) {
            if (content->type == kind::string) {
                add_text(*content, i, vocab);
            } else if (content->type == kind::array) {
                for (const auto & part : content->children) {
                    const node * type = part.find("type");
                    const node * value = part.find("text");
                    if (type && type->type == kind::string && type->text == "text" &&
                        value && value->type == kind::string) {
                        add_text(*value, i, vocab);
                    }
                }
            }
        }
        if (role == "assistant") {
            const node * reasoning = msg.find("reasoning_content");
            if (reasoning && reasoning->type == kind::string) {
                add_text(*reasoning, i, vocab);
            }
        }
        auto add_function = [&](const node & fn) {
            const auto & arguments = require(fn, "arguments", kind::string);
            // Opaque or unfinished arguments stay byte-for-byte intact.
            node inner;
            try {
                inner = server_compress_json::parse(arguments.text);
            } catch (const std::invalid_argument &) {
                return;
            }
            add_arguments(inner, i, vocab, arguments);
        };
        if (const node * tool_calls = msg.find("tool_calls"); tool_calls && !is_null(tool_calls)) {
            for (const auto & tc : tool_calls->children) {
                add_function(*tc.find("function"));
            }
        }
        if (const node * fc = msg.find("function_call"); fc && !is_null(fc)) {
            add_function(*fc);
        }
    }
}

void server_compress_plan::add_arguments(const node & value, size_t message,
        const llama_vocab * vocab, const node & outer) {
    if (value.type == kind::string) {
        add_text(value, message, vocab, &outer);
    } else {
        // Object children contain only values, never keys.
        for (const auto & child : value.children) {
            add_arguments(child, message, vocab, outer);
        }
    }
}

void server_compress_plan::add_text(const node & value, size_t message,
        const llama_vocab * vocab, const node * outer) {
    if (value.text.empty()) {
        return;
    }
    text field;
    field.message = message;
    field.tokens = common_tokenize(vocab, value.text, false, false);
    if (field.tokens.empty()) {
        return;
    }

    std::string pieces;
    std::vector<size_t> ends = {0};
    for (llama_token token : field.tokens) {
        pieces += common_token_to_piece(vocab, token, true);
        ends.push_back(pieces.size());
    }
    const bool leading_space = pieces.size() == value.text.size() + 1 && pieces[0] == ' ' &&
        pieces.compare(1, value.text.size(), value.text) == 0;
    const bool exact = pieces == value.text || leading_space;
    if (leading_space) {
        for (auto & end : ends) {
            end -= end != 0;
        }
    }

    std::vector<size_t> boundaries = {0};
    for (const auto & unit : value.units) {
        boundaries.push_back(unit.text_end);
    }
    const size_t first_block = blocks_.size();
    auto append = [&](size_t byte_begin, size_t byte_end, size_t token_begin, size_t token_end) {
        if (byte_begin == byte_end) {
            return;
        }
        range source = server_compress_json::map_string_range(value, byte_begin, byte_end);
        if (outer) {
            source = server_compress_json::map_string_range(*outer, source.begin, source.end);
        }
        const size_t id = blocks_.size();
        blocks_.push_back({source, message, token_end - token_begin, 0, 0, false, false});
        field.token_blocks.insert(field.token_blocks.end(), token_end - token_begin, id);
    };

    if (exact) {
        for (size_t begin = 0; begin < field.tokens.size();) {
            size_t end = std::min(begin + block_tokens, field.tokens.size());
            while (end < field.tokens.size() &&
                    (ends[end] == ends[begin] || !std::binary_search(boundaries.begin(), boundaries.end(), ends[end]))) {
                ++end;
            }
            append(ends[begin], ends[end], begin, end);
            begin = end;
        }
    } else {
        // Some vocabularies normalize text. Keep the original Unicode bytes;
        // tokenize fixed code-point blocks instead of trying to invert them.
        field.tokens.clear();
        for (size_t begin = 0; begin < value.units.size();) {
            const size_t end = std::min(begin + block_tokens, value.units.size());
            const size_t byte_begin = value.units[begin].text_begin;
            const size_t byte_end = value.units[end - 1].text_end;
            const auto tokens = common_tokenize(vocab, value.text.substr(byte_begin, byte_end - byte_begin), false, false);
            const size_t token_begin = field.tokens.size();
            field.tokens.insert(field.tokens.end(), tokens.begin(), tokens.end());
            append(byte_begin, byte_end, token_begin, field.tokens.size());
            begin = end;
        }
    }
    if (blocks_.size() != first_block) {
        blocks_[first_block].pinned = true;
        blocks_.back().pinned = true;
    }
    if (field.tokens.size() != field.token_blocks.size()) {
        throw std::runtime_error("compression token-to-source mapping failed");
    }
    texts_.push_back(std::move(field));
}

bool server_compress_plan::needs_scores(size_t index) const {
    const auto & field = texts_.at(index);
    if (!groups_[message_groups_[field.message]].pinned) {
        return true;
    }
    for (size_t block_id : field.token_blocks) {
        if (!blocks_[block_id].pinned) {
            return true;
        }
    }
    return false;
}

void server_compress_plan::add_scores(size_t text_index, size_t begin, const std::vector<float> & nll) {
    const auto & field = texts_.at(text_index);
    if (begin > field.tokens.size() || nll.size() > field.tokens.size() - begin) {
        throw std::logic_error("compression score range is out of bounds");
    }
    for (size_t i = 0; i < nll.size(); ++i) {
        if (std::isfinite(nll[i])) {
            auto & block = blocks_[field.token_blocks[begin + i]];
            block.loss_sum += nll[i];
            ++block.loss_count;
        }
    }
}

void server_compress_plan::rank() {
    actions_.clear();
    next_action_ = 0;
    std::vector<double> sums(groups_.size(), 0);
    std::vector<size_t> counts(groups_.size(), 0);
    for (size_t i = 0; i < blocks_.size(); ++i) {
        const auto & block = blocks_[i];
        const size_t group_id = message_groups_[block.message];
        sums[group_id] += block.loss_sum;
        counts[group_id] += block.loss_count;
        if (!block.pinned) {
            const double score = block.loss_count ? block.loss_sum / block.loss_count : std::numeric_limits<double>::infinity();
            actions_.push_back({i, block.tokens, block.source.begin, score, false});
        }
    }
    const auto & messages = root_.find("messages")->children;
    for (size_t i = 0; i < groups_.size(); ++i) {
        const auto & group = groups_[i];
        if (!group.pinned) {
            // Metadata-only historical turns can be removed without model work.
            const double score = counts[i] ? sums[i] / counts[i] : 0;
            actions_.push_back({i, group.estimated_tokens,
                                messages[group.messages.front()].source.begin, score, true});
        }
    }
    std::stable_sort(actions_.begin(), actions_.end(), [](const action & a, const action & b) {
        if (a.score != b.score) {
            return a.score < b.score;
        }
        return a.order < b.order;
    });
}

bool server_compress_plan::remove_tokens(size_t estimate) {
    bool changed = false;
    size_t removed = 0;
    while (next_action_ < actions_.size() && removed < estimate) {
        const auto & action = actions_[next_action_++];
        if (action.whole_group) {
            auto & group = groups_[action.index];
            if (group.removed) {
                continue;
            }
            group.removed = true;
        } else {
            auto & block = blocks_[action.index];
            if (block.removed || groups_[message_groups_[block.message]].removed) {
                continue;
            }
            block.removed = true;
        }
        changed = true;
        removed += std::max(size_t(1), action.tokens);
    }
    return changed;
}

std::string server_compress_plan::minimum() const { return render(true); }
std::string server_compress_plan::render() const { return render(false); }

std::string server_compress_plan::render(bool minimum) const {
    const auto & messages = root_.find("messages")->children;
    std::vector<bool> removed(messages.size(), false);
    for (const auto & group : groups_) {
        if (group.removed || (minimum && !group.pinned)) {
            for (size_t message : group.messages) {
                removed[message] = true;
            }
        }
    }
    std::vector<range> ranges;
    for (const auto & block : blocks_) {
        if (!removed[block.message] && (block.removed || (minimum && !block.pinned))) {
            ranges.push_back(block.source);
        }
    }
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
            ranges.push_back({messages[first].source.begin, messages[end].source.begin});
        } else if (first > 0) {
            ranges.push_back({messages[first - 1].source.end, messages.back().source.end});
        } else {
            throw std::logic_error("compression must retain a message");
        }
        first = end;
    }
    return server_compress_json::erase_ranges(raw_, std::move(ranges));
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
