#include "server-rerot.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <utility>

namespace {

constexpr std::string_view k_open_ol  = "<ol>";
constexpr std::string_view k_close_ol = "</ol>";
constexpr std::string_view k_open_li  = "<li>";
constexpr std::string_view k_close_li = "</li>";

bool ascii_space(unsigned char ch) {
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' || ch == '\v';
}

void skip_ascii_space(std::string_view text, size_t & pos, size_t end) {
    while (pos < end && ascii_space(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
}

std::string trim_ascii_space(std::string_view text) {
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && ascii_space(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    while (end > begin && ascii_space(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}

size_t trailing_opener_prefix(std::string_view text) {
    const size_t max_len = std::min(text.size(), k_open_ol.size() - 1);
    for (size_t len = max_len; len > 0; --len) {
        if (text.substr(text.size() - len) == k_open_ol.substr(0, len)) {
            return len;
        }
    }
    return 0;
}

size_t trailing_marker_prefix(std::string_view text, std::string_view marker) {
    if (marker.empty()) {
        return 0;
    }
    const size_t max_len = std::min(text.size(), marker.size() - 1);
    for (size_t len = max_len; len > 0; --len) {
        if (text.substr(text.size() - len) == marker.substr(0, len)) {
            return len;
        }
    }
    return 0;
}

bool only_ascii_space(std::string_view text) {
    return std::all_of(text.begin(), text.end(), [](char ch) {
        return ascii_space(static_cast<unsigned char>(ch));
    });
}

bool ascii_iequal(char lhs, char rhs) {
    return std::tolower(static_cast<unsigned char>(lhs)) ==
           std::tolower(static_cast<unsigned char>(rhs));
}

std::string normalize_lane_title(std::string_view raw) {
    std::string title = trim_ascii_space(raw);
    size_t pos = 0;
    static constexpr std::string_view lane = "lane";
    if (title.size() >= lane.size() &&
        std::equal(lane.begin(), lane.end(), title.begin(), ascii_iequal)) {
        pos = lane.size();
        while (pos < title.size() && ascii_space(static_cast<unsigned char>(title[pos]))) {
            ++pos;
        }
        const size_t digit_begin = pos;
        while (pos < title.size() && std::isdigit(static_cast<unsigned char>(title[pos]))) {
            ++pos;
        }
        if (pos > digit_begin) {
            while (pos < title.size() && ascii_space(static_cast<unsigned char>(title[pos]))) {
                ++pos;
            }
            bool has_separator = false;
            if (pos < title.size() && (title[pos] == ':' || title[pos] == '-' || title[pos] == '.')) {
                ++pos;
                has_separator = true;
            } else if (pos + 3 <= title.size() &&
                       static_cast<unsigned char>(title[pos])     == 0xEF &&
                       static_cast<unsigned char>(title[pos + 1]) == 0xBC &&
                       static_cast<unsigned char>(title[pos + 2]) == 0x9A) {
                pos += 3; // UTF-8 full-width colon
                has_separator = true;
            }
            if (has_separator) {
                while (pos < title.size() && ascii_space(static_cast<unsigned char>(title[pos]))) {
                    ++pos;
                }
                title.erase(0, pos);
            }
        }
    }

    std::string escaped;
    escaped.reserve(title.size());
    for (char ch : title) {
        switch (ch) {
            case '&':  escaped += "&amp;";  break;
            case '<':  escaped += "&lt;";   break;
            case '>':  escaped += "&gt;";   break;
            case '"': escaped += "&quot;"; break;
            case '\'': escaped += "&#39;"; break;
            default:   escaped.push_back(ch); break;
        }
    }
    return escaped;
}

} // namespace

server_rerot_planner_parser::server_rerot_planner_parser() {
    reset();
}

void server_rerot_planner_parser::reset() {
    state_ = server_rerot_parser_state::public_text;
    opener_candidate_.clear();
    list_bytes_.clear();
    items_.clear();
    error_.clear();
}

server_rerot_parser_state server_rerot_planner_parser::state() const {
    return state_;
}

bool server_rerot_planner_parser::complete() const {
    return state_ == server_rerot_parser_state::complete;
}

bool server_rerot_planner_parser::failed() const {
    return state_ == server_rerot_parser_state::failed;
}

const std::vector<std::string> & server_rerot_planner_parser::items() const {
    return items_;
}

const std::string & server_rerot_planner_parser::error() const {
    return error_;
}

server_rerot_parser_step server_rerot_planner_parser::consume(std::string_view bytes) {
    if (state_ == server_rerot_parser_state::complete) {
        server_rerot_parser_step step;
        step.malformed = true;
        step.error = "planner emitted bytes after the completed </ol> record";
        return step;
    }
    if (state_ == server_rerot_parser_state::failed) {
        server_rerot_parser_step step;
        step.write_visibility = llama_rerot_visibility::pending_record;
        step.malformed = true;
        step.error = error_;
        return step;
    }
    if (state_ == server_rerot_parser_state::list_pending) {
        return consume_list_bytes(bytes, false);
    }
    return consume_before_list(bytes);
}

server_rerot_parser_step server_rerot_planner_parser::consume_before_list(std::string_view bytes) {
    server_rerot_parser_step step;

    if (!opener_candidate_.empty()) {
        std::string joined = opener_candidate_;
        joined.append(bytes.data(), bytes.size());

        if (joined.size() < k_open_ol.size() && k_open_ol.substr(0, joined.size()) == joined) {
            opener_candidate_ = std::move(joined);
            state_ = server_rerot_parser_state::opening_candidate;
            step.write_visibility = llama_rerot_visibility::pending_record;
            return step;
        }

        if (joined.size() >= k_open_ol.size() &&
            std::string_view(joined).substr(0, k_open_ol.size()) == k_open_ol) {
            opener_candidate_.clear();
            state_ = server_rerot_parser_state::list_pending;
            list_bytes_ = std::move(joined);
            step.write_visibility = llama_rerot_visibility::pending_record;
            step.record_opened = true;
            return consume_list_bytes({}, true);
        }

        // The bytes resident from earlier tokens are no longer ambiguous. They
        // can become PUBLIC atomically, while this token is classified afresh.
        opener_candidate_.clear();
        state_ = server_rerot_parser_state::public_text;
        step.release_previous_pending = true;
    }

    const size_t opener = bytes.find(k_open_ol);
    if (opener != std::string_view::npos) {
        state_ = server_rerot_parser_state::list_pending;
        list_bytes_.assign(bytes.substr(opener));
        step.write_visibility = llama_rerot_visibility::pending_record;
        step.record_opened = true;

        auto list_step = consume_list_bytes({}, true);
        list_step.release_previous_pending = step.release_previous_pending;
        return list_step;
    }

    const size_t suffix = trailing_opener_prefix(bytes);
    if (suffix != 0) {
        opener_candidate_.assign(bytes.substr(bytes.size() - suffix));
        state_ = server_rerot_parser_state::opening_candidate;
        step.write_visibility = llama_rerot_visibility::pending_record;
        return step;
    }

    state_ = server_rerot_parser_state::public_text;
    step.write_visibility = llama_rerot_visibility::public_live;
    return step;
}

server_rerot_parser_step server_rerot_planner_parser::consume_list_bytes(
        std::string_view bytes,
        bool opened_now) {
    server_rerot_parser_step step;
    step.write_visibility = llama_rerot_visibility::pending_record;
    step.record_opened = opened_now;

    if (!bytes.empty()) {
        list_bytes_.append(bytes.data(), bytes.size());
    }

    finish_record(step);
    return step;
}

bool server_rerot_planner_parser::finish_record(server_rerot_parser_step & step) {
    if (list_bytes_.size() < k_open_ol.size() ||
        std::string_view(list_bytes_).substr(0, k_open_ol.size()) != k_open_ol) {
        return fail(step, "internal planner parser lost the <ol> record boundary");
    }

    const size_t nested = list_bytes_.find(k_open_ol, k_open_ol.size());
    const size_t close = list_bytes_.find(k_close_ol, k_open_ol.size());
    if (nested != std::string::npos && (close == std::string::npos || nested < close)) {
        return fail(step, "nested <ol> is not allowed in a RERoT planner record");
    }
    if (close == std::string::npos) {
        return false;
    }

    const size_t close_end = close + k_close_ol.size();
    if (!only_ascii_space(std::string_view(list_bytes_).substr(close_end))) {
        return fail(step, "planner emitted non-whitespace bytes after </ol> in the same token");
    }

    std::vector<std::string> parsed;
    size_t pos = k_open_ol.size();
    skip_ascii_space(list_bytes_, pos, close);

    while (pos < close) {
        if (close - pos < k_open_li.size() ||
            std::string_view(list_bytes_).substr(pos, k_open_li.size()) != k_open_li) {
            return fail(step, "RERoT planner record must contain only direct <li> children");
        }
        pos += k_open_li.size();

        const size_t item_close = list_bytes_.find(k_close_li, pos);
        if (item_close == std::string::npos || item_close > close) {
            return fail(step, "RERoT planner record contains an unclosed <li>");
        }

        const std::string_view body(list_bytes_.data() + pos, item_close - pos);
        if (body.find('<') != std::string_view::npos) {
            return fail(step, "nested tags are not allowed inside RERoT <li> titles");
        }

        std::string title = trim_ascii_space(body);
        if (title.empty()) {
            return fail(step, "RERoT planner <li> titles must not be empty");
        }
        parsed.push_back(std::move(title));

        pos = item_close + k_close_li.size();
        skip_ascii_space(list_bytes_, pos, close);
    }

    if (parsed.empty()) {
        return fail(step, "RERoT planner <ol> must contain at least one <li>");
    }

    items_ = std::move(parsed);
    state_ = server_rerot_parser_state::complete;
    step.record_closed = true;
    step.items = items_;
    return true;
}

bool server_rerot_planner_parser::fail(server_rerot_parser_step & step, std::string message) {
    state_ = server_rerot_parser_state::failed;
    error_ = std::move(message);
    step.write_visibility = llama_rerot_visibility::pending_record;
    step.malformed = true;
    step.error = error_;
    return false;
}

server_rerot_marker_parser::server_rerot_marker_parser(std::string marker)
    : marker_(std::move(marker)) {
    reset();
}

void server_rerot_marker_parser::reset() {
    candidate_.clear();
    error_.clear();
    state_ = marker_.empty()
        ? server_rerot_marker_state::failed
        : server_rerot_marker_state::public_text;
    if (marker_.empty()) {
        error_ = "RERoT private marker must not be empty";
    }
}

server_rerot_marker_state server_rerot_marker_parser::state() const {
    return state_;
}

bool server_rerot_marker_parser::complete() const {
    return state_ == server_rerot_marker_state::complete;
}

bool server_rerot_marker_parser::failed() const {
    return state_ == server_rerot_marker_state::failed;
}

const std::string & server_rerot_marker_parser::error() const {
    return error_;
}

bool server_rerot_marker_parser::fail(server_rerot_marker_step & step, std::string message) {
    state_ = server_rerot_marker_state::failed;
    error_ = std::move(message);
    step.write_visibility = llama_rerot_visibility::pending_record;
    step.malformed = true;
    step.error = error_;
    return false;
}

server_rerot_marker_step server_rerot_marker_parser::consume(std::string_view bytes) {
    server_rerot_marker_step step;

    if (state_ == server_rerot_marker_state::failed) {
        step.malformed = true;
        step.error = error_;
        step.write_visibility = llama_rerot_visibility::pending_record;
        return step;
    }
    if (state_ == server_rerot_marker_state::complete) {
        fail(step, "RERoT Lane emitted bytes after its private end marker");
        return step;
    }

    std::string combined;
    std::string_view scan = bytes;
    if (!candidate_.empty()) {
        combined = candidate_;
        combined.append(bytes.data(), bytes.size());
        scan = combined;

        if (scan.size() < marker_.size() && marker_.substr(0, scan.size()) == scan) {
            candidate_.assign(scan);
            state_ = server_rerot_marker_state::marker_candidate;
            step.write_visibility = llama_rerot_visibility::pending_record;
            return step;
        }

        if (scan.size() >= marker_.size() && scan.substr(0, marker_.size()) == marker_) {
            if (!only_ascii_space(scan.substr(marker_.size()))) {
                fail(step, "RERoT Lane emitted non-whitespace bytes after its private end marker");
                return step;
            }
            candidate_.clear();
            state_ = server_rerot_marker_state::complete;
            step.write_visibility = llama_rerot_visibility::pending_record;
            step.marker_closed = true;
            return step;
        }

        // Earlier tokenizer tokens only looked like the marker. Release their
        // pending run before classifying the current token independently.
        candidate_.clear();
        state_ = server_rerot_marker_state::public_text;
        step.release_previous_pending = true;
        scan = bytes;
    }

    const size_t full = scan.find(marker_);
    if (full != std::string_view::npos) {
        if (!only_ascii_space(scan.substr(full + marker_.size()))) {
            fail(step, "RERoT Lane emitted non-whitespace bytes after its private end marker");
            return step;
        }
        state_ = server_rerot_marker_state::complete;
        step.write_visibility = llama_rerot_visibility::pending_record;
        step.marker_closed = true;
        return step;
    }

    const size_t suffix = trailing_marker_prefix(scan, marker_);
    if (suffix != 0) {
        candidate_.assign(scan.substr(scan.size() - suffix));
        state_ = server_rerot_marker_state::marker_candidate;
        step.write_visibility = llama_rerot_visibility::pending_record;
        return step;
    }

    state_ = server_rerot_marker_state::public_text;
    step.write_visibility = llama_rerot_visibility::public_live;
    return step;
}

std::string_view server_rerot_planner_prompt() {
    static constexpr std::string_view prompt =
        "让我先分析这个问题的规模与难度，并在分析完成后用\n"
        "<ol>\n"
        "<li>Lane 1 部分的问题描述</li>\n"
        "<li>Lane 2 部分的问题描述</li>\n"
        "...\n"
        "</ol>\n"
        "的方式多线程推理。如果问题简单也可以只有一个 Lane。\n\n"
        "各 Lane 应尽量是可以并行推进的互补部分；共同前提、依赖关系和必要定义应先在列表前说明。\n"
        "公开推理引用其他部分时，请使用章节标题、命题、公式或具体结论，不依赖“上面、下面、前者、后一段”等版面相对指代。\n"
        "完整输出 </ol> 后立即停止规划，不继续展开 Lane 正文。";
    return prompt;
}

server_rerot_runtime::server_rerot_runtime(
        llama_memory_t memory,
        llama_rerot_frontier_mode frontier_mode,
        uint32_t first_internal_seq,
        uint32_t max_seq)
    : memory_(memory),
      frontier_mode_(frontier_mode),
      first_internal_seq_(first_internal_seq),
      max_seq_(std::max(first_internal_seq, max_seq)) {
    for (uint32_t seq = first_internal_seq_; seq < max_seq_; ++seq) {
        free_internal_seqs_.push_back(static_cast<llama_seq_id>(seq));
    }
}

uint64_t server_rerot_runtime::adopt_root(
        int root_task_id,
        int response_task_id,
        int physical_slot,
        llama_seq_id exec_seq,
        llama_pos storage_pos_next) {
    if (physical_slot < 0 || exec_seq < 0 || storage_pos_next < 0) {
        return 0;
    }

    release_slot(physical_slot);

    uint64_t episode_id = next_episode_id_++;
    if (episode_id == 0) {
        episode_id = next_episode_id_++;
    }

    auto inserted = episodes_.emplace(episode_id, server_rerot_episode(episode_id));
    if (!inserted.second) {
        return 0;
    }

    auto & current = inserted.first->second;
    current.root_task_id = root_task_id;
    current.response_task_id = response_task_id;
    current.base_prefix_end = storage_pos_next;

    server_rerot_node_runtime root;
    root.id = current.document.root();
    root.physical_slot = physical_slot;
    root.exec_seq = exec_seq;
    root.storage_pos_next = storage_pos_next;
    current.nodes.push_back(std::move(root));
    current.running.insert(current.document.root());

    slot_to_episode_[physical_slot] = episode_id;
    return episode_id;
}

void server_rerot_runtime::release_slot(int physical_slot) {
    const auto slot_it = slot_to_episode_.find(physical_slot);
    if (slot_it == slot_to_episode_.end()) {
        return;
    }

    const uint64_t episode_id = slot_it->second;
    auto ep_it = episodes_.find(episode_id);
    if (ep_it != episodes_.end()) {
        auto & current = ep_it->second;
        for (auto & current_node : current.nodes) {
            if (current_node.physical_slot != physical_slot) {
                continue;
            }
            clear_sequence_control(current_node.exec_seq);
            current.running.erase(current_node.id);
            current.starting.erase(current_node.id);
            current_node.physical_slot = -1;
            current_node.exec_seq = -1;
        }
    }

    slot_to_episode_.erase(slot_it);

    // Logical episode lifetime is explicit. A fork can legitimately have no
    // bound physical slot while all children are queued, so never infer death
    // from an empty slot map.
}

server_rerot_episode * server_rerot_runtime::episode(uint64_t episode_id) {
    const auto it = episodes_.find(episode_id);
    return it == episodes_.end() ? nullptr : &it->second;
}

const server_rerot_episode * server_rerot_runtime::episode(uint64_t episode_id) const {
    const auto it = episodes_.find(episode_id);
    return it == episodes_.end() ? nullptr : &it->second;
}

server_rerot_episode * server_rerot_runtime::episode_for_slot(int physical_slot) {
    const auto it = slot_to_episode_.find(physical_slot);
    return it == slot_to_episode_.end() ? nullptr : episode(it->second);
}

const server_rerot_episode * server_rerot_runtime::episode_for_slot(int physical_slot) const {
    const auto it = slot_to_episode_.find(physical_slot);
    return it == slot_to_episode_.end() ? nullptr : episode(it->second);
}

server_rerot_node_runtime * server_rerot_runtime::node(
        uint64_t episode_id,
        llama_rerot_node_id node_id) {
    auto * current = episode(episode_id);
    return current && node_id < current->nodes.size() ? &current->nodes[node_id] : nullptr;
}

const server_rerot_node_runtime * server_rerot_runtime::node(
        uint64_t episode_id,
        llama_rerot_node_id node_id) const {
    const auto * current = episode(episode_id);
    return current && node_id < current->nodes.size() ? &current->nodes[node_id] : nullptr;
}

uint64_t server_rerot_runtime::next_publish_epoch(server_rerot_episode & episode) {
    if (episode.publish_epoch == std::numeric_limits<uint64_t>::max()) {
        fail_episode(episode, "RERoT publish epoch overflow");
        return 0;
    }
    return ++episode.publish_epoch;
}

llama_rerot_run_id server_rerot_runtime::ensure_run(
        server_rerot_episode & episode,
        server_rerot_node_runtime & node,
        llama_rerot_visibility visibility,
        llama_pos storage_pos) {
    std::optional<llama_rerot_run_id> * active = nullptr;
    switch (visibility) {
        case llama_rerot_visibility::public_live:     active = &node.public_run; break;
        case llama_rerot_visibility::private_control: active = &node.private_run; break;
        case llama_rerot_visibility::pending_record:  active = &node.pending_record; break;
        case llama_rerot_visibility::normal: return LLAMA_REROT_RUN_INVALID;
    }

    if (active->has_value()) {
        const auto * current_run = episode.document.run(**active);
        if (current_run && current_run->visibility == visibility &&
            int64_t(current_run->storage_pos0) + int64_t(current_run->token_count) == int64_t(storage_pos)) {
            return current_run->id;
        }
        active->reset();
    }

    const uint64_t publish_epoch = visibility == llama_rerot_visibility::public_live
        ? episode.publish_epoch
        : 0;
    const auto run_id = episode.document.append_run(
        node.id, visibility, storage_pos, 0, publish_epoch);
    *active = run_id;
    return run_id;
}

bool server_rerot_runtime::release_false_pending(
        server_rerot_episode & episode,
        server_rerot_node_runtime & node) {
    if (!node.pending_record.has_value()) {
        return fail_episode(episode, "RERoT parser released a missing pending run");
    }

    const auto run_id = *node.pending_record;
    const auto * run = episode.document.run(run_id);
    if (!run || run->visibility != llama_rerot_visibility::pending_record || run->token_count == 0) {
        return fail_episode(episode, "RERoT parser released an invalid pending run");
    }

    const uint64_t publish_epoch = next_publish_epoch(episode);
    if (publish_epoch == 0) {
        return false;
    }

    if (memory_) {
        const size_t changed = llama_memory_rerot_reclassify_run(
            memory_, episode.id, run_id,
            LLAMA_REROT_KV_PENDING_RECORD,
            LLAMA_REROT_KV_PUBLIC_LIVE,
            publish_epoch);
        if (changed == 0) {
            return fail_episode(episode, "failed to atomically release a false control-marker prefix");
        }
    }

    if (!episode.document.reclassify_run(
            run_id,
            llama_rerot_visibility::pending_record,
            llama_rerot_visibility::public_live,
            publish_epoch)) {
        return fail_episode(episode, "logical false-marker publication failed");
    }

    node.pending_record.reset();
    node.public_run = run_id;
    episode.topology_barrier_pending = true;
    ++episode.layout_epoch;
    return true;
}

std::optional<server_rerot_token_plan> server_rerot_runtime::plan_private_token(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        llama_pos storage_pos) {
    auto * current = episode(episode_id);
    auto * current_node = node(episode_id, node_id);
    if (!current || !current_node || current->hard_aborted || storage_pos < 0) {
        return std::nullopt;
    }

    server_rerot_token_plan plan;
    plan.storage_pos = storage_pos;
    plan.visibility = llama_rerot_visibility::private_control;
    plan.run_id = ensure_run(*current, *current_node, plan.visibility, storage_pos);
    return plan.valid() ? std::optional<server_rerot_token_plan>(std::move(plan)) : std::nullopt;
}

std::optional<server_rerot_token_plan> server_rerot_runtime::plan_heading_token(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        llama_pos storage_pos) {
    auto * current = episode(episode_id);
    auto * current_node = node(episode_id, node_id);
    if (!current || !current_node || current->hard_aborted || storage_pos < 0) {
        return std::nullopt;
    }

    server_rerot_token_plan plan;
    plan.storage_pos = storage_pos;
    plan.visibility = llama_rerot_visibility::pending_record;
    plan.run_id = ensure_run(*current, *current_node, plan.visibility, storage_pos);
    return plan.valid() ? std::optional<server_rerot_token_plan>(std::move(plan)) : std::nullopt;
}

std::optional<server_rerot_token_plan> server_rerot_runtime::plan_generated_token(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        llama_pos storage_pos,
        std::string_view token_bytes) {
    auto * current = episode(episode_id);
    auto * current_node = node(episode_id, node_id);
    if (!current || !current_node || current->hard_aborted || storage_pos < 0) {
        return std::nullopt;
    }

    server_rerot_parser_step parser_step;
    server_rerot_marker_step marker_step;
    if (current_node->planner_armed) {
        parser_step = current_node->parser.consume(token_bytes);
        if (parser_step.malformed) {
            fail_episode(*current, parser_step.error.empty()
                ? "malformed RERoT planner record"
                : parser_step.error);
            return std::nullopt;
        }
        if (parser_step.release_previous_pending && !release_false_pending(*current, *current_node)) {
            return std::nullopt;
        }
    } else {
        marker_step = current_node->exit_parser.consume(token_bytes);
        if (marker_step.malformed) {
            fail_episode(*current, marker_step.error.empty()
                ? "malformed RERoT private end marker"
                : marker_step.error);
            return std::nullopt;
        }
        if (marker_step.release_previous_pending && !release_false_pending(*current, *current_node)) {
            return std::nullopt;
        }
        parser_step.write_visibility = marker_step.write_visibility;
    }

    server_rerot_token_plan plan;
    plan.storage_pos = storage_pos;
    plan.visibility = parser_step.write_visibility;
    plan.parser_step = std::move(parser_step);
    plan.marker_step = std::move(marker_step);
    plan.run_id = ensure_run(*current, *current_node, plan.visibility, storage_pos);
    if (!plan.valid()) {
        fail_episode(*current, "failed to allocate a logical RERoT run");
        return std::nullopt;
    }
    return plan;
}

bool server_rerot_runtime::build_reader_view_desc(
        const server_rerot_episode & episode,
        const server_rerot_node_runtime & node,
        llama_rerot_run_id query_run,
        std::vector<uint32_t> & ordered_runs,
        llama_rerot_reader_view_desc & desc) const {
    const auto view = episode.document.build_view(node.id);
    ordered_runs.clear();
    ordered_runs.reserve(view.runs.size() + 1);
    for (const auto & current_run : view.runs) {
        ordered_runs.push_back(current_run.run_id);
    }
    if (std::find(ordered_runs.begin(), ordered_runs.end(), query_run) == ordered_runs.end()) {
        ordered_runs.push_back(query_run);
    }

    desc.episode_id = episode.id;
    desc.reader_node_id = node.id;
    desc.query_run_id = query_run;
    desc.frontier = episode.frontier;
    desc.frontier_mode = frontier_mode_;
    desc.stamp = {
        episode.topology_epoch,
        episode.publish_epoch,
        episode.layout_epoch,
    };
    desc.ordered_run_ids = ordered_runs.data();
    desc.n_ordered_runs = ordered_runs.size();
    return !ordered_runs.empty();
}

bool server_rerot_runtime::install_token_plan(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        const server_rerot_token_plan & plan) {
    auto * current = episode(episode_id);
    auto * current_node = node(episode_id, node_id);
    if (!current || !current_node || current->hard_aborted || !plan.valid()) {
        return false;
    }
    if (!memory_) {
        return true;
    }

    llama_rerot_kv_visibility visibility = LLAMA_REROT_KV_PUBLIC_LIVE;
    switch (plan.visibility) {
        case llama_rerot_visibility::public_live:     visibility = LLAMA_REROT_KV_PUBLIC_LIVE; break;
        case llama_rerot_visibility::private_control: visibility = LLAMA_REROT_KV_PRIVATE_CONTROL; break;
        case llama_rerot_visibility::pending_record:  visibility = LLAMA_REROT_KV_PENDING_RECORD; break;
        case llama_rerot_visibility::normal: return false;
    }

    const llama_rerot_kv_write_tag tag = {
        current->id,
        current_node->id,
        plan.run_id,
        plan.visibility == llama_rerot_visibility::public_live ? current->publish_epoch : 0,
        current->frontier,
        visibility,
    };
    if (!llama_memory_rerot_set_write_tag(memory_, current_node->exec_seq, &tag)) {
        return fail_episode(*current, "failed to install RERoT KV write tag");
    }

    std::vector<uint32_t> ordered_runs;
    llama_rerot_reader_view_desc desc = {};
    if (!build_reader_view_desc(*current, *current_node, plan.run_id, ordered_runs, desc) ||
        !llama_memory_rerot_set_reader_view(memory_, current_node->exec_seq, &desc)) {
        llama_memory_rerot_clear_write_tag(memory_, current_node->exec_seq);
        return fail_episode(*current, "failed to install RERoT reader view");
    }
    return true;
}

bool server_rerot_runtime::publish_pending_record(
        server_rerot_episode & episode,
        server_rerot_node_runtime & node,
        llama_rerot_run_id run_id,
        const std::vector<std::string> & items) {
    if (!node.pending_record.has_value() || *node.pending_record != run_id || items.empty()) {
        return fail_episode(episode, "planner closed a missing or empty pending record");
    }

    if (!publish_pending_run(episode, node, run_id, true)) {
        return false;
    }
    node.planner_armed = false;

    if (items.size() == 1) {
        if (!episode.document.set_node_state(node.id, llama_rerot_node_state::terminal_running)) {
            return fail_episode(episode, "failed to enter RERoT N=1 terminal state");
        }
        return true;
    }

    const llama_rerot_node_id parent_id = node.id;
    if (!episode.document.set_node_state(parent_id, llama_rerot_node_state::forked)) {
        return fail_episode(episode, "failed to freeze RERoT fork parent");
    }
    episode.running.erase(parent_id);
    episode.forked_this_frontier.push_back(parent_id);

    for (const auto & title : items) {
        const auto child_id = episode.document.create_child(
            parent_id, title, llama_rerot_node_state::queued);
        if (child_id != episode.nodes.size()) {
            return fail_episode(episode, "RERoT node ids lost dense runtime alignment");
        }
        server_rerot_node_runtime child;
        child.id = child_id;
        episode.nodes.push_back(std::move(child));
        episode.ready_queue.push_back(child_id);
    }

    return true;
}

bool server_rerot_runtime::publish_pending_run(
        server_rerot_episode & episode,
        server_rerot_node_runtime & node,
        llama_rerot_run_id run_id,
        bool topology_change) {
    if (!node.pending_record.has_value() || *node.pending_record != run_id) {
        return fail_episode(episode, "attempted to publish a non-current pending run");
    }
    const auto * run = episode.document.run(run_id);
    if (!run || run->visibility != llama_rerot_visibility::pending_record || run->token_count == 0) {
        return fail_episode(episode, "attempted to publish an empty or invalid pending run");
    }

    const uint64_t publish_epoch = next_publish_epoch(episode);
    if (publish_epoch == 0) {
        return false;
    }

    if (memory_) {
        const size_t changed = llama_memory_rerot_publish_run(memory_, episode.id, run_id, publish_epoch);
        if (changed == 0) {
            return fail_episode(episode, "failed to atomically publish a pending RERoT run");
        }
    }
    if (!episode.document.publish_run(run_id, publish_epoch)) {
        return fail_episode(episode, "logical pending-run publication failed");
    }

    node.pending_record.reset();
    node.public_run = run_id;
    if (topology_change) {
        episode.topology_barrier_pending = true;
        ++episode.topology_epoch;
        ++episode.layout_epoch;
    }
    return true;
}

bool server_rerot_runtime::finalize_exit_marker(
        server_rerot_episode & episode,
        server_rerot_node_runtime & node,
        llama_rerot_run_id run_id) {
    if (!node.pending_record.has_value() || *node.pending_record != run_id) {
        return fail_episode(episode, "private end marker does not own the current pending run");
    }
    const auto * run = episode.document.run(run_id);
    if (!run || run->visibility != llama_rerot_visibility::pending_record || run->token_count == 0) {
        return fail_episode(episode, "private end marker run is empty or invalid");
    }

    if (memory_) {
        const size_t changed = llama_memory_rerot_reclassify_run(
            memory_, episode.id, run_id,
            LLAMA_REROT_KV_PENDING_RECORD,
            LLAMA_REROT_KV_PRIVATE_CONTROL,
            0);
        if (changed == 0) {
            return fail_episode(episode, "failed to atomically privatize </think>");
        }
    }
    if (!episode.document.reclassify_run(
            run_id,
            llama_rerot_visibility::pending_record,
            llama_rerot_visibility::private_control,
            0)) {
        return fail_episode(episode, "logical </think> privatization failed");
    }

    node.pending_record.reset();
    node.private_run = run_id;
    node.exit_intent = true;
    ++episode.layout_epoch;
    return true;
}

bool server_rerot_runtime::commit_token(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        const server_rerot_token_plan & plan) {
    auto * current = episode(episode_id);
    auto * current_node = node(episode_id, node_id);
    if (!current || !current_node || current->hard_aborted || !plan.valid()) {
        return false;
    }

    const auto * run = current->document.run(plan.run_id);
    if (!run || run->owner != node_id || run->visibility != plan.visibility ||
        int64_t(run->storage_pos0) + int64_t(run->token_count) != int64_t(plan.storage_pos)) {
        return fail_episode(*current, "RERoT token commit does not extend its planned run");
    }

    if (!current->document.set_run_token_count(plan.run_id, run->token_count + 1)) {
        return fail_episode(*current, "failed to extend RERoT logical run");
    }
    current_node->storage_pos_next = plan.storage_pos + 1;

    if (plan.parser_step.record_closed) {
        return publish_pending_record(
            *current, *current_node, plan.run_id, plan.parser_step.items);
    }
    if (plan.marker_step.marker_closed) {
        return finalize_exit_marker(*current, *current_node, plan.run_id);
    }
    return true;
}

bool server_rerot_runtime::publish_heading(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        llama_rerot_run_id run_id) {
    auto * current = episode(episode_id);
    auto * current_node = node(episode_id, node_id);
    if (!current || !current_node || current->hard_aborted) {
        return false;
    }
    return publish_pending_run(*current, *current_node, run_id, true);
}

std::optional<llama_seq_id> server_rerot_runtime::alloc_internal_seq() {
    if (free_internal_seqs_.empty()) {
        return std::nullopt;
    }
    const auto seq_id = free_internal_seqs_.front();
    free_internal_seqs_.pop_front();
    return seq_id;
}

void server_rerot_runtime::free_internal_seq(llama_seq_id seq_id) {
    if (seq_id < static_cast<llama_seq_id>(first_internal_seq_) ||
        seq_id >= static_cast<llama_seq_id>(max_seq_)) {
        return;
    }
    if (std::find(free_internal_seqs_.begin(), free_internal_seqs_.end(), seq_id) == free_internal_seqs_.end()) {
        free_internal_seqs_.push_back(seq_id);
    }
}

bool server_rerot_runtime::ensure_archive_seq(
        server_rerot_episode & episode,
        llama_seq_id source_seq) {
    if (episode.archive_seq >= 0) {
        return true;
    }

    const auto archive_seq = alloc_internal_seq();
    if (!archive_seq.has_value()) {
        return fail_episode(episode, "RERoT internal seq-id arena exhausted while creating archive");
    }
    episode.archive_seq = *archive_seq;

    if (memory_ && episode.base_prefix_end > 0) {
        llama_memory_seq_cp_attention(
            memory_, source_seq, episode.archive_seq, 0, episode.base_prefix_end);
    }
    return true;
}

void server_rerot_runtime::archive_public_runs(
        server_rerot_episode & episode,
        const server_rerot_node_runtime & node,
        llama_seq_id archive_seq) {
    if (!memory_ || archive_seq < 0 || node.exec_seq < 0) {
        return;
    }

    const auto * doc_node = episode.document.node(node.id);
    if (!doc_node) {
        return;
    }
    for (const auto run_id : doc_node->runs) {
        const auto * run = episode.document.run(run_id);
        if (!run || run->visibility != llama_rerot_visibility::public_live || run->token_count == 0) {
            continue;
        }
        // Select by stable logical metadata rather than storage ranges. The
        // physical KV layout may already have changed under TriAttention, but
        // every still-resident cell of this published run keeps its
        // (episode_id, run_id) identity across eviction/compaction.
        llama_memory_rerot_add_run_ref(memory_, episode.id, run_id, archive_seq);
    }
}

bool server_rerot_runtime::retire_node(
        server_rerot_episode & episode,
        server_rerot_node_runtime & node,
        int * released_slot) {
    if (released_slot) {
        *released_slot = -1;
    }
    if (node.physical_slot < 0 || node.exec_seq < 0 || node.pending_record.has_value()) {
        return fail_episode(episode, "invalid RERoT Lane retirement state");
    }
    if (!ensure_archive_seq(episode, node.exec_seq)) {
        return false;
    }

    archive_public_runs(episode, node, episode.archive_seq);
    if (memory_) {
        if (!llama_memory_seq_rm_attention(memory_, node.exec_seq, -1, -1) ||
            !llama_memory_seq_rm_recurrent(memory_, node.exec_seq, -1, -1)) {
            return fail_episode(episode, "failed to release retiring RERoT execution sequence");
        }
        clear_sequence_control(node.exec_seq);
    }

    const int physical_slot = node.physical_slot;
    const auto slot_it = slot_to_episode_.find(physical_slot);
    if (slot_it != slot_to_episode_.end() && slot_it->second == episode.id) {
        slot_to_episode_.erase(slot_it);
    }

    episode.running.erase(node.id);
    episode.starting.erase(node.id);
    node.exit_intent = false;
    node.physical_slot = -1;
    node.exec_seq = -1;
    if (!episode.document.set_node_state(node.id, llama_rerot_node_state::retired)) {
        return fail_episode(episode, "failed to mark retiring RERoT Lane as RETIRED");
    }

    episode.topology_barrier_pending = true;
    ++episode.topology_epoch;
    ++episode.layout_epoch;
    if (released_slot) {
        *released_slot = physical_slot;
    }
    return true;
}

bool server_rerot_runtime::freeze_fork_parent(
        uint64_t episode_id,
        llama_rerot_node_id parent_id) {
    auto * current = episode(episode_id);
    auto * parent = node(episode_id, parent_id);
    if (!current || !parent || current->hard_aborted) {
        return false;
    }
    const auto * parent_doc = current->document.node(parent_id);
    if (!parent_doc || parent_doc->state != llama_rerot_node_state::forked ||
        parent->physical_slot < 0 || parent->exec_seq < 0 || parent_doc->children.empty()) {
        return fail_episode(*current, "invalid RERoT parent freeze state");
    }

    std::vector<llama_seq_id> allocated;
    const bool need_archive = current->archive_seq < 0;
    const size_t need_count = parent_doc->children.size() + (need_archive ? 1 : 0);
    allocated.reserve(need_count);
    for (size_t i = 0; i < need_count; ++i) {
        const auto seq_id = alloc_internal_seq();
        if (!seq_id.has_value()) {
            for (const auto allocated_id : allocated) {
                free_internal_seq(allocated_id);
            }
            return fail_episode(*current, "RERoT internal seq-id arena exhausted during fork");
        }
        allocated.push_back(*seq_id);
    }

    size_t next_alloc = 0;
    llama_seq_id archive_seq = current->archive_seq;
    if (need_archive) {
        archive_seq = allocated[next_alloc++];
    }

    std::vector<std::pair<llama_rerot_node_id, llama_seq_id>> parked;
    parked.reserve(parent_doc->children.size());
    for (const auto child_id : parent_doc->children) {
        parked.emplace_back(child_id, allocated[next_alloc++]);
    }

    if (memory_) {
        if (need_archive && current->base_prefix_end > 0) {
            llama_memory_seq_cp_attention(
                memory_, parent->exec_seq, archive_seq, 0, current->base_prefix_end);
        }

        archive_public_runs(*current, *parent, archive_seq);

        for (const auto & child : parked) {
            llama_memory_seq_cp_recurrent(memory_, parent->exec_seq, child.second, -1, -1);
        }

        if (!llama_memory_seq_rm_attention(memory_, parent->exec_seq, -1, -1) ||
            !llama_memory_seq_rm_recurrent(memory_, parent->exec_seq, -1, -1)) {
            return fail_episode(*current, "failed to release parent execution sequence after fork");
        }
        clear_sequence_control(parent->exec_seq);
    }

    if (need_archive) {
        current->archive_seq = archive_seq;
    }
    for (const auto & child : parked) {
        auto * child_runtime = node(episode_id, child.first);
        if (!child_runtime) {
            return fail_episode(*current, "missing RERoT child runtime during fork freeze");
        }
        child_runtime->parked_seq = child.second;
        child_runtime->storage_pos_next = parent->storage_pos_next;
    }

    const int released_slot = parent->physical_slot;
    parent->physical_slot = -1;
    parent->exec_seq = -1;
    current->running.erase(parent_id);
    slot_to_episode_.erase(released_slot);
    current->topology_barrier_pending = true;
    ++current->topology_epoch;
    ++current->layout_epoch;
    return true;
}

bool server_rerot_runtime::admit_next_child(
        uint64_t episode_id,
        int physical_slot,
        llama_seq_id exec_seq,
        llama_rerot_node_id * admitted_node) {
    if (admitted_node) {
        *admitted_node = LLAMA_REROT_NODE_INVALID;
    }
    auto * current = episode(episode_id);
    if (!current || current->hard_aborted || physical_slot < 0 || exec_seq < 0 || current->ready_queue.empty()) {
        return false;
    }
    if (slot_to_episode_.find(physical_slot) != slot_to_episode_.end()) {
        return fail_episode(*current, "RERoT attempted to admit into a bound physical slot");
    }

    const auto child_id = current->ready_queue.front();
    auto * child = node(episode_id, child_id);
    const auto * child_doc = current->document.node(child_id);
    if (!child || !child_doc || child_doc->state != llama_rerot_node_state::queued || child->parked_seq < 0) {
        return fail_episode(*current, "RERoT ready queue contains an invalid parked child");
    }

    if (memory_) {
        if (current->archive_seq < 0) {
            return fail_episode(*current, "RERoT child admission has no archive sequence");
        }
        if (current->base_prefix_end > 0) {
            llama_memory_seq_cp_attention(
                memory_, current->archive_seq, exec_seq, 0, current->base_prefix_end);
        }

        // The copied recurrent state is positioned at the parent's final
        // planner token. Add a reference to the latest PUBLIC parent run so
        // attention and recurrent seq_pos_max stay aligned even though the
        // intervening private planner prompt is intentionally invisible.
        const auto * parent_doc = current->document.node(child_doc->parent);
        const llama_rerot_run * anchor = nullptr;
        if (parent_doc) {
            for (const auto run_id : parent_doc->runs) {
                const auto * run = current->document.run(run_id);
                if (!run || run->visibility != llama_rerot_visibility::public_live || run->token_count == 0) {
                    continue;
                }
                const int64_t end = int64_t(run->storage_pos0) + int64_t(run->token_count);
                const int64_t anchor_end = anchor
                    ? int64_t(anchor->storage_pos0) + int64_t(anchor->token_count)
                    : -1;
                if (end > anchor_end) {
                    anchor = run;
                }
            }
        }
        if (!anchor || int64_t(anchor->storage_pos0) + int64_t(anchor->token_count) != child->storage_pos_next ||
            llama_memory_rerot_add_run_ref(memory_, current->id, anchor->id, exec_seq) == 0) {
            return fail_episode(*current, "RERoT child admission could not anchor the fork frontier");
        }

        llama_memory_seq_cp_recurrent(memory_, child->parked_seq, exec_seq, -1, -1);
        if (!llama_memory_seq_rm_recurrent(memory_, child->parked_seq, -1, -1)) {
            return fail_episode(*current, "failed to release parked recurrent sequence after admission");
        }
    }

    free_internal_seq(child->parked_seq);
    child->parked_seq = -1;
    child->physical_slot = physical_slot;
    child->exec_seq = exec_seq;

    current->ready_queue.pop_front();
    current->starting.insert(child_id);
    if (!current->document.set_node_state(child_id, llama_rerot_node_state::starting)) {
        return fail_episode(*current, "failed to enter RERoT child STARTING state");
    }
    slot_to_episode_[physical_slot] = episode_id;
    if (admitted_node) {
        *admitted_node = child_id;
    }
    return true;
}

bool server_rerot_runtime::complete_admission(
        uint64_t episode_id,
        llama_rerot_node_id node_id) {
    auto * current = episode(episode_id);
    auto * child = node(episode_id, node_id);
    const auto * child_doc = current ? current->document.node(node_id) : nullptr;
    if (!current || !child || !child_doc || current->hard_aborted ||
        child_doc->state != llama_rerot_node_state::starting ||
        child->physical_slot < 0 || child->exec_seq < 0) {
        return current ? fail_episode(*current, "invalid RERoT child admission completion") : false;
    }

    current->starting.erase(node_id);
    current->running.insert(node_id);
    if (!current->document.set_node_state(node_id, llama_rerot_node_state::planning)) {
        return fail_episode(*current, "failed to enter RERoT child PLANNING state");
    }
    current->topology_barrier_pending = true;
    ++current->topology_epoch;
    ++current->layout_epoch;
    return true;
}

server_rerot_frontier_result server_rerot_runtime::finish_frontier(uint64_t episode_id) {
    server_rerot_frontier_result result;
    result.episode_id = episode_id;

    auto * current = episode(episode_id);
    if (!current) {
        result.hard_aborted = true;
        result.abort_reason = "unknown RERoT episode";
        return result;
    }

    result.completed_frontier = current->frontier;
    result.forked = std::move(current->forked_this_frontier);
    current->forked_this_frontier.clear();
    std::sort(result.forked.begin(), result.forked.end(), [&](auto lhs, auto rhs) {
        return current->document.tree_path(lhs) < current->document.tree_path(rhs);
    });

    std::vector<llama_rerot_node_id> exits;
    exits.reserve(current->running.size());
    for (const auto node_id : current->running) {
        const auto * current_node = node(episode_id, node_id);
        if (current_node && current_node->exit_intent) {
            exits.push_back(node_id);
        }
    }
    std::sort(exits.begin(), exits.end(), [&](auto lhs, auto rhs) {
        return current->document.tree_path(lhs) < current->document.tree_path(rhs);
    });

    llama_rerot_node_id final_candidate = LLAMA_REROT_NODE_INVALID;
    if (!exits.empty() &&
        exits.size() == current->running.size() &&
        current->ready_queue.empty() &&
        current->starting.empty()) {
        // Same-frontier simultaneous exits are not judged. Stable tree-path
        // order alone chooses the Lane that remains live for the final acquire
        // fence. It must not be retired before Stage 7 refreshes its logits.
        final_candidate = exits.back();
    }

    for (const auto node_id : exits) {
        if (node_id == final_candidate) {
            continue;
        }
        auto * current_node = node(episode_id, node_id);
        if (!current_node) {
            fail_episode(*current, "RERoT exit set references a missing node");
            break;
        }
        int released_slot = -1;
        if (!retire_node(*current, *current_node, &released_slot)) {
            break;
        }
        result.retired.push_back(node_id);
        if (released_slot >= 0) {
            result.released_slots.push_back(released_slot);
        }
    }

    if (!current->hard_aborted && final_candidate != LLAMA_REROT_NODE_INVALID) {
        result.final_node = final_candidate;
        current->finalizing = true;
        const auto * survivor = node(episode_id, final_candidate);
        if (!survivor || !survivor->exit_intent ||
            current->running.find(final_candidate) == current->running.end()) {
            fail_episode(*current, "RERoT final survivor lost its live execution state");
            result.final_node = LLAMA_REROT_NODE_INVALID;
        }
    }

    result.topology_barrier = current->topology_barrier_pending;
    current->topology_barrier_pending = false;

    if (!current->hard_aborted) {
        advance_frontier(episode_id);
    }

    result.hard_aborted = current->hard_aborted;
    result.abort_reason = current->abort_reason;
    return result;
}

bool server_rerot_runtime::detach_node(
        uint64_t episode_id,
        llama_rerot_node_id node_id) {
    auto * current = episode(episode_id);
    auto * current_node = node(episode_id, node_id);
    if (!current || !current_node) {
        return false;
    }

    const int physical_slot = current_node->physical_slot;
    const llama_seq_id exec_seq = current_node->exec_seq;
    clear_sequence_control(exec_seq);
    current_node->physical_slot = -1;
    current_node->exec_seq = -1;
    current->running.erase(node_id);
    current->starting.erase(node_id);
    if (physical_slot >= 0) {
        const auto it = slot_to_episode_.find(physical_slot);
        if (it != slot_to_episode_.end() && it->second == episode_id) {
            slot_to_episode_.erase(it);
        }
    }
    return true;
}

bool server_rerot_runtime::erase_episode(uint64_t episode_id) {
    auto it = episodes_.find(episode_id);
    if (it == episodes_.end()) {
        return false;
    }

    auto & current = it->second;
    for (auto & current_node : current.nodes) {
        if (current_node.exec_seq >= 0) {
            clear_sequence_control(current_node.exec_seq);
            if (memory_) {
                llama_memory_seq_rm(memory_, current_node.exec_seq, -1, -1);
            }
        }
        if (current_node.parked_seq >= 0) {
            if (memory_) {
                llama_memory_seq_rm(memory_, current_node.parked_seq, -1, -1);
            }
            free_internal_seq(current_node.parked_seq);
            current_node.parked_seq = -1;
        }
    }
    if (current.archive_seq >= 0) {
        if (memory_) {
            llama_memory_seq_rm(memory_, current.archive_seq, -1, -1);
        }
        free_internal_seq(current.archive_seq);
        current.archive_seq = -1;
    }

    for (auto slot_it = slot_to_episode_.begin(); slot_it != slot_to_episode_.end();) {
        if (slot_it->second == episode_id) {
            slot_it = slot_to_episode_.erase(slot_it);
        } else {
            ++slot_it;
        }
    }
    episodes_.erase(it);
    return true;
}

bool server_rerot_runtime::has_ready_nodes(uint64_t episode_id) const {
    const auto * current = episode(episode_id);
    return current && !current->ready_queue.empty();
}

bool server_rerot_runtime::parent_has_unadmitted_children(
        uint64_t episode_id,
        llama_rerot_node_id parent_id) const {
    const auto * current = episode(episode_id);
    const auto * parent = current ? current->document.node(parent_id) : nullptr;
    if (!current || !parent) {
        return false;
    }
    return std::any_of(parent->children.begin(), parent->children.end(), [&](auto child_id) {
        const auto * child = current->document.node(child_id);
        return child && (child->state == llama_rerot_node_state::queued ||
                         child->state == llama_rerot_node_state::starting);
    });
}

std::string server_rerot_runtime::heading_text(
        uint64_t episode_id,
        llama_rerot_node_id node_id) const {
    const auto * current = episode(episode_id);
    const auto * current_node = current ? current->document.node(node_id) : nullptr;
    if (!current_node || node_id == current->document.root()) {
        return {};
    }
    const uint32_t heading_level = std::min<uint32_t>(6, std::max<uint32_t>(1, current_node->depth));
    const std::string tag = "h" + std::to_string(heading_level);
    return "<" + tag + ">" + normalize_lane_title(current_node->title) + "</" + tag + ">\n";
}

void server_rerot_runtime::advance_frontier(uint64_t episode_id) {
    auto * current = episode(episode_id);
    if (!current || current->hard_aborted) {
        return;
    }
    if (current->frontier == std::numeric_limits<uint64_t>::max()) {
        fail_episode(*current, "RERoT frontier overflow");
        return;
    }
    ++current->frontier;
}

void server_rerot_runtime::clear_sequence_control(llama_seq_id seq_id) {
    if (!memory_ || seq_id < 0) {
        return;
    }
    llama_memory_rerot_clear_write_tag(memory_, seq_id);
    llama_memory_rerot_clear_reader_view(memory_, seq_id);
}

bool server_rerot_runtime::fail_episode(server_rerot_episode & episode, std::string reason) {
    episode.hard_aborted = true;
    episode.finalizing = false;
    episode.abort_reason = std::move(reason);
    return false;
}

bool server_rerot_runtime::hard_abort(uint64_t episode_id, std::string reason) {
    auto * current = episode(episode_id);
    return current ? fail_episode(*current, std::move(reason)) : false;
}

