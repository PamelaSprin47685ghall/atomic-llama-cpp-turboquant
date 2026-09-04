#include "server-rerot.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <mutex>
#include <utility>

namespace {

constexpr std::string_view k_open_ol  = "<ol>";
constexpr std::string_view k_close_ol = "</ol>";
constexpr std::string_view k_open_li  = "<li>";
constexpr std::string_view k_close_li = "</li>";
constexpr std::string_view k_private_control_tags[] = {"<blockquote>", "</blockquote>"};

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
        step.marker_closed = true;
        step.write_visibility = llama_rerot_visibility::pending_record;
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
        candidate_.clear();
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

void server_rerot_control_tag_filter::consume(std::string & bytes) {
    if (!pending_.empty()) {
        pending_.append(bytes);
        bytes.swap(pending_);
        pending_.clear();
    }

    while (!bytes.empty()) {
        size_t first = std::string::npos;
        std::string_view matched;
        for (const auto tag : k_private_control_tags) {
            const size_t pos = bytes.find(tag);
            if (pos < first) {
                first = pos;
                matched = tag;
            }
        }
        if (first != std::string::npos) {
            bytes.erase(first, matched.size());
            continue;
        }

        size_t hold = 0;
        for (const auto tag : k_private_control_tags) {
            hold = std::max(hold, trailing_marker_prefix(bytes, tag));
        }
        if (hold != 0) {
            pending_.assign(bytes, bytes.size() - hold, hold);
            bytes.resize(bytes.size() - hold);
        }
        return;
    }
}

void server_rerot_control_tag_filter::reset() {
    pending_.clear();
}

server_rerot_stream_lines server_rerot_line_mux::append(
        llama_rerot_node_id node_id,
        llama_rerot_run_id run_id,
        std::string_view bytes,
        const llama_rerot_document & document) {
    server_rerot_stream_lines result;
    if (node_id == LLAMA_REROT_NODE_INVALID || run_id == LLAMA_REROT_RUN_INVALID) {
        result.ok = false;
        result.error = "invalid RERoT line-stream node/run";
        return result;
    }

    auto & lane = lanes_[node_id];
    if (!bytes.empty()) {
        if (!lane.blocked.empty() && lane.blocked.back().run_id == run_id) {
            lane.blocked.back().bytes.append(bytes);
        } else {
            lane.blocked.push_back({run_id, std::string(bytes)});
        }
    }
    return drain_lane(node_id, document, false);
}

server_rerot_stream_lines server_rerot_line_mux::drain(
        llama_rerot_node_id node_id,
        const llama_rerot_document & document) {
    return drain_lane(node_id, document, false);
}

server_rerot_stream_lines server_rerot_line_mux::finish(
        llama_rerot_node_id node_id,
        const llama_rerot_document & document) {
    return drain_lane(node_id, document, true);
}

bool server_rerot_line_mux::empty() const {
    return lanes_.empty();
}

server_rerot_stream_lines server_rerot_line_mux::drain_lane(
        llama_rerot_node_id node_id,
        const llama_rerot_document & document,
        bool finish) {
    server_rerot_stream_lines result;
    const auto lane_it = lanes_.find(node_id);
    if (lane_it == lanes_.end()) {
        return result;
    }

    auto & lane = lane_it->second;
    while (!lane.blocked.empty()) {
        const auto * run = document.run(lane.blocked.front().run_id);
        if (!run) {
            result.ok = false;
            result.error = "RERoT line stream references a missing run";
            return result;
        }
        if (run->visibility == llama_rerot_visibility::pending_record) {
            if (finish) {
                result.ok = false;
                result.error =
                    "RERoT Lane finished with unresolved streaming bytes: node=" +
                    std::to_string(node_id) +
                    " run=" + std::to_string(run->id) +
                    " blocked_segments=" + std::to_string(lane.blocked.size()) +
                    " blocked_bytes=" + std::to_string(lane.blocked.front().bytes.size()) +
                    " partial_bytes=" + std::to_string(lane.partial_line.size());
            }
            return result;
        }

        std::string bytes = std::move(lane.blocked.front().bytes);
        lane.blocked.pop_front();
        if (run->visibility == llama_rerot_visibility::private_control) {
            continue;
        }
        if (run->visibility != llama_rerot_visibility::public_live) {
            result.ok = false;
            result.error = "RERoT line stream encountered normal visibility";
            return result;
        }

        lane.partial_line += bytes;
        for (size_t newline = lane.partial_line.find('\n');
             newline != std::string::npos;
             newline = lane.partial_line.find('\n')) {
            std::string line = lane.partial_line.substr(0, newline + 1);
            lane.partial_line.erase(0, newline + 1);
            for (const std::string_view tag : {"<blockquote>", "</blockquote>"}) {
                for (size_t pos = line.find(tag); pos != std::string::npos; pos = line.find(tag)) {
                    line.erase(pos, tag.size());
                }
            }
            if (!line.empty() && line != "\n") {
                result.lines.push_back(std::move(line));
            }
        }
    }

    if (finish && !lane.partial_line.empty()) {
        for (const std::string_view tag : {"<blockquote>", "</blockquote>"}) {
            for (size_t pos = lane.partial_line.find(tag);
                 pos != std::string::npos;
                 pos = lane.partial_line.find(tag)) {
                lane.partial_line.erase(pos, tag.size());
            }
        }
        if (!lane.partial_line.empty() && !only_ascii_space(lane.partial_line)) {
            lane.partial_line.push_back('\n');
            result.lines.push_back(std::move(lane.partial_line));
        }
    }
    if (finish || (lane.blocked.empty() && lane.partial_line.empty())) {
        lanes_.erase(lane_it);
    }
    return result;
}

std::string_view server_rerot_planner_prompt() {
    static constexpr std::string_view prompt =
        "我按依赖把请求归成合理粒度的并行目标：独立才分成多个 li，有依赖就合在一个 li。每个 li 需要一段实质推理，只写具体目标，不展开答案；同一方案的并列要点归在一起。以 ol 开头、/ol 结尾；不能并行就一个 li。\n";
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
        llama_pos storage_pos_next,
        uint64_t requested_episode_id) {
    if (physical_slot < 0 || exec_seq < 0 || storage_pos_next < 0) {
        return 0;
    }

    release_slot(physical_slot);

    uint64_t episode_id = requested_episode_id;
    if (episode_id == 0) {
        episode_id = next_episode_id_++;
        if (episode_id == 0) {
            episode_id = next_episode_id_++;
        }
    } else if (episode_id >= next_episode_id_) {
        next_episode_id_ = episode_id + 1;
        if (next_episode_id_ == 0) {
            next_episode_id_ = 1;
        }
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
        // Rejected speculative tokens can leave storage-position gaps. A
        // logical pending record may therefore span several physical runs;
        // its resolution handles every pending run owned by this Lane.
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

bool server_rerot_runtime::resolve_pending_record_runs(
        server_rerot_episode & episode,
        server_rerot_node_runtime & node,
        llama_rerot_run_id current_run_id,
        llama_rerot_visibility replacement,
        uint64_t publish_epoch,
        uint64_t & resolved_tokens) {
    resolved_tokens = 0;
    if (!node.pending_record.has_value() ||
        *node.pending_record != current_run_id ||
        (replacement != llama_rerot_visibility::public_live &&
         replacement != llama_rerot_visibility::private_control) ||
        (replacement == llama_rerot_visibility::public_live) != (publish_epoch != 0)) {
        return fail_episode(episode, "invalid pending-record resolution request");
    }

    const auto * logical_node = episode.document.node(node.id);
    if (!logical_node) {
        return fail_episode(episode, "pending-record owner is missing");
    }

    std::vector<std::pair<llama_rerot_run_id, uint32_t>> pending_runs;
    bool found_current = false;
    for (const auto candidate_id : logical_node->runs) {
        const auto * candidate = episode.document.run(candidate_id);
        if (!candidate) {
            return fail_episode(episode, "pending-record run is missing");
        }
        if (candidate->visibility != llama_rerot_visibility::pending_record) {
            continue;
        }
        if (candidate->token_count == 0) {
            return fail_episode(episode, "pending-record run is empty");
        }
        pending_runs.emplace_back(candidate_id, candidate->token_count);
        found_current = found_current || candidate_id == current_run_id;
        resolved_tokens += candidate->token_count;
    }
    if (!found_current || pending_runs.empty() || episode.pending_tokens < resolved_tokens) {
        return fail_episode(episode, "pending-record run set is incomplete");
    }

    if (memory_) {
        const auto replacement_kv =
            replacement == llama_rerot_visibility::public_live
                ? LLAMA_REROT_KV_PUBLIC_LIVE
                : LLAMA_REROT_KV_PRIVATE_CONTROL;
        for (const auto & pending : pending_runs) {
            const size_t changed = llama_memory_rerot_reclassify_run(
                memory_, episode.id, pending.first,
                LLAMA_REROT_KV_PENDING_RECORD,
                replacement_kv,
                publish_epoch);
            if (changed != pending.second) {
                return fail_episode(
                    episode,
                    "failed to resolve every physical run in a pending RERoT record");
            }
        }
    }

    for (const auto & pending : pending_runs) {
        if (!episode.document.reclassify_run(
                pending.first,
                llama_rerot_visibility::pending_record,
                replacement,
                publish_epoch)) {
            return fail_episode(
                episode,
                "failed to resolve every logical run in a pending RERoT record");
        }
    }
    return true;
}

bool server_rerot_runtime::release_false_pending(
        server_rerot_episode & episode,
        server_rerot_node_runtime & node) {
    if (!node.pending_record.has_value()) {
        return fail_episode(episode, "RERoT parser released a missing pending run");
    }

    const auto run_id = *node.pending_record;
    const uint64_t publish_epoch = next_publish_epoch(episode);
    if (publish_epoch == 0) {
        return false;
    }

    uint64_t published_tokens = 0;
    if (!resolve_pending_record_runs(
            episode, node, run_id,
            llama_rerot_visibility::public_live,
            publish_epoch,
            published_tokens)) {
        return false;
    }
    episode.pending_tokens -= published_tokens;
    episode.generated_public_tokens += published_tokens;

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
    auto plans = plan_private_span(episode_id, node_id, storage_pos, 1);
    if (!plans.has_value()) {
        return std::nullopt;
    }
    return std::move(plans->front());
}

std::optional<std::vector<server_rerot_token_plan>> server_rerot_runtime::plan_private_span(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        llama_pos storage_pos,
        size_t token_count) {
    auto * current = episode(episode_id);
    auto * current_node = node(episode_id, node_id);
    if (!current || !current_node || current->hard_aborted || storage_pos < 0 ||
        token_count == 0 ||
        token_count > size_t(std::numeric_limits<llama_pos>::max() - storage_pos) + 1) {
        return std::nullopt;
    }

    const llama_rerot_run_id run_id = ensure_run(
        *current, *current_node, llama_rerot_visibility::private_control, storage_pos);
    if (run_id == LLAMA_REROT_RUN_INVALID) {
        return std::nullopt;
    }

    std::vector<server_rerot_token_plan> plans;
    plans.reserve(token_count);
    for (size_t i = 0; i < token_count; ++i) {
        server_rerot_token_plan plan;
        plan.storage_pos = storage_pos + static_cast<llama_pos>(i);
        plan.visibility = llama_rerot_visibility::private_control;
        plan.run_id = run_id;
        plans.push_back(std::move(plan));
    }
    return plans;
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
    plan.is_heading = true;
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
        // A node may fork via <ol>...</ol> OR conclude directly with </think>.
        // Both detectors track the token stream in parallel.
        marker_step = current_node->exit_parser.consume(token_bytes);
        if (marker_step.malformed) {
            fail_episode(*current, marker_step.error.empty()
                ? "malformed RERoT private end marker"
                : marker_step.error);
            return std::nullopt;
        }

        parser_step = current_node->parser.consume(token_bytes);
        if (parser_step.malformed) {
            // Free-form worker thought may contain nested lists or imperfect
            // markup. Never crash the whole episode with HTTP 500: disarm the
            // planner, release any pending bytes as public text, and let the
            // lane continue reasoning.
            current_node->planner_armed = false;
            if (current_node->pending_record.has_value() &&
                !release_false_pending(*current, *current_node)) {
                return std::nullopt;
            }
            parser_step.write_visibility = llama_rerot_visibility::public_live;
            parser_step.malformed = false;
        }

        const bool candidate_alive =
            current_node->exit_parser.state() == server_rerot_marker_state::marker_candidate ||
            current_node->parser.state() == server_rerot_parser_state::opening_candidate ||
            current_node->parser.state() == server_rerot_parser_state::list_pending;

        const bool release_pending =
            (marker_step.release_previous_pending || parser_step.release_previous_pending) &&
            !candidate_alive;

        if (release_pending) {
            parser_step.release_previous_pending = true;
            if (!release_false_pending(*current, *current_node)) {
                return std::nullopt;
            }
        }

        if (marker_step.marker_closed) {
            current_node->planner_armed = false;
            parser_step.write_visibility = marker_step.write_visibility;
        } else if (parser_step.record_closed) {
            // Completed <ol> record; visibility handled by parser_step
        } else if (candidate_alive) {
            parser_step.write_visibility = llama_rerot_visibility::pending_record;
        } else {
            parser_step.write_visibility = llama_rerot_visibility::public_live;
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

std::optional<server_rerot_token_plan> server_rerot_runtime::plan_serial_token(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        llama_pos storage_pos) {
    auto * current = episode(episode_id);
    auto * current_node = node(episode_id, node_id);
    if (!current || !current_node || current->hard_aborted || !current->serial_tail ||
        current->serial_node != node_id || storage_pos < 0) {
        return std::nullopt;
    }

    server_rerot_token_plan plan;
    plan.storage_pos = storage_pos;
    plan.visibility = llama_rerot_visibility::public_live;
    plan.run_id = ensure_run(*current, *current_node, plan.visibility, storage_pos);
    return plan.valid() ? std::optional<server_rerot_token_plan>(std::move(plan)) : std::nullopt;
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
        const server_rerot_token_plan & plan,
        size_t * span_count_out) {
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
        std::fprintf(stderr,
            "RERoT install failed: episode=%llu node=%u seq=%d run=%u stage=write_tag\n",
            static_cast<unsigned long long>(current->id),
            current_node->id, current_node->exec_seq, plan.run_id);
        return fail_episode(*current, "failed to install RERoT KV write tag");
    }

    std::vector<uint32_t> ordered_runs;
    llama_rerot_reader_view_desc desc = {};
    if (!build_reader_view_desc(*current, *current_node, plan.run_id, ordered_runs, desc)) {
        std::fprintf(stderr,
            "RERoT install failed: episode=%llu node=%u seq=%d run=%u stage=build_view\n",
            static_cast<unsigned long long>(current->id),
            current_node->id, current_node->exec_seq, plan.run_id);
        llama_memory_rerot_clear_write_tag(memory_, current_node->exec_seq);
        return fail_episode(*current, "failed to build RERoT reader view");
    }
    if (span_count_out) {
        *span_count_out = ordered_runs.size();
    }
    if (!llama_memory_rerot_set_reader_view(memory_, current_node->exec_seq, &desc)) {
        std::fprintf(stderr,
            "RERoT install failed: episode=%llu node=%u seq=%d run=%u stage=set_view ordered=%zu\n",
            static_cast<unsigned long long>(current->id),
            current_node->id, current_node->exec_seq, plan.run_id, ordered_runs.size());
        llama_memory_rerot_clear_write_tag(memory_, current_node->exec_seq);
        return fail_episode(*current, "failed to install RERoT reader view");
    }
    return true;
}

bool server_rerot_runtime::install_token_plan_write_only(
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

    llama_memory_rerot_clear_reader_view(memory_, current_node->exec_seq);
    const llama_rerot_kv_write_tag tag = {
        current->id,
        current_node->id,
        plan.run_id,
        plan.visibility == llama_rerot_visibility::public_live ? current->publish_epoch : 0,
        current->frontier,
        visibility,
    };
    if (!llama_memory_rerot_set_write_tag(memory_, current_node->exec_seq, &tag)) {
        return fail_episode(*current, "failed to install RERoT write-only tag");
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
        child.enqueue_frontier = episode.frontier;
        episode.nodes.push_back(std::move(child));
        episode.ready_queue.push_back(child_id);
        if (episode.queue_peak < (uint64_t) episode.ready_queue.size()) {
            episode.queue_peak = (uint64_t) episode.ready_queue.size();
        }
    }

    return true;
}

bool server_rerot_runtime::publish_pending_run(
        server_rerot_episode & episode,
        server_rerot_node_runtime & node,
        llama_rerot_run_id run_id,
        bool topology_change) {
    const uint64_t publish_epoch = next_publish_epoch(episode);
    if (publish_epoch == 0) {
        return false;
    }

    uint64_t published_tokens = 0;
    if (!resolve_pending_record_runs(
            episode, node, run_id,
            llama_rerot_visibility::public_live,
            publish_epoch,
            published_tokens)) {
        return false;
    }
    episode.pending_tokens -= published_tokens;
    episode.generated_public_tokens += published_tokens;

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
    uint64_t privatized_tokens = 0;
    if (!resolve_pending_record_runs(
            episode, node, run_id,
            llama_rerot_visibility::private_control,
            0,
            privatized_tokens)) {
        return false;
    }
    episode.pending_tokens -= privatized_tokens;
    episode.generated_private_tokens += privatized_tokens;

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

    // Episode-level hard-resource accounting (§20). Planner injection,
    // headings, and model tokens all draw from one global budget.
    if (plan.is_heading) {
        ++current->forced_heading_tokens;
    }
    switch (plan.visibility) {
        case llama_rerot_visibility::public_live:     ++current->generated_public_tokens;  break;
        case llama_rerot_visibility::private_control: ++current->generated_private_tokens; break;
        case llama_rerot_visibility::pending_record:  ++current->pending_tokens;            break;
        case llama_rerot_visibility::normal: break;
    }
    if (check_hard_limits(*current)) {
        return false;
    }

    if (plan.parser_step.record_closed) {
        if (!publish_pending_record(
                *current, *current_node, plan.run_id, plan.parser_step.items)) {
            return false;
        }
        // A fork can cross the node/queue budget even though the closing token
        // itself was within budget. Never start the new topology half-funded.
        if (check_hard_limits(*current)) {
            return false;
        }
        return true;
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
        return fail_episode(episode, "rerot_resource_exhausted: RERoT internal seq-id arena exhausted while creating archive");
    }
    episode.archive_seq = *archive_seq;

    if (memory_ && episode.base_prefix_end > 0) {
        llama_memory_seq_cp_attention(
            memory_, source_seq, episode.archive_seq, 0, episode.base_prefix_end);
    }
    return true;
}

bool server_rerot_runtime::sync_public_archive(
        uint64_t episode_id,
        std::vector<llama_seq_id> * semantic_seq_ids_out) {
    auto * current = episode(episode_id);
    if (!current || current->hard_aborted) {
        return false;
    }

    if (current->archive_seq < 0) {
        const auto source = std::find_if(
            current->nodes.begin(), current->nodes.end(),
            [](const server_rerot_node_runtime & node) {
                return node.exec_seq >= 0;
            });
        if (source == current->nodes.end() ||
            !ensure_archive_seq(*current, source->exec_seq)) {
            return false;
        }
    }

    if (semantic_seq_ids_out) {
        semantic_seq_ids_out->clear();
        semantic_seq_ids_out->push_back(current->archive_seq);
    }
    for (const auto & node : current->nodes) {
        if (node.exec_seq >= 0) {
            archive_public_runs(*current, node, current->archive_seq);
            if (semantic_seq_ids_out) {
                semantic_seq_ids_out->push_back(node.exec_seq);
            }
        }
    }
    if (semantic_seq_ids_out) {
        std::sort(semantic_seq_ids_out->begin(), semantic_seq_ids_out->end());
        semantic_seq_ids_out->erase(
            std::unique(semantic_seq_ids_out->begin(), semantic_seq_ids_out->end()),
            semantic_seq_ids_out->end());
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
            return fail_episode(*current, "rerot_resource_exhausted: RERoT internal seq-id arena exhausted during fork");
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

    // Global FIFO (§16.2): the key is (enqueue_frontier, tree_path). No
    // scoring, no priority, no preemption, and no overtake by later subtrees.
    auto best_it = current->ready_queue.begin();
    if (!node(episode_id, *best_it)) {
        return fail_episode(*current, "RERoT ready queue contains an invalid parked child");
    }
    {
        uint64_t best_frontier = node(episode_id, *best_it)->enqueue_frontier;
        std::vector<uint32_t> best_path = current->document.tree_path(*best_it);
        for (auto it = best_it + 1; it != current->ready_queue.end(); ++it) {
            if (!node(episode_id, *it)) {
                continue;
            }
            const uint64_t frontier = node(episode_id, *it)->enqueue_frontier;
            std::vector<uint32_t> path = current->document.tree_path(*it);
            if (frontier < best_frontier ||
                (frontier == best_frontier && path < best_path)) {
                best_it = it;
                best_frontier = frontier;
                best_path = std::move(path);
            }
        }
    }
    const auto child_id = *best_it;
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

    current->ready_queue.erase(best_it);
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

    // Once the serial tail owns decode (§22) the parallel scheduler is off.
    if (current->serial_tail) {
        result.completed_frontier = current->frontier;
        return result;
    }

    // A hard-aborted episode never runs a partial frontier subset (§§19-20):
    // no retirements, no survivor, no topology work.
    if (current->hard_aborted) {
        result.completed_frontier = current->frontier;
        result.hard_aborted = true;
        result.abort_reason = current->abort_reason;
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
    // Preserve the first failure. Follow-up cleanup paths commonly observe the
    // failed operation and call hard_abort() again with a generic wrapper;
    // replacing the original parser/backend/resource cause makes diagnosis
    // impossible and can repeat destructive sequence cleanup.
    if (episode.hard_aborted) {
        return false;
    }

    // Episode-level HARD_ABORT (§20): cancel RUNNING and STARTING, drop QUEUED
    // descriptors, release parked/archive refs, choose no survivor, emit no
    // answer. The logical tree is kept for diagnostics until erase_episode.
    episode.hard_aborted = true;
    episode.finalizing = false;
    episode.fence_refreshed = false;
    episode.serial_tail = false;
    episode.serial_node = LLAMA_REROT_NODE_INVALID;
    episode.abort_reason = std::move(reason);
    episode.ready_queue.clear();
    episode.starting.clear();
    episode.running.clear();
    episode.forked_this_frontier.clear();
    episode.topology_barrier_pending = false;
    for (auto & current_node : episode.nodes) {
        current_node.exit_intent = false;
        if (current_node.exec_seq >= 0) {
            clear_sequence_control(current_node.exec_seq);
            if (memory_) {
                llama_memory_seq_rm(memory_, current_node.exec_seq, -1, -1);
            }
            current_node.exec_seq = -1;
        }
        if (current_node.parked_seq >= 0) {
            if (memory_) {
                llama_memory_seq_rm(memory_, current_node.parked_seq, -1, -1);
            }
            free_internal_seq(current_node.parked_seq);
            current_node.parked_seq = -1;
        }
        if (current_node.physical_slot >= 0) {
            const auto slot_it = slot_to_episode_.find(current_node.physical_slot);
            if (slot_it != slot_to_episode_.end() && slot_it->second == episode.id) {
                slot_to_episode_.erase(slot_it);
            }
            current_node.physical_slot = -1;
        }
    }
    if (episode.archive_seq >= 0) {
        if (memory_) {
            llama_memory_seq_rm(memory_, episode.archive_seq, -1, -1);
        }
        free_internal_seq(episode.archive_seq);
        episode.archive_seq = -1;
    }
    return false;
}

bool server_rerot_runtime::check_hard_limits(server_rerot_episode & episode) {
    const auto & limits = episode.hard_limits;
    // Visibility counters are mutually exclusive. Forced headings are tracked
    // as a diagnostic subset of PENDING tokens, so adding that counter again
    // would charge every heading twice.
    const uint64_t total_tokens = episode.generated_public_tokens +
                                  episode.generated_private_tokens +
                                  episode.pending_tokens;
    if (limits.max_total_tokens != 0 && total_tokens > limits.max_total_tokens) {
        fail_episode(episode, "rerot_resource_exhausted: episode token budget exceeded");
        return true;
    }
    if (limits.max_nodes != 0 && episode.document.node_count() > limits.max_nodes) {
        fail_episode(episode, "rerot_resource_exhausted: episode node budget exceeded");
        return true;
    }
    if (limits.max_queue_descriptors != 0 && episode.queue_peak > limits.max_queue_descriptors) {
        fail_episode(episode, "rerot_resource_exhausted: episode queue descriptor budget exceeded");
        return true;
    }
    if (limits.max_frontiers != 0 && episode.frontier > limits.max_frontiers) {
        fail_episode(episode, "rerot_resource_exhausted: episode frontier budget exceeded");
        return true;
    }
    return false;
}

bool server_rerot_runtime::hard_abort(uint64_t episode_id, std::string reason) {
    auto * current = episode(episode_id);
    return current ? fail_episode(*current, std::move(reason)) : false;
}

bool server_rerot_runtime::begin_frontier(uint64_t episode_id) {
    auto * current = episode(episode_id);
    if (!current || current->hard_aborted || current->serial_tail) {
        return false;
    }
    // One frontier is one atomic budget unit (§19.3): never run a partial Lane
    // subset when the episode already crossed a hard limit.
    if (check_hard_limits(*current)) {
        return false;
    }
    return true;
}

void server_rerot_runtime::set_hard_limits(uint64_t episode_id, server_rerot_hard_limits limits) {
    auto * current = episode(episode_id);
    if (!current) {
        return;
    }
    current->hard_limits = limits;
    check_hard_limits(*current);
}

bool server_rerot_runtime::refresh_final_fence(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        std::vector<uint32_t> * ordered_runs_out) {
    auto * current = episode(episode_id);
    auto * survivor = node(episode_id, node_id);
    if (!current || !survivor) {
        return false;
    }
    if (current->hard_aborted || !current->finalizing || current->serial_tail) {
        return false;
    }
    // The natural-final preconditions (§21.2) must still hold on stable shared
    // memory: nothing queued or starting, exactly this Lane still RUNNING.
    if (!current->ready_queue.empty() || !current->starting.empty() ||
        current->running.size() != 1 || current->running.count(node_id) != 1 ||
        !survivor->exit_intent || survivor->exec_seq < 0) {
        return fail_episode(*current, "RERoT final fence lost its single-survivor precondition");
    }
    // Stable-view re-evaluation (§21.4): the fence decode observes every
    // public write committed through the final frontier, including the last
    // writes of already-retired sibling Lanes.
    const auto view = current->document.build_view(node_id);
    if (ordered_runs_out) {
        ordered_runs_out->clear();
        ordered_runs_out->reserve(view.runs.size());
        for (const auto & view_run : view.runs) {
            ordered_runs_out->push_back(view_run.run_id);
        }
    }
    if (memory_) {
        const auto * doc_node = current->document.node(node_id);
        if (!doc_node || doc_node->runs.empty()) {
            return fail_episode(*current, "RERoT final survivor has no query run");
        }
        std::vector<uint32_t> ordered_runs;
        llama_rerot_reader_view_desc desc = {};
        if (!build_reader_view_desc(*current, *survivor, doc_node->runs.back(), ordered_runs, desc) ||
            !llama_memory_rerot_set_reader_view(memory_, survivor->exec_seq, &desc)) {
            return fail_episode(*current, "failed to install RERoT final fence view");
        }
    }
    current->fence_refreshed = true;
    return true;
}

bool server_rerot_runtime::complete_serial_tail(uint64_t episode_id, llama_rerot_node_id node_id) {
    auto * current = episode(episode_id);
    auto * survivor = node(episode_id, node_id);
    if (!current || !survivor) {
        return false;
    }
    if (current->hard_aborted || !current->finalizing || !current->fence_refreshed ||
        current->serial_tail || current->running.size() != 1 ||
        current->running.count(node_id) != 1) {
        return current ? fail_episode(*current, "invalid RERoT serial tail transition") : false;
    }
    current->serial_tail = true;
    current->serial_node = node_id;
    return true;
}

int server_rerot_runtime::response_task_id(uint64_t episode_id) const {
    const auto * current = episode(episode_id);
    return current ? current->response_task_id : -1;
}

llama_rerot_frontier_mode server_rerot_runtime::frontier_mode() const {
    return frontier_mode_;
}

// ---------------------------------------------------------------------------
// Versioned episode persistence + shared-memory log truncation (§§25,A.8-A.10)
// ---------------------------------------------------------------------------
//
// Physical KV ownership stays in llama_kv_cells and PAC-DFS spans are always
// re-derived via document.build_view (validated on every load), so this codec
// persists logical ownership only. Frontier mode is intentionally NOT stored:
// it is a runtime/view-installation property, and a restored episode adopts
// the loading runtime's mode with views reinstalled before decode.

namespace {

bool rerot_state_set_error(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
    return false;
}

constexpr uint32_t k_rerot_known_caps =
    LLAMA_REROT_STATE_CAP_REROT | LLAMA_REROT_STATE_CAP_REROT_TREE |
    LLAMA_REROT_STATE_CAP_REROT_PRIVATE | LLAMA_REROT_STATE_CAP_REROT_MTP |
    LLAMA_REROT_STATE_CAP_HYBRID_REC | LLAMA_REROT_STATE_CAP_SPARSE_KV |
    LLAMA_REROT_STATE_CAP_TRIATTENTION;

constexpr uint8_t k_rerot_flag_barrier   = 1u << 0;
constexpr uint8_t k_rerot_flag_finalize  = 1u << 1;
constexpr uint8_t k_rerot_flag_aborted   = 1u << 2;
constexpr uint8_t k_rerot_flag_fence     = 1u << 3;
constexpr uint8_t k_rerot_flag_serial    = 1u << 4;

struct rerot_blob_writer {
    std::vector<uint8_t> buf;
    void u8(uint8_t v) { buf.push_back(v); }
    void u32(uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            buf.push_back(static_cast<uint8_t>(v >> (8 * i)));
        }
    }
    void u64(uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            buf.push_back(static_cast<uint8_t>(v >> (8 * i)));
        }
    }
    void i32(int32_t v) { u32(static_cast<uint32_t>(v)); }
    void str(const std::string & s) {
        u64(static_cast<uint64_t>(s.size()));
        buf.insert(buf.end(), s.begin(), s.end());
    }
    void blob(const std::vector<uint8_t> & v) {
        u64(static_cast<uint64_t>(v.size()));
        buf.insert(buf.end(), v.begin(), v.end());
    }
    void str_vec(const std::vector<std::string> & v) {
        u32(static_cast<uint32_t>(v.size()));
        for (const auto & s : v) {
            str(s);
        }
    }
};

struct rerot_blob_reader {
    const uint8_t * p = nullptr;
    size_t n = 0;
    size_t off = 0;
    bool ok = true;

    bool need(size_t k) {
        if (!ok || k > n - off) {
            ok = false;
            return false;
        }
        return true;
    }
    uint8_t u8() {
        uint8_t v = 0;
        if (need(1)) {
            v = p[off];
            off += 1;
        }
        return v;
    }
    uint32_t u32() {
        uint32_t v = 0;
        if (need(4)) {
            for (int i = 0; i < 4; ++i) {
                v |= static_cast<uint32_t>(p[off + i]) << (8 * i);
            }
            off += 4;
        }
        return v;
    }
    uint64_t u64() {
        uint64_t v = 0;
        if (need(8)) {
            for (int i = 0; i < 8; ++i) {
                v |= static_cast<uint64_t>(p[off + i]) << (8 * i);
            }
            off += 8;
        }
        return v;
    }
    int32_t i32() { return static_cast<int32_t>(u32()); }
    std::string str() {
        std::string s;
        const uint64_t len = u64();
        if (!ok || len > n - off) {
            ok = false;
            return s;
        }
        s.assign(reinterpret_cast<const char *>(p + off), static_cast<size_t>(len));
        off += static_cast<size_t>(len);
        return s;
    }
    std::vector<uint8_t> blob() {
        std::vector<uint8_t> v;
        const uint64_t len = u64();
        if (!ok || len > n - off) {
            ok = false;
            return v;
        }
        v.assign(p + off, p + off + static_cast<size_t>(len));
        off += static_cast<size_t>(len);
        return v;
    }
};

uint8_t rerot_visibility_to_u8(llama_rerot_visibility v) {
    return static_cast<uint8_t>(v);
}

bool rerot_visibility_from_u8(uint8_t v, llama_rerot_visibility & out) {
    if (v != 1 && v != 2 && v != 3) {
        return false;
    }
    out = static_cast<llama_rerot_visibility>(v);
    return true;
}

bool rerot_node_state_from_u8(uint8_t v, llama_rerot_node_state & out) {
    if (v > static_cast<uint8_t>(llama_rerot_node_state::retired)) {
        return false;
    }
    out = static_cast<llama_rerot_node_state>(v);
    return true;
}

void rerot_write_planner_snapshot(rerot_blob_writer & w, const server_rerot_planner_snapshot & s) {
    w.u8(static_cast<uint8_t>(s.state));
    w.str(s.opener_candidate);
    w.str(s.list_bytes);
    w.str_vec(s.items);
    w.str(s.error);
}

bool rerot_read_planner_snapshot(rerot_blob_reader & r, server_rerot_planner_snapshot & s) {
    const uint8_t state = r.u8();
    std::string opener = r.str();
    std::string list = r.str();
    const uint32_t n_items = r.u32();
    std::vector<std::string> items;
    if (!r.ok || n_items > (r.n - r.off)) {
        r.ok = false;
        return false;
    }
    items.reserve(n_items);
    for (uint32_t i = 0; i < n_items; ++i) {
        items.push_back(r.str());
        if (!r.ok) {
            return false;
        }
    }
    std::string error = r.str();
    if (!r.ok || state > static_cast<uint8_t>(server_rerot_parser_state::failed)) {
        r.ok = false;
        return false;
    }
    s.state = static_cast<server_rerot_parser_state>(state);
    s.opener_candidate = std::move(opener);
    s.list_bytes = std::move(list);
    s.items = std::move(items);
    s.error = std::move(error);
    return true;
}

void rerot_write_marker_snapshot(rerot_blob_writer & w, const server_rerot_marker_snapshot & s) {
    w.str(s.marker);
    w.str(s.candidate);
    w.u8(static_cast<uint8_t>(s.state));
    w.str(s.error);
}

bool rerot_read_marker_snapshot(rerot_blob_reader & r, server_rerot_marker_snapshot & s) {
    std::string marker = r.str();
    std::string candidate = r.str();
    const uint8_t state = r.u8();
    std::string error = r.str();
    if (!r.ok || state > static_cast<uint8_t>(server_rerot_marker_state::failed)) {
        r.ok = false;
        return false;
    }
    s.marker = std::move(marker);
    s.candidate = std::move(candidate);
    s.state = static_cast<server_rerot_marker_state>(state);
    s.error = std::move(error);
    return true;
}

} // namespace

server_rerot_planner_snapshot server_rerot_planner_parser::snapshot() const {
    server_rerot_planner_snapshot s;
    s.state = state_;
    s.opener_candidate = opener_candidate_;
    s.list_bytes = list_bytes_;
    s.items = items_;
    s.error = error_;
    return s;
}

bool server_rerot_planner_parser::restore(const server_rerot_planner_snapshot & snap, std::string * error) {
    if (snap.state > server_rerot_parser_state::failed) {
        return rerot_state_set_error(error, "RERoT planner snapshot has an out-of-range state");
    }
    if (snap.opener_candidate.size() > k_open_ol.size() - 1) {
        return rerot_state_set_error(error, "RERoT planner snapshot has an oversized opener candidate");
    }
    if (!snap.opener_candidate.empty() &&
        k_open_ol.substr(0, snap.opener_candidate.size()) != snap.opener_candidate) {
        return rerot_state_set_error(error, "RERoT planner snapshot opener is not a prefix of <ol>");
    }
    if (snap.state == server_rerot_parser_state::opening_candidate && snap.opener_candidate.empty()) {
        return rerot_state_set_error(error, "RERoT planner snapshot is candidate-armed without candidate bytes");
    }
    if (snap.state == server_rerot_parser_state::public_text && !snap.opener_candidate.empty()) {
        return rerot_state_set_error(error, "RERoT planner snapshot leaves residue candidate bytes in public text");
    }
    const bool listed = snap.state == server_rerot_parser_state::list_pending ||
                        snap.state == server_rerot_parser_state::complete ||
                        snap.state == server_rerot_parser_state::failed;
    if (listed) {
        if (snap.list_bytes.size() < k_open_ol.size() ||
            std::string_view(snap.list_bytes).substr(0, k_open_ol.size()) != k_open_ol) {
            return rerot_state_set_error(error, "RERoT planner snapshot list state lost the <ol> boundary");
        }
    } else if (!snap.list_bytes.empty()) {
        return rerot_state_set_error(error, "RERoT planner snapshot leaves residue list bytes before <ol>");
    }
    if (snap.state == server_rerot_parser_state::complete && snap.items.empty()) {
        return rerot_state_set_error(error, "RERoT planner snapshot is complete without items");
    }
    if (snap.state == server_rerot_parser_state::failed && snap.error.empty()) {
        return rerot_state_set_error(error, "RERoT planner snapshot failed without an error");
    }
    state_ = snap.state;
    opener_candidate_ = snap.opener_candidate;
    list_bytes_ = snap.list_bytes;
    items_ = snap.items;
    error_ = snap.error;
    return true;
}

const std::string & server_rerot_marker_parser::marker() const {
    return marker_;
}

server_rerot_marker_snapshot server_rerot_marker_parser::snapshot() const {
    server_rerot_marker_snapshot s;
    s.marker = marker_;
    s.candidate = candidate_;
    s.state = state_;
    s.error = error_;
    return s;
}

bool server_rerot_marker_parser::restore(const server_rerot_marker_snapshot & snap, std::string * error) {
    if (snap.marker.empty()) {
        return rerot_state_set_error(error, "RERoT marker snapshot has an empty marker");
    }
    if (snap.state > server_rerot_marker_state::failed) {
        return rerot_state_set_error(error, "RERoT marker snapshot has an out-of-range state");
    }
    if (snap.candidate.size() >= snap.marker.size()) {
        return rerot_state_set_error(error, "RERoT marker snapshot has an oversized marker candidate");
    }
    if (!snap.candidate.empty() &&
        snap.marker.substr(0, snap.candidate.size()) != snap.candidate) {
        return rerot_state_set_error(error, "RERoT marker snapshot candidate is not a prefix of the marker");
    }
    if (snap.state == server_rerot_marker_state::marker_candidate && snap.candidate.empty()) {
        return rerot_state_set_error(error, "RERoT marker snapshot is candidate-armed without candidate bytes");
    }
    if ((snap.state == server_rerot_marker_state::public_text ||
         snap.state == server_rerot_marker_state::complete) &&
        !snap.candidate.empty()) {
        return rerot_state_set_error(error, "RERoT marker snapshot leaves a residue candidate");
    }
    if (snap.state == server_rerot_marker_state::failed && snap.error.empty()) {
        return rerot_state_set_error(error, "RERoT marker snapshot failed without an error");
    }
    marker_ = snap.marker;
    candidate_ = snap.candidate;
    state_ = snap.state;
    error_ = snap.error;
    return true;
}

void server_rerot_episode_demote_to_logical(server_rerot_episode & episode) {
    // Episode-level demotion (A.8.2): transient physical slot bindings go,
    // logical membership and lineage stay for re-admission.
    for (auto & node : episode.nodes) {
        node.physical_slot = -1;
    }
}

std::vector<uint8_t> server_rerot_episode_save(
        const server_rerot_episode & episode,
        const server_rerot_state_fingerprints & fp,
        std::string * error_out) {
    if (episode.id == 0) {
        rerot_state_set_error(error_out, "RERoT episode save refused: episode id is zero");
        return {};
    }
    if ((fp.caps & ~k_rerot_known_caps) != 0 ||
        (fp.caps & LLAMA_REROT_STATE_CAP_REROT) == 0) {
        rerot_state_set_error(
            error_out,
            "RERoT episode save refused: capability bitmap is unknown or lacks the REROT bit");
        return {};
    }
    if (episode.nodes.size() != episode.document.node_count()) {
        rerot_state_set_error(error_out, "RERoT episode save refused: node runtime/document alignment lost");
        return {};
    }
    for (size_t i = 0; i < episode.nodes.size(); ++i) {
        if (episode.nodes[i].id != static_cast<llama_rerot_node_id>(i)) {
            rerot_state_set_error(error_out, "RERoT episode save refused: node runtime ids are not dense");
            return {};
        }
        const auto & node = episode.nodes[i];
        const bool bound = node.physical_slot >= 0;
        const bool member = episode.running.count(node.id) != 0 || episode.starting.count(node.id) != 0;
        if (bound && !member) {
            rerot_state_set_error(error_out, "RERoT episode save refused: transient slot binding outside running/starting "
                "(detach the node or complete admission first; refusing a partial snapshot)");
            return {};
        }
        if (episode.document.node(node.id) == nullptr) {
            rerot_state_set_error(error_out, "RERoT episode save refused: runtime node without document state");
            return {};
        }
    }
    if (episode.nodes.size() > std::numeric_limits<uint32_t>::max() ||
        episode.document.run_count() > std::numeric_limits<uint32_t>::max()) {
        rerot_state_set_error(error_out, "RERoT episode save refused: topology exceeds the u32 codec");
        return {};
    }
    for (size_t i = 0; i < episode.document.run_count(); ++i) {
        const auto * run = episode.document.run(static_cast<llama_rerot_run_id>(i));
        if (run->visibility == llama_rerot_visibility::normal) {
            rerot_state_set_error(error_out, "RERoT episode save refused: run " + std::to_string(i) +
                " is untagged (normal visibility); only classified public/private/pending runs persist");
            return {};
        }
    }

    rerot_blob_writer w;
    w.u32(LLAMA_REROT_STATE_MAGIC);
    w.u32(LLAMA_REROT_STATE_VERSION);
    w.u32(fp.caps);
    w.u64(fp.model_fp);
    w.u64(fp.rope_fp);
    w.u64(fp.tri_fp);
    w.u64(episode.id);
    w.u64(episode.frontier);
    w.u64(episode.publish_epoch);
    w.u64(episode.topology_epoch);
    w.u64(episode.layout_epoch);
    w.i32(episode.base_prefix_end);
    w.i32(episode.root_task_id);
    w.i32(episode.response_task_id);
    w.i32(episode.archive_seq);
    uint8_t flags = 0;
    if (episode.topology_barrier_pending) { flags |= k_rerot_flag_barrier; }
    if (episode.finalizing)               { flags |= k_rerot_flag_finalize; }
    if (episode.hard_aborted)             { flags |= k_rerot_flag_aborted; }
    if (episode.fence_refreshed)          { flags |= k_rerot_flag_fence; }
    if (episode.serial_tail)              { flags |= k_rerot_flag_serial; }
    w.u8(flags);
    w.u32(episode.serial_node);
    w.str(episode.abort_reason);
    w.u64(episode.generated_public_tokens);
    w.u64(episode.generated_private_tokens);
    w.u64(episode.forced_heading_tokens);
    w.u64(episode.pending_tokens);
    w.u64(episode.queue_peak);
    w.u64(episode.hard_limits.max_total_tokens);
    w.u64(episode.hard_limits.max_nodes);
    w.u64(episode.hard_limits.max_queue_descriptors);
    w.u64(episode.hard_limits.max_frontiers);

    const auto write_id_vec = [&](const auto & c) {
        w.u32(static_cast<uint32_t>(c.size()));
        for (const auto id : c) {
            w.u32(id);
        }
    };
    write_id_vec(episode.forked_this_frontier);
    w.u32(static_cast<uint32_t>(episode.ready_queue.size()));
    for (const auto id : episode.ready_queue) {
        w.u32(id);
    }
    write_id_vec(episode.running);
    write_id_vec(episode.starting);

    w.u32(static_cast<uint32_t>(episode.document.node_count()));
    for (size_t i = 0; i < episode.document.node_count(); ++i) {
        const auto * node = episode.document.node(static_cast<llama_rerot_node_id>(i));
        w.u32(node->id);
        w.u32(node->parent);
        w.u32(node->depth);
        w.u32(node->child_index);
        w.u8(static_cast<uint8_t>(node->state));
        w.str(node->title);
        w.u32(static_cast<uint32_t>(node->children.size()));
        for (const auto child : node->children) {
            w.u32(child);
        }
        w.u32(static_cast<uint32_t>(node->runs.size()));
        for (const auto run : node->runs) {
            w.u32(run);
        }
    }

    w.u32(static_cast<uint32_t>(episode.document.run_count()));
    for (size_t i = 0; i < episode.document.run_count(); ++i) {
        const auto * run = episode.document.run(static_cast<llama_rerot_run_id>(i));
        w.u32(run->id);
        w.u32(run->owner);
        w.u8(rerot_visibility_to_u8(run->visibility));
        w.i32(run->storage_pos0);
        w.u32(run->token_count);
        w.u64(run->publish_epoch);
    }

    w.u32(static_cast<uint32_t>(episode.nodes.size()));
    for (const auto & node : episode.nodes) {
        w.u32(node.id);
        w.i32(node.physical_slot);
        w.i32(node.exec_seq);
        w.i32(node.parked_seq);
        w.i32(node.storage_pos_next);
        const std::optional<llama_rerot_run_id> refs[3] = {node.public_run, node.private_run, node.pending_record};
        for (const auto & ref : refs) {
            w.u8(ref.has_value() ? 1 : 0);
            w.u32(ref.has_value() ? *ref : 0);
        }
        rerot_write_planner_snapshot(w, node.parser.snapshot());
        rerot_write_marker_snapshot(w, node.exit_parser.snapshot());
        w.u8(node.planner_armed ? 1 : 0);
        w.u64(node.enqueue_frontier);
        w.u8(node.exit_intent ? 1 : 0);
        w.blob(node.sampler_blob);
        w.blob(node.mtp_blob);
        w.u64(node.view_stamp.topology_epoch);
        w.u64(node.view_stamp.publish_epoch);
        w.u64(node.view_stamp.layout_epoch);
    }
    return std::move(w.buf);
}

namespace {

struct rerot_node_blob {
    llama_rerot_node_id id = LLAMA_REROT_NODE_INVALID;
    llama_rerot_node_id parent = LLAMA_REROT_NODE_INVALID;
    uint32_t depth = 0;
    uint32_t child_index = 0;
    llama_rerot_node_state state = llama_rerot_node_state::planning;
    std::string title;
    std::vector<llama_rerot_node_id> children;
    std::vector<llama_rerot_run_id> runs;
};

struct rerot_run_blob {
    llama_rerot_run_id id = LLAMA_REROT_RUN_INVALID;
    llama_rerot_node_id owner = LLAMA_REROT_NODE_INVALID;
    llama_rerot_visibility visibility = llama_rerot_visibility::public_live;
    llama_pos storage_pos0 = 0;
    uint32_t token_count = 0;
    uint64_t publish_epoch = 0;
};

struct rerot_runtime_blob {
    llama_rerot_node_id id = LLAMA_REROT_NODE_INVALID;
    int physical_slot = -1;
    llama_seq_id exec_seq = -1;
    llama_seq_id parked_seq = -1;
    llama_pos storage_pos_next = 0;
    std::optional<llama_rerot_run_id> refs[3];
    server_rerot_planner_snapshot planner;
    server_rerot_marker_snapshot marker;
    bool planner_armed = true;
    uint64_t enqueue_frontier = 0;
    bool exit_intent = false;
    std::vector<uint8_t> sampler_blob;
    std::vector<uint8_t> mtp_blob;
    llama_rerot_view_stamp view_stamp = {0, 0, 0};
};

} // namespace

bool server_rerot_episode_load(
        const uint8_t * data,
        size_t size,
        const server_rerot_state_fingerprints & expected_fp,
        server_rerot_episode * episode_out,
        std::string * error_out) {
    if (data == nullptr || size == 0 || episode_out == nullptr) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: empty blob or null target");
    }
    rerot_blob_reader r{data, size, 0, true};
    const uint32_t magic = r.u32();
    if (!r.ok) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: blob too short for a state header");
    }
    if (magic != LLAMA_REROT_STATE_MAGIC) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: magic mismatch "
            "(not a RERoT episode blob; old-binary-new-state must never be best-effort)");
    }
    const uint32_t version = r.u32();
    if (version != LLAMA_REROT_STATE_VERSION) {
        if (version > LLAMA_REROT_STATE_VERSION) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: old binary cannot read new RERoT state "
                "(blob v" + std::to_string(version) + ", binary v" + std::to_string(LLAMA_REROT_STATE_VERSION) +
                "); upgrade the binary, never best-effort restore");
        }
        return rerot_state_set_error(error_out, "RERoT episode load refused: legacy RERoT state v" +
            std::to_string(version) + " is unsupported; refusing best-effort upgrade");
    }
    const uint32_t caps = r.u32();
    const uint64_t model_fp = r.u64();
    const uint64_t rope_fp = r.u64();
    const uint64_t tri_fp = r.u64();
    if (!r.ok) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: truncated state header");
    }
    if ((caps & ~k_rerot_known_caps) != 0) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: unknown state capability bits "
            "(blob needs a newer feature set); upgrade the binary");
    }
    if ((caps & LLAMA_REROT_STATE_CAP_REROT) == 0) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: blob lacks the REROT capability bit");
    }
    if (caps != expected_fp.caps) {
        return rerot_state_set_error(
            error_out,
            "RERoT episode load refused: state capability bitmap mismatch "
            "(saved mechanisms differ from the active runtime); refusing best-effort restore");
    }
    if (model_fp != expected_fp.model_fp) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: model fingerprint mismatch "
            "(different arch or context-train size); refusing best-effort restore");
    }
    if (rope_fp != expected_fp.rope_fp) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: RoPE fingerprint mismatch "
            "(different rope base/scale); refusing best-effort restore");
    }
    if (tri_fp != expected_fp.tri_fp) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: Tri-calibration fingerprint mismatch "
            "(different Tri ratio/enabled state); refusing best-effort restore");
    }

    const uint64_t episode_id = r.u64();
    const uint64_t frontier = r.u64();
    const uint64_t publish_epoch = r.u64();
    const uint64_t topology_epoch = r.u64();
    const uint64_t layout_epoch = r.u64();
    const llama_pos base_prefix_end = r.i32();
    const int root_task_id = r.i32();
    const int response_task_id = r.i32();
    const llama_seq_id archive_seq = r.i32();
    const uint8_t flags = r.u8();
    const llama_rerot_node_id serial_node = r.u32();
    std::string abort_reason = r.str();
    const uint64_t gen_public = r.u64();
    const uint64_t gen_private = r.u64();
    const uint64_t forced_heading = r.u64();
    const uint64_t pending_tokens = r.u64();
    const uint64_t queue_peak = r.u64();
    server_rerot_hard_limits limits;
    limits.max_total_tokens = r.u64();
    limits.max_nodes = r.u64();
    limits.max_queue_descriptors = r.u64();
    limits.max_frontiers = r.u64();
    if (!r.ok) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: truncated episode scalars");
    }
    if (episode_id == 0 || frontier == 0 || publish_epoch == 0 || topology_epoch == 0 || layout_epoch == 0) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: zero episode id or epoch");
    }
    if (base_prefix_end < 0 || (flags & 0xE0u) != 0) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: corrupt prefix end or flag bits");
    }

    const auto read_id_vec = [&](std::vector<llama_rerot_node_id> & out) {
        const uint32_t n = r.u32();
        if (!r.ok || n > (r.n - r.off)) {
            r.ok = false;
            return;
        }
        out.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            out[i] = r.u32();
        }
    };
    std::vector<llama_rerot_node_id> forked, running_vec, starting_vec;
    read_id_vec(forked);
    const uint32_t n_ready = r.u32();
    std::deque<llama_rerot_node_id> ready;
    if (!r.ok || n_ready > (r.n - r.off)) {
        r.ok = false;
    } else {
        for (uint32_t i = 0; i < n_ready; ++i) {
            ready.push_back(r.u32());
        }
    }
    read_id_vec(running_vec);
    read_id_vec(starting_vec);
    if (!r.ok) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: truncated queue/membership lists");
    }

    const uint32_t n_nodes = r.u32();
    if (!r.ok || n_nodes == 0 || n_nodes > (r.n - r.off)) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: corrupt node count");
    }
    std::vector<rerot_node_blob> node_blobs;
    node_blobs.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; ++i) {
        rerot_node_blob nb;
        nb.id = r.u32();
        nb.parent = r.u32();
        nb.depth = r.u32();
        nb.child_index = r.u32();
        const uint8_t st = r.u8();
        nb.title = r.str();
        const uint32_t n_children = r.u32();
        if (!r.ok || n_children > (r.n - r.off) || !rerot_node_state_from_u8(st, nb.state)) {
            r.ok = false;
            break;
        }
        nb.children.resize(n_children);
        for (uint32_t k = 0; k < n_children; ++k) {
            nb.children[k] = r.u32();
        }
        const uint32_t n_runs = r.u32();
        if (!r.ok || n_runs > (r.n - r.off)) {
            r.ok = false;
            break;
        }
        nb.runs.resize(n_runs);
        for (uint32_t k = 0; k < n_runs; ++k) {
            nb.runs[k] = r.u32();
        }
        node_blobs.push_back(std::move(nb));
    }
    if (!r.ok) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: truncated node topology");
    }

    const uint32_t n_runs = r.u32();
    if (!r.ok || n_runs > (r.n - r.off)) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: corrupt run count");
    }
    std::vector<rerot_run_blob> run_blobs;
    run_blobs.reserve(n_runs);
    for (uint32_t i = 0; i < n_runs; ++i) {
        rerot_run_blob rb;
        rb.id = r.u32();
        rb.owner = r.u32();
        const uint8_t vis = r.u8();
        rb.storage_pos0 = r.i32();
        rb.token_count = r.u32();
        rb.publish_epoch = r.u64();
        if (!r.ok || !rerot_visibility_from_u8(vis, rb.visibility)) {
            r.ok = false;
            break;
        }
        if (rb.storage_pos0 < 0) {
            r.ok = false;
            break;
        }
        run_blobs.push_back(std::move(rb));
    }
    if (!r.ok) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: truncated run metadata");
    }

    const uint32_t n_rt = r.u32();
    if (!r.ok || n_rt != n_nodes) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: runtime/document node count skew");
    }
    std::vector<rerot_runtime_blob> rt_blobs;
    rt_blobs.reserve(n_rt);
    for (uint32_t i = 0; i < n_rt; ++i) {
        rerot_runtime_blob rb;
        rb.id = r.u32();
        rb.physical_slot = r.i32();
        rb.exec_seq = r.i32();
        rb.parked_seq = r.i32();
        rb.storage_pos_next = r.i32();
        for (auto & ref : rb.refs) {
            const uint8_t has = r.u8();
            const uint32_t id = r.u32();
            if (!r.ok || (has != 0 && has != 1)) {
                r.ok = false;
                break;
            }
            if (has != 0) {
                ref = id;
            }
        }
        if (!r.ok || !rerot_read_planner_snapshot(r, rb.planner) || !rerot_read_marker_snapshot(r, rb.marker)) {
            r.ok = false;
            break;
        }
        rb.planner_armed = r.u8() != 0;
        rb.enqueue_frontier = r.u64();
        rb.exit_intent = r.u8() != 0;
        rb.sampler_blob = r.blob();
        rb.mtp_blob = r.blob();
        rb.view_stamp.topology_epoch = r.u64();
        rb.view_stamp.publish_epoch = r.u64();
        rb.view_stamp.layout_epoch = r.u64();
        if (!r.ok || rb.storage_pos_next < 0 || rb.physical_slot < -1 ||
            rb.exec_seq < -1 || rb.parked_seq < -1) {
            r.ok = false;
            break;
        }
        rt_blobs.push_back(std::move(rb));
    }
    if (!r.ok) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: truncated Lane runtime state");
    }
    if (r.off != r.n) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: trailing garbage after episode state");
    }

    for (uint32_t i = 0; i < n_nodes; ++i) {
        const auto & nb = node_blobs[i];
        if (nb.id != i) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: node ids are not dense");
        }
        if (i == 0) {
            if (nb.parent != LLAMA_REROT_NODE_INVALID || nb.depth != 0 || nb.child_index != 0) {
                return rerot_state_set_error(error_out, "RERoT episode load refused: corrupt root linkage");
            }
            if (!nb.title.empty()) {
                return rerot_state_set_error(error_out, "RERoT episode load refused: root carries an unrepresentable title");
            }
        } else if (nb.parent >= i) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: child precedes its parent");
        }
        for (const auto child : nb.children) {
            if (child >= n_nodes || child <= i) {
                return rerot_state_set_error(error_out, "RERoT episode load refused: child backlink out of range");
            }
        }
        for (const auto run : nb.runs) {
            if (run >= n_runs) {
                return rerot_state_set_error(error_out, "RERoT episode load refused: node references an unknown run");
            }
        }
    }
    for (uint32_t i = 0; i < n_runs; ++i) {
        const auto & rb = run_blobs[i];
        if (rb.id != i || rb.owner >= n_nodes) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: run density/owner out of range");
        }
        const bool wants_epoch = rb.visibility == llama_rerot_visibility::public_live;
        if (wants_epoch == (rb.publish_epoch == 0)) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: run publish epoch breaks the visibility contract");
        }
    }
    for (const auto id : forked) {
        if (id >= n_nodes) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: fork record references an unknown node");
        }
    }
    for (const auto id : ready) {
        if (id >= n_nodes) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: FIFO queue references an unknown node");
        }
    }
    for (const auto id : running_vec) {
        if (id >= n_nodes) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: running set references an unknown node");
        }
    }
    for (const auto id : starting_vec) {
        if (id >= n_nodes) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: starting set references an unknown node");
        }
    }

    server_rerot_episode rebuilt(episode_id);
    rebuilt.document.reset(episode_id);
    if (!rebuilt.document.set_node_state(0, node_blobs[0].state)) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: cannot restore root state");
    }
    try {
        for (uint32_t i = 1; i < n_nodes; ++i) {
            const auto & nb = node_blobs[i];
            const auto made = rebuilt.document.create_child(nb.parent, nb.title, nb.state);
            if (made != i) {
                return rerot_state_set_error(error_out, "RERoT episode load refused: child id skew during rebuild");
            }
            const auto * check = rebuilt.document.node(i);
            if (check == nullptr || check->depth != nb.depth || check->child_index != nb.child_index) {
                return rerot_state_set_error(error_out, "RERoT episode load refused: path metadata mismatch during rebuild");
            }
        }
        for (uint32_t i = 0; i < n_nodes; ++i) {
            const auto * check = rebuilt.document.node(i);
            if (check == nullptr || check->children != node_blobs[i].children) {
                return rerot_state_set_error(error_out, "RERoT episode load refused: child order mismatch during rebuild");
            }
        }
        for (uint32_t i = 0; i < n_runs; ++i) {
            const auto & rb = run_blobs[i];
            const auto made = rebuilt.document.append_run(
                rb.owner, rb.visibility, rb.storage_pos0, rb.token_count, rb.publish_epoch);
            if (made != i) {
                return rerot_state_set_error(error_out, "RERoT episode load refused: run id skew during rebuild");
            }
        }
        for (uint32_t i = 0; i < n_nodes; ++i) {
            const auto * check = rebuilt.document.node(i);
            if (check == nullptr || check->runs != node_blobs[i].runs) {
                return rerot_state_set_error(error_out, "RERoT episode load refused: run order mismatch during rebuild");
            }
        }
    } catch (const std::exception & ex) {
        return rerot_state_set_error(error_out, std::string("RERoT episode load refused: rebuild threw: ") + ex.what());
    }
    {
        std::string verr;
        if (!rebuilt.document.validate(&verr)) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: document validation failed: " + verr);
        }
    }

    rebuilt.nodes.clear();
    rebuilt.nodes.reserve(n_rt);
    for (uint32_t i = 0; i < n_rt; ++i) {
        const auto & sb = rt_blobs[i];
        if (sb.id != i) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: Lane runtime ids are not dense");
        }
        server_rerot_node_runtime node;
        node.id = sb.id;
        node.physical_slot = sb.physical_slot;
        node.exec_seq = sb.exec_seq;
        node.parked_seq = sb.parked_seq;
        node.storage_pos_next = sb.storage_pos_next;
        node.public_run = sb.refs[0];
        node.private_run = sb.refs[1];
        node.pending_record = sb.refs[2];
        std::string perr;
        if (!node.parser.restore(sb.planner, &perr)) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: planner parser: " + perr);
        }
        if (!node.exit_parser.restore(sb.marker, &perr)) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: end-marker parser: " + perr);
        }
        node.planner_armed = sb.planner_armed;
        node.enqueue_frontier = sb.enqueue_frontier;
        node.exit_intent = sb.exit_intent;
        node.sampler_blob = sb.sampler_blob;
        node.mtp_blob = sb.mtp_blob;
        node.view_stamp = sb.view_stamp;
        const auto check_ref = [&](const std::optional<llama_rerot_run_id> & ref, llama_rerot_visibility want) {
            if (!ref.has_value()) {
                return true;
            }
            const auto * run = rebuilt.document.run(*ref);
            return run != nullptr && run->owner == node.id && run->visibility == want;
        };
        if (!check_ref(node.public_run, llama_rerot_visibility::public_live) ||
            !check_ref(node.private_run, llama_rerot_visibility::private_control) ||
            !check_ref(node.pending_record, llama_rerot_visibility::pending_record)) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: Lane active-run pointer mismatch");
        }
        rebuilt.nodes.push_back(std::move(node));
    }

    rebuilt.id = episode_id;
    rebuilt.root_task_id = root_task_id;
    rebuilt.response_task_id = response_task_id;
    rebuilt.frontier = frontier;
    rebuilt.publish_epoch = publish_epoch;
    rebuilt.topology_epoch = topology_epoch;
    rebuilt.layout_epoch = layout_epoch;
    rebuilt.base_prefix_end = base_prefix_end;
    rebuilt.ready_queue = std::move(ready);
    rebuilt.running = std::set<llama_rerot_node_id>(running_vec.begin(), running_vec.end());
    rebuilt.starting = std::set<llama_rerot_node_id>(starting_vec.begin(), starting_vec.end());
    rebuilt.archive_seq = archive_seq;
    rebuilt.topology_barrier_pending = (flags & k_rerot_flag_barrier) != 0;
    rebuilt.finalizing = (flags & k_rerot_flag_finalize) != 0;
    rebuilt.hard_aborted = (flags & k_rerot_flag_aborted) != 0;
    rebuilt.abort_reason = std::move(abort_reason);
    rebuilt.forked_this_frontier = std::move(forked);
    rebuilt.generated_public_tokens = gen_public;
    rebuilt.generated_private_tokens = gen_private;
    rebuilt.forced_heading_tokens = forced_heading;
    rebuilt.pending_tokens = pending_tokens;
    rebuilt.queue_peak = queue_peak;
    rebuilt.hard_limits = limits;
    rebuilt.fence_refreshed = (flags & k_rerot_flag_fence) != 0;
    rebuilt.serial_tail = (flags & k_rerot_flag_serial) != 0;
    rebuilt.serial_node = serial_node;
    if (rebuilt.serial_node != LLAMA_REROT_NODE_INVALID && rebuilt.serial_node >= n_nodes) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: serial node out of range");
    }

    *episode_out = std::move(rebuilt);
    return true;
}

bool server_rerot_truncate_oldest_public(
        server_rerot_episode & episode,
        uint64_t max_tokens_to_remove,
        server_rerot_shift_result * result_out,
        std::string * error_out) {
    if (result_out) {
        *result_out = server_rerot_shift_result{};
    }
    if (episode.id == 0) {
        return rerot_state_set_error(error_out, "RERoT shift refused: invalid episode");
    }
    if (max_tokens_to_remove == 0) {
        if (result_out) {
            result_out->new_layout_epoch = episode.layout_epoch;
            result_out->new_publish_epoch = episode.publish_epoch;
        }
        return true;
    }
    if (episode.layout_epoch == std::numeric_limits<uint64_t>::max() ||
        episode.publish_epoch == std::numeric_limits<uint64_t>::max()) {
        return rerot_state_set_error(error_out, "RERoT shift refused: publish/layout epoch exhausted");
    }
    std::set<llama_rerot_run_id> pinned;
    for (const auto & node : episode.nodes) {
        if (node.public_run.has_value()) {
            pinned.insert(*node.public_run);
        }
        if (node.private_run.has_value()) {
            pinned.insert(*node.private_run);
        }
        if (node.pending_record.has_value()) {
            pinned.insert(*node.pending_record);
        }
    }
    uint64_t removed = 0;
    uint32_t n_trunc = 0;
    while (removed < max_tokens_to_remove) {
        const llama_rerot_run * best = nullptr;
        for (size_t i = 0; i < episode.document.run_count(); ++i) {
            const auto * run = episode.document.run(static_cast<llama_rerot_run_id>(i));
            if (run == nullptr || run->visibility != llama_rerot_visibility::public_live || run->token_count == 0) {
                continue;
            }
            if (run->storage_pos0 < episode.base_prefix_end) {
                continue;
            }
            if (pinned.count(run->id) != 0) {
                continue;
            }
            if (best == nullptr || run->storage_pos0 < best->storage_pos0 ||
                (run->storage_pos0 == best->storage_pos0 &&
                 (run->publish_epoch < best->publish_epoch ||
                  (run->publish_epoch == best->publish_epoch && run->id < best->id)))) {
                best = run;
            }
        }
        if (best == nullptr) {
            break;
        }
        const uint64_t count = best->token_count;
        const auto id = best->id;
        best = nullptr;
        if (!episode.document.set_run_token_count(id, 0)) {
            return rerot_state_set_error(error_out, "RERoT shift failed: cannot empty run " + std::to_string(id));
        }
        removed += count;
        ++n_trunc;
    }
    if (removed == 0) {
        if (result_out) {
            result_out->new_layout_epoch = episode.layout_epoch;
            result_out->new_publish_epoch = episode.publish_epoch;
        }
        return true;
    }
    ++episode.layout_epoch;
    ++episode.publish_epoch;
    episode.topology_barrier_pending = true;
    if (result_out) {
        result_out->tokens_removed = removed;
        result_out->runs_truncated = n_trunc;
        result_out->runs_emptied = n_trunc;
        result_out->new_layout_epoch = episode.layout_epoch;
        result_out->new_publish_epoch = episode.publish_epoch;
    }
    return true;
}

bool server_rerot_runtime::save_episode(
        uint64_t episode_id,
        const server_rerot_state_fingerprints & fp,
        std::vector<uint8_t> * blob_out,
        std::string * error_out) const {
    if (blob_out == nullptr) {
        return rerot_state_set_error(error_out, "RERoT episode save refused: null blob target");
    }
    const auto * current = episode(episode_id);
    if (current == nullptr) {
        return rerot_state_set_error(error_out, "RERoT episode save refused: unknown episode " + std::to_string(episode_id));
    }
    auto blob = server_rerot_episode_save(*current, fp, error_out);
    if (blob.empty()) {
        return false;
    }
    *blob_out = std::move(blob);
    return true;
}

bool server_rerot_runtime::load_episode(
        const uint8_t * data,
        size_t size,
        const server_rerot_state_fingerprints & expected_fp,
        uint64_t * episode_id_out,
        std::string * error_out) {
    server_rerot_episode staged(1);
    if (!server_rerot_episode_load(data, size, expected_fp, &staged, error_out)) {
        return false;
    }
    if (episodes_.count(staged.id) != 0) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: episode " +
            std::to_string(staged.id) + " is already live; erase it first");
    }
    std::set<llama_seq_id> live_seqs;
    for (const auto & kv : episodes_) {
        const auto & ep = kv.second;
        if (ep.archive_seq >= 0) {
            live_seqs.insert(ep.archive_seq);
        }
        for (const auto & node : ep.nodes) {
            if (node.exec_seq >= 0) {
                live_seqs.insert(node.exec_seq);
            }
            if (node.parked_seq >= 0) {
                live_seqs.insert(node.parked_seq);
            }
        }
    }
    if (staged.archive_seq != -1 &&
        (staged.archive_seq < static_cast<llama_seq_id>(first_internal_seq_) ||
         staged.archive_seq >= static_cast<llama_seq_id>(max_seq_) ||
         live_seqs.count(staged.archive_seq) != 0)) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: archive seq out of arena or already bound");
    }
    for (const auto & node : staged.nodes) {
        if (node.exec_seq != -1 &&
            (node.exec_seq < 0 || node.exec_seq >= static_cast<llama_seq_id>(max_seq_) ||
             node.exec_seq >= static_cast<llama_seq_id>(LLAMA_MAX_SEQ) ||
             live_seqs.count(node.exec_seq) != 0)) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: exec seq out of range or already bound");
        }
        if (node.parked_seq != -1 &&
            (node.parked_seq < static_cast<llama_seq_id>(first_internal_seq_) ||
             node.parked_seq >= static_cast<llama_seq_id>(max_seq_) ||
             live_seqs.count(node.parked_seq) != 0)) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: parked seq out of arena or already bound");
        }
        if (node.physical_slot >= 0 && slot_to_episode_.count(node.physical_slot) != 0) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: physical slot " +
                std::to_string(node.physical_slot) + " already bound; demote before swap");
        }
    }
    const auto erase_free = [&](llama_seq_id seq) {
        free_internal_seqs_.erase(
            std::remove(free_internal_seqs_.begin(), free_internal_seqs_.end(), seq),
            free_internal_seqs_.end());
    };
    if (staged.archive_seq >= 0) {
        erase_free(staged.archive_seq);
    }
    for (const auto & node : staged.nodes) {
        if (node.parked_seq >= 0) {
            erase_free(node.parked_seq);
        }
        if (node.physical_slot >= 0) {
            slot_to_episode_[node.physical_slot] = staged.id;
        }
    }
    if (staged.id >= next_episode_id_) {
        next_episode_id_ = staged.id + 1;
        if (next_episode_id_ == 0) {
            next_episode_id_ = 1;
        }
    }
    const uint64_t restored = staged.id;
    episodes_.emplace(restored, std::move(staged));
    if (episode_id_out) {
        *episode_id_out = restored;
    }
    return true;
}

bool server_rerot_runtime::demote_episode(uint64_t episode_id) {
    auto * current = episode(episode_id);
    if (current == nullptr) {
        return false;
    }
    for (const auto & node : current->nodes) {
        if (node.physical_slot >= 0) {
            const auto it = slot_to_episode_.find(node.physical_slot);
            if (it != slot_to_episode_.end() && it->second == episode_id) {
                slot_to_episode_.erase(it);
            }
        }
    }
    server_rerot_episode_demote_to_logical(*current);
    return true;
}

namespace {

class server_rerot_chronicle_registry {
public:
    static server_rerot_chronicle_registry & instance() {
        static server_rerot_chronicle_registry reg;
        return reg;
    }

    void register_mapping(
            std::string_view chronicle,
            std::string_view canonical,
            std::string_view content,
            uint64_t episode_id) {
        if (chronicle.empty() || canonical.empty() || chronicle == canonical) {
            return;
        }

        std::string norm_chronicle = normalize_for_lookup(chronicle);
        const uint64_t key = hash_text(norm_chronicle);

        std::lock_guard<std::mutex> lock(mutex_);
        if (entries_.size() >= MAX_ENTRIES) {
            evict_oldest();
        }

        entry_t e;
        e.chronicle_normalized = std::move(norm_chronicle);
        e.canonical_reasoning = std::string(canonical);
        e.final_content = std::string(content);
        e.episode_id = episode_id;
        e.timestamp_us = ggml_time_us();

        entries_[key] = std::move(e);
        lru_order_.push_back(key);
    }

    std::optional<std::string> resolve(std::string_view incoming) const {
        if (incoming.empty()) {
            return std::nullopt;
        }

        std::string norm_incoming = normalize_for_lookup(incoming);
        const uint64_t key = hash_text(norm_incoming);

        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(key);
        if (it == entries_.end()) {
            return std::nullopt;
        }

        // Exact byte-for-byte match to guarantee zero false mappings on unrecorded or modified thoughts
        if (it->second.chronicle_normalized != norm_incoming) {
            return std::nullopt;
        }

        return it->second.canonical_reasoning;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.clear();
        lru_order_.clear();
    }

private:
    static constexpr size_t MAX_ENTRIES = 1024;

    struct entry_t {
        std::string chronicle_normalized;
        std::string canonical_reasoning;
        std::string final_content;
        uint64_t episode_id = 0;
        int64_t timestamp_us = 0;
    };

    static std::string normalize_for_lookup(std::string_view text) {
        return trim_ascii_space(text);
    }

    static uint64_t hash_text(std::string_view text) {
        uint64_t h = 0xcbf29ce484222325ULL;
        for (unsigned char c : text) {
            h = (h ^ c) * 0x100000001b3ULL;
        }
        return h;
    }

    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, entry_t> entries_;
    std::vector<uint64_t> lru_order_;

    void evict_oldest() {
        while (!lru_order_.empty()) {
            const uint64_t old_key = lru_order_.front();
            lru_order_.erase(lru_order_.begin());
            auto it = entries_.find(old_key);
            if (it != entries_.end()) {
                entries_.erase(it);
                break;
            }
        }
    }
};

} // namespace

void server_rerot_register_chronicle_mapping(
        std::string_view chronicle_reasoning,
        std::string_view canonical_reasoning,
        std::string_view final_content,
        uint64_t episode_id) {
    server_rerot_chronicle_registry::instance().register_mapping(
        chronicle_reasoning, canonical_reasoning, final_content, episode_id);
}

std::optional<std::string> server_rerot_resolve_canonical_reasoning(
        std::string_view incoming_reasoning) {
    return server_rerot_chronicle_registry::instance().resolve(incoming_reasoning);
}

void server_rerot_clear_chronicle_registry() {
    server_rerot_chronicle_registry::instance().clear();
}

