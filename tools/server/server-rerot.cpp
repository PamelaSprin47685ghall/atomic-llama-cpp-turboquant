#include "server-rerot.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "peg-parser.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <unordered_set>
#include <utility>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {

constexpr std::string_view k_open_ol  = "<ol>";
constexpr std::string_view k_close_ol = "</ol>";
constexpr std::string_view k_open_li  = "<li>";
constexpr std::string_view k_close_li = "</li>";

bool ascii_space(unsigned char ch) {
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' || ch == '\v';
}

bool valid_child_close_marker(std::string_view marker) {
    if (marker.size() != 11 || marker[0] != '<' || marker[1] != '/' ||
        marker.back() != '>') {
        return false;
    }
    for (size_t i = 2; i < 10; ++i) {
        const unsigned char ch = static_cast<unsigned char>(marker[i]);
        if (!((ch >= '0' && ch <= '9') ||
              (ch >= 'A' && ch <= 'Z') ||
              (ch >= 'a' && ch <= 'z'))) {
            return false;
        }
    }
    return true;
}

constexpr uint32_t k_hand_seed_magic = 0x33454553; // 'SEE3'
constexpr uint32_t k_sampler_snap_magic = 0x4C50534D; // 'MSPL'

void checkpoint_fill_from_hand_seed(
        server_rerot_prebranch_checkpoint & cp,
        std::vector<uint8_t> seed) {
    cp.gdn_recurrent_states = seed;
    cp.conv1d_states.clear();
    if (seed.size() < sizeof(uint32_t) * 3 + sizeof(llama_pos)) {
        return;
    }
    size_t off = 0;
    auto read_u32 = [&](uint32_t & out) -> bool {
        if (off + sizeof(uint32_t) > seed.size()) {
            return false;
        }
        std::memcpy(&out, seed.data() + off, sizeof(uint32_t));
        off += sizeof(uint32_t);
        return true;
    };
    uint32_t magic = 0;
    if (!read_u32(magic) || magic != k_hand_seed_magic) {
        return;
    }
    if (off + sizeof(llama_pos) > seed.size()) {
        return;
    }
    off += sizeof(llama_pos);
    uint32_t n_conv = 0;
    if (!read_u32(n_conv)) {
        return;
    }
    const size_t conv_begin = off;
    for (uint32_t i = 0; i < n_conv; ++i) {
        uint32_t sz = 0;
        if (!read_u32(sz) || off + sz > seed.size()) {
            return;
        }
        off += sz;
    }
    cp.conv1d_states.assign(
        seed.begin() + static_cast<std::ptrdiff_t>(conv_begin),
        seed.begin() + static_cast<std::ptrdiff_t>(off));
}

bool dag_node_started(const llama_rerot_node * node) {
    if (node == nullptr) {
        return false;
    }
    switch (node->state) {
        case llama_rerot_node_state::starting:
        case llama_rerot_node_state::running:
        case llama_rerot_node_state::retired:
        case llama_rerot_node_state::ready_suspended:
        case llama_rerot_node_state::terminal_running:
        case llama_rerot_node_state::forked:
            return true;
        default:
            return false;
    }
}

bool valid_native_end_marker(std::string_view marker) {
    // Accepts any non-empty, non-null terminal marker (e.g. </think>,
    // [/THINK], <|end|>, <|im_end|>, etc.) without assuming XML brackets.
    return !marker.empty() && marker.find('\0') == std::string_view::npos;
}

std::string sanitize_plan_intent(
        std::string intent,
        std::string_view think_start,
        std::string_view think_end = {}) {
    for (char & c : intent) {
        if (c == '\r' || c == '\n') {
            c = ' ';
        }
    }
    if (!think_end.empty()) {
        size_t pos = 0;
        while ((pos = intent.find(think_end, pos)) != std::string::npos) {
            intent.replace(pos, think_end.size(), "[end]");
            pos += 5;
        }
    }
    if (!think_start.empty()) {
        size_t pos = 0;
        while ((pos = intent.find(think_start, pos)) != std::string::npos) {
            intent.replace(pos, think_start.size(), "[start]");
            pos += 7;
        }
    }
    return intent;
}

bool dag_node_in_started_set(const server_rerot_episode & ep, llama_rerot_node_id nid) {
    if (nid == 0) {
        return true;
    }
    if (nid >= ep.nodes.size()) {
        return false;
    }
    const auto & nr = ep.nodes[nid];
    if (nr.is_sealed) {
        return true;
    }
    if (ep.running.count(nid) != 0 || ep.starting.count(nid) != 0 ||
        ep.suspended.count(nid) != 0) {
        return true;
    }
    const auto * dn = ep.document.node(nid);
    if (!dn) {
        return false;
    }
    switch (dn->state) {
        case llama_rerot_node_state::starting:
        case llama_rerot_node_state::running:
        case llama_rerot_node_state::terminal_running:
        case llama_rerot_node_state::ready_suspended:
        case llama_rerot_node_state::retired:
            return true;
        default:
            return false;
    }
}

void skip_json_ws(std::string_view text, size_t & pos) {
    while (pos < text.size() && ascii_space(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
}

bool skip_json_string(std::string_view text, size_t & pos) {
    if (pos >= text.size() || text[pos] != '"') {
        return false;
    }
    ++pos;
    while (pos < text.size()) {
        if (text[pos] == '\\') {
            pos += (pos + 1 < text.size()) ? 2 : 1;
            continue;
        }
        if (text[pos] == '"') {
            ++pos;
            return true;
        }
        ++pos;
    }
    return false;
}

bool json_value_has_duplicate_keys(std::string_view text, size_t & pos);

bool json_object_has_duplicate_keys(std::string_view text, size_t & pos) {
    if (pos >= text.size() || text[pos] != '{') {
        return true;
    }
    ++pos;
    skip_json_ws(text, pos);
    std::unordered_set<std::string> keys;
    if (pos < text.size() && text[pos] == '}') {
        ++pos;
        return false;
    }
    while (pos < text.size()) {
        skip_json_ws(text, pos);
        const size_t key_begin = pos;
        if (!skip_json_string(text, pos)) {
            return true;
        }
        if (!keys.insert(std::string(text.substr(key_begin, pos - key_begin))).second) {
            return true;
        }
        skip_json_ws(text, pos);
        if (pos >= text.size() || text[pos] != ':') {
            return true;
        }
        ++pos;
        skip_json_ws(text, pos);
        if (json_value_has_duplicate_keys(text, pos)) {
            return true;
        }
        skip_json_ws(text, pos);
        if (pos < text.size() && text[pos] == ',') {
            ++pos;
            continue;
        }
        if (pos < text.size() && text[pos] == '}') {
            ++pos;
            return false;
        }
        return true;
    }
    return true;
}

bool json_array_has_duplicate_keys(std::string_view text, size_t & pos) {
    if (pos >= text.size() || text[pos] != '[') {
        return true;
    }
    ++pos;
    skip_json_ws(text, pos);
    if (pos < text.size() && text[pos] == ']') {
        ++pos;
        return false;
    }
    while (pos < text.size()) {
        skip_json_ws(text, pos);
        if (json_value_has_duplicate_keys(text, pos)) {
            return true;
        }
        skip_json_ws(text, pos);
        if (pos < text.size() && text[pos] == ',') {
            ++pos;
            continue;
        }
        if (pos < text.size() && text[pos] == ']') {
            ++pos;
            return false;
        }
        return true;
    }
    return true;
}

bool json_value_has_duplicate_keys(std::string_view text, size_t & pos) {
    skip_json_ws(text, pos);
    if (pos >= text.size()) {
        return true;
    }
    const char ch = text[pos];
    if (ch == '{') {
        return json_object_has_duplicate_keys(text, pos);
    }
    if (ch == '[') {
        return json_array_has_duplicate_keys(text, pos);
    }
    if (ch == '"') {
        return !skip_json_string(text, pos);
    }
    if (ch == 't' || ch == 'f' || ch == 'n' || ch == '-' || (ch >= '0' && ch <= '9')) {
        while (pos < text.size() && !ascii_space(static_cast<unsigned char>(text[pos])) &&
               text[pos] != ',' && text[pos] != '}' && text[pos] != ']') {
            ++pos;
        }
        return false;
    }
    return true;
}

bool json_text_has_duplicate_keys(std::string_view text) {
    size_t pos = 0;
    skip_json_ws(text, pos);
    if (json_value_has_duplicate_keys(text, pos)) {
        return true;
    }
    skip_json_ws(text, pos);
    return pos != text.size();
}

bool json_object_has_only_keys(
        const json & object,
        std::initializer_list<const char *> allowed,
        std::string * error,
        const char * context) {
    for (auto it = object.begin(); it != object.end(); ++it) {
        bool ok = false;
        for (const char * key : allowed) {
            if (it.key() == key) {
                ok = true;
                break;
            }
        }
        if (!ok) {
            if (error) {
                *error = std::string("unexpected field '") + it.key() + "' in " + context;
            }
            return false;
        }
    }
    return true;
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

server_rerot_marker_parser::server_rerot_marker_parser(std::string marker, bool native_end)
    : marker_(std::move(marker)), native_end_(native_end) {
    reset();
}

void server_rerot_marker_parser::reset() {
    candidate_.clear();
    error_.clear();
    state_ = server_rerot_marker_state::public_text;
    if (marker_.empty()) {
        return;
    }
    if (native_end_) {
        if (!valid_native_end_marker(marker_)) {
            state_ = server_rerot_marker_state::failed;
            error_ = "RERoT native source-end marker is not a reasoning-end tag";
        }
        return;
    }
    if (!valid_child_close_marker(marker_)) {
        state_ = server_rerot_marker_state::failed;
        error_ = "RERoT child close marker must contain exactly 8 base62 characters";
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
    if (marker_.empty()) {
        return step;
    }

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
        step.public_prefix_bytes = full;
        step.marker_closed = true;
        return step;
    }

    const size_t suffix = trailing_marker_prefix(scan, marker_);
    if (suffix >= 1) {
        candidate_.assign(scan.substr(scan.size() - suffix));
        state_ = server_rerot_marker_state::marker_candidate;
        step.write_visibility = llama_rerot_visibility::pending_record;
        step.public_prefix_bytes = scan.size() - suffix;
        return step;
    }

    state_ = server_rerot_marker_state::public_text;
    step.write_visibility = llama_rerot_visibility::public_live;
    return step;
}

server_rerot_stream_lines server_rerot_line_mux::append(
        llama_rerot_node_id node_id,
        llama_rerot_run_id run_id,
        std::string_view bytes,
        const llama_rerot_document & document,
        size_t public_prefix_bytes) {
    server_rerot_stream_lines result;
    if (node_id == LLAMA_REROT_NODE_INVALID ||
        run_id == LLAMA_REROT_RUN_INVALID ||
        public_prefix_bytes > bytes.size()) {
        result.ok = false;
        result.error = "invalid RERoT line-stream node/run or public prefix";
        return result;
    }

    auto & lane = lanes_[node_id];
    const auto append_segment = [&](std::string_view segment_bytes, bool presentation_public) {
        if (segment_bytes.empty()) {
            return;
        }
        if (!lane.blocked.empty() &&
            lane.blocked.back().run_id == run_id &&
            lane.blocked.back().presentation_public == presentation_public) {
            lane.blocked.back().bytes.append(segment_bytes);
        } else {
            lane.blocked.push_back({run_id, std::string(segment_bytes), presentation_public});
        }
    };
    append_segment(bytes.substr(0, public_prefix_bytes), true);
    append_segment(bytes.substr(public_prefix_bytes), false);
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
        const bool presentation_public = lane.blocked.front().presentation_public;
        const auto * run = document.run(lane.blocked.front().run_id);
        if (!run) {
            result.ok = false;
            result.error = "RERoT line stream references a missing run";
            return result;
        }
        if (!presentation_public &&
            run->visibility == llama_rerot_visibility::pending_record) {
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
        if (!presentation_public &&
            run->visibility == llama_rerot_visibility::private_control) {
            continue;
        }
        if (!presentation_public &&
            run->visibility != llama_rerot_visibility::public_live) {
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
            if (!line.empty() && line != "\n") {
                result.lines.push_back(std::move(line));
            }
        }
    }

    if (finish && !lane.partial_line.empty()) {
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
        "我先把回答中可以同时展开、工作量大致相当的并列对象或章节列成一个平面的 HTML 有序列表："
        "以 <ol> 开头，每项只写一个简短 <li> 标题，不提前展开内容，以 </ol> 结尾。"
        "一个 <li> 只对应一个可独立展开的对象或子主题；同一项里仍有互不依赖的部分时，我继续把它们分成并列项。"
        "只有确实无法并行拆解时才写一个 <li>。输出完 </ol> 后我再展开分析。";
    return prompt;
}

std::string server_rerot_child_contract(
        std::string_view title,
        std::string_view close_marker) {
    if (!valid_child_close_marker(close_marker)) return {};
    const std::string clean_title = normalize_lane_title(title);
    if (clean_title.empty()) return {};
    return "\n我现在专注于自己的唯一子任务『" + clean_title + "』。"
        "其他公开章节只作为我的参考背景，我不接管它们，也不重新回答整个用户问题。"
        "正文中普通列表和标题只是内容呈现；详尽完成本项内容后，我直接输出 " + std::string(close_marker) +
        " 结束当前工作块，不再发散。\n";
}

std::string server_rerot_child_planner_prompt(std::string_view title) {
    const std::string clean_title = normalize_lane_title(title);
    if (clean_title.empty()) return {};
    return "我只针对当前唯一任务『" + clean_title + "』自省是否还存在两个或更多可以独立并行展开的子任务。"
        "输出一个平面的 HTML 有序列表：以 <ol> 开头，每项只写一个简短 <li> 标题，不提前展开内容，以 </ol> 结尾。"
        "如果无需拆分，我也只输出一个概括当前任务的 <li>。不列出兄弟章节，不重述整个用户问题。";
}

std::string server_rerot_child_worker_prompt(
        std::string_view title,
        std::string_view close_marker) {
    if (!valid_child_close_marker(close_marker)) return {};
    const std::string clean_title = normalize_lane_title(title);
    if (clean_title.empty()) return {};
    return "\n我现在开始专注于自己的唯一子任务『" + clean_title + "』的详尽推导。"
        "其他公开章节仅作为参考背景，我不越界接管，也不重新回答整个用户问题。"
        "正文中普通列表、标题和代码块只是内容呈现，不具有调度含义。"
        "详尽完成本项内容后，我直接输出 " + std::string(close_marker) +
        " 结束当前段落，不再继续其他章节。\n";
}

std::string server_rerot_format_fixed_entry(
        const std::string & node_label,
        const std::string & intent,
        bool is_synthesis,
        std::string_view think_end,
        std::string_view think_start) {
    // AGENTS.md §04: F_i = CLOSE_PREVIOUS + HANDOFF_TO_i + OPEN_CURRENT.
    // Tags must come from the target template; this helper only concatenates
    // the structural close/open around a target-only identity payload.
    std::string frame;
    frame.append(think_end.data(), think_end.size());
    frame += "\n";
    if (is_synthesis) {
        frame += "synthesis:";
        frame += node_label.empty() ? "0" : node_label;
    } else {
        frame += "lane:";
        frame += node_label;
    }
    if (!intent.empty()) {
        // AGENTS.md §04.10: serialize intent cleanly, flattening line breaks
        // and disarming any raw think markers to prevent boundary forgery.
        std::string clean_intent;
        clean_intent.reserve(intent.size());
        for (char c : intent) {
            if (c == '\r' || c == '\n') {
                clean_intent += ' ';
            } else {
                clean_intent += c;
            }
        }
        if (!think_end.empty()) {
            size_t pos = 0;
            while ((pos = clean_intent.find(think_end, pos)) != std::string::npos) {
                clean_intent.replace(pos, think_end.size(), "[end]");
                pos += 5;
            }
        }
        if (!think_start.empty()) {
            size_t pos = 0;
            while ((pos = clean_intent.find(think_start, pos)) != std::string::npos) {
                clean_intent.replace(pos, think_start.size(), "[start]");
                pos += 7;
            }
        }
        frame += "\nintent:";
        frame += clean_intent;
    }
    frame += "\n";
    frame.append(think_start.data(), think_start.size());
    frame += "\n";
    return frame;
}

std::string server_rerot_format_plan_prefix(
        const server_rerot_routing_decision & decision,
        std::string_view think_start) {
    if (!decision.is_dag() || decision.questions.empty()) {
        return {};
    }
    std::vector<const server_rerot_dag_plan_item *> ordered;
    ordered.reserve(decision.questions.size());
    for (const auto & q : decision.questions) {
        ordered.push_back(&q);
    }
    std::stable_sort(ordered.begin(), ordered.end(), [](const auto * a, const auto * b) {
        return a->plan_rank < b->plan_rank;
    });

    std::string prefix = "plan:\n";
    for (const auto * q : ordered) {
        prefix += "- ";
        prefix += q->id;
        prefix += ": ";
        prefix += sanitize_plan_intent(q->intent, think_start);
        prefix += "\n";
    }
    return prefix;
}

std::string_view server_rerot_routing_probe_prompt() {
    static constexpr std::string_view prompt =
        "Choose whether this request continues as a single answer or a DAG of independent sub-questions. "
        "Output only JSON: {\"strategy\":\"simple\",\"payload\":{}} or "
        "{\"strategy\":\"dag\",\"payload\":{\"questions\":[{\"id\":\"...\",\"intent\":\"...\"}],\"depends_on\":[]}}.";
    return prompt;
}

std::string server_rerot_routing_grammar() {
    const auto schema = nlohmann::ordered_json::parse(server_rerot_routing_schema_json());
    return json_schema_to_grammar(schema);
}

std::string server_rerot_source_end_grammar(std::string_view close_marker) {
    if (!valid_native_end_marker(close_marker)) {
        return {};
    }
    const std::string close(close_marker);
    auto arena = build_peg_parser([&](common_peg_parser_builder & builder) {
        return builder.sequence({
            builder.chars("[^ \\t\\r\\n]", 1, 1),
            builder.until_one_of({close}),
            builder.literal(close),
            builder.end(),
        });
    });
    common_grammar_options options;
    options.dotall = true;
    return build_grammar([&](const common_grammar_builder & grammar_builder) {
        arena.build_grammar(grammar_builder, false);
    }, options);
}

std::string server_rerot_routing_schema_json() {
    // Exact schema from AGENTS.md §02.4
    return R"({
  "oneOf": [
    {
      "type": "object",
      "required": ["strategy", "payload"],
      "additionalProperties": false,
      "properties": {
        "strategy": {"const": "simple"},
        "payload": {"const": {}}
      }
    },
    {
      "type": "object",
      "required": ["strategy", "payload"],
      "additionalProperties": false,
      "properties": {
        "strategy": {"const": "dag"},
        "payload": {"$ref": "#/$defs/DagPayload"}
      }
    }
  ],
  "$defs": {
    "DagPayload": {
      "type": "object",
      "required": ["questions", "depends_on"],
      "additionalProperties": false,
      "properties": {
        "questions": {
          "type": "array",
          "minItems": 1,
          "items": {
            "type": "object",
            "required": ["id", "intent"],
            "additionalProperties": false,
            "properties": {
              "id": {"type": "string", "minLength": 1},
              "intent": {"type": "string", "minLength": 1}
            }
          }
        },
        "depends_on": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["id", "depends_on_id"],
            "additionalProperties": false,
            "properties": {
              "id": {"type": "string", "minLength": 1},
              "depends_on_id": {"type": "string", "minLength": 1}
            }
          }
        }
      }
    }
  }
})";
}

server_rerot_routing_decision server_rerot_parse_routing_decision(const std::string & json_str) {
    server_rerot_routing_decision result;
    if (json_text_has_duplicate_keys(json_str)) {
        result.error = "duplicate JSON object members are not allowed";
        return result;
    }
    json root_json;
    try {
        root_json = json::parse(json_str);
    } catch (const std::exception & e) {
        result.error = "Invalid JSON syntax: " + std::string(e.what());
        return result;
    }

    if (!root_json.is_object() || !root_json.contains("strategy") || !root_json.contains("payload")) {
        result.error = "Missing strategy or payload";
        return result;
    }
    if (!json_object_has_only_keys(root_json, {"strategy", "payload"}, &result.error, "routing root")) {
        return result;
    }

    std::string strategy = root_json["strategy"].is_string() ? root_json["strategy"].get<std::string>() : "";
    if (strategy == "simple") {
        if (!root_json["payload"].is_object() || !root_json["payload"].empty()) {
            result.error = "simple payload must be empty object {}";
            return result;
        }
        result.strategy = server_rerot_routing_decision::strategy_type::simple;
        return result;
    }

    if (strategy == "dag") {
        const auto & payload = root_json["payload"];
        if (!payload.is_object() || !payload.contains("questions") || !payload.contains("depends_on")) {
            result.error = "dag payload missing questions or depends_on";
            return result;
        }
        if (!json_object_has_only_keys(payload, {"questions", "depends_on"}, &result.error, "dag payload")) {
            return result;
        }

        const auto & q_arr = payload["questions"];
        if (!q_arr.is_array() || q_arr.empty()) {
            result.error = "questions must be a non-empty array";
            return result;
        }

        auto is_all_ws = [](const std::string & s) {
            return std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); });
        };

        std::unordered_set<std::string> known_ids;
        uint32_t rank = 0;
        for (const auto & item : q_arr) {
            if (!item.is_object() || !item.contains("id") || !item.contains("intent")) {
                result.error = "question item missing id or intent";
                return result;
            }
            if (!json_object_has_only_keys(item, {"id", "intent"}, &result.error, "question")) {
                return result;
            }
            if (!item["id"].is_string() || !item["intent"].is_string()) {
                result.error = "id and intent must be strings";
                return result;
            }
            std::string qid = item["id"].get<std::string>();
            std::string intent = item["intent"].get<std::string>();
            if (qid.empty() || is_all_ws(qid) || intent.empty() || is_all_ws(intent)) {
                result.error = "id and intent must not be empty or whitespace-only";
                return result;
            }
            if (qid == "0") {
                result.error = "id '0' is reserved for main synthesis";
                return result;
            }
            if (known_ids.count(qid)) {
                result.error = "duplicate question id: " + qid;
                return result;
            }
            known_ids.insert(qid);
            result.questions.push_back({qid, intent, rank++});
        }

        const auto & dep_arr = payload["depends_on"];
        if (!dep_arr.is_array()) {
            result.error = "depends_on must be an array";
            return result;
        }

        std::set<std::pair<std::string, std::string>> seen_edges;
        for (const auto & dep : dep_arr) {
            if (!dep.is_object() || !dep.contains("id") || !dep.contains("depends_on_id")) {
                result.error = "depends_on item missing id or depends_on_id";
                return result;
            }
            if (!json_object_has_only_keys(dep, {"id", "depends_on_id"}, &result.error, "depends_on")) {
                return result;
            }
            if (!dep["id"].is_string() || !dep["depends_on_id"].is_string()) {
                result.error = "dependency endpoints must be strings";
                return result;
            }
            std::string to_id = dep["id"].get<std::string>();
            std::string from_id = dep["depends_on_id"].get<std::string>();

            if (!known_ids.count(to_id) || !known_ids.count(from_id)) {
                result.error = "unknown endpoint in dependency: " + from_id + " -> " + to_id;
                return result;
            }
            if (to_id == from_id) {
                result.error = "self-loop dependency: " + to_id;
                return result;
            }
            if (seen_edges.count({from_id, to_id})) {
                result.error = "duplicate dependency edge: " + from_id + " -> " + to_id;
                return result;
            }
            seen_edges.insert({from_id, to_id});
            result.dependencies.push_back({from_id, to_id});
        }

        // Kahn algorithm topological check
        std::unordered_map<std::string, int> in_degree;
        std::unordered_map<std::string, std::vector<std::string>> adj;
        for (const auto & qid : known_ids) {
            in_degree[qid] = 0;
        }
        for (const auto & edge : result.dependencies) {
            in_degree[edge.to_id]++;
            adj[edge.from_id].push_back(edge.to_id);
        }

        std::queue<std::string> q;
        for (const auto & [qid, deg] : in_degree) {
            if (deg == 0) q.push(qid);
        }
        size_t visited = 0;
        while (!q.empty()) {
            std::string u = q.front();
            q.pop();
            visited++;
            for (const auto & v : adj[u]) {
                if (--in_degree[v] == 0) q.push(v);
            }
        }

        if (visited != known_ids.size()) {
            result.error = "cycle detected in dependency graph";
            return result;
        }

        result.strategy = server_rerot_routing_decision::strategy_type::dag;
        return result;
    }

    result.error = "unknown strategy: " + strategy;
    return result;
}

std::string server_rerot_child_grammar(std::string_view close_marker) {
    if (!valid_child_close_marker(close_marker)) {
        return {};
    }
    const std::string close(close_marker);
    auto arena = build_peg_parser([&](common_peg_parser_builder & builder) {
        // Newlines are ordinary content, not a scheduler boundary. Only this
        // child's exact closing marker ends the thread; no line/token budget
        // can stand in for completing its assigned work.
        return builder.sequence({
            builder.chars("[^ \\t\\r\\n]", 1, 1),
            builder.until_one_of({close}),
            builder.literal(close),
            builder.end(),
        });
    });
    common_grammar_options options;
    options.dotall = true;
    return build_grammar([&](const common_grammar_builder & grammar_builder) {
        arena.build_grammar(grammar_builder, false);
    }, options);
}

std::string_view server_rerot_planner_grammar() {
    static constexpr std::string_view grammar =
        "root ::= item-text \"</li>\" ws (\"<li>\" item-text \"</li>\" ws)* \"</ol>\"\n"
        "item-text ::= ws [^< \\t\\r\\n] [^<]*\n"
        "ws ::= [ \\t\\r\\n]*\n";
    return grammar;
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
    if (first_internal_seq_ > 0) {
        set_pen_capacity(first_internal_seq_);
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
    root.pen_id = physical_slot;
    root.physical_slot = physical_slot;
    root.exec_seq = exec_seq;
    root.storage_pos_next = storage_pos_next;
    current.nodes.push_back(std::move(root));
    current.running.insert(current.document.root());

    if (physical_slot >= 0 && (size_t) physical_slot < pens_.size()) {
        pens_[physical_slot].state = server_pen_state::running;
        pens_[physical_slot].person = episode_id;
        pens_[physical_slot].episode_id = episode_id;
        pens_[physical_slot].node_id = current.document.root();
        pens_[physical_slot].exec_seq = exec_seq;
    }

    slot_to_episode_[physical_slot] = episode_id;
    return episode_id;
}

void server_rerot_runtime::set_pen_capacity(uint32_t total_pens) {
    pens_.clear();
    pens_.resize(total_pens);
    for (uint32_t i = 0; i < total_pens; ++i) {
        pens_[i].id = static_cast<rerot_pen_id>(i);
        pens_[i].state = server_pen_state::free;
        pens_[i].exec_seq = first_internal_seq_ + i < max_seq_ ? first_internal_seq_ + i : i;
    }
}

uint32_t server_rerot_runtime::pen_capacity() const {
    return static_cast<uint32_t>(pens_.size());
}

uint32_t server_rerot_runtime::pens_allocated() const {
    uint32_t count = 0;
    for (const auto & p : pens_) {
        if (p.state != server_pen_state::free) {
            ++count;
        }
    }
    return count;
}

uint32_t server_rerot_runtime::pens_running() const {
    uint32_t count = 0;
    for (const auto & p : pens_) {
        if (p.state == server_pen_state::running) {
            ++count;
        }
    }
    return count;
}

const server_pen * server_rerot_runtime::pen(rerot_pen_id id) const {
    if (id >= 0 && (size_t) id < pens_.size()) {
        return &pens_[id];
    }
    return nullptr;
}

server_pen * server_rerot_runtime::pen(rerot_pen_id id) {
    if (id >= 0 && (size_t) id < pens_.size()) {
        return &pens_[id];
    }
    return nullptr;
}

std::vector<rerot_pen_id> server_rerot_runtime::pens_for_person(rerot_person_id person) const {
    std::vector<rerot_pen_id> result;
    for (const auto & p : pens_) {
        if (p.state != server_pen_state::free && p.person == person) {
            result.push_back(p.id);
        }
    }
    return result;
}

std::optional<rerot_pen_id> server_rerot_runtime::allocate_pen(
        rerot_person_id person,
        uint64_t episode_id,
        llama_rerot_node_id node_id) {
    for (auto & p : pens_) {
        if (p.state == server_pen_state::free) {
            p.state = server_pen_state::allocated;
            p.person = person;
            p.episode_id = episode_id;
            p.node_id = node_id;
            return p.id;
        }
    }
    return std::nullopt;
}

void server_rerot_runtime::free_pen(rerot_pen_id pen_id) {
    if (pen_id >= 0 && (size_t) pen_id < pens_.size()) {
        pens_[pen_id].state = server_pen_state::free;
        pens_[pen_id].person = 0;
        pens_[pen_id].episode_id = 0;
        pens_[pen_id].node_id = LLAMA_REROT_NODE_INVALID;
    }
}

uint32_t server_rerot_runtime::pens_suspended() const {
    uint32_t count = 0;
    for (const auto & [_, ep] : episodes_) {
        count += ep.suspended.size();
    }
    return count;
}

uint32_t server_rerot_runtime::pen_queue_depth() const {
    uint32_t count = 0;
    for (const auto & [_, ep] : episodes_) {
        count += static_cast<uint32_t>(ep.ready_queue.size());
    }
    return count;
}

bool server_rerot_runtime::suspend_pen(rerot_pen_id pen_id) {
    if (pen_id < 0 || (size_t) pen_id >= pens_.size()) {
        return false;
    }
    auto & p = pens_[pen_id];
    if (p.state == server_pen_state::free) {
        return false;
    }
    const uint64_t ep_id = p.episode_id;
    const llama_rerot_node_id nid = p.node_id;
    auto * ep = episode(ep_id);
    auto * n = node(ep_id, nid);
    if (!ep || !n || ep->hard_aborted) {
        return false;
    }

    // Preserve hand/private state + sampler/MTP binding in the node runtime (§B.8.4)
    ep->running.erase(nid);
    ep->starting.erase(nid);
    ep->suspended.insert(nid);
    ep->document.set_node_state(nid, llama_rerot_node_state::ready_suspended);

    n->pen_id = -1;
    n->physical_slot = -1;
    n->exec_seq = -1;

    // Release pen row back to free pool so another person or child can allocate it
    p.state = server_pen_state::free;
    p.person = 0;
    p.episode_id = 0;
    p.node_id = LLAMA_REROT_NODE_INVALID;
    slot_to_episode_.erase(pen_id);

    ep->topology_barrier_pending = true;
    ++ep->topology_epoch;
    ++ep->layout_epoch;
    return true;
}

bool server_rerot_runtime::resume_pen(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        rerot_pen_id pen_id,
        llama_seq_id exec_seq) {
    if (pen_id < 0 || (size_t) pen_id >= pens_.size() || exec_seq < 0) {
        return false;
    }
    auto * ep = episode(episode_id);
    auto * n = node(episode_id, node_id);
    if (!ep || !n || ep->hard_aborted || ep->suspended.count(node_id) == 0) {
        return false;
    }
    if (ep->is_dag && !ep->c_base.valid()) {
        return false;
    }
    auto & p = pens_[pen_id];
    if (p.state != server_pen_state::allocated && p.state != server_pen_state::free) {
        return false;
    }

    p.state = server_pen_state::running;
    p.person = episode_id;
    p.episode_id = episode_id;
    p.node_id = node_id;
    p.exec_seq = exec_seq;
    slot_to_episode_[pen_id] = episode_id;

    n->pen_id = pen_id;
    n->physical_slot = pen_id;
    n->exec_seq = exec_seq;

    if (memory_ && n->parked_seq >= 0) {
        llama_memory_seq_cp_attention(memory_, n->parked_seq, exec_seq, -1, -1);
        if (!n->hand_seed.empty() &&
            !llama_memory_rerot_apply_hand_seed(
                memory_, exec_seq, n->hand_seed.data(), n->hand_seed.size())) {
            return false;
        }
    }

    ep->suspended.erase(node_id);
    ep->running.insert(node_id);
    const llama_rerot_node_state resume_state = ep->is_dag
        ? llama_rerot_node_state::running
        : (n->planner_armed
            ? llama_rerot_node_state::planning
            : llama_rerot_node_state::terminal_running);
    ep->document.set_node_state(node_id, resume_state);

    ep->topology_barrier_pending = true;
    ++ep->topology_epoch;
    ++ep->layout_epoch;
    return true;
}

size_t server_rerot_runtime::schedule_pens(const std::vector<uint64_t> & ready_people) {
    if (pens_.empty() || ready_people.empty()) {
        return 0;
    }
    size_t newly_admitted = 0;

    auto try_fill_one = [&](uint64_t person_id, server_rerot_episode * ep) -> bool {
        auto try_resume = [&]() -> bool {
            if (ep->suspended.empty()) {
                return false;
            }
            const llama_rerot_node_id suspended_nid = *ep->suspended.begin();
            auto pen_opt = allocate_pen(person_id, person_id, suspended_nid);
            if (!pen_opt) {
                return false;
            }
            const llama_seq_id seq = pens_[*pen_opt].exec_seq;
            if (resume_pen(person_id, suspended_nid, *pen_opt, seq)) {
                ++newly_admitted;
                return true;
            }
            free_pen(*pen_opt);
            return false;
        };
        auto try_admit = [&]() -> bool {
            if (ep->ready_queue.empty()) {
                return false;
            }
            auto pen_opt = allocate_pen(person_id, person_id, ep->ready_queue.front());
            if (!pen_opt) {
                return false;
            }
            llama_rerot_node_id admitted = LLAMA_REROT_NODE_INVALID;
            const llama_seq_id seq = pens_[*pen_opt].exec_seq;
            if (admit_next_child(person_id, *pen_opt, seq, &admitted)) {
                ++newly_admitted;
                return true;
            }
            free_pen(*pen_opt);
            return false;
        };
        // DAG W>P: drain queued STARTING (F_i) before resuming already-running workers.
        if (ep->is_dag) {
            if (try_admit()) {
                return true;
            }
            return try_resume();
        }
        if (try_resume()) {
            return true;
        }
        return try_admit();
    };

    // Pass 1 (§B.8.2): For each runnable person currently having 0 pens (by arrival/age), allocate 1 pen
    for (uint64_t person_id : ready_people) {
        auto * ep = episode(person_id);
        if (!ep || ep->hard_aborted || !has_ready_nodes(person_id)) {
            continue;
        }
        if (!pens_for_person(person_id).empty()) {
            continue; // already has at least one pen
        }
        if (!try_fill_one(person_id, ep)) {
            if (pens_allocated() >= pens_.size()) {
                return newly_admitted; // all pens exhausted
            }
        }
    }

    // Pass 2 (§B.8.2): Round-robin across people with ready children until P exhausted or no runnable work
    bool any_admitted = true;
    while (any_admitted) {
        any_admitted = false;
        for (uint64_t person_id : ready_people) {
            auto * ep = episode(person_id);
            if (!ep || ep->hard_aborted || !has_ready_nodes(person_id)) {
                continue;
            }
            if (pens_allocated() >= pens_.size()) {
                return newly_admitted; // all P pens exhausted
            }
            if (try_fill_one(person_id, ep)) {
                any_admitted = true;
            }
        }
    }
    return newly_admitted;
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
            current_node.pen_id = -1;
            current_node.physical_slot = -1;
            current_node.exec_seq = -1;
        }
    }

    if (physical_slot >= 0 && (size_t) physical_slot < pens_.size()) {
        pens_[physical_slot].state = server_pen_state::free;
        pens_[physical_slot].person = 0;
        pens_[physical_slot].episode_id = 0;
        pens_[physical_slot].node_id = LLAMA_REROT_NODE_INVALID;
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
        llama_pos storage_pos,
        llama_rerot_segment_kind kind) {
    std::optional<llama_rerot_run_id> * active = nullptr;
    switch (visibility) {
        case llama_rerot_visibility::public_live:     active = &node.public_run; break;
        case llama_rerot_visibility::private_control: active = &node.private_run; break;
        case llama_rerot_visibility::pending_record:  active = &node.pending_record; break;
        case llama_rerot_visibility::normal: return LLAMA_REROT_RUN_INVALID;
    }

    if (active->has_value()) {
        const auto * current_run = episode.document.run(**active);
        // Runs are keyed by owner + visibility + contiguity + segment kind:
        // adjacent FRAME/BODY (or BODY/source_end) spans share visibility and
        // positions but must never merge, or formal-P text would dissolve
        // into exportable BODY and end boundaries into body text.
        if (current_run && current_run->visibility == visibility &&
            current_run->kind == kind &&
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
        node.id, visibility, storage_pos, 0, publish_epoch, kind);
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
        (current_run_id != LLAMA_REROT_RUN_INVALID && *node.pending_record != current_run_id) ||
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

    const auto kind = current->probing
        ? llama_rerot_segment_kind::probe_control
        : llama_rerot_segment_kind::body;
    const llama_rerot_run_id run_id = ensure_run(
        *current, *current_node, llama_rerot_visibility::private_control, storage_pos, kind);
    if (run_id == LLAMA_REROT_RUN_INVALID) {
        return std::nullopt;
    }

    std::vector<server_rerot_token_plan> plans;
    plans.reserve(token_count);
    for (size_t i = 0; i < token_count; ++i) {
        server_rerot_token_plan plan;
        plan.storage_pos = storage_pos + static_cast<llama_pos>(i);
        plan.visibility = llama_rerot_visibility::private_control;
        plan.segment_kind = kind;
        plan.run_id = run_id;
        plans.push_back(std::move(plan));
    }
    if (current->probing) {
        current->probe_tokens += token_count;
    }
    return plans;
}

std::optional<std::vector<server_rerot_token_plan>> server_rerot_runtime::plan_frame_span(
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
        *current, *current_node, llama_rerot_visibility::pending_record, storage_pos,
        llama_rerot_segment_kind::frame);
    if (run_id == LLAMA_REROT_RUN_INVALID) {
        return std::nullopt;
    }

    std::vector<server_rerot_token_plan> plans;
    plans.reserve(token_count);
    for (size_t i = 0; i < token_count; ++i) {
        server_rerot_token_plan plan;
        plan.storage_pos = storage_pos + static_cast<llama_pos>(i);
        plan.visibility = llama_rerot_visibility::pending_record;
        plan.segment_kind = llama_rerot_segment_kind::frame;
        plan.event_origin = llama_rerot_event_origin::runtime_frame;
        plan.run_id = run_id;
        plans.push_back(std::move(plan));
    }
    current->frame_tokens += token_count;
    return plans;
}

std::optional<std::vector<server_rerot_token_plan>> server_rerot_runtime::plan_public_span(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        llama_pos storage_pos,
        size_t token_count,
        llama_rerot_segment_kind kind) {
    auto * current = episode(episode_id);
    auto * current_node = node(episode_id, node_id);
    if (!current || !current_node || current->hard_aborted || storage_pos < 0 ||
        token_count == 0 ||
        token_count > size_t(std::numeric_limits<llama_pos>::max() - storage_pos) + 1) {
        return std::nullopt;
    }

    const llama_rerot_run_id run_id = ensure_run(
        *current, *current_node, llama_rerot_visibility::public_live, storage_pos, kind);
    if (run_id == LLAMA_REROT_RUN_INVALID) {
        return std::nullopt;
    }

    std::vector<server_rerot_token_plan> plans;
    plans.reserve(token_count);
    for (size_t i = 0; i < token_count; ++i) {
        server_rerot_token_plan plan;
        plan.storage_pos = storage_pos + static_cast<llama_pos>(i);
        plan.visibility = llama_rerot_visibility::public_live;
        plan.segment_kind = kind;
        plan.run_id = run_id;
        plans.push_back(std::move(plan));
    }
    // Forced framing forwards (rebuild suffix + plan text) are real compute:
    // ledger them here so they can never hide inside useful-BODY accounting.
    // On commit they additionally enter generated_public_tokens by visibility,
    // exactly like published worker FRAME — never as useful BODY, since the
    // export set keys on segment kind, not visibility.
    current->frame_tokens += token_count;
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

    if (current->probing) {
        server_rerot_token_plan plan;
        plan.storage_pos = storage_pos;
        plan.visibility = llama_rerot_visibility::private_control;
        plan.segment_kind = llama_rerot_segment_kind::probe_control;
        plan.run_id = ensure_run(
            *current, *current_node, plan.visibility, storage_pos, plan.segment_kind);
        ++current->probe_tokens;
        return plan.valid() ? std::optional<server_rerot_token_plan>(std::move(plan)) : std::nullopt;
    }

    if (current->is_dag) {
        server_rerot_token_plan plan;
        plan.storage_pos = storage_pos;
        const auto * logical = current->document.node(node_id);
        const bool starting = logical && logical->state == llama_rerot_node_state::starting;
        if (starting) {
            plan.visibility = llama_rerot_visibility::pending_record;
            plan.segment_kind = llama_rerot_segment_kind::frame;
            plan.event_origin = llama_rerot_event_origin::runtime_frame;
            plan.run_id = ensure_run(
                *current, *current_node, plan.visibility, storage_pos, plan.segment_kind);
            return plan.valid() ? std::optional<server_rerot_token_plan>(std::move(plan)) : std::nullopt;
        }
        if (current_node->exit_parser.marker().empty()) {
            plan.visibility = llama_rerot_visibility::pending_record;
            plan.segment_kind = llama_rerot_segment_kind::body;
            plan.run_id = ensure_run(
                *current, *current_node, plan.visibility, storage_pos, plan.segment_kind);
            return plan.valid() ? std::optional<server_rerot_token_plan>(std::move(plan)) : std::nullopt;
        }
        auto marker_step = current_node->exit_parser.consume(token_bytes);
        if (marker_step.malformed) {
            fail_episode(*current, marker_step.error.empty()
                ? "malformed RERoT native source-end marker"
                : marker_step.error);
            return std::nullopt;
        }
        plan.marker_step = marker_step;
        if (marker_step.marker_closed) {
            plan.visibility = llama_rerot_visibility::private_control;
            plan.segment_kind = llama_rerot_segment_kind::source_end;
            plan.event_origin = current_node->stage_role == llama_rerot_stage_role::synthesis
                ? llama_rerot_event_origin::synthesis_source
                : llama_rerot_event_origin::worker_source;
            ++current->source_end_tokens;
        } else {
            plan.visibility = llama_rerot_visibility::pending_record;
            plan.segment_kind = llama_rerot_segment_kind::body;
        }
        plan.run_id = ensure_run(
            *current, *current_node, plan.visibility, storage_pos, plan.segment_kind);
        return plan.valid() ? std::optional<server_rerot_token_plan>(std::move(plan)) : std::nullopt;
    }

    server_rerot_parser_step parser_step;
    server_rerot_marker_step marker_step;
    if (current_node->planner_armed) {
        // A child or root may recursively fork or close its own random delimiter.
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

        // The list parser owns all bytes of an unfinished structural record.
        // Marker-like substrings inside it must not escape as public prefixes
        // or be moved ahead of the record when its bytes are rendered.
        if (current_node->parser.state() == server_rerot_parser_state::opening_candidate ||
            current_node->parser.state() == server_rerot_parser_state::list_pending ||
            parser_step.record_closed) {
            marker_step.public_prefix_bytes = 0;
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
            if (current_node->pending_record.has_value() &&
                !release_false_pending(*current, *current_node)) {
                return std::nullopt;
            }
        }

        if (marker_step.marker_closed) {
            current_node->planner_armed = false;
            parser_step.write_visibility = llama_rerot_visibility::private_control;
        } else if (parser_step.record_closed) {
            // Completed <ol> record; visibility handled by parser_step (pending_record, will be published in commit_token)
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

        const bool candidate_alive =
            current_node->exit_parser.state() == server_rerot_marker_state::marker_candidate;

        if (marker_step.release_previous_pending && !candidate_alive) {
            if (current_node->pending_record.has_value() &&
                !release_false_pending(*current, *current_node)) {
                return std::nullopt;
            }
        }

        if (marker_step.marker_closed) {
            parser_step.write_visibility = llama_rerot_visibility::private_control;
        } else if (candidate_alive) {
            parser_step.write_visibility = llama_rerot_visibility::pending_record;
        } else {
            parser_step.write_visibility = llama_rerot_visibility::public_live;
        }
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
    const auto view = episode.is_dag
        ? build_dag_view_for_reader(episode.id, node.id)
        : episode.document.build_view(node.id);
    ordered_runs.clear();
    ordered_runs.reserve(view.runs.size() + 1);
    // Explicit research control, never selected as an automatic fallback.
    // With local-state this separates lexical sharing from native branch
    // continuation while leaving storage, DDVR and execution batching intact.
    const bool ancestors_only = std::getenv("LLAMA_REROT_ANCESTORS_ONLY") != nullptr;
    std::unordered_set<llama_rerot_node_id> ancestors;
    if (ancestors_only) {
        const auto * ancestor = episode.document.node(node.id);
        while (ancestor && ancestors.insert(ancestor->id).second) {
            ancestor = episode.document.node(ancestor->parent);
        }
    }
    for (const auto & current_run : view.runs) {
        if (ancestors_only && ancestors.count(current_run.owner) == 0) continue;
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
        child.exit_parser = server_rerot_marker_parser("</" + random_string(8) + ">");
        // A child does not infer scheduler intent from ordinary HTML. Recursive
        // planning is armed only by the explicit PRIVATE planner phase.
        child.planner_armed = false;
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
    if (node.pending_record.has_value()) {
        uint64_t privatized_tokens = 0;
        if (!resolve_pending_record_runs(
                episode, node, *node.pending_record,
                llama_rerot_visibility::private_control,
                0,
                privatized_tokens)) {
            return false;
        }
        episode.pending_tokens -= privatized_tokens;
        episode.generated_private_tokens += privatized_tokens;
        node.pending_record.reset();
    }

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
    current_node->last_write_public = (plan.visibility == llama_rerot_visibility::public_live);

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
    if (current->is_dag) {
        note_dag_step_commit(*current, node_id);
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
        if (current->is_dag) {
            if (plan.event_origin == llama_rerot_event_origin::runtime_frame ||
                plan.event_origin == llama_rerot_event_origin::foreign_export ||
                plan.event_origin == llama_rerot_event_origin::unknown) {
                return true;
            }
            current_node->exit_intent = true;
            if (current_node->stage_role == llama_rerot_stage_role::synthesis) {
                // Keep the synthesis pen bound for ordinary serial content.
                return true;
            }
            return seal_dag_node(current->id, current_node->id, plan.event_origin);
        }
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

void server_rerot_runtime::release_probe_seq(server_rerot_episode & episode) {
    if (episode.probe_seq < 0) {
        return;
    }
    const llama_seq_id probe = episode.probe_seq;
    if (memory_) {
        llama_memory_seq_rm_attention(memory_, probe, -1, -1);
        llama_memory_seq_rm_recurrent(memory_, probe, -1, -1);
        llama_memory_seq_rm(memory_, probe, -1, -1);
    }
    clear_sequence_control(probe);
    free_internal_seq(probe);
    episode.probe_seq = -1;
    auto * root = episode.nodes.empty() ? nullptr : &episode.nodes[0];
    if (root && root->exec_seq == probe) {
        root->exec_seq = episode.c0.seq_id >= 0 ? episode.c0.seq_id : -1;
        if (root->pen_id >= 0 && (size_t) root->pen_id < pens_.size()) {
            pens_[root->pen_id].exec_seq = root->exec_seq;
        }
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
    // Run refs are located by stable (episode, run) identity, not by
    // sequence: a sealed passivated worker's cells live on its parked seq.
    if (!memory_ || archive_seq < 0 || (node.exec_seq < 0 && node.parked_seq < 0)) {
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
    // Retirement never drops uncommitted spans: the DAG caller retires sealed
    // workers only after cohort publication cleared their pending records.
    if (node.pending_record.has_value()) {
        return fail_episode(episode, "invalid RERoT Lane retirement state");
    }
    if (node.physical_slot >= 0 && node.exec_seq >= 0) {
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

        if (physical_slot >= 0 && (size_t) physical_slot < pens_.size()) {
            pens_[physical_slot].state = server_pen_state::free;
            pens_[physical_slot].person = 0;
            pens_[physical_slot].episode_id = 0;
            pens_[physical_slot].node_id = LLAMA_REROT_NODE_INVALID;
        }

        episode.running.erase(node.id);
        episode.starting.erase(node.id);
        node.exit_intent = false;
        node.fence = {};
        node.pen_id = -1;
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

    // Sealed DAG passivation: no live executor, but the parked seq still pins
    // attention + lineage for the archived PUBLIC history. Archive from the
    // parked source, release it, and clear the marker so the retired
    // notification below emits exactly once.
    if (!(episode.is_dag && node.is_sealed && node.physical_slot < 0 &&
          node.exec_seq < 0 && node.parked_seq >= 0)) {
        return fail_episode(episode, "invalid RERoT Lane retirement state");
    }
    if (!ensure_archive_seq(episode, node.parked_seq)) {
        return false;
    }
    archive_public_runs(episode, node, episode.archive_seq);
    if (memory_) {
        llama_memory_seq_rm(memory_, node.parked_seq, -1, -1);
    }
    free_internal_seq(node.parked_seq);
    node.parked_seq = -1;

    episode.running.erase(node.id);
    episode.starting.erase(node.id);
    episode.suspended.erase(node.id);
    node.exit_intent = false;
    node.fence = {};
    node.pen_id = -1;
    node.physical_slot = -1;
    node.exec_seq = -1;
    if (!episode.document.set_node_state(node.id, llama_rerot_node_state::retired)) {
        return fail_episode(episode, "failed to mark retiring RERoT Lane as RETIRED");
    }

    episode.topology_barrier_pending = true;
    ++episode.topology_epoch;
    ++episode.layout_epoch;
    return true;
}

bool server_rerot_runtime::initialize_dag(
        uint64_t episode_id,
        const server_rerot_routing_decision & decision,
        std::string * error_out) {
    auto * ep = episode(episode_id);
    if (!ep) {
        if (error_out) *error_out = "episode not found";
        return false;
    }
    if (!decision.is_dag()) {
        if (error_out) *error_out = "invalid DAG decision: " + decision.error;
        return false;
    }
    if (ep->hard_aborted) {
        if (error_out) *error_out = "episode is hard aborted";
        return false;
    }
    if (ep->is_dag) {
        if (error_out) *error_out = "DAG already initialized (recursive nested DAG is not a production path)";
        return false;
    }

    // A pre-routing archive keeper (if any) pins ordinary-prefix cells; the
    // formal P is rebuilt after this point, so drop it here. The union is
    // recreated on demand by ensure_archive_seq from post-rebuild runs.
    if (ep->archive_seq >= 0) {
        if (memory_) {
            llama_memory_seq_rm(memory_, ep->archive_seq, -1, -1);
        }
        clear_sequence_control(ep->archive_seq);
        free_internal_seq(ep->archive_seq);
        ep->archive_seq = -1;
    }

    ep->is_dag = true;
    ep->strategy_decided = true;
    ep->document.set_dag_mode(true);
    ep->document.set_stage_role(0, llama_rerot_stage_role::planner);
    if (!ep->nodes.empty()) {
        ep->nodes[0].stage_role = llama_rerot_stage_role::planner;
        ep->nodes[0].planner_armed = false;
    }

    std::unordered_map<std::string, llama_rerot_node_id> str_to_nid;

    for (const auto & q : decision.questions) {
        auto nid = ep->document.create_child(ep->document.root(), q.intent, llama_rerot_node_state::queued);
        ep->document.set_plan_rank(nid, q.plan_rank);
        ep->document.set_stage_role(nid, llama_rerot_stage_role::worker);

        server_rerot_node_runtime nr;
        nr.id = nid;
        nr.string_id = q.id;
        nr.intent = q.intent;
        nr.planner_armed = false;
        nr.stage_role = llama_rerot_stage_role::worker;
        if (!ep->source_end_marker.empty()) {
            nr.exit_parser = server_rerot_marker_parser(ep->source_end_marker, true);
        }

        if (nid >= ep->nodes.size()) {
            ep->nodes.resize(nid + 1);
        }
        ep->nodes[nid] = std::move(nr);
        str_to_nid[q.id] = nid;
    }

    for (const auto & dep : decision.dependencies) {
        auto from_nid = str_to_nid.at(dep.from_id);
        auto to_nid = str_to_nid.at(dep.to_id);
        if (!ep->document.add_edge(from_nid, to_nid, error_out)) {
            return false;
        }
    }

    for (const auto & q : decision.questions) {
        if (!ep->document.add_edge(0, str_to_nid.at(q.id), error_out)) {
            return false;
        }
    }

    auto synth = ep->document.create_child(ep->document.root(), "0.synthesize", llama_rerot_node_state::queued);
    ep->document.set_plan_rank(synth, static_cast<uint32_t>(decision.questions.size()));
    ep->document.set_stage_role(synth, llama_rerot_stage_role::synthesis);
    if (synth >= ep->nodes.size()) {
        ep->nodes.resize(synth + 1);
    }
    server_rerot_node_runtime synth_nr;
    synth_nr.id = synth;
    synth_nr.string_id = "0";
    synth_nr.intent = "0.synthesize";
    synth_nr.planner_armed = false;
    synth_nr.stage_role = llama_rerot_stage_role::synthesis;
    if (!ep->source_end_marker.empty()) {
        synth_nr.exit_parser = server_rerot_marker_parser(ep->source_end_marker, true);
    }
    ep->nodes[synth] = std::move(synth_nr);
    ep->synthesis_node = synth;
    for (const auto & q : decision.questions) {
        if (!ep->document.add_edge(str_to_nid.at(q.id), synth, error_out)) {
            return false;
        }
    }

    for (size_t i = 0; i < ep->nodes.size(); ++i) {
        const auto * doc_n = ep->document.node(static_cast<llama_rerot_node_id>(i));
        if (doc_n) {
            ep->nodes[i].remaining_preds = static_cast<uint32_t>(doc_n->predecessors.size());
        }
    }

    // Plan submission completes 0.plan. Decrement worker remaining_preds exactly once.
    ep->nodes[0].remaining_preds = 0;
    ep->nodes[0].is_sealed = true;
    ep->document.set_node_state(0, llama_rerot_node_state::retired);
    const auto * root_doc = ep->document.node(0);
    if (root_doc) {
        for (const auto succ : root_doc->successors) {
            if (succ < ep->nodes.size() && ep->nodes[succ].remaining_preds > 0) {
                --ep->nodes[succ].remaining_preds;
            }
        }
    }

    // Descriptors are published here (nodes, edges, remaining_preds, and the
    // root plan seal above), but no ready work exists yet. Physical readiness
    // is distinct from logical plan commit: workers cannot yield, admit, or
    // form a cohort until the formal base is captured after the forced root
    // P injection. activate_dag_frontier enqueues once c_base is valid.
    ep->ready_queue.clear();

    // The ordinary prefix is model-PUBLIC control, never API BODY: flip any
    // root BODY runs to FRAME so the reasoning export (which skips frame,
    // source_end, and probe_control runs) cannot leak template/prompt text.
    // Visibility, positions, counts, and epochs are untouched, so
    // build_dag_view keeps the full P for attention. probe_control (and any
    // other non-body kind) is left alone.
    const auto * root_runs = ep->document.node(0);
    if (!root_runs) {
        if (error_out) {
            *error_out = "root document state is missing";
        }
        return false;
    }
    for (const auto rid : root_runs->runs) {
        const auto * run = ep->document.run(rid);
        if (!run || run->kind != llama_rerot_segment_kind::body) {
            continue;
        }
        if (!ep->document.set_run_segment_kind(rid, llama_rerot_segment_kind::frame)) {
            return fail_episode(*ep, "RERoT DAG init could not frame the ordinary prefix");
        }
    }

    return true;
}

void server_rerot_runtime::set_dag_protocol_markers(
        uint64_t episode_id,
        std::string_view source_end_marker,
        std::string_view think_start_marker) {
    auto * ep = episode(episode_id);
    if (!ep) {
        return;
    }
    ep->source_end_marker = std::string(source_end_marker);
    ep->think_start_marker = std::string(think_start_marker);
}

bool server_rerot_runtime::capture_c0(uint64_t episode_id, llama_seq_id seq_id, llama_pos n_prompt_tokens) {
    auto * ep = episode(episode_id);
    if (!ep || seq_id < 0 || n_prompt_tokens < 0) {
        return false;
    }
    ep->c0.seq_id = seq_id;
    ep->c0.n_prompt_tokens = n_prompt_tokens;
    ep->c0.captured = true;
    auto * root = node(episode_id, 0);
    if (memory_ && seq_id >= 0) {
        const size_t seed_size =
            llama_memory_rerot_capture_hand_seed(memory_, seq_id, nullptr, 0);
        if (seed_size > 0) {
            std::vector<uint8_t> seed(seed_size);
            if (llama_memory_rerot_capture_hand_seed(
                    memory_, seq_id, seed.data(), seed.size()) != seed.size()) {
                return false;
            }
            if (root) {
                root->hand_seed = seed;
            }
            checkpoint_fill_from_hand_seed(ep->c0, std::move(seed));
        }
    }
    if (ep->c0.gdn_recurrent_states.empty() && root && !root->hand_seed.empty()) {
        checkpoint_fill_from_hand_seed(ep->c0, root->hand_seed);
    }
    return ep->c0.valid();
}

void server_rerot_runtime::capture_checkpoint_sampler(
        server_rerot_prebranch_checkpoint & cp,
        uint32_t seed,
        const llama_tokens & prev) {
    cp.sampler_snapshot_bytes.clear();
    const uint32_t n_prev = static_cast<uint32_t>(prev.size());
    const size_t bytes =
        sizeof(uint32_t) * 3 + sizeof(llama_token) * prev.size();
    cp.sampler_snapshot_bytes.resize(bytes);
    uint8_t * p = cp.sampler_snapshot_bytes.data();
    auto write_u32 = [&](uint32_t v) {
        std::memcpy(p, &v, sizeof(uint32_t));
        p += sizeof(uint32_t);
    };
    write_u32(k_sampler_snap_magic);
    write_u32(seed);
    write_u32(n_prev);
    if (!prev.empty()) {
        std::memcpy(p, prev.data(), sizeof(llama_token) * prev.size());
    }
}

bool server_rerot_runtime::arm_isolated_probe(uint64_t episode_id, llama_seq_id c0_seq) {
    auto * ep = episode(episode_id);
    auto * root = node(episode_id, 0);
    if (!ep || !root || c0_seq < 0 || !ep->c0.valid()) {
        return false;
    }
    if (ep->probe_seq >= 0) {
        return root->exec_seq == ep->probe_seq;
    }
    llama_seq_id skipped = -1;
    auto probe = alloc_internal_seq();
    if (probe.has_value() && *probe == c0_seq) {
        skipped = *probe;
        probe = alloc_internal_seq();
    }
    if (skipped >= 0) {
        free_internal_seq(skipped);
    }
    if (!probe.has_value()) {
        return false;
    }
    if (memory_) {
        llama_memory_seq_cp_attention(memory_, c0_seq, *probe, -1, -1);
        llama_memory_seq_cp_recurrent(memory_, c0_seq, *probe, -1, -1);
    }
    ep->probe_seq = *probe;
    root->exec_seq = *probe;
    if (root->pen_id >= 0 && (size_t) root->pen_id < pens_.size()) {
        pens_[root->pen_id].exec_seq = *probe;
    }
    return true;
}

bool server_rerot_runtime::discard_isolated_probe(uint64_t episode_id, llama_seq_id c0_seq) {
    auto * ep = episode(episode_id);
    auto * root = node(episode_id, 0);
    if (!ep || ep->probe_seq < 0) {
        return false;
    }
    release_probe_seq(*ep);
    // Physical removal alone leaves stale spans: probe tokens were planned as
    // probe_control runs on the wiped positions. Retract them (ids stay dense
    // for save/load; probe_tokens keeps the real-work accounting).
    for (size_t i = 0; i < ep->document.run_count(); ++i) {
        const auto rid = static_cast<llama_rerot_run_id>(i);
        const auto * run = ep->document.run(rid);
        if (run && run->kind == llama_rerot_segment_kind::probe_control && run->token_count > 0) {
            if (!ep->document.set_run_token_count(rid, 0)) {
                return fail_episode(*ep, "RERoT probe discard could not retract probe spans");
            }
        }
    }
    if (root) {
        root->exec_seq = c0_seq;
        if (root->pen_id >= 0 && (size_t) root->pen_id < pens_.size()) {
            pens_[root->pen_id].exec_seq = c0_seq;
        }
        if (ep->c0.n_prompt_tokens >= 0) {
            root->storage_pos_next = ep->c0.n_prompt_tokens;
        }
    }
    return true;
}

bool server_rerot_runtime::capture_c_base(uint64_t episode_id) {
    auto * ep = episode(episode_id);
    auto * root = node(episode_id, 0);
    if (!ep || !root) {
        return false;
    }
    // Snapshot current root hand_seed + watermark. Overlay onto C0 extras
    // (sampler/conv) but never keep a stale C0 seed when the root seed is newer.
    if (ep->c0.valid()) {
        ep->c_base = ep->c0;
    } else {
        ep->c_base.clear();
    }
    if (root->exec_seq >= 0) {
        ep->c_base.seq_id = root->exec_seq;
    }
    if (root->storage_pos_next >= 0) {
        ep->c_base.n_prompt_tokens = root->storage_pos_next;
    }
    if (memory_ && root->exec_seq >= 0) {
        const size_t seed_size =
            llama_memory_rerot_capture_hand_seed(memory_, root->exec_seq, nullptr, 0);
        if (seed_size > 0) {
            root->hand_seed.resize(seed_size);
            if (llama_memory_rerot_capture_hand_seed(
                    memory_,
                    root->exec_seq,
                    root->hand_seed.data(),
                    root->hand_seed.size()) != root->hand_seed.size()) {
                return false;
            }
        }
    }
    if (!root->hand_seed.empty()) {
        checkpoint_fill_from_hand_seed(ep->c_base, root->hand_seed);
    }
    ep->c_base.captured = true;
    return ep->c_base.valid();
}

bool server_rerot_runtime::activate_dag_frontier(uint64_t episode_id) {
    auto * ep = episode(episode_id);
    if (!ep || !ep->is_dag) {
        return false;
    }
    // No cohort, ready work, or time-slicing before the formal base exists:
    // lineage, storage watermarks, and sampler clones are undefined until
    // the forced root P injection is captured.
    if (!ep->c_base.valid()) {
        return false;
    }
    const auto eligible = get_eligible_dag_nodes(episode_id);
    for (const auto nid : eligible) {
        if (std::find(ep->ready_queue.begin(), ep->ready_queue.end(), nid) == ep->ready_queue.end()) {
            ep->ready_queue.push_back(nid);
        }
        if (nid < ep->nodes.size()) {
            ep->nodes[nid].enqueue_frontier = ep->frontier;
        }
    }
    yield_dag_pen_for_ready(episode_id);
    snapshot_dag_logical_step(episode_id);
    return true;
}

bool server_rerot_runtime::has_free_pen() const {
    for (const auto & p : pens_) {
        if (p.state == server_pen_state::free) {
            return true;
        }
    }
    return false;
}

void server_rerot_runtime::snapshot_dag_logical_step(uint64_t episode_id) {
    auto * ep = episode(episode_id);
    if (!ep || !ep->is_dag || ep->hard_aborted || ep->serial_tail) {
        return;
    }
    if (!ep->c_base.valid()) {
        return;
    }
    if (!ep->dag_step_cohort.empty()) {
        return;
    }
    const auto consider = [&](llama_rerot_node_id nid) {
        if (nid == 0 || nid >= ep->nodes.size()) {
            return;
        }
        const auto & nr = ep->nodes[nid];
        if (nr.is_sealed) {
            return;
        }
        if (nr.stage_role != llama_rerot_stage_role::worker &&
            nr.stage_role != llama_rerot_stage_role::synthesis) {
            return;
        }
        ep->dag_step_cohort.insert(nid);
    };
    for (const auto nid : ep->running) {
        consider(nid);
    }
    for (const auto nid : ep->starting) {
        consider(nid);
    }
    for (const auto nid : ep->suspended) {
        consider(nid);
    }
    for (const auto nid : ep->ready_queue) {
        consider(nid);
    }
    ep->dag_step_committed.clear();
    // This-step BODY stays PENDING until the cohort publishes at the step
    // boundary. The PUBLIC watermark is frozen at the last committed frontier
    // (advance_frontier) and never moves here: every foreign PUBLIC run —
    // BODY and FRAME alike — stamped above it stays invisible to later
    // physical slices of this step; only the reader's own runs are exempt
    // (document build_dag_view). Mid-step FRAME publication bumps the publish
    // counter but not the frozen watermark, so admitting or completing a
    // worker can never leak a newly completed entry into this step's world.
}

bool server_rerot_runtime::has_open_dag_logical_step(uint64_t episode_id) const {
    const auto * ep = episode(episode_id);
    return ep && ep->is_dag && !ep->dag_step_cohort.empty();
}

// Batch-construction guard: a bound member already done for the open logical
// step (sealed or committed) must not be fed again while other cohort members
// are still pending. No new logical step is inferred from physical order.
bool server_rerot_runtime::dag_step_node_done(uint64_t episode_id, llama_rerot_node_id node_id) const {
    const auto * ep = episode(episode_id);
    if (!ep || !ep->is_dag || ep->dag_step_cohort.empty()) {
        return false;
    }
    return dag_step_member_done(*ep, node_id);
}

bool server_rerot_runtime::dag_step_member_done(
        const server_rerot_episode & episode,
        llama_rerot_node_id node_id) const {
    if (node_id >= episode.nodes.size()) {
        return true;
    }
    if (episode.nodes[node_id].is_sealed) {
        return true;
    }
    return episode.dag_step_committed.count(node_id) != 0;
}

bool server_rerot_runtime::dag_logical_step_complete(uint64_t episode_id) const {
    const auto * ep = episode(episode_id);
    if (!ep || !ep->is_dag) {
        return true;
    }
    if (ep->dag_step_cohort.empty()) {
        return true;
    }
    for (const auto nid : ep->dag_step_cohort) {
        if (!dag_step_member_done(*ep, nid)) {
            return false;
        }
    }
    return true;
}

std::optional<llama_rerot_node_id> server_rerot_runtime::dag_step_next_pending(
        uint64_t episode_id) const {
    const auto * ep = episode(episode_id);
    if (!ep || ep->dag_step_cohort.empty()) {
        return std::nullopt;
    }
    for (const auto nid : ep->dag_step_cohort) {
        if (dag_step_member_done(*ep, nid)) {
            continue;
        }
        const auto * n = node(episode_id, nid);
        if (!n || n->physical_slot >= 0) {
            continue;
        }
        if (ep->suspended.count(nid) != 0 ||
            std::find(ep->ready_queue.begin(), ep->ready_queue.end(), nid) != ep->ready_queue.end()) {
            return nid;
        }
    }
    return std::nullopt;
}

void server_rerot_runtime::note_dag_step_commit(
        server_rerot_episode & episode,
        llama_rerot_node_id node_id) {
    if (!episode.is_dag || episode.dag_step_cohort.empty()) {
        return;
    }
    if (episode.dag_step_cohort.count(node_id) == 0) {
        return;
    }
    episode.dag_step_committed.insert(node_id);
}

void server_rerot_runtime::rollback_dag_step_publish(
        server_rerot_episode & episode,
        const std::vector<dag_step_publish_stage> & staged) {
    // Unwind staged cohort publishes so a mid-cohort failure leaves no half
    // frontier behind. The logical document is the transaction record;
    // physical reclassification is attempted best-effort (the episode is
    // already aborted and its cells are dropped with it).
    for (const auto & entry : staged) {
        auto * n = node(episode.id, entry.node_id);
        const auto * dn = episode.document.node(entry.node_id);
        if (dn && entry.epoch != 0) {
            for (const auto rid : dn->runs) {
                const auto * run = episode.document.run(rid);
                if (run && run->owner == entry.node_id &&
                    run->visibility == llama_rerot_visibility::public_live &&
                    run->publish_epoch == entry.epoch) {
                    if (memory_) {
                        llama_memory_rerot_reclassify_run(memory_, episode.id, rid,
                            LLAMA_REROT_KV_PUBLIC_LIVE, LLAMA_REROT_KV_PENDING_RECORD, 0);
                    }
                    episode.document.reclassify_run(rid,
                        llama_rerot_visibility::public_live,
                        llama_rerot_visibility::pending_record, 0);
                }
            }
        }
        if (n) {
            n->pending_record = entry.pending;
            n->public_run = entry.published_over;
        }
    }
}

bool server_rerot_runtime::publish_dag_step_bodies(server_rerot_episode & episode) {
    const uint64_t epoch_before = episode.publish_epoch;
    const uint64_t pending_before = episode.pending_tokens;
    const uint64_t public_before = episode.generated_public_tokens;
    std::vector<dag_step_publish_stage> staged;
    staged.reserve(episode.dag_step_cohort.size());
    for (const auto nid : episode.dag_step_cohort) {
        auto * n = node(episode.id, nid);
        if (!n || !n->pending_record.has_value()) {
            continue;
        }
        if (n->exit_parser.state() == server_rerot_marker_state::marker_candidate) {
            continue;
        }
        const auto * run = episode.document.run(*n->pending_record);
        if (!run || run->kind != llama_rerot_segment_kind::body) {
            continue;
        }
        dag_step_publish_stage entry;
        entry.node_id = nid;
        entry.pending = n->pending_record;
        entry.published_over = n->public_run;
        const auto pending_id = *n->pending_record;
        const uint64_t epoch_before_member = episode.publish_epoch;
        if (!publish_pending_run(episode, *n, pending_id, false)) {
            // The failing member may itself be partially reclassified, but
            // only if the counter actually advanced for it. On counter
            // overflow the epoch retains UINT64_MAX and must match nothing:
            // reusing it would revert an older committed run of the same
            // node stamped at max.
            entry.epoch = episode.publish_epoch != epoch_before_member
                ? episode.publish_epoch
                : 0;
            staged.push_back(entry);
            rollback_dag_step_publish(episode, staged);
            episode.publish_epoch = epoch_before;
            episode.pending_tokens = pending_before;
            episode.generated_public_tokens = public_before;
            return false;
        }
        if (const auto * done = episode.document.run(pending_id)) {
            entry.epoch = done->publish_epoch;
        }
        staged.push_back(entry);
    }
    return true;
}

bool server_rerot_runtime::prepare_dag_prefix_rebuild(
        uint64_t episode_id, llama_pos base, std::string * error_out) {
    const auto refuse = [&](const std::string & msg) {
        if (error_out) {
            *error_out = msg;
        }
        return false;
    };
    auto * ep = episode(episode_id);
    auto * root = node(episode_id, 0);
    if (!ep || !root) {
        return refuse("episode not found");
    }
    if (base < 0) {
        return refuse("RERoT prefix rebuild refused: negative prefix base");
    }
    if (ep->hard_aborted) {
        return refuse("RERoT prefix rebuild refused: episode is hard aborted");
    }
    if (ep->is_dag) {
        return refuse("RERoT prefix rebuild refused: DAG already initialized; rebuild is pre-routing only");
    }
    if (ep->c0.valid() && base > ep->c0.n_prompt_tokens) {
        return refuse("RERoT prefix rebuild refused: base extends past the C0 watermark");
    }
    const auto * root_doc = ep->document.node(0);
    if (!root_doc) {
        return refuse("RERoT prefix rebuild refused: root document state is missing");
    }
    // Shrink every root logical run to [0, base): fully rebuilt spans are
    // removed, straddling spans are cut. Physical truncation is the caller's
    // job (memory + prompt tape); this fixes the logical record to match.
    for (const auto rid : root_doc->runs) {
        const auto * run = ep->document.run(rid);
        if (!run || run->token_count == 0) {
            continue;
        }
        const int64_t end = int64_t(run->storage_pos0) + int64_t(run->token_count);
        if (run->storage_pos0 >= base) {
            if (!ep->document.set_run_token_count(rid, 0)) {
                return fail_episode(*ep, "RERoT prefix rebuild could not retract root spans");
            }
        } else if (end > base) {
            if (!ep->document.set_run_token_count(
                    rid, static_cast<uint32_t>(base - run->storage_pos0))) {
                return fail_episode(*ep, "RERoT prefix rebuild could not shrink root spans");
            }
        }
    }
    if (root->pending_record.has_value()) {
        return fail_episode(*ep, "RERoT prefix rebuild found a root pending record (no admission may precede the rebuild)");
    }
    // Clear run refs left over emptied runs. Work counters are cumulative and
    // stay: the replayed suffix is real work and is counted on commit.
    const auto ref_live = [&](const std::optional<llama_rerot_run_id> & ref) {
        if (!ref.has_value()) {
            return true;
        }
        const auto * run = ep->document.run(*ref);
        return run && run->owner == 0 && run->token_count > 0;
    };
    if (!ref_live(root->public_run)) {
        root->public_run.reset();
    }
    if (!ref_live(root->private_run)) {
        root->private_run.reset();
    }
    // Stale lineage pins pre-rebuild cells; both are rebuilt on demand after
    // the suffix replay (archive via ensure_archive_seq, parked via yield).
    if (ep->archive_seq >= 0) {
        if (memory_) {
            llama_memory_seq_rm(memory_, ep->archive_seq, -1, -1);
        }
        clear_sequence_control(ep->archive_seq);
        free_internal_seq(ep->archive_seq);
        ep->archive_seq = -1;
    }
    if (root->parked_seq >= 0) {
        if (memory_) {
            llama_memory_seq_rm(memory_, root->parked_seq, -1, -1);
        }
        free_internal_seq(root->parked_seq);
        root->parked_seq = -1;
    }
    root->storage_pos_next = base;
    return true;
}

bool server_rerot_runtime::yield_dag_pen_for_ready(uint64_t episode_id, bool resource_pressure) {
    auto * ep = episode(episode_id);
    if (!ep || !ep->is_dag || ep->hard_aborted || pens_.empty()) {
        return false;
    }
    if (!ep->c_base.valid()) {
        return false;
    }
    if (ep->ready_queue.empty() && ep->suspended.empty()) {
        return false;
    }
    if (has_free_pen() && !resource_pressure) {
        return false;
    }

    rerot_pen_id victim_pen = -1;
    server_rerot_node_runtime * victim = nullptr;
    const auto consider_victim = [&](llama_rerot_node_id nid, bool require_committed) -> bool {
        if (nid == 0) {
            return false;
        }
        auto * n = node(episode_id, nid);
        const auto * dn = ep->document.node(nid);
        if (!n || !dn || n->pen_id < 0) {
            return false;
        }
        if (n->stage_role != llama_rerot_stage_role::worker &&
            n->stage_role != llama_rerot_stage_role::synthesis) {
            return false;
        }
        const bool starting = ep->starting.count(nid) != 0;
        if (starting) {
            if (!require_committed || ep->dag_step_committed.count(nid) == 0) {
                return false;
            }
        } else if (dn->state != llama_rerot_node_state::running &&
                   dn->state != llama_rerot_node_state::terminal_running) {
            return false;
        }
        if (require_committed && ep->dag_step_committed.count(nid) == 0) {
            return false;
        }
        victim_pen = n->pen_id;
        victim = n;
        return true;
    };
    if (!ep->dag_step_cohort.empty()) {
        for (auto it = ep->running.rbegin(); it != ep->running.rend(); ++it) {
            if (consider_victim(*it, true)) {
                break;
            }
        }
        if (victim_pen < 0) {
            for (auto it = ep->starting.rbegin(); it != ep->starting.rend(); ++it) {
                if (consider_victim(*it, true)) {
                    break;
                }
            }
        }
    }
    if (victim_pen < 0) {
        for (auto it = ep->running.rbegin(); it != ep->running.rend(); ++it) {
            if (ep->starting.count(*it) != 0) {
                continue;
            }
            if (consider_victim(*it, false)) {
                break;
            }
        }
    }
    if (victim_pen < 0 || !victim) {
        return false;
    }
    if (memory_ && victim->exec_seq >= 0) {
        if (victim->parked_seq < 0) {
            const auto parked = alloc_internal_seq();
            if (!parked.has_value()) {
                return false;
            }
            victim->parked_seq = *parked;
        }
        llama_memory_seq_cp_attention(memory_, victim->exec_seq, victim->parked_seq, -1, -1);
        const size_t seed_size =
            llama_memory_rerot_capture_hand_seed(memory_, victim->exec_seq, nullptr, 0);
        if (seed_size > 0) {
            victim->hand_seed.resize(seed_size);
            if (llama_memory_rerot_capture_hand_seed(
                    memory_,
                    victim->exec_seq,
                    victim->hand_seed.data(),
                    victim->hand_seed.size()) != victim->hand_seed.size()) {
                return false;
            }
        }
        llama_memory_seq_rm_attention(memory_, victim->exec_seq, -1, -1);
        llama_memory_seq_rm_recurrent(memory_, victim->exec_seq, -1, -1);
    }
    return suspend_pen(victim_pen);
}

std::vector<llama_rerot_node_id> server_rerot_runtime::get_eligible_dag_nodes(uint64_t episode_id) const {
    const auto * ep = episode(episode_id);
    if (!ep || !ep->is_dag) {
        return {};
    }
    if (!ep->c_base.valid()) {
        return {};
    }

    std::vector<llama_rerot_node_id> eligible;
    for (size_t i = 1; i < ep->nodes.size(); ++i) {
        const auto & nr = ep->nodes[i];
        const auto * doc_node = ep->document.node(static_cast<llama_rerot_node_id>(i));
        if (!doc_node) continue;

        // Must not be already started or sealed. queued + remaining_preds==0 only.
        if (nr.is_sealed ||
            ep->running.count(static_cast<llama_rerot_node_id>(i)) ||
            ep->starting.count(static_cast<llama_rerot_node_id>(i)) ||
            ep->suspended.count(static_cast<llama_rerot_node_id>(i))) {
            continue;
        }
        if (doc_node->state != llama_rerot_node_state::queued) {
            continue;
        }
        if (nr.remaining_preds == 0) {
            eligible.push_back(static_cast<llama_rerot_node_id>(i));
        }
    }
    return eligible;
}

bool server_rerot_runtime::seal_dag_node(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        llama_rerot_event_origin origin) {
    auto * ep = episode(episode_id);
    if (!ep || node_id >= ep->nodes.size()) {
        return false;
    }

    if (origin == llama_rerot_event_origin::runtime_frame ||
        origin == llama_rerot_event_origin::foreign_export ||
        origin == llama_rerot_event_origin::unknown) {
        return false;
    }

    auto & nr = ep->nodes[node_id];
    if (nr.is_sealed) {
        return true; // Idempotent: duplicate callbacks never decrement twice
    }
    // Abort is terminal, never a natural completion: a late source-end
    // callback on a dead episode must not resurrect scheduling state.
    if (ep->hard_aborted) {
        return false;
    }
    // Only a started worker/synthesis stage can seal. Sealing a queued or
    // blocked node would unlock successors without any computation.
    if (!dag_node_in_started_set(*ep, node_id)) {
        return fail_episode(*ep, "RERoT seal references a node that never started");
    }

    nr.is_sealed = true;
    nr.completion_origin = origin;
    ep->document.set_node_state(node_id, llama_rerot_node_state::retired);
    ep->running.erase(node_id);
    ep->starting.erase(node_id);
    ep->suspended.erase(node_id);

    const auto * doc_n = ep->document.node(node_id);
    if (doc_n) {
        for (const auto succ : doc_n->successors) {
            if (succ < ep->nodes.size() && ep->nodes[succ].remaining_preds > 0) {
                --ep->nodes[succ].remaining_preds;
            }
        }
    }

    // Sealed workers passivate immediately: park attention + PENDING refs and
    // free the executor now, so a pending peer is never starved behind a
    // finished lane for the rest of the cohort. Retirement (archive + release
    // + exactly-once notification) still waits for cohort publication in
    // finish_frontier, which retires sealed parked workers too. Synthesis
    // stays bound for serial content.
    if (nr.stage_role == llama_rerot_stage_role::worker) {
        if (memory_ && nr.exec_seq >= 0) {
            if (nr.parked_seq < 0) {
                const auto parked = alloc_internal_seq();
                if (!parked.has_value()) {
                    return fail_episode(*ep, "rerot_resource_exhausted: no parked seq to passivate sealed worker");
                }
                nr.parked_seq = *parked;
            }
            llama_memory_seq_cp_attention(memory_, nr.exec_seq, nr.parked_seq, -1, -1);
            const size_t seed_size =
                llama_memory_rerot_capture_hand_seed(memory_, nr.exec_seq, nullptr, 0);
            if (seed_size > 0) {
                nr.hand_seed.resize(seed_size);
                if (llama_memory_rerot_capture_hand_seed(
                        memory_, nr.exec_seq, nr.hand_seed.data(), nr.hand_seed.size()) !=
                    nr.hand_seed.size()) {
                    return fail_episode(*ep, "failed to capture sealed worker hand seed");
                }
            }
            llama_memory_seq_rm_attention(memory_, nr.exec_seq, -1, -1);
            llama_memory_seq_rm_recurrent(memory_, nr.exec_seq, -1, -1);
        }
        if (nr.pen_id >= 0 || nr.physical_slot >= 0 || nr.exec_seq >= 0) {
            const int slot = nr.physical_slot;
            const rerot_pen_id pen = nr.pen_id;
            nr.pen_id = -1;
            nr.physical_slot = -1;
            nr.exec_seq = -1;
            if (pen >= 0 && (size_t) pen < pens_.size()) {
                pens_[pen].state = server_pen_state::free;
                pens_[pen].person = 0;
                pens_[pen].episode_id = 0;
                pens_[pen].node_id = LLAMA_REROT_NODE_INVALID;
            }
            if (slot >= 0) {
                const auto slot_it = slot_to_episode_.find(slot);
                if (slot_it != slot_to_episode_.end() && slot_it->second == ep->id) {
                    slot_to_episode_.erase(slot_it);
                }
            }
        }
        // The parked seq doubles as the awaiting-retire marker (released with
        // the exactly-once notification in finish_frontier), including
        // null-memory fixtures where there is no physical state to park.
        if (nr.parked_seq < 0) {
            const auto parked = alloc_internal_seq();
            if (!parked.has_value()) {
                return fail_episode(*ep, "rerot_resource_exhausted: no parked seq to passivate sealed worker");
            }
            nr.parked_seq = *parked;
        }
    }

    // Refresh ready_queue with newly eligible nodes. No yield here: seal runs
    // inside the commit path and must only stage its own completed state
    // (passivation above) — never select another unfinished row as a yield
    // victim. Central scheduling runs after the slice commits.
    const auto eligible = get_eligible_dag_nodes(episode_id);
    for (const auto nid : eligible) {
        if (std::find(ep->ready_queue.begin(), ep->ready_queue.end(), nid) == ep->ready_queue.end()) {
            ep->ready_queue.push_back(nid);
        }
    }

    return true;
}

llama_rerot_reader_view server_rerot_runtime::build_dag_view_for_reader(
        uint64_t episode_id,
        llama_rerot_node_id reader) const {
    const auto * ep = episode(episode_id);
    if (!ep) {
        throw std::out_of_range("episode not found");
    }

    std::vector<llama_rerot_node_id> started;
    started.push_back(0);
    for (size_t i = 1; i < ep->nodes.size(); ++i) {
        const auto nid = static_cast<llama_rerot_node_id>(i);
        if (dag_node_in_started_set(*ep, nid)) {
            started.push_back(nid);
        }
    }

    return ep->document.build_dag_view(reader, started, ep->frozen_read_publish_epoch);
}

bool server_rerot_runtime::freeze_fork_parent(
        uint64_t episode_id,
        llama_rerot_node_id parent_id) {
    auto * current = episode(episode_id);
    auto * parent = node(episode_id, parent_id);
    if (!current || !parent || current->hard_aborted) {
        return false;
    }
    if (current->is_dag) {
        return fail_episode(*current,
            "rerot_protocol_error: nested HTML fork is retired; recursive nested DAG is not a production path");
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

    std::vector<uint8_t> parent_seed = parent->hand_seed;
    if (memory_) {
        const size_t seed_size =
            llama_memory_rerot_capture_hand_seed(
                memory_, parent->exec_seq, nullptr, 0);
        if (seed_size == 0) {
            return fail_episode(
                *current,
                "failed to size the parent recurrent hand seed");
        }
        parent_seed.resize(seed_size);
        if (llama_memory_rerot_capture_hand_seed(
                memory_,
                parent->exec_seq,
                parent_seed.data(),
                parent_seed.size()) != parent_seed.size()) {
            return fail_episode(
                *current,
                "failed to capture the parent recurrent hand seed");
        }
        parent->hand_seed = parent_seed;

        if (need_archive && current->base_prefix_end > 0) {
            llama_memory_seq_cp_attention(
                memory_, parent->exec_seq, archive_seq, 0, current->base_prefix_end);
        }
        archive_public_runs(*current, *parent, archive_seq);

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
        child_runtime->hand_seed = parent_seed;
    }

    const int released_slot = parent->physical_slot;
    parent->pen_id = -1;
    parent->physical_slot = -1;
    parent->exec_seq = -1;
    current->running.erase(parent_id);
    if (released_slot >= 0 && (size_t) released_slot < pens_.size()) {
        pens_[released_slot].state = server_pen_state::free;
        pens_[released_slot].person = 0;
        pens_[released_slot].episode_id = 0;
        pens_[released_slot].node_id = LLAMA_REROT_NODE_INVALID;
    }
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
        llama_rerot_node_id * admitted_node,
        llama_rerot_node_id prefer_node) {
    if (admitted_node) {
        *admitted_node = LLAMA_REROT_NODE_INVALID;
    }
    auto * current = episode(episode_id);
    if (!current || current->hard_aborted || physical_slot < 0 || exec_seq < 0 || current->ready_queue.empty()) {
        return false;
    }
    // Workers cannot admit before the formal base is captured: lineage,
    // storage watermarks, and sampler clones do not exist yet. This is a
    // scheduling no-op (the caller retries after capture), never an abort.
    // The source root formal-P forced execution does not go through here.
    if (current->is_dag && !current->c_base.valid()) {
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
    if (prefer_node != LLAMA_REROT_NODE_INVALID) {
        auto prefer_it = std::find(current->ready_queue.begin(), current->ready_queue.end(), prefer_node);
        if (prefer_it != current->ready_queue.end()) {
            best_it = prefer_it;
        }
    }
    if (prefer_node == LLAMA_REROT_NODE_INVALID || best_it == current->ready_queue.begin()) {
        uint64_t best_frontier = node(episode_id, *best_it)->enqueue_frontier;
        std::vector<uint32_t> best_path = current->document.tree_path(*best_it);
        for (auto it = current->ready_queue.begin(); it != current->ready_queue.end(); ++it) {
            if (prefer_node != LLAMA_REROT_NODE_INVALID && *best_it == prefer_node) {
                break;
            }
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
    if (!child || !child_doc || child_doc->state != llama_rerot_node_state::queued) {
        return fail_episode(*current, "RERoT ready queue contains an invalid parked child");
    }
    if (!current->is_dag && child->parked_seq < 0) {
        return fail_episode(*current, "RERoT ready queue contains an invalid parked child");
    }

    if (memory_) {
        if (!current->is_dag && current->archive_seq < 0) {
            return fail_episode(*current, "RERoT child admission has no archive sequence");
        }
        if (current->base_prefix_end > 0 && current->archive_seq >= 0) {
            llama_memory_seq_cp_attention(
                memory_, current->archive_seq, exec_seq, 0, current->base_prefix_end);
        }

        if (!current->is_dag) {
        // Transfer every parked visible run to the physical execution
        // sequence, then release the logical parking reference. The previous
        // single-anchor reconstruction dropped sibling-specific sparse runs
        // and leaked parked refs across admissions.
        llama_memory_seq_cp_attention(
            memory_, child->parked_seq, exec_seq, -1, -1);
        if (!llama_memory_seq_rm_attention(
                memory_, child->parked_seq, -1, -1)) {
            return fail_episode(
                *current,
                "failed to transfer parked attention state during admission");
        }
        }

        const auto * parent_doc = current->document.node(child_doc->parent);
        const llama_rerot_run * anchor = nullptr;
        if (parent_doc) {
            for (const auto run_id : parent_doc->runs) {
                const auto * run = current->document.run(run_id);
                if (!run ||
                    run->visibility != llama_rerot_visibility::public_live ||
                    run->token_count == 0) {
                    continue;
                }
                const int64_t end =
                    int64_t(run->storage_pos0) + int64_t(run->token_count);
                const int64_t anchor_end = anchor
                    ? int64_t(anchor->storage_pos0) +
                        int64_t(anchor->token_count)
                    : -1;
                if (end > anchor_end) {
                    anchor = run;
                }
            }
        }
        if (current->is_dag) {
            if (child->parked_seq >= 0) {
                llama_memory_seq_cp_attention(
                    memory_, child->parked_seq, exec_seq, -1, -1);
                if (!llama_memory_seq_rm_attention(
                        memory_, child->parked_seq, -1, -1)) {
                    return fail_episode(
                        *current, "failed to restore parked DAG attention state");
                }
                if (!child->hand_seed.empty() &&
                    !llama_memory_rerot_apply_hand_seed(
                        memory_, exec_seq, child->hand_seed.data(), child->hand_seed.size())) {
                    return fail_episode(*current, "failed to restore parked DAG hand seed");
                }
            } else {
            const auto & seed = !current->c_base.gdn_recurrent_states.empty()
                ? current->c_base.gdn_recurrent_states
                : current->c0.gdn_recurrent_states;
            if (!seed.empty() &&
                !llama_memory_rerot_apply_hand_seed(
                    memory_, exec_seq, seed.data(), seed.size())) {
                return fail_episode(*current, "failed to restore DAG C_base hand seed");
            }
            }
        } else if (!anchor ||
            int64_t(anchor->storage_pos0) + int64_t(anchor->token_count) !=
                child->storage_pos_next ||
            llama_memory_rerot_add_run_ref(
                memory_, current->id, anchor->id, exec_seq) == 0) {
            return fail_episode(
                *current,
                "RERoT child admission could not anchor the fork frontier");
        } else if (child->hand_seed.empty() ||
            !llama_memory_rerot_apply_hand_seed(
                memory_,
                exec_seq,
                child->hand_seed.data(),
                child->hand_seed.size())) {
            return fail_episode(
                *current,
                "failed to restore child recurrent hand seed");
        }
    }

    // Fresh DAG stages generate after the formal common prefix P, never at
    // position 0: anchor storage to the C_base watermark (the rebuilt P end
    // once a tools-induced prefix rebuild moved it). Admission is refused
    // until c_base is valid, so no fallback exists here; a resumed (parked)
    // stage keeps its own watermark so committed spans stay contiguous.
    if (current->is_dag && child->parked_seq < 0 && current->c_base.valid() &&
        current->c_base.n_prompt_tokens >= 0) {
        child->storage_pos_next = current->c_base.n_prompt_tokens;
    }
    if (child->parked_seq >= 0) {
        free_internal_seq(child->parked_seq);
    }
    child->parked_seq = -1;
    child->pen_id = physical_slot;
    child->physical_slot = physical_slot;
    child->exec_seq = exec_seq;

    current->ready_queue.erase(best_it);
    current->starting.insert(child_id);
    if (!current->document.set_node_state(child_id, llama_rerot_node_state::starting)) {
        return fail_episode(*current, "failed to enter RERoT child STARTING state");
    }
    if (physical_slot >= 0 && (size_t) physical_slot < pens_.size()) {
        pens_[physical_slot].state = server_pen_state::allocated;
        pens_[physical_slot].person = episode_id;
        pens_[physical_slot].episode_id = episode_id;
        pens_[physical_slot].node_id = child_id;
        pens_[physical_slot].exec_seq = exec_seq;
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
    if (child->pen_id >= 0 && (size_t) child->pen_id < pens_.size()) {
        pens_[child->pen_id].state = server_pen_state::running;
    }
    const llama_rerot_node_state next_state = current->is_dag
        ? llama_rerot_node_state::running
        : llama_rerot_node_state::planning;
    if (!current->document.set_node_state(node_id, next_state)) {
        return fail_episode(*current, current->is_dag
            ? "failed to enter RERoT DAG RUNNING state"
            : "failed to enter RERoT child PLANNING state");
    }
    // No scheduling side effects here: this runs inside the per-row commit
    // path, and yielding here could evict a peer whose host commit/sample
    // has not run yet. The central admit/yield loop runs after all row
    // commits plus pending-decision capture in the slice.
    current->topology_barrier_pending = true;
    ++current->topology_epoch;
    ++current->layout_epoch;
    return true;
}

bool server_rerot_runtime::begin_worker(
        uint64_t episode_id,
        llama_rerot_node_id node_id) {
    auto * current = episode(episode_id);
    auto * lane = node(episode_id, node_id);
    const auto * logical = current ? current->document.node(node_id) : nullptr;
    if (!current || !lane || !logical || current->hard_aborted ||
        logical->state != llama_rerot_node_state::planning ||
        lane->planner_armed || lane->pending_record.has_value()) {
        return false;
    }
    lane->parser.reset();
    if (!current->document.set_node_state(node_id, llama_rerot_node_state::terminal_running)) {
        return fail_episode(*current, "failed to enter RERoT direct worker state");
    }
    return true;
}

bool server_rerot_runtime::arm_planner(
        uint64_t episode_id,
        llama_rerot_node_id node_id) {
    auto * current = episode(episode_id);
    auto * lane = node(episode_id, node_id);
    const auto * logical = current ? current->document.node(node_id) : nullptr;
    if (!current || !lane || !logical || current->hard_aborted ||
        logical->state != llama_rerot_node_state::planning ||
        lane->planner_armed || lane->pending_record.has_value()) {
        return false;
    }
    lane->parser.reset();
    lane->planner_armed = true;
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

    const bool dag_step_ready = !current->is_dag || dag_logical_step_complete(episode_id);
    if (current->is_dag && !dag_step_ready) {
        result.topology_barrier = current->topology_barrier_pending;
        result.hard_aborted = current->hard_aborted;
        result.abort_reason = current->abort_reason;
        return result;
    }

    if (current->is_dag && !publish_dag_step_bodies(*current)) {
        result.hard_aborted = true;
        result.abort_reason = current->abort_reason;
        return result;
    }
    if (current->is_dag) {
        current->dag_step_cohort.clear();
        current->dag_step_committed.clear();
    }

    // Shared-RBB is not the DAG default (AGENTS.md §05). The host hook is a
    // no-op atomicity check on the HTML path only.
    if (memory_ && !current->is_dag && !current->running.empty()) {
        std::vector<llama_seq_id> candidate_seqs;
        std::vector<uint8_t> is_public_write;
        candidate_seqs.reserve(current->running.size());
        is_public_write.reserve(current->running.size());
        for (const auto nid : current->running) {
            const auto * n = node(episode_id, nid);
            if (n && n->exec_seq >= 0) {
                candidate_seqs.push_back(n->exec_seq);
                is_public_write.push_back(n->last_write_public ? 1 : 0);
            }
        }
        if (!candidate_seqs.empty()) {
            llama_memory_rerot_commit_rbb_frontier(
                memory_, (uint32_t) current->root_task_id,
                candidate_seqs.data(), is_public_write.data(), candidate_seqs.size());
        }
    }

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
    if (current->is_dag) {
        for (size_t i = 1; i < current->nodes.size(); ++i) {
            if (i == current->synthesis_node) {
                continue;
            }
            auto * sealed = node(episode_id, static_cast<llama_rerot_node_id>(i));
            if (!sealed || !sealed->is_sealed) {
                continue;
            }
            // Retirement waits for cohort publication: pending spans are
            // never retired early. Fully released nodes were already notified.
            if (sealed->pending_record.has_value()) {
                continue;
            }
            const bool live = sealed->physical_slot >= 0 && sealed->exec_seq >= 0;
            const bool parked = sealed->physical_slot < 0 && sealed->exec_seq < 0 &&
                sealed->parked_seq >= 0;
            if (!live && !parked) {
                continue;
            }
            int released_slot = -1;
            if (!retire_node(*current, *sealed, &released_slot)) {
                break;
            }
            result.retired.push_back(static_cast<llama_rerot_node_id>(i));
            if (released_slot >= 0) {
                result.released_slots.push_back(released_slot);
            }
        }
        auto * synth = current->synthesis_node != LLAMA_REROT_NODE_INVALID
            ? node(episode_id, current->synthesis_node)
            : nullptr;
        if (synth && !synth->is_sealed) {
            result.synthesis_node = current->synthesis_node;
            if (synth->remaining_preds == 0 &&
                std::find(current->ready_queue.begin(), current->ready_queue.end(),
                          current->synthesis_node) == current->ready_queue.end() &&
                current->running.count(current->synthesis_node) == 0 &&
                current->starting.count(current->synthesis_node) == 0) {
                current->ready_queue.push_back(current->synthesis_node);
                synth->enqueue_frontier = current->frontier;
            }
            if (synth->exit_intent && current->running.count(current->synthesis_node) != 0) {
                final_candidate = current->synthesis_node;
            }
        }
    } else if (!exits.empty() &&
        exits.size() == current->running.size() &&
        current->ready_queue.empty() &&
        current->starting.empty() &&
        current->suspended.empty()) {
        // Same-frontier simultaneous exits are not judged. Stable tree-path
        // order alone chooses the Lane that remains live for the final acquire
        // fence. It must not be retired before Stage 7 refreshes its logits.
        final_candidate = exits.back();
    }

    if (!current->is_dag) {
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
    }

    if (!current->hard_aborted && final_candidate != LLAMA_REROT_NODE_INVALID) {
        result.final_node = final_candidate;
        current->finalizing = true;
        const auto * survivor = node(episode_id, final_candidate);
        if (!survivor ||
            current->running.find(final_candidate) == current->running.end() ||
            (!current->is_dag && !survivor->exit_intent)) {
            fail_episode(*current, "RERoT final survivor lost its live execution state");
            result.final_node = LLAMA_REROT_NODE_INVALID;
        }
    }

    result.topology_barrier = current->topology_barrier_pending;
    current->topology_barrier_pending = false;

    if (!current->hard_aborted) {
        advance_frontier(episode_id);
        if (current->is_dag) {
            activate_dag_frontier(episode_id);
        }
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
    current_node->pen_id = -1;
    current_node->physical_slot = -1;
    current_node->exec_seq = -1;
    current->running.erase(node_id);
    current->starting.erase(node_id);
    if (physical_slot >= 0) {
        if ((size_t) physical_slot < pens_.size()) {
            pens_[physical_slot].state = server_pen_state::free;
            pens_[physical_slot].person = 0;
            pens_[physical_slot].episode_id = 0;
            pens_[physical_slot].node_id = LLAMA_REROT_NODE_INVALID;
        }
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
    release_probe_seq(current);
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
    for (auto & p : pens_) {
        if (p.episode_id == episode_id || p.person == episode_id) {
            if (p.exec_seq >= 0) {
                clear_sequence_control(p.exec_seq);
                if (memory_) {
                    llama_memory_seq_rm(memory_, p.exec_seq, -1, -1);
                }
            }
            p.state = server_pen_state::free;
            p.person = 0;
            p.episode_id = 0;
            p.node_id = LLAMA_REROT_NODE_INVALID;
        }
    }
    episodes_.erase(it);
    return true;
}

bool server_rerot_runtime::has_ready_nodes(uint64_t episode_id) const {
    const auto * current = episode(episode_id);
    return current && (!current->ready_queue.empty() || !current->suspended.empty());
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
    // Guide §4.3: root child (depth 1 in document tree) corresponds to <h1>, depth 2 to <h2>, etc.
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
    if (current->is_dag) {
        current->frozen_read_publish_epoch = current->publish_epoch;
    }
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
    episode.suspended.clear();
    episode.forked_this_frontier.clear();
    episode.topology_barrier_pending = false;
    episode.dag_step_cohort.clear();
    episode.dag_step_committed.clear();
    release_probe_seq(episode);
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
            if ((size_t) current_node.physical_slot < pens_.size()) {
                pens_[current_node.physical_slot].state = server_pen_state::free;
                pens_[current_node.physical_slot].person = 0;
                pens_[current_node.physical_slot].episode_id = 0;
                pens_[current_node.physical_slot].node_id = LLAMA_REROT_NODE_INVALID;
            }
            const auto slot_it = slot_to_episode_.find(current_node.physical_slot);
            if (slot_it != slot_to_episode_.end() && slot_it->second == episode.id) {
                slot_to_episode_.erase(slot_it);
            }
            current_node.pen_id = -1;
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
    if (current->is_dag) {
        snapshot_dag_logical_step(episode_id);
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

bool server_rerot_runtime::track_fence_token(
        uint64_t episode_id, llama_rerot_node_id node_id,
        const server_rerot_token_plan & plan, llama_token token) {
    auto * current = episode(episode_id);
    auto * lane = node(episode_id, node_id);
    if (!current || !lane || current->hard_aborted || !plan.valid()) {
        return false;
    }
    if (lane->control_id().empty()) {
        return true;
    }
    const bool candidate = lane->exit_parser.state() == server_rerot_marker_state::marker_candidate ||
        plan.marker_step.marker_closed;
    if (!candidate || plan.marker_step.release_previous_pending) {
        lane->fence = {};
    }
    if (!candidate) {
        return true;
    }
    auto & fence = lane->fence;
    if (fence.prepared || (plan.visibility != llama_rerot_visibility::pending_record &&
        plan.visibility != llama_rerot_visibility::private_control)) {
        return fail_episode(*current, "invalid RERoT closing-marker checkpoint row");
    }
    if (fence.rows.empty() && memory_ && llama_memory_seq_get_recurrent_used(memory_, lane->exec_seq) > 0) {
        const size_t size = llama_memory_rerot_capture_hand_seed(memory_, lane->exec_seq, nullptr, 0);
        if (size == 0) {
            return fail_episode(*current, "cannot capture RERoT pre-marker hand");
        }
        fence.hand.resize(size);
        if (llama_memory_rerot_capture_hand_seed(memory_, lane->exec_seq,
                fence.hand.data(), size) != size) {
            return fail_episode(*current, "incomplete RERoT pre-marker hand");
        }
    }
    if (!fence.rows.empty() && plan.storage_pos != fence.rows.back().pos + 1) {
        return fail_episode(*current, "non-contiguous RERoT closing-marker checkpoint");
    }
    fence.rows.push_back({token, plan.storage_pos, plan.run_id});
    return true;
}

bool server_rerot_runtime::prepare_final_fence(uint64_t episode_id, llama_rerot_node_id node_id) {
    auto * current = episode(episode_id);
    auto * lane = node(episode_id, node_id);
    if (!current || !lane || current->hard_aborted || current->serial_tail ||
        !current->finalizing || !current->fence_refreshed || lane->fence.prepared ||
        !lane->exit_intent || lane->fence.rows.empty() ||
        !current->ready_queue.empty() || !current->starting.empty() || !current->suspended.empty() ||
        current->running.size() != 1 || current->running.count(node_id) != 1) {
        return false;
    }
    auto & fence = lane->fence;
    // Every original row must now be PRIVATE, including partial marker rows
    // that were PENDING during sampling. No public write may be replayed.
    for (size_t i = 0; i < fence.rows.size(); ++i) {
        const auto & row = fence.rows[i];
        const auto * run = current->document.run(row.run);
        if (!run || run->owner != node_id || run->visibility != llama_rerot_visibility::private_control ||
            row.pos < run->storage_pos0 || int64_t(row.pos) >= int64_t(run->storage_pos0) + run->token_count ||
            (i && row.pos != fence.rows[i - 1].pos + 1)) {
            return fail_episode(*current, "RERoT fence would replay a non-private or missing row");
        }
    }
    if (fence.rows.back().pos + 1 != lane->storage_pos_next) {
        return fail_episode(*current, "RERoT fence is not the survivor's exact suffix");
    }
    if (memory_) {
        const bool recurrent = llama_memory_seq_get_recurrent_used(memory_, lane->exec_seq) > 0;
        if (recurrent && (fence.hand.empty() || !llama_memory_rerot_apply_hand_seed(
                memory_, lane->exec_seq, fence.hand.data(), fence.hand.size()))) {
            return fail_episode(*current, "failed to restore RERoT pre-marker hand");
        }
        // Only this Lane's private suffix is removed. PUBLIC archive refs,
        // other people and the current shared brain are not rolled back.
        if (!llama_memory_seq_rm_attention(memory_, lane->exec_seq,
                fence.rows.front().pos, lane->storage_pos_next)) {
            return fail_episode(*current, "failed to replace RERoT fence KV suffix");
        }
    }
    fence.cursor = 0;
    fence.prepared = true;
    return true;
}

std::optional<server_rerot_token_plan> server_rerot_runtime::plan_final_fence_token(
        uint64_t episode_id, llama_rerot_node_id node_id) const {
    const auto * current = episode(episode_id);
    const auto * lane = node(episode_id, node_id);
    if (!current || !lane || current->hard_aborted || current->serial_tail ||
        !current->finalizing || !current->fence_refreshed || !lane->fence.prepared ||
        lane->fence.cursor >= lane->fence.rows.size()) {
        return std::nullopt;
    }
    const auto & row = lane->fence.rows[lane->fence.cursor];
    server_rerot_token_plan plan;
    plan.storage_pos = row.pos;
    plan.run_id = row.run;
    plan.visibility = llama_rerot_visibility::private_control;
    return plan;
}

bool server_rerot_runtime::commit_final_fence_token(
        uint64_t episode_id, llama_rerot_node_id node_id, const server_rerot_token_plan & plan) {
    const auto expected = plan_final_fence_token(episode_id, node_id);
    if (!expected || plan.storage_pos != expected->storage_pos || plan.run_id != expected->run_id ||
        plan.visibility != llama_rerot_visibility::private_control) {
        return false;
    }
    // Numerical re-evaluation only: no parser event, logical token, run
    // extension, publish/frontier epoch or client-visible stream is added.
    auto & fence = node(episode_id, node_id)->fence;
    ++fence.cursor;
    if (fence.complete()) {
        std::vector<uint8_t>().swap(fence.hand);
    }
    return true;
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
    // memory: nothing queued, starting, or suspended, exactly this Lane still RUNNING.
    if (!current->ready_queue.empty() || !current->starting.empty() || !current->suspended.empty() ||
        current->running.size() != 1 || current->running.count(node_id) != 1 ||
        (!current->is_dag && !survivor->exit_intent) || survivor->exec_seq < 0) {
        return fail_episode(*current, "RERoT final fence lost its single-survivor precondition");
    }
    // Install the stable view through the final public frontier. This method
    // does NOT decode/re-evaluate the closing sequence; neither does the core
    // refresh barrier (synchronize only). Full §21.4 close replay still needs
    // a causal checkpoint and must not double-apply recurrent transitions.
    const auto view = current->is_dag
        ? build_dag_view_for_reader(current->id, node_id)
        : current->document.build_view(node_id);
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
        (!current->is_dag && !survivor->fence.complete()) ||
        current->serial_tail || current->running.size() != 1 ||
        current->running.count(node_id) != 1) {
        return current ? fail_episode(*current, "invalid RERoT serial tail transition") : false;
    }
    current->serial_tail = true;
    current->serial_node = node_id;
    return true;
}

bool server_rerot_runtime::continue_unforked_root(
        uint64_t episode_id,
        llama_rerot_node_id node_id) {
    auto * current = episode(episode_id);
    auto * root = node(episode_id, node_id);
    const auto * document_node = current ? current->document.node(node_id) : nullptr;
    if (!current || !root || !document_node || node_id != current->document.root() ||
        !root->control_id().empty() || current->hard_aborted || current->finalizing ||
        current->serial_tail || !current->ready_queue.empty() ||
        !current->starting.empty() || !current->suspended.empty() ||
        current->running.size() != 1 || current->running.count(node_id) != 1 ||
        document_node->state != llama_rerot_node_state::terminal_running ||
        root->exit_intent || root->exec_seq < 0) {
        return current ? fail_episode(
            *current, "invalid RERoT N=1 root serial transition") : false;
    }
    current->serial_tail = true;
    current->serial_node = node_id;
    return true;
}

bool server_rerot_runtime::validate_serial_tail_state(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        std::string * error_out) const {
    const auto * current = episode(episode_id);
    if (!current) {
        if (error_out) *error_out = "episode not found";
        return false;
    }
    if (current->hard_aborted) {
        if (error_out) *error_out = "episode is hard aborted";
        return false;
    }
    if (!current->serial_tail || current->serial_node != node_id) {
        if (error_out) *error_out = "episode is not in serial tail state for this node";
        return false;
    }
    if (!current->ready_queue.empty() || !current->starting.empty() || !current->suspended.empty()) {
        if (error_out) *error_out = "serial tail has unadmitted or suspended nodes";
        return false;
    }
    if (current->running.size() != 1 || current->running.count(node_id) != 1) {
        if (error_out) *error_out = "serial tail must have exactly one survivor running";
        return false;
    }
    const auto * survivor = node(episode_id, node_id);
    if (!survivor || survivor->exec_seq < 0) {
        if (error_out) *error_out = "survivor node or exec seq invalid";
        return false;
    }
    return true;
}

bool server_rerot_runtime::freeze_serial_coordinates(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        std::string * error_out) {
    if (!validate_serial_tail_state(episode_id, node_id, error_out)) {
        return false;
    }
    auto * current = episode(episode_id);
    auto * survivor = node(episode_id, node_id);
    if (!current || !survivor) {
        return false;
    }
    // Increment layout epoch to freeze survivor virtual coordinates (§22)
    ++current->layout_epoch;
    if (memory_) {
        const auto * doc_node = current->document.node(node_id);
        if (doc_node && !doc_node->runs.empty()) {
            std::vector<uint32_t> ordered_runs;
            llama_rerot_reader_view_desc desc = {};
            if (build_reader_view_desc(*current, *survivor, doc_node->runs.back(), ordered_runs, desc)) {
                llama_memory_rerot_set_reader_view(memory_, survivor->exec_seq, &desc);
            }
        }
    }
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
constexpr uint8_t k_rerot_flag_dag       = 1u << 5;

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
    if (v > static_cast<uint8_t>(llama_rerot_node_state::ready_suspended)) {
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
    w.u8(s.native_end ? 1 : 0);
}

bool rerot_read_marker_snapshot(rerot_blob_reader & r, server_rerot_marker_snapshot & s) {
    std::string marker = r.str();
    std::string candidate = r.str();
    const uint8_t state = r.u8();
    std::string error = r.str();
    const uint8_t native_end = r.u8();
    if (!r.ok || state > static_cast<uint8_t>(server_rerot_marker_state::failed) || native_end > 1) {
        r.ok = false;
        return false;
    }
    s.marker = std::move(marker);
    s.candidate = std::move(candidate);
    s.state = static_cast<server_rerot_marker_state>(state);
    s.error = std::move(error);
    s.native_end = native_end != 0;
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
    s.native_end = native_end_;
    return s;
}

bool server_rerot_marker_parser::restore(const server_rerot_marker_snapshot & snap, std::string * error) {
    if (!snap.marker.empty()) {
        if (snap.native_end) {
            if (!valid_native_end_marker(snap.marker)) {
                return rerot_state_set_error(
                    error, "RERoT state uses an invalid native source-end marker");
            }
        } else if (!valid_child_close_marker(snap.marker)) {
            return rerot_state_set_error(
                error, "RERoT state uses an obsolete or malformed child delimiter");
        }
    }
    if (snap.marker.empty() &&
        (snap.state != server_rerot_marker_state::public_text ||
         !snap.candidate.empty() || !snap.error.empty())) {
        return rerot_state_set_error(error, "RERoT unarmed marker has active parser state");
    }
    if (snap.state > server_rerot_marker_state::failed) {
        return rerot_state_set_error(error, "RERoT marker snapshot has an out-of-range state");
    }
    if (!snap.marker.empty() && snap.candidate.size() >= snap.marker.size()) {
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
    native_end_ = snap.native_end;
    return true;
}

void server_rerot_episode_demote_to_logical(server_rerot_episode & episode) {
    // Episode-level demotion (A.8.2): transient physical slot/pen bindings go,
    // logical membership and lineage stay for re-admission.
    for (auto & node : episode.nodes) {
        node.physical_slot = -1;
        node.pen_id = -1;
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
    if (episode.probing || episode.probe_seq >= 0) {
        rerot_state_set_error(error_out, "RERoT episode save refused: isolated probe is in flight");
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
    if (episode.is_dag)                   { flags |= k_rerot_flag_dag; }
    w.u8(flags);
    w.u32(episode.serial_node);
    w.u32(episode.synthesis_node);
    w.str(episode.source_end_marker);
    w.str(episode.think_start_marker);
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
    w.u64(episode.frozen_read_publish_epoch);
    w.i32(episode.probe_seq);
    w.u8(episode.probing ? 1 : 0);
    w.u8(episode.strategy_decided ? 1 : 0);
    w.u64(episode.probe_tokens);
    w.u64(episode.frame_tokens);
    w.u64(episode.source_end_tokens);
    const auto write_checkpoint = [&](const server_rerot_prebranch_checkpoint & cp) {
        w.i32(cp.task_id);
        w.i32(cp.seq_id);
        w.i32(cp.n_prompt_tokens);
        w.u8(cp.captured ? 1 : 0);
        w.blob(cp.gdn_recurrent_states);
        w.blob(cp.conv1d_states);
        w.blob(cp.sampler_snapshot_bytes);
    };
    write_checkpoint(episode.c0);
    write_checkpoint(episode.c_base);
    w.u32(static_cast<uint32_t>(episode.dag_step_cohort.size()));
    for (const auto id : episode.dag_step_cohort) {
        w.u32(id);
    }
    w.u32(static_cast<uint32_t>(episode.dag_step_committed.size()));
    for (const auto id : episode.dag_step_committed) {
        w.u32(id);
    }

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
    write_id_vec(episode.suspended);

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
        w.u32(node->plan_rank);
        w.u8(static_cast<uint8_t>(node->stage_role));
        write_id_vec(node->predecessors);
        write_id_vec(node->successors);
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
        w.u8(static_cast<uint8_t>(run->kind));
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
        w.blob(node.hand_seed);
        w.blob(node.fence.hand);
        w.u8(node.fence.prepared ? 1 : 0);
        w.u32(static_cast<uint32_t>(node.fence.cursor));
        w.u32(static_cast<uint32_t>(node.fence.rows.size()));
        for (const auto & row : node.fence.rows) {
            w.i32(row.token);
            w.i32(row.pos);
            w.u32(row.run);
        }
        w.u64(node.view_stamp.topology_epoch);
        w.u64(node.view_stamp.publish_epoch);
        w.u64(node.view_stamp.layout_epoch);
        w.u32(node.remaining_preds);
        w.u8(static_cast<uint8_t>(node.stage_role));
        w.u8(node.is_sealed ? 1 : 0);
        w.str(node.string_id);
        w.str(node.intent);
        w.u8(static_cast<uint8_t>(node.completion_origin));
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
    uint32_t plan_rank = 0;
    llama_rerot_stage_role stage_role = llama_rerot_stage_role::planner;
    std::vector<llama_rerot_node_id> predecessors;
    std::vector<llama_rerot_node_id> successors;
};

struct rerot_run_blob {
    llama_rerot_run_id id = LLAMA_REROT_RUN_INVALID;
    llama_rerot_node_id owner = LLAMA_REROT_NODE_INVALID;
    llama_rerot_visibility visibility = llama_rerot_visibility::public_live;
    llama_rerot_segment_kind kind = llama_rerot_segment_kind::body;
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
    std::vector<uint8_t> hand_seed;
    server_rerot_fence_checkpoint fence;
    llama_rerot_view_stamp view_stamp = {0, 0, 0};
    uint32_t remaining_preds = 0;
    llama_rerot_stage_role stage_role = llama_rerot_stage_role::planner;
    bool is_sealed = false;
    std::string string_id;
    std::string intent;
    llama_rerot_event_origin completion_origin = llama_rerot_event_origin::unknown;
};

} // namespace

bool server_rerot_validate_sampler_snapshot(
        const std::vector<uint8_t> & bytes, std::string * error_out) {
    if (bytes.empty()) {
        return true;
    }
    constexpr size_t k_head = sizeof(uint32_t) * 3;
    if (bytes.size() < k_head) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: truncated sampler snapshot");
    }
    uint32_t magic = 0;
    uint32_t n_prev = 0;
    std::memcpy(&magic, bytes.data(), sizeof(uint32_t));
    std::memcpy(&n_prev, bytes.data() + sizeof(uint32_t) * 2, sizeof(uint32_t));
    if (magic != k_sampler_snap_magic) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: sampler snapshot magic mismatch");
    }
    if (bytes.size() != k_head + size_t(n_prev) * sizeof(llama_token)) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: sampler snapshot length mismatch");
    }
    return true;
}

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
    const llama_rerot_node_id synthesis_node = r.u32();
    std::string source_end_marker = r.str();
    std::string think_start_marker = r.str();
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
    const uint64_t frozen_read_publish_epoch = r.u64();
    const llama_seq_id probe_seq = r.i32();
    const bool probing = r.u8() != 0;
    const bool strategy_decided = r.u8() != 0;
    const uint64_t probe_tokens = r.u64();
    const uint64_t frame_tokens = r.u64();
    const uint64_t source_end_tokens = r.u64();
    const auto read_checkpoint = [&](server_rerot_prebranch_checkpoint & cp) {
        cp.task_id = r.i32();
        cp.seq_id = r.i32();
        cp.n_prompt_tokens = r.i32();
        cp.captured = r.u8() != 0;
        cp.gdn_recurrent_states = r.blob();
        cp.conv1d_states = r.blob();
        cp.sampler_snapshot_bytes = r.blob();
    };
    server_rerot_prebranch_checkpoint c0;
    server_rerot_prebranch_checkpoint c_base;
    read_checkpoint(c0);
    read_checkpoint(c_base);
    if (!r.ok) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: truncated episode scalars");
    }
    // The snapshot carries seed + history only: it cannot restore a sampler
    // standalone, so it is validated as an opaque encoding, never decoded
    // into a reseed. Full sampler continuity lives on the in-process
    // demotion path (transport-owned sampler clones).
    if (!server_rerot_validate_sampler_snapshot(c0.sampler_snapshot_bytes, error_out) ||
        !server_rerot_validate_sampler_snapshot(c_base.sampler_snapshot_bytes, error_out)) {
        return false;
    }
    if (episode_id == 0 || frontier == 0 || publish_epoch == 0 || topology_epoch == 0 || layout_epoch == 0) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: zero episode id or epoch");
    }
    if (base_prefix_end < 0 || (flags & 0xC0u) != 0) {
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
    std::vector<llama_rerot_node_id> dag_step_cohort_vec;
    std::vector<llama_rerot_node_id> dag_step_committed_vec;
    read_id_vec(dag_step_cohort_vec);
    read_id_vec(dag_step_committed_vec);
    std::vector<llama_rerot_node_id> forked, running_vec, starting_vec, suspended_vec;
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
    read_id_vec(suspended_vec);
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
        nb.plan_rank = r.u32();
        const uint8_t role = r.u8();
        if (!r.ok || role > static_cast<uint8_t>(llama_rerot_stage_role::synthesis)) {
            r.ok = false;
            break;
        }
        nb.stage_role = static_cast<llama_rerot_stage_role>(role);
        read_id_vec(nb.predecessors);
        read_id_vec(nb.successors);
        if (!r.ok) {
            break;
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
        const uint8_t kind = r.u8();
        if (!r.ok || !rerot_visibility_from_u8(vis, rb.visibility) ||
            kind > static_cast<uint8_t>(llama_rerot_segment_kind::probe_control)) {
            r.ok = false;
            break;
        }
        rb.kind = static_cast<llama_rerot_segment_kind>(kind);
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
        rb.hand_seed = r.blob();
        rb.fence.hand = r.blob();
        const uint8_t fence_prepared = r.u8();
        rb.fence.prepared = fence_prepared != 0;
        rb.fence.cursor = r.u32();
        const uint32_t n_fence_rows = r.u32();
        if (!r.ok || fence_prepared > 1 || rb.fence.cursor > n_fence_rows ||
            n_fence_rows > (r.n - r.off) / 12 ||
            (!rb.fence.prepared && rb.fence.cursor != 0) ||
            (rb.fence.prepared && (!rb.exit_intent || n_fence_rows == 0))) {
            r.ok = false;
            break;
        }
        for (uint32_t j = 0; j < n_fence_rows; ++j) {
            server_rerot_fence_row row;
            row.token = r.i32();
            row.pos = r.i32();
            row.run = r.u32();
            if (!r.ok || row.token < 0 || row.pos < 0 || row.run >= run_blobs.size() ||
                run_blobs[row.run].owner != rb.id ||
                row.pos < run_blobs[row.run].storage_pos0 ||
                int64_t(row.pos) >= int64_t(run_blobs[row.run].storage_pos0) + run_blobs[row.run].token_count ||
                (j && int64_t(row.pos) != int64_t(rb.fence.rows.back().pos) + 1) ||
                (rb.fence.prepared && run_blobs[row.run].visibility != llama_rerot_visibility::private_control)) {
                r.ok = false;
                break;
            }
            rb.fence.rows.push_back(row);
        }
        rb.view_stamp.topology_epoch = r.u64();
        rb.view_stamp.publish_epoch = r.u64();
        rb.view_stamp.layout_epoch = r.u64();
        rb.remaining_preds = r.u32();
        const uint8_t stage_role = r.u8();
        const uint8_t sealed = r.u8();
        rb.string_id = r.str();
        rb.intent = r.str();
        const uint8_t origin = r.u8();
        if (!r.ok || stage_role > static_cast<uint8_t>(llama_rerot_stage_role::synthesis) ||
            sealed > 1 || origin > static_cast<uint8_t>(llama_rerot_event_origin::forced_abort)) {
            r.ok = false;
            break;
        }
        rb.stage_role = static_cast<llama_rerot_stage_role>(stage_role);
        rb.is_sealed = sealed != 0;
        rb.completion_origin = static_cast<llama_rerot_event_origin>(origin);
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
    for (const auto id : suspended_vec) {
        if (id >= n_nodes) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: suspended set references an unknown node");
        }
    }
    for (const auto id : dag_step_cohort_vec) {
        if (id >= n_nodes) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: DAG step cohort references an unknown node");
        }
    }
    for (const auto id : dag_step_committed_vec) {
        if (id >= n_nodes) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: DAG step commit record references an unknown node");
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
                rb.owner, rb.visibility, rb.storage_pos0, rb.token_count, rb.publish_epoch, rb.kind);
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
    const bool is_dag = (flags & k_rerot_flag_dag) != 0;
    if (is_dag) {
        rebuilt.document.set_dag_mode(true);
        for (uint32_t i = 0; i < n_nodes; ++i) {
            const auto & nb = node_blobs[i];
            rebuilt.document.set_plan_rank(i, nb.plan_rank);
            rebuilt.document.set_stage_role(i, nb.stage_role);
            for (const auto pred : nb.predecessors) {
                std::string eerr;
                if (!rebuilt.document.add_edge(pred, i, &eerr)) {
                    return rerot_state_set_error(error_out, "RERoT episode load refused: DAG edge restore: " + eerr);
                }
            }
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
        node.pen_id = sb.physical_slot;
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
        node.hand_seed = sb.hand_seed;
        node.fence = sb.fence;
        node.view_stamp = sb.view_stamp;
        node.remaining_preds = sb.remaining_preds;
        node.stage_role = sb.stage_role;
        node.is_sealed = sb.is_sealed;
        node.string_id = sb.string_id;
        node.intent = sb.intent;
        node.completion_origin = sb.completion_origin;
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
    rebuilt.suspended = std::set<llama_rerot_node_id>(suspended_vec.begin(), suspended_vec.end());
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
    rebuilt.is_dag = is_dag;
    rebuilt.strategy_decided = strategy_decided;
    rebuilt.probing = probing;
    rebuilt.probe_seq = probe_seq;
    rebuilt.frozen_read_publish_epoch = frozen_read_publish_epoch;
    rebuilt.probe_tokens = probe_tokens;
    rebuilt.frame_tokens = frame_tokens;
    rebuilt.source_end_tokens = source_end_tokens;
    rebuilt.c0 = std::move(c0);
    rebuilt.c_base = std::move(c_base);
    rebuilt.dag_step_cohort = std::set<llama_rerot_node_id>(
        dag_step_cohort_vec.begin(), dag_step_cohort_vec.end());
    rebuilt.dag_step_committed = std::set<llama_rerot_node_id>(
        dag_step_committed_vec.begin(), dag_step_committed_vec.end());
    rebuilt.synthesis_node = synthesis_node;
    rebuilt.source_end_marker = std::move(source_end_marker);
    rebuilt.think_start_marker = think_start_marker.empty() ? std::string("<think>") : std::move(think_start_marker);
    if (rebuilt.probing || rebuilt.probe_seq >= 0) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: probe-in-flight state is not restorable");
    }
    if (rebuilt.synthesis_node != LLAMA_REROT_NODE_INVALID && rebuilt.synthesis_node >= n_nodes) {
        return rerot_state_set_error(error_out, "RERoT episode load refused: synthesis node out of range");
    }
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
    for (size_t i = 0; i < episode.document.run_count(); ++i) {
        const auto * run = episode.document.run(static_cast<llama_rerot_run_id>(i));
        if (run == nullptr) {
            continue;
        }
        if (run->kind == llama_rerot_segment_kind::frame) {
            pinned.insert(run->id);
        }
        if (episode.is_dag &&
            run->visibility == llama_rerot_visibility::public_live &&
            dag_node_started(episode.document.node(run->owner))) {
            pinned.insert(run->id);
        }
    }
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
        if (node.physical_slot >= 0 &&
            ((size_t) node.physical_slot >= pens_.size() ||
             pens_[node.physical_slot].state != server_pen_state::free)) {
            return rerot_state_set_error(error_out, "RERoT episode load refused: physical slot " +
                std::to_string(node.physical_slot) + " is not free in this runtime; demote before swap");
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
            // Rebind the pen arena symmetrically: the slot map alone would
            // leave a phantom binding that the allocator hands out while
            // admission still sees it as bound.
            auto & p = pens_[node.physical_slot];
            p.state = staged.running.count(node.id) != 0
                ? server_pen_state::running
                : server_pen_state::allocated;
            p.person = staged.id;
            p.episode_id = staged.id;
            p.node_id = node.id;
            p.exec_seq = node.exec_seq;
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
            if ((size_t) node.physical_slot < pens_.size()) {
                pens_[node.physical_slot].state = server_pen_state::free;
                pens_[node.physical_slot].person = 0;
                pens_[node.physical_slot].episode_id = 0;
                pens_[node.physical_slot].node_id = LLAMA_REROT_NODE_INVALID;
            }
            const auto it = slot_to_episode_.find(node.physical_slot);
            if (it != slot_to_episode_.end() && it->second == episode_id) {
                slot_to_episode_.erase(it);
            }
        }
    }
    server_rerot_episode_demote_to_logical(*current);
    return true;
}

bool server_rerot_runtime::context_shift(
        uint64_t episode_id,
        uint64_t max_tokens_to_remove,
        server_rerot_shift_result * result_out,
        std::string * error_out) {
    auto * current = episode(episode_id);
    if (current == nullptr) {
        return rerot_state_set_error(error_out, "RERoT context shift failed: unknown episode");
    }
    if (current->hard_aborted) {
        return rerot_state_set_error(error_out, "RERoT context shift failed: episode is hard aborted");
    }

    server_rerot_shift_result local_res;
    if (!server_rerot_truncate_oldest_public(*current, max_tokens_to_remove, &local_res, error_out)) {
        return false;
    }

    if (local_res.tokens_removed > 0) {
        current->topology_barrier_pending = true;
    }

    if (result_out) {
        *result_out = local_res;
    }
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

