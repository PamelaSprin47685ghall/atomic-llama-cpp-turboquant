#include "server-rerot.h"

#include <algorithm>
#include <cctype>
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

bool only_ascii_space(std::string_view text) {
    return std::all_of(text.begin(), text.end(), [](char ch) {
        return ascii_space(static_cast<unsigned char>(ch));
    });
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

