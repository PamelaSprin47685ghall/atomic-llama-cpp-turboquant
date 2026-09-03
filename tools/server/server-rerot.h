#pragma once

#include "llama-rerot.h"

#include <cstdint>
#include <deque>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Server-side control-plane helpers for Recursive Elastic Ring-of-Thought.
//
// The model-facing planner protocol is byte-oriented, while KV visibility is
// token/cell-oriented. The parser therefore deliberately classifies an entire
// sampled token conservatively: if any suffix could become the structural
// `<ol>` opener, that token is PENDING until the prefix is resolved. This may
// temporarily hide harmless bytes in the same token from foreign readers, but
// it never exposes an incomplete structural record.

enum class server_rerot_parser_state : uint8_t {
    public_text = 0,
    opening_candidate,
    list_pending,
    complete,
    failed,
};

struct server_rerot_parser_step {
    // Visibility to use when writing the sampled token that supplied this byte
    // chunk. PRIVATE planner injection is handled by the runtime before model
    // sampling and therefore never originates from this parser.
    llama_rerot_visibility write_visibility = llama_rerot_visibility::public_live;

    // A PENDING opener candidate from one or more previous tokens was proven
    // not to be `<ol>`. The runtime must atomically reclassify the previous
    // pending run to PUBLIC before installing the write tag for this token.
    bool release_previous_pending = false;

    // Structural events are committed only after the current token has been
    // decoded, so its KV cell participates in the same atomic publication.
    bool record_opened = false;
    bool record_closed = false;
    bool malformed = false;

    // Populated only when record_closed is true. Titles are trimmed at their
    // outer ASCII whitespace but otherwise preserved byte-for-byte.
    std::vector<std::string> items;
    std::string error;
};

class server_rerot_planner_parser {
public:
    server_rerot_planner_parser();

    void reset();

    // Consume exactly the bytes represented by one sampled token. Token
    // boundaries may occur anywhere inside `<ol>`, `<li>`, `</li>`, or `</ol>`.
    server_rerot_parser_step consume(std::string_view bytes);

    server_rerot_parser_state state() const;
    bool complete() const;
    bool failed() const;

    const std::vector<std::string> & items() const;
    const std::string & error() const;

private:
    server_rerot_parser_step consume_before_list(std::string_view bytes);
    server_rerot_parser_step consume_list_bytes(std::string_view bytes, bool opened_now);
    bool finish_record(server_rerot_parser_step & step);
    bool fail(server_rerot_parser_step & step, std::string message);

    server_rerot_parser_state state_ = server_rerot_parser_state::public_text;

    // At most "<ol". These bytes live in one pending run in KV until a later
    // token either completes `<ol>` or disproves the candidate.
    std::string opener_candidate_;

    // Begins exactly at the structural `<ol>` opener. Bytes preceding the
    // opener in the same tokenizer token are conservatively PENDING in KV but
    // are intentionally not part of the semantic list record parsed here.
    std::string list_bytes_;

    std::vector<std::string> items_;
    std::string error_;
};

enum class server_rerot_marker_state : uint8_t {
    public_text = 0,
    marker_candidate,
    complete,
    failed,
};

struct server_rerot_marker_step {
    llama_rerot_visibility write_visibility = llama_rerot_visibility::public_live;
    bool release_previous_pending = false;
    bool marker_closed = false;
    bool malformed = false;
    std::string error;
};

// Byte-stream detector for a private runtime marker such as </think>. A
// tokenizer token containing all or part of the marker is held PENDING until
// the marker is either completed or disproved.
class server_rerot_marker_parser {
public:
    explicit server_rerot_marker_parser(std::string marker = "</think>");

    void reset();
    server_rerot_marker_step consume(std::string_view bytes);

    server_rerot_marker_state state() const;
    bool complete() const;
    bool failed() const;
    const std::string & error() const;

private:
    bool fail(server_rerot_marker_step & step, std::string message);

    std::string marker_;
    std::string candidate_;
    server_rerot_marker_state state_ = server_rerot_marker_state::public_text;
    std::string error_;
};

// Fixed private control prompt injected after the user's ordinary prompt and
// before planner-visible generation. It is tokenized by the server so the
// runtime stays independent of tokenizer/chat-template machinery.
std::string_view server_rerot_planner_prompt();

struct server_rerot_token_plan {
    llama_rerot_run_id run_id = LLAMA_REROT_RUN_INVALID;
    llama_pos storage_pos = -1;
    llama_rerot_visibility visibility = llama_rerot_visibility::normal;
    server_rerot_parser_step parser_step;
    server_rerot_marker_step marker_step;

    bool valid() const {
        return run_id != LLAMA_REROT_RUN_INVALID && storage_pos >= 0 &&
               visibility != llama_rerot_visibility::normal;
    }
};

struct server_rerot_node_runtime {
    llama_rerot_node_id id = LLAMA_REROT_NODE_INVALID;
    int physical_slot = -1;
    llama_seq_id exec_seq = -1;
    llama_seq_id parked_seq = -1;
    llama_pos storage_pos_next = 0;

    std::optional<llama_rerot_run_id> public_run;
    std::optional<llama_rerot_run_id> private_run;
    std::optional<llama_rerot_run_id> pending_record;

    server_rerot_planner_parser parser;
    server_rerot_marker_parser exit_parser;
    bool planner_armed = true;
    bool exit_intent = false;
};

struct server_rerot_frontier_result {
    uint64_t episode_id = 0;
    uint64_t completed_frontier = 0;
    std::vector<llama_rerot_node_id> forked;
    std::vector<llama_rerot_node_id> retired;
    std::vector<int> released_slots;
    llama_rerot_node_id final_node = LLAMA_REROT_NODE_INVALID;
    bool topology_barrier = false;
    bool hard_aborted = false;
    std::string abort_reason;

    bool natural_final() const {
        return final_node != LLAMA_REROT_NODE_INVALID && !hard_aborted;
    }
};

struct server_rerot_episode {
    uint64_t id = 0;
    int root_task_id = -1;
    int response_task_id = -1;

    uint64_t frontier = 1;
    uint64_t publish_epoch = 1;
    uint64_t topology_epoch = 1;
    uint64_t layout_epoch = 1;
    llama_pos base_prefix_end = 0;

    llama_rerot_document document;
    std::vector<server_rerot_node_runtime> nodes;

    std::deque<llama_rerot_node_id> ready_queue;
    std::set<llama_rerot_node_id> running;
    std::set<llama_rerot_node_id> starting;

    llama_seq_id archive_seq = -1;

    bool topology_barrier_pending = false;
    bool finalizing = false;
    bool hard_aborted = false;
    std::string abort_reason;

    std::vector<llama_rerot_node_id> forked_this_frontier;

    explicit server_rerot_episode(uint64_t episode_id = 1)
        : id(episode_id), document(episode_id) {
    }
};

// Logical/server control plane for RERoT. It owns no physical KV indices.
// Physical mutations go through llama_memory_rerot_* and are keyed solely by
// stable episode/run ids. Passing a null memory is supported by unit tests and
// exercises the same logical state transitions without resident-cell checks.
class server_rerot_runtime {
public:
    explicit server_rerot_runtime(
        llama_memory_t memory,
        llama_rerot_frontier_mode frontier_mode = LLAMA_REROT_FRONTIER_STRONG,
        uint32_t first_internal_seq = 0,
        uint32_t max_seq = 256);

    uint64_t adopt_root(
        int root_task_id,
        int response_task_id,
        int physical_slot,
        llama_seq_id exec_seq,
        llama_pos storage_pos_next);

    void release_slot(int physical_slot);

    server_rerot_episode * episode(uint64_t episode_id);
    const server_rerot_episode * episode(uint64_t episode_id) const;
    server_rerot_episode * episode_for_slot(int physical_slot);
    const server_rerot_episode * episode_for_slot(int physical_slot) const;

    server_rerot_node_runtime * node(uint64_t episode_id, llama_rerot_node_id node_id);
    const server_rerot_node_runtime * node(uint64_t episode_id, llama_rerot_node_id node_id) const;

    // Planner injection bypasses the XML parser and is always PRIVATE.
    std::optional<server_rerot_token_plan> plan_private_token(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        llama_pos storage_pos);

    // Forced headings are kept PENDING until every heading token has decoded.
    std::optional<server_rerot_token_plan> plan_heading_token(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        llama_pos storage_pos);

    // Classify one sampled model token. Any required release of an earlier
    // false opener candidate is committed before this plan is returned.
    std::optional<server_rerot_token_plan> plan_generated_token(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        llama_pos storage_pos,
        std::string_view token_bytes);

    // Install sequence-scoped write/view control immediately before decode.
    bool install_token_plan(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        const server_rerot_token_plan & plan);

    // Commit one successfully decoded token. A completed planner list is
    // atomically published here, then N=1 or N>1 topology mutation occurs.
    bool commit_token(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        const server_rerot_token_plan & plan);

    bool publish_heading(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        llama_rerot_run_id run_id);

    // Fork-time parking/freezing and queued-child admission. These operations
    // use component-selective sequence APIs: public attention gets an archive
    // keeper; recurrent state gets parked COW ids; physical server slots are
    // freed without moving K/V.
    bool freeze_fork_parent(uint64_t episode_id, llama_rerot_node_id parent_id);
    bool admit_next_child(
        uint64_t episode_id,
        int physical_slot,
        llama_seq_id exec_seq,
        llama_rerot_node_id * admitted_node);

    // Marks a fully injected child heading as RUNNING. The private planner
    // prefix may be injected before or after this transition, but ordinary
    // generated tokens are only legal once the node is RUNNING.
    bool complete_admission(uint64_t episode_id, llama_rerot_node_id node_id);

    // Apply all EXIT_INTENTs after every token in a frontier has committed.
    // Simultaneous exits use stable tree-path ordering. If no queued, starting,
    // or other running work remains, the last ordered exit is the natural
    // final survivor.
    server_rerot_frontier_result finish_frontier(uint64_t episode_id);

    // Detach a physical executor without destroying the logical episode. Used
    // after a fork parent or retired leaf has had its memory safely parked or
    // archived by the caller/runtime.
    bool detach_node(uint64_t episode_id, llama_rerot_node_id node_id);

    bool erase_episode(uint64_t episode_id);
    bool has_ready_nodes(uint64_t episode_id) const;
    bool parent_has_unadmitted_children(
        uint64_t episode_id,
        llama_rerot_node_id parent_id) const;

    std::string heading_text(uint64_t episode_id, llama_rerot_node_id node_id) const;

    void advance_frontier(uint64_t episode_id);
    void clear_sequence_control(llama_seq_id seq_id);

    bool hard_abort(uint64_t episode_id, std::string reason);

private:
    llama_rerot_run_id ensure_run(
        server_rerot_episode & episode,
        server_rerot_node_runtime & node,
        llama_rerot_visibility visibility,
        llama_pos storage_pos);

    bool release_false_pending(
        server_rerot_episode & episode,
        server_rerot_node_runtime & node);

    bool publish_pending_record(
        server_rerot_episode & episode,
        server_rerot_node_runtime & node,
        llama_rerot_run_id run_id,
        const std::vector<std::string> & items);

    bool publish_pending_run(
        server_rerot_episode & episode,
        server_rerot_node_runtime & node,
        llama_rerot_run_id run_id,
        bool topology_change);

    bool retire_node(
        server_rerot_episode & episode,
        server_rerot_node_runtime & node,
        int * released_slot);

    bool ensure_archive_seq(server_rerot_episode & episode, llama_seq_id source_seq);
    void archive_public_runs(
        server_rerot_episode & episode,
        const server_rerot_node_runtime & node,
        llama_seq_id archive_seq);

    bool finalize_exit_marker(
        server_rerot_episode & episode,
        server_rerot_node_runtime & node,
        llama_rerot_run_id run_id);

    bool build_reader_view_desc(
        const server_rerot_episode & episode,
        const server_rerot_node_runtime & node,
        llama_rerot_run_id query_run,
        std::vector<uint32_t> & ordered_runs,
        llama_rerot_reader_view_desc & desc) const;

    uint64_t next_publish_epoch(server_rerot_episode & episode);
    std::optional<llama_seq_id> alloc_internal_seq();
    void free_internal_seq(llama_seq_id seq_id);
    bool fail_episode(server_rerot_episode & episode, std::string reason);

    llama_memory_t memory_ = nullptr;
    llama_rerot_frontier_mode frontier_mode_ = LLAMA_REROT_FRONTIER_STRONG;
    uint64_t next_episode_id_ = 1;
    uint32_t first_internal_seq_ = 0;
    uint32_t max_seq_ = 256;
    std::deque<llama_seq_id> free_internal_seqs_;
    std::unordered_map<uint64_t, server_rerot_episode> episodes_;
    std::unordered_map<int, uint64_t> slot_to_episode_;
};

