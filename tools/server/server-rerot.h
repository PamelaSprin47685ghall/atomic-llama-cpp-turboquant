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

// Versioned persistence snapshot for the planner XML parser (§A.8). Captures
// the exact byte-level state (partial "<ol" candidate, accumulated list bytes,
// completed items, error) so a restored episode resumes token classification
// without re-emitting planner bytes. Empty strings/vectors encode absent.
struct server_rerot_planner_snapshot {
    server_rerot_parser_state state = server_rerot_parser_state::public_text;
    std::string opener_candidate;
    std::string list_bytes;
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

    server_rerot_planner_snapshot snapshot() const;
    // Restore a snapshot produced by snapshot(). Returns false (the caller
    // must treat the whole episode restore as corrupt, never best-effort)
    // when the snapshot violates parser invariants: out-of-range state,
    // oversized or non-prefix opener candidate, list bytes without the "<ol>"
    // boundary, completed without items, or failed without an error.
    bool restore(const server_rerot_planner_snapshot & snap, std::string * error = nullptr);

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
// Versioned persistence snapshot for the private end-marker parser (§A.8).
// The marker text itself is part of the snapshot: a restore with a different
// marker is corrupt, never silently accepted.
struct server_rerot_marker_snapshot {
    std::string marker;
    std::string candidate;
    server_rerot_marker_state state = server_rerot_marker_state::public_text;
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
    const std::string & marker() const;

    server_rerot_marker_snapshot snapshot() const;
    // Restore a snapshot produced by snapshot(). Returns false when the
    // snapshot violates marker invariants (empty marker, out-of-range state,
    // non-prefix candidate, completed with a residue candidate, or failed
    // without an error).
    bool restore(const server_rerot_marker_snapshot & snap, std::string * error = nullptr);

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

struct server_rerot_hard_limits {
    // Zero means unbounded. When any bound is crossed the whole episode takes
    // HARD_ABORT with finish_reason rerot_resource_exhausted: no survivor, no
    // answer, no tool call. There is deliberately no per-child, per-depth, or
    // final-answer reservation (§20).
    uint64_t max_total_tokens = 0;
    uint64_t max_nodes = 0;
    uint64_t max_queue_descriptors = 0;
    uint64_t max_frontiers = 0;
};

struct server_rerot_token_plan {
    llama_rerot_run_id run_id = LLAMA_REROT_RUN_INVALID;
    llama_pos storage_pos = -1;
    llama_rerot_visibility visibility = llama_rerot_visibility::normal;
    // True when this plan carries runtime-forced heading bytes. Headings are
    // PENDING until fully injected, then atomically published by
    // publish_heading (§16). Used only for budget accounting.
    bool is_heading = false;
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
    // Global FIFO key (§16.2). Assigned at fork publication from the current
    // frontier. Admission order is (enqueue_frontier, tree_path); no scoring.
    uint64_t enqueue_frontier = 0;
    bool exit_intent = false;

    // A.8 opaque per-lane extension state. Filled by the server integration
    // (sampler bytes, MTP checkpoint); empty means none. Persisted verbatim
    // and never interpreted here.
    std::vector<uint8_t> sampler_blob;
    std::vector<uint8_t> mtp_blob;
    // Last installed frontier-view stamp for this Lane's exec seq (A.6 MTP
    // binding). All-zero means no view installed yet.
    llama_rerot_view_stamp view_stamp = {0, 0, 0};
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

// Caller-supplied save/load identity for versioned episode persistence
// (§A.8.1): capability bitmap (LLAMA_REROT_STATE_CAP_* from llama.h) plus
// model / RoPE / Tri-calibration fingerprints. The context half (see
// llama_rerot_context_fingerprints, implemented in src/llama-context.cpp)
// derives these from the live model and cparams; server integration passes
// them through verbatim so save and load validate the same triple.
struct server_rerot_state_fingerprints {
    uint32_t caps = 0;
    uint64_t model_fp = 0;
    uint64_t rope_fp = 0;
    uint64_t tri_fp = 0;
};

// Shared-memory log-truncation outcome (§A.9): whole oldest PUBLIC runs
// dropped episode-wide. PAC-DFS re-dense-packs on the next build_view;
// layout (+publish) epochs move so MTP drafts bound to older stamps report
// stale. Active private causal state is never a truncation candidate and Tri
// physical residency is never touched by the truncation itself.
struct server_rerot_shift_result {
    uint64_t tokens_removed = 0;
    uint32_t runs_truncated = 0;
    uint32_t runs_emptied = 0;
    uint64_t new_layout_epoch = 0;
    uint64_t new_publish_epoch = 0;
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

    // Episode-level hard-resource counters (§20). Only totals are tracked:
    // no per-child, per-depth, or final-answer reservation exists.
    uint64_t generated_public_tokens = 0;
    uint64_t generated_private_tokens = 0;
    uint64_t forced_heading_tokens = 0;
    uint64_t pending_tokens = 0;
    uint64_t queue_peak = 0;
    server_rerot_hard_limits hard_limits;

    // Final acquire fence (§21.4) then serial tail (§22). refresh_final_fence
    // rebuilds the survivor view on stable shared memory; complete_serial_tail
    // retires the parallel scheduler and pins output to the survivor, which
    // keeps writing the root response stream (response_task_id).
    bool fence_refreshed = false;
    bool serial_tail = false;
    llama_rerot_node_id serial_node = LLAMA_REROT_NODE_INVALID;

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
        llama_pos storage_pos_next,
        uint64_t requested_episode_id = 0);

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

    // Final-fence continuation. The planner and private end-marker parsers are
    // permanently retired at this point; body/tool tokens remain PUBLIC so
    // segmented DDVR continues to expose the stable shared document.
    std::optional<server_rerot_token_plan> plan_serial_token(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        llama_pos storage_pos);

    // Install sequence-scoped write/view control immediately before decode.
    bool install_token_plan(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        const server_rerot_token_plan & plan,
        size_t * span_count_out = nullptr);

    // Before the first fork the root owns one ordinary contiguous sequence.
    // Tag writes transactionally but leave its stock membership attention
    // path intact; DDVR is activated only once archive/foreign runs exist.
    bool install_token_plan_write_only(
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
    // Materialize one archive ownership ref for every currently resident
    // PUBLIC run. Used by TriAttention as the episode's single semantic union.
    bool sync_public_archive(
        uint64_t episode_id,
        std::vector<llama_seq_id> * semantic_seq_ids_out = nullptr);
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

    // Atomic budget gate (§19.3). Call once before decoding a frontier batch:
    // when the episode already crossed any hard limit the whole episode takes
    // HARD_ABORT (finish_reason rerot_resource_exhausted) instead of running
    // a partial Lane subset. Returns false when aborted or unknown.
    bool begin_frontier(uint64_t episode_id);
    void set_hard_limits(uint64_t episode_id, server_rerot_hard_limits limits);

    // Final acquire fence (§21.4). Rebuilds the survivor reader view after all
    // same-frontier public writes and exits have committed, and reinstalls it
    // on the survivor exec seq so the fence </think> decode observes stable
    // shared memory. On success fills ordered_runs_out with the fence view.
    bool refresh_final_fence(
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        std::vector<uint32_t> * ordered_runs_out);

    // Serial tail transition (§22). Disables the parallel scheduler and pins
    // ordinary tool-call / body decode to the fence survivor. The survivor
    // keeps writing the episode root response stream.
    bool complete_serial_tail(uint64_t episode_id, llama_rerot_node_id node_id);

    // Response ownership (§18.3): the HTTP/SSE stream belongs to the episode
    // root request, never to an internal Lane slot.
    int response_task_id(uint64_t episode_id) const;
    llama_rerot_frontier_mode frontier_mode() const;

    // Versioned episode persistence (§A.8, episode-level demotion unit).
    // Save serializes the logical episode (never physical KV indices); load
    // validates magic/version/caps/fingerprints and rebuilds PAC-DFS spans
    // from logical ownership. See the free-function wire contract at the end
    // of this header for accept/reject rules.
    bool save_episode(
        uint64_t episode_id,
        const server_rerot_state_fingerprints & fp,
        std::vector<uint8_t> * blob_out,
        std::string * error_out = nullptr) const;
    bool load_episode(
        const uint8_t * data,
        size_t size,
        const server_rerot_state_fingerprints & expected_fp,
        uint64_t * episode_id_out = nullptr,
        std::string * error_out = nullptr);
    // Episode-level demotion (§A.8.2): release transient physical slot
    // bindings so the episode can be swapped as a logical unit.
    // Tree/runs/epochs/queue/counters/lineage are preserved; demoted
    // RUNNING/STARTING members await re-admission. Returns false for unknown
    // episodes.
    bool demote_episode(uint64_t episode_id);

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
    // Returns true when a hard limit was crossed (episode aborted with
    // finish_reason rerot_resource_exhausted). False otherwise.
    bool check_hard_limits(server_rerot_episode & episode);

    llama_memory_t memory_ = nullptr;
    llama_rerot_frontier_mode frontier_mode_ = LLAMA_REROT_FRONTIER_STRONG;
    uint64_t next_episode_id_ = 1;
    uint32_t first_internal_seq_ = 0;
    uint32_t max_seq_ = 256;
    std::deque<llama_seq_id> free_internal_seqs_;
    std::unordered_map<uint64_t, server_rerot_episode> episodes_;
    std::unordered_map<int, uint64_t> slot_to_episode_;
};

// ---------------------------------------------------------------------------
// Versioned episode persistence + shared-memory log truncation (§§25,A.8-A.10)
// ---------------------------------------------------------------------------
//
// Wire format (little-endian POD; the episode carries no tensor bytes, so no
// float/endian-sensitive payload appears):
//   u32 magic (LLAMA_REROT_STATE_MAGIC) | u32 version (LLAMA_REROT_STATE_VERSION)
//   u32 caps (LLAMA_REROT_STATE_CAP_* bitmap supplied by the caller)
//   u64 model_fp | u64 rope_fp | u64 tri_fp (caller fingerprints)
//   u64 episode_id | u64 frontier | u64 publish/topology/layout epochs
//   i32 base_prefix_end | i32 root/response task ids | i32 archive_seq
//   u8 flags (topology_barrier, finalizing, hard_aborted, fence_refreshed,
//     serial_tail) | u32 serial_node | string abort_reason
//   u64 x5 counters (public/private/forced-heading/pending/queue_peak)
//   u64 x4 hard limits (total_tokens/nodes/queue/frontiers)
//   u32-counted vectors: forked_this_frontier, ready_queue (FIFO order
//     preserved verbatim), running, starting
//   document nodes (u32 count; per node: u32 id/parent/depth/child_index,
//     u8 state, string title, u32-counted children[], u32-counted runs[])
//   document runs (u32 count; per run: u32 id/owner, u8 visibility,
//     i32 storage_pos0, u32 token_count, u64 publish_epoch)
//   node runtimes (u32 count, must equal node count; per runtime: u32 id,
//     i32 physical_slot/exec_seq/parked_seq/storage_pos_next, optional run
//     refs (u8 present + u32 id) x3, planner snapshot, marker snapshot,
//     opaque sampler/mtp blobs (u64 length + bytes), view stamp (3 x u64))
// Strings/blobs are u64 length + bytes. No physical KV cell/key indices are
// ever stored: restore rebuilds PAC-DFS spans from logical ownership via
// build_view and recomputes virtual positions densely. Exec/parked/archive
// seq ids ARE stored as KV/recurrent lineage handles (A.8), not as semantic
// addresses.
//
// Load rules (never best-effort): magic mismatch, version skew in either
// direction (an old binary reading a new state and a new binary reading a
// legacy state are both rejected — the latter must not silently upgrade),
// unknown caps bits, a missing REROT bit, or any model/rope/tri fingerprint
// mismatch is an explicit error naming the offending field. Corrupt topology
// (non-dense ids, bad parent/child backlinks, run owner/visibility contract
// violations, parser invariant violations) fails document validation and is
// rejected with the first violated invariant. SAVE refuses only genuinely
// inconsistent transient state with an explicit reason: a bound physical slot
// outside running/starting (detach the node or complete admission first),
// runtime/document misalignment, or untagged (normal-visibility) runs.
// STARTING admission heading injection spans frontiers and persists verbatim.

// Serialize one logical episode. Returns an empty vector (with *error_out
// set) when the episode holds genuinely unserializable transient state;
// never emits a partial blob.
std::vector<uint8_t> server_rerot_episode_save(
    const server_rerot_episode & episode,
    const server_rerot_state_fingerprints & fp,
    std::string * error_out = nullptr);

// Deserialize and validate a blob produced by server_rerot_episode_save into
// *episode_out (untouched on failure). Virtual positions are NOT stored;
// callers observe them via document.build_view(reader) after a successful
// load, which re-dense-packs from logical ownership.
bool server_rerot_episode_load(
    const uint8_t * data,
    size_t size,
    const server_rerot_state_fingerprints & expected_fp,
    server_rerot_episode * episode_out,
    std::string * error_out = nullptr);

// Episode-level demotion helper (§A.8.2): clear transient physical slot
// bindings on a detached episode struct. Tree/runs/epochs/queue/counters/
// lineage are preserved; demoted RUNNING/STARTING members await re-admission
// by the runtime (which additionally drops its slot map entries).
void server_rerot_episode_demote_to_logical(server_rerot_episode & episode);

// Shared-memory log truncation (§A.9): drop whole oldest PUBLIC runs
// episode-wide until tokens_removed >= max_tokens_to_remove or no eligible
// run remains. Eligibility: visibility public_live, token_count > 0, storage
// entirely at or after base_prefix_end (the common system/user prefix is
// always kept), and NOT referenced by any live Lane's active
// public/private/pending run pointer (active causal tails are never cut).
// Whole-run granularity is deliberate: runs are the semantic log segments,
// and the frozen document offers no partial front-slice primitive. Sets
// topology_barrier_pending, bumps layout (+publish) epochs, and fills
// *result_out (no-op success with zeros when nothing is eligible). Returns
// false only for a missing/invalid episode or epoch overflow. Physical KV
// cells of dropped runs become unreachable (excluded from future
// ordered_runs) and are reclaimed lazily via Tri/lifecycle — Tri eviction
// itself stays physical-only and separate, and is never invoked here.
bool server_rerot_truncate_oldest_public(
    server_rerot_episode & episode,
    uint64_t max_tokens_to_remove,
    server_rerot_shift_result * result_out = nullptr,
    std::string * error_out = nullptr);

// ---------------------------------------------------------------------------
// Context-half envelope + shift applier, implemented in src/llama-context.cpp
// (declared here so server integration has a single persistence header; the
// core TU includes this header for one compiler-checked source of truth).
// ---------------------------------------------------------------------------
struct llama_context;

// Derive the save/load identity triple (+caps) for the calling context from
// the live model and cparams. The model_fp covers arch + ctx-train size, the
// rope_fp covers rope base/scale, and the tri_fp covers the Tri ratio/enabled
// calibration. Returns zeros (with REROT bit clear) when RERoT is disabled.
server_rerot_state_fingerprints llama_rerot_context_fingerprints(const struct llama_context * ctx);

// Serialize the active episode's context control state (episode id, frontier
// mode, topology/publish/layout epochs, per-seq MTP view stamps, caps,
// fingerprint triple) into a versioned envelope. Physical KV/recurrent tensor
// bytes are deliberately omitted: the logical episode (server blob + this
// envelope) is authoritative and resident classified cells, write tags, and
// reader views are transient execution state reinstalled on restore via
// set_write_tag / set_frontier_views (episode-level demotion). Returns false
// (with *error_out set) when RERoT is disabled, no episode is active, or a
// genuinely unserializable transient is present (backend sampler state bound
// to an episode exec seq, which has no byte-serializable form here).
bool llama_rerot_context_save_envelope(
    struct llama_context * ctx,
    std::vector<uint8_t> & blob_out,
    std::string * error_out = nullptr);

// Restore an envelope produced by llama_rerot_context_save_envelope into a
// context with no active episode (fresh restore target) or with the SAME
// episode already active (idempotent re-restore). Validates magic/version/
// caps/fingerprint triple explicitly and rejects old-binary-new-state,
// legacy, unknown-feature, and any fingerprint mismatch without best effort.
// Restoring atop a DIFFERENT active episode is rejected (end it first).
// Memory tensor state is not touched: pair with server_rerot runtime
// load_episode for the logical tree, then reinstall views before decode.
bool llama_rerot_context_load_envelope(
    struct llama_context * ctx,
    const uint8_t * data,
    size_t size,
    std::string * error_out = nullptr);

// Mirror a server-side log truncation into context control state (§A.9):
// adopt (never rewind) the post-truncation layout/publish epochs and roll
// every installed per-seq view stamp forward to them so MTP drafts bound to
// pre-shift stamps report stale (caller restores its checkpoint, drops
// uncommitted draft state, and re-drafts). Touches no seq data (active
// private causal state is never cut) and never invokes Tri reclaim.
// Returns false without mutation when RERoT is disabled, no episode is
// active, or the episode id does not match.
bool llama_rerot_context_apply_shift(
    struct llama_context * ctx,
    uint64_t episode_id,
    uint64_t tokens_removed,
    uint64_t new_layout_epoch,
    uint64_t new_publish_epoch,
    std::string * error_out = nullptr);

