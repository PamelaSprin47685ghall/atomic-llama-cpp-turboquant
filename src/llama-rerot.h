#pragma once

#include "llama.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

// Internal, experimental support for Recursive Elastic Ring-of-Thought (RERoT).
//
// This header deliberately contains only stable logical identifiers and pure
// view/DDVR helpers. Physical KV ownership remains in llama_kv_cells and server
// scheduling remains in tools/server/server-rerot.*.

using llama_rerot_node_id = uint32_t;
using llama_rerot_run_id  = uint32_t;

static constexpr llama_rerot_node_id LLAMA_REROT_NODE_INVALID =
    std::numeric_limits<llama_rerot_node_id>::max();
static constexpr llama_rerot_run_id LLAMA_REROT_RUN_INVALID =
    std::numeric_limits<llama_rerot_run_id>::max();

enum class llama_rerot_visibility : uint8_t {
    normal = 0,
    public_live,
    private_control,
    pending_record,
};

// Metadata attached to the physical KV cell by llama_kv_cells. Keeping this
// POD-like and index-local lets existing save/copy/pack code move it together
// with position, extents, shifts, and sequence references.
struct llama_kv_rerot_meta {
    uint64_t episode_id = 0;
    llama_rerot_node_id node_id = LLAMA_REROT_NODE_INVALID;
    llama_rerot_run_id run_id = LLAMA_REROT_RUN_INVALID;
    uint64_t publish_epoch = 0;
    uint64_t frontier = 0;
    llama_rerot_visibility visibility = llama_rerot_visibility::normal;

    void reset() {
        episode_id = 0;
        node_id = LLAMA_REROT_NODE_INVALID;
        run_id = LLAMA_REROT_RUN_INVALID;
        publish_epoch = 0;
        frontier = 0;
        visibility = llama_rerot_visibility::normal;
    }

    bool active() const {
        return episode_id != 0;
    }

    bool operator==(const llama_kv_rerot_meta & other) const {
        return episode_id == other.episode_id &&
               node_id == other.node_id &&
               run_id == other.run_id &&
               publish_epoch == other.publish_epoch &&
               frontier == other.frontier &&
               visibility == other.visibility;
    }

    bool operator!=(const llama_kv_rerot_meta & other) const {
        return !(*this == other);
    }
};

enum class llama_rerot_node_state : uint8_t {
    planning = 0,
    terminal_running,
    forked,
    queued,
    starting,
    running,
    retired,
    ready_suspended,
};

struct llama_rerot_run {
    llama_rerot_run_id id = LLAMA_REROT_RUN_INVALID;
    llama_rerot_node_id owner = LLAMA_REROT_NODE_INVALID;
    llama_rerot_visibility visibility = llama_rerot_visibility::normal;

    llama_pos storage_pos0 = 0;
    uint32_t token_count = 0;

    // Publication-epoch contract (enforced by append_run and validate,
    // mirroring the KV cell rules): public_live requires a non-zero publish
    // epoch, while normal, private_control, and pending_record require zero.
    // A pending run becomes public only through publish_run/reclassify_run,
    // which assign the epoch atomically with the visibility change.
    uint64_t publish_epoch = 0;
};

struct llama_rerot_node {
    llama_rerot_node_id id = LLAMA_REROT_NODE_INVALID;
    llama_rerot_node_id parent = LLAMA_REROT_NODE_INVALID;

    uint32_t depth = 0;
    uint32_t child_index = 0;
    std::string title;

    llama_rerot_node_state state = llama_rerot_node_state::planning;

    // Both vectors preserve semantic/document order.
    std::vector<llama_rerot_node_id> children;
    std::vector<llama_rerot_run_id> runs;
};

struct llama_rerot_view_run {
    // Logical span triple (run_id, virtual_pos0, token_count) flattened from
    // one PAC-DFS reader view. This struct is the only view-to-layout input
    // and MUST never carry physical KV indices (cell/key/seq ids): the KV
    // cache resolves run_ids to resident cells itself, so TriAttention
    // compaction never invalidates a previously built view. The remaining
    // fields mirror the source run for span flattening without consulting
    // physical state.
    llama_rerot_run_id run_id = LLAMA_REROT_RUN_INVALID;
    llama_rerot_node_id owner = LLAMA_REROT_NODE_INVALID;

    llama_pos storage_pos0 = 0;
    llama_pos virtual_pos0 = 0;
    uint32_t token_count = 0;
    uint64_t publish_epoch = 0;
};

struct llama_rerot_reader_view {
    uint64_t episode_id = 0;
    llama_rerot_node_id reader = LLAMA_REROT_NODE_INVALID;
    std::vector<llama_rerot_view_run> runs;
    llama_pos query_virtual_pos = 0;
};

// A reader view installed on an execution sequence. The logical document is
// owned by the scheduler; the KV cache only needs the stable ordered run ids
// and enough clock information to decide current-frontier visibility.
struct llama_rerot_reader_state {
    uint64_t episode_id = 0;
    llama_rerot_node_id reader = LLAMA_REROT_NODE_INVALID;
    llama_rerot_run_id query_run = LLAMA_REROT_RUN_INVALID;
    uint64_t frontier = 0;

    uint64_t topology_epoch = 0;
    uint64_t publish_epoch = 0;
    uint64_t layout_epoch = 0;

    llama_rerot_frontier_mode frontier_mode = LLAMA_REROT_FRONTIER_STRONG;
    std::vector<llama_rerot_run_id> ordered_runs;

    void reset() {
        episode_id = 0;
        reader = LLAMA_REROT_NODE_INVALID;
        query_run = LLAMA_REROT_RUN_INVALID;
        frontier = 0;
        topology_epoch = 0;
        publish_epoch = 0;
        layout_epoch = 0;
        frontier_mode = LLAMA_REROT_FRONTIER_STRONG;
        ordered_runs.clear();
    }

    bool active() const {
        return episode_id != 0;
    }
};

// Backend-neutral indexed attention layout. Each group is one copy of an
// original query evaluated at an effective RoPE position. Entries select a
// physical K/V row and the query group to use for that row. Query offsets make
// all entries for one output query contiguous, allowing one global online
// softmax per query/head without materializing or rephasing K.
struct llama_rerot_attn_group {
    uint32_t query_index = 0;
    llama_pos effective_pos = 0;
};

struct llama_rerot_attn_entry {
    uint32_t key_index = 0;
    uint32_t group_index = 0;
};

struct llama_rerot_attn_layout {
    uint32_t n_queries = 0;
    std::vector<llama_rerot_attn_group> groups;
    std::vector<llama_rerot_attn_entry> entries;
    std::vector<uint32_t> query_offsets;

    bool empty() const {
        return n_queries == 0;
    }

    bool validate(uint32_t n_keys, std::string * error = nullptr) const;
};

// Minimal physical-key description used by the pure layout builder. The KV
// cache fills `owned_by_reader` from sequence references; no physical-cell
// registry is retained by the logical scheduler.
struct llama_rerot_key_record {
    uint32_t key_index = 0;
    llama_pos storage_pos = 0;
    bool owned_by_reader = false;
    llama_kv_rerot_meta meta;
};

struct llama_rerot_query_layout {
    std::vector<llama_rerot_attn_group> groups;
    std::vector<llama_rerot_attn_entry> entries;
    llama_pos query_virtual_pos = 0;
};

// Build the layout for one query token. Untagged keys form the common/serial
// prefix; tagged keys are ordered by `reader.ordered_runs`. The returned query
// groups are local to this query and are remapped when multiple query layouts
// are combined by the KV cache.
llama_rerot_query_layout llama_rerot_build_query_layout(
    const llama_rerot_reader_state & reader,
    llama_pos query_storage_pos,
    const std::vector<llama_rerot_key_record> & keys);

// Pure logical document/tree model. It never stores physical KV indices.
class llama_rerot_document {
public:
    explicit llama_rerot_document(uint64_t episode_id = 1);

    void reset(uint64_t episode_id);

    uint64_t episode_id() const;
    llama_rerot_node_id root() const;

    llama_rerot_node_id create_child(
        llama_rerot_node_id parent,
        std::string title,
        llama_rerot_node_state state = llama_rerot_node_state::queued);

    llama_rerot_run_id append_run(
        llama_rerot_node_id owner,
        llama_rerot_visibility visibility,
        llama_pos storage_pos0,
        uint32_t token_count,
        uint64_t publish_epoch = 0);

    bool set_run_token_count(llama_rerot_run_id run_id, uint32_t token_count);
    bool publish_run(llama_rerot_run_id run_id, uint64_t publish_epoch);
    bool reclassify_run(
        llama_rerot_run_id run_id,
        llama_rerot_visibility expected,
        llama_rerot_visibility replacement,
        uint64_t publish_epoch = 0);
    bool set_node_state(llama_rerot_node_id node_id, llama_rerot_node_state state);

    const llama_rerot_node * node(llama_rerot_node_id node_id) const;
    const llama_rerot_run * run(llama_rerot_run_id run_id) const;

    size_t node_count() const;
    size_t run_count() const;

    bool is_ancestor(llama_rerot_node_id ancestor, llama_rerot_node_id descendant) const;
    std::vector<uint32_t> tree_path(llama_rerot_node_id node_id) const;

    // Path-Anchored Cyclic DFS render of the whole tree for one reader: at
    // every node on the reader path the reader-branch child renders last and
    // preceding/following siblings keep document order; off-path subtrees
    // render in stable document order. The render depends only on tree
    // topology, run ownership/visibility, and token counts -- never on node
    // scheduling state (queued/running/retired) or queue order. Output runs
    // densely tile virtual positions [0, L) with each visible run exactly
    // once; query_virtual_pos == L. Throws std::out_of_range for an unknown
    // reader.
    llama_rerot_reader_view build_view(llama_rerot_node_id reader) const;

    // Verifies episode/node/run id density and backlinks, the
    // publish-epoch/visibility contract, and -- for EVERY node as reader --
    // PAC-DFS density ([0, L), no overlap), exactly-once/completeness of
    // visible runs, view/source field fidelity, and sibling-block ordering
    // with own-subtree-last. Returns false and describes the first violated
    // invariant.
    bool validate(std::string * error = nullptr) const;

private:
    bool run_visible_to(const llama_rerot_run & run, llama_rerot_node_id reader) const;

    void render_node(
        llama_rerot_node_id node_id,
        llama_rerot_node_id reader,
        std::vector<llama_rerot_view_run> & out,
        llama_pos & virtual_pos) const;

    uint64_t episode_id_ = 0;
    std::vector<llama_rerot_node> nodes_;
    std::vector<llama_rerot_run> runs_;
};

// ---------------------------------------------------------------------------
// DDVR mathematical reference helpers
// ---------------------------------------------------------------------------

enum class llama_rerot_rope_layout : uint8_t {
    half = 0,
    interleaved,
};

struct llama_rerot_rope_config {
    uint32_t head_dim = 0;
    uint32_t rotary_dim = 0;
    double theta = 10000.0;
    double freq_scale = 1.0;
    llama_rerot_rope_layout layout = llama_rerot_rope_layout::half;

    // Number of rotary pairs assigned to each position axis. If all entries
    // are zero, all pairs use axis 0. For Qwen3.5 text IMRoPE, the first three
    // axes receive the text position and axis 3 remains zero.
    std::array<uint32_t, 4> axis_pair_count = { 0, 0, 0, 0 };
};

using llama_rerot_rope_pos = std::array<int64_t, 4>;

struct llama_rerot_ddvr_span {
    uint32_t key_begin = 0;
    uint32_t key_count = 0;
    int64_t storage_pos0 = 0;
    int64_t virtual_pos0 = 0;
};

llama_rerot_rope_pos llama_rerot_text_position(int64_t pos);

// Applies RoPE in place. Non-rotary tail dimensions are preserved.
bool llama_rerot_rope_apply(
    float * vector,
    size_t vector_size,
    const llama_rerot_rope_pos & position,
    const llama_rerot_rope_config & config,
    std::string * error = nullptr);

// Both functions consume unrotated query/key vectors. The materialized version
// rotates every key to its virtual position; the q-side version stores keys at
// writer-local positions and rotates one query per span. Both perform one
// global softmax over all visible keys.
std::vector<float> llama_rerot_ddvr_attention_materialized(
    const std::vector<float> & raw_query,
    const std::vector<float> & raw_keys,
    const std::vector<float> & values,
    uint32_t value_dim,
    int64_t query_virtual_pos,
    const std::vector<llama_rerot_ddvr_span> & spans,
    const llama_rerot_rope_config & config,
    float scale = 0.0f);

std::vector<float> llama_rerot_ddvr_attention_qside(
    const std::vector<float> & raw_query,
    const std::vector<float> & raw_keys,
    const std::vector<float> & values,
    uint32_t value_dim,
    int64_t query_virtual_pos,
    const std::vector<llama_rerot_ddvr_span> & spans,
    const llama_rerot_rope_config & config,
    float scale = 0.0f);

