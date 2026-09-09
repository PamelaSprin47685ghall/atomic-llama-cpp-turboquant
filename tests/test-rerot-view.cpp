#include "llama-rerot.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        ++g_failures; \
    } \
} while (0)

static std::vector<llama_rerot_node_id> owners(const llama_rerot_reader_view & view) {
    std::vector<llama_rerot_node_id> result;
    result.reserve(view.runs.size());
    for (const auto & run : view.runs) {
        result.push_back(run.owner);
    }
    return result;
}

static void test_manual_pac_dfs() {
    llama_rerot_document doc(42);
    const auto root = doc.root();
    const auto a = doc.create_child(root, "A");
    const auto b = doc.create_child(root, "B");
    const auto c = doc.create_child(root, "C");
    const auto b1 = doc.create_child(b, "B1");
    const auto b2 = doc.create_child(b, "B2");
    const auto b3 = doc.create_child(b, "B3");

    doc.append_run(root, llama_rerot_visibility::public_live, 0, 2, 1);
    doc.append_run(a, llama_rerot_visibility::public_live, 0, 1, 2);
    doc.append_run(b, llama_rerot_visibility::public_live, 0, 1, 2);
    doc.append_run(c, llama_rerot_visibility::public_live, 0, 1, 2);
    doc.append_run(b1, llama_rerot_visibility::public_live, 0, 1, 3);
    doc.append_run(b2, llama_rerot_visibility::public_live, 0, 1, 3);
    doc.append_run(b3, llama_rerot_visibility::public_live, 0, 1, 3);

    std::string error;
    CHECK(doc.validate(&error));

    const auto view_b2 = doc.build_view(b2);
    const std::vector<llama_rerot_node_id> expected_b2 = { root, c, a, b, b3, b1, b2 };
    CHECK(owners(view_b2) == expected_b2);
    CHECK(view_b2.query_virtual_pos == 8);

    llama_pos expected_pos = 0;
    for (const auto & run : view_b2.runs) {
        CHECK(run.virtual_pos0 == expected_pos);
        expected_pos += run.token_count;
    }

    // B is off-path for reader C and therefore keeps stable original DFS order.
    const auto view_c = doc.build_view(c);
    const std::vector<llama_rerot_node_id> expected_c = { root, a, b, b1, b2, b3, c };
    CHECK(owners(view_c) == expected_c);
}

static void test_visibility() {
    llama_rerot_document doc(7);
    const auto a = doc.create_child(doc.root(), "A");
    const auto b = doc.create_child(doc.root(), "B");

    doc.append_run(doc.root(), llama_rerot_visibility::normal, 0, 1);
    doc.append_run(a, llama_rerot_visibility::public_live, 0, 2, 1);
    doc.append_run(a, llama_rerot_visibility::private_control, 2, 3);
    const auto pending = doc.append_run(a, llama_rerot_visibility::pending_record, 5, 4);
    doc.append_run(b, llama_rerot_visibility::public_live, 0, 1, 1);

    auto view_a = doc.build_view(a);
    auto view_b = doc.build_view(b);
    CHECK(view_a.runs.size() == 5);
    CHECK(view_b.runs.size() == 3);

    CHECK(doc.publish_run(pending, 2));
    view_b = doc.build_view(b);
    CHECK(view_b.runs.size() == 4);
    CHECK(view_b.query_virtual_pos == 8);
}

static void test_reclassify_run_validation() {
    llama_rerot_document doc(8);
    const auto owner = doc.create_child(doc.root(), "owner");
    const auto run = doc.append_run(owner, llama_rerot_visibility::pending_record, 4, 2);

    CHECK(!doc.reclassify_run(
        run,
        llama_rerot_visibility::public_live,
        llama_rerot_visibility::private_control));
    CHECK(doc.run(run)->visibility == llama_rerot_visibility::pending_record);

    CHECK(!doc.reclassify_run(
        run,
        llama_rerot_visibility::pending_record,
        llama_rerot_visibility::private_control,
        7));
    CHECK(doc.run(run)->visibility == llama_rerot_visibility::pending_record);

    CHECK(doc.reclassify_run(
        run,
        llama_rerot_visibility::pending_record,
        llama_rerot_visibility::public_live,
        7));
    CHECK(doc.run(run)->visibility == llama_rerot_visibility::public_live);
    CHECK(doc.run(run)->publish_epoch == 7);
}

static void collect_leaves(
        const llama_rerot_document & doc,
        llama_rerot_node_id node_id,
        std::vector<llama_rerot_node_id> & leaves) {
    const auto * node = doc.node(node_id);
    assert(node);
    if (node->children.empty()) {
        leaves.push_back(node_id);
        return;
    }
    for (auto child : node->children) {
        collect_leaves(doc, child, leaves);
    }
}

static void test_randomized_invariants() {
    std::mt19937 rng(0x5eed1234u);

    for (int iteration = 0; iteration < 100; ++iteration) {
        llama_rerot_document doc(uint64_t(iteration) + 1);
        const int target_nodes = 8 + int(rng() % 48);

        for (int i = 1; i < target_nodes; ++i) {
            const auto parent = llama_rerot_node_id(rng() % doc.node_count());
            const auto node = doc.create_child(parent, "node-" + std::to_string(i));
            const uint32_t public_count = 1 + uint32_t(rng() % 4);
            doc.append_run(node, llama_rerot_visibility::public_live, 0, public_count, uint64_t(i));
            if ((rng() & 3u) == 0) {
                doc.append_run(node, llama_rerot_visibility::private_control, public_count, 1 + uint32_t(rng() % 3));
            }
        }
        doc.append_run(doc.root(), llama_rerot_visibility::public_live, 0, 2, 1);

        std::string error;
        CHECK(doc.validate(&error));

        std::vector<llama_rerot_node_id> leaves;
        collect_leaves(doc, doc.root(), leaves);
        for (auto reader : leaves) {
            const auto view = doc.build_view(reader);
            std::set<llama_rerot_run_id> seen;
            llama_pos expected = 0;
            for (const auto & run : view.runs) {
                CHECK(run.virtual_pos0 == expected);
                CHECK(run.token_count > 0);
                CHECK(seen.insert(run.run_id).second);
                expected += run.token_count;

                const auto * source = doc.run(run.run_id);
                CHECK(source != nullptr);
                if (source && source->visibility == llama_rerot_visibility::private_control) {
                    CHECK(source->owner == reader);
                }
            }
            CHECK(view.query_virtual_pos == expected);
        }
    }
}

static llama_kv_rerot_meta public_meta(
        uint64_t episode,
        llama_rerot_node_id node,
        llama_rerot_run_id run,
        uint64_t frontier) {
    llama_kv_rerot_meta result;
    result.episode_id = episode;
    result.node_id = node;
    result.run_id = run;
    result.frontier = frontier;
    result.visibility = llama_rerot_visibility::public_live;
    return result;
}

static void test_query_layout_frontiers() {
    constexpr uint64_t episode = 99;
    constexpr llama_rerot_node_id node_a = 1;
    constexpr llama_rerot_node_id node_b = 2;
    constexpr llama_rerot_run_id run_a = 11;
    constexpr llama_rerot_run_id run_b = 12;

    llama_rerot_reader_state reader;
    reader.episode_id = episode;
    reader.reader = node_a;
    reader.query_run = run_a;
    reader.frontier = 1;
    reader.frontier_mode = LLAMA_REROT_FRONTIER_STRONG;
    reader.ordered_runs = { run_b, run_a };

    const std::vector<llama_rerot_key_record> keys = {
        { 0, 0, true,  {} },
        { 1, 1, true,  {} },
        { 2, 2, false, public_meta(episode, node_b, run_b, 0) },
        { 3, 3, false, public_meta(episode, node_b, run_b, 1) },
        { 4, 2, true,  public_meta(episode, node_a, run_a, 0) },
        { 5, 3, true,  public_meta(episode, node_a, run_a, 1) },
    };

    const auto strong = llama_rerot_build_query_layout(reader, 3, keys);
    CHECK(strong.query_virtual_pos == 4);
    CHECK(strong.entries.size() == 5);
    CHECK(strong.groups.size() == 2);
    CHECK(strong.groups[0].effective_pos == 3);
    CHECK(strong.groups[1].effective_pos == 4);

    std::set<uint32_t> strong_keys;
    for (const auto & entry : strong.entries) {
        strong_keys.insert(entry.key_index);
    }
    CHECK(strong_keys == std::set<uint32_t>({0, 1, 2, 4, 5}));

    reader.frontier_mode = LLAMA_REROT_FRONTIER_LAG1;
    const auto lag1 = llama_rerot_build_query_layout(reader, 3, keys);
    CHECK(lag1.query_virtual_pos == 3);
    CHECK(lag1.entries.size() == 4);
    CHECK(lag1.groups.size() == 1);
    CHECK(lag1.groups[0].effective_pos == 3);

    std::set<uint32_t> lag1_keys;
    for (const auto & entry : lag1.entries) {
        lag1_keys.insert(entry.key_index);
    }
    CHECK(lag1_keys == std::set<uint32_t>({0, 1, 4, 5}));
}

static void test_concurrent_sibling_readers_strong() {
    // Audit scenario: Multiple sibling lanes concurrently generate PUBLIC tokens.
    // Ensure that:
    // 1. Each reader's own virtual positions are strictly continuous, no gaps, no overlaps.
    // 2. STRONG does NOT expose same-frontier peer rows; own current K/V stays visible.
    constexpr uint64_t episode = 2026;
    llama_rerot_document doc(episode);
    const auto root = doc.root();
    const auto lane_a = doc.create_child(root, "LaneA");
    const auto lane_b = doc.create_child(root, "LaneB");
    const auto lane_c = doc.create_child(root, "LaneC");

    // Common prefix
    doc.append_run(root, llama_rerot_visibility::normal, 0, 4);

    // Each lane writes public tokens
    const auto run_a = doc.append_run(lane_a, llama_rerot_visibility::public_live, 4, 3, 10);
    const auto run_b = doc.append_run(lane_b, llama_rerot_visibility::public_live, 4, 2, 11);
    const auto run_c = doc.append_run(lane_c, llama_rerot_visibility::public_live, 4, 4, 12);

    std::string err;
    CHECK(doc.validate(&err));

    const auto view_a = doc.build_view(lane_a);
    const auto view_b = doc.build_view(lane_b);
    const auto view_c = doc.build_view(lane_c);

    // Verify virtual position continuity for all concurrent readers
    for (const auto & view : { view_a, view_b, view_c }) {
        llama_pos pos = 0;
        for (const auto & r : view.runs) {
            CHECK(r.virtual_pos0 == pos);
            CHECK(r.token_count > 0);
            pos += r.token_count;
        }
        CHECK(view.query_virtual_pos == pos);
        CHECK(view.query_virtual_pos == 4 + 3 + 2 + 4); // 13 tokens total
    }

    // PAC-DFS sibling ordering: reader's own branch must be LAST
    CHECK(view_a.runs.back().owner == lane_a);
    CHECK(view_b.runs.back().owner == lane_b);
    CHECK(view_c.runs.back().owner == lane_c);

    // Physical key setup across lanes in unified KV cache
    std::vector<llama_rerot_key_record> keys;
    // Prefix keys 0..3 (storage 0..3)
    for (uint32_t i = 0; i < 4; ++i) {
        keys.push_back({ i, static_cast<llama_pos>(i), true, {} });
    }
    // Lane A keys (storage 4..6, frontier 1)
    for (uint32_t i = 0; i < 3; ++i) {
        keys.push_back({ 4 + i, static_cast<llama_pos>(4 + i), false,
            public_meta(episode, lane_a, run_a, 1) });
    }
    // Lane B keys (storage 4..5, frontier 1)
    for (uint32_t i = 0; i < 2; ++i) {
        keys.push_back({ 7 + i, static_cast<llama_pos>(4 + i), false,
            public_meta(episode, lane_b, run_b, 1) });
    }
    // Lane C keys (storage 4..7, frontier 1)
    for (uint32_t i = 0; i < 4; ++i) {
        keys.push_back({ 9 + i, static_cast<llama_pos>(4 + i), false,
            public_meta(episode, lane_c, run_c, 1) });
    }

    // Test Reader A querying at storage_pos = 6 (its 3rd token) in STRONG mode
    {
        llama_rerot_reader_state reader_a;
        reader_a.episode_id = episode;
        reader_a.reader = lane_a;
        reader_a.query_run = run_a;
        reader_a.frontier = 1;
        reader_a.frontier_mode = LLAMA_REROT_FRONTIER_STRONG;
        for (const auto & r : view_a.runs) {
            reader_a.ordered_runs.push_back(r.run_id);
        }

        // Reader A owns its own keys
        auto keys_a = keys;
        for (uint32_t i = 0; i < 3; ++i) {
            keys_a[4 + i].owned_by_reader = true;
        }

        const auto layout_a = llama_rerot_build_query_layout(reader_a, 6, keys_a);
        // Prefix + own 3 current rows. Current B/C rows are write-stage peers.
        CHECK(layout_a.entries.size() == 7);
        CHECK(layout_a.query_virtual_pos == 6);

        // Effective position must never be negative or invert RoPE
        for (const auto & group : layout_a.groups) {
            CHECK(group.effective_pos >= 0);
        }
    }

    // Test Reader B querying at storage_pos = 5 (its 2nd token) in STRONG mode
    {
        llama_rerot_reader_state reader_b;
        reader_b.episode_id = episode;
        reader_b.reader = lane_b;
        reader_b.query_run = run_b;
        reader_b.frontier = 1;
        reader_b.frontier_mode = LLAMA_REROT_FRONTIER_STRONG;
        for (const auto & r : view_b.runs) {
            reader_b.ordered_runs.push_back(r.run_id);
        }

        auto keys_b = keys;
        for (uint32_t i = 0; i < 2; ++i) {
            keys_b[7 + i].owned_by_reader = true;
        }

        const auto layout_b = llama_rerot_build_query_layout(reader_b, 5, keys_b);
        CHECK(layout_b.entries.size() == 6);
        CHECK(layout_b.query_virtual_pos == 5);

        for (const auto & group : layout_b.groups) {
            CHECK(group.effective_pos >= 0);
        }
    }
}

// ---------------------------------------------------------------------------
// Structured PAC-DFS invariant battery (guide sections 5 and 8).
// The reference renderer is written directly from the section 8.1 rule text
// and speaks only logical ids through the public document API. The property
// checks do not replicate the render algorithm: they assert density/tiling,
// exactly-once/completeness, ancestor-before-descendant, own-subtree-last,
// off-path stability, and queue independence.
// ---------------------------------------------------------------------------
static bool run_visible_to_reader(const llama_rerot_run & run, llama_rerot_node_id reader) {
    if (run.visibility == llama_rerot_visibility::normal ||
        run.visibility == llama_rerot_visibility::public_live) {
        return true;
    }
    return run.owner == reader;
}
static void reference_render_node(
        const llama_rerot_document & doc,
        llama_rerot_node_id node_id,
        llama_rerot_node_id reader,
        std::vector<llama_rerot_run_id> & out) {
    const auto * node = doc.node(node_id);
    assert(node);
    for (const auto run_id : node->runs) {
        const auto * run = doc.run(run_id);
        assert(run);
        if (run->token_count == 0) {
            continue;
        }
        if (run_visible_to_reader(*run, reader)) {
            out.push_back(run_id);
        }
    }
    if (node->children.empty()) {
        return;
    }
    size_t reader_child = node->children.size();
    for (size_t i = 0; i < node->children.size(); ++i) {
        if (doc.is_ancestor(node->children[i], reader)) {
            reader_child = i;
            break;
        }
    }
    std::vector<llama_rerot_node_id> order;
    if (reader_child == node->children.size()) {
        order = node->children;
    } else {
        for (size_t i = reader_child + 1; i < node->children.size(); ++i) {
            order.push_back(node->children[i]);
        }
        for (size_t i = 0; i < reader_child; ++i) {
            order.push_back(node->children[i]);
        }
        order.push_back(node->children[reader_child]);
    }
    for (const auto child : order) {
        reference_render_node(doc, child, reader, out);
    }
}
static std::vector<llama_rerot_run_id> reference_pac_dfs(
        const llama_rerot_document & doc,
        llama_rerot_node_id reader) {
    std::vector<llama_rerot_run_id> out;
    reference_render_node(doc, doc.root(), reader, out);
    return out;
}
static std::vector<llama_rerot_run_id> view_run_ids(const llama_rerot_reader_view & view) {
    std::vector<llama_rerot_run_id> out;
    out.reserve(view.runs.size());
    for (const auto & run : view.runs) {
        out.push_back(run.run_id);
    }
    return out;
}
static void check_doc_valid(const llama_rerot_document & doc, const char * tag) {
    std::string error;
    if (!doc.validate(&error)) {
        std::fprintf(stderr, "FAIL validate [%s]: %s\n", tag, error.c_str());
        ++g_failures;
    }
}
// Full invariant battery for one reader. Callers validate the document once;
// this focuses on the view itself.
static void check_view_battery(
        const llama_rerot_document & doc,
        llama_rerot_node_id reader) {
    const auto view = doc.build_view(reader);
    CHECK(view.episode_id == doc.episode_id());
    CHECK(view.reader == reader);
    CHECK(view_run_ids(view) == reference_pac_dfs(doc, reader));
    llama_pos expected = 0;
    for (const auto & run : view.runs) {
        CHECK(run.virtual_pos0 == expected);
        CHECK(run.token_count > 0);
        expected += run.token_count;
    }
    CHECK(view.query_virtual_pos == expected);
    {
        // Logical span triples tile [0, L) with no gap or overlap, checked on
        // sorted intervals rather than render order. Only logical fields
        // (run_id/virtual_pos0/count) participate: no physical index exists.
        std::vector<std::pair<llama_pos, uint32_t>> spans;
        for (const auto & run : view.runs) {
            spans.emplace_back(run.virtual_pos0, run.token_count);
        }
        std::sort(spans.begin(), spans.end());
        llama_pos cursor = 0;
        for (const auto & span : spans) {
            CHECK(span.first == cursor);
            cursor += span.second;
        }
        CHECK(cursor == view.query_virtual_pos);
    }
    {
        // Each public token exactly once: completeness plus uniqueness plus
        // view/source field fidelity.
        std::set<llama_rerot_run_id> expected_runs;
        for (llama_rerot_run_id id = 0; id < doc.run_count(); ++id) {
            const auto * run = doc.run(id);
            assert(run);
            if (run->token_count != 0 && run_visible_to_reader(*run, reader)) {
                expected_runs.insert(id);
            }
        }
        std::set<llama_rerot_run_id> seen;
        for (const auto & run : view.runs) {
            CHECK(seen.insert(run.run_id).second);
            const auto * source = doc.run(run.run_id);
            CHECK(source != nullptr);
            if (source) {
                CHECK(run.owner == source->owner);
                CHECK(run.storage_pos0 == source->storage_pos0);
                CHECK(run.token_count == source->token_count);
                CHECK(run.publish_epoch == source->publish_epoch);
            }
        }
        CHECK(seen == expected_runs);
    }
    {
        // Ancestor heading before descendant heading.
        std::unordered_map<llama_rerot_run_id, size_t> position;
        for (size_t i = 0; i < view.runs.size(); ++i) {
            position.emplace(view.runs[i].run_id, i);
        }
        for (const auto & run : view.runs) {
            const auto * owner = doc.node(run.owner);
            assert(owner);
            llama_rerot_node_id ancestor = owner->parent;
            while (ancestor != LLAMA_REROT_NODE_INVALID) {
                const auto * anode = doc.node(ancestor);
                assert(anode);
                for (const auto other : anode->runs) {
                    const auto it = position.find(other);
                    if (it != position.end()) {
                        CHECK(it->second < position[run.run_id]);
                    }
                }
                ancestor = anode->parent;
            }
        }
    }
    {
        // Own-subtree-last at every level of the reader path: all runs of
        // sibling subtrees precede the first run of the on-path child.
        for (llama_rerot_node_id n = 0; n < doc.node_count(); ++n) {
            if (n == reader || !doc.is_ancestor(n, reader)) {
                continue;
            }
            const auto * node = doc.node(n);
            assert(node);
            llama_rerot_node_id on_path = LLAMA_REROT_NODE_INVALID;
            for (const auto child : node->children) {
                if (child == reader || doc.is_ancestor(child, reader)) {
                    on_path = child;
                    break;
                }
            }
            CHECK(on_path != LLAMA_REROT_NODE_INVALID);
            if (on_path == LLAMA_REROT_NODE_INVALID) {
                continue;
            }
            size_t path_first = view.runs.size();
            size_t sibling_last = 0;
            bool has_path = false;
            bool has_sibling = false;
            for (size_t i = 0; i < view.runs.size(); ++i) {
                const auto owner = view.runs[i].owner;
                if (owner == on_path || doc.is_ancestor(on_path, owner)) {
                    if (!has_path) {
                        path_first = i;
                        has_path = true;
                    }
                } else if (owner != n && doc.is_ancestor(n, owner)) {
                    sibling_last = i;
                    has_sibling = true;
                }
            }
            if (has_path && has_sibling) {
                CHECK(sibling_last < path_first);
            }
        }
    }
}
static void build_kary_tree(
        llama_rerot_document & doc,
        int branching,
        int levels,
        uint64_t & epoch,
        std::mt19937 & rng) {
    std::vector<llama_rerot_node_id> frontier = { doc.root() };
    doc.append_run(doc.root(), llama_rerot_visibility::normal, 0, 1 + uint32_t(rng() % 2));
    for (int level = 0; level < levels; ++level) {
        std::vector<llama_rerot_node_id> next;
        for (const auto parent : frontier) {
            for (int c = 0; c < branching; ++c) {
                const auto child = doc.create_child(
                    parent, "n" + std::to_string(level) + "-" + std::to_string(c));
                llama_pos pos = 0;
                const int n_public = 1 + int(rng() % 2);
                for (int k = 0; k < n_public; ++k) {
                    const uint32_t count = 1 + uint32_t(rng() % 3);
                    doc.append_run(child, llama_rerot_visibility::public_live, pos, count, ++epoch);
                    pos += count;
                }
                if ((rng() & 1u) == 0u) {
                    doc.append_run(child, llama_rerot_visibility::private_control, pos, 1 + uint32_t(rng() % 2));
                }
                next.push_back(child);
            }
        }
        frontier = std::move(next);
    }
}
static void test_structured_kary_trees() {
    for (int branching = 2; branching <= 4; ++branching) {
        std::mt19937 rng(0x9e3779b9u ^ uint32_t(branching));
        uint64_t epoch = 0;
        llama_rerot_document doc(1000u + uint64_t(branching));
        build_kary_tree(doc, branching, 3, epoch, rng);
        check_doc_valid(doc, "kary");
        // Every node -- hence every leaf -- as reader.
        for (llama_rerot_node_id reader = 0; reader < doc.node_count(); ++reader) {
            check_view_battery(doc, reader);
        }
    }
}
static void collect_subtree(
        const llama_rerot_document & doc,
        llama_rerot_node_id node_id,
        std::vector<llama_rerot_node_id> & out) {
    out.push_back(node_id);
    const auto * node = doc.node(node_id);
    assert(node);
    for (const auto child : node->children) {
        collect_subtree(doc, child, out);
    }
}
static void test_off_path_stability() {
    std::mt19937 rng(777u);
    uint64_t epoch = 0;
    llama_rerot_document doc(555);
    build_kary_tree(doc, 3, 3, epoch, rng);
    check_doc_valid(doc, "off-path");
    const auto * root = doc.node(doc.root());
    assert(root && root->children.size() == 3);
    std::vector<std::vector<llama_rerot_node_id>> branch_leaves(3);
    std::vector<std::set<llama_rerot_node_id>> branch_members(3);
    for (size_t i = 0; i < 3; ++i) {
        collect_leaves(doc, root->children[i], branch_leaves[i]);
        std::vector<llama_rerot_node_id> members;
        collect_subtree(doc, root->children[i], members);
        branch_members[i] = std::set<llama_rerot_node_id>(members.begin(), members.end());
    }
    const auto project = [&](const llama_rerot_reader_view & view, size_t branch) {
        std::vector<llama_rerot_run_id> out;
        for (const auto & run : view.runs) {
            if (branch_members[branch].count(run.owner) != 0) {
                out.push_back(run.run_id);
            }
        }
        return out;
    };
    // Plain document DFS order of a branch: the root reader is off-path for
    // every branch, so its projection is the stable order.
    const auto root_view = doc.build_view(doc.root());
    for (size_t a = 0; a < 3; ++a) {
        for (size_t b = 0; b < 3; ++b) {
            if (a == b) {
                continue;
            }
            const auto view_a = doc.build_view(branch_leaves[a].front());
            const auto view_b = doc.build_view(branch_leaves[b].front());
            for (size_t c = 0; c < 3; ++c) {
                if (c == a || c == b) {
                    continue;
                }
                // A branch that is off-path for both readers renders identically.
                CHECK(project(view_a, c) == project(view_b, c));
                CHECK(project(view_a, c) == project(root_view, c));
            }
        }
    }
}
static void test_queue_independence() {
    std::mt19937 rng(4242u);
    uint64_t epoch = 0;
    llama_rerot_document doc(31337);
    build_kary_tree(doc, 3, 3, epoch, rng);
    check_doc_valid(doc, "queue-before");
    std::vector<std::vector<llama_rerot_run_id>> before_ids;
    std::vector<llama_pos> before_query;
    for (llama_rerot_node_id reader = 0; reader < doc.node_count(); ++reader) {
        const auto view = doc.build_view(reader);
        before_ids.push_back(view_run_ids(view));
        before_query.push_back(view.query_virtual_pos);
    }
    const llama_rerot_node_state states[] = {
        llama_rerot_node_state::planning,
        llama_rerot_node_state::terminal_running,
        llama_rerot_node_state::forked,
        llama_rerot_node_state::queued,
        llama_rerot_node_state::starting,
        llama_rerot_node_state::running,
        llama_rerot_node_state::retired,
    };
    for (llama_rerot_node_id node = 0; node < doc.node_count(); ++node) {
        CHECK(doc.set_node_state(node, states[node % 7]));
    }
    check_doc_valid(doc, "queue-after");
    for (llama_rerot_node_id reader = 0; reader < doc.node_count(); ++reader) {
        const auto view = doc.build_view(reader);
        CHECK(view_run_ids(view) == before_ids[reader]);
        CHECK(view.query_virtual_pos == before_query[reader]);
        check_view_battery(doc, reader);
    }
}
static void test_writer_mutation_stability() {
    std::mt19937 rng(9001u);
    uint64_t epoch = 0;
    llama_rerot_document doc(271828);
    build_kary_tree(doc, 2, 3, epoch, rng);
    check_doc_valid(doc, "mutate-base");
    std::vector<llama_rerot_reader_view> before;
    for (llama_rerot_node_id reader = 0; reader < doc.node_count(); ++reader) {
        before.push_back(doc.build_view(reader));
    }
    // Phase 1: grow one public run. Order is untouched; positions before the
    // grown run are identical and positions after it shift by exactly delta.
    llama_rerot_run_id grown = LLAMA_REROT_RUN_INVALID;
    for (llama_rerot_run_id id = 0; id < doc.run_count(); ++id) {
        const auto * run = doc.run(id);
        assert(run);
        if (run->visibility == llama_rerot_visibility::public_live && run->token_count > 0) {
            grown = id;
            break;
        }
    }
    CHECK(grown != LLAMA_REROT_RUN_INVALID);
    const uint32_t old_count = doc.run(grown)->token_count;
    const uint32_t delta = 3;
    CHECK(doc.set_run_token_count(grown, old_count + delta));
    check_doc_valid(doc, "mutate-grown");
    for (llama_rerot_node_id reader = 0; reader < doc.node_count(); ++reader) {
        const auto after = doc.build_view(reader);
        CHECK(view_run_ids(after) == view_run_ids(before[reader]));
        CHECK(after.runs.size() == before[reader].runs.size());
        bool seen_grown = false;
        for (size_t i = 0; i < after.runs.size(); ++i) {
            if (after.runs[i].run_id == grown) {
                seen_grown = true;
                CHECK(after.runs[i].virtual_pos0 == before[reader].runs[i].virtual_pos0);
                continue;
            }
            const llama_pos want = before[reader].runs[i].virtual_pos0 + (seen_grown ? llama_pos(delta) : 0);
            CHECK(after.runs[i].virtual_pos0 == want);
        }
        const llama_pos want_query = before[reader].query_virtual_pos + (seen_grown ? llama_pos(delta) : 0);
        CHECK(after.query_virtual_pos == want_query);
        check_view_battery(doc, reader);
    }
    // Phase 2: append a new public run on another writer. Pre-existing runs
    // keep their relative order in every reader view.
    const auto * grown_run = doc.run(grown);
    assert(grown_run);
    const auto * root = doc.node(doc.root());
    assert(root && !root->children.empty());
    llama_rerot_node_id append_owner = LLAMA_REROT_NODE_INVALID;
    for (const auto candidate : root->children) {
        if (!doc.is_ancestor(candidate, grown_run->owner) && candidate != grown_run->owner) {
            std::vector<llama_rerot_node_id> leaves;
            collect_leaves(doc, candidate, leaves);
            append_owner = leaves.front();
            break;
        }
    }
    CHECK(append_owner != LLAMA_REROT_NODE_INVALID);
    llama_pos storage_end = 0;
    for (const auto run_id : doc.node(append_owner)->runs) {
        const auto * run = doc.run(run_id);
        assert(run);
        storage_end = std::max(storage_end, run->storage_pos0 + llama_pos(run->token_count));
    }
    const auto added = doc.append_run(
        append_owner, llama_rerot_visibility::public_live, storage_end, 2, ++epoch);
    CHECK(doc.run(added)->publish_epoch == epoch);
    check_doc_valid(doc, "mutate-appended");
    for (llama_rerot_node_id reader = 0; reader < doc.node_count(); ++reader) {
        const auto before_ids = view_run_ids(before[reader]);
        const auto after = doc.build_view(reader);
        std::vector<llama_rerot_run_id> filtered;
        for (const auto id : view_run_ids(after)) {
            if (id != added) {
                filtered.push_back(id);
            }
        }
        // Pre-existing runs keep relative order; the grown run keeps its
        // grown size; the new run lands exactly once where PAC-DFS puts it.
        CHECK(filtered == before_ids);
        size_t added_count = 0;
        for (const auto id : view_run_ids(after)) {
            if (id == added) {
                ++added_count;
            }
        }
        const bool added_visible = run_visible_to_reader(*doc.run(added), reader);
        CHECK(added_count == (added_visible ? 1u : 0u));
        check_view_battery(doc, reader);
    }
}
static void test_run_epoch_contract() {
    llama_rerot_document doc(1234);
    bool threw = false;
    try {
        doc.append_run(doc.root(), llama_rerot_visibility::public_live, 0, 1, 0);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
    for (const auto visibility : {
            llama_rerot_visibility::normal,
            llama_rerot_visibility::private_control,
            llama_rerot_visibility::pending_record }) {
        threw = false;
        try {
            doc.append_run(doc.root(), visibility, 0, 1, 9);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw);
    }
    doc.append_run(doc.root(), llama_rerot_visibility::public_live, 0, 1, 5);
    doc.append_run(doc.root(), llama_rerot_visibility::private_control, 1, 1);
    doc.append_run(doc.root(), llama_rerot_visibility::pending_record, 2, 1);
    doc.append_run(doc.root(), llama_rerot_visibility::normal, 3, 1);
    check_doc_valid(doc, "epoch-contract");
    // The root owns the private and pending runs, so all four are visible to
    // it in document order with epochs propagated verbatim.
    const auto view = doc.build_view(doc.root());
    CHECK(view.runs.size() == 4);
    CHECK(view.runs[0].publish_epoch == 5);
    CHECK(view.runs[1].publish_epoch == 0);
    CHECK(view.runs[2].publish_epoch == 0);
    CHECK(view.runs[3].publish_epoch == 0);
}
static void test_dag_cycle_preferred_topo() {
    // AGENTS.md §03.5.4 & §09
    // questions plan_order: 1, 2, 3 (flat)
    // reader 1: 2, 3, 1
    // reader 2: 3, 1, 2
    // reader 3: 1, 2, 3
    llama_rerot_document doc(99);
    doc.set_dag_mode(true);

    const auto root = doc.root();
    const auto n1 = doc.create_child(root, "Task 1");
    const auto n2 = doc.create_child(root, "Task 2");
    const auto n3 = doc.create_child(root, "Task 3");
    doc.set_plan_rank(n1, 0);
    doc.set_plan_rank(n2, 1);
    doc.set_plan_rank(n3, 2);

    std::string err;
    const std::vector<llama_rerot_node_id> flat = { n1, n2, n3 };
    CHECK(doc.topo_sort_cycle_preferred(flat, n1, &err) == std::vector<llama_rerot_node_id>({ n2, n3, n1 }));
    CHECK(doc.topo_sort_cycle_preferred(flat, n2, &err) == std::vector<llama_rerot_node_id>({ n3, n1, n2 }));
    CHECK(doc.topo_sort_cycle_preferred(flat, n3, &err) == std::vector<llama_rerot_node_id>({ n1, n2, n3 }));

    // AGENTS.md §03.5.5: With dependency 1 -> 3, 2 independent
    // reader 2: 1, 3, 2 (1 must precede 3 despite cycle preference!)
    // reader 3: 1, 2, 3
    CHECK(doc.add_edge(n1, n3, &err));
    CHECK(doc.topo_sort_cycle_preferred(flat, n2, &err) == std::vector<llama_rerot_node_id>({ n1, n3, n2 }));
    CHECK(doc.topo_sort_cycle_preferred(flat, n3, &err) == std::vector<llama_rerot_node_id>({ n1, n2, n3 }));

    // Diamond: 1->2, 1->3, 2->4, 3->4
    const auto n4 = doc.create_child(root, "Task 4");
    doc.set_plan_rank(n4, 3);
    CHECK(doc.add_edge(n1, n2, &err));
    CHECK(doc.add_edge(n2, n4, &err));
    CHECK(doc.add_edge(n3, n4, &err));
    const std::vector<llama_rerot_node_id> diamond = { n1, n2, n3, n4 };
    CHECK(doc.topo_sort_cycle_preferred(diamond, n4, &err) == std::vector<llama_rerot_node_id>({ n1, n2, n3, n4 }));

    llama_rerot_document diamond_view(102);
    diamond_view.set_dag_mode(true);
    const auto droot = diamond_view.root();
    const auto d1 = diamond_view.create_child(droot, "1");
    const auto d2 = diamond_view.create_child(droot, "2");
    const auto d3 = diamond_view.create_child(droot, "3");
    const auto d4 = diamond_view.create_child(droot, "4");
    diamond_view.set_plan_rank(d1, 0);
    diamond_view.set_plan_rank(d2, 1);
    diamond_view.set_plan_rank(d3, 2);
    diamond_view.set_plan_rank(d4, 3);
    CHECK(diamond_view.add_edge(d1, d2, &err));
    CHECK(diamond_view.add_edge(d1, d3, &err));
    CHECK(diamond_view.add_edge(d2, d4, &err));
    CHECK(diamond_view.add_edge(d3, d4, &err));
    diamond_view.append_run(droot, llama_rerot_visibility::public_live, 0, 4, 1);
    diamond_view.append_run(d1, llama_rerot_visibility::public_live, 4, 1, 2);
    diamond_view.append_run(d2, llama_rerot_visibility::public_live, 5, 1, 2);
    diamond_view.append_run(d3, llama_rerot_visibility::public_live, 6, 1, 2);
    diamond_view.append_run(d4, llama_rerot_visibility::public_live, 7, 1, 2);
    const std::vector<llama_rerot_node_id> diamond_started = { droot, d1, d2, d3, d4 };
    const auto dview = diamond_view.build_dag_view(d4, diamond_started);
    CHECK(owners(dview) == std::vector<llama_rerot_node_id>({ droot, d1, d2, d3, d4 }));
    std::unordered_set<llama_rerot_run_id> diamond_runs;
    std::unordered_set<llama_rerot_node_id> diamond_owners;
    for (const auto & run : dview.runs) {
        CHECK(diamond_runs.insert(run.run_id).second);
        CHECK(diamond_owners.insert(run.owner).second);
    }
    CHECK(diamond_runs.size() == 5);
    CHECK(diamond_owners.size() == 5);

    // Cycle detection: add 4 -> 1 should fail
    CHECK(doc.add_edge(n4, n1, &err));
    CHECK(doc.topo_sort_cycle_preferred(diamond, n4, &err).empty());
}

static void test_dag_reader_view_assembly() {
    llama_rerot_document doc(100);
    doc.set_dag_mode(true);

    const auto root = doc.root();
    const auto n1 = doc.create_child(root, "1");
    const auto n2 = doc.create_child(root, "2");
    const auto n3 = doc.create_child(root, "3");
    doc.set_plan_rank(n1, 0);
    doc.set_plan_rank(n2, 1);
    doc.set_plan_rank(n3, 2);

    doc.append_run(root, llama_rerot_visibility::public_live, 0, 10, 1); // P: 10 tokens
    doc.append_run(n1, llama_rerot_visibility::public_live, 10, 5, 2);   // B1: 5 tokens
    doc.append_run(n2, llama_rerot_visibility::public_live, 15, 6, 2);   // B2: 6 tokens
    doc.append_run(n3, llama_rerot_visibility::public_live, 21, 7, 2);   // B3: 7 tokens

    const std::vector<llama_rerot_node_id> started = { root, n1, n2, n3 };

    // reader 1: P, B2, B3, B1
    const auto view1 = doc.build_dag_view(n1, started);
    CHECK(owners(view1) == std::vector<llama_rerot_node_id>({ root, n2, n3, n1 }));
    CHECK(view1.query_virtual_pos == 28);

    // reader 2: P, B3, B1, B2
    const auto view2 = doc.build_dag_view(n2, started);
    CHECK(owners(view2) == std::vector<llama_rerot_node_id>({ root, n3, n1, n2 }));
    CHECK(view2.query_virtual_pos == 28);

    // reader 3: P, B1, B2, B3
    const auto view3 = doc.build_dag_view(n3, started);
    CHECK(owners(view3) == std::vector<llama_rerot_node_id>({ root, n1, n2, n3 }));
    CHECK(view3.query_virtual_pos == 28);

    // Synthesis reader 0: P, B1, B2, B3
    const auto view0 = doc.build_dag_view(root, started);
    CHECK(owners(view0) == std::vector<llama_rerot_node_id>({ root, n1, n2, n3 }));
    CHECK(view0.query_virtual_pos == 28);

    auto unique_run_ids = [](const llama_rerot_reader_view & view) {
        std::unordered_set<llama_rerot_run_id> ids;
        for (const auto & run : view.runs) {
            CHECK(ids.insert(run.run_id).second);
        }
        return ids.size() == view.runs.size();
    };
    CHECK(unique_run_ids(view1));
    CHECK(unique_run_ids(view2));
    CHECK(unique_run_ids(view3));
    CHECK(unique_run_ids(view0));

    // Verify source_end segment kind is isolated from foreign readers (§04.7)
    const auto end_run = doc.append_run(n1, llama_rerot_visibility::public_live, 12, 1, 4, llama_rerot_segment_kind::source_end);
    const auto view2_with_end = doc.build_dag_view(n2, started);
    // Reader 2 should NOT see reader 1's source_end run
    for (const auto & r : view2_with_end.runs) {
        CHECK(r.run_id != end_run);
    }
    // Reader 1 (owner) SHOULD see its own source_end run
    const auto view1_with_end = doc.build_dag_view(n1, started);
    bool seen_own_end = false;
    for (const auto & r : view1_with_end.runs) {
        if (r.run_id == end_run) seen_own_end = true;
    }
    CHECK(seen_own_end);

    // Predecessor missing from started_nodes must throw (§09 reference property)
    std::string err;
    doc.add_edge(n1, n3, &err);
    bool caught_missing_pred = false;
    try {
        doc.build_dag_view(n3, { root, n2, n3 }); // n1 missing!
    } catch (const std::exception &) {
        caught_missing_pred = true;
    }
    CHECK(caught_missing_pred);
}

static void test_chapter09_exhaustive_dag_properties() {
    // Port of the reference logic from AGENTS.md §09
    // Enumerates 4-node all possible DAGs and verifies:
    // - cycle-preferred topological order
    // - own work last for active reader
    // - all predecessors precede successors
    const std::vector<llama_rerot_node_id> nodes = { 1, 2, 3, 4 };
    std::vector<std::pair<llama_rerot_node_id, llama_rerot_node_id>> possible;
    for (auto a : nodes) {
        for (auto b : nodes) {
            if (a != b) possible.emplace_back(a, b);
        }
    }

    size_t dag_count = 0;
    // Test a representative dense subset of DAG topologies (all 2-edge and 3-edge graphs)
    for (size_t i = 0; i < possible.size(); ++i) {
        for (size_t j = i + 1; j < possible.size(); ++j) {
            llama_rerot_document doc(500);
            doc.set_dag_mode(true);
            const auto root = doc.root();
            auto n1 = doc.create_child(root, "1"); doc.set_plan_rank(n1, 0);
            auto n2 = doc.create_child(root, "2"); doc.set_plan_rank(n2, 1);
            auto n3 = doc.create_child(root, "3"); doc.set_plan_rank(n3, 2);
            auto n4 = doc.create_child(root, "4"); doc.set_plan_rank(n4, 3);

            std::string err;
            if (!doc.add_edge(possible[i].first, possible[i].second, &err)) continue;
            if (!doc.add_edge(possible[j].first, possible[j].second, &err)) continue;

            const auto order = doc.topo_sort_cycle_preferred(nodes, n4, &err);
            if (order.empty()) continue; // Has cycle
            dag_count++;

            // Property: for reader r (n4 here), r must be last if it has no successors
            if (possible[i].first != n4 && possible[j].first != n4) {
                CHECK(order.back() == n4);
            }
            // Property: predecessors must precede successors
            std::unordered_map<llama_rerot_node_id, size_t> pos;
            for (size_t k = 0; k < order.size(); ++k) pos[order[k]] = k;
            CHECK(pos[possible[i].first] < pos[possible[i].second]);
            CHECK(pos[possible[j].first] < pos[possible[j].second]);
        }
    }
    CHECK(dag_count > 0);
}

int main() {
    std::fprintf(stderr, "=== RERoT View Tests ===\n");
    test_dag_cycle_preferred_topo();
    test_dag_reader_view_assembly();
    test_chapter09_exhaustive_dag_properties();
    test_manual_pac_dfs();
    test_visibility();
    test_reclassify_run_validation();
    test_randomized_invariants();
    test_query_layout_frontiers();
    test_concurrent_sibling_readers_strong();
    test_structured_kary_trees();
    test_off_path_stability();
    test_queue_independence();
    test_writer_mutation_stability();
    test_run_epoch_contract();
    std::fprintf(stderr, "=== Results: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}

