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
    CHECK(strong.query_virtual_pos == 5);
    CHECK(strong.entries.size() == 6);
    CHECK(strong.groups.size() == 2);
    CHECK(strong.groups[0].effective_pos == 3);
    CHECK(strong.groups[1].effective_pos == 5);

    std::set<uint32_t> strong_keys;
    for (const auto & entry : strong.entries) {
        strong_keys.insert(entry.key_index);
    }
    CHECK(strong_keys == std::set<uint32_t>({0, 1, 2, 3, 4, 5}));

    reader.frontier_mode = LLAMA_REROT_FRONTIER_LAG1;
    const auto lag1 = llama_rerot_build_query_layout(reader, 3, keys);
    CHECK(lag1.query_virtual_pos == 4);
    CHECK(lag1.entries.size() == 5);
    CHECK(lag1.groups.size() == 2);
    CHECK(lag1.groups[0].effective_pos == 3);
    CHECK(lag1.groups[1].effective_pos == 4);

    std::set<uint32_t> lag1_keys;
    for (const auto & entry : lag1.entries) {
        lag1_keys.insert(entry.key_index);
    }
    CHECK(lag1_keys == std::set<uint32_t>({0, 1, 2, 4, 5}));
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
int main() {
    std::fprintf(stderr, "=== RERoT View Tests ===\n");
    test_manual_pac_dfs();
    test_visibility();
    test_reclassify_run_validation();
    test_randomized_invariants();
    test_query_layout_frontiers();
    test_structured_kary_trees();
    test_off_path_stability();
    test_queue_independence();
    test_writer_mutation_stability();
    test_run_epoch_contract();
    std::fprintf(stderr, "=== Results: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}

