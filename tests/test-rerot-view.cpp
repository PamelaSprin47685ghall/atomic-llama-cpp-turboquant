#include "llama-rerot.h"

#include <algorithm>
#include <cassert>
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

int main() {
    std::fprintf(stderr, "=== RERoT View Tests ===\n");
    test_manual_pac_dfs();
    test_visibility();
    test_randomized_invariants();
    std::fprintf(stderr, "=== Results: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}

