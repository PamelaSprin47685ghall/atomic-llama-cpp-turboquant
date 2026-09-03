#include "server-rerot.h"

#include <cstdio>
#include <string>

static int g_failures = 0;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        ++g_failures; \
    } \
} while (0)

static bool commit_private(
        server_rerot_runtime & runtime,
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        llama_pos pos) {
    const auto plan = runtime.plan_private_token(episode_id, node_id, pos);
    return plan.has_value() && runtime.commit_token(episode_id, node_id, *plan);
}

static bool commit_generated(
        server_rerot_runtime & runtime,
        uint64_t episode_id,
        llama_rerot_node_id node_id,
        llama_pos pos,
        const std::string & bytes) {
    const auto plan = runtime.plan_generated_token(episode_id, node_id, pos, bytes);
    return plan.has_value() && runtime.commit_token(episode_id, node_id, *plan);
}

static void start_child(
        server_rerot_runtime & runtime,
        uint64_t episode_id,
        int slot,
        llama_rerot_node_id expected_node) {
    llama_rerot_node_id admitted = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(episode_id, slot, slot, &admitted));
    CHECK(admitted == expected_node);

    auto * node = runtime.node(episode_id, admitted);
    CHECK(node != nullptr);
    if (!node) {
        return;
    }

    const auto heading = runtime.plan_heading_token(
        episode_id, admitted, node->storage_pos_next);
    CHECK(heading.has_value());
    if (!heading) {
        return;
    }
    CHECK(runtime.commit_token(episode_id, admitted, *heading));
    CHECK(runtime.publish_heading(episode_id, admitted, heading->run_id));

    node = runtime.node(episode_id, admitted);
    CHECK(node != nullptr);
    if (node) {
        CHECK(commit_private(runtime, episode_id, admitted, node->storage_pos_next));
    }
    CHECK(runtime.complete_admission(episode_id, admitted));
}

static void make_terminal(
        server_rerot_runtime & runtime,
        uint64_t episode_id,
        llama_rerot_node_id node_id) {
    auto * node = runtime.node(episode_id, node_id);
    CHECK(node != nullptr);
    if (!node) {
        return;
    }
    CHECK(commit_generated(runtime, episode_id, node_id, node->storage_pos_next,
        "<ol><li>Lane 1: directly solve this section</li></ol>"));
    const auto * episode = runtime.episode(episode_id);
    const auto * logical = episode ? episode->document.node(node_id) : nullptr;
    CHECK(logical != nullptr);
    CHECK(logical && logical->state == llama_rerot_node_state::terminal_running);
}

static void request_exit(
        server_rerot_runtime & runtime,
        uint64_t episode_id,
        llama_rerot_node_id node_id) {
    auto * node = runtime.node(episode_id, node_id);
    CHECK(node != nullptr);
    if (!node) {
        return;
    }
    CHECK(commit_generated(runtime, episode_id, node_id, node->storage_pos_next,
        "final observation </thi"));
    node = runtime.node(episode_id, node_id);
    CHECK(node != nullptr);
    if (node) {
        CHECK(commit_generated(runtime, episode_id, node_id, node->storage_pos_next, "nk>"));
        CHECK(node->exit_intent);
    }
}

static void test_recursive_queue_and_last_survivor() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 32);
    const uint64_t episode_id = runtime.adopt_root(100, 100, 0, 0, 10);
    CHECK(episode_id != 0);

    CHECK(commit_private(runtime, episode_id, 0, 10));
    CHECK(commit_generated(runtime, episode_id, 0, 11, "Shared scale analysis. "));
    CHECK(commit_generated(runtime, episode_id, 0, 12,
        "<ol><li>Lane 1: Algebra & invariants</li><li>Lane 2：Systems path</li></ol>"));

    auto frontier = runtime.finish_frontier(episode_id);
    CHECK(frontier.forked.size() == 1 && frontier.forked[0] == 0);
    CHECK(!frontier.natural_final());
    CHECK(runtime.freeze_fork_parent(episode_id, 0));
    CHECK(runtime.has_ready_nodes(episode_id));

    const auto * episode = runtime.episode(episode_id);
    CHECK(episode != nullptr);
    CHECK(episode && episode->document.node_count() == 3);
    CHECK(runtime.heading_text(episode_id, 1) == "<h1>Algebra &amp; invariants</h1>\n");
    CHECK(runtime.heading_text(episode_id, 2) == "<h1>Systems path</h1>\n");

    start_child(runtime, episode_id, 0, 1);
    make_terminal(runtime, episode_id, 1);
    request_exit(runtime, episode_id, 1);
    frontier = runtime.finish_frontier(episode_id);
    CHECK(frontier.retired.size() == 1 && frontier.retired[0] == 1);
    CHECK(frontier.released_slots.size() == 1 && frontier.released_slots[0] == 0);
    CHECK(!frontier.natural_final());
    const auto * retired_first = runtime.node(episode_id, 1);
    CHECK(retired_first != nullptr);
    CHECK(retired_first && retired_first->physical_slot == -1);
    CHECK(retired_first && retired_first->exec_seq == -1);

    start_child(runtime, episode_id, 0, 2);
    make_terminal(runtime, episode_id, 2);
    request_exit(runtime, episode_id, 2);
    frontier = runtime.finish_frontier(episode_id);
    CHECK(frontier.natural_final());
    CHECK(frontier.final_node == 2);
    CHECK(frontier.retired.empty());
    CHECK(frontier.released_slots.empty());
    episode = runtime.episode(episode_id);
    CHECK(episode && episode->finalizing);
    CHECK(episode && episode->running.size() == 1 && episode->running.count(2) == 1);
    const auto * final_lane = runtime.node(episode_id, 2);
    CHECK(final_lane != nullptr);
    CHECK(final_lane && final_lane->physical_slot == 0);
    CHECK(final_lane && final_lane->exec_seq == 0);
    CHECK(final_lane && final_lane->exit_intent);
    const auto * final_doc = episode ? episode->document.node(2) : nullptr;
    CHECK(final_doc != nullptr);
    CHECK(final_doc && final_doc->state == llama_rerot_node_state::terminal_running);

    CHECK(runtime.erase_episode(episode_id));
    CHECK(runtime.episode(episode_id) == nullptr);
}

static void test_same_frontier_exit_tie_break() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 4, 20);
    const uint64_t episode_id = runtime.adopt_root(200, 200, 0, 0, 0);
    CHECK(episode_id != 0);
    CHECK(commit_generated(runtime, episode_id, 0, 0,
        "<ol><li>Lane 1: First</li><li>Lane 2: Second</li></ol>"));
    auto frontier = runtime.finish_frontier(episode_id);
    CHECK(frontier.forked.size() == 1);
    CHECK(runtime.freeze_fork_parent(episode_id, 0));

    start_child(runtime, episode_id, 0, 1);
    start_child(runtime, episode_id, 1, 2);
    make_terminal(runtime, episode_id, 1);
    make_terminal(runtime, episode_id, 2);
    request_exit(runtime, episode_id, 2); // commit in reverse physical order
    request_exit(runtime, episode_id, 1);

    frontier = runtime.finish_frontier(episode_id);
    CHECK(frontier.retired.size() == 1);
    CHECK(frontier.retired[0] == 1);
    CHECK(frontier.released_slots.size() == 1 && frontier.released_slots[0] == 0);
    CHECK(frontier.final_node == 2);
    CHECK(frontier.natural_final());

    const auto * episode = runtime.episode(episode_id);
    CHECK(episode != nullptr);
    CHECK(episode && episode->running.size() == 1 && episode->running.count(2) == 1);
    const auto * first = runtime.node(episode_id, 1);
    const auto * second = runtime.node(episode_id, 2);
    CHECK(first && first->physical_slot == -1 && first->exec_seq == -1);
    CHECK(second && second->physical_slot == 1 && second->exec_seq == 1);
    CHECK(second && second->exit_intent);
    CHECK(runtime.erase_episode(episode_id));
}

static void test_internal_seq_exhaustion_aborts_whole_episode() {
    // One archive id plus two child parked ids are required, but only two ids
    // are available. This is a hard global-resource failure, not truncation of
    // the public plan.
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 10);
    const uint64_t episode_id = runtime.adopt_root(300, 300, 0, 0, 0);
    CHECK(commit_generated(runtime, episode_id, 0, 0,
        "<ol><li>A</li><li>B</li></ol>"));
    runtime.finish_frontier(episode_id);
    CHECK(!runtime.freeze_fork_parent(episode_id, 0));
    const auto * episode = runtime.episode(episode_id);
    CHECK(episode && episode->hard_aborted);
    CHECK(episode && !episode->abort_reason.empty());
    CHECK(runtime.erase_episode(episode_id));
}

int main() {
    std::fprintf(stderr, "=== RERoT Runtime Tests ===\n");
    test_recursive_queue_and_last_survivor();
    test_same_frontier_exit_tie_break();
    test_internal_seq_exhaustion_aborts_whole_episode();
    std::fprintf(stderr, "=== Results: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
