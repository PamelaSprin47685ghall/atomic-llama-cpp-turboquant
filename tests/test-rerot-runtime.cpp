#include "server-rerot.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

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
    CHECK(heading->is_heading);
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

static bool view_contains_run(const llama_rerot_reader_view & view, llama_rerot_run_id run_id) {
    for (const auto & entry : view.runs) {
        if (entry.run_id == run_id) {
            return true;
        }
    }
    return false;
}

static void test_n1_no_fork_disarm_forever() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t episode_id = runtime.adopt_root(100, 101, 0, 0, 10);
    CHECK(episode_id != 0);
    CHECK(runtime.response_task_id(episode_id) == 101);

    CHECK(commit_private(runtime, episode_id, 0, 10));
    CHECK(commit_generated(runtime, episode_id, 0, 11, "Scale analysis. "));

    auto * node = runtime.node(episode_id, 0);
    CHECK(node != nullptr && node->planner_armed);
    CHECK(commit_generated(runtime, episode_id, 0, 12, "<ol><li>Only Lane</li></ol>"));

    // N=1: atomic publish, no fork, planner disarmed forever for this node.
    node = runtime.node(episode_id, 0);
    const auto * episode = runtime.episode(episode_id);
    CHECK(node != nullptr && !node->planner_armed);
    CHECK(node != nullptr && !node->pending_record.has_value());
    CHECK(episode && episode->document.node_count() == 1);
    const auto * logical = episode ? episode->document.node(0) : nullptr;
    CHECK(logical && logical->state == llama_rerot_node_state::terminal_running);

    auto frontier = runtime.finish_frontier(episode_id);
    CHECK(frontier.forked.empty());
    CHECK(!frontier.natural_final());
    CHECK(!frontier.hard_aborted);

    // Later planner-shaped text is ordinary body content, never a re-fork.
    node = runtime.node(episode_id, 0);
    CHECK(commit_generated(runtime, episode_id, 0, node->storage_pos_next, "Body mentions "));
    node = runtime.node(episode_id, 0);
    CHECK(commit_generated(runtime, episode_id, 0, node->storage_pos_next, "<ol><li>Again</li></ol>"));
    node = runtime.node(episode_id, 0);
    CHECK(node != nullptr && !node->planner_armed);
    episode = runtime.episode(episode_id);
    CHECK(episode && episode->document.node_count() == 1);
    CHECK(episode && episode->ready_queue.empty());

    // Single RUNNING Lane with empty queue/starting exits into natural final,
    // then fence refresh and serial tail transition keep the root response.
    request_exit(runtime, episode_id, 0);
    frontier = runtime.finish_frontier(episode_id);
    CHECK(frontier.natural_final());
    CHECK(frontier.final_node == 0);
    CHECK(frontier.retired.empty());

    std::vector<uint32_t> fence_runs;
    CHECK(runtime.refresh_final_fence(episode_id, 0, &fence_runs));
    CHECK(!fence_runs.empty());
    CHECK(runtime.complete_serial_tail(episode_id, 0));
    episode = runtime.episode(episode_id);
    CHECK(episode && episode->serial_tail && episode->serial_node == 0);
    CHECK(runtime.response_task_id(episode_id) == 101);
    CHECK(runtime.erase_episode(episode_id));
}

static void test_n2_strong_uptake() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t episode_id = runtime.adopt_root(110, 110, 0, 0, 0);
    CHECK(commit_generated(runtime, episode_id, 0, 0,
        "<ol><li>Alpha</li><li>Beta</li></ol>"));
    CHECK(runtime.finish_frontier(episode_id).forked.size() == 1);
    CHECK(runtime.freeze_fork_parent(episode_id, 0));

    start_child(runtime, episode_id, 0, 1);
    start_child(runtime, episode_id, 1, 2);

    // Lane 1 publishes one PUBLIC finding plus PRIVATE control state.
    auto * lane1 = runtime.node(episode_id, 1);
    CHECK(lane1 != nullptr);
    const llama_pos pos = lane1 ? lane1->storage_pos_next : 0;
    CHECK(commit_generated(runtime, episode_id, 1, pos, "result ALPHA "));
    lane1 = runtime.node(episode_id, 1);
    CHECK(lane1 && lane1->public_run.has_value() && lane1->private_run.has_value());
    const auto public_run = lane1 ? *lane1->public_run : LLAMA_REROT_RUN_INVALID;
    const auto private_run = lane1 ? *lane1->private_run : LLAMA_REROT_RUN_INVALID;

    // The sibling PAC-DFS view takes up PUBLIC lexical material immediately
    // (strong frontier) but never the foreign PRIVATE control run.
    const auto * episode = runtime.episode(episode_id);
    CHECK(episode != nullptr);
    const auto view2 = episode ? episode->document.build_view(2) : llama_rerot_reader_view{};
    CHECK(view_contains_run(view2, public_run));
    CHECK(!view_contains_run(view2, private_run));
    const auto view1 = episode ? episode->document.build_view(1) : llama_rerot_reader_view{};
    CHECK(view_contains_run(view1, private_run));

    CHECK(runtime.erase_episode(episode_id));
}

static void test_queue_fifo_five_children_one_slot() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t episode_id = runtime.adopt_root(120, 120, 0, 0, 0);
    CHECK(commit_generated(runtime, episode_id, 0, 0,
        "<ol><li>T1</li><li>T2</li><li>T3</li><li>T4</li><li>T5</li></ol>"));
    CHECK(runtime.finish_frontier(episode_id).forked.size() == 1);
    CHECK(runtime.freeze_fork_parent(episode_id, 0));

    const auto * episode = runtime.episode(episode_id);
    CHECK(episode && episode->document.node_count() == 6);
    CHECK(episode && episode->ready_queue.size() == 5 && episode->queue_peak == 5);
    CHECK(runtime.parent_has_unadmitted_children(episode_id, 0));

    // One physical slot: every admission follows <li> order with no scoring.
    for (llama_rerot_node_id expected = 1; expected <= 5; ++expected) {
        start_child(runtime, episode_id, 0, expected);
        make_terminal(runtime, episode_id, expected);
        request_exit(runtime, episode_id, expected);
        const auto frontier = runtime.finish_frontier(episode_id);
        if (expected < 5) {
            CHECK(!frontier.natural_final());
            CHECK(frontier.retired.size() == 1 && frontier.retired[0] == expected);
            CHECK(frontier.released_slots.size() == 1 && frontier.released_slots[0] == 0);
        } else {
            CHECK(frontier.natural_final());
            CHECK(frontier.final_node == expected);
            CHECK(frontier.retired.empty());
        }
    }
    CHECK(!runtime.parent_has_unadmitted_children(episode_id, 0));
    CHECK(runtime.erase_episode(episode_id));
}

static void test_nested_fork_keeps_fifo() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t episode_id = runtime.adopt_root(130, 130, 0, 0, 0);
    CHECK(commit_generated(runtime, episode_id, 0, 0,
        "<ol><li>A-branch</li><li>B-branch</li></ol>"));
    CHECK(runtime.finish_frontier(episode_id).forked.size() == 1);
    CHECK(runtime.freeze_fork_parent(episode_id, 0));

    // Admit A only; B stays queued at the older enqueue frontier.
    start_child(runtime, episode_id, 0, 1);

    // A recursively forks two grandchildren while B is still queued.
    auto * lane_a = runtime.node(episode_id, 1);
    CHECK(lane_a != nullptr);
    CHECK(commit_generated(runtime, episode_id, 1, lane_a->storage_pos_next,
        "<ol><li>A1</li><li>A2</li></ol>"));
    const auto * episode = runtime.episode(episode_id);
    const auto * logical_a = episode ? episode->document.node(1) : nullptr;
    CHECK(logical_a && logical_a->state == llama_rerot_node_state::forked);
    CHECK(runtime.freeze_fork_parent(episode_id, 1));

    // Later-enqueued grandchildren must not overtake the older sibling B:
    // key order is (enqueue_frontier, tree_path).
    std::vector<llama_rerot_node_id> order;
    for (int i = 0; i < 3; ++i) {
        llama_rerot_node_id admitted = LLAMA_REROT_NODE_INVALID;
        CHECK(runtime.admit_next_child(episode_id, 0, 0, &admitted));
        order.push_back(admitted);
        auto * node = runtime.node(episode_id, admitted);
        const auto heading = runtime.plan_heading_token(episode_id, admitted, node->storage_pos_next);
        CHECK(heading.has_value());
        CHECK(runtime.commit_token(episode_id, admitted, *heading));
        CHECK(runtime.publish_heading(episode_id, admitted, heading->run_id));
        node = runtime.node(episode_id, admitted);
        CHECK(commit_private(runtime, episode_id, admitted, node->storage_pos_next));
        CHECK(runtime.complete_admission(episode_id, admitted));
        make_terminal(runtime, episode_id, admitted);
        request_exit(runtime, episode_id, admitted);
        const auto frontier = runtime.finish_frontier(episode_id);
        if (i < 2) {
            CHECK(!frontier.natural_final());
            CHECK(frontier.retired.size() == 1 && frontier.retired[0] == admitted);
        } else {
            CHECK(frontier.natural_final());
            CHECK(frontier.final_node == admitted);
        }
    }
    CHECK(order == std::vector<llama_rerot_node_id>({2, 3, 4}));
    CHECK(runtime.erase_episode(episode_id));
}

static void test_pending_invisible_until_atomic_publish() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t episode_id = runtime.adopt_root(140, 140, 0, 0, 0);
    CHECK(commit_generated(runtime, episode_id, 0, 0,
        "<ol><li>A</li><li>B</li></ol>"));
    CHECK(runtime.finish_frontier(episode_id).forked.size() == 1);
    CHECK(runtime.freeze_fork_parent(episode_id, 0));
    start_child(runtime, episode_id, 0, 1);
    start_child(runtime, episode_id, 1, 2);

    // Lane 1 opens a nested planner record but has not closed it: PENDING.
    auto * lane1 = runtime.node(episode_id, 1);
    CHECK(commit_generated(runtime, episode_id, 1, lane1->storage_pos_next, "intro <ol><li>sub"));
    lane1 = runtime.node(episode_id, 1);
    CHECK(lane1 && lane1->pending_record.has_value());
    const auto pending_run = *lane1->pending_record;

    const auto * episode = runtime.episode(episode_id);
    const auto own_view = episode->document.build_view(1);
    const auto foreign_view = episode->document.build_view(2);
    CHECK(view_contains_run(own_view, pending_run));
    CHECK(!view_contains_run(foreign_view, pending_run));

    // Atomic </ol> publish flips the same run PUBLIC for every reader at once.
    lane1 = runtime.node(episode_id, 1);
    CHECK(commit_generated(runtime, episode_id, 1, lane1->storage_pos_next, "task</li></ol>"));
    lane1 = runtime.node(episode_id, 1);
    CHECK(lane1 && !lane1->pending_record.has_value() && !lane1->planner_armed);
    episode = runtime.episode(episode_id);
    const auto foreign_after = episode->document.build_view(2);
    CHECK(view_contains_run(foreign_after, pending_run));
    const auto * run = episode->document.run(pending_run);
    CHECK(run && run->visibility == llama_rerot_visibility::public_live);

    CHECK(runtime.erase_episode(episode_id));
}

static void test_lag1_delays_same_frontier_peer() {
    // Same-frontier peer visibility is the only strong/lag1 difference and it
    // lives in the pure query-layout builder the reader views feed.
    const uint64_t episode_id = 7;
    const llama_rerot_run_id k_peer_run = 9;
    const llama_rerot_run_id k_own_run = 10;

    auto make_reader = [&](llama_rerot_frontier_mode mode, uint64_t frontier) {
        llama_rerot_reader_state reader;
        reader.episode_id = episode_id;
        reader.reader = 2;
        reader.query_run = k_own_run;
        reader.frontier = frontier;
        reader.frontier_mode = mode;
        reader.ordered_runs = {k_peer_run, k_own_run};
        return reader;
    };
    auto make_key = [&](uint32_t index, llama_rerot_node_id owner, llama_rerot_run_id run) {
        llama_rerot_key_record key;
        key.key_index = index;
        key.storage_pos = 0;
        key.owned_by_reader = true;
        key.meta.episode_id = episode_id;
        key.meta.node_id = owner;
        key.meta.run_id = run;
        key.meta.publish_epoch = 3;
        key.meta.frontier = 5;
        key.meta.visibility = llama_rerot_visibility::public_live;
        return key;
    };
    const std::vector<llama_rerot_key_record> keys = {
        make_key(0, 1, k_peer_run),
        make_key(1, 2, k_own_run),
    };

    const auto strong = llama_rerot_build_query_layout(
        make_reader(LLAMA_REROT_FRONTIER_STRONG, 5), 0, keys);
    CHECK(strong.entries.size() == 2);

    const auto lag_same = llama_rerot_build_query_layout(
        make_reader(LLAMA_REROT_FRONTIER_LAG1, 5), 0, keys);
    CHECK(lag_same.entries.size() == 1);

    // Next frontier the lag1 reader takes up the same peer write.
    const auto lag_next = llama_rerot_build_query_layout(
        make_reader(LLAMA_REROT_FRONTIER_LAG1, 6), 0, keys);
    CHECK(lag_next.entries.size() == 2);

    std::string error;
    CHECK(strong.query_virtual_pos == 1);

    // The runtime threads the configured mode into every installed view.
    server_rerot_runtime strong_rt(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    server_rerot_runtime lag_rt(nullptr, LLAMA_REROT_FRONTIER_LAG1, 8, 64);
    CHECK(strong_rt.frontier_mode() == LLAMA_REROT_FRONTIER_STRONG);
    CHECK(lag_rt.frontier_mode() == LLAMA_REROT_FRONTIER_LAG1);
    (void) error;
}

static void test_final_fence_sees_last_sibling_write() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t episode_id = runtime.adopt_root(150, 151, 0, 0, 0);
    CHECK(runtime.response_task_id(episode_id) == 151);
    CHECK(commit_generated(runtime, episode_id, 0, 0,
        "<ol><li>Alpha</li><li>Beta</li></ol>"));
    CHECK(runtime.finish_frontier(episode_id).forked.size() == 1);
    CHECK(runtime.freeze_fork_parent(episode_id, 0));
    start_child(runtime, episode_id, 0, 1);
    start_child(runtime, episode_id, 1, 2);
    make_terminal(runtime, episode_id, 1);
    make_terminal(runtime, episode_id, 2);

    // Lane 1 writes its last PUBLIC finding, then both Lanes exit together.
    auto * lane1 = runtime.node(episode_id, 1);
    CHECK(commit_generated(runtime, episode_id, 1, lane1->storage_pos_next, "SECRET42 "));
    lane1 = runtime.node(episode_id, 1);
    const auto secret_run = lane1 && lane1->public_run.has_value()
        ? *lane1->public_run
        : LLAMA_REROT_RUN_INVALID;
    CHECK(secret_run != LLAMA_REROT_RUN_INVALID);
    request_exit(runtime, episode_id, 1);
    auto * lane2 = runtime.node(episode_id, 2);
    CHECK(commit_generated(runtime, episode_id, 2, lane2->storage_pos_next, "unrelated "));
    request_exit(runtime, episode_id, 2);

    const auto frontier = runtime.finish_frontier(episode_id);
    CHECK(frontier.natural_final());
    CHECK(frontier.final_node == 2);
    CHECK(frontier.retired.size() == 1 && frontier.retired[0] == 1);

    // The fence re-evaluation observes stable shared memory: SECRET42 is
    // visible to the survivor even though its writer already retired.
    std::vector<uint32_t> fence_runs;
    CHECK(runtime.refresh_final_fence(episode_id, 2, &fence_runs));
    CHECK(std::find(fence_runs.begin(), fence_runs.end(), secret_run) != fence_runs.end());
    CHECK(runtime.complete_serial_tail(episode_id, 2));
    const auto * episode = runtime.episode(episode_id);
    CHECK(episode && episode->serial_tail && episode->serial_node == 2);
    CHECK(runtime.response_task_id(episode_id) == 151);

    // The parallel scheduler is off: further frontier calls are no-ops.
    const auto idle = runtime.finish_frontier(episode_id);
    CHECK(!idle.hard_aborted && idle.final_node == LLAMA_REROT_NODE_INVALID);

    CHECK(runtime.erase_episode(episode_id));
}

static void test_hard_abort_cancels_everything() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t episode_id = runtime.adopt_root(160, 160, 0, 0, 0);
    CHECK(runtime.begin_frontier(episode_id));

    server_rerot_hard_limits limits;
    limits.max_total_tokens = 2;
    runtime.set_hard_limits(episode_id, limits);

    CHECK(commit_private(runtime, episode_id, 0, 0));
    CHECK(commit_generated(runtime, episode_id, 0, 1, "hi "));
    // Crossing the global budget hard-aborts the whole episode: no survivor,
    // no answer, finish_reason rerot_resource_exhausted.
    CHECK(!commit_generated(runtime, episode_id, 0, 2, "over "));
    const auto * episode = runtime.episode(episode_id);
    CHECK(episode && episode->hard_aborted);
    CHECK(episode && episode->abort_reason.find("rerot_resource_exhausted") != std::string::npos);
    CHECK(episode && episode->running.empty() && episode->ready_queue.empty());
    CHECK(!runtime.begin_frontier(episode_id));

    const auto frontier = runtime.finish_frontier(episode_id);
    CHECK(frontier.hard_aborted);
    CHECK(!frontier.natural_final());
    CHECK(frontier.final_node == LLAMA_REROT_NODE_INVALID);
    CHECK(!runtime.refresh_final_fence(episode_id, 0, nullptr));
    CHECK(runtime.erase_episode(episode_id));
}

static void test_queue_budget_aborts_instead_of_truncating() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t episode_id = runtime.adopt_root(170, 170, 0, 0, 0);
    server_rerot_hard_limits limits;
    limits.max_queue_descriptors = 1;
    runtime.set_hard_limits(episode_id, limits);

    // The public plan is never silently truncated: two descriptors against a
    // budget of one aborts the episode instead of dropping a child.
    CHECK(!commit_generated(runtime, episode_id, 0, 0, "<ol><li>A</li><li>B</li></ol>"));
    const auto * episode = runtime.episode(episode_id);
    CHECK(episode && episode->hard_aborted);
    CHECK(episode && episode->abort_reason.find("rerot_resource_exhausted") != std::string::npos);
    CHECK(runtime.erase_episode(episode_id));
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
    CHECK(episode && episode->abort_reason.find("rerot_resource_exhausted") != std::string::npos);
    CHECK(episode && !episode->abort_reason.empty());
    CHECK(episode && episode->running.empty() && episode->ready_queue.empty());
    const auto frontier = runtime.finish_frontier(episode_id);
    CHECK(frontier.hard_aborted && !frontier.natural_final());
    CHECK(runtime.erase_episode(episode_id));
}

int main() {
    std::fprintf(stderr, "=== RERoT Runtime Tests ===\n");
    test_n1_no_fork_disarm_forever();
    test_n2_strong_uptake();
    test_queue_fifo_five_children_one_slot();
    test_nested_fork_keeps_fifo();
    test_pending_invisible_until_atomic_publish();
    test_lag1_delays_same_frontier_peer();
    test_final_fence_sees_last_sibling_write();
    test_hard_abort_cancels_everything();
    test_queue_budget_aborts_instead_of_truncating();
    test_recursive_queue_and_last_survivor();
    test_same_frontier_exit_tie_break();
    test_internal_seq_exhaustion_aborts_whole_episode();
    std::fprintf(stderr, "=== Results: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
