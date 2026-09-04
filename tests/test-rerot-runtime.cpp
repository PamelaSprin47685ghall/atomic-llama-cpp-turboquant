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
        "final observation </blockquo"));
    node = runtime.node(episode_id, node_id);
    CHECK(node != nullptr);
    if (node) {
        CHECK(commit_generated(runtime, episode_id, node_id, node->storage_pos_next, "te>"));
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

static void test_line_mux_completion_order_and_visibility() {
    llama_rerot_document document(7);
    const auto lane_a = document.create_child(document.root(), "A");
    const auto lane_b = document.create_child(document.root(), "B");
    const auto public_a = document.append_run(
        lane_a, llama_rerot_visibility::public_live, 0, 1, 1);
    const auto public_b = document.append_run(
        lane_b, llama_rerot_visibility::public_live, 0, 1, 1);

    server_rerot_line_mux mux;
    std::vector<std::string> observed;
    auto collect = [&](std::string label, server_rerot_stream_lines result) {
        CHECK(result.ok);
        for (auto & line : result.lines) {
            observed.push_back(label + ":" + line);
        }
    };

    collect("A", mux.append(lane_a, public_a, "A partial", document));
    collect("B", mux.append(lane_b, public_b, "B first\nB second", document));
    collect("A", mux.append(lane_a, public_a, " done\n", document));
    collect("B", mux.append(lane_b, public_b, "\n", document));
    CHECK(observed == std::vector<std::string>({
        "B:B first\n",
        "A:A partial done\n",
        "B:B second\n",
    }));

    const auto hidden = document.append_run(
        lane_a, llama_rerot_visibility::pending_record, 10, 1);
    collect("A", mux.append(lane_a, hidden, "<ol>\n", document));
    CHECK(document.reclassify_run(
        hidden,
        llama_rerot_visibility::pending_record,
        llama_rerot_visibility::private_control));
    collect("A", mux.drain(lane_a, document));
    CHECK(observed.size() == 3);

    const auto delayed = document.append_run(
        lane_a, llama_rerot_visibility::pending_record, 11, 1);
    collect("A", mux.append(lane_a, delayed, "delayed", document));
    CHECK(document.reclassify_run(
        delayed,
        llama_rerot_visibility::pending_record,
        llama_rerot_visibility::public_live,
        2));
    collect("A", mux.drain(lane_a, document));
    CHECK(observed.size() == 3);
    collect("A", mux.append(lane_a, delayed, " line\n", document));
    CHECK(observed.back() == "A:delayed line\n");

    const auto tail = document.append_run(
        lane_b, llama_rerot_visibility::public_live, 20, 1, 3);
    collect("B", mux.append(lane_b, tail, "terminal remainder", document));
    collect("B", mux.finish(lane_b, document));
    CHECK(observed.back() == "B:terminal remainder\n");
    CHECK(mux.empty());

    server_rerot_line_mux invalid;
    const auto missing = invalid.append(lane_a, 9999, "bad\n", document);
    CHECK(!missing.ok);
}

static void test_marker_token_preserves_public_prefix() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t episode_id = runtime.adopt_root(96, 96, 0, 0, 0);
    server_rerot_line_mux mux;

    auto plan = runtime.plan_generated_token(
        episode_id, 0, 0, "answer</p");
    CHECK(plan.has_value());
    CHECK(plan && runtime.commit_token(episode_id, 0, *plan));
    if (!plan) {
        return;
    }
    auto ready = mux.append(
        0, plan->run_id, "answer</p", runtime.episode(episode_id)->document);
    CHECK(ready.ok && ready.lines.empty());

    auto * node = runtime.node(episode_id, 0);
    CHECK(node != nullptr);
    plan = runtime.plan_generated_token(
        episode_id, 0, node ? node->storage_pos_next : 1, "></");
    CHECK(plan.has_value());
    CHECK(plan && plan->marker_step.public_prefix_bytes == 1);
    CHECK(plan && runtime.commit_token(episode_id, 0, *plan));
    if (!plan) {
        return;
    }
    ready = mux.append(
        0,
        plan->run_id,
        "></",
        runtime.episode(episode_id)->document,
        plan->marker_step.public_prefix_bytes);
    CHECK(ready.ok && ready.lines.empty());

    node = runtime.node(episode_id, 0);
    CHECK(node != nullptr);
    plan = runtime.plan_generated_token(
        episode_id, 0, node ? node->storage_pos_next : 2, "blockquote>");
    CHECK(plan.has_value());
    CHECK(plan && plan->marker_step.marker_closed);
    CHECK(plan && runtime.commit_token(episode_id, 0, *plan));
    if (!plan) {
        return;
    }
    ready = mux.append(
        0,
        plan->run_id,
        "blockquote>",
        runtime.episode(episode_id)->document,
        plan->marker_step.public_prefix_bytes);
    CHECK(ready.ok && ready.lines.empty());

    ready = mux.finish(0, runtime.episode(episode_id)->document);
    CHECK(ready.ok);
    CHECK(ready.lines == std::vector<std::string>({"answer</p>\n"}));
    CHECK(mux.empty());
    CHECK(runtime.erase_episode(episode_id));
}

static void test_split_pending_record_resolution() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t episode_id = runtime.adopt_root(98, 98, 0, 0, 10);
    server_rerot_line_mux mux;
    std::vector<std::string> emitted;

    auto first = runtime.plan_generated_token(
        episode_id, 0, 10, "<ol>\n<li>split ");
    CHECK(first.has_value());
    CHECK(first && runtime.commit_token(episode_id, 0, *first));
    if (!first) {
        return;
    }
    auto ready = mux.append(0, first->run_id, "<ol>\n<li>split ",
        runtime.episode(episode_id)->document);
    CHECK(ready.ok && ready.lines.empty());

    auto closing = runtime.plan_generated_token(
        episode_id, 0, 12, "record</li></ol>\n");
    CHECK(closing.has_value());
    CHECK(closing && closing->run_id != first->run_id);
    CHECK(closing && closing->parser_step.record_closed);
    CHECK(closing && runtime.commit_token(episode_id, 0, *closing));
    if (!closing) {
        return;
    }
    ready = mux.append(0, closing->run_id, "record</li></ol>\n",
        runtime.episode(episode_id)->document);
    CHECK(ready.ok);
    emitted.insert(emitted.end(), ready.lines.begin(), ready.lines.end());
    CHECK(emitted == std::vector<std::string>({
        "<ol>\n",
        "<li>split record</li></ol>\n",
    }));

    const auto * episode = runtime.episode(episode_id);
    CHECK(episode && episode->pending_tokens == 0);
    CHECK(episode && episode->generated_public_tokens == 2);
    CHECK(episode && episode->document.run(first->run_id)->visibility ==
        llama_rerot_visibility::public_live);
    CHECK(episode && episode->document.run(closing->run_id)->visibility ==
        llama_rerot_visibility::public_live);

    auto * node = runtime.node(episode_id, 0);
    CHECK(node != nullptr);
    auto public_line = runtime.plan_generated_token(
        episode_id, 0, node->storage_pos_next, "public line\n");
    CHECK(public_line.has_value());
    CHECK(public_line && runtime.commit_token(episode_id, 0, *public_line));
    if (!public_line) {
        return;
    }
    ready = mux.append(0, public_line->run_id, "public line\n",
        runtime.episode(episode_id)->document);
    CHECK(ready.ok && ready.lines == std::vector<std::string>({"public line\n"}));

    node = runtime.node(episode_id, 0);
    auto marker_a = runtime.plan_generated_token(
        episode_id, 0, node->storage_pos_next, "</blockquo");
    CHECK(marker_a.has_value());
    CHECK(marker_a && runtime.commit_token(episode_id, 0, *marker_a));
    if (!marker_a) {
        return;
    }
    ready = mux.append(0, marker_a->run_id, "</blockquo",
        runtime.episode(episode_id)->document);
    CHECK(ready.ok && ready.lines.empty());

    node = runtime.node(episode_id, 0);
    auto marker_b = runtime.plan_generated_token(
        episode_id, 0, node->storage_pos_next + 1, "te>");
    CHECK(marker_b.has_value());
    CHECK(marker_b && marker_b->run_id != marker_a->run_id);
    CHECK(marker_b && marker_b->marker_step.marker_closed);
    CHECK(marker_b && runtime.commit_token(episode_id, 0, *marker_b));
    if (!marker_b) {
        return;
    }
    ready = mux.append(0, marker_b->run_id, "te>",
        runtime.episode(episode_id)->document);
    CHECK(ready.ok && ready.lines.empty());
    CHECK(mux.empty());

    episode = runtime.episode(episode_id);
    CHECK(episode && episode->pending_tokens == 0);
    CHECK(episode && episode->generated_private_tokens == 2);
    CHECK(episode && episode->document.run(marker_a->run_id)->visibility ==
        llama_rerot_visibility::private_control);
    CHECK(episode && episode->document.run(marker_b->run_id)->visibility ==
        llama_rerot_visibility::private_control);
    CHECK(runtime.erase_episode(episode_id));

    server_rerot_runtime prefix_runtime(
        nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t prefix_episode = prefix_runtime.adopt_root(97, 97, 0, 0, 20);
    server_rerot_line_mux prefix_mux;

    auto prefix_a = prefix_runtime.plan_generated_token(
        prefix_episode, 0, 20, "ordinary <");
    CHECK(prefix_a.has_value());
    CHECK(prefix_a && prefix_runtime.commit_token(prefix_episode, 0, *prefix_a));
    if (!prefix_a) {
        return;
    }
    ready = prefix_mux.append(0, prefix_a->run_id, "ordinary <",
        prefix_runtime.episode(prefix_episode)->document);
    CHECK(ready.ok && ready.lines.empty());

    auto prefix_b = prefix_runtime.plan_generated_token(
        prefix_episode, 0, 22, "o");
    CHECK(prefix_b.has_value());
    CHECK(prefix_b && prefix_b->run_id != prefix_a->run_id);
    CHECK(prefix_b && prefix_runtime.commit_token(prefix_episode, 0, *prefix_b));
    if (!prefix_b) {
        return;
    }
    ready = prefix_mux.append(0, prefix_b->run_id, "o",
        prefix_runtime.episode(prefix_episode)->document);
    CHECK(ready.ok && ready.lines.empty());

    auto released = prefix_runtime.plan_generated_token(
        prefix_episode, 0, 24, "x\n");
    CHECK(released.has_value());
    CHECK(released && released->parser_step.release_previous_pending);
    CHECK(released && prefix_runtime.commit_token(prefix_episode, 0, *released));
    if (!released) {
        return;
    }
    ready = prefix_mux.append(0, released->run_id, "x\n",
        prefix_runtime.episode(prefix_episode)->document);
    CHECK(ready.ok);
    CHECK(ready.lines == std::vector<std::string>({"ordinary <ox\n"}));
    CHECK(prefix_mux.empty());

    episode = prefix_runtime.episode(prefix_episode);
    CHECK(episode && episode->pending_tokens == 0);
    CHECK(episode && episode->generated_public_tokens == 3);
    CHECK(episode && episode->document.run(prefix_a->run_id)->visibility ==
        llama_rerot_visibility::public_live);
    CHECK(episode && episode->document.run(prefix_b->run_id)->visibility ==
        llama_rerot_visibility::public_live);
    CHECK(prefix_runtime.erase_episode(prefix_episode));
}

static void test_private_span_reserves_one_contiguous_run() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t episode_id = runtime.adopt_root(99, 99, 0, 0, 10);
    const auto plans = runtime.plan_private_span(episode_id, 0, 10, 4);
    CHECK(plans.has_value());
    CHECK(plans && plans->size() == 4);
    CHECK(!runtime.plan_private_span(episode_id, 0, 10, 0).has_value());
    if (!plans || plans->size() != 4) {
        return;
    }

    const llama_rerot_run_id run_id = plans->front().run_id;
    for (size_t i = 0; i < plans->size(); ++i) {
        CHECK((*plans)[i].run_id == run_id);
        CHECK((*plans)[i].storage_pos == 10 + static_cast<llama_pos>(i));
        CHECK(runtime.commit_token(episode_id, 0, (*plans)[i]));
    }

    const auto * episode = runtime.episode(episode_id);
    const auto * node = runtime.node(episode_id, 0);
    const auto * run = episode ? episode->document.run(run_id) : nullptr;
    CHECK(run && run->token_count == 4);
    CHECK(node && node->storage_pos_next == 14);
    CHECK(episode && episode->generated_private_tokens == 4);
    CHECK(runtime.erase_episode(episode_id));
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
    // The structural record starts PENDING but is accounted as PUBLIC after
    // atomic N=1 publication; the forced/private prelude remains separate.
    CHECK(episode && episode->pending_tokens == 0);
    CHECK(episode && episode->generated_private_tokens == 1);
    CHECK(episode && episode->generated_public_tokens == 2);

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
    const std::string original_reason = episode ? episode->abort_reason : std::string();
    CHECK(!runtime.hard_abort(episode_id, "generic wrapper must not replace first cause"));
    episode = runtime.episode(episode_id);
    CHECK(episode && episode->abort_reason == original_reason);
    CHECK(episode && episode->running.empty() && episode->ready_queue.empty());
    CHECK(!runtime.begin_frontier(episode_id));

    const auto frontier = runtime.finish_frontier(episode_id);
    CHECK(frontier.hard_aborted);
    CHECK(!frontier.natural_final());
    CHECK(frontier.final_node == LLAMA_REROT_NODE_INVALID);
    CHECK(!runtime.refresh_final_fence(episode_id, 0, nullptr));
    CHECK(runtime.erase_episode(episode_id));
}

static void test_requested_episode_ids_are_stable() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);

    const uint64_t requested = runtime.adopt_root(165, 165, 0, 0, 0, 42);
    CHECK(requested == 42);
    CHECK(runtime.episode(42) != nullptr);
    std::vector<llama_seq_id> semantic_seq_ids;
    CHECK(runtime.sync_public_archive(42, &semantic_seq_ids));
    CHECK(semantic_seq_ids.size() == 2);
    CHECK(std::find(semantic_seq_ids.begin(), semantic_seq_ids.end(), 0) !=
          semantic_seq_ids.end());
    CHECK(runtime.adopt_root(166, 166, 1, 1, 0, 42) == 0);
    CHECK(runtime.erase_episode(42));

    const uint64_t automatic = runtime.adopt_root(167, 167, 0, 0, 0);
    CHECK(automatic == 43);
    CHECK(runtime.erase_episode(automatic));
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

static server_rerot_state_fingerprints test_state_fingerprints() {
    server_rerot_state_fingerprints fp;
    fp.caps =
        LLAMA_REROT_STATE_CAP_REROT |
        LLAMA_REROT_STATE_CAP_REROT_TREE |
        LLAMA_REROT_STATE_CAP_REROT_PRIVATE |
        LLAMA_REROT_STATE_CAP_REROT_MTP |
        LLAMA_REROT_STATE_CAP_HYBRID_REC |
        LLAMA_REROT_STATE_CAP_SPARSE_KV |
        LLAMA_REROT_STATE_CAP_TRIATTENTION;
    fp.model_fp = 0x1122334455667788ULL;
    fp.rope_fp = 0x8877665544332211ULL;
    fp.tri_fp = 0x0f1e2d3c4b5a6978ULL;
    return fp;
}

static void test_episode_state_round_trip_and_fingerprint() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t episode_id = runtime.adopt_root(280, 281, 0, 0, 10, 77);
    CHECK(episode_id == 77);
    CHECK(commit_private(runtime, episode_id, 0, 10));
    CHECK(commit_generated(runtime, episode_id, 0, 11,
        "<ol><li>Persist this terminal task</li></ol>"));

    auto * lane = runtime.node(episode_id, 0);
    CHECK(lane != nullptr);
    if (lane) {
        lane->sampler_blob = {1, 2, 3, 4};
        lane->mtp_blob = {9, 8, 7};
        lane->view_stamp = {4, 5, 6};
    }

    const auto fp = test_state_fingerprints();
    std::vector<uint8_t> blob;
    std::string error;
    CHECK(runtime.save_episode(episode_id, fp, &blob, &error));
    CHECK(!blob.empty());
    CHECK(error.empty());

    server_rerot_runtime restored(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    uint64_t restored_id = 0;
    CHECK(restored.load_episode(blob.data(), blob.size(), fp, &restored_id, &error));
    CHECK(restored_id == episode_id);
    const auto * restored_episode = restored.episode(restored_id);
    const auto * restored_lane = restored.node(restored_id, 0);
    CHECK(restored_episode != nullptr);
    CHECK(restored_episode && restored_episode->root_task_id == 280);
    CHECK(restored_episode && restored_episode->response_task_id == 281);
    CHECK(restored_episode && restored_episode->base_prefix_end == 10);
    CHECK(restored_lane != nullptr);
    CHECK(restored_lane && restored_lane->sampler_blob == std::vector<uint8_t>({1, 2, 3, 4}));
    CHECK(restored_lane && restored_lane->mtp_blob == std::vector<uint8_t>({9, 8, 7}));
    CHECK(restored_lane && restored_lane->view_stamp.topology_epoch == 4);
    CHECK(restored_lane && restored_lane->view_stamp.publish_epoch == 5);
    CHECK(restored_lane && restored_lane->view_stamp.layout_epoch == 6);
    CHECK(restored_episode && restored_episode->document.validate(&error));

    server_rerot_runtime wrong_model(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    auto wrong_fp = fp;
    ++wrong_fp.model_fp;
    error.clear();
    CHECK(!wrong_model.load_episode(blob.data(), blob.size(), wrong_fp, nullptr, &error));
    CHECK(error.find("model fingerprint mismatch") != std::string::npos);

    server_rerot_runtime wrong_caps(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    wrong_fp = fp;
    wrong_fp.caps &= ~LLAMA_REROT_STATE_CAP_REROT_MTP;
    error.clear();
    CHECK(!wrong_caps.load_episode(blob.data(), blob.size(), wrong_fp, nullptr, &error));
    CHECK(error.find("capability bitmap mismatch") != std::string::npos);

    wrong_fp = fp;
    wrong_fp.caps = LLAMA_REROT_STATE_CAP_NONE;
    blob.clear();
    error.clear();
    CHECK(!runtime.save_episode(episode_id, wrong_fp, &blob, &error));
    CHECK(error.find("capability bitmap") != std::string::npos);
    CHECK(blob.empty());
}

static void test_context_shift_truncates_only_unpinned_public_runs() {
    server_rerot_episode episode(290);
    server_rerot_node_runtime root;
    root.id = 0;
    episode.nodes.push_back(std::move(root));
    episode.base_prefix_end = 10;
    episode.publish_epoch = 3;
    episode.layout_epoch = 7;

    const auto prefix = episode.document.append_run(
        0, llama_rerot_visibility::public_live, 0, 10, 1);
    const auto old_public = episode.document.append_run(
        0, llama_rerot_visibility::public_live, 10, 4, 2);
    const auto active_public = episode.document.append_run(
        0, llama_rerot_visibility::public_live, 14, 3, 3);
    CHECK(prefix != LLAMA_REROT_RUN_INVALID);
    CHECK(old_public != LLAMA_REROT_RUN_INVALID);
    CHECK(active_public != LLAMA_REROT_RUN_INVALID);
    episode.nodes[0].public_run = active_public;

    server_rerot_shift_result result;
    std::string error;
    CHECK(server_rerot_truncate_oldest_public(episode, 3, &result, &error));
    CHECK(error.empty());
    CHECK(result.tokens_removed == 4);
    CHECK(result.runs_truncated == 1);
    CHECK(result.runs_emptied == 1);
    CHECK(result.new_layout_epoch == 8);
    CHECK(result.new_publish_epoch == 4);
    CHECK(episode.topology_barrier_pending);
    CHECK(episode.document.run(prefix)->token_count == 10);
    CHECK(episode.document.run(old_public)->token_count == 0);
    CHECK(episode.document.run(active_public)->token_count == 3);

    const uint64_t layout_after = episode.layout_epoch;
    const uint64_t publish_after = episode.publish_epoch;
    CHECK(server_rerot_truncate_oldest_public(episode, 100, &result, &error));
    CHECK(result.tokens_removed == 0);
    CHECK(episode.layout_epoch == layout_after);
    CHECK(episode.publish_epoch == publish_after);
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

static void test_chronicle_to_canonical_mapping_registry() {
    server_rerot_clear_chronicle_registry();

    const std::string chronicle =
        "<ol>\n<li>Task B</li>\n<li>Task A</li>\n</ol>\n"
        "<h1>Task B</h1>\nResult B\n"
        "<h1>Task A</h1>\nResult A\n";
    const std::string canonical =
        "<ol>\n<li>Task A</li>\n<li>Task B</li>\n</ol>\n"
        "<h1>Task A</h1>\nResult A\n"
        "<h1>Task B</h1>\nResult B\n";

    // Unregistered thoughts return nullopt (never blindly mapped)
    CHECK(!server_rerot_resolve_canonical_reasoning(chronicle).has_value());
    CHECK(!server_rerot_resolve_canonical_reasoning("completely unknown thought").has_value());
    CHECK(!server_rerot_resolve_canonical_reasoning("").has_value());

    // Register mapping
    server_rerot_register_chronicle_mapping(chronicle, canonical, "Final answer", 42);

    // Exact match resolves to canonical PAC-DFS document
    auto resolved = server_rerot_resolve_canonical_reasoning(chronicle);
    CHECK(resolved.has_value());
    CHECK(resolved.value() == canonical);

    // Whitespace trimming variations match safely
    CHECK(server_rerot_resolve_canonical_reasoning("  \n" + chronicle + "\n  ").value() == canonical);

    // Tampered or modified thoughts are strictly rejected (no false mappings)
    const std::string tampered =
        "<ol>\n<li>Task B</li>\n<li>Task A</li>\n</ol>\n"
        "<h1>Task B</h1>\nTampered Result B\n"
        "<h1>Task A</h1>\nResult A\n";
    CHECK(!server_rerot_resolve_canonical_reasoning(tampered).has_value());

    // Identical chronicle and canonical are skipped as no-ops
    server_rerot_register_chronicle_mapping("same", "same", "", 43);
    CHECK(!server_rerot_resolve_canonical_reasoning("same").has_value());

    server_rerot_clear_chronicle_registry();
    CHECK(!server_rerot_resolve_canonical_reasoning(chronicle).has_value());
}

int main() {
    std::fprintf(stderr, "=== RERoT Runtime Tests ===\n");
    test_chronicle_to_canonical_mapping_registry();
    test_line_mux_completion_order_and_visibility();
    test_marker_token_preserves_public_prefix();
    test_split_pending_record_resolution();
    test_private_span_reserves_one_contiguous_run();
    test_n1_no_fork_disarm_forever();
    test_n2_strong_uptake();
    test_queue_fifo_five_children_one_slot();
    test_nested_fork_keeps_fifo();
    test_pending_invisible_until_atomic_publish();
    test_lag1_delays_same_frontier_peer();
    test_final_fence_sees_last_sibling_write();
    test_hard_abort_cancels_everything();
    test_requested_episode_ids_are_stable();
    test_queue_budget_aborts_instead_of_truncating();
    test_recursive_queue_and_last_survivor();
    test_same_frontier_exit_tie_break();
    test_episode_state_round_trip_and_fingerprint();
    test_context_shift_truncates_only_unpinned_public_runs();
    test_internal_seq_exhaustion_aborts_whole_episode();
    std::fprintf(stderr, "=== Results: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
