#include "server-rerot.h"
#include "llama-context.h"
#include "llama-model.h"

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

static void test_people_pen_scheduler() {
    // Phase 1 acceptance gate (§§B.0, B.4.2, B.8, B.13 Phase 1)
    // 1. B=3, P=9: single person can take all 9 pens (no per-person cap)
    {
        server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 10, 100);
        runtime.set_pen_capacity(9);
        CHECK(runtime.pen_capacity() == 9);
        CHECK(runtime.pens_allocated() == 0);

        const uint64_t ep_a = runtime.adopt_root(101, 101, 0, 10, 0, 1);
        CHECK(ep_a == 1);
        CHECK(runtime.pens_allocated() == 1);

        // Fork into 10 children
        std::string list_xml = "<ol>";
        for (int i = 0; i < 10; ++i) {
            list_xml += "<li>Task A" + std::to_string(i) + "</li>";
        }
        list_xml += "</ol>";
        CHECK(commit_generated(runtime, ep_a, 0, 0, list_xml));
        runtime.finish_frontier(ep_a);
        CHECK(runtime.freeze_fork_parent(ep_a, 0));
        // Parent frozen: pen 0 is freed, 10 children in ready_queue
        CHECK(runtime.pens_allocated() == 0);
        const auto * ep_a_ptr = runtime.episode(ep_a);
        CHECK(ep_a_ptr && ep_a_ptr->ready_queue.size() == 10);

        // Schedule pens for Person A: should get all 9 pens
        const size_t admitted = runtime.schedule_pens({ep_a});
        CHECK(admitted == 9);
        CHECK(runtime.pens_allocated() == 9);
        CHECK(runtime.pens_for_person(ep_a).size() == 9);
        // 10th child remains safely in ready_queue with zero descriptor loss
        CHECK(ep_a_ptr->ready_queue.size() == 1);
        CHECK(runtime.has_ready_nodes(ep_a));
        CHECK(runtime.erase_episode(ep_a));
        CHECK(runtime.pens_allocated() == 0);
    }

    // 2. B=3, P=9: three people progress 3/3/3
    {
        server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 10, 100);
        runtime.set_pen_capacity(9);

        const uint64_t ep1 = runtime.adopt_root(1, 1, 0, 10, 0, 10);
        const uint64_t ep2 = runtime.adopt_root(2, 2, 1, 11, 0, 20);
        const uint64_t ep3 = runtime.adopt_root(3, 3, 2, 12, 0, 30);
        CHECK(runtime.pens_allocated() == 3);

        // Each forks into 4 children
        for (uint64_t ep : {ep1, ep2, ep3}) {
            CHECK(commit_generated(runtime, ep, 0, 0, "<ol><li>1</li><li>2</li><li>3</li><li>4</li></ol>"));
            runtime.finish_frontier(ep);
            CHECK(runtime.freeze_fork_parent(ep, 0));
        }
        CHECK(runtime.pens_allocated() == 0);

        // Schedule across all 3 people with P=9:
        // Pass 1: 1/1/1 (3 pens)
        // Pass 2: +1/+1/+1 (6 pens), then +1/+1/+1 (9 pens) -> 3/3/3!
        const size_t admitted = runtime.schedule_pens({ep1, ep2, ep3});
        CHECK(admitted == 9);
        CHECK(runtime.pens_allocated() == 9);
        CHECK(runtime.pens_for_person(ep1).size() == 3);
        CHECK(runtime.pens_for_person(ep2).size() == 3);
        CHECK(runtime.pens_for_person(ep3).size() == 3);

        // Each person has exactly 1 child remaining in ready_queue
        CHECK(runtime.episode(ep1)->ready_queue.size() == 1);
        CHECK(runtime.episode(ep2)->ready_queue.size() == 1);
        CHECK(runtime.episode(ep3)->ready_queue.size() == 1);

        CHECK(runtime.erase_episode(ep1));
        CHECK(runtime.erase_episode(ep2));
        CHECK(runtime.erase_episode(ep3));
        CHECK(runtime.pens_allocated() == 0);
    }

    // 3. B=3, P=5: sum of allocated pens is always <= 5
    {
        server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 10, 100);
        runtime.set_pen_capacity(5);
        CHECK(runtime.pen_capacity() == 5);

        const uint64_t ep1 = runtime.adopt_root(1, 1, 0, 10, 0, 100);
        const uint64_t ep2 = runtime.adopt_root(2, 2, 1, 11, 0, 200);
        const uint64_t ep3 = runtime.adopt_root(3, 3, 2, 12, 0, 300);

        for (uint64_t ep : {ep1, ep2, ep3}) {
            CHECK(commit_generated(runtime, ep, 0, 0, "<ol><li>A</li><li>B</li><li>C</li><li>D</li></ol>"));
            runtime.finish_frontier(ep);
            CHECK(runtime.freeze_fork_parent(ep, 0));
        }

        const size_t admitted = runtime.schedule_pens({ep1, ep2, ep3});
        CHECK(admitted == 5);
        CHECK(runtime.pens_allocated() == 5);
        CHECK(runtime.pens_allocated() <= 5);

        // Total pens across all people is exactly 5
        const size_t sum_pens = runtime.pens_for_person(ep1).size() +
                                runtime.pens_for_person(ep2).size() +
                                runtime.pens_for_person(ep3).size();
        CHECK(sum_pens == 5);

        // 4. Pen return when person has no ready work:
        // When a node finishes, its pen is returned immediately
        auto p1_pens = runtime.pens_for_person(ep1);
        CHECK(!p1_pens.empty());
        const int p1_slot = p1_pens[0];
        runtime.release_slot(p1_slot);
        CHECK(runtime.pens_allocated() == 4);
        CHECK(runtime.pen(p1_slot)->state == server_pen_state::free);

        // New work can immediately acquire the returned pen
        const size_t re_admitted = runtime.schedule_pens({ep1});
        CHECK(re_admitted == 1);
        CHECK(runtime.pens_allocated() == 5);

        CHECK(runtime.erase_episode(ep1));
        CHECK(runtime.erase_episode(ep2));
        CHECK(runtime.erase_episode(ep3));
        CHECK(runtime.pens_allocated() == 0);
    }

    // 5. Child count far larger than P (50 children, P=4): zero descriptor loss
    {
        server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 10, 200);
        runtime.set_pen_capacity(4);

        const uint64_t ep = runtime.adopt_root(1, 1, 0, 10, 0, 500);
        std::string big_list = "<ol>";
        for (int i = 0; i < 50; ++i) {
            big_list += "<li>Task " + std::to_string(i) + "</li>";
        }
        big_list += "</ol>";

        CHECK(commit_generated(runtime, ep, 0, 0, big_list));
        runtime.finish_frontier(ep);
        CHECK(runtime.freeze_fork_parent(ep, 0));

        const auto * ep_ptr = runtime.episode(ep);
        CHECK(ep_ptr && ep_ptr->ready_queue.size() == 50);

        // Schedule: exactly 4 admitted, 46 remain queued
        size_t admitted = runtime.schedule_pens({ep});
        CHECK(admitted == 4);
        CHECK(runtime.pens_allocated() == 4);
        CHECK(ep_ptr->ready_queue.size() == 46);

        // Verify descriptors are strictly intact in FIFO order
        for (size_t i = 0; i < 46; ++i) {
            llama_rerot_node_id q_node = ep_ptr->ready_queue[i];
            const auto * node_doc = ep_ptr->document.node(q_node);
            CHECK(node_doc != nullptr);
            std::string expected_title = "Task " + std::to_string(i + 4);
            CHECK(node_doc->title == expected_title);
        }

        CHECK(runtime.erase_episode(ep));
        CHECK(runtime.pens_allocated() == 0);
    }
}

struct test_stub_model : public llama_model {
    test_stub_model() : llama_model(llama_model_default_params()) {
        hparams.n_ctx_train = 4096;
        arch = LLM_ARCH_QWEN35;
    }
    void load_stats(llama_model_loader &) override {}
    void load_hparams(llama_model_loader &) override {}
    void load_vocab(llama_model_loader &) override {}
    bool load_tensors(llama_model_loader &) override { return true; }
    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override {
        return nullptr;
    }
};

static void test_context_multi_episode_isolation() {
    // Phase 2 acceptance gate (§§B.5, B.13 Phase 2)
    test_stub_model model;
    llama_cparams cparams = {};
    cparams.rerot_enabled = true;
    cparams.rerot_frontier = LLAMA_REROT_FRONTIER_STRONG;
    llama_context ctx(model, cparams, true);

    // 1. Same context simultaneously begins 3 episodes
    CHECK(llama_rerot_episode_begin(&ctx, 101, nullptr));
    CHECK(llama_rerot_episode_begin(&ctx, 102, nullptr));
    CHECK(llama_rerot_episode_begin(&ctx, 103, nullptr));

    CHECK(llama_rerot_is_active(&ctx, 101));
    CHECK(llama_rerot_is_active(&ctx, 102));
    CHECK(llama_rerot_is_active(&ctx, 103));
    CHECK(!llama_rerot_is_active(&ctx, 999));

    // Duplicate begin of already active episode fails
    CHECK(!llama_rerot_episode_begin(&ctx, 101, nullptr));

    // 2. Sequence binding constraint: same seq cannot be bound to two different people/episodes
    llama_rerot_write_tag tag101 = {};
    tag101.episode_id = 101;
    tag101.node_id = 1;
    tag101.run_id = 1;
    tag101.publish_epoch = 1;
    tag101.frontier = 1;
    tag101.visibility = LLAMA_REROT_KV_PUBLIC_LIVE;
    CHECK(llama_rerot_set_write_tag(&ctx, 5, &tag101));

    llama_rerot_write_tag tag102 = {};
    tag102.episode_id = 102;
    tag102.node_id = 2;
    tag102.run_id = 2;
    tag102.publish_epoch = 1;
    tag102.frontier = 1;
    tag102.visibility = LLAMA_REROT_KV_PUBLIC_LIVE;
    CHECK(!llama_rerot_set_write_tag(&ctx, 5, &tag102)); // Rejected: seq 5 bound to episode 101

    uint32_t run_ids[] = {1};
    llama_rerot_frontier_reader_view v102 = {5, 102, 2, 2, 1, LLAMA_REROT_FRONTIER_STRONG, {1, 1, 1}, run_ids, 1};
    CHECK(!llama_rerot_set_frontier_views(&ctx, &v102, 1)); // Rejected: seq 5 bound to episode 101

    // 3. Ending A does not clear B/C views/tags/stamps
    llama_rerot_frontier_reader_view v_ep102 = {6, 102, 1, 1, 1, LLAMA_REROT_FRONTIER_STRONG, {10, 20, 30}, run_ids, 1};
    CHECK(llama_rerot_set_frontier_views(&ctx, &v_ep102, 1));
    llama_rerot_frontier_reader_view v_ep103 = {7, 103, 1, 1, 1, LLAMA_REROT_FRONTIER_STRONG, {15, 25, 35}, run_ids, 1};
    CHECK(llama_rerot_set_frontier_views(&ctx, &v_ep103, 1));

    // End episode 101
    llama_rerot_episode_end(&ctx, 101);
    CHECK(!llama_rerot_is_active(&ctx, 101));
    CHECK(llama_rerot_is_active(&ctx, 102));
    CHECK(llama_rerot_is_active(&ctx, 103));

    // Stamps for 102 and 103 are untouched
    llama_rerot_view_stamp stamp102 = {10, 20, 30};
    CHECK(!llama_rerot_mtp_is_stale(&ctx, 6, &stamp102));
    llama_rerot_view_stamp stamp103 = {15, 25, 35};
    CHECK(!llama_rerot_mtp_is_stale(&ctx, 7, &stamp103));

    // 4. A topology bump in 102 does not make 103 MTP stale
    llama_rerot_frontier_reader_view v_ep102_bump = {6, 102, 1, 1, 2, LLAMA_REROT_FRONTIER_STRONG, {11, 20, 30}, run_ids, 1};
    CHECK(llama_rerot_set_frontier_views(&ctx, &v_ep102_bump, 1));
    CHECK(llama_rerot_mtp_is_stale(&ctx, 6, &stamp102)); // Seq 6 is stale
    CHECK(!llama_rerot_mtp_is_stale(&ctx, 7, &stamp103)); // Seq 7 is NOT stale!

    // 5. Each episode can independently save, load, and end
    std::vector<uint8_t> env102;
    std::string err;
    CHECK(llama_rerot_context_save_envelope_episode(&ctx, 102, env102, &err));
    CHECK(!env102.empty());

    std::vector<uint8_t> env103;
    CHECK(llama_rerot_context_save_envelope_episode(&ctx, 103, env103, &err));
    CHECK(!env103.empty());

    // End episode 102
    llama_rerot_episode_end(&ctx, 102);
    CHECK(!llama_rerot_is_active(&ctx, 102));
    CHECK(llama_rerot_is_active(&ctx, 103));

    // Restore episode 102: restores alongside active 103
    CHECK(llama_rerot_context_load_envelope(&ctx, env102.data(), env102.size(), &err));
    CHECK(llama_rerot_is_active(&ctx, 102));
    CHECK(llama_rerot_is_active(&ctx, 103));

    llama_rerot_view_stamp restored_stamp102 = {11, 20, 30};
    CHECK(!llama_rerot_mtp_is_stale(&ctx, 6, &restored_stamp102));
    CHECK(!llama_rerot_mtp_is_stale(&ctx, 7, &stamp103));

    llama_rerot_episode_end(&ctx, 102);
    llama_rerot_episode_end(&ctx, 103);
    CHECK(!llama_rerot_is_active(&ctx, 0));
}

static void test_multi_person_multi_pen_bxp_stress() {
    std::fprintf(stderr, "--- test_multi_person_multi_pen_bxp_stress (§B.0, §B.8, §B.12, §B.16) ---\n");
    // B=6 people, P=18 pens
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 10, 250);
    runtime.set_pen_capacity(18);
    CHECK(runtime.pen_capacity() == 18);

    std::vector<uint64_t> people;
    const std::vector<int> child_counts = { 5, 2, 4, 1, 3, 3 }; // ragged distribution
    for (size_t i = 0; i < child_counts.size(); ++i) {
        const uint64_t ep = runtime.adopt_root(
            int(i + 1), int(i + 1), int(i), llama_seq_id(10 + i), 0, uint64_t(100 + i));
        people.push_back(ep);
    }
    CHECK(runtime.pens_allocated() == 6);

    // People 0, 1, 2, 4, 5 fork into their respective child counts. Person 3 (N=1) continues terminal.
    for (size_t i = 0; i < people.size(); ++i) {
        const uint64_t ep = people[i];
        const int n_children = child_counts[i];
        if (n_children == 1) {
            // N=1: commit single item, disarms planner forever
            CHECK(commit_generated(runtime, ep, 0, 0, "<ol><li>Single task</li></ol>"));
            runtime.finish_frontier(ep);
            CHECK(runtime.episode(ep)->document.node(0)->state == llama_rerot_node_state::terminal_running);
            continue;
        }

        std::string xml = "<ol>";
        for (int c = 0; c < n_children; ++c) {
            xml += "<li>Task " + std::to_string(c) + "</li>";
        }
        xml += "</ol>";
        CHECK(commit_generated(runtime, ep, 0, 0, xml));
        runtime.finish_frontier(ep);
        CHECK(runtime.freeze_fork_parent(ep, 0));
    }

    // Schedule pens across all people:
    // Person 3 holds 1 pen.
    // Persons 0 (5), 1 (2), 2 (4), 4 (3), 5 (3) need 17 pens.
    // Total pens allocated = 1 + 17 = 18 pens (exactly P=18!).
    const size_t admitted = runtime.schedule_pens(people);
    CHECK(admitted == 17);
    CHECK(runtime.pens_allocated() == 18);
    CHECK(runtime.pens_allocated() <= runtime.pen_capacity());

    for (size_t i = 0; i < people.size(); ++i) {
        const uint64_t ep = people[i];
        const auto p_pens = runtime.pens_for_person(ep);
        if (child_counts[i] == 1) {
            CHECK(p_pens.size() == 1);
        } else {
            CHECK(p_pens.size() == size_t(child_counts[i]));
            for (size_t c = 1; c <= size_t(child_counts[i]); ++c) {
                CHECK(runtime.complete_admission(ep, llama_rerot_node_id(c)));
            }
        }
    }

    // Advance 3 frontiers and simulate completion of children
    for (size_t i = 0; i < people.size(); ++i) {
        const uint64_t ep = people[i];
        const auto p_pens = runtime.pens_for_person(ep);
        if (child_counts[i] == 1) {
            make_terminal(runtime, ep, 0);
            request_exit(runtime, ep, 0);
            auto f = runtime.finish_frontier(ep);
            CHECK(f.natural_final());
        } else {
            // Retire all children except the last one
            for (size_t c = 1; c < size_t(child_counts[i]); ++c) {
                make_terminal(runtime, ep, llama_rerot_node_id(c));
                request_exit(runtime, ep, llama_rerot_node_id(c));
            }
            auto f = runtime.finish_frontier(ep);
            CHECK(f.retired.size() == size_t(child_counts[i] - 1));

            // Retire the final child
            const llama_rerot_node_id last_child = llama_rerot_node_id(child_counts[i]);
            make_terminal(runtime, ep, last_child);
            request_exit(runtime, ep, last_child);
            auto final_f = runtime.finish_frontier(ep);
            CHECK(final_f.natural_final());
            CHECK(final_f.final_node == last_child);
        }
    }

    // Erase all episodes and verify full pen reclamation
    for (uint64_t ep : people) {
        CHECK(runtime.erase_episode(ep));
    }
    CHECK(runtime.pens_allocated() == 0);
    CHECK(runtime.pens_running() == 0);
}

static void test_triattention_multi_person_pressure() {
    std::fprintf(stderr, "--- test_triattention_multi_person_pressure (§B.9, §B.11, §B.13 Phase 7) ---\n");
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 10, 100);
    runtime.set_pen_capacity(9);

    const uint64_t ep1 = runtime.adopt_root(1, 1, 0, 10, 0, 201);
    const uint64_t ep2 = runtime.adopt_root(2, 2, 1, 11, 0, 202);
    const uint64_t ep3 = runtime.adopt_root(3, 3, 2, 12, 0, 203);
    CHECK(runtime.pens_allocated() == 3);

    // Person 1 creates 500 public tokens, Person 2 creates 200, Person 3 creates 800
    auto * ep1_ptr = runtime.episode(ep1);
    auto * ep2_ptr = runtime.episode(ep2);
    auto * ep3_ptr = runtime.episode(ep3);
    CHECK(ep1_ptr && ep2_ptr && ep3_ptr);

    const auto r1 = ep1_ptr->document.append_run(0, llama_rerot_visibility::public_live, 0, 500, 1);
    const auto r1_active = ep1_ptr->document.append_run(0, llama_rerot_visibility::public_live, 500, 10, 2);
    runtime.node(ep1, 0)->public_run = r1_active;

    const auto r2 = ep2_ptr->document.append_run(0, llama_rerot_visibility::public_live, 0, 200, 1);
    const auto r3 = ep3_ptr->document.append_run(0, llama_rerot_visibility::public_live, 0, 800, 1);
    CHECK(r1 != LLAMA_REROT_RUN_INVALID && r2 != LLAMA_REROT_RUN_INVALID && r3 != LLAMA_REROT_RUN_INVALID);

    // Simulate TriAttention truncation on Person 1 only (drain 500 tokens)
    server_rerot_shift_result shift_res;
    std::string err;
    CHECK(server_rerot_truncate_oldest_public(*ep1_ptr, 450, &shift_res, &err));
    CHECK(shift_res.tokens_removed == 500);
    CHECK(ep1_ptr->document.run(r1)->token_count == 0);

    // Invariant: Person 2 and Person 3 public runs and token counts remain completely intact
    CHECK(ep2_ptr->document.run(r2)->token_count == 200);
    CHECK(ep3_ptr->document.run(r3)->token_count == 800);

    // Invariant: Recurrent pen queueing / shortage does NOT invoke TriAttention reclaim (§B.9.1)
    // Add 10 queued children to Person 2 with pen_capacity=9. Total pens needed = 13 > 9.
    std::string list_xml = "<ol>";
    for (int i = 0; i < 10; ++i) list_xml += "<li>Subtask " + std::to_string(i) + "</li>";
    list_xml += "</ol>";
    CHECK(commit_generated(runtime, ep2, 0, 200, list_xml));
    runtime.finish_frontier(ep2);
    CHECK(runtime.freeze_fork_parent(ep2, 0));

    // Ready queue contains 10 children. Scheduling allocates remaining pens without touching KV runs!
    const size_t admitted = runtime.schedule_pens({ep1, ep2, ep3});
    CHECK(admitted > 0);
    CHECK(runtime.pens_allocated() <= runtime.pen_capacity());
    CHECK(ep2_ptr->document.run(r2)->token_count == 200); // KV untouched!
    CHECK(ep3_ptr->document.run(r3)->token_count == 800); // KV untouched!

    CHECK(runtime.erase_episode(ep1));
    CHECK(runtime.erase_episode(ep2));
    CHECK(runtime.erase_episode(ep3));
    CHECK(runtime.pens_allocated() == 0);
}

static void test_shared_prefix_multi_branch_union_and_preemption() {
    std::fprintf(stderr, "--- test_shared_prefix_multi_branch_union_and_preemption (§A.8, §A.11, §A.24, §B.8) ---\n");
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 10, 100);
    runtime.set_pen_capacity(8);

    // 1. Setup 3 branches sharing a common prefix (e.g. 100 prefix tokens)
    const uint64_t ep1 = runtime.adopt_root(1, 1, 0, 10, 100, 101);
    const uint64_t ep2 = runtime.adopt_root(2, 2, 1, 11, 100, 102);
    const uint64_t ep3 = runtime.adopt_root(3, 3, 2, 12, 100, 103);
    CHECK(runtime.pens_allocated() == 3);

    // Branch 1 forks into 3 children, Branch 2 forks into 2 children
    CHECK(commit_generated(runtime, ep1, 0, 100, "<ol><li>B1-1</li><li>B1-2</li><li>B1-3</li></ol>"));
    CHECK(commit_generated(runtime, ep2, 0, 100, "<ol><li>B2-1</li><li>B2-2</li></ol>"));
    runtime.finish_frontier(ep1);
    runtime.finish_frontier(ep2);

    CHECK(runtime.freeze_fork_parent(ep1, 0));
    CHECK(runtime.freeze_fork_parent(ep2, 0));

    // Ep3 remains running (1 pen). Ep1 needs 3 pens, Ep2 needs 2 pens. Total needed = 6 pens.
    size_t admitted = runtime.schedule_pens({ep1, ep2, ep3});
    CHECK(admitted == 5); // 3 for ep1 + 2 for ep2
    CHECK(runtime.pens_allocated() == 6); // 1 for ep3 + 3 for ep1 + 2 for ep2

    CHECK(runtime.pens_for_person(ep1).size() == 3);
    CHECK(runtime.pens_for_person(ep2).size() == 2);
    CHECK(runtime.pens_for_person(ep3).size() == 1);

    // Complete admission for all children
    CHECK(runtime.complete_admission(ep1, 1));
    CHECK(runtime.complete_admission(ep1, 2));
    CHECK(runtime.complete_admission(ep1, 3));
    CHECK(runtime.complete_admission(ep2, 1));
    CHECK(runtime.complete_admission(ep2, 2));

    // 2. Preemption / demotion of Episode 2 (§A.8.2, §A.11):
    // Episode 2 is demoted to logical state. Its physical pen bindings are released.
    auto * ep2_ptr = runtime.episode(ep2);
    CHECK(ep2_ptr != nullptr);
    for (int pen_id : runtime.pens_for_person(ep2)) {
        runtime.release_slot(pen_id);
    }
    server_rerot_episode_demote_to_logical(*ep2_ptr);
    CHECK(runtime.pens_allocated() == 4); // 3 for ep1 + 1 for ep3
    CHECK(runtime.pens_for_person(ep2).empty());

    // Verify Ep1 and Ep3 are completely untouched and can continue
    CHECK(runtime.pens_for_person(ep1).size() == 3);
    CHECK(runtime.pens_for_person(ep3).size() == 1);

    // 3. Cancellation of Episode 1 (§A.15):
    // Hard-abort and erase Episode 1. All its pens must be cleanly returned.
    runtime.hard_abort(ep1, "test_cancelled");
    CHECK(runtime.erase_episode(ep1));
    CHECK(runtime.pens_allocated() == 1); // Only Ep3 remains
    CHECK(runtime.pens_for_person(ep3).size() == 1);

    // Ep3 finishes naturally
    make_terminal(runtime, ep3, 0);
    request_exit(runtime, ep3, 0);
    auto f3 = runtime.finish_frontier(ep3);
    CHECK(f3.natural_final());
    CHECK(f3.final_node == 0);

    CHECK(runtime.erase_episode(ep2));
    CHECK(runtime.erase_episode(ep3));
    CHECK(runtime.pens_allocated() == 0);
}

static void test_hand_seed_and_final_fence_continuation() {
    std::fprintf(stderr, "--- test_hand_seed_and_final_fence_continuation (§16.1, §21.4, §22) ---\n");
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 10, 100);
    runtime.set_pen_capacity(4);

    const uint64_t ep = runtime.adopt_root(1, 1, 0, 10, 0, 1234);
    CHECK(runtime.pens_allocated() == 1);

    // Give parent node a fake hand_seed (mock conv tail bytes)
    const std::vector<uint8_t> expected_seed = { 0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04 };
    auto * parent_node = runtime.node(ep, 0);
    CHECK(parent_node != nullptr);
    parent_node->hand_seed = expected_seed;

    // Fork into 2 children
    CHECK(commit_generated(runtime, ep, 0, 0, "<ol><li>Child 1</li><li>Child 2</li></ol>"));
    runtime.finish_frontier(ep);

    // Freeze parent
    CHECK(runtime.freeze_fork_parent(ep, 0));
    CHECK(runtime.pens_allocated() == 0);

    const auto * ep_ptr = runtime.episode(ep);
    CHECK(ep_ptr != nullptr);
    CHECK(ep_ptr->ready_queue.size() == 2);

    // Verify child descriptors inherited parent hand_seed
    for (const auto child_id : ep_ptr->ready_queue) {
        const auto * c_rt = runtime.node(ep, child_id);
        CHECK(c_rt != nullptr && c_rt->hand_seed == expected_seed);
    }

    // Test Episode Save & Load roundtrip with hand_seed
    server_rerot_state_fingerprints fp;
    fp.caps = LLAMA_REROT_STATE_CAP_REROT | LLAMA_REROT_STATE_CAP_REROT_TREE |
              LLAMA_REROT_STATE_CAP_REROT_PRIVATE | LLAMA_REROT_STATE_CAP_REROT_MTP |
              LLAMA_REROT_STATE_CAP_HYBRID_REC | LLAMA_REROT_STATE_CAP_SPARSE_KV |
              LLAMA_REROT_STATE_CAP_TRIATTENTION;
    fp.model_fp = 0x1111;
    fp.rope_fp = 0x2222;
    fp.tri_fp = 0x3333;

    std::string err;
    const auto blob = server_rerot_episode_save(*ep_ptr, fp, &err);
    CHECK(err.empty());
    CHECK(!blob.empty());

    server_rerot_episode loaded_ep;
    CHECK(server_rerot_episode_load(blob.data(), blob.size(), fp, &loaded_ep, &err));
    CHECK(err.empty());

    // Verify nodes in loaded episode preserve hand_seed
    for (const auto & n : loaded_ep.nodes) {
        if (n.id == 0 || n.id == 1 || n.id == 2) {
            CHECK(n.hand_seed == expected_seed);
        }
    }

    // Schedule pens for children
    const size_t admitted = runtime.schedule_pens({ep});
    CHECK(admitted == 2);
    CHECK(runtime.pens_allocated() == 2);

    // Complete admission for child 1 (node 1) and child 2 (node 2)
    CHECK(runtime.complete_admission(ep, 1));
    CHECK(runtime.complete_admission(ep, 2));

    // Finish child 1
    make_terminal(runtime, ep, 1);
    request_exit(runtime, ep, 1);
    auto f1 = runtime.finish_frontier(ep);
    CHECK(f1.retired.size() == 1 && f1.retired[0] == 1);
    CHECK(f1.released_slots.size() == 1);

    // Child 2 marks exit intent and initiates final fence
    make_terminal(runtime, ep, 2);
    request_exit(runtime, ep, 2);
    auto f2 = runtime.finish_frontier(ep);
    CHECK(f2.natural_final());
    CHECK(f2.final_node == 2);

    auto * mut_ep = runtime.episode(ep);
    CHECK(mut_ep && mut_ep->finalizing);

    std::vector<uint32_t> ordered_runs;
    CHECK(runtime.refresh_final_fence(ep, 2, &ordered_runs));
    CHECK(mut_ep->fence_refreshed);

    // Complete serial tail transition
    CHECK(runtime.complete_serial_tail(ep, 2));
    CHECK(mut_ep->serial_tail);
    CHECK(mut_ep->serial_node == 2);

    CHECK(runtime.erase_episode(ep));
}

static void test_frontier_boundary_pen_yield_and_resume() {
    // §B.8.4, §B.15, Phase 1 & 2: Frontier-boundary pen yield and resume
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 10, 100);
    runtime.set_pen_capacity(2);
    CHECK(runtime.pen_capacity() == 2);
    CHECK(runtime.pens_allocated() == 0);
    CHECK(runtime.pens_suspended() == 0);

    const uint64_t ep = runtime.adopt_root(1, 1, 0, 10, 0, 1000);
    CHECK(runtime.pens_allocated() == 1);

    CHECK(commit_generated(runtime, ep, 0, 0, "<ol><li>Child 1</li><li>Child 2</li></ol>"));
    runtime.finish_frontier(ep);
    CHECK(runtime.freeze_fork_parent(ep, 0));
    CHECK(runtime.pens_allocated() == 0);

    // Schedule: both children get pens (0 and 1)
    const size_t admitted = runtime.schedule_pens({ep});
    CHECK(admitted == 2);
    CHECK(runtime.pens_allocated() == 2);
    CHECK(runtime.pens_running() == 0); // STARTING

    // Complete admission for both
    CHECK(runtime.complete_admission(ep, 1));
    CHECK(runtime.complete_admission(ep, 2));
    CHECK(runtime.pens_running() == 2);
    CHECK(runtime.pens_suspended() == 0);

    const auto * ep_ptr = runtime.episode(ep);
    CHECK(ep_ptr->running.count(1) == 1);
    CHECK(ep_ptr->running.count(2) == 1);
    CHECK(ep_ptr->suspended.empty());

    // Child 1 yields pen 0 at frontier boundary (§B.8.4)
    CHECK(runtime.suspend_pen(0));
    CHECK(runtime.pens_suspended() == 1);
    CHECK(runtime.pens_running() == 1);
    CHECK(runtime.pens_allocated() == 1);
    CHECK(ep_ptr->running.count(1) == 0);
    CHECK(ep_ptr->running.count(2) == 1);
    CHECK(ep_ptr->suspended.count(1) == 1);
    CHECK(runtime.node(ep, 1)->pen_id == -1);
    CHECK(runtime.node(ep, 1)->physical_slot == -1);
    CHECK(ep_ptr->document.node(1)->state == llama_rerot_node_state::ready_suspended);
    CHECK(runtime.has_ready_nodes(ep));

    // Finish a normal frontier without exits: Child 1 stays suspended
    auto f1 = runtime.finish_frontier(ep);
    CHECK(!f1.natural_final());
    CHECK(f1.final_node == LLAMA_REROT_NODE_INVALID);

    // Now resume Child 1 with pen 0
    auto pen0_opt = runtime.allocate_pen(ep, ep, 1);
    CHECK(pen0_opt.has_value() && *pen0_opt == 0);
    CHECK(runtime.resume_pen(ep, 1, 0, 10));
    CHECK(runtime.pens_suspended() == 0);
    CHECK(runtime.pens_running() == 2);
    CHECK(runtime.pens_allocated() == 2);
    CHECK(ep_ptr->suspended.empty());
    CHECK(ep_ptr->running.count(1) == 1);
    CHECK(runtime.node(ep, 1)->pen_id == 0);

    // Both children mark exit intent
    request_exit(runtime, ep, 1);
    request_exit(runtime, ep, 2);
    auto f2 = runtime.finish_frontier(ep);
    CHECK(f2.natural_final());
    CHECK(f2.final_node == 2); // tie-break survivor

    CHECK(runtime.erase_episode(ep));
    CHECK(runtime.pens_allocated() == 0);
    CHECK(runtime.pens_suspended() == 0);
}

static void test_multi_person_b_greater_than_p_fairness() {
    // §B.8.4, §B.8.2: 4 runnable child lanes across 2 people competing for P=2 pens
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 10, 100);
    runtime.set_pen_capacity(2);
    CHECK(runtime.pen_capacity() == 2);

    const uint64_t ep1 = runtime.adopt_root(1, 1, 0, 10, 0, 101);
    const uint64_t ep2 = runtime.adopt_root(2, 2, 1, 11, 0, 102);
    CHECK(runtime.pens_allocated() == 2);

    // Fork both into 2 children each
    CHECK(commit_generated(runtime, ep1, 0, 0, "<ol><li>P1-A</li><li>P1-B</li></ol>"));
    runtime.finish_frontier(ep1);
    CHECK(runtime.freeze_fork_parent(ep1, 0));

    CHECK(commit_generated(runtime, ep2, 0, 0, "<ol><li>P2-A</li><li>P2-B</li></ol>"));
    runtime.finish_frontier(ep2);
    CHECK(runtime.freeze_fork_parent(ep2, 0));
    CHECK(runtime.pens_allocated() == 0);

    // Schedule across {ep1, ep2}:
    // Pass 1 allocates pen 0 to ep1 (child 1) and pen 1 to ep2 (child 1)
    const size_t admitted1 = runtime.schedule_pens({ep1, ep2});
    CHECK(admitted1 == 2);
    CHECK(runtime.pens_allocated() == 2);
    CHECK(runtime.pens_for_person(ep1).size() == 1);
    CHECK(runtime.pens_for_person(ep2).size() == 1);

    // Complete admission for both
    CHECK(runtime.complete_admission(ep1, 1));
    CHECK(runtime.complete_admission(ep2, 1));
    CHECK(runtime.pens_running() == 2);

    // Both ep1 and ep2 still have 1 child in ready_queue (P1-B and P2-B)
    CHECK(runtime.episode(ep1)->ready_queue.size() == 1);
    CHECK(runtime.episode(ep2)->ready_queue.size() == 1);

    // Ep1 yields pen 0 at frontier boundary (§B.8.4)
    const auto p1_pens = runtime.pens_for_person(ep1);
    CHECK(p1_pens.size() == 1);
    CHECK(runtime.suspend_pen(p1_pens[0]));
    CHECK(runtime.pens_suspended() == 1);
    CHECK(runtime.pens_allocated() == 1);
    CHECK(runtime.pens_for_person(ep1).empty());

    // Schedule across {ep2}: ep2 gets pen 0 for its second child (child 2)
    const size_t admitted2 = runtime.schedule_pens({ep2});
    CHECK(admitted2 == 1);
    CHECK(runtime.pens_allocated() == 2);
    CHECK(runtime.pens_for_person(ep2).size() == 2); // ep2 now owns both pens
    CHECK(runtime.episode(ep2)->ready_queue.empty());
    CHECK(runtime.complete_admission(ep2, 2));
    CHECK(runtime.pens_running() == 2);

    // Ep2 child 1 and child 2 exit
    request_exit(runtime, ep2, 1);
    request_exit(runtime, ep2, 2);
    auto f2 = runtime.finish_frontier(ep2);
    CHECK(f2.natural_final());
    CHECK(runtime.erase_episode(ep2));
    CHECK(runtime.pens_allocated() == 0); // both pens free

    // Now ep1 schedules: resumes suspended child 1 and admits child 2
    CHECK(runtime.pens_suspended() == 1);
    CHECK(runtime.episode(ep1)->ready_queue.size() == 1);
    const size_t admitted3 = runtime.schedule_pens({ep1});
    CHECK(admitted3 == 2); // 1 resumed + 1 admitted
    CHECK(runtime.pens_allocated() == 2);
    CHECK(runtime.pens_suspended() == 0); // suspended child 1 was resumed!
    CHECK(runtime.episode(ep1)->ready_queue.empty()); // child 2 admitted!
    CHECK(runtime.complete_admission(ep1, 2));
    CHECK(runtime.pens_running() == 2);

    // Ep1 completes both children
    request_exit(runtime, ep1, 1);
    request_exit(runtime, ep1, 2);
    auto f1 = runtime.finish_frontier(ep1);
    CHECK(f1.natural_final());
    CHECK(runtime.erase_episode(ep1));

    CHECK(runtime.pens_allocated() == 0);
    CHECK(runtime.pens_suspended() == 0);
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

static void test_pen_capacity_and_multi_episode_allocation() {
    // 1. Constructor auto-initializes pen capacity from first_internal_seq
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 4, 32);
    CHECK(runtime.pen_capacity() == 4);
    CHECK(runtime.pens_allocated() == 0);
    CHECK(runtime.pens_running() == 0);
    CHECK(runtime.pens_suspended() == 0);

    // 2. Explicit capacity expansion
    runtime.set_pen_capacity(6);
    CHECK(runtime.pen_capacity() == 6);
    for (int32_t i = 0; i < 6; ++i) {
        const auto * p = runtime.pen(i);
        CHECK(p != nullptr);
        CHECK(p->id == i);
        CHECK(p->state == server_pen_state::free);
    }

    // 3. Adopt root for Episode 1 on slot 0
    const uint64_t ep1 = runtime.adopt_root(10, 10, 0, 0, 0);
    CHECK(ep1 != 0);
    CHECK(runtime.pens_allocated() == 1);
    CHECK(runtime.pens_running() == 1);
    const auto * p0 = runtime.pen(0);
    CHECK(p0 && p0->state == server_pen_state::running);
    CHECK(p0 && p0->person == ep1 && p0->episode_id == ep1);

    // 4. Adopt root for Episode 2 on slot 1
    const uint64_t ep2 = runtime.adopt_root(20, 20, 1, 1, 0);
    CHECK(ep2 != 0);
    CHECK(ep2 != ep1);
    CHECK(runtime.pens_allocated() == 2);
    CHECK(runtime.pens_running() == 2);
    const auto * p1 = runtime.pen(1);
    CHECK(p1 && p1->state == server_pen_state::running);
    CHECK(p1 && p1->person == ep2 && p1->episode_id == ep2);

    CHECK(runtime.pens_for_person(ep1).size() == 1);
    CHECK(runtime.pens_for_person(ep2).size() == 1);

    // 5. Suspend Pen 0 (Episode 1 yields pen row back to free pool, node enters suspended set)
    CHECK(runtime.suspend_pen(0));
    CHECK(runtime.pens_suspended() == 1);
    CHECK(runtime.pens_running() == 1);
    CHECK(runtime.pens_allocated() == 1); // Pen 0 is free for another lane/person; only Pen 1 allocated
    CHECK(runtime.pen(0)->state == server_pen_state::free);

    // 6. Release slot 1 (Episode 2 finishes)
    runtime.release_slot(1);
    CHECK(runtime.pen(1)->state == server_pen_state::free);
    CHECK(runtime.pens_for_person(ep2).empty());
    CHECK(runtime.pens_running() == 0);

    // 7. Resume Episode 1 suspended node onto slot 1
    CHECK(runtime.resume_pen(ep1, 0, 1, 1));
    CHECK(runtime.pens_suspended() == 0);
    CHECK(runtime.pens_running() == 1);
    CHECK(runtime.pens_allocated() == 1);
    CHECK(runtime.pen(1)->state == server_pen_state::running);
    CHECK(runtime.pen(1)->person == ep1);

    // Cleanup
    CHECK(runtime.erase_episode(ep1));
    CHECK(runtime.erase_episode(ep2));
}

static void test_recurrent_only_pressure_isolation() {
    std::fprintf(stderr, "--- test_recurrent_only_pressure_isolation (§B.9, Gate 13 of DoD A.30) ---\n");
    // P=2 pens, 2 episodes running
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 2, 32);
    const uint64_t ep1 = runtime.adopt_root(10, 10, 0, 0, 0, 301);
    const uint64_t ep2 = runtime.adopt_root(11, 11, 1, 1, 0, 302);
    CHECK(runtime.pens_allocated() == 2);
    CHECK(runtime.pens_running() == 2);

    auto * ep1_ptr = runtime.episode(ep1);
    auto * ep2_ptr = runtime.episode(ep2);
    CHECK(ep1_ptr && ep2_ptr);

    // Episode 1 has 400 public tokens, Episode 2 has 300 public tokens
    const auto r1 = ep1_ptr->document.append_run(0, llama_rerot_visibility::public_live, 0, 400, 1);
    const auto r2 = ep2_ptr->document.append_run(0, llama_rerot_visibility::public_live, 0, 300, 1);
    CHECK(r1 != LLAMA_REROT_RUN_INVALID && r2 != LLAMA_REROT_RUN_INVALID);

    // Episode 1 forks 6 children. Total demand = 7 pens, but P=2.
    // This is purely recurrent pen shortage.
    std::string list_xml = "<ol>";
    for (int i = 0; i < 6; ++i) {
        list_xml += "<li>Recurrent task " + std::to_string(i) + "</li>";
    }
    list_xml += "</ol>";
    CHECK(commit_generated(runtime, ep1, 0, 400, list_xml));
    runtime.finish_frontier(ep1);
    CHECK(runtime.freeze_fork_parent(ep1, 0));

    // Queue depth must be 6
    CHECK(ep1_ptr->ready_queue.size() == 6);
    CHECK(runtime.pen_queue_depth() == 6);

    // Invariant (Gate 13 of DoD A.30): recurrent-only shortage must NOT trigger TriAttention KV reclaim!
    // All token counts and runs across ep1 and ep2 remain 100% intact.
    CHECK(ep1_ptr->document.run(r1)->token_count == 400);
    CHECK(ep2_ptr->document.run(r2)->token_count == 300);

    // Try scheduling: 1 pen was freed by parent freeze, so 1 child admitted
    size_t admitted = runtime.schedule_pens({ep1, ep2});
    CHECK(admitted == 1);
    CHECK(runtime.pens_allocated() == 2);
    CHECK(ep1_ptr->ready_queue.size() == 5);
    CHECK(runtime.pen_queue_depth() == 5);

    // Invariant: still no KV reclaim!
    CHECK(ep1_ptr->document.run(r1)->token_count == 400);
    CHECK(ep2_ptr->document.run(r2)->token_count == 300);

    CHECK(runtime.erase_episode(ep1));
    CHECK(runtime.erase_episode(ep2));
}

static void test_multi_episode_concurrent_final_fence_and_coordinate_freeze() {
    std::fprintf(stderr, "--- test_multi_episode_concurrent_final_fence_and_coordinate_freeze (§21.4, §22) ---\n");
    // 4 pens
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 4, 32);
    const uint64_t ep1 = runtime.adopt_root(20, 20, 0, 0, 0, 401);
    const uint64_t ep2 = runtime.adopt_root(21, 21, 1, 1, 0, 402);
    CHECK(runtime.pens_allocated() == 2);

    auto * ep1_ptr = runtime.episode(ep1);
    auto * ep2_ptr = runtime.episode(ep2);
    CHECK(ep1_ptr && ep2_ptr);

    // Episode 1 finishes planning with N=1 (no fork)
    std::string n1_xml = "<ol><li>Single task</li></ol>";
    CHECK(commit_generated(runtime, ep1, 0, 0, n1_xml));
    runtime.finish_frontier(ep1);

    // Episode 1 survivor plans a generated token and emits </think>
    CHECK(commit_generated(runtime, ep1, 0, 10, "conclusion"));
    runtime.finish_frontier(ep1);

    auto * node1 = runtime.node(ep1, 0);
    CHECK(node1 != nullptr);
    node1->exit_intent = true;
    ep1_ptr->finalizing = true;

    // Refresh final fence on Episode 1
    std::vector<uint32_t> fence_runs;
    CHECK(runtime.refresh_final_fence(ep1, 0, &fence_runs));
    CHECK(!fence_runs.empty());
    CHECK(runtime.complete_serial_tail(ep1, 0));

    // Validate serial tail state and execute coordinate freeze on Episode 1
    std::string err;
    CHECK(runtime.validate_serial_tail_state(ep1, 0, &err));
    const uint64_t ep1_old_epoch = ep1_ptr->layout_epoch;
    CHECK(runtime.freeze_serial_coordinates(ep1, 0, &err));
    CHECK(ep1_ptr->layout_epoch == ep1_old_epoch + 1);

    // While Episode 1 is in serial tail, Episode 2 continues running concurrently
    CHECK(ep2_ptr->serial_tail == false);
    CHECK(commit_generated(runtime, ep2, 0, 0, "concurrent reasoning in ep2"));
    runtime.finish_frontier(ep2);

    // Invariant: Episode 1 serial tail is not corrupted by Episode 2 execution
    CHECK(runtime.validate_serial_tail_state(ep1, 0, &err));

    // Now Episode 2 finishes its reasoning and reaches final fence
    auto * node2 = runtime.node(ep2, 0);
    CHECK(node2 != nullptr);
    node2->exit_intent = true;
    ep2_ptr->finalizing = true;

    std::vector<uint32_t> ep2_fence_runs;
    CHECK(runtime.refresh_final_fence(ep2, 0, &ep2_fence_runs));
    CHECK(runtime.complete_serial_tail(ep2, 0));
    CHECK(runtime.validate_serial_tail_state(ep2, 0, &err));
    CHECK(runtime.freeze_serial_coordinates(ep2, 0, &err));

    // Both episodes reached serial tail independently
    CHECK(ep1_ptr->serial_tail == true);
    CHECK(ep2_ptr->serial_tail == true);

    CHECK(runtime.erase_episode(ep1));
    CHECK(runtime.erase_episode(ep2));
}

int main() {
    std::fprintf(stderr, "=== RERoT Runtime Tests ===\n");
    test_recurrent_only_pressure_isolation();
    test_multi_episode_concurrent_final_fence_and_coordinate_freeze();
    test_pen_capacity_and_multi_episode_allocation();
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
    test_people_pen_scheduler();
    test_context_multi_episode_isolation();
    test_hand_seed_and_final_fence_continuation();
    test_shared_prefix_multi_branch_union_and_preemption();
    test_multi_person_multi_pen_bxp_stress();
    test_triattention_multi_person_pressure();
    test_frontier_boundary_pen_yield_and_resume();
    test_multi_person_b_greater_than_p_fairness();
    std::fprintf(stderr, "=== Results: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
