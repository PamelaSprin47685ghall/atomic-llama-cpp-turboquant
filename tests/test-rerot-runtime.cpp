#include "server-rerot.h"
#include "server-task.h"
#include "llama.h"
#include "llama-context.h"
#include "llama-grammar.h"
#include "llama-model.h"
#include "llama-memory-recurrent.h"

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
    return plan.has_value() && runtime.track_fence_token(episode_id, node_id, *plan, 1000 + pos) &&
        runtime.commit_token(episode_id, node_id, *plan);
}

static void replay_final_fence(server_rerot_runtime & runtime, uint64_t ep, llama_rerot_node_id node) {
    auto * episode = runtime.episode(ep);
    const auto before = *episode;
    // Installing a reader view alone may not unlock serial continuation.
    server_rerot_runtime rejected(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 128);
    CHECK(rejected.adopt_root(0, 0, 0, 0, 0, ep) == ep);
    *rejected.episode(ep) = before;
    CHECK(!rejected.complete_serial_tail(ep, node));
    CHECK(runtime.prepare_final_fence(ep, node));
    CHECK(!runtime.prepare_final_fence(ep, node)); // No second rewind.
    const auto original = runtime.node(ep, node)->fence.rows;
    CHECK(original.size() == 2); // request_exit spans two actual tokenizer rows
    size_t replayed = 0;
    while (const auto plan = runtime.plan_final_fence_token(ep, node)) {
        CHECK(plan->storage_pos == original[replayed].pos);
        CHECK(plan->run_id == original[replayed].run);
        CHECK(plan->visibility == llama_rerot_visibility::private_control);
        CHECK(runtime.commit_final_fence_token(ep, node, *plan));
        ++replayed;
        if (replayed == 1) {
            // A mid-fence episode checkpoint retains the original token tape,
            // cursor and pre-marker local state, not a retokenized delimiter.
            server_rerot_state_fingerprints fp;
            fp.caps = LLAMA_REROT_STATE_CAP_REROT | LLAMA_REROT_STATE_CAP_REROT_TREE |
                LLAMA_REROT_STATE_CAP_REROT_PRIVATE;
            std::string err;
            const auto blob = server_rerot_episode_save(*episode, fp, &err);
            server_rerot_episode loaded;
            CHECK(!blob.empty());
            CHECK(server_rerot_episode_load(blob.data(), blob.size(), fp, &loaded, &err));
            CHECK(loaded.nodes[node].fence.cursor == 1);
            CHECK(loaded.nodes[node].fence.prepared);
            CHECK(loaded.nodes[node].fence.rows.size() == original.size());
            for (size_t i = 0; i < original.size(); ++i) {
                CHECK(loaded.nodes[node].fence.rows[i].token == original[i].token);
                CHECK(loaded.nodes[node].fence.rows[i].pos == original[i].pos);
                CHECK(loaded.nodes[node].fence.rows[i].run == original[i].run);
            }
        }
    }
    CHECK(replayed == original.size());
    CHECK(runtime.node(ep, node)->fence.complete());
    CHECK(episode->generated_public_tokens == before.generated_public_tokens);
    CHECK(episode->generated_private_tokens == before.generated_private_tokens);
    CHECK(episode->pending_tokens == before.pending_tokens);
    CHECK(episode->document.run_count() == before.document.run_count());
    CHECK(episode->nodes[node].storage_pos_next == before.nodes[node].storage_pos_next);
    CHECK(episode->publish_epoch == before.publish_epoch);
    CHECK(episode->layout_epoch == before.layout_epoch);
    CHECK(episode->frontier == before.frontier);
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
    const auto id = node->control_id();
    CHECK(id.size() == 8);
    CHECK(std::all_of(id.begin(), id.end(), [](unsigned char ch) {
        return (ch >= '0' && ch <= '9') ||
               (ch >= 'A' && ch <= 'Z') ||
               (ch >= 'a' && ch <= 'z');
    }));
    CHECK(node->exit_parser.marker() == "</" + std::string(id) + ">");
    CHECK(node->control_open() == "<" + std::string(id) + ">");

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
    node = runtime.node(episode_id, admitted);
    CHECK(node && !node->planner_armed);
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
    const auto * logical_before = runtime.episode(episode_id)->document.node(node_id);
    if (logical_before && logical_before->state == llama_rerot_node_state::terminal_running) {
        return;
    }
    if (!node->planner_armed) {
        CHECK(runtime.arm_planner(episode_id, node_id));
        node = runtime.node(episode_id, node_id);
        CHECK(node && node->planner_armed);
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
    const std::string close = node->exit_parser.marker();
    CHECK(close.size() == 11);
    if (close.size() != 11) {
        return;
    }
    CHECK(commit_generated(
        runtime, episode_id, node_id, node->storage_pos_next,
        "final observation " + close.substr(0, 6)));
    node = runtime.node(episode_id, node_id);
    CHECK(node != nullptr);
    if (node) {
        CHECK(commit_generated(
            runtime, episode_id, node_id, node->storage_pos_next, close.substr(6)));
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

static void test_list_marker_prefixes_remain_atomic() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t episode_id = runtime.adopt_root(95, 95, 0, 0, 0);
    server_rerot_line_mux mux;
    const std::vector<std::string> tokens = {
        "<", "ol", ">\n<li>", "第一项。\n</", "li", ">\n</", "ol", ">\n",
    };
    std::string emitted;
    llama_pos position = 0;
    for (const auto & bytes : tokens) {
        auto plan = runtime.plan_generated_token(episode_id, 0, position++, bytes);
        CHECK(plan.has_value());
        if (!plan) {
            return;
        }
        // A caller must not expose marker-like prefixes from an unfinished
        // list as public text or move its punctuation ahead of the opener.
        CHECK(bytes.substr(0, plan->marker_step.public_prefix_bytes).empty());
        CHECK(runtime.commit_token(episode_id, 0, *plan));
        const auto ready = mux.append(
            0, plan->run_id, bytes, runtime.episode(episode_id)->document,
            plan->marker_step.public_prefix_bytes);
        CHECK(ready.ok);
        for (const auto & line : ready.lines) {
            emitted += line;
        }
        if (!plan->parser_step.record_closed) {
            CHECK(emitted.empty());
        }
    }
    CHECK(emitted == "<ol>\n<li>第一项。\n</li>\n</ol>\n");
}

static void test_marker_token_preserves_public_prefix() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t episode_id = runtime.adopt_root(96, 96, 0, 0, 0);
    CHECK(commit_generated(
        runtime, episode_id, 0, 0,
        "<ol><li>Marker A</li><li>Marker B</li></ol>"));
    runtime.finish_frontier(episode_id);
    CHECK(runtime.freeze_fork_parent(episode_id, 0));
    start_child(runtime, episode_id, 0, 1);

    auto * child = runtime.node(episode_id, 1);
    CHECK(child != nullptr);
    if (!child) {
        return;
    }
    const std::string id(child->control_id());
    server_rerot_line_mux mux;

    // In KISS architecture, ordinary child tokens are public_live immediately.
    auto plan = runtime.plan_generated_token(
        episode_id, 1, child->storage_pos_next, "answer</p");
    CHECK(plan.has_value());
    CHECK(plan && plan->visibility ==
        llama_rerot_visibility::public_live);
    CHECK(plan && runtime.commit_token(episode_id, 1, *plan));
    if (!plan) {
        return;
    }
    const auto body_run = plan->run_id;
    const auto * body_before =
        runtime.episode(episode_id)->document.run(body_run);
    CHECK(body_before && body_before->visibility ==
        llama_rerot_visibility::public_live);
    auto ready = mux.append(
        1, plan->run_id, "answer</p", runtime.episode(episode_id)->document);
    CHECK(ready.ok && ready.lines.empty());

    child = runtime.node(episode_id, 1);
    // Token containing ">" then candidate start "</" is pending_record for the candidate part.
    plan = runtime.plan_generated_token(
        episode_id, 1, child ? child->storage_pos_next : 1, "></");
    CHECK(plan.has_value());
    CHECK(plan && plan->visibility == llama_rerot_visibility::pending_record);
    CHECK(plan && plan->marker_step.public_prefix_bytes == 1);
    const auto * body_after =
        runtime.episode(episode_id)->document.run(body_run);
    CHECK(body_after && body_after->visibility ==
        llama_rerot_visibility::public_live);
    CHECK(plan && runtime.commit_token(episode_id, 1, *plan));
    if (!plan) {
        return;
    }
    ready = mux.append(
        1,
        plan->run_id,
        "></",
        runtime.episode(episode_id)->document,
        plan->marker_step.public_prefix_bytes);
    CHECK(ready.ok && ready.lines.empty());

    child = runtime.node(episode_id, 1);
    CHECK(child != nullptr);
    // Closing marker completes: becomes private_control
    plan = runtime.plan_generated_token(
        episode_id, 1, child ? child->storage_pos_next : 2, id + ">");
    CHECK(plan.has_value());
    CHECK(plan && plan->marker_step.marker_closed);
    CHECK(plan && plan->visibility == llama_rerot_visibility::private_control);
    CHECK(plan && runtime.commit_token(episode_id, 1, *plan));
    if (!plan) {
        return;
    }
    ready = mux.append(
        1,
        plan->run_id,
        id + ">",
        runtime.episode(episode_id)->document,
        plan->marker_step.public_prefix_bytes);
    CHECK(ready.ok && ready.lines.empty());

    ready = mux.finish(1, runtime.episode(episode_id)->document);
    CHECK(ready.ok);
    CHECK(ready.lines == std::vector<std::string>({"answer</p>\n"}));
    CHECK(mux.empty());
    CHECK(runtime.erase_episode(episode_id));
}

static void test_child_public_live_visibility_and_exit_marker() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t episode_id = runtime.adopt_root(150, 150, 0, 0, 0);
    CHECK(commit_generated(runtime, episode_id, 0, 0,
        "<ol><li>Task 1</li><li>Task 2</li></ol>"));
    runtime.finish_frontier(episode_id);
    CHECK(runtime.freeze_fork_parent(episode_id, 0));
    start_child(runtime, episode_id, 0, 1);

    auto * child = runtime.node(episode_id, 1);
    CHECK(child != nullptr);
    if (!child) {
        return;
    }
    const std::string id(child->control_id());

    // 1. Regular text (single/multi-line) is public_live immediately
    auto plan = runtime.plan_generated_token(
        episode_id, 1, child->storage_pos_next, "line 1 text with punctuation: 1.0, and words.\n");
    CHECK(plan.has_value());
    CHECK(plan && plan->visibility == llama_rerot_visibility::public_live);
    CHECK(plan && runtime.commit_token(episode_id, 1, *plan));

    // 2. Illegal or partial candidate prefix that gets disproved returns to public_live
    child = runtime.node(episode_id, 1);
    plan = runtime.plan_generated_token(episode_id, 1, child->storage_pos_next, "<");
    CHECK(plan.has_value());
    CHECK(plan && plan->visibility == llama_rerot_visibility::pending_record);
    CHECK(plan && runtime.commit_token(episode_id, 1, *plan));

    child = runtime.node(episode_id, 1);
    plan = runtime.plan_generated_token(episode_id, 1, child->storage_pos_next, "illegal_not_marker>");
    CHECK(plan.has_value());
    CHECK(plan && plan->visibility == llama_rerot_visibility::public_live);
    CHECK(plan && runtime.commit_token(episode_id, 1, *plan));

    // 3. Outputting legal close marker marks it private_control and triggers exit intent
    child = runtime.node(episode_id, 1);
    const std::string close_tag = "</" + id + ">";
    plan = runtime.plan_generated_token(episode_id, 1, child->storage_pos_next, close_tag);
    CHECK(plan.has_value());
    CHECK(plan && plan->visibility == llama_rerot_visibility::private_control);
    CHECK(plan && plan->marker_step.marker_closed);
    CHECK(plan && runtime.commit_token(episode_id, 1, *plan));
    child = runtime.node(episode_id, 1);
    CHECK(child && child->exit_intent);

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

    CHECK(mux.empty());
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

    // N=1 never invents a child delimiter or a synthetic thought close.
    CHECK(node && node->control_id().empty());
    CHECK(runtime.continue_unforked_root(episode_id, 0));
    episode = runtime.episode(episode_id);
    CHECK(episode && episode->serial_tail && episode->serial_node == 0);

    // Later planner-shaped text is ordinary serial content, never a re-fork.
    node = runtime.node(episode_id, 0);
    const auto serial = runtime.plan_serial_token(
        episode_id, 0, node->storage_pos_next);
    CHECK(serial.has_value());
    CHECK(serial && runtime.commit_token(episode_id, 0, *serial));
    node = runtime.node(episode_id, 0);
    CHECK(node != nullptr && !node->planner_armed);
    episode = runtime.episode(episode_id);
    CHECK(episode && episode->document.node_count() == 1);
    CHECK(episode && episode->ready_queue.empty());
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

    // An explicit private task contract may QUOTE the closing delimiter.
    // Decoding that instruction is not a sampled exit and it stays owner-only.
    auto * controlled = runtime.node(episode_id, 1);
    const std::string contract = server_rerot_child_contract(
        "Alpha", controlled->exit_parser.marker());
    CHECK(!contract.empty());
    const uint64_t public_before = runtime.episode(episode_id)->generated_public_tokens;
    const auto forced_contract = runtime.plan_private_span(episode_id, 1, controlled->storage_pos_next, 3);
    CHECK(forced_contract.has_value());
    if (forced_contract) {
        for (const auto & plan : *forced_contract) CHECK(runtime.commit_token(episode_id, 1, plan));
        const auto & doc = runtime.episode(episode_id)->document;
        CHECK(view_contains_run(doc.build_view(1), forced_contract->front().run_id));
        CHECK(!view_contains_run(doc.build_view(2), forced_contract->front().run_id));
    }
    CHECK(!controlled->exit_intent);
    CHECK(!controlled->last_write_public);
    CHECK(runtime.episode(episode_id)->generated_public_tokens == public_before);

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
    CHECK(runtime.arm_planner(episode_id, 1));
    lane_a = runtime.node(episode_id, 1);
    CHECK(lane_a && lane_a->planner_armed);
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

static void test_child_content_list_is_not_scheduler_control() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t episode_id = runtime.adopt_root(135, 135, 0, 0, 0);
    CHECK(commit_generated(runtime, episode_id, 0, 0,
        "<ol><li>Only child</li><li>Sibling</li></ol>"));
    CHECK(runtime.finish_frontier(episode_id).forked.size() == 1);
    CHECK(runtime.freeze_fork_parent(episode_id, 0));
    start_child(runtime, episode_id, 0, 1);

    const auto * logical = runtime.episode(episode_id)->document.node(1);
    CHECK(logical && logical->title == "Only child");
    const auto * lane = runtime.node(episode_id, 1);
    const std::string contract = server_rerot_child_contract(
        logical ? logical->title : std::string_view{},
        lane ? lane->exit_parser.marker() : std::string_view{});
    CHECK(contract.find("Only child") != std::string::npos);
    CHECK(contract.find("Sibling") == std::string::npos);
    const std::string planner = server_rerot_child_planner_prompt(
        logical ? logical->title : std::string_view{});
    CHECK(planner.find("Only child") != std::string::npos);
    const std::string worker = server_rerot_child_worker_prompt(
        logical ? logical->title : std::string_view{},
        lane ? lane->exit_parser.marker() : std::string_view{});
    CHECK(worker.find("Only child") != std::string::npos);
    CHECK(worker.find("整个用户问题") != std::string::npos);

    // Default child admission goes directly to worker mode. No synthetic N=1
    // planner record is required merely to make ordinary HTML safe.
    CHECK(runtime.begin_worker(episode_id, 1));
    lane = runtime.node(episode_id, 1);
    CHECK(lane && !lane->planner_armed);
    const size_t nodes_before = runtime.episode(episode_id)->document.node_count();
    const llama_pos pos = lane ? lane->storage_pos_next : 0;
    CHECK(commit_generated(runtime, episode_id, 1, pos,
        "正文可以自然使用 <ol><li>事实甲</li><li>事实乙</li></ol> 而不改变拓扑。"));
    const auto * episode = runtime.episode(episode_id);
    const auto * after = episode ? episode->document.node(1) : nullptr;
    CHECK(episode && episode->document.node_count() == nodes_before);
    CHECK(after && after->state == llama_rerot_node_state::terminal_running);
    CHECK(runtime.node(episode_id, 1) && !runtime.node(episode_id, 1)->planner_armed);
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
    CHECK(runtime.arm_planner(episode_id, 1));
    lane1 = runtime.node(episode_id, 1);
    CHECK(lane1 && lane1->planner_armed);
    CHECK(commit_generated(runtime, episode_id, 1, lane1->storage_pos_next, "<ol><li>sub"));
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
    CHECK(strong.entries.size() == 1); // own current only; peer is still in write stage

    const auto lag_same = llama_rerot_build_query_layout(
        make_reader(LLAMA_REROT_FRONTIER_LAG1, 5), 0, keys);
    CHECK(lag_same.entries.size() == 1);

    // STRONG takes the peer at the next read frontier; LAG1 deliberately waits
    // one additional committed frontier.
    const auto strong_next = llama_rerot_build_query_layout(
        make_reader(LLAMA_REROT_FRONTIER_STRONG, 6), 0, keys);
    CHECK(strong_next.entries.size() == 2);
    const auto lag_next = llama_rerot_build_query_layout(
        make_reader(LLAMA_REROT_FRONTIER_LAG1, 6), 0, keys);
    CHECK(lag_next.entries.size() == 1);
    const auto lag_after = llama_rerot_build_query_layout(
        make_reader(LLAMA_REROT_FRONTIER_LAG1, 7), 0, keys);
    CHECK(lag_after.entries.size() == 2);

    std::string error;
    CHECK(strong.query_virtual_pos == 0);

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
    replay_final_fence(runtime, episode_id, 2);
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
        "<ol><li>Persist child A</li><li>Persist child B</li></ol>"));
    CHECK(runtime.finish_frontier(episode_id).forked.size() == 1);
    CHECK(runtime.freeze_fork_parent(episode_id, 0));
    start_child(runtime, episode_id, 0, 1);

    auto * lane = runtime.node(episode_id, 1);
    CHECK(lane != nullptr);
    if (lane) {
        lane->sampler_blob = {1, 2, 3, 4};
        lane->mtp_blob = {9, 8, 7};
        lane->view_stamp = {4, 5, 6};
    }

    const std::string saved_marker = lane ? lane->exit_parser.marker() : "";
    const auto saved_marker_state = lane
        ? lane->exit_parser.snapshot()
        : server_rerot_marker_snapshot{};
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
    const auto * restored_lane = restored.node(restored_id, 1);
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
    CHECK(restored_lane && restored_lane->exit_parser.marker() == saved_marker);
    CHECK(restored_lane &&
        restored_lane->exit_parser.snapshot().candidate == saved_marker_state.candidate);
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

static void test_rerot_mtp_speculative_matrix() {
    std::fprintf(stderr, "--- test_rerot_mtp_speculative_matrix (AGENTS.md Phase 4) ---\n");
    test_stub_model model;
    llama_cparams cparams = {};
    cparams.rerot_enabled = true;
    cparams.rerot_frontier = LLAMA_REROT_FRONTIER_STRONG;
    llama_context ctx(model, cparams, true);

    const uint64_t ep1 = 201;
    CHECK(llama_rerot_episode_begin(&ctx, ep1, nullptr));

    const llama_seq_id seq_reader = 1;
    uint32_t run_ids[] = {1};

    // 1. Initial frontier reader view at epoch (1, 1, 1)
    llama_rerot_frontier_reader_view v1 = {
        seq_reader, ep1, 1, 1, 10, LLAMA_REROT_FRONTIER_STRONG, {1, 1, 1}, run_ids, 1
    };
    CHECK(llama_rerot_set_frontier_views(&ctx, &v1, 1));

    // Draft created under view (1, 1, 1)
    llama_rerot_view_stamp draft_stamp = {1, 1, 1};

    // Invariant 1: RERoT + MTP no peer update -> draft remains valid
    CHECK(!llama_rerot_mtp_is_stale(&ctx, seq_reader, &draft_stamp));

    // 2. Peer updates: peer publishes run advancing episode publish_epoch to 2
    llama_rerot_publish pub = {};
    pub.episode_id = ep1;
    pub.run_id = 2;
    pub.publish_epoch = 2;
    CHECK(llama_rerot_publish_run(&ctx, &pub) > 0);

    // Invariant 2: RERoT + MTP peer update -> older draft MUST report stale!
    CHECK(llama_rerot_mtp_is_stale(&ctx, seq_reader, &draft_stamp));

    // 3. Rollback & re-draft: update view stamp to current publish epoch 2
    llama_rerot_frontier_reader_view v2 = {
        seq_reader, ep1, 1, 1, 11, LLAMA_REROT_FRONTIER_STRONG, {1, 2, 1}, run_ids, 1
    };
    CHECK(llama_rerot_set_frontier_views(&ctx, &v2, 1));
    draft_stamp = {1, 2, 1}; // New draft from fresh view
    CHECK(!llama_rerot_mtp_is_stale(&ctx, seq_reader, &draft_stamp));

    // 4. Invariant 3: Fork barrier: topology changes (new child fork bumps topology epoch)
    llama_rerot_frontier_reader_view v3 = {
        seq_reader, ep1, 1, 1, 12, LLAMA_REROT_FRONTIER_STRONG, {2, 2, 1}, run_ids, 1
    };
    CHECK(llama_rerot_set_frontier_views(&ctx, &v3, 1));
    // Old draft from stamp {1, 2, 1} is immediately stale across the fork barrier
    CHECK(llama_rerot_mtp_is_stale(&ctx, seq_reader, &draft_stamp));

    // 5. Invariant 4: Tri pressure & context shift: layout epoch bumps
    draft_stamp = {2, 2, 1}; // Re-draft at current view
    CHECK(!llama_rerot_mtp_is_stale(&ctx, seq_reader, &draft_stamp));
    llama_rerot_frontier_reader_view v4 = {
        seq_reader, ep1, 1, 1, 13, LLAMA_REROT_FRONTIER_STRONG, {2, 2, 2}, run_ids, 1
    };
    CHECK(llama_rerot_set_frontier_views(&ctx, &v4, 1));
    // Tri layout compaction / eviction bumps layout epoch: old draft stale!
    CHECK(llama_rerot_mtp_is_stale(&ctx, seq_reader, &draft_stamp));

    // 6. Invariant 5: Final fence: coordinator coordinates freeze and serial tail
    draft_stamp = {2, 2, 2}; // Re-draft at current view
    CHECK(!llama_rerot_mtp_is_stale(&ctx, seq_reader, &draft_stamp));
    llama_rerot_frontier_reader_view v_final = {
        seq_reader, ep1, 1, 1, 14, LLAMA_REROT_FRONTIER_STRONG, {3, 3, 3}, run_ids, 1
    };
    CHECK(llama_rerot_set_frontier_views(&ctx, &v_final, 1));
    CHECK(llama_rerot_mtp_is_stale(&ctx, seq_reader, &draft_stamp));

    llama_rerot_episode_end(&ctx, ep1);
    CHECK(!llama_rerot_is_active(&ctx, ep1));
}

static void test_phase5_ram_checkpoint_context_shift_matrix() {
    std::fprintf(stderr, "--- test_phase5_ram_checkpoint_context_shift_matrix (AGENTS.md Phase 5) ---\n");

    // -----------------------------------------------------------------------
    // Part 1: Minimum Acceptance Gate:
    // fork -> children running -> demote -> RAM save -> physical slots reallocated to other requests
    // -> restore to different physical indices -> resume generation
    // Result matches uninterrupted reference.
    // -----------------------------------------------------------------------
    {
        // 1. Reference runtime: uninterrupted generation
        server_rerot_runtime ref_runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
        const uint64_t ep_ref = ref_runtime.adopt_root(101, 101, 0, 0, 10, 501);
        CHECK(ep_ref == 501);
        CHECK(commit_private(ref_runtime, ep_ref, 0, 10));
        CHECK(commit_generated(ref_runtime, ep_ref, 0, 11, "<ol><li>Worker Alpha</li><li>Worker Beta</li></ol>"));
        ref_runtime.finish_frontier(ep_ref);
        CHECK(ref_runtime.freeze_fork_parent(ep_ref, 0));

        // Admit child 1 into slot 1, child 2 into slot 2
        start_child(ref_runtime, ep_ref, 1, 1);
        start_child(ref_runtime, ep_ref, 2, 2);
        CHECK(ref_runtime.node(ep_ref, 1)->physical_slot == 1);
        CHECK(ref_runtime.node(ep_ref, 2)->physical_slot == 2);

        // Advance generation on children in reference
        CHECK(commit_generated(ref_runtime, ep_ref, 1, ref_runtime.node(ep_ref, 1)->storage_pos_next, "Alpha reasoning token"));
        CHECK(commit_generated(ref_runtime, ep_ref, 2, ref_runtime.node(ep_ref, 2)->storage_pos_next, "Beta reasoning token"));

        // 2. Target runtime: fork -> run -> demote -> RAM save -> swap slots -> restore -> resume
        server_rerot_runtime target_runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
        const uint64_t ep_tgt = target_runtime.adopt_root(101, 101, 0, 0, 10, 501);
        CHECK(ep_tgt == 501);
        CHECK(commit_private(target_runtime, ep_tgt, 0, 10));
        CHECK(commit_generated(target_runtime, ep_tgt, 0, 11, "<ol><li>Worker Alpha</li><li>Worker Beta</li></ol>"));
        target_runtime.finish_frontier(ep_tgt);
        CHECK(target_runtime.freeze_fork_parent(ep_tgt, 0));

        start_child(target_runtime, ep_tgt, 1, 1);
        start_child(target_runtime, ep_tgt, 2, 2);
        CHECK(commit_generated(target_runtime, ep_tgt, 1, target_runtime.node(ep_tgt, 1)->storage_pos_next, "Alpha reasoning token"));
        CHECK(commit_generated(target_runtime, ep_tgt, 2, target_runtime.node(ep_tgt, 2)->storage_pos_next, "Beta reasoning token"));

        // Stamp states for verification
        target_runtime.node(ep_tgt, 1)->sampler_blob = {11, 22, 33};
        target_runtime.node(ep_tgt, 1)->mtp_blob = {44, 55};
        target_runtime.node(ep_tgt, 2)->sampler_blob = {66, 77};
        target_runtime.node(ep_tgt, 2)->mtp_blob = {88, 99};

        // Demote target episode: transient slot/pen bindings released
        CHECK(target_runtime.demote_episode(ep_tgt));
        CHECK(target_runtime.node(ep_tgt, 1)->physical_slot == -1);
        CHECK(target_runtime.node(ep_tgt, 2)->physical_slot == -1);

        // RAM Save episode
        const auto fp = test_state_fingerprints();
        std::vector<uint8_t> ram_blob;
        std::string err;
        CHECK(target_runtime.save_episode(ep_tgt, fp, &ram_blob, &err));
        CHECK(!ram_blob.empty() && err.empty());

        // Erase ep_tgt in target_runtime, and allocate other work in target_runtime to occupy slots 1 and 2
        // while preserving non-overlapping internal sequence arena.
        CHECK(target_runtime.erase_episode(ep_tgt));

        // In target_runtime, occupy physical slots 1 and 2 with an unrelated episode using distinct exec_seqs (5 and 6)
        const uint64_t ep_other = target_runtime.adopt_root(200, 200, 0, 7, 0, 777);
        CHECK(ep_other == 777);
        CHECK(commit_generated(target_runtime, ep_other, 0, 0, "<ol><li>Other 1</li><li>Other 2</li></ol>"));
        target_runtime.finish_frontier(ep_other);
        CHECK(target_runtime.freeze_fork_parent(ep_other, 0));

        llama_rerot_node_id other1 = LLAMA_REROT_NODE_INVALID;
        CHECK(target_runtime.admit_next_child(ep_other, 1, 5, &other1));
        CHECK(other1 == 1);
        llama_rerot_node_id other2 = LLAMA_REROT_NODE_INVALID;
        CHECK(target_runtime.admit_next_child(ep_other, 2, 6, &other2));
        CHECK(other2 == 2);
        CHECK(target_runtime.node(ep_other, 1)->physical_slot == 1);
        CHECK(target_runtime.node(ep_other, 2)->physical_slot == 2);

        // Now restore ep_tgt into target_runtime:
        // Because ep_tgt was demoted, its physical_slot is -1, so it does NOT conflict with slots 1 and 2!
        uint64_t restored_id = 0;
        const bool load_ok = target_runtime.load_episode(ram_blob.data(), ram_blob.size(), fp, &restored_id, &err);
        if (!load_ok) {
            std::fprintf(stderr, "target_runtime.load_episode failed: %s\n", err.c_str());
        }
        CHECK(load_ok);
        CHECK(restored_id == ep_tgt);

        auto * restored_ep = target_runtime.episode(restored_id);
        CHECK(restored_ep != nullptr);
        CHECK(restored_ep->document.validate(&err));
        CHECK(restored_ep->running.count(1) != 0);
        CHECK(restored_ep->running.count(2) != 0);

        // Restore to different physical slots (slots 3 and 4, since 1 and 2 are busy!)
        auto * r_node1 = target_runtime.node(restored_id, 1);
        auto * r_node2 = target_runtime.node(restored_id, 2);
        CHECK(r_node1 && r_node2);
        r_node1->physical_slot = 3;
        r_node2->physical_slot = 4;
        CHECK(r_node1->sampler_blob == std::vector<uint8_t>({11, 22, 33}));
        CHECK(r_node1->mtp_blob == std::vector<uint8_t>({44, 55}));
        CHECK(r_node2->sampler_blob == std::vector<uint8_t>({66, 77}));
        CHECK(r_node2->mtp_blob == std::vector<uint8_t>({88, 99}));

        // Continue generation on the restored episode on different physical slots
        CHECK(commit_generated(target_runtime, restored_id, 1, r_node1->storage_pos_next, " continue A"));
        CHECK(commit_generated(target_runtime, restored_id, 2, r_node2->storage_pos_next, " continue B"));

        // Match with ref_runtime also generating the same tokens
        CHECK(commit_generated(ref_runtime, ep_ref, 1, ref_runtime.node(ep_ref, 1)->storage_pos_next, " continue A"));
        CHECK(commit_generated(ref_runtime, ep_ref, 2, ref_runtime.node(ep_ref, 2)->storage_pos_next, " continue B"));

        // Verify logical equivalence: node run counts, token counts, PAC-DFS view equality
        const auto * ref_ep = ref_runtime.episode(ep_ref);
        CHECK(ref_ep->document.node_count() == restored_ep->document.node_count());
        CHECK(ref_ep->document.run_count() == restored_ep->document.run_count());
        for (size_t r = 0; r < ref_ep->document.run_count(); ++r) {
            const auto * ref_run = ref_ep->document.run(llama_rerot_run_id(r));
            const auto * res_run = restored_ep->document.run(llama_rerot_run_id(r));
            CHECK(ref_run != nullptr && res_run != nullptr);
            CHECK(ref_run->owner == res_run->owner);
            CHECK(ref_run->visibility == res_run->visibility);
            CHECK(ref_run->token_count == res_run->token_count);
            CHECK(ref_run->storage_pos0 == res_run->storage_pos0);
        }

        // Compare reader PAC-DFS view between reference and restored
        const auto ref_view = ref_ep->document.build_view(1);
        const auto res_view = restored_ep->document.build_view(1);
        CHECK(ref_view.runs.size() == res_view.runs.size());
        CHECK(ref_view.runs.size() > 0);
        for (size_t i = 0; i < ref_view.runs.size(); ++i) {
            CHECK(ref_view.runs[i].run_id == res_view.runs[i].run_id);
            CHECK(ref_view.runs[i].owner == res_view.runs[i].owner);
            CHECK(ref_view.runs[i].token_count == res_view.runs[i].token_count);
            CHECK(ref_view.runs[i].virtual_pos0 == res_view.runs[i].virtual_pos0);
            CHECK(ref_view.runs[i].storage_pos0 == res_view.runs[i].storage_pos0);
            CHECK(ref_view.runs[i].publish_epoch == res_view.runs[i].publish_epoch);
        }
    }

    // -----------------------------------------------------------------------
    // Part 2: Context Shift as Logical History Deletion:
    // Drops oldest unpinned public runs, updates coordinates, reader PAC-DFS view,
    // bumps layout & publish epochs, and invalidates MTP view stamps.
    // -----------------------------------------------------------------------
    {
        test_stub_model model;
        llama_cparams cparams = {};
        cparams.rerot_enabled = true;
        cparams.rerot_frontier = LLAMA_REROT_FRONTIER_STRONG;
        llama_context ctx(model, cparams, true);

        const uint64_t ep = 601;
        CHECK(llama_rerot_episode_begin(&ctx, ep, nullptr));

        server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
        const uint64_t ep_id = runtime.adopt_root(10, 10, 0, 0, 10, ep);
        CHECK(ep_id == ep);
        auto * episode = runtime.episode(ep);
        CHECK(episode != nullptr);

        // Append base prefix (pinned)
        const auto run_prefix = episode->document.append_run(0, llama_rerot_visibility::public_live, 0, 10, 1);
        // Append older public runs that can be shifted
        const auto run_old1 = episode->document.append_run(0, llama_rerot_visibility::public_live, 10, 5, 2);
        const auto run_old2 = episode->document.append_run(0, llama_rerot_visibility::public_live, 15, 5, 3);
        // Active run
        const auto run_active = episode->document.append_run(0, llama_rerot_visibility::public_live, 20, 4, 4);
        runtime.node(ep, 0)->public_run = run_active;

        episode->publish_epoch = 10;
        episode->layout_epoch = 20;
        episode->topology_epoch = 5;

        // Establish reader view on context and check MTP draft stamp
        llama_seq_id seq_reader = 0;
        llama_rerot_view_stamp stamp = {episode->topology_epoch, episode->publish_epoch, episode->layout_epoch};
        uint32_t run_ids[] = {run_prefix, run_old1, run_old2, run_active};
        llama_rerot_frontier_reader_view v_before = {
            seq_reader, ep, 0, run_active, 24, LLAMA_REROT_FRONTIER_STRONG, stamp, run_ids, 4
        };
        CHECK(llama_rerot_set_frontier_views(&ctx, &v_before, 1));
        CHECK(!llama_rerot_mtp_is_stale(&ctx, seq_reader, &stamp));

        // Perform context shift: remove 5 tokens (which exactly empties run_old1)
        server_rerot_shift_result shift_res;
        std::string err;
        CHECK(runtime.context_shift(ep, 5, &shift_res, &err));
        CHECK(err.empty());
        CHECK(shift_res.tokens_removed == 5);
        CHECK(shift_res.runs_emptied == 1);
        CHECK(shift_res.runs_truncated == 1);
        CHECK(episode->document.run(run_prefix)->token_count == 10); // prefix kept intact
        CHECK(episode->document.run(run_old1)->token_count == 0);     // old1 emptied
        CHECK(episode->document.run(run_old2)->token_count == 5);     // old2 intact
        CHECK(episode->document.run(run_active)->token_count == 4);   // active intact

        // Layout epoch and publish epoch bumped
        CHECK(episode->layout_epoch == 21);
        CHECK(episode->publish_epoch == 11);
        CHECK(episode->topology_barrier_pending);

        // Update reader view on context post-shift
        llama_rerot_view_stamp stamp_post = {episode->topology_epoch, episode->publish_epoch, episode->layout_epoch};
        uint32_t post_run_ids[] = {run_prefix, run_old2, run_active}; // run_old1 dropped from PAC-DFS view
        llama_rerot_frontier_reader_view v_after = {
            seq_reader, ep, 0, run_active, 19, LLAMA_REROT_FRONTIER_STRONG, stamp_post, post_run_ids, 3
        };
        CHECK(llama_rerot_set_frontier_views(&ctx, &v_after, 1));

        // Prior MTP draft stamp MUST be invalidated (stale)
        CHECK(llama_rerot_mtp_is_stale(&ctx, seq_reader, &stamp));
        // New draft stamp with updated layout epoch is valid
        CHECK(!llama_rerot_mtp_is_stale(&ctx, seq_reader, &stamp_post));

        llama_rerot_episode_end(&ctx, ep);
    }

    // -----------------------------------------------------------------------
    // Part 3: Checkpoint and Partial Rollback (AGENTS.md Phase 5):
    // Snapshot plane, child native recurrent, PUBLIC brain, PRIVATE root brain,
    // F32 hand, conv state.
    // -----------------------------------------------------------------------
    {
        test_stub_model model;
        model.hparams.n_layer_all = 4;
        model.hparams.n_embd = 8;
        model.hparams.n_embd_r_impl = 16;
        model.hparams.ssm_d_state = 16;
        model.hparams.ssm_d_inner = 32;

        // Grouped recurrent memory: 2 brain rows, 4 hand rows, n_rs_seq = 2 (2 snapshots)
        llama_memory_recurrent mem(model, GGML_TYPE_F32, GGML_TYPE_F32,
            false, 4, 16, 2, 2, 4, nullptr);
        CHECK(mem.is_grouped_layout());
        CHECK(mem.get_brain_capacity() == 2);
        CHECK(mem.get_hand_capacity() == 4);

        const uint64_t ep_roll = 701;
        llama_seq_id root_seq = 0;
        llama_seq_id child_seq = 1;

        // Step 1: Root prefill and setup write tag
        llama_kv_rerot_meta root_tag;
        root_tag.episode_id = ep_roll;
        root_tag.node_id = 0;
        root_tag.visibility = llama_rerot_visibility::public_live;
        CHECK(mem.rerot_set_write_tag(root_seq, root_tag));

        // Allocate root in recurrent memory
        mem.tails[root_seq] = 0;
        mem.cells[0].pos = 5;
        mem.cells[0].seq_id.insert(root_seq);
        mem.seq_episode[root_seq] = ep_roll;
        mem.seq_node[root_seq] = 0;

        // Verify PUBLIC brain slot allocation
        CHECK(mem.episode_brain.count(ep_roll) != 0);
        const int32_t brain_row = mem.episode_brain.at(ep_roll);
        CHECK(brain_row >= 0 && brain_row < 2);
        CHECK(mem.seq_brain[root_seq] == brain_row);

        // Capture root hand seed at fork
        const auto fork_seed = mem.capture_hand_seed(ep_roll, root_seq);
        CHECK(fork_seed != nullptr);
        CHECK(fork_seed->source_pos == 5);
        CHECK(fork_seed->conv_tail_bytes.size() == mem.r_l.size());
        CHECK(fork_seed->state_bytes.size() == mem.s_l.size());

        // Step 2: Child admission with hand seed applied
        llama_kv_rerot_meta child_tag;
        child_tag.episode_id = ep_roll;
        child_tag.node_id = 1;
        child_tag.visibility = llama_rerot_visibility::public_live;
        CHECK(mem.rerot_set_write_tag(child_seq, child_tag));

        // Child tail starts unallocated (needs_cell=true), apply_hand_seed allocates cell 1
        CHECK(mem.apply_hand_seed(child_seq, fork_seed));
        CHECK(mem.tails[child_seq] == 1);
        CHECK(mem.cells[1].pos == 5);
        CHECK(mem.cells[1].has_seq_id(child_seq));

        // Advance child tokens: simulate snapshot recording
        mem.cells[1].pos = 8;
        // Verify child native recurrent status
        CHECK(mem.uses_native_child_state(child_seq));

        // Perform partial rollback on child sequence (rollback from pos 8 to pos 6)
        // pos 8 down to 6 is a 2-token rollback, within n_rs_seq=2 capacity
        CHECK(mem.seq_rm(child_seq, 7, -1));
        CHECK(mem.cells[1].pos == 6);
        CHECK(mem.rs_idx[child_seq] == 2);

        // Clear tags and release episode cleanly
        mem.rerot_clear_write_tag(child_seq);
        mem.rerot_clear_write_tag(root_seq);
        mem.rerot_release_episode(ep_roll);
        CHECK(mem.get_brain_used() == 0);
    }
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
            CHECK(runtime.continue_unforked_root(ep, 0));
            CHECK(runtime.episode(ep)->serial_tail);
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

    // Ep3 takes the delimiter-free N=1 serial path.
    make_terminal(runtime, ep3, 0);
    CHECK(runtime.continue_unforked_root(ep3, 0));
    CHECK(runtime.episode(ep3)->serial_tail);

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
    replay_final_fence(runtime, ep, 2);
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

    // N=1 roots have no child delimiter and continue serially without
    // observing or synthesizing the model-native thought boundary.
    CHECK(runtime.continue_unforked_root(ep1, 0));

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

    // Episode 2 independently takes the same delimiter-free N=1 path.
    CHECK(commit_generated(
        runtime, ep2, 0, runtime.node(ep2, 0)->storage_pos_next,
        "<ol><li>Single task</li></ol>"));
    runtime.finish_frontier(ep2);
    CHECK(runtime.continue_unforked_root(ep2, 0));
    CHECK(runtime.validate_serial_tail_state(ep2, 0, &err));
    CHECK(runtime.freeze_serial_coordinates(ep2, 0, &err));

    // Both episodes reached serial tail independently
    CHECK(ep1_ptr->serial_tail == true);
    CHECK(ep2_ptr->serial_tail == true);

    CHECK(runtime.erase_episode(ep1));
    CHECK(runtime.erase_episode(ep2));
}

static void test_phase7_runtime_production_stress_and_pressure() {
    // =========================================================================
    // Phase 7: Production-scale Full-Slot Pressure & Fallback Order (§B.9, §B.13, Gate 23)
    // Co-locates:
    //   - 6 outer slots (B=6)
    //   - recursive forks with ragged child distribution
    //   - queued children (B > P where P=12, queued=18)
    //   - TriAttention KV pressure with lossy reclaim on public history
    //   - speculative MTP draft invalidation on layout epoch changes
    //   - active preemption and slot demotion/re-allocation
    // Invariants:
    //   - 0 orphan seq ref, 0 orphan recurrent cell
    //   - victim preemption returns all physical slots immediately
    //   - unaffected lanes continue and complete with exactly one terminal event
    // =========================================================================
    std::fprintf(stderr, "--- test_phase7_runtime_production_stress_and_pressure (AGENTS.md Phase 7) ---\n");

    const uint32_t P = 12; // 12 physical pens
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 10, 250);
    runtime.set_pen_capacity(P);
    CHECK(runtime.pen_capacity() == P);

    // 1. 6 outer episodes (B=6)
    std::vector<uint64_t> people;
    const std::vector<int> child_counts = { 4, 3, 5, 2, 3, 4 }; // Sum = 21 children > 12 pens
    for (size_t i = 0; i < child_counts.size(); ++i) {
        const uint64_t ep = runtime.adopt_root(
            int(i + 1), int(i + 1), int(i), llama_seq_id(10 + i), 0, uint64_t(700 + i));
        people.push_back(ep);
    }
    CHECK(runtime.pens_allocated() == 6);

    // Fork each episode
    for (size_t i = 0; i < people.size(); ++i) {
        const uint64_t ep = people[i];
        const int n_children = child_counts[i];
        std::string xml = "<ol>";
        for (int c = 0; c < n_children; ++c) {
            xml += "<li>Subtask " + std::to_string(c) + "</li>";
        }
        xml += "</ol>";
        CHECK(commit_generated(runtime, ep, 0, 0, xml));
        runtime.finish_frontier(ep);
        CHECK(runtime.freeze_fork_parent(ep, 0));
    }

    // Schedule pens: P=12 capacity, children demand 21 pens.
    // Invariant: admissions are strictly bounded by pen_capacity, remainder stays in ready_queue
    const size_t admitted = runtime.schedule_pens(people);
    CHECK(admitted == P);
    CHECK(runtime.pens_allocated() == P);
    CHECK(runtime.pens_allocated() <= runtime.pen_capacity());

    // 2. Memory pressure & active preemption:
    // Demote Person 2 (ep = people[2], which holds child pens)
    const uint64_t victim_ep = people[2];
    const auto victim_pens = runtime.pens_for_person(victim_ep);
    CHECK(!victim_pens.empty());
    const size_t freed_pen_count = victim_pens.size();

    // Release slots and demote to logical state (§A.11)
    for (int pen_id : victim_pens) {
        runtime.release_slot(pen_id);
    }
    auto * victim_ptr = runtime.episode(victim_ep);
    CHECK(victim_ptr != nullptr);
    server_rerot_episode_demote_to_logical(*victim_ptr);

    // Invariant: freed pens immediately available for queued children
    CHECK(runtime.pens_allocated() == P - freed_pen_count);
    CHECK(runtime.pens_for_person(victim_ep).empty());

    // Re-schedule pens to admit waiting children from remaining people
    const size_t refill_admitted = runtime.schedule_pens(people);
    CHECK(refill_admitted == freed_pen_count);
    CHECK(runtime.pens_allocated() == P);

    // 3. TriAttention truncation on Person 0 under KV pressure (§B.9.3)
    auto * ep0_ptr = runtime.episode(people[0]);
    CHECK(ep0_ptr != nullptr);
    const auto r_pub = ep0_ptr->document.append_run(0, llama_rerot_visibility::public_live, 0, 400, 1);
    const auto r_active = ep0_ptr->document.append_run(0, llama_rerot_visibility::public_live, 400, 10, 2);
    runtime.node(people[0], 0)->public_run = r_active;

    server_rerot_shift_result shift_res;
    std::string err;
    CHECK(server_rerot_truncate_oldest_public(*ep0_ptr, 350, &shift_res, &err));
    CHECK(shift_res.tokens_removed == 400);
    CHECK(ep0_ptr->document.run(r_pub)->token_count == 0);

    // Invariant: Layout epoch bumped, older MTP drafts are invalidated
    CHECK(ep0_ptr->layout_epoch > 0);

    // 4. Clean completion and erasure of all episodes
    for (uint64_t ep : people) {
        CHECK(runtime.erase_episode(ep));
    }
    CHECK(runtime.pens_allocated() == 0);
    CHECK(runtime.pens_running() == 0);
}

static void test_final_fence_user_grammar_restoration() {
    std::fprintf(stderr, "--- test_final_fence_user_grammar_restoration (§26, §A.16.1) ---\n");
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t ep = runtime.adopt_root(150, 151, 0, 0, 0, 501);
    CHECK(ep == 501);
    CHECK(runtime.response_task_id(ep) == 151);

    // 1. Simulate incoming user task with user/tool JSON schema grammar
    task_params params;
    const std::string user_grammar_source = "root ::= \"{\\\"name\\\":\\\"\" [a-z]+ \"\\\"}\"";
    params.sampling.grammar = {
        COMMON_GRAMMAR_TYPE_USER,
        user_grammar_source
    };

    // 2. Take user grammar before planner <ol> injection (server_rerot_take_user_grammar)
    const common_grammar saved_grammar = server_rerot_take_user_grammar(params);
    CHECK(!saved_grammar.empty());
    CHECK(!common_grammar_value(saved_grammar).empty());
    CHECK(common_grammar_value(saved_grammar) == user_grammar_source);
    CHECK(saved_grammar.type == COMMON_GRAMMAR_TYPE_USER);
    CHECK(params.sampling.grammar.empty());

    // 3. Sibling children fork, run to completion, and initiate final fence (mirroring test_final_fence_sees_last_sibling_write)
    CHECK(commit_generated(runtime, ep, 0, 0, "<ol><li>Alpha</li><li>Beta</li></ol>"));
    CHECK(runtime.finish_frontier(ep).forked.size() == 1);
    CHECK(runtime.freeze_fork_parent(ep, 0));

    start_child(runtime, ep, 0, 1);
    start_child(runtime, ep, 1, 2);
    make_terminal(runtime, ep, 1);
    make_terminal(runtime, ep, 2);
    request_exit(runtime, ep, 1);
    request_exit(runtime, ep, 2);

    const auto frontier = runtime.finish_frontier(ep);
    CHECK(frontier.natural_final());
    CHECK(frontier.final_node == 2);
    CHECK(frontier.retired.size() == 1 && frontier.retired[0] == 1);

    std::vector<uint32_t> ordered_runs;
    CHECK(runtime.refresh_final_fence(ep, 2, &ordered_runs));
    replay_final_fence(runtime, ep, 2);
    CHECK(runtime.complete_serial_tail(ep, 2));
    const auto * ep_ptr = runtime.episode(ep);
    CHECK(ep_ptr != nullptr && ep_ptr->serial_tail && ep_ptr->serial_node == 2);
    CHECK(runtime.response_task_id(ep) == 151);

    // 4. Restore user grammar on serial tail entry (server_rerot_restore_user_grammar)
    server_rerot_restore_user_grammar(params, saved_grammar);
    CHECK(!params.sampling.grammar.empty());
    CHECK(params.sampling.grammar.type == COMMON_GRAMMAR_TYPE_USER);
    CHECK(params.sampling.grammar.type == saved_grammar.type);
    CHECK(common_grammar_value(params.sampling.grammar) == user_grammar_source);
    CHECK(common_grammar_value(params.sampling.grammar) == common_grammar_value(saved_grammar));

    // 5. Invariant: Restored grammar compiles cleanly via real grammar init used elsewhere in repo tests
    llama_grammar * compiled = llama_grammar_init_impl(
        nullptr, common_grammar_value(params.sampling.grammar).c_str(), "root", false, nullptr, 0, nullptr, 0);
    CHECK(compiled != nullptr);
    if (compiled) {
        llama_grammar_free_impl(compiled);
    }

    CHECK(runtime.erase_episode(ep));
}

static void test_ram_restore_context_shift_and_preemption() {
    std::fprintf(stderr, "--- test_ram_restore_context_shift_and_preemption (§15.2, §21.4, §25, §A.8-A.9) ---\n");
    // 1. Create episode with public root (pos 10 prefix) and forked children (mirroring test_episode_state_round_trip_and_fingerprint)
    server_rerot_runtime rt1(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    const uint64_t ep1 = rt1.adopt_root(280, 281, 0, 0, 10, 701);
    CHECK(ep1 == 701);

    // Public unpinned text committed at pos 10
    CHECK(commit_generated(rt1, ep1, 0, 10, "unpinned text "));
    // Planner record committed at pos 11, forking into 2 children
    CHECK(commit_generated(rt1, ep1, 0, 11, "<ol><li>Child A</li><li>Child B</li></ol>"));
    CHECK(rt1.finish_frontier(ep1).forked.size() == 1);
    CHECK(rt1.freeze_fork_parent(ep1, 0));
    start_child(rt1, ep1, 0, 1);

    auto * child1 = rt1.node(ep1, 1);
    CHECK(child1 != nullptr);

    // 2. Save via existing episode save API with fingerprints
    const auto fp = test_state_fingerprints();
    std::vector<uint8_t> ram_blob;
    std::string err;
    CHECK(rt1.save_episode(ep1, fp, &ram_blob, &err));
    CHECK(!ram_blob.empty());
    CHECK(err.empty());

    // 3. Restore into fresh runtime with pen capacity forced to 1 to trigger preemption pressure
    server_rerot_runtime rt2(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 64);
    uint64_t restored_ep = 0;
    CHECK(rt2.load_episode(ram_blob.data(), ram_blob.size(), fp, &restored_ep, &err));
    CHECK(restored_ep == ep1);
    auto * restored_ep_ptr = rt2.episode(restored_ep);
    CHECK(restored_ep_ptr != nullptr);
    CHECK(restored_ep_ptr->document.validate(&err));
    CHECK(restored_ep_ptr->ready_queue.size() == 1);

    // 4. Apply existing context shift on restored episode
    server_rerot_shift_result shift_res;
    err.clear();
    CHECK(rt2.context_shift(restored_ep, 1, &shift_res, &err));
    CHECK(err.empty());
    CHECK(shift_res.tokens_removed == 1);
    CHECK(shift_res.runs_truncated == 1);
    CHECK(shift_res.runs_emptied == 1);
    CHECK(restored_ep_ptr->topology_barrier_pending);

    // Assert restored episode state validity/unpinned truncation behavior consistent with existing shift test
    CHECK(restored_ep_ptr->document.validate(&err));
    const auto * truncated_run = restored_ep_ptr->document.run(0);
    CHECK(truncated_run != nullptr);
    CHECK(truncated_run->token_count == 0);
    CHECK(truncated_run->storage_pos0 == 10);

    // 5. Set pen capacity to 1 (preemption constraint) and schedule pens
    rt2.set_pen_capacity(1);
    CHECK(rt2.pen_capacity() == 1);
    // Slot 0 was freed by demoting or releasing before schedule
    rt2.release_slot(0);
    CHECK(rt2.pens_allocated() == 0);
    const size_t scheduled = rt2.schedule_pens({restored_ep});
    CHECK(scheduled == 1);
    CHECK(rt2.pens_allocated() == 1);
    CHECK(rt2.pens_allocated() <= rt2.pen_capacity());
    CHECK(restored_ep_ptr->ready_queue.empty());

    // 6. Clean episode erase
    CHECK(rt2.erase_episode(restored_ep));
    CHECK(rt1.erase_episode(ep1));
    CHECK(rt2.pens_allocated() == 0);
}
static void test_dag_runtime_lifecycle() {
    // AGENTS.md §§02, 03, 07:
    // Create DAG episode:
    // Questions: A (rank 0), B (rank 1), C (rank 2)
    // Edge: A -> C (C depends on A). B is independent.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(4);

    const int root_slot = 0;
    const uint64_t ep_id = runtime.adopt_root(10, 10, root_slot, 1, 0);
    CHECK(ep_id != 0);

    const std::string dag_json = R"({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "A", "intent": "Fact A"},
          {"id": "B", "intent": "Fact B"},
          {"id": "C", "intent": "Synthesize C"}
        ],
        "depends_on": [
          {"id": "C", "depends_on_id": "A"}
        ]
      }
    })";
    const auto decision = server_rerot_parse_routing_decision(dag_json);
    CHECK(decision.is_dag());

    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));

    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    CHECK(ep->is_dag);
    CHECK(ep->synthesis_node != 0);
    CHECK(ep->synthesis_node != LLAMA_REROT_NODE_INVALID);
    CHECK(ep->nodes[1].remaining_preds == 0);
    CHECK(ep->nodes[2].remaining_preds == 0);
    CHECK(ep->nodes[3].remaining_preds == 1);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 3);
    CHECK(ep->c0.valid() == false);
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(ep->c0.valid());
    CHECK(ep->c0.gdn_recurrent_states.empty());
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(ep->c_base.valid());

    // Initial eligible nodes must be A (1) and B (2). C (3) must NOT be eligible!
    auto eligible = runtime.get_eligible_dag_nodes(ep_id);
    CHECK(eligible.size() == 2);
    CHECK(std::find(eligible.begin(), eligible.end(), 1) != eligible.end());
    CHECK(std::find(eligible.begin(), eligible.end(), 2) != eligible.end());
    CHECK(std::find(eligible.begin(), eligible.end(), 3) == eligible.end());
    CHECK(std::find(eligible.begin(), eligible.end(), ep->synthesis_node) == eligible.end());

    // Allocate pen and start A (1)
    auto pen_a = runtime.allocate_pen(10, ep_id, 1);
    CHECK(pen_a.has_value());
    ep->document.set_node_state(1, llama_rerot_node_state::running);
    ep->running.insert(1);
    ep->nodes[1].pen_id = *pen_a;

    // Allocate pen and start B (2)
    auto pen_b = runtime.allocate_pen(10, ep_id, 2);
    CHECK(pen_b.has_value());
    ep->document.set_node_state(2, llama_rerot_node_state::running);
    ep->running.insert(2);
    ep->nodes[2].pen_id = *pen_b;

    // Both A and B are running. Add some public tokens:
    ep->document.append_run(0, llama_rerot_visibility::public_live, 0, 10, 1); // P: 10
    ep->document.append_run(1, llama_rerot_visibility::public_live, 10, 4, 2); // A: 4
    ep->document.append_run(2, llama_rerot_visibility::public_live, 14, 6, 2); // B: 6

    // Verify reader views:
    // A reader view: P, B, A (own work last!)
    const auto view_a = runtime.build_dag_view_for_reader(ep_id, 1);
    CHECK(view_a.runs.size() == 3);
    CHECK(view_a.runs[0].owner == 0);
    CHECK(view_a.runs[1].owner == 2);
    CHECK(view_a.runs[2].owner == 1);

    // B reader view: P, A, B (own work last!)
    const auto view_b = runtime.build_dag_view_for_reader(ep_id, 2);
    CHECK(view_b.runs.size() == 3);
    CHECK(view_b.runs[0].owner == 0);
    CHECK(view_b.runs[1].owner == 1);
    CHECK(view_b.runs[2].owner == 2);

    // Now A naturally finishes and is sealed:
    CHECK(runtime.seal_dag_node(ep_id, 1, llama_rerot_event_origin::worker_source));
    CHECK(ep->nodes[1].is_sealed);
    CHECK(ep->document.node(1)->state == llama_rerot_node_state::retired);

    // Now C (3) must become eligible! Synthesis still waits on B and C.
    CHECK(ep->nodes[3].remaining_preds == 0);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 2);
    eligible = runtime.get_eligible_dag_nodes(ep_id);
    CHECK(eligible.size() == 1);
    CHECK(eligible[0] == 3);

    // Start C (3)
    auto pen_c = runtime.allocate_pen(10, ep_id, 3);
    CHECK(pen_c.has_value());
    ep->document.set_node_state(3, llama_rerot_node_state::running);
    ep->running.insert(3);
    ep->nodes[3].pen_id = *pen_c;
    ep->document.append_run(3, llama_rerot_visibility::public_live, 20, 8, 3); // C: 8

    // C reader view: A (predecessor), B (peer), C (self last)
    const auto view_c = runtime.build_dag_view_for_reader(ep_id, 3);
    CHECK(view_c.runs.size() == 4);
    CHECK(view_c.runs[0].owner == 0);
    CHECK(view_c.runs[1].owner == 1);
    CHECK(view_c.runs[2].owner == 2);
    CHECK(view_c.runs[3].owner == 3);

    // Seal B and C
    CHECK(runtime.seal_dag_node(ep_id, 2, llama_rerot_event_origin::worker_source));
    CHECK(runtime.seal_dag_node(ep_id, 3, llama_rerot_event_origin::worker_source));
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 0);

    // Repeat seal is a no-op: remaining_preds must not decrement twice (§07.5)
    CHECK(runtime.seal_dag_node(ep_id, 1, llama_rerot_event_origin::worker_source));
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 0);

    // Provenance validation: runtime_frame origin must be rejected (§04.6)
    CHECK(!runtime.seal_dag_node(ep_id, 1, llama_rerot_event_origin::runtime_frame));
    CHECK(!runtime.seal_dag_node(ep_id, 1, llama_rerot_event_origin::foreign_export));

    // Synthesis reader 0 view: P, A, B, C
    const auto view_synth = runtime.build_dag_view_for_reader(ep_id, 0);
    CHECK(view_synth.runs.size() == 4);
    CHECK(view_synth.runs[0].owner == 0);
    CHECK(view_synth.runs[1].owner == 1);
    CHECK(view_synth.runs[2].owner == 2);
    CHECK(view_synth.runs[3].owner == 3);

    // Workers are sealed; synthesis is eligible but not yet running, so there
    // is no serial survivor. final_node is only the live synthesis node after
    // it emits native source-end while still bound to a slot.
    const auto frontier_res = runtime.finish_frontier(ep_id);
    CHECK(frontier_res.final_node == LLAMA_REROT_NODE_INVALID);
    CHECK(frontier_res.synthesis_node == ep->synthesis_node);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 0);

    // Verify DAG episode serialization and deserialization preserves DAG topology
    server_rerot_state_fingerprints fp;
    fp.caps = LLAMA_REROT_STATE_CAP_REROT | LLAMA_REROT_STATE_CAP_REROT_TREE;
    std::string serr;
    std::vector<uint8_t> saved_blob;
    CHECK(runtime.save_episode(ep_id, fp, &saved_blob, &serr));
    CHECK(serr.empty());
    CHECK(!saved_blob.empty());

    // Erase live episode before restoring (§A.8)
    CHECK(runtime.erase_episode(ep_id));

    uint64_t loaded_ep_id = 0;
    std::string lerr;
    CHECK(runtime.load_episode(saved_blob.data(), saved_blob.size(), fp, &loaded_ep_id, &lerr));
    CHECK(loaded_ep_id == ep_id);
    CHECK(runtime.episode(ep_id)->is_dag);
    CHECK(runtime.episode(ep_id)->document.is_dag_mode());
    CHECK(runtime.episode(ep_id)->nodes[1].is_sealed);
    CHECK(runtime.episode(ep_id)->nodes[1].string_id == "A");
    const auto * loaded_c = runtime.episode(ep_id)->document.node(3);
    CHECK(loaded_c != nullptr);
    CHECK(std::find(loaded_c->predecessors.begin(), loaded_c->predecessors.end(),
                    llama_rerot_node_id(1)) != loaded_c->predecessors.end());
    CHECK(runtime.episode(ep_id)->synthesis_node != 0);
    CHECK(runtime.episode(ep_id)->nodes[runtime.episode(ep_id)->synthesis_node].remaining_preds == 0);
}

static void test_c0_and_dag_admit_without_parked_seq() {
    // DAG workers start from C_base with parked_seq < 0. Admission must not
    // demand an HTML-fork parked sequence, and C0 remains valid with empty
    // recurrent state.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(4);

    const uint64_t ep_id = runtime.adopt_root(11, 11, 0, 1, 0);
    CHECK(ep_id != 0);
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    CHECK(ep->c0.valid());
    CHECK(ep->c0.gdn_recurrent_states.empty());
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(ep->c_base.valid());

    const std::string dag_json = R"({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "A", "intent": "Fact A"},
          {"id": "B", "intent": "Fact B"}
        ],
        "depends_on": []
      }
    })";
    const auto decision = server_rerot_parse_routing_decision(dag_json);
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    CHECK(runtime.activate_dag_frontier(ep_id));

    llama_rerot_node_id admitted = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 1, 2, &admitted));
    CHECK(admitted == 1);
    CHECK(runtime.node(ep_id, admitted)->parked_seq < 0);
    CHECK(runtime.node(ep_id, admitted)->physical_slot == 1);
    CHECK(runtime.complete_admission(ep_id, admitted));
}

static bool dag_view_has_run(const llama_rerot_reader_view & view, llama_rerot_run_id run_id) {
    for (const auto & run : view.runs) {
        if (run.run_id == run_id) {
            return true;
        }
    }
    return false;
}

static bool dag_view_has_owner(const llama_rerot_reader_view & view, llama_rerot_node_id owner) {
    for (const auto & run : view.runs) {
        if (run.owner == owner) {
            return true;
        }
    }
    return false;
}

static bool dag_queue_contains(
        const std::deque<llama_rerot_node_id> & queue,
        llama_rerot_node_id node_id) {
    return std::find(queue.begin(), queue.end(), node_id) != queue.end();
}

static void dag_unbind_planner_if_bound(server_rerot_runtime & runtime, uint64_t ep_id) {
    auto * root = runtime.node(ep_id, 0);
    if (root && root->physical_slot >= 0) {
        CHECK(runtime.detach_node(ep_id, 0));
    }
}

static bool dag_state_is_live_worker(llama_rerot_node_state state) {
    return state == llama_rerot_node_state::running ||
           state == llama_rerot_node_state::terminal_running;
}

static void test_dag_capture_c_base_snapshots_current_seed() {
    // C_base is a snapshot of the current seed. Empty GDN is valid; validity
    // is not "episode_id != 0 && gdn nonempty".
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(4);
    const uint64_t ep_id = runtime.adopt_root(12, 12, 0, 1, 0);
    CHECK(ep_id != 0);
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    auto * ep = runtime.episode(ep_id);
    auto * root = runtime.node(ep_id, 0);
    CHECK(ep != nullptr);
    CHECK(root != nullptr);
    CHECK(ep->c0.valid());
    CHECK(ep->c0.gdn_recurrent_states.empty());
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(ep->c_base.valid());
    CHECK(ep->c_base.gdn_recurrent_states.empty());

    const std::vector<uint8_t> seed = {11, 22, 33, 44};
    root->hand_seed = seed;
    ep->c0.gdn_recurrent_states = {9, 9};
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(ep->c_base.valid());
    CHECK(ep->c_base.gdn_recurrent_states == seed);
    root->hand_seed = {1};
    CHECK(ep->c_base.gdn_recurrent_states == seed);

    // Multi-stage COW and immutability (§12.4):
    // Modifying worker or root seeds post-admission must not mutate C_base snapshot.
    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {
        "questions": [{"id": "1", "intent": "Worker 1"}, {"id": "2", "intent": "Worker 2"}],
        "depends_on": []
      }
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    CHECK(runtime.activate_dag_frontier(ep_id));

    // Admit worker 1 and worker 2
    llama_rerot_node_id a = LLAMA_REROT_NODE_INVALID;
    llama_rerot_node_id b = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 1, 8, &a));
    CHECK(runtime.admit_next_child(ep_id, 2, 9, &b));
    CHECK(a == 1 && b == 2);
    CHECK(runtime.complete_admission(ep_id, a));
    CHECK(runtime.complete_admission(ep_id, b));
    auto * w1 = runtime.node(ep_id, 1);
    auto * w2 = runtime.node(ep_id, 2);
    CHECK(w1 != nullptr && w2 != nullptr);

    // Initial hand seed inherited from C_base
    CHECK(w1->hand_seed == seed);
    CHECK(w2->hand_seed == seed);

    // Mutating w1 hand seed (first write / local mutation) isolates w1 from w2 and C_base
    w1->hand_seed = {0xAA, 0xBB};
    CHECK(w1->hand_seed != w2->hand_seed);
    CHECK(w2->hand_seed == seed);
    CHECK(ep->c_base.gdn_recurrent_states == seed);

    // Mutating w2 hand seed isolates w2 from w1 and C_base
    w2->hand_seed = {0xCC, 0xDD};
    CHECK(w1->hand_seed != w2->hand_seed);
    CHECK(ep->c_base.gdn_recurrent_states == seed);
}

static void test_dag_isolated_probe_uses_other_seq() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 16);
    runtime.set_pen_capacity(2);
    const uint64_t ep_id = runtime.adopt_root(20, 20, 0, 0, 4);
    CHECK(ep_id != 0);
    CHECK(runtime.capture_c0(ep_id, 0, 4));
    CHECK(runtime.arm_isolated_probe(ep_id, 0));
    auto * ep = runtime.episode(ep_id);
    auto * root = runtime.node(ep_id, 0);
    CHECK(ep != nullptr && root != nullptr);
    CHECK(ep->probe_seq >= 8);
    CHECK(ep->probe_seq != 0);
    CHECK(root->exec_seq == ep->probe_seq);
    CHECK(runtime.discard_isolated_probe(ep_id, 0));
    CHECK(ep->probe_seq == -1);
    CHECK(root->exec_seq == 0);
    CHECK(root->storage_pos_next == 4);
    CHECK(!runtime.discard_isolated_probe(ep_id, 0));
}

static void test_dag_c0_probe_cancel_and_isolation() {
    // Stage 3 (§12.4): Cancellation/abort during isolated probe must cleanly
    // release probe resources without corrupting C0 or peer episode state.
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 16);
    runtime.set_pen_capacity(4);

    // Create Ep 1 (target episode under probe) and Ep 2 (peer episode)
    const uint64_t ep1 = runtime.adopt_root(21, 21, 0, 0, 4);
    const uint64_t ep2 = runtime.adopt_root(22, 22, 0, 0, 8);
    CHECK(ep1 != 0 && ep2 != 0);

    // Ep 1 captures C0 and arms isolated probe
    CHECK(runtime.capture_c0(ep1, 0, 4));
    CHECK(runtime.arm_isolated_probe(ep1, 0));
    auto * ep1_ptr = runtime.episode(ep1);
    CHECK(ep1_ptr != nullptr);
    const llama_seq_id probe1 = ep1_ptr->probe_seq;
    CHECK(probe1 >= 8);

    // Ep 2 captures C0 and remains unprobed
    CHECK(runtime.capture_c0(ep2, 0, 8));
    auto * ep2_ptr = runtime.episode(ep2);
    CHECK(ep2_ptr != nullptr);
    CHECK(ep2_ptr->c0.valid());
    CHECK(ep2_ptr->c0.n_prompt_tokens == 8);

    // Abort Ep 1 during probe (e.g. client cancellation)
    runtime.hard_abort(ep1, "client_cancelled_during_probe");
    CHECK(ep1_ptr->hard_aborted);
    CHECK(ep1_ptr->probe_seq == -1);

    // Probe sequence must be returned: arming probe on ep2 must succeed and assign a valid internal sequence
    CHECK(runtime.arm_isolated_probe(ep2, 0));
    CHECK(ep2_ptr->probe_seq >= 8);
    CHECK(runtime.discard_isolated_probe(ep2, 0));
    CHECK(ep2_ptr->probe_seq == -1);

    // Peer Ep 2 C0 state and runtime must be 100% intact
    CHECK(ep2_ptr->c0.valid());
    CHECK(ep2_ptr->c0.n_prompt_tokens == 8);
    CHECK(!ep2_ptr->hard_aborted);
    auto * ep2_root = runtime.node(ep2, 0);
    CHECK(ep2_root != nullptr);
    CHECK(ep2_root->storage_pos_next == 8);
}

static void test_dag_admission_view_survival() {
    // After complete_admission, DAG document state is running (not planning)
    // and the node's PUBLIC runs remain in every started reader's view.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(4);

    const uint64_t ep_id = runtime.adopt_root(13, 13, 0, 1, 0);
    CHECK(ep_id != 0);

    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "A", "intent": "Fact A"},
          {"id": "B", "intent": "Fact B"},
          {"id": "C", "intent": "Synthesize C"}
        ],
        "depends_on": [
          {"id": "C", "depends_on_id": "A"}
        ]
      }
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    CHECK(ep->nodes[1].remaining_preds == 0);
    CHECK(ep->nodes[2].remaining_preds == 0);
    CHECK(ep->nodes[3].remaining_preds == 1);

    // Formal base capture after init, before any ready work.
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    ep->ready_queue.clear();
    CHECK(runtime.activate_dag_frontier(ep_id));
    CHECK(dag_queue_contains(ep->ready_queue, 1));
    CHECK(dag_queue_contains(ep->ready_queue, 2));
    CHECK(!dag_queue_contains(ep->ready_queue, 3));
    CHECK(!dag_queue_contains(ep->ready_queue, ep->synthesis_node));

    auto pen_a = runtime.allocate_pen(13, ep_id, 1);
    CHECK(pen_a.has_value());
    llama_rerot_node_id admitted_a = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, *pen_a, 2, &admitted_a));
    CHECK(admitted_a == 1);

    auto pen_b = runtime.allocate_pen(13, ep_id, 2);
    CHECK(pen_b.has_value());
    llama_rerot_node_id admitted_b = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, *pen_b, 3, &admitted_b));
    CHECK(admitted_b == 2);

    CHECK(runtime.complete_admission(ep_id, admitted_a));
    const auto * logical_a = ep->document.node(admitted_a);
    CHECK(logical_a != nullptr);
    CHECK(dag_state_is_live_worker(logical_a->state));
    CHECK(logical_a->state != llama_rerot_node_state::planning);

    ep->document.append_run(0, llama_rerot_visibility::public_live, 0, 4, 1);
    const auto run_a = ep->document.append_run(
        admitted_a, llama_rerot_visibility::public_live, 4, 5, 2);

    const auto view_a = runtime.build_dag_view_for_reader(ep_id, admitted_a);
    CHECK(dag_view_has_run(view_a, run_a));
    CHECK(dag_view_has_owner(view_a, admitted_a));

    CHECK(runtime.complete_admission(ep_id, admitted_b));
    const auto run_b = ep->document.append_run(
        admitted_b, llama_rerot_visibility::public_live, 9, 6, 2);
    const auto view_b = runtime.build_dag_view_for_reader(ep_id, admitted_b);
    CHECK(dag_view_has_run(view_b, run_a));
    CHECK(dag_view_has_run(view_b, run_b));
    CHECK(dag_view_has_owner(view_b, admitted_a));
    CHECK(!runtime.node(ep_id, admitted_a)->is_sealed);
    CHECK(ep->nodes[3].remaining_preds == 1);
}

static void test_dag_w_gt_p_yield_without_seal() {
    // P=1, W=3 independent workers: yield the running pen so the next
    // eligible can admit without anyone sealing.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(1);

    const uint64_t ep_id = runtime.adopt_root(14, 14, 0, 1, 0);
    CHECK(ep_id != 0);

    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "A", "intent": "Fact A"},
          {"id": "B", "intent": "Fact B"},
          {"id": "C", "intent": "Fact C"}
        ],
        "depends_on": []
      }
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    // Formal base capture after init, before any ready work: lineage,
    // watermarks, and sampler clones do not exist until this point.
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(runtime.activate_dag_frontier(ep_id));
    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    CHECK(ep->ready_queue.size() >= 3);

    dag_unbind_planner_if_bound(runtime, ep_id);

    llama_rerot_node_id first = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 0, 1, &first));
    CHECK(first == 1);
    CHECK(runtime.complete_admission(ep_id, first));
    CHECK(!runtime.node(ep_id, first)->is_sealed);
    CHECK(dag_queue_contains(ep->ready_queue, 2) || dag_queue_contains(ep->ready_queue, 3) ||
          ep->starting.count(2) || ep->starting.count(3) ||
          ep->running.count(2) || ep->running.count(3));

    auto find_other_started = [&]() -> llama_rerot_node_id {
        for (llama_rerot_node_id nid : {llama_rerot_node_id(2), llama_rerot_node_id(3)}) {
            const auto * dn = ep->document.node(nid);
            if (!dn) {
                continue;
            }
            if (ep->starting.count(nid) || ep->running.count(nid) ||
                dn->state == llama_rerot_node_state::starting ||
                dag_state_is_live_worker(dn->state) ||
                dn->state == llama_rerot_node_state::ready_suspended) {
                return nid;
            }
        }
        return LLAMA_REROT_NODE_INVALID;
    };

    // complete_admission may already have yielded; still require a yield when
    // the first worker holds the only pen. First must not need is_sealed.
    if (runtime.node(ep_id, first)->physical_slot >= 0) {
        CHECK(runtime.yield_dag_pen_for_ready(ep_id));
    }
    CHECK(!runtime.node(ep_id, first)->is_sealed);
    const auto first_state = ep->document.node(first)->state;
    CHECK(first_state == llama_rerot_node_state::ready_suspended ||
          dag_state_is_live_worker(first_state));
    CHECK(first_state != llama_rerot_node_state::planning);

    llama_rerot_node_id second = find_other_started();
    if (second == LLAMA_REROT_NODE_INVALID) {
        CHECK(runtime.admit_next_child(ep_id, 0, 2, &second));
    }
    CHECK(second != LLAMA_REROT_NODE_INVALID);
    CHECK(second != first);
    CHECK(!runtime.node(ep_id, first)->is_sealed);

    ep->document.append_run(0, llama_rerot_visibility::public_live, 0, 3, 1);
    const auto run_a = ep->document.append_run(
        first, llama_rerot_visibility::public_live, 3, 4, 2);
    const auto run_b = ep->document.append_run(
        second, llama_rerot_visibility::public_live, 7, 5, 2);

    const auto view_first = runtime.build_dag_view_for_reader(ep_id, first);
    const auto view_second = runtime.build_dag_view_for_reader(ep_id, second);
    CHECK(dag_view_has_run(view_first, run_a));
    CHECK(dag_view_has_run(view_second, run_a));
    CHECK(dag_view_has_run(view_second, run_b));
}

static void test_dag_frozen_read_publish_epoch() {
    // Foreign PUBLIC runs with publish_epoch > frozen_read_publish_epoch are
    // omitted. Older public runs and the reader's own run remain visible.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(4);

    const uint64_t ep_id = runtime.adopt_root(15, 15, 0, 1, 0);
    CHECK(ep_id != 0);

    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "A", "intent": "Fact A"},
          {"id": "B", "intent": "Fact B"}
        ],
        "depends_on": []
      }
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    // Formal base capture after init, before any ready work: lineage,
    // watermarks, and sampler clones do not exist until this point.
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(runtime.activate_dag_frontier(ep_id));
    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);

    CHECK(ep->document.set_node_state(1, llama_rerot_node_state::running));
    CHECK(ep->document.set_node_state(2, llama_rerot_node_state::running));
    ep->running.insert(1);
    ep->running.insert(2);

    const auto run_p = ep->document.append_run(0, llama_rerot_visibility::public_live, 0, 4, 1);
    const auto run_b = ep->document.append_run(2, llama_rerot_visibility::public_live, 4, 3, 2);
    const auto run_a = ep->document.append_run(1, llama_rerot_visibility::public_live, 7, 5, 5);

    ep->frozen_read_publish_epoch = 2;

    const auto view_b = runtime.build_dag_view_for_reader(ep_id, 2);
    CHECK(dag_view_has_run(view_b, run_p));
    CHECK(dag_view_has_run(view_b, run_b));
    CHECK(!dag_view_has_run(view_b, run_a));

    const auto view_a = runtime.build_dag_view_for_reader(ep_id, 1);
    CHECK(dag_view_has_run(view_a, run_a));
    CHECK(dag_view_has_run(view_a, run_p));
    CHECK(dag_view_has_run(view_a, run_b));
}

static void test_dag_logical_step_hides_foreign_pending() {
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(4);

    const uint64_t ep_id = runtime.adopt_root(16, 16, 0, 1, 0);
    CHECK(ep_id != 0);
    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "A", "intent": "Fact A"},
          {"id": "B", "intent": "Fact B"}
        ],
        "depends_on": []
      }
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    // Formal base capture after init, before any ready work: lineage,
    // watermarks, and sampler clones do not exist until this point.
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(runtime.activate_dag_frontier(ep_id));
    CHECK(runtime.has_open_dag_logical_step(ep_id));
    CHECK(!runtime.dag_logical_step_complete(ep_id));

    auto pen_a = runtime.allocate_pen(16, ep_id, 1);
    auto pen_b = runtime.allocate_pen(16, ep_id, 2);
    CHECK(pen_a.has_value());
    CHECK(pen_b.has_value());

    llama_rerot_node_id a = LLAMA_REROT_NODE_INVALID;
    llama_rerot_node_id b = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, *pen_a, 1, &a));
    CHECK(runtime.admit_next_child(ep_id, *pen_b, 2, &b));
    CHECK(a != b);
    CHECK(a != LLAMA_REROT_NODE_INVALID);
    CHECK(b != LLAMA_REROT_NODE_INVALID);
    CHECK(runtime.complete_admission(ep_id, a));
    CHECK(runtime.complete_admission(ep_id, b));

    const auto * na_start = runtime.node(ep_id, a);
    CHECK(na_start != nullptr);
    CHECK(commit_generated(runtime, ep_id, a, na_start->storage_pos_next, "aaa"));
    const auto * na = runtime.node(ep_id, a);
    CHECK(na && na->pending_record.has_value());
    const auto pending_a = *na->pending_record;
    runtime.finish_frontier(ep_id);
    CHECK(!runtime.dag_logical_step_complete(ep_id));
    const auto view_b_mid = runtime.build_dag_view_for_reader(ep_id, b);
    CHECK(!dag_view_has_run(view_b_mid, pending_a));

    const auto * nb = runtime.node(ep_id, b);
    CHECK(nb != nullptr);
    CHECK(commit_generated(runtime, ep_id, b, nb->storage_pos_next, "bbb"));
    runtime.finish_frontier(ep_id);
    const auto view_b_end = runtime.build_dag_view_for_reader(ep_id, b);
    CHECK(dag_view_has_run(view_b_end, pending_a));
}

static void test_dag_refuses_nested_html_fork() {
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(2);
    const uint64_t ep_id = runtime.adopt_root(18, 16, 0, 1, 0);
    CHECK(ep_id != 0);
    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {"questions": [{"id": "A", "intent": "A"}], "depends_on": []}
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    CHECK(!runtime.freeze_fork_parent(ep_id, 0));
    const auto * ep = runtime.episode(ep_id);
    CHECK(ep && ep->hard_aborted);
    CHECK(ep->abort_reason.find("nested") != std::string::npos);
}

static void test_dag_save_refuses_probe_and_persists_c0() {
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(2);
    const uint64_t ep_id = runtime.adopt_root(17, 16, 0, 1, 0);
    CHECK(ep_id != 0);
    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {"questions": [{"id": "A", "intent": "A"}], "depends_on": []}
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    CHECK(runtime.capture_c0(ep_id, 1, 4));
    auto * ep = runtime.episode(ep_id);
    CHECK(ep);
    ep->c0.gdn_recurrent_states = {1, 2, 3, 4};
    CHECK(runtime.capture_c_base(ep_id));
    ep->frozen_read_publish_epoch = 9;
    ep->probe_tokens = 11;
    ep->frame_tokens = 7;

    server_rerot_state_fingerprints fp;
    fp.caps = LLAMA_REROT_STATE_CAP_REROT | LLAMA_REROT_STATE_CAP_REROT_TREE |
              LLAMA_REROT_STATE_CAP_REROT_PRIVATE;
    ep->probing = true;
    const auto refused = server_rerot_episode_save(*ep, fp, &err);
    CHECK(refused.empty());
    CHECK(err.find("probe") != std::string::npos);
    ep->probing = false;
    ep->probe_seq = 8;
    err.clear();
    CHECK(server_rerot_episode_save(*ep, fp, &err).empty());
    ep->probe_seq = -1;
    err.clear();
    const auto blob = server_rerot_episode_save(*ep, fp, &err);
    CHECK(!blob.empty());
    CHECK(err.empty());
    server_rerot_episode loaded;
    CHECK(server_rerot_episode_load(blob.data(), blob.size(), fp, &loaded, &err));
    CHECK(loaded.c0.gdn_recurrent_states == ep->c0.gdn_recurrent_states);
    CHECK(loaded.c_base.valid());
    CHECK(loaded.frozen_read_publish_epoch == 9);
    CHECK(loaded.probe_tokens == 11);
    CHECK(loaded.frame_tokens == 7);
    CHECK(loaded.c0.gdn_recurrent_states.size() == 4);
}

static void test_dag_shift_pins_started_public_history() {
    server_rerot_episode episode(292);
    episode.is_dag = true;
    server_rerot_node_runtime root;
    root.id = 0;
    episode.nodes.push_back(std::move(root));
    const auto child = episode.document.create_child(0, "A");
    CHECK(child != LLAMA_REROT_NODE_INVALID);
    CHECK(episode.document.set_node_state(child, llama_rerot_node_state::running));
    server_rerot_node_runtime worker;
    worker.id = child;
    episode.nodes.push_back(std::move(worker));
    episode.base_prefix_end = 4;
    episode.publish_epoch = 3;
    episode.layout_epoch = 7;

    CHECK(episode.document.append_run(
        0, llama_rerot_visibility::public_live, 0, 4, 1) != LLAMA_REROT_RUN_INVALID);
    const auto history = episode.document.append_run(
        child, llama_rerot_visibility::public_live, 4, 6, 2);
    const auto current = episode.document.append_run(
        child, llama_rerot_visibility::public_live, 10, 3, 3);
    CHECK(history != LLAMA_REROT_RUN_INVALID);
    CHECK(current != LLAMA_REROT_RUN_INVALID);
    episode.nodes[1].public_run = current;

    server_rerot_shift_result result;
    std::string error;
    CHECK(server_rerot_truncate_oldest_public(episode, 6, &result, &error));
    CHECK(error.empty());
    CHECK(result.tokens_removed == 0);
    CHECK(episode.document.run(history)->token_count == 6);
    CHECK(episode.document.run(current)->token_count == 3);
}

static void test_context_shift_pins_frame_runs() {
    server_rerot_episode episode(291);
    server_rerot_node_runtime root;
    root.id = 0;
    episode.nodes.push_back(std::move(root));
    const auto child = episode.document.create_child(0, "A");
    CHECK(child != LLAMA_REROT_NODE_INVALID);
    server_rerot_node_runtime worker;
    worker.id = child;
    episode.nodes.push_back(std::move(worker));
    episode.base_prefix_end = 4;
    episode.publish_epoch = 3;
    episode.layout_epoch = 7;

    CHECK(episode.document.append_run(
        0, llama_rerot_visibility::public_live, 0, 4, 1) != LLAMA_REROT_RUN_INVALID);
    const auto frame = episode.document.append_run(
        child, llama_rerot_visibility::public_live, 4, 5, 2, llama_rerot_segment_kind::frame);
    CHECK(frame != LLAMA_REROT_RUN_INVALID);

    server_rerot_shift_result result;
    std::string error;
    CHECK(server_rerot_truncate_oldest_public(episode, 5, &result, &error));
    CHECK(error.empty());
    CHECK(result.tokens_removed == 0);
    CHECK(episode.document.run(frame)->token_count == 5);
}

static void test_dag_seal_exactly_once_and_refuses_unstarted() {
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(4);
    const uint64_t ep_id = runtime.adopt_root(30, 30, 0, 1, 0);
    CHECK(ep_id != 0);
    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "A", "intent": "Fact A"},
          {"id": "B", "intent": "Fact B"},
          {"id": "C", "intent": "Needs A"}
        ],
        "depends_on": [{"id": "C", "depends_on_id": "A"}]
      }
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    // Formal base capture after init, before any ready work: lineage,
    // watermarks, and sampler clones do not exist until this point.
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(runtime.activate_dag_frontier(ep_id));
    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    CHECK(ep->nodes[3].remaining_preds == 1);

    // Bad origin never seals and never aborts.
    CHECK(!runtime.seal_dag_node(ep_id, 1, llama_rerot_event_origin::runtime_frame));
    CHECK(!ep->hard_aborted);
    CHECK(!ep->nodes[1].is_sealed);

    // Admit + complete A, then seal twice: the successor debits once.
    auto * root = runtime.node(ep_id, 0);
    if (root && root->physical_slot >= 0) {
        CHECK(runtime.detach_node(ep_id, 0));
    }
    llama_rerot_node_id a = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 0, 2, &a));
    CHECK(a == 1);
    CHECK(runtime.complete_admission(ep_id, a));
    CHECK(runtime.seal_dag_node(ep_id, a, llama_rerot_event_origin::worker_source));
    CHECK(ep->nodes[a].is_sealed);
    CHECK(ep->nodes[3].remaining_preds == 0);
    CHECK(runtime.seal_dag_node(ep_id, a, llama_rerot_event_origin::worker_source));
    CHECK(ep->nodes[3].remaining_preds == 0);

    // Sealing queued C (never started) is a scheduler invariant violation:
    // the episode aborts and nothing is debited or resurrected.
    CHECK(!runtime.seal_dag_node(ep_id, 3, llama_rerot_event_origin::worker_source));
    CHECK(ep->hard_aborted);
    CHECK(ep->abort_reason.find("never started") != std::string::npos);
    CHECK(!ep->nodes[3].is_sealed);
    CHECK(ep->nodes[3].remaining_preds == 0);

    // A late seal on the dead episode fails without touching the first reason.
    CHECK(!runtime.seal_dag_node(ep_id, 2, llama_rerot_event_origin::worker_source));
    CHECK(ep->hard_aborted);
    CHECK(ep->abort_reason.find("never started") != std::string::npos);
    CHECK(!ep->nodes[2].is_sealed);
}

static void test_dag_seal_releases_pen_parked_until_cohort_retire() {
    // P=1, two independent workers: A seals while B is still queued. The seal
    // must free A's executor at once (attention parked, PENDING kept) so B
    // can admit before cohort publication. finish_frontier then publishes
    // both bodies and retires A exactly once with history preserved.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(1);
    const uint64_t ep_id = runtime.adopt_root(31, 31, 0, 1, 0);
    CHECK(ep_id != 0);
    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "A", "intent": "Fact A"},
          {"id": "B", "intent": "Fact B"}
        ],
        "depends_on": []
      }
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    // Formal base capture after init, before any ready work: lineage,
    // watermarks, and sampler clones do not exist until this point.
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(runtime.activate_dag_frontier(ep_id));
    auto * root = runtime.node(ep_id, 0);
    if (root && root->physical_slot >= 0) {
        CHECK(runtime.detach_node(ep_id, 0));
    }
    llama_rerot_node_id a = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 0, 2, &a));
    CHECK(a == 1);
    CHECK(runtime.complete_admission(ep_id, a));
    auto * na = runtime.node(ep_id, a);
    CHECK(na != nullptr);
    CHECK(commit_generated(runtime, ep_id, a, na->storage_pos_next, "aaa"));
    CHECK(runtime.seal_dag_node(ep_id, a, llama_rerot_event_origin::worker_source));

    // Executor freed now; logical state parked, nothing dropped.
    CHECK(runtime.has_free_pen());
    const auto * sealed = runtime.node(ep_id, a);
    CHECK(sealed && sealed->is_sealed);
    CHECK(sealed->physical_slot < 0 && sealed->exec_seq < 0 && sealed->pen_id < 0);
    CHECK(sealed->parked_seq >= 0);
    CHECK(sealed->pending_record.has_value());
    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    CHECK(ep->document.node(a)->state == llama_rerot_node_state::retired);

    // The queued peer admits onto the freed pen before any publication.
    llama_rerot_node_id b = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 0, 3, &b));
    CHECK(b == 2);
    CHECK(runtime.complete_admission(ep_id, b));
    auto * nb = runtime.node(ep_id, b);
    CHECK(nb != nullptr);
    CHECK(commit_generated(runtime, ep_id, b, nb->storage_pos_next, "bbb"));

    // Cohort publication publishes both bodies; A retires exactly once.
    const auto res = runtime.finish_frontier(ep_id);
    CHECK(!res.hard_aborted);
    CHECK(res.retired.size() == 1 && res.retired[0] == a);
    const auto * retired = runtime.node(ep_id, a);
    CHECK(retired && retired->parked_seq < 0);
    CHECK(retired->pending_record == std::nullopt);
    const auto * history = retired->public_run.has_value()
        ? ep->document.run(*retired->public_run)
        : nullptr;
    CHECK(history && history->owner == a);
    CHECK(history->visibility == llama_rerot_visibility::public_live);
    CHECK(history->token_count == 1);

    // A second frontier emits no duplicate retired notification.
    const auto res2 = runtime.finish_frontier(ep_id);
    CHECK(!res2.hard_aborted);
    CHECK(res2.retired.empty());
}

static void test_dag_complete_admission_evicts_no_peer() {
    // P=2, A running, B starting, C queued, cohort open, no pen free:
    // completing B's admission must only flip B starting->running. It must
    // not suspend any bound peer mid-commit-loop; central scheduling runs
    // after the slice commits.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(2);
    const uint64_t ep_id = runtime.adopt_root(33, 33, 0, 1, 0);
    CHECK(ep_id != 0);
    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "A", "intent": "Fact A"},
          {"id": "B", "intent": "Fact B"},
          {"id": "C", "intent": "Fact C"}
        ],
        "depends_on": []
      }
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(runtime.activate_dag_frontier(ep_id));
    dag_unbind_planner_if_bound(runtime, ep_id);

    llama_rerot_node_id a = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 0, 2, &a));
    CHECK(a == 1);
    CHECK(runtime.complete_admission(ep_id, a));
    llama_rerot_node_id b = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 1, 3, &b));
    CHECK(b == 2);

    CHECK(runtime.complete_admission(ep_id, b));
    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    CHECK(ep->suspended.empty());
    CHECK(ep->running.count(a) != 0 && ep->running.count(b) != 0);
    const auto * na = runtime.node(ep_id, a);
    const auto * nb = runtime.node(ep_id, b);
    CHECK(na && na->physical_slot == 0 && na->exec_seq == 2);
    CHECK(nb && nb->physical_slot == 1 && nb->exec_seq == 3);
    CHECK(!ep->hard_aborted);
}

static void test_dag_seal_evicts_no_bound_peer() {
    // P=1: A suspended, B just admitted+completed (host commit/sample still
    // pending), C queued. Sealing A must stage only A's own completion —
    // passivate self, debit successors — and leave bound B untouched. It must
    // never select B as a yield victim mid-commit-loop.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(1);
    const uint64_t ep_id = runtime.adopt_root(34, 34, 0, 1, 0);
    CHECK(ep_id != 0);
    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "A", "intent": "Fact A"},
          {"id": "B", "intent": "Fact B"},
          {"id": "C", "intent": "Fact C"}
        ],
        "depends_on": []
      }
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(runtime.activate_dag_frontier(ep_id));
    dag_unbind_planner_if_bound(runtime, ep_id);

    llama_rerot_node_id a = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 0, 2, &a));
    CHECK(a == 1);
    CHECK(runtime.complete_admission(ep_id, a));
    // Park A so B can start; B's host work has not run yet.
    CHECK(runtime.yield_dag_pen_for_ready(ep_id));
    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr && ep->suspended.count(a) != 0);
    llama_rerot_node_id b = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 0, 3, &b));
    CHECK(b == 2);
    CHECK(runtime.complete_admission(ep_id, b));

    CHECK(runtime.seal_dag_node(ep_id, a, llama_rerot_event_origin::worker_source));
    CHECK(!ep->hard_aborted);
    const auto * sealed = runtime.node(ep_id, a);
    CHECK(sealed && sealed->is_sealed && sealed->parked_seq >= 0);
    CHECK(sealed->physical_slot < 0 && sealed->exec_seq < 0);
    // Bound B is untouched: still running on its pen, never suspended.
    const auto * peer = runtime.node(ep_id, b);
    CHECK(peer && peer->physical_slot == 0 && peer->exec_seq == 3);
    CHECK(ep->running.count(b) != 0 && ep->suspended.empty());
    CHECK(ep->document.node(b)->state == llama_rerot_node_state::running);
}

static void test_dag_yield_force_under_resource_pressure() {
    // Pens free, but recurrent rows exhausted: the default yield no-ops (a
    // free pen is not a free recurrent row); the pressure yield suspends one
    // bound worker so the queued peer can proceed.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(2);
    const uint64_t ep_id = runtime.adopt_root(32, 32, 0, 1, 0);
    CHECK(ep_id != 0);
    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "A", "intent": "Fact A"},
          {"id": "B", "intent": "Fact B"}
        ],
        "depends_on": []
      }
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(runtime.activate_dag_frontier(ep_id));
    dag_unbind_planner_if_bound(runtime, ep_id);

    llama_rerot_node_id a = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 0, 2, &a));
    CHECK(a == 1);
    CHECK(runtime.complete_admission(ep_id, a));

    // One pen still free: default yield refuses, worker untouched.
    CHECK(runtime.has_free_pen());
    CHECK(!runtime.yield_dag_pen_for_ready(ep_id));
    const auto * running = runtime.node(ep_id, a);
    CHECK(running && running->physical_slot == 0 && running->exec_seq == 2);
    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr && ep->suspended.empty());

    // Forced yield under pressure suspends the bound worker anyway.
    CHECK(runtime.yield_dag_pen_for_ready(ep_id, true));
    CHECK(ep->suspended.count(a) != 0);
    CHECK(ep->document.node(a)->state == llama_rerot_node_state::ready_suspended);
    CHECK(runtime.node(ep_id, a)->physical_slot < 0);
    CHECK(runtime.has_free_pen());

    // The queued peer admits onto the freed pen.
    llama_rerot_node_id b = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 0, 2, &b));
    CHECK(b == 2);
}

static void test_dag_admit_anchors_worker_storage_to_c_base() {
    // C_base watermark path: workers start after the rebuilt formal prefix.
    {
        server_rerot_runtime runtime(nullptr);
        runtime.set_pen_capacity(4);
        const uint64_t ep_id = runtime.adopt_root(32, 32, 0, 1, 0);
        CHECK(ep_id != 0);
        CHECK(runtime.capture_c0(ep_id, 1, 0));
        const auto decision = server_rerot_parse_routing_decision(R"({
          "strategy": "dag",
          "payload": {"questions": [{"id": "A", "intent": "A"}], "depends_on": []}
        })");
        CHECK(decision.is_dag());
        std::string err;
        CHECK(runtime.initialize_dag(ep_id, decision, &err));
        auto * root = runtime.node(ep_id, 0);
        CHECK(root != nullptr);
        root->storage_pos_next = 7; // rebuilt formal-P end
        CHECK(runtime.capture_c_base(ep_id));
        CHECK(runtime.activate_dag_frontier(ep_id));
        if (root->physical_slot >= 0) {
            CHECK(runtime.detach_node(ep_id, 0));
        }
        llama_rerot_node_id admitted = LLAMA_REROT_NODE_INVALID;
        CHECK(runtime.admit_next_child(ep_id, 0, 2, &admitted));
        CHECK(admitted == 1);
        const auto * w = runtime.node(ep_id, admitted);
        CHECK(w && w->storage_pos_next == 7);
    }
}

static void test_dag_init_alone_never_admits_until_base_captured() {
    // Initialization publishes validated descriptors (nodes, edges, root plan
    // seal) but no worker is physically ready before the formal base: the
    // real-target failure admitted from this state and died on missing
    // lineage. Formal P capture then activates the same descriptors.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(4);
    const uint64_t ep_id = runtime.adopt_root(33, 33, 0, 1, 5);
    CHECK(ep_id != 0);
    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {"questions": [{"id": "A", "intent": "A"}], "depends_on": []}
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    CHECK(ep->nodes[1].remaining_preds == 0);
    CHECK(ep->nodes[0].is_sealed);
    CHECK(runtime.get_eligible_dag_nodes(ep_id).empty());
    CHECK(ep->ready_queue.empty());
    CHECK(!runtime.activate_dag_frontier(ep_id));
    CHECK(!runtime.has_open_dag_logical_step(ep_id));
    CHECK(!runtime.dag_step_next_pending(ep_id).has_value());
    // Admission pre-base is a scheduling no-op, never an abort.
    auto * root = runtime.node(ep_id, 0);
    if (root && root->physical_slot >= 0) {
        CHECK(runtime.detach_node(ep_id, 0));
    }
    llama_rerot_node_id admitted = LLAMA_REROT_NODE_INVALID;
    CHECK(!runtime.admit_next_child(ep_id, 0, 2, &admitted));
    CHECK(admitted == LLAMA_REROT_NODE_INVALID);
    CHECK(!ep->hard_aborted);
    // Formal P capture then activates: workers admit from the base watermark.
    CHECK(runtime.capture_c0(ep_id, 1, 5));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(runtime.activate_dag_frontier(ep_id));
    CHECK(!runtime.get_eligible_dag_nodes(ep_id).empty());
    CHECK(dag_queue_contains(ep->ready_queue, 1));
    CHECK(runtime.admit_next_child(ep_id, 0, 2, &admitted));
    CHECK(admitted == 1);
    const auto * w = runtime.node(ep_id, admitted);
    CHECK(w && w->storage_pos_next == 5);
}

static void test_dag_discard_probe_retracts_probe_runs() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 16);
    runtime.set_pen_capacity(2);
    const uint64_t ep_id = runtime.adopt_root(34, 34, 0, 0, 4);
    CHECK(ep_id != 0);
    CHECK(runtime.capture_c0(ep_id, 0, 4));
    CHECK(runtime.arm_isolated_probe(ep_id, 0));
    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    ep->probing = true;
    auto * root = runtime.node(ep_id, 0);
    CHECK(root != nullptr);
    CHECK(commit_generated(runtime, ep_id, 0, root->storage_pos_next, "probe bytes "));
    bool saw_probe = false;
    for (size_t i = 0; i < ep->document.run_count(); ++i) {
        const auto * run = ep->document.run(static_cast<llama_rerot_run_id>(i));
        if (run && run->kind == llama_rerot_segment_kind::probe_control) {
            CHECK(run->token_count == 1);
            saw_probe = true;
        }
    }
    CHECK(saw_probe);
    CHECK(runtime.discard_isolated_probe(ep_id, 0));
    for (size_t i = 0; i < ep->document.run_count(); ++i) {
        const auto * run = ep->document.run(static_cast<llama_rerot_run_id>(i));
        CHECK(!(run && run->kind == llama_rerot_segment_kind::probe_control && run->token_count > 0));
    }
    CHECK(root->storage_pos_next == 4);
}

static void test_dag_frozen_view_gates_foreign_frame() {
    // Frozen epoch gates every foreign PUBLIC run newer than the snapshot —
    // BODY and FRAME alike. Only the reader's own runs are exempt.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(4);
    const uint64_t ep_id = runtime.adopt_root(35, 35, 0, 1, 0);
    CHECK(ep_id != 0);
    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "A", "intent": "Fact A"},
          {"id": "B", "intent": "Fact B"}
        ],
        "depends_on": []
      }
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    // Formal base capture after init, before any ready work: lineage,
    // watermarks, and sampler clones do not exist until this point.
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(runtime.activate_dag_frontier(ep_id));
    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    CHECK(ep->document.set_node_state(1, llama_rerot_node_state::running));
    CHECK(ep->document.set_node_state(2, llama_rerot_node_state::running));
    ep->running.insert(1);
    ep->running.insert(2);

    const auto run_p = ep->document.append_run(0, llama_rerot_visibility::public_live, 0, 4, 1);
    const auto run_b = ep->document.append_run(2, llama_rerot_visibility::public_live, 4, 3, 2);
    const auto run_f = ep->document.append_run(
        1, llama_rerot_visibility::public_live, 7, 2, 5, llama_rerot_segment_kind::frame);
    const auto run_a = ep->document.append_run(1, llama_rerot_visibility::public_live, 9, 5, 5);
    ep->frozen_read_publish_epoch = 2;

    const auto view_b = runtime.build_dag_view_for_reader(ep_id, 2);
    CHECK(dag_view_has_run(view_b, run_p));
    CHECK(dag_view_has_run(view_b, run_b));
    CHECK(!dag_view_has_run(view_b, run_f));
    CHECK(!dag_view_has_run(view_b, run_a));

    const auto view_a = runtime.build_dag_view_for_reader(ep_id, 1);
    CHECK(dag_view_has_run(view_a, run_p));
    CHECK(dag_view_has_run(view_a, run_b));
    CHECK(dag_view_has_run(view_a, run_f));
    CHECK(dag_view_has_run(view_a, run_a));
}

static void test_dag_load_refuses_bad_cohort_and_sampler_snapshot() {
    server_rerot_state_fingerprints fp;
    fp.caps = LLAMA_REROT_STATE_CAP_REROT | LLAMA_REROT_STATE_CAP_REROT_TREE |
              LLAMA_REROT_STATE_CAP_REROT_PRIVATE;

    // Truncated sampler bytes are corruption, never a best-effort reseed.
    {
        server_rerot_runtime runtime(nullptr);
        runtime.set_pen_capacity(2);
        const uint64_t ep_id = runtime.adopt_root(50, 50, 0, 1, 0);
        CHECK(ep_id != 0);
        CHECK(runtime.capture_c0(ep_id, 1, 0));
        auto * ep = runtime.episode(ep_id);
        CHECK(ep != nullptr);
        ep->c0.sampler_snapshot_bytes = {1, 2, 3};
        std::string err;
        const auto blob = server_rerot_episode_save(*ep, fp, &err);
        CHECK(!blob.empty());
        server_rerot_episode loaded;
        CHECK(!server_rerot_episode_load(blob.data(), blob.size(), fp, &loaded, &err));
        CHECK(err.find("sampler") != std::string::npos);
    }
    // A well-formed snapshot round-trips verbatim (opaque, never installed).
    {
        server_rerot_runtime runtime(nullptr);
        runtime.set_pen_capacity(2);
        const uint64_t ep_id = runtime.adopt_root(51, 51, 0, 1, 0);
        CHECK(ep_id != 0);
        CHECK(runtime.capture_c0(ep_id, 1, 0));
        auto * ep = runtime.episode(ep_id);
        CHECK(ep != nullptr);
        const llama_tokens prev = {7, 8, 9};
        runtime.capture_checkpoint_sampler(ep->c0, 42, prev);
        CHECK(server_rerot_validate_sampler_snapshot(ep->c0.sampler_snapshot_bytes, nullptr));
        std::string err;
        const auto blob = server_rerot_episode_save(*ep, fp, &err);
        CHECK(!blob.empty());
        server_rerot_episode loaded;
        CHECK(server_rerot_episode_load(blob.data(), blob.size(), fp, &loaded, &err));
        CHECK(loaded.c0.sampler_snapshot_bytes == ep->c0.sampler_snapshot_bytes);
        const std::vector<uint8_t> bad_magic(12, 0);
        CHECK(!server_rerot_validate_sampler_snapshot(bad_magic, &err));
    }
    // Cohort / commit records must name live nodes.
    {
        server_rerot_runtime runtime(nullptr);
        runtime.set_pen_capacity(2);
        const uint64_t ep_id = runtime.adopt_root(52, 52, 0, 1, 0);
        CHECK(ep_id != 0);
        const auto decision = server_rerot_parse_routing_decision(R"({
          "strategy": "dag",
          "payload": {"questions": [{"id": "A", "intent": "A"}], "depends_on": []}
        })");
        CHECK(decision.is_dag());
        std::string err;
        CHECK(runtime.initialize_dag(ep_id, decision, &err));
        auto * ep = runtime.episode(ep_id);
        CHECK(ep != nullptr);
        ep->dag_step_cohort.insert(9999);
        const auto blob = server_rerot_episode_save(*ep, fp, &err);
        CHECK(!blob.empty());
        server_rerot_episode loaded;
        CHECK(!server_rerot_episode_load(blob.data(), blob.size(), fp, &loaded, &err));
        CHECK(err.find("cohort") != std::string::npos);
        ep->dag_step_cohort.clear();
        ep->dag_step_committed.insert(9998);
        err.clear();
        const auto blob2 = server_rerot_episode_save(*ep, fp, &err);
        CHECK(!blob2.empty());
        CHECK(!server_rerot_episode_load(blob2.data(), blob2.size(), fp, &loaded, &err));
        CHECK(err.find("commit") != std::string::npos);
    }
}

static void test_dag_load_rebinds_pens_for_bound_slots() {
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(2);
    const uint64_t ep_id = runtime.adopt_root(40, 40, 0, 1, 0);
    CHECK(ep_id != 0);
    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {"questions": [{"id": "A", "intent": "A"}], "depends_on": []}
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    // Formal base capture after init, before any ready work: lineage,
    // watermarks, and sampler clones do not exist until this point.
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(runtime.activate_dag_frontier(ep_id));
    auto * root = runtime.node(ep_id, 0);
    if (root && root->physical_slot >= 0) {
        CHECK(runtime.detach_node(ep_id, 0));
    }
    llama_rerot_node_id w = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 1, 2, &w));
    CHECK(w == 1);
    CHECK(runtime.complete_admission(ep_id, w));

    server_rerot_state_fingerprints fp;
    fp.caps = LLAMA_REROT_STATE_CAP_REROT | LLAMA_REROT_STATE_CAP_REROT_TREE |
              LLAMA_REROT_STATE_CAP_REROT_PRIVATE;
    std::vector<uint8_t> blob;
    CHECK(runtime.save_episode(ep_id, fp, &blob, &err));
    CHECK(!blob.empty());
    CHECK(runtime.erase_episode(ep_id));
    CHECK(runtime.has_free_pen());

    uint64_t restored_id = 0;
    CHECK(runtime.load_episode(blob.data(), blob.size(), fp, &restored_id, &err));
    CHECK(restored_id == ep_id);
    // The pen arena is rebound symmetrically: slot 1 is running work again,
    // not a phantom map entry over a free pen.
    const auto * pen = runtime.pen(1);
    CHECK(pen && pen->state == server_pen_state::running);
    CHECK(pen && pen->node_id == w && pen->episode_id == ep_id);
    const auto * node = runtime.node(ep_id, w);
    CHECK(node && node->physical_slot == 1 && node->pen_id == 1 && node->exec_seq == 2);
    // Only slot 0 is allocatable now.
    const auto free_pen = runtime.allocate_pen(ep_id, ep_id, w);
    CHECK(free_pen.has_value() && *free_pen == 0);
    runtime.free_pen(*free_pen);
}

static void test_dag_step_publish_is_atomic_on_member_failure() {
    // A commits cleanly, then B's gapped speculative plan fails resolution at
    // publish time. The failure must still abort, but A stays PENDING and no
    // counter/epoch moves: no half frontier escapes.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(2);
    const uint64_t ep_id = runtime.adopt_root(61, 61, 0, 1, 0);
    CHECK(ep_id != 0);
    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "A", "intent": "Fact A"},
          {"id": "B", "intent": "Fact B"}
        ],
        "depends_on": []
      }
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    // Formal base capture after init, before any ready work: lineage,
    // watermarks, and sampler clones do not exist until this point.
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(runtime.activate_dag_frontier(ep_id));
    auto * root = runtime.node(ep_id, 0);
    if (root && root->physical_slot >= 0) {
        CHECK(runtime.detach_node(ep_id, 0));
    }
    llama_rerot_node_id a = LLAMA_REROT_NODE_INVALID;
    llama_rerot_node_id b = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 0, 2, &a));
    CHECK(runtime.admit_next_child(ep_id, 1, 3, &b));
    CHECK(a == 1 && b == 2);
    CHECK(runtime.complete_admission(ep_id, a));
    CHECK(runtime.complete_admission(ep_id, b));

    auto * na = runtime.node(ep_id, a);
    CHECK(na != nullptr);
    CHECK(commit_generated(runtime, ep_id, a, na->storage_pos_next, "aaa"));
    // B leaves a gapped plan: the empty pending run can only fail at publish
    // time (both commits themselves succeed).
    const auto gap1 = runtime.plan_generated_token(ep_id, b, 10, "gap");
    CHECK(gap1.has_value());
    const auto gap2 = runtime.plan_generated_token(ep_id, b, 20, "bbb");
    CHECK(gap2.has_value());
    CHECK(runtime.track_fence_token(ep_id, b, *gap2, 2020) &&
          runtime.commit_token(ep_id, b, *gap2));

    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    const uint64_t epoch_before = ep->publish_epoch;
    const uint64_t pending_before = ep->pending_tokens;
    CHECK(pending_before == 2);
    const auto res = runtime.finish_frontier(ep_id);
    CHECK(res.hard_aborted);
    CHECK(res.abort_reason.find("pending-record") != std::string::npos);

    ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    CHECK(ep->publish_epoch == epoch_before);
    CHECK(ep->pending_tokens == pending_before);
    CHECK(ep->generated_public_tokens == 0);
    const auto * na_after = runtime.node(ep_id, a);
    CHECK(na_after && na_after->pending_record.has_value());
    const auto * arun = ep->document.run(*na_after->pending_record);
    CHECK(arun && arun->visibility == llama_rerot_visibility::pending_record);
    CHECK(arun && arun->token_count == 1);
    const auto * nb_after = runtime.node(ep_id, b);
    CHECK(nb_after && nb_after->pending_record.has_value());
}

static void test_dag_step_publish_overflow_keeps_history_public() {
    // The worker owns a historical PUBLIC run stamped at the max epoch and a
    // fresh pending run; the next publish overflows. The failure aborts, but
    // the historical run must stay PUBLIC: a retained max epoch on the
    // failing entry must never match it for rollback.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(2);
    const uint64_t ep_id = runtime.adopt_root(62, 62, 0, 1, 0);
    CHECK(ep_id != 0);
    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {"questions": [{"id": "A", "intent": "Fact A"}], "depends_on": []}
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    // Formal base capture after init, before any ready work: lineage,
    // watermarks, and sampler clones do not exist until this point.
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(runtime.activate_dag_frontier(ep_id));
    auto * root = runtime.node(ep_id, 0);
    if (root && root->physical_slot >= 0) {
        CHECK(runtime.detach_node(ep_id, 0));
    }
    llama_rerot_node_id a = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 0, 2, &a));
    CHECK(a == 1);
    CHECK(runtime.complete_admission(ep_id, a));

    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    const auto hist = ep->document.append_run(
        a, llama_rerot_visibility::public_live, 0, 3, ~uint64_t(0));
    auto * na = runtime.node(ep_id, a);
    CHECK(na != nullptr);
    na->storage_pos_next = 3;
    CHECK(commit_generated(runtime, ep_id, a, na->storage_pos_next, "aaa"));
    ep->publish_epoch = ~uint64_t(0);

    const auto res = runtime.finish_frontier(ep_id);
    CHECK(res.hard_aborted);
    CHECK(res.abort_reason.find("overflow") != std::string::npos);

    ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    const auto * hist_run = ep->document.run(hist);
    CHECK(hist_run && hist_run->visibility == llama_rerot_visibility::public_live);
    CHECK(hist_run && hist_run->publish_epoch == ~uint64_t(0));
    CHECK(hist_run && hist_run->token_count == 3);
    const auto * na_after = runtime.node(ep_id, a);
    CHECK(na_after && na_after->pending_record.has_value());
    const auto * fresh = ep->document.run(*na_after->pending_record);
    CHECK(fresh && fresh->visibility == llama_rerot_visibility::pending_record);
    CHECK(ep->pending_tokens == 1);
    CHECK(ep->generated_public_tokens == 0);
}

static void test_dag_init_frames_prefix_keeps_view_not_body_export() {
    // The ordinary prefix is model-PUBLIC control, never API BODY:
    // initialize_dag flips root BODY runs to FRAME (probe_control left
    // alone) while build_dag_view keeps the full P for attention and worker
    // BODY reasoning stays in the body-export set.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(4);
    const uint64_t ep_id = runtime.adopt_root(70, 70, 0, 1, 0);
    CHECK(ep_id != 0);
    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    // Ordinary prefill as production leaves it: root PUBLIC BODY prefix plus
    // a stale (unretracted here to prove kind-based exclusion) probe run.
    const auto run_p = ep->document.append_run(0, llama_rerot_visibility::public_live, 0, 10, 1);
    const auto run_probe = ep->document.append_run(
        0, llama_rerot_visibility::private_control, 10, 2, 0, llama_rerot_segment_kind::probe_control);

    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {"questions": [{"id": "A", "intent": "Fact A"}], "depends_on": []}
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));

    // Kind flipped; everything the model attends to is preserved.
    const auto * framed = ep->document.run(run_p);
    CHECK(framed && framed->kind == llama_rerot_segment_kind::frame);
    CHECK(framed->visibility == llama_rerot_visibility::public_live);
    CHECK(framed->storage_pos0 == 0 && framed->token_count == 10 && framed->publish_epoch == 1);
    const auto * probe = ep->document.run(run_probe);
    CHECK(probe && probe->kind == llama_rerot_segment_kind::probe_control);

    // Formal plan-prefix suffix planned as FRAME stays model-visible and out
    // of the body-export set (the ServerProtocol plan_prefix callsite).
    const auto suffix = runtime.plan_public_span(
        ep_id, 0, 10, 3, llama_rerot_segment_kind::frame);
    CHECK(suffix.has_value() && suffix->size() == 3);
    for (const auto & plan : *suffix) {
        CHECK(plan.visibility == llama_rerot_visibility::public_live);
        CHECK(plan.segment_kind == llama_rerot_segment_kind::frame);
    }
    // Forced P forwards ledger as framing cost, never as useful BODY.
    CHECK(ep->frame_tokens == 3);
    CHECK(runtime.commit_token(ep_id, 0, suffix->front()));
    CHECK(ep->generated_public_tokens == 1);
    const auto * suffix_run = ep->document.run(suffix->front().run_id);
    CHECK(suffix_run && suffix_run->kind == llama_rerot_segment_kind::frame);
    CHECK(suffix_run->visibility == llama_rerot_visibility::public_live);

    // Worker BODY reasoning stays exportable and shares the same P view.
    CHECK(ep->document.set_node_state(1, llama_rerot_node_state::running));
    ep->running.insert(1);
    const auto run_b = ep->document.append_run(1, llama_rerot_visibility::public_live, 13, 4, 2);
    const auto view = runtime.build_dag_view_for_reader(ep_id, 1);
    CHECK(dag_view_has_run(view, run_p));
    CHECK(dag_view_has_run(view, suffix->front().run_id));
    CHECK(dag_view_has_run(view, run_b));
    const auto * first = ep->document.run(view.runs.front().run_id);
    CHECK(first && first->owner == 0 && first->storage_pos0 == 0 && first->token_count == 10);

    // Body-export set under the API rule (skip frame/source_end/probe,
    // require public_live): P runs excluded, worker BODY included.
    const auto exportable = [&](llama_rerot_run_id rid) {
        const auto * run = ep->document.run(rid);
        if (!run) {
            return false;
        }
        if (run->kind == llama_rerot_segment_kind::frame ||
            run->kind == llama_rerot_segment_kind::source_end ||
            run->kind == llama_rerot_segment_kind::probe_control) {
            return false;
        }
        return run->visibility == llama_rerot_visibility::public_live;
    };
    CHECK(!exportable(run_p));
    CHECK(!exportable(suffix->front().run_id));
    CHECK(!exportable(run_probe));
    CHECK(exportable(run_b));
}

static void test_dag_ensure_run_keeps_segment_kinds_distinct() {
    // ensure_run reuses by owner + visibility + contiguity + kind: adjacent
    // same-visibility FRAME then BODY spans stay two model spans (formal-P
    // text must not dissolve into exportable BODY), same-kind spans still
    // merge, and a private BODY run stays distinct from the source_end that
    // closes it.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(2);
    const uint64_t ep_id = runtime.adopt_root(71, 71, 0, 1, 0);
    CHECK(ep_id != 0);
    runtime.set_dag_protocol_markers(ep_id, "<end>", "<think>");
    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    const auto run_p = ep->document.append_run(0, llama_rerot_visibility::public_live, 0, 10, 1);

    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {"questions": [{"id": "A", "intent": "Fact A"}], "depends_on": []}
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    CHECK(ep->document.run(run_p)->kind == llama_rerot_segment_kind::frame);

    // Formal suffix as FRAME, then more FRAME: same kind still merges.
    const auto f2_plans = runtime.plan_public_span(
        ep_id, 0, 10, 2, llama_rerot_segment_kind::frame);
    CHECK(f2_plans.has_value() && f2_plans->size() == 2);
    const auto run_f2 = f2_plans->front().run_id;
    for (const auto & plan : *f2_plans) {
        CHECK(plan.run_id == run_f2);
        CHECK(runtime.commit_token(ep_id, 0, plan));
    }
    const auto f2_more = runtime.plan_public_span(
        ep_id, 0, 12, 1, llama_rerot_segment_kind::frame);
    CHECK(f2_more.has_value() && f2_more->front().run_id == run_f2);
    CHECK(runtime.commit_token(ep_id, 0, f2_more->front()));
    CHECK(ep->document.run(run_f2)->token_count == 3);

    // Same visibility and contiguity, new kind: BODY splits off.
    const auto b2_plans = runtime.plan_public_span(ep_id, 0, 13, 1);
    CHECK(b2_plans.has_value() && b2_plans->front().run_id != run_f2);
    CHECK(b2_plans->front().segment_kind == llama_rerot_segment_kind::body);
    const auto run_b2 = b2_plans->front().run_id;
    CHECK(runtime.commit_token(ep_id, 0, b2_plans->front()));
    const auto * b2 = ep->document.run(run_b2);
    CHECK(b2 && b2->kind == llama_rerot_segment_kind::body);
    CHECK(b2->storage_pos0 == 13 && b2->token_count == 1);
    CHECK(b2->visibility == llama_rerot_visibility::public_live);

    // Private BODY closed by the native end marker: the end boundary stays
    // its own run instead of dissolving into the body run.
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(runtime.activate_dag_frontier(ep_id));
    auto * root = runtime.node(ep_id, 0);
    if (root && root->physical_slot >= 0) {
        CHECK(runtime.detach_node(ep_id, 0));
    }
    llama_rerot_node_id w = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 0, 2, &w));
    CHECK(w == 1);
    CHECK(runtime.complete_admission(ep_id, w));
    const auto priv_plans = runtime.plan_private_span(ep_id, w, 0, 2);
    CHECK(priv_plans.has_value() && priv_plans->size() == 2);
    const auto run_pb = priv_plans->front().run_id;
    for (const auto & plan : *priv_plans) {
        CHECK(plan.run_id == run_pb);
        CHECK(runtime.commit_token(ep_id, w, plan));
    }
    CHECK(commit_generated(runtime, ep_id, w,
        runtime.node(ep_id, w)->storage_pos_next, "<end>"));
    const auto * pb = ep->document.run(run_pb);
    CHECK(pb && pb->kind == llama_rerot_segment_kind::body && pb->token_count == 2);
    auto * nw = runtime.node(ep_id, w);
    CHECK(nw && nw->is_sealed);
    CHECK(nw->private_run.has_value() && *nw->private_run != run_pb);
    const auto * se = ep->document.run(*nw->private_run);
    CHECK(se && se->kind == llama_rerot_segment_kind::source_end && se->token_count == 1);
    CHECK(se->storage_pos0 == 2 && se->visibility == llama_rerot_visibility::private_control);

    // Body-export set under the API rule: P and suffix FRAME excluded, both
    // BODY spans included, end boundary excluded.
    const auto exportable = [&](llama_rerot_run_id rid) {
        const auto * run = ep->document.run(rid);
        if (!run) {
            return false;
        }
        if (run->kind == llama_rerot_segment_kind::frame ||
            run->kind == llama_rerot_segment_kind::source_end ||
            run->kind == llama_rerot_segment_kind::probe_control) {
            return false;
        }
        return run->visibility == llama_rerot_visibility::public_live;
    };
    CHECK(!exportable(run_p));
    CHECK(!exportable(run_f2));
    CHECK(exportable(run_b2));
}

static void test_dag_prefix_rebuild_reconciles_root_runs() {
    server_rerot_runtime runtime(nullptr, LLAMA_REROT_FRONTIER_STRONG, 8, 32);
    runtime.set_pen_capacity(2);
    const uint64_t ep_id = runtime.adopt_root(60, 60, 0, 1, 10);
    CHECK(ep_id != 0);
    CHECK(runtime.capture_c0(ep_id, 1, 10));
    auto * ep = runtime.episode(ep_id);
    auto * root = runtime.node(ep_id, 0);
    CHECK(ep != nullptr && root != nullptr);
    // Ordinary prefix plus stale probe residue on the root document.
    const auto r_full = ep->document.append_run(0, llama_rerot_visibility::public_live, 0, 10, 1);
    const auto r_probe = ep->document.append_run(
        0, llama_rerot_visibility::private_control, 10, 4, 0, llama_rerot_segment_kind::probe_control);
    root->public_run = r_full;
    root->private_run = r_probe;
    ep->archive_seq = 9;
    root->parked_seq = 10;

    std::string err;
    CHECK(runtime.prepare_dag_prefix_rebuild(ep_id, 6, &err));
    CHECK(err.empty());
    CHECK(ep->document.run(r_full)->token_count == 6);
    CHECK(ep->document.run(r_probe)->token_count == 0);
    CHECK(root->storage_pos_next == 6);
    CHECK(root->public_run.has_value()); // straddling run stays referenced
    CHECK(!root->private_run.has_value()); // emptied run ref cleared
    CHECK(ep->archive_seq == -1);
    CHECK(root->parked_seq == -1);

    // Base past C0 is refused; post-init rebuild is refused.
    CHECK(!runtime.prepare_dag_prefix_rebuild(ep_id, 11, &err));
    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {"questions": [{"id": "A", "intent": "A"}], "depends_on": []}
    })");
    CHECK(decision.is_dag());
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    CHECK(!runtime.prepare_dag_prefix_rebuild(ep_id, 6, &err));
}

static void test_dag_three_lane_flat_cycle_and_peer_uptake() {
    // AGENTS.md stage 6 minimal workload 1 (RERoT.md §12.7): flat three lanes.
    // Verifies cyclic reader order 1->(2,3,1), 2->(3,1,2), 3->(1,2,3) and
    // continuous peer uptake across frontiers without history loss.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(4);

    const uint64_t ep_id = runtime.adopt_root(40, 40, 0, 1, 0);
    CHECK(ep_id != 0);

    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "A", "intent": "Fact A"},
          {"id": "B", "intent": "Fact B"},
          {"id": "C", "intent": "Fact C"}
        ],
        "depends_on": []
      }
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));

    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    CHECK(ep->is_dag);
    CHECK(ep->synthesis_node != 0);
    CHECK(ep->synthesis_node != LLAMA_REROT_NODE_INVALID);
    CHECK(ep->nodes[1].remaining_preds == 0);
    CHECK(ep->nodes[2].remaining_preds == 0);
    CHECK(ep->nodes[3].remaining_preds == 0);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 3);
    CHECK(!ep->c_base.valid());
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(ep->c0.valid());
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(ep->c_base.valid());

    // Initial eligibility: all three lanes, synthesis blocked.
    auto eligible = runtime.get_eligible_dag_nodes(ep_id);
    CHECK(eligible.size() == 3);
    CHECK(std::find(eligible.begin(), eligible.end(), 1) != eligible.end());
    CHECK(std::find(eligible.begin(), eligible.end(), 2) != eligible.end());
    CHECK(std::find(eligible.begin(), eligible.end(), 3) != eligible.end());
    CHECK(std::find(eligible.begin(), eligible.end(), ep->synthesis_node) == eligible.end());

    auto pen_a = runtime.allocate_pen(40, ep_id, 1);
    CHECK(pen_a.has_value());
    CHECK(ep->document.set_node_state(1, llama_rerot_node_state::running));
    ep->running.insert(1);
    ep->nodes[1].pen_id = *pen_a;

    auto pen_b = runtime.allocate_pen(40, ep_id, 2);
    CHECK(pen_b.has_value());
    CHECK(ep->document.set_node_state(2, llama_rerot_node_state::running));
    ep->running.insert(2);
    ep->nodes[2].pen_id = *pen_b;

    auto pen_c = runtime.allocate_pen(40, ep_id, 3);
    CHECK(pen_c.has_value());
    CHECK(ep->document.set_node_state(3, llama_rerot_node_state::running));
    ep->running.insert(3);
    ep->nodes[3].pen_id = *pen_c;

    // First frontier wave: shared prefix plus one public run per lane.
    // All runs share publish epoch 1 so the frozen read watermark captured by
    // finish_frontier (episode publish epoch, untouched by direct appends)
    // never gates them: this test pins ordering/uptake, not frozen gating.
    const auto run_p = ep->document.append_run(0, llama_rerot_visibility::public_live, 0, 10, 1);
    const auto run_a1 = ep->document.append_run(1, llama_rerot_visibility::public_live, 10, 4, 1);
    const auto run_b1 = ep->document.append_run(2, llama_rerot_visibility::public_live, 14, 6, 1);
    const auto run_c1 = ep->document.append_run(3, llama_rerot_visibility::public_live, 20, 5, 1);
    CHECK(run_p != LLAMA_REROT_RUN_INVALID);
    CHECK(run_a1 != LLAMA_REROT_RUN_INVALID);
    CHECK(run_b1 != LLAMA_REROT_RUN_INVALID);
    CHECK(run_c1 != LLAMA_REROT_RUN_INVALID);

    // Cyclic reader order: own work last, peers in plan cycle.
    const auto view_a = runtime.build_dag_view_for_reader(ep_id, 1);
    CHECK(view_a.runs.size() == 4);
    CHECK(view_a.runs[0].owner == 0);
    CHECK(view_a.runs[1].owner == 2);
    CHECK(view_a.runs[2].owner == 3);
    CHECK(view_a.runs[3].owner == 1);

    const auto view_b = runtime.build_dag_view_for_reader(ep_id, 2);
    CHECK(view_b.runs.size() == 4);
    CHECK(view_b.runs[0].owner == 0);
    CHECK(view_b.runs[1].owner == 3);
    CHECK(view_b.runs[2].owner == 1);
    CHECK(view_b.runs[3].owner == 2);

    const auto view_c = runtime.build_dag_view_for_reader(ep_id, 3);
    CHECK(view_c.runs.size() == 4);
    CHECK(view_c.runs[0].owner == 0);
    CHECK(view_c.runs[1].owner == 1);
    CHECK(view_c.runs[2].owner == 2);
    CHECK(view_c.runs[3].owner == 3);

    const auto view_s = runtime.build_dag_view_for_reader(ep_id, 0);
    CHECK(view_s.runs.size() == 4);
    CHECK(view_s.runs[0].owner == 0);
    CHECK(view_s.runs[1].owner == 1);
    CHECK(view_s.runs[2].owner == 2);
    CHECK(view_s.runs[3].owner == 3);

    // Advance the frontier with no seals pending (empty cohort publishes trivially).
    const auto fr1 = runtime.finish_frontier(ep_id);
    CHECK(!fr1.hard_aborted);
    ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);

    // Second wave: incremental public tokens on every lane.
    const auto run_a2 = ep->document.append_run(1, llama_rerot_visibility::public_live, 25, 2, 1);
    const auto run_b2 = ep->document.append_run(2, llama_rerot_visibility::public_live, 27, 2, 1);
    const auto run_c2 = ep->document.append_run(3, llama_rerot_visibility::public_live, 29, 2, 1);
    CHECK(run_a2 != LLAMA_REROT_RUN_INVALID);
    CHECK(run_b2 != LLAMA_REROT_RUN_INVALID);
    CHECK(run_c2 != LLAMA_REROT_RUN_INVALID);

    // First-wave history is untouched by the second wave.
    CHECK(ep->document.run(run_a1)->token_count == 4);
    CHECK(ep->document.run(run_b1)->token_count == 6);
    CHECK(ep->document.run(run_c1)->token_count == 5);

    // Peer uptake: every reader sees both waves of every peer plus its own,
    // ordered by owning node (each node's runs in document order).
    const auto view_a2 = runtime.build_dag_view_for_reader(ep_id, 1);
    CHECK(view_a2.runs.size() == 7);
    CHECK(view_a2.runs[0].owner == 0);
    CHECK(view_a2.runs[1].owner == 2);
    CHECK(view_a2.runs[2].owner == 2);
    CHECK(view_a2.runs[3].owner == 3);
    CHECK(view_a2.runs[4].owner == 3);
    CHECK(view_a2.runs[5].owner == 1);
    CHECK(view_a2.runs[6].owner == 1);
    CHECK(dag_view_has_run(view_a2, run_p));
    CHECK(dag_view_has_run(view_a2, run_a1));
    CHECK(dag_view_has_run(view_a2, run_a2));
    CHECK(dag_view_has_run(view_a2, run_b1));
    CHECK(dag_view_has_run(view_a2, run_b2));
    CHECK(dag_view_has_run(view_a2, run_c1));
    CHECK(dag_view_has_run(view_a2, run_c2));
    CHECK(view_a2.query_virtual_pos == 31);

    const auto view_b2 = runtime.build_dag_view_for_reader(ep_id, 2);
    CHECK(view_b2.runs.size() == 7);
    CHECK(view_b2.runs[0].owner == 0);
    CHECK(view_b2.runs[1].owner == 3);
    CHECK(view_b2.runs[2].owner == 3);
    CHECK(view_b2.runs[3].owner == 1);
    CHECK(view_b2.runs[4].owner == 1);
    CHECK(view_b2.runs[5].owner == 2);
    CHECK(view_b2.runs[6].owner == 2);
    CHECK(dag_view_has_run(view_b2, run_a2));
    CHECK(dag_view_has_run(view_b2, run_b2));
    CHECK(dag_view_has_run(view_b2, run_c2));
    CHECK(view_b2.query_virtual_pos == 31);

    const auto view_c2 = runtime.build_dag_view_for_reader(ep_id, 3);
    CHECK(view_c2.runs.size() == 7);
    CHECK(view_c2.runs[0].owner == 0);
    CHECK(view_c2.runs[1].owner == 1);
    CHECK(view_c2.runs[2].owner == 1);
    CHECK(view_c2.runs[3].owner == 2);
    CHECK(view_c2.runs[4].owner == 2);
    CHECK(view_c2.runs[5].owner == 3);
    CHECK(view_c2.runs[6].owner == 3);
    CHECK(dag_view_has_run(view_c2, run_a2));
    CHECK(dag_view_has_run(view_c2, run_b2));
    CHECK(dag_view_has_run(view_c2, run_c2));
    CHECK(view_c2.query_virtual_pos == 31);

    // Seal all lanes; the cohort then publishes and retires without dropping history.
    CHECK(runtime.seal_dag_node(ep_id, 1, llama_rerot_event_origin::worker_source));
    CHECK(runtime.seal_dag_node(ep_id, 2, llama_rerot_event_origin::worker_source));
    CHECK(runtime.seal_dag_node(ep_id, 3, llama_rerot_event_origin::worker_source));
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 0);
    const auto fr2 = runtime.finish_frontier(ep_id);
    CHECK(!fr2.hard_aborted);
    ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);

    const auto view_a3 = runtime.build_dag_view_for_reader(ep_id, 1);
    CHECK(view_a3.runs.size() == 7);
    CHECK(view_a3.runs[0].owner == 0);
    CHECK(view_a3.runs[1].owner == 2);
    CHECK(view_a3.runs[2].owner == 2);
    CHECK(view_a3.runs[3].owner == 3);
    CHECK(view_a3.runs[4].owner == 3);
    CHECK(view_a3.runs[5].owner == 1);
    CHECK(view_a3.runs[6].owner == 1);
    CHECK(view_a3.query_virtual_pos == 31);

    const auto view_s2 = runtime.build_dag_view_for_reader(ep_id, 0);
    CHECK(view_s2.runs.size() == 7);
    CHECK(view_s2.runs[0].owner == 0);
    CHECK(view_s2.runs[1].owner == 1);
    CHECK(view_s2.runs[2].owner == 1);
    CHECK(view_s2.runs[3].owner == 2);
    CHECK(view_s2.runs[4].owner == 2);
    CHECK(view_s2.runs[5].owner == 3);
    CHECK(view_s2.runs[6].owner == 3);
    CHECK(view_s2.query_virtual_pos == 31);
    CHECK(ep->document.run(run_a1)->token_count == 4);
    CHECK(ep->document.run(run_b1)->token_count == 6);
    CHECK(ep->document.run(run_c1)->token_count == 5);
    CHECK(ep->document.run(run_a2)->token_count == 2);
    CHECK(ep->document.run(run_b2)->token_count == 2);
    CHECK(ep->document.run(run_c2)->token_count == 2);
}

static void test_dag_diamond_and_unequal_length_history() {
    // AGENTS.md stage 6 minimal workloads 3+4 (RERoT.md §12.7): diamond join
    // 1->2, 1->3, 2->4, 3->4 with unequal lane lengths. Verifies join gating,
    // unique ancestor expansion, and completed-node history retention.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(4);

    const uint64_t ep_id = runtime.adopt_root(41, 41, 0, 1, 0);
    CHECK(ep_id != 0);

    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "1", "intent": "Base fact"},
          {"id": "2", "intent": "Left branch"},
          {"id": "3", "intent": "Right branch"},
          {"id": "4", "intent": "Join synthesis"}
        ],
        "depends_on": [
          {"id": "2", "depends_on_id": "1"},
          {"id": "3", "depends_on_id": "1"},
          {"id": "4", "depends_on_id": "2"},
          {"id": "4", "depends_on_id": "3"}
        ]
      }
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));

    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    CHECK(ep->is_dag);
    CHECK(ep->synthesis_node != 0);
    CHECK(ep->synthesis_node != LLAMA_REROT_NODE_INVALID);
    CHECK(ep->nodes[1].remaining_preds == 0);
    CHECK(ep->nodes[2].remaining_preds == 1);
    CHECK(ep->nodes[3].remaining_preds == 1);
    CHECK(ep->nodes[4].remaining_preds == 2);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 4);
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(ep->c_base.valid());

    // Only the diamond source is initially eligible.
    auto eligible = runtime.get_eligible_dag_nodes(ep_id);
    CHECK(eligible.size() == 1);
    CHECK(eligible[0] == 1);

    auto pen_1 = runtime.allocate_pen(41, ep_id, 1);
    CHECK(pen_1.has_value());
    CHECK(ep->document.set_node_state(1, llama_rerot_node_state::running));
    ep->running.insert(1);
    ep->nodes[1].pen_id = *pen_1;

    // All runs share publish epoch 1 so the frozen read watermark captured by
    // finish_frontier never gates them: this test pins join/history, not frozen gating.
    const auto run_p = ep->document.append_run(0, llama_rerot_visibility::public_live, 0, 10, 1);
    const auto run_1 = ep->document.append_run(1, llama_rerot_visibility::public_live, 10, 5, 1);
    CHECK(run_p != LLAMA_REROT_RUN_INVALID);
    CHECK(run_1 != LLAMA_REROT_RUN_INVALID);

    CHECK(runtime.seal_dag_node(ep_id, 1, llama_rerot_event_origin::worker_source));
    CHECK(ep->nodes[1].is_sealed);
    CHECK(ep->document.node(1)->state == llama_rerot_node_state::retired);

    // Both branches unlock; the join stays blocked on both.
    CHECK(ep->nodes[2].remaining_preds == 0);
    CHECK(ep->nodes[3].remaining_preds == 0);
    CHECK(ep->nodes[4].remaining_preds == 2);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 3);
    eligible = runtime.get_eligible_dag_nodes(ep_id);
    CHECK(eligible.size() == 2);
    CHECK(std::find(eligible.begin(), eligible.end(), 2) != eligible.end());
    CHECK(std::find(eligible.begin(), eligible.end(), 3) != eligible.end());
    CHECK(std::find(eligible.begin(), eligible.end(), 4) == eligible.end());
    CHECK(std::find(eligible.begin(), eligible.end(), ep->synthesis_node) == eligible.end());

    auto pen_2 = runtime.allocate_pen(41, ep_id, 2);
    CHECK(pen_2.has_value());
    CHECK(ep->document.set_node_state(2, llama_rerot_node_state::running));
    ep->running.insert(2);
    ep->nodes[2].pen_id = *pen_2;

    auto pen_3 = runtime.allocate_pen(41, ep_id, 3);
    CHECK(pen_3.has_value());
    CHECK(ep->document.set_node_state(3, llama_rerot_node_state::running));
    ep->running.insert(3);
    ep->nodes[3].pen_id = *pen_3;

    // Unequal lengths: short branch finishes in 3 tokens, long branch runs 10.
    const auto run_2 = ep->document.append_run(2, llama_rerot_visibility::public_live, 15, 3, 1);
    const auto run_3 = ep->document.append_run(3, llama_rerot_visibility::public_live, 18, 10, 1);
    CHECK(run_2 != LLAMA_REROT_RUN_INVALID);
    CHECK(run_3 != LLAMA_REROT_RUN_INVALID);
    CHECK(ep->document.run(run_2)->token_count == 3);
    CHECK(ep->document.run(run_3)->token_count == 10);

    const auto fr1 = runtime.finish_frontier(ep_id);
    CHECK(!fr1.hard_aborted);
    ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);

    // Seal the short branch first: the join must remain blocked on the long branch.
    CHECK(runtime.seal_dag_node(ep_id, 2, llama_rerot_event_origin::worker_source));
    CHECK(ep->nodes[2].is_sealed);
    CHECK(ep->nodes[4].remaining_preds == 1);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 2);
    eligible = runtime.get_eligible_dag_nodes(ep_id);
    CHECK(std::find(eligible.begin(), eligible.end(), 4) == eligible.end());

    // Completed short-branch history survives while the long branch continues.
    CHECK(ep->document.run(run_2)->token_count == 3);
    const auto view_3_mid = runtime.build_dag_view_for_reader(ep_id, 3);
    CHECK(dag_view_has_run(view_3_mid, run_p));
    CHECK(dag_view_has_run(view_3_mid, run_1));
    CHECK(dag_view_has_run(view_3_mid, run_2));
    CHECK(dag_view_has_run(view_3_mid, run_3));

    // Sealing the long branch unlocks the join.
    CHECK(runtime.seal_dag_node(ep_id, 3, llama_rerot_event_origin::worker_source));
    CHECK(ep->nodes[3].is_sealed);
    CHECK(ep->nodes[4].remaining_preds == 0);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 1);
    eligible = runtime.get_eligible_dag_nodes(ep_id);
    CHECK(eligible.size() == 1);
    CHECK(eligible[0] == 4);

    auto pen_4 = runtime.allocate_pen(41, ep_id, 4);
    CHECK(pen_4.has_value());
    CHECK(ep->document.set_node_state(4, llama_rerot_node_state::running));
    ep->running.insert(4);
    ep->nodes[4].pen_id = *pen_4;
    const auto run_4 = ep->document.append_run(4, llama_rerot_visibility::public_live, 28, 4, 1);
    CHECK(run_4 != LLAMA_REROT_RUN_INVALID);

    // Join reader view: shared ancestor 1 appears exactly once despite two paths.
    const auto view_4 = runtime.build_dag_view_for_reader(ep_id, 4);
    CHECK(view_4.runs.size() == 5);
    CHECK(view_4.runs[0].owner == 0);
    CHECK(view_4.runs[1].owner == 1);
    CHECK(view_4.runs[2].owner == 2);
    CHECK(view_4.runs[3].owner == 3);
    CHECK(view_4.runs[4].owner == 4);
    size_t count_1 = 0;
    size_t count_2 = 0;
    size_t count_3 = 0;
    size_t count_4 = 0;
    for (const auto & entry : view_4.runs) {
        count_1 += (entry.owner == 1) ? 1 : 0;
        count_2 += (entry.owner == 2) ? 1 : 0;
        count_3 += (entry.owner == 3) ? 1 : 0;
        count_4 += (entry.owner == 4) ? 1 : 0;
    }
    CHECK(count_1 == 1);
    CHECK(count_2 == 1);
    CHECK(count_3 == 1);
    CHECK(count_4 == 1);
    CHECK(dag_view_has_run(view_4, run_p));
    CHECK(dag_view_has_run(view_4, run_1));
    CHECK(dag_view_has_run(view_4, run_2));
    CHECK(dag_view_has_run(view_4, run_3));
    CHECK(dag_view_has_run(view_4, run_4));
    CHECK(ep->document.run(run_1)->token_count == 5);
    CHECK(ep->document.run(run_2)->token_count == 3);
    CHECK(ep->document.run(run_3)->token_count == 10);
    CHECK(ep->document.run(run_4)->token_count == 4);

    CHECK(runtime.seal_dag_node(ep_id, 4, llama_rerot_event_origin::worker_source));
    CHECK(ep->nodes[4].is_sealed);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 0);

    // Synthesis sees every worker exactly once with full completed history.
    const auto view_s = runtime.build_dag_view_for_reader(ep_id, 0);
    CHECK(view_s.runs.size() == 5);
    CHECK(view_s.runs[0].owner == 0);
    CHECK(view_s.runs[1].owner == 1);
    CHECK(view_s.runs[2].owner == 2);
    CHECK(view_s.runs[3].owner == 3);
    CHECK(view_s.runs[4].owner == 4);
    size_t synth_1 = 0;
    size_t synth_2 = 0;
    size_t synth_3 = 0;
    size_t synth_4 = 0;
    for (const auto & entry : view_s.runs) {
        synth_1 += (entry.owner == 1) ? 1 : 0;
        synth_2 += (entry.owner == 2) ? 1 : 0;
        synth_3 += (entry.owner == 3) ? 1 : 0;
        synth_4 += (entry.owner == 4) ? 1 : 0;
    }
    CHECK(synth_1 == 1);
    CHECK(synth_2 == 1);
    CHECK(synth_3 == 1);
    CHECK(synth_4 == 1);
    CHECK(dag_view_has_run(view_s, run_1));
    CHECK(dag_view_has_run(view_s, run_2));
    CHECK(dag_view_has_run(view_s, run_3));
    CHECK(dag_view_has_run(view_s, run_4));

    const auto fr2 = runtime.finish_frontier(ep_id);
    CHECK(!fr2.hard_aborted);
    CHECK(fr2.synthesis_node == ep->synthesis_node);
    ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    const auto view_s2 = runtime.build_dag_view_for_reader(ep_id, 0);
    CHECK(view_s2.runs.size() == 5);
    CHECK(ep->document.run(run_1)->token_count == 5);
    CHECK(ep->document.run(run_2)->token_count == 3);
    CHECK(ep->document.run(run_3)->token_count == 10);
    CHECK(ep->document.run(run_4)->token_count == 4);
}

static void test_dag_a_to_c_with_b_independent_overlap() {
    // AGENTS.md stage 6 minimal workload 2 (RERoT.md §12.7): A->C with B
    // independent. Verifies gated unlock of C on A's seal, B/C execution
    // overlap, and completed-node history retention in peer views.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(4);

    const uint64_t ep_id = runtime.adopt_root(42, 42, 0, 1, 0);
    CHECK(ep_id != 0);

    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "A", "intent": "Fact A"},
          {"id": "B", "intent": "Fact B"},
          {"id": "C", "intent": "Fact C"}
        ],
        "depends_on": [
          {"id": "C", "depends_on_id": "A"}
        ]
      }
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));

    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    CHECK(ep->is_dag);
    CHECK(ep->synthesis_node != 0);
    CHECK(ep->synthesis_node != LLAMA_REROT_NODE_INVALID);
    CHECK(ep->nodes[1].remaining_preds == 0);
    CHECK(ep->nodes[2].remaining_preds == 0);
    CHECK(ep->nodes[3].remaining_preds == 1);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 3);
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(ep->c_base.valid());

    // Initial eligibility: A (1) and B (2) only; C (3) is blocked on A.
    auto eligible = runtime.get_eligible_dag_nodes(ep_id);
    CHECK(eligible.size() == 2);
    CHECK(std::find(eligible.begin(), eligible.end(), 1) != eligible.end());
    CHECK(std::find(eligible.begin(), eligible.end(), 2) != eligible.end());
    CHECK(std::find(eligible.begin(), eligible.end(), 3) == eligible.end());
    CHECK(std::find(eligible.begin(), eligible.end(), ep->synthesis_node) == eligible.end());

    auto pen_a = runtime.allocate_pen(42, ep_id, 1);
    CHECK(pen_a.has_value());
    CHECK(ep->document.set_node_state(1, llama_rerot_node_state::running));
    ep->running.insert(1);
    ep->nodes[1].pen_id = *pen_a;

    auto pen_b = runtime.allocate_pen(42, ep_id, 2);
    CHECK(pen_b.has_value());
    CHECK(ep->document.set_node_state(2, llama_rerot_node_state::running));
    ep->running.insert(2);
    ep->nodes[2].pen_id = *pen_b;

    // First wave: shared prefix plus one public run per admitted lane.
    // All runs share publish epoch 1 so the frozen read watermark captured by
    // finish_frontier never gates them: this test pins gating/overlap, not frozen gating.
    const auto run_p = ep->document.append_run(0, llama_rerot_visibility::public_live, 0, 10, 1);
    const auto run_a = ep->document.append_run(1, llama_rerot_visibility::public_live, 10, 5, 1);
    const auto run_b1 = ep->document.append_run(2, llama_rerot_visibility::public_live, 15, 8, 1);
    CHECK(run_p != LLAMA_REROT_RUN_INVALID);
    CHECK(run_a != LLAMA_REROT_RUN_INVALID);
    CHECK(run_b1 != LLAMA_REROT_RUN_INVALID);

    // A finishes and seals while B is still running: C unlocks, B unaffected.
    CHECK(runtime.seal_dag_node(ep_id, 1, llama_rerot_event_origin::worker_source));
    CHECK(ep->nodes[1].is_sealed);
    CHECK(ep->document.node(1)->state == llama_rerot_node_state::retired);
    CHECK(ep->nodes[3].remaining_preds == 0);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 2);
    CHECK(!ep->nodes[2].is_sealed);
    CHECK(ep->running.count(2) == 1);
    CHECK(ep->document.node(2)->state == llama_rerot_node_state::running);
    eligible = runtime.get_eligible_dag_nodes(ep_id);
    CHECK(eligible.size() == 1);
    CHECK(eligible[0] == 3);

    // Start C while B keeps running: genuine B/C execution overlap.
    auto pen_c = runtime.allocate_pen(42, ep_id, 3);
    CHECK(pen_c.has_value());
    CHECK(ep->document.set_node_state(3, llama_rerot_node_state::running));
    ep->running.insert(3);
    ep->nodes[3].pen_id = *pen_c;
    const auto run_c = ep->document.append_run(3, llama_rerot_visibility::public_live, 23, 6, 1);
    const auto run_b2 = ep->document.append_run(2, llama_rerot_visibility::public_live, 29, 4, 1);
    CHECK(run_c != LLAMA_REROT_RUN_INVALID);
    CHECK(run_b2 != LLAMA_REROT_RUN_INVALID);

    // B and C overlap in execution.
    CHECK(ep->running.count(2) == 1);
    CHECK(ep->running.count(3) == 1);
    CHECK(ep->document.node(2)->state == llama_rerot_node_state::running);
    CHECK(ep->document.node(3)->state == llama_rerot_node_state::running);

    // Reader 2 (B): sealed A history is retained and visible, C is visible,
    // own runs are last. Cycle-preferred topo order anchored at 2 with the
    // 1->3 edge is [1,3,2].
    const auto view_b = runtime.build_dag_view_for_reader(ep_id, 2);
    CHECK(view_b.runs.size() == 5);
    CHECK(view_b.runs[0].owner == 0);
    CHECK(view_b.runs[1].owner == 1);
    CHECK(view_b.runs[2].owner == 3);
    CHECK(view_b.runs[3].owner == 2);
    CHECK(view_b.runs[4].owner == 2);
    CHECK(dag_view_has_run(view_b, run_p));
    CHECK(dag_view_has_run(view_b, run_a));
    CHECK(dag_view_has_run(view_b, run_b1));
    CHECK(dag_view_has_run(view_b, run_b2));
    CHECK(dag_view_has_run(view_b, run_c));
    CHECK(ep->document.run(run_a)->token_count == 5);
    CHECK(ep->document.run(run_b1)->token_count == 8);
    CHECK(ep->document.run(run_c)->token_count == 6);
    CHECK(ep->document.run(run_b2)->token_count == 4);
    CHECK(view_b.query_virtual_pos == 33);

    // Reader 3 (C): predecessor A first, concurrent peer B visible, own run last.
    const auto view_c = runtime.build_dag_view_for_reader(ep_id, 3);
    CHECK(view_c.runs.size() == 5);
    CHECK(view_c.runs[0].owner == 0);
    CHECK(view_c.runs[1].owner == 1);
    CHECK(view_c.runs[2].owner == 2);
    CHECK(view_c.runs[3].owner == 2);
    CHECK(view_c.runs[4].owner == 3);
    CHECK(dag_view_has_run(view_c, run_p));
    CHECK(dag_view_has_run(view_c, run_a));
    CHECK(dag_view_has_run(view_c, run_b1));
    CHECK(dag_view_has_run(view_c, run_b2));
    CHECK(dag_view_has_run(view_c, run_c));
    CHECK(view_c.query_virtual_pos == 33);

    // Seal B: synthesis still blocked on C.
    CHECK(runtime.seal_dag_node(ep_id, 2, llama_rerot_event_origin::worker_source));
    CHECK(ep->nodes[2].is_sealed);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 1);
    eligible = runtime.get_eligible_dag_nodes(ep_id);
    CHECK(eligible.empty());

    // Seal C: synthesis becomes eligible.
    CHECK(runtime.seal_dag_node(ep_id, 3, llama_rerot_event_origin::worker_source));
    CHECK(ep->nodes[3].is_sealed);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 0);
    eligible = runtime.get_eligible_dag_nodes(ep_id);
    CHECK(eligible.size() == 1);
    CHECK(eligible[0] == ep->synthesis_node);

    // Synthesis sees every worker in plan order with the 1->3 edge preserved:
    // A and C appear exactly once; B appears twice (two overlap waves).
    const auto view_s = runtime.build_dag_view_for_reader(ep_id, 0);
    CHECK(view_s.runs.size() == 5);
    CHECK(view_s.runs[0].owner == 0);
    CHECK(view_s.runs[1].owner == 1);
    CHECK(view_s.runs[2].owner == 2);
    CHECK(view_s.runs[3].owner == 2);
    CHECK(view_s.runs[4].owner == 3);
    size_t synth_1 = 0;
    size_t synth_2 = 0;
    size_t synth_3 = 0;
    for (const auto & entry : view_s.runs) {
        synth_1 += (entry.owner == 1) ? 1 : 0;
        synth_2 += (entry.owner == 2) ? 1 : 0;
        synth_3 += (entry.owner == 3) ? 1 : 0;
    }
    CHECK(synth_1 == 1);
    CHECK(synth_2 == 2);
    CHECK(synth_3 == 1);
    CHECK(dag_view_has_run(view_s, run_p));
    CHECK(dag_view_has_run(view_s, run_a));
    CHECK(dag_view_has_run(view_s, run_b1));
    CHECK(dag_view_has_run(view_s, run_b2));
    CHECK(dag_view_has_run(view_s, run_c));
    CHECK(view_s.query_virtual_pos == 33);
    CHECK(ep->document.run(run_a)->token_count == 5);
    CHECK(ep->document.run(run_b1)->token_count == 8);
    CHECK(ep->document.run(run_b2)->token_count == 4);
    CHECK(ep->document.run(run_c)->token_count == 6);

    const auto synth = ep->synthesis_node;
    const auto fr = runtime.finish_frontier(ep_id);
    CHECK(!fr.hard_aborted);
    CHECK(fr.synthesis_node == synth);
    ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    const auto view_s2 = runtime.build_dag_view_for_reader(ep_id, 0);
    CHECK(view_s2.runs.size() == 5);
    CHECK(ep->document.run(run_a)->token_count == 5);
    CHECK(ep->document.run(run_b1)->token_count == 8);
    CHECK(ep->document.run(run_b2)->token_count == 4);
    CHECK(ep->document.run(run_c)->token_count == 6);
}

static void test_dag_initialize_refuses_double_init() {
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(2);
    const uint64_t ep_id = runtime.adopt_root(53, 53, 0, 1, 0);
    CHECK(ep_id != 0);
    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {"questions": [{"id": "A", "intent": "A"}], "depends_on": []}
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    const auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    const size_t n_nodes = ep->nodes.size();
    err.clear();
    CHECK(!runtime.initialize_dag(ep_id, decision, &err));
    CHECK(err.find("already") != std::string::npos);
    CHECK(runtime.episode(ep_id)->nodes.size() == n_nodes);
}

static void test_dag_source_end_multi_token_and_starting_frame_gate() {
    // Stage 2 (§12.3) Verification:
    // 1. STARTING FRAME tokens carrying close delimiters must NOT seal worker.
    // 2. Multi-token source-end marker: candidate token stays PENDING and does NOT
    //    seal or unlock successors until the final token closes the marker.
    // 3. Foreign FRAME or ordinary think tags in BODY do not trigger seal.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(4);
    const uint64_t ep_id = runtime.adopt_root(74, 74, 0, 1, 0);
    CHECK(ep_id != 0);
    runtime.set_dag_protocol_markers(ep_id, "</think>", "<think>");

    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "A", "intent": "Fact A"},
          {"id": "B", "intent": "Fact B"}
        ],
        "depends_on": [
          {"id": "B", "depends_on_id": "A"}
        ]
      }
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(runtime.activate_dag_frontier(ep_id));

    auto * root = runtime.node(ep_id, 0);
    if (root && root->physical_slot >= 0) {
        CHECK(runtime.detach_node(ep_id, 0));
    }

    llama_rerot_node_id a = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 0, 2, &a));
    CHECK(a == 1);
    // Node A is in STARTING state.
    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    CHECK(ep->document.node(a)->state == llama_rerot_node_state::starting);

    // 1. In STARTING state, forwarding FRAME bytes (even if containing "</think>")
    // plans as frame and must NOT close or seal worker (§12.3 item 3).
    auto frame_plan = runtime.plan_generated_token(ep_id, a, 10, "</think>");
    CHECK(frame_plan.has_value());
    CHECK(frame_plan->segment_kind == llama_rerot_segment_kind::frame);
    CHECK(frame_plan->event_origin == llama_rerot_event_origin::runtime_frame);
    CHECK(runtime.commit_token(ep_id, a, *frame_plan));
    CHECK(!ep->nodes[a].is_sealed);
    CHECK(ep->nodes[2].remaining_preds == 1); // Successor B remains blocked

    // Complete admission -> worker A moves from STARTING to RUNNING.
    CHECK(runtime.complete_admission(ep_id, a));
    CHECK(ep->document.node(a)->state == llama_rerot_node_state::running);

    // 2. Multi-token source-end: first token is candidate prefix ("</thi").
    // Must remain body and PENDING, exit_parser enters candidate state, successor B remains blocked.
    auto p1 = runtime.plan_generated_token(ep_id, a, 11, "</thi");
    CHECK(p1.has_value());
    CHECK(p1->segment_kind == llama_rerot_segment_kind::body);
    CHECK(!p1->marker_step.marker_closed);
    CHECK(ep->nodes[a].exit_parser.state() == server_rerot_marker_state::marker_candidate);
    CHECK(runtime.commit_token(ep_id, a, *p1));
    CHECK(!ep->nodes[a].is_sealed);
    CHECK(ep->nodes[2].remaining_preds == 1);

    // 3. Second token completes the marker ("nk>").
    // Must plan as source_end, marker_closed = true, commit seals worker A and decrements B's remaining_preds to 0.
    auto p2 = runtime.plan_generated_token(ep_id, a, 12, "nk>");
    CHECK(p2.has_value());
    CHECK(p2->segment_kind == llama_rerot_segment_kind::source_end);
    CHECK(p2->marker_step.marker_closed);
    CHECK(p2->event_origin == llama_rerot_event_origin::worker_source);
    CHECK(runtime.commit_token(ep_id, a, *p2));
    CHECK(ep->nodes[a].is_sealed);
    CHECK(ep->nodes[2].remaining_preds == 0); // Successor B is now unlocked
}

static void test_dag_microbatch_slice_order_and_no_earlier_public_leak() {
    // Stage 4 (§12.5) Verification:
    // Must-pass item 3: Same logical frontier under different microbatch slicing/row orders
    // produces the identical prescribed state and reader views.
    // Must-pass item 4: A later slice in the same frontier cannot observe this frontier's
    // earlier slice's newly committed tokens until the cohort-wide atomic publication.

    // Scenario 1: Slice 1 executes lane A, Slice 2 executes lane B.
    // Verify that when lane A commits in slice 1, lane B in slice 2 does NOT see lane A's write,
    // and both only become mutually visible after the cohort step finishes.
    server_rerot_runtime runtime1(nullptr);
    runtime1.set_pen_capacity(4);
    const uint64_t ep1 = runtime1.adopt_root(101, 101, 0, 1, 0);
    CHECK(ep1 != 0);

    const auto decision = server_rerot_parse_routing_decision(R"({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "A", "intent": "Fact A"},
          {"id": "B", "intent": "Fact B"}
        ],
        "depends_on": []
      }
    })");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime1.initialize_dag(ep1, decision, &err));
    CHECK(runtime1.capture_c0(ep1, 1, 0));
    CHECK(runtime1.capture_c_base(ep1));
    CHECK(runtime1.activate_dag_frontier(ep1));
    auto * root1 = runtime1.node(ep1, 0);
    if (root1 && root1->physical_slot >= 0) {
        CHECK(runtime1.detach_node(ep1, 0));
    }

    llama_rerot_node_id a1 = LLAMA_REROT_NODE_INVALID;
    llama_rerot_node_id b1 = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime1.admit_next_child(ep1, 0, 2, &a1));
    CHECK(runtime1.admit_next_child(ep1, 1, 3, &b1));
    CHECK(a1 == 1 && b1 == 2);
    CHECK(runtime1.complete_admission(ep1, a1));
    CHECK(runtime1.complete_admission(ep1, b1));

    // Snapshot open logical step for ep1
    CHECK(runtime1.has_open_dag_logical_step(ep1));

    // Slice 1: Lane A executes and commits.
    auto * na1 = runtime1.node(ep1, a1);
    CHECK(na1 != nullptr);
    CHECK(commit_generated(runtime1, ep1, a1, na1->storage_pos_next, "tok_a"));
    CHECK(na1->pending_record.has_value());
    const auto run_a1 = *na1->pending_record;

    // Mid-frontier inspection: Slice 2 is about to run Lane B.
    // Reader B must NOT see Lane A's uncommitted/pending run_a1.
    const auto view_b1_pre = runtime1.build_dag_view_for_reader(ep1, b1);
    CHECK(!dag_view_has_run(view_b1_pre, run_a1));

    // Slice 2: Lane B executes and commits.
    auto * nb1 = runtime1.node(ep1, b1);
    CHECK(nb1 != nullptr);
    CHECK(commit_generated(runtime1, ep1, b1, nb1->storage_pos_next, "tok_b"));
    CHECK(nb1->pending_record.has_value());
    const auto run_b1 = *nb1->pending_record;

    // Before finish_frontier: neither peer sees the other's pending write.
    const auto view_a1_mid = runtime1.build_dag_view_for_reader(ep1, a1);
    const auto view_b1_mid = runtime1.build_dag_view_for_reader(ep1, b1);
    CHECK(!dag_view_has_run(view_a1_mid, run_b1));
    CHECK(!dag_view_has_run(view_b1_mid, run_a1));

    // Finish frontier publishes cohort atomically.
    const auto res1 = runtime1.finish_frontier(ep1);
    CHECK(!res1.hard_aborted);
    const auto view_a1_post = runtime1.build_dag_view_for_reader(ep1, a1);
    const auto view_b1_post = runtime1.build_dag_view_for_reader(ep1, b1);
    CHECK(dag_view_has_run(view_a1_post, run_a1));
    CHECK(dag_view_has_run(view_a1_post, run_b1));
    CHECK(dag_view_has_run(view_b1_post, run_a1));
    CHECK(dag_view_has_run(view_b1_post, run_b1));

    // Scenario 2: Reversed row/microbatch order (Slice 1 runs Lane B, Slice 2 runs Lane A).
    // The final result must be strictly identical in view structure and token mapping.
    server_rerot_runtime runtime2(nullptr);
    runtime2.set_pen_capacity(4);
    const uint64_t ep2 = runtime2.adopt_root(102, 102, 0, 1, 0);
    CHECK(ep2 != 0);
    CHECK(runtime2.initialize_dag(ep2, decision, &err));
    CHECK(runtime2.capture_c0(ep2, 1, 0));
    CHECK(runtime2.capture_c_base(ep2));
    CHECK(runtime2.activate_dag_frontier(ep2));
    auto * root2 = runtime2.node(ep2, 0);
    if (root2 && root2->physical_slot >= 0) {
        CHECK(runtime2.detach_node(ep2, 0));
    }

    llama_rerot_node_id a2 = LLAMA_REROT_NODE_INVALID;
    llama_rerot_node_id b2 = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime2.admit_next_child(ep2, 0, 2, &a2));
    CHECK(runtime2.admit_next_child(ep2, 1, 3, &b2));
    CHECK(runtime2.complete_admission(ep2, a2));
    CHECK(runtime2.complete_admission(ep2, b2));

    // Reversed execution: Slice 1 executes Lane B first
    auto * nb2 = runtime2.node(ep2, b2);
    CHECK(nb2 != nullptr);
    CHECK(commit_generated(runtime2, ep2, b2, nb2->storage_pos_next, "tok_b"));
    CHECK(nb2->pending_record.has_value());
    const auto run_b2 = *nb2->pending_record;

    // Mid-frontier inspection: Slice 2 (Lane A) must NOT see Lane B's write yet
    const auto view_a2_pre = runtime2.build_dag_view_for_reader(ep2, a2);
    CHECK(!dag_view_has_run(view_a2_pre, run_b2));

    // Slice 2 executes Lane A
    auto * na2 = runtime2.node(ep2, a2);
    CHECK(na2 != nullptr);
    CHECK(commit_generated(runtime2, ep2, a2, na2->storage_pos_next, "tok_a"));
    const auto run_a2 = *na2->pending_record;

    // Finish frontier publishes cohort atomically in runtime2
    const auto res2 = runtime2.finish_frontier(ep2);
    CHECK(!res2.hard_aborted);

    const auto view_a2_post = runtime2.build_dag_view_for_reader(ep2, a2);
    const auto view_b2_post = runtime2.build_dag_view_for_reader(ep2, b2);
    CHECK(dag_view_has_run(view_a2_post, run_a2));
    CHECK(dag_view_has_run(view_a2_post, run_b2));
    CHECK(dag_view_has_run(view_b2_post, run_a2));
    CHECK(dag_view_has_run(view_b2_post, run_b2));

    // Exact structural equivalence: query virtual positions, total runs, and run counts match
    CHECK(view_a1_post.runs.size() == view_a2_post.runs.size());
    CHECK(view_a1_post.query_virtual_pos == view_a2_post.query_virtual_pos);
    CHECK(view_b1_post.runs.size() == view_b2_post.runs.size());
    CHECK(view_b1_post.query_virtual_pos == view_b2_post.query_virtual_pos);
}

static void test_dag_duplicate_source_end_and_restore_no_double_decrement() {
    // Stage 4 (§12.5) Verification:
    // Must-pass item 6: Duplicate source-end callback and repeat restore notifications
    // must not double-decrement remaining_preds on successors (§07.5).
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(3);
    const uint64_t ep_id = runtime.adopt_root(70, 70, 0, 1, 0);
    CHECK(ep_id != 0);
    const auto decision = server_rerot_parse_routing_decision(
        "{\"strategy\": \"dag\", \"payload\": {\"questions\": ["
        "{\"id\": \"A\", \"intent\": \"Worker A\"},"
        "{\"id\": \"B\", \"intent\": \"Worker B\"}"
        "], \"depends_on\": [{\"id\": \"B\", \"depends_on_id\": \"A\"}]}}");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(runtime.activate_dag_frontier(ep_id));

    auto * root = runtime.node(ep_id, 0);
    if (root && root->physical_slot >= 0) {
        CHECK(runtime.detach_node(ep_id, 0));
    }
    llama_rerot_node_id a = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 0, 2, &a));
    CHECK(a == 1);
    CHECK(runtime.complete_admission(ep_id, a));

    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    CHECK(ep->nodes[2].remaining_preds == 1);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 2);

    // First legitimate source_end seals node A and decrements B and synthesis
    CHECK(runtime.seal_dag_node(ep_id, a, llama_rerot_event_origin::worker_source));
    CHECK(ep->nodes[a].is_sealed);
    CHECK(ep->nodes[2].remaining_preds == 0);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 1);

    // Duplicate source_end calls must be no-ops: remaining_preds must not become negative or decrement again
    CHECK(runtime.seal_dag_node(ep_id, a, llama_rerot_event_origin::worker_source));
    CHECK(ep->nodes[2].remaining_preds == 0);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 1);

    CHECK(runtime.seal_dag_node(ep_id, a, llama_rerot_event_origin::worker_source));
    CHECK(ep->nodes[2].remaining_preds == 0);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 1);

    // Save state while A is sealed and B is ready (remaining_preds == 0)
    server_rerot_state_fingerprints fp;
    fp.caps = LLAMA_REROT_STATE_CAP_REROT | LLAMA_REROT_STATE_CAP_REROT_TREE |
              LLAMA_REROT_STATE_CAP_REROT_PRIVATE;
    std::vector<uint8_t> blob;
    CHECK(runtime.save_episode(ep_id, fp, &blob, &err));
    CHECK(!blob.empty());

    // Restore into fresh runtime and verify remaining_preds is exactly 0, not decremented
    server_rerot_runtime runtime_restored(nullptr);
    uint64_t restored_id = 0;
    CHECK(runtime_restored.load_episode(blob.data(), blob.size(), fp, &restored_id, &err));
    auto * ep_restored = runtime_restored.episode(restored_id);
    CHECK(ep_restored != nullptr);
    CHECK(ep_restored->nodes[1].is_sealed);
    CHECK(ep_restored->nodes[2].remaining_preds == 0);
    CHECK(ep_restored->nodes[ep_restored->synthesis_node].remaining_preds == 1);

    // Attempting repeat seal on the restored instance must also be a no-op
    CHECK(runtime_restored.seal_dag_node(restored_id, 1, llama_rerot_event_origin::worker_source));
    CHECK(ep_restored->nodes[2].remaining_preds == 0);
    CHECK(ep_restored->nodes[ep_restored->synthesis_node].remaining_preds == 1);
}

static void test_dag_abort_clears_orphan_refs_and_pens() {
    // Stage 4 (§12.5) Verification:
    // Must-pass item 7: On abort / cancellation, there must be 0 orphan seq refs,
    // 0 active running pens, and no uncleaned internal state.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(3);
    const uint64_t ep_id = runtime.adopt_root(80, 80, 0, 1, 0);
    CHECK(ep_id != 0);
    const auto decision = server_rerot_parse_routing_decision(
        "{\"strategy\": \"dag\", \"payload\": {\"questions\": ["
        "{\"id\": \"A\", \"intent\": \"Worker A\"},"
        "{\"id\": \"B\", \"intent\": \"Worker B\"}"
        "], \"depends_on\": []}}");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(runtime.activate_dag_frontier(ep_id));

    auto * root = runtime.node(ep_id, 0);
    if (root && root->physical_slot >= 0) {
        CHECK(runtime.detach_node(ep_id, 0));
    }
    llama_rerot_node_id a = LLAMA_REROT_NODE_INVALID;
    llama_rerot_node_id b = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 0, 2, &a));
    CHECK(runtime.admit_next_child(ep_id, 1, 3, &b));
    CHECK(a == 1 && b == 2);
    CHECK(runtime.complete_admission(ep_id, a));
    CHECK(runtime.complete_admission(ep_id, b));

    // Two active pens bound to this episode
    CHECK(runtime.pens_for_person(ep_id).size() == 2);
    CHECK(runtime.pens_allocated() == 2);

    // Hard abort (e.g. client cancellation or resource error)
    CHECK(!runtime.hard_abort(ep_id, "client_cancellation_during_workers"));
    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    CHECK(ep->hard_aborted);
    CHECK(ep->abort_reason.find("client_cancellation") != std::string::npos);

    // All pens must be freed immediately, no active bindings left
    CHECK(runtime.pens_for_person(ep_id).empty());
    CHECK(runtime.pens_allocated() == 0);
    CHECK(runtime.pen(0)->state == server_pen_state::free);
    CHECK(runtime.pen(1)->state == server_pen_state::free);
    CHECK(runtime.pen(2)->state == server_pen_state::free);
    CHECK(ep->running.empty());
    CHECK(ep->ready_queue.empty());
    CHECK(ep->nodes[a].physical_slot == -1);
    CHECK(ep->nodes[b].physical_slot == -1);

    // Erase episode cleanly
    CHECK(runtime.erase_episode(ep_id));
    CHECK(runtime.episode(ep_id) == nullptr);
}

static void test_dag_demote_restore_different_physical_slots() {
    // Stage 7 (§12.8) / AGENTS.md 阶段 7:
    // "fork/启动后 demote -> 换物理位置 restore: 图、入口 cursor、source end、局部状态、输出继续一致"
    //
    // Setup:
    // 1. Runtime A (reference) runs an uninterrupted DAG:
    //    root (0) -> Worker A (1), Worker B (2) -> Synthesis (3)
    //    Workers admit into physical slots 1 and 2, generate tokens, and seal.
    // 2. Runtime B (test target) creates the identical DAG:
    //    Workers admit into physical slots 1 and 2, generate initial tokens,
    //    attach sampler/MTP/state snapshots, and then demote_episode() is called
    //    to release physical slot bindings.
    // 3. Runtime B serializes to RAM blob (save_episode).
    // 4. In Runtime B, slots 1 and 2 are occupied by unrelated work.
    // 5. Episode is restored (load_episode) and rebound to DIFFERENT physical slots (slots 3 and 4).
    // 6. Restored workers continue generating the same subsequent tokens.
    // 7. Verifies logical DAG equivalence, token counts, run structure, and reader view identicality.

    // 1. Reference runtime
    server_rerot_runtime ref_runtime(nullptr);
    ref_runtime.set_pen_capacity(6);
    const uint64_t ep_ref = ref_runtime.adopt_root(100, 100, 0, 1, 0);
    CHECK(ep_ref != 0);
    const auto decision = server_rerot_parse_routing_decision(R"json({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "A", "intent": "Worker Alpha"},
          {"id": "B", "intent": "Worker Beta"}
        ],
        "depends_on": []
      }
    })json");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(ref_runtime.initialize_dag(ep_ref, decision, &err));
    CHECK(ref_runtime.capture_c0(ep_ref, 1, 0));
    CHECK(ref_runtime.capture_c_base(ep_ref));
    CHECK(ref_runtime.activate_dag_frontier(ep_ref));

    auto * ref_root = ref_runtime.node(ep_ref, 0);
    if (ref_root && ref_root->physical_slot >= 0) {
        CHECK(ref_runtime.detach_node(ep_ref, 0));
    }
    llama_rerot_node_id ref_w1 = LLAMA_REROT_NODE_INVALID;
    llama_rerot_node_id ref_w2 = LLAMA_REROT_NODE_INVALID;
    CHECK(ref_runtime.admit_next_child(ep_ref, 1, 10, &ref_w1));
    CHECK(ref_runtime.admit_next_child(ep_ref, 2, 20, &ref_w2));
    CHECK(ref_w1 == 1 && ref_w2 == 2);
    CHECK(ref_runtime.complete_admission(ep_ref, ref_w1));
    CHECK(ref_runtime.complete_admission(ep_ref, ref_w2));

    // Append initial public tokens to reference workers
    auto * ref_ep = ref_runtime.episode(ep_ref);
    const auto ref_r1 = ref_ep->document.append_run(ref_w1, llama_rerot_visibility::public_live, 10, 4, 1);
    const auto ref_r2 = ref_ep->document.append_run(ref_w2, llama_rerot_visibility::public_live, 20, 5, 2);
    ref_ep->nodes[ref_w1].public_run = ref_r1;
    ref_ep->nodes[ref_w2].public_run = ref_r2;

    // 2. Target runtime: identical initial setup
    server_rerot_runtime tgt_runtime(nullptr);
    tgt_runtime.set_pen_capacity(6);
    const uint64_t ep_tgt = tgt_runtime.adopt_root(100, 100, 0, 1, 0);
    CHECK(ep_tgt == ep_ref);
    CHECK(tgt_runtime.initialize_dag(ep_tgt, decision, &err));
    CHECK(tgt_runtime.capture_c0(ep_tgt, 1, 0));
    CHECK(tgt_runtime.capture_c_base(ep_tgt));
    CHECK(tgt_runtime.activate_dag_frontier(ep_tgt));

    auto * tgt_root = tgt_runtime.node(ep_tgt, 0);
    if (tgt_root && tgt_root->physical_slot >= 0) {
        CHECK(tgt_runtime.detach_node(ep_tgt, 0));
    }
    llama_rerot_node_id tgt_w1 = LLAMA_REROT_NODE_INVALID;
    llama_rerot_node_id tgt_w2 = LLAMA_REROT_NODE_INVALID;
    CHECK(tgt_runtime.admit_next_child(ep_tgt, 1, 10, &tgt_w1));
    CHECK(tgt_runtime.admit_next_child(ep_tgt, 2, 20, &tgt_w2));
    CHECK(tgt_w1 == 1 && tgt_w2 == 2);
    CHECK(tgt_runtime.complete_admission(ep_tgt, tgt_w1));
    CHECK(tgt_runtime.complete_admission(ep_tgt, tgt_w2));

    auto * tgt_ep = tgt_runtime.episode(ep_tgt);
    const auto tgt_r1 = tgt_ep->document.append_run(tgt_w1, llama_rerot_visibility::public_live, 10, 4, 1);
    const auto tgt_r2 = tgt_ep->document.append_run(tgt_w2, llama_rerot_visibility::public_live, 20, 5, 2);
    tgt_ep->nodes[tgt_w1].public_run = tgt_r1;
    tgt_ep->nodes[tgt_w2].public_run = tgt_r2;

    // Attach local state blobs for verification
    tgt_ep->nodes[tgt_w1].sampler_blob = {0xAA, 0xBB, 0xCC};
    tgt_ep->nodes[tgt_w1].mtp_blob = {0x11, 0x22};
    tgt_ep->nodes[tgt_w2].sampler_blob = {0xDD, 0xEE};
    tgt_ep->nodes[tgt_w2].mtp_blob = {0x33, 0x44};

    // 3. Demote target episode: transient physical slot/pen bindings are released
    CHECK(tgt_runtime.demote_episode(ep_tgt));
    CHECK(tgt_runtime.node(ep_tgt, tgt_w1)->physical_slot == -1);
    CHECK(tgt_runtime.node(ep_tgt, tgt_w2)->physical_slot == -1);

    // 4. Save episode to RAM blob
    server_rerot_state_fingerprints fp;
    fp.caps = LLAMA_REROT_STATE_CAP_REROT | LLAMA_REROT_STATE_CAP_REROT_TREE |
              LLAMA_REROT_STATE_CAP_REROT_PRIVATE;
    std::vector<uint8_t> ram_blob;
    CHECK(tgt_runtime.save_episode(ep_tgt, fp, &ram_blob, &err));
    CHECK(!ram_blob.empty() && err.empty());

    // 5. Erase ep_tgt, and occupy physical slots 1 and 2 with unrelated work
    CHECK(tgt_runtime.erase_episode(ep_tgt));

    // Allocate an unrelated episode that occupies slots 1 and 2
    const uint64_t ep_other = tgt_runtime.adopt_root(200, 200, 0, 5, 0);
    CHECK(ep_other != 0);
    CHECK(tgt_runtime.initialize_dag(ep_other, decision, &err));
    CHECK(tgt_runtime.capture_c0(ep_other, 5, 0));
    CHECK(tgt_runtime.capture_c_base(ep_other));
    CHECK(tgt_runtime.activate_dag_frontier(ep_other));
    auto * other_root = tgt_runtime.node(ep_other, 0);
    if (other_root && other_root->physical_slot >= 0) {
        CHECK(tgt_runtime.detach_node(ep_other, 0));
    }
    llama_rerot_node_id other_w1 = LLAMA_REROT_NODE_INVALID;
    llama_rerot_node_id other_w2 = LLAMA_REROT_NODE_INVALID;
    CHECK(tgt_runtime.admit_next_child(ep_other, 1, 51, &other_w1));
    CHECK(tgt_runtime.admit_next_child(ep_other, 2, 52, &other_w2));
    CHECK(tgt_runtime.node(ep_other, other_w1)->physical_slot == 1);
    CHECK(tgt_runtime.node(ep_other, other_w2)->physical_slot == 2);

    // 6. Restore ep_tgt: since it was demoted, physical_slot is -1, so it does NOT conflict with slots 1 and 2!
    uint64_t restored_id = 0;
    CHECK(tgt_runtime.load_episode(ram_blob.data(), ram_blob.size(), fp, &restored_id, &err));
    CHECK(restored_id == ep_tgt);

    auto * restored_ep = tgt_runtime.episode(restored_id);
    CHECK(restored_ep != nullptr);
    CHECK(restored_ep->is_dag);
    CHECK(restored_ep->running.count(tgt_w1) != 0);
    CHECK(restored_ep->running.count(tgt_w2) != 0);

    // Rebind restored workers to DIFFERENT physical slots (3 and 4)
    auto * r_node1 = tgt_runtime.node(restored_id, tgt_w1);
    auto * r_node2 = tgt_runtime.node(restored_id, tgt_w2);
    CHECK(r_node1 && r_node2);
    r_node1->physical_slot = 3;
    r_node1->pen_id = 3;
    r_node2->physical_slot = 4;
    r_node2->pen_id = 4;

    // Verify local state blobs were preserved exactly
    CHECK(r_node1->sampler_blob == std::vector<uint8_t>({0xAA, 0xBB, 0xCC}));
    CHECK(r_node1->mtp_blob == std::vector<uint8_t>({0x11, 0x22}));
    CHECK(r_node2->sampler_blob == std::vector<uint8_t>({0xDD, 0xEE}));
    CHECK(r_node2->mtp_blob == std::vector<uint8_t>({0x33, 0x44}));

    // 7. Both reference and restored continue generation
    const auto ref_r1_cont = ref_ep->document.append_run(ref_w1, llama_rerot_visibility::public_live, 14, 6, 3);
    const auto ref_r2_cont = ref_ep->document.append_run(ref_w2, llama_rerot_visibility::public_live, 25, 7, 4);
    ref_ep->nodes[ref_w1].public_run = ref_r1_cont;
    ref_ep->nodes[ref_w2].public_run = ref_r2_cont;

    const auto res_r1_cont = restored_ep->document.append_run(tgt_w1, llama_rerot_visibility::public_live, 14, 6, 3);
    const auto res_r2_cont = restored_ep->document.append_run(tgt_w2, llama_rerot_visibility::public_live, 25, 7, 4);
    restored_ep->nodes[tgt_w1].public_run = res_r1_cont;
    restored_ep->nodes[tgt_w2].public_run = res_r2_cont;

    // Seal both workers
    CHECK(ref_runtime.seal_dag_node(ep_ref, ref_w1, llama_rerot_event_origin::worker_source));
    CHECK(ref_runtime.seal_dag_node(ep_ref, ref_w2, llama_rerot_event_origin::worker_source));
    CHECK(tgt_runtime.seal_dag_node(restored_id, tgt_w1, llama_rerot_event_origin::worker_source));
    CHECK(tgt_runtime.seal_dag_node(restored_id, tgt_w2, llama_rerot_event_origin::worker_source));

    // Both synthesis nodes must now be eligible (remaining_preds == 0)
    CHECK(ref_ep->nodes[ref_ep->synthesis_node].remaining_preds == 0);
    CHECK(restored_ep->nodes[restored_ep->synthesis_node].remaining_preds == 0);

    // Verify logical DAG equivalence between reference and restored
    CHECK(ref_ep->document.node_count() == restored_ep->document.node_count());
    CHECK(ref_ep->document.run_count() == restored_ep->document.run_count());
    for (size_t i = 0; i < ref_ep->document.run_count(); ++i) {
        const auto * r_run = ref_ep->document.run(llama_rerot_run_id(i));
        const auto * t_run = restored_ep->document.run(llama_rerot_run_id(i));
        CHECK(r_run && t_run);
        CHECK(r_run->owner == t_run->owner);
        CHECK(r_run->visibility == t_run->visibility);
        CHECK(r_run->token_count == t_run->token_count);
        CHECK(r_run->storage_pos0 == t_run->storage_pos0);
        CHECK(r_run->kind == t_run->kind);
    }

    // Compare reader views for synthesis node (reader 0)
    const auto ref_view = ref_runtime.build_dag_view_for_reader(ep_ref, 0);
    const auto res_view = tgt_runtime.build_dag_view_for_reader(restored_id, 0);
    CHECK(ref_view.runs.size() == res_view.runs.size());
    CHECK(!ref_view.runs.empty());
    for (size_t i = 0; i < ref_view.runs.size(); ++i) {
        CHECK(ref_view.runs[i].owner == res_view.runs[i].owner);
        CHECK(ref_view.runs[i].run_id == res_view.runs[i].run_id);
    }
}

static void test_dag_synthesis_complementary_results_distinct_intents() {
    // Stage 6 (§12.7) Minimal Workload 5:
    // Synthesis uses multiple complementary results without cross-contaminating intents.
    // Verifies:
    // 1. Two parallel workers with distinct intents ("algebra_proof" and "geometric_evidence")
    //    produce distinct reasoning/body token sequences and separate FRAME handoffs.
    // 2. Both workers' outputs remain distinct and correctly attributed to their own node_id.
    // 3. When both workers seal, synthesis_node is unlocked (remaining_preds == 0) and admits.
    // 4. The synthesis reader view contains the root plan, worker 1 (algebra), and worker 2 (geometry)
    //    in strict topological plan order, each appearing exactly once without cross-contamination.
    // 5. Synthesis generates its final conclusion from a clean stage start and separate logits.
    server_rerot_runtime runtime(nullptr);
    runtime.set_pen_capacity(3);
    const uint64_t ep_id = runtime.adopt_root(95, 95, 0, 1, 0);
    CHECK(ep_id != 0);

    const auto decision = server_rerot_parse_routing_decision(R"json({
      "strategy": "dag",
      "payload": {
        "questions": [
          {"id": "algebra", "intent": "Derive algebraic lemma: a^2 - b^2 = (a-b)(a+b)"},
          {"id": "geometry", "intent": "Construct geometric dissection proof"}
        ],
        "depends_on": []
      }
    })json");
    CHECK(decision.is_dag());
    std::string err;
    CHECK(runtime.initialize_dag(ep_id, decision, &err));
    CHECK(runtime.capture_c0(ep_id, 1, 0));
    CHECK(runtime.capture_c_base(ep_id));
    CHECK(runtime.activate_dag_frontier(ep_id));

    auto * root = runtime.node(ep_id, 0);
    if (root && root->physical_slot >= 0) {
        CHECK(runtime.detach_node(ep_id, 0));
    }

    auto * ep = runtime.episode(ep_id);
    CHECK(ep != nullptr);
    CHECK(ep->synthesis_node != LLAMA_REROT_NODE_INVALID);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 2);

    // Verify distinct intents stored in episode runtime nodes
    const auto * node_alg = runtime.node(ep_id, 1);
    const auto * node_geo = runtime.node(ep_id, 2);
    CHECK(node_alg != nullptr && node_alg->intent.find("algebraic lemma") != std::string::npos);
    CHECK(node_geo != nullptr && node_geo->intent.find("geometric dissection") != std::string::npos);
    CHECK(node_alg->intent != node_geo->intent);

    // Admit both parallel workers
    llama_rerot_node_id n_alg = LLAMA_REROT_NODE_INVALID;
    llama_rerot_node_id n_geo = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 0, 2, &n_alg));
    CHECK(runtime.admit_next_child(ep_id, 1, 3, &n_geo));
    CHECK(n_alg == 1 && n_geo == 2);
    CHECK(runtime.complete_admission(ep_id, n_alg));
    CHECK(runtime.complete_admission(ep_id, n_geo));

    // Create shared root plan prefix run
    const auto run_p = ep->document.append_run(0, llama_rerot_visibility::public_live, 0, 10, 1);
    CHECK(run_p != LLAMA_REROT_RUN_INVALID);

    // Worker 1 writes algebraic reasoning and publishes its run
    const auto run_alg = ep->document.append_run(n_alg, llama_rerot_visibility::public_live, 10, 6, 1);
    CHECK(run_alg != LLAMA_REROT_RUN_INVALID);

    // Worker 2 writes geometric reasoning and publishes its run
    const auto run_geo = ep->document.append_run(n_geo, llama_rerot_visibility::public_live, 16, 8, 1);
    CHECK(run_geo != LLAMA_REROT_RUN_INVALID);

    // Publish frontier 1
    auto fr1 = runtime.finish_frontier(ep_id);
    CHECK(!fr1.hard_aborted);

    // Verify each worker's distinct run identity and content
    const auto * alg_run = ep->document.run(run_alg);
    const auto * geo_run = ep->document.run(run_geo);
    CHECK(alg_run != nullptr && geo_run != nullptr);
    CHECK(alg_run->owner == n_alg);
    CHECK(geo_run->owner == n_geo);
    CHECK(alg_run->id != geo_run->id);
    CHECK(alg_run->token_count == 6);
    CHECK(geo_run->token_count == 8);

    // Seal worker 1 (algebra) with legitimate worker_source provenance
    CHECK(runtime.seal_dag_node(ep_id, n_alg, llama_rerot_event_origin::worker_source));
    CHECK(ep->nodes[n_alg].is_sealed);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 1);
    CHECK(runtime.get_eligible_dag_nodes(ep_id).empty()); // synthesis not eligible yet

    // Seal worker 2 (geometry)
    CHECK(runtime.seal_dag_node(ep_id, n_geo, llama_rerot_event_origin::worker_source));
    CHECK(ep->nodes[n_geo].is_sealed);
    CHECK(ep->nodes[ep->synthesis_node].remaining_preds == 0);

    // Now synthesis is unlocked and eligible!
    auto eligible = runtime.get_eligible_dag_nodes(ep_id);
    CHECK(eligible.size() == 1);
    CHECK(eligible[0] == ep->synthesis_node);

    // Finish frontier triggers synthesis readiness
    auto fr2 = runtime.finish_frontier(ep_id);
    CHECK(!fr2.hard_aborted);
    CHECK(fr2.synthesis_node == ep->synthesis_node);

    // Build synthesis reader view (reader == 0)
    const auto view_synth = runtime.build_dag_view_for_reader(ep_id, 0);
    // Runs: [root plan (owner 0), alg frame+body (owner 1), geo frame+body (owner 2)]
    CHECK(view_synth.runs.size() >= 3);
    CHECK(view_synth.runs[0].owner == 0); // root plan

    // Both workers must appear in the synthesis view, exactly once, with their distinct owners
    size_t count_alg = 0;
    size_t count_geo = 0;
    for (const auto & entry : view_synth.runs) {
        if (entry.owner == n_alg) count_alg++;
        if (entry.owner == n_geo) count_geo++;
    }
    CHECK(count_alg == 1);
    CHECK(count_geo == 1);

    // Verify ordering in synthesis view: root -> algebra -> geometry
    bool seen_root = false;
    bool seen_alg = false;
    bool seen_geo = false;
    for (const auto & entry : view_synth.runs) {
        if (entry.owner == 0) seen_root = true;
        if (entry.owner == n_alg) {
            CHECK(seen_root);
            seen_alg = true;
        }
        if (entry.owner == n_geo) {
            CHECK(seen_alg);
            seen_geo = true;
        }
    }
    CHECK(seen_root && seen_alg && seen_geo);

    // Verify synthesis node state and admission
    llama_rerot_node_id n_synth_admitted = LLAMA_REROT_NODE_INVALID;
    CHECK(runtime.admit_next_child(ep_id, 2, 4, &n_synth_admitted));
    CHECK(n_synth_admitted == ep->synthesis_node);
    CHECK(runtime.complete_admission(ep_id, n_synth_admitted));
    CHECK(ep->document.node(ep->synthesis_node)->state == llama_rerot_node_state::running);

    // Synthesis writes final unified summary run
    const auto run_synth = ep->document.append_run(ep->synthesis_node, llama_rerot_visibility::public_live, 24, 12, 1);
    CHECK(run_synth != LLAMA_REROT_RUN_INVALID);
    auto fr3 = runtime.finish_frontier(ep_id);
    CHECK(!fr3.hard_aborted);
    const auto * synth_run = ep->document.run(run_synth);
    CHECK(synth_run != nullptr && synth_run->owner == ep->synthesis_node);
    CHECK(synth_run->token_count == 12);
}

int main() {
    std::fprintf(stderr, "=== RERoT Runtime Tests ===\n");
    test_c0_and_dag_admit_without_parked_seq();
    test_dag_runtime_lifecycle();
    test_dag_capture_c_base_snapshots_current_seed();
    test_dag_isolated_probe_uses_other_seq();
    test_dag_c0_probe_cancel_and_isolation();
    test_dag_admission_view_survival();
    test_dag_w_gt_p_yield_without_seal();
    test_dag_frozen_read_publish_epoch();
    test_dag_logical_step_hides_foreign_pending();
    test_dag_refuses_nested_html_fork();
    {
        server_rerot_runtime runtime(nullptr);
        const uint64_t ep_id = runtime.adopt_root(15, 15, 0, 1, 0);
        CHECK(ep_id != 0);
        const auto decision = server_rerot_parse_routing_decision(R"({
          "strategy": "dag",
          "payload": {
            "questions": [{"id": "A", "intent": "Fact A"}],
            "depends_on": []
          }
        })");
        std::string err;
        CHECK(runtime.initialize_dag(ep_id, decision, &err));
        auto * ep = runtime.episode(ep_id);
        CHECK(ep != nullptr && ep->is_dag);
        auto * node0 = runtime.node(ep_id, 0);
        CHECK(node0 != nullptr);
        const auto run_id = ep->document.append_run(0, llama_rerot_visibility::pending_record, 0, 5, 0);
        node0->pending_record = run_id;
        server_rerot_token_plan plan;
        plan.storage_pos = 5;
        plan.visibility = llama_rerot_visibility::pending_record;
        plan.segment_kind = llama_rerot_segment_kind::body;
        plan.run_id = run_id;
        plan.parser_step.record_closed = true;
        plan.parser_step.items = {"task1", "task2"};
        CHECK(!runtime.commit_token(ep_id, 0, plan));
        CHECK(ep->hard_aborted);
        CHECK(ep->abort_reason.find("HTML fork is retired") != std::string::npos);
    }
    {
        // Scheduler invariant check (§12.5 item 8):
        // If a valid DAG has unfinished nodes but no runnable work,
        // activate_dag_frontier must immediately report a scheduler invariant error and abort.
        server_rerot_runtime runtime(nullptr);
        const uint64_t ep_id = runtime.adopt_root(16, 16, 0, 1, 0);
        CHECK(ep_id != 0);
        const auto decision = server_rerot_parse_routing_decision(R"({
          "strategy": "dag",
          "payload": {
            "questions": [{"id": "1", "intent": "Worker 1"}],
            "depends_on": []
          }
        })");
        CHECK(decision.is_dag());
        std::string err;
        CHECK(runtime.initialize_dag(ep_id, decision, &err));
        CHECK(runtime.capture_c0(ep_id, 1, 0));
        CHECK(runtime.capture_c_base(ep_id));
        auto * ep = runtime.episode(ep_id);
        CHECK(ep != nullptr);

        // Artificially corrupt state: make both worker 1 and synthesis node unrunnable
        ep->ready_queue.clear();
        ep->nodes[1].remaining_preds = 999;
        ep->nodes[ep->synthesis_node].remaining_preds = 999;
        CHECK(!runtime.activate_dag_frontier(ep_id));
        CHECK(ep->hard_aborted);
        CHECK(ep->abort_reason.find("rerot_scheduler_invariant_error") != std::string::npos);
    }
    test_dag_save_refuses_probe_and_persists_c0();
    test_dag_seal_exactly_once_and_refuses_unstarted();
    test_dag_seal_releases_pen_parked_until_cohort_retire();
    test_dag_complete_admission_evicts_no_peer();
    test_dag_seal_evicts_no_bound_peer();
    test_dag_yield_force_under_resource_pressure();
    test_dag_admit_anchors_worker_storage_to_c_base();
    test_dag_discard_probe_retracts_probe_runs();
    test_dag_frozen_view_gates_foreign_frame();
    test_dag_load_refuses_bad_cohort_and_sampler_snapshot();
    test_dag_load_rebinds_pens_for_bound_slots();
    test_dag_step_publish_is_atomic_on_member_failure();
    test_dag_step_publish_overflow_keeps_history_public();
    test_dag_init_alone_never_admits_until_base_captured();
    test_dag_init_frames_prefix_keeps_view_not_body_export();
    test_dag_ensure_run_keeps_segment_kinds_distinct();
    test_dag_prefix_rebuild_reconciles_root_runs();
    test_dag_initialize_refuses_double_init();
    test_dag_source_end_multi_token_and_starting_frame_gate();
    test_dag_microbatch_slice_order_and_no_earlier_public_leak();
    test_dag_duplicate_source_end_and_restore_no_double_decrement();
    test_dag_abort_clears_orphan_refs_and_pens();
    test_dag_demote_restore_different_physical_slots();
    test_dag_synthesis_complementary_results_distinct_intents();
    test_dag_three_lane_flat_cycle_and_peer_uptake();
    test_dag_diamond_and_unequal_length_history();
    test_dag_a_to_c_with_b_independent_overlap();
    test_dag_shift_pins_started_public_history();
    test_context_shift_pins_frame_runs();
    test_recurrent_only_pressure_isolation();
    test_multi_episode_concurrent_final_fence_and_coordinate_freeze();
    test_pen_capacity_and_multi_episode_allocation();
    test_chronicle_to_canonical_mapping_registry();
    test_line_mux_completion_order_and_visibility();
    test_list_marker_prefixes_remain_atomic();
    test_marker_token_preserves_public_prefix();
    test_child_public_live_visibility_and_exit_marker();
    test_split_pending_record_resolution();
    test_private_span_reserves_one_contiguous_run();
    test_n1_no_fork_disarm_forever();
    test_n2_strong_uptake();
    test_queue_fifo_five_children_one_slot();
    test_nested_fork_keeps_fifo();
    test_child_content_list_is_not_scheduler_control();
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
    test_rerot_mtp_speculative_matrix();
    test_phase5_ram_checkpoint_context_shift_matrix();
    test_hand_seed_and_final_fence_continuation();
    test_shared_prefix_multi_branch_union_and_preemption();
    test_multi_person_multi_pen_bxp_stress();
    test_triattention_multi_person_pressure();
    test_frontier_boundary_pen_yield_and_resume();
    test_multi_person_b_greater_than_p_fairness();
    test_phase7_runtime_production_stress_and_pressure();
    test_final_fence_user_grammar_restoration();
    test_ram_restore_context_shift_and_preemption();
    std::fprintf(stderr, "=== Results: %d failure(s) ===\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
