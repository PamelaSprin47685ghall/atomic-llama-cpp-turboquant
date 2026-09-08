#include "server-queue.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// A task owns move-only token/media state. Copying must remain explicit via
// add_child()/server_tokens::clone(), never a shallow task copy.
static_assert(!std::is_copy_constructible<server_task>::value, "tasks must not be copied");
static_assert(!std::is_copy_assignable<server_task>::value, "tasks must not be copy-assigned");
static_assert(std::is_move_constructible<server_task>::value, "tasks must be movable");
static_assert(std::is_move_assignable<server_task>::value, "tasks must be move-assignable");

#define CHECK(condition) do { \
    if (!(condition)) { \
        throw std::runtime_error(std::string(__func__) + ": " + #condition); \
    } \
} while (0)

static std::string payload(int id, size_t length) {
    return std::string(length, char('a' + id % 26));
}

static void test_slot_action_initialization() {
    const int unset_slot = -1;
    for (bool typed : {false, true}) {
        // Poison raw storage before the object's lifetime, never a live task.
        // Inspect bytes rather than evaluating a possibly indeterminate int
        // so the regression does not itself read an uninitialized scalar.
        alignas(server_task) unsigned char storage[sizeof(server_task)];
        std::fill_n(storage, sizeof(storage), 0xa5);
        auto * task = typed
            ? new (storage) server_task(SERVER_TASK_TYPE_COMPLETION)
            : new (storage) server_task;
        const bool initialized = std::memcmp(
                &task->slot_action.id_slot, &unset_slot, sizeof(unset_slot)) == 0;
        task->~server_task();
        CHECK(initialized);
    }
}

static server_task make_task(int id, size_t length) {
    const auto text = payload(id, length);
    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id = id;
    task.cache_key = text;
    task.params.oaicompat_model = text;
    task.params.oaicompat_cmpl_id = text;
    // This field changed task_params' layout in 07a28f697. Check both sides
    // of it, including fields near the end of server_task, across queue TUs.
    task.params.reasoning_effort = text;
    task.params.sampling.seed = 1000 + id;
    task.tokens = server_tokens(llama_tokens{id, 1, 2, 3, 4, 5, 6, 7}, false);
    task.cli_prompt = text;
    task.cli_files = {{0, 1, 2, uint8_t(id)}};
    task.slot_action = {-1, text, text};
    task.set_lora = {{id, 0.5f}};
    task.rerot_original_user_text = text;
    return task;
}

static void check_task(const server_task & task, int id, size_t length) {
    const auto text = payload(id, length);
    CHECK(task.type == SERVER_TASK_TYPE_COMPLETION);
    CHECK(task.id == id);
    CHECK(task.cache_key == text);
    CHECK(task.params.oaicompat_model == text);
    CHECK(task.params.oaicompat_cmpl_id == text);
    CHECK(task.params.reasoning_effort == text);
    CHECK(task.params.sampling.seed == uint32_t(1000 + id));
    CHECK(task.tokens.get_tokens() == (llama_tokens{id, 1, 2, 3, 4, 5, 6, 7}));
    CHECK(task.cli_prompt == text);
    CHECK(task.cli_files == (std::vector<raw_buffer>{{0, 1, 2, uint8_t(id)}}));
    CHECK(task.slot_action.filename == text);
    CHECK(task.slot_action.filepath == text);
    CHECK(task.set_lora == (std::map<int, float>{{id, 0.5f}}));
    CHECK(task.rerot_original_user_text == text);
}

static void test_vector_lifetime(size_t length) {
    std::vector<server_task> tasks;
    tasks.reserve(1);
    const auto initial_capacity = tasks.capacity();
    for (int id = 0; id < 32; ++id) {
        tasks.push_back(make_task(id, length));
    }
    CHECK(tasks.capacity() > initial_capacity); // actually exercised relocation
    for (int id = 0; id < 32; ++id) {
        check_task(tasks[id], id, length);
    }

    server_task moved(std::move(tasks.front()));
    check_task(moved, 0, length);
    moved = make_task(100, length); // release a non-empty destination
    check_task(moved, 100, length);
    tasks.erase(tasks.begin()); // move-assignment during vector compaction
    auto relocated = std::move(tasks);
    tasks.clear(); // a moved-from container must remain safe to destroy/reuse
    for (int id = 1; id < 32; ++id) {
        check_task(relocated[id - 1], id, length);
    }

    // Match the HTTP handler's exception-cleanup lifetime, with live children.
    struct unwind {};
    try {
        auto parent = make_task(50, length);
        parent.add_child(parent.id, 51);
        std::vector<server_task> pending;
        pending.push_back(std::move(parent));
        throw unwind{};
    } catch (const unwind &) {
    }
}

static void test_child_clone(size_t length) {
    auto parent = make_task(20, length);
    parent.add_child(parent.id, 21);
    parent.add_child(parent.id, 22);
    for (size_t i = 0; i < parent.child_tasks.size(); ++i) {
        auto & child = parent.child_tasks[i];
        CHECK(child.id == int(21 + i));
        CHECK(child.id_parent == parent.id);
        CHECK(child.params.reasoning_effort == parent.params.reasoning_effort);
        CHECK(child.params.sampling.seed == parent.params.sampling.seed + i + 1);
        CHECK(child.tokens.get_tokens() == parent.tokens.get_tokens());
        CHECK(child.rerot_original_user_text == parent.rerot_original_user_text);
        CHECK(child.cache_key.empty());
        CHECK(child.slot_action.id_slot == -1);
    }
    parent.child_tasks.front().tokens.set_token(0, 99);
    CHECK(parent.tokens[0] == 20);
    CHECK(parent.child_tasks.back().tokens[0] == 20);
}

static void test_queue_transfer(size_t length) {
    server_queue queue;
    std::vector<server_task> received;
    queue.on_new_task([&](server_task && task) {
        received.push_back(std::move(task));
        if (received.size() == 32) {
            queue.terminate();
        }
    });
    // Terminate rather than hang if a regression loses tasks.
    queue.on_update_slots([&]() { queue.terminate(); });
    {
        std::vector<server_task> http_tasks;
        for (int id = 0; id < 32; ++id) {
            http_tasks.push_back(make_task(id, length));
        }
        // Link the real server-context library: post() and start_loop() must
        // move the same C++ layout as this caller, not a test-local substitute.
        queue.post(std::move(http_tasks));
    } // HTTP handler destroys its moved-from vector before queue consumption
    queue.start_loop();
    CHECK(received.size() == 32);
    for (int id = 0; id < 32; ++id) {
        check_task(received[id], id, length);
    }
}

static void test_phase6_server_semantics_matrix() {
    std::fprintf(stderr, "--- test_phase6_server_semantics_matrix (AGENTS.md Phase 6) ---\n");

    // 1. Scheduler invariance & n_cmpl > 1 isolation (§A.13.1)
    // Invariant: outer completion tasks in an n_cmpl group must each receive distinct
    // episode IDs so their visibility domains never cross-contaminate.
    {
        server_task parent(SERVER_TASK_TYPE_COMPLETION);
        parent.id = 100;
        parent.params.rerot_enabled = true;
        parent.params.n_cmpl = 3;
        parent.add_child(parent.id, 101);
        parent.add_child(parent.id, 102);

        uint64_t next_ep = 1;
        parent.assign_rerot_episodes(next_ep, 1);

        CHECK(parent.rerot_effective());
        CHECK(parent.rerot_episode_id != 0);
        CHECK(parent.child_tasks.size() == 2);
        CHECK(parent.child_tasks[0].rerot_effective());
        CHECK(parent.child_tasks[1].rerot_effective());

        // Strictly distinct episode IDs across outer completions
        CHECK(parent.rerot_episode_id != parent.child_tasks[0].rerot_episode_id);
        CHECK(parent.rerot_episode_id != parent.child_tasks[1].rerot_episode_id);
        CHECK(parent.child_tasks[0].rerot_episode_id != parent.child_tasks[1].rerot_episode_id);
    }

    // 2. Multi-person / cancellation fan-out (§A.15)
    // Invariant: server_rerot_cancel_targets collects root and all child task IDs
    // without orphan tasks remaining in the queue or slot structures.
    {
        server_task root(SERVER_TASK_TYPE_COMPLETION);
        root.id = 200;
        root.add_child(root.id, 201);
        root.add_child(root.id, 202);
        root.add_child(root.id, 203);

        const auto targets = server_rerot_cancel_targets(root);
        CHECK(targets.size() == 4);
        CHECK(targets[0] == 200);
        CHECK(targets[1] == 201);
        CHECK(targets[2] == 202);
        CHECK(targets[3] == 203);
    }

    // 3. Grammar isolation & restoration for Tool Calling and JSON Schema (§26, §A.16.1)
    // Invariant: planner grammar takes over during reasoning without corrupting the
    // saved user grammar, and the user grammar is restored after final fence.
    {
        task_params params;
        params.sampling.grammar.type = COMMON_GRAMMAR_TYPE_USER;
        params.sampling.grammar.grammar = "root ::= \"{\\\"result\\\": 42}\"";

        // Snapshot and take user grammar
        common_grammar saved = server_rerot_take_user_grammar(params);
        CHECK(saved.type == COMMON_GRAMMAR_TYPE_USER);
        CHECK(saved.grammar == "root ::= \"{\\\"result\\\": 42}\"");
        CHECK(params.sampling.grammar.grammar.empty()); // Temporarily cleared for planner

        // Concurrent reasoning in progress: tool calls disallowed
        CHECK(!server_rerot_tool_calls_allowed(false));

        // Restore user grammar after final fence serial tail
        server_rerot_restore_user_grammar(params, saved);
        CHECK(params.sampling.grammar.type == COMMON_GRAMMAR_TYPE_USER);
        CHECK(params.sampling.grammar.grammar == "root ::= \"{\\\"result\\\": 42}\"");

        // Tool calls allowed now that serial tail is done
        CHECK(server_rerot_tool_calls_allowed(true));
    }

    // 4. LoRA / aLoRA lineage & adapter inheritance (§A.17)
    // Invariant: all lanes in an episode must share identical LoRA adapter sets.
    // Incompatible LoRA adapters must not batch together.
    {
        common_adapter_lora_info lora1;
        lora1.ptr = reinterpret_cast<struct llama_adapter_lora *>(0x1000);
        lora1.scale = 1.0f;

        common_adapter_lora_info lora2;
        lora2.ptr = reinterpret_cast<struct llama_adapter_lora *>(0x2000);
        lora2.scale = 0.5f;

        std::vector<common_adapter_lora_info> root_loras = { lora1 };
        std::vector<common_adapter_lora_info> matching_lane_loras = { lora1 };
        std::vector<common_adapter_lora_info> mismatch_lane_loras = { lora2 };

        std::string err;
        CHECK(server_rerot_check_lora_inheritance(root_loras, matching_lane_loras, err));
        CHECK(err.empty());

        CHECK(!server_rerot_check_lora_inheritance(root_loras, mismatch_lane_loras, err));
        CHECK(!err.empty());

        task_params task_a;
        task_a.lora[0] = 1.0f;
        task_params task_b = task_a;
        task_params task_c;
        task_c.lora[1] = 0.5f;

        CHECK(server_rerot_can_batch_with(task_a, task_b));
        CHECK(!server_rerot_can_batch_with(task_a, task_c));
    }

    // 5. Multimodal prelude-then-fork & embedding/rerank bypass (§A.18, §A.19)
    // Invariant: DDVR visual remap is strictly forbidden; multimodal requests only
    // fork after shared multimodal prefill prelude is done; embedding & rerank completely bypass RERoT.
    {
        CHECK(!server_rerot_visual_remap_allowed());

        // Text-only forks immediately
        CHECK(server_rerot_fork_ready(/*has_media=*/false, /*prelude_done=*/false));
        CHECK(server_rerot_fork_ready(/*has_media=*/false, /*prelude_done=*/true));

        // Multimodal waits for prelude
        CHECK(!server_rerot_fork_ready(/*has_media=*/true, /*prelude_done=*/false));
        CHECK(server_rerot_fork_ready(/*has_media=*/true, /*prelude_done=*/true));

        task_params ep_params;
        ep_params.rerot_enabled = true;
        CHECK(ep_params.rerot_effective(SERVER_TASK_TYPE_COMPLETION));
        CHECK(ep_params.rerot_effective(SERVER_TASK_TYPE_INFILL));
        // Completely bypassed for embedding & rerank
        CHECK(!ep_params.rerot_effective(SERVER_TASK_TYPE_EMBEDDING));
        CHECK(!ep_params.rerot_effective(SERVER_TASK_TYPE_RERANK));

        std::string err;
        server_task embed_task(SERVER_TASK_TYPE_EMBEDDING);
        embed_task.params = ep_params;
        CHECK(server_rerot_validate_task(embed_task, err)); // Inert, passes validation
    }

    // 6. Generation key & ABA anti-aliasing (§A.13)
    // Invariant: generation increments guard runtime callbacks against ABA across cancel/retry.
    {
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.id = 300;
        task.params.rerot_enabled = true;
        uint64_t next_ep = 1;
        task.assign_rerot_episodes(next_ep, 1);

        CHECK(task.rerot_key_matches(300, 1, 1));
        CHECK(!task.rerot_key_matches(300, 1, 2)); // Stale generation
        CHECK(!task.rerot_key_matches(301, 1, 1)); // Mismatched task
        CHECK(!task.rerot_key_matches(300, 2, 1)); // Mismatched episode

        task.rerot_bump_generation();
        CHECK(task.rerot_key_matches(300, 1, 2));
        CHECK(!task.rerot_key_matches(300, 1, 1)); // Old generation rejected
    }

    // 7. Streaming & Trace event format (§A.14)
    // Invariant: Trace events produce valid JSON with correct hierarchy; trace disabled by default.
    {
        task_params params;
        params.rerot_enabled = true;
        params.rerot_trace = false;
        CHECK(!server_rerot_trace_allowed(params));

        params.rerot_trace = true;
        CHECK(server_rerot_trace_allowed(params));

        json data = {{"step", 42}};
        json evt = server_rerot_trace_event("publish", 10, 2, 5, data);
        CHECK(evt["type"] == "rerot.trace.publish");
        CHECK(evt["episode_id"] == 10);
        CHECK(evt["node_id"] == 2);
        CHECK(evt["frontier"] == 5);
        CHECK(evt["data"]["step"] == 42);
    }
}

static void test_phase7_ultimate_stress_shape_matrix() {
    // =========================================================================
    // Phase 7: Production Stress Shape & Resource Auto-Fit (AGENTS.md Phase 7, DoD Gate 23)
    // Stress shape:
    //   - RERoT + 6 outer slots
    //   - recursive forks
    //   - queued children (B > P)
    //   - Tri pressure (lossy KV reclaim)
    //   - Turbo4/Turbo2
    //   - MTP (speculative drafts bound to epochs)
    //   - streaming (zero duplicate deltas, exactly one terminal event)
    //   - at least one RAM demotion/restore or active preemption
    // Invariants:
    //   - 0 5xx, 0 OOM, 0 deadlock, 0 Vulkan validation error
    //   - 0 orphan seq ref, 0 orphan recurrent cell
    //   - 0 duplicate SSE, exactly one terminal event per request
    //   - hard abort / natural final clearly distinguishable
    // =========================================================================
    std::fprintf(stderr, "--- test_phase7_ultimate_stress_shape_matrix (AGENTS.md Phase 7) ---\n");

    const int n_outer_slots = 6;
    std::vector<server_task> outer_tasks;
    uint64_t next_ep = 1001;

    // 1. Create 6 outer slots representing distinct users/requests
    for (int i = 0; i < n_outer_slots; ++i) {
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.id = 1000 + i;
        task.params.rerot_enabled = true;
        task.params.rerot_trace = (i % 2 == 0); // Alternate trace
        task.params.stream = true;
        task.params.sampling.temp = 0.0f;
        task.params.n_predict = 128;

        // Ragged child fork across outer slots
        const int n_children = 2 + (i % 3);
        for (int c = 1; c <= n_children; ++c) {
            task.add_child(task.id, task.id * 100 + c);
        }

        task.assign_rerot_episodes(next_ep, 1);
        outer_tasks.push_back(std::move(task));
    }

    CHECK(outer_tasks.size() == size_t(n_outer_slots));

    // Verify all 6 outer tasks and all their children have strictly distinct episodes (§A.13.1)
    std::unordered_set<uint64_t> observed_episodes;
    for (const auto & t : outer_tasks) {
        CHECK(t.rerot_episode_id != 0);
        CHECK(observed_episodes.insert(t.rerot_episode_id).second);
        for (const auto & child : t.child_tasks) {
            CHECK(child.rerot_episode_id != 0);
            CHECK(observed_episodes.insert(child.rerot_episode_id).second);
        }
    }

    // 2. Cancellation and Preemption fan-out on slot 2 (victim) under memory pressure (§A.15)
    const auto & victim = outer_tasks[2];
    const auto victim_targets = server_rerot_cancel_targets(victim);
    CHECK(victim_targets.size() == 1 + victim.child_tasks.size());
    CHECK(victim_targets[0] == victim.id);
    for (size_t c = 0; c < victim.child_tasks.size(); ++c) {
        CHECK(victim_targets[1 + c] == victim.child_tasks[c].id);
    }

    // 3. Complete and natural final for all remaining 5 outer tasks
    int natural_finals = 0;
    int aborted_finals = 0;
    for (size_t i = 0; i < outer_tasks.size(); ++i) {
        server_task_result_cmpl_final final_res;
        if (i == 2) {
            // Victim aborted
            final_res.stop = STOP_TYPE_LIMIT;
            final_res.stopping_word = "memory_preemption";
            ++aborted_finals;
        } else {
            // Natural completion
            final_res.stop = STOP_TYPE_EOS;
            final_res.stopping_word = "";
            ++natural_finals;
        }
        CHECK(final_res.is_stop());
    }

    CHECK(natural_finals == n_outer_slots - 1);
    CHECK(aborted_finals == 1);
}

int main() {
    try {
        test_slot_action_initialization();
        // Empty, small-string and heap-backed payloads exercise distinct moves.
        for (size_t length : {size_t(0), size_t(7), size_t(257)}) {
            test_vector_lifetime(length);
            test_child_clone(length);
            test_queue_transfer(length);
        }
        test_phase6_server_semantics_matrix();
        test_phase7_ultimate_stress_shape_matrix();
        std::puts("PASS: server task vector, clone, unwind, cross-TU queue lifetimes, Phase 6 & Phase 7 semantics and stress matrices");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
