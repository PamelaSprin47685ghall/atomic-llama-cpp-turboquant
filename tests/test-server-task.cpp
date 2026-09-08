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

int main() {
    try {
        test_slot_action_initialization();
        // Empty, small-string and heap-backed payloads exercise distinct moves.
        for (size_t length : {size_t(0), size_t(7), size_t(257)}) {
            test_vector_lifetime(length);
            test_child_clone(length);
            test_queue_transfer(length);
        }
        std::puts("PASS: server task vector, clone, unwind and cross-TU queue lifetimes");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
