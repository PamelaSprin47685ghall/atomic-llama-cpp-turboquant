// Compare queued prompt scores with independent, single-token model evaluations.
// Run with a small decoder GGUF as argv[1] or LLAMA_TEST_MODEL.

#include "server-context.h"
#include "server-queue.h"
#include "server-task.h"

#include "common.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static std::vector<float> reference_scores(llama_context * ctx, const llama_tokens & tokens) {
    llama_memory_clear(llama_get_memory(ctx), true);
    std::vector<float> result(tokens.size(), std::numeric_limits<float>::quiet_NaN());
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        llama_token token = tokens[i];
        llama_pos position = (llama_pos) i;
        llama_seq_id sequence = 0;
        llama_seq_id * sequences = &sequence;
        int32_t n_sequences = 1;
        int8_t output = 1;
        llama_batch batch = {1, &token, nullptr, &position, &n_sequences, &sequences, &output};
        const int status = llama_decode(ctx, batch);
        llama_synchronize(ctx);
        require(status == 0, "reference decode failed");
        const float * logits = llama_get_logits_ith(ctx, 0);
        require(logits != nullptr, "reference logits are missing");
        const double maximum = *std::max_element(logits, logits + n_vocab);
        long double denominator = 0.0;
        for (int32_t j = 0; j < n_vocab; ++j) {
            denominator += std::exp((long double) logits[j] - maximum);
        }
        result[i + 1] = (float) (std::log(denominator) - ((long double) logits[tokens[i + 1]] - maximum));
    }
    llama_memory_clear(llama_get_memory(ctx), true);
    return result;
}

static void compare_scores(server_task_result * result, const std::vector<float> & expected) {
    require(result != nullptr && !result->is_error(), "score task failed");
    const auto * score = dynamic_cast<server_task_result_score *>(result);
    require(score != nullptr, "score task returned the wrong result type");
    require(score->nll.size() == expected.size(), "score count does not match the input");
    if (expected.empty()) {
        return;
    }
    require(std::isnan(score->nll[0]), "the first input token must have no score");
    for (size_t i = 1; i < expected.size(); ++i) {
        const float tolerance = 2e-3f + 1e-4f * std::abs(expected[i]);
        if (!std::isfinite(score->nll[i]) || std::abs(score->nll[i] - expected[i]) > tolerance) {
            std::fprintf(stderr, "score mismatch at %zu: expected %.8f, got %.8f\n", i, expected[i], score->nll[i]);
            throw std::runtime_error("queued score differs from the single-token reference");
        }
    }
}

static server_task make_task(server_response_reader & reader, server_task_type type, const llama_tokens & tokens) {
    server_task task(type);
    task.id = reader.get_new_id();
    task.tokens = server_tokens(tokens, false);
    return task;
}

static std::function<bool()> deadline() {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    return [end]() { return std::chrono::steady_clock::now() > end; };
}

struct running_server {
    server_context & server;
    std::thread thread;

    explicit running_server(server_context & server) : server(server), thread([&server]() { server.start_loop(); }) {}
    ~running_server() {
        server.terminate();
        thread.join();
    }
};

struct running_queue {
    server_queue & queue;
    std::thread thread;

    explicit running_queue(server_queue & queue) : queue(queue), thread([&queue]() { queue.start_loop(20); }) {}
    ~running_queue() {
        queue.terminate();
        thread.join();
    }
};

static void test_queue_activity() {
    server_queue queue;
    std::mutex mutex;
    std::condition_variable condition;
    bool started = false;
    bool sleeping = false;
    size_t sleep_count = 0;
    size_t wake_count = 0;
    queue.on_new_task([](server_task &&) {});
    queue.on_update_slots([&]() {
        std::lock_guard<std::mutex> lock(mutex);
        started = true;
        condition.notify_all();
    });
    queue.on_sleeping_state([&](bool value) {
        std::lock_guard<std::mutex> lock(mutex);
        sleeping = value;
        if (value) {
            ++sleep_count;
        } else {
            ++wake_count;
        }
        condition.notify_all();
    });

    std::optional<server_queue::model_activity> activity(queue.acquire_model_activity());
    running_queue loop(queue);
    {
        // Releasing another guard, including a moved guard, must not end this lease.
        auto nested = queue.acquire_model_activity();
        auto moved = std::move(nested);
    }
    {
        std::unique_lock<std::mutex> lock(mutex);
        require(condition.wait_for(lock, std::chrono::seconds(3), [&] { return started; }), "queue did not start");
        // The queue checks its idle deadline at least once per second.
        require(!condition.wait_for(lock, std::chrono::milliseconds(1100), [&] { return sleep_count != 0; }),
                "model slept while read-only work held an activity guard");
    }
    activity.reset();
    {
        std::unique_lock<std::mutex> lock(mutex);
        require(condition.wait_for(lock, std::chrono::seconds(3), [&] { return sleeping; }),
                "model did not sleep after the activity ended");
    }
    {
        auto awakened = queue.acquire_model_activity();
        require(!queue.is_sleeping(), "activity acquisition returned before the model woke");
        std::lock_guard<std::mutex> lock(mutex);
        require(!sleeping && wake_count == 1, "activity acquisition did not finish the wake callback");
    }
    {
        std::unique_lock<std::mutex> lock(mutex);
        require(condition.wait_for(lock, std::chrono::seconds(3), [&] { return sleep_count == 2; }),
                "model did not sleep after the awakened activity ended");
    }
}

static void test_scores(const char * model_path) {
    common_params params;
    params.model.path = model_path;
    params.n_gpu_layers = 0;
    params.n_ctx = 512;
    params.n_parallel = 2;
    params.n_parallel_explicit = true;
    params.n_batch = 64;
    params.n_ubatch = 64;
    params.n_ctx_checkpoints = 0;
    params.kv_unified = true;
    params.prefill_priority = true;
    params.cpuparams.n_threads = 2;
    params.cpuparams_batch.n_threads = 2;

    server_context server;
    require(server.load_model(params), "cannot load the test model");
    const int32_t n_ctx = server.get_meta().slot_n_ctx;
    llama_context * ctx = server.get_llama_context();
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    std::string text;
    for (int i = 0; i < 256; ++i) {
        text += "Once upon a time a little fox found a red book. The book had many stories. ";
    }
    llama_tokens tokens = common_tokenize(vocab, text, true, true);
    require((int32_t) tokens.size() > n_ctx, "test input is shorter than the context");

    // The one-token case has no output rows; 33 ends with a zero-output batch.
    // Requests may arrive together, but each scoring window must own shared KV.
    const std::vector<size_t> lengths = {0, 1, 32, 33, 65, 97, (size_t) n_ctx};
    std::vector<llama_tokens> windows;
    std::vector<std::vector<float>> expected;
    for (const size_t length : lengths) {
        windows.emplace_back(tokens.begin(), tokens.begin() + length);
        expected.push_back(reference_scores(ctx, windows.back()));
    }

    running_server loop(server);
    {
        auto reader = server.get_response_reader();
        std::vector<server_task> tasks;
        for (size_t i = 0; i < windows.size(); ++i) {
            tasks.push_back(make_task(reader, SERVER_TASK_TYPE_SCORE, windows[i]));
        }
        tasks.push_back(make_task(reader, SERVER_TASK_TYPE_METRICS, {}));
        reader.post_tasks(std::move(tasks));
        size_t received_scores = 0;
        bool received_metrics = false;
        const auto should_stop = deadline();
        while (reader.has_next()) {
            auto result = reader.next(should_stop);
            require(result && !result->is_error(), "parallel score tasks failed");
            if (const auto * metrics = dynamic_cast<server_task_result_metrics *>(result.get())) {
                require(metrics->n_processing_slots == 1, "shared KV must have only one active score window");
                received_metrics = true;
            } else {
                compare_scores(result.get(), expected.at(result->index));
                ++received_scores;
            }
        }
        require(received_metrics && received_scores == windows.size(), "parallel score results are missing");
    }
    {
        auto reader = server.get_response_reader();
        llama_tokens oversized(tokens.begin(), tokens.begin() + n_ctx + 1);
        reader.post_task(make_task(reader, SERVER_TASK_TYPE_SCORE, oversized));
        auto result = reader.next(deadline());
        require(result && result->is_error(), "an oversized score task must fail");
        const auto * error = dynamic_cast<server_task_result_error *>(result.get());
        require(error && error->err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE, "wrong context-limit error");
    }
    for (bool score_first : {false, true}) {
        // Neither request fits beside the other in shared KV. With prefill
        // priority enabled, concurrent scoring would interrupt generation.
        // Check both arrival orders, including a backend-sampled completion.
        auto reader = server.get_response_reader();
        std::vector<server_task> tasks;
        auto score = make_task(reader, SERVER_TASK_TYPE_SCORE, windows.back());
        llama_tokens prompt(tokens.begin(), tokens.begin() + n_ctx - 8);
        auto completion = make_task(reader, SERVER_TASK_TYPE_COMPLETION, prompt);
        completion.params.n_predict = 4;
        completion.params.sampling.temp = 0.0f;
        completion.params.sampling.backend_sampling = true;
        if (score_first) {
            tasks.push_back(std::move(score));
            tasks.push_back(std::move(completion));
        } else {
            tasks.push_back(std::move(completion));
            tasks.push_back(std::move(score));
        }
        reader.post_tasks(std::move(tasks));
        auto results = reader.wait_for_all(deadline());
        require(!results.is_terminated && !results.error, "score and completion tasks failed");
        compare_scores(results.results[score_first ? 0 : 1].get(), expected.back());
        const auto * completion_result = dynamic_cast<server_task_result_cmpl_final *>(results.results[score_first ? 1 : 0].get());
        require(completion_result && completion_result->n_decoded > 0, "completion was interrupted by scoring");
    }
    {
        // The last completion leaves a nearly full idle prompt in shared KV.
        auto reader = server.get_response_reader();
        reader.post_task(make_task(reader, SERVER_TASK_TYPE_METRICS, {}));
        auto result = reader.next(deadline());
        const auto * metrics = dynamic_cast<server_task_result_metrics *>(result.get());
        require(metrics && metrics->n_processing_slots == 0, "completion did not release its slot");
        bool cached = false;
        for (const auto & slot : metrics->slots_data) {
            cached = cached || slot.value("n_prompt_tokens", 0) >= n_ctx - 8;
        }
        require(cached, "test did not leave an idle prompt in shared KV");
    }
    {
        auto reader = server.get_response_reader();
        reader.post_task(make_task(reader, SERVER_TASK_TYPE_SCORE, windows.back()));
        auto result = reader.next(deadline());
        compare_scores(result.get(), expected.back());
    }
    {
        auto reader = server.get_response_reader();
        reader.post_task(make_task(reader, SERVER_TASK_TYPE_SCORE, windows.back()));
        reader.stop();
    }
    {
        // A score after completion and cancellation must use fresh prompt logits.
        auto reader = server.get_response_reader();
        reader.post_task(make_task(reader, SERVER_TASK_TYPE_SCORE, windows[3]));
        auto result = reader.next(deadline());
        compare_scores(result.get(), expected[3]);
    }
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--queue-only") {
        try {
            test_queue_activity();
            std::printf("SERVER_QUEUE_ACTIVITY_TEST PASS\n");
            return 0;
        } catch (const std::exception & error) {
            std::fprintf(stderr, "SERVER_QUEUE_ACTIVITY_TEST FAIL: %s\n", error.what());
            return 1;
        }
    }
    const char * model_path = argc > 1 ? argv[1] : std::getenv("LLAMA_TEST_MODEL");
    if (!model_path || !*model_path) {
        std::fprintf(stderr, "test-server-score: skipped; set LLAMA_TEST_MODEL or supply a decoder GGUF\n");
        return 77;
    }
    common_init();
    llama_backend_init();
    int status = 0;
    try {
        test_scores(model_path);
        std::printf("SERVER_SCORE_TEST PASS\n");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "SERVER_SCORE_TEST FAIL: %s\n", error.what());
        status = 1;
    }
    llama_backend_free();
    return status;
}
