// Compare queued prompt scores with independent, single-token model evaluations.
// Run with a small decoder GGUF as argv[1] or LLAMA_TEST_MODEL.

#include "server-context.h"
#include "server-compress-stream.h"
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

using stream_text = server_compress_plan::text;
using stream_losses = std::vector<std::vector<float>>;

static stream_text make_text(llama_tokens tokens) {
    stream_text text;
    text.tokens = std::move(tokens);
    return text;
}

static void compare_losses(const std::vector<float> & actual, const std::vector<float> & expected) {
    require(actual.size() == expected.size(), "mapped stream score count is wrong");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::isnan(expected[i])) {
            require(std::isnan(actual[i]), "a stream token without a predecessor must have no score");
        } else {
            const float tolerance = 2e-3f + 1e-4f * std::abs(expected[i]);
            require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) <= tolerance,
                    "mapped stream score differs from the causal reference");
        }
    }
}

static stream_losses reference_stream(llama_context * ctx, const std::vector<stream_text> & texts,
        const llama_tokens & separator, llama_token bos, bool reverse) {
    // Evaluate the complete short stream one token at a time. This reference
    // does not use the production window helper or reset at text boundaries.
    llama_tokens stream;
    if (bos != LLAMA_TOKEN_NULL) {
        stream.push_back(bos);
    }
    std::vector<size_t> starts(texts.size(), 0);
    bool first = true;
    for (size_t i = 0; i < texts.size(); ++i) {
        const size_t index = reverse ? texts.size() - i - 1 : i;
        const auto & tokens = texts[index].tokens;
        if (tokens.empty()) {
            continue;
        }
        if (!first) {
            stream.insert(stream.end(), separator.begin(), separator.end());
        }
        first = false;
        starts[index] = stream.size();
        stream.insert(stream.end(), tokens.begin(), tokens.end());
    }
    require(stream.size() <= llama_n_ctx(ctx), "reference stream exceeds the test context");
    const auto losses = reference_scores(ctx, stream);
    stream_losses result;
    for (size_t i = 0; i < texts.size(); ++i) {
        result.emplace_back(losses.begin() + starts[i], losses.begin() + starts[i] + texts[i].tokens.size());
    }
    return result;
}

static stream_losses score_stream(server_context & server, const std::vector<stream_text> & texts,
        const llama_tokens & separator, size_t capacity, size_t overlap, llama_token bos, bool reverse) {
    server_compress_stream stream(texts, separator, capacity, overlap, bos, reverse);
    server_compress_window window;
    stream_losses result;
    std::vector<size_t> received(texts.size(), 0);
    for (const auto & text : texts) {
        result.emplace_back(text.tokens.size(), std::numeric_limits<float>::quiet_NaN());
    }
    while (stream.next(window)) {
        require(window.tokens.size() <= capacity, "stream window exceeds inference capacity");
        require(window.score_offset < window.tokens.size(), "stream window has no target tokens");
        require(window.begin == received.at(window.text_index), "stream targets skipped or repeated source tokens");
        const size_t count = window.tokens.size() - window.score_offset;
        const auto & original = texts.at(window.text_index).tokens;
        require(window.begin + count <= original.size(), "stream target leaves its natural text block");
        require(std::equal(window.tokens.begin() + window.score_offset, window.tokens.end(), original.begin() + window.begin),
                "stream changed the token order inside a natural block");

        auto reader = server.get_response_reader();
        reader.post_task(make_task(reader, SERVER_TASK_TYPE_SCORE, window.tokens));
        auto raw = reader.next(deadline());
        require(raw && !raw->is_error(), "stream score task failed");
        const auto * score = dynamic_cast<server_task_result_score *>(raw.get());
        require(score && score->nll.size() == window.tokens.size(), "stream score result has the wrong shape");
        std::copy(score->nll.begin() + window.score_offset, score->nll.end(), result[window.text_index].begin() + window.begin);
        received[window.text_index] += count;
    }
    for (size_t i = 0; i < texts.size(); ++i) {
        require(received[i] == texts[i].tokens.size(), "stream omitted a natural text block");
    }
    return result;
}

static std::vector<server_compress_window> stream_windows(const std::vector<stream_text> & texts,
        const llama_tokens & separator, size_t capacity, size_t overlap, llama_token bos, bool reverse) {
    server_compress_stream stream(texts, separator, capacity, overlap, bos, reverse);
    std::vector<server_compress_window> result;
    server_compress_window window;
    while (stream.next(window)) {
        result.push_back(window);
    }
    return result;
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

    const llama_tokens separator = common_tokenize(vocab, "\n\n", false, false);
    const llama_token bos = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : LLAMA_TOKEN_NULL;
    const auto a = common_tokenize(vocab, "The little fox found a red book. The book had many stories.", false, false);
    const auto b = common_tokenize(vocab, "The red book belonged to the little fox. He read it every day.", false, false);
    const auto alternative = common_tokenize(vocab, "A big bear played with a blue ball beside a lake.", false, false);
    require(!a.empty() && !b.empty() && !alternative.empty(), "stream test tokenization failed");
    std::vector<stream_text> texts = {make_text(a), make_text({}), make_text(b)};
    auto changed = texts;
    for (size_t i = 0; i < b.size(); ++i) {
        changed[2].tokens[i] = alternative[i % alternative.size()];
    }
    require(changed[2].tokens != texts[2].tokens, "context fixture did not change its tokens");
    const auto forward_reference = reference_stream(ctx, texts, separator, bos, false);
    const auto reverse_reference = reference_stream(ctx, texts, separator, bos, true);
    const auto changed_forward_reference = reference_stream(ctx, changed, separator, bos, false);
    const auto changed_reverse_reference = reference_stream(ctx, changed, separator, bos, true);

    running_server loop(server);
    {
        const auto forward = score_stream(server, texts, separator, n_ctx, 32, bos, false);
        const auto reverse = score_stream(server, texts, separator, n_ctx, 32, bos, true);
        const auto changed_forward = score_stream(server, changed, separator, n_ctx, 32, bos, false);
        const auto changed_reverse = score_stream(server, changed, separator, n_ctx, 32, bos, true);
        for (size_t i = 0; i < texts.size(); ++i) {
            compare_losses(forward[i], forward_reference[i]);
            compare_losses(reverse[i], reverse_reference[i]);
            compare_losses(changed_forward[i], changed_forward_reference[i]);
            compare_losses(changed_reverse[i], changed_reverse_reference[i]);
        }
        // A is unchanged. B can affect A only when block reversal puts B first.
        // Do not assume the small fixture model understands any semantic ranking.
        compare_losses(forward[0], changed_forward[0]);
        float context_difference = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::isfinite(reverse[0][i]) && std::isfinite(changed_reverse[0][i])) {
                context_difference = std::max(context_difference, std::abs(reverse[0][i] - changed_reverse[0][i]));
            }
        }
        require(context_difference > 1e-4f, "reversed scoring did not use the preceding natural block");
        std::printf("SERVER_COMPRESS_STREAM_CONTEXT_DELTA %.6f\n", context_difference);
    }
    {
        const std::vector<stream_text> single = {make_text(windows[5])};
        const auto forward_windows = stream_windows(single, separator, 32, 8, bos, false);
        const auto reverse_windows = stream_windows(single, separator, 32, 8, bos, true);
        require(forward_windows.size() > 1 && forward_windows.size() == reverse_windows.size(),
                "long block did not cross inference windows");
        for (size_t i = 0; i < forward_windows.size(); ++i) {
            require(forward_windows[i].tokens == reverse_windows[i].tokens &&
                    forward_windows[i].begin == reverse_windows[i].begin &&
                    forward_windows[i].score_offset == reverse_windows[i].score_offset,
                    "reversing one natural block reversed or realigned its capacity chunks");
        }
        const auto forward = score_stream(server, single, separator, 32, 8, bos, false);
        const auto reverse = score_stream(server, single, separator, 32, 8, bos, true);
        compare_losses(forward[0], reverse[0]);

        // Different neighbor lengths must not change the local window phase.
        // Once a window's prefix lies wholly inside the long middle block,
        // forward and reverse receive exactly the same model input.
        const std::vector<stream_text> surrounded = {
            make_text(llama_tokens(tokens.begin(), tokens.begin() + 3)),
            make_text(windows[5]),
            make_text(llama_tokens(tokens.begin() + 17, tokens.begin() + 34)),
        };
        const auto surrounded_forward = stream_windows(surrounded, separator, 32, 8, bos, false);
        const auto surrounded_reverse = stream_windows(surrounded, separator, 32, 8, bos, true);
        size_t checked = 0;
        for (const auto & window : surrounded_forward) {
            if (window.text_index != 1 || window.begin < 32) {
                continue;
            }
            const auto match = std::find_if(surrounded_reverse.begin(), surrounded_reverse.end(), [&](const server_compress_window & other) {
                return other.text_index == 1 && other.begin == window.begin;
            });
            require(match != surrounded_reverse.end() && match->tokens == window.tokens && match->score_offset == window.score_offset,
                    "window phase created a false forward/reverse contrast inside a long block");
            ++checked;
        }
        require(checked != 0, "phase test did not reach the interior of its long block");
        const auto surrounded_forward_scores = score_stream(server, surrounded, separator, 32, 8, bos, false);
        const auto surrounded_reverse_scores = score_stream(server, surrounded, separator, 32, 8, bos, true);
        compare_losses(
            std::vector<float>(surrounded_forward_scores[1].begin() + 32, surrounded_forward_scores[1].end()),
            std::vector<float>(surrounded_reverse_scores[1].begin() + 32, surrounded_reverse_scores[1].end()));
    }
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
