// Test native scoring blocks and source edits with a real GGUF tokenizer.
// Injected losses test ranking rules, not model quality. No inference is needed.

#include "server-compress.h"

#include "common.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
using server_compress_json::deletion_result;

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static bool is_subsequence(const std::string & shorter, const std::string & original) {
    size_t next = 0;
    for (char byte : original) {
        if (next < shorter.size() && shorter[next] == byte) {
            ++next;
        }
    }
    return next == shorter.size();
}

static std::string check_provenance(const deletion_result & result, const std::string & original) {
    size_t end = 0;
    for (const auto & marker : result.markers) {
        require(marker.begin >= end && marker.end <= result.text.size(), "marker provenance is out of order");
        require(result.text.substr(marker.begin, marker.end - marker.begin) == "␡", "marker provenance has wrong bytes");
        end = marker.end;
    }
    const auto retained = server_compress_json::erase_ranges(result.text, result.markers);
    require(is_subsequence(retained, original), "source bytes changed outside inserted markers");
    server_compress_json::parse(result.text); // Strict Unicode and JSON, independent of nlohmann.
    return retained;
}

static std::string source(const json & body) {
    return " \n" + body.dump(2) + "\t\n";
}

static json call(const char * id, const char * name, const std::string & arguments) {
    return {{"id", id}, {"type", "function"}, {"function", {{"name", name}, {"arguments", arguments}}}};
}

static void set_scores(server_compress_plan & plan, size_t index, float forward, float reverse) {
    const size_t size = plan.texts().at(index).tokens.size();
    plan.add_scores(index, 0, std::vector<float>(size, forward), false);
    plan.add_scores(index, 0, std::vector<float>(size, reverse), true);
}

static void test_native_blocks(const llama_vocab * vocab) {
    std::string long_item;
    for (size_t i = 0; i < 256; ++i) {
        long_item += " a small source item remains one block.";
    }
    const std::string arguments = "{ \"path\": \"src/a.cpp\", \"left\": \"first value\", \"right\": \"second value\", \"count\": 7 }";
    const json body = {{"messages", json::array({
        {{"role", "system"}, {"content", "System context only."}},
        {{"role", "developer"}, {"content", "Developer context only."}},
        {{"role", "user"}, {"content", json::array({
            {{"type", "text"}, {"text", long_item}},
            {{"type", "text"}, {"text", "Second user item."}},
        })}},
        // Place reasoning last in JSON to test logical order, not key order.
        {{"role", "assistant"}, {"tool_calls", json::array({call("one", "write", arguments)})},
         {"content", json::array({
             {{"type", "text"}, {"text", "First visible item."}},
             {{"type", "text"}, {"text", "Second visible item."}},
         })}, {"reasoning_content", "Reasoning comes first."}},
        {{"role", "tool"}, {"tool_call_id", "one"}, {"content", "Tool result."}},
    })}};
    const auto raw = source(body);
    server_compress_plan plan(raw, vocab);
    const std::vector<std::string> expected = {
        "System context only.", "Developer context only.", long_item, "Second user item.",
        "Reasoning comes first.", "First visible item.", "Second visible item.",
        "write\n" + arguments, "Tool result.",
    };
    const std::vector<size_t> messages = {0, 1, 2, 2, 3, 3, 3, 3, 4};
    require(plan.texts().size() == expected.size(), "native items were split or combined into synthetic blocks");
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto & block = plan.texts()[i];
        require(block.tokens == common_tokenize(vocab, expected[i], false, false), "native block token order differs");
        require(block.message == messages[i], "native block lost its source message");
        require(block.token_units.size() == block.tokens.size(), "token/source map has wrong size");
        require(block.forward_nll.size() == block.tokens.size() && block.reverse_nll.size() == block.tokens.size(),
                "two score buffers must cover the same native block");
    }
    require(plan.texts()[2].tokens.size() > 512, "long native-item fixture is too short");
    for (size_t i : {size_t(0), size_t(1)}) {
        const auto & units = plan.texts()[i].token_units;
        require(std::all_of(units.begin(), units.end(), [](size_t value) {
            return value == std::numeric_limits<size_t>::max();
        }), "instruction blocks must provide context without editable units");
    }
    const auto unchanged = plan.render();
    require(unchanged.text == raw && unchanged.markers.empty(), "unmodified plan changed original bytes");
}

static void test_single_large_block(const llama_vocab * vocab) {
    const std::string prefix = "User ␡ prefix.";
    const std::string suffix = " Final instruction.";
    std::string content = prefix;
    for (size_t i = 0; i < 2048; ++i) {
        content += " a";
    }
    content += suffix;
    const json body = {{"messages", json::array({{{"role", "user"}, {"content", content}}})}};
    const auto raw = source(body);
    server_compress_plan plan(raw, vocab);
    require(plan.texts().size() == 1 && plan.texts()[0].tokens.size() > 512, "one large field must remain one scoring block");
    set_scores(plan, 0, 5.0f, 5.0f);
    plan.rank();
    require(plan.remove_tokens(1), "a large native block must support internal deletion");
    const auto first = plan.render();
    const auto retained = check_provenance(first, raw);
    require(first.markers.size() == 1, "one small deletion must have one marker");
    require(retained.size() < raw.size() && raw.size() - retained.size() < 32,
            "internal deletion still behaves like a fixed-size chunk");
    require(plan.removed_messages() == 0, "the current user message was removed");
    const auto minimum = plan.minimum();
    check_provenance(minimum, raw);
    const auto after = json::parse(minimum.text);
    const auto text = after["messages"][0]["content"].get<std::string>();
    require(text.size() < content.size() && minimum.markers.size() == 1, "adjacent internal deletions did not coalesce");
    require(text.compare(0, prefix.size(), prefix) == 0 &&
            text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0,
            "protected edges or the original user marker changed");
}

static void test_nested_arguments(const llama_vocab * vocab) {
    std::string code = "ORIGINAL-␡-BEGIN\n";
    std::string note = "NOTE-BEGIN ";
    for (size_t i = 0; i < 192; ++i) {
        code += "const char * value = \"中文 😀 \\\\ cache\";\n";
        note += "The previous value did not change. ";
    }
    code += "CONTENT-END";
    note += "NOTE-END";
    const json values = {{"path", "src/a.cpp"}, {"content", code}, {"mode", 384},
                         {"options", {{"note", note}, {"enabled", true}, {"threshold", 1.5}, {"tags", {"a", "b"}}}}};
    const std::string arguments = values.dump(2, ' ', true);
    const auto tool = call("write_latest", "write", arguments);
    const json body = {{"stream", false}, {"extra", {{"keep", "user ␡ metadata"}}},
                      {"messages", json::array({
        {{"role", "system"}, {"content", "Keep the source contract. ␡"}},
        {{"role", "user"}, {"content", "Write the file and inspect the result."}},
        {{"role", "assistant"}, {"content", nullptr}, {"tool_calls", json::array({tool})}},
        {{"role", "tool"}, {"tool_call_id", "write_latest"}, {"content", "Done."}},
    })}};
    const auto raw = source(body);
    server_compress_plan plan(raw, vocab);
    require(plan.texts().size() == 4, "one tool call with several argument leaves must remain one native block");
    require(plan.texts()[2].tokens == common_tokenize(vocab, "write\n" + arguments, false, false),
            "tool block must include the name and complete serialized argument JSON");
    const auto result = plan.minimum();
    check_provenance(result, raw);
    const auto after = json::parse(result.text);
    require(after["messages"].size() == 4 && after["messages"][0] == body["messages"][0], "protected message shells changed");
    require(after["stream"] == body["stream"] && after["extra"] == body["extra"], "unknown top-level fields changed");
    require(after["messages"][3] == body["messages"][3], "short tool result changed");
    const auto new_arguments = after["messages"][2]["tool_calls"][0]["function"]["arguments"].get<std::string>();
    server_compress_json::parse(new_arguments);
    const auto edited = json::parse(new_arguments);
    require(edited.size() == values.size() && edited["options"].size() == values["options"].size(), "argument keys were removed");
    require(edited["path"] == values["path"] && edited["mode"] == values["mode"], "short path or numeric argument changed");
    for (const auto * key : {"enabled", "threshold", "tags"}) {
        require(edited["options"][key] == values["options"][key], "non-text argument structure changed");
    }
    const auto new_code = edited["content"].get<std::string>();
    require(new_code.size() < code.size() && edited["options"]["note"].get<std::string>().size() < note.size(),
            "large argument leaves were not internally shortened");
    require(new_code.find("ORIGINAL-␡-BEGIN") == 0, "the original marker in nested arguments was changed");
    require(result.markers.size() >= 2, "distinct long argument values must mark their own deletions");
    auto expected_call = tool;
    expected_call["function"]["arguments"] = new_arguments;
    require(after["messages"][2]["tool_calls"][0] == expected_call, "tool call metadata changed");
}

static std::vector<size_t> interior_units(const server_compress_plan::text & field, size_t position) {
    const size_t none = std::numeric_limits<size_t>::max();
    std::vector<size_t> units;
    for (size_t i = position; i + 64 < field.tokens.size(); ++i) {
        if (i >= 64 && field.token_units[i] != none &&
                (units.empty() || units.back() != field.token_units[i])) {
            units.push_back(field.token_units[i]);
            // A continuous low-score region can pay for a marker before the
            // one-token net saving is reached. This is a test fixture, not a
            // partition of the native scoring block.
            if (units.size() == 16) {
                return units;
            }
        }
    }
    throw std::runtime_error("tokenizer fixture has no interior source region");
}

static std::string ranked_edit(const std::string & raw, const llama_vocab * vocab,
                               const std::vector<size_t> & first, const std::vector<size_t> & second,
                               float first_f, float first_r,
                               float second_f, float second_r) {
    server_compress_plan plan(raw, vocab);
    const auto & block = plan.texts().at(0);
    std::vector<float> forward(block.tokens.size(), 100.0f);
    std::vector<float> reverse(block.tokens.size(), 0.0f);
    for (size_t i = 0; i < block.tokens.size(); ++i) {
        if (std::find(first.begin(), first.end(), block.token_units[i]) != first.end()) {
            forward[i] = first_f;
            reverse[i] = first_r;
        } else if (std::find(second.begin(), second.end(), block.token_units[i]) != second.end()) {
            forward[i] = second_f;
            reverse[i] = second_r;
        }
    }
    // Offset calls exercise score attachment as used by overlapping score windows.
    const size_t middle = block.tokens.size() / 2;
    for (bool backwards : {false, true}) {
        const auto & values = backwards ? reverse : forward;
        plan.add_scores(0, middle, std::vector<float>(values.begin() + middle, values.end()), backwards);
        plan.add_scores(0, 0, std::vector<float>(values.begin(), values.begin() + middle), backwards);
    }
    plan.rank();
    require(plan.remove_tokens(1), "ranked fixture did not remove a source region");
    const auto result = plan.render();
    check_provenance(result, raw);
    require(result.markers.size() == 1, "one continuous ranked region must produce one marker");
    return result.text;
}

static void test_delta_and_tie_break(const llama_vocab * vocab) {
    std::string content;
    for (size_t i = 0; i < 512; ++i) {
        content += " a";
    }
    const auto raw = source({{"messages", json::array({{{"role", "user"}, {"content", json::array({
        {{"type", "text"}, {"text", content}},
        {{"type", "text"}, {"text", "A second native context item."}},
    })}}})}});
    server_compress_plan plan(raw, vocab);
    require(plan.texts().size() == 2, "delta ranking fixture must have two native blocks");
    const auto & field = plan.texts()[0];
    const auto first = interior_units(field, field.tokens.size() / 3);
    const auto second = interior_units(field, 2 * field.tokens.size() / 3);
    require(first.back() < second.front(), "ranking fixtures selected overlapping source regions");
    const auto first_only = ranked_edit(raw, vocab, first, second, 0, 100, 100, 0);
    const auto second_only = ranked_edit(raw, vocab, first, second, 100, 0, 0, 100);
    require(first_only != second_only, "controlled scores did not select distinct source positions");
    require(ranked_edit(raw, vocab, first, second, 1, 1, 50, 60) == second_only,
            "forward-minus-reverse delta must rank before forward surprisal");
    require(ranked_edit(raw, vocab, first, second, 5, 5, 2, 2) == second_only,
            "equal deltas must use forward surprisal as the tie-break");
    require(ranked_edit(raw, vocab, first, second, 2, 2, 2, 2) == first_only,
            "equal score pairs must keep deterministic source order");
}

static void test_single_block_forward_fallback(const llama_vocab * vocab) {
    std::string content;
    for (size_t i = 0; i < 512; ++i) {
        content += " a";
    }
    const auto raw = source({{"messages", json::array({{{"role", "user"}, {"content", content}}})}});
    server_compress_plan plan(raw, vocab);
    require(plan.texts().size() == 1, "single-block fallback fixture must have one native block");
    const auto & field = plan.texts()[0];
    const auto first = interior_units(field, field.tokens.size() / 3);
    const auto second = interior_units(field, 2 * field.tokens.size() / 3);
    const auto first_only = ranked_edit(raw, vocab, first, second, 0, 100, 100, 0);
    require(ranked_edit(raw, vocab, first, second, 1, 1, 2, 2.0001f) == first_only,
            "reverse rounding noise must not override forward ranking for a single native block");
}

static void test_tool_group_ranking(const llama_vocab * vocab) {
    json messages = json::array({{{"role", "system"}, {"content", "Protected ␡ context."}},
                                {{"role", "user"}, {"content", "Inspect these files."}}});
    for (const auto * id : {"first", "second", "latest"}) {
        messages.push_back({{"role", "assistant"}, {"content", nullptr},
                            {"tool_calls", json::array({call(id, "read", "{\"path\":\"a.txt\"}")})}});
        messages.push_back({{"role", "tool"}, {"tool_call_id", id}, {"content", "unchanged"}});
    }
    const auto raw = source({{"messages", messages}});
    server_compress_plan plan(raw, vocab);
    for (size_t i = 0; i < plan.texts().size(); ++i) {
        const size_t message = plan.texts()[i].message;
        const bool second = message == 4 || message == 5;
        set_scores(plan, i, second ? 20.0f : 1.0f, second ? 40.0f : 1.0f);
    }
    plan.rank();
    require(plan.remove_tokens(1), "old tool exchange could not be removed");
    const auto result = plan.render();
    check_provenance(result, raw);
    require(result.markers.empty() && plan.removed_messages() == 2, "tool exchange deletion must be structural and atomic");
    json expected = json::array();
    for (size_t i = 0; i < messages.size(); ++i) {
        if (i != 4 && i != 5) {
            expected.push_back(messages[i]);
        }
    }
    require(json::parse(result.text)["messages"] == expected,
            "delta ranking removed the wrong exchange or changed retained messages");
}

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : std::getenv("LLAMA_TEST_MODEL");
    if (!model_path || !*model_path) {
        std::fprintf(stderr, "test-server-compress: skipped; set LLAMA_TEST_MODEL or supply a GGUF\n");
        return 77;
    }
    llama_backend_init();
    int status = 0;
    try {
        auto params = llama_model_default_params();
        params.vocab_only = true;
        params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_path, params), llama_model_free);
        require(model != nullptr, "cannot load the test model tokenizer");
        const auto * vocab = llama_model_get_vocab(model.get());
        test_native_blocks(vocab);
        test_single_large_block(vocab);
        test_nested_arguments(vocab);
        test_delta_and_tie_break(vocab);
        test_single_block_forward_fallback(vocab);
        test_tool_group_ranking(vocab);
        std::printf("SERVER_COMPRESS_PLAN_TEST PASS\n");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "SERVER_COMPRESS_PLAN_TEST FAIL: %s\n", error.what());
        status = 1;
    }
    llama_backend_free();
    return status;
}
