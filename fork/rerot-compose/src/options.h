#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tq::rerot_compose {

struct task_input {
    std::string id;
    std::string instruction;
};

struct options {
    std::string model;
    std::string prompt;
    std::string chat_template;
    std::string synthesis = "Use the preceding work to answer the original question.";
    std::vector<task_input> tasks;
    std::vector<std::pair<std::string, std::string>> edges;
    std::uint32_t parallel = 4;
    std::uint32_t context = 8192;
    std::uint32_t total_kv = 0;
    std::uint32_t batch = 512;
    std::uint32_t ubatch = 128;
    std::uint32_t threads = 4;
    std::uint32_t gpu_layers = 0;
    std::uint32_t worker_tokens = 256;
    std::uint32_t final_tokens = 256;
    std::uint32_t seed = 0;
    std::uint32_t top_k = 40;
    float temperature = 0.0f;
    std::string split = "layer";
    std::string devices;
    bool phase_memory = true;
    bool json = false;
    bool help = false;
};

inline std::uint32_t integer(std::string_view value, const std::string & name,
                             bool zero = false, std::uint32_t limit = INT32_MAX) {
    std::uint32_t number = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
    if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
        (!zero && number == 0) || number > limit) {
        throw std::invalid_argument(name + " has an invalid unsigned decimal value");
    }
    return number;
}

inline float real(std::string_view value, const std::string & name) {
    float number = 0.0f;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
    if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
        !std::isfinite(number) || number < 0.0f) {
        throw std::invalid_argument(name + " requires a finite non-negative number");
    }
    return number;
}

inline bool valid_id(std::string_view text) {
    if (text.empty()) { return false; }
    for (const unsigned char c : text) {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.')) {
            return false;
        }
    }
    return true;
}

inline std::pair<std::string, std::string> assignment(const std::string & value,
                                                       const std::string & name) {
    const auto split = value.find('=');
    if (split == std::string::npos || split == 0 || split + 1 == value.size()) {
        throw std::invalid_argument(name + " requires NAME=VALUE");
    }
    const auto id = value.substr(0, split);
    if (!valid_id(id)) { throw std::invalid_argument(name + " has an invalid stage name"); }
    return {id, value.substr(split + 1)};
}

inline options parse(int argc, char ** argv) {
    options result;
    std::unordered_set<std::string> seen;
    std::unordered_set<std::string> ids;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--help") { result.help = true; continue; }
        if (arg == "--json") { result.json = true; continue; }
        if (arg == "--phase-memory" || arg == "--no-phase-memory") {
            if (!seen.insert("phase-memory").second) {
                throw std::invalid_argument("phase-memory was specified more than once");
            }
            result.phase_memory = arg == "--phase-memory";
            continue;
        }
        if (arg == "--mtp" || arg == "--latebind" || arg == "--tp5") {
            throw std::invalid_argument(arg + " requires an execution provider that is not installed in this program");
        }
        const std::unordered_set<std::string> allowed = {
            "--model", "--prompt", "--stage", "--after", "--synthesis", "--chat-template",
            "--parallel", "--ctx-size", "--total-kv", "--batch-size", "--ubatch-size",
            "--threads", "--gpu-layers", "--worker-tokens", "--final-tokens", "--seed",
            "--temperature", "--top-k", "--split-mode", "--devices"};
        if (!allowed.count(arg)) { throw std::invalid_argument("unknown option: " + arg); }
        if (arg != "--stage" && arg != "--after" && !seen.insert(arg).second) {
            throw std::invalid_argument("option was specified more than once: " + arg);
        }
        if (++index == argc) { throw std::invalid_argument("missing value for " + arg); }
        const std::string value = argv[index];
        if (arg == "--stage") {
            auto entry = assignment(value, arg);
            if (!ids.insert(entry.first).second) { throw std::invalid_argument("duplicate stage: " + entry.first); }
            result.tasks.push_back({std::move(entry.first), std::move(entry.second)});
        } else if (arg == "--after") {
            auto entry = assignment(value, arg);
            if (!valid_id(entry.second)) { throw std::invalid_argument("--after requires CHILD=PARENT"); }
            result.edges.emplace_back(std::move(entry.second), std::move(entry.first));
        } else if (arg == "--model") { result.model = value; }
        else if (arg == "--prompt") { result.prompt = value; }
        else if (arg == "--synthesis") { result.synthesis = value; }
        else if (arg == "--chat-template") { result.chat_template = value; }
        else if (arg == "--devices") { result.devices = value; }
        else if (arg == "--split-mode") { result.split = value; }
        else if (arg == "--parallel") { result.parallel = integer(value, arg); }
        else if (arg == "--ctx-size") { result.context = integer(value, arg); }
        else if (arg == "--total-kv") { result.total_kv = integer(value, arg); }
        else if (arg == "--batch-size") { result.batch = integer(value, arg); }
        else if (arg == "--ubatch-size") { result.ubatch = integer(value, arg); }
        else if (arg == "--threads") { result.threads = integer(value, arg); }
        else if (arg == "--gpu-layers") { result.gpu_layers = integer(value, arg, true); }
        else if (arg == "--worker-tokens") { result.worker_tokens = integer(value, arg); }
        else if (arg == "--final-tokens") { result.final_tokens = integer(value, arg); }
        else if (arg == "--seed") { result.seed = integer(value, arg, true, UINT32_MAX - 1); }
        else if (arg == "--top-k") { result.top_k = integer(value, arg); }
        else { result.temperature = real(value, arg); }
    }
    if (result.help) { return result; }
    if (result.model.empty() || result.prompt.empty() || result.tasks.empty()) {
        throw std::invalid_argument("--model, --prompt, and at least one --stage are required");
    }
    if (result.synthesis.empty()) { throw std::invalid_argument("--synthesis must not be empty"); }
    if (result.ubatch > result.batch) { throw std::invalid_argument("--ubatch-size exceeds --batch-size"); }
    if (result.split != "none" && result.split != "layer" && result.split != "row") {
        throw std::invalid_argument("--split-mode requires none, layer, or row");
    }
    for (const auto & edge : result.edges) {
        if (!ids.count(edge.first) || !ids.count(edge.second)) {
            throw std::invalid_argument("--after names an unknown stage");
        }
    }
    return result;
}

} // namespace tq::rerot_compose
