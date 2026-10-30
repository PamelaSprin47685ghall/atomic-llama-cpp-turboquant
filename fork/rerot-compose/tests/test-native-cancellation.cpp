#include "native.h"

#include <ggml-backend.h>
#include <tq/rerot/episode.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool valid, const char * message) {
    if (!valid) { throw std::runtime_error(message); }
}

template <typename Operation>
void failure(Operation operation, const std::string & expected) {
    try { operation(); }
    catch (const std::exception & error) {
        require(error.what() == expected, error.what());
        return;
    }
    throw std::runtime_error("native execution did not report failure");
}

class backend_owner {
public:
    backend_owner() { ggml_backend_load_all(); llama_backend_init(); }
    ~backend_owner() { llama_backend_free(); }
};

void compute_case(const tq::rerot_compose::options & config, bool throw_query) {
    bool armed = false;
    std::uint32_t polls = 0;
    tq::rerot_compose::native_execution native(config, 3, 1, [&]() {
        if (!armed) { return false; }
        if (++polls < 3) { return false; }
        if (throw_query) { throw std::runtime_error("cancel query failed"); }
        return true;
    });
    // Two C checks run before submission. The third query runs inside the
    // native CPU graph. The ordinary abort case must report A's status 2.
    const std::vector<llama_token> input(32, 0);
    armed = true;
    const std::string expected = throw_query ? "cancel query failed" : "cancelled during native execution";
    failure([&] { native.prefill(input); }, expected);
    require(native.drained(), "cancelled native work was not drained");
    require(polls == 3, "cancellation did not remain sticky");
    const std::string retry = throw_query ? expected : "cancelled before base prefill";
    failure([&] { native.prefill(input); }, retry);
    require(native.drained() && polls == 3, "a cancelled executor submitted more work");
}

void partial_frame_case(const tq::rerot_compose::options & config) {
    tq::rerot_compose::native_execution native(config, 3, 1);
    native.prefill({0});
    native.start(1);
    tq::rerot::Episode episode(tq::rerot::Plan({{"worker", "work"}}), {}, {{1, 0, 1}});
    const auto step = episode.begin_step();
    // B allows sparse int64 storage positions. The first view is valid for A;
    // only the next FRAME wave exceeds A's int32 position domain.
    episode.prepare(step, 1, {
        {2, tq::rerot::SegmentKind::frame, tq::rerot::Visibility::public_kv, 1, 1},
        {3, tq::rerot::SegmentKind::frame, tq::rerot::Visibility::public_kv, std::int64_t(INT32_MAX) + 1, 1}});
    const std::vector<tq::rerot_compose::input_row> rows = {
        {1, 1, 0, true, episode.view(step, 1, {2, 0})},
        {1, 2, 0, true, episode.view(step, 1, {3, 0})}};
    failure([&] { static_cast<void>(native.forward(step.frontier(), rows)); },
            "reader position exceeds A's position domain");
    require(native.drained(), "a rejected later FRAME wave left earlier native work undrained");
    failure([&] { native.start(2); }, "invalid native stage start");
    failure([&] { static_cast<void>(native.forward(step.frontier(), {rows.front()})); }, "invalid native frontier");
    require(native.drained(), "a failed FRAME executor submitted more work");
    episode.abort(step);
    episode.confirm_drained(step);
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 2) { throw std::invalid_argument("usage: test-native-cancellation MODEL.gguf"); }
        backend_owner backend;
        tq::rerot_compose::options config;
        config.model = argv[1];
        config.context = 256;
        config.batch = 64;
        config.ubatch = 32;
        config.threads = 2;

        failure([&] { tq::rerot_compose::native_execution invalid(config, 3, UINT32_MAX); },
                "task graph exceeds A's logical sequence capacity");
        std::uint32_t loading_polls = 0;
        failure([&] {
            tq::rerot_compose::native_execution loading(config, 3, 1, [&] { return ++loading_polls >= 2; });
        }, "cancelled during model loading");
        require(loading_polls == 2, "model loading did not retain its cancellation");
        compute_case(config, false);
        compute_case(config, true);
        partial_frame_case(config);
        std::cout << "PASS: pen bounds, real model-load cancellation, CPU decode abort, drain, sticky stop,"
                     " callback exception transfer, and partial FRAME failure\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
