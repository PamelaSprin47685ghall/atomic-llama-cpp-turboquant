#include <tq/scheduling/prefill.h>

#include <iostream>
#include <stdexcept>
#include <string>

using tq::scheduling::prefill_selector;
using tq::scheduling::selection;
using tq::scheduling::slot_state;
using tq::scheduling::stage;

static void check(bool value, const char * message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

static void prompt_is(const selection & selected, std::uint32_t id) {
    check(selected.next == stage::prompt, "expected prompt phase");
    check(selected.prompt_slot == id, "wrong prompt slot");
    check(selected.generation_slots.empty(), "generation must pause during prompt");
}

static void test_sticky_and_slot_order() {
    prefill_selector selector;
    prompt_is(selector.choose({{8, stage::prompt}, {2, stage::prompt}, {4, stage::generation}}), 8);
    prompt_is(selector.choose({{1, stage::prompt}, {8, stage::prompt}, {2, stage::prompt}}), 8);
    prompt_is(selector.choose({{1, stage::prompt}, {8, stage::generation}, {2, stage::prompt}}), 1);
    prompt_is(selector.choose({{8, stage::generation}, {2, stage::prompt}}), 2);
    const auto decode = selector.choose({{8, stage::generation}, {2, stage::generation}, {1, stage::idle}});
    check(decode.next == stage::generation && !decode.prompt_slot, "expected generation phase");
    check(decode.generation_slots == std::vector<std::uint32_t>({8, 2}), "generation order changed");
    const auto empty = selector.choose({});
    check(empty.next == stage::generation && empty.generation_slots.empty(), "empty input must select no work");
}

static void test_ownership_and_instances() {
    prefill_selector first;
    prefill_selector second;
    std::vector<slot_state> input{{9, stage::prompt}};
    auto result = first.choose(input);
    input.front().id = 2;
    result.prompt_slot = 88;
    const std::vector<slot_state> both{{2, stage::prompt}, {9, stage::prompt}};
    prompt_is(first.choose(both), 9);
    prompt_is(second.choose(both), 2);
    first.reset();
    prompt_is(first.choose(both), 2);

    auto decode = first.choose({{3, stage::generation}});
    decode.generation_slots.front() = 77;
    check(first.choose({{3, stage::generation}}).generation_slots.front() == 3,
          "result mutation changed selector");
}

static void test_invalid_input_does_not_commit() {
    prefill_selector selector;
    prompt_is(selector.choose({{9, stage::prompt}}), 9);
    for (const auto & bad : {
            std::vector<slot_state>{{1, stage::prompt}, {1, stage::generation}},
            std::vector<slot_state>{{1, static_cast<stage>(99)}}}) {
        bool rejected = false;
        try {
            selector.choose(bad);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        check(rejected, "invalid input was accepted");
        prompt_is(selector.choose({{1, stage::prompt}, {9, stage::prompt}}), 9);
    }
}

static void test_generation_can_starve() {
    prefill_selector selector;
    for (std::uint32_t id = 1; id < 20; ++id) {
        prompt_is(selector.choose({{0, stage::generation}, {id, stage::prompt}}), id);
    }
    const auto resumed = selector.choose({{0, stage::generation}});
    check(resumed.generation_slots == std::vector<std::uint32_t>({0}), "generation did not resume");
}

int main() {
    try {
        test_sticky_and_slot_order();
        test_ownership_and_instances();
        test_invalid_input_does_not_commit();
        test_generation_can_starve();
        std::cout << "prefill selector tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
