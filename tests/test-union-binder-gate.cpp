#include "../src/llama-union-binder.h"

#include <cstdio>

//
// test-union-binder-gate
//
// The value contract of every GGML_TP5_* on/off knob in the union binder, as
// pure functions of the environment string. No model, no context, no backend:
// this test runs in every CPU build, so the parser contract is provable with
// zero fixtures.
//
// Contract (llama-union-binder.h): exactly "1" is ON. Unset, "", "0", "true",
// "01", " 1", "2" and every other string are OFF. The umbrella gate alone
// keeps an unset CPU default; the rs_z sub-lane has no default and is OFF
// whenever the value is not exactly "1".
//
// Every case fails with its own message, so a broken parser names the input.

namespace {

int g_fail = 0;
int g_checks_run = 0;

void check(bool cond, const char * what, const char * value, bool expected) {
    ++g_checks_run;
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s(value=%s) expected %s\n", what,
                     value == nullptr ? "<unset>" : value, expected ? "ON" : "OFF");
        ++g_fail;
    }
}

} // namespace

int main() {
    // the exact-one contract, shared by the umbrella gate (explicit values)
    // and the rs_z runtime lane
    struct exact_case {
        const char * value;
        bool         expected;
    };
    const exact_case exact[] = {
        { nullptr, false },
        { "",      false },
        { "0",     false },
        { "true",  false },
        { "01",    false },
        { " 1",    false },
        { "1 ",    false },
        { "2",     false },
        { "-1",    false },
        { "1",     true  },
    };
    for (const auto & c : exact) {
        check(llama_union_gate_exact_one(c.value) == c.expected,
              "llama_union_gate_exact_one", c.value, c.expected);
    }

    // the umbrella gate: unset keeps the caller's CPU default, every explicit
    // value follows the exact-one contract
    struct binder_case {
        const char * value;
        bool         cpu_default;
        bool         expected;
    };
    const binder_case binder[] = {
        { nullptr, false, false },
        { nullptr, true,  true  },
        { "1",     false, true  },
        { "1",     true,  true  },
        { "0",     false, false },
        { "0",     true,  false },
        { "",      true,  false },
        { "true",  true,  false },
        { "01",    true,  false },
        { " 1",    true,  false },
        { "2",     true,  false },
    };
    for (const auto & c : binder) {
        check(llama_union_binder_enabled(c.value, c.cpu_default) == c.expected,
              "llama_union_binder_enabled", c.value, c.expected);
    }

    // the rs_z runtime lane reads its own variable through the same helper,
    // with no default: only "1" arms it. This is the exact expression the
    // context constructor evaluates.
    for (const auto & c : exact) {
        check((c.value != nullptr && llama_union_gate_exact_one(c.value)) == c.expected,
              "rs_z lane gate", c.value, c.expected);
    }

    if (g_checks_run == 0) {
        std::fprintf(stderr, "UNION_BINDER_GATE_TEST UNPROVEN reason=no_check_ran checks=0\n");
        return 2;
    }
    if (g_fail != 0) {
        std::printf("UNION_BINDER_GATE_TEST FAIL %d case(s)\n", g_fail);
        return 1;
    }
    std::printf("UNION_BINDER_GATE_TEST PASS %zu exact-one, %zu umbrella cases\n",
                sizeof(exact) / sizeof(exact[0]), sizeof(binder) / sizeof(binder[0]));
    return 0;
}
