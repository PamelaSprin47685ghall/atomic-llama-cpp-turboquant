// CPU-only decision test for llama_free's relay retain policy.
//
// The policy lives in src/llama-relay-free.h so every branch is testable
// without a llama_context, a relay provider, HIP, or a model. The provider
// queries themselves are covered by the opt-in five-GPU relay test.

#include "llama-relay-free.h"
#include "ggml-relay-token-guard.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

int failures = 0;
int checks   = 0;

bool expect(bool condition, const char * expression, int line) {
    ++checks;
    if (!condition) {
        std::fprintf(stderr, "FAIL line %d: %s\n", line, expression);
        ++failures;
    }
    return condition;
}

#define CHECK(expr) expect((expr), #expr, __LINE__)

llama_relay_free_facts settled() {
    llama_relay_free_facts facts;
    facts.have_snapshot   = true;
    facts.status_failed   = false;
    facts.status_finished = true;
    facts.status_epoch    = 2;
    facts.status_channels = 4;
    facts.cpu_result      = 1;
    facts.cpu.epoch       = 2;
    facts.cpu.channels    = 4;
    facts.cpu.outputs_released  = 4;
    facts.cpu.borrowed_channel  = UINT32_MAX;
    facts.cpu.active_token      = 0;
    facts.cpu.tokens_outstanding = 0;
    return facts;
}

void test_settled() {
    const auto decision = llama_relay_free_decide(settled());
    CHECK(!decision.retain);
    CHECK(!decision.abort_relay);
}

void test_epoch_zero() {
    auto facts = settled();
    facts.status_epoch = 0;
    facts.status_channels = 0;
    facts.cpu.epoch = 0;
    facts.cpu.channels = 0;
    facts.cpu.outputs_released = 0;
    const auto decision = llama_relay_free_decide(facts);
    CHECK(!decision.retain);
    CHECK(!decision.abort_relay);
}

void test_prepare_window() {
    // prepare marks the new channels released and sets outputs_released to the
    // new count; epoch 0 has no begun epoch. This must not look like an
    // unreleased output.
    auto facts = settled();
    facts.status_epoch = 0;
    facts.status_channels = 4;
    facts.cpu.epoch = 0;
    facts.cpu.channels = 4;
    facts.cpu.outputs_released = 4;
    const auto decision = llama_relay_free_decide(facts);
    CHECK(!decision.retain);
    CHECK(!decision.abort_relay);
}

void test_prepare_window_after_epoch() {
    // prepare changes channels after epoch 2 and resets outputs_released to
    // the new count. The provider is still settled, not "outputs outstanding".
    auto facts = settled();
    facts.status_epoch = 2;
    facts.status_channels = 5;
    facts.cpu.epoch = 2;
    facts.cpu.channels = 5;
    facts.cpu.outputs_released = 5;
    const auto decision = llama_relay_free_decide(facts);
    CHECK(!decision.retain);
    CHECK(!decision.abort_relay);
}

void test_borrow() {
    auto facts = settled();
    facts.cpu.borrowed_channel = 1;
    const auto decision = llama_relay_free_decide(facts);
    CHECK(decision.retain);
    CHECK(decision.abort_relay);
    CHECK(std::strcmp(decision.reason, "a relay CPU borrow is still outstanding") == 0);
}

void test_active_token() {
    auto facts = settled();
    facts.cpu.active_token = 7;
    const auto decision = llama_relay_free_decide(facts);
    CHECK(decision.retain);
    CHECK(decision.abort_relay);
    CHECK(std::strcmp(decision.reason, "relay completion tokens are still outstanding") == 0);
}

void test_settled_token_not_released() {
    auto facts = settled();
    facts.cpu.active_token = 0;
    facts.cpu.tokens_outstanding = 1;
    const auto decision = llama_relay_free_decide(facts);
    CHECK(decision.retain);
    CHECK(decision.abort_relay);
    CHECK(std::strcmp(decision.reason, "relay completion tokens are still outstanding") == 0);
}

void test_releases_outstanding() {
    auto facts = settled();
    facts.cpu.outputs_released = 3;
    const auto decision = llama_relay_free_decide(facts);
    CHECK(decision.retain);
    CHECK(decision.abort_relay);
    CHECK(std::strcmp(decision.reason, "relay output releases are still outstanding") == 0);
}

void test_failed_provider_skips_releases() {
    auto facts = settled();
    facts.status_failed = true;
    facts.cpu.outputs_released = 3;
    const auto decision = llama_relay_free_decide(facts);
    CHECK(!decision.retain);
    CHECK(!decision.abort_relay);
}

void test_contended() {
    auto facts = settled();
    facts.cpu_result = 0;
    const auto decision = llama_relay_free_decide(facts);
    CHECK(decision.retain);
    CHECK(decision.abort_relay);
    CHECK(std::strcmp(decision.reason, "relay CPU state could not be proven") == 0);
}

void test_unsupported_falls_back() {
    auto facts = settled();
    facts.cpu_result = -1;
    facts.cpu = {};
    const auto decision = llama_relay_free_decide(facts);
    CHECK(!decision.retain);
    CHECK(!decision.abort_relay);
}

void test_epoch_mismatch() {
    auto facts = settled();
    facts.cpu.epoch = 3;
    const auto decision = llama_relay_free_decide(facts);
    CHECK(decision.retain);
    CHECK(decision.abort_relay);
    CHECK(std::strcmp(decision.reason, "relay CPU state could not be proven") == 0);
}

void test_channels_mismatch() {
    auto facts = settled();
    facts.cpu.channels = 5;
    const auto decision = llama_relay_free_decide(facts);
    CHECK(decision.retain);
    CHECK(decision.abort_relay);
    CHECK(std::strcmp(decision.reason, "relay CPU state could not be proven") == 0);
}

void test_unfinished_epoch() {
    auto facts = settled();
    facts.status_finished = false;
    const auto decision = llama_relay_free_decide(facts);
    CHECK(decision.retain);
    CHECK(decision.abort_relay);
    CHECK(std::strcmp(decision.reason, "an async/native submission is still unproven") == 0);
}

void test_borrow_precedes_unfinished() {
    auto facts = settled();
    facts.status_finished = false;
    facts.cpu.borrowed_channel = 2;
    const auto decision = llama_relay_free_decide(facts);
    CHECK(decision.retain);
    CHECK(decision.abort_relay);
    CHECK(std::strcmp(decision.reason, "a relay CPU borrow is still outstanding") == 0);
}

void test_no_snapshot() {
    auto facts = settled();
    facts.have_snapshot = false;
    facts.cpu_result = -1;
    facts.cpu = {};
    const auto decision = llama_relay_free_decide(facts);
    CHECK(decision.retain);
    CHECK(decision.abort_relay);
    CHECK(std::strcmp(decision.reason, "an async/native submission is still unproven") == 0);
}

void test_has_unproven_only() {
    auto facts = settled();
    facts.has_unproven = true;
    const auto decision = llama_relay_free_decide(facts);
    CHECK(decision.retain);
    CHECK(!decision.abort_relay);
    CHECK(std::strcmp(decision.reason, "an async/native submission is still unproven") == 0);
}

void test_has_unproven_and_borrow() {
    auto facts = settled();
    facts.has_unproven = true;
    facts.cpu.borrowed_channel = 3;
    const auto decision = llama_relay_free_decide(facts);
    CHECK(decision.retain);
    CHECK(decision.abort_relay);
    CHECK(std::strcmp(decision.reason, "a relay CPU borrow is still outstanding") == 0);
}

void test_token_order_contract() {
    // The provider's prepare/begin gate blocks on either map. A token record
    // is created by begin, remains after wait settles it, and is retired only
    // by release. A second release is a no-op; wait after release is UNKNOWN.
    CHECK(!relay_token_blocks_new_epoch(0, 0));
    CHECK(relay_token_blocks_new_epoch(1, 0));
    CHECK(relay_token_blocks_new_epoch(0, 1));
    CHECK(relay_token_blocks_new_epoch(1, 1));
    CHECK(relay_token_release_is_noop(0));
    CHECK(!relay_token_release_is_noop(1));
    CHECK(relay_token_wait_after_release_is_unknown(0));
    CHECK(!relay_token_wait_after_release_is_unknown(1));

    size_t tails = 0;
    size_t states = 0;
    auto begin_allowed = [&]() { return !relay_token_blocks_new_epoch(tails, states); };
    CHECK(begin_allowed());
    tails = 1; states = 1; // begin created the token record
    CHECK(!begin_allowed());
    // wait_token settles success/failure but does not retire the record.
    CHECK(!begin_allowed());
    tails = 0; states = 0; // release_token
    CHECK(begin_allowed());
    tails = 0; states = 0; // second release is idempotent
    CHECK(begin_allowed());
    CHECK(relay_token_wait_after_release_is_unknown(states));
}

} // namespace

int main() {
    test_settled();
    test_epoch_zero();
    test_prepare_window();
    test_prepare_window_after_epoch();
    test_borrow();
    test_active_token();
    test_settled_token_not_released();
    test_releases_outstanding();
    test_failed_provider_skips_releases();
    test_contended();
    test_unsupported_falls_back();
    test_epoch_mismatch();
    test_channels_mismatch();
    test_unfinished_epoch();
    test_borrow_precedes_unfinished();
    test_no_snapshot();
    test_has_unproven_only();
    test_has_unproven_and_borrow();
    test_token_order_contract();

    if (failures != 0) {
        std::fprintf(stderr, "RELAY_FREE_DECISION_TEST FAIL (%d failures / %d checks)\n", failures, checks);
        return 1;
    }
    std::printf("RELAY_FREE_DECISION_TEST PASS (%d checks)\n", checks);
    return 0;
}
