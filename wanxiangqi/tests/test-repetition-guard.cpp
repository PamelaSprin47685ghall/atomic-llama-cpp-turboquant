#include "o200k-codec.h"
#include "repetition-corpus.h"
#include "repetition-envelope.h"
#include "repetition-guard.h"

#include "ggml.h"

#include <cstdio>
#include <vector>

static void test_envelope_metadata() {
    wanxiangqi_o200k_codec codec;

    GGML_ASSERT(codec.vocabulary_size() == wanxiangqi_repetition_envelope::vocabulary_size);
    GGML_ASSERT(wanxiangqi_repetition_envelope::half_life == 256);
}

static void test_repository_corpus_filter() {
    GGML_ASSERT(wanxiangqi_repetition_corpus_path("tools/server/server-context.cpp"));
    GGML_ASSERT(wanxiangqi_repetition_corpus_path("docs/build.md"));
    GGML_ASSERT(!wanxiangqi_repetition_corpus_path("docs/ops/Vulkan.csv"));
    GGML_ASSERT(!wanxiangqi_repetition_corpus_path("vendor/nlohmann/json.hpp"));
    GGML_ASSERT(!wanxiangqi_repetition_corpus_path("tests/fixtures/generated.cpp"));
    GGML_ASSERT(!wanxiangqi_repetition_corpus_path("tests/snapshots/model.schema"));
    GGML_ASSERT(!wanxiangqi_repetition_corpus_path("package-lock.json"));
    GGML_ASSERT(wanxiangqi_repetition_corpus_text("int main() { return 0; }\n"));
    GGML_ASSERT(!wanxiangqi_repetition_corpus_text("// Auto-generated source\nint value = 1;\n"));
    GGML_ASSERT(!wanxiangqi_repetition_corpus_text("// generated with scripts/gen-data.py\nint value = 1;\n"));
    GGML_ASSERT(!wanxiangqi_repetition_corpus_text("// Pre-computed rotation matrices\nstatic float data[] = { 1 };\n"));
}

static void test_strict_envelope_boundaries() {
    GGML_ASSERT(wanxiangqi_repetition_classify(
        wanxiangqi_repetition_envelope::minimum_weighted_distinct_count) == wanxiangqi_repetition_state::normal);
    GGML_ASSERT(wanxiangqi_repetition_classify(
        wanxiangqi_repetition_envelope::maximum_weighted_distinct_count) == wanxiangqi_repetition_state::normal);
}

static void test_repetitive_sequence() {
    wanxiangqi_repetition_detector detector;
    auto state = wanxiangqi_repetition_state::normal;

    for (size_t i = 0; i < wanxiangqi_repetition_envelope::half_life * 8 && state == wanxiangqi_repetition_state::normal; ++i) {
        state = detector.push(42);
    }

    GGML_ASSERT(state == wanxiangqi_repetition_state::too_repetitive);
}

static void test_low_repetition_sequence() {
    wanxiangqi_repetition_detector detector;
    auto state = wanxiangqi_repetition_state::normal;

    for (size_t token = 0; token < wanxiangqi_repetition_envelope::half_life * 64 && state == wanxiangqi_repetition_state::normal; ++token) {
        state = detector.push(static_cast<int>(token));
    }

    GGML_ASSERT(state == wanxiangqi_repetition_state::too_random);
}

static void test_guard_streams_text_deltas() {
    wanxiangqi_repetition_guard guard;
    auto state = wanxiangqi_repetition_state::normal;

    for (size_t i = 0; i < wanxiangqi_repetition_envelope::half_life * 8 && state == wanxiangqi_repetition_state::normal; ++i) {
        state = guard.observe(" hello");
    }

    GGML_ASSERT(state == wanxiangqi_repetition_state::too_repetitive);
}

int main() {
    test_envelope_metadata();
    test_repository_corpus_filter();
    test_strict_envelope_boundaries();
    test_repetitive_sequence();
    test_low_repetition_sequence();
    test_guard_streams_text_deltas();
    std::printf("repetition guard tests passed\n");
    return 0;
}
