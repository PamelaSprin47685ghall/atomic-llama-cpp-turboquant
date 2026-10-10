#pragma once

#include "ggml.h"
#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Host intervals only. A decode scope encloses its ubatch scopes: never add
// them together or describe async submission time as pure GPU execution time.
// No additional scheduler waits, allocations, or GPU commands are introduced.
struct llama_decode_cost {
    static bool enabled() {
        static const bool on = [] {
            const char * e = std::getenv("GGML_DECODE_COST");
            return e && std::strcmp(e, "0") != 0 && std::strcmp(e, "off") != 0;
        }();
        return on;
    }
    bool on;
    const char * scope;
    const void * ctx;
    int rows;
    int type;
    int reuse = -1;
    int64_t begin;
    int64_t last;
    std::array<int64_t, 5> us{};
    llama_decode_cost(const char * s, const void * c, int n, int t)
        : on(enabled()), scope(s), ctx(c), rows(n), type(t),
          begin(on ? ggml_time_us() : 0), last(begin) {}
    void mark(size_t i) {
        if (!on) return;
        const int64_t now = ggml_time_us();
        us.at(i) += now - last;
        last = now;
    }
    ~llama_decode_cost() {
        if (!on) return;
        const int64_t end = ggml_time_us();
        std::fprintf(stderr, "[decode-cost] scope=%s ctx=%p type=%d rows=%d reuse=%d "
                "setup_us=%" PRId64 " graph_us=%" PRId64 " alloc_us=%" PRId64
                " input_us=%" PRId64 " submit_us=%" PRId64 " finish_us=%" PRId64
                " total_us=%" PRId64 "\n", scope, ctx, type, rows, reuse,
                us[0], us[1], us[2], us[3], us[4], end - last, end - begin);
    }
};
