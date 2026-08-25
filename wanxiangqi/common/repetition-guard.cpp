#include "repetition-guard.h"

#include "o200k-codec.h"
#include "repetition-envelope.h"

#include <cmath>
#include <utility>
#include <vector>

const char * wanxiangqi_repetition_state_name(wanxiangqi_repetition_state state) {
    switch (state) {
        case wanxiangqi_repetition_state::normal:         return "normal";
        case wanxiangqi_repetition_state::too_repetitive: return "too-repetitive";
        case wanxiangqi_repetition_state::too_random:     return "too-random";
    }
    return "unknown";
}

wanxiangqi_repetition_state wanxiangqi_repetition_classify(double weighted_distinct_count) {
    if (weighted_distinct_count < wanxiangqi_repetition_envelope::lower_weighted_distinct_count) {
        return wanxiangqi_repetition_state::too_repetitive;
    }
    if (weighted_distinct_count > wanxiangqi_repetition_envelope::upper_weighted_distinct_count) {
        return wanxiangqi_repetition_state::too_random;
    }
    return wanxiangqi_repetition_state::normal;
}

wanxiangqi_repetition_detector::wanxiangqi_repetition_detector() {
    reset();
}

void wanxiangqi_repetition_detector::reset() {
    weighted_distinct = wanxiangqi_repetition_envelope::normal_weighted_distinct_count;
    token_step = 0;
    last_seen.clear();
}

wanxiangqi_repetition_state wanxiangqi_repetition_detector::push(int token) {
    ++token_step;
    const auto found = last_seen.find(token);
    const double replacement = found == last_seen.end()
        ? 1.0
        : 1.0 - std::pow(wanxiangqi_repetition_envelope::lambda, static_cast<double>(token_step - found->second));

    weighted_distinct = wanxiangqi_repetition_envelope::lambda * weighted_distinct + replacement;
    last_seen[token] = token_step;
    return wanxiangqi_repetition_classify(weighted_distinct);
}

double wanxiangqi_repetition_detector::weighted_distinct_count() const {
    return weighted_distinct;
}

int64_t wanxiangqi_repetition_detector::step() const {
    return token_step;
}

struct wanxiangqi_repetition_guard::impl {
    wanxiangqi_o200k_codec codec;
    wanxiangqi_repetition_detector detector;
};

wanxiangqi_repetition_guard::wanxiangqi_repetition_guard()
    : pimpl(std::make_unique<impl>()) {
}

wanxiangqi_repetition_guard::~wanxiangqi_repetition_guard() = default;

wanxiangqi_repetition_guard::wanxiangqi_repetition_guard(wanxiangqi_repetition_guard &&) noexcept = default;

wanxiangqi_repetition_guard & wanxiangqi_repetition_guard::operator=(wanxiangqi_repetition_guard &&) noexcept = default;

void wanxiangqi_repetition_guard::reset() {
    pimpl->detector.reset();
}

wanxiangqi_repetition_state wanxiangqi_repetition_guard::observe(const std::string & text) {
    if (text.empty()) {
        return wanxiangqi_repetition_classify(pimpl->detector.weighted_distinct_count());
    }

    const std::vector<int> tokens = pimpl->codec.encode(text);
    for (int token : tokens) {
        pimpl->detector.push(token);
    }

    return wanxiangqi_repetition_classify(pimpl->detector.weighted_distinct_count());
}
