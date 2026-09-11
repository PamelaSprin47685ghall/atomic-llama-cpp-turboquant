#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

enum class wanxiangqi_repetition_state {
    normal,
    too_repetitive,
    too_random,
};

const char * wanxiangqi_repetition_state_name(wanxiangqi_repetition_state state);
wanxiangqi_repetition_state wanxiangqi_repetition_classify(double weighted_distinct_count);

class wanxiangqi_repetition_detector {
public:
    wanxiangqi_repetition_detector();

    void reset();
    wanxiangqi_repetition_state push(int token);

    double weighted_distinct_count() const;
    int64_t step() const;

private:
    double weighted_distinct = 0.0;
    int64_t token_step = 0;
    std::unordered_map<int, int64_t> last_seen;
};

class wanxiangqi_repetition_guard {
public:
    wanxiangqi_repetition_guard();
    ~wanxiangqi_repetition_guard();

    wanxiangqi_repetition_guard(wanxiangqi_repetition_guard &&) noexcept;
    wanxiangqi_repetition_guard & operator=(wanxiangqi_repetition_guard &&) noexcept;

    wanxiangqi_repetition_guard(const wanxiangqi_repetition_guard &) = delete;
    wanxiangqi_repetition_guard & operator=(const wanxiangqi_repetition_guard &) = delete;

    void reset();
    wanxiangqi_repetition_state observe(const std::string & text);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
