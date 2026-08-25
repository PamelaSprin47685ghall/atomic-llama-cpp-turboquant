#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class wanxiangqi_o200k_codec {
public:
    wanxiangqi_o200k_codec();
    ~wanxiangqi_o200k_codec();

    wanxiangqi_o200k_codec(wanxiangqi_o200k_codec &&) noexcept;
    wanxiangqi_o200k_codec & operator=(wanxiangqi_o200k_codec &&) noexcept;

    wanxiangqi_o200k_codec(const wanxiangqi_o200k_codec &) = delete;
    wanxiangqi_o200k_codec & operator=(const wanxiangqi_o200k_codec &) = delete;

    std::vector<int> encode(const std::string & text) const;
    std::size_t vocabulary_size() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
