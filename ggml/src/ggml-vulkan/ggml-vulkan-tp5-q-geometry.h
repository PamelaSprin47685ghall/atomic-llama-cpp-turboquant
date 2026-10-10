#pragma once
#include <cstdint>
#include <cstring>

// Split at whole-wave boundaries only. Per-row arithmetic always uses16 lanes.
inline uint32_t vk_tp5_q_threads(const char * text) {
    if (!text || std::strcmp(text,"256")==0) return 256;
    if (std::strcmp(text,"128")==0) return 128;
    if (std::strcmp(text,"64")==0) return 64;
    return 0;
}
inline uint32_t vk_tp5_q_groups(uint32_t rank, uint32_t threads) {
    if (threads!=64 && threads!=128 && threads!=256) return 0;
    const uint32_t rows=threads/16;
    return rank && rank<=512 && rank%rows==0 ? rank/rows : 0;
}
