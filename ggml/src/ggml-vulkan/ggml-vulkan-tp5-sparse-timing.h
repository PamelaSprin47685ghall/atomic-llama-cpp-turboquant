#pragma once
#include <cstdint>
#include <cstdlib>
#include <string>

// Selection of existing timestamp IDs, not a modification of graph commands.
// Unselected queries must never be waited on or read as completed samples.
struct vk_tp5_sparse_timing {
    bool sparse = false;
    bool valid = true;
    uint32_t stride = 0;
    uint32_t residue = 0;

    static vk_tp5_sparse_timing parse(const char * text) {
        vk_tp5_sparse_timing s;
        if (!text) return s;
        s.sparse = true;
        if (std::string(text) == "span") return s;
        const std::string value(text);
        const auto colon = value.find(':');
        if (colon == std::string::npos || colon == 0 || colon + 1 == value.size()) { s.valid = false; return s; }
        const auto integer = [](const std::string & x, uint32_t & out) {
            if (x.empty() || x.size() > 3 || x.find_first_not_of("0123456789") != std::string::npos) return false;
            out = (uint32_t) std::strtoul(x.c_str(), nullptr, 10);
            return true;
        };
        s.valid = integer(value.substr(0, colon), s.stride) && integer(value.substr(colon+1), s.residue) &&
                  s.stride >= 1 && s.stride <= 97 && s.residue < s.stride;
        return s;
    }
    bool stage(uint32_t s) const { return !sparse || (stride && s % stride == residue); }
    bool query(uint32_t q, uint32_t stages) const {
        if (!valid || q >= 5 * stages + 3) return false;
        return !sparse || q == 0 || q == 5 * stages + 2 || stage(q / 5);
    }
};

inline const vk_tp5_sparse_timing & vk_tp5_timing_selection() {
    static const auto s = vk_tp5_sparse_timing::parse(std::getenv("GGML_TP5_GPU_TIMING_SPARSE"));
    return s;
}
