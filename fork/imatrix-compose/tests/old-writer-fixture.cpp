#include "calib-dump.h"

#include <filesystem>
#include <iostream>
#include <map>

int main(int argc, char ** argv) {
    if (argc != 2 || std::filesystem::exists(argv[1])) {
        std::cerr << "usage: old-writer-fixture NEW-CALIBRATION.bin\n";
        return 2;
    }
    const std::uint32_t endian = 1;
    if (*reinterpret_cast<const unsigned char *>(&endian) != 1) {
        std::cerr << "the old writer requires a little-endian host\n";
        return 2;
    }
    wxq::calib_meta meta;
    meta.max_ctx = 0;
    meta.act_stride = 0;
    meta.act_tokens = 0;
    meta.route_stride = 1;
    meta.flags = wxq::CALIB_COV | wxq::CALIB_HIDDEN | wxq::CALIB_EXPERT_IN;
    meta.n_decoded = 9;
    meta.n_total = 9;
    meta.n_ctx_resets = 2;
    meta.complete = 1;
    wxq::calib_layer layer;
    layer.n_expert = 2;
    layer.n_embd = 2;
    layer.n_ff = 3;
    layer.n_pos = 4;
    layer.n_route = 9;
    layer.n_cov = 4;
    layer.counts = {9, 6};
    layer.act_sum = {10, 20};
    layer.exp_n = {2, 1};
    layer.exp_mean = {-1, 2, 3, -2};
    layer.exp_diag = {5, 8, 9, 4};
    layer.hid_n = {2, 1};
    layer.hid_sum = {4, 8, 12, 3, 6, 9};
    layer.cov = {10, -3, 12345, 20};
    const std::map<int, wxq::calib_layer> layers = {{3, layer}, {7, layer}};
    std::string error;
    if (!wxq::write_calib(argv[1], meta, layers, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    std::cout << "Generated a format fixture with the unchanged WXQCAL03 writer. This is not model calibration data.\n";
    return 0;
}
