#include <tq/calibration/reader.h>

#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

using tq::calibration::reader;
using tq::calibration::read_limits;

namespace {

void check(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

void u64(std::string & out, std::uint64_t value, unsigned bytes = 8) {
    for (unsigned i = 0; i < bytes; ++i) { out.push_back(static_cast<char>(value >> (i * 8))); }
}

void f64(std::string & out, double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    u64(out, bits);
}

void set_u64(std::string & out, std::size_t offset, std::uint64_t value, unsigned bytes = 8) {
    for (unsigned i = 0; i < bytes; ++i) { out.at(offset + i) = static_cast<char>(value >> (i * 8)); }
}

std::string header(std::uint32_t layers, std::uint32_t flags = 7) {
    std::string out = "WXQCAL03";
    for (auto value : {layers, 0u, 0u, 0u, 1u, flags}) { u64(out, value, 4); }
    for (auto value : {9u, 9u, 2u}) { u64(out, value); }
    u64(out, 1);
    return out;
}

std::string layer(std::uint32_t id = 3) {
    std::string out;
    for (auto value : {id, 2u, 2u, 3u}) { u64(out, value, 4); }
    for (auto value : {4u, 9u, 4u}) { u64(out, value); }
    for (auto value : {9u, 6u}) { u64(out, value); }
    for (auto value : {10.0, 20.0}) { f64(out, value); }
    for (auto value : {2u, 1u}) { u64(out, value); }
    for (auto value : {-1.0, 2.0, 3.0, -2.0}) { f64(out, value); }
    for (auto value : {5.0, 8.0, 9.0, 4.0}) { f64(out, value); }
    for (auto value : {2u, 1u}) { u64(out, value); }
    for (auto value : {4.0, 8.0, 12.0, 3.0, 6.0, 9.0}) { f64(out, value); }
    for (auto value : {10.0, -3.0, -3.0, 20.0}) { f64(out, value); }
    return out;
}

void read_all(const std::string & bytes, read_limits limits = {}) {
    std::istringstream input(bytes);
    reader source(input, limits);
    while (source.next()) {}
}

void rejects(const std::function<void()> & action) {
    bool failed = false;
    try { action(); } catch (const std::exception &) { failed = true; }
    check(failed, "invalid input was accepted");
}

void run() {
    static_assert(!std::is_copy_constructible<reader>::value, "reader position has one owner");
    const std::string valid = header(1) + layer();
    {
        std::istringstream input(header(2) + layer(3) + layer(7));
        reader source(input);
        auto first = source.next();
        auto second = source.next();
        check(first && second && first->id == 3 && second->id == 7, "layer identities changed");
        check(first->expert_sum[0] == -1 && first->expert_sum2[3] == 4 && first->hidden_sum2[5] == 9, "moments changed");
        check(first->route_counts[0] == 9 && first->expert_counts[0] == 2, "independent sample populations were merged");
        first->expert_counts[0] = 99;
        check(second->expert_counts[0] == 2, "layer arrays alias each other");
        auto copy = source.header();
        copy.layers = 99;
        check(copy.layers == 99 && source.header().layers == 2 && !source.next() && !source.next(), "header ownership or terminal EOF failed");
    }
    for (std::size_t size = 0; size < valid.size(); ++size) { rejects([&] { read_all(valid.substr(0, size)); }); }
    rejects([&] { read_all(valid + "tail"); });
    rejects([&] { read_all(header(2) + layer(3) + layer(3)); });
    for (unsigned kind = 0; kind < 12; ++kind) {
        std::string bad = valid;
        if (kind == 0) { bad[0] = '!'; }
        if (kind == 1) { set_u64(bad, 28, 8, 4); }
        if (kind == 2) { bad[56] = 2; }
        if (kind == 3) { bad[57] = 1; }
        if (kind == 4) { set_u64(bad, 8, 4097, 4); }
        if (kind == 5) { set_u64(bad, 68, 0, 4); }
        if (kind == 6) { set_u64(bad, 80, 10); }
        if (kind == 7) { set_u64(bad, 104, 10); }
        if (kind == 8) { set_u64(bad, 136, 5); }
        if (kind == 9) { set_u64(bad, 136, 0); }
        if (kind == 10) { set_u64(bad, 120, 0x7ff0000000000000ULL); }
        if (kind == 11) { set_u64(bad, 184, 0xbff0000000000000ULL); }
        rejects([&] { read_all(bad); });
    }
    read_limits bounds;
    bounds.max_layer_bytes = valid.size() - 64;
    read_all(valid, bounds);
    --bounds.max_layer_bytes;
    rejects([&] { read_all(valid, bounds); });
    bounds = {};
    bounds.max_file_bytes = valid.size() - 1;
    rejects([&] { read_all(valid, bounds); });
    std::string huge = header(1) + layer();
    set_u64(huge, 68, UINT32_MAX, 4);
    set_u64(huge, 72, UINT32_MAX, 4);
    rejects([&] { read_all(huge); });
    {
        std::istringstream input(valid + "x");
        reader source(input);
        rejects([&] { source.next(); });
        rejects([&] { source.next(); });
    }
    std::string partial = valid;
    partial[56] = 0;
    set_u64(partial, 40, 100);
    read_all(partial);
    read_all(header(0));
    rejects([&] { read_all(header(0) + "x"); });
}

} // namespace

int main() {
    try { run(); std::cout << "calibration reader: all checks passed\n"; return 0; }
    catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
