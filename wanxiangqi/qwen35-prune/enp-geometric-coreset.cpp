#include "enp-geometric-coreset.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace qwen35_prune {
namespace {

constexpr int SKETCH_DIM = 16;
constexpr int SIGNATURE_DIM = 16;

static uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static void make_sketch(const float * x, int width, float * out) {
    std::fill(out, out + SKETCH_DIM, 0.0f);
    for (int i = 0; i < width; ++i) {
        const uint64_t h = mix64((uint64_t) i + 0x454e50434f524553ULL);
        const int bucket = (int) (h & (SKETCH_DIM - 1));
        const float sign = (h >> 8) & 1ULL ? 1.0f : -1.0f;
        out[bucket] += sign * x[i];
    }
}

static uint32_t sketch_signature(const float * sketch) {
    uint32_t out = 0;
    for (int i = 0; i < SIGNATURE_DIM; ++i) {
        if (sketch[i] >= 0.0f) out |= 1u << i;
    }
    return out;
}

static double l2_distance(const float * a, const float * b, int width) {
    long double sum = 0.0L;
    for (int i = 0; i < width; ++i) {
        const long double d = (long double) a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt((double) sum);
}

static double vector_norm2(const float * x, int width) {
    long double sum = 0.0L;
    for (int i = 0; i < width; ++i) sum += (long double) x[i] * x[i];
    return (double) sum;
}

struct pair_candidate {
    int a = -1;
    int b = -1;
    double sketch_d2 = 0.0;
    uint64_t tie = 0;
};

static double sketch_distance2(const float * a, const float * b) {
    double sum = 0.0;
    for (int i = 0; i < SKETCH_DIM; ++i) {
        const double d = (double) a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

static std::vector<std::pair<int, int>> choose_pairs(
        int n,
        int merges,
        const std::function<const float *(int)> & sketch,
        const std::function<double(int)> & norm2,
        const std::function<uint64_t(int)> & source_id) {
    if (merges < 0 || merges > n / 2) throw std::runtime_error("invalid geometric coreset merge count");
    if (merges == 0) return {};

    std::vector<int> order((size_t) n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const uint32_t sa = sketch_signature(sketch(a));
        const uint32_t sb = sketch_signature(sketch(b));
        if (sa != sb) return sa < sb;
        if (norm2(a) != norm2(b)) return norm2(a) < norm2(b);
        return source_id(a) < source_id(b);
    });

    if (merges == n / 2) {
        std::vector<std::pair<int, int>> out;
        out.reserve((size_t) merges);
        for (int i = 0; i < 2 * merges; i += 2) out.emplace_back(order[(size_t) i], order[(size_t) i + 1]);
        return out;
    }

    std::vector<pair_candidate> candidates;
    candidates.reserve((size_t) std::max(0, n - 1));
    for (int i = 0; i + 1 < n; ++i) {
        const int a = order[(size_t) i];
        const int b = order[(size_t) i + 1];
        candidates.push_back({a, b, sketch_distance2(sketch(a), sketch(b)),
                              std::min(source_id(a), source_id(b))});
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const pair_candidate & a, const pair_candidate & b) {
        if (a.sketch_d2 != b.sketch_d2) return a.sketch_d2 < b.sketch_d2;
        return a.tie < b.tie;
    });

    std::vector<uint8_t> used((size_t) n, 0);
    std::vector<std::pair<int, int>> out;
    out.reserve((size_t) merges);
    for (const auto & c : candidates) {
        if (used[(size_t) c.a] || used[(size_t) c.b]) continue;
        used[(size_t) c.a] = used[(size_t) c.b] = 1;
        out.emplace_back(c.a, c.b);
        if ((int) out.size() == merges) return out;
    }

    // A path-greedy matching can leave isolated vertices.  Fill any remaining
    // merges deterministically from the unused sorted vertices; proof validity
    // is unaffected because original-space distance is measured after pairing.
    std::vector<int> remaining;
    for (int id : order) if (!used[(size_t) id]) remaining.push_back(id);
    size_t pos = 0;
    while ((int) out.size() < merges) {
        if (pos + 1 >= remaining.size()) throw std::runtime_error("failed to complete geometric coreset pairing");
        out.emplace_back(remaining[pos], remaining[pos + 1]);
        pos += 2;
    }
    return out;
}

static enp_geometric_cluster merge_clusters(
        const enp_geometric_cluster & a,
        const enp_geometric_cluster & b) {
    if (a.center.size() != b.center.size() || a.center.empty() || a.weight <= 0 || b.weight <= 0) {
        throw std::runtime_error("invalid geometric coreset cluster merge");
    }
    const double d = l2_distance(a.center.data(), b.center.data(), (int) a.center.size());
    const bool keep_a = a.weight > b.weight || (a.weight == b.weight && a.source_id <= b.source_id);
    const auto & keep = keep_a ? a : b;
    const auto & move = keep_a ? b : a;
    enp_geometric_cluster out = keep;
    out.weight = a.weight + b.weight;
    out.transport = a.transport + b.transport + (double) move.weight * d;
    out.radius = std::max(keep.radius, move.radius + d);
    return out;
}

} // namespace

enp_geometric_stream::enp_geometric_stream(int width, int max_points) {
    initialize(width, max_points);
}

void enp_geometric_stream::initialize(int width, int max_points) {
    if (width <= 0 || max_points <= 0) throw std::runtime_error("invalid geometric coreset dimensions");
    width_ = width;
    max_points_ = max_points;
    size_ = 0;
    population_ = 0;
    max_norm2_ = 0.0;
    const size_t capacity = (size_t) 2 * max_points_;
    centers_.assign(capacity * width_, 0.0f);
    sketches_.assign(capacity * SKETCH_DIM, 0.0f);
    center_norm2_.assign(capacity, 0.0);
    weights_.assign(capacity, 0);
    transport_.assign(capacity, 0.0);
    radius_.assign(capacity, 0.0);
    source_id_.assign(capacity, 0);
}

void enp_geometric_stream::add(const float * x) {
    if (!x || width_ <= 0 || max_points_ <= 0) throw std::runtime_error("uninitialized geometric coreset stream");
    if (size_ >= 2 * max_points_) throw std::runtime_error("geometric coreset stream overflow");
    const int dst = size_++;
    std::copy_n(x, width_, centers_.data() + (size_t) dst * width_);
    make_sketch(x, width_, sketches_.data() + (size_t) dst * SKETCH_DIM);
    const double norm2 = vector_norm2(x, width_);
    center_norm2_[(size_t) dst] = norm2;
    weights_[(size_t) dst] = 1;
    transport_[(size_t) dst] = 0.0;
    radius_[(size_t) dst] = 0.0;
    source_id_[(size_t) dst] = population_++;
    max_norm2_ = std::max(max_norm2_, norm2);
    if (size_ == 2 * max_points_) compress_to(max_points_);
}

void enp_geometric_stream::compress_to(int target) {
    if (target <= 0 || target > size_ || size_ - target > size_ / 2) {
        throw std::runtime_error("invalid geometric coreset compression target");
    }
    const int old_size = size_;
    const int merges = old_size - target;
    if (merges == 0) return;
    const auto pairs = choose_pairs(
        old_size, merges,
        [&](int i) { return sketches_.data() + (size_t) i * SKETCH_DIM; },
        [&](int i) { return center_norm2_[(size_t) i]; },
        [&](int i) { return source_id_[(size_t) i]; });

    std::vector<int> mate((size_t) old_size, -1);
    for (const auto & [a, b] : pairs) {
        mate[(size_t) a] = b;
        mate[(size_t) b] = a;
    }

    std::vector<float> centers_new((size_t) 2 * max_points_ * width_, 0.0f);
    std::vector<float> sketches_new((size_t) 2 * max_points_ * SKETCH_DIM, 0.0f);
    std::vector<double> center_norm2_new((size_t) 2 * max_points_, 0.0);
    std::vector<int64_t> weights_new((size_t) 2 * max_points_, 0);
    std::vector<double> transport_new((size_t) 2 * max_points_, 0.0);
    std::vector<double> radius_new((size_t) 2 * max_points_, 0.0);
    std::vector<uint64_t> source_id_new((size_t) 2 * max_points_, 0);

    int out = 0;
    for (int i = 0; i < old_size; ++i) {
        const int j = mate[(size_t) i];
        if (j >= 0 && i > j) continue;
        int keep = i;
        int move = -1;
        double merged_transport = transport_[(size_t) i];
        double merged_radius = radius_[(size_t) i];
        int64_t merged_weight = weights_[(size_t) i];
        if (j >= 0) {
            const bool keep_i = weights_[(size_t) i] > weights_[(size_t) j] ||
                (weights_[(size_t) i] == weights_[(size_t) j] && source_id_[(size_t) i] <= source_id_[(size_t) j]);
            keep = keep_i ? i : j;
            move = keep_i ? j : i;
            const double d = l2_distance(
                centers_.data() + (size_t) i * width_,
                centers_.data() + (size_t) j * width_, width_);
            merged_weight = weights_[(size_t) i] + weights_[(size_t) j];
            merged_transport = transport_[(size_t) i] + transport_[(size_t) j] +
                (double) weights_[(size_t) move] * d;
            merged_radius = std::max(radius_[(size_t) keep], radius_[(size_t) move] + d);
        }

        std::copy_n(centers_.data() + (size_t) keep * width_, width_,
                    centers_new.data() + (size_t) out * width_);
        std::copy_n(sketches_.data() + (size_t) keep * SKETCH_DIM, SKETCH_DIM,
                    sketches_new.data() + (size_t) out * SKETCH_DIM);
        center_norm2_new[(size_t) out] = center_norm2_[(size_t) keep];
        weights_new[(size_t) out] = merged_weight;
        transport_new[(size_t) out] = merged_transport;
        radius_new[(size_t) out] = merged_radius;
        source_id_new[(size_t) out] = source_id_[(size_t) keep];
        ++out;
    }
    if (out != target) throw std::runtime_error("geometric coreset compression produced wrong size");
    size_ = out;
    centers_.swap(centers_new);
    sketches_.swap(sketches_new);
    center_norm2_.swap(center_norm2_new);
    weights_.swap(weights_new);
    transport_.swap(transport_new);
    radius_.swap(radius_new);
    source_id_.swap(source_id_new);
}

std::vector<enp_geometric_cluster> enp_geometric_stream::finalize() {
    if (size_ > max_points_) compress_to(max_points_);
    std::vector<enp_geometric_cluster> out;
    out.reserve((size_t) size_);
    for (int i = 0; i < size_; ++i) {
        enp_geometric_cluster c;
        c.center.assign(centers_.data() + (size_t) i * width_, centers_.data() + (size_t) (i + 1) * width_);
        c.weight = weights_[(size_t) i];
        c.transport = transport_[(size_t) i];
        c.radius = radius_[(size_t) i];
        c.source_id = source_id_[(size_t) i];
        out.push_back(std::move(c));
    }
    std::stable_sort(out.begin(), out.end(), [](const enp_geometric_cluster & a, const enp_geometric_cluster & b) {
        return a.source_id < b.source_id;
    });
    return out;
}

std::vector<enp_geometric_cluster> enp_geometric_coarsen(
        const std::vector<enp_geometric_cluster> & input,
        int target) {
    if (input.empty() || target <= 0 || target > (int) input.size()) {
        throw std::runtime_error("invalid geometric coreset coarsen target");
    }
    std::vector<enp_geometric_cluster> current = input;
    const int width = (int) current.front().center.size();
    for (const auto & c : current) {
        if ((int) c.center.size() != width || c.weight <= 0 || c.transport < 0.0 || c.radius < 0.0) {
            throw std::runtime_error("invalid geometric coreset input cluster");
        }
    }
    while ((int) current.size() > target) {
        const int n = (int) current.size();
        const int next = std::max(target, (n + 1) / 2);
        const int merges = n - next;
        std::vector<std::array<float, SKETCH_DIM>> sketches((size_t) n);
        std::vector<double> norms((size_t) n, 0.0);
        for (int i = 0; i < n; ++i) {
            make_sketch(current[(size_t) i].center.data(), width, sketches[(size_t) i].data());
            norms[(size_t) i] = vector_norm2(current[(size_t) i].center.data(), width);
        }
        const auto pairs = choose_pairs(
            n, merges,
            [&](int i) { return sketches[(size_t) i].data(); },
            [&](int i) { return norms[(size_t) i]; },
            [&](int i) { return current[(size_t) i].source_id; });
        std::vector<int> mate((size_t) n, -1);
        for (const auto & [a, b] : pairs) {
            mate[(size_t) a] = b;
            mate[(size_t) b] = a;
        }
        std::vector<enp_geometric_cluster> reduced;
        reduced.reserve((size_t) next);
        for (int i = 0; i < n; ++i) {
            const int j = mate[(size_t) i];
            if (j >= 0 && i > j) continue;
            reduced.push_back(j >= 0 ? merge_clusters(current[(size_t) i], current[(size_t) j]) : current[(size_t) i]);
        }
        if ((int) reduced.size() != next) throw std::runtime_error("geometric coreset coarsen produced wrong size");
        current.swap(reduced);
    }
    std::stable_sort(current.begin(), current.end(), [](const enp_geometric_cluster & a, const enp_geometric_cluster & b) {
        return a.source_id < b.source_id;
    });
    return current;
}

double enp_geometric_total_weight(const std::vector<enp_geometric_cluster> & clusters) {
    long double sum = 0.0L;
    for (const auto & c : clusters) sum += c.weight;
    return (double) sum;
}

double enp_geometric_total_transport(const std::vector<enp_geometric_cluster> & clusters) {
    long double sum = 0.0L;
    for (const auto & c : clusters) sum += c.transport;
    return (double) sum;
}

double enp_geometric_max_radius(const std::vector<enp_geometric_cluster> & clusters) {
    double out = 0.0;
    for (const auto & c : clusters) out = std::max(out, c.radius);
    return out;
}

} // namespace qwen35_prune
