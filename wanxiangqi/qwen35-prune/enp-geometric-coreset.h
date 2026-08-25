#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace qwen35_prune {

struct enp_geometric_cluster {
    std::vector<float> center;
    int64_t weight = 0;
    double transport = 0.0;
    double radius = 0.0;
    uint64_t source_id = 0;
};

// Task-blind streaming hidden-state coreset.  Representatives are always
// medoids (actual observed hidden states).  Whenever two weighted clusters are
// merged, the losing cluster is transported to the heavier medoid and the
// exact original-space L2 distance is accumulated.  Therefore transport and
// radius remain deterministic upper bounds independent of the sketch used to
// choose merge candidates.
class enp_geometric_stream {
public:
    enp_geometric_stream() = default;
    enp_geometric_stream(int width, int max_points);

    void initialize(int width, int max_points);
    void add(const float * x);
    std::vector<enp_geometric_cluster> finalize();

    int width() const { return width_; }
    int max_points() const { return max_points_; }
    uint64_t population() const { return population_; }
    double max_norm2() const { return max_norm2_; }

private:
    // Must match the deterministic sketch used by the implementation.
    static constexpr int SKETCH_DIM = 16;

    int width_ = 0;
    int max_points_ = 0;
    int size_ = 0;
    uint64_t population_ = 0;
    double max_norm2_ = 0.0;

    std::vector<float> centers_;
    std::vector<float> sketches_;
    std::vector<double> center_norm2_;
    std::vector<int64_t> weights_;
    std::vector<double> transport_;
    std::vector<double> radius_;
    std::vector<uint64_t> source_id_;

    void compress_to(int target);
};

// Deterministically coarsen an already certified coreset.  The returned
// representatives are a subset of the input medoids and inherit/increase the
// original transport/radius bounds, so the result can be used for adaptive-R
// ENP scoring without weakening the proof chain.
std::vector<enp_geometric_cluster> enp_geometric_coarsen(
        const std::vector<enp_geometric_cluster> & input,
        int target);

double enp_geometric_total_weight(const std::vector<enp_geometric_cluster> & clusters);
double enp_geometric_total_transport(const std::vector<enp_geometric_cluster> & clusters);
double enp_geometric_max_radius(const std::vector<enp_geometric_cluster> & clusters);

} // namespace qwen35_prune
