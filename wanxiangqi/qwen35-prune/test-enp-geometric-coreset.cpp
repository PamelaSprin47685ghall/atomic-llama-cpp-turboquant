#include "enp-geometric-coreset.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

using namespace qwen35_prune;

static void require(bool ok, const char * msg) {
    if (!ok) throw std::runtime_error(msg);
}

int main() {
    constexpr int width = 4;
    enp_geometric_stream stream(width, 4);
    std::vector<std::vector<float>> input;
    for (int i = 0; i < 19; ++i) {
        std::vector<float> x = {
            (float) i,
            (float) (i % 3),
            (float) ((i * 7) % 5),
            (float) (i % 2 ? -1 : 1),
        };
        input.push_back(x);
        stream.add(x.data());
    }
    const auto c4 = stream.finalize();
    require(c4.size() == 4, "stream did not finish at requested coreset size");
    require((int64_t) enp_geometric_total_weight(c4) == 19, "coreset weight not conserved");
    require(enp_geometric_total_transport(c4) >= 0.0, "negative transport");
    require(enp_geometric_max_radius(c4) >= 0.0, "negative radius");

    std::set<std::vector<float>> observed(input.begin(), input.end());
    for (const auto & c : c4) {
        require(observed.count(c.center) == 1, "representative is not an observed hidden-state medoid");
        require(c.weight > 0, "non-positive cluster weight");
    }

    const double transport4 = enp_geometric_total_transport(c4);
    const double radius4 = enp_geometric_max_radius(c4);
    const auto c2 = enp_geometric_coarsen(c4, 2);
    require(c2.size() == 2, "adaptive coarsening size mismatch");
    require((int64_t) enp_geometric_total_weight(c2) == 19, "coarsening lost weight");
    require(enp_geometric_total_transport(c2) + 1e-12 >= transport4, "coarsening decreased transport bound");
    require(enp_geometric_max_radius(c2) + 1e-12 >= radius4, "coarsening decreased radius bound");
    for (const auto & c : c2) require(observed.count(c.center) == 1, "coarsened representative stopped being a medoid");

    // Exact-fallback mode: if R is at least the finite population, no point is
    // merged and the geometric approximation collapses to the full empirical
    // distribution with zero transport/radius error.
    enp_geometric_stream exact(width, 32);
    for (const auto & x : input) exact.add(x.data());
    const auto ce = exact.finalize();
    require(ce.size() == input.size(), "full-population coreset unexpectedly merged points");
    require((int64_t) enp_geometric_total_weight(ce) == 19, "full-population weight mismatch");
    require(enp_geometric_total_transport(ce) == 0.0, "full-population fallback has transport error");
    require(enp_geometric_max_radius(ce) == 0.0, "full-population fallback has radius error");
    for (const auto & c : ce) require(c.weight == 1, "full-population representative is not an exact atom");

    enp_geometric_stream identical(width, 2);
    const float same[width] = {1.0f, -2.0f, 3.0f, 4.0f};
    for (int i = 0; i < 17; ++i) identical.add(same);
    const auto ci = identical.finalize();
    require((int64_t) enp_geometric_total_weight(ci) == 17, "identical-point weight mismatch");
    require(enp_geometric_total_transport(ci) == 0.0, "identical points acquired transport error");
    require(enp_geometric_max_radius(ci) == 0.0, "identical points acquired radius error");

    std::cout << "enp-geometric-coreset: OK\n";
    return 0;
}
