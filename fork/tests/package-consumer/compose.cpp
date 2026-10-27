#include <tq/compose/sampler.h>

#include <stdexcept>

int main() {
    const float values[] = {1.0f, 3.0f, 2.0f, -1.0f};
    tq::compose::sampling_options options;
    options.top_k = 2;
    tq::compose::sampler sample(options);
    if (sample.sample(values, 4) != 1 || sample.stats().compact_rows != 1) {
        throw std::runtime_error("installed composer failed");
    }
}
