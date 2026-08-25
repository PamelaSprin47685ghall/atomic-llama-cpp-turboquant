#include "gdn-joint-replay.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace qwen35_prune;

int main() {
    gdn_joint_replay_data d;
    d.tokens = 1;
    d.conv_silu.assign(8192, 0.0f);
    d.beta.assign(32, 1.0f);
    d.gate_log_decay.assign(32, -1.0f);
    d.z.assign(4096, 1.0f);
    d.teacher_final.assign(4096, 0.0f);

    gdn_joint_candidate c;
    for (int h = 0; h < 16; ++h) {
        c.qk[(size_t) h].resize(64);
        for (int i = 0; i < 64; ++i) c.qk[(size_t) h][(size_t) i] = i;
    }
    c.v.resize(64);
    for (int i = 0; i < 64; ++i) c.v[(size_t) i] = i;

    // q=k=e0 and v=e0 for every head. From zero state with beta=1,
    // candidate recurrent output at retained coordinate 0 is 1/sqrt(64).
    for (int h = 0; h < 16; ++h) {
        d.conv_silu[(size_t) h * 128] = 1.0f;
        d.conv_silu[2048 + (size_t) h * 128] = 1.0f;
    }
    for (int h = 0; h < 32; ++h) d.conv_silu[4096 + (size_t) h * 128] = 1.0f;

    gdn_joint_replay_weights w;
    w.norm_gamma.fill(1.0f);
    w.ssm_out_column_energy.fill(1.0);
    w.rms_epsilon = 0.0;
    // RMS normalization of a 64-vector with only x0=1/8 produces x0=8.
    // Gate SiLU(1) then gives the teacher feature below.
    const double silu1 = 1.0 / (1.0 + std::exp(-1.0));
    for (int h = 0; h < 32; ++h) d.teacher_final[(size_t) h * 128] = (float) (8.0 * silu1);

    const auto s = gdn_score_joint_candidate(d, c, w);
    if (!(s.retained_dynamic_sse < 1e-10)) return 1;
    // All omitted teacher coordinates are zero, so the complete score is zero.
    if (!(s.weighted_sse < 1e-10)) return 1;
    std::cout << "gdn-joint-replay: OK\n";
    return 0;
}
