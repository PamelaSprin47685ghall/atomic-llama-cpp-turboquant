#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace qwen35_prune {

constexpr int GDN_JOINT_DIM_SRC = 128;
constexpr int GDN_JOINT_DIM_DST = 64;
constexpr int GDN_JOINT_QK_HEADS = 16;
constexpr int GDN_JOINT_V_HEADS = 32;

struct gdn_joint_replay_data {
    int tokens = 0;
    // Post-conv, post-SiLU, before Q/K L2 normalization. Layout per token:
    // Q[16*128], K[16*128], V[32*128].
    std::vector<float> conv_silu;
    std::vector<float> beta;
    std::vector<float> gate_log_decay;
    std::vector<float> z;
    // Teacher feature after recurrent output -> RMSNorm(gamma) -> SiLU(z),
    // before ssm_out. Layout per token: 32*128.
    std::vector<float> teacher_final;
};

struct gdn_joint_candidate {
    std::array<std::vector<int>, GDN_JOINT_QK_HEADS> qk;
    std::vector<int> v;
};

struct gdn_joint_replay_weights {
    std::array<float, GDN_JOINT_DIM_SRC> norm_gamma {};
    std::array<double, GDN_JOINT_V_HEADS * GDN_JOINT_DIM_SRC> ssm_out_column_energy {};
    double rms_epsilon = 1e-6;
    double l2_epsilon = 1e-6;
};

struct gdn_joint_replay_score {
    double weighted_sse = 0.0;
    double teacher_energy = 0.0;
    double retained_dynamic_sse = 0.0;
    double omitted_teacher_energy = 0.0;
    uint64_t feature_values = 0;

    double relative_error() const;
};

gdn_joint_replay_score gdn_score_joint_candidate(
        const gdn_joint_replay_data & data,
        const gdn_joint_candidate & candidate,
        const gdn_joint_replay_weights & weights);

} // namespace qwen35_prune
