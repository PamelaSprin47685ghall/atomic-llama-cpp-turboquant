#include "gdn-joint-replay.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace qwen35_prune {
namespace {

static void validate_subset(const std::vector<int> & s, int expected, const char * what) {
    if ((int) s.size() != expected) throw std::runtime_error(std::string(what) + " subset size mismatch");
    std::unordered_set<int> seen;
    for (int x : s) {
        if (x < 0 || x >= GDN_JOINT_DIM_SRC || !seen.insert(x).second) {
            throw std::runtime_error(std::string(what) + " subset is invalid");
        }
    }
}

static double silu(double x) {
    if (x >= 0.0) return x / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return x * e / (1.0 + e);
}

} // namespace

double gdn_joint_replay_score::relative_error() const {
    return teacher_energy > 0.0 ? std::sqrt(weighted_sse / teacher_energy) : 0.0;
}

gdn_joint_replay_score gdn_score_joint_candidate(
        const gdn_joint_replay_data & data,
        const gdn_joint_candidate & candidate,
        const gdn_joint_replay_weights & weights) {
    if (data.tokens <= 0 ||
        data.conv_silu.size() != (size_t) data.tokens * 8192 ||
        data.beta.size() != (size_t) data.tokens * GDN_JOINT_V_HEADS ||
        data.gate_log_decay.size() != (size_t) data.tokens * GDN_JOINT_V_HEADS ||
        data.z.size() != (size_t) data.tokens * GDN_JOINT_V_HEADS * GDN_JOINT_DIM_SRC ||
        data.teacher_final.size() != (size_t) data.tokens * GDN_JOINT_V_HEADS * GDN_JOINT_DIM_SRC) {
        throw std::runtime_error("GDN joint replay data shape mismatch");
    }
    if (!(weights.rms_epsilon >= 0.0) || !(weights.l2_epsilon >= 0.0)) {
        throw std::runtime_error("GDN joint replay epsilon is invalid");
    }
    for (const auto & s : candidate.qk) validate_subset(s, GDN_JOINT_DIM_DST, "Q/K");
    validate_subset(candidate.v, GDN_JOINT_DIM_DST, "V");

    // State is key-coordinate x value-coordinate for each value head.
    std::vector<double> state((size_t) GDN_JOINT_V_HEADS * GDN_JOINT_DIM_DST * GDN_JOINT_DIM_DST, 0.0);
    std::array<std::array<double, GDN_JOINT_DIM_DST>, GDN_JOINT_QK_HEADS> qn {};
    std::array<std::array<double, GDN_JOINT_DIM_DST>, GDN_JOINT_QK_HEADS> kn {};
    std::array<double, GDN_JOINT_DIM_DST> vv {};
    std::array<double, GDN_JOINT_DIM_DST> pred {};
    std::array<double, GDN_JOINT_DIM_DST> sq {};
    std::array<double, GDN_JOINT_DIM_DST> delta {};
    std::array<double, GDN_JOINT_DIM_DST> output {};
    gdn_joint_replay_score score;
    const double inv_sqrt_dim = 1.0 / std::sqrt((double) GDN_JOINT_DIM_DST);

    for (int t = 0; t < data.tokens; ++t) {
        const float * conv = data.conv_silu.data() + (size_t) t * 8192;
        for (int h = 0; h < GDN_JOINT_QK_HEADS; ++h) {
            long double q2 = 0.0, k2 = 0.0;
            for (int j = 0; j < GDN_JOINT_DIM_DST; ++j) {
                const int src = candidate.qk[(size_t) h][(size_t) j];
                const double q = conv[(size_t) h * GDN_JOINT_DIM_SRC + src];
                const double k = conv[2048 + (size_t) h * GDN_JOINT_DIM_SRC + src];
                qn[(size_t) h][(size_t) j] = q;
                kn[(size_t) h][(size_t) j] = k;
                q2 += q * q;
                k2 += k * k;
            }
            const double qi = 1.0 / std::max(std::sqrt((double) q2), weights.l2_epsilon);
            const double ki = 1.0 / std::max(std::sqrt((double) k2), weights.l2_epsilon);
            for (int j = 0; j < GDN_JOINT_DIM_DST; ++j) {
                qn[(size_t) h][(size_t) j] *= qi;
                kn[(size_t) h][(size_t) j] *= ki;
            }
        }

        const float * teacher = data.teacher_final.data() +
                (size_t) t * GDN_JOINT_V_HEADS * GDN_JOINT_DIM_SRC;
        // The omitted-coordinate term depends only on V selection. Include it
        // once per token before scoring retained dynamic error.
        std::array<uint8_t, GDN_JOINT_DIM_SRC> retained {};
        for (int x : candidate.v) retained[(size_t) x] = 1;
        for (int vh = 0; vh < GDN_JOINT_V_HEADS; ++vh) {
            for (int i = 0; i < GDN_JOINT_DIM_SRC; ++i) {
                const double target = teacher[(size_t) vh * GDN_JOINT_DIM_SRC + i];
                const double w = weights.ssm_out_column_energy[(size_t) vh * GDN_JOINT_DIM_SRC + i];
                const double e = w * target * target;
                score.teacher_energy += e;
                if (!retained[(size_t) i]) {
                    score.omitted_teacher_energy += e;
                    score.weighted_sse += e;
                }
            }
        }

        for (int vh = 0; vh < GDN_JOINT_V_HEADS; ++vh) {
            const int qh = vh % GDN_JOINT_QK_HEADS;
            const auto & q = qn[(size_t) qh];
            const auto & k = kn[(size_t) qh];
            for (int j = 0; j < GDN_JOINT_DIM_DST; ++j) {
                vv[(size_t) j] = conv[4096 + (size_t) vh * GDN_JOINT_DIM_SRC +
                                             candidate.v[(size_t) j]];
                pred[(size_t) j] = 0.0;
                sq[(size_t) j] = 0.0;
            }

            double * s = state.data() + (size_t) vh * GDN_JOINT_DIM_DST * GDN_JOINT_DIM_DST;
            const double decay = std::exp(data.gate_log_decay[(size_t) t * GDN_JOINT_V_HEADS + vh]);
            const double beta = data.beta[(size_t) t * GDN_JOINT_V_HEADS + vh];
            double kq = 0.0;
            for (int i = 0; i < GDN_JOINT_DIM_DST; ++i) kq += k[(size_t) i] * q[(size_t) i];

            // Compute S_old^T k and S_old^T q together for cache locality.
            for (int i = 0; i < GDN_JOINT_DIM_DST; ++i) {
                const double ki = k[(size_t) i];
                const double qi = q[(size_t) i];
                const double * row = s + (size_t) i * GDN_JOINT_DIM_DST;
                for (int j = 0; j < GDN_JOINT_DIM_DST; ++j) {
                    pred[(size_t) j] += row[j] * ki;
                    sq[(size_t) j] += row[j] * qi;
                }
            }
            for (int j = 0; j < GDN_JOINT_DIM_DST; ++j) {
                pred[(size_t) j] *= decay;
                sq[(size_t) j] *= decay;
                delta[(size_t) j] = beta * (vv[(size_t) j] - pred[(size_t) j]);
                output[(size_t) j] = (sq[(size_t) j] + delta[(size_t) j] * kq) * inv_sqrt_dim;
            }
            for (int i = 0; i < GDN_JOINT_DIM_DST; ++i) {
                const double ki = k[(size_t) i];
                double * row = s + (size_t) i * GDN_JOINT_DIM_DST;
                for (int j = 0; j < GDN_JOINT_DIM_DST; ++j) {
                    row[j] = decay * row[j] + ki * delta[(size_t) j];
                }
            }

            long double norm2 = 0.0;
            for (double x : output) norm2 += x * x;
            const double rms_scale = 1.0 / std::sqrt((double) (norm2 / GDN_JOINT_DIM_DST) + weights.rms_epsilon);
            const float * z = data.z.data() +
                    ((size_t) t * GDN_JOINT_V_HEADS + vh) * GDN_JOINT_DIM_SRC;
            for (int j = 0; j < GDN_JOINT_DIM_DST; ++j) {
                const int src = candidate.v[(size_t) j];
                const double feature = output[(size_t) j] * rms_scale *
                        weights.norm_gamma[(size_t) src] * silu(z[src]);
                const double target = teacher[(size_t) vh * GDN_JOINT_DIM_SRC + src];
                const double w = weights.ssm_out_column_energy[(size_t) vh * GDN_JOINT_DIM_SRC + src];
                const double err = feature - target;
                const double e = w * err * err;
                score.retained_dynamic_sse += e;
                score.weighted_sse += e;
                ++score.feature_values;
            }
        }
    }
    return score;
}

} // namespace qwen35_prune
