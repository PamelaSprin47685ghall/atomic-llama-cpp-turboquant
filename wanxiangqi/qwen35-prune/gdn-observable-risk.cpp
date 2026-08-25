#include "gdn-observable-risk.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qwen35_prune {
namespace {

static double logsumexp_scaled(const float * logits, int n, double inv_temperature) {
    if (!logits || n <= 0) throw std::runtime_error("observable risk logsumexp received empty logits");
    double max_value = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        const double v = (double) logits[i] * inv_temperature;
        if (!std::isfinite(v)) throw std::runtime_error("observable risk encountered non-finite logit");
        max_value = std::max(max_value, v);
    }
    long double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += std::exp((double) logits[i] * inv_temperature - max_value);
    return max_value + std::log((double) sum);
}

static double logsumexp_mapped_scaled(
        const float * teacher_logits,
        int teacher_vocab,
        const int * map,
        int student_vocab,
        double inv_temperature) {
    double max_value = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < student_vocab; ++j) {
        const int id = map[j];
        if (id < 0 || id >= teacher_vocab) throw std::runtime_error("observable risk vocabulary map out of range");
        const double v = (double) teacher_logits[id] * inv_temperature;
        if (!std::isfinite(v)) throw std::runtime_error("observable risk encountered non-finite mapped teacher logit");
        max_value = std::max(max_value, v);
    }
    long double sum = 0.0;
    for (int j = 0; j < student_vocab; ++j) {
        sum += std::exp((double) teacher_logits[map[j]] * inv_temperature - max_value);
    }
    return max_value + std::log((double) sum);
}

} // namespace

void gdn_observable_risk_sum::add(const gdn_observable_risk_sample & sample) {
    ++samples;
    student_nll += sample.student_nll;
    teacher_conditional_kl += sample.teacher_conditional_kl;
    teacher_conditional_entropy += sample.teacher_conditional_entropy;
    teacher_retained_mass += sample.teacher_retained_mass;
    top1_agree += sample.top1_agree ? 1u : 0u;
}

gdn_observable_risk_sample gdn_measure_observable_risk_sample(
        const float * teacher_logits,
        int teacher_vocab,
        const float * student_logits,
        int student_vocab,
        const int * student_to_teacher,
        int next_student_token,
        double distill_temperature) {
    if (!teacher_logits || !student_logits || !student_to_teacher || teacher_vocab <= 0 || student_vocab <= 0) {
        throw std::runtime_error("observable risk received invalid logits/vocabulary");
    }
    if (next_student_token < 0 || next_student_token >= student_vocab) {
        throw std::runtime_error("observable risk next token is outside student vocabulary");
    }
    if (!(distill_temperature > 0.0) || !std::isfinite(distill_temperature)) {
        throw std::runtime_error("observable risk temperature must be finite and positive");
    }

    const double student_logz = logsumexp_scaled(student_logits, student_vocab, 1.0);
    gdn_observable_risk_sample result;
    result.student_nll = student_logz - (double) student_logits[next_student_token];

    const double teacher_full_logz = logsumexp_scaled(teacher_logits, teacher_vocab, 1.0);
    const double teacher_retained_logz = logsumexp_mapped_scaled(
            teacher_logits, teacher_vocab, student_to_teacher, student_vocab, 1.0);
    result.teacher_retained_mass = std::exp(teacher_retained_logz - teacher_full_logz);

    const double inv_temperature = 1.0 / distill_temperature;
    const double teacher_conditional_logz = logsumexp_mapped_scaled(
            teacher_logits, teacher_vocab, student_to_teacher, student_vocab, inv_temperature);
    const double student_distill_logz = logsumexp_scaled(student_logits, student_vocab, inv_temperature);

    int teacher_top1 = 0;
    int student_top1 = 0;
    double teacher_top1_logit = -std::numeric_limits<double>::infinity();
    double student_top1_logit = -std::numeric_limits<double>::infinity();
    long double kl = 0.0;
    long double entropy = 0.0;
    for (int j = 0; j < student_vocab; ++j) {
        const double t_scaled = (double) teacher_logits[student_to_teacher[j]] * inv_temperature;
        const double s_scaled = (double) student_logits[j] * inv_temperature;
        const double log_pt = t_scaled - teacher_conditional_logz;
        const double log_ps = s_scaled - student_distill_logz;
        const double pt = std::exp(log_pt);
        kl += (long double) pt * (log_pt - log_ps);
        entropy -= (long double) pt * log_pt;
        if (t_scaled > teacher_top1_logit) {
            teacher_top1_logit = t_scaled;
            teacher_top1 = j;
        }
        if (s_scaled > student_top1_logit) {
            student_top1_logit = s_scaled;
            student_top1 = j;
        }
    }
    result.teacher_conditional_kl = std::max(0.0, (double) kl);
    result.teacher_conditional_entropy = (double) entropy;
    result.top1_agree = teacher_top1 == student_top1;
    return result;
}

gdn_prepared_observable_teacher gdn_prepare_observable_teacher(
        const float * teacher_logits,
        int teacher_vocab,
        int student_vocab,
        const int * student_to_teacher,
        double distill_temperature) {
    if (!teacher_logits || !student_to_teacher || teacher_vocab <= 0 || student_vocab <= 0) {
        throw std::runtime_error("observable teacher preparation received invalid logits/vocabulary");
    }
    if (!(distill_temperature > 0.0) || !std::isfinite(distill_temperature)) {
        throw std::runtime_error("observable teacher preparation temperature must be finite and positive");
    }

    gdn_prepared_observable_teacher out;
    out.student_vocab = student_vocab;
    out.distill_temperature = distill_temperature;
    out.teacher_probability.resize((size_t) student_vocab);

    const double teacher_full_logz = logsumexp_scaled(teacher_logits, teacher_vocab, 1.0);
    const double teacher_retained_logz = logsumexp_mapped_scaled(
            teacher_logits, teacher_vocab, student_to_teacher, student_vocab, 1.0);
    out.teacher_retained_mass = std::exp(teacher_retained_logz - teacher_full_logz);

    const double inv_temperature = 1.0 / distill_temperature;
    const double teacher_conditional_logz = logsumexp_mapped_scaled(
            teacher_logits, teacher_vocab, student_to_teacher, student_vocab, inv_temperature);
    long double entropy = 0.0;
    double top1_logit = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < student_vocab; ++j) {
        const int id = student_to_teacher[j];
        if (id < 0 || id >= teacher_vocab) throw std::runtime_error("observable teacher preparation map out of range");
        const double scaled = (double) teacher_logits[id] * inv_temperature;
        const double logp = scaled - teacher_conditional_logz;
        const double p = std::exp(logp);
        out.teacher_probability[(size_t) j] = (float) p;
        entropy -= (long double) p * logp;
        if (scaled > top1_logit) {
            top1_logit = scaled;
            out.teacher_top1 = j;
        }
    }
    out.teacher_conditional_entropy = (double) entropy;
    return out;
}

gdn_observable_risk_sample gdn_measure_observable_risk_prepared(
        const gdn_prepared_observable_teacher & teacher,
        const float * student_logits,
        int next_student_token) {
    if (!student_logits || teacher.student_vocab <= 0 ||
        teacher.teacher_probability.size() != (size_t) teacher.student_vocab) {
        throw std::runtime_error("prepared observable risk received invalid teacher/student data");
    }
    if (next_student_token < 0 || next_student_token >= teacher.student_vocab) {
        throw std::runtime_error("prepared observable risk next token is outside student vocabulary");
    }
    if (!(teacher.distill_temperature > 0.0) || !std::isfinite(teacher.distill_temperature)) {
        throw std::runtime_error("prepared observable risk teacher temperature is invalid");
    }

    gdn_observable_risk_sample out;
    const double student_logz = logsumexp_scaled(student_logits, teacher.student_vocab, 1.0);
    out.student_nll = student_logz - (double) student_logits[next_student_token];
    out.teacher_retained_mass = teacher.teacher_retained_mass;
    out.teacher_conditional_entropy = teacher.teacher_conditional_entropy;

    const double inv_temperature = 1.0 / teacher.distill_temperature;
    const double student_distill_logz = logsumexp_scaled(
            student_logits, teacher.student_vocab, inv_temperature);
    int student_top1 = 0;
    double student_top1_logit = -std::numeric_limits<double>::infinity();
    long double cross_entropy = 0.0;
    for (int j = 0; j < teacher.student_vocab; ++j) {
        const double scaled = (double) student_logits[j] * inv_temperature;
        const double log_ps = scaled - student_distill_logz;
        cross_entropy -= (long double) teacher.teacher_probability[(size_t) j] * log_ps;
        if (scaled > student_top1_logit) {
            student_top1_logit = scaled;
            student_top1 = j;
        }
    }
    out.teacher_conditional_kl = std::max(
            0.0, (double) cross_entropy - teacher.teacher_conditional_entropy);
    out.top1_agree = teacher.teacher_top1 == student_top1;
    return out;
}

} // namespace qwen35_prune
