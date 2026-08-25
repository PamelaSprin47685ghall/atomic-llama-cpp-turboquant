#pragma once

#include <cstdint>
#include <vector>

namespace qwen35_prune {

struct gdn_observable_risk_sample {
    double student_nll = 0.0;
    double teacher_conditional_kl = 0.0;
    double teacher_conditional_entropy = 0.0;
    double teacher_retained_mass = 0.0;
    bool top1_agree = false;
};

struct gdn_observable_risk_sum {
    uint64_t samples = 0;
    double student_nll = 0.0;
    double teacher_conditional_kl = 0.0;
    double teacher_conditional_entropy = 0.0;
    double teacher_retained_mass = 0.0;
    uint64_t top1_agree = 0;

    void add(const gdn_observable_risk_sample & sample);
};

struct gdn_prepared_observable_teacher {
    int student_vocab = 0;
    double distill_temperature = 1.0;
    double teacher_conditional_entropy = 0.0;
    double teacher_retained_mass = 0.0;
    int teacher_top1 = -1;
    std::vector<float> teacher_probability;
};


// Compare the *observable* next-token functions of teacher and student.
// The teacher distribution is restricted to the student's retained vocabulary
// and renormalized before KL is computed. This conditions out the separately
// fixed vocabulary-pruning decision and measures only the function mismatch
// that the GDN weights can actually repair.
gdn_observable_risk_sample gdn_measure_observable_risk_sample(
        const float * teacher_logits,
        int teacher_vocab,
        const float * student_logits,
        int student_vocab,
        const int * student_to_teacher,
        int next_student_token,
        double distill_temperature);

// Prepare the teacher side once when many student candidates are evaluated on
// the same empirical prefixes. The cached distribution is exactly the teacher
// distribution restricted to the retained student vocabulary and normalized
// at distill_temperature. Subsequent candidate scores therefore require only
// a student forward pass.
gdn_prepared_observable_teacher gdn_prepare_observable_teacher(
        const float * teacher_logits,
        int teacher_vocab,
        int student_vocab,
        const int * student_to_teacher,
        double distill_temperature);

gdn_observable_risk_sample gdn_measure_observable_risk_prepared(
        const gdn_prepared_observable_teacher & teacher,
        const float * student_logits,
        int next_student_token);

} // namespace qwen35_prune
