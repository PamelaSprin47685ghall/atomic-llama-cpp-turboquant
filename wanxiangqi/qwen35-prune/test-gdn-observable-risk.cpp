#include "gdn-observable-risk.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace qwen35_prune;

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    const float teacher[] = {2.0f, -3.0f, 1.0f, 0.5f, -4.0f};
    const int map[] = {0, 2, 3};
    const float student_same[] = {2.0f, 1.0f, 0.5f};
    const auto exact = gdn_measure_observable_risk_sample(
            teacher, 5, student_same, 3, map, 1, 1.0);
    require(exact.teacher_conditional_kl < 1e-12, "identical retained logits should have zero KL");
    require(exact.teacher_retained_mass > 0.99 && exact.teacher_retained_mass < 1.0,
            "retained teacher mass is out of expected range");
    require(exact.top1_agree, "identical retained logits should agree on top1");

    // Logit shifts must not change either categorical distribution.
    const float student_shifted[] = {12.0f, 11.0f, 10.5f};
    const auto shifted = gdn_measure_observable_risk_sample(
            teacher, 5, student_shifted, 3, map, 1, 1.0);
    require(shifted.teacher_conditional_kl < 1e-12, "logit shift changed observable KL");
    require(std::abs(shifted.student_nll - exact.student_nll) < 1e-12,
            "logit shift changed student NLL");

    const float student_bad[] = {-2.0f, 3.0f, 0.0f};
    const auto bad = gdn_measure_observable_risk_sample(
            teacher, 5, student_bad, 3, map, 1, 1.0);
    require(bad.teacher_conditional_kl > 0.5, "mismatched logits should have positive KL");
    require(!bad.top1_agree, "mismatched logits unexpectedly agree on top1");

    const auto prepared_teacher = gdn_prepare_observable_teacher(teacher, 5, 3, map, 1.0);
    const auto prepared_bad = gdn_measure_observable_risk_prepared(prepared_teacher, student_bad, 1);
    require(std::abs(prepared_bad.student_nll - bad.student_nll) < 1e-12,
            "prepared teacher changed student NLL");
    require(std::abs(prepared_bad.teacher_conditional_kl - bad.teacher_conditional_kl) < 2e-7,
            "prepared teacher changed conditional KL");
    require(std::abs(prepared_bad.teacher_conditional_entropy - bad.teacher_conditional_entropy) < 1e-12,
            "prepared teacher changed conditional entropy");
    require(std::abs(prepared_bad.teacher_retained_mass - bad.teacher_retained_mass) < 1e-12,
            "prepared teacher changed retained mass");
    require(prepared_bad.top1_agree == bad.top1_agree,
            "prepared teacher changed top1 agreement");

    gdn_observable_risk_sum sum;
    sum.add(exact);
    sum.add(bad);
    require(sum.samples == 2, "observable risk accumulator sample count mismatch");
    std::cout << "gdn-observable-risk: OK\n";
    return 0;
}
