// C++ port of the 16 check groups of `tp5_local_plan.py`
// (SHA-256 59129982...59e657, guide 54.10) against the recorded fixture
// numbers (guide 54.6, 54.8, 54.9). The dense-GLU oracle and the two-row
// gather identity check are ported verbatim in arithmetic; the Python
// reference keeps its own recorded outputs, and the live cross-check
// (`crosscheck-plan.py`) compares this port against the Python executable.

#include <tq/relay/detail/arithmetic.h>
#include <tq/relay/plan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using tq::relay::compile_glu_plan;
using tq::relay::delivery;
using tq::relay::glu_plan_result;
using tq::relay::glu_spec;
using tq::relay::index_range;
using tq::relay::join_operation;
using tq::relay::output_distribution;
using tq::relay::plan_decision;
using tq::relay::plan_error;
using tq::relay::shard_strategy;
using tq::relay::storage_layout;

namespace {

void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

glu_spec baseline_spec() {
    glu_spec spec;
    spec.ranks          = 5;
    spec.hidden         = 2048;
    spec.intermediate   = 512;
    spec.active_rows    = 2;
    spec.block_elements = 256;
    spec.block_bytes    = 144;
    spec.layout         = storage_layout::independent_blocks_no_padding;
    spec.strategy       = shard_strategy::gather_intermediate;
    spec.output         = output_distribution::sharded;
    return spec;
}

void expect_rejection(const glu_spec & spec,
                      plan_decision    decision,
                      plan_error       code,
                      const char *     field,
                      const char *     message) {
    const auto result = compile_glu_plan(spec);
    check(result.decision == decision, message);
    check(result.error == code, message);
    check(result.error_field == field, message);
    check(result.ranks.empty() && result.collectives.empty(), "rejection must carry no partial plan");
}

void check_baseline_numbers() {
    const auto plan = compile_glu_plan(baseline_spec());
    check(plan.decision == plan_decision::feasible, "baseline must be feasible");
    // Recorded outputs, guide 54.6 and 54.8.
    const index_range middle[] = {
        { 0,   103 },
        { 103, 206 },
        { 206, 308 },
        { 308, 410 },
        { 410, 512 }
    };
    const index_range hidden[] = {
        { 0,    410  },
        { 410,  820  },
        { 820,  1230 },
        { 1230, 1639 },
        { 1639, 2048 }
    };
    const std::uint64_t weights[] = { 355392, 355392, 353088, 352800, 352800 };
    for (int rank = 0; rank < 5; ++rank) {
        check(plan.ranks[rank].rank == static_cast<std::uint32_t>(rank), "rank identity drifted");
        check(plan.ranks[rank].gate_up_output == middle[rank], "wrong middle cover");
        check(plan.ranks[rank].down_output == hidden[rank], "wrong hidden cover");
        check(plan.ranks[rank].down_input == index_range{ 0, 512 }, "down must read the full middle");
        check(plan.ranks[rank].source_weight_bytes == weights[rank], "wrong source weight accounting");
    }
    check(plan.totals.source_weight_bytes == 1769472, "wrong baseline weight total");
    check(plan.totals.linear_macs == 6291456, "wrong baseline MAC total");
    check(plan.totals.endpoint_send_bytes == 4096, "wrong baseline send total");
    check(plan.totals.endpoint_receive_bytes == 20480, "wrong baseline receive total");
    check(plan.input_bytes_per_rank == 16384, "wrong replicated input bytes");
    check(plan.output_ranges == std::vector<index_range>(hidden, hidden + 5), "output boundary drifted");
    check(plan.collectives.size() == 1, "baseline has extra collectives");
    check(plan.collectives[0].operation == join_operation::gather, "wrong middle operation");
    check(plan.collectives[0].distribution == delivery::broadcast, "wrong middle delivery");
    check(plan.collectives[0].assembled_value_bytes == 4096, "wrong assembled middle bytes");
}

void check_replicated_output_counts_the_final_gather() {
    auto spec       = baseline_spec();
    spec.output     = output_distribution::replicated;
    const auto plan = compile_glu_plan(spec);
    check(plan.decision == plan_decision::feasible, "replicated must be feasible");
    check(plan.collectives.size() == 2, "missing final hidden gather");
    check(plan.totals.endpoint_send_bytes == 20480, "missing final send bytes");
    check(plan.totals.endpoint_receive_bytes == 102400, "missing final receive bytes");
    check(plan.totals.source_weight_bytes == 1769472, "boundary changed the weights");
    check(plan.totals.linear_macs == 6291456, "boundary changed the work");
}

void check_split_512_is_infeasible() {
    auto spec     = baseline_spec();
    spec.strategy = shard_strategy::split_intermediate;
    expect_rejection(spec, plan_decision::infeasible, plan_error::insufficient_units, "intermediate",
                     "two quantization blocks cannot become five nonempty input shards");
}

void check_aligned_paired_partition() {
    for (const auto output : { output_distribution::sharded, output_distribution::replicated }) {
        auto spec         = baseline_spec();
        spec.intermediate = 1280;
        spec.strategy     = shard_strategy::split_intermediate;
        spec.output       = output;
        const auto plan   = compile_glu_plan(spec);
        check(plan.decision == plan_decision::feasible, "aligned split must be feasible");
        check(plan.totals.source_weight_bytes == 4423680, "wrong paired source bytes");
        check(plan.totals.endpoint_send_bytes == 81920, "wrong sum input bytes");
        const std::uint64_t expected_receive = output == output_distribution::sharded ? 16384 : 81920;
        check(plan.totals.endpoint_receive_bytes == expected_receive, "sum output distribution drifted");
        check(plan.collectives[0].operation == join_operation::sum, "split must SUM, not gather");
        for (const auto & rank : plan.ranks) {
            check(rank.down_input == rank.gate_up_output, "paired split must share the range");
            check(rank.down_input.end - rank.down_input.begin == 256, "block-aligned split drifted");
        }
    }
}

void check_row_block_alignment() {
    auto spec   = baseline_spec();
    spec.hidden = 2049;
    expect_rejection(spec, plan_decision::infeasible, plan_error::row_block_alignment, "gate_up_row",
                     "no-padding storage rejects an incomplete source row block");
}

void check_uint64_overflow() {
    auto spec           = baseline_spec();
    spec.hidden         = static_cast<std::uint64_t>(1) << 63;
    spec.intermediate   = 5;
    spec.block_elements = 1;
    spec.block_bytes    = 4;
    expect_rejection(spec, plan_decision::invalid, plan_error::size_overflow, "gate_up_row",
                     "derived byte count overflowed uint64");
    // The checked helpers themselves must fail cleanly at the boundary.
    std::uint64_t out = 0;
    check(!tq::relay::detail::checked_mul(std::numeric_limits<std::uint64_t>::max(), 2, out), "mul overflow passed");
    check(!tq::relay::detail::checked_add(std::numeric_limits<std::uint64_t>::max(), 1, out), "add overflow passed");
    check(tq::relay::detail::checked_mul(std::numeric_limits<std::uint64_t>::max(), 1, out) &&
              out == std::numeric_limits<std::uint64_t>::max(),
          "boundary multiply rejected");
    check(tq::relay::detail::checked_add(0, 0, out) && out == 0, "zero add rejected");

    // Regression for the three-factor checked product: the error field must
    // survive the internal move. gate_up_weight_bytes = 2 * width * hrow
    // overflows first here, before any other derived count.
    auto three_factor           = baseline_spec();
    three_factor.hidden         = static_cast<std::uint64_t>(1) << 32;
    three_factor.intermediate   = 5ULL << 32;
    three_factor.active_rows    = 1;
    three_factor.block_elements = static_cast<std::uint64_t>(1) << 32;
    three_factor.block_bytes    = static_cast<std::uint64_t>(1) << 32;
    expect_rejection(three_factor, plan_decision::invalid, plan_error::size_overflow, "gate_up_weight_bytes",
                     "three-factor overflow must name the intact error field");
}

void check_invalid_counts() {
    auto spec  = baseline_spec();
    spec.ranks = 4;
    expect_rejection(spec, plan_decision::invalid, plan_error::rank_count, "ranks", "rank count must stay five");
    for (std::uint64_t glu_spec::* member : { &glu_spec::hidden, &glu_spec::intermediate, &glu_spec::active_rows,
                                              &glu_spec::block_elements, &glu_spec::block_bytes }) {
        auto zeroed       = baseline_spec();
        zeroed.*member    = 0;
        const auto result = compile_glu_plan(zeroed);
        check(result.decision == plan_decision::invalid && result.error == plan_error::out_of_range,
              "zero count was not rejected");
    }
}

// The dense GLU oracle of guide 54.12: deterministic small fixture, an
// independent full-dense reference, and a plan-driven assembly. Both
// partition algebras must reproduce the reference within 1e-12.
using matrix = std::vector<std::vector<double>>;

matrix dense_fixture_x(std::uint64_t h, std::uint64_t rows) {
    matrix x(rows, std::vector<double>(h));
    for (std::uint64_t t = 0; t < rows; ++t) {
        for (std::uint64_t i = 0; i < h; ++i) {
            x[t][i] = (static_cast<double>(((t + 1) * (i + 2)) % 9) - 4.0) * 0.05;
        }
    }
    return x;
}

matrix dense_fixture_gate(std::uint64_t h, std::uint64_t f) {
    matrix gate(f, std::vector<double>(h));
    for (std::uint64_t j = 0; j < f; ++j) {
        for (std::uint64_t i = 0; i < h; ++i) {
            gate[j][i] = (static_cast<double>((j * 5 + i * 3) % 13) - 6.0) * 0.03;
        }
    }
    return gate;
}

matrix dense_fixture_up(std::uint64_t h, std::uint64_t f) {
    matrix up(f, std::vector<double>(h));
    for (std::uint64_t j = 0; j < f; ++j) {
        for (std::uint64_t i = 0; i < h; ++i) {
            up[j][i] = (static_cast<double>((j * 7 + i * 2) % 11) - 5.0) * 0.02;
        }
    }
    return up;
}

matrix dense_fixture_down(std::uint64_t h, std::uint64_t f) {
    matrix down(h, std::vector<double>(f));
    for (std::uint64_t i = 0; i < h; ++i) {
        for (std::uint64_t j = 0; j < f; ++j) {
            down[i][j] = (static_cast<double>((i * 4 + j * 3) % 17) - 8.0) * 0.04;
        }
    }
    return down;
}

double sigmoid(double value) {
    return 1.0 / (1.0 + std::exp(-value));
}

matrix dense_reference(const matrix & x, const matrix & gate, const matrix & up, const matrix & down) {
    const std::uint64_t rows = x.size();
    const std::uint64_t h    = x[0].size();
    const std::uint64_t f    = gate.size();
    matrix              result(rows, std::vector<double>(h, 0.0));
    for (std::uint64_t t = 0; t < rows; ++t) {
        std::vector<double> middle(f, 0.0);
        for (std::uint64_t j = 0; j < f; ++j) {
            double g = 0.0;
            double u = 0.0;
            for (std::uint64_t i = 0; i < h; ++i) {
                g += gate[j][i] * x[t][i];
                u += up[j][i] * x[t][i];
            }
            middle[j] = sigmoid(g) * u;
        }
        for (std::uint64_t i = 0; i < h; ++i) {
            double value = 0.0;
            for (std::uint64_t j = 0; j < f; ++j) {
                value += down[i][j] * middle[j];
            }
            result[t][i] = value;
        }
    }
    return result;
}

matrix dense_from_plan(const glu_plan_result & plan,
                       const matrix &          x,
                       const matrix &          gate,
                       const matrix &          up,
                       const matrix &          down) {
    const std::uint64_t rows = x.size();
    const std::uint64_t h    = plan.spec.hidden;
    const std::uint64_t f    = plan.spec.intermediate;
    matrix              result(rows, std::vector<double>(h, 0.0));
    // Local gate/up and GLU per rank.
    std::vector<matrix> fragments;
    fragments.reserve(plan.ranks.size());
    for (const auto & rank : plan.ranks) {
        const std::uint64_t begin = rank.gate_up_output.begin;
        const std::uint64_t end   = rank.gate_up_output.end;
        matrix              local(rows, std::vector<double>(end - begin, 0.0));
        for (std::uint64_t t = 0; t < rows; ++t) {
            for (std::uint64_t feature = begin; feature < end; ++feature) {
                double g = 0.0;
                double u = 0.0;
                for (std::uint64_t i = 0; i < h; ++i) {
                    g += gate[feature][i] * x[t][i];
                    u += up[feature][i] * x[t][i];
                }
                local[t][feature - begin] = sigmoid(g) * u;
            }
        }
        fragments.push_back(std::move(local));
    }
    if (plan.spec.strategy == shard_strategy::gather_intermediate) {
        matrix joined(rows, std::vector<double>(f, 0.0));
        for (std::size_t rank = 0; rank < plan.ranks.size(); ++rank) {
            const std::uint64_t begin = plan.ranks[rank].gate_up_output.begin;
            for (std::uint64_t t = 0; t < rows; ++t) {
                std::copy(fragments[rank][t].begin(), fragments[rank][t].end(), joined[t].begin() + begin);
            }
        }
        for (const auto & rank : plan.ranks) {
            const std::uint64_t begin = rank.down_output.begin;
            const std::uint64_t end   = rank.down_output.end;
            for (std::uint64_t t = 0; t < rows; ++t) {
                for (std::uint64_t feature = begin; feature < end; ++feature) {
                    double value = 0.0;
                    for (std::uint64_t j = 0; j < f; ++j) {
                        value += down[feature][j] * joined[t][j];
                    }
                    result[t][feature] = value;
                }
            }
        }
    } else {
        for (std::size_t rank = 0; rank < plan.ranks.size(); ++rank) {
            const std::uint64_t begin = plan.ranks[rank].down_input.begin;
            for (std::uint64_t t = 0; t < rows; ++t) {
                for (std::uint64_t feature = 0; feature < h; ++feature) {
                    double partial = 0.0;
                    for (std::uint64_t j = begin; j < plan.ranks[rank].down_input.end; ++j) {
                        partial += down[feature][j] * fragments[rank][t][j - begin];
                    }
                    result[t][feature] += partial;
                }
            }
        }
    }
    return result;
}

double max_abs_error(const matrix & actual, const matrix & expected) {
    double largest = 0.0;
    for (std::size_t t = 0; t < actual.size(); ++t) {
        for (std::size_t i = 0; i < actual[t].size(); ++i) {
            check(std::isfinite(actual[t][i]) && std::isfinite(expected[t][i]), "nonfinite dense value");
            largest = std::max(largest, std::abs(actual[t][i] - expected[t][i]));
        }
    }
    return largest;
}

void check_dense_oracle() {
    const std::uint64_t h         = 10;
    const std::uint64_t f         = 15;
    const std::uint64_t rows      = 2;
    const auto          x         = dense_fixture_x(h, rows);
    const auto          gate      = dense_fixture_gate(h, f);
    const auto          up        = dense_fixture_up(h, f);
    const auto          down      = dense_fixture_down(h, f);
    const auto          reference = dense_reference(x, gate, up, down);
    double              largest   = 0.0;
    for (const auto strategy : { shard_strategy::gather_intermediate, shard_strategy::split_intermediate }) {
        for (const auto output : { output_distribution::sharded, output_distribution::replicated }) {
            auto spec           = baseline_spec();
            spec.hidden         = h;
            spec.intermediate   = f;
            spec.active_rows    = rows;
            spec.block_elements = 1;
            spec.block_bytes    = 4;
            spec.strategy       = strategy;
            spec.output         = output;
            const auto plan     = compile_glu_plan(spec);
            check(plan.decision == plan_decision::feasible, "small fixture must be feasible");
            const auto actual = dense_from_plan(plan, x, gate, up, down);
            largest           = std::max(largest, max_abs_error(actual, reference));
        }
    }
    check(largest <= 1e-12, "dense GLU oracle disagrees with the plan algebra");
}

void check_two_row_gather_placement() {
    // Guide 54.13: two active rows must be placed row-strided into the
    // assembled matrix. A naive flat copy of each rank's two local rows
    // produces wrong global positions even though every address stays
    // inside the buffer.
    const std::uint64_t f            = 15;
    const std::uint64_t rows         = 2;
    const std::uint64_t ranges[5][2] = {
        { 0,  3  },
        { 3,  6  },
        { 6,  9  },
        { 9,  12 },
        { 12, 15 }
    };
    std::vector<double> correct(rows * f, -1.0);
    std::vector<double> wrong(rows * f, -1.0);
    for (const auto & range : ranges) {
        const std::uint64_t begin = range[0];
        const std::uint64_t end   = range[1];
        std::vector<double> local;
        for (std::uint64_t row = 0; row < rows; ++row) {
            for (std::uint64_t feature = begin; feature < end; ++feature) {
                local.push_back(static_cast<double>(row * 1000 + feature));
            }
        }
        for (std::uint64_t row = 0; row < rows; ++row) {
            for (std::uint64_t local_index = 0; local_index < end - begin; ++local_index) {
                correct[row * f + begin + local_index] = local[row * (end - begin) + local_index];
            }
        }
        // The flat-copy bug: paste both rows contiguously from `begin`.
        std::copy(local.begin(), local.end(), wrong.begin() + begin);
    }
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t feature = 0; feature < f; ++feature) {
            check(correct[row * f + feature] == static_cast<double>(row * 1000 + feature),
                  "row-strided placement drifted");
        }
    }
    check(correct != wrong, "two-row gather did not expose the flat-copy bug");
}

}  // namespace

int main() {
    check_baseline_numbers();
    check_replicated_output_counts_the_final_gather();
    check_split_512_is_infeasible();
    check_aligned_paired_partition();
    check_row_block_alignment();
    check_uint64_overflow();
    check_invalid_counts();
    check_dense_oracle();
    check_two_row_gather_placement();
    std::cout << "PASS: local GLU plan checks; synthetic CPU geometry only\n";
    return 0;
}
