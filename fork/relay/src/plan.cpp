#include <tq/relay/detail/arithmetic.h>
#include <tq/relay/plan.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace tq::relay {

namespace {

constexpr std::uint32_t REQUIRED_RANKS     = 5;
constexpr std::uint32_t WIRE_ELEMENT_BYTES = 4;  // f32 payload contract

const char * decision_names[] = { "feasible", "infeasible", "invalid" };
const char * error_names[]    = { "NONE",          "OUT_OF_RANGE",        "RANK_COUNT",         "INVALID_ENUM",
                                  "SIZE_OVERFLOW", "PARTITION_ALIGNMENT", "INSUFFICIENT_UNITS", "ROW_BLOCK_ALIGNMENT" };
const char * join_names[]     = { "sum", "gather" };
const char * delivery_names[] = { "scatter", "broadcast" };
const char * value_names[]    = { "middle_activation", "output_hidden" };

template <std::size_t N> const char * name_of(const char * const (&names)[N], std::size_t index) noexcept {
    return index < N ? names[index] : "unknown";
}

// Internal rejection carrier; converted to a value at the public boundary.
class plan_rejection : public std::runtime_error {
  public:
    plan_rejection(plan_decision decision, plan_error code, std::string field, std::string detail) :
        std::runtime_error(detail),
        decision_(decision),
        code_(code),
        field_(std::move(field)),
        detail_(std::move(detail)) {}

    plan_decision decision_;
    plan_error    code_;
    std::string   field_;
    std::string   detail_;
};

[[noreturn]] void reject(plan_decision decision, plan_error code, std::string field, std::string detail) {
    throw plan_rejection(decision, code, std::move(field), std::move(detail));
}

std::uint64_t product(std::string field, std::uint64_t a, std::uint64_t b) {
    std::uint64_t out = 0;
    if (!detail::checked_mul(a, b, out)) {
        reject(plan_decision::invalid, plan_error::size_overflow, std::move(field), "product exceeds UINT64_MAX");
    }
    return out;
}

std::uint64_t product(std::string field, std::uint64_t a, std::uint64_t b, std::uint64_t c) {
    // Compute the inner product from a copy first: function argument
    // evaluation order is unspecified, so moving field into the outer call
    // must not race the inner read.
    const std::uint64_t ab = product(field, a, b);
    return product(std::move(field), ab, c);
}

std::uint64_t product(std::string field, std::uint64_t a, std::uint64_t b, std::uint64_t c, std::uint64_t d) {
    const std::uint64_t ab = product(field, a, b);
    const std::uint64_t cd = product(field, c, d);
    return product(std::move(field), ab, cd);
}

std::uint64_t sum(std::string field, std::uint64_t a, std::uint64_t b) {
    std::uint64_t out = 0;
    if (!detail::checked_add(a, b, out)) {
        reject(plan_decision::invalid, plan_error::size_overflow, std::move(field), "sum exceeds UINT64_MAX");
    }
    return out;
}

// Largest balanced split of `size` into ranks shards of `unit` elements.
// Rejections mirror the Python reference: whole units first, then at least
// one unit per nonempty rank.
std::vector<index_range> partition(std::uint64_t size, std::uint64_t unit, const char * field) {
    if (size % unit != 0) {
        reject(plan_decision::infeasible, plan_error::partition_alignment, field, "axis does not contain whole units");
    }
    const std::uint64_t units = size / unit;
    if (units < REQUIRED_RANKS) {
        reject(plan_decision::infeasible, plan_error::insufficient_units, field,
               "five nonempty shards require at least five indivisible units");
    }
    const std::uint64_t      base  = units / REQUIRED_RANKS;
    const std::uint64_t      extra = units % REQUIRED_RANKS;
    std::vector<index_range> result;
    result.reserve(REQUIRED_RANKS);
    std::uint64_t begin = 0;
    for (std::uint32_t rank = 0; rank < REQUIRED_RANKS; ++rank) {
        const std::uint64_t end = begin + (base + (rank < extra ? 1 : 0)) * unit;
        result.push_back({ begin, end });
        begin = end;
    }
    return result;
}

// Whole source row bytes for a row whose input axis has `features`
// elements, in the synthetic independent-block no-padding layout.
std::uint64_t row_bytes(std::uint64_t features,
                        std::uint64_t block_elements,
                        std::uint64_t block_bytes,
                        const char *  field) {
    if (features % block_elements != 0) {
        reject(plan_decision::infeasible, plan_error::row_block_alignment, field,
               "a row must contain whole blocks with no padding");
    }
    return product(field, features / block_elements, block_bytes);
}

collective_plan collective(std::uint32_t                    step,
                           collective_value                 value,
                           join_operation                   operation,
                           delivery                         distribution,
                           std::uint64_t                    features,
                           std::uint64_t                    active_rows,
                           const std::vector<index_range> & producer_ranges,
                           const std::vector<index_range> & consumer_ranges) {
    collective_plan result;
    result.step               = step;
    result.value              = value;
    result.operation          = operation;
    result.distribution       = distribution;
    result.features           = features;
    result.active_rows        = active_rows;
    result.wire_element_bytes = WIRE_ELEMENT_BYTES;
    result.producer_ranges    = producer_ranges;
    result.consumer_ranges    = consumer_ranges;
    for (const auto & range : producer_ranges) {
        result.producer_bytes.push_back(
            product("producer_bytes", range.end - range.begin, active_rows, WIRE_ELEMENT_BYTES));
        result.endpoint_send_bytes =
            sum("endpoint_send_bytes", result.endpoint_send_bytes, result.producer_bytes.back());
    }
    for (const auto & range : consumer_ranges) {
        result.consumer_bytes.push_back(
            product("consumer_bytes", range.end - range.begin, active_rows, WIRE_ELEMENT_BYTES));
        result.endpoint_receive_bytes =
            sum("endpoint_receive_bytes", result.endpoint_receive_bytes, result.consumer_bytes.back());
    }
    result.assembled_value_bytes = product("assembled_value_bytes", features, active_rows, WIRE_ELEMENT_BYTES);
    return result;
}

void validate_spec(const glu_spec & spec) {
    if (spec.ranks != REQUIRED_RANKS) {
        reject(plan_decision::invalid, plan_error::rank_count, "ranks", "this teaching planner requires five ranks");
    }
    const std::uint64_t counts[] = { spec.hidden, spec.intermediate, spec.active_rows, spec.block_elements,
                                     spec.block_bytes };
    const char *        names[]  = { "hidden", "intermediate", "active_rows", "block_elements", "block_bytes" };
    for (std::size_t i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i) {
        if (counts[i] == 0) {
            reject(plan_decision::invalid, plan_error::out_of_range, names[i], "expected a positive count");
        }
    }
    switch (spec.layout) {
        case storage_layout::independent_blocks_no_padding:
            break;
        default:
            reject(plan_decision::invalid, plan_error::invalid_enum, "storage_layout",
                   "unsupported value for this schema");
    }
    switch (spec.strategy) {
        case shard_strategy::gather_intermediate:
        case shard_strategy::split_intermediate:
            break;
        default:
            reject(plan_decision::invalid, plan_error::invalid_enum, "strategy", "unsupported value for this schema");
    }
    switch (spec.output) {
        case output_distribution::sharded:
        case output_distribution::replicated:
            break;
        default:
            reject(plan_decision::invalid, plan_error::invalid_enum, "output_layout",
                   "unsupported value for this schema");
    }
}

}  // namespace

const char * to_string(plan_decision decision) noexcept {
    return name_of(decision_names, static_cast<std::size_t>(decision));
}

const char * to_string(plan_error error) noexcept {
    return name_of(error_names, static_cast<std::size_t>(error));
}

const char * to_string(join_operation operation) noexcept {
    return name_of(join_names, static_cast<std::size_t>(operation));
}

const char * to_string(delivery value) noexcept {
    return name_of(delivery_names, static_cast<std::size_t>(value));
}

const char * to_string(collective_value value) noexcept {
    return name_of(value_names, static_cast<std::size_t>(value));
}

glu_plan_result compile_glu_plan(const glu_spec & input) {
    try {
        const glu_spec spec = input;
        validate_spec(spec);
        const std::uint64_t h           = spec.hidden;
        const std::uint64_t f           = spec.intermediate;
        const std::uint64_t rows        = spec.active_rows;
        const std::uint64_t block       = spec.block_elements;
        const std::uint64_t block_bytes = spec.block_bytes;
        const std::uint64_t hrow        = row_bytes(h, block, block_bytes, "gate_up_row");
        const std::uint64_t frow        = row_bytes(f, block, block_bytes, "down_row");
        const bool          gather      = spec.strategy == shard_strategy::gather_intermediate;

        const std::uint64_t            unit   = gather ? 1 : block;
        const std::vector<index_range> middle = partition(f, unit, "intermediate");
        std::vector<index_range>       hidden_ranges;
        if (gather || spec.output == output_distribution::sharded) {
            hidden_ranges = partition(h, 1, "output_hidden");
        }
        const std::vector<index_range> consumers = spec.output == output_distribution::sharded ?
                                                       hidden_ranges :
                                                       std::vector<index_range>(REQUIRED_RANKS, index_range{ 0, h });

        glu_plan_result plan;
        plan.decision             = plan_decision::feasible;
        plan.spec                 = spec;
        plan.output               = spec.output;
        plan.input_bytes_per_rank = product("input_bytes_per_rank", rows, h, WIRE_ELEMENT_BYTES);

        for (std::uint32_t rank = 0; rank < REQUIRED_RANKS; ++rank) {
            rank_plan part;
            part.rank                         = rank;
            part.gate_up_output               = middle[rank];
            const std::uint64_t width         = middle[rank].end - middle[rank].begin;
            const std::uint64_t gate_up_bytes = product("gate_up_weight_bytes", 2, width, hrow);
            const std::uint64_t gate_up_macs  = product("gate_up_macs", 2, rows, h, width);
            std::uint64_t       down_macs     = 0;
            if (gather) {
                part.down_input      = { 0, f };
                part.down_output     = hidden_ranges[rank];
                part.down_value_kind = rank_value_kind::complete_output_rows;
                part.down_weight_bytes =
                    product("down_weight_bytes", part.down_output.end - part.down_output.begin, frow);
                down_macs = product("down_macs", rows, part.down_output.end - part.down_output.begin, f);
            } else {
                part.down_input      = middle[rank];
                part.down_output     = { 0, h };
                part.down_value_kind = rank_value_kind::full_hidden_partial_sum;
                part.down_weight_bytes =
                    product("down_weight_bytes", h, row_bytes(width, block, block_bytes, "down_input_slice"));
                down_macs = product("down_macs", rows, h, width);
            }
            part.gate_up_weight_bytes = gate_up_bytes;
            part.source_weight_bytes  = sum("rank_weight_bytes", gate_up_bytes, part.down_weight_bytes);
            part.linear_macs          = sum("rank_linear_macs", gate_up_macs, down_macs);
            plan.ranks.push_back(part);
        }

        if (gather) {
            plan.collectives.push_back(collective(0, collective_value::middle_activation, join_operation::gather,
                                                  delivery::broadcast, f, rows, middle,
                                                  std::vector<index_range>(REQUIRED_RANKS, index_range{ 0, f })));
            if (spec.output == output_distribution::replicated) {
                plan.collectives.push_back(collective(1, collective_value::output_hidden, join_operation::gather,
                                                      delivery::broadcast, h, rows, hidden_ranges, consumers));
            }
        } else {
            plan.collectives.push_back(
                collective(0, collective_value::output_hidden, join_operation::sum,
                           spec.output == output_distribution::sharded ? delivery::scatter : delivery::broadcast, h,
                           rows, std::vector<index_range>(REQUIRED_RANKS, index_range{ 0, h }), consumers));
        }
        plan.output_ranges = consumers;

        const std::uint64_t gate_up_total = product("gate_up_total", 2, f, hrow);
        const std::uint64_t down_total    = product("down_total", h, frow);
        plan.totals.source_weight_bytes   = sum("source_weight_bytes", gate_up_total, down_total);
        plan.totals.linear_macs           = product("linear_macs", 3, rows, h, f);
        for (const auto & item : plan.collectives) {
            plan.totals.endpoint_send_bytes =
                sum("all_endpoint_send_bytes", plan.totals.endpoint_send_bytes, item.endpoint_send_bytes);
            plan.totals.endpoint_receive_bytes =
                sum("all_endpoint_receive_bytes", plan.totals.endpoint_receive_bytes, item.endpoint_receive_bytes);
        }

        // Internal conservation checks: the planner must not silently drop
        // or double-count rank work (guide 54.8, 29.4).
        std::uint64_t rank_weight_sum = 0;
        std::uint64_t rank_macs_sum   = 0;
        for (const auto & part : plan.ranks) {
            rank_weight_sum = sum("rank_weight_bytes", rank_weight_sum, part.source_weight_bytes);
            rank_macs_sum   = sum("rank_linear_macs", rank_macs_sum, part.linear_macs);
        }
        if (rank_weight_sum != plan.totals.source_weight_bytes || rank_macs_sum != plan.totals.linear_macs) {
            throw std::logic_error("internal plan coverage error");
        }
        return plan;
    } catch (const plan_rejection & rejection) {
        glu_plan_result result;
        result.decision     = rejection.decision_;
        result.error        = rejection.code_;
        result.error_field  = rejection.field_;
        result.error_detail = rejection.detail_;
        return result;
    }
}

}  // namespace tq::relay
