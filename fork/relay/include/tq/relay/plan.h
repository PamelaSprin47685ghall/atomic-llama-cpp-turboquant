#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tq::relay {

// A local dense SwiGLU layout contract (guide 51, 54). This is a port of the
// verified Python local plan checker `tp5_local_plan.py` (recorded SHA-256
// 5912998214d411bda905a25f58878bf9550c8e7e7cfa9ad43733a5736b59e657) with the
// same arithmetic and the same rejection codes. It answers one narrow
// question: can five nonempty ranks split the given synthetic storage
// geometry without breaking an independent per-row block? `feasible` is not
// a GPU or model qualification (guide 54.1, 54.14).

enum class storage_layout {
    independent_blocks_no_padding,  // blocks along each row's input axis; no row padding
};

enum class shard_strategy {
    gather_intermediate,  // split gate/up output rows, gather the middle, split down output rows
    split_intermediate,   // paired input-axis split; down produces full-hidden partial sums
};

enum class output_distribution {
    sharded,     // disjoint complete feature ranges; the final gather is not included
    replicated,  // every consumer gets the complete output; the final gather is included
};

enum class plan_decision {
    feasible,    // the synthetic storage geometry supports the request
    infeasible,  // the geometry is legal but the chosen strategy cannot satisfy it
    invalid,     // the request violates the input contract
};

enum class plan_error {
    none,
    out_of_range,         // a count is zero
    rank_count,           // ranks != 5
    invalid_enum,         // an enum value outside the declared set
    size_overflow,        // a derived byte or work count exceeds uint64
    partition_alignment,  // the axis does not contain whole units
    insufficient_units,   // fewer indivisible units than nonempty shards
    row_block_alignment,  // a row does not contain whole blocks
};

const char * to_string(plan_decision decision) noexcept;
const char * to_string(plan_error error) noexcept;

struct glu_spec {
    std::uint32_t       ranks          = 5;
    std::uint64_t       hidden         = 0;
    std::uint64_t       intermediate   = 0;
    std::uint64_t       active_rows    = 0;
    std::uint64_t       block_elements = 0;
    std::uint64_t       block_bytes    = 0;
    storage_layout      layout         = storage_layout::independent_blocks_no_padding;
    shard_strategy      strategy       = shard_strategy::gather_intermediate;
    output_distribution output         = output_distribution::sharded;
};

// Half-open [begin, end) over logical coordinates.
struct index_range {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

inline bool operator==(const index_range & a, const index_range & b) noexcept {
    return a.begin == b.begin && a.end == b.end;
}

enum class rank_value_kind {
    complete_output_rows,     // down produces complete output rows
    full_hidden_partial_sum,  // down produces a full-hidden partial sum
};

struct rank_plan {
    std::uint32_t   rank = 0;
    index_range     gate_up_output;  // middle feature range owned by this rank
    index_range     down_input;
    index_range     down_output;
    rank_value_kind down_value_kind      = rank_value_kind::complete_output_rows;
    std::uint64_t   gate_up_weight_bytes = 0;
    std::uint64_t   down_weight_bytes    = 0;
    std::uint64_t   source_weight_bytes  = 0;
    std::uint64_t   linear_macs          = 0;
};

enum class join_operation { sum, gather };
enum class delivery { scatter, broadcast };
enum class collective_value { middle_activation, output_hidden };

const char * to_string(join_operation operation) noexcept;
const char * to_string(delivery delivery) noexcept;
const char * to_string(collective_value value) noexcept;

struct collective_plan {
    std::uint32_t              step               = 0;
    collective_value           value              = collective_value::middle_activation;
    join_operation             operation          = join_operation::gather;
    delivery                   distribution       = delivery::broadcast;
    std::uint64_t              features           = 0;
    std::uint64_t              active_rows        = 0;
    std::uint32_t              wire_element_bytes = 4;  // f32 payload contract
    std::vector<index_range>   producer_ranges;
    std::vector<index_range>   consumer_ranges;
    std::vector<std::uint64_t> producer_bytes;
    std::vector<std::uint64_t> consumer_bytes;
    std::uint64_t              endpoint_send_bytes    = 0;
    std::uint64_t              endpoint_receive_bytes = 0;
    std::uint64_t              assembled_value_bytes  = 0;
};

struct glu_plan_totals {
    std::uint64_t source_weight_bytes    = 0;
    std::uint64_t linear_macs            = 0;
    std::uint64_t endpoint_send_bytes    = 0;
    std::uint64_t endpoint_receive_bytes = 0;
};

struct glu_plan_result {
    plan_decision decision = plan_decision::invalid;
    plan_error    error    = plan_error::none;
    std::string   error_field;  // which field or derived quantity failed
    std::string   error_detail;

    // Filled only when decision == feasible.
    glu_spec                     spec;
    std::uint64_t                input_bytes_per_rank = 0;  // replicated complete X; delivery is external
    output_distribution          output               = output_distribution::sharded;
    std::vector<index_range>     output_ranges;             // disjoint complete feature ranges
    std::vector<rank_plan>       ranks;
    std::vector<collective_plan> collectives;
    glu_plan_totals              totals;
};

// Compile the five-rank layout. Never throws: every rejection is a value
// (guide 6.4). Internal consistency violations (rank sums diverging from
// totals) are programming errors and terminate via std::logic_error, the
// same treatment as the Python reference's RuntimeError.
glu_plan_result compile_glu_plan(const glu_spec & spec);

}  // namespace tq::relay
