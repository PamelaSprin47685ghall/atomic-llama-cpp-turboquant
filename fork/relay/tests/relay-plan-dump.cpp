// Test-only tool: compile the four fixed local-GLU fixtures with the C++
// planner and print a minimal JSON summary that crosscheck-plan.py compares
// against the verified Python reference tp5_local_plan.py. Not installed.

#include <tq/relay/plan.h>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using tq::relay::compile_glu_plan;
using tq::relay::glu_spec;
using tq::relay::output_distribution;
using tq::relay::plan_decision;
using tq::relay::plan_error;
using tq::relay::shard_strategy;
using tq::relay::storage_layout;

namespace {

glu_spec fixture(const char * name) {
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
    const std::string key(name);
    if (key == "glu-gather-replicated") {
        spec.output = output_distribution::replicated;
    } else if (key == "glu-middle-infeasible") {
        spec.strategy = shard_strategy::split_intermediate;
    } else if (key == "glu-middle-aligned") {
        spec.intermediate = 1280;
        spec.strategy     = shard_strategy::split_intermediate;
    }
    return spec;
}

// Minimal JSON emitter for this tool's fixed shape: strings are the only
// values that can contain escape-worthy characters.
void write_string(const std::string & value) {
    std::cout << '"';
    for (const unsigned char byte : value) {
        switch (byte) {
            case '"':
                std::cout << "\\\"";
                break;
            case '\\':
                std::cout << "\\\\";
                break;
            case '\n':
                std::cout << "\\n";
                break;
            case '\r':
                std::cout << "\\r";
                break;
            case '\t':
                std::cout << "\\t";
                break;
            default:
                if (byte < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", byte);
                    std::cout << buffer;
                } else {
                    std::cout << static_cast<char>(byte);
                }
        }
    }
    std::cout << '"';
}

void write_key(const char * key) {
    write_string(key);
}

void write_ranges(const std::vector<tq::relay::index_range> & ranges) {
    std::cout << '[';
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << '[' << ranges[i].begin << ',' << ranges[i].end << ']';
    }
    std::cout << ']';
}

void write_uint_array(const char * key, const std::vector<std::uint64_t> & values) {
    std::cout << ',';
    write_key(key);
    std::cout << ":[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << values[i];
    }
    std::cout << ']';
}

void dump(const char * name) {
    const auto                          result = compile_glu_plan(fixture(name));
    std::vector<std::uint64_t>          per_rank_weight;
    std::vector<std::uint64_t>          per_rank_macs;
    std::vector<tq::relay::index_range> middle_ranges;
    std::vector<tq::relay::index_range> output_ranges;
    for (const auto & rank : result.ranks) {
        per_rank_weight.push_back(rank.source_weight_bytes);
        per_rank_macs.push_back(rank.linear_macs);
        middle_ranges.push_back(rank.gate_up_output);
        output_ranges.push_back(rank.down_output);
    }
    std::cout << "{\"name\":";
    write_string(name);
    std::cout << ",\"decision\":";
    write_string(tq::relay::to_string(result.decision));
    if (result.decision == plan_decision::feasible) {
        std::cout << ",\"error\":null,\"source_weight_bytes\":" << result.totals.source_weight_bytes
                  << ",\"linear_macs\":" << result.totals.linear_macs
                  << ",\"endpoint_send_bytes\":" << result.totals.endpoint_send_bytes
                  << ",\"endpoint_receive_bytes\":" << result.totals.endpoint_receive_bytes;
        write_uint_array("per_rank_source_weight_bytes", per_rank_weight);
        write_uint_array("per_rank_linear_macs", per_rank_macs);
        std::cout << ",\"middle_ranges\":";
        write_ranges(middle_ranges);
        std::cout << ",\"output_ranges\":";
        write_ranges(output_ranges);
    } else {
        std::cout << ",\"error\":";
        write_string(tq::relay::to_string(result.error));
        std::cout << ",\"error_field\":";
        write_string(result.error_field);
    }
    std::cout << "}\n";
}

}  // namespace

int main() {
    std::cout << '[';
    const char * names[] = { "glu-gather", "glu-gather-replicated", "glu-middle-infeasible", "glu-middle-aligned" };
    for (std::size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        dump(names[i]);
    }
    std::cout << "]\n";
    return 0;
}
