// Evidence plumbing checks that need no GPU: case registry coverage,
// pattern determinism and distinctness, and strict JSON serialization.

#include "evidence.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace tq::relay::probe;

namespace {

void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_case_registry() {
    check(case_count() == 11, "case registry size drifted");
    const case_id all[] = {
        case_id::device_identity,         case_id::mapped_host_access,
        case_id::cpu_to_gpu_publish,      case_id::gpu_to_cpu_publish,
        case_id::producer_delay_ordering, case_id::ticket_semantics,
        case_id::multi_group_completion,  case_id::abort_stop,
        case_id::host_system_rmw_u64_add, case_id::host_system_rmw_u64_cas,
        case_id::five_rank_cross_device,
    };
    for (std::size_t i = 0; i < case_count(); ++i) {
        check(case_name(all[i]) != nullptr && case_name(all[i])[0] != '\0', "case name must exist");
        for (std::size_t j = i + 1; j < case_count(); ++j) {
            check(std::string(case_name(all[i])) != case_name(all[j]), "case names must be unique");
        }
    }
}

void check_patterns() {
    check(pattern_word(1, 0, 7) == pattern_word(1, 0, 7), "pattern must be deterministic");
    check(pattern_word(1, 0, 7) != pattern_word(1, 1, 7), "index must change the pattern");
    check(pattern_word(1, 0, 7) != pattern_word(2, 0, 7), "generation must change the pattern");
    check(pattern_word(1, 0, 7) != pattern_word(1, 0, 8), "seed must change the pattern");
}

void check_json() {
    probe_report report;
    report.schema                 = 1;
    report.build_id               = "test";
    report.memory_flags_requested = "quotes \" and \\ backslash\nnewline";
    device_record device;
    device.ordinal          = 0;
    std::string device_name = "device ";
    device_name += '\x00';
    device_name += " control";
    device.name       = device_name;
    device.pci_bus_id = "0000:03:00.0";
    device.cases.push_back({ case_id::device_identity, outcome::pass, "detail \"escaped\"" });
    report.devices.push_back(device);
    report.qualification = outcome::fail;
    report.limits.push_back("limit \t tab");
    const std::string json = to_json(report);
    check(!json.empty() && json.front() == '{' && json.back() == '}', "report must be a JSON object");
    check(json.find("\"schema\":1,\"target\":") != std::string::npos, "top-level pairs must be comma separated");
    check(json.find("\"rank\":0,\"pci_bdf\":") != std::string::npos, "device pairs must be comma separated");
    std::size_t open = 0;
    for (const char ch : json) {
        open += ch == '{' ? 1 : 0;
        open -= ch == '}' ? 1 : 0;
    }
    check(open == 0, "braces must balance");
    check(json.find("\\\"escaped\\\"") != std::string::npos, "detail escaping broken");
    check(json.find("\\n") != std::string::npos, "newline escaping broken");
    check(json.find("\\t") != std::string::npos, "tab escaping broken");
    check(json.find("\\u0000") != std::string::npos, "control escaping broken");
    check(json.find("\"qualification\":\"fail\"") != std::string::npos, "qualification serialization broken");
}

}  // namespace

int main() {
    check_case_registry();
    check_patterns();
    check_json();
    std::cout << "PASS: probe evidence plumbing checks\n";
    return 0;
}
