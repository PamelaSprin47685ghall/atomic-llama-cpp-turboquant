// Evidence plumbing checks that need no GPU: case registry coverage,
// pattern determinism and distinctness, and strict JSON serialization.

#include "evidence.h"

#include <dirent.h>
#include <stdlib.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tq::relay::probe;

namespace {

void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_case_registry() {
    check(case_count() == 18, "case registry size drifted");
    const case_id all[] = {
        case_id::device_identity,           case_id::mapped_host_access,
        case_id::cpu_to_gpu_publish,        case_id::gpu_to_cpu_publish,
        case_id::producer_delay_ordering,   case_id::ticket_semantics,
        case_id::multi_group_completion,    case_id::abort_stop,
        case_id::host_system_rmw_u64_add,   case_id::host_system_rmw_u64_cas,
        case_id::five_rank_cross_device,    case_id::multi_consumer_output_credit,
        case_id::delayed_consumer_ordering, case_id::two_bank_rotation,
        case_id::varying_live_bytes,        case_id::five_stream_relay_cycle,
        case_id::offset_addressing,         case_id::multi_group_ordering,
    };
    for (std::size_t i = 0; i < case_count(); ++i) {
        check(case_name(all[i]) != nullptr && case_name(all[i])[0] != '\0', "case name must exist");
        for (std::size_t j = i + 1; j < case_count(); ++j) {
            check(std::string(case_name(all[i])) != case_name(all[j]), "case names must be unique");
        }
    }
}

void check_exclusive_write() {
    char   dir_template[] = "/tmp/tq-relay-probe-write-XXXXXX";
    char * dir            = mkdtemp(dir_template);
    check(dir != nullptr, "mkdtemp failed");
    const std::string base    = std::string(dir) + "/evidence.json";
    const std::string content = "{\"evidence\":true}\n";

    // First publish succeeds and the file content is complete on disk.
    const write_result first = write_file_exclusive(base, content);
    check(first.code == 0, "first exclusive write must succeed");
    std::ifstream input(base);
    std::string   on_disk((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    check(on_disk == content, "published content must match what was written");

    // A second publish refuses to replace the existing file and leaves the
    // original content untouched.
    const write_result second = write_file_exclusive(base, "{\"replaced\":true}\n");
    check(second.code != 0, "existing evidence must not be replaced");
    check(second.message.find("exist") != std::string::npos, "refusal must explain the existing file");
    input.clear();
    input.seekg(0);
    on_disk.assign((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    check(on_disk == content, "the original evidence must stay intact");

    // No temporary names may remain, and the directory must contain exactly
    // the one published file.
    DIR * listing = opendir(dir);
    check(listing != nullptr, "opendir failed");
    std::vector<std::string> entries;
    while (dirent * entry = readdir(listing)) {
        const std::string name = entry->d_name;
        if (name != "." && name != "..") {
            entries.push_back(name);
        }
    }
    closedir(listing);
    check(entries.size() == 1 && entries[0] == "evidence.json", "temporary names must not remain");

    check(unlink(base.c_str()) == 0, "test cleanup unlink failed");
    check(rmdir(dir) == 0, "test cleanup rmdir failed");
}

void check_bench_json() {
    bench_report report;
    report.schema                     = 1;
    report.build_id                   = "test";
    report.hip_runtime_version_string = "7.2.4";
    report.device_ordinal             = 0;
    report.device_name                = "bench \"device\"";
    report.gcn_arch_name              = "gfx1030";
    report.rounds                     = 3;
    report.measurements.push_back({ "host_publish_to_device_ack", "ns", 3, 100, 200, 300, 400 });
    report.measurements.push_back({ "host_five_way_fold", "bytes_per_second", 1, 5, 5, 5, 5 });
    report.limits.push_back("limit \\ backslash");
    const std::string json = to_json(report);
    check(!json.empty() && json.front() == '{' && json.back() == '}', "bench report must be a JSON object");
    check(json.find("\"schema\":1,\"target\":\"tp5-relay-bench\"") != std::string::npos,
          "bench top-level pairs must be comma separated");
    check(json.find("\"name\":\"host_publish_to_device_ack\",\"unit\":\"ns\",\"samples\":3,\"min\":100,"
                    "\"median\":200,\"p90\":300,\"max\":400") != std::string::npos,
          "measurement pairs must be comma separated");
    check(json.find("\\\"device\\\"") != std::string::npos, "bench name escaping broken");
    check(json.find("\\\\ backslash") != std::string::npos, "bench limit escaping broken");
    std::size_t open = 0;
    for (const char ch : json) {
        open += ch == '{' ? 1 : 0;
        open -= ch == '}' ? 1 : 0;
    }
    check(open == 0, "bench braces must balance");
}

void check_patterns() {
    const std::uint64_t first  = pattern_word(1, 0, 7);
    const std::uint64_t second = pattern_word(1, 0, 7);
    check(first == second, "pattern must be deterministic");
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
    device.cases.push_back({ case_id::five_rank_cross_device, outcome::pass, "rank 0 of 5" });
    report.devices.push_back(device);
    report.five_rank_status = outcome::not_run;
    report.five_rank_detail = "requires \"five\" devices\n";
    report.qualification    = outcome::fail;
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
    check(json.find("\"five_rank_status\":\"not_run\"") != std::string::npos, "five-rank status serialization broken");
    check(json.find("\"five_rank_detail\":\"requires \\\"five\\\" devices\\n\"") != std::string::npos,
          "five-rank detail escaping broken");
    check(json.find("\"five_rank_cross_device\":{\"result\":\"pass\",\"detail\":\"rank 0 of 5\"}") != std::string::npos,
          "five-rank per-device case serialization broken");
}

}  // namespace

int main() {
    try {
        check_case_registry();
        check_patterns();
        check_json();
        check_bench_json();
        check_exclusive_write();
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "PASS: probe evidence plumbing checks\n";
    return 0;
}
