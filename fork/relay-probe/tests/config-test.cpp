// Evidence plumbing checks that need no GPU: case registry coverage,
// pattern determinism and distinctness, and strict JSON serialization.

#include "evidence.h"

#include <dirent.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
    check(case_count() == 25, "case registry size drifted");
    const case_id all[] = {
        case_id::device_identity,
        case_id::mapped_host_access,
        case_id::cpu_to_gpu_publish,
        case_id::gpu_to_cpu_publish,
        case_id::producer_delay_ordering,
        case_id::ticket_semantics,
        case_id::multi_group_completion,
        case_id::abort_stop,
        case_id::host_system_rmw_u64_add,
        case_id::host_system_rmw_u64_cas,
        case_id::five_rank_cross_device,
        case_id::multi_consumer_output_credit,
        case_id::delayed_consumer_ordering,
        case_id::two_bank_rotation,
        case_id::varying_live_bytes,
        case_id::five_stream_relay_cycle,
        case_id::offset_addressing,
        case_id::multi_group_ordering,
        case_id::native_sync_deadline,
        case_id::same_stream_inverse_dependency,
        case_id::dual_window_independent_progress,
        case_id::service_loop_independent_progress,
        case_id::nonfinite_reduction_gate,
        case_id::host_wait_deadline_no_reuse,
        case_id::five_rank_indexing,
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

    // A symlink at the target is refused too and stays a symlink: the
    // documented contract is "an existing file or symlink is never
    // replaced", not "an existing regular file".
    const std::string  victim         = std::string(dir) + "/victim.json";
    const std::string  alias          = std::string(dir) + "/alias.json";
    const write_result victim_written = write_file_exclusive(victim, "{\"victim\":true}\n");
    check(victim_written.code == 0, "victim write must succeed");
    check(::symlink("victim.json", alias.c_str()) == 0, "symlink setup failed");
    const write_result through_link = write_file_exclusive(alias, "{\"replaced\":true}\n");
    check(through_link.code != 0, "a symlink target must not be replaced");
    check(through_link.message.find("refusing") != std::string::npos, "the symlink refusal must be explained");
    struct stat link_info{};
    check(::lstat(alias.c_str(), &link_info) == 0 && S_ISLNK(link_info.st_mode), "the symlink must survive");
    input.close();
    input.open(victim);
    on_disk.assign((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    check(on_disk == "{\"victim\":true}\n", "the victim file must stay intact");
    check(::unlink(alias.c_str()) == 0 && ::unlink(victim.c_str()) == 0, "symlink test cleanup failed");

    // A path whose directory does not exist refuses cleanly with a message
    // naming the temporary file creation failure, not a crash or a stray
    // file somewhere else.
    const write_result missing_dir = write_file_exclusive(std::string(dir) + "/no/such/evidence.json", content);
    check(missing_dir.code != 0, "a missing directory must be refused");
    check(missing_dir.message.find("cannot exclusively create") != std::string::npos,
          "the missing-directory refusal must be explained");

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

// The README tells the operator that stdout can restore evidence when the
// process is killed mid-run. That claim rests on the probe's stdout mode:
// with _IONBF the C standard requires each write to be transmitted to the
// destination as soon as possible, so a SIGKILL after a completed line
// cannot lose it. This test kills a child exactly like that and checks the
// file, pinning the contract to the libc this project runs on.
void check_stdout_kill_survival() {
    char   dir_template[] = "/tmp/tq-relay-probe-stdout-XXXXXX";
    char * dir            = mkdtemp(dir_template);
    check(dir != nullptr, "mkdtemp failed");
    const std::string path  = std::string(dir) + "/out.txt";
    const pid_t       child = fork();
    check(child >= 0, "fork failed");
    if (child == 0) {
        FILE * out = std::freopen(path.c_str(), "w", stdout);
        if (out == nullptr) {
            _exit(2);
        }
        (void) setvbuf(stdout, nullptr, _IONBF, 0);
        std::printf("survives-kill line\n");
        // Die before exit-time flushing or any buffer fill could help.
        (void) raise(SIGKILL);
        _exit(1);
    }
    int status = 0;
    check(waitpid(child, &status, 0) == child, "waitpid failed");
    check(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL, "child must die by SIGKILL");
    std::ifstream     input(path);
    const std::string content((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    check(content.find("survives-kill line") != std::string::npos,
          "stdout written before the kill must survive SIGKILL");
    check(unlink(path.c_str()) == 0, "stdout test unlink failed");
    check(rmdir(dir) == 0, "stdout test rmdir failed");
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

void check_fold_gate() {
    // The gate rests on two invariants: the five weights are pairwise
    // distinct (a swapped rank row still breaks the identity) and their sum
    // is exactly 1 mod 2^64 (a clean fold reproduces the shared pattern).
    std::uint64_t weight_sum = 0;
    for (std::uint32_t rank = 0; rank < relay_ranks; ++rank) {
        const std::uint64_t weight = rank_fold_weight(rank);
        weight_sum += weight;
        for (std::uint32_t other = 0; other < rank; ++other) {
            check(weight != rank_fold_weight(other), "rank weights must be pairwise distinct");
        }
    }
    check(weight_sum == 1, "rank weights must sum to exactly 1 mod 2^64");

    // A clean fold reproduces the shared pattern at every word and reports
    // no problem.
    const std::uint64_t identities[][2] = {
        { 1,    7  },
        { 2,    0  },
        { 2000, 13 }
    };
    for (const auto & identity : identities) {
        const std::uint64_t        generation = identity[0];
        const std::uint32_t        seed       = static_cast<std::uint32_t>(identity[1]);
        std::vector<std::uint64_t> partials(relay_ranks * relay_payload_words);
        for (std::uint32_t rank = 0; rank < relay_ranks; ++rank) {
            for (std::uint32_t i = 0; i < relay_payload_words; ++i) {
                partials[rank * relay_payload_words + i] = pattern_word(generation, i, seed) * rank_fold_weight(rank);
            }
        }
        std::vector<std::uint64_t> out(relay_payload_words);
        check(!fold_partials(partials.data(), generation, seed, out.data()).has_value(),
              "a clean fold must pass the gate");
        for (std::uint32_t i = 0; i < relay_payload_words; ++i) {
            check(out[i] == pattern_word(generation, i, seed), "a clean fold must reproduce the shared pattern");
        }
    }

    // Corruption must be rejected before publication. Build one clean set
    // and mutate it through every shape of damage the gate exists to catch:
    // a high-bit flip (the odd weights are a bijection, where a power-of-two
    // weight would have shifted this bit out of the word), a quiet-NaN bit
    // pattern (for synthetic words just another wrong word), and one rank
    // row copied over another (a missing contribution; the distinct weights
    // make every rank pair detectable). Each mutation must name the word and
    // generation, and none may write through to the output.
    constexpr std::uint64_t generation = 1;
    constexpr std::uint32_t seed       = 7;
    const auto              clean      = [&]() {
        std::vector<std::uint64_t> partials(relay_ranks * relay_payload_words);
        for (std::uint32_t rank = 0; rank < relay_ranks; ++rank) {
            for (std::uint32_t i = 0; i < relay_payload_words; ++i) {
                partials[rank * relay_payload_words + i] = pattern_word(generation, i, seed) * rank_fold_weight(rank);
            }
        }
        return partials;
    };
    std::vector<std::uint64_t> out(relay_payload_words);
    const auto                 expect_rejection = [&](std::vector<std::uint64_t> & partials) {
        std::fill(out.begin(), out.end(), 0);
        const auto problem = fold_partials(partials.data(), generation, seed, out.data());
        check(problem.has_value(), "a corrupted fold must be rejected");
        check(problem->find("reduction rejected word") != std::string::npos, "the rejection must name the fold gate");
        check(problem->find("generation 1") != std::string::npos, "the rejection must name the generation");
        for (std::uint64_t word : out) {
            check(word == 0, "a rejected fold must not write through to the output");
        }
    };
    {
        std::vector<std::uint64_t> partials = clean();
        partials[2 * relay_payload_words + 5] ^= 1ULL << 63;
        expect_rejection(partials);
    }
    {
        std::vector<std::uint64_t> partials   = clean();
        partials[3 * relay_payload_words + 1] = 0x7FF8000000000000ULL;  // quiet NaN
        expect_rejection(partials);
    }
    {
        std::vector<std::uint64_t> partials = clean();
        for (std::uint32_t i = 0; i < relay_payload_words; ++i) {
            partials[1 * relay_payload_words + i] = partials[0 * relay_payload_words + i];
        }
        expect_rejection(partials);
    }
}

void check_five_rank_records() {
    // Success: five pass records with the per-rank wording.
    const auto passed = five_rank_case_records("", 5);
    check(passed.size() == 5, "success must produce five records");
    for (std::size_t rank = 0; rank < passed.size(); ++rank) {
        check(passed[rank].id == case_id::five_rank_cross_device, "record id must be the five-rank case");
        check(passed[rank].result == outcome::pass, "success records must pass");
        check(passed[rank].detail.find("rank " + std::to_string(rank) + " of 5") != std::string::npos,
              "success detail must name the rank");
    }

    // Failure pinned to rank 2: that record carries the raw attribution,
    // the others the participated wording.
    const std::string failure = "rank 2 drain failed: an error";
    const auto        failed  = five_rank_case_records(failure, 2);
    check(failed.size() == 5, "failure must produce five records");
    for (std::size_t rank = 0; rank < failed.size(); ++rank) {
        check(failed[rank].result == outcome::fail, "failure records must fail");
        if (rank == 2) {
            check(failed[rank].detail == failure, "the failed rank must carry the raw attribution");
        } else {
            check(failed[rank].detail ==
                      "rank " + std::to_string(rank) + " of 5 participated; case failed (see five_rank_detail)",
                  "non-failed ranks must carry the participated wording");
        }
    }

    // Sentinel: the failure is not pinned to a specific rank, so every
    // record uses the participated wording and none claims the attribution.
    const auto unpinned = five_rank_case_records("guard word changed in round 3", 5);
    check(unpinned.size() == 5, "unpinned failure must produce five records");
    for (const auto & record : unpinned) {
        check(record.result == outcome::fail, "unpinned failure records must fail");
        check(record.detail.find("participated; case failed") != std::string::npos,
              "unpinned failure must not be attributed to a rank");
    }
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
        check_fold_gate();
        check_five_rank_records();
        check_json();
        check_bench_json();
        check_exclusive_write();
        check_stdout_kill_survival();
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "PASS: probe evidence plumbing checks\n";
    return 0;
}
