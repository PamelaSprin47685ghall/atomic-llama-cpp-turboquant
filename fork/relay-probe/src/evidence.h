#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tq::relay::probe {

// One capability outcome. Unknown or skipped are real states; they are not
// silently promoted to pass (guide 31.1).
enum class outcome {
    pass,
    fail,
    unsupported,
    not_run,
};

const char * to_string(outcome value) noexcept;

// Case registry. Names are part of the evidence record; renumbering or
// renaming a case changes what an old report can claim.
enum class case_id {
    device_identity,
    mapped_host_access,
    cpu_to_gpu_publish,
    gpu_to_cpu_publish,
    producer_delay_ordering,
    ticket_semantics,
    multi_group_completion,
    abort_stop,
    host_system_rmw_u64_add,
    host_system_rmw_u64_cas,
    five_rank_cross_device,
    multi_consumer_output_credit,
    delayed_consumer_ordering,
    two_bank_rotation,
    varying_live_bytes,
    five_stream_relay_cycle,
    offset_addressing,
    multi_group_ordering,
    native_sync_deadline,
    same_stream_inverse_dependency,
    dual_window_independent_progress,
    service_loop_independent_progress,
    nonfinite_reduction_gate,
    host_wait_deadline_no_reuse,
};

const char * case_name(case_id value) noexcept;
std::size_t  case_count() noexcept;

// Deterministic payload patterns. Both host and device derive the same
// word from (generation, word index, seed); a mismatch after an acquire
// catches torn, stale, or aliased reads without any probabilistic checksum.
// constexpr makes the definition usable in HIP device code without dragging
// HIP macros into this CPU-only header.
constexpr std::uint64_t pattern_word(std::uint64_t generation, std::uint32_t index, std::uint32_t seed) noexcept {
    // SplitMix64-style mixing; deterministic across host and device since
    // both use 64-bit wraparound arithmetic on the same inputs.
    std::uint64_t value = generation ^ 0x9E3779B97F4A7C15ULL;
    value += static_cast<std::uint64_t>(index) * 0xBF58476D1CE4E5B9ULL;
    value += static_cast<std::uint64_t>(seed) * 0x94D049BB133111EBULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

// The emulated relay fold geometry: RANKS rows of PAYLOAD_WORDS words per
// generation. The rank weights below are pairwise distinct and sum to
// exactly 1 mod 2^64, so a correct five-way fold reproduces the shared
// pattern word for word, while a missing, duplicated, torn, or corrupted
// partial breaks the identity deterministically (guide 35.5, 39.7 F04,
// 41.6). For these synthetic words a quiet-NaN bit pattern is just another
// wrong word and is caught the same way.
constexpr std::uint32_t relay_ranks         = 5;
constexpr std::uint32_t relay_payload_words = 8;

// Pairwise distinct ODD weights that sum to 1 mod 2^64. Each property earns
// its place: oddness makes the per-rank multiplication a bijection on
// 64-bit words, so flipping ANY bit of any producer partial — a power-of-two
// weight would shift high bits out of the word and hide them — changes the
// fold; distinctness makes a duplicated row (a missing contribution) change
// the fold for every rank pair. Row ORDER is not part of the gate: a SUM
// fold is commutative, and the per-rank binding is proven separately by the
// per-rank ready words before the fold runs.
constexpr std::uint64_t rank_fold_weight(std::uint32_t rank) noexcept {
    if (rank == 0) {
        return 1;
    }
    if (rank < 4) {
        return 1 + 2 * rank;  // 3, 5, 7
    }
    return 0ULL - 15ULL;      // -(1 + 3 + 5 + 7) mod 2^64; keeps the sum at 1
}

// Host five-way fold with the publication gate. `rank_partials` holds
// relay_ranks rows of relay_payload_words words; a correct fold writes the
// shared pattern into `out` and returns nullopt. Any word that fails the
// exact identity returns a diagnostic naming the word and generation, and
// nothing may be published from it.
std::optional<std::string> fold_partials(const std::uint64_t * rank_partials,
                                         std::uint64_t         generation,
                                         std::uint32_t         seed,
                                         std::uint64_t *       out);

struct case_record {
    case_id     id     = case_id::device_identity;
    outcome     result = outcome::not_run;
    std::string detail;
};

// Per-rank records for the five-rank case (guide 47.1): the failed rank
// carries the attribution detail; the others record that they participated
// but the case failed. `failed_rank == 5` is the "not a specific rank"
// sentinel used when the failure cannot be pinned to one device (e.g. the
// shared host allocation itself failed). Ranks are always 0..4.
std::vector<case_record> five_rank_case_records(const std::string & detail, std::uint32_t failed_rank);

struct device_record {
    std::int32_t             ordinal = -1;
    std::string              name;
    std::string              pci_bus_id;
    std::string              gcn_arch;
    std::uint64_t            total_vram_bytes    = 0;
    bool                     can_map_host_memory = false;
    std::vector<case_record> cases;
};

struct probe_report {
    std::uint32_t              schema = 1;
    std::string                target = "tp5-mapped-relay";
    std::string                build_id;
    int                        hip_runtime_version = 0;
    std::string                hip_runtime_version_string;
    std::string                compiler_version;
    std::string                host_arch;
    std::uint32_t              control_word_bits = 64;
    std::string                memory_flags_requested;
    std::uint32_t              hot_reuse_rounds = 0;
    std::vector<device_record> devices;
    outcome                    five_rank_status = outcome::not_run;
    std::string                five_rank_detail;
    outcome                    qualification = outcome::fail;
    std::string                qualification_detail;
    std::vector<std::string>   limits;
};

// Serialize the report as strict JSON (finite values only; strings escaped).
std::string to_json(const probe_report & report);

// Bench: measurements are observations for the cost model (guide 14), never
// qualification gates. Latencies use the host steady clock.
struct bench_measurement {
    std::string   name;
    std::string   unit;  // "ns" or "bytes_per_second"
    std::uint64_t samples = 0;
    std::uint64_t min     = 0;
    std::uint64_t median  = 0;
    std::uint64_t p90     = 0;
    std::uint64_t max     = 0;
};

struct bench_report {
    std::uint32_t                  schema = 1;
    std::string                    target = "tp5-relay-bench";
    std::string                    build_id;
    std::string                    hip_runtime_version_string;
    std::int32_t                   device_ordinal = -1;
    std::string                    device_name;
    std::string                    gcn_arch_name;
    std::uint32_t                  rounds = 0;
    std::vector<bench_measurement> measurements;
    std::vector<std::string>       limits;
};

std::string to_json(const bench_report & report);

// Exclusive complete-file publish (guide 16.3, 42.5): write `content` to a
// same-directory temporary file, fsync it, then hard-link it into place, so
// `path` either appears complete or does not appear at all. An existing file
// or symlink is never replaced. The temporary name is removed on every path;
// its removal after a successful link is best effort and does not invalidate
// the published evidence. Durability of the directory entry across a crash
// is not claimed.
struct write_result {
    int         code = 0;  // 0 published; 2 refusal or I/O failure
    std::string message;
};

write_result write_file_exclusive(const std::string & path, const std::string & content);

}  // namespace tq::relay::probe
