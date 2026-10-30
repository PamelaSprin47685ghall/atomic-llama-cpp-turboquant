#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace tq::tp5 {

enum class mode { relay, reference };
enum class wire_type { f32, f16 };

struct options {
    std::string                            model;
    std::vector<std::string>               prompts;
    // Empty selects normal generation. Otherwise each request consumes these
    // fixed next-input tokens while still recording the unmodified logits.
    std::vector<std::vector<std::int32_t>> teacher_tokens;
    // Empty keeps the loader's default HC layout. Explicit offsets select the
    // full mirrored HC97 layout and one immutable owner plan per context.
    std::vector<std::uint32_t>              hc_owner_offsets;
    mode                                   execution           = mode::relay;
    wire_type                              wire                = wire_type::f32;
    std::uint32_t                          windows             = 2;
    std::uint32_t                          predict             = 32;
    std::uint32_t                          context_per_request = 512;
    std::uint32_t                          threads             = 4;
    std::uint32_t                          deadline_seconds    = 300;
    bool                                   flash_attention     = false;
    bool                                   capture_logits      = false;
    bool                                   ignore_eos          = false;
    // Controlled consumer delay for lifetime/independent-progress acceptance.
    std::uint32_t                          delay_window        = 0;
    std::uint32_t                          delay_publish_ms    = 0;
    // Cancel one request after this many accepted output tokens; -1 disables.
    std::int32_t                           cancel_request      = -1;
    std::uint32_t                          cancel_after        = 1;
    // Explicit failure injection before a matching CPU publication.
    std::int32_t                           abort_window        = -1;
    std::uint64_t                          abort_channel       = 0;
    std::uint64_t                          abort_epoch         = 1;
};

struct token_record {
    std::int32_t       input_token    = 0;
    std::int32_t       sampled_token  = 0;
    std::int32_t       emitted_token  = 0;
    std::uint64_t      input_position = 0;
    std::uint64_t      committed_ns   = 0;
    // Owned full-vocabulary model output, present only when requested.
    std::vector<float> logits;
};

struct request_result {
    std::uint32_t             id     = 0;
    std::uint32_t             window = 0;
    std::vector<std::int32_t> prompt_tokens;
    std::vector<token_record> tokens;
    std::string               text;
    std::string               stop_reason;
    std::uint64_t             admitted_ns = 0;
    std::uint64_t             finished_ns = 0;
};

struct window_result {
    std::int32_t                            hc_owner_offset = -1;
    std::uint32_t                id                 = 0;
    // Actual public context getters, not echoes of requested options.
    std::uint32_t                context_tokens     = 0;
    std::uint32_t                sequence_tokens    = 0;
    std::uint32_t                kv_capacity_tokens = 0;
    std::uint64_t                decode_calls       = 0;
    // Host call intervals, including native work/waits performed inside each call.
    std::uint64_t                decode_submit_ns   = 0;
    std::uint64_t                native_wait_ns     = 0;
    std::uint64_t                result_commit_ns   = 0;
    std::uint64_t                input_rows         = 0;
    std::uint64_t                epochs             = 0;
    std::uint64_t                channels           = 0;
    std::uint64_t                cpu_reductions     = 0;
    std::uint64_t                input_retirements  = 0;
    std::uint64_t                output_retirements = 0;
    std::uint64_t                first_submit_ns    = 0;
    std::uint64_t                last_finish_ns     = 0;
    std::array<std::uint64_t, 5> rank_graphs{};
    std::array<std::uint64_t, 5> rank_compute_nodes{};
    std::array<std::uint64_t, 5> rank_relay_nodes{};
    std::array<std::uint64_t, 5> rank_published_bytes{};
    std::array<std::uint64_t, 5> rank_consumed_bytes{};
    // Valid wire payload lengths derived from true v2 events, not PCIe/cache-line
    // measurements. Noncomputed inputs do not contribute to packed/read bytes.
    std::array<std::uint64_t, 5> rank_input_packed_payload_bytes{};
    std::array<std::uint64_t, 5> rank_output_consumed_payload_bytes{};
    std::uint64_t                cpu_input_payload_bytes = 0;
    std::uint64_t                cpu_output_payload_bytes = 0;
    std::array<std::string, 5>   pci_bdf{};
    std::uint32_t                hip_runtime_version = 0;
    bool                         native_drained      = true;
};

struct run_result {
    std::string                            hc_weight_layout;
    std::vector<std::uint32_t>              hc_owner_offsets;
    std::string                 execution_mode;
    std::string                 numeric_policy;
    std::string                 wire_format;
    std::string                 model_description;
    bool                        flash_attention             = false;
    std::uint32_t               vocabulary                  = 0;
    std::uint64_t               started_ns                  = 0;
    std::uint64_t               finished_ns                 = 0;
    std::uint64_t               max_simultaneous_windows    = 0;
    std::uint64_t               reserved_tokens_after_close = 0;
    std::uint32_t               active_requests_after_close = 0;
    std::vector<request_result> requests;
    std::vector<window_result>  windows;
};

// Initializes its own model and two independently mutable contexts. The input
// options are copied/validated before device work. Errors never return partial
// results as success. The run drains before releasing state and reservations.
// An unproved native drain terminates the isolated CLI process without running
// destructors that would free in-flight owners; use the CLI as the GPU domain.
run_result run(const options & requested);

}  // namespace tq::tp5
