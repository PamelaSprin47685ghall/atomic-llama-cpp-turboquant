#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_RELAY_ABI_VERSION 1
#define GGML_RELAY_RANKS       5

// A returned work view borrows provider storage. Only its output is writable.
// Publish or abort the view before requesting another event on this backend.
enum ggml_relay_event_kind {
    GGML_RELAY_INPUT_READY     = 1,
    GGML_RELAY_OUTPUT_RELEASED = 2,
};

struct ggml_relay_event {
    uint32_t                   abi_version;
    enum ggml_relay_event_kind kind;
    uint64_t                   epoch;
    uint64_t                   plan_generation;
    uint32_t                   channel;
    uint32_t                   ranks;
    uint64_t                   elements;
    const float *              inputs[GGML_RELAY_RANKS];
    float *                    output;
};

// An explicit transport representation; compute tensors remain F32.
// F16 stores IEEE-754 binary16 little-endian values. All ranks and the output
// use the same representation, selected before the first prepared graph.
enum ggml_relay_wire_type {
    GGML_RELAY_WIRE_F32 = 0,
    GGML_RELAY_WIRE_F16 = 1,
};

#define GGML_RELAY_EVENT_V2_ABI_VERSION 2

struct ggml_relay_event_v2 {
    uint32_t                   abi_version;
    enum ggml_relay_event_kind kind;
    uint64_t                   epoch;
    uint64_t                   plan_generation;
    uint32_t                   channel;
    uint32_t                   ranks;
    uint64_t                   elements;
    enum ggml_relay_wire_type  wire_type;
    // Actual COMPUTE contributors, never a ready/ack mask. INPUT_READY still
    // requires all five real arrivals. Unset inputs contain positive zeros.
    uint32_t                   computed_mask;
    // Extent of EACH input and of the output, not the sum over ranks.
    uint64_t                   wire_bytes;
    const void *               inputs[GGML_RELAY_RANKS];
    void *                     output;
};

// Masks describe this epoch, not all work ever submitted to a device.
// completed_mask covers native work only. A result reader can still be live.
// Graph, node, byte and reduction counters accumulate over this provider's life.
// rank_graphs counts prepared GGML stages accepted directly or in a native
// replay. rank_relay_nodes counts relay kernel nodes accepted by those paths,
// not host launch API calls. Neither counter proves native completion.
struct ggml_relay_status {
    uint32_t abi_version;
    uint32_t ranks;
    uint64_t epoch;
    uint64_t plan_generation;
    uint64_t channels;
    uint32_t attempted_mask;
    uint32_t accepted_mask;
    uint32_t unknown_mask;
    uint32_t completed_mask;
    bool     sealed;
    bool     failed;
    bool     finished;
    int32_t  error_code;
    uint64_t rank_graphs[GGML_RELAY_RANKS];
    uint64_t rank_compute_nodes[GGML_RELAY_RANKS];
    uint64_t rank_relay_nodes[GGML_RELAY_RANKS];
    uint64_t rank_published_bytes[GGML_RELAY_RANKS];
    uint64_t rank_consumed_bytes[GGML_RELAY_RANKS];
    uint64_t cpu_reductions;
    uint32_t hip_runtime_version;
    char     pci_bdf[GGML_RELAY_RANKS][32];
    char     error[256];
};

// V1 payload views remain F32. Poll/publish reject an F16 provider; use V2.
// Snapshot and abort are format-independent. Byte counters count valid wire
// payload at real arrivals/releases, including prepared zero inputs; they are
// protocol counters, not measurements of PCIe transactions.
// Poll is finite and never waits for a submission lock or a missing rank.
// Returns 1 for an event, 0 for no event, and -1 for unsupported/failed.
GGML_API int  ggml_backend_relay_poll(ggml_backend_t backend, struct ggml_relay_event * event);
// Payload writes must be complete before this call. The event must match the
// outstanding INPUT_READY view, including epoch, plan, channel and output.
// After abort this call returns the CPU borrow, returns false, and never
// publishes a success generation. Abort alone does not end a CPU borrow.
GGML_API bool ggml_backend_relay_publish(ggml_backend_t backend, const struct ggml_relay_event * event);
// False also permits temporary contention; retry without holding a provider
// view or lock. Capability discovery must precede concurrent execution.
GGML_API bool ggml_backend_relay_snapshot(ggml_backend_t backend, struct ggml_relay_status * status);
// Experimental HC4/K320 context owner offset (0..4), configured exactly once
// after context creation and before its first real graph_compute/submit.
// Requires GGML_TP5_HC_CONTEXT_OWNER=1 before model loading: 97 complete BF16
// HC pairs are physically mirrored and immutable across contexts. The private
// execution plan preserves 97 HC joins and all 181 relay channels. Unsupported
// graphs fail before submission; they never fall back to five contributors.
// Graph reservation alone does not freeze this setter. Old event/provider ABI
// and ordinary contexts remain unchanged. Callers must serialize configuration.
GGML_API bool ggml_backend_relay_set_hc_owner_offset(ggml_backend_t backend, uint32_t offset);

// Select once before the first prepare/submission, with no concurrent service.
// F32 is the default. Unsupported formats or a started provider return false.
GGML_API bool ggml_backend_relay_set_wire_type(ggml_backend_t backend, enum ggml_relay_wire_type wire_type);
// V2 uses the same borrow/release protocol as V1. event_size must be at least
// sizeof(ggml_relay_event_v2); only that prefix is read/written. INPUT_READY is
// returned only after all five real ready words. Payload pointers and every
// identity/format/extent/mask field must match the outstanding view on publish.
// The caller folds only computed inputs in ascending rank order with F32
// accumulation, checks finite inputs/sum/encoded output, then publishes. An
// empty computed mask produces positive zero; skipped ranks still participate
// in all five producer/consumer ledger facts.
GGML_API int  ggml_backend_relay_poll_v2(
        ggml_backend_t backend, struct ggml_relay_event_v2 * event, size_t event_size);
GGML_API bool ggml_backend_relay_publish_v2(
        ggml_backend_t backend, const struct ggml_relay_event_v2 * event, size_t event_size);
// Sticky failure: stop normal publication and make bounded device waits exit.
GGML_API void ggml_backend_relay_abort(ggml_backend_t backend, int32_t error_code);

#ifdef __cplusplus
}
#endif
