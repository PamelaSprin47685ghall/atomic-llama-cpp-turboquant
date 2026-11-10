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
    // A plan was evicted or flushed. The event carries only its generation
    // and epoch; channel and payload fields are zero. Old consumers that do
    // not know this kind ignore it.
    GGML_RELAY_PLAN_RETIRED    = 3,
};

// TP5 pillar-5 plan identity. Seven coarse fields plus the channel count.
// Two epochs with the same key must be interchangeable for every captured
// graph. `channels` is part of the identity: a 180- and a 181-channel plan
// never share a plan slot. Zero lets the provider use the prepared stage
// count; any nonzero value must equal the count being prepared.
struct ggml_relay_plan_key {
    uint64_t model;       // model identity inside one process
    uint32_t window;      // service window ordinal (provider is per window)
    uint32_t n_tokens;    // rows in the stage graph
    uint32_t n_outputs;   // active output-gather width
    int32_t  rs_class;    // nextn offset << 8 | rs_row_armed << 4 | gtype
    uint32_t n_kv_bucket; // padded KV width bucket
    uint64_t alloc_gen;   // gallocr placement generation
    uint32_t channels;    // relay channel count
};

// Optional diagnostic view. The provider owns the slots; this is a copy.
struct ggml_relay_plan_snapshot {
    uint32_t valid_mask;
    uint32_t channels[8];
    uint64_t generation[8];
    uint64_t hits;
    uint64_t misses;
    uint64_t inserts;
    uint64_t evictions;
    uint64_t pinned_refusals;
    uint64_t flushes;
    uint64_t stale_generation;
    uint64_t pool_bytes;
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
// sealed records that finish sealed the epoch's submission at that time; it is
// not a current-health or success flag. A later abort/failure may leave
// sealed=true and failed=true. That combination is legal and does not mean
// success: callers must check failed (and poll() returning -1), never sealed
// alone. failed is sticky. finish/publish success publication and abort()'s
// sticky abort claim share one lock-free atomic claim word: a success CAS that
// increments first is ordered before a later abort; an abort fetch_or that
// sets the sticky bit first makes the success CAS fail. snapshot() reads the
// claim with a no-op CAS. The claim orders abort against host success
// publication; general fail() is provider-mutex ordered.
// error_code is the value passed to abort() when a host abort claimed the
// failure; otherwise it is a POSIX errno from the provider. Callers must not
// interpret one space as the other.
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
// A mismatched event returns false and keeps the borrow: the provider cannot
// prove the CPU reader stopped. Retry with the saved view.
GGML_API bool ggml_backend_relay_publish(ggml_backend_t backend, const struct ggml_relay_event * event);
// False also permits temporary contention; retry without holding a provider
// view or lock. A finished epoch whose outputs are all released has no
// pending fact; the returned status is what a full observation would produce.
// Capability discovery must precede concurrent execution.
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
// abort() is non-blocking: two RMWs (one pending-error CAS and one fetch_or to
// set the sticky abort bit), then the device abort word is published. It may
// then try_lock the provider mutex (non-blocking) to consume the abort into
// status; it never waits on that lock or a HIP launch/JIT/EventRecord holder.
// A return guarantees the claim/device word, not that status.failed is set;
// the next consuming entry sets it. The pending error is visible before the
// claim bit: poll(), snapshot(), prepare/submit/begin and finish() fail closed
// once they observe it. A host success that already passed that check and
// wins the claim CAS is still ordered before this abort by the claim word.
// The backend must outlive this call and must not be destroyed concurrently,
// as for any ggml_backend_t operation; the claim does not protect object
// lifetime.
GGML_API void ggml_backend_relay_abort(ggml_backend_t backend, int32_t error_code);

// Pillar 5: describe the next relay plan before its graph is submitted.
// The key is stored on the Meta backend and passed to the provider's
// prepare_ex when that capability exists. A null key or a backend without a
// relay provider is a no-op. Callers serialize configuration; llama calls it
// from the decode thread before graph compute.
GGML_API void ggml_backend_meta_set_plan_key(
        ggml_backend_t backend, const struct ggml_relay_plan_key * key);
// Optional diagnostic copy of the provider's plan cache. Returns false when
// the backend is not a relay or the provider predates the plan cache. This
// does not extend ggml_relay_status.
GGML_API bool ggml_backend_relay_plan_snapshot(
        ggml_backend_t backend, struct ggml_relay_plan_snapshot * snapshot);

// Per-submission provider token result. The numeric values match the generic
// backend completion result, so a Meta consumer can translate without
// inventing a second success code.
enum ggml_relay_token_result {
    GGML_RELAY_TOKEN_SUCCESS = 0,
    GGML_RELAY_TOKEN_FAILED  = 1,
    GGML_RELAY_TOKEN_UNKNOWN = 2,
};

// CPU-side facts for the current provider epoch: the single work borrow, the
// output-release drain, and the provider completion tokens. Returns 1 on
// success, 0 when the provider mutex is temporarily contended, and -1 when the
// backend is not a relay or the provider predates this query (ABI < 5).
// This does not extend ggml_relay_status. borrowed_channel is UINT32_MAX when
// no work view is outstanding. A borrow survives abort: the CPU reader may
// still own the mapped payload. Publish the saved INPUT_READY view to return
// it, or retain the owner. The provider never revokes a borrow on its own.
// tokens_outstanding counts every token record that release_token has not
// retired, including settled records; active_token is the current record or 0.
struct ggml_relay_cpu_status {
    uint64_t epoch;
    uint64_t channels;
    uint64_t outputs_released;
    uint64_t active_token;
    uint64_t tokens_outstanding;
    uint32_t borrowed_channel;
};
GGML_API int ggml_backend_relay_cpu_snapshot(
        ggml_backend_t backend, struct ggml_relay_cpu_status * status);

#ifdef __cplusplus
}
#endif
