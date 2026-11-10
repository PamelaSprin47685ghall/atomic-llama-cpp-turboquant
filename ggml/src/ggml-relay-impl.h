#pragma once

#include "ggml-relay.h"

// ABI 5 is the single unified table. The token trio occupies the first three
// appended slots and the CPU borrow/drain query is appended last. ABI 4 is
// ambiguous: two incompatible candidates used it for borrow_snapshot and for
// the token trio, so no ABI 4 provider is accepted. ABI 5 requires all four
// fields below.
#define GGML_RELAY_PROVIDER_ABI_VERSION 5

// Internal backend capability. No model, request or B-library types cross it.
// HIP owns buffers, queues, system-scope flags and completion observations.
// The external composer owns the CPU reducer and drives the public events.
struct ggml_relay_provider_i {
    uint32_t abi_version;
    void * (*create)(ggml_backend_t * backends, size_t ranks);
    // Return false and retain all native owners when completion is unproved.
    bool (*destroy)(void * provider);
    // Channel-major array: terminals[channel * GGML_RELAY_RANKS + rank].
    // Prepare all allocations before the first graph is submitted.
    // The owner calls prepare on every graph/layout/binding redefinition,
    // including unchanged terminal bindings. Metadata stays immutable until
    // the next prepare; input tensor contents may change after native drain.
    bool (*prepare)(void * provider, ggml_tensor * const * terminals, size_t channels);
    // Stage-major array: graphs[stage * GGML_RELAY_RANKS + rank]. HIP owns
    // warmup, complete-epoch capture/replay and native acceptance/drain. A
    // failure never permits a caller to continue execution on another path.
    // A true return is the only continuation permit: the provider clears
    // sealed if its plan-pin release or final observation fails. finish
    // claims a success on the lock-free abort claim word; an abort that claims
    // first makes it fail, an abort after the claim is a later failure.
    bool (*submit)(void * provider, ggml_cgraph * const * graphs, size_t stages);
    int (*poll)(void * provider, ggml_relay_event * event);
    bool (*publish)(void * provider, const ggml_relay_event * event);
    bool (*snapshot)(void * provider, ggml_relay_status * status);
    void (*abort)(void * provider, int32_t error_code);
    // ABI 3: prepare with the caller's plan identity. `key` may be null, in
    // which case the provider keeps the legacy behavior. The channel count in
    // the key is authoritative only after the provider validates it against
    // `channels`. Providers that predate ABI 3 leave this field absent; the
    // consumer must check abi_version before calling it.
    bool (*prepare_ex)(void * provider, ggml_tensor * const * terminals, size_t channels,
                       const ggml_relay_plan_key * key);
    // ABI 3: optional plan-cache diagnostic. Returns false when unavailable.
    bool (*plan_snapshot)(void * provider, ggml_relay_plan_snapshot * snapshot);
    // ABI 5: atomically submit and return a provider-owned per-plan completion
    // token. wait_token blocks on the provider completion event and reports
    // success/failure for exactly that token; it never polls or substitutes a
    // global epoch. release_token retires the token record. The token result
    // uses ggml_relay_token_result. Every token must be released before the
    // next begin: the record owns the lane tail events, and re-recording them
    // for a new epoch would retarget a later wait_token.
    bool (*submit_token)(void * provider, ggml_cgraph * const * graphs, size_t stages,
                         uint64_t * token);
    int  (*wait_token)(void * provider, uint64_t token, int * error);
    void (*release_token)(void * provider, uint64_t token);
    // ABI 5: CPU borrow/drain/token facts. False on provider-mutex contention;
    // the consumer retries and treats persistent contention as unproven.
    bool (*cpu_snapshot)(void * provider, ggml_relay_cpu_status * status);
};

typedef const ggml_relay_provider_i * (*ggml_backend_relay_get_interface_t)(void);

// Separate optional capability: the existing provider-v2 table and F32 ABI
// remain layout-compatible with old consumers/backends.
#define GGML_RELAY_WIRE_PROVIDER_ABI_VERSION 1
struct ggml_relay_wire_provider_i {
    uint32_t abi_version;
    bool (*set_wire_type)(void * provider, enum ggml_relay_wire_type wire_type);
    int (*poll_v2)(void * provider, ggml_relay_event_v2 * event, size_t event_size);
    bool (*publish_v2)(void * provider, const ggml_relay_event_v2 * event, size_t event_size);
};
typedef const ggml_relay_wire_provider_i * (*ggml_backend_relay_get_wire_interface_t)(void);
