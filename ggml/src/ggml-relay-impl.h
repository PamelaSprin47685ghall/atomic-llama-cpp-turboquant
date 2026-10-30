#pragma once

#include "ggml-relay.h"

#define GGML_RELAY_PROVIDER_ABI_VERSION 2

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
    bool (*submit)(void * provider, ggml_cgraph * const * graphs, size_t stages);
    int (*poll)(void * provider, ggml_relay_event * event);
    bool (*publish)(void * provider, const ggml_relay_event * event);
    bool (*snapshot)(void * provider, ggml_relay_status * status);
    void (*abort)(void * provider, int32_t error_code);
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
