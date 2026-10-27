#pragma once

#include "../ggml-relay-impl.h"

#if defined(GGML_USE_HIP)
// Strict relay replay is independent of GGML_HIP_GRAPHS (which must stay off).
// Each exact complete frame first runs directly to warm backend resources,
// then captures one entire stream-zero epoch per rank. Two provider-owned
// entries retain alternating private-arena bindings; any topology, tensor
// metadata, source/output address, fusion-use or numeric-policy change misses.
// Capture creates no native acceptance. All five executables must exist before
// the first launch; publication seals only after five actual accepted launches.
// Relay kernels read a mapped activation word, immutable until native tails
// and CPU readers drain. Executables die before their scratch/mapped storage;
// failed partial launches never fall back or release unproved native owners.
const ggml_relay_provider_i * ggml_backend_hip_relay_get_interface();
const ggml_relay_wire_provider_i * ggml_backend_hip_relay_get_wire_interface();
#endif
