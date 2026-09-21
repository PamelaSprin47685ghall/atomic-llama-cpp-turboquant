// Narrow internal bridge between the Vulkan backend and the TP5 collective
// transport (TP5.md 5.4, 14.4). The transport needs the already-initialized
// VkDevice/queue of each rank backend; it must not create its own devices.
//
// This header is private to ggml-vulkan: it is not installed and must not be
// included from llama model code. Only these declarations are exposed.

#pragma once

#include "ggml-backend.h"
#include "ggml-impl.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <array>

// Opaque device handle implemented in ggml-vulkan.cpp as vk_device.
// The typedef here must match ggml-vulkan.cpp's exactly.
struct vk_device_struct;
using vk_device = std::shared_ptr<vk_device_struct>;

struct vk_buffer_struct;
using vk_buffer = std::shared_ptr<vk_buffer_struct>;

// Extract the vk_device from a live Vulkan backend. Returns nullptr when the
// backend is not a Vulkan backend. The returned shared_ptr keeps the device
// alive; the transport holds one per rank for its lifetime.
vk_device ggml_vk_tp5_backend_device(ggml_backend_t backend);

// Device capabilities the transport must query (TP5.md 13.1).
struct vk_tp5_device_caps {
    bool external_memory_fd      = false;
    bool external_memory_dma_buf = false;
    bool device_coherent_memory = false;
    bool external_semaphore_fd   = false;
    bool timeline_semaphore      = false;
    bool timeline_semaphore_features = false;
    uint64_t max_timeline_semaphore_value_difference = 0;
    bool timeline_opaque_fd_export = false;
    bool timeline_opaque_fd_import = false;
    uint32_t timeline_opaque_fd_compatible = 0;
    uint32_t timeline_opaque_fd_export_from_import = 0;
    uint32_t timeline_opaque_fd_features = 0;
    uint64_t max_storage_buffer_range = 0;
    uint64_t min_storage_buffer_offset_alignment = 0;
    bool     storage_buffer_array_dynamic_indexing   = false;
    uint32_t max_storage_buffer_descriptors          = 0;
    float    timestamp_period                        = 0;
    uint32_t timestamp_valid_bits                    = 0;
    char device_uuid[16] = {};
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    char name[256] = {};
};
vk_tp5_device_caps ggml_vk_tp5_device_caps(vk_device device);

// Raw device access for the transport's Vulkan calls (allocation, import,
// barriers, command buffers). The transport serializes against the queue
// handle above; it never touches the device's own mutex-protected state.
struct VkPhysicalDevice_T * ggml_vk_tp5_vk_physical_device(vk_device device);
struct VkDevice_T * ggml_vk_tp5_vk_device(vk_device device);

// Additional bridge points consumed by ggml-vulkan-collective.cpp.
// (declared extern in the collective TU; defined in ggml-vulkan.cpp)
void ggml_vk_tp5_get_queue(vk_device device, struct VkQueue_T ** q, uint32_t * family);
void ggml_vk_tp5_mem_props(vk_device device, void * out); // VkPhysicalDeviceMemoryProperties*
bool ggml_vk_tp5_tensor_dev_ref(struct ggml_tensor * t, struct VkBuffer_T ** buf,
                                uint64_t * off, uint64_t * size, vk_buffer * owner = nullptr);
uint64_t ggml_vk_tp5_get_tensor_bda(struct ggml_tensor * t);

static constexpr size_t VK_TP5_RELAY_ROUTE_MAX = 128;
static constexpr size_t VK_TP5_RELAY_ROUTE_STRIDE = 256;
static constexpr uint32_t VK_TP5_RELAY_ROUTE_KEEP_LOCAL = 1u << 0;
struct vk_tp5_relay_route_entry {
    uint32_t bank     = 0;
    uint32_t ready    = 0;
    uint32_t epoch    = 0;
    uint32_t reserved = 0;
};
static_assert(sizeof(vk_tp5_relay_route_entry) == 16);
static_assert(VK_TP5_RELAY_ROUTE_STRIDE % sizeof(vk_tp5_relay_route_entry) == 0);

struct vk_tp5_relay_payload_binding {
    struct VkBuffer_T * bank[2] = {nullptr, nullptr};
    uint64_t            bytes[2] = {0, 0};
    uint64_t            generation = 0;
};

// Configure the next graph's terminal producer, without recording or waiting.
// A packed companion is available only after actual recording or validated
// replay. Taking it consumes the record, so repeated allreduces repack F32.
void ggml_vk_tp5_set_wire_output(ggml_backend_t backend, struct ggml_tensor * tensor,
                                 bool compute_consumer = false, size_t relay_stage = SIZE_MAX,
                                 bool relay_f32 = false);
// Mark one replay graph as eligible to publish its leading HC consumer recipe
// for Exact LateBind. This is deliberately independent of producer-wire: the
// legacy RELAY P1 path must be able to keep its original producer unchanged.
void ggml_vk_tp5_set_latebind_capture(ggml_backend_t backend, const struct ggml_cgraph * graph, bool enabled,
                                     bool define_program = false);
struct vk_tp5_graph_program;
// Definition-time lookup. The returned program owns immutable descriptors and
// buffers, so its commands may be lowered into an independent primary CB.
std::shared_ptr<const vk_tp5_graph_program> ggml_vk_tp5_graph_program(ggml_backend_t backend, void * first_cb);
bool ggml_vk_tp5_take_wire_output(ggml_backend_t       backend,
                                  struct ggml_tensor * tensor,
                                  struct VkBuffer_T ** buf,
                                  uint64_t *           off,
                                  uint64_t *           size,
                                  vk_buffer *          owner,
                                  bool *               relay_direct = nullptr,
                                  size_t               relay_stage = SIZE_MAX,
                                  size_t *             relay_stage_out = nullptr);
bool ggml_vk_tp5_get_relay_ready(ggml_backend_t backend, size_t stage, volatile uint32_t ** ready_ptr);
bool ggml_vk_tp5_update_relay_route(ggml_backend_t backend, size_t stage, uint32_t bank, uint64_t epoch,
                                    const vk_tp5_relay_payload_binding & payload,
                                    volatile uint32_t ** ready_ptr, bool keep_local = false);
void ggml_vk_tp5_clear_relay_payload_binding(ggml_backend_t backend);

struct vk_tp5_hc_binding {
    struct VkBuffer_T * buffer = nullptr;
    uint64_t            offset = 0;
    uint64_t            size   = 0;
    vk_buffer           owner;
};

struct vk_tp5_hc_sum {
    uint32_t                         width   = 0;
    float                            epsilon = 0;
    uint32_t                         streams = 0;
    uint32_t                         late_rank = 0;
    vk_tp5_hc_binding                block;
    // residual, gamma, combined output, normalized output, inject weights/input
    std::array<vk_tp5_hc_binding, 6> bindings;
    // Exact LateBind-TP: W_down and the post-SiLU low-rank output. These are
    // populated only for the first HC consumer after a TP boundary.
    vk_tp5_hc_binding                down_weight;
    vk_tp5_hc_binding                lo;
    vk_tp5_hc_binding                quantized;
};

// A recipe exists only for the committed first CB of a replay entry. Ordinary
// execution retains this CB; a chain may omit it only after producing both HC
// outputs in the preceding collective's P2, using the exact block binding.
bool ggml_vk_tp5_hc_consumer(ggml_backend_t backend, void * first_cb, vk_tp5_hc_sum * recipe);
// relay=false selects the five-slot TIMELINE sum consumer; relay=true selects
// the bounded local-VRAM generation wait + CPU-reduced F32 consumer.
bool ggml_vk_tp5_hc_sum_pipeline(vk_device                         device,
                                 bool                              quantized,
                                 bool                              relay,
                                 struct VkPipeline_T **            pipeline,
                                 struct VkPipelineLayout_T **      layout,
                                 struct VkDescriptorSetLayout_T ** dsl);

// Transfer (SDMA) queue for the collective's cross-device P2P copies.
// Returns false when the device has no distinct transfer queue, in which case
// the transport falls back to the compute queue. The graphics ring is not used:
// P2P copies there stall the GFX ring (measured ring gfx_0.0.0 timeouts).
bool ggml_vk_tp5_transfer_queue(vk_device device, struct VkQueue_T ** q, uint32_t * family);

void ggml_vk_tp5_sync_device(vk_device device);
void ggml_vk_tp5_flush_async(ggml_backend_t backend);
void ggml_vk_tp5_mark_queue_submitted(ggml_backend_t backend);
