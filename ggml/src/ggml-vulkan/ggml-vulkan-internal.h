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

// Configure the next graph's terminal producer, without recording or waiting.
// A packed companion is available only after actual recording or validated
// replay. Taking it consumes the record, so repeated allreduces repack F32.
void ggml_vk_tp5_set_wire_output(ggml_backend_t backend, struct ggml_tensor * tensor);
bool ggml_vk_tp5_take_wire_output(ggml_backend_t       backend,
                                  struct ggml_tensor * tensor,
                                  struct VkBuffer_T ** buf,
                                  uint64_t *           off,
                                  uint64_t *           size,
                                  vk_buffer *          owner);

struct vk_tp5_hc_binding {
    struct VkBuffer_T * buffer = nullptr;
    uint64_t            offset = 0;
    uint64_t            size   = 0;
    vk_buffer           owner;
};

struct vk_tp5_hc_sum {
    uint32_t                         width   = 0;
    float                            epsilon = 0;
    vk_tp5_hc_binding                block;
    // residual, gamma, combined output, normalized output, inject weights/input
    std::array<vk_tp5_hc_binding, 6> bindings;
    vk_tp5_hc_binding                quantized;
};

// A recipe exists only for the committed first CB of a replay entry. Ordinary
// execution retains this CB; a chain may omit it only after producing both HC
// outputs in the preceding collective's P2, using the exact block binding.
bool ggml_vk_tp5_hc_consumer(ggml_backend_t backend, void * first_cb, vk_tp5_hc_sum * recipe);
bool ggml_vk_tp5_hc_sum_pipeline(vk_device                         device,
                                 bool                              quantized,
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
