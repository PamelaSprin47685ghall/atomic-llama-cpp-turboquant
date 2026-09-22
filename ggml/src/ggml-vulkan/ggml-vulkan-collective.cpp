// TP5 collective transport for the Vulkan backend (TP5.md sections 11-14).
//
// Implements:
//   ggml_backend_comm_init             -> ggml_backend_vk_tp5_comm_init
//   ggml_backend_comm_free             -> ggml_backend_vk_tp5_comm_free
//   ggml_backend_comm_allreduce_tensor -> ggml_backend_vk_tp5_allreduce_tensor
//
// Hardware-proven 2-stage execution (matches test-vulkan-p2p-allreduce.cpp):
//   Phase 1: Local tensor -> canonical wire copy -> PUSH to own slot + 4 peer slots.
//   BARRIER: In host mode, wait for all 5 Phase 1 fences to ensure all 20 PCIe
//            DMA PUSH transfers have 100% physically landed in destination VRAM!
//   Phase 2: Local compute sum (reads 5 slots in fixed rank order, FP32 acc) ->
//            writes back into local tensor. Wait for all 5 Phase 2 fences.
//
// Cross-device DMA cannot be synchronized by a local pipeline barrier: host fence
// separation (or SYNC_FD timeline) across phases is mathematically required.

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vulkan-internal.h"
#include "ggml-vulkan-tp5-command.h"
#include "ggml-vulkan-shaders.hpp"
#include "ggml-vulkan.h"
#include "ggml-tp5-profile.h"
#include "ggml-vulkan-collective.hpp"
#include "ggml-vulkan-relay.h"
#include "ggml-vulkan-tp5-rows.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <fcntl.h>
#include <xf86drm.h>
#include <climits>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

struct tp5_fd {
    int fd = -1;
    explicit tp5_fd(int f = -1) : fd(f) {}
    tp5_fd(const tp5_fd &) = delete;
    tp5_fd & operator=(const tp5_fd &) = delete;
    ~tp5_fd() { if (fd >= 0) ::close(fd); }
    int release() { int f = fd; fd = -1; return f; }
};

namespace {

enum class tp5_wire_type { F32, F16 };
enum class tp5_sync_mode { HOST, SYNCFD, TIMELINE, GPUFLAG, DRM, STAR, RELAY };

static constexpr size_t TP5_MAILBOX_BANKS = 2;
static constexpr size_t TP5_LATE_MAX_FLOATS = 8192; // up to 4 rows * 4 streams * rank <= 512
enum class tp5_numerical_mode {
    REFERENCE,        // 未启用 LateBind（REFERENCE/回退模式）：常规无 LateBind 路径，完全遵循原图/标准 AllReduce 数值
    EXACT_F32,        // 调度收益但数学严格等价（EXACT_F32 模式）：启用 LateBind 解耦提前搬运，但 Q 充分统计量保持 FP32 精确路径无量化
    AGGRESSIVE_Q8,    // Aggressive 模式：启用 LateBind + Q8 激活量化 + Q8xQ8 整数点积 + FP16 sidecar 紧凑传输
    P1A_NOSIDECAR_Q8  // P1-A 对照器具：不带 sidecar 的 aggressive Q8 HC（同精度 Q8dot/量化/fold，在 Y 就绪后本地串行执行）
};

enum class tp5_numerical_reason {
    NONE,
    DISABLED_BY_ENV,
    NON_RELAY_SYNC,
    NON_F32_WIRE,
    NO_LATE_TENSORS,
    EXACT_Q_REQUESTED,
    MISSING_HARDWARE_INT_DOT,
    UNSUPPORTED_WAVE32,
    UNALIGNED_LATE_SHAPE,
    PIPELINE_UNAVAILABLE,
    P1A_BENCH_REQUESTED
};

static inline const char * tp5_numerical_mode_name(tp5_numerical_mode mode) {
    switch (mode) {
        case tp5_numerical_mode::REFERENCE:        return "reference";
        case tp5_numerical_mode::EXACT_F32:        return "exact-f32";
        case tp5_numerical_mode::AGGRESSIVE_Q8:    return "aggressive-q8";
        case tp5_numerical_mode::P1A_NOSIDECAR_Q8: return "p1a-nosidecar-q8";
        default:                                   return "unknown";
    }
}

static inline const char * tp5_numerical_mode_desc(tp5_numerical_mode mode) {
    switch (mode) {
        case tp5_numerical_mode::REFERENCE:        return "reference-no-latebind";
        case tp5_numerical_mode::EXACT_F32:        return "exact-f32-strict-math";
        case tp5_numerical_mode::AGGRESSIVE_Q8:    return "aggressive-q8-quantized";
        case tp5_numerical_mode::P1A_NOSIDECAR_Q8: return "p1a-nosidecar-q8-bench";
        default:                                   return "unknown";
    }
}

static inline const char * tp5_numerical_reason_name(tp5_numerical_reason reason) {
    switch (reason) {
        case tp5_numerical_reason::NONE:                     return "none";
        case tp5_numerical_reason::DISABLED_BY_ENV:          return "disabled-by-env";
        case tp5_numerical_reason::NON_RELAY_SYNC:           return "non-relay-sync";
        case tp5_numerical_reason::NON_F32_WIRE:             return "non-f32-wire";
        case tp5_numerical_reason::NO_LATE_TENSORS:          return "no-late-tensors";
        case tp5_numerical_reason::EXACT_Q_REQUESTED:        return "exact-q-requested";
        case tp5_numerical_reason::MISSING_HARDWARE_INT_DOT: return "missing-hardware-int-dot";
        case tp5_numerical_reason::UNSUPPORTED_WAVE32:       return "unsupported-wave32";
        case tp5_numerical_reason::UNALIGNED_LATE_SHAPE:     return "unaligned-late-shape";
        case tp5_numerical_reason::PIPELINE_UNAVAILABLE:     return "pipeline-unavailable";
        case tp5_numerical_reason::P1A_BENCH_REQUESTED:      return "p1a-bench-requested";
        default:                                             return "unknown";
    }
}

struct tp5_numerical_spec {
    bool latebind_env_enabled    = false;
    bool is_relay_sync           = false;
    bool is_f32_wire             = false;
    bool has_late_tensors        = false;
    bool exact_q_requested       = false;
    bool p1a_nosidecar_requested = false;
    bool hw_int_dot              = false;
    bool hw_wave32               = false;
    bool shape_aligned           = false;
    bool pipeline_ready          = false;
};

static inline std::pair<tp5_numerical_mode, tp5_numerical_reason> tp5_resolve_numerical_mode(
        const tp5_numerical_spec & spec) {
    if (spec.p1a_nosidecar_requested) {
        return { tp5_numerical_mode::P1A_NOSIDECAR_Q8, tp5_numerical_reason::NONE };
    }
    if (!spec.latebind_env_enabled) {
        return { tp5_numerical_mode::REFERENCE, tp5_numerical_reason::DISABLED_BY_ENV };
    }
    if (!spec.is_relay_sync) {
        return { tp5_numerical_mode::REFERENCE, tp5_numerical_reason::NON_RELAY_SYNC };
    }
    if (!spec.is_f32_wire) {
        return { tp5_numerical_mode::REFERENCE, tp5_numerical_reason::NON_F32_WIRE };
    }
    if (!spec.has_late_tensors) {
        return { tp5_numerical_mode::REFERENCE, tp5_numerical_reason::NO_LATE_TENSORS };
    }
    if (spec.exact_q_requested) {
        return { tp5_numerical_mode::EXACT_F32, tp5_numerical_reason::EXACT_Q_REQUESTED };
    }
    if (!spec.hw_int_dot) {
        return { tp5_numerical_mode::EXACT_F32, tp5_numerical_reason::MISSING_HARDWARE_INT_DOT };
    }
    if (!spec.hw_wave32) {
        return { tp5_numerical_mode::EXACT_F32, tp5_numerical_reason::UNSUPPORTED_WAVE32 };
    }
    if (!spec.shape_aligned) {
        return { tp5_numerical_mode::EXACT_F32, tp5_numerical_reason::UNALIGNED_LATE_SHAPE };
    }
    if (!spec.pipeline_ready) {
        return { tp5_numerical_mode::EXACT_F32, tp5_numerical_reason::PIPELINE_UNAVAILABLE };
    }
    return { tp5_numerical_mode::AGGRESSIVE_Q8, tp5_numerical_reason::NONE };
}

static constexpr size_t TP5_RELAY_HEADER_BYTES = 64;
static constexpr size_t TP5_LATE_Q_CONTROL_BYTES = 64;
static constexpr size_t TP5_LATE_Q_READY_WORD = 0;
static constexpr size_t TP5_LATE_Q_COUNTER_WORD = 1;

// LateBind Q sidecar layout helper (single source of truth for both CPU and GPU push constants).
// In aggressive Q8 fast path, the 64-byte control area lives at B + 64 + L, and payload starts at B + 128 + L.
// In exact fallback path, control uses status[6] (host-imported RAM), and payload broadcast destination is B + 64 + L.
static inline size_t tp5_late_q_control_offset(size_t late_host_offset) {
    return TP5_RELAY_HEADER_BYTES + late_host_offset;
}

static inline volatile uint32_t * tp5_late_q_control_ptr(void * bcast_host, size_t late_host_offset) {
    if (!bcast_host || late_host_offset == 0) return nullptr;
    return (volatile uint32_t *) ((char *) bcast_host + tp5_late_q_control_offset(late_host_offset));
}

static inline const volatile uint32_t * tp5_late_q_control_cptr(const void * bcast_host, size_t late_host_offset) {
    if (!bcast_host || late_host_offset == 0) return nullptr;
    return (const volatile uint32_t *) ((const char *) bcast_host + tp5_late_q_control_offset(late_host_offset));
}

static inline size_t tp5_late_q_bcast_payload_offset(size_t late_host_offset, bool late_q8_fast) {
    return TP5_RELAY_HEADER_BYTES + late_host_offset + (late_q8_fast ? TP5_LATE_Q_CONTROL_BYTES : 0);
}

static inline uint32_t tp5_late_q_payload_word_offset(size_t late_host_offset, bool late_q8_fast) {
    return (uint32_t) (tp5_late_q_bcast_payload_offset(late_host_offset, late_q8_fast) / 4);
}

static bool tp5_latebind_hc_enabled() {
    const char * env = getenv("GGML_TP5_LATEBIND");
    return env && (strcmp(env, "hc-down") == 0 || strcmp(env, "1") == 0 || strcmp(env, "on") == 0);
}

static bool tp5_latebind_fused_finalize_enabled() {
    const char * env = getenv("GGML_TP5_LATEBIND_FUSED_FINALIZE");
    return env && atoi(env) != 0;
}

static bool tp5_latebind_exact_q_requested() {
    const char * env = getenv("GGML_TP5_LATEBIND_EXACT_Q");
    return env && atoi(env) != 0;
}

// P1-A experimental comparison instrument: aggressive Q8 HC without sidecar.
// Measurement only; defaults to OFF. When OFF, existing paths are completely untouched.
static bool tp5_p1a_nosidecar_q8_requested() {
    const char * env = getenv("GGML_TP5_P1A_NOSIDECAR_Q8");
    return env && (strcmp(env, "1") == 0 || strcmp(env, "on") == 0 || strcmp(env, "true") == 0);
}

static size_t tp5_mailbox_bank(uint64_t epoch) {
    return (size_t) ((epoch - 1) & 1);
}

static std::string tp5_discover_dri_render_node(ggml_backend_t backend, vk_device device) {
    // Method 1: Query VkPhysicalDeviceDrmPropertiesEXT and PCIBusInfo directly from physical device
    if (device) {
        VkPhysicalDevice phys_dev = (VkPhysicalDevice) ggml_vk_tp5_vk_physical_device(device);
        if (phys_dev != VK_NULL_HANDLE) {
            VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            VkPhysicalDeviceDrmPropertiesEXT drm_props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT};
            VkPhysicalDevicePCIBusInfoPropertiesEXT pci_props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
            drm_props.pNext = &pci_props;
            props2.pNext = &drm_props;
            vkGetPhysicalDeviceProperties2(phys_dev, &props2);
            if (drm_props.renderMinor >= 128) {
                char path[64];
                snprintf(path, sizeof(path), "/dev/dri/renderD%d", (int) drm_props.renderMinor);
                if (access(path, R_OK | W_OK) == 0) {
                    return std::string(path);
                }
            }
            char by_path[128];
            snprintf(by_path, sizeof(by_path), "/dev/dri/by-path/pci-%04x:%02x:%02x.%x-render",
                     pci_props.pciDomain, pci_props.pciBus, pci_props.pciDevice, pci_props.pciFunction);
            char resolved[PATH_MAX];
            if (realpath(by_path, resolved) && access(resolved, R_OK | W_OK) == 0) {
                return std::string(resolved);
            }
        }
    }

    // Method 2: Match PCI address from ggml_backend_dev_props against libdrm drmGetDevices2
    if (backend) {
        ggml_backend_dev_t bdev = ggml_backend_get_device(backend);
        if (bdev) {
            ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(bdev, &props);
            if (props.device_id && props.device_id[0]) {
                drmDevicePtr devices[64];
                int num_devs = drmGetDevices2(0, devices, 64);
                if (num_devs > 0) {
                    std::string matched_node;
                    for (int d = 0; d < num_devs; ++d) {
                        if (devices[d]->bustype == DRM_BUS_PCI && devices[d]->businfo.pci) {
                            char pci_buf[32];
                            snprintf(pci_buf, sizeof(pci_buf), "%04x:%02x:%02x.%x",
                                     devices[d]->businfo.pci->domain,
                                     devices[d]->businfo.pci->bus,
                                     devices[d]->businfo.pci->dev,
                                     devices[d]->businfo.pci->func);
                            if (strcasecmp(pci_buf, props.device_id) == 0) {
                                if (devices[d]->available_nodes & (1 << DRM_NODE_RENDER)) {
                                    matched_node = devices[d]->nodes[DRM_NODE_RENDER];
                                }
                                break;
                            }
                        }
                    }
                    drmFreeDevices(devices, num_devs);
                    if (!matched_node.empty() && access(matched_node.c_str(), R_OK | W_OK) == 0) {
                        return matched_node;
                    }
                }

                // Method 3: Check /dev/dri/by-path/pci-<PCI>-render
                char by_path[128];
                snprintf(by_path, sizeof(by_path), "/dev/dri/by-path/pci-%s-render", props.device_id);
                char resolved[PATH_MAX];
                if (realpath(by_path, resolved) && access(resolved, R_OK | W_OK) == 0) {
                    return std::string(resolved);
                }
            }
        }
    }

    return "";
}

static size_t tp5_plan_slot(size_t rank, size_t bank) {
    return rank * TP5_MAILBOX_BANKS + bank;
}

static VkDeviceSize tp5_bank_bytes(size_t n_ranks, VkDeviceSize stride) {
    return (VkDeviceSize) n_ranks * stride;
}

static VkDeviceSize tp5_flags_byte_offset(size_t n_ranks, VkDeviceSize stride) {
    return (VkDeviceSize) TP5_MAILBOX_BANKS * tp5_bank_bytes(n_ranks, stride);
}

static VkDeviceSize tp5_flags_region_bytes(size_t n_ranks, tp5_sync_mode sync_mode) {
    const size_t n_words = (sync_mode == tp5_sync_mode::GPUFLAG) ? (2 * n_ranks + 1) : n_ranks;
    return (VkDeviceSize) n_words * sizeof(uint32_t);
}

static uint32_t tp5_default_spin_max() {
    constexpr long max_bounded_spin = 100000000L;
    const char * env = getenv("GGML_TP5_SPIN_MAX");
    if (env && env[0]) {
        const long v = strtol(env, nullptr, 10);
        // This fixed startup budget covers the complete GPU -> CPU -> GPU
        // handoff. It is bounded above and is never enlarged after a timeout.
        if (v > 0) return (uint32_t) std::min<long>(v, max_bounded_spin);
    }
    // Keep a missed handoff below the amdgpu watchdog window.  The host
    // protocol reports this bounded failure through status[2].
    return (uint32_t) max_bounded_spin;
}

static uint32_t tp5_default_relay_handoff_timeout_ms() {
    constexpr long max_bounded_timeout_ms = 10000L;
    const char * env = getenv("GGML_TP5_RELAY_HANDOFF_TIMEOUT_MS");
    if (env && env[0]) {
        const long v = strtol(env, nullptr, 10);
        if (v > 0) return (uint32_t) std::min<long>(v, max_bounded_timeout_ms);
    }
    return 2000u;
}

struct tp5_rank {
    vk_device device;
    VkDevice vkdev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;

    // Pipeline executable statistics (RADV codegen evidence: VGPR/SGPR/LDS/
    // spill) for the TP5 collective kernels, gated on the same extension as
    // GGML_VK_PIPELINE_STATS. Filter selects by pipeline name substring;
    // empty filter matches every TP5 kernel.
    bool pipeline_stats = false;
    char pipeline_stats_filter[64] {};

    uint32_t queue_family = 0;
    // Distinct SDMA transfer queue when available; cross-device PUSH copies run here
    // (the graphics ring stalls on P2P copies: measured ring gfx_0.0.0 timeouts).
    VkQueue transfer_queue = VK_NULL_HANDLE;
    uint32_t transfer_family = 0;
    bool has_transfer = false;
    vk_tp5_device_caps caps{};

    VkCommandPool cmd_pool = VK_NULL_HANDLE;

    // Optional one-shot replay-chain diagnostic. Never reset while in flight.
    VkQueryPool                  timing_pool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> timing_markers;
    bool                         timing_reported = false;

    // Fences: separate Phase 1 and Phase 2
    VkFence fence_p1 = VK_NULL_HANDLE;
    VkFence fence_p2 = VK_NULL_HANDLE;
    VkFence fence_ring[4] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};

    // Mailbox: device-local buffer with slots (slot r = data pushed by rank r)
    VkBuffer mailbox_buf = VK_NULL_HANDLE;
    VkDeviceMemory mailbox_mem = VK_NULL_HANDLE;
    VkDeviceSize mailbox_bytes = 0;
    VkDeviceSize mailbox_alloc_bytes = 0; // exporter vkAllocateMemory size (import must match)
    int mailbox_export_fd = -1;

    struct inbox {
        VkBuffer       buf         = VK_NULL_HANDLE;
        VkDeviceMemory mem         = VK_NULL_HANDLE;
        VkDeviceSize   alloc_bytes = 0;
        int            fd          = -1;
    };

    std::vector<inbox> inboxes;  // one allocation per sender and bank; no shared reservation

    struct imported {
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        int peer = -1;
        size_t         bank = 0;
    };
    std::vector<imported> imports;

    // Star AllReduce BDA & Broadcast Resources (imported via VK_EXT_external_memory_host)
    uint64_t bda_addr[TP5_MAILBOX_BANKS] = {0, 0};
    VkBuffer host_import_buf[TP5_MAILBOX_BANKS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory host_import_mem[TP5_MAILBOX_BANKS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkBuffer bcast_buf[TP5_MAILBOX_BANKS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory bcast_mem[TP5_MAILBOX_BANKS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    void * bcast_host[TP5_MAILBOX_BANKS] = {nullptr, nullptr};
    VkPipeline bda_push_pipe = VK_NULL_HANDLE;
    VkPipelineLayout bda_push_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout bda_push_dsl = VK_NULL_HANDLE;
    VkPipeline relay_copy_pipe = VK_NULL_HANDLE;
    VkPipelineLayout relay_copy_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout relay_copy_dsl = VK_NULL_HANDLE;
    VkPipeline late_inject_pipe = VK_NULL_HANDLE;
    VkPipelineLayout late_inject_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout late_inject_dsl = VK_NULL_HANDLE;
    VkPipeline late_q_pipe = VK_NULL_HANDLE;
    VkPipelineLayout late_q_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout late_q_dsl = VK_NULL_HANDLE;
    VkPipeline late_act_q8_pipe = VK_NULL_HANDLE;
    VkPipelineLayout late_act_q8_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout late_act_q8_dsl = VK_NULL_HANDLE;
    VkPipeline late_q8dot_pipe = VK_NULL_HANDLE;
    VkPipelineLayout late_q8dot_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout late_q8dot_dsl = VK_NULL_HANDLE;
    VkPipeline late_publish_pipe = VK_NULL_HANDLE;
    VkPipelineLayout late_publish_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout late_publish_dsl = VK_NULL_HANDLE;
    VkPipeline late_publish_f16_pipe = VK_NULL_HANDLE;
    VkPipelineLayout late_publish_f16_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout late_publish_f16_dsl = VK_NULL_HANDLE;
    VkPipeline late_norm_pipe = VK_NULL_HANDLE;
    VkPipelineLayout late_norm_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout late_norm_dsl = VK_NULL_HANDLE;
    VkPipeline late_lo_pipe = VK_NULL_HANDLE;
    VkPipelineLayout late_lo_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout late_lo_dsl = VK_NULL_HANDLE;
    VkPipeline late_lo_q8_pipe = VK_NULL_HANDLE;
    VkPipelineLayout late_lo_q8_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout late_lo_q8_dsl = VK_NULL_HANDLE;
    VkPipeline late_pack_pipe = VK_NULL_HANDLE;
    VkPipelineLayout late_pack_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout late_pack_dsl = VK_NULL_HANDLE;
    VkPipeline late_up_q8dot_pipe = VK_NULL_HANDLE;
    VkPipelineLayout late_up_q8dot_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout late_up_q8dot_dsl = VK_NULL_HANDLE;


    // Pre-allocated static descriptor sets and command buffers for Star AllReduce
    VkDescriptorSet star_ds_pack[TP5_MAILBOX_BANKS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet star_ds_bda[TP5_MAILBOX_BANKS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet star_ds_sum[TP5_MAILBOX_BANKS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkCommandBuffer star_cmd_p1[TP5_MAILBOX_BANKS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkCommandBuffer star_cmd_p2[TP5_MAILBOX_BANKS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::vector<VkCommandBuffer> star_chain_p1[TP5_MAILBOX_BANKS];
    std::vector<VkCommandBuffer> star_chain_p2[TP5_MAILBOX_BANKS];
    std::vector<VkDescriptorSet> star_chain_ds_pack[TP5_MAILBOX_BANKS];
    std::vector<VkDescriptorSet> star_chain_ds_sum[TP5_MAILBOX_BANKS];

    // Wire staging buffer (canonical wire dtype)
    VkBuffer wire_buf = VK_NULL_HANDLE;
    VkDeviceMemory wire_mem = VK_NULL_HANDLE;
    uint64_t wire_bda = 0;

    // Semaphores (for syncfd mode)
    VkSemaphore sem_p1_done = VK_NULL_HANDLE;
    std::vector<VkSemaphore> wait_sems;
    PFN_vkGetSemaphoreFdKHR pfn_get_sem_fd = nullptr;
    PFN_vkImportSemaphoreFdKHR pfn_import_sem_fd = nullptr;

    // Timeline semaphores (for timeline mode, persistent across workspace resize)
    VkSemaphore timeline_sem = VK_NULL_HANDLE; // own timeline semaphore (rank signals this)
    int timeline_export_fd = -1;
    std::vector<VkSemaphore> peer_timeline_sems; // imported peer timeline semaphores (size n_ranks)
    // Linux Native DRM Syncobj Timeline (Mode 3: bypasses Vulkan timeline wrapper overhead)
    int dri_fd = -1;
    uint32_t own_syncobj = 0;
    std::vector<uint32_t> peer_syncobjs;
    VkSemaphore host_ready_sem = VK_NULL_HANDLE;
    uint32_t host_ready_syncobj = 0;
    VkEvent host_ready_event[TP5_MAILBOX_BANKS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};

    PFN_vkSignalSemaphore pfn_signal_semaphore = nullptr;
    PFN_vkWaitSemaphores pfn_wait_semaphores = nullptr;
    PFN_vkGetSemaphoreCounterValue pfn_get_sem_counter = nullptr;
    // Highest own-timeline value known to have been accepted by vkQueueSubmit.
    // It is the only safe drain target after a partial multi-rank submission.
    uint64_t last_submitted_timeline_value = 0;

    // Pipelines
    VkPipeline sum_pipe = VK_NULL_HANDLE;
    VkPipeline pack_pipe = VK_NULL_HANDLE;
    VkPipeline pack_vec_pipe = VK_NULL_HANDLE;
    VkPipeline flag_pipe = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
    VkPipelineLayout flag_pipe_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkDescriptorSetLayout flag_dsl = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    // Borrowed from the backend device, retained by this rank's device owner.
    VkPipeline            hc_sum_pipe[2]      = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkPipelineLayout      hc_sum_layout[2]    = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDescriptorSetLayout hc_sum_dsl[2]       = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    // P2P parallel multicast compute shader: writes to local + 4 peer mailboxes concurrently
    VkPipeline            push_pipe           = VK_NULL_HANDLE;
    VkPipelineLayout      push_layout         = VK_NULL_HANDLE;
    VkDescriptorSetLayout push_dsl            = VK_NULL_HANDLE;
    // Descriptor-only 128-bit RELAY P1 transport.  This keeps the imported
    // host allocation out of BDA/global-BO-list policy while retaining the
    // coalesced one-WG copy pattern that was faster than scalar producer
    // stores on the target discrete GPUs.
    VkPipeline            relay_p1_copy_pipe   = VK_NULL_HANDLE;
    VkPipelineLayout      relay_p1_copy_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout relay_p1_copy_dsl    = VK_NULL_HANDLE;

    // Per-epoch params for GPUFLAG replay-safe dynamic seq (host-coherent)
    VkBuffer epoch_buf = VK_NULL_HANDLE;
    VkDeviceMemory epoch_mem = VK_NULL_HANDLE;
    uint32_t * epoch_host = nullptr;
};

// Own all pointer targets passed to vkQueueSubmit. A chain sizes its array
// before initializing these records; initialized records must not be moved.
struct tp5_timeline_batch {
    VkSemaphore                   waits[8]{};
    uint64_t                      values[8]{};
    VkPipelineStageFlags          stages[8]{};
    uint64_t                      signal = 0;
    bool                          is_star_batch = false;
    bool                          is_relay_batch = false;
    VkTimelineSemaphoreSubmitInfo timeline{ VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
    VkSubmitInfo                  submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };

    void init(tp5_rank &              rank,
              size_t                  rank_index,
              size_t                  n_ranks,
              uint64_t                epoch,
              bool                    phase1,
              const VkCommandBuffer * command,
              bool                    is_star = false,
              bool                    is_relay = false) {
        is_star_batch = is_star;
        is_relay_batch = is_relay;
        // R=2 credit is already implied by the existing dependency chain:
        // peer P2(e-2) -> peer P1(e-1) signal -> local P2(e-1) wait
        // -> local P1(e), ordered by the P1 compute/transfer barrier.
        // Thus P1(e) may overlap peer P2(e-1), but cannot overwrite a bank
        // still being read by peer P2(e-2). This is NOT valid for one bank.
        static_assert(TP5_MAILBOX_BANKS == 2, "transitive credit requires two banks");
        const uint64_t wait_value = phase1 ? 0 : 2 * epoch - 1;
        uint32_t       count      = 0;
        if (wait_value != 0) {
            if (is_relay) {
                // RELAY has no semaphore gate here. Its already-queued P2 (or
                // fused first consumer) performs the bounded local-VRAM
                // generation poll inside the compute dispatch itself.
            } else if (is_star) {
                waits[0]  = rank.host_ready_sem;
                values[0] = epoch;
                stages[0] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
                count     = 1;
            } else {
                for (size_t peer = 0; peer < n_ranks; ++peer) {
                    if (peer == rank_index)
                        continue;
                    waits[count]  = rank.peer_timeline_sems[peer];
                    values[count] = wait_value;
                    stages[count] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
                    ++count;
                }
            }
        }
        signal                             = 2 * epoch - (phase1 ? 1 : 0);
        timeline.waitSemaphoreValueCount   = count;
        timeline.pWaitSemaphoreValues      = count ? values : nullptr;
        timeline.signalSemaphoreValueCount = 1;
        timeline.pSignalSemaphoreValues    = &signal;
        submit.pNext                       = &timeline;
        submit.waitSemaphoreCount          = count;
        submit.pWaitSemaphores             = count ? waits : nullptr;
        submit.pWaitDstStageMask           = count ? stages : nullptr;
        submit.commandBufferCount          = 1;
        submit.pCommandBuffers             = command;
        submit.signalSemaphoreCount        = 1;
        submit.pSignalSemaphores           = &rank.timeline_sem;
    }

    void patch_epoch(uint64_t wait_epoch, uint64_t sig) {
        const uint64_t wait_val = (is_star_batch || is_relay_batch) ? wait_epoch : (wait_epoch ? 2 * wait_epoch - 1 : 0);
        for (uint32_t i = 0; i < timeline.waitSemaphoreValueCount; ++i) {
            values[i] = wait_val;
        }
        signal = sig;
    }
};

struct tp5_rank_chain {
    std::vector<tp5_timeline_batch> batches;
    std::vector<VkSubmitInfo>       submits;
    std::vector<VkCommandBuffer>    compute;

};

struct tp5_binding_key {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    bool operator==(const tp5_binding_key & o) const {
        return buf == o.buf && offset == o.offset && size == o.size;
    }
};

struct tp5_comm;
static bool tp5_drain_epoch(tp5_comm & c, uint64_t epoch, uint64_t timeout_ns = 5000000000ULL);
static bool tp5_drain_submitted(tp5_comm & c, uint64_t timeout_ns = 5000000000ULL);
static bool tp5_gpuflag_drain_epoch(tp5_comm & c, size_t ring_idx);
static bool tp5_gpuflag_drain_all(tp5_comm & c);

struct tp5_hc_key {
    uint32_t                       width        = 0;
    uint32_t                       epsilon_bits = 0;
    uint32_t                       streams      = 0;
    uint32_t                       late_rank    = 0;
    uint32_t                       capacity_rows = 1;
    std::array<tp5_binding_key, 6> bindings{};
    tp5_binding_key                down_weight{};
    tp5_binding_key                lo{};
    tp5_binding_key                up_weight{};
    tp5_binding_key                mixed{};
    tp5_binding_key                quantized{};

    bool operator==(const tp5_hc_key & other) const {
        return width == other.width && epsilon_bits == other.epsilon_bits &&
               streams == other.streams && late_rank == other.late_rank &&
               capacity_rows == other.capacity_rows &&
               bindings == other.bindings && down_weight == other.down_weight &&
               lo == other.lo && up_weight == other.up_weight && mixed == other.mixed &&
               quantized == other.quantized;
    }
};

struct tp5_plan_key {
    size_t n_elems = 0;
    tp5_wire_type wire = tp5_wire_type::F32;
    VkDeviceSize stride = 0;
    uint64_t workspace_gen = 0;
    std::vector<tp5_binding_key> bindings;
    std::vector<tp5_binding_key> packed_bindings;
    std::array<tp5_hc_key, 8>    hc{};
    std::array<tp5_hc_key, 8>    late{};
    std::array<bool, 8>           relay_direct{};

    bool operator==(const tp5_plan_key & o) const {
        if (n_elems != o.n_elems || wire != o.wire || stride != o.stride || workspace_gen != o.workspace_gen) {
            return false;
        }
        if (bindings.size() != o.bindings.size()) return false;
        for (size_t i = 0; i < bindings.size(); ++i) {
            if (!(bindings[i] == o.bindings[i])) return false;
        }
        if (packed_bindings.size() != o.packed_bindings.size())
            return false;
        for (size_t i = 0; i < packed_bindings.size(); ++i) {
            if (!(packed_bindings[i] == o.packed_bindings[i]))
                return false;
        }
        return hc == o.hc && late == o.late && relay_direct == o.relay_direct;
    }
};

struct tp5_compiled_chain {
    bool                                          valid         = false;
    size_t                                        n_stages      = 0;
    bool                                          isolate_bo    = false;
    size_t                                        start_bank    = 0;
    uint64_t                                      workspace_gen = 0;
    uint64_t                                      plans_gen     = 0;
    std::vector<size_t>                           stage_plan_indices;
    std::vector<tp5_plan_key>                     stage_keys;
    std::vector<std::vector<std::vector<void *>>> stage_compute_cbs;

    void invalidate() {
        valid = false;
        stage_plan_indices.clear();
        stage_keys.clear();
        stage_compute_cbs.clear();
    }
};

struct tp5_p1_rank_binding {
    VkBuffer     buf       = VK_NULL_HANDLE;
    VkDeviceSize offset    = 0;
    VkDeviceSize size      = 0;
    bool         is_packed = false;

    bool operator==(const tp5_p1_rank_binding & o) const {
        return buf == o.buf && offset == o.offset && size == o.size && is_packed == o.is_packed;
    }
};

struct tp5_p1_key {
    size_t                           n_elems       = 0;
    tp5_wire_type                    wire          = tp5_wire_type::F32;
    VkDeviceSize                     stride        = 0;
    uint64_t                         workspace_gen = 0;
    std::vector<tp5_p1_rank_binding> effective_bindings;

    bool operator==(const tp5_p1_key & o) const {
        return n_elems == o.n_elems && wire == o.wire && stride == o.stride && workspace_gen == o.workspace_gen &&
               effective_bindings == o.effective_bindings;
    }
};

struct tp5_p1_resources {
    tp5_p1_key                    key;
    std::vector<vk_buffer>        owners;
    std::vector<VkCommandBuffer> cmd_p1;
    std::vector<vk_tp5_command_tape> definitions;
    std::vector<VkDescriptorSet>  ds_pack;
    std::vector<VkDescriptorSet>  ds_relay_pack;
    std::vector<VkDescriptorSet>  ds_relay_copy;
    std::vector<VkDescriptorSet>  ds_flag;
    std::vector<VkDescriptorSet>  ds_push;
    std::vector<VkDevice>         devices;
    std::vector<VkCommandPool>    cmd_pools;
    std::vector<VkDescriptorPool> desc_pools;

    ~tp5_p1_resources() {
        for (size_t idx = 0; idx < cmd_p1.size(); ++idx) {
            const size_t i = idx / TP5_MAILBOX_BANKS;
            if (i >= devices.size() || i >= cmd_pools.size()) {
                break;
            }
            VkDevice      dev = devices[i];
            VkCommandPool cp  = cmd_pools[i];
            if (dev != VK_NULL_HANDLE && cp != VK_NULL_HANDLE && cmd_p1[idx] != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(dev, cp, 1, &cmd_p1[idx]);
                cmd_p1[idx] = VK_NULL_HANDLE;
            }
        }
        for (size_t i = 0; i < ds_pack.size(); ++i) {
            if (i >= devices.size() || i >= desc_pools.size()) {
                break;
            }
            VkDevice         dev = devices[i];
            VkDescriptorPool dp  = desc_pools[i];
            if (dev != VK_NULL_HANDLE && dp != VK_NULL_HANDLE && ds_pack[i] != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(dev, dp, 1, &ds_pack[i]);
                ds_pack[i] = VK_NULL_HANDLE;
            }
        }
        for (size_t idx = 0; idx < ds_relay_pack.size(); ++idx) {
            const size_t i = idx / TP5_MAILBOX_BANKS;
            if (i >= devices.size() || i >= desc_pools.size()) {
                break;
            }
            VkDevice         dev = devices[i];
            VkDescriptorPool dp  = desc_pools[i];
            if (dev != VK_NULL_HANDLE && dp != VK_NULL_HANDLE && ds_relay_pack[idx] != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(dev, dp, 1, &ds_relay_pack[idx]);
                ds_relay_pack[idx] = VK_NULL_HANDLE;
            }
        }
        for (size_t idx = 0; idx < ds_relay_copy.size(); ++idx) {
            const size_t i = idx / TP5_MAILBOX_BANKS;
            if (i >= devices.size() || i >= desc_pools.size()) {
                break;
            }
            VkDevice         dev = devices[i];
            VkDescriptorPool dp  = desc_pools[i];
            if (dev != VK_NULL_HANDLE && dp != VK_NULL_HANDLE && ds_relay_copy[idx] != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(dev, dp, 1, &ds_relay_copy[idx]);
                ds_relay_copy[idx] = VK_NULL_HANDLE;
            }
        }
        for (size_t i = 0; i < ds_flag.size(); ++i) {
            if (i >= devices.size() || i >= desc_pools.size()) {
                break;
            }
            VkDevice         dev = devices[i];
            VkDescriptorPool dp  = desc_pools[i];
            if (dev != VK_NULL_HANDLE && dp != VK_NULL_HANDLE && ds_flag[i] != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(dev, dp, 1, &ds_flag[i]);
                ds_flag[i] = VK_NULL_HANDLE;
            }
        }
        for (size_t idx = 0; idx < ds_push.size(); ++idx) {
            const size_t i = idx / TP5_MAILBOX_BANKS;
            if (i >= devices.size() || i >= desc_pools.size()) {
                break;
            }
            VkDevice         dev = devices[i];
            VkDescriptorPool dp  = desc_pools[i];
            if (dev != VK_NULL_HANDLE && dp != VK_NULL_HANDLE && ds_push[idx] != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(dev, dp, 1, &ds_push[idx]);
                ds_push[idx] = VK_NULL_HANDLE;
            }
        }
        owners.clear();
    }
};

struct tp5_cached_plan {
    tp5_plan_key key;
    // Recorded descriptors/CBs must not outlive their VkBuffer objects, even
    // after an epoch drains and before a cache entry is replayed or evicted.
    std::vector<vk_buffer>       owners;
    std::shared_ptr<tp5_p1_resources> p1;
    std::vector<VkCommandBuffer> cmd_p2;
    std::vector<VkCommandBuffer> cmd_late_pre;
    std::vector<vk_tp5_command_tape> pre_definitions;
    std::vector<vk_tp5_command_tape> p2_definitions;
    std::vector<size_t> late_inject_end;
    std::vector<size_t> late_q_begin;
    std::vector<size_t> late_q_contract_begin;
    std::vector<size_t> late_norm_end;
    std::vector<size_t> late_lo_begin;
    std::vector<VkDescriptorSet> ds_sum;
    std::vector<VkDescriptorSet> relay_ds;
    std::vector<VkDescriptorSet> late_inject_ds;
    std::vector<VkDescriptorSet> late_q_ds;
    std::vector<VkDescriptorSet> late_act_q8_ds;
    std::vector<VkDescriptorSet> late_q8dot_ds;
    std::vector<VkDescriptorSet> late_publish_ds;
    std::vector<VkDescriptorSet> late_norm_ds;
    std::vector<VkDescriptorSet> late_lo_ds;
    std::vector<VkDescriptorSet> late_lo_q8_ds;
    std::vector<VkDescriptorSet> late_up_q8dot_ds;
    std::vector<VkBuffer>        late_scatter_buf;
    std::vector<VkDeviceMemory>  late_scatter_mem;
    std::vector<VkBuffer>        late_rho_buf;
    std::vector<VkDeviceMemory>  late_rho_mem;
    std::vector<VkBuffer>        late_sidecar_buf;
    std::vector<VkDeviceMemory>  late_sidecar_mem;
    std::vector<VkBuffer>        late_act_q8_buf;
    std::vector<VkDeviceMemory>  late_act_q8_mem;
    std::vector<VkBuffer>        late_lo_q8_buf;
    std::vector<VkDeviceMemory>  late_lo_q8_mem;
    // P1-B packed weights: definition-time repack of W_down/W_up into
    // uint32-word layout; the pack dispatch is recorded at the head of
    // cmd_late_pre and self-disables via a persistent done flag.
    std::vector<VkBuffer>        late_down_packed_buf;
    std::vector<VkDeviceMemory>  late_down_packed_mem;
    std::vector<VkBuffer>        late_up_packed_buf;
    std::vector<VkDeviceMemory>  late_up_packed_mem;
    std::vector<VkDescriptorSet> late_pack_ds;
    tp5_numerical_mode           numerical_mode = tp5_numerical_mode::REFERENCE;
    tp5_numerical_reason         numerical_reason = tp5_numerical_reason::NONE;
    bool                         late_q8_fast = false;
    bool                         late_sidecar_f16 = false;
    std::vector<VkCommandBuffer> star_cmd_p1;
    std::vector<VkCommandBuffer> star_cmd_p2;
    uint64_t last_used_call = 0;
};

struct tensor_dev_ref {
    VkBuffer      buf    = VK_NULL_HANDLE;
    VkDeviceSize  offset = 0;
    VkDeviceSize  size   = 0;
    vk_buffer     owner;
    VkBuffer      packed_buf    = VK_NULL_HANDLE;
    VkDeviceSize  packed_offset = 0;
    VkDeviceSize  packed_size   = 0;
    vk_buffer     packed_owner;
    vk_tp5_hc_sum hc;
    vk_tp5_hc_sum late;
    bool          relay_direct = false;
    bool          ok = false;
};

struct tp5_linear_program {
    std::vector<vk_device> device_owners;
    std::vector<VkDevice> devices;
    std::vector<VkCommandPool> pools;
    std::vector<VkCommandBuffer> commands;
    std::vector<std::shared_ptr<const vk_tp5_graph_program>> graphs;
    // Definition-time semantic sequence record for validation & regression testing:
    std::vector<tp5_latebind_semantic_step> late_steps;
    ~tp5_linear_program() {
        // Owners release this object only after native drain, or before its
        // first submission. Rebuilds retain old programs until that drain.
        for (size_t i = 0; i < pools.size(); ++i)
            if (pools[i]) vkDestroyCommandPool(devices[i], pools[i], nullptr);
    }
};

struct tp5_comm {
    size_t n_ranks = 0;
    std::vector<tp5_rank> ranks;
    std::vector<ggml_backend_t> backends;
    std::vector<tp5_rank_chain> chain_scratch;
    uint64_t                    timing_chain       = 0;
    uint64_t                    chain_calls        = 0;
    size_t                      timing_stages      = 0;
    std::vector<uint8_t>        timing_late_stage;
    std::vector<uint8_t>        timing_direct_stage;
    bool                        timing_submitted   = false;
    tp5_wire_type wire = tp5_wire_type::F32;
    tp5_sync_mode sync_mode = tp5_sync_mode::HOST;

    bool cmd_replay_enabled = true;
    bool                        isolate_mailbox    = false;
    uint32_t spin_max = 100000000u;
    uint32_t relay_handoff_timeout_ms = 2000u;
    size_t max_elems = 0;
    uint64_t workspace_gen = 1;

    // Star AllReduce Host RAM & Infrastructure
    void * star_host_raw[TP5_MAILBOX_BANKS] = {nullptr, nullptr};
    void * star_host_aligned[TP5_MAILBOX_BANKS] = {nullptr, nullptr};
    volatile uint32_t * star_host_flag[TP5_MAILBOX_BANKS] = {nullptr, nullptr};
    bool relay_bank_used[TP5_MAILBOX_BANKS] = {false, false};
    size_t star_host_alloc_size = 0;
    size_t star_rank_stride = 0;
    size_t late_host_offset = 0;
    size_t late_max_floats = TP5_LATE_MAX_FLOATS;
    std::unique_ptr<tp5_avx2_pool> avx2_pool;
    std::unique_ptr<tp5_drm_signaler> drm_signaler;
    std::unique_ptr<tp5_submit_pool> submit_pool;
    bool star_chain_compiled[TP5_MAILBOX_BANKS] = {false, false};
    size_t star_chain_stages = 0;
    size_t star_chain_max_elems = 0;
    std::unique_ptr<tp5_drm_waiter> drm_waiter;

    // Bounded immutable per-binding plan cache
    // For 96 distinct subgraphs in Qwen4EXP TP5, keep 256 entries to eliminate LRU thrashing.
    static constexpr size_t MAX_CACHED_PLANS = 256;
    std::vector<tp5_cached_plan> cached_plans;
    size_t                       last_hit_idx = 0;
    uint64_t                     plans_gen    = 1;
    tp5_compiled_chain           compiled_chain;
    std::shared_ptr<tp5_linear_program> linear_program;
    std::vector<std::shared_ptr<tp5_linear_program>> retired_linear_programs;

    // Retained scratch buffers for chain validation/plan resolution across calls
    std::vector<tensor_dev_ref> chain_refs;
    std::vector<tp5_plan_key>   chain_keys;
    std::vector<size_t>         chain_plan_indices;

    uint64_t allreduce_calls = 0;
    uint64_t host_waits = 0;
    uint64_t last_drained_epoch = 0;
    static constexpr uint64_t MAX_OUTSTANDING_EPOCHS = 128;
    std::array<std::array<volatile uint32_t *, 8>, MAX_OUTSTANDING_EPOCHS> relay_ready_routes{};

    struct tp5_in_flight_slot {
        uint64_t epoch = 0;
        std::vector<vk_buffer> owners;
        std::shared_ptr<tp5_p1_resources> p1;
    };
    tp5_in_flight_slot in_flight_ring[MAX_OUTSTANDING_EPOCHS];

    std::mutex mutex;
    bool failed = false;
    std::string fail_reason;

    void fail(const std::string & why) {
        if (!failed) {
            failed = true;
            fail_reason = why;
            fprintf(stderr, "ggml-vulkan-collective: FAILURE: %s\n", why.c_str());
        }
    }

    void destroy_plan(tp5_cached_plan & plan) {
        invalidate_chain();
        plan.p1.reset();
        for (size_t idx = 0; idx < plan.cmd_p2.size(); ++idx) {
            const size_t i = idx / TP5_MAILBOX_BANKS;
            if (i >= n_ranks)
                break;
            tp5_rank & r = ranks[i];
            if (r.vkdev == VK_NULL_HANDLE) {
                continue;
            }
            if (idx < plan.cmd_p2.size() && plan.cmd_p2[idx] != VK_NULL_HANDLE && r.cmd_pool != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(r.vkdev, r.cmd_pool, 1, &plan.cmd_p2[idx]);
                plan.cmd_p2[idx] = VK_NULL_HANDLE;
            }
            if (idx < plan.cmd_late_pre.size() && plan.cmd_late_pre[idx] != VK_NULL_HANDLE &&
                r.cmd_pool != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(r.vkdev, r.cmd_pool, 1, &plan.cmd_late_pre[idx]);
                plan.cmd_late_pre[idx] = VK_NULL_HANDLE;
            }
            if (idx < plan.star_cmd_p1.size() && plan.star_cmd_p1[idx] != VK_NULL_HANDLE && r.cmd_pool != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(r.vkdev, r.cmd_pool, 1, &plan.star_cmd_p1[idx]);
                plan.star_cmd_p1[idx] = VK_NULL_HANDLE;
            }
            if (idx < plan.star_cmd_p2.size() && plan.star_cmd_p2[idx] != VK_NULL_HANDLE && r.cmd_pool != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(r.vkdev, r.cmd_pool, 1, &plan.star_cmd_p2[idx]);
                plan.star_cmd_p2[idx] = VK_NULL_HANDLE;
            }
            if (idx < plan.ds_sum.size() && plan.ds_sum[idx] != VK_NULL_HANDLE && r.desc_pool != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(r.vkdev, r.desc_pool, 1, &plan.ds_sum[idx]);
                plan.ds_sum[idx] = VK_NULL_HANDLE;
            }
            if (idx < plan.relay_ds.size() && plan.relay_ds[idx] != VK_NULL_HANDLE && r.desc_pool != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(r.vkdev, r.desc_pool, 1, &plan.relay_ds[idx]);
                plan.relay_ds[idx] = VK_NULL_HANDLE;
            }
            if (idx < plan.late_q_ds.size() && plan.late_q_ds[idx] != VK_NULL_HANDLE && r.desc_pool != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(r.vkdev, r.desc_pool, 1, &plan.late_q_ds[idx]);
                plan.late_q_ds[idx] = VK_NULL_HANDLE;
            }
            if (idx < plan.late_act_q8_ds.size() && plan.late_act_q8_ds[idx] != VK_NULL_HANDLE &&
                r.desc_pool != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(r.vkdev, r.desc_pool, 1, &plan.late_act_q8_ds[idx]);
                plan.late_act_q8_ds[idx] = VK_NULL_HANDLE;
            }
            if (idx < plan.late_q8dot_ds.size() && plan.late_q8dot_ds[idx] != VK_NULL_HANDLE &&
                r.desc_pool != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(r.vkdev, r.desc_pool, 1, &plan.late_q8dot_ds[idx]);
                plan.late_q8dot_ds[idx] = VK_NULL_HANDLE;
            }
            if (idx < plan.late_publish_ds.size() && plan.late_publish_ds[idx] != VK_NULL_HANDLE &&
                r.desc_pool != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(r.vkdev, r.desc_pool, 1, &plan.late_publish_ds[idx]);
                plan.late_publish_ds[idx] = VK_NULL_HANDLE;
            }
            if (idx < plan.late_norm_ds.size() && plan.late_norm_ds[idx] != VK_NULL_HANDLE &&
                r.desc_pool != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(r.vkdev, r.desc_pool, 1, &plan.late_norm_ds[idx]);
                plan.late_norm_ds[idx] = VK_NULL_HANDLE;
            }
            if (idx < plan.late_lo_ds.size() && plan.late_lo_ds[idx] != VK_NULL_HANDLE &&
                r.desc_pool != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(r.vkdev, r.desc_pool, 1, &plan.late_lo_ds[idx]);
                plan.late_lo_ds[idx] = VK_NULL_HANDLE;
            }
            if (idx < plan.late_lo_q8_ds.size() && plan.late_lo_q8_ds[idx] != VK_NULL_HANDLE &&
                r.desc_pool != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(r.vkdev, r.desc_pool, 1, &plan.late_lo_q8_ds[idx]);
                plan.late_lo_q8_ds[idx] = VK_NULL_HANDLE;
            }
            if (idx < plan.late_up_q8dot_ds.size() && plan.late_up_q8dot_ds[idx] != VK_NULL_HANDLE &&
                r.desc_pool != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(r.vkdev, r.desc_pool, 1, &plan.late_up_q8dot_ds[idx]);
                plan.late_up_q8dot_ds[idx] = VK_NULL_HANDLE;
            }
        }
        for (size_t idx = 0; idx < plan.late_inject_ds.size(); ++idx) {
            const size_t i = idx / TP5_MAILBOX_BANKS;
            if (i >= n_ranks) break;
            auto & r = ranks[i];
            if (idx < plan.late_inject_ds.size() && plan.late_inject_ds[idx] != VK_NULL_HANDLE &&
                r.vkdev != VK_NULL_HANDLE && r.desc_pool != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(r.vkdev, r.desc_pool, 1, &plan.late_inject_ds[idx]);
                plan.late_inject_ds[idx] = VK_NULL_HANDLE;
            }
        }
        for (size_t i = 0; i < plan.late_scatter_buf.size() && i < n_ranks; ++i) {
            auto & r = ranks[i];
            if (plan.late_scatter_buf[i] != VK_NULL_HANDLE && r.vkdev != VK_NULL_HANDLE) {
                vkDestroyBuffer(r.vkdev, plan.late_scatter_buf[i], nullptr);
                plan.late_scatter_buf[i] = VK_NULL_HANDLE;
            }
            if (i < plan.late_scatter_mem.size() && plan.late_scatter_mem[i] != VK_NULL_HANDLE &&
                r.vkdev != VK_NULL_HANDLE) {
                vkFreeMemory(r.vkdev, plan.late_scatter_mem[i], nullptr);
                plan.late_scatter_mem[i] = VK_NULL_HANDLE;
            }
            if (i < plan.late_rho_buf.size() && plan.late_rho_buf[i] && r.vkdev) {
                vkDestroyBuffer(r.vkdev, plan.late_rho_buf[i], nullptr);
                plan.late_rho_buf[i] = VK_NULL_HANDLE;
            }
            if (i < plan.late_rho_mem.size() && plan.late_rho_mem[i] && r.vkdev) {
                vkFreeMemory(r.vkdev, plan.late_rho_mem[i], nullptr);
                plan.late_rho_mem[i] = VK_NULL_HANDLE;
            }
            if (i < plan.late_sidecar_buf.size() && plan.late_sidecar_buf[i] != VK_NULL_HANDLE &&
                r.vkdev != VK_NULL_HANDLE) {
                vkDestroyBuffer(r.vkdev, plan.late_sidecar_buf[i], nullptr);
                plan.late_sidecar_buf[i] = VK_NULL_HANDLE;
            }
            if (i < plan.late_sidecar_mem.size() && plan.late_sidecar_mem[i] != VK_NULL_HANDLE &&
                r.vkdev != VK_NULL_HANDLE) {
                vkFreeMemory(r.vkdev, plan.late_sidecar_mem[i], nullptr);
                plan.late_sidecar_mem[i] = VK_NULL_HANDLE;
            }
            if (i < plan.late_act_q8_buf.size() && plan.late_act_q8_buf[i] != VK_NULL_HANDLE &&
                r.vkdev != VK_NULL_HANDLE) {
                vkDestroyBuffer(r.vkdev, plan.late_act_q8_buf[i], nullptr);
                plan.late_act_q8_buf[i] = VK_NULL_HANDLE;
            }
            if (i < plan.late_act_q8_mem.size() && plan.late_act_q8_mem[i] != VK_NULL_HANDLE &&
                r.vkdev != VK_NULL_HANDLE) {
                vkFreeMemory(r.vkdev, plan.late_act_q8_mem[i], nullptr);
                plan.late_act_q8_mem[i] = VK_NULL_HANDLE;
            }
            if (i < plan.late_lo_q8_buf.size() && plan.late_lo_q8_buf[i] != VK_NULL_HANDLE &&
                r.vkdev != VK_NULL_HANDLE) {
                vkDestroyBuffer(r.vkdev, plan.late_lo_q8_buf[i], nullptr);
                plan.late_lo_q8_buf[i] = VK_NULL_HANDLE;
            }
            if (i < plan.late_lo_q8_mem.size() && plan.late_lo_q8_mem[i] != VK_NULL_HANDLE &&
                r.vkdev != VK_NULL_HANDLE) {
                vkFreeMemory(r.vkdev, plan.late_lo_q8_mem[i], nullptr);
                plan.late_lo_q8_mem[i] = VK_NULL_HANDLE;
            }
            if (i < plan.late_down_packed_buf.size() && plan.late_down_packed_buf[i] != VK_NULL_HANDLE &&
                r.vkdev != VK_NULL_HANDLE) {
                vkDestroyBuffer(r.vkdev, plan.late_down_packed_buf[i], nullptr);
                plan.late_down_packed_buf[i] = VK_NULL_HANDLE;
            }
            if (i < plan.late_down_packed_mem.size() && plan.late_down_packed_mem[i] != VK_NULL_HANDLE &&
                r.vkdev != VK_NULL_HANDLE) {
                vkFreeMemory(r.vkdev, plan.late_down_packed_mem[i], nullptr);
                plan.late_down_packed_mem[i] = VK_NULL_HANDLE;
            }
            if (i < plan.late_up_packed_buf.size() && plan.late_up_packed_buf[i] != VK_NULL_HANDLE &&
                r.vkdev != VK_NULL_HANDLE) {
                vkDestroyBuffer(r.vkdev, plan.late_up_packed_buf[i], nullptr);
                plan.late_up_packed_buf[i] = VK_NULL_HANDLE;
            }
            if (i < plan.late_up_packed_mem.size() && plan.late_up_packed_mem[i] != VK_NULL_HANDLE &&
                r.vkdev != VK_NULL_HANDLE) {
                vkFreeMemory(r.vkdev, plan.late_up_packed_mem[i], nullptr);
                plan.late_up_packed_mem[i] = VK_NULL_HANDLE;
            }
        }
        plan.owners.clear();
    }

    void invalidate_chain() {
        compiled_chain.invalidate();
        plans_gen++;
    }

    void clear_cached_plans(bool queues_drained = false) {
        invalidate_chain();
        if ((sync_mode == tp5_sync_mode::TIMELINE || sync_mode == tp5_sync_mode::DRM ||
             sync_mode == tp5_sync_mode::RELAY) && !queues_drained) {
            if (!tp5_drain_epoch(*this, allreduce_calls)) {
                fail("clear_cached_plans: drain failed before plan destruction");
                return;
            }
        }
        if (sync_mode == tp5_sync_mode::GPUFLAG) {
            tp5_gpuflag_drain_all(*this);
        } else if (sync_mode != tp5_sync_mode::TIMELINE && sync_mode != tp5_sync_mode::DRM &&
                   sync_mode != tp5_sync_mode::RELAY) {
        for (auto & r : ranks) {
            if (r.vkdev != VK_NULL_HANDLE) {
                vkDeviceWaitIdle(r.vkdev);
            }
        }
        }
        // Cover actual submissions, including a partial rank submit whose
        // epoch never reached allreduce_calls. Never free emitted CB owners
        // on the strength of a logical epoch alone.
        if ((linear_program || !retired_linear_programs.empty()) && !tp5_drain_submitted(*this)) {
            fail("linear definition drain failed before resource release");
            return;
        }
        linear_program.reset();
        retired_linear_programs.clear();
        for (auto & plan : cached_plans) {
            destroy_plan(plan);
        }
        cached_plans.clear();
    }

    void evict_lru_plan() {
        if (cached_plans.empty()) return;
        size_t lru_idx = 0;
        uint64_t min_call = cached_plans[0].last_used_call;
        for (size_t i = 1; i < cached_plans.size(); ++i) {
            if (cached_plans[i].last_used_call < min_call) {
                min_call = cached_plans[i].last_used_call;
                lru_idx = i;
            }
        }
        if (sync_mode == tp5_sync_mode::TIMELINE || sync_mode == tp5_sync_mode::DRM ||
            sync_mode == tp5_sync_mode::RELAY) {
            if (!tp5_drain_epoch(*this, allreduce_calls)) {
                fail("evict_lru_plan: drain failed before plan eviction");
                return;
            }
        }
        if (sync_mode == tp5_sync_mode::GPUFLAG) {
            tp5_gpuflag_drain_all(*this);
        } else if (sync_mode != tp5_sync_mode::TIMELINE && sync_mode != tp5_sync_mode::DRM &&
                   sync_mode != tp5_sync_mode::RELAY) {
        for (auto & r : ranks) {
            if (r.vkdev != VK_NULL_HANDLE) {
                vkDeviceWaitIdle(r.vkdev);
            }
        }
        }
        if ((linear_program || !retired_linear_programs.empty()) && !tp5_drain_submitted(*this)) {
            fail("linear definition drain failed before plan eviction");
            return;
        }
        linear_program.reset();
        retired_linear_programs.clear();
        destroy_plan(cached_plans[lru_idx]);
        cached_plans.erase(cached_plans.begin() + lru_idx);
    }
};

uint32_t find_memory_type(const VkPhysicalDeviceMemoryProperties & props,
                          uint32_t type_bits, VkMemoryPropertyFlags req) {
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            ((props.memoryTypes[i].propertyFlags & req) == req)) {
            return i;
        }
    }
    return UINT32_MAX;
}

static void tp5_print_pipeline_statistics(VkDevice dev, VkPipeline pipeline, const char * pipe_name) {
    // Mirrors the main ggml-vulkan stats reporting (GGML_VK_PIPELINE_STATS)
    // for the TP5 collective kernels. Requires the pipeline to have been
    // created with VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR.
    PFN_vkGetPipelineExecutablePropertiesKHR get_props =
        (PFN_vkGetPipelineExecutablePropertiesKHR) vkGetDeviceProcAddr(
            dev, "vkGetPipelineExecutablePropertiesKHR");
    PFN_vkGetPipelineExecutableStatisticsKHR get_stats =
        (PFN_vkGetPipelineExecutableStatisticsKHR) vkGetDeviceProcAddr(
            dev, "vkGetPipelineExecutableStatisticsKHR");
    if (!get_props || !get_stats) return;

    VkPipelineInfoKHR pinfo{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    pinfo.pipeline = pipeline;
    uint32_t n_exec = 0;
    if (get_props(dev, &pinfo, &n_exec, nullptr) != VK_SUCCESS || n_exec == 0) return;
    std::vector<VkPipelineExecutablePropertiesKHR> props(n_exec,
        VkPipelineExecutablePropertiesKHR{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
    if (get_props(dev, &pinfo, &n_exec, props.data()) != VK_SUCCESS) return;

    for (uint32_t e = 0; e < n_exec; ++e) {
        fprintf(stderr, "tp5: pipeline stats for %s [%.*s]:\n", pipe_name,
                (int) sizeof(props[e].name), props[e].name);
        VkPipelineExecutableInfoKHR einfo{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
        einfo.pipeline = pipeline;
        einfo.executableIndex = e;
        uint32_t n_stats = 0;
        if (get_stats(dev, &einfo, &n_stats, nullptr) != VK_SUCCESS || n_stats == 0) continue;
        std::vector<VkPipelineExecutableStatisticKHR> stats(n_stats,
            VkPipelineExecutableStatisticKHR{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
        if (get_stats(dev, &einfo, &n_stats, stats.data()) != VK_SUCCESS) continue;
        for (const auto & st : stats) {
            fprintf(stderr, "tp5:   %s: ", st.name);
            switch (st.format) {
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                    fprintf(stderr, "%s", st.value.b32 ? "true" : "false"); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                    fprintf(stderr, "%lld", (long long) st.value.i64); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                    fprintf(stderr, "%llu", (unsigned long long) st.value.u64); break;
                case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
                    fprintf(stderr, "%g", st.value.f64); break;
                default: break;
            }
            fprintf(stderr, "\n");
        }
    }
}

static bool tp5_create_shader_module(VkDevice dev,
                                     const unsigned char * spv, uint64_t len,
                                     VkShaderModule * out) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = (size_t) len;
    ci.pCode = (const uint32_t *) (const void *) spv;
    return vkCreateShaderModule(dev, &ci, nullptr, out) == VK_SUCCESS;
}

bool tp5_alloc_host_visible_buffer(tp5_rank & r, VkDeviceSize size, VkBuffer & buf, VkDeviceMemory & mem, void ** mapped,
                                  VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  uint32_t * memory_type = nullptr) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(r.vkdev, &bci, nullptr, &buf) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(r.vkdev, buf, &req);
    VkPhysicalDeviceMemoryProperties props{};
    ggml_vk_tp5_mem_props(r.device, &props);
    uint32_t mt = find_memory_type(props, req.memoryTypeBits, required);
    if (mt == UINT32_MAX) {
        vkDestroyBuffer(r.vkdev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mt;
    if (vkAllocateMemory(r.vkdev, &ai, nullptr, &mem) != VK_SUCCESS) {
        vkDestroyBuffer(r.vkdev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        return false;
    }
    if (vkBindBufferMemory(r.vkdev, buf, mem, 0) != VK_SUCCESS) {
        vkDestroyBuffer(r.vkdev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        vkFreeMemory(r.vkdev, mem, nullptr);
        mem = VK_NULL_HANDLE;
        return false;
    }
    if (mapped) {
        void * ptr = nullptr;
        if (vkMapMemory(r.vkdev, mem, 0, size, 0, &ptr) != VK_SUCCESS) {
            vkDestroyBuffer(r.vkdev, buf, nullptr);
            buf = VK_NULL_HANDLE;
            vkFreeMemory(r.vkdev, mem, nullptr);
            mem = VK_NULL_HANDLE;
            return false;
        }
        *mapped = ptr;
    }
    if (memory_type) *memory_type = mt;
    return true;
}

bool tp5_build_rank_pipelines(tp5_comm & c, tp5_rank & r) {
    {
        VkDescriptorSetLayoutBinding b[4] = {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t) c.n_ranks, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,                    VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,                    VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,                    VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        };
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 4;
        ci.pBindings = b;
        if (vkCreateDescriptorSetLayout(r.vkdev, &ci, nullptr, &r.dsl) != VK_SUCCESS) return false;
    }
    {
        VkPushConstantRange        pc{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 };
        VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        ci.setLayoutCount = 1;
        ci.pSetLayouts = &r.dsl;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &pc;
        if (vkCreatePipelineLayout(r.vkdev, &ci, nullptr, &r.pipe_layout) != VK_SUCCESS) return false;
    }
    VkShaderModule mod_sum = VK_NULL_HANDLE, mod_pack = VK_NULL_HANDLE, mod_pack_vec = VK_NULL_HANDLE;
    const unsigned char * sum_spv = c.wire == tp5_wire_type::F16 ? tp5_sum_f16_data : tp5_sum_f32_data;
    const uint64_t sum_len = c.wire == tp5_wire_type::F16 ? tp5_sum_f16_len : tp5_sum_f32_len;
    if (!tp5_create_shader_module(r.vkdev, sum_spv, sum_len, &mod_sum)) return false;
    if (c.wire == tp5_wire_type::F16 &&
        !tp5_create_shader_module(r.vkdev, tp5_pack_f16_data, tp5_pack_f16_len, &mod_pack)) {
        vkDestroyShaderModule(r.vkdev, mod_sum, nullptr);
        return false;
    }
    if (c.wire == tp5_wire_type::F16 &&
        !tp5_create_shader_module(r.vkdev, tp5_pack_f16_vec_data, tp5_pack_f16_vec_len, &mod_pack_vec)) {
        vkDestroyShaderModule(r.vkdev, mod_sum, nullptr);
        if (mod_pack != VK_NULL_HANDLE) vkDestroyShaderModule(r.vkdev, mod_pack, nullptr);
        return false;
    }
    const uint32_t                 n_slots = (uint32_t) c.n_ranks;
    const VkSpecializationMapEntry slot_map{ 0, 0, sizeof(n_slots) };
    const VkSpecializationInfo     slot_spec{ 1, &slot_map, sizeof(n_slots), &n_slots };
    auto mk = [&](VkShaderModule m, VkPipeline * p) -> bool {
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = m;
        ci.stage.pName = "main";
        ci.stage.pSpecializationInfo = m == mod_sum ? &slot_spec : nullptr;
        ci.layout = r.pipe_layout;
        return vkCreateComputePipelines(r.vkdev, VK_NULL_HANDLE, 1, &ci, nullptr, p) == VK_SUCCESS;
    };
    bool ok = mk(mod_sum, &r.sum_pipe);
    if (ok && c.wire == tp5_wire_type::F16) ok = mk(mod_pack, &r.pack_pipe);
    if (ok && c.wire == tp5_wire_type::F16) ok = mk(mod_pack_vec, &r.pack_vec_pipe);
    vkDestroyShaderModule(r.vkdev, mod_sum, nullptr);
    if (mod_pack != VK_NULL_HANDLE) vkDestroyShaderModule(r.vkdev, mod_pack, nullptr);
    if (mod_pack_vec != VK_NULL_HANDLE) vkDestroyShaderModule(r.vkdev, mod_pack_vec, nullptr);
    if (!ok) return false;

    if (c.sync_mode == tp5_sync_mode::GPUFLAG) {
        VkDescriptorSetLayoutBinding fb[9] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo fci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        fci.bindingCount = 9;
        fci.pBindings = fb;
        if (vkCreateDescriptorSetLayout(r.vkdev, &fci, nullptr, &r.flag_dsl) != VK_SUCCESS) return false;

        VkPushConstantRange fpc{VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
        VkPipelineLayoutCreateInfo fpli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        fpli.setLayoutCount = 1;
        fpli.pSetLayouts = &r.flag_dsl;
        fpli.pushConstantRangeCount = 1;
        fpli.pPushConstantRanges = &fpc;
        if (vkCreatePipelineLayout(r.vkdev, &fpli, nullptr, &r.flag_pipe_layout) != VK_SUCCESS) return false;

        VkShaderModule mod_flag = VK_NULL_HANDLE;
        if (!tp5_create_shader_module(r.vkdev, tp5_gpuflag_data, tp5_gpuflag_len, &mod_flag)) return false;
        VkComputePipelineCreateInfo fci_pipe{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        fci_pipe.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fci_pipe.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        fci_pipe.stage.module = mod_flag;
        fci_pipe.stage.pName = "main";
        fci_pipe.layout = r.flag_pipe_layout;
        ok = vkCreateComputePipelines(r.vkdev, VK_NULL_HANDLE, 1, &fci_pipe, nullptr, &r.flag_pipe) == VK_SUCCESS;
        vkDestroyShaderModule(r.vkdev, mod_flag, nullptr);
        if (!ok) return false;

        if (!tp5_alloc_host_visible_buffer(r, 16, r.epoch_buf, r.epoch_mem, (void **) &r.epoch_host)) return false;
    }

    if (c.n_ranks == 5 && c.wire == tp5_wire_type::F16) {
        VkDescriptorSetLayoutBinding pb[6] = {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        };
        VkDescriptorSetLayoutCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        pci.bindingCount = 6;
        pci.pBindings = pb;
        if (vkCreateDescriptorSetLayout(r.vkdev, &pci, nullptr, &r.push_dsl) != VK_SUCCESS) return false;

        VkPushConstantRange ppc{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
        VkPipelineLayoutCreateInfo ppli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        ppli.setLayoutCount = 1;
        ppli.pSetLayouts = &r.push_dsl;
        ppli.pushConstantRangeCount = 1;
        ppli.pPushConstantRanges = &ppc;
        if (vkCreatePipelineLayout(r.vkdev, &ppli, nullptr, &r.push_layout) != VK_SUCCESS) return false;

        VkShaderModule mod_push = VK_NULL_HANDLE;
        if (!tp5_create_shader_module(r.vkdev, tp5_p2p_push_f16_data, tp5_p2p_push_f16_len, &mod_push)) return false;
        VkComputePipelineCreateInfo pci_pipe{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pci_pipe.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pci_pipe.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pci_pipe.stage.module = mod_push;
        pci_pipe.stage.pName = "main";
        pci_pipe.layout = r.push_layout;
        ok = vkCreateComputePipelines(r.vkdev, VK_NULL_HANDLE, 1, &pci_pipe, nullptr, &r.push_pipe) == VK_SUCCESS;
        vkDestroyShaderModule(r.vkdev, mod_push, nullptr);
        if (!ok) return false;
    }

    if (c.sync_mode == tp5_sync_mode::STAR) {
        struct {
            uint64_t src_bda_addr;
            uint64_t dst_bda_addr;
            uint64_t flag_bda_addr;
            uint32_t n_uvec4;
            uint32_t rank_idx;
            uint32_t seq_val;
            uint32_t mode;
        } bda_pc_dummy;
        VkPushConstantRange bda_pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bda_pc_dummy)};
        VkPipelineLayoutCreateInfo bda_pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        bda_pli.setLayoutCount = 0;
        bda_pli.pSetLayouts = nullptr;
        bda_pli.pushConstantRangeCount = 1;
        bda_pli.pPushConstantRanges = &bda_pcr;
        if (vkCreatePipelineLayout(r.vkdev, &bda_pli, nullptr, &r.bda_push_layout) != VK_SUCCESS) return false;

        VkShaderModule mod_bda = VK_NULL_HANDLE;
        if (!tp5_create_shader_module(r.vkdev, tp5_bda_push_f16_data, tp5_bda_push_f16_len, &mod_bda)) return false;
        VkComputePipelineCreateInfo bda_pipe_ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        bda_pipe_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        bda_pipe_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        bda_pipe_ci.stage.module = mod_bda;
        bda_pipe_ci.stage.pName = "main";
        bda_pipe_ci.layout = r.bda_push_layout;
        ok = vkCreateComputePipelines(r.vkdev, VK_NULL_HANDLE, 1, &bda_pipe_ci, nullptr, &r.bda_push_pipe) == VK_SUCCESS;
        vkDestroyShaderModule(r.vkdev, mod_bda, nullptr);
        if (!ok) return false;
    }

    if (c.sync_mode == tp5_sync_mode::RELAY) {
        VkDescriptorSetLayoutBinding rb[3] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo rdci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        rdci.bindingCount = 3;
        rdci.pBindings = rb;
        if (vkCreateDescriptorSetLayout(r.vkdev, &rdci, nullptr, &r.relay_copy_dsl) != VK_SUCCESS) return false;
        VkPushConstantRange rpc{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
        VkPipelineLayoutCreateInfo rlci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        rlci.setLayoutCount = 1;
        rlci.pSetLayouts = &r.relay_copy_dsl;
        rlci.pushConstantRangeCount = 1;
        rlci.pPushConstantRanges = &rpc;
        if (vkCreatePipelineLayout(r.vkdev, &rlci, nullptr, &r.relay_copy_layout) != VK_SUCCESS) return false;
        VkShaderModule relay_mod = VK_NULL_HANDLE;
        if (!tp5_create_shader_module(r.vkdev, tp5_relay_copy_f32_data, tp5_relay_copy_f32_len, &relay_mod)) return false;
        VkComputePipelineCreateInfo relay_ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        relay_ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        relay_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        relay_ci.stage.module = relay_mod;
        relay_ci.stage.pName = "main";
        relay_ci.layout = r.relay_copy_layout;
        ok = vkCreateComputePipelines(r.vkdev, VK_NULL_HANDLE, 1, &relay_ci, nullptr, &r.relay_copy_pipe) == VK_SUCCESS;
        vkDestroyShaderModule(r.vkdev, relay_mod, nullptr);
        if (!ok) return false;

        VkDescriptorSetLayoutBinding p1b[2] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo p1dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        p1dci.bindingCount = 2;
        p1dci.pBindings    = p1b;
        if (vkCreateDescriptorSetLayout(r.vkdev, &p1dci, nullptr, &r.relay_p1_copy_dsl) != VK_SUCCESS)
            return false;
        VkPushConstantRange p1pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
        VkPipelineLayoutCreateInfo p1pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        p1pli.setLayoutCount         = 1;
        p1pli.pSetLayouts            = &r.relay_p1_copy_dsl;
        p1pli.pushConstantRangeCount = 1;
        p1pli.pPushConstantRanges    = &p1pc;
        if (vkCreatePipelineLayout(r.vkdev, &p1pli, nullptr, &r.relay_p1_copy_layout) != VK_SUCCESS)
            return false;
        VkShaderModule p1copy_mod = VK_NULL_HANDLE;
        if (!tp5_create_shader_module(r.vkdev, tp5_copy_u128_data, tp5_copy_u128_len, &p1copy_mod))
            return false;
        VkComputePipelineCreateInfo p1ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        p1ci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        p1ci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        p1ci.stage.module = p1copy_mod;
        p1ci.stage.pName  = "main";
        p1ci.layout       = r.relay_p1_copy_layout;
        ok = vkCreateComputePipelines(r.vkdev, VK_NULL_HANDLE, 1, &p1ci, nullptr, &r.relay_p1_copy_pipe) == VK_SUCCESS;
        vkDestroyShaderModule(r.vkdev, p1copy_mod, nullptr);
        if (!ok) return false;

        if (tp5_latebind_hc_enabled()) {
            const auto make_late = [&](uint32_t bindings, uint32_t pc_bytes,
                                       const unsigned char * spv, uint64_t spv_len,
                                       VkDescriptorSetLayout & dsl, VkPipelineLayout & layout,
                                       VkPipeline & pipeline, const char * pipe_name,
                                       uint32_t required_subgroup = 0) -> bool {
                std::vector<VkDescriptorSetLayoutBinding> bs(bindings);
                for (uint32_t i = 0; i < bindings; ++i) {
                    bs[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
                }
                VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
                dci.bindingCount = bindings;
                dci.pBindings    = bs.data();
                if (vkCreateDescriptorSetLayout(r.vkdev, &dci, nullptr, &dsl) != VK_SUCCESS)
                    return false;
                VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_bytes};
                VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
                pli.setLayoutCount         = 1;
                pli.pSetLayouts            = &dsl;
                pli.pushConstantRangeCount = 1;
                pli.pPushConstantRanges    = &pcr;
                if (vkCreatePipelineLayout(r.vkdev, &pli, nullptr, &layout) != VK_SUCCESS)
                    return false;
                VkShaderModule mod = VK_NULL_HANDLE;
                if (!tp5_create_shader_module(r.vkdev, spv, spv_len, &mod))
                    return false;
                // Capture statistics when the extension is live so the same
                // RADV codegen evidence (VGPR/SGPR/LDS/spill) is available for
                // the TP5 collective kernels as for the main pipelines
                // (roadmap §4.3: judge codegen, not shader names).
                const bool want_stats = r.pipeline_stats && pipe_name != nullptr &&
                                        (!r.pipeline_stats_filter[0] ||
                                         strstr(pipe_name, r.pipeline_stats_filter));
                VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
                pci.flags = want_stats ? VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR : 0;
                pci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                pci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
                pci.stage.module = mod;
                pci.stage.pName  = "main";
                VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup_ci{
                    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT};
                if (required_subgroup != 0) {
                    subgroup_ci.requiredSubgroupSize = required_subgroup;
                    pci.stage.pNext = &subgroup_ci;
                }
                pci.layout       = layout;
                const bool made =
                    vkCreateComputePipelines(r.vkdev, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline) == VK_SUCCESS;
                vkDestroyShaderModule(r.vkdev, mod, nullptr);
                if (made && want_stats) tp5_print_pipeline_statistics(r.vkdev, pipeline, pipe_name);
                return made;
            };
            if (!make_late(4, 12, tp5_hc_late_inject_data, tp5_hc_late_inject_len,
                           r.late_inject_dsl, r.late_inject_layout, r.late_inject_pipe, "tp5_hc_late_inject") ||
                !make_late(7, 20, tp5_hc_late_q_data, tp5_hc_late_q_len,
                           r.late_q_dsl, r.late_q_layout, r.late_q_pipe, "tp5_hc_late_q") ||
                !make_late(3, 4, tp5_hc_publish_data, tp5_hc_publish_len,
                           r.late_publish_dsl, r.late_publish_layout, r.late_publish_pipe, "tp5_hc_publish") ||
                !make_late(9, 28, tp5_hc_resume_norm_data, tp5_hc_resume_norm_len,
                           r.late_norm_dsl, r.late_norm_layout, r.late_norm_pipe, "tp5_hc_resume_norm") ||
                !make_late(4, 32, tp5_hc_resume_lo_data, tp5_hc_resume_lo_len,
                           r.late_lo_dsl, r.late_lo_layout, r.late_lo_pipe, "tp5_hc_resume_lo")) {
                return false;
            }
            const bool q8_wave32 =
                !tp5_latebind_exact_q_requested() && r.caps.integer_dot_product &&
                r.caps.subgroup_size_control && r.caps.subgroup_min_size <= 32u &&
                r.caps.subgroup_max_size >= 32u;
            if (q8_wave32) {
                const bool q8_ok =
                    make_late(2, 4, tp5_hc_late_pack_data, tp5_hc_late_pack_len,
                              r.late_pack_dsl, r.late_pack_layout, r.late_pack_pipe, "tp5_hc_late_pack") &&
                    make_late(6, 16, tp5_hc_late_act_q8_data, tp5_hc_late_act_q8_len,
                              r.late_act_q8_dsl, r.late_act_q8_layout, r.late_act_q8_pipe, "tp5_hc_late_act_q8", 32u) &&
                    make_late(3, 28, tp5_hc_late_q_q8dot_data, tp5_hc_late_q_q8dot_len,
                              r.late_q8dot_dsl, r.late_q8dot_layout, r.late_q8dot_pipe, "tp5_hc_late_q_q8dot", 32u) &&
                    make_late(5, 32, tp5_hc_resume_lo_q8_data, tp5_hc_resume_lo_q8_len,
                              r.late_lo_q8_dsl, r.late_lo_q8_layout, r.late_lo_q8_pipe, "tp5_hc_resume_lo_q8", 32u) &&
                    make_late(5, 20, tp5_hc_late_up_q8dot_data, tp5_hc_late_up_q8dot_len,
                              r.late_up_q8dot_dsl, r.late_up_q8dot_layout, r.late_up_q8dot_pipe, "tp5_hc_late_up_q8dot", 32u);
                if (!q8_ok) {
                    if (r.late_pack_pipe) { vkDestroyPipeline(r.vkdev, r.late_pack_pipe, nullptr);
                                           vkDestroyPipelineLayout(r.vkdev, r.late_pack_layout, nullptr);
                                           vkDestroyDescriptorSetLayout(r.vkdev, r.late_pack_dsl, nullptr);
                                           r.late_pack_pipe = VK_NULL_HANDLE;
                                           r.late_pack_layout = VK_NULL_HANDLE;
                                           r.late_pack_dsl = VK_NULL_HANDLE; }
                    if (r.late_act_q8_pipe) vkDestroyPipeline(r.vkdev, r.late_act_q8_pipe, nullptr);
                    if (r.late_q8dot_pipe) vkDestroyPipeline(r.vkdev, r.late_q8dot_pipe, nullptr);
                    if (r.late_lo_q8_pipe) vkDestroyPipeline(r.vkdev, r.late_lo_q8_pipe, nullptr);
                    if (r.late_up_q8dot_pipe) vkDestroyPipeline(r.vkdev, r.late_up_q8dot_pipe, nullptr);
                    if (r.late_act_q8_layout) vkDestroyPipelineLayout(r.vkdev, r.late_act_q8_layout, nullptr);
                    if (r.late_q8dot_layout) vkDestroyPipelineLayout(r.vkdev, r.late_q8dot_layout, nullptr);
                    if (r.late_lo_q8_layout) vkDestroyPipelineLayout(r.vkdev, r.late_lo_q8_layout, nullptr);
                    if (r.late_up_q8dot_layout) vkDestroyPipelineLayout(r.vkdev, r.late_up_q8dot_layout, nullptr);
                    if (r.late_act_q8_dsl) vkDestroyDescriptorSetLayout(r.vkdev, r.late_act_q8_dsl, nullptr);
                    if (r.late_q8dot_dsl) vkDestroyDescriptorSetLayout(r.vkdev, r.late_q8dot_dsl, nullptr);
                    if (r.late_lo_q8_dsl) vkDestroyDescriptorSetLayout(r.vkdev, r.late_lo_q8_dsl, nullptr);
                    if (r.late_up_q8dot_dsl) vkDestroyDescriptorSetLayout(r.vkdev, r.late_up_q8dot_dsl, nullptr);
                    r.late_act_q8_pipe = r.late_q8dot_pipe = VK_NULL_HANDLE;
                    r.late_lo_q8_pipe = r.late_up_q8dot_pipe = VK_NULL_HANDLE;
                    r.late_act_q8_layout = r.late_q8dot_layout = VK_NULL_HANDLE;
                    r.late_lo_q8_layout = r.late_up_q8dot_layout = VK_NULL_HANDLE;
                    r.late_act_q8_dsl = r.late_q8dot_dsl = VK_NULL_HANDLE;
                    r.late_lo_q8_dsl = r.late_up_q8dot_dsl = VK_NULL_HANDLE;
                    fprintf(stderr,
                            "[tp5-latebind] aggressive Q8/wave32 pipeline unavailable; falling back to F32 Q\n");
                }
            }
        }
    }

    // Two sum sets plus one pack set per cached binding; reserve one temporary
    // plan as well. The SUM/pack layout contains one input descriptor per rank.
    const uint32_t             plan_capacity = tp5_comm::MAX_CACHED_PLANS + 1;
    const uint32_t             sets_per_plan = TP5_MAILBOX_BANKS * 2 + 1 +
                                               (c.sync_mode == tp5_sync_mode::RELAY && c.wire == tp5_wire_type::F16 ?
                                                    TP5_MAILBOX_BANKS : 0) +
                                               (c.sync_mode == tp5_sync_mode::RELAY ? TP5_MAILBOX_BANKS : 0) +
                                               (c.sync_mode == tp5_sync_mode::GPUFLAG ? 1 : 0) +
                                               (c.sync_mode == tp5_sync_mode::RELAY ? TP5_MAILBOX_BANKS : 0) +
                                               (c.sync_mode == tp5_sync_mode::RELAY && tp5_latebind_hc_enabled() ? 17 : 0);
    VkDescriptorPoolSize       ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                             plan_capacity * (std::max(uint32_t(c.n_ranks + 3), 12u) * sets_per_plan +
                                              (c.sync_mode == tp5_sync_mode::GPUFLAG ? 5 : 0)) };
    VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ci.maxSets       = plan_capacity * sets_per_plan;
    ci.poolSizeCount = 1;
    ci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(r.vkdev, &ci, nullptr, &r.desc_pool) != VK_SUCCESS) return false;

    if (c.sync_mode == tp5_sync_mode::STAR) {
        for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
            VkDescriptorSetAllocateInfo ds_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ds_ai.descriptorPool = r.desc_pool;
            ds_ai.descriptorSetCount = 1;

            ds_ai.pSetLayouts = &r.dsl;
            vkAllocateDescriptorSets(r.vkdev, &ds_ai, &r.star_ds_pack[b]);
            vkAllocateDescriptorSets(r.vkdev, &ds_ai, &r.star_ds_sum[b]);

            VkCommandBufferAllocateInfo cb_ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cb_ai.commandPool = r.cmd_pool;
            cb_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cb_ai.commandBufferCount = 1;
            vkAllocateCommandBuffers(r.vkdev, &cb_ai, &r.star_cmd_p1[b]);
            vkAllocateCommandBuffers(r.vkdev, &cb_ai, &r.star_cmd_p2[b]);
        }
        for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
            r.star_chain_p1[b].resize(128, VK_NULL_HANDLE);
            r.star_chain_p2[b].resize(128, VK_NULL_HANDLE);
            r.star_chain_ds_pack[b].resize(128, VK_NULL_HANDLE);
            r.star_chain_ds_sum[b].resize(128, VK_NULL_HANDLE);

            for (size_t s = 0; s < 128; ++s) {
                VkDescriptorSetAllocateInfo ds_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                ds_ai.descriptorPool = r.desc_pool;
                ds_ai.descriptorSetCount = 1;
                ds_ai.pSetLayouts = &r.dsl;
                vkAllocateDescriptorSets(r.vkdev, &ds_ai, &r.star_chain_ds_pack[b][s]);
                vkAllocateDescriptorSets(r.vkdev, &ds_ai, &r.star_chain_ds_sum[b][s]);
            }

            VkCommandBufferAllocateInfo chain_ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            chain_ai.commandPool = r.cmd_pool;
            chain_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            chain_ai.commandBufferCount = 128;
            vkAllocateCommandBuffers(r.vkdev, &chain_ai, r.star_chain_p1[b].data());
            vkAllocateCommandBuffers(r.vkdev, &chain_ai, r.star_chain_p2[b].data());
        }
    }

    return true;
}

static void tp5_update_star_sum_descriptor(tp5_rank & r, VkDescriptorSet ds, VkBuffer bcast_buf,
                                           VkBuffer out_tensor_buf, VkDeviceSize out_offset, VkDeviceSize out_size,
                                           VkDeviceSize payload_bytes, uint32_t n_slots) {
    VkDescriptorBufferInfo in_infos[8];
    for (uint32_t s = 0; s < n_slots; ++s) {
        in_infos[s] = VkDescriptorBufferInfo{bcast_buf, 0, payload_bytes};
    }
    VkDescriptorBufferInfo out_info{out_tensor_buf, out_offset, out_size};
    VkWriteDescriptorSet w[2] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, n_slots, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          nullptr, in_infos, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          nullptr, &out_info, nullptr },
    };
    vkUpdateDescriptorSets(r.vkdev, 2, w, 0, nullptr);
}

// Bind sender payloads directly, as slices of a contiguous mailbox or as
// separate sender/bank BOs. Both representations preserve rank addition order.
void tp5_update_sum_descriptor(tp5_rank &      r,
                               VkDescriptorSet ds,
                               VkBuffer        out_tensor_buf,
                               VkDeviceSize    out_offset,
                               VkDeviceSize    out_size,
                               uint32_t        n_slots,
                               size_t          bank,
                               VkDeviceSize    stride,
                               VkDeviceSize    payload_bytes,
                               VkDeviceSize    flags_byte_offset,
                               VkDeviceSize    flags_bytes,
                               bool            bind_epoch) {
    VkDescriptorBufferInfo in_infos[8];
    for (uint32_t s = 0; s < n_slots; ++s) {
        in_infos[s] = r.inboxes.empty() ?
                          VkDescriptorBufferInfo{ r.mailbox_buf, (bank * n_slots + s) * stride, payload_bytes } :
                          VkDescriptorBufferInfo{ r.inboxes[tp5_plan_slot(s, bank)].buf, 0, payload_bytes };
    }
    VkDescriptorBufferInfo out_info{out_tensor_buf, out_offset, out_size};
    VkDescriptorBufferInfo flag_info{r.mailbox_buf, flags_byte_offset, flags_bytes};
    VkWriteDescriptorSet   w[4] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, n_slots, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         nullptr,                                                                                                         in_infos, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1, 0, 1,       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr,
         &out_info,                                                                                                                 nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 2, 0, 1,       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr,
         &flag_info,                                                                                                                nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 3, 0, 1,       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr,
         nullptr,                                                                                                                   nullptr },
    };
    VkDescriptorBufferInfo epoch_info{r.epoch_buf, 0, 16};
    if (bind_epoch && r.epoch_buf != VK_NULL_HANDLE) {
        w[3].pBufferInfo = &epoch_info;
    } else {
        w[3].descriptorCount = 0;
    }
    vkUpdateDescriptorSets(r.vkdev, bind_epoch ? 4u : 3u, w, 0, nullptr);
}

void tp5_update_flag_descriptor(tp5_rank & r, size_t my_rank, VkDescriptorSet ds, VkDeviceSize mailbox_bytes) {
    VkDescriptorBufferInfo own{r.mailbox_buf, 0, mailbox_bytes};
    VkDescriptorBufferInfo peer_infos[8];
    VkWriteDescriptorSet w[9] = {};
    for (uint32_t i = 0; i < 9; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = ds;
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &own;
    }
    w[0].pBufferInfo = &own;
    for (const auto & im : r.imports) {
        const size_t peer = (size_t) im.peer;
        if (peer >= 8) continue;
        peer_infos[peer].buffer = im.buf;
        peer_infos[peer].offset = 0;
        peer_infos[peer].range = mailbox_bytes;
        uint32_t binding = (peer < my_rank) ? (uint32_t) (peer + 1) : (uint32_t) peer;
        if (binding > 0 && binding < 8) {
            w[binding].pBufferInfo = &peer_infos[peer];
        }
    }
    VkDescriptorBufferInfo epoch_info{r.epoch_buf, 0, 16};
    w[8].pBufferInfo = &epoch_info;
    vkUpdateDescriptorSets(r.vkdev, 9, w, 0, nullptr);
}

struct tp5_flag_pc {
    uint32_t mode;
    uint32_t flags_base_u32;
    uint32_t ready_off_u32;
};

static void tp5_record_flag_dispatch(VkCommandBuffer     cmd,
                                     tp5_rank &          r,
                                     VkDescriptorSet     ds_flag,
                                     const tp5_flag_pc & pc) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.flag_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.flag_pipe_layout, 0, 1, &ds_flag, 0, nullptr);
    vkCmdPushConstants(cmd, r.flag_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, 1, 1, 1);
}

void tp5_update_pack_descriptor(tp5_rank &      r,
                                VkDescriptorSet ds,
                                VkBuffer        in_tensor_buf,
                                VkDeviceSize    in_offset,
                                VkDeviceSize    in_size,
                                uint32_t        n_slots,
                                VkBuffer        out_buf = VK_NULL_HANDLE,
                                VkDeviceSize    out_offset = 0,
                                VkDeviceSize    out_size = VK_WHOLE_SIZE) {
    if (ds == VK_NULL_HANDLE) return;
    VkDescriptorBufferInfo in_infos[8];
    std::fill_n(in_infos, n_slots, VkDescriptorBufferInfo{ in_tensor_buf, in_offset, in_size });
    VkDescriptorBufferInfo out_info{out_buf != VK_NULL_HANDLE ? out_buf : r.wire_buf, out_offset, out_size};
    VkWriteDescriptorSet   w[2] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, n_slots, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         nullptr,                                                                                                         in_infos, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1, 0, 1,       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr,
         &out_info,                                                                                                                 nullptr },
    };
    vkUpdateDescriptorSets(r.vkdev, 2, w, 0, nullptr);
}

static bool tp5_alloc_host_import_buffer(tp5_rank & r, void * host_ptr, VkDeviceSize size,
                                         bool with_device_address, VkBuffer & buf, VkDeviceMemory & mem,
                                         uint64_t & bda_addr) {
    bda_addr = 0;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (with_device_address) {
        bci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    bci.pNext = &ext;

    if (vkCreateBuffer(r.vkdev, &bci, nullptr, &buf) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(r.vkdev, buf, &req);

    auto vkGetMemoryHostPointerPropertiesEXT = (PFN_vkGetMemoryHostPointerPropertiesEXT)
        vkGetDeviceProcAddr(r.vkdev, "vkGetMemoryHostPointerPropertiesEXT");
    if (!vkGetMemoryHostPointerPropertiesEXT) {
        vkDestroyBuffer(r.vkdev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryHostPointerPropertiesEXT host_props{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (vkGetMemoryHostPointerPropertiesEXT(r.vkdev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                            host_ptr, &host_props) != VK_SUCCESS) {
        vkDestroyBuffer(r.vkdev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        return false;
    }

    VkPhysicalDeviceMemoryProperties props{};
    ggml_vk_tp5_mem_props(r.device, &props);

    uint32_t comp_bits = req.memoryTypeBits & host_props.memoryTypeBits;
    uint32_t mt = find_memory_type(props, comp_bits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) mt = find_memory_type(props, comp_bits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (mt == UINT32_MAX) mt = find_memory_type(props, comp_bits, 0);
    if (mt == UINT32_MAX) {
        vkDestroyBuffer(r.vkdev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        return false;
    }

    VkMemoryAllocateFlagsInfo flags_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    if (with_device_address) {
        flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    }

    VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    imp.pNext = with_device_address ? &flags_info : nullptr;
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    imp.pHostPointer = host_ptr;

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &imp;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mt;

    if (vkAllocateMemory(r.vkdev, &ai, nullptr, &mem) != VK_SUCCESS) {
        vkDestroyBuffer(r.vkdev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        return false;
    }

    if (vkBindBufferMemory(r.vkdev, buf, mem, 0) != VK_SUCCESS) {
        vkDestroyBuffer(r.vkdev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        vkFreeMemory(r.vkdev, mem, nullptr);
        mem = VK_NULL_HANDLE;
        return false;
    }

    if (with_device_address) {
        auto vkGetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)
            vkGetDeviceProcAddr(r.vkdev, "vkGetBufferDeviceAddressKHR");
        if (!vkGetBufferDeviceAddressKHR) {
            vkGetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)
                vkGetDeviceProcAddr(r.vkdev, "vkGetBufferDeviceAddress");
        }
        if (vkGetBufferDeviceAddressKHR) {
            VkBufferDeviceAddressInfo dai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            dai.buffer = buf;
            bda_addr = vkGetBufferDeviceAddressKHR(r.vkdev, &dai);
        }
        if (bda_addr == 0) {
            fprintf(stderr, "tp5_alloc_host_import_buffer: WARN bda_addr is 0 for buf=%p\n", (void*)buf);
        }
    }
    return true;
}

bool tp5_alloc_device_buffer(tp5_rank & r, VkDeviceSize size,
                             VkBufferUsageFlags usage, bool exportable,
                             VkBuffer & buf, VkDeviceMemory & mem,
                             int * export_fd) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    if (exportable) bci.pNext = &ext;

    if (vkCreateBuffer(r.vkdev, &bci, nullptr, &buf) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(r.vkdev, buf, &req);

    VkPhysicalDeviceMemoryProperties props{};
    ggml_vk_tp5_mem_props(r.device, &props);

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;

    VkExportMemoryAllocateInfo exp_ai{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exp_ai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    if (exportable) ai.pNext = &exp_ai;

    VkMemoryAllocateFlagsInfo flags_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        flags_info.pNext = ai.pNext;
        ai.pNext = &flags_info;
    }

    uint32_t mt = find_memory_type(props, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mt == UINT32_MAX) {
        vkDestroyBuffer(r.vkdev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        return false;
    }
    ai.memoryTypeIndex = mt;

    if (vkAllocateMemory(r.vkdev, &ai, nullptr, &mem) != VK_SUCCESS) {
        vkDestroyBuffer(r.vkdev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        return false;
    }
    if (vkBindBufferMemory(r.vkdev, buf, mem, 0) != VK_SUCCESS) {
        vkDestroyBuffer(r.vkdev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        vkFreeMemory(r.vkdev, mem, nullptr);
        mem = VK_NULL_HANDLE;
        return false;
    }

    if (exportable && export_fd) {
        auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR) vkGetDeviceProcAddr(r.vkdev, "vkGetMemoryFdKHR");
        if (!vkGetMemoryFdKHR) {
            vkDestroyBuffer(r.vkdev, buf, nullptr);
            buf = VK_NULL_HANDLE;
            vkFreeMemory(r.vkdev, mem, nullptr);
            mem = VK_NULL_HANDLE;
            return false;
        }
        VkMemoryGetFdInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
        gi.memory = mem;
        gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        if (vkGetMemoryFdKHR(r.vkdev, &gi, export_fd) != VK_SUCCESS || *export_fd < 0) {
            vkDestroyBuffer(r.vkdev, buf, nullptr);
            buf = VK_NULL_HANDLE;
            vkFreeMemory(r.vkdev, mem, nullptr);
            mem = VK_NULL_HANDLE;
            return false;
        }
    }
    return true;
}

// Import peer DMA-BUF with FULL storage and transfer usage (matches POC).
// import_alloc_size must equal the exporter's vkAllocateMemory size exactly.
bool tp5_import_peer_buffer(tp5_rank & r, int peer_fd, VkDeviceSize buf_size,
                            VkDeviceSize import_alloc_size,
                            VkBuffer & buf, VkDeviceMemory & mem) {
    tp5_fd fd_guard(peer_fd);
    auto vkGetMemoryFdPropertiesKHR = (PFN_vkGetMemoryFdPropertiesKHR)
        vkGetDeviceProcAddr(r.vkdev, "vkGetMemoryFdPropertiesKHR");
    if (!vkGetMemoryFdPropertiesKHR) return false;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = buf_size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    bci.pNext = &ext;
    if (vkCreateBuffer(r.vkdev, &bci, nullptr, &buf) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(r.vkdev, buf, &req);

    VkMemoryFdPropertiesKHR fd_props{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    if (vkGetMemoryFdPropertiesKHR(r.vkdev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd_guard.fd, &fd_props) != VK_SUCCESS) {
        vkDestroyBuffer(r.vkdev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        return false;
    }

    const uint32_t compatible = req.memoryTypeBits & fd_props.memoryTypeBits;
    if (compatible == 0) {
        vkDestroyBuffer(r.vkdev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        return false;
    }
    VkPhysicalDeviceMemoryProperties props{};
    ggml_vk_tp5_mem_props(r.device, &props);
    uint32_t mt = find_memory_type(props, compatible, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mt == UINT32_MAX) mt = find_memory_type(props, compatible, 0);
    if (mt == UINT32_MAX) {
        vkDestroyBuffer(r.vkdev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        return false;
    }

    VkImportMemoryFdInfoKHR imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    imp.fd = fd_guard.fd;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = import_alloc_size;
    ai.memoryTypeIndex = mt;
    ai.pNext = &imp;
    if (vkAllocateMemory(r.vkdev, &ai, nullptr, &mem) != VK_SUCCESS) {
        vkDestroyBuffer(r.vkdev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        return false;
    }
    // Vulkan has now taken ownership of fd_guard.fd
    fd_guard.release();

    if (vkBindBufferMemory(r.vkdev, buf, mem, 0) != VK_SUCCESS) {
        vkDestroyBuffer(r.vkdev, buf, nullptr);
        buf = VK_NULL_HANDLE;
        vkFreeMemory(r.vkdev, mem, nullptr);
        mem = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

static void tp5_destroy_inboxes(tp5_rank & r) {
    for (auto & box : r.inboxes) {
        if (box.fd >= 0)
            ::close(box.fd);
        if (box.buf)
            vkDestroyBuffer(r.vkdev, box.buf, nullptr);
        if (box.mem)
            vkFreeMemory(r.vkdev, box.mem, nullptr);
    }
    r.inboxes.clear();
}

void tp5_destroy_rank(tp5_rank & r) {
    if (r.vkdev == VK_NULL_HANDLE) return;

    if (r.timing_pool) {
        vkDestroyQueryPool(r.vkdev, r.timing_pool, nullptr);
        r.timing_pool = VK_NULL_HANDLE;
    }
    if (r.fence_p1) { vkDestroyFence(r.vkdev, r.fence_p1, nullptr); r.fence_p1 = VK_NULL_HANDLE; }
    if (r.fence_p2) { vkDestroyFence(r.vkdev, r.fence_p2, nullptr); r.fence_p2 = VK_NULL_HANDLE; }
    for (size_t f = 0; f < 4; ++f) {
        if (r.fence_ring[f]) { vkDestroyFence(r.vkdev, r.fence_ring[f], nullptr); r.fence_ring[f] = VK_NULL_HANDLE; }
    }
    if (r.sem_p1_done) { vkDestroySemaphore(r.vkdev, r.sem_p1_done, nullptr); r.sem_p1_done = VK_NULL_HANDLE; }
    for (auto s : r.wait_sems) if (s) vkDestroySemaphore(r.vkdev, s, nullptr);
    r.wait_sems.clear();

    for (size_t j = 0; j < r.peer_timeline_sems.size(); ++j) {
        if (r.peer_timeline_sems[j] != VK_NULL_HANDLE && r.peer_timeline_sems[j] != r.timeline_sem) {
            vkDestroySemaphore(r.vkdev, r.peer_timeline_sems[j], nullptr);
        }
    }
    r.peer_timeline_sems.clear();
    if (r.timeline_export_fd >= 0) { ::close(r.timeline_export_fd); r.timeline_export_fd = -1; }
    if (r.timeline_sem != VK_NULL_HANDLE) {
        vkDestroySemaphore(r.vkdev, r.timeline_sem, nullptr);
        r.timeline_sem = VK_NULL_HANDLE;
    }
    if (r.dri_fd >= 0) {
        if (r.own_syncobj) {
            drmSyncobjDestroy(r.dri_fd, r.own_syncobj);
            r.own_syncobj = 0;
        }
        for (uint32_t handle : r.peer_syncobjs) {
            if (handle) drmSyncobjDestroy(r.dri_fd, handle);
        }
        r.peer_syncobjs.clear();
        ::close(r.dri_fd);
        r.dri_fd = -1;
    }
    for (auto & im : r.imports) {
        if (im.buf) { vkDestroyBuffer(r.vkdev, im.buf, nullptr); im.buf = VK_NULL_HANDLE; }
        if (im.mem) { vkFreeMemory(r.vkdev, im.mem, nullptr); im.mem = VK_NULL_HANDLE; }
    }
    r.imports.clear();

    tp5_destroy_inboxes(r);
    if (r.wire_buf) { vkDestroyBuffer(r.vkdev, r.wire_buf, nullptr); r.wire_buf = VK_NULL_HANDLE; }
    if (r.wire_mem) { vkFreeMemory(r.vkdev, r.wire_mem, nullptr); r.wire_mem = VK_NULL_HANDLE; }

    if (r.mailbox_export_fd >= 0) { ::close(r.mailbox_export_fd); r.mailbox_export_fd = -1; }
    if (r.mailbox_buf) { vkDestroyBuffer(r.vkdev, r.mailbox_buf, nullptr); r.mailbox_buf = VK_NULL_HANDLE; }
    if (r.mailbox_mem) { vkFreeMemory(r.vkdev, r.mailbox_mem, nullptr); r.mailbox_mem = VK_NULL_HANDLE; }

    if (r.epoch_host) { vkUnmapMemory(r.vkdev, r.epoch_mem); r.epoch_host = nullptr; }
    if (r.epoch_buf) { vkDestroyBuffer(r.vkdev, r.epoch_buf, nullptr); r.epoch_buf = VK_NULL_HANDLE; }
    if (r.epoch_mem) { vkFreeMemory(r.vkdev, r.epoch_mem, nullptr); r.epoch_mem = VK_NULL_HANDLE; }

    if (r.sum_pipe) { vkDestroyPipeline(r.vkdev, r.sum_pipe, nullptr); r.sum_pipe = VK_NULL_HANDLE; }
    if (r.pack_pipe) { vkDestroyPipeline(r.vkdev, r.pack_pipe, nullptr); r.pack_pipe = VK_NULL_HANDLE; }
    if (r.pack_vec_pipe) { vkDestroyPipeline(r.vkdev, r.pack_vec_pipe, nullptr); r.pack_vec_pipe = VK_NULL_HANDLE; }
    if (r.flag_pipe) { vkDestroyPipeline(r.vkdev, r.flag_pipe, nullptr); r.flag_pipe = VK_NULL_HANDLE; }
    if (r.push_pipe) { vkDestroyPipeline(r.vkdev, r.push_pipe, nullptr); r.push_pipe = VK_NULL_HANDLE; }
    if (r.relay_copy_pipe) { vkDestroyPipeline(r.vkdev, r.relay_copy_pipe, nullptr); r.relay_copy_pipe = VK_NULL_HANDLE; }
    if (r.relay_p1_copy_pipe) { vkDestroyPipeline(r.vkdev, r.relay_p1_copy_pipe, nullptr); r.relay_p1_copy_pipe = VK_NULL_HANDLE; }
    if (r.late_inject_pipe) { vkDestroyPipeline(r.vkdev, r.late_inject_pipe, nullptr); r.late_inject_pipe = VK_NULL_HANDLE; }
    if (r.late_q_pipe) { vkDestroyPipeline(r.vkdev, r.late_q_pipe, nullptr); r.late_q_pipe = VK_NULL_HANDLE; }
    if (r.late_act_q8_pipe) { vkDestroyPipeline(r.vkdev, r.late_act_q8_pipe, nullptr); r.late_act_q8_pipe = VK_NULL_HANDLE; }
    if (r.late_pack_pipe) { vkDestroyPipeline(r.vkdev, r.late_pack_pipe, nullptr); r.late_pack_pipe = VK_NULL_HANDLE; }
    if (r.late_pack_layout) { vkDestroyPipelineLayout(r.vkdev, r.late_pack_layout, nullptr); r.late_pack_layout = VK_NULL_HANDLE; }
    if (r.late_pack_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.late_pack_dsl, nullptr); r.late_pack_dsl = VK_NULL_HANDLE; }
    if (r.late_q8dot_pipe) { vkDestroyPipeline(r.vkdev, r.late_q8dot_pipe, nullptr); r.late_q8dot_pipe = VK_NULL_HANDLE; }
    if (r.late_publish_pipe) { vkDestroyPipeline(r.vkdev, r.late_publish_pipe, nullptr); r.late_publish_pipe = VK_NULL_HANDLE; }
    if (r.late_norm_pipe) { vkDestroyPipeline(r.vkdev, r.late_norm_pipe, nullptr); r.late_norm_pipe = VK_NULL_HANDLE; }
    if (r.late_lo_pipe) { vkDestroyPipeline(r.vkdev, r.late_lo_pipe, nullptr); r.late_lo_pipe = VK_NULL_HANDLE; }
    if (r.late_lo_q8_pipe) { vkDestroyPipeline(r.vkdev, r.late_lo_q8_pipe, nullptr); r.late_lo_q8_pipe = VK_NULL_HANDLE; }
    if (r.late_up_q8dot_pipe) { vkDestroyPipeline(r.vkdev, r.late_up_q8dot_pipe, nullptr); r.late_up_q8dot_pipe = VK_NULL_HANDLE; }
    if (r.pipe_layout) { vkDestroyPipelineLayout(r.vkdev, r.pipe_layout, nullptr); r.pipe_layout = VK_NULL_HANDLE; }
    if (r.flag_pipe_layout) { vkDestroyPipelineLayout(r.vkdev, r.flag_pipe_layout, nullptr); r.flag_pipe_layout = VK_NULL_HANDLE; }
    if (r.push_layout) { vkDestroyPipelineLayout(r.vkdev, r.push_layout, nullptr); r.push_layout = VK_NULL_HANDLE; }
    if (r.relay_copy_layout) { vkDestroyPipelineLayout(r.vkdev, r.relay_copy_layout, nullptr); r.relay_copy_layout = VK_NULL_HANDLE; }
    if (r.relay_p1_copy_layout) { vkDestroyPipelineLayout(r.vkdev, r.relay_p1_copy_layout, nullptr); r.relay_p1_copy_layout = VK_NULL_HANDLE; }
    if (r.late_inject_layout) { vkDestroyPipelineLayout(r.vkdev, r.late_inject_layout, nullptr); r.late_inject_layout = VK_NULL_HANDLE; }
    if (r.late_q_layout) { vkDestroyPipelineLayout(r.vkdev, r.late_q_layout, nullptr); r.late_q_layout = VK_NULL_HANDLE; }
    if (r.late_act_q8_layout) { vkDestroyPipelineLayout(r.vkdev, r.late_act_q8_layout, nullptr); r.late_act_q8_layout = VK_NULL_HANDLE; }
    if (r.late_q8dot_layout) { vkDestroyPipelineLayout(r.vkdev, r.late_q8dot_layout, nullptr); r.late_q8dot_layout = VK_NULL_HANDLE; }
    if (r.late_publish_layout) { vkDestroyPipelineLayout(r.vkdev, r.late_publish_layout, nullptr); r.late_publish_layout = VK_NULL_HANDLE; }
    if (r.late_norm_layout) { vkDestroyPipelineLayout(r.vkdev, r.late_norm_layout, nullptr); r.late_norm_layout = VK_NULL_HANDLE; }
    if (r.late_lo_layout) { vkDestroyPipelineLayout(r.vkdev, r.late_lo_layout, nullptr); r.late_lo_layout = VK_NULL_HANDLE; }
    if (r.late_lo_q8_layout) { vkDestroyPipelineLayout(r.vkdev, r.late_lo_q8_layout, nullptr); r.late_lo_q8_layout = VK_NULL_HANDLE; }
    if (r.late_up_q8dot_layout) { vkDestroyPipelineLayout(r.vkdev, r.late_up_q8dot_layout, nullptr); r.late_up_q8dot_layout = VK_NULL_HANDLE; }
    if (r.dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.dsl, nullptr); r.dsl = VK_NULL_HANDLE; }
    if (r.flag_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.flag_dsl, nullptr); r.flag_dsl = VK_NULL_HANDLE; }
    if (r.push_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.push_dsl, nullptr); r.push_dsl = VK_NULL_HANDLE; }
    if (r.relay_copy_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.relay_copy_dsl, nullptr); r.relay_copy_dsl = VK_NULL_HANDLE; }
    if (r.relay_p1_copy_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.relay_p1_copy_dsl, nullptr); r.relay_p1_copy_dsl = VK_NULL_HANDLE; }
    if (r.late_inject_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.late_inject_dsl, nullptr); r.late_inject_dsl = VK_NULL_HANDLE; }
    if (r.late_q_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.late_q_dsl, nullptr); r.late_q_dsl = VK_NULL_HANDLE; }
    if (r.late_act_q8_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.late_act_q8_dsl, nullptr); r.late_act_q8_dsl = VK_NULL_HANDLE; }
    if (r.late_q8dot_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.late_q8dot_dsl, nullptr); r.late_q8dot_dsl = VK_NULL_HANDLE; }
    if (r.late_publish_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.late_publish_dsl, nullptr); r.late_publish_dsl = VK_NULL_HANDLE; }
    if (r.late_norm_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.late_norm_dsl, nullptr); r.late_norm_dsl = VK_NULL_HANDLE; }
    if (r.late_lo_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.late_lo_dsl, nullptr); r.late_lo_dsl = VK_NULL_HANDLE; }
    if (r.late_lo_q8_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.late_lo_q8_dsl, nullptr); r.late_lo_q8_dsl = VK_NULL_HANDLE; }
    if (r.late_up_q8dot_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.late_up_q8dot_dsl, nullptr); r.late_up_q8dot_dsl = VK_NULL_HANDLE; }
    if (r.desc_pool) { vkDestroyDescriptorPool(r.vkdev, r.desc_pool, nullptr); r.desc_pool = VK_NULL_HANDLE; }
    if (r.cmd_pool) { vkDestroyCommandPool(r.vkdev, r.cmd_pool, nullptr); r.cmd_pool = VK_NULL_HANDLE; }

    // Clean up Star AllReduce BDA & Broadcast resources
    for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
        if (r.bcast_host[b]) {
            vkUnmapMemory(r.vkdev, r.bcast_mem[b]);
            r.bcast_host[b] = nullptr;
        }
        if (r.bcast_buf[b]) {
            vkDestroyBuffer(r.vkdev, r.bcast_buf[b], nullptr);
            r.bcast_buf[b] = VK_NULL_HANDLE;
        }
        if (r.bcast_mem[b]) {
            vkFreeMemory(r.vkdev, r.bcast_mem[b], nullptr);
            r.bcast_mem[b] = VK_NULL_HANDLE;
        }
        if (r.host_import_buf[b]) {
            vkDestroyBuffer(r.vkdev, r.host_import_buf[b], nullptr);
            r.host_import_buf[b] = VK_NULL_HANDLE;
        }
        if (r.host_import_mem[b]) {
            vkFreeMemory(r.vkdev, r.host_import_mem[b], nullptr);
            r.host_import_mem[b] = VK_NULL_HANDLE;
        }
        r.bda_addr[b] = 0;
    }
    for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
        if (r.host_ready_event[b] != VK_NULL_HANDLE) {
            vkDestroyEvent(r.vkdev, r.host_ready_event[b], nullptr);
            r.host_ready_event[b] = VK_NULL_HANDLE;
        }
    }
    if (r.bda_push_pipe) { vkDestroyPipeline(r.vkdev, r.bda_push_pipe, nullptr); r.bda_push_pipe = VK_NULL_HANDLE; }
    if (r.bda_push_layout) { vkDestroyPipelineLayout(r.vkdev, r.bda_push_layout, nullptr); r.bda_push_layout = VK_NULL_HANDLE; }
    if (r.bda_push_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.bda_push_dsl, nullptr); r.bda_push_dsl = VK_NULL_HANDLE; }
    if (r.relay_copy_pipe) { vkDestroyPipeline(r.vkdev, r.relay_copy_pipe, nullptr); r.relay_copy_pipe = VK_NULL_HANDLE; }
    if (r.relay_copy_layout) { vkDestroyPipelineLayout(r.vkdev, r.relay_copy_layout, nullptr); r.relay_copy_layout = VK_NULL_HANDLE; }
    if (r.relay_copy_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.relay_copy_dsl, nullptr); r.relay_copy_dsl = VK_NULL_HANDLE; }

}

bool tp5_setup_workspace(tp5_comm & c, size_t max_elems) {
    // Every rank slot and both bank descriptors must satisfy all devices'
    // storage-buffer alignment, including odd-sized F16 payloads.
    const size_t wire_b    = c.wire == tp5_wire_type::F16 ? 2 : 4;
    size_t       alignment = 4;
    for (const auto & r : c.ranks) {
        alignment = std::max(alignment, (size_t) r.caps.min_storage_buffer_offset_alignment);
    }
    if (max_elems > (UINT32_MAX - alignment) / wire_b) {
        c.fail("workspace exceeds shader addressing range");
        return false;
    }
    max_elems = ((max_elems * wire_b + alignment - 1) / alignment) * alignment / wire_b;
    const bool   late_enabled = c.sync_mode == tp5_sync_mode::RELAY &&
                              c.wire == tp5_wire_type::F32 && tp5_latebind_hc_enabled();
    bool late_q8_workspace = late_enabled && !tp5_latebind_exact_q_requested();
    for (const auto & r : c.ranks) {
        late_q8_workspace =
            late_q8_workspace && r.late_act_q8_pipe != VK_NULL_HANDLE && r.late_q8dot_pipe != VK_NULL_HANDLE &&
            r.late_lo_q8_pipe != VK_NULL_HANDLE && r.late_up_q8dot_pipe != VK_NULL_HANDLE;
    }
    const size_t late_align   = std::max<size_t>(64, alignment);
    const size_t late_offset  =
        ((max_elems * sizeof(float) + late_align - 1) / late_align) * late_align;
    const size_t late_control = late_q8_workspace ? TP5_LATE_Q_CONTROL_BYTES : 0;
    const size_t late_bytes   = late_enabled ? TP5_LATE_MAX_FLOATS *
        (late_q8_workspace ? sizeof(ggml_fp16_t) : sizeof(float)) : 0;
    c.late_host_offset        = late_offset;
    if (late_enabled && late_q8_workspace) {
        GGML_ASSERT(tp5_late_q_control_offset(late_offset) % 64 == 0);
        GGML_ASSERT(tp5_late_q_bcast_payload_offset(late_offset, true) % 64 == 0);
        GGML_ASSERT(tp5_late_q_control_offset(late_offset) + TP5_LATE_Q_CONTROL_BYTES <=
                    tp5_late_q_bcast_payload_offset(late_offset, true));
    }
    for (const auto & r : c.ranks) {
        const uint64_t relay_header = c.sync_mode == tp5_sync_mode::RELAY ? 64u : 0u;
        const uint64_t required =
            late_enabled ? relay_header + late_offset + late_control + late_bytes :
                           (uint64_t) max_elems * sizeof(float) + relay_header;
        if (required > r.caps.max_storage_buffer_range) {
            c.fail("tensor or mailbox slot exceeds maxStorageBufferRange");
            return false;
        }
    }
    // Clear all cached plans and drain in-flight GPU execution before modifying workspace buffers
    if (c.sync_mode == tp5_sync_mode::TIMELINE || c.sync_mode == tp5_sync_mode::DRM ||
        c.sync_mode == tp5_sync_mode::RELAY) {
        if (!tp5_drain_epoch(c, c.allreduce_calls)) {
            c.fail("setup_workspace: drain failed before workspace reallocation");
            return false;
        }
    } else if (c.sync_mode == tp5_sync_mode::GPUFLAG) {
        tp5_gpuflag_drain_all(c);
    } else if (c.sync_mode == tp5_sync_mode::STAR) {
        for (auto & r : c.ranks) {
            if (r.vkdev) vkDeviceWaitIdle(r.vkdev);
        }
    }
    c.clear_cached_plans();
    if (c.failed) return false;
    if (c.sync_mode == tp5_sync_mode::RELAY) {
        for (auto backend : c.backends) {
            ggml_vk_tp5_clear_relay_payload_binding(backend);
        }
    }
    c.workspace_gen++;
    for (bool & used : c.relay_bank_used) used = false;

    if (c.sync_mode == tp5_sync_mode::STAR || c.sync_mode == tp5_sync_mode::RELAY) {
        const size_t page_size = 4096;
        const size_t rank_payload_bytes =
            late_enabled ? late_offset + late_control + late_bytes : max_elems * sizeof(float);
        const size_t rank_stride = ((rank_payload_bytes + 64 + page_size - 1) / page_size) * page_size;
        const size_t stage_stride = (c.n_ranks + 1) * rank_stride;
        // Stages reuse two banks; no command addresses a stage-indexed slot.
        // Importing 128 unused copies makes amdgpu walk that entire USERPTR
        // range at every submit, especially after prefill grows the workspace.
        const size_t total_host_size = stage_stride;
        const size_t total_bcast_size = rank_stride;
        c.star_rank_stride = rank_stride;
        c.star_host_alloc_size = total_host_size;

        for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
            // Imported memory must be released before its backing host pages.
            for (auto & r : c.ranks) {
                if (r.host_import_buf[b]) vkDestroyBuffer(r.vkdev, r.host_import_buf[b], nullptr);
                if (r.host_import_mem[b]) vkFreeMemory(r.vkdev, r.host_import_mem[b], nullptr);
                r.host_import_buf[b] = VK_NULL_HANDLE;
                r.host_import_mem[b] = VK_NULL_HANDLE;
            }
            if (c.star_host_raw[b]) {
                free(c.star_host_raw[b]);
                c.star_host_raw[b] = nullptr;
                c.star_host_aligned[b] = nullptr;
            }
            void * ptr = nullptr;
            const size_t total_alloc_size = total_host_size + 4096; // Data slots + 64-bit atomic flag page
            if (posix_memalign(&ptr, page_size, total_alloc_size) != 0 || !ptr) {
                c.fail("posix_memalign failed for Star Host RAM");
                return false;
            }
            memset(ptr, 0, total_alloc_size);
            c.star_host_raw[b] = ptr;
            c.star_host_aligned[b] = ptr;
            c.star_host_flag[b] = (volatile uint32_t *)((char *)ptr + total_host_size);

            for (size_t i = 0; i < c.n_ranks; ++i) {
                tp5_rank & r = c.ranks[i];
                if (r.host_import_buf[b]) {
                    vkDestroyBuffer(r.vkdev, r.host_import_buf[b], nullptr);
                    r.host_import_buf[b] = VK_NULL_HANDLE;
                }
                if (r.host_import_mem[b]) {
                    vkFreeMemory(r.vkdev, r.host_import_mem[b], nullptr);
                    r.host_import_mem[b] = VK_NULL_HANDLE;
                }
                if (!tp5_alloc_host_import_buffer(r, (char *) ptr + i * rank_stride, rank_stride,
                                                  c.sync_mode == tp5_sync_mode::STAR,
                                                  r.host_import_buf[b], r.host_import_mem[b], r.bda_addr[b])) {
                    c.fail("alloc_host_import_buffer failed on rank " + std::to_string(i));
                    return false;
                }

                if (r.bcast_host[b]) {
                    vkUnmapMemory(r.vkdev, r.bcast_mem[b]);
                    r.bcast_host[b] = nullptr;
                }
                if (r.bcast_buf[b]) {
                    vkDestroyBuffer(r.vkdev, r.bcast_buf[b], nullptr);
                    r.bcast_buf[b] = VK_NULL_HANDLE;
                }
                if (r.bcast_mem[b]) {
                    vkFreeMemory(r.vkdev, r.bcast_mem[b], nullptr);
                    r.bcast_mem[b] = VK_NULL_HANDLE;
                }
                bool bcast_ok = false;
                if (c.sync_mode == tp5_sync_mode::RELAY) {
                    static const bool force_uncached = [] {
                        const char * env = getenv("GGML_TP5_RELAY_FORCE_UNCACHED");
                        return env && atoi(env) != 0;
                    }();
                    // Prefer cached device-coherent local VRAM. The AMD
                    // coherence bit already provides automatic host<->device
                    // visibility when paired with HOST_COHERENT; forcing
                    // DEVICE_UNCACHED defeats the tiny doorbell working set's
                    // chance to stay in the GPU cache hierarchy.
                    const VkMemoryPropertyFlags relay_cached =
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD;
                    const VkMemoryPropertyFlags relay_uncached =
                        relay_cached | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
                    bool used_uncached = force_uncached;
                    if (!force_uncached) {
                        bcast_ok = tp5_alloc_host_visible_buffer(r, total_bcast_size, r.bcast_buf[b], r.bcast_mem[b],
                                                                 &r.bcast_host[b], relay_cached);
                    }
                    if (!bcast_ok) {
                        used_uncached = true;
                        bcast_ok = tp5_alloc_host_visible_buffer(r, total_bcast_size, r.bcast_buf[b], r.bcast_mem[b],
                                                                 &r.bcast_host[b], relay_uncached);
                    }
                    if (bcast_ok && getenv("GGML_TP5_PROFILE")) {
                        fprintf(stderr, "[tp5-relay-config] rank=%zu bank=%zu bcast_cache=%s\n",
                                i, b, used_uncached ? "uncached" : "cached");
                    }
                } else {
                    bcast_ok = tp5_alloc_host_visible_buffer(
                        r, total_bcast_size, r.bcast_buf[b], r.bcast_mem[b], &r.bcast_host[b],
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                }
                if (!bcast_ok) {
                    c.fail("alloc_host_visible_buffer failed for bcast_buf on rank " + std::to_string(i));
                    return false;
                }
            }
        }

        for (auto & r : c.ranks) {
            if (r.wire_buf) { vkDestroyBuffer(r.vkdev, r.wire_buf, nullptr); r.wire_buf = VK_NULL_HANDLE; }
            if (r.wire_mem) { vkFreeMemory(r.vkdev, r.wire_mem, nullptr); r.wire_mem = VK_NULL_HANDLE; }
            r.wire_bda = 0;
            const size_t total_wire_size = rank_stride;
            VkBufferUsageFlags wire_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            if (c.sync_mode == tp5_sync_mode::STAR) {
                wire_usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            }
            if (!tp5_alloc_device_buffer(r, total_wire_size, wire_usage,
                    false, r.wire_buf, r.wire_mem, nullptr)) {
                c.fail("wire buffer allocation failed");
                return false;
            }
            if (c.sync_mode == tp5_sync_mode::STAR) {
                auto vkGetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)
                    vkGetDeviceProcAddr(r.vkdev, "vkGetBufferDeviceAddressKHR");
                if (!vkGetBufferDeviceAddressKHR) {
                    vkGetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)
                        vkGetDeviceProcAddr(r.vkdev, "vkGetBufferDeviceAddress");
                }
                if (vkGetBufferDeviceAddressKHR) {
                    VkBufferDeviceAddressInfo dai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
                    dai.buffer = r.wire_buf;
                    r.wire_bda = vkGetBufferDeviceAddressKHR(r.vkdev, &dai);
                }
            }
        }
        // STAR never reads a peer VRAM mailbox. With BDA enabled, RADV's
        // global BO list would nevertheless include those unused imports and
        // introduce inter-rank reservation dependencies on every submission.
        if (c.sync_mode == tp5_sync_mode::RELAY && c.allreduce_calls != 0) {
            // setup_workspace native-drained every prior real submission
            // before replacing host-import memory. Reconstruct the two
            // completed bank generations so the next prequeued chain can
            // retain its no-host-wait handoff across a workspace resize.
            for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
                const uint64_t completed = tp5_mailbox_bank(c.allreduce_calls) == b ?
                    c.allreduce_calls : c.allreduce_calls - 1;
                for (size_t i = 0; i < c.n_ranks; ++i) {
                    auto * status = (volatile uint32_t *) ((char *) c.star_host_aligned[b] +
                                                           i * c.star_rank_stride + c.star_rank_stride - 64);
                    status[1] = (uint32_t) completed;
                    status[3] = (uint32_t) completed;
                }
            }
            std::atomic_thread_fence(std::memory_order_release);
#if defined(__x86_64__) || defined(_M_X64)
            _mm_sfence();
#endif
        }
        c.max_elems = max_elems;
        return true;
    }

    const VkDeviceSize wire_bytes = (VkDeviceSize) max_elems * wire_b;
    const VkDeviceSize stride = wire_bytes;
    const VkDeviceSize flags_bytes = tp5_flags_region_bytes(c.n_ranks, c.sync_mode);
    const VkDeviceSize mailbox_bytes =
        (c.isolate_mailbox ? 0 : (VkDeviceSize) TP5_MAILBOX_BANKS * c.n_ranks * stride) + flags_bytes;

    for (auto & r : c.ranks) {
        for (auto & im : r.imports) {
            if (im.buf) { vkDestroyBuffer(r.vkdev, im.buf, nullptr); im.buf = VK_NULL_HANDLE; }
            if (im.mem) { vkFreeMemory(r.vkdev, im.mem, nullptr); im.mem = VK_NULL_HANDLE; }
        }
        r.imports.clear();
        tp5_destroy_inboxes(r);
        if (r.wire_buf) { vkDestroyBuffer(r.vkdev, r.wire_buf, nullptr); r.wire_buf = VK_NULL_HANDLE; }
        if (r.wire_mem) { vkFreeMemory(r.vkdev, r.wire_mem, nullptr); r.wire_mem = VK_NULL_HANDLE; }
        if (r.mailbox_export_fd >= 0) { ::close(r.mailbox_export_fd); r.mailbox_export_fd = -1; }
        if (r.mailbox_buf) { vkDestroyBuffer(r.vkdev, r.mailbox_buf, nullptr); r.mailbox_buf = VK_NULL_HANDLE; }
        if (r.mailbox_mem) { vkFreeMemory(r.vkdev, r.mailbox_mem, nullptr); r.mailbox_mem = VK_NULL_HANDLE; }

        // Mailbox: exported DMA-BUF with full TRANSFER + STORAGE usage (matches POC)
        if (!tp5_alloc_device_buffer(r, mailbox_bytes,
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     !c.isolate_mailbox, r.mailbox_buf, r.mailbox_mem, &r.mailbox_export_fd)) {
            c.fail("mailbox allocation failed");
            return false;
        }
        {
            VkMemoryRequirements mreq{};
            vkGetBufferMemoryRequirements(r.vkdev, r.mailbox_buf, &mreq);
            r.mailbox_alloc_bytes = mreq.size;
        }
        if (c.isolate_mailbox) {
            r.inboxes.resize(c.n_ranks * TP5_MAILBOX_BANKS);
            for (auto & box : r.inboxes) {
                if (!tp5_alloc_device_buffer(r, stride,
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                             true, box.buf, box.mem, &box.fd)) {
                    c.fail("isolated inbox allocation failed");
                    return false;
                }
                VkMemoryRequirements req{};
                vkGetBufferMemoryRequirements(r.vkdev, box.buf, &req);
                box.alloc_bytes = req.size;
            }
        }
        if (!tp5_alloc_device_buffer(r, wire_bytes,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                false, r.wire_buf, r.wire_mem, nullptr)) {
            c.fail("wire buffer allocation failed");
            return false;
        }
    }

    for (size_t i = 0; i < c.n_ranks; ++i) {
        for (size_t j = 0; j < c.n_ranks; ++j) {
            if (i == j) continue;
            for (size_t b = 0; b < (c.isolate_mailbox ? TP5_MAILBOX_BANKS : 1); ++b) {
                const auto * box = c.isolate_mailbox ? &c.ranks[j].inboxes[tp5_plan_slot(i, b)] : nullptr;
                tp5_fd       fd(::dup(box ? box->fd : c.ranks[j].mailbox_export_fd));
                if (fd.fd < 0) {
                    c.fail("dup of peer mailbox fd failed");
                    return false;
                }
                tp5_rank::imported im;
                im.peer = (int) j;
                im.bank = b;
                if (!tp5_import_peer_buffer(c.ranks[i], fd.release(), box ? stride : mailbox_bytes,
                                            box ? box->alloc_bytes : c.ranks[j].mailbox_alloc_bytes, im.buf, im.mem)) {
                    c.fail("import of peer mailbox failed (rank " + std::to_string(i) + " <- rank " +
                           std::to_string(j) + ")");
                    return false;
                }
                c.ranks[i].imports.push_back(im);
            }
        }
    }
    for (auto & r : c.ranks) {
        for (auto & box : r.inboxes) {
            if (box.fd >= 0) {
                ::close(box.fd);
                box.fd = -1;
            }
        }
    }

    c.max_elems = max_elems;
    if (getenv("GGML_TP5_DEBUG_MAILBOX")) {
        fprintf(stderr, "tp5 mailbox: max_elems=%zu bytes=%llu alloc:",
                max_elems, (unsigned long long) mailbox_bytes);
        for (size_t i = 0; i < c.n_ranks; ++i) {
            fprintf(stderr, " r%zu=%llu", i, (unsigned long long) c.ranks[i].mailbox_alloc_bytes);
        }
        fprintf(stderr, "\n");
    }
    return true;
}

tensor_dev_ref tp5_tensor_dev_ref(ggml_tensor * t) {
    tensor_dev_ref r{};
    r.ok = ggml_vk_tp5_tensor_dev_ref(t, &r.buf, &r.offset, &r.size, &r.owner);
    return r;
}

struct tp5_sum_pc {
    uint32_t n_elems;
    uint32_t n_slots;
    uint32_t use_flags;
    uint32_t ready_off_u32;
};

static bool tp5_hc_consumer_ref(tp5_comm &       c,
                                size_t           rank,
                                void *           first_cb,
                                tensor_dev_ref & ref,
                                size_t           n_elems,
                                tp5_hc_key &     key) {
    const bool relay = c.sync_mode == tp5_sync_mode::RELAY;
    static const bool relay_fused_hc = [] {
        const char * env = getenv("GGML_TP5_RELAY_FUSED_HC");
        return env && atoi(env) != 0;
    }();
    if (c.n_ranks != 5 || c.wire != tp5_wire_type::F16 ||
        (c.sync_mode != tp5_sync_mode::TIMELINE && !(relay && relay_fused_hc)))
        return false;
    vk_tp5_hc_sum hc;
    if (!ggml_vk_tp5_hc_consumer(c.backends[rank], first_cb, &hc) || hc.width != n_elems ||
        hc.block.buffer != ref.buf || hc.block.offset != ref.offset || hc.block.size != ref.size)
        return false;
    auto & r = c.ranks[rank];
    const bool has_quantized = (hc.quantized.buffer != nullptr);
    const uint32_t required_descriptors = has_quantized ? 13 : 12;
    if (r.caps.max_storage_buffer_descriptors < required_descriptors)
        return false;
    for (const auto & binding : hc.bindings) {
        if (!binding.buffer || !binding.owner || !binding.size || binding.size > r.caps.max_storage_buffer_range ||
            binding.offset % std::max(uint64_t(4), r.caps.min_storage_buffer_offset_alignment) != 0)
            return false;
        // Stream zero commits the sum while every stream reads HC inputs.
        // Unlike separate kernels, a residual alias of the sum is unsafe.
        if (binding.buffer == ref.buf && (binding.offset <= ref.offset ? ref.offset - binding.offset < binding.size :
                                                                         binding.offset - ref.offset < ref.size))
            return false;
    }
    if (has_quantized) {
        const auto & qb = hc.quantized;
        if (!qb.owner || !qb.size || qb.size > r.caps.max_storage_buffer_range ||
            qb.offset % std::max(uint64_t(4), r.caps.min_storage_buffer_offset_alignment) != 0)
            return false;
        if (qb.buffer == ref.buf && (qb.offset <= ref.offset ? ref.offset - qb.offset < qb.size :
                                                               qb.offset - ref.offset < ref.size))
            return false;
    }
    const size_t pipe_idx = has_quantized ? 1 : 0;
    if (!r.hc_sum_pipe[pipe_idx] &&
        !ggml_vk_tp5_hc_sum_pipeline(r.device, has_quantized, relay, &r.hc_sum_pipe[pipe_idx],
                                     &r.hc_sum_layout[pipe_idx], &r.hc_sum_dsl[pipe_idx]))
        return false;
    ref.hc    = std::move(hc);
    key.width = ref.hc.width;
    key.streams = ref.hc.streams;
    key.late_rank = 0;
    std::memcpy(&key.epsilon_bits, &ref.hc.epsilon, sizeof(key.epsilon_bits));
    for (size_t i = 0; i < key.bindings.size(); ++i) {
        const auto & binding = ref.hc.bindings[i];
        key.bindings[i]      = { binding.buffer, binding.offset, binding.size };
    }
    if (has_quantized) {
        key.quantized = { ref.hc.quantized.buffer, ref.hc.quantized.offset, ref.hc.quantized.size };
    } else {
        key.quantized = {};
    }
    return true;
}

static bool tp5_late_consumer_ref(tp5_comm &       c,
                                  size_t           rank,
                                  void *           first_cb,
                                  tensor_dev_ref & ref,
                                  size_t           n_elems,
                                  tp5_hc_key &     key) {
    if (!tp5_latebind_hc_enabled() || c.sync_mode != tp5_sync_mode::RELAY ||
        c.wire != tp5_wire_type::F32 || c.n_ranks != 5)
        return false;
    vk_tp5_hc_sum hc;
    if (!ggml_vk_tp5_hc_consumer(c.backends[rank], first_cb, &hc) ||
        hc.width == 0 || (hc.width % 256u) != 0u || hc.width > 4096 ||
        hc.streams != 4 || n_elems == 0 || (n_elems % hc.width) != 0 ||
        hc.late_rank == 0 || hc.late_rank % 16 != 0 || hc.quantized.buffer != nullptr ||
        size_t(hc.streams) * hc.late_rank > 2048 ||
        hc.block.buffer != ref.buf || hc.block.offset != ref.offset || hc.block.size != ref.size)
        return false;
    const size_t capacity_rows = n_elems / hc.width;
    if (capacity_rows == 0 || capacity_rows > VK_TP5_DIRECT_COLUMN_TILE ||
        size_t(capacity_rows) * hc.streams * hc.late_rank > TP5_LATE_MAX_FLOATS)
        return false;
    auto & r = c.ranks[rank];
    if (r.caps.max_storage_buffer_descriptors < 9) return false;
    const auto valid = [&](const vk_tp5_hc_binding & binding) {
        return binding.buffer && binding.owner && binding.size &&
               binding.size <= r.caps.max_storage_buffer_range &&
               binding.offset % std::max(uint64_t(16), r.caps.min_storage_buffer_offset_alignment) == 0;
    };
    for (const auto & binding : hc.bindings) {
        if (!valid(binding) ||
            (binding.buffer == ref.buf &&
             (binding.offset <= ref.offset ? ref.offset - binding.offset < binding.size :
                                             binding.offset - ref.offset < ref.size)))
            return false;
    }
    if (ref.offset % 16 != 0) return false;
    if (!valid(hc.down_weight) || !valid(hc.lo) || !valid(hc.up_weight) || !valid(hc.mixed))
        return false;
    const uint64_t down_expected =
        uint64_t(hc.late_rank) * ggml_row_size(GGML_TYPE_Q8_0, uint64_t(hc.streams) * hc.width);
    const uint64_t up_expected =
        uint64_t(hc.streams) * hc.width * ggml_row_size(GGML_TYPE_Q8_0, hc.late_rank);
    if (hc.lo.size < uint64_t(capacity_rows) * hc.late_rank * sizeof(float) ||
        hc.mixed.size < uint64_t(capacity_rows) * hc.width * sizeof(float) ||
        hc.bindings[0].size < uint64_t(capacity_rows) * hc.streams * hc.width * sizeof(float) ||
        hc.bindings[1].size < uint64_t(capacity_rows) * hc.width * sizeof(float) ||
        hc.bindings[2].size < uint64_t(capacity_rows) * hc.streams * hc.width * sizeof(float) ||
        hc.bindings[3].size < uint64_t(capacity_rows) * hc.streams * hc.width * sizeof(float) ||
        hc.down_weight.size < down_expected || hc.up_weight.size < up_expected)
        return false;
    const auto program = ggml_vk_tp5_graph_program(c.backends[rank], first_cb);
    if (!program || !program->commands.valid || program->hc_down_end <= program->hc_norm_end ||
        program->hc_down_end > program->commands.code.size())
        return false;

    ref.late       = std::move(hc);
    ref.late.capacity_rows = uint32_t(capacity_rows);
    key.width      = ref.late.width;
    key.streams    = ref.late.streams;
    key.late_rank  = ref.late.late_rank;
    key.capacity_rows = uint32_t(capacity_rows);
    std::memcpy(&key.epsilon_bits, &ref.late.epsilon, sizeof(key.epsilon_bits));
    for (size_t i = 0; i < key.bindings.size(); ++i) {
        const auto & binding = ref.late.bindings[i];
        key.bindings[i] = { binding.buffer, binding.offset, binding.size };
    }
    key.down_weight = { ref.late.down_weight.buffer, ref.late.down_weight.offset, ref.late.down_weight.size };
    key.lo          = { ref.late.lo.buffer, ref.late.lo.offset, ref.late.lo.size };
    key.up_weight   = { ref.late.up_weight.buffer, ref.late.up_weight.offset, ref.late.up_weight.size };
    key.mixed       = { ref.late.mixed.buffer, ref.late.mixed.offset, ref.late.mixed.size };
    key.quantized   = {};
    return true;
}

static void tp5_update_hc_descriptor(tp5_rank &             rank,
                                     VkDescriptorSet        set,
                                     const tensor_dev_ref & output,
                                     size_t                 bank,
                                     VkDeviceSize           stride,
                                     VkDeviceSize           payload,
                                     bool                   relay = false) {
    VkDescriptorBufferInfo infos[13]{};
    constexpr uint32_t     hc_slots[] = { 0, 6, 7, 8, 9, 10 };
    for (size_t i = 0; i < output.hc.bindings.size(); ++i) {
        const auto & binding = output.hc.bindings[i];
        infos[hc_slots[i]]   = { binding.buffer, binding.offset, binding.size };
    }
    for (size_t i = 0; i < 5; ++i) {
        if (relay) {
            // Keep the 12/13-binding HC layout identical to TIMELINE. Only
            // binding 1 is the inbox and binding 2 is the host-imported status
            // range; duplicate the inbox into the remaining inactive slots.
            infos[i + 1] = { rank.bcast_buf[bank], 0, 64 + output.size };
        } else {
            infos[i + 1] = rank.inboxes.empty() ?
                               VkDescriptorBufferInfo{ rank.mailbox_buf, (bank * 5 + i) * stride, payload } :
                               VkDescriptorBufferInfo{ rank.inboxes[tp5_plan_slot(i, bank)].buf, 0, payload };
        }
    }
    if (relay) {
        infos[2] = { rank.host_import_buf[bank], stride - 64, 64 };
    }
    infos[11] = { output.buf, output.offset, output.size };
    const bool has_quantized = (output.hc.quantized.buffer != nullptr);
    const uint32_t count = has_quantized ? 13 : 12;
    if (has_quantized) {
        infos[12] = { output.hc.quantized.buffer, output.hc.quantized.offset, output.hc.quantized.size };
    }
    VkWriteDescriptorSet writes[13]{};
    for (uint32_t i = 0; i < count; ++i) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = set;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &infos[i];
    }
    vkUpdateDescriptorSets(rank.vkdev, count, writes, 0, nullptr);
}

static void tp5_update_storage_set(VkDevice dev, VkDescriptorSet set,
                                   const VkDescriptorBufferInfo * infos, uint32_t count) {
    VkWriteDescriptorSet writes[12]{};
    GGML_ASSERT(count <= 12);
    for (uint32_t i = 0; i < count; ++i) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = set;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &infos[i];
    }
    vkUpdateDescriptorSets(dev, count, writes, 0, nullptr);
}

static void tp5_update_push_descriptor(tp5_comm & c, tp5_rank & r, size_t rank_idx, VkDescriptorSet ds,
                                       VkBuffer src_buf, VkDeviceSize src_off, VkDeviceSize payload, size_t bank,
                                       VkDeviceSize slot_off) {
    VkDescriptorBufferInfo infos[6]{};
    infos[0] = { src_buf, src_off, payload };
    // Binding 1: Local mailbox
    infos[1] = { c.isolate_mailbox ? r.inboxes[tp5_plan_slot(rank_idx, bank)].buf : r.mailbox_buf,
                 c.isolate_mailbox ? 0 : slot_off,
                 payload };
    // Binding 2..5: 4 peer imports
    size_t peer_idx = 0;
    for (size_t p = 0; p < r.imports.size(); ++p) {
        if (c.isolate_mailbox && r.imports[p].bank != bank) continue;
        if (peer_idx < 4) {
            infos[2 + peer_idx] = { r.imports[p].buf, c.isolate_mailbox ? 0 : slot_off, payload };
            ++peer_idx;
        }
    }
    VkWriteDescriptorSet writes[6]{};
    for (uint32_t i = 0; i < 2 + (uint32_t) peer_idx; ++i) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = ds;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &infos[i];
    }
    vkUpdateDescriptorSets(r.vkdev, 2 + (uint32_t) peer_idx, writes, 0, nullptr);
}

bool tp5_record_plan(tp5_comm & c, tp5_cached_plan & plan, const std::vector<tensor_dev_ref> & trefs,
                     size_t n_elems, VkDeviceSize flags_base) {
    const bool define_inline = c.sync_mode == tp5_sync_mode::RELAY &&
                                c.wire == tp5_wire_type::F32 && tp5_latebind_hc_enabled();
    plan.owners.clear();
    for (size_t i = 0; i < c.n_ranks; ++i) {
        if (trefs[i].owner)
            plan.owners.push_back(trefs[i].owner);
        if (trefs[i].packed_owner)
            plan.owners.push_back(trefs[i].packed_owner);
        if (trefs[i].hc.width) {
            for (const auto & binding : trefs[i].hc.bindings)
                plan.owners.push_back(binding.owner);
            if (trefs[i].hc.quantized.buffer && trefs[i].hc.quantized.owner)
                plan.owners.push_back(trefs[i].hc.quantized.owner);
        }
        if (trefs[i].late.late_rank) {
            for (const auto & binding : trefs[i].late.bindings)
                plan.owners.push_back(binding.owner);
            plan.owners.push_back(trefs[i].late.down_weight.owner);
            plan.owners.push_back(trefs[i].late.lo.owner);
            plan.owners.push_back(trefs[i].late.up_weight.owner);
            plan.owners.push_back(trefs[i].late.mixed.owner);
        }
    }
    const size_t wire_b = c.wire == tp5_wire_type::F16 ? 2 : 4;
    // Copies use whole words; an odd F16 element count leaves a padding half
    // in the aligned wire allocation which the sum shader never reads.
    const VkDeviceSize payload        = ((VkDeviceSize) n_elems * wire_b + 3) & ~VkDeviceSize(3);
    const VkDeviceSize stride = (VkDeviceSize) c.max_elems * wire_b;
    const VkDeviceSize tensor_bytes = (VkDeviceSize) n_elems * sizeof(float);
    const VkDeviceSize payload_bytes = (VkDeviceSize) c.n_ranks * stride;
    const VkDeviceSize flags_bytes = tp5_flags_region_bytes(c.n_ranks, c.sync_mode);
    const VkDeviceSize mailbox_bytes  = flags_base + flags_bytes;
    const uint32_t flags_base_u32 = (uint32_t) (flags_base / sizeof(uint32_t));
    const bool gpuflag = (c.sync_mode == tp5_sync_mode::GPUFLAG);
    const tp5_flag_pc flag_pc_base{0u, flags_base_u32, 0u};

    bool relay_direct_all = c.sync_mode == tp5_sync_mode::RELAY && !trefs.empty();
    for (size_t i = 0; relay_direct_all && i < c.n_ranks; ++i)
        relay_direct_all = trefs[i].relay_direct;

    std::shared_ptr<tp5_p1_resources> p1_res;
    if (!relay_direct_all) {
    // P1 effective key & candidate reuse
    tp5_p1_key p1_key;
    p1_key.n_elems       = n_elems;
    p1_key.wire          = c.wire;
    p1_key.stride        = stride;
    p1_key.workspace_gen = c.workspace_gen;
    p1_key.effective_bindings.resize(c.n_ranks);
    for (size_t i = 0; i < c.n_ranks; ++i) {
        if (trefs[i].packed_buf != VK_NULL_HANDLE) {
            p1_key.effective_bindings[i] = { trefs[i].packed_buf, trefs[i].packed_offset, trefs[i].packed_size, true };
        } else {
            p1_key.effective_bindings[i] = { trefs[i].buf, trefs[i].offset, trefs[i].size, false };
        }
    }

    bool         share_p1_allowed = true;
    const char * share_p1_env     = getenv("GGML_TP5_SHARE_P1");
    if (share_p1_env && atoi(share_p1_env) == 0) {
        share_p1_allowed = false;
    }

    if (share_p1_allowed) {
        for (const auto & existing_plan : c.cached_plans) {
            if (existing_plan.p1 && existing_plan.p1->key == p1_key) {
                p1_res = existing_plan.p1;
                break;
            }
        }
    }

    if (!p1_res) {
        auto new_p1 = std::make_shared<tp5_p1_resources>();
        new_p1->key = p1_key;
        new_p1->cmd_p1.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
        if (define_inline) {
            new_p1->definitions.resize(c.n_ranks * TP5_MAILBOX_BANKS);
        }
        if (c.wire == tp5_wire_type::F16) {
            if (c.sync_mode == tp5_sync_mode::RELAY) {
                new_p1->ds_relay_pack.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
            } else {
                new_p1->ds_pack.resize(c.n_ranks, VK_NULL_HANDLE);
            }
        }
        if (c.sync_mode == tp5_sync_mode::RELAY) {
            new_p1->ds_relay_copy.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
        }
        if (gpuflag) {
            new_p1->ds_flag.resize(c.n_ranks, VK_NULL_HANDLE);
        }
        new_p1->devices.resize(c.n_ranks, VK_NULL_HANDLE);
        new_p1->cmd_pools.resize(c.n_ranks, VK_NULL_HANDLE);
        new_p1->desc_pools.resize(c.n_ranks, VK_NULL_HANDLE);
        for (size_t i = 0; i < c.n_ranks; ++i) {
            new_p1->devices[i]    = c.ranks[i].vkdev;
            new_p1->cmd_pools[i]  = c.ranks[i].cmd_pool;
            new_p1->desc_pools[i] = c.ranks[i].desc_pool;
            if (trefs[i].packed_buf != VK_NULL_HANDLE) {
                if (trefs[i].packed_owner) {
                    new_p1->owners.push_back(trefs[i].packed_owner);
                }
            } else {
                if (trefs[i].owner) {
                    new_p1->owners.push_back(trefs[i].owner);
                }
            }
        }

        for (size_t i = 0; i < c.n_ranks; ++i) {
            tp5_rank &                  r = c.ranks[i];
            VkCommandBufferAllocateInfo cba{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, r.cmd_pool,
                                             VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
            VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, r.desc_pool, 1,
                                            &r.dsl };
            for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
                const size_t idx = tp5_plan_slot(i, b);
                if (vkAllocateCommandBuffers(r.vkdev, &cba, &new_p1->cmd_p1[idx]) != VK_SUCCESS) {
                    c.fail("allocation of P1 command buffer failed on rank " + std::to_string(i));
                    return false;
                }
            }
            if (c.wire == tp5_wire_type::F16) {
                if (trefs[i].packed_buf == VK_NULL_HANDLE) {
                    if (c.sync_mode == tp5_sync_mode::RELAY) {
                        for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
                            const size_t idx = tp5_plan_slot(i, b);
                            if (vkAllocateDescriptorSets(r.vkdev, &ai, &new_p1->ds_relay_pack[idx]) != VK_SUCCESS) {
                                c.fail("allocation of RELAY pack descriptor set failed on rank " + std::to_string(i));
                                return false;
                            }
                        }
                    } else if (vkAllocateDescriptorSets(r.vkdev, &ai, &new_p1->ds_pack[i]) != VK_SUCCESS) {
                        c.fail("allocation of pack descriptor set failed on rank " + std::to_string(i));
                        return false;
                    }
                }
                if (c.sync_mode != tp5_sync_mode::RELAY && r.push_pipe != VK_NULL_HANDLE && c.n_ranks == 5) {
                    new_p1->ds_push.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
                    VkDescriptorSetAllocateInfo pai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, r.desc_pool, 1,
                                                    &r.push_dsl };
                    for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
                        const size_t idx = tp5_plan_slot(i, b);
                        if (vkAllocateDescriptorSets(r.vkdev, &pai, &new_p1->ds_push[idx]) != VK_SUCCESS) {
                            c.fail("allocation of push descriptor set failed on rank " + std::to_string(i));
                            return false;
                        }
                    }
                }
            }
            if (c.sync_mode == tp5_sync_mode::RELAY) {
                VkDescriptorSetAllocateInfo cai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, r.desc_pool,
                                                 1, &r.relay_p1_copy_dsl};
                for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
                    const size_t idx = tp5_plan_slot(i, b);
                    if (vkAllocateDescriptorSets(r.vkdev, &cai, &new_p1->ds_relay_copy[idx]) != VK_SUCCESS) {
                        c.fail("allocation of RELAY P1 vector-copy descriptor failed on rank " + std::to_string(i));
                        return false;
                    }
                }
            }
            if (gpuflag) {
                VkDescriptorSetAllocateInfo fai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, r.desc_pool,
                                                 1, &r.flag_dsl };
                if (vkAllocateDescriptorSets(r.vkdev, &fai, &new_p1->ds_flag[i]) != VK_SUCCESS) {
                    c.fail("allocation of flag descriptor set failed on rank " + std::to_string(i));
                    return false;
                }
                tp5_update_flag_descriptor(r, i, new_p1->ds_flag[i], mailbox_bytes);
            }
        }

        VkCommandBufferUsageFlags cb_flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        VkCommandBufferBeginInfo  beg{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, cb_flags, nullptr };
        static const bool allow_parallel_push = [] {
            const char * env = getenv("GGML_TP5_PARALLEL_PUSH");
            return env && (atoi(env) != 0);
        }();
        for (size_t i = 0; i < c.n_ranks; ++i) {
            tp5_rank & r          = c.ranks[i];
            const bool has_packed = (trefs[i].packed_buf != VK_NULL_HANDLE);
            if (c.sync_mode != tp5_sync_mode::RELAY && c.wire == tp5_wire_type::F16 && !has_packed) {
                tp5_update_pack_descriptor(r, new_p1->ds_pack[i], trefs[i].buf, trefs[i].offset, tensor_bytes,
                                           (uint32_t) c.n_ranks);
            }
            for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
                const size_t       idx      = tp5_plan_slot(i, b);
                const VkDeviceSize bank_off = (VkDeviceSize) b * payload_bytes;
                const VkDeviceSize slot_off = bank_off + (VkDeviceSize) i * stride;
                VkCommandBuffer    cmd      = new_p1->cmd_p1[idx];
                vk_tp5_capture_scope p1_scope(define_inline ? &new_p1->definitions[idx] : nullptr);
                vk_tp5_register_source(cmd);
                if (vkBeginCommandBuffer(cmd, &beg) != VK_SUCCESS) {
                    c.fail("begin cmd_p1 failed on rank " + std::to_string(i));
                    return false;
                }

                if (c.sync_mode == tp5_sync_mode::RELAY) {
                    const bool copy_payload = c.wire == tp5_wire_type::F32 || has_packed;
                    const VkBuffer source_buf = has_packed ? trefs[i].packed_buf : trefs[i].buf;
                    const VkDeviceSize source_offset = has_packed ? trefs[i].packed_offset : trefs[i].offset;
                    const VkDeviceSize source_bytes = c.wire == tp5_wire_type::F16 ? payload : tensor_bytes;
                    const bool vector_copy =
                        copy_payload && r.relay_p1_copy_pipe != VK_NULL_HANDLE &&
                        idx < new_p1->ds_relay_copy.size() && new_p1->ds_relay_copy[idx] != VK_NULL_HANDLE &&
                        source_bytes >= 16 && source_bytes <= 16384 &&
                        source_bytes % 16 == 0 && source_offset % 16 == 0;
                    VkMemoryBarrier mb_pre{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                           VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                                           copy_payload ?
                                               (vector_copy ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_TRANSFER_READ_BIT) :
                                               VK_ACCESS_SHADER_READ_BIT};
                    tp5_cmd_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         copy_payload ?
                                             (vector_copy ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT :
                                                            VK_PIPELINE_STAGE_TRANSFER_BIT) :
                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         0, 1, &mb_pre, 0, nullptr, 0, nullptr);

                    if (copy_payload) {
                        if (vector_copy) {
                            VkDescriptorBufferInfo infos[2] = {
                                {source_buf, source_offset, source_bytes},
                                {r.host_import_buf[b], 0, source_bytes},
                            };
                            tp5_update_storage_set(r.vkdev, new_p1->ds_relay_copy[idx], infos, 2);
                            tp5_cmd_bind_pipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.relay_p1_copy_pipe);
                            tp5_cmd_bind_descriptors(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.relay_p1_copy_layout, 0, 1,
                                                    &new_p1->ds_relay_copy[idx], 0, nullptr);
                            const uint32_t n_uvec4 = (uint32_t) (source_bytes / 16);
                            tp5_cmd_push(cmd, r.relay_p1_copy_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                               sizeof(n_uvec4), &n_uvec4);
                            // Decode-sized payloads stay in one workgroup.
                            // Larger transfers use the driver's transfer path
                            // instead of turning the compute queue into a DMA
                            // engine.
                            tp5_cmd_dispatch(cmd, 1, 1, 1);
                            VkMemoryBarrier mb_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                                    VK_ACCESS_SHADER_WRITE_BIT,
                                                    VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT};
                            tp5_cmd_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                                 VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                                                 0, 1, &mb_host, 0, nullptr, 0, nullptr);
                        } else {
                            VkBufferCopy copy{source_offset, 0, source_bytes};
                            tp5_cmd_copy(cmd, source_buf, r.host_import_buf[b], 1, &copy);
                            VkMemoryBarrier mb_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                                    VK_ACCESS_TRANSFER_WRITE_BIT,
                                                    VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT};
                            tp5_cmd_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                 VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                                                 0, 1, &mb_host, 0, nullptr, 0, nullptr);
                        }
                    } else {
                        if (new_p1->ds_relay_pack[idx] == VK_NULL_HANDLE) {
                            c.fail("missing RELAY pack descriptor on rank " + std::to_string(i));
                            return false;
                        }
                        tp5_update_pack_descriptor(r, new_p1->ds_relay_pack[idx], trefs[i].buf, trefs[i].offset,
                                                   tensor_bytes, (uint32_t) c.n_ranks, r.host_import_buf[b], 0, payload);
                        const bool vector_pack =
                            r.pack_vec_pipe != VK_NULL_HANDLE && n_elems >= 4 && n_elems % 4 == 0 &&
                            trefs[i].offset % 16 == 0;
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          vector_pack ? r.pack_vec_pipe : r.pack_pipe);
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipe_layout, 0, 1,
                                                &new_p1->ds_relay_pack[idx], 0, nullptr);
                        const uint32_t n = vector_pack ? (uint32_t) (n_elems / 4) : (uint32_t) n_elems;
                        vkCmdPushConstants(cmd, r.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(n), &n);
                        const uint32_t groups = vector_pack ?
                            (n_elems <= 4096 ? 1u : (n + 63u) / 64u) :
                            (n + 255u) / 256u;
                        vkCmdDispatch(cmd, groups, 1, 1);
                        VkMemoryBarrier mb_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                                VK_ACCESS_SHADER_WRITE_BIT,
                                                VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT};
                        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                                             0, 1, &mb_host, 0, nullptr, 0, nullptr);
                    }

                    // The route table is only for direct producers. The fallback
                    // publishes the conventional host-import completion word.
                    tp5_cmd_fill(cmd, r.host_import_buf[b], c.star_rank_stride - 64, sizeof(uint32_t), 1u);
                    VkMemoryBarrier mb_ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                             VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
                    tp5_cmd_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                         0, 1, &mb_ready, 0, nullptr, 0, nullptr);
                    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
                        c.fail("end cmd_p1 RELAY failed on rank " + std::to_string(i));
                        return false;
                    }
                    continue;
                }

                if (c.sync_mode == tp5_sync_mode::STAR) {
                    VkMemoryBarrier mb_pre{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                            VK_ACCESS_SHADER_WRITE_BIT,
                                            VK_ACCESS_SHADER_READ_BIT };
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         0, 1, &mb_pre, 0, nullptr, 0, nullptr);

                    uint64_t stage_bda_src = 0;
                    auto pfn_bda = (PFN_vkGetBufferDeviceAddressKHR) vkGetDeviceProcAddr(r.vkdev, "vkGetBufferDeviceAddressKHR");
                    if (!pfn_bda) pfn_bda = (PFN_vkGetBufferDeviceAddressKHR) vkGetDeviceProcAddr(r.vkdev, "vkGetBufferDeviceAddress");
                    if (pfn_bda) {
                        VkBufferDeviceAddressInfo dai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr,
                                                     has_packed ? trefs[i].packed_buf : trefs[i].buf};
                        stage_bda_src = pfn_bda(r.vkdev, &dai) + (has_packed ? trefs[i].packed_offset : trefs[i].offset);
                    }
                    uint64_t dst_bda = r.bda_addr[b];
                    uint32_t n_vec4 = has_packed ? (uint32_t)(n_elems / 8) : (uint32_t)(n_elems / 4);
                    uint64_t flag_bda = dst_bda + c.star_rank_stride - 64;

                    // Dispatch 1: payload data. A terminal producer may have
                    // already emitted the canonical F16 companion, in which
                    // case P1 copies it directly instead of re-reading F32 and
                    // converting it again.
                    struct {
                        uint64_t src_bda;
                        uint64_t dst_bda;
                        uint64_t flag_bda;
                        uint32_t n_vec4;
                        uint32_t rank_idx;
                        uint32_t seq_val;
                        uint32_t mode;
                    } bda_pc{stage_bda_src, dst_bda, flag_bda, n_vec4, (uint32_t)n_elems, 1u,
                             c.wire == tp5_wire_type::F32 ? 5u : (has_packed ? 2u : 0u)};
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.bda_push_pipe);
                    vkCmdPushConstants(cmd, r.bda_push_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bda_pc), &bda_pc);
                    vkCmdDispatch(cmd, (uint32_t)(n_elems + (has_packed ? 511 : 255)) / (has_packed ? 512 : 256),
                                  1, 1);

                    // The completion flag must not become host-visible before
                    // every payload store.
                    VkMemoryBarrier mb_host{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                             VK_ACCESS_SHADER_WRITE_BIT,
                                             VK_ACCESS_HOST_READ_BIT };
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         0, 1, &mb_host, 0, nullptr, 0, nullptr);

                    bda_pc.mode = 1;
                    vkCmdPushConstants(cmd, r.bda_push_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(bda_pc), &bda_pc);
                    vkCmdDispatch(cmd, 1, 1, 1);
                    VkMemoryBarrier mb_flag{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                             VK_ACCESS_SHADER_WRITE_BIT,
                                             VK_ACCESS_HOST_READ_BIT };
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                         0, 1, &mb_flag, 0, nullptr, 0, nullptr);
                    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
                        c.fail("end cmd_p1 star failed on rank " + std::to_string(i));
                        return false;
                    }
                    continue;
                }
                if (has_packed) {
                    const bool use_parallel_push = allow_parallel_push &&
                                                   (r.push_pipe != VK_NULL_HANDLE && c.n_ranks == 5 &&
                                                    (payload % 16 == 0) && r.imports.size() >= 4 &&
                                                    (c.wire == tp5_wire_type::F16));
                    if (use_parallel_push) {
                        tp5_update_push_descriptor(c, r, i, new_p1->ds_push[idx], trefs[i].packed_buf,
                                                   trefs[i].packed_offset, payload, b, slot_off);
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.push_pipe);
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.push_layout, 0, 1,
                                                &new_p1->ds_push[idx], 0, nullptr);
                        const uint32_t n_uvec4 = (uint32_t) (payload / 16);
                        vkCmdPushConstants(cmd, r.push_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n_uvec4);
                        vkCmdDispatch(cmd, (n_uvec4 + 63) / 64, 1, 1);
                    } else {
                    VkMemoryBarrier mb_pre{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                                            VK_ACCESS_TRANSFER_READ_BIT };
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                         1, &mb_pre, 0, nullptr, 0, nullptr);

                    VkBufferCopy local_cp{ trefs[i].packed_offset, c.isolate_mailbox ? 0 : slot_off, payload };
                    vkCmdCopyBuffer(cmd, trefs[i].packed_buf,
                                    c.isolate_mailbox ? r.inboxes[tp5_plan_slot(i, b)].buf : r.mailbox_buf, 1,
                                    &local_cp);
                    for (size_t p = 0; p < r.imports.size(); ++p) {
                        if (c.isolate_mailbox && r.imports[p].bank != b) {
                            continue;
                        }
                        VkBufferCopy peer_cp{ trefs[i].packed_offset, c.isolate_mailbox ? 0 : slot_off, payload };
                        vkCmdCopyBuffer(cmd, trefs[i].packed_buf, r.imports[p].buf, 1, &peer_cp);
                    }
                    }
                } else {
                    VkMemoryBarrier mb_pre{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                                            VkAccessFlags(c.wire == tp5_wire_type::F32 ?
                                                              VK_ACCESS_TRANSFER_READ_BIT |
                                                                  VK_ACCESS_TRANSFER_WRITE_BIT :
                                                              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT) };
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         (c.wire == tp5_wire_type::F32 ? VK_PIPELINE_STAGE_TRANSFER_BIT :
                                                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
                                         0, 1, &mb_pre, 0, nullptr, 0, nullptr);

                    if (gpuflag) {
                        tp5_flag_pc wait_pc = flag_pc_base;
                        wait_pc.mode        = 0u;
                        tp5_record_flag_dispatch(cmd, r, new_p1->ds_flag[i], wait_pc);
                        VkMemoryBarrier mb_flag_wait{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                                      VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT };
                        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb_flag_wait, 0, nullptr, 0,
                                             nullptr);
                    }

                    if (c.wire == tp5_wire_type::F32) {
                        VkBufferCopy cp{ trefs[i].offset, 0, tensor_bytes };
                        vkCmdCopyBuffer(cmd, trefs[i].buf, r.wire_buf, 1, &cp);
                    } else {
                        const bool vector_pack =
                            r.pack_vec_pipe != VK_NULL_HANDLE && n_elems >= 4 && n_elems % 4 == 0 &&
                            trefs[i].offset % 16 == 0;
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          vector_pack ? r.pack_vec_pipe : r.pack_pipe);
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipe_layout, 0, 1,
                                                &new_p1->ds_pack[i], 0, nullptr);
                        uint32_t n = vector_pack ? (uint32_t) (n_elems / 4) : (uint32_t) n_elems;
                        vkCmdPushConstants(cmd, r.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n);
                        const uint32_t groups = vector_pack ?
                            (n_elems <= 4096 ? 1u : (n + 63u) / 64u) :
                            (n + 255u) / 256u;
                        vkCmdDispatch(cmd, groups, 1, 1);
                    }

                    VkMemoryBarrier mb_wire{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                             (c.wire == tp5_wire_type::F32 ? VK_ACCESS_TRANSFER_WRITE_BIT :
                                                                             VK_ACCESS_SHADER_WRITE_BIT),
                                             (c.wire == tp5_wire_type::F16 && r.push_pipe != VK_NULL_HANDLE ?
                                                  VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_TRANSFER_READ_BIT) };
                    vkCmdPipelineBarrier(cmd,
                                         (c.wire == tp5_wire_type::F32 ? VK_PIPELINE_STAGE_TRANSFER_BIT :
                                                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
                                         (c.wire == tp5_wire_type::F16 && r.push_pipe != VK_NULL_HANDLE && allow_parallel_push ?
                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT),
                                         0, 1, &mb_wire, 0, nullptr, 0, nullptr);

                    const bool use_parallel_push = allow_parallel_push &&
                                                   (r.push_pipe != VK_NULL_HANDLE && c.n_ranks == 5 &&
                                                    (payload % 16 == 0) && r.imports.size() >= 4 &&
                                                    (c.wire == tp5_wire_type::F16));
                    if (use_parallel_push) {
                        tp5_update_push_descriptor(c, r, i, new_p1->ds_push[idx], r.wire_buf, 0, payload, b, slot_off);
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.push_pipe);
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.push_layout, 0, 1,
                                                &new_p1->ds_push[idx], 0, nullptr);
                        const uint32_t n_uvec4 = (uint32_t) (payload / 16);
                        vkCmdPushConstants(cmd, r.push_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n_uvec4);
                        vkCmdDispatch(cmd, (n_uvec4 + 63) / 64, 1, 1);
                    } else {
                    VkBufferCopy local_cp{ 0, c.isolate_mailbox ? 0 : slot_off, payload };
                    vkCmdCopyBuffer(cmd, r.wire_buf,
                                    c.isolate_mailbox ? r.inboxes[tp5_plan_slot(i, b)].buf : r.mailbox_buf, 1,
                                    &local_cp);
                    for (size_t p = 0; p < r.imports.size(); ++p) {
                        if (c.isolate_mailbox && r.imports[p].bank != b) {
                            continue;
                        }
                        VkBufferCopy peer_cp{ 0, c.isolate_mailbox ? 0 : slot_off, payload };
                        vkCmdCopyBuffer(cmd, r.wire_buf, r.imports[p].buf, 1, &peer_cp);
                    }
                    }
                }

                const bool any_p_push = allow_parallel_push &&
                                        (r.push_pipe != VK_NULL_HANDLE && c.n_ranks == 5 &&
                                         (payload % 16 == 0) && r.imports.size() >= 4 &&
                                         (c.wire == tp5_wire_type::F16));
                VkMemoryBarrier mb_p1{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                       any_p_push ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_TRANSFER_WRITE_BIT,
                                       VK_ACCESS_SHADER_READ_BIT };
                vkCmdPipelineBarrier(cmd,
                                     any_p_push ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                     &mb_p1, 0, nullptr, 0, nullptr);

                if (gpuflag) {
                    tp5_flag_pc pub_pc = flag_pc_base;
                    pub_pc.mode        = 1u;
                    tp5_record_flag_dispatch(cmd, r, new_p1->ds_flag[i], pub_pc);
                    VkMemoryBarrier mb_flag_pub{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                                                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT };
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                         0, 1, &mb_flag_pub, 0, nullptr, 0, nullptr);
                }

                if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
                    c.fail("end cmd_p1 failed on rank " + std::to_string(i));
                    return false;
                }
            }
        }
        p1_res = new_p1;
    }
    }

    plan.p1 = p1_res;
    plan.cmd_p2.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
    if (define_inline) plan.p2_definitions.resize(c.n_ranks * TP5_MAILBOX_BANKS);
    plan.ds_sum.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
    bool late_plan = c.sync_mode == tp5_sync_mode::RELAY && !trefs.empty();
    for (size_t i = 0; late_plan && i < c.n_ranks; ++i)
        late_plan = trefs[i].late.late_rank != 0;
    tp5_numerical_spec num_spec;
    num_spec.latebind_env_enabled = tp5_latebind_hc_enabled();
    num_spec.is_relay_sync        = (c.sync_mode == tp5_sync_mode::RELAY);
    num_spec.is_f32_wire          = (c.wire == tp5_wire_type::F32);
    num_spec.has_late_tensors     = late_plan;
    num_spec.exact_q_requested    = tp5_latebind_exact_q_requested();
    num_spec.p1a_nosidecar_requested = tp5_p1a_nosidecar_q8_requested();
    num_spec.hw_int_dot           = true;
    num_spec.hw_wave32            = true;
    num_spec.shape_aligned        = true;
    num_spec.pipeline_ready       = true;
    for (size_t i = 0; i < c.n_ranks; ++i) {
        num_spec.hw_int_dot     = num_spec.hw_int_dot && c.ranks[i].caps.integer_dot_product;
        num_spec.hw_wave32      = num_spec.hw_wave32 && c.ranks[i].caps.subgroup_size_control &&
                                  c.ranks[i].caps.subgroup_min_size <= 32u && c.ranks[i].caps.subgroup_max_size >= 32u;
        num_spec.shape_aligned  = num_spec.shape_aligned && trefs[i].late.streams == 4u &&
                                  trefs[i].late.width % 32u == 0u && trefs[i].late.late_rank % 16u == 0u;
        num_spec.pipeline_ready = num_spec.pipeline_ready &&
                                  c.ranks[i].late_act_q8_pipe != VK_NULL_HANDLE && c.ranks[i].late_q8dot_pipe != VK_NULL_HANDLE &&
                                  c.ranks[i].late_lo_q8_pipe != VK_NULL_HANDLE && c.ranks[i].late_up_q8dot_pipe != VK_NULL_HANDLE;
    }
    const auto resolved = tp5_resolve_numerical_mode(num_spec);
    plan.numerical_mode  = resolved.first;
    plan.numerical_reason = resolved.second;
    plan.late_q8_fast    = (plan.numerical_mode == tp5_numerical_mode::AGGRESSIVE_Q8 ||
                            plan.numerical_mode == tp5_numerical_mode::P1A_NOSIDECAR_Q8);
    // The F16 sidecar format is a property of the Q8DOT producer, not of the
    // aggressive schedule: P1-A dispatches the same tp5_hc_latebind Q8DOT,
    // which packs F16 pairs into bcast VRAM and publishes the Q control
    // region. Keying the format off the mode name left P1-A with F32/exact
    // consumers: the handoff polled status[6] (written only by the exact-path
    // publisher) and timed out, and LO decoded F16 words as F32 bits.
    plan.late_sidecar_f16 = plan.late_q8_fast;

    static std::atomic<uint32_t> reported_modes_mask{0};
    const uint32_t mode_bit = 1u << (uint32_t) plan.numerical_mode;
    if ((reported_modes_mask.fetch_or(mode_bit, std::memory_order_relaxed) & mode_bit) == 0) {
        fprintf(stderr,
                "[tp5-numerical-mode] mode=%s desc=%s reason=%s wire=%s late=%s direct=%s\n",
                tp5_numerical_mode_name(plan.numerical_mode),
                tp5_numerical_mode_desc(plan.numerical_mode),
                tp5_numerical_reason_name(plan.numerical_reason),
                c.wire == tp5_wire_type::F32 ? "f32" : "f16",
                late_plan ? "yes" : "no",
                relay_direct_all ? "direct" : "p1");
    }

    if (late_plan) {
        static std::atomic<bool> late_reported{false};
        if (!late_reported.exchange(true, std::memory_order_relaxed)) {
            fprintf(stderr,
                    "[tp5-latebind] hc-down active producer=%s q=%s mode=%s width=%u streams=%u rank=%u sidecar=%zuB "
                    "(GGML_TP5_REPLICATE_ATTN remains independent)\n",
                    relay_direct_all ? "direct" : "p1",
                    plan.late_q8_fast ? "q8dot-approx" : "f32-exact",
                    tp5_numerical_mode_name(plan.numerical_mode),
                    trefs[0].late.width, trefs[0].late.streams, trefs[0].late.late_rank,
                    size_t(trefs[0].late.streams) * trefs[0].late.late_rank *
                        (plan.late_sidecar_f16 ? sizeof(ggml_fp16_t) : sizeof(float)));
        }
        plan.cmd_late_pre.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
        plan.pre_definitions.resize(c.n_ranks * TP5_MAILBOX_BANKS);
        plan.late_inject_end.resize(c.n_ranks * TP5_MAILBOX_BANKS);
        plan.late_q_begin.resize(c.n_ranks * TP5_MAILBOX_BANKS);
        plan.late_q_contract_begin.resize(c.n_ranks * TP5_MAILBOX_BANKS);
        plan.late_norm_end.resize(c.n_ranks * TP5_MAILBOX_BANKS);
        plan.late_lo_begin.resize(c.n_ranks * TP5_MAILBOX_BANKS);
        plan.late_inject_ds.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
        if (plan.late_q8_fast) {
            plan.late_act_q8_ds.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
            plan.late_q8dot_ds.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
            plan.late_lo_q8_ds.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
            plan.late_up_q8dot_ds.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
            plan.late_down_packed_buf.resize(c.n_ranks, VK_NULL_HANDLE);
            plan.late_down_packed_mem.resize(c.n_ranks, VK_NULL_HANDLE);
            plan.late_up_packed_buf.resize(c.n_ranks, VK_NULL_HANDLE);
            plan.late_up_packed_mem.resize(c.n_ranks, VK_NULL_HANDLE);
            plan.late_pack_ds.resize(2 * c.n_ranks, VK_NULL_HANDLE);
        } else {
            plan.late_q_ds.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
            plan.late_publish_ds.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
            plan.late_lo_ds.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
        }
        plan.late_norm_ds.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
        plan.late_scatter_buf.resize(c.n_ranks, VK_NULL_HANDLE);
        plan.late_scatter_mem.resize(c.n_ranks, VK_NULL_HANDLE);
        plan.late_rho_buf.resize(c.n_ranks, VK_NULL_HANDLE);
        plan.late_rho_mem.resize(c.n_ranks, VK_NULL_HANDLE);
        if (plan.late_q8_fast) {
            plan.late_act_q8_buf.resize(c.n_ranks, VK_NULL_HANDLE);
            plan.late_act_q8_mem.resize(c.n_ranks, VK_NULL_HANDLE);
            plan.late_lo_q8_buf.resize(c.n_ranks, VK_NULL_HANDLE);
            plan.late_lo_q8_mem.resize(c.n_ranks, VK_NULL_HANDLE);
        } else {
            plan.late_sidecar_buf.resize(c.n_ranks, VK_NULL_HANDLE);
            plan.late_sidecar_mem.resize(c.n_ranks, VK_NULL_HANDLE);
        }
    }
    if (c.sync_mode == tp5_sync_mode::RELAY) {
        plan.relay_ds.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
    }

    // Allocate per-plan Phase 2 command buffers and descriptor sets (2 mailbox banks × n_ranks)
    for (size_t i = 0; i < c.n_ranks; ++i) {
        tp5_rank &                  r = c.ranks[i];
        VkCommandBufferAllocateInfo cba{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, r.cmd_pool,
                                         VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
        VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, r.desc_pool, 1,
                                        &r.dsl };
        VkDescriptorSetAllocateInfo sum_ai = ai;
        if (trefs[i].hc.width) {
            const size_t pipe_idx = (trefs[i].hc.quantized.buffer != nullptr) ? 1 : 0;
            sum_ai.pSetLayouts = &r.hc_sum_dsl[pipe_idx];
        }
        if (late_plan) {
            const auto & late = trefs[i].late;
            const size_t max_rows = VK_TP5_DIRECT_COLUMN_TILE;
            const size_t max_late_count = max_rows * size_t(trefs[i].late.streams) * trefs[i].late.late_rank;
            if (!tp5_alloc_device_buffer(r, max_rows * 4 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false,
                                         plan.late_rho_buf[i], plan.late_rho_mem[i], nullptr)) {
                c.fail("allocation of LateBind rho buffer failed on rank " + std::to_string(i));
                return false;
            }
            if (!tp5_alloc_device_buffer(r, max_rows * 4 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false,
                                         plan.late_scatter_buf[i], plan.late_scatter_mem[i], nullptr)) {
                c.fail("allocation of LateBind scatter buffer failed on rank " + std::to_string(i));
                return false;
            }
            if (plan.late_q8_fast) {
                const size_t act_elems = max_rows * size_t(late.streams) * late.width;
                const VkDeviceSize act_q8_bytes = (VkDeviceSize) ggml_row_size(GGML_TYPE_Q8_0, act_elems);
                if (act_q8_bytes == 0 || act_q8_bytes > r.caps.max_storage_buffer_range ||
                    !tp5_alloc_device_buffer(r, act_q8_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false,
                                             plan.late_act_q8_buf[i], plan.late_act_q8_mem[i], nullptr)) {
                    c.fail("allocation of LateBind Q8 activation buffer failed on rank " + std::to_string(i));
                    return false;
                }
                const VkDeviceSize lo_q8_bytes =
                    (VkDeviceSize) ggml_row_size(GGML_TYPE_Q8_0, max_rows * late.late_rank);
                if (lo_q8_bytes == 0 || lo_q8_bytes > r.caps.max_storage_buffer_range ||
                    !tp5_alloc_device_buffer(r, lo_q8_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false,
                                             plan.late_lo_q8_buf[i], plan.late_lo_q8_mem[i], nullptr)) {
                    c.fail("allocation of LateBind Q8 LO buffer failed on rank " + std::to_string(i));
                    return false;
                }
                // P1-B packed weight buffers: one extra flag block at index 0.
                const uint64_t weight_blocks =
                    uint64_t(late.late_rank) * late.streams * late.width / 32u;
                const VkDeviceSize packed_bytes =
                    VkDeviceSize(weight_blocks + 1u) * (sizeof(float) + 8u * sizeof(uint32_t));
                if (packed_bytes == 0 || packed_bytes > 2u * r.caps.max_storage_buffer_range ||
                    !tp5_alloc_device_buffer(r, packed_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false,
                                             plan.late_down_packed_buf[i], plan.late_down_packed_mem[i],
                                             nullptr) ||
                    !tp5_alloc_device_buffer(r, packed_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false,
                                             plan.late_up_packed_buf[i], plan.late_up_packed_mem[i],
                                             nullptr)) {
                    c.fail("allocation of LateBind packed weight buffers failed on rank " + std::to_string(i));
                    return false;
                }
                // The pack done flag must start at zero: device-local memory
                // is uninitialized, and a freed earlier plan's buffer could
                // otherwise alias with flag==1 and silently skip this plan's
                // pack (stale weights). A dedicated one-shot fill CB with a
                // fence wait runs only here, at plan creation (cold path).
                if (r.late_pack_pipe != VK_NULL_HANDLE) {
                    VkCommandBuffer clear_cmd = VK_NULL_HANDLE;
                    VkCommandBufferAllocateInfo clear_ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                                         nullptr, r.cmd_pool,
                                                         VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
                    if (vkAllocateCommandBuffers(r.vkdev, &clear_ai, &clear_cmd) != VK_SUCCESS) {
                        c.fail("allocation of LateBind pack clear CB failed on rank " + std::to_string(i));
                    return false;
                }
                    VkFence clear_fence = VK_NULL_HANDLE;
                    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
                    if (vkCreateFence(r.vkdev, &fci, nullptr, &clear_fence) != VK_SUCCESS) {
                        vkFreeCommandBuffers(r.vkdev, r.cmd_pool, 1, &clear_cmd);
                        c.fail("creation of LateBind pack clear fence failed on rank " + std::to_string(i));
                    return false;
                }
                    VkCommandBufferBeginInfo clear_bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                    if (vkBeginCommandBuffer(clear_cmd, &clear_bi) == VK_SUCCESS &&
                        vkEndCommandBuffer(clear_cmd) == VK_SUCCESS) {
                        vkCmdFillBuffer(clear_cmd, plan.late_down_packed_buf[i], 0, 4u, 0u);
                        vkCmdFillBuffer(clear_cmd, plan.late_up_packed_buf[i], 0, 4u, 0u);
                        VkSubmitInfo clear_si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                        clear_si.commandBufferCount = 1;
                        clear_si.pCommandBuffers = &clear_cmd;
                        if (vkQueueSubmit(r.queue, 1, &clear_si, clear_fence) == VK_SUCCESS &&
                            vkWaitForFences(r.vkdev, 1, &clear_fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS) {
                            // Flag words are zero before any pack replay.
                        } else {
                            c.fail("submission of LateBind pack clear failed on rank " + std::to_string(i));
                        }
                    } else {
                        c.fail("recording of LateBind pack clear failed on rank " + std::to_string(i));
                    }
                    vkDestroyFence(r.vkdev, clear_fence, nullptr);
                    vkFreeCommandBuffers(r.vkdev, r.cmd_pool, 1, &clear_cmd);
                    if (c.failed) return false;
                }
            } else {
                const VkDeviceSize late_bytes = (VkDeviceSize) max_late_count * sizeof(float);
                if (!tp5_alloc_device_buffer(r, late_bytes,
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                             false, plan.late_sidecar_buf[i], plan.late_sidecar_mem[i], nullptr)) {
                    c.fail("allocation of LateBind local sidecar failed on rank " + std::to_string(i));
                    return false;
                }
            }
            for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
                VkDescriptorSetAllocateInfo inj_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                                    r.desc_pool, 1, &r.late_inject_dsl};
                if (vkAllocateDescriptorSets(r.vkdev, &inj_ai,
                                             &plan.late_inject_ds[tp5_plan_slot(i, b)]) != VK_SUCCESS) {
                    c.fail("allocation of LateBind inject descriptor failed on rank " + std::to_string(i));
                    return false;
                }
            }
            if (plan.late_q8_fast && r.late_pack_dsl != VK_NULL_HANDLE) {
                // P1-B pack descriptors are per-rank: both bank replays of
                // cmd_late_pre bind the same read-only weight/packed pair.
                // The pack dispatch self-disables via the persistent done
                // flag, so the second replay is a no-op guard read.
                VkDescriptorSetAllocateInfo pk_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                                   r.desc_pool, 1, &r.late_pack_dsl};
                if (vkAllocateDescriptorSets(r.vkdev, &pk_ai, &plan.late_pack_ds[i]) != VK_SUCCESS) {
                    c.fail("allocation of LateBind pack descriptor failed on rank " + std::to_string(i));
                    return false;
                }
            }
        }
        for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
            const size_t idx = tp5_plan_slot(i, b);
            if (vkAllocateCommandBuffers(r.vkdev, &cba, &plan.cmd_p2[idx]) != VK_SUCCESS) {
                c.fail("allocation of P2 command buffer failed on rank " + std::to_string(i));
                return false;
            }
            if (late_plan && vkAllocateCommandBuffers(r.vkdev, &cba, &plan.cmd_late_pre[idx]) != VK_SUCCESS) {
                c.fail("allocation of LateBind precompute command buffer failed on rank " + std::to_string(i));
                return false;
            }
            if (vkAllocateDescriptorSets(r.vkdev, &sum_ai, &plan.ds_sum[idx]) != VK_SUCCESS) {
                c.fail("allocation of sum descriptor set failed on rank " + std::to_string(i));
                return false;
            }
            if (late_plan) {
                VkDescriptorSetAllocateInfo n_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, r.desc_pool,
                                                  1, &r.late_norm_dsl};
                if (vkAllocateDescriptorSets(r.vkdev, &n_ai, &plan.late_norm_ds[idx]) != VK_SUCCESS) {
                    c.fail("allocation of LateBind descriptor set failed on rank " + std::to_string(i));
                    return false;
                }
                if (plan.late_q8_fast) {
                    VkDescriptorSetAllocateInfo a8_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                                      r.desc_pool, 1, &r.late_act_q8_dsl};
                    VkDescriptorSetAllocateInfo q8_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                                      r.desc_pool, 1, &r.late_q8dot_dsl};
                    VkDescriptorSetAllocateInfo lo8_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                                       r.desc_pool, 1, &r.late_lo_q8_dsl};
                    VkDescriptorSetAllocateInfo up8_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                                       r.desc_pool, 1, &r.late_up_q8dot_dsl};
                    if (vkAllocateDescriptorSets(r.vkdev, &a8_ai, &plan.late_act_q8_ds[idx]) != VK_SUCCESS ||
                        vkAllocateDescriptorSets(r.vkdev, &q8_ai, &plan.late_q8dot_ds[idx]) != VK_SUCCESS ||
                        vkAllocateDescriptorSets(r.vkdev, &lo8_ai, &plan.late_lo_q8_ds[idx]) != VK_SUCCESS ||
                        vkAllocateDescriptorSets(r.vkdev, &up8_ai, &plan.late_up_q8dot_ds[idx]) != VK_SUCCESS) {
                        c.fail("allocation of LateBind aggressive Q8 descriptor set failed on rank " +
                               std::to_string(i));
                        return false;
                    }
                } else {
                    VkDescriptorSetAllocateInfo q_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                                      r.desc_pool, 1, &r.late_q_dsl};
                    VkDescriptorSetAllocateInfo pub_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                                        r.desc_pool, 1, &r.late_publish_dsl};
                    VkDescriptorSetAllocateInfo l_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                                      r.desc_pool, 1, &r.late_lo_dsl};
                    if (vkAllocateDescriptorSets(r.vkdev, &q_ai, &plan.late_q_ds[idx]) != VK_SUCCESS ||
                        vkAllocateDescriptorSets(r.vkdev, &pub_ai, &plan.late_publish_ds[idx]) != VK_SUCCESS ||
                        vkAllocateDescriptorSets(r.vkdev, &l_ai, &plan.late_lo_ds[idx]) != VK_SUCCESS) {
                        c.fail("allocation of LateBind exact descriptor set failed on rank " +
                               std::to_string(i));
                        return false;
                    }
                }
            }
            if (c.sync_mode == tp5_sync_mode::RELAY && !trefs[i].hc.width) {
                VkDescriptorSetAllocateInfo relay_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                relay_ai.descriptorPool = r.desc_pool;
                relay_ai.descriptorSetCount = 1;
                relay_ai.pSetLayouts = &r.relay_copy_dsl;
                if (vkAllocateDescriptorSets(r.vkdev, &relay_ai, &plan.relay_ds[idx]) != VK_SUCCESS) {
                    c.fail("allocation of RELAY descriptor set failed on rank " + std::to_string(i));
                    return false;
                }
            }
        }
    }

    if (late_plan) {
        const uint32_t profile_spin = getenv("GGML_TP5_PROFILE") ? 1u : 0u;
        (void) profile_spin;
        for (size_t i = 0; i < c.n_ranks; ++i) {
            tp5_rank & r = c.ranks[i];
            const auto & late = trefs[i].late;
            const size_t max_rows = VK_TP5_DIRECT_COLUMN_TILE;
            const size_t late_count = max_rows * size_t(late.streams) * late.late_rank;
            VkDescriptorBufferInfo scatter_info{plan.late_scatter_buf[i], 0, max_rows * 4 * sizeof(float)};

            for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
                const size_t idx = tp5_plan_slot(i, b);
                VkDescriptorBufferInfo inject_infos[4] = {
                    {late.bindings[4].buffer, late.bindings[4].offset, late.bindings[4].size},
                    {late.bindings[5].buffer, late.bindings[5].offset, late.bindings[5].size},
                    scatter_info,
                    {r.bcast_buf[b], 0, (VkDeviceSize) c.star_rank_stride},
                };
                tp5_update_storage_set(r.vkdev, plan.late_inject_ds[idx], inject_infos, 4);
                if (plan.late_q8_fast) {
                    const size_t act_elems = max_rows * size_t(late.streams) * late.width;
                    const VkDeviceSize act_q8_bytes = (VkDeviceSize) ggml_row_size(GGML_TYPE_Q8_0, act_elems);
                    // P1-B packed weight layout: (blocks + 1 flag) * 36B.
                    const uint64_t weight_blocks =
                        uint64_t(late.late_rank) * late.streams * late.width / 32u;
                    const VkDeviceSize packed_bytes =
                        VkDeviceSize(weight_blocks + 1u) * (sizeof(float) + 8u * sizeof(uint32_t));
                    VkDescriptorBufferInfo act_q8_infos[6] = {
                        {late.bindings[1].buffer, late.bindings[1].offset, late.bindings[1].size},
                        {late.bindings[0].buffer, late.bindings[0].offset, late.bindings[0].size},
                        {trefs[i].buf, trefs[i].offset, trefs[i].size},
                        scatter_info,
                        {plan.late_act_q8_buf[i], 0, act_q8_bytes},
                        {r.bcast_buf[b], 0, (VkDeviceSize) c.star_rank_stride},
                    };
                    tp5_update_storage_set(r.vkdev, plan.late_act_q8_ds[idx], act_q8_infos, 6);
                    VkDescriptorBufferInfo q8dot_infos[3] = {
                        {plan.late_down_packed_buf[i], 0, packed_bytes},
                        {plan.late_act_q8_buf[i], 0, act_q8_bytes},
                        {r.bcast_buf[b], 0, (VkDeviceSize) c.star_rank_stride},
                    };
                    tp5_update_storage_set(r.vkdev, plan.late_q8dot_ds[idx], q8dot_infos, 3);
                } else {
                    VkDescriptorBufferInfo q_infos[7] = {
                        {late.down_weight.buffer, late.down_weight.offset, late.down_weight.size},
                        {late.bindings[1].buffer, late.bindings[1].offset, late.bindings[1].size},
                        {late.bindings[0].buffer, late.bindings[0].offset, late.bindings[0].size},
                        {trefs[i].buf, trefs[i].offset, trefs[i].size},
                        scatter_info,
                        {plan.late_sidecar_buf[i], 0, (VkDeviceSize) (late_count * sizeof(float))},
                        {r.bcast_buf[b], 0, (VkDeviceSize) c.star_rank_stride},
                    };
                    tp5_update_storage_set(r.vkdev, plan.late_q_ds[idx], q_infos, 7);
                    VkDescriptorBufferInfo late_publish_infos[3] = {
                        {plan.late_sidecar_buf[i], 0, (VkDeviceSize) (late_count * sizeof(float))},
                        {r.host_import_buf[b], (VkDeviceSize) c.late_host_offset,
                         (VkDeviceSize) (late_count * sizeof(float))},
                        {r.host_import_buf[b], (VkDeviceSize) c.star_rank_stride - 64, 64},
                    };
                    tp5_update_storage_set(r.vkdev, plan.late_publish_ds[idx], late_publish_infos, 3);
                }

                VkDescriptorBufferInfo norm_infos[9] = {
                    {r.bcast_buf[b], 0, (VkDeviceSize) c.star_rank_stride},
                    {late.bindings[0].buffer, late.bindings[0].offset, late.bindings[0].size},
                    {late.bindings[1].buffer, late.bindings[1].offset, late.bindings[1].size},
                    {late.bindings[2].buffer, late.bindings[2].offset, late.bindings[2].size},
                    {late.bindings[3].buffer, late.bindings[3].offset, late.bindings[3].size},
                    scatter_info,
                    {plan.late_rho_buf[i], 0, max_rows * 4 * sizeof(float)},
                    {r.host_import_buf[b], 0, (VkDeviceSize) c.star_rank_stride},
                    {trefs[i].buf, trefs[i].offset, trefs[i].size},
                };
                tp5_update_storage_set(r.vkdev, plan.late_norm_ds[idx], norm_infos, 9);
                VkDescriptorBufferInfo lo_infos[4] = {
                    {r.bcast_buf[b], 0, (VkDeviceSize) c.star_rank_stride},
                    {plan.late_rho_buf[i], 0, max_rows * 4 * sizeof(float)},
                    {late.lo.buffer, late.lo.offset, late.lo.size},
                    {r.host_import_buf[b], 0, (VkDeviceSize) c.star_rank_stride},
                };
                if (plan.late_q8_fast) {
                    const VkDeviceSize lo_q8_bytes =
                        (VkDeviceSize) ggml_row_size(GGML_TYPE_Q8_0, max_rows * late.late_rank);
                    const uint64_t weight_blocks =
                        uint64_t(late.late_rank) * late.streams * late.width / 32u;
                    const VkDeviceSize packed_bytes =
                        VkDeviceSize(weight_blocks + 1u) * (sizeof(float) + 8u * sizeof(uint32_t));
                    VkDescriptorBufferInfo lo_q8_infos[5] = {
                        lo_infos[0], lo_infos[1], lo_infos[2], lo_infos[3],
                        {plan.late_lo_q8_buf[i], 0, lo_q8_bytes},
                    };
                    tp5_update_storage_set(r.vkdev, plan.late_lo_q8_ds[idx], lo_q8_infos, 5);
                    VkDescriptorBufferInfo up_q8_infos[5] = {
                        {plan.late_up_packed_buf[i], 0, packed_bytes},
                        {plan.late_lo_q8_buf[i], 0, lo_q8_bytes},
                        {late.bindings[3].buffer, late.bindings[3].offset, late.bindings[3].size},
                        {late.mixed.buffer, late.mixed.offset, late.mixed.size},
                        {r.bcast_buf[b], 0, (VkDeviceSize) c.star_rank_stride},
                    };
                    tp5_update_storage_set(r.vkdev, plan.late_up_q8dot_ds[idx], up_q8_infos, 5);
                } else {
                    tp5_update_storage_set(r.vkdev, plan.late_lo_ds[idx], lo_infos, 4);
                }

                VkCommandBuffer cmd = plan.cmd_late_pre[idx];
                vk_tp5_capture_scope pre_scope(&plan.pre_definitions[idx]);
                vk_tp5_register_source(cmd);
                VkCommandBufferUsageFlags cb_flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
                VkCommandBufferBeginInfo beg_late{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, cb_flags,
                                                   nullptr};
                if (vkBeginCommandBuffer(cmd, &beg_late) != VK_SUCCESS) {
                    c.fail("begin LateBind precompute failed on rank " + std::to_string(i));
                    return false;
                }
                VkMemoryBarrier mb_in{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                      VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                                      VK_ACCESS_SHADER_READ_BIT};
                tp5_cmd_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb_in, 0, nullptr, 0, nullptr);

                // P1-B: one-shot weight repack (W_down then W_up) into the
                // uint32-word layout consumed by Q8DOT/UP_Q8DOT. Recorded at
                // the head of every cmd_late_pre replay; after the first
                // execution the persistent done flag makes each replay a
                // single guard-read no-op. Weights are immutable model data,
                // so the pack result stays valid for the plan's lifetime.
                if (plan.late_q8_fast && r.late_pack_pipe != VK_NULL_HANDLE &&
                    plan.late_pack_ds[i] != VK_NULL_HANDLE) {
                    const uint64_t weight_blocks =
                        uint64_t(late.late_rank) * late.streams * late.width / 32u;
                    const VkDeviceSize packed_bytes =
                        VkDeviceSize(weight_blocks + 1u) * (sizeof(float) + 8u * sizeof(uint32_t));
                    for (int which = 0; which < 2; ++which) {
                        const VkBuffer src_buf  = which == 0 ? late.down_weight.buffer : late.up_weight.buffer;
                        const VkDeviceSize src_off =
                            which == 0 ? late.down_weight.offset : late.up_weight.offset;
                        const VkBuffer dst_buf  =
                            which == 0 ? plan.late_down_packed_buf[i] : plan.late_up_packed_buf[i];
                        // Two pack descriptor sets: [0]=down, [1]=up.
                        VkDescriptorBufferInfo pack_infos[2] = {
                            {src_buf, src_off, (VkDeviceSize) ggml_row_size(
                                 GGML_TYPE_Q8_0, weight_blocks * 32u)},
                            {dst_buf, 0, packed_bytes},
                        };
                        tp5_update_storage_set(r.vkdev, plan.late_pack_ds[i + which * c.n_ranks],
                                               pack_infos, 2);
                        tp5_cmd_bind_pipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_pack_pipe);
                        tp5_cmd_bind_descriptors(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_pack_layout, 0, 1,
                                                &plan.late_pack_ds[i + which * c.n_ranks], 0, nullptr);
                        struct { uint32_t blocks; } pack_pc{uint32_t(weight_blocks)};
                        static_assert(sizeof(pack_pc) == 4);
                        tp5_cmd_push(cmd, r.late_pack_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                     sizeof(pack_pc), &pack_pc);
                        tp5_cmd_dispatch(cmd, uint32_t((weight_blocks + 255u) / 256u), 1, 1);
                        VkMemoryBarrier mb_pack{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
                        tp5_cmd_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb_pack,
                                             0, nullptr, 0, nullptr);
                    }
                }

                tp5_cmd_bind_pipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_inject_pipe);
                tp5_cmd_bind_descriptors(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_inject_layout, 0, 1,
                                        &plan.late_inject_ds[idx], 0, nullptr);
                struct { uint32_t width, streams, active_rows; } inject_pc{
                    late.width, late.streams, late.capacity_rows ? late.capacity_rows : 1u};
                static_assert(sizeof(inject_pc) == 12);
                tp5_cmd_push(cmd, r.late_inject_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             sizeof(inject_pc), &inject_pc);
                tp5_cmd_dispatch(cmd, late.streams, 1, 1);
                // The suffix barrier belongs to Q, not to the hoisted
                // scatter. It must remain after the terminal z_p producer.
                plan.late_inject_end[idx] = plan.pre_definitions[idx].code.size();
                VkMemoryBarrier mb_scatter{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                                            VK_ACCESS_SHADER_READ_BIT};
                tp5_cmd_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb_scatter,
                                     0, nullptr, 0, nullptr);
                // Semantic split used by the linear definition: everything
                // before this point prepares scatter and its visibility. The
                // expensive Q sufficient-statistic work may then overlap the
                // already-resident Y consumer instead of serializing in front
                // of it.
                plan.late_q_begin[idx] = plan.pre_definitions[idx].code.size();

                if (plan.late_q8_fast) {
                    tp5_cmd_bind_pipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_act_q8_pipe);
                    tp5_cmd_bind_descriptors(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_act_q8_layout, 0, 1,
                                            &plan.late_act_q8_ds[idx], 0, nullptr);
                    struct { uint32_t width, streams, my_rank, active_rows; } act_pc{
                        late.width, late.streams, (uint32_t) i, late.capacity_rows ? late.capacity_rows : 1u};
                    static_assert(sizeof(act_pc) == 16);
                    tp5_cmd_push(cmd, r.late_act_q8_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                 sizeof(act_pc), &act_pc);
                    tp5_cmd_dispatch(cmd, late.width / 64u, 1, 1);

                    // The visibility barrier must cover every row the shader
                    // can write (capacity rows; word 5 only shrinks the loop),
                    // matching the max_rows-sized allocation. A single-row
                    // range left tokens 2..4 unordered w.r.t. Q8DOT.
                    const VkDeviceSize act_q8_bytes = (VkDeviceSize) ggml_row_size(
                        GGML_TYPE_Q8_0,
                        size_t(late.capacity_rows ? late.capacity_rows : 1u) * late.streams * late.width);
                    VkBufferMemoryBarrier act_ready{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
                                                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                    plan.late_act_q8_buf[i], 0, act_q8_bytes};
                    tp5_cmd_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         0, 0, nullptr, 1, &act_ready, 0, nullptr);
                    plan.late_q_contract_begin[idx] = plan.pre_definitions[idx].code.size();

                    tp5_cmd_bind_pipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_q8dot_pipe);
                    tp5_cmd_bind_descriptors(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_q8dot_layout, 0, 1,
                                            &plan.late_q8dot_ds[idx], 0, nullptr);
                    const uint32_t q8_workgroups = (late.late_rank + 15u) / 16u;
                    struct {
                        uint32_t width, rank_dim, streams, rows_per_wg, n_workgroups, late_word_offset, active_rows;
                    } q8_pc{late.width, late.late_rank, late.streams, 16u, q8_workgroups,
                            tp5_late_q_payload_word_offset(c.late_host_offset, true),
                            late.capacity_rows ? late.capacity_rows : 1u};
                    static_assert(sizeof(q8_pc) == 28);
                    tp5_cmd_push(cmd, r.late_q8dot_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                 sizeof(q8_pc), &q8_pc);
                    tp5_cmd_dispatch(cmd, q8_workgroups, 1, 1);
                } else {
                    plan.late_q_contract_begin[idx] = plan.late_q_begin[idx];
                    tp5_cmd_bind_pipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_q_pipe);
                    tp5_cmd_bind_descriptors(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_q_layout, 0, 1,
                                            &plan.late_q_ds[idx], 0, nullptr);
                    struct { uint32_t width, rank_dim, streams, my_rank, active_rows; } q_pc{
                        late.width, late.late_rank, late.streams, (uint32_t) i,
                        late.capacity_rows ? late.capacity_rows : 1u};
                    static_assert(sizeof(q_pc) == 20);
                    tp5_cmd_push(cmd, r.late_q_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(q_pc), &q_pc);
                    tp5_cmd_dispatch(cmd, late.late_rank, 1, 1);
                    // Exact-Q fallback keeps the compact publisher dispatch.
                    VkBufferMemoryBarrier mb_q_copy{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
                                                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                    plan.late_sidecar_buf[i], 0,
                                                    (VkDeviceSize) (late_count * sizeof(float))};
                    tp5_cmd_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         0, 0, nullptr, 1, &mb_q_copy, 0, nullptr);
                    const uint32_t late_uvec4 = (uint32_t) ((late_count * sizeof(float)) / 16u);
                    if (late_count % 4u != 0u || late_uvec4 == 0u) {
                        c.fail("LateBind sidecar is not 128-bit copy aligned on rank " + std::to_string(i));
                        return false;
                    }
                    tp5_cmd_bind_pipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_publish_pipe);
                    tp5_cmd_bind_descriptors(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_publish_layout, 0, 1,
                                            &plan.late_publish_ds[idx], 0, nullptr);
                    tp5_cmd_push(cmd, r.late_publish_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(late_uvec4), &late_uvec4);
                    tp5_cmd_dispatch(cmd, 1, 1, 1);
                }

                if (!plan.late_q8_fast) {
                    // Exact fallback publishes from imported system RAM and
                    // therefore keeps the established Vulkan HOST-domain
                    // barrier. Aggressive Q8 writes cached device-coherent
                    // bcast VRAM instead; its payload/ready protocol is
                    // shader-ordered and CPU-visible without a queue-wide
                    // COMPUTE->HOST execution dependency.
                    VkBufferMemoryBarrier host_ranges[2]{};
                    for (auto & barrier : host_ranges) {
                        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                        barrier.buffer = r.host_import_buf[b];
                    }
                    host_ranges[0].offset = (VkDeviceSize) c.late_host_offset;
                    host_ranges[0].size   = (VkDeviceSize) (late_count * sizeof(float));
                    host_ranges[1].offset = (VkDeviceSize) c.star_rank_stride - 64;
                    host_ranges[1].size   = 64;
                    tp5_cmd_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_HOST_BIT,
                                         0, 0, nullptr, 2, host_ranges, 0, nullptr);
                }
                if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
                    c.fail("end LateBind precompute failed on rank " + std::to_string(i));
                    return false;
                }
            }
        }
    }

    // Record P2 once per (rank, mailbox bank). SIMULTANEOUS_USE allows pipelined resubmit.
    VkCommandBufferUsageFlags cb_flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    VkCommandBufferBeginInfo  beg{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, cb_flags, nullptr };
    for (size_t i = 0; i < c.n_ranks; ++i) {
        tp5_rank & r = c.ranks[i];
        for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
            const size_t    idx = tp5_plan_slot(i, b);
            VkCommandBuffer cmd = plan.cmd_p2[idx];
            vk_tp5_capture_scope p2_scope(define_inline ? &plan.p2_definitions[idx] : nullptr);
            vk_tp5_register_source(cmd);
            if (vkBeginCommandBuffer(cmd, &beg) != VK_SUCCESS) {
                c.fail("begin cmd_p2 failed on rank " + std::to_string(i));
                return false;
            }

            if (c.sync_mode == tp5_sync_mode::STAR || c.sync_mode == tp5_sync_mode::RELAY) {
                const size_t bcast_bytes = tensor_bytes; // CPU accumulation produces F32, regardless of wire type.
                if (c.sync_mode == tp5_sync_mode::RELAY) {
                    const uint32_t profile_spin = getenv("GGML_TP5_PROFILE") ? 1u : 0u;
                    const auto record_relay_copy = [&]() {
                        VkDescriptorBufferInfo src_info{r.bcast_buf[b], 0, 64 + tensor_bytes};
                        VkDescriptorBufferInfo dst_info{trefs[i].buf, trefs[i].offset, tensor_bytes};
                        VkDescriptorBufferInfo status_info{r.host_import_buf[b], c.star_rank_stride - 64, 64};
                        VkWriteDescriptorSet writes[3]{};
                        for (uint32_t w = 0; w < 3; ++w) {
                            writes[w].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                            writes[w].dstSet = plan.relay_ds[idx];
                            writes[w].dstBinding = w;
                            writes[w].descriptorCount = 1;
                            writes[w].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        }
                        writes[0].pBufferInfo = &src_info;
                        writes[1].pBufferInfo = &dst_info;
                        writes[2].pBufferInfo = &status_info;
                        vkUpdateDescriptorSets(r.vkdev, 3, writes, 0, nullptr);
                        tp5_cmd_bind_pipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.relay_copy_pipe);
                        tp5_cmd_bind_descriptors(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.relay_copy_layout, 0, 1,
                                                &plan.relay_ds[idx], 0, nullptr);
                        struct { uint32_t n_elems; uint32_t spin_max; uint32_t dst_offset_words;
                                 uint32_t profile_spin; } relay_pc{
                            (uint32_t)n_elems, c.spin_max, 0u, profile_spin};
                        static_assert(sizeof(relay_pc) == 16);
                        tp5_cmd_push(cmd, r.relay_copy_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                           sizeof(relay_pc), &relay_pc);
                        tp5_cmd_dispatch(cmd, 1, 1, 1);
                    };
                    if (late_plan) {
                        const auto & late = trefs[i].late;
                        // Each consumer waits only for the data it uses.
                        // Norm consumes y_epoch; LO consumes q_epoch after
                        // norm has completed. No standalone P2 wait/copy.
                        const uint32_t status_offset = (uint32_t) ((c.star_rank_stride - 64) / 4);
                        tp5_cmd_bind_pipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_norm_pipe);
                        tp5_cmd_bind_descriptors(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_norm_layout, 0, 1,
                                                &plan.late_norm_ds[idx], 0, nullptr);
                        struct {
                            uint32_t width, streams;
                            float epsilon;
                            uint32_t spin_max, status_word_offset, profile_spin, active_rows;
                        } norm_pc{late.width, late.streams, late.epsilon, c.spin_max, status_offset, profile_spin,
                                  late.capacity_rows ? late.capacity_rows : 1u};
                        static_assert(sizeof(norm_pc) == 28);
                        tp5_cmd_push(cmd, r.late_norm_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                     sizeof(norm_pc), &norm_pc);
                        tp5_cmd_dispatch(cmd, late.streams, 1, 1);
                        // End the Y consumer prefix immediately after the norm
                        // dispatch. The rho->LO dependency barrier belongs to
                        // the LO suffix; keeping it out of this prefix allows
                        // Q to issue while the four norm workgroups are waiting
                        // for / consuming Y.
                        plan.late_norm_end[idx] = plan.p2_definitions[idx].code.size();
                        VkMemoryBarrier mb_norm_lo{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                                   VK_ACCESS_SHADER_WRITE_BIT,
                                                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
                        tp5_cmd_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                             1, &mb_norm_lo, 0, nullptr, 0, nullptr);
                        plan.late_lo_begin[idx] = plan.p2_definitions[idx].code.size();

                        struct {
                            uint32_t rank_dim, streams, late_word_offset, spin_max,
                                     status_word_offset, profile_spin, sidecar_f16, active_rows;
                        }
                            lo_pc{late.late_rank, late.streams,
                                  tp5_late_q_payload_word_offset(c.late_host_offset, plan.late_q8_fast),
                                  c.spin_max, status_offset, profile_spin,
                                  plan.late_sidecar_f16 ? 1u : 0u,
                                  late.capacity_rows ? late.capacity_rows : 1u};
                        static_assert(sizeof(lo_pc) == 32);
                        if (plan.late_q8_fast) {
                            tp5_cmd_bind_pipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_lo_q8_pipe);
                            tp5_cmd_bind_descriptors(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_lo_q8_layout, 0, 1,
                                                    &plan.late_lo_q8_ds[idx], 0, nullptr);
                            tp5_cmd_push(cmd, r.late_lo_q8_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                         sizeof(lo_pc), &lo_pc);
                            tp5_cmd_dispatch(cmd, 1, 1, 1);

                            // Same capacity-coverage rule as the ACT_Q8 barrier:
                            // the buffer is allocated for max_rows tokens and the
                            // shader writes token < capacity_rows; the barrier
                            // must order all of them for the UP_Q8DOT reader.
                            const VkDeviceSize lo_q8_bytes = (VkDeviceSize) ggml_row_size(
                                GGML_TYPE_Q8_0,
                                size_t(late.capacity_rows ? late.capacity_rows : 1u) * late.late_rank);
                            VkBufferMemoryBarrier lo_q8_ready{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
                                                             VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                                             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                             plan.late_lo_q8_buf[i], 0, lo_q8_bytes};
                            tp5_cmd_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                                 0, 0, nullptr, 1, &lo_q8_ready, 0, nullptr);

                            tp5_cmd_bind_pipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_up_q8dot_pipe);
                            tp5_cmd_bind_descriptors(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_up_q8dot_layout,
                                                    0, 1, &plan.late_up_q8dot_ds[idx], 0, nullptr);
                            struct { uint32_t width, rank_dim, streams, rows_per_wg, active_rows; } up_pc{
                                late.width, late.late_rank, late.streams, 24u,
                                late.capacity_rows ? late.capacity_rows : 1u};
                            static_assert(sizeof(up_pc) == 20);
                            tp5_cmd_push(cmd, r.late_up_q8dot_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                         sizeof(up_pc), &up_pc);
                            tp5_cmd_dispatch(cmd, (late.width + 23u) / 24u, 1, 1);
                        } else {
                            tp5_cmd_bind_pipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_lo_pipe);
                            tp5_cmd_bind_descriptors(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.late_lo_layout, 0, 1,
                                                    &plan.late_lo_ds[idx], 0, nullptr);
                            tp5_cmd_push(cmd, r.late_lo_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                         sizeof(lo_pc), &lo_pc);
                            tp5_cmd_dispatch(cmd, 1, 1, 1);
                        }
                    } else if (trefs[i].hc.width) {
                        // Fast RELAY path: the downstream HC consumer itself
                        // owns the bounded doorbell wait. Once the generation
                        // arrives it continues directly with inject/RMS/norm,
                        // so communication does not terminate one dispatch
                        // merely to launch the next compute dispatch.
                        const size_t pipe_idx = (trefs[i].hc.quantized.buffer != nullptr) ? 1 : 0;
                        tp5_update_hc_descriptor(r, plan.ds_sum[idx], trefs[i], b, stride, payload, true);
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.hc_sum_pipe[pipe_idx]);
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.hc_sum_layout[pipe_idx], 0, 1,
                                                &plan.ds_sum[idx], 0, nullptr);
                        struct {
                            uint32_t width;
                            float    epsilon;
                            uint32_t spin_max;
                            uint32_t reserved;
                        } relay_hc_pc{trefs[i].hc.width, trefs[i].hc.epsilon, c.spin_max, profile_spin};
                        static_assert(sizeof(relay_hc_pc) == 16);
                        vkCmdPushConstants(cmd, r.hc_sum_layout[pipe_idx], VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                           sizeof(relay_hc_pc), &relay_hc_pc);
                        vkCmdDispatch(cmd, 2, 1, 1);
                    } else {
                        // Generic fallback for the few collective boundaries
                        // whose first consumer is not an HC prefix.
                        record_relay_copy();
                    }
                    // Only order P2 outputs into subsequent GPU work. Host
                    // visibility of status[3] is supplied transitively by the
                    // next P1's COMPUTE->HOST publish barrier before its flag
                    // is observed, so paying a HOST scope here on every stage
                    // would serialize the queue for no additional credit.
                    VkMemoryBarrier mb_post{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                            VK_ACCESS_SHADER_WRITE_BIT,
                                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
                    tp5_cmd_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         0, 1, &mb_post, 0, nullptr, 0, nullptr);
                } else {
                    VkBufferCopy cp{ 0, trefs[i].offset, bcast_bytes };
                    vkCmdCopyBuffer(cmd, r.bcast_buf[b], trefs[i].buf, 1, &cp);
                    VkMemoryBarrier mb_post{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                             VK_ACCESS_TRANSFER_WRITE_BIT,
                                             VK_ACCESS_SHADER_READ_BIT };
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb_post, 0, nullptr, 0, nullptr);
                }
                if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
                    c.fail("end cmd_p2 star failed on rank " + std::to_string(i));
                    return false;
                }
                continue;
            }
            if (!gpuflag) {
                const bool      hc = trefs[i].hc.width != 0;
                VkMemoryBarrier mb_p2_in{
                    VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                    VkAccessFlags(VK_ACCESS_TRANSFER_WRITE_BIT | (hc ? VK_ACCESS_SHADER_WRITE_BIT : 0)),
                    VkAccessFlags(VK_ACCESS_SHADER_READ_BIT | (hc ? VK_ACCESS_SHADER_WRITE_BIT : 0))
                };
                vkCmdPipelineBarrier(cmd,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT | (hc ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : 0),
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb_p2_in, 0, nullptr, 0, nullptr);
            }

            if (trefs[i].hc.width) {
                const size_t pipe_idx = (trefs[i].hc.quantized.buffer != nullptr) ? 1 : 0;
                tp5_update_hc_descriptor(r, plan.ds_sum[idx], trefs[i], b, stride, payload);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.hc_sum_pipe[pipe_idx]);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.hc_sum_layout[pipe_idx], 0, 1, &plan.ds_sum[idx],
                                        0, nullptr);

                const struct {
                    uint32_t width;
                    float    epsilon;
                } pc{ trefs[i].hc.width, trefs[i].hc.epsilon };

                vkCmdPushConstants(cmd, r.hc_sum_layout[pipe_idx], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, 2, 1, 1);
            } else {
                tp5_update_sum_descriptor(r, plan.ds_sum[idx], trefs[i].buf, trefs[i].offset, tensor_bytes,
                                          (uint32_t) c.n_ranks, b, stride, payload, flags_base, flags_bytes, gpuflag);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.sum_pipe);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipe_layout, 0, 1, &plan.ds_sum[idx], 0,
                                        nullptr);

                tp5_sum_pc pc{
                    (uint32_t) n_elems,
                    (uint32_t) c.n_ranks,
                    gpuflag ? 1u : 0u,
                    0u,
                };
                vkCmdPushConstants(cmd, r.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, (uint32_t) ((n_elems + 255) / 256), 1, 1);
            }

            if (gpuflag) {
                VkMemoryBarrier mb_sum{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                                        VK_ACCESS_SHADER_WRITE_BIT };
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                     1, &mb_sum, 0, nullptr, 0, nullptr);
                tp5_flag_pc done_pc = flag_pc_base;
                done_pc.mode        = 2u;
                tp5_record_flag_dispatch(cmd, r, plan.p1->ds_flag[i], done_pc);
            }

            VkMemoryBarrier mb_post{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                         VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT };
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb_post,
                                 0, nullptr, 0, nullptr);

            if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
                c.fail("end cmd_p2 failed on rank " + std::to_string(i));
                return false;
            }
        }
    }
    return true;
}

bool tp5_wait_all_p1(tp5_comm & c) {
    for (size_t i = 0; i < c.n_ranks; ++i) {
        tp5_rank & r = c.ranks[i];
        if (vkWaitForFences(r.vkdev, 1, &r.fence_p1, VK_TRUE, 5000000000ULL) != VK_SUCCESS) {
            c.fail("Phase 1 fence wait failed on rank " + std::to_string(i));
            return false;
        }
        c.host_waits++;
        if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
            prof->host_wait_count++;
        }
        vkResetFences(r.vkdev, 1, &r.fence_p1);
    }
    return true;
}

bool tp5_wait_all_p2(tp5_comm & c) {
    for (size_t i = 0; i < c.n_ranks; ++i) {
        tp5_rank & r = c.ranks[i];
        if (vkWaitForFences(r.vkdev, 1, &r.fence_p2, VK_TRUE, 5000000000ULL) != VK_SUCCESS) {
            c.fail("Phase 2 fence wait failed on rank " + std::to_string(i));
            return false;
        }
        if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
            prof->host_wait_count++;
        }
        vkResetFences(r.vkdev, 1, &r.fence_p2);
    }
    return true;
}

// Cross-GPU P2P writes complete on the producer queue; consumers need a global
// device idle before Phase 2 reads peer-filled mailbox slots (RADV PCIe mesh).
bool tp5_p2p_visibility_barrier(tp5_comm & c) {
    for (auto & r : c.ranks) {
        vkDeviceWaitIdle(r.vkdev);
    }
    return true;
}

static void tp5_note_timeline_submission(tp5_rank & r, uint64_t signal_value) {
    r.last_submitted_timeline_value = std::max(r.last_submitted_timeline_value, signal_value);
}

static bool tp5_wait_timeline_value(tp5_comm & c, size_t rank_index, uint64_t target_val, uint64_t timeout_ns) {
    if (target_val == 0) {
        return true;
    }
    tp5_rank & r = c.ranks[rank_index];
    if (r.timeline_sem == VK_NULL_HANDLE || r.pfn_wait_semaphores == nullptr) {
        c.fail("native timeline wait unavailable on rank " + std::to_string(rank_index));
        return false;
    }
    if (c.sync_mode == tp5_sync_mode::DRM && r.own_syncobj && r.dri_fd >= 0) {
        uint32_t first = 0;
        auto t_wait_start = std::chrono::high_resolution_clock::now();
        uint64_t target_val_mut = target_val;
        if (tp5_drm_syncobj_ops::timeline_wait(r.dri_fd, &r.own_syncobj, &target_val_mut, 1, (int64_t) timeout_ns,
                                               DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT, &first) == 0) {
            auto t_wait_end = std::chrono::high_resolution_clock::now();
            if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
                prof->host_wait_count++;
                prof->host_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(t_wait_end - t_wait_start).count();
            }
            return true;
        }
    }
    if (r.pfn_get_sem_counter) {
        uint64_t cur = 0;
        if (r.pfn_get_sem_counter(r.vkdev, r.timeline_sem, &cur) == VK_SUCCESS && cur >= target_val) {
            return true;
        }
    }
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &r.timeline_sem;
    wi.pValues = &target_val;
    auto t_wait_start = std::chrono::high_resolution_clock::now();
    if (r.pfn_wait_semaphores(r.vkdev, &wi, timeout_ns) != VK_SUCCESS) {
        c.fail("native timeline wait failed or timed out for value " + std::to_string(target_val) +
               " on rank " + std::to_string(rank_index));
        return false;
    }
    auto t_wait_end = std::chrono::high_resolution_clock::now();
    if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
        prof->host_wait_count++;
        prof->host_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(t_wait_end - t_wait_start).count();
    }
    return true;
}

bool tp5_drain_epoch(tp5_comm & c, uint64_t epoch, uint64_t timeout_ns) {
    if (epoch == 0 || epoch <= c.last_drained_epoch) return true;
    const uint64_t target_val = 2 * epoch;
    for (size_t i = 0; i < c.n_ranks; ++i) {
        if (!tp5_wait_timeline_value(c, i, target_val, timeout_ns)) {
            return false;
        }
    }
    c.last_drained_epoch = std::max(c.last_drained_epoch, epoch);
    // Retire completed buffer owners in the ring up to the drained epoch
    for (size_t s = 0; s < tp5_comm::MAX_OUTSTANDING_EPOCHS; ++s) {
        if (c.in_flight_ring[s].epoch > 0 && c.in_flight_ring[s].epoch <= c.last_drained_epoch) {
            c.in_flight_ring[s].owners.clear();
            c.in_flight_ring[s].p1.reset();
            c.in_flight_ring[s].epoch = 0;
        }
    }
    return true;
}

bool tp5_drain_submitted(tp5_comm & c, uint64_t timeout_ns) {
    for (size_t i = 0; i < c.n_ranks; ++i) {
        if (!tp5_wait_timeline_value(c, i, c.ranks[i].last_submitted_timeline_value, timeout_ns)) {
            return false;
        }
    }
    return true;
}

void tp5_update_epoch_bufs(tp5_comm & c, uint32_t seq) {
    for (size_t i = 0; i < c.n_ranks; ++i) {
        tp5_rank & r = c.ranks[i];
        if (!r.epoch_host) continue;
        r.epoch_host[0] = seq;
        r.epoch_host[1] = c.spin_max;
        r.epoch_host[2] = (uint32_t) i;
        r.epoch_host[3] = (uint32_t) c.n_ranks;
    }
}

static bool tp5_gpuflag_drain_epoch(tp5_comm & c, size_t ring_idx) {
    for (size_t i = 0; i < c.n_ranks; ++i) {
        tp5_rank & r = c.ranks[i];
        if (r.fence_ring[ring_idx] == VK_NULL_HANDLE) continue;
        if (vkWaitForFences(r.vkdev, 1, &r.fence_ring[ring_idx], VK_TRUE, 5000000000ULL) != VK_SUCCESS) {
            c.fail("gpuflag ring drain failed on rank " + std::to_string(i));
            return false;
        }
        vkResetFences(r.vkdev, 1, &r.fence_ring[ring_idx]);
    }
    return true;
}

static bool tp5_gpuflag_drain_all(tp5_comm & c) {
    if (c.allreduce_calls == 0) return true;
    for (size_t r = 0; r < tp5_comm::MAX_OUTSTANDING_EPOCHS; ++r) {
        if (!tp5_gpuflag_drain_epoch(c, r)) return false;
        c.in_flight_ring[r].owners.clear();
        c.in_flight_ring[r].p1.reset();
        c.in_flight_ring[r].epoch = 0;
    }
    return true;
}

// ===========================================================================
// Two-Stage Mesh AllReduce: Proven Hardware Synchronization
// ===========================================================================

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2,f16c")))
static size_t tp5_star_sum_avx2(const void * const * ranks, float * dst, size_t begin, size_t end) {
    size_t i = begin;
    for (; i + 8 <= end; i += 8) {
        __m256 sum = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) ((const uint16_t *) ranks[0] + i)));
        for (size_t r = 1; r < 5; ++r) {
            sum = _mm256_add_ps(sum, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) ((const uint16_t *) ranks[r] + i))));
        }
        _mm256_storeu_ps(dst + i, sum);
    }
    return i;
}

__attribute__((target("avx2,f16c")))
static size_t tp5_star_sum_broadcast_avx2(const void * const * ranks, float * const * dsts,
                                          size_t begin, size_t end) {
    size_t i = begin;
    for (; i + 8 <= end; i += 8) {
        __m256 sum = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) ((const uint16_t *) ranks[0] + i)));
        for (size_t r = 1; r < 5; ++r) {
            sum = _mm256_add_ps(sum, _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i *) ((const uint16_t *) ranks[r] + i))));
        }
        for (size_t r = 0; r < 5; ++r) {
            _mm256_storeu_ps(dsts[r] + i, sum);
        }
    }
    return i;
}

__attribute__((target("avx2,f16c")))
static size_t tp5_star_sum_broadcast_f16_avx2(const void * const * ranks, uint16_t * const * dsts,
                                              size_t begin, size_t end) {
    size_t i = begin;
    for (; i + 8 <= end; i += 8) {
        __m256 sum = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) ((const uint16_t *) ranks[0] + i)));
        for (size_t r = 1; r < 5; ++r) {
            sum = _mm256_add_ps(sum, _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i *) ((const uint16_t *) ranks[r] + i))));
        }
        const __m128i packed = _mm256_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT);
        for (size_t r = 0; r < 5; ++r) {
            _mm_storeu_si128((__m128i *) (dsts[r] + i), packed);
        }
    }
    return i;
}

__attribute__((target("avx2")))
static size_t tp5_star_sum_broadcast_f32_avx2(const void * const * ranks, float * const * dsts,
                                              size_t begin, size_t end) {
    size_t i = begin;
    for (; i + 8 <= end; i += 8) {
        __m256 sum = _mm256_loadu_ps((const float *) ranks[0] + i);
        for (size_t r = 1; r < 5; ++r) {
            sum = _mm256_add_ps(sum, _mm256_loadu_ps((const float *) ranks[r] + i));
        }
        for (size_t r = 0; r < 5; ++r) {
            _mm256_storeu_ps(dsts[r] + i, sum);
        }
    }
    return i;
}

__attribute__((target("avx2")))
static size_t tp5_star_sum_f32_avx2(const void * const * ranks, float * dst, size_t begin, size_t end) {
    size_t i = begin;
    for (; i + 8 <= end; i += 8) {
        __m256 sum = _mm256_loadu_ps((const float *) ranks[0] + i);
        for (size_t r = 1; r < 5; ++r) {
            sum = _mm256_add_ps(sum, _mm256_loadu_ps((const float *) ranks[r] + i));
        }
        _mm256_storeu_ps(dst + i, sum);
    }
    return i;
}
#endif

static void tp5_avx2_accumulate_star(const void * const * rank_ptrs, void * dst_ptr, size_t n_elems,
                                     size_t worker_id, size_t num_workers, bool out_f32) {
    // 22-core AVX2 accumulation over 5 GPU rank input buffers into dst_ptr
    // Each rank buffer has n_elems float16_t (uint16_t) elements.
    const size_t chunk_size = (n_elems + num_workers - 1) / num_workers;
    const size_t start = worker_id * chunk_size;
    const size_t end = std::min(start + chunk_size, n_elems);
    if (start >= end) return;

    const uint16_t * r0 = (const uint16_t *) rank_ptrs[0];
    const uint16_t * r1 = (const uint16_t *) rank_ptrs[1];
    const uint16_t * r2 = (const uint16_t *) rank_ptrs[2];
    const uint16_t * r3 = (const uint16_t *) rank_ptrs[3];
    const uint16_t * r4 = (const uint16_t *) rank_ptrs[4];
    uint16_t * dst16 = (uint16_t *) dst_ptr;
    float * dst32 = (float *) dst_ptr;

    size_t i = start;
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    if (out_f32 && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c")) {
        i = tp5_star_sum_avx2(rank_ptrs, dst32, i, end);
    }
#endif
#if defined(__AVX2__) && defined(__F16C__)
    for (; i + 8 <= end; i += 8) {
        __m256 v0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(r0 + i)));
        __m256 v1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(r1 + i)));
        __m256 v2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(r2 + i)));
        __m256 v3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(r3 + i)));
        __m256 v4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(r4 + i)));

        __m256 sum = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(v0, v1), v2), v3);
        sum = _mm256_add_ps(sum, v4);

        if (out_f32) {
            _mm256_storeu_ps(dst32 + i, sum);
        } else {
            __m128i out_f16 = _mm256_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(dst16 + i), out_f16);
        }
    }
#endif
    for (; i < end; ++i) {
        float s = ggml_fp16_to_fp32(r0[i]) + ggml_fp16_to_fp32(r1[i]) +
                  ggml_fp16_to_fp32(r2[i]) + ggml_fp16_to_fp32(r3[i]) +
                  ggml_fp16_to_fp32(r4[i]);
        if (out_f32) {
            dst32[i] = s;
        } else {
            dst16[i] = ggml_fp32_to_fp16(s);
        }
    }
}

static void tp5_accumulate_broadcast_star(const void * const * rank_ptrs, float * const * dsts, size_t n_elems) {
    const uint16_t * r0 = (const uint16_t *) rank_ptrs[0];
    const uint16_t * r1 = (const uint16_t *) rank_ptrs[1];
    const uint16_t * r2 = (const uint16_t *) rank_ptrs[2];
    const uint16_t * r3 = (const uint16_t *) rank_ptrs[3];
    const uint16_t * r4 = (const uint16_t *) rank_ptrs[4];
    size_t i = 0;
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c")) {
        i = tp5_star_sum_broadcast_avx2(rank_ptrs, dsts, 0, n_elems);
    }
#endif
    for (; i < n_elems; ++i) {
        const float sum = ggml_fp16_to_fp32(r0[i]) + ggml_fp16_to_fp32(r1[i]) +
                          ggml_fp16_to_fp32(r2[i]) + ggml_fp16_to_fp32(r3[i]) +
                          ggml_fp16_to_fp32(r4[i]);
        for (size_t r = 0; r < 5; ++r) {
            dsts[r][i] = sum;
        }
    }
}

static void tp5_accumulate_broadcast_star_f16(const void * const * rank_ptrs, uint16_t * const * dsts,
                                              size_t n_elems) {
    const uint16_t * r0 = (const uint16_t *) rank_ptrs[0];
    const uint16_t * r1 = (const uint16_t *) rank_ptrs[1];
    const uint16_t * r2 = (const uint16_t *) rank_ptrs[2];
    const uint16_t * r3 = (const uint16_t *) rank_ptrs[3];
    const uint16_t * r4 = (const uint16_t *) rank_ptrs[4];
    size_t i = 0;
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c")) {
        i = tp5_star_sum_broadcast_f16_avx2(rank_ptrs, dsts, 0, n_elems);
    }
#endif
    for (; i < n_elems; ++i) {
        const float sum = ggml_fp16_to_fp32(r0[i]) + ggml_fp16_to_fp32(r1[i]) +
                          ggml_fp16_to_fp32(r2[i]) + ggml_fp16_to_fp32(r3[i]) +
                          ggml_fp16_to_fp32(r4[i]);
        const uint16_t h = ggml_fp32_to_fp16(sum);
        for (size_t r = 0; r < 5; ++r) dsts[r][i] = h;
    }
}

static void tp5_accumulate_broadcast_star_f32(const void * const * rank_ptrs, float * const * dsts, size_t n_elems) {
    const float * r0 = (const float *) rank_ptrs[0];
    const float * r1 = (const float *) rank_ptrs[1];
    const float * r2 = (const float *) rank_ptrs[2];
    const float * r3 = (const float *) rank_ptrs[3];
    const float * r4 = (const float *) rank_ptrs[4];
    size_t i = 0;
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    if (__builtin_cpu_supports("avx2")) {
        i = tp5_star_sum_broadcast_f32_avx2(rank_ptrs, dsts, 0, n_elems);
    }
#endif
    for (; i < n_elems; ++i) {
        const float sum = r0[i] + r1[i] + r2[i] + r3[i] + r4[i];
        for (size_t r = 0; r < 5; ++r) {
            dsts[r][i] = sum;
        }
    }
}

static void tp5_accumulate_star_f32(const void * const * rank_ptrs, float * dst, size_t n_elems,
                                    size_t worker_id, size_t num_workers) {
    const size_t chunk_size = (n_elems + num_workers - 1) / num_workers;
    const size_t start = worker_id * chunk_size;
    const size_t end   = std::min(start + chunk_size, n_elems);
    if (start >= end) return;
    const float * r0 = (const float *) rank_ptrs[0];
    const float * r1 = (const float *) rank_ptrs[1];
    const float * r2 = (const float *) rank_ptrs[2];
    const float * r3 = (const float *) rank_ptrs[3];
    const float * r4 = (const float *) rank_ptrs[4];
    size_t i = start;
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    if (__builtin_cpu_supports("avx2")) {
        i = tp5_star_sum_f32_avx2(rank_ptrs, dst, i, end);
    }
#endif
    for (; i < end; ++i) {
        dst[i] = r0[i] + r1[i] + r2[i] + r3[i] + r4[i];
    }
}

struct tp5_star_times {
    double wait_us = 0;
    double sum_us = 0;
    double broadcast_us = 0;
    double first_ready_us = 0;
    double ready_skew_us = 0;
    double arm_us = 0;
    double cpu_data_us = 0;
    double sidecar_wait_us = 0;
    double sidecar_data_us = 0;
    double y_publish_us = 0;
    double q_publish_us = 0;
    double generation_us = 0;
    double total_us = 0;
    uint64_t poll_iters = 0;
    uint32_t retired_epoch = 0;
    std::array<double, 8> rank_ready_us{};
    std::array<uint32_t, 8> retired_spin{};
    std::array<uint32_t, 8> retired_q_spin{};
};

static void tp5_profile_atomic_max(std::atomic<uint64_t> & dst, uint64_t value) {
    uint64_t current = dst.load(std::memory_order_relaxed);
    while (current < value &&
           !dst.compare_exchange_weak(current, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
    }
}

// Host failure must release every already queued P2 without waiting for its
// full spin budget. The shader polls status[2] and turns its destination into
// NaNs; real timeline values still prove completion before destruction.
static void tp5_relay_request_abort(tp5_comm & c) {
    for (size_t bank = 0; bank < TP5_MAILBOX_BANKS; ++bank) {
        for (size_t i = 0; i < c.n_ranks; ++i) {
            auto * status = (volatile uint32_t *) ((char *) c.star_host_aligned[bank] +
                                                   i * c.star_rank_stride + c.star_rank_stride - 64);
            status[2] = status[1] == 0u ? UINT32_MAX : status[1];
            status[4] = UINT32_MAX;
        }
    }
    std::atomic_thread_fence(std::memory_order_release);
#if defined(__x86_64__) || defined(_M_X64)
    _mm_sfence();
#endif
}

// Arm one bank for its next generation without a host wait. For epoch e > 2,
// the observed producer-ready state (or legacy P1-ready state) for e-1 is
// published after P2(e-2) on the same queue, so the following handoff has
// transitive ownership credit for reusing bank(e). Any broken ordering fails
// closed.
static bool tp5_relay_arm_bank(tp5_comm & c, uint64_t epoch, uint32_t late_rows = 0) {
    if (epoch == 0) {
        c.fail("RELAY cannot arm generation zero");
        return false;
    }
    const size_t   bank              = tp5_mailbox_bank(epoch);
    const uint32_t previous_expected = epoch > TP5_MAILBOX_BANKS ? (uint32_t) (epoch - TP5_MAILBOX_BANKS) : 0u;
    std::atomic_thread_fence(std::memory_order_acquire);
    for (size_t i = 0; i < c.n_ranks; ++i) {
        auto * status = (volatile uint32_t *) ((char *) c.star_host_aligned[bank] + i * c.star_rank_stride +
                                               c.star_rank_stride - 64);
        if (status[2] != 0u) {
            c.fail("RELAY previous GPU wait failed on rank " + std::to_string(i));
            return false;
        }
        if (status[1] != previous_expected) {
            c.fail("RELAY bank generation mismatch on rank " + std::to_string(i) +
                   " (expected=" + std::to_string(previous_expected) +
                   ", found=" + std::to_string(status[1]) + ")");
            return false;
        }
        if (previous_expected != 0u && status[3] != previous_expected) {
            c.fail("RELAY transitive bank credit missing on rank " + std::to_string(i) +
                   " (epoch=" + std::to_string(epoch) + ")");
            return false;
        }
        if (previous_expected == 0u && status[3] != 0u) {
            c.fail("RELAY stale completion on rank " + std::to_string(i));
            return false;
        }
        status[0] = 0u;
        status[1] = (uint32_t) epoch;
        status[2] = 0u;
        status[3] = 0u;
        status[4] = 0u;
        status[5] = 0u;
        status[6] = 0u; // LateBind sidecar ready
        status[7] = 0u; // Q point-of-use spin samples, separate from Y
        // Aggressive Q8 sidecar lives in this rank's device-coherent bcast
        // VRAM. Reset its local ready/counter in the unified Q control region
        // (B + 64 + L) while the bank has transitive reuse credit, before the
        // next primary stream can touch it.
        if (c.ranks[i].bcast_host[bank] && c.late_host_offset) {
            auto * qctrl = tp5_late_q_control_ptr(c.ranks[i].bcast_host[bank], c.late_host_offset);
            qctrl[TP5_LATE_Q_READY_WORD]   = 0u; // Q sidecar ready
            qctrl[TP5_LATE_Q_COUNTER_WORD] = 0u; // Q8 workgroup completion counter
        }
        // Publish the runtime active-row count (RELAY header word 5) while the
        // bank provably has no in-flight reader: the credit checks above prove
        // the previous epoch's late kernels completed, and the next consumer
        // of this bank cannot pass its generation wait until a later handoff
        // publishes it. LateBind kernels bound their token loops with this
        // word so a maximum-capacity definition executes only useful rows.
        if (c.ranks[i].bcast_host[bank]) {
            ((volatile uint32_t *) c.ranks[i].bcast_host[bank])[5] = late_rows;
        }
    }
    c.relay_bank_used[bank] = false;
    std::atomic_thread_fence(std::memory_order_release);
#if defined(__x86_64__) || defined(_M_X64)
    _mm_sfence();
#endif
    return true;
}

// The previous chain pre-arms its successor from the final P1 flag. Accept
// that ready state without a host wait; otherwise arm from already completed
// predecessor credit. Mixed generations are a protocol failure, never a wait.
static bool tp5_relay_ensure_armed_epoch(tp5_comm & c, uint64_t epoch, uint32_t late_rows = 0) {
    if (epoch == 0) {
        c.fail("RELAY cannot ensure generation zero");
        return false;
    }
    const size_t bank = tp5_mailbox_bank(epoch);
    const uint32_t expected = (uint32_t) epoch;
    bool armed = false;
    bool unarmed = false;
    std::atomic_thread_fence(std::memory_order_acquire);
    for (size_t i = 0; i < c.n_ranks; ++i) {
        const auto * status = (const volatile uint32_t *) ((const char *) c.star_host_aligned[bank] +
                                                            i * c.star_rank_stride + c.star_rank_stride - 64);
        if (status[1] == expected) {
            armed = true;
            const auto * qctrl = tp5_late_q_control_cptr(c.ranks[i].bcast_host[bank], c.late_host_offset);
            if (status[0] != 0u || status[2] != 0u || status[3] != 0u ||
                status[6] != 0u || (qctrl && (qctrl[TP5_LATE_Q_READY_WORD] != 0u || qctrl[TP5_LATE_Q_COUNTER_WORD] != 0u))) {
                c.fail("RELAY pre-armed bank is not idle on rank " + std::to_string(i) +
                       " (epoch=" + std::to_string(epoch) + ", flag=" + std::to_string(status[0]) +
                       ", error=" + std::to_string(status[2]) + ", done=" + std::to_string(status[3]) +
                       ", late_ready=" + std::to_string(status[6]) +
                       ", q8_ready=" + std::to_string(qctrl ? qctrl[TP5_LATE_Q_READY_WORD] : UINT32_MAX) +
                       ", q8_count=" + std::to_string(qctrl ? qctrl[TP5_LATE_Q_COUNTER_WORD] : UINT32_MAX) + ")");
                return false;
            }
        } else {
            unarmed = true;
        }
    }
    if (armed && unarmed) {
        c.fail("RELAY bank generation differs across ranks for epoch " + std::to_string(epoch));
        return false;
    }
    return armed || tp5_relay_arm_bank(c, epoch, late_rows);
}

static bool tp5_relay_direct_stage(const tp5_plan_key & key, size_t n_ranks) {
    if (n_ranks == 0 || n_ranks > key.relay_direct.size())
        return false;
    for (size_t i = 0; i < n_ranks; ++i) {
        if (!key.relay_direct[i])
            return false;
    }
    return true;
}

static bool tp5_late_stage(const tp5_plan_key & key, size_t n_ranks) {
    if (!tp5_latebind_hc_enabled() || n_ranks == 0 || n_ranks > key.late.size())
        return false;
    uint32_t streams = 0;
    uint32_t rank_dim = 0;
    uint32_t capacity_rows = 0;
    for (size_t i = 0; i < n_ranks; ++i) {
        if (key.late[i].late_rank == 0 || key.late[i].streams != 4)
            return false;
        if (i == 0) {
            streams       = key.late[i].streams;
            rank_dim      = key.late[i].late_rank;
            capacity_rows = key.late[i].capacity_rows ? key.late[i].capacity_rows : 1u;
        } else if (key.late[i].streams != streams || key.late[i].late_rank != rank_dim ||
                   (key.late[i].capacity_rows ? key.late[i].capacity_rows : 1u) != capacity_rows) {
            return false;
        }
    }
    return size_t(capacity_rows) * streams * rank_dim <= TP5_LATE_MAX_FLOATS;
}

static bool tp5_relay_publish_generation(tp5_comm & c, size_t bank, uint32_t word) {
    // word 0 is Y, word 2 is Q. Both are data readiness, never CPU queue
    // scheduling. Preserve the established payload -> fence -> epoch order.
    if (c.n_ranks != 5 || bank >= TP5_MAILBOX_BANKS || (word != 0 && word != 2)) {
        c.fail("invalid RELAY publication route"); return false;
    }
    uint32_t epochs[8] = {};
    for (size_t i = 0; i < c.n_ranks; ++i) {
        const auto * status = (const volatile uint32_t *) ((const char *) c.star_host_aligned[bank] +
            i * c.star_rank_stride + c.star_rank_stride - 64);
        if (!status[1] || status[2]) {
            c.fail("RELAY generation publication rejected on rank " + std::to_string(i));
            return false;
        }
        epochs[i] = status[1];
        if (i && epochs[i] != epochs[0]) {
            c.fail("RELAY publication epochs disagree across ranks"); return false;
        }
    }
#if defined(__x86_64__) || defined(_M_X64)
    _mm_sfence();
#endif
    for (size_t i = 0; i < c.n_ranks; ++i)
        ((volatile uint32_t *) c.ranks[i].bcast_host[bank])[word] = epochs[i];
#if defined(__x86_64__) || defined(_M_X64)
    _mm_sfence();
#endif
    static const bool force_flush = [] {
        const char * v = getenv("GGML_TP5_RELAY_FORCE_FLUSH");
        return v && atoi(v) != 0;
    }();
    if (force_flush) for (size_t i = 0; i < c.n_ranks; ++i) {
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = c.ranks[i].bcast_mem[bank];
        range.size = VK_WHOLE_SIZE;
        if (vkFlushMappedMemoryRanges(c.ranks[i].vkdev, 1, &range) != VK_SUCCESS) {
            c.fail("RELAY diagnostic generation flush failed"); return false;
        }
    }
    c.relay_bank_used[bank] = true;
    return true;
}

static bool tp5_star_handoff(tp5_comm & c, size_t bank, size_t n_elems, tp5_star_times * times = nullptr,
                             bool relay = false, uint64_t arm_next_epoch = 0,
                             volatile uint32_t * const * direct_ready = nullptr,
                             size_t late_count = 0, bool late_sidecar_f16 = false,
                             uint32_t late_rows = 0) {
    struct relay_abort_guard {
        tp5_comm & comm;
        bool active;
        ~relay_abort_guard() {
            if (active) tp5_relay_request_abort(comm);
        }
    } abort_guard{c, relay};
    if (bank >= TP5_MAILBOX_BANKS || c.n_ranks != 5 || n_elems == 0 || n_elems > c.max_elems ||
        n_elems > UINT32_MAX) {
        c.fail("STAR/RELAY handoff exceeds its defined active payload capacity");
        return false;
    }
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + (relay ? std::chrono::milliseconds(c.relay_handoff_timeout_ms) :
                                           std::chrono::seconds(2));
    const void * rank_ptrs[5];
    volatile uint32_t * flags[5] = {};
    std::array<bool, 8> ready_seen{};
    for (size_t i = 0; i < c.n_ranks; ++i) {
        rank_ptrs[i] = (const char *) c.star_host_aligned[bank] + i * c.star_rank_stride;
        flags[i] = direct_ready ? direct_ready[i] :
            (volatile uint32_t *) ((char *) c.star_host_aligned[bank] + (i + 1) * c.star_rank_stride - 64);
        if (!flags[i]) {
            c.fail("RELAY direct producer missing ready route on rank " + std::to_string(i));
            return false;
        }
    }
    uint32_t pending = c.n_ranks == 32 ? UINT32_MAX : ((1u << c.n_ranks) - 1u);
    uint64_t poll_iters = 0;
    for (uint64_t spin = 0; pending != 0u; ++spin) {
        ++poll_iters;
        for (size_t i = 0; i < c.n_ranks; ++i) {
            const uint32_t bit = 1u << i;
            if ((pending & bit) && *flags[i] == 1u) {
                if (times && !ready_seen[i]) {
                    ready_seen[i] = true;
                    times->rank_ready_us[i] = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - start).count();
                }
                if (!direct_ready)
                    *flags[i] = 0u;
                pending &= ~bit;
            }
        }
        if (pending == 0u) {
            break;
        }
        // The readiness path must be L3 polling, not a clock syscall/clock
        // read per iteration. Keep the wait time-bounded while checking the
        // deadline only periodically.
        if ((spin & 1023u) == 0u && std::chrono::steady_clock::now() >= deadline) {
            size_t timed_out_rank = 0;
            while (timed_out_rank < c.n_ranks && (pending & (1u << timed_out_rank)) == 0u) {
                ++timed_out_rank;
            }
            c.fail("STAR payload timeout on rank " + std::to_string(timed_out_rank));
            if (getenv("GGML_TP5_RELAY_DEBUG") && timed_out_rank < c.n_ranks) {
                const auto * status = (const volatile uint32_t *) ((const char *) c.star_host_aligned[bank] +
                    timed_out_rank * c.star_rank_stride + c.star_rank_stride - 64);
                fprintf(stderr, "[tp5-relay-debug] payload-timeout bank=%zu rank=%zu flag=%u status=[%u,%u,%u,%u]\n",
                        bank, timed_out_rank, *flags[timed_out_rank], status[0], status[1], status[2], status[3]);
            }
            return false;
        }
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#endif
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    const auto ready = times ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (times) {
        times->poll_iters = poll_iters;
        double first = times->rank_ready_us[0];
        double last  = times->rank_ready_us[0];
        for (size_t i = 1; i < c.n_ranks; ++i) {
            first = std::min(first, times->rank_ready_us[i]);
            last  = std::max(last, times->rank_ready_us[i]);
        }
        times->first_ready_us = first;
        times->ready_skew_us  = last - first;
    }
    // Producer-ready/P1-ready for e is observed only after P2(e-1) on the
    // same queue. Re-arm bank(e+1) before the CPU reduction so an already
    // resident P2(e+1) can observe its generation without another submit.
    if (times && relay && arm_next_epoch > TP5_MAILBOX_BANKS) {
        const uint32_t retired_epoch = (uint32_t) (arm_next_epoch - TP5_MAILBOX_BANKS);
        const size_t retired_bank = tp5_mailbox_bank(arm_next_epoch);
        times->retired_epoch = retired_epoch;
        for (size_t i = 0; i < c.n_ranks; ++i) {
            const auto * status = (const volatile uint32_t *) ((const char *) c.star_host_aligned[retired_bank] +
                i * c.star_rank_stride + c.star_rank_stride - 64);
            times->retired_spin[i] = status[5];
            times->retired_q_spin[i] = status[7];
        }
    }
    const auto arm_start = times ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (relay && arm_next_epoch != 0 && !tp5_relay_arm_bank(c, arm_next_epoch, late_rows)) {
        return false;
    }
    const auto data_start = times ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (times) {
        times->arm_us = std::chrono::duration<double, std::micro>(data_start - arm_start).count();
    }
    void * sum = (char *) c.star_host_aligned[bank] + 5 * c.star_rank_stride;
    float * bcast_ptrs[5] = {};
    for (size_t i = 0; i < c.n_ranks; ++i) {
        bcast_ptrs[i] = (float *) ((char *) c.ranks[i].bcast_host[bank] + (relay ? 64 : 0));
        if (relay) {
            auto * header = (volatile uint32_t *) c.ranks[i].bcast_host[bank];
            header[0] = 0; // generation doorbell; CPU publishes it last
            header[1] = 0; // fused multi-workgroup completion counter
            if (late_count) header[2] = 0; // independent Q generation
            header[3] = uint32_t(n_elems); // useful F32 elements; published with the generation
        }
    }
    // RELAY decode payloads fit in L1/L2. Fan the reduction directly into the
    // five BAR mappings; keep STAR's established sum+memcpy path unchanged.
    const bool out_f32 = true;
    if (relay && n_elems < 262144) {
        if (c.wire == tp5_wire_type::F32)
            tp5_accumulate_broadcast_star_f32(rank_ptrs, bcast_ptrs, n_elems);
        else
            tp5_accumulate_broadcast_star(rank_ptrs, bcast_ptrs, n_elems);
    } else {
        if (c.avx2_pool && n_elems >= 262144) {
            if (!c.avx2_pool->parallel_for([&](size_t id, size_t count) {
                    if (c.wire == tp5_wire_type::F32)
                        tp5_accumulate_star_f32(rank_ptrs, (float *) sum, n_elems, id, count);
                    else
                        tp5_avx2_accumulate_star(rank_ptrs, sum, n_elems, id, count, out_f32);
                })) {
                c.fail("STAR CPU reduction timeout");
                return false;
            }
        } else {
            if (c.wire == tp5_wire_type::F32)
                tp5_accumulate_star_f32(rank_ptrs, (float *) sum, n_elems, 0, 1);
            else
                tp5_avx2_accumulate_star(rank_ptrs, sum, n_elems, 0, 1, out_f32);
        }
        const size_t bcast_bytes = n_elems * sizeof(float);
        for (size_t i = 0; i < c.n_ranks; ++i) {
            std::memcpy(bcast_ptrs[i], sum, bcast_bytes);
        }
    }
    const auto y_reduced = times ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (relay && late_count != 0) {
        const size_t late_wire_bytes =
            late_count * (late_sidecar_f16 ? sizeof(ggml_fp16_t) : sizeof(float));
        const size_t late_payload_end =
            tp5_late_q_bcast_payload_offset(c.late_host_offset, late_sidecar_f16) + late_wire_bytes;
        if (late_count > c.late_max_floats || late_payload_end > c.star_rank_stride) {
            c.fail("RELAY LateBind sidecar exceeds workspace");
            return false;
        }
        // Do not keep a ready activation behind the sidecar collective.
        // GPU Q work continues, then norm resumes as soon as this store lands.
        if (!tp5_relay_publish_generation(c, bank, 0)) return false;
        const auto q_wait_start = times ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (times) times->y_publish_us = std::chrono::duration<double, std::micro>(q_wait_start - y_reduced).count();
        uint32_t late_pending = c.n_ranks == 32 ? UINT32_MAX : ((1u << c.n_ranks) - 1u);
        for (uint64_t spin = 0; late_pending != 0u; ++spin) {
            for (size_t i = 0; i < c.n_ranks; ++i) {
                const uint32_t bit = 1u << i;
                if ((late_pending & bit) == 0u)
                    continue;
                const auto * status = (const volatile uint32_t *) ((const char *) c.star_host_aligned[bank] +
                    i * c.star_rank_stride + c.star_rank_stride - 64);
                if (status[2] != 0u) {
                    c.fail("RELAY sidecar aborted on rank " + std::to_string(i)); return false;
                }
                const auto * qctrl = tp5_late_q_control_cptr(c.ranks[i].bcast_host[bank], c.late_host_offset);
                const bool q_ready = late_sidecar_f16 ?
                    (qctrl && qctrl[TP5_LATE_Q_READY_WORD] == 1u) :
                    (status[6] == 1u);
                if (q_ready) {
                    late_pending &= ~bit;
                }
            }
            if (late_pending == 0u)
                break;
            if ((spin & 1023u) == 0u && std::chrono::steady_clock::now() >= deadline) {
                size_t rank = 0;
                while (rank < c.n_ranks && (late_pending & (1u << rank)) == 0u)
                    ++rank;
                c.fail("RELAY LateBind sidecar timeout on rank " + std::to_string(rank));
                return false;
            }
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#endif
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        const auto q_data_start = times ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (times) times->sidecar_wait_us = std::chrono::duration<double, std::micro>(q_data_start - q_wait_start).count();
        const void * late_ptrs[5] = {};
        float *      late_bcast_f32[5] = {};
        uint16_t *   late_bcast_f16[5] = {};
        for (size_t i = 0; i < c.n_ranks; ++i) {
            void * dst = (char *) c.ranks[i].bcast_host[bank] +
                         tp5_late_q_bcast_payload_offset(c.late_host_offset, late_sidecar_f16);
            late_bcast_f32[i] = (float *) dst;
            late_bcast_f16[i] = (uint16_t *) dst;
            late_ptrs[i] = late_sidecar_f16 ?
                (const void *) late_bcast_f16[i] :
                (const void *) ((const char *) c.star_host_aligned[bank] +
                                i * c.star_rank_stride + c.late_host_offset);
        }
        if (late_sidecar_f16)
            tp5_accumulate_broadcast_star_f16(late_ptrs, late_bcast_f16, late_count);
        else
            tp5_accumulate_broadcast_star_f32(late_ptrs, late_bcast_f32, late_count);
        const auto q_reduced = times ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (times) times->sidecar_data_us = std::chrono::duration<double, std::micro>(q_reduced - q_data_start).count();
        if (!tp5_relay_publish_generation(c, bank, 2)) return false;
        if (times) times->q_publish_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - q_reduced).count();
    }
    const auto reduced = times ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (times) {
        times->cpu_data_us = std::chrono::duration<double, std::micro>(y_reduced - data_start).count() +
                             times->sidecar_data_us;
    }
#if defined(__x86_64__) || defined(_M_X64)
    // LateBind already ordered each independent publication in its helper.
    if (!(relay && late_count))
        _mm_sfence(); // Order ordinary BAR payload stores before generation.
#endif
    if (relay && late_count == 0) {
        // bcast_mem is always HOST_COHERENT + DEVICE_COHERENT_AMD (cached
        // device-local VRAM is preferred; uncached is fallback only). The
        // coherent memory-domain contract removes the need for a Vulkan flush
        // call between host stores and the polling shader. Ordering is purely
        // payload stores -> sfence -> generation stores -> sfence.
        uint32_t seqs[5] = {};
        for (size_t i = 0; i < c.n_ranks; ++i) {
            const auto * status = (volatile uint32_t *) ((char *) c.star_host_aligned[bank] + i * c.star_rank_stride +
                                                         c.star_rank_stride - 64);
            const uint32_t seq = status[1];
            if (seq == 0u) {
                c.fail("RELAY missing epoch on rank " + std::to_string(i));
                return false;
            }
            seqs[i] = seq;
        }
        for (size_t i = 0; i < c.n_ranks; ++i) {
            ((volatile uint32_t *) c.ranks[i].bcast_host[bank])[0] = seqs[i];
        }
#if defined(__x86_64__) || defined(_M_X64)
        _mm_sfence();
#endif
        static const bool relay_force_flush = [] {
            const char * env = getenv("GGML_TP5_RELAY_FORCE_FLUSH");
            return env && atoi(env) != 0;
        }();
        if (relay_force_flush) {
            for (size_t i = 0; i < c.n_ranks; ++i) {
                VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
                range.memory = c.ranks[i].bcast_mem[bank];
                range.offset = 0;
                range.size = VK_WHOLE_SIZE;
                if (vkFlushMappedMemoryRanges(c.ranks[i].vkdev, 1, &range) != VK_SUCCESS) {
                    c.fail("RELAY payload-doorbell diagnostic flush failed on rank " + std::to_string(i));
                    return false;
                }
            }
        }
        c.relay_bank_used[bank] = true;
    }
    if (times) {
        const auto finished = std::chrono::steady_clock::now();
        times->wait_us = std::chrono::duration<double, std::micro>(ready - start).count();
        times->sum_us = std::chrono::duration<double, std::micro>(reduced - ready).count();
        times->broadcast_us = std::chrono::duration<double, std::micro>(finished - reduced).count();
        times->generation_us = late_count ? times->y_publish_us + times->q_publish_us : times->broadcast_us;
        times->total_us = std::chrono::duration<double, std::micro>(finished - start).count();
    }
    abort_guard.active = false;
    return true;
}

static vk_tp5_relay_payload_binding tp5_relay_payload_binding(const tp5_comm & c, const tp5_rank & r) {
    vk_tp5_relay_payload_binding payload;
    for (size_t bank = 0; bank < TP5_MAILBOX_BANKS; ++bank) {
        payload.bank[bank]  = r.host_import_buf[bank];
        payload.bytes[bank] = c.star_rank_stride;
    }
    payload.generation = c.workspace_gen;
    return payload;
}

// Lower immutable model/transport definitions into one primary command
// buffer per rank. This is definition-time only: no secondary CB execution,
// no CB-index skip convention, and no IR traversal on a warm token.
static bool tp5_define_linear_chain(tp5_comm & c,
        const std::vector<std::vector<std::vector<void *>>> & source_cbs,
        size_t n_stages, bool capture) try {
    // Bound definition churn (for example CHAIN_CACHE=0 or changing shapes).
    // The ordinary warm 48/96-stage program never enters this path.
    if (c.retired_linear_programs.size() >= 8) {
        if (!tp5_drain_submitted(c)) {
            c.fail("LateBind definition retirement could not drain submitted work"); return false;
        }
        c.retired_linear_programs.clear();
    }
    auto linear = std::make_shared<tp5_linear_program>();
    linear->devices.resize(c.n_ranks, VK_NULL_HANDLE);
    linear->pools.resize(c.n_ranks, VK_NULL_HANDLE);
    linear->commands.resize(c.n_ranks, VK_NULL_HANDLE);
    linear->graphs.resize((n_stages + 1) * c.n_ranks);
    // Resolve every rank before recording or submitting any of them.
    for (size_t s = 0; s <= n_stages; ++s) for (size_t r = 0; r < c.n_ranks; ++r) {
        if (source_cbs[s][r].empty()) {
            c.fail("LateBind linear definition has an empty source graph");
            return false;
        }
        auto graph = ggml_vk_tp5_graph_program(c.backends[r], source_cbs[s][r].front());
        if (!graph || !graph->commands.valid) {
            c.fail("LateBind source definition unavailable at stage " + std::to_string(s) +
                   " rank " + std::to_string(r) + ": " +
                   (graph ? graph->commands.rejection : "graph was not recorded for inline lowering"));
            return false;
        }
        if (graph->commands.sources.size() != source_cbs[s][r].size()) {
            c.fail("LateBind source CB count changed before definition");
            return false;
        }
        for (size_t k = 0; k < source_cbs[s][r].size(); ++k) {
            if (graph->commands.sources[k] != (VkCommandBuffer) source_cbs[s][r][k]) {
                c.fail("LateBind source CB identity changed before definition");
                return false;
            }
        }
        linear->graphs[s * c.n_ranks + r] = std::move(graph);
    }
    const auto same_binding = [](const vk_tp5_hc_binding & a, const tp5_binding_key & b) {
        return a.buffer == b.buf && a.offset == b.offset && a.size == b.size;
    };
    for (size_t r = 0; r < c.n_ranks; ++r) {
        auto & rank = c.ranks[r];
        linear->device_owners.push_back(rank.device);
        linear->devices[r] = rank.vkdev;
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.queueFamilyIndex = rank.queue_family;
        if (vkCreateCommandPool(rank.vkdev, &pci, nullptr, &linear->pools[r]) != VK_SUCCESS) {
            c.fail("LateBind linear command pool allocation failed"); return false;
        }
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = linear->pools[r];
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(rank.vkdev, &ai, &linear->commands[r]) != VK_SUCCESS) {
            c.fail("LateBind linear primary allocation failed"); return false;
        }
        const VkCommandBuffer cmd = linear->commands[r];
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
            c.fail("LateBind linear primary begin failed"); return false;
        }
        size_t dispatches = 0, barriers = 0, copies = 0, hoisted = 0, late_sites = 0, q8_sites = 0, p1a_sites = 0;
        const auto emit = [&](const vk_tp5_command_tape & tape, size_t first = 0, size_t last = SIZE_MAX) {
            if (last == SIZE_MAX) last = tape.code.size();
            if (!tape.emit(cmd, first, last)) {
                c.fail("invalid LateBind command definition: " + tape.rejection); return false;
            }
            for (size_t k = first; k < last; ++k) {
                dispatches += tape.code[k].type == vk_tp5_command_tape::kind::dispatch;
                barriers += tape.code[k].type == vk_tp5_command_tape::kind::barrier;
                copies += tape.code[k].type == vk_tp5_command_tape::kind::copy;
            }
            return true;
        };
        const auto timestamp = [&](size_t q) {
            if (!capture) return;
            if (q == 0) vkCmdResetQueryPool(cmd, rank.timing_pool, 0, (uint32_t) (5 * n_stages + 3));
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, rank.timing_pool, (uint32_t) q);
        };
        for (size_t s = 0; s <= n_stages; ++s) {
            const auto & graph = *linear->graphs[s * c.n_ranks + r];
            tp5_cached_plan * outgoing = s < n_stages ? &c.cached_plans[c.chain_plan_indices[s]] : nullptr;
            const uint64_t epoch = c.allreduce_calls + s + 1;
            const size_t slot = tp5_plan_slot(r, tp5_mailbox_bank(epoch));
            const bool late_out = outgoing && outgoing->key.late[r].late_rank != 0;
            const bool direct_out = outgoing && tp5_relay_direct_stage(outgoing->key, c.n_ranks);
            if (late_out && outgoing->numerical_mode == tp5_numerical_mode::P1A_NOSIDECAR_Q8) ++p1a_sites;
            else if (late_out && outgoing->late_q8_fast) ++q8_sites;
            const bool scatter_source_matches = late_out && s < n_stages &&
                graph.normalized_tensor != nullptr &&
                linear->graphs[(s + 1) * c.n_ranks + r]->hc.inject_input_tensor == graph.normalized_tensor &&
                same_binding(graph.normalized, outgoing->key.late[r].bindings[5]);
            bool injected = false;
            size_t model_start = 0;
            timestamp(5 * s);
            if (s > 0) {
                auto & incoming = c.cached_plans[c.chain_plan_indices[s - 1]];
                const size_t prev_slot = tp5_plan_slot(r, tp5_mailbox_bank(epoch - 1));
                if (incoming.p2_definitions.size() <= prev_slot) {
                    c.fail("missing inline P2 definition"); return false;
                }
                if (incoming.key.late[r].late_rank != 0) {
                    ++late_sites;
                    if (graph.hc.late_rank == 0 || graph.hc_down_end <= graph.hc_norm_end ||
                        graph.hc_down_end > graph.commands.code.size() ||
                        (incoming.late_q8_fast &&
                         (graph.hc_up_end <= graph.hc_down_end || graph.hc_up_end > graph.commands.code.size()))) {
                        c.fail("LateBind HC semantic range is unavailable"); return false;
                    }
                    const size_t cut = incoming.late_norm_end[prev_slot];
                    const size_t lo_begin = incoming.late_lo_begin[prev_slot];
                    if (!cut || lo_begin <= cut || lo_begin > incoming.p2_definitions[prev_slot].code.size()) {
                        c.fail("LateBind norm semantic split is invalid"); return false;
                    }
                    // First emit only the norm->consumer visibility barrier.
                    // That makes normalized safe for the next stage's scatter
                    // without forcing us through the Q-dependent LO wait.
                    if (!emit(incoming.p2_definitions[prev_slot], cut, lo_begin)) return false;
                    if (scatter_source_matches) {
                        if (!emit(outgoing->pre_definitions[slot], 0, outgoing->late_inject_end[slot])) return false;
                        injected = true;
                        ++hoisted;
                    }
                    // Then consume Q. Aggressive suffix also performs Q8
                    // W_up/fold, so skip the original F32 W_up semantic range.
                    if (!emit(incoming.p2_definitions[prev_slot], lo_begin)) return false;
                    model_start = incoming.late_q8_fast ? graph.hc_up_end : graph.hc_down_end;
                } else if (!emit(incoming.p2_definitions[prev_slot])) return false;
            }
            timestamp(5 * s + 1);
            if (late_out && !injected && graph.hc_norm_end > model_start &&
                scatter_source_matches) {
                if (!emit(graph.commands, model_start, graph.hc_norm_end) ||
                    !emit(outgoing->pre_definitions[slot], 0, outgoing->late_inject_end[slot])) return false;
                model_start = graph.hc_norm_end;
                injected = true; ++hoisted;
            }
            if (!emit(graph.commands, model_start)) return false;
            timestamp(5 * s + 2);
            if (!outgoing) break;
            if (!direct_out) {
                if (!outgoing->p1 || outgoing->p1->definitions.size() <= slot) {
                    c.fail("missing inline P1 definition"); return false;
                }
                // P1 must lead every late precompute. The CPU can begin the
                // Y reduction as soon as its payload is visible, while this
                // queue continues with scatter/Q preparation. Moving scatter
                // before P1 hid a transport-critical dependency behind local
                // work and inflated the handoff tail.
                if (!emit(outgoing->p1->definitions[slot])) return false;
            }
            timestamp(5 * s + 3);
            if (late_out) {
                const auto & pre = outgoing->pre_definitions[slot];
                const auto & p2  = outgoing->p2_definitions[slot];
                const size_t inject_end = outgoing->late_inject_end[slot];
                const size_t q_begin    = outgoing->late_q_begin[slot];
                const size_t q_contract = outgoing->late_q_contract_begin[slot];
                const size_t norm_end   = outgoing->late_norm_end[slot];
                const size_t lo_begin   = outgoing->late_lo_begin[slot];
                if (!inject_end || q_begin <= inject_end || q_begin > pre.code.size() ||
                    q_contract < q_begin || q_contract > pre.code.size() ||
                    !norm_end || norm_end > p2.code.size() ||
                    lo_begin <= norm_end || lo_begin > p2.code.size()) {
                    c.fail("LateBind linear semantic split is invalid"); return false;
                }
                if (outgoing->numerical_mode == tp5_numerical_mode::P1A_NOSIDECAR_Q8) {
                    // P1-A measurement instrument: no-sidecar aggressive Q8 HC
                    // Y is ready -> combine/RMS (late_norm) -> norm_ready -> ACT_Q8 -> act_ready ->
                    // down Q8DOT -> q_local_ready -> LO_Q8 -> lo_ready -> UP_Q8DOT.
                    // Emit through lo_begin: the p2 segment [norm_end, lo_begin)
                    // is exactly the norm_ready global memory barrier. Without
                    // it, ACT_Q8 could read local_z (trefs) before late_norm's
                    // sum_output write to the same buffer is visible — the
                    // BARRIER_NORM_ACT contract would be a label with no command.
                    if (!emit(p2, 0, lo_begin)) {
                        return false;
                    }
                    linear->late_steps.push_back({tp5_latebind_semantic_step::kind::WRITE_TREFS, "late_norm"});
                    linear->late_steps.push_back({tp5_latebind_semantic_step::kind::BARRIER_NORM_ACT, "norm_ready"});

                    if (!emit(pre, injected ? inject_end : 0, q_contract)) {
                        return false;
                    }
                    linear->late_steps.push_back({tp5_latebind_semantic_step::kind::DISPATCH_ACT_Q8, "norm_act_q8"});
                    linear->late_steps.push_back({tp5_latebind_semantic_step::kind::BARRIER_ACT_BUF, "act_ready"});

                    if (!emit(pre, q_contract)) {
                        return false;
                    }
                    linear->late_steps.push_back({tp5_latebind_semantic_step::kind::DISPATCH_Q8DOT, "down_q8dot_local"});
                    linear->late_steps.push_back({tp5_latebind_semantic_step::kind::BARRIER_LOCAL_Q, "q_local_ready"});

                    linear->late_steps.push_back({tp5_latebind_semantic_step::kind::DISPATCH_LO_Q8, "lo_q8_local"});
                    linear->late_steps.push_back({tp5_latebind_semantic_step::kind::BARRIER_LO_BUF, "lo_ready"});
                    linear->late_steps.push_back({tp5_latebind_semantic_step::kind::DISPATCH_UP_Q8DOT, "up_q8dot_fold"});
                } else if (outgoing->late_q8_fast) {
                    if (q_contract <= q_begin) {
                        c.fail("LateBind aggressive Q8 semantic split is missing"); return false;
                    }
                    // P1 -> scatter -> ACT-Q8 -> ACT visibility -> WAR barrier (trefs) -> norm(Y)
                    // -> Q8dot.
                    // The WAR barrier on trefs (SHADER_READ -> SHADER_WRITE) guarantees late_act_q8
                    // finishes reading z_p before late_norm overwrites it with canonical y.
                    // Symmetrically with the exact-F32 branch, this barrier is placed strictly
                    // before norm, preserving the zero-barrier overlap between norm and Q8dot.
                    if (!emit(pre, injected ? inject_end : 0, q_contract)) {
                        return false;
                    }
                    linear->late_steps.push_back({tp5_latebind_semantic_step::kind::READ_TREFS, "late_act_q8"});
                    linear->late_steps.push_back({tp5_latebind_semantic_step::kind::BARRIER_ACT_BUF, "act_ready"});

                    VkBufferMemoryBarrier q_norm_barrier{
                        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
                        VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                        outgoing->key.bindings[r].buf,
                        outgoing->key.bindings[r].offset,
                        outgoing->key.bindings[r].size
                    };
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         0, 0, nullptr, 1, &q_norm_barrier, 0, nullptr);
                    ++barriers;
                    linear->late_steps.push_back({tp5_latebind_semantic_step::kind::BARRIER_WAR_TREFS, "q_norm_barrier"});

                    if (!emit(p2, 0, norm_end)) {
                        return false;
                    }
                    linear->late_steps.push_back({tp5_latebind_semantic_step::kind::WRITE_TREFS, "late_norm"});

                    if (!emit(pre, q_contract)) {
                        return false;
                    }
                    linear->late_steps.push_back({tp5_latebind_semantic_step::kind::DISPATCH_Q8DOT, "late_q8dot"});
                } else {
                    // Exact fallback: late_q reads z_p from outgoing->key.bindings[r],
                    // while norm overwrites the same buffer with canonical y. To prevent
                    // the RAW hazard, late_q and sidecar publication must strictly precede
                    // norm, separated by an execution/memory barrier on the tensor buffer.
                    if (!emit(pre, injected ? inject_end : 0)) {
                        return false;
                    }
                    linear->late_steps.push_back({tp5_latebind_semantic_step::kind::READ_TREFS, "late_q"});

                    VkBufferMemoryBarrier q_norm_barrier{
                        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr,
                        VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                        outgoing->key.bindings[r].buf,
                        outgoing->key.bindings[r].offset,
                        outgoing->key.bindings[r].size
                    };
                    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         0, 0, nullptr, 1, &q_norm_barrier, 0, nullptr);
                    ++barriers;
                    linear->late_steps.push_back({tp5_latebind_semantic_step::kind::BARRIER_WAR_TREFS, "q_norm_barrier"});

                    if (!emit(p2, 0, norm_end)) {
                        return false;
                    }
                    linear->late_steps.push_back({tp5_latebind_semantic_step::kind::WRITE_TREFS, "late_norm"});
                }
            }
            timestamp(5 * s + 4);
        }
        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            c.fail("LateBind linear primary end failed"); return false;
        }
        if (!linear->late_steps.empty()) {
            std::string war_err;
            if (p1a_sites > 0) {
                if (!tp5_validate_p1a_schedule(linear->late_steps, war_err)) {
                    c.fail("P1-A linear schedule validation failed on rank " + std::to_string(r) + ": " + war_err);
                    return false;
                }
            } else if (!tp5_validate_latebind_war_schedule(q8_sites > 0, linear->late_steps, war_err)) {
                c.fail("LateBind linear WAR schedule validation failed on rank " + std::to_string(r) + ": " + war_err);
                return false;
            }
        }
        const char * active_num_mode = (late_sites == 0) ? "reference" :
                                       (p1a_sites > 0 ? "p1a-nosidecar-q8" :
                                       (q8_sites > 0 ? "aggressive-q8" : "exact-f32"));
        const char * active_num_desc = (late_sites == 0) ? "reference-no-latebind" :
                                       (p1a_sites > 0 ? "p1a-nosidecar-q8-bench" :
                                       (q8_sites > 0 ? "aggressive-q8-quantized" : "exact-f32-strict-math"));
        fprintf(stderr, "[tp5-linear-definition] rank=%zu stages=%zu primary_cbs=1 mode=%s desc=%s late=%zu "
                        "q8_fast=%zu scatter_early=%zu overlap=norm-q sidecar_pub=fused "
                        "dispatches=%zu barriers=%zu copies=%zu timing=%d\n",
                r, n_stages, active_num_mode, active_num_desc, late_sites, q8_sites, hoisted, dispatches, barriers, copies, capture ? 1 : 0);
    }
    if (c.linear_program) c.retired_linear_programs.push_back(std::move(c.linear_program));
    c.linear_program = std::move(linear);
    for (size_t r = 0; r < c.n_ranks; ++r)
        c.chain_scratch[r].compute.assign(1, c.linear_program->commands[r]);
    return true;
} catch (const std::exception & error) {
    c.fail(std::string("LateBind definition failed before submission: ") + error.what());
    return false;
} catch (...) {
    c.fail("LateBind definition failed before submission");
    return false;
}

static bool tp5_relay_submit_epoch_chain(
        tp5_comm & c, const std::vector<std::vector<std::vector<void *>>> & stage_compute_cbs,
        const std::vector<tp5_plan_key> & keys, const std::vector<size_t> & active_elems,
        size_t n_stages, uint32_t active_rows, uint32_t capacity_rows) {
    ggml_tp5_profile * prof = ggml_tp5_profile_active();
    static const bool relay_stage_detail = [] {
        const char * env = getenv("GGML_TP5_PROFILE_RELAY_STAGES");
        return env && atoi(env) != 0;
    }();
    // Each rank submits one ordered command-buffer stream:
    // direct producer: [compute+host-payload(0), P2(1), compute+host-payload(1), ...]
    // fallback:        [compute(0), P1(1), P2(1), compute(1), P1(2), ...]
    // P2 is resident before the CPU observes either route-ready or the legacy
    // P1 flag. Two mailbox banks are reused only after transitive completion
    // credit from the following producer publication.
    if (n_stages == 0 || keys.size() != n_stages || active_elems.size() != n_stages ||
        stage_compute_cbs.size() != n_stages + 1 ||
        c.chain_scratch.size() != c.n_ranks) {
        c.fail("RELAY epoch chain scratch shape invalid");
        return false;
    }
    const uint64_t first_epoch = c.allreduce_calls + 1;
    const uint64_t last_epoch  = first_epoch + n_stages - 1;
    if (last_epoch > UINT32_MAX) {
        c.fail("RELAY epoch exceeds 32-bit doorbell generation");
        return false;
    }

    // The first P2 can start as soon as P1 does. A prior chain has already
    // armed this generation; the initial chain arms it from idle state.
    if (!tp5_relay_ensure_armed_epoch(c, first_epoch)) {
        return false;
    }

    // Publish the runtime active-row count into the first bank's RELAY header
    // word 5 before any rank is submitted. The bank is provably idle here:
    // ensure_armed_epoch verified flag/error/done zero. Later banks in this
    // chain receive the same word during their pre-arm in tp5_star_handoff,
    // after the previous reader's transitive credit. LateBind kernels read
    // word 5 to bound their token loops, so a maximum-capacity definition
    // executes only the useful rows (M3 invariant 2).
    {
        const uint32_t late_rows = active_rows ? active_rows : capacity_rows;
        const size_t first_bank = tp5_mailbox_bank(first_epoch);
        for (size_t i = 0; i < c.n_ranks; ++i) {
            if (c.ranks[i].bcast_host[first_bank]) {
                ((volatile uint32_t *) c.ranks[i].bcast_host[first_bank])[5] = late_rows;
            }
        }
        std::atomic_thread_fence(std::memory_order_release);
#if defined(__x86_64__) || defined(_M_X64)
        _mm_sfence();
#endif
    }

    const uint32_t chain_late_rows = active_rows ? active_rows : capacity_rows;

    // Patch every capacity-aware dispatch, including the non-reducing tail,
    // before any rank is submitted. A smaller active prefix is legal only if
    // each source definition has classified every dispatch it owns.
    if (active_rows && capacity_rows) {
        for (size_t s = 0; s <= n_stages; ++s) {
            for (size_t r = 0; r < c.n_ranks; ++r) {
                const auto & source = stage_compute_cbs[s][r];
                if (source.empty()) {
                    if (active_rows < capacity_rows) {
                        c.fail("predefined row tail/source graph is empty");
                        return false;
                    }
                    continue;
                }
                if (!ggml_vk_tp5_update_predefined_dispatches(
                        c.backends[r], source.front(), s, active_rows, capacity_rows)) {
                    if (active_rows < capacity_rows) {
                        c.fail("predefined row dispatch patch rejected stage=" + std::to_string(s) +
                               " rank=" + std::to_string(r));
                        return false;
                    }
                }
            }
        }
    }

    const auto route_begin = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    size_t n_direct_stages = 0;
    size_t n_late_stages   = 0;
    for (size_t s = 0; s < n_stages; ++s) {
        auto & direct_ready = c.relay_ready_routes[s];
        direct_ready.fill(nullptr);
        const bool direct = tp5_relay_direct_stage(keys[s], c.n_ranks);
        const bool late = tp5_late_stage(keys[s], c.n_ranks);
        // A stage drops P1 only when EVERY rank has a direct producer. In a
        // mixed stage the direct-capable ranks still run their recorded direct
        // shader, followed by P1, so P1 must be able to read the local partial.
        // Patch their routes as well: a previous all-direct replay may have
        // left keep_local=false and a different bank/generation in the slot.
        const bool keep_local = !direct || late;
        if (direct)
            ++n_direct_stages;
        if (late)
            ++n_late_stages;
        const uint64_t epoch = first_epoch + s;
        const size_t   bank  = tp5_mailbox_bank(epoch);
        for (size_t i = 0; i < c.n_ranks; ++i) {
            if (!keys[s].relay_direct[i]) {
                continue;
            }
            volatile uint32_t * ready = nullptr;
            const auto payload = tp5_relay_payload_binding(c, c.ranks[i]);
            if (!ggml_vk_tp5_update_relay_route(c.backends[i], s, (uint32_t) bank, epoch, payload, &ready,
                                                keep_local, active_elems[s]) || !ready) {
                c.fail("RELAY direct route update failed stage=" + std::to_string(s) +
                       " rank=" + std::to_string(i));
                return false;
            }
            direct_ready[i] = direct ? ready : nullptr;
        }
    }
    std::atomic_thread_fence(std::memory_order_release);
    const auto route_end = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    const uint64_t final_signal = 2 * last_epoch;
    // Allocate diagnostic storage before the first rank is submitted; an
    // allocation failure must never strand an already queued GPU waiter.
    std::vector<tp5_star_times> stage_times;
    std::vector<std::array<uint32_t, 8>> stage_spins, stage_q_spins;
    if (prof) {
        try {
            stage_times.resize(n_stages);
            stage_spins.resize(n_stages);
            stage_q_spins.resize(n_stages);
            for (auto & spins : stage_spins) spins.fill(UINT32_MAX);
            for (auto & spins : stage_q_spins) spins.fill(UINT32_MAX);
        } catch (...) {
            c.fail("RELAY profile allocation failed before submission"); return false;
        }
    }
    bool submitted_any = false;
    double submit_wall_us = 0.0;
    for (size_t i = 0; i < c.n_ranks; ++i) {
        auto & scratch = c.chain_scratch[i];
        const size_t min_collective_cbs = 2 * n_stages - n_direct_stages + n_late_stages;
        const bool linear = c.wire == tp5_wire_type::F32 && tp5_latebind_hc_enabled();
        if ((linear ? scratch.compute.size() != 1 : scratch.compute.size() < min_collective_cbs) ||
            scratch.compute.size() > UINT32_MAX) {
            if (submitted_any) tp5_relay_request_abort(c);
            c.fail("RELAY epoch chain command layout invalid on rank " + std::to_string(i));
            return false;
        }
        VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        timeline.signalSemaphoreValueCount = 1;
        timeline.pSignalSemaphoreValues    = &final_signal;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.pNext                = &timeline;
        submit.commandBufferCount   = (uint32_t) scratch.compute.size();
        submit.pCommandBuffers      = scratch.compute.data();
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores    = &c.ranks[i].timeline_sem;
        if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
            ++prof->queue_submits;
            ++prof->submit_batches;
        }
        const auto submit_begin = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (vkQueueSubmit(c.ranks[i].queue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS) {
            tp5_relay_request_abort(c);
            c.fail("RELAY full-chain submit failed on rank " + std::to_string(i));
            return false;
        }
        if (prof) {
            submit_wall_us += std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - submit_begin).count();
        }
        ggml_vk_tp5_mark_queue_submitted(c.backends[i]);
        tp5_note_timeline_submission(c.ranks[i], final_signal);
        submitted_any = true;
    }

    for (size_t s = 0; s < n_stages; ++s) {
        const uint64_t epoch = first_epoch + s;
        // e+1 reuses bank(e-1). A direct producer's route-ready (or legacy
        // P1(e)'s flag) is observed only after a COMPUTE->HOST publication on
        // the same queue, so it is also transitive proof that P2(e-1)
        // completed. Arm e+1 before reducing/publishing e.
        const uint64_t arm_next_epoch = epoch < UINT32_MAX ? epoch + 1 : 0;
        const bool direct = tp5_relay_direct_stage(keys[s], c.n_ranks);
        const bool late   = tp5_late_stage(keys[s], c.n_ranks);
        const size_t stage_rows = (capacity_rows != 0 && active_rows != 0) ? active_rows : 1;
        const size_t late_count = late ? stage_rows * size_t(keys[s].late[0].streams) * keys[s].late[0].late_rank : 0;
        const bool late_sidecar_f16 =
            late && c.cached_plans[c.chain_plan_indices[s]].late_sidecar_f16;
        tp5_star_times * step = prof ? &stage_times[s] : nullptr;
        if (!tp5_star_handoff(c, tp5_mailbox_bank(epoch), active_elems[s], step, true,
                              arm_next_epoch, direct ? c.relay_ready_routes[s].data() : nullptr,
                              late_count, late_sidecar_f16, chain_late_rows)) {
            fprintf(stderr, "[tp5-relay-chain] handoff failed stage=%zu epoch=%llu bank=%zu\n",
                    s, (unsigned long long) epoch, tp5_mailbox_bank(epoch));
            return false;
        }
        if (prof && step->retired_epoch >= first_epoch && step->retired_epoch <= last_epoch) {
            const size_t retired_stage = (size_t) (step->retired_epoch - first_epoch);
            if (retired_stage < stage_spins.size()) {
                stage_spins[retired_stage] = step->retired_spin;
                stage_q_spins[retired_stage] = step->retired_q_spin;
            }
        }
    }

    // Never wait for the final P2 merely to profile it. If it already
    // completed by the time host handoffs finish, opportunistically sample it;
    // otherwise its spin count remains pending/unknown for this chain.
    if (prof && n_stages != 0) {
        const size_t last_stage = n_stages - 1;
        const size_t bank = tp5_mailbox_bank(last_epoch);
        for (size_t i = 0; i < c.n_ranks; ++i) {
            const auto * status = (const volatile uint32_t *) ((const char *) c.star_host_aligned[bank] +
                i * c.star_rank_stride + c.star_rank_stride - 64);
            if (status[3] == (uint32_t) last_epoch) {
                stage_spins[last_stage][i] = status[5];
                stage_q_spins[last_stage][i] = status[7];
            }
        }
    }

    // Match TIMELINE's asynchronous chain contract: the final P2 and graph
    // tail remain in flight behind their real final timeline value. Capacity
    // backpressure and teardown drain that actual submission; no per-chain
    // host wait or second P2 submission is permitted here.
    c.allreduce_calls = last_epoch;
    if (prof) {
        prof->collective_calls += n_stages;
        ++prof->relay_chains;
        prof->relay_direct_stages += n_direct_stages;
        prof->relay_fallback_stages += n_stages - n_direct_stages;
        prof->relay_route_patch_us += (uint64_t) (std::chrono::duration<double, std::micro>(
            route_end - route_begin).count() + 0.5);
        prof->relay_submit_us += (uint64_t) (submit_wall_us + 0.5);

        for (size_t s = 0; s < n_stages; ++s) {
            const auto & step = stage_times[s];
            const uint64_t wait_us = (uint64_t) (step.wait_us + 0.5);
            const uint64_t skew_us = (uint64_t) (step.ready_skew_us + 0.5);
            prof->relay_ready_wait_us += wait_us;
            prof->relay_ready_skew_us += skew_us;
            prof->relay_arm_us += (uint64_t) (step.arm_us + 0.5);
            prof->relay_cpu_data_us += (uint64_t) (step.cpu_data_us + 0.5);
            prof->relay_sidecar_wait_us += (uint64_t) (step.sidecar_wait_us + 0.5);
            prof->relay_sidecar_data_us += (uint64_t) (step.sidecar_data_us + 0.5);
            prof->relay_y_publish_us += (uint64_t) (step.y_publish_us + 0.5);
            prof->relay_q_publish_us += (uint64_t) (step.q_publish_us + 0.5);
            prof->relay_generation_us += (uint64_t) (step.generation_us + 0.5);
            prof->relay_handoff_total_us += (uint64_t) (step.total_us + 0.5);
            prof->relay_poll_iters += step.poll_iters;
            tp5_profile_atomic_max(prof->relay_ready_wait_max_us, wait_us);
            tp5_profile_atomic_max(prof->relay_ready_skew_max_us, skew_us);

            uint64_t spin_sum = 0;
            uint64_t spin_max = 0;
            uint64_t spin_n = 0;
            for (size_t i = 0; i < c.n_ranks; ++i) {
                const uint32_t spins = stage_spins[s][i];
                if (spins == UINT32_MAX)
                    continue;
                spin_sum += spins;
                spin_max = std::max<uint64_t>(spin_max, spins);
                ++spin_n;
            }
            prof->relay_gpu_spin_iters += spin_sum;
            prof->relay_gpu_spin_samples += spin_n;
            tp5_profile_atomic_max(prof->relay_gpu_spin_max, spin_max);
            uint64_t q_sum = 0, q_max = 0, q_n = 0;
            if (tp5_late_stage(keys[s], c.n_ranks)) {
                for (size_t i = 0; i < c.n_ranks; ++i) {
                    const uint32_t spins = stage_q_spins[s][i];
                    if (spins == UINT32_MAX) continue;
                    q_sum += spins; q_max = std::max<uint64_t>(q_max, spins); ++q_n;
                }
                prof->relay_q_spin_iters += q_sum;
                prof->relay_q_spin_samples += q_n;
                tp5_profile_atomic_max(prof->relay_q_spin_max, q_max);
            }

            if (relay_stage_detail) {
                fprintf(stderr,
                        "[tp5-relay-stage] exec=%llu stage=%zu epoch=%llu direct=%d wire=%s elems=%zu "
                        "ready_first_us=%.3f ready_all_us=%.3f skew_us=%.3f arm_us=%.3f "
                        "cpu_data_us=%.3f generation_us=%.3f total_us=%.3f polls=%llu "
                        "gpu_spin_avg=%.1f gpu_spin_max=%llu gpu_spin_n=%llu "
                        "ready_rank_us=",
                        (unsigned long long) prof->graph_exec_id, s,
                        (unsigned long long) (first_epoch + s),
                        tp5_relay_direct_stage(keys[s], c.n_ranks) ? 1 : 0,
                        c.wire == tp5_wire_type::F32 ? "f32" : "f16", keys[s].n_elems,
                        step.first_ready_us, step.wait_us, step.ready_skew_us, step.arm_us,
                        step.cpu_data_us, step.generation_us, step.total_us,
                        (unsigned long long) step.poll_iters,
                        spin_n ? double(spin_sum) / double(spin_n) : -1.0,
                        (unsigned long long) spin_max, (unsigned long long) spin_n);
                for (size_t i = 0; i < c.n_ranks; ++i) {
                    fprintf(stderr, "%s%.3f", i ? "," : "", step.rank_ready_us[i]);
                }
                fputc('\n', stderr);
                fprintf(stderr, "[tp5-latebind-stage] exec=%llu stage=%zu sidecar_wait_us=%.3f "
                                "sidecar_data_us=%.3f y_publish_us=%.3f q_publish_us=%.3f "
                                "q_spin_avg=%.1f q_spin_max=%llu q_spin_n=%llu\n",
                        (unsigned long long) prof->graph_exec_id, s, step.sidecar_wait_us,
                        step.sidecar_data_us, step.y_publish_us, step.q_publish_us,
                        q_n ? double(q_sum) / double(q_n) : -1.0,
                        (unsigned long long) q_max, (unsigned long long) q_n);
            }
        }
    }
    return true;
}

bool tp5_allreduce_star(tp5_comm & c, ggml_tensor ** tensors, size_t n_elems) {

    // Executes the 6-Pillar Star AllReduce:
    // Pillar 1: GPU BDA upstream push -> GPU writes directly to host-imported RAM
    // Pillar 2: DRM Syncobj wait -> Host waits for all ranks' Phase 1 timeline signal
    // Pillar 3: 22-core AVX2 vectorized accumulate directly in L3 cache
    // Pillar 4: CPU Root Complex broadcast -> Copies accumulated sum to each GPU's host-visible broadcast buffer
    // Pillar 5: Async DRM Syncobj signal handoff -> non-blocking background signal to unblock Phase 2 compute
    // Pillar 6: Overlapped double-buffering -> alternating mailbox banks (epoch & 1)
    const uint64_t epoch = ++c.allreduce_calls;
    const size_t bank = tp5_mailbox_bank(epoch);

    static double acc_flush_us = 0;
    static double acc_pure_sub_us = 0;
    static double acc_rec_us = 0;
    static double acc_p1_sub_us = 0;
    static double acc_wait_us = 0;
    static double acc_avx2_us = 0;
    static double acc_bcast_us = 0;
    static double acc_p2_sub_us = 0;
    static uint64_t call_cnt = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    auto t_flush_start = std::chrono::high_resolution_clock::now();
    for (auto backend : c.backends) {
        ggml_vk_tp5_flush_async(backend);
    }
    auto t_flush_done = std::chrono::high_resolution_clock::now();

    if (c.max_elems < n_elems) {
        if (!tp5_setup_workspace(c, n_elems)) {
            c.fail("tp5_allreduce_star: workspace setup failed");
            return false;
        }
    }

    const size_t payload = n_elems * sizeof(uint16_t);

    std::vector<tensor_dev_ref> trefs(c.n_ranks);
    for (size_t j = 0; j < c.n_ranks; ++j) {
        trefs[j] = tp5_tensor_dev_ref(tensors[j]);
        if (!trefs[j].ok) {
            c.fail("tensor " + std::to_string(j) + " is not in a Vulkan buffer");
            return false;
        }
    }

    tp5_plan_key key;
    key.n_elems = n_elems;
    key.wire = c.wire;
    key.stride = c.star_rank_stride;
    key.workspace_gen = c.workspace_gen;
    key.bindings.resize(c.n_ranks);
    for (size_t j = 0; j < c.n_ranks; ++j) {
        key.bindings[j].buf = trefs[j].buf;
        key.bindings[j].offset = trefs[j].offset;
        key.bindings[j].size = trefs[j].size;
    }

    tp5_cached_plan * plan = nullptr;
    bool plan_cache_hit = false;
    size_t & last_hit_idx = c.last_hit_idx;
    if (c.cmd_replay_enabled) {
        if (last_hit_idx < c.cached_plans.size() && c.cached_plans[last_hit_idx].key == key &&
            !c.cached_plans[last_hit_idx].star_cmd_p1.empty()) {
            plan = &c.cached_plans[last_hit_idx];
            plan->last_used_call = c.allreduce_calls;
            plan_cache_hit = true;
        } else {
            for (size_t idx = 0; idx < c.cached_plans.size(); ++idx) {
                if (c.cached_plans[idx].key == key && !c.cached_plans[idx].star_cmd_p1.empty()) {
                    plan = &c.cached_plans[idx];
                    plan->last_used_call = c.allreduce_calls;
                    last_hit_idx = idx;
                    plan_cache_hit = true;
                    break;
                }
            }
        }
    }

    auto t_rec_start = std::chrono::high_resolution_clock::now();
    struct star_rank_submit_t {
        tp5_rank * rk;
        VkSubmitInfo submits[2];
        VkTimelineSemaphoreSubmitInfo tsi[2];
        uint64_t sig_p1;
        uint64_t wait_ready;
        uint64_t sig_p2;
        VkPipelineStageFlags wait_stage;
        VkCommandBuffer cmd_p1;
        VkCommandBuffer cmd_p2;
    };
    std::vector<star_rank_submit_t> star_submits(c.n_ranks);

    if (!plan_cache_hit) {
        tp5_cached_plan new_plan;
        new_plan.key = key;
        new_plan.star_cmd_p1.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
        new_plan.star_cmd_p2.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);

        for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
            for (size_t i = 0; i < c.n_ranks; ++i) {
                tp5_rank & r = c.ranks[i];
                const size_t bslot = tp5_plan_slot(i, b);

                VkCommandBufferAllocateInfo cb_ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                cb_ai.commandPool = r.cmd_pool;
                cb_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                cb_ai.commandBufferCount = 1;
                vkAllocateCommandBuffers(r.vkdev, &cb_ai, &new_plan.star_cmd_p1[bslot]);
                vkAllocateCommandBuffers(r.vkdev, &cb_ai, &new_plan.star_cmd_p2[bslot]);

                VkCommandBuffer cmd_p1 = new_plan.star_cmd_p1[bslot];
                VkCommandBufferBeginInfo bi_p1{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                vkBeginCommandBuffer(cmd_p1, &bi_p1);

                VkMemoryBarrier mb_pre{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                        VK_ACCESS_SHADER_WRITE_BIT,
                                        VK_ACCESS_SHADER_READ_BIT };
                vkCmdPipelineBarrier(cmd_p1, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0, 1, &mb_pre, 0, nullptr, 0, nullptr);

                uint64_t stage_bda_src = ggml_vk_tp5_get_tensor_bda(tensors[i]);
                uint64_t dst_bda = r.bda_addr[b];
                vkCmdBindPipeline(cmd_p1, VK_PIPELINE_BIND_POINT_COMPUTE, r.bda_push_pipe);

                uint32_t n_vec4 = (uint32_t)(n_elems / 4);
                uint64_t flag_bda = dst_bda + c.star_rank_stride - 64;

                // STAR/legacy fallback payload. F32 wire is a literal copy;
                // F16 wire uses the established in-flight conversion.
                struct {
                    uint64_t src_bda;
                    uint64_t dst_bda;
                    uint64_t flag_bda;
                    uint32_t n_vec4;
                    uint32_t rank_idx;
                    uint32_t seq_val;
                    uint32_t mode;
                } bda_pc{stage_bda_src, dst_bda, flag_bda, n_vec4, (uint32_t)n_elems, 1u,
                         c.wire == tp5_wire_type::F32 ? 5u : 0u};
                vkCmdBindPipeline(cmd_p1, VK_PIPELINE_BIND_POINT_COMPUTE, r.bda_push_pipe);
                vkCmdPushConstants(cmd_p1, r.bda_push_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bda_pc), &bda_pc);
                vkCmdDispatch(cmd_p1, (uint32_t)(n_elems + 255) / 256, 1, 1);

                // Hardware Pipeline Barrier: Enforces complete cache drain and PCIe visibility before flag write
                VkMemoryBarrier mb_host{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                         VK_ACCESS_SHADER_WRITE_BIT,
                                         VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
                vkCmdPipelineBarrier(cmd_p1, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0, 1, &mb_host, 0, nullptr, 0, nullptr);

                // Dispatch 2: Flag Write (Mode 1, guaranteed 100% data arrival in Host memory)
                bda_pc.mode = 1;
                vkCmdPushConstants(cmd_p1, r.bda_push_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bda_pc), &bda_pc);
                vkCmdDispatch(cmd_p1, 1, 1, 1);
                vkEndCommandBuffer(cmd_p1);

                VkCommandBuffer cmd_p2 = new_plan.star_cmd_p2[bslot];
                VkCommandBufferBeginInfo bi_p2{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                vkBeginCommandBuffer(cmd_p2, &bi_p2);

                VkBuffer dst_buf = VK_NULL_HANDLE;
                VkDeviceSize dst_off = 0, dst_size = 0;
                ggml_vk_tp5_tensor_dev_ref(tensors[i], &dst_buf, &dst_off, &dst_size);

                VkDescriptorSet ds_sum = r.star_ds_sum[b];
                const size_t bcast_bytes = (tensors[0]->type == GGML_TYPE_F32) ? n_elems * sizeof(float) : payload;

                if (tensors[0]->type == GGML_TYPE_F32) {
                    VkBufferCopy cp{0, dst_off, bcast_bytes};
                    vkCmdCopyBuffer(cmd_p2, r.bcast_buf[b], dst_buf, 1, &cp);
                } else {
                    tp5_update_star_sum_descriptor(r, ds_sum, r.bcast_buf[b], dst_buf, dst_off, dst_size, payload, (uint32_t)c.n_ranks);
                    vkCmdBindPipeline(cmd_p2, VK_PIPELINE_BIND_POINT_COMPUTE, r.sum_pipe);
                    vkCmdBindDescriptorSets(cmd_p2, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipe_layout, 0, 1, &ds_sum, 0, nullptr);
                    tp5_sum_pc pc_sum{(uint32_t) n_elems, 1, 0, 0};
                    vkCmdPushConstants(cmd_p2, r.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_sum), &pc_sum);
                    vkCmdDispatch(cmd_p2, (uint32_t)(n_elems + 255) / 256, 1, 1);
                }

                VkMemoryBarrier mb_post{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                         VK_ACCESS_TRANSFER_WRITE_BIT,
                                         VK_ACCESS_SHADER_READ_BIT };
                vkCmdPipelineBarrier(cmd_p2,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb_post, 0, nullptr, 0, nullptr);
                vkEndCommandBuffer(cmd_p2);
            }
        }
        for (size_t j = 0; j < c.n_ranks; ++j) {
            if (trefs[j].owner) new_plan.owners.push_back(trefs[j].owner);
        }
        if (c.cached_plans.size() >= tp5_comm::MAX_CACHED_PLANS) {
            c.evict_lru_plan();
        }
        c.cached_plans.push_back(std::move(new_plan));
        plan = &c.cached_plans.back();
        last_hit_idx = c.cached_plans.size() - 1;
    }

    // Assembly using persistent pre-recorded command buffers (Zero vkResetCommandBuffer, Zero recording!)
    for (size_t i = 0; i < c.n_ranks; ++i) {
        tp5_rank & r = c.ranks[i];
        const size_t bslot = tp5_plan_slot(i, bank);
        VkCommandBuffer cmd_p1 = plan->star_cmd_p1[bslot];
        VkCommandBuffer cmd_p2 = plan->star_cmd_p2[bslot];

        star_submits[i].rk = &r;
        star_submits[i].sig_p1 = 2 * epoch - 1;
        star_submits[i].wait_ready = epoch;
        star_submits[i].sig_p2 = 2 * epoch;
        star_submits[i].wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        star_submits[i].cmd_p1 = cmd_p1;
        star_submits[i].cmd_p2 = cmd_p2;

        star_submits[i].tsi[0] = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        star_submits[i].tsi[0].signalSemaphoreValueCount = 1;
        star_submits[i].tsi[0].pSignalSemaphoreValues = &star_submits[i].sig_p1;

        star_submits[i].submits[0] = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        star_submits[i].submits[0].pNext = &star_submits[i].tsi[0];
        star_submits[i].submits[0].commandBufferCount = 1;
        star_submits[i].submits[0].pCommandBuffers = &star_submits[i].cmd_p1;
        star_submits[i].submits[0].signalSemaphoreCount = 1;
        star_submits[i].submits[0].pSignalSemaphores = &r.timeline_sem;

        star_submits[i].tsi[1] = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        star_submits[i].tsi[1].waitSemaphoreValueCount = 0;
        star_submits[i].tsi[1].pWaitSemaphoreValues = nullptr;
        star_submits[i].tsi[1].signalSemaphoreValueCount = 1;
        star_submits[i].tsi[1].pSignalSemaphoreValues = &star_submits[i].sig_p2;

        star_submits[i].submits[1] = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        star_submits[i].submits[1].pNext = &star_submits[i].tsi[1];
        star_submits[i].submits[1].waitSemaphoreCount = 0;
        star_submits[i].submits[1].pWaitSemaphores = nullptr;
        star_submits[i].submits[1].pWaitDstStageMask = nullptr;
        star_submits[i].submits[1].commandBufferCount = 1;
        star_submits[i].submits[1].pCommandBuffers = &star_submits[i].cmd_p2;
        star_submits[i].submits[1].signalSemaphoreCount = 1;
        star_submits[i].submits[1].pSignalSemaphores = &r.timeline_sem;
    }
    auto t_rec_done = std::chrono::high_resolution_clock::now();

    auto t_sub_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < c.n_ranks; ++i) {
        tp5_rank & r = *star_submits[i].rk;
        if (vkQueueSubmit(r.queue, 1, &star_submits[i].submits[0], VK_NULL_HANDLE) != VK_SUCCESS) {
            c.fail("tp5_allreduce_star: merged submit failed on rank " + std::to_string(i));
            return false;
        }
        ggml_vk_tp5_mark_queue_submitted(c.backends[i]);
    }
    auto t_sub_done = std::chrono::high_resolution_clock::now();
    auto t_after_p1 = t_sub_done;

    tp5_star_times handoff;
    if (!tp5_star_handoff(c, bank, n_elems, &handoff)) {
        return false;
    }
    const auto t_after_bcast = std::chrono::high_resolution_clock::now();

    // =========================================================================
    // PILLAR 6: ATOMIC ASYNC SIGNAL HANDOFF (Wakes up GPU Phase 2 via timeline sem)
    // ZERO risk of GPU event hangs, completely thread-safe & monotonic!
    // =========================================================================
    // Phase 2 is submitted only after the CPU publishes the broadcast payload.
    {
        for (size_t i = 0; i < c.n_ranks; ++i) {
            tp5_rank & r = *star_submits[i].rk;
            if (vkQueueSubmit(r.queue, 1, &star_submits[i].submits[1], VK_NULL_HANDLE) != VK_SUCCESS) {
                c.fail("tp5_allreduce_star: Phase 2 submit failed on rank " + std::to_string(i));
                return false;
            }
            ggml_vk_tp5_mark_queue_submitted(c.backends[i]);
        }
    }
    auto t_after_p2 = std::chrono::high_resolution_clock::now();

    acc_flush_us += std::chrono::duration<double, std::micro>(t_flush_done - t_flush_start).count();
    acc_pure_sub_us += std::chrono::duration<double, std::micro>(t_sub_done - t_sub_start).count();
    acc_rec_us += std::chrono::duration<double, std::micro>(t_rec_done - t_rec_start).count();
    acc_p1_sub_us += std::chrono::duration<double, std::micro>(t_after_p1 - t_start).count();
    acc_wait_us   += handoff.wait_us;
    acc_avx2_us   += handoff.sum_us;
    acc_bcast_us  += handoff.broadcast_us;
    acc_p2_sub_us += std::chrono::duration<double, std::micro>(t_after_p2 - t_after_bcast).count();
    call_cnt++;

    if (call_cnt % 96 == 0 && ggml_tp5_profile_active()) {
        fprintf(stderr, "[tp5-star-step] calls=%llu elems=%zu flush_us=%.3f record_us=%.3f "
                        "p1_submit_us=%.3f flag_wait_us=%.3f sum_us=%.3f broadcast_us=%.3f "
                        "p2_submit_us=%.3f (host wall times; P2 completion excluded)\n",
                (unsigned long long) call_cnt, n_elems, acc_flush_us / call_cnt, acc_rec_us / call_cnt,
                acc_pure_sub_us / call_cnt, acc_wait_us / call_cnt, acc_avx2_us / call_cnt,
                acc_bcast_us / call_cnt, acc_p2_sub_us / call_cnt);
    }
return true;
}

bool tp5_allreduce_mesh(tp5_comm & c, ggml_tensor ** tensors, size_t n_elems) {
    if (c.failed) return false;
    if (n_elems == 0) return true;
    if (c.allreduce_calls >= (UINT64_MAX / 2 - 1)) {
        c.fail("epoch overflow guard triggered");
        return false;
    }

    // Idempotent: submit only real pending producers/uploads, including callers
    // using the registry directly rather than going through meta.
    for (auto backend : c.backends)
        ggml_vk_tp5_flush_async(backend);

    if (n_elems > c.max_elems) {
        if (!tp5_setup_workspace(c, n_elems)) return false;
    }

    const size_t wire_b = c.wire == tp5_wire_type::F16 ? 2 : 4;
    const VkDeviceSize stride = (VkDeviceSize) c.max_elems * wire_b;
    const VkDeviceSize tensor_bytes = (VkDeviceSize) n_elems * sizeof(float);
    const VkDeviceSize flags_base   = c.isolate_mailbox ? 0 : tp5_flags_byte_offset(c.n_ranks, stride);
    const uint32_t seq = (uint32_t) (c.allreduce_calls + 1);

    std::vector<tensor_dev_ref> trefs(c.n_ranks);
    for (size_t j = 0; j < c.n_ranks; ++j) {
        trefs[j] = tp5_tensor_dev_ref(tensors[j]);
        if (!trefs[j].ok) {
            c.fail("tensor " + std::to_string(j) + " is not in a Vulkan buffer");
            return false;
        }
        if (trefs[j].size < tensor_bytes) {
            c.fail("tensor " + std::to_string(j) + " smaller than payload");
            return false;
        }
        if (trefs[j].offset % std::max(uint64_t(4), c.ranks[j].caps.min_storage_buffer_offset_alignment) != 0 ||
            tensor_bytes > c.ranks[j].caps.max_storage_buffer_range) {
            c.fail("tensor binding exceeds storage-buffer limits on rank " + std::to_string(j));
            return false;
        }
    }

    tp5_plan_key key;
    key.n_elems = n_elems;
    key.wire = c.wire;
    key.stride = stride;
    key.workspace_gen = c.workspace_gen;
    key.bindings.resize(c.n_ranks);
    key.packed_bindings.resize(c.n_ranks);
    key.relay_direct.fill(false);
    size_t one_shot_relay_stage = SIZE_MAX;
    for (size_t j = 0; j < c.n_ranks; ++j) {
        if ((c.wire == tp5_wire_type::F16 &&
             (c.sync_mode == tp5_sync_mode::TIMELINE || c.sync_mode == tp5_sync_mode::RELAY)) ||
            (c.wire == tp5_wire_type::F32 && c.sync_mode == tp5_sync_mode::RELAY)) {
            uint64_t poff = 0, psize = 0;
            bool direct = false;
            size_t direct_stage = SIZE_MAX;
            if (ggml_vk_tp5_take_wire_output(c.backends[j], tensors[j], &trefs[j].packed_buf, &poff, &psize,
                                             &trefs[j].packed_owner, &direct, SIZE_MAX, &direct_stage)) {
                trefs[j].relay_direct = direct && c.sync_mode == tp5_sync_mode::RELAY;
                if (trefs[j].relay_direct) {
                    if (one_shot_relay_stage == SIZE_MAX)
                        one_shot_relay_stage = direct_stage;
                    if (direct_stage != one_shot_relay_stage)
                        trefs[j].relay_direct = false;
                } else if (c.wire == tp5_wire_type::F16) {
                    trefs[j].packed_offset            = (VkDeviceSize) poff;
                    trefs[j].packed_size              = (VkDeviceSize) psize;
                    const VkDeviceSize expected_bytes = ((VkDeviceSize) n_elems * 2 + 3) & ~VkDeviceSize(3);
                    if (!trefs[j].packed_buf || !trefs[j].packed_owner || n_elems % 2 != 0 ||
                        trefs[j].packed_size < expected_bytes ||
                        trefs[j].packed_offset %
                                std::max(uint64_t(4), c.ranks[j].caps.min_storage_buffer_offset_alignment) !=
                            0 ||
                        expected_bytes > c.ranks[j].caps.max_storage_buffer_range) {
                        trefs[j].packed_buf = VK_NULL_HANDLE;
                        trefs[j].packed_owner.reset();
                        trefs[j].packed_offset = 0;
                        trefs[j].packed_size   = 0;
                    }
                } else {
                    trefs[j].packed_buf = VK_NULL_HANDLE;
                    trefs[j].packed_owner.reset();
                    trefs[j].packed_offset = 0;
                    trefs[j].packed_size   = 0;
                }
            }
        }
        key.bindings[j].buf = trefs[j].buf;
        key.bindings[j].offset = trefs[j].offset;
        key.bindings[j].size = trefs[j].size;
        key.packed_bindings[j].buf    = trefs[j].packed_buf;
        key.packed_bindings[j].offset = trefs[j].packed_offset;
        key.packed_bindings[j].size   = trefs[j].packed_size;
        key.relay_direct[j]           = trefs[j].relay_direct;
    }
    const bool one_shot_direct = tp5_relay_direct_stage(key, c.n_ranks) && one_shot_relay_stage != SIZE_MAX;

    tp5_cached_plan * plan = nullptr;
    tp5_cached_plan one_shot;
    bool is_cached = false;
    bool plan_cache_hit = false;
    struct one_shot_retain_guard {
        tp5_comm & comm;
        tp5_cached_plan & plan;
        bool & cached;
        bool armed = false;

        ~one_shot_retain_guard() {
            if (armed && !cached) {
                comm.cached_plans.push_back(std::move(plan));
            }
        }
    } retain_one_shot{c, one_shot, is_cached};

    size_t & last_hit_idx = c.last_hit_idx;
    if (c.cmd_replay_enabled) {
        if (last_hit_idx < c.cached_plans.size() && c.cached_plans[last_hit_idx].key == key) {
            plan = &c.cached_plans[last_hit_idx];
            plan->last_used_call = c.allreduce_calls;
            is_cached = true;
            plan_cache_hit = true;
            if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
                prof->collective_plan_hits++;
            }
        } else for (size_t idx = 0; idx < c.cached_plans.size(); ++idx) {
            auto & p = c.cached_plans[idx];
            if (p.key == key) {
                last_hit_idx = idx;
                plan = &p;
                plan->last_used_call = c.allreduce_calls;
                is_cached = true;
                plan_cache_hit = true;
                if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
                    prof->collective_plan_hits++;
                }
                break;
            }
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    if (!plan) {
        if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
            prof->collective_plan_misses++;
        }
        tp5_cached_plan new_plan;
        new_plan.key            = key;
        new_plan.last_used_call = c.allreduce_calls;

        if (!tp5_record_plan(c, new_plan, trefs, n_elems, flags_base)) {
            c.destroy_plan(new_plan);
            return false;
        }

        if (c.cmd_replay_enabled) {
            if (c.cached_plans.size() >= tp5_comm::MAX_CACHED_PLANS) {
                c.evict_lru_plan();
                if (c.failed) {
                    c.destroy_plan(new_plan);
                    return false;
                }
            }
            c.cached_plans.push_back(std::move(new_plan));
            plan = &c.cached_plans.back();
            is_cached = true;
        } else {
            one_shot = std::move(new_plan);
            plan = &one_shot;
            is_cached = false;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    const uint64_t epoch = (uint64_t) (c.allreduce_calls + 1);

    // Bound retained resources independently of the two payload banks.
    const auto t_bp0 = t1;
    if ((c.sync_mode == tp5_sync_mode::TIMELINE || c.sync_mode == tp5_sync_mode::RELAY) &&
        c.allreduce_calls >= tp5_comm::MAX_OUTSTANDING_EPOCHS) {
        uint64_t drain_target = c.allreduce_calls + 1 - tp5_comm::MAX_OUTSTANDING_EPOCHS;
        if (drain_target > c.last_drained_epoch) {
            if (!tp5_drain_epoch(c, drain_target)) return false;
        }
    }
    // Maintain ring of in-flight buffer owners: drain slot's previous epoch before reusing,
    // then retain input buffer shared_ptrs so VkBuffer/VkDeviceMemory stay alive until P2 completes.
    if (c.sync_mode == tp5_sync_mode::TIMELINE || c.sync_mode == tp5_sync_mode::RELAY) {
        size_t ring_idx = (size_t)((epoch - 1) % tp5_comm::MAX_OUTSTANDING_EPOCHS);
        auto & slot = c.in_flight_ring[ring_idx];
        if (slot.epoch > 0) {
            if (!tp5_drain_epoch(c, slot.epoch)) return false;
        }
        slot.epoch = epoch;
        slot.owners.clear();
        slot.p1 = plan->p1;
        for (size_t j = 0; j < c.n_ranks; ++j) {
            if (trefs[j].owner)
                slot.owners.push_back(trefs[j].owner);
            if (trefs[j].packed_owner)
                slot.owners.push_back(trefs[j].packed_owner);
        }
    } else if (c.sync_mode == tp5_sync_mode::GPUFLAG) {
        size_t ring_idx = (size_t)((epoch - 1) % tp5_comm::MAX_OUTSTANDING_EPOCHS);
        auto & slot = c.in_flight_ring[ring_idx];
        if (slot.epoch > 0) {
            size_t drain_ring_idx = (size_t)((slot.epoch - 1) % tp5_comm::MAX_OUTSTANDING_EPOCHS);
            if (!tp5_gpuflag_drain_epoch(c, drain_ring_idx)) return false;
            slot.owners.clear();
            slot.p1.reset();
            slot.epoch = 0;
        }
        slot.epoch = epoch;
        slot.owners.clear();
        slot.p1 = plan->p1;
        for (size_t j = 0; j < c.n_ranks; ++j) {
            if (trefs[j].owner)
                slot.owners.push_back(trefs[j].owner);
            if (trefs[j].packed_owner)
                slot.owners.push_back(trefs[j].packed_owner);
        }
        tp5_update_epoch_bufs(c, seq);
    }

    const auto t_bp1 = std::chrono::high_resolution_clock::now();

    // Runtime active-row count for LateBind kernels (RELAY header word 5).
    // Non-predefined callers publish none; the late kernels then fall back to
    // their push-constant capacity (single-row legacy behavior).
    uint32_t one_shot_rows = 0;
    if (c.sync_mode == tp5_sync_mode::RELAY) {
        for (size_t i = 0; i < c.n_ranks; ++i) {
            uint32_t rows = 0, cap = 0;
            if (ggml_vk_tp5_predefined_rows(c.backends[i], &rows, &cap) && rows) {
                one_shot_rows = rows;
                break;
            }
        }
    }

    if (c.sync_mode == tp5_sync_mode::RELAY && !tp5_relay_ensure_armed_epoch(c, epoch)) {
        return false;
    }

    // Submit Phase 1 on all ranks
    retain_one_shot.armed = !is_cached;
    const bool p1_on_transfer = false;
    static const bool merge_submit = [] {
        const char * env = getenv("GGML_TP5_MERGE_SUBMIT");
        return env ? (atoi(env) != 0) : true;
    }();
    if (c.sync_mode == tp5_sync_mode::RELAY) {
        const size_t bank = tp5_mailbox_bank(epoch);
        volatile uint32_t * direct_ready[8] = {};
        // Publish the runtime active-row count (RELAY header word 5) before
        // this epoch's command buffers are submitted. Non-predefined callers
        // publish no frame; word 5 stays 0 and the late kernels fall back to
        // their push-constant capacity (single-row legacy behavior).
        {
            for (size_t i = 0; i < c.n_ranks; ++i) {
                if (c.ranks[i].bcast_host[bank]) {
                    ((volatile uint32_t *) c.ranks[i].bcast_host[bank])[5] = one_shot_rows;
                }
            }
            std::atomic_thread_fence(std::memory_order_release);
#if defined(__x86_64__) || defined(_M_X64)
            _mm_sfence();
#endif
        }
        if (one_shot_direct) {
            for (size_t i = 0; i < c.n_ranks; ++i) {
                if (!ggml_vk_tp5_get_relay_ready(c.backends[i], one_shot_relay_stage, &direct_ready[i]) ||
                    !direct_ready[i]) {
                    c.fail("RELAY one-shot direct ready route unavailable on rank " + std::to_string(i));
                    return false;
                }
            }
        }
        for (size_t i = 0; i < c.n_ranks; ++i) {
            auto * status = (volatile uint32_t *) ((char *) c.star_host_aligned[bank] +
                                                   i * c.star_rank_stride + c.star_rank_stride - 64);
            if (status[0] != 0u || status[1] != (uint32_t) epoch || status[2] != 0u || status[3] != 0u) {
                c.fail("RELAY mailbox status not idle on rank " + std::to_string(i) +
                       " (epoch=" + std::to_string(epoch) + ", expected=" + std::to_string(status[1]) +
                       ", flag=" + std::to_string(status[0]) + ", error=" + std::to_string(status[2]) +
                       ", done=" + std::to_string(status[3]) + ")");
                return false;
            }
        }
        // Direct producer already wrote/published the host payload in the
        // preceding model CB, so only P2 exists here. Fallback retains the
        // legacy P1+P2 pair. In both cases P2 is resident before CPU handoff.
        const uint64_t final_signal = 2 * epoch;
        for (size_t i = 0; i < c.n_ranks; ++i) {
            tp5_rank & r = c.ranks[i];
            const size_t bslot = tp5_plan_slot(i, bank);
            const VkCommandBuffer commands[2] = {
                one_shot_direct ? plan->cmd_p2[bslot] : plan->p1->cmd_p1[bslot],
                plan->cmd_p2[bslot],
            };
            VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
            timeline.signalSemaphoreValueCount = 1;
            timeline.pSignalSemaphoreValues    = &final_signal;
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.pNext                = &timeline;
            submit.commandBufferCount   = one_shot_direct ? 1u : 2u;
            submit.pCommandBuffers      = commands;
            submit.signalSemaphoreCount = 1;
            submit.pSignalSemaphores    = &r.timeline_sem;
            if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
                ++prof->queue_submits;
                ++prof->submit_batches;
            }
            if (vkQueueSubmit(r.queue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS) {
                tp5_relay_request_abort(c);
                c.fail("RELAY full one-shot submit failed on rank " + std::to_string(i));
                return false;
            }
            ggml_vk_tp5_mark_queue_submitted(c.backends[i]);
            tp5_note_timeline_submission(r, final_signal);
        }
    } else if (c.sync_mode == tp5_sync_mode::TIMELINE || c.sync_mode == tp5_sync_mode::DRM) {
        if (merge_submit && c.n_ranks <= 8) {
            // Merged 2-stage submit (AGENTS.md P2): submit Phase 1 + Phase 2 together in one
            // vkQueueSubmit per rank (2 VkSubmitInfo entries). Reduces ioctl/kernel submissions
            // by 50% without altering timeline dependencies (P2 waits on peers' P1 signals).
            tp5_timeline_batch b1[8];
            tp5_timeline_batch b2[8];
            for (size_t i = 0; i < c.n_ranks; ++i) {
                tp5_rank & r = c.ranks[i];
                const size_t bslot = tp5_plan_slot(i, tp5_mailbox_bank(epoch));
                b1[i].init(r, i, c.n_ranks, epoch, true, &plan->p1->cmd_p1[bslot]);
                b2[i].init(r, i, c.n_ranks, epoch, false, &plan->cmd_p2[bslot]);
                VkSubmitInfo submits[2] = { b1[i].submit, b2[i].submit };

                if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
                    prof->queue_submits++;
                    prof->submit_batches += 2;
                }
                if (vkQueueSubmit(r.queue, 2, submits, VK_NULL_HANDLE) != VK_SUCCESS) {
                    c.fail("Phase 1+2 merged timeline submit failed on rank " + std::to_string(i));
                    return false;
                }
                ggml_vk_tp5_mark_queue_submitted(c.backends[i]);
                tp5_note_timeline_submission(r, b2[i].signal);
            }
        } else {
            // Split submit: Phase 1 enqueued on all ranks first; Phase 2 enqueued in second loop below.
            for (size_t i = 0; i < c.n_ranks; ++i) {
                tp5_rank & r = c.ranks[i];
                const size_t bslot = tp5_plan_slot(i, tp5_mailbox_bank(epoch));
                tp5_timeline_batch batch;
                batch.init(r, i, c.n_ranks, epoch, true, &plan->p1->cmd_p1[bslot]);
                const VkSubmitInfo & si = batch.submit;

                if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
                    prof->queue_submits++;
                    prof->submit_batches++;
                }
                if (vkQueueSubmit(r.queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
                    c.fail("Phase 1 timeline submit failed on rank " + std::to_string(i));
                    return false;
                }
                tp5_note_timeline_submission(r, batch.signal);
            }
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    if (c.sync_mode == tp5_sync_mode::TIMELINE || c.sync_mode == tp5_sync_mode::DRM) {
        // Zero host wait, zero per-AR export/import, zero vkDeviceWaitIdle:
        // Phase 2 compute stages on each rank wait on peer timeline semaphores over PCIe.
    } else if (c.sync_mode == tp5_sync_mode::SYNCFD) {
        for (size_t src = 0; src < c.n_ranks; ++src) {
            tp5_rank & r_src = c.ranks[src];
            VkSemaphoreGetFdInfoKHR gfi{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
            gfi.semaphore = r_src.sem_p1_done;
            gfi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
            int sync_fd = -1;
            if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
                prof->fd_exports++;
            }
            VkResult res = r_src.pfn_get_sem_fd(r_src.vkdev, &gfi, &sync_fd);
            if (res != VK_SUCCESS) {
                c.fail("GetSemaphoreFdKHR failed on rank " + std::to_string(src));
                return false;
            }
            for (size_t dst = 0; dst < c.n_ranks; ++dst) {
                if (dst == src) continue;
                tp5_rank & r_dst = c.ranks[dst];
                size_t slot_idx = (src < dst) ? src : (src - 1);
                int fd_for_import = -1;
                if (sync_fd >= 0) {
                    fd_for_import = ::dup(sync_fd);
                    if (fd_for_import < 0) {
                        ::close(sync_fd);
                        c.fail("dup sync_fd failed for rank " + std::to_string(dst));
                        return false;
                    }
                }

                VkImportSemaphoreFdInfoKHR ifi{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR};
                ifi.semaphore = r_dst.wait_sems[slot_idx];
                ifi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
                ifi.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
                ifi.fd = fd_for_import;
                if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
                    prof->fd_imports++;
                }
                VkResult imp_res = r_dst.pfn_import_sem_fd(r_dst.vkdev, &ifi);
                if (imp_res != VK_SUCCESS) {
                    if (fd_for_import >= 0) {
                        ::close(fd_for_import);
                    }
                    if (sync_fd >= 0) {
                        ::close(sync_fd);
                    }
                    c.fail("ImportSemaphoreFdKHR failed on rank " + std::to_string(dst));
                    return false;
                }
            }
            if (sync_fd >= 0) {
                ::close(sync_fd);
            }
        }
    } else if (c.sync_mode == tp5_sync_mode::RELAY) {
        tp5_star_times relay_times;
        const uint64_t arm_next_epoch = epoch < UINT32_MAX ? epoch + 1 : 0;
        volatile uint32_t * direct_ready[8] = {};
        if (one_shot_direct) {
            for (size_t i = 0; i < c.n_ranks; ++i) {
                if (!ggml_vk_tp5_get_relay_ready(c.backends[i], one_shot_relay_stage, &direct_ready[i]) ||
                    !direct_ready[i]) {
                    c.fail("RELAY one-shot direct ready route disappeared on rank " + std::to_string(i));
                    return false;
                }
            }
        }
        if (!tp5_star_handoff(c, tp5_mailbox_bank(epoch), n_elems, &relay_times, true, arm_next_epoch,
                              one_shot_direct ? direct_ready : nullptr,
                              0, false, one_shot_rows)) {
            return false;
        }
    } else if (c.sync_mode != tp5_sync_mode::GPUFLAG) {
        if (!tp5_wait_all_p1(c)) {
            return false;
        }
    }
    if (c.sync_mode != tp5_sync_mode::TIMELINE && c.sync_mode != tp5_sync_mode::DRM &&
        c.sync_mode != tp5_sync_mode::GPUFLAG && c.sync_mode != tp5_sync_mode::RELAY) {
    tp5_p2p_visibility_barrier(c);
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    auto t4 = std::chrono::high_resolution_clock::now();

    // Submit Phase 2 on all ranks
    if (c.sync_mode == tp5_sync_mode::TIMELINE || c.sync_mode == tp5_sync_mode::DRM) {
        if (!merge_submit || c.n_ranks > 8) {
            // Split submit path only: under merged submit, Phase 2 was already submitted in the merged loop.
            for (size_t i = 0; i < c.n_ranks; ++i) {
                tp5_rank & r = c.ranks[i];
                const size_t       bslot = tp5_plan_slot(i, tp5_mailbox_bank(epoch));
                tp5_timeline_batch batch;
                batch.init(r, i, c.n_ranks, epoch, false, &plan->cmd_p2[bslot]);
                const VkSubmitInfo & si = batch.submit;

                if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
                    prof->queue_submits++;
                    prof->submit_batches++;
                }
                if (vkQueueSubmit(r.queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
                    // Do NOT call vkQueueWaitIdle on prior ranks: avoid deadlock on missing signals
                    c.fail("Phase 2 timeline submit failed on rank " + std::to_string(i));
                    return false;
                }
                ggml_vk_tp5_mark_queue_submitted(c.backends[i]);
                tp5_note_timeline_submission(r, batch.signal);
            }
        }
    } else if (c.sync_mode == tp5_sync_mode::RELAY) {
        // P2 is already resident behind the producer (direct) or P1 (fallback).
    } else if (c.sync_mode == tp5_sync_mode::GPUFLAG) {
        size_t ring_idx = (size_t)((epoch - 1) % tp5_comm::MAX_OUTSTANDING_EPOCHS);
        for (size_t i = 0; i < c.n_ranks; ++i) {
            tp5_rank & r = c.ranks[i];
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers    = &plan->cmd_p2[tp5_plan_slot(i, tp5_mailbox_bank(epoch))];
            vkResetFences(r.vkdev, 1, &r.fence_ring[ring_idx]);
            if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
                prof->queue_submits++;
                prof->submit_batches++;
            }
            if (vkQueueSubmit(r.queue, 1, &si, r.fence_ring[ring_idx]) != VK_SUCCESS) {
                c.fail("Phase 2 gpuflag submit failed on rank " + std::to_string(i));
                return false;
            }
        }
    } else {
    for (size_t i = 0; i < c.n_ranks; ++i) {
        tp5_rank & r = c.ranks[i];
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        VkPipelineStageFlags wait_stages[8] = {
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        };
        if (c.sync_mode == tp5_sync_mode::SYNCFD) {
            si.waitSemaphoreCount = (uint32_t) r.wait_sems.size();
            si.pWaitSemaphores = r.wait_sems.data();
            si.pWaitDstStageMask = wait_stages;
        }
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &plan->cmd_p2[tp5_plan_slot(i, tp5_mailbox_bank(epoch))];
        vkResetFences(r.vkdev, 1, &r.fence_p2);
        if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
            prof->queue_submits++;
            prof->submit_batches++;
        }
        if (vkQueueSubmit(r.queue, 1, &si, r.fence_p2) != VK_SUCCESS) {
            c.fail("Phase 2 submit failed on rank " + std::to_string(i));
            return false;
        }
    }

    if (!tp5_wait_all_p2(c)) {
        return false;
    }
    }

    auto t5 = std::chrono::high_resolution_clock::now();

    if (!is_cached) {
        if (c.sync_mode == tp5_sync_mode::TIMELINE || c.sync_mode == tp5_sync_mode::RELAY) {
            if (!tp5_drain_epoch(c, epoch)) {
                c.fail("one_shot plan drain failed before destruction");
                return false;
            }
        }
        retain_one_shot.armed = false;
        c.destroy_plan(one_shot);
    }

    static uint64_t total_p1_rec = 0, total_p1_sub = 0, total_p1_wait = 0, total_p2_rec = 0, total_p2_sub = 0, total_bp = 0;
    static int stat_count = 0;
    const auto bp_us = std::chrono::duration_cast<std::chrono::microseconds>(t_bp1 - t_bp0).count();
    total_bp      += bp_us;
    if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
        prof->backpressure_us += (uint64_t) bp_us;
        prof->collective_calls++;
    }
    total_p1_rec  += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    total_p1_sub  += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t_bp1).count();
    total_p1_wait += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    total_p2_rec  += std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    total_p2_sub  += std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
    stat_count++;

    static int tp5_profile_batch = 96;
    if (stat_count == 1) {
        const char * env = getenv("GGML_TP5_PROFILE_BATCH");
        if (env) {
            tp5_profile_batch = std::max(1, atoi(env));
        }
    }
    if (stat_count % tp5_profile_batch == 0) {
        if (c.sync_mode == tp5_sync_mode::TIMELINE) {
            fprintf(stderr, "[tp5-profile] %d allreduces (async timeline): "
                            "p1_rec=%.2f ms, p1_sub=%.2f ms, cpu_backpressure=%.2f ms, p2_rec=%.2f ms, p2_sub=%.2f ms | CPU SUBMIT+BP TOTAL=%.2f ms\n",
                    tp5_profile_batch,
                    total_p1_rec  / 1000.0,
                    total_p1_sub  / 1000.0,
                    total_bp      / 1000.0,
                    total_p2_rec  / 1000.0,
                    total_p2_sub  / 1000.0,
                    (total_p1_rec + total_p1_sub + total_bp + total_p2_rec + total_p2_sub) / 1000.0);
        } else {
        fprintf(stderr, "[tp5-profile] %d allreduces cost: "
                        "p1_rec=%.2f ms, p1_sub=%.2f ms, p1_wait=%.2f ms, p2_rec=%.2f ms, p2_sub=%.2f ms | COLLECTIVE TOTAL=%.2f ms\n",
                tp5_profile_batch,
                total_p1_rec  / 1000.0,
                total_p1_sub  / 1000.0,
                total_p1_wait / 1000.0,
                total_p2_rec  / 1000.0,
                total_p2_sub  / 1000.0,
                (total_p1_rec + total_bp + total_p1_sub + total_p1_wait + total_p2_rec + total_p2_sub) / 1000.0);
        }
        total_p1_rec = total_p1_sub = total_p1_wait = total_p2_rec = total_p2_sub = total_bp = 0;
    }

    c.allreduce_calls++;
    return true;
}

} // namespace

// Timestamp boundaries deliberately serialize adjacent regions. This measures
// a diagnostic chain, not an uninstrumented throughput run. No new host wait or
// semaphore is introduced, and no clocks from different devices are compared.
static bool tp5_prepare_gpu_timing(tp5_comm & c, size_t n_stages) {
    const uint32_t count = (uint32_t) (5 * n_stages + 3);
    for (auto & r : c.ranks) {
        if (!r.caps.timestamp_valid_bits || r.caps.timestamp_period <= 0)
            return false;
        VkQueryPoolCreateInfo qi{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        qi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = count;
        if (vkCreateQueryPool(r.vkdev, &qi, nullptr, &r.timing_pool) != VK_SUCCESS)
            return false;
        r.timing_markers.resize(count);
        VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        ai.commandPool        = r.cmd_pool;
        ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = count;
        if (vkAllocateCommandBuffers(r.vkdev, &ai, r.timing_markers.data()) != VK_SUCCESS)
            return false;
        for (uint32_t q = 0; q < count; ++q) {
            VkCommandBuffer          cb = r.timing_markers[q];
            VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS)
                return false;
            if (q == 0)
                vkCmdResetQueryPool(cb, r.timing_pool, 0, count);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, r.timing_pool, q);
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                 nullptr, 0, nullptr, 0, nullptr);
            if (vkEndCommandBuffer(cb) != VK_SUCCESS)
                return false;
        }
    }
    c.timing_stages = n_stages;
    return true;
}

static void tp5_poll_gpu_timing(tp5_comm & c) {
    if (!c.timing_submitted)
        return;
    const uint32_t count = (uint32_t) (5 * c.timing_stages + 3);

    struct sample {
        uint64_t ticks;
        uint64_t available;
    };

    for (size_t rank = 0; rank < c.n_ranks; ++rank) {
        auto & r = c.ranks[rank];
        if (r.timing_reported)
            continue;
        std::vector<sample> samples(count);
        const VkResult      result =
            vkGetQueryPoolResults(r.vkdev, r.timing_pool, 0, count, samples.size() * sizeof(sample), samples.data(),
                                  sizeof(sample), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (result == VK_NOT_READY)
            continue;
        if (result != VK_SUCCESS) {
            fprintf(stderr, "[tp5-gpu-timing] rank=%zu query error=%d\n", rank, (int) result);
            r.timing_reported = true;
            continue;
        }
        if (std::any_of(samples.begin(), samples.end(), [](const sample & s) { return !s.available; }))
            continue;
        const uint64_t mask =
            r.caps.timestamp_valid_bits >= 64 ? UINT64_MAX : (uint64_t(1) << r.caps.timestamp_valid_bits) - 1;
        const auto us = [&](size_t begin, size_t end) {
            return double((samples[end].ticks - samples[begin].ticks) & mask) * r.caps.timestamp_period / 1000.0;
        };
        double sum = 0, compute = 0, push = 0, gaps = 0, push_gap = 0;
        double relay_p1 = 0, relay_late_pre = 0;
        for (size_t s = 0; s < c.timing_stages; ++s) {
            const size_t q   = 5 * s;
            const double gap = s ? us(q - 1, q) : 0;
            const bool late_stage =
                s < c.timing_late_stage.size() && c.timing_late_stage[s] != 0;
            const bool direct_stage =
                s < c.timing_direct_stage.size() && c.timing_direct_stage[s] != 0;
            const double slot_a = us(q + 2, q + 3);
            const double slot_b = us(q + 3, q + 4);
            const double relay_p1_us = direct_stage ? 0.0 : (late_stage ? slot_a : slot_b);
            const double relay_late_us = late_stage ? slot_b : 0.0;
            sum += us(q, q + 1);
            compute += us(q + 1, q + 2);
            push_gap += slot_a;
            push += slot_b;
            relay_p1 += relay_p1_us;
            relay_late_pre += relay_late_us;
            gaps += gap;
            fprintf(stderr,
                    "[tp5-gpu-stage] rank=%zu stage=%zu gap_us=%.3f sum_us=%.3f compute_us=%.3f push_us=%.3f "
                    "push_gap_us=%.3f\n",
                    rank, s, gap, us(q, q + 1), us(q + 1, q + 2), us(q + 3, q + 4), us(q + 2, q + 3));
            if (c.sync_mode == tp5_sync_mode::RELAY) {
                fprintf(stderr,
                        "[tp5-relay-gpu-stage] rank=%zu stage=%zu p2_us=%.3f compute_us=%.3f "
                        "p1_us=%.3f late_pre_us=%.3f direct=%d late=%d gap_us=%.3f\n",
                        rank, s, us(q, q + 1), us(q + 1, q + 2), relay_p1_us,
                        relay_late_us, direct_stage ? 1 : 0, late_stage ? 1 : 0, gap);
            }
        }
        sum += us(count - 3, count - 2);
        const double tail_compute = us(count - 2, count - 1);
        compute += tail_compute;
        gaps += us(count - 4, count - 3);
        fprintf(stderr,
                "[tp5-gpu-timing] rank=%zu stages=%zu span_us=%.3f sum_us=%.3f compute_us=%.3f push_us=%.3f "
                "gaps_us=%.3f tail_compute_us=%.3f push_gap_us=%.3f\n",
                rank, c.timing_stages, us(0, count - 1), sum, compute, push, gaps, tail_compute, push_gap);
        if (c.sync_mode == tp5_sync_mode::RELAY) {
            fprintf(stderr,
                    "[tp5-relay-gpu] rank=%zu stages=%zu span_us=%.3f p2_us=%.3f compute_us=%.3f "
                    "p1_us=%.3f late_pre_us=%.3f gaps_us=%.3f tail_compute_us=%.3f\n",
                    rank, c.timing_stages, us(0, count - 1), sum, compute, relay_p1, relay_late_pre, gaps,
                    tail_compute);
        }
        r.timing_reported = true;
    }
}

bool ggml_backend_vk_tp5_submit_epoch_chain(void * comm_handle,
                                            const std::vector<std::vector<std::vector<void *>>> & stage_compute_cbs,
                                            const std::vector<std::vector<ggml_tensor *>> & stage_tensors,
                                            const ggml_predefined_frame * frame,
                                            uint32_t capacity_rows, uint32_t capacity_outputs) {
    if (!comm_handle) return false;
    tp5_comm & c = *reinterpret_cast<tp5_comm *>(comm_handle);
    std::lock_guard<std::mutex> lock(c.mutex);
    if (c.failed) {
        return false;
    }
    if ((c.sync_mode != tp5_sync_mode::TIMELINE && c.sync_mode != tp5_sync_mode::DRM &&
         c.sync_mode != tp5_sync_mode::STAR && c.sync_mode != tp5_sync_mode::RELAY) || !c.cmd_replay_enabled)
        return false;

    const size_t n_stages = stage_tensors.size();
    const uint32_t active_rows = frame ? frame->active_tokens : 0;
    if (frame) {
        if (frame->version != GGML_PREDEFINED_ABI_VERSION || frame->phase >= GGML_PREDEFINED_PHASE_COUNT ||
            frame->active_sequences == 0 || active_rows == 0 || capacity_rows == 0 || capacity_outputs == 0 ||
            active_rows > capacity_rows || frame->active_outputs > capacity_outputs) {
            return false;
        }
    } else if (capacity_rows != 0 || capacity_outputs != 0) {
        return false;
    }
    const bool predefined_rows = frame != nullptr;
    // N reductions have N+1 compute groups; the final group follows the last
    // SUM and may be empty for collective-only callers. Never drop tail work.
    if (n_stages == 0 || n_stages > tp5_comm::MAX_OUTSTANDING_EPOCHS || stage_compute_cbs.size() != n_stages + 1)
        return false;
    if (n_stages > UINT64_MAX / 2 - 1 - c.allreduce_calls)
        return false;
    for (const auto & stage : stage_compute_cbs) {
        if (stage.size() != c.n_ranks)
            return false;
        for (const auto & rank : stage) {
            for (void * cb : rank)
                if (!cb)
                    return false;
        }
    }

    tp5_poll_gpu_timing(c);

    // Validate the ENTIRE chain before any submission or workspace mutation.
    // In particular, a later larger tensor must not destroy an earlier plan.
    size_t                      max_elems = c.max_elems;
    c.chain_refs.resize(n_stages * c.n_ranks);
    c.chain_keys.resize(n_stages);
    c.chain_plan_indices.resize(n_stages);
    auto & refs = c.chain_refs;
    auto & keys = c.chain_keys;
    std::vector<size_t> active_elems(n_stages);
    for (size_t s = 0; s < n_stages; ++s) {
        if (stage_tensors[s].size() != c.n_ranks)
            return false;
        auto & key = keys[s];
        key.wire   = c.wire;
        key.stride        = 0;
        key.workspace_gen = 0;
        key.hc.fill(tp5_hc_key{});
        key.late.fill(tp5_hc_key{});
        key.relay_direct.fill(false);
        key.bindings.resize(c.n_ranks);
        key.packed_bindings.resize(c.n_ranks);
        for (size_t j = 0; j < c.n_ranks; ++j) {
            ggml_tensor * t = stage_tensors[s][j];
            if (!t || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || ggml_nelements(t) <= 0)
                return false;
            const size_t n_elems = (size_t) ggml_nelements(t);
            if (j == 0)
                key.n_elems = n_elems;
            if (n_elems != key.n_elems)
                return false;
            auto & ref = refs[s * c.n_ranks + j];
            ref        = tp5_tensor_dev_ref(t);
            if (!ref.ok || ref.size < ggml_nbytes(t))
                return false;
            if (ref.offset % std::max(uint64_t(4), c.ranks[j].caps.min_storage_buffer_offset_alignment) != 0)
                return false;
            if (ggml_nbytes(t) > c.ranks[j].caps.max_storage_buffer_range)
                return false;
            if (c.sync_mode != tp5_sync_mode::STAR &&
                (c.wire == tp5_wire_type::F16 || c.sync_mode == tp5_sync_mode::RELAY)) {
                uint64_t poff = 0, psize = 0;
                bool direct = false;
                if (ggml_vk_tp5_take_wire_output(c.backends[j], t, &ref.packed_buf, &poff, &psize,
                                                 &ref.packed_owner, &direct, s)) {
                    ref.relay_direct = direct && c.sync_mode == tp5_sync_mode::RELAY;
                    if (!ref.relay_direct && c.wire == tp5_wire_type::F16) {
                        ref.packed_offset                 = (VkDeviceSize) poff;
                        ref.packed_size                   = (VkDeviceSize) psize;
                        const VkDeviceSize expected_bytes = ((VkDeviceSize) n_elems * 2 + 3) & ~VkDeviceSize(3);
                        if (!ref.packed_buf || !ref.packed_owner || n_elems % 2 != 0 ||
                            ref.packed_size < expected_bytes ||
                            ref.packed_offset %
                                    std::max(uint64_t(4), c.ranks[j].caps.min_storage_buffer_offset_alignment) != 0 ||
                            expected_bytes > c.ranks[j].caps.max_storage_buffer_range) {
                            ref.packed_buf = VK_NULL_HANDLE;
                            ref.packed_owner.reset();
                            ref.packed_offset = 0;
                            ref.packed_size   = 0;
                        }
                    } else if (!ref.relay_direct) {
                        ref.packed_buf = VK_NULL_HANDLE;
                        ref.packed_owner.reset();
                        ref.packed_offset = 0;
                        ref.packed_size   = 0;
                    }
                }
            }
            key.bindings[j]        = { ref.buf, ref.offset, ref.size };
            key.packed_bindings[j] = { ref.packed_buf, ref.packed_offset, ref.packed_size };
            key.relay_direct[j]    = ref.relay_direct;
            const auto & consumer  = stage_compute_cbs[s + 1][j];
            if (!consumer.empty() && c.sync_mode == tp5_sync_mode::RELAY && tp5_latebind_hc_enabled()) {
                tp5_late_consumer_ref(c, j, consumer.front(), ref, n_elems, key.late[j]);
            }
            if (consumer.size() > 1 && c.sync_mode != tp5_sync_mode::STAR) {
                tp5_hc_consumer_ref(c, j, consumer.front(), ref, n_elems, key.hc[j]);
            }
            max_elems = std::max(max_elems, n_elems);
        }
        if (!predefined_rows) {
            active_elems[s] = key.n_elems;
        } else {
            if (capacity_rows == 0 || key.n_elems % capacity_rows != 0) {
                return false;
            }
            const size_t width = key.n_elems / capacity_rows;
            if (width == 0 || active_rows > SIZE_MAX / width) {
                return false;
            }
            active_elems[s] = width * active_rows;
            if (active_elems[s] == 0 || active_elems[s] > key.n_elems) {
                return false;
            }
        }
        bool late_all = tp5_latebind_hc_enabled() && c.sync_mode == tp5_sync_mode::RELAY;
        for (size_t j = 0; j < c.n_ranks; ++j) {
            late_all = late_all && key.late[j].late_rank != 0 &&
                       key.late[j].late_rank == key.late[0].late_rank &&
                       key.late[j].streams == key.late[0].streams &&
                       key.late[j].width == key.late[0].width &&
                       key.late[j].epsilon_bits == key.late[0].epsilon_bits;
        }
        if (late_all) {
            // The exact LateBind finalizer subsumes the old HC prefix. Keep a
            // single replacement recipe so command-buffer skipping is uniform
            // across all five ranks.
            key.hc.fill(tp5_hc_key{});
            for (size_t j = 0; j < c.n_ranks; ++j)
                refs[s * c.n_ranks + j].hc = {};
        } else {
            key.late.fill(tp5_hc_key{});
            for (size_t j = 0; j < c.n_ranks; ++j)
                refs[s * c.n_ranks + j].late = {};
        }
    }
    if (max_elems > c.max_elems && !tp5_setup_workspace(c, max_elems)) {
        return false;
    }
    const size_t       wire_b     = c.wire == tp5_wire_type::F16 ? 2 : 4;
    const VkDeviceSize stride     = (VkDeviceSize) c.max_elems * wire_b;
    const VkDeviceSize flags_base = c.isolate_mailbox ? 0 : tp5_flags_byte_offset(c.n_ranks, stride);
    for (size_t s = 0; s < n_stages; ++s) {
        auto & key        = keys[s];
        key.stride = stride;
        key.workspace_gen = c.workspace_gen;
    }

    // Plan resolution: try fast index reuse if plans_gen matches, else search/insert once
    for (size_t s = 0; s < n_stages; ++s) {
        const auto & key       = keys[s];
        size_t       found_idx = SIZE_MAX;
        bool         is_hit    = false;

        // Fast path: if previous compilation had this stage index under the same plans_gen
        if (c.compiled_chain.valid && c.compiled_chain.plans_gen == c.plans_gen &&
            s < c.compiled_chain.stage_plan_indices.size()) {
            const size_t prev_idx = c.compiled_chain.stage_plan_indices[s];
            if (prev_idx < c.cached_plans.size() && c.cached_plans[prev_idx].key == key) {
                found_idx = prev_idx;
                is_hit    = true;
            }
        }

        if (found_idx == SIZE_MAX) {
            for (size_t p = 0; p < c.cached_plans.size(); ++p) {
                if (c.cached_plans[p].key == key) {
                    found_idx = p;
                    is_hit    = true;
                    break;
                }
            }
        }

        if (found_idx != SIZE_MAX) {
            c.chain_plan_indices[s] = found_idx;
        } else {
            if (c.cached_plans.size() >= tp5_comm::MAX_CACHED_PLANS) {
                c.clear_cached_plans();
                if (c.failed) {
                    return false;
                }
                s = (size_t) -1;
                continue;
            }
            tp5_cached_plan new_p;
            new_p.key = key;
            new_p.cmd_p2.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
            new_p.ds_sum.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
            std::vector<tensor_dev_ref> stage_refs(refs.begin() + s * c.n_ranks, refs.begin() + (s + 1) * c.n_ranks);
            if (!tp5_record_plan(c, new_p, stage_refs, new_p.key.n_elems, flags_base)) {
                c.destroy_plan(new_p);
                return false;
            }
            found_idx = c.cached_plans.size();
            c.cached_plans.push_back(std::move(new_p));
            c.chain_plan_indices[s] = found_idx;
            is_hit                  = false;
        }
        if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
            if (is_hit) {
                ++prof->collective_plan_hits;
            } else
                ++prof->collective_plan_misses;
        }
        c.cached_plans[found_idx].last_used_call = c.allreduce_calls + s + 1;
    }

    const uint64_t last_epoch = c.allreduce_calls + n_stages;
    const uint64_t retire_before =
        last_epoch > tp5_comm::MAX_OUTSTANDING_EPOCHS ? last_epoch - tp5_comm::MAX_OUTSTANDING_EPOCHS : 0;
    const auto bp_start = std::chrono::steady_clock::now();
    if (!tp5_drain_epoch(c, retire_before)) {
        return false;
    }
    if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
        prof->backpressure_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - bp_start).count();
    }

    // Build all ranks before submitting any rank; allocations and pointer
    // fixups cannot fail halfway through an inter-device dependency chain.
    size_t n_direct_stages = 0;
    size_t n_late_stages   = 0;
    if (c.sync_mode == tp5_sync_mode::RELAY) {
        for (size_t s = 0; s < n_stages; ++s) {
            if (tp5_relay_direct_stage(keys[s], c.n_ranks)) {
                ++n_direct_stages;
            }
            if (tp5_late_stage(keys[s], c.n_ranks)) {
                ++n_late_stages;
            }
        }
    }
    const bool capture_requested = ++c.chain_calls == c.timing_chain;
    const bool capture           = capture_requested && tp5_prepare_gpu_timing(c, n_stages);
    if (capture) {
        c.timing_late_stage.assign(n_stages, 0);
        c.timing_direct_stage.assign(n_stages, 0);
        for (size_t s = 0; s < n_stages; ++s) {
            c.timing_late_stage[s] = tp5_late_stage(keys[s], c.n_ranks) ? 1u : 0u;
            c.timing_direct_stage[s] = tp5_relay_direct_stage(keys[s], c.n_ranks) ? 1u : 0u;
        }
    }
    if (capture_requested && !capture) {
        fprintf(stderr, "[tp5-gpu-timing] capture unavailable; submitting original chain\n");
    }
    const bool isolate_bo = c.isolate_mailbox && c.sync_mode != tp5_sync_mode::STAR && c.sync_mode != tp5_sync_mode::RELAY;
    const char * chain_cache_env   = getenv("GGML_TP5_CHAIN_CACHE");
    // This reuses only the submission layout after the caller rebuilt and
    // compared current stage command handles, plan keys, and generations; it
    // is not a topology lock or a graph-validation bypass. RELAY needs the
    // same reusable full-chain layout to keep every P2 prequeued.
    const bool   allow_chain_cache = chain_cache_env == nullptr || atoi(chain_cache_env) != 0;
    const bool linear = c.sync_mode == tp5_sync_mode::RELAY && c.wire == tp5_wire_type::F32 &&
                        tp5_latebind_hc_enabled();
    if (!linear && n_late_stages > 0) {
        c.fail("LateBind stage requested in non-linear submit_epoch_chain; LateBind requires RELAY F32 linear program lowering");
        return false;
    }

    bool can_reuse_chain =
        allow_chain_cache && !capture && c.compiled_chain.valid && c.compiled_chain.n_stages == n_stages &&
        c.compiled_chain.isolate_bo == isolate_bo &&
        c.compiled_chain.start_bank == tp5_mailbox_bank(c.allreduce_calls + 1) &&
        c.compiled_chain.workspace_gen == c.workspace_gen && c.compiled_chain.plans_gen == c.plans_gen &&
        c.compiled_chain.stage_plan_indices == c.chain_plan_indices && c.compiled_chain.stage_keys == keys &&
        c.compiled_chain.stage_compute_cbs == stage_compute_cbs;

    if (linear) {
        if (!can_reuse_chain || !c.linear_program) {
            c.compiled_chain.invalidate();
            if (!tp5_define_linear_chain(c, stage_compute_cbs, n_stages, capture)) return false;
            if (allow_chain_cache && !capture) {
                c.compiled_chain.valid = true;
                c.compiled_chain.n_stages = n_stages;
                c.compiled_chain.isolate_bo = isolate_bo;
                c.compiled_chain.start_bank = tp5_mailbox_bank(c.allreduce_calls + 1);
                c.compiled_chain.workspace_gen = c.workspace_gen;
                c.compiled_chain.plans_gen = c.plans_gen;
                c.compiled_chain.stage_plan_indices = c.chain_plan_indices;
                c.compiled_chain.stage_keys = keys;
                c.compiled_chain.stage_compute_cbs = stage_compute_cbs;
            }
        }
        // The only changing execution input is the already-existing epoch
        // mailbox. There are no per-stage CB batches to patch here.
    } else if (can_reuse_chain) {
        // Warm path: patch only epoch values
        for (size_t i = 0; i < c.n_ranks; ++i) {
            auto & scratch = c.chain_scratch[i];
            for (size_t s = 0; s < n_stages; ++s) {
                const uint64_t epoch = c.allreduce_calls + s + 1;
                if (isolate_bo && s > 0) {
                    scratch.batches[2 * s].patch_epoch(epoch - 1, 2 * (epoch - 1));
                    scratch.batches[2 * s + 1].patch_epoch(0, 2 * epoch - 1);
                } else {
                    scratch.batches[s].patch_epoch(s > 0 ? epoch - 1 : 0, 2 * epoch - 1);
                }
            }
            if (isolate_bo) {
                scratch.batches[2 * n_stages].patch_epoch(last_epoch, 2 * last_epoch);
            } else {
                scratch.batches[n_stages].patch_epoch(last_epoch, 2 * last_epoch);
            }
        }
    } else {
        // Invalidate compiled chain before rebuilding scratch
        c.compiled_chain.invalidate();

        // Cold / Rebuild path
        for (size_t i = 0; i < c.n_ranks; ++i) {
            auto & scratch = c.chain_scratch[i];
            scratch.batches.resize((isolate_bo ? 2 : 1) * (n_stages + 1));
            scratch.submits.clear();
            scratch.submits.reserve(scratch.batches.size());
            size_t n_compute = 0;
            for (const auto & stage : stage_compute_cbs)
                n_compute += stage[i].size();
            for (size_t s = 0; s < n_stages; ++s) {
                const auto & stage_key = c.cached_plans[c.chain_plan_indices[s]].key;
                // LateBind stages are strictly lowered via tp5_define_linear_chain.
                // It is impossible for a late stage to reach this non-linear rebuild path.
                // Fail-closed immediately if any late key is unexpectedly observed.
                if (stage_key.late[i].late_rank) {
                    c.fail("LateBind stage key detected in non-linear epoch chain rebuild; LateBind requires linear program lowering");
                    return false;
                } else if (stage_key.hc[i].width) {
                    --n_compute;
                }
            }
            scratch.compute.resize(n_compute + 2 * n_stages - n_direct_stages +
                                   (capture ? 5 * n_stages + 3 : 0));
            const auto append_segment = [&](size_t slot, size_t begin, size_t end, uint64_t wait_epoch,
                                            uint64_t signal) {
                if (begin == end)
                    return;
                auto & batch = scratch.batches[slot];
                batch.init(c.ranks[i], i, c.n_ranks, wait_epoch ? wait_epoch : 1, wait_epoch == 0,
                           scratch.compute.data() + begin, c.sync_mode == tp5_sync_mode::STAR,
                           c.sync_mode == tp5_sync_mode::RELAY);
                batch.signal                             = signal;
                batch.timeline.signalSemaphoreValueCount = signal ? 1 : 0;
                batch.submit.signalSemaphoreCount        = signal ? 1 : 0;
                batch.submit.commandBufferCount          = (uint32_t) (end - begin);
                scratch.submits.push_back(batch.submit);
            };
            size_t cursor = 0;
            for (size_t s = 0; s < n_stages; ++s) {
                const uint64_t epoch = c.allreduce_calls + s + 1;
                const size_t   bslot = tp5_plan_slot(i, tp5_mailbox_bank(epoch));
                const size_t   first = cursor;
                if (capture)
                    scratch.compute[cursor++] = c.ranks[i].timing_markers[5 * s];
                if (s > 0) {
                    const size_t previous_slot = tp5_plan_slot(i, tp5_mailbox_bank(epoch - 1));
                    scratch.compute[cursor++]  = c.cached_plans[c.chain_plan_indices[s - 1]].cmd_p2[previous_slot];
                }
                if (capture)
                    scratch.compute[cursor++] = c.ranks[i].timing_markers[5 * s + 1];
                size_t first_compute = 0;
                if (s > 0) {
                    const auto & prev_key = c.cached_plans[c.chain_plan_indices[s - 1]].key;
                    if (prev_key.late[i].late_rank) {
                        c.fail("LateBind stage key detected in non-linear compute offset");
                        return false;
                    }
                    first_compute = prev_key.hc[i].width ? 1 : 0;
                }
                for (size_t cb = first_compute; cb < stage_compute_cbs[s][i].size(); ++cb) {
                    scratch.compute[cursor++] = (VkCommandBuffer) stage_compute_cbs[s][i][cb];
                }
                if (capture)
                    scratch.compute[cursor++] = c.ranks[i].timing_markers[5 * s + 2];
                const size_t compute_end = cursor;
                const auto & current_plan = c.cached_plans[c.chain_plan_indices[s]];
                if (current_plan.key.late[i].late_rank != 0) {
                    c.fail("LateBind plan detected in non-linear submit_epoch_chain");
                    return false;
                }
                const bool direct_stage =
                    c.sync_mode == tp5_sync_mode::RELAY &&
                    tp5_relay_direct_stage(current_plan.key, c.n_ranks);
                if (capture)
                    scratch.compute[cursor++] = c.ranks[i].timing_markers[5 * s + 3];
                if (!direct_stage) {
                    scratch.compute[cursor++] = current_plan.p1->cmd_p1[bslot];
                }
                if (capture)
                    scratch.compute[cursor++] = c.ranks[i].timing_markers[5 * s + 4];

                if (isolate_bo && s > 0) {
                    // A real completed-SUM signal stops Mesa from merging this
                    // local-only job back into the following imported-BO PUSH.
                    // No extra host wait or queue API call is introduced.
                    append_segment(2 * s, first, compute_end, epoch - 1, 2 * (epoch - 1));
                    append_segment(2 * s + 1, compute_end, cursor, 0, 2 * epoch - 1);
                    continue;
                }
                auto & batch = scratch.batches[s];
                // Direct RELAY: SUM(s-1) -> compute+host-payload(s), with no
                // P1 command. Fallback modes append P1 as before.
                batch.init(c.ranks[i], i, c.n_ranks, s > 0 ? epoch - 1 : epoch, s == 0, scratch.compute.data() + first,
                           c.sync_mode == tp5_sync_mode::STAR, c.sync_mode == tp5_sync_mode::RELAY);
                batch.signal                    = 2 * epoch - 1;
                batch.submit.commandBufferCount = (uint32_t) (cursor - first);
                scratch.submits.push_back(batch.submit);
            }
            // The final SUM and dependent model tail publish completion together.
            // Intermediate P2(e)
            // is covered by the following ready value 2*(e+1)-1 > 2*e, so existing
            // epoch retirement waits remain valid without a signal-only batch.
            const size_t last_slot   = tp5_plan_slot(i, tp5_mailbox_bank(last_epoch));
            auto &       final       = scratch.batches[n_stages];
            const size_t final_first = cursor;
            if (capture)
                scratch.compute[cursor++] = c.ranks[i].timing_markers[5 * n_stages];
            scratch.compute[cursor++] = c.cached_plans[c.chain_plan_indices.back()].cmd_p2[last_slot];
            if (capture)
                scratch.compute[cursor++] = c.ranks[i].timing_markers[5 * n_stages + 1];
            const auto & last_key = c.cached_plans[c.chain_plan_indices.back()].key;
            if (last_key.late[i].late_rank) {
                c.fail("LateBind tail stage detected in non-linear submit_epoch_chain");
                return false;
            }
            const size_t first_tail = last_key.hc[i].width ? 1 : 0;
            for (size_t cb = first_tail; cb < stage_compute_cbs.back()[i].size(); ++cb) {
                scratch.compute[cursor++] = (VkCommandBuffer) stage_compute_cbs.back()[i][cb];
            }
            if (capture)
                scratch.compute[cursor++] = c.ranks[i].timing_markers[5 * n_stages + 2];
            GGML_ASSERT(cursor == scratch.compute.size());
            if (isolate_bo) {
                append_segment(2 * n_stages, final_first, cursor, last_epoch, 2 * last_epoch);
                continue;
            }
            final.init(c.ranks[i], i, c.n_ranks, last_epoch, false, scratch.compute.data() + final_first,
                       c.sync_mode == tp5_sync_mode::STAR, c.sync_mode == tp5_sync_mode::RELAY);
            final.submit.commandBufferCount = (uint32_t) (cursor - final_first);
            scratch.submits.push_back(final.submit);
        }

        if (allow_chain_cache && !capture) {
            c.compiled_chain.valid              = true;
            c.compiled_chain.n_stages           = n_stages;
            c.compiled_chain.isolate_bo         = isolate_bo;
            c.compiled_chain.start_bank         = tp5_mailbox_bank(c.allreduce_calls + 1);
            c.compiled_chain.workspace_gen      = c.workspace_gen;
            c.compiled_chain.plans_gen          = c.plans_gen;
            c.compiled_chain.stage_plan_indices = c.chain_plan_indices;
            c.compiled_chain.stage_keys         = keys;
            c.compiled_chain.stage_compute_cbs  = stage_compute_cbs;
        }
    }

    // Retain EVERY rank before submission; ring reclamation above ensures
    // these assignments cannot drop owners referenced by pending commands.
    for (size_t s = 0; s < n_stages; ++s) {
        const uint64_t epoch = c.allreduce_calls + s + 1;
        auto &         slot  = c.in_flight_ring[(epoch - 1) % tp5_comm::MAX_OUTSTANDING_EPOCHS];
        GGML_ASSERT(slot.epoch == 0 || slot.epoch <= c.last_drained_epoch);
        slot.owners.clear();
        slot.p1 = c.cached_plans[c.chain_plan_indices[s]].p1;
        for (size_t j = 0; j < c.n_ranks; ++j) {
            const auto & r = refs[s * c.n_ranks + j];
            if (r.owner)
                slot.owners.push_back(r.owner);
            if (r.packed_owner)
                slot.owners.push_back(r.packed_owner);
        }
        slot.epoch = epoch;
    }
    // Cached compute bypasses graph_compute_async, so pending host input copies
    // and transfer-queue dependencies must be published before the chain.
    const auto prefix_start = capture ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    for (auto backend : c.backends)
        ggml_vk_tp5_flush_async(backend);
    if (predefined_rows && active_rows < capacity_rows) {
        // Dynamic useful-row transport is implemented only by RELAY direct
        // producers: their host-indirect row dispatch and P2 payload header
        // both consume active_elems. A legacy P1 or other transport would copy
        // padded rows and is therefore not a maximum-capacity execution.
        if (c.sync_mode != tp5_sync_mode::RELAY) {
            return false;
        }
        for (const auto & key : keys) {
            if (!tp5_relay_direct_stage(key, c.n_ranks)) {
                return false;
            }
        }
    }
    if (c.sync_mode == tp5_sync_mode::RELAY) {
        if (!tp5_relay_submit_epoch_chain(
                c, stage_compute_cbs, keys, active_elems, n_stages, active_rows, capacity_rows)) {
            return false;
        }
        if (capture)
            c.timing_submitted = true;
        return true;
    }
    const auto prefix_end        = capture ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    int64_t    submit_wall_us[8] = {};
    const bool measure_submit = capture || ggml_tp5_profile_active();
    tp5_star_times chain_handoff;
    double signal_us = 0.0;
    // STAR is submitted once per rank. Default STAR releases each dependent
    // segment with the host timeline semaphore.
    for (size_t i = 0; i < c.n_ranks; ++i) {
        auto & submits = c.chain_scratch[i].submits;
        if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
            ++prof->queue_submits;
            prof->submit_batches += submits.size();
        }
        const auto submit_start = measure_submit ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (vkQueueSubmit(c.ranks[i].queue, (uint32_t) submits.size(), submits.data(), VK_NULL_HANDLE) != VK_SUCCESS) {
            c.fail("epoch chain submit failed on rank " + std::to_string(i));
            return false;
        }
        if (measure_submit) {
            submit_wall_us[i] += std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - submit_start).count();
        }
        ggml_vk_tp5_mark_queue_submitted(c.backends[i]);
        uint64_t last_signal = 0;
        for (const auto & batch : c.chain_scratch[i].batches) {
            last_signal = std::max(last_signal, batch.signal);
        }
        tp5_note_timeline_submission(c.ranks[i], last_signal);
    }
    if (c.sync_mode == tp5_sync_mode::STAR || c.sync_mode == tp5_sync_mode::RELAY) {
        for (size_t s = 0; s < n_stages; ++s) {
            const uint64_t epoch = c.allreduce_calls + s + 1;
            tp5_star_times step;
            if (!tp5_star_handoff(c, tp5_mailbox_bank(epoch), keys[s].n_elems, &step,
                                  c.sync_mode == tp5_sync_mode::RELAY)) {
                return false;
            }
            chain_handoff.wait_us += step.wait_us;
            chain_handoff.sum_us += step.sum_us;
            chain_handoff.broadcast_us += step.broadcast_us;
            const auto signal_start = measure_submit ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (c.sync_mode == tp5_sync_mode::STAR) for (size_t i = 0; i < c.n_ranks; ++i) {
                auto & rank = c.ranks[i];
                if (!rank.pfn_signal_semaphore || rank.host_ready_sem == VK_NULL_HANDLE) {
                    c.fail("STAR host timeline signal unavailable on rank " + std::to_string(i));
                    return false;
                }
                VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
                signal.semaphore = rank.host_ready_sem;
                signal.value = epoch;
                if (rank.pfn_signal_semaphore(rank.vkdev, &signal) != VK_SUCCESS) {
                    c.fail("STAR host timeline signal failed on rank " + std::to_string(i));
                    return false;
                }
            }
            if (measure_submit) {
                signal_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - signal_start).count();
            }
        }
    }
    if (measure_submit && (c.sync_mode == tp5_sync_mode::STAR || c.sync_mode == tp5_sync_mode::RELAY)) {
        int64_t submit_us = 0;
        for (size_t i = 0; i < c.n_ranks; ++i) submit_us += submit_wall_us[i];
        fprintf(stderr, "[tp5-star-chain] stages=%zu submit_us=%lld flag_wait_us=%.3f sum_us=%.3f broadcast_us=%.3f signal_us=%.3f\n",
                n_stages, (long long) submit_us, chain_handoff.wait_us, chain_handoff.sum_us, chain_handoff.broadcast_us, signal_us);
    }
    if (capture) {
        const auto prefix_us = std::chrono::duration_cast<std::chrono::microseconds>(prefix_end - prefix_start).count();
        fprintf(stderr, "[tp5-host-timing] prefix_flush_us=%lld\n", (long long) prefix_us);
        for (size_t i = 0; i < c.n_ranks; ++i) {
            const auto & scratch = c.chain_scratch[i];
            const size_t markers = c.ranks[i].timing_markers.size();
            const size_t collective_cbs =
                c.sync_mode == tp5_sync_mode::RELAY ?
                    2 * n_stages - n_direct_stages + n_late_stages : 2 * n_stages;
            fprintf(stderr,
                    "[tp5-host-timing] rank=%zu batches=%zu compute_cbs=%zu collective_cbs=%zu marker_cbs=%zu "
                    "submit_wall_us=%lld\n",
                    i, scratch.submits.size(), scratch.compute.size() - collective_cbs - markers,
                    collective_cbs, markers,
                    (long long) submit_wall_us[i]);
        }
    }
    c.allreduce_calls = last_epoch;
    if (capture)
        c.timing_submitted = true;
    if (ggml_tp5_profile * prof = ggml_tp5_profile_active())
        prof->collective_calls += n_stages;
    return true;
}

// ---------------------------------------------------------------------------
// Registry entry points
// ---------------------------------------------------------------------------

extern "C" {

void * ggml_backend_vk_tp5_comm_init(ggml_backend_t * backends, size_t n) {
    const char * star_spin_env = getenv("GGML_TP5_STAR_SPIN");
    if (star_spin_env && atoi(star_spin_env) != 0) {
        fprintf(stderr, "ggml-vulkan-collective: GGML_TP5_STAR_SPIN is retired; use timeline or star\n");
        return nullptr;
    }
    if (n < 1 || n > 8) {
        fprintf(stderr, "ggml-vulkan-collective: need 2..8 backends, got %zu\n", n);
        return nullptr;
    }
    auto * c = new tp5_comm();
    c->n_ranks = n;
    c->spin_max = tp5_default_spin_max();
    c->relay_handoff_timeout_ms = tp5_default_relay_handoff_timeout_ms();
    c->ranks.resize(n);
    c->chain_scratch.resize(n);
    c->backends.assign(backends, backends + n);
    // Plan addresses stay stable while a complete chain is assembled.
    c->cached_plans.reserve(tp5_comm::MAX_CACHED_PLANS);

    const char * timing_env = getenv("GGML_TP5_GPU_TIMING");
    if (timing_env && atoi(timing_env) > 0)
        c->timing_chain = (uint64_t) atoi(timing_env);

    const char * wire_env = getenv("GGML_TP5_WIRE");
    c->wire = (wire_env && strcmp(wire_env, "f32") == 0) ? tp5_wire_type::F32 : tp5_wire_type::F16;

    const char * sync_env = getenv("GGML_TP5_SYNC");
    if (sync_env && (strcmp(sync_env, "star") == 0 || strcmp(sync_env, "l3_star") == 0)) {
        c->sync_mode = tp5_sync_mode::STAR;
    } else if (sync_env && strcmp(sync_env, "relay") == 0) {
        c->sync_mode = tp5_sync_mode::RELAY;
    } else if (sync_env && strcmp(sync_env, "timeline") == 0) {
        c->sync_mode = tp5_sync_mode::TIMELINE;
    } else if (sync_env && strcmp(sync_env, "host") == 0) {
        c->sync_mode = tp5_sync_mode::HOST;
    } else if (sync_env && strcmp(sync_env, "syncfd") == 0) {
        c->sync_mode = tp5_sync_mode::SYNCFD;
    } else if (sync_env && strcmp(sync_env, "drm") == 0) {
        c->sync_mode = tp5_sync_mode::DRM;
    } else if (sync_env && (strcmp(sync_env, "gpu") == 0 || strcmp(sync_env, "gpuflag") == 0)) {
        fprintf(stderr, "ggml-vulkan-collective: sync mode 'gpuflag' is experimental and unsafe under current driver memory model; falling back to timeline\n");
        c->sync_mode = tp5_sync_mode::TIMELINE;
    } else {
        c->sync_mode = (n == 5) ? tp5_sync_mode::STAR : tp5_sync_mode::TIMELINE;
    }

    const char * replay_env = getenv("GGML_TP5_CMD_REPLAY");
    const char * isolate_env = getenv("GGML_TP5_ISOLATE_BO");
    c->isolate_mailbox       = isolate_env && atoi(isolate_env) != 0 && c->sync_mode == tp5_sync_mode::TIMELINE;
    if (!replay_env) replay_env = getenv("GGML_VK_CMD_REPLAY");
    c->cmd_replay_enabled = !replay_env || (atoi(replay_env) != 0 && strcmp(replay_env, "off") != 0);

    for (size_t i = 0; i < n; ++i) {
        tp5_rank & r = c->ranks[i];
        r.device = ggml_vk_tp5_backend_device(backends[i]);
        if (!r.device) {
            fprintf(stderr, "ggml-vulkan-collective: backend %zu is not Vulkan\n", i);
            for (auto & rr : c->ranks) tp5_destroy_rank(rr);
            delete c;
            return nullptr;
        }
        r.vkdev = ggml_vk_tp5_vk_device(r.device);
        r.caps = ggml_vk_tp5_device_caps(r.device);

        // Same evidence channel as GGML_VK_PIPELINE_STATS, extended to the
        // TP5 collective kernels: real RADV codegen statistics (VGPR/SGPR/
        // LDS/spill) instead of shader-name-based reasoning.
        if (const char * stats_env = getenv("GGML_TP5_PIPELINE_STATS")) {
            if (r.caps.pipeline_executable_properties) {
                r.pipeline_stats = true;
                snprintf(r.pipeline_stats_filter, sizeof(r.pipeline_stats_filter), "%s", stats_env);
            } else {
                fprintf(stderr, "ggml-vulkan-collective: GGML_TP5_PIPELINE_STATS requested but "
                                "VK_KHR_pipeline_executable_properties is unavailable; ignoring\n");
            }
        }
        if (c->sync_mode == tp5_sync_mode::RELAY &&
            (!r.caps.device_coherent_memory || r.caps.vendor_id != 0x1002)) {
            fprintf(stderr, "ggml-vulkan-collective: RELAY requires AMD device-coherent memory on rank %zu\n", i);
            for (auto & rr : c->ranks) tp5_destroy_rank(rr);
            delete c;
            return nullptr;
        }
        const size_t required_storage_descriptors =
            std::max(c->n_ranks + 3, c->sync_mode == tp5_sync_mode::GPUFLAG ? size_t(9) : size_t(0));
        if (!r.caps.storage_buffer_array_dynamic_indexing ||
            r.caps.max_storage_buffer_descriptors < required_storage_descriptors) {
            fprintf(
                stderr,
                "ggml-vulkan-collective: rank %zu needs uniform storage-buffer array indexing and %zu descriptors\n", i,
                required_storage_descriptors);
            for (auto & rr : c->ranks)
                tp5_destroy_rank(rr);
            delete c;
            return nullptr;
        }
        if (!r.caps.external_memory_dma_buf) {
            fprintf(stderr, "ggml-vulkan-collective: rank %zu lacks DMA-BUF\n", i);
            for (auto & rr : c->ranks) tp5_destroy_rank(rr);
            delete c;
            return nullptr;
        }

        if (c->sync_mode == tp5_sync_mode::SYNCFD) {
            if (!r.caps.external_semaphore_fd) {
                fprintf(stderr, "ggml-vulkan-collective: rank %zu lacks external semaphore SYNC_FD support\n", i);
                for (auto & rr : c->ranks) tp5_destroy_rank(rr);
                delete c;
                return nullptr;
            }
        }

        if (c->sync_mode == tp5_sync_mode::TIMELINE || c->sync_mode == tp5_sync_mode::STAR || c->sync_mode == tp5_sync_mode::RELAY) {
            if (!r.caps.timeline_semaphore || !r.caps.timeline_semaphore_features) {
                fprintf(stderr, "ggml-vulkan-collective: rank %zu lacks timeline semaphore support (ext=%d feat=%d)\n",
                        i, (int) r.caps.timeline_semaphore, (int) r.caps.timeline_semaphore_features);
                for (auto & rr : c->ranks) tp5_destroy_rank(rr);
                delete c;
                return nullptr;
            }
            if (!r.caps.external_semaphore_fd || !r.caps.timeline_opaque_fd_export || !r.caps.timeline_opaque_fd_import) {
                fprintf(stderr, "ggml-vulkan-collective: rank %zu lacks external timeline OPAQUE_FD support (ext_fd=%d exp=%d imp=%d compat=0x%x)\n",
                        i, (int) r.caps.external_semaphore_fd, (int) r.caps.timeline_opaque_fd_export, (int) r.caps.timeline_opaque_fd_import,
                        r.caps.timeline_opaque_fd_compatible);
                for (auto & rr : c->ranks) tp5_destroy_rank(rr);
                delete c;
                return nullptr;
            }
            const uint64_t req_diff = 2ULL * tp5_comm::MAX_OUTSTANDING_EPOCHS;
            if (r.caps.max_timeline_semaphore_value_difference < req_diff) {
                fprintf(stderr, "ggml-vulkan-collective: rank %zu maxTimelineDifference (%llu) is less than required window difference (%llu)\n",
                        i, (unsigned long long) r.caps.max_timeline_semaphore_value_difference, (unsigned long long) req_diff);
                for (auto & rr : c->ranks) tp5_destroy_rank(rr);
                delete c;
                return nullptr;
            }
        }

        ggml_vk_tp5_get_queue(r.device, &r.queue, &r.queue_family);
        r.has_transfer = ggml_vk_tp5_transfer_queue(r.device, &r.transfer_queue, &r.transfer_family);

        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpi.queueFamilyIndex = r.queue_family;
        if (vkCreateCommandPool(r.vkdev, &cpi, nullptr, &r.cmd_pool) != VK_SUCCESS) {
            for (auto & rr : c->ranks) tp5_destroy_rank(rr);
            delete c;
            return nullptr;
        }

        if (c->sync_mode == tp5_sync_mode::HOST) {
            VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            vkCreateFence(r.vkdev, &fi, nullptr, &r.fence_p1);
        }
        if (c->sync_mode == tp5_sync_mode::HOST || c->sync_mode == tp5_sync_mode::SYNCFD ||
            c->sync_mode == tp5_sync_mode::GPUFLAG) {
            VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            vkCreateFence(r.vkdev, &fi, nullptr, &r.fence_p2);
        }
        if (c->sync_mode == tp5_sync_mode::GPUFLAG) {
            for (size_t f = 0; f < 4; ++f) {
                VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
                vkCreateFence(r.vkdev, &fi, nullptr, &r.fence_ring[f]);
            }
        }

        r.pfn_get_sem_fd = (PFN_vkGetSemaphoreFdKHR) vkGetDeviceProcAddr(r.vkdev, "vkGetSemaphoreFdKHR");
        r.pfn_import_sem_fd = (PFN_vkImportSemaphoreFdKHR) vkGetDeviceProcAddr(r.vkdev, "vkImportSemaphoreFdKHR");
        r.pfn_wait_semaphores = (PFN_vkWaitSemaphores) vkGetDeviceProcAddr(r.vkdev, "vkWaitSemaphores");
        if (!r.pfn_wait_semaphores) {
            r.pfn_wait_semaphores = (PFN_vkWaitSemaphores) vkGetDeviceProcAddr(r.vkdev, "vkWaitSemaphoresKHR");
        }
        r.pfn_get_sem_counter = (PFN_vkGetSemaphoreCounterValue) vkGetDeviceProcAddr(r.vkdev, "vkGetSemaphoreCounterValue");
        if (!r.pfn_get_sem_counter) {
            r.pfn_get_sem_counter = (PFN_vkGetSemaphoreCounterValue) vkGetDeviceProcAddr(r.vkdev, "vkGetSemaphoreCounterValueKHR");
        }
        r.pfn_signal_semaphore = (PFN_vkSignalSemaphore) vkGetDeviceProcAddr(r.vkdev, "vkSignalSemaphore");
        if (!r.pfn_signal_semaphore) {
            r.pfn_signal_semaphore = (PFN_vkSignalSemaphore) vkGetDeviceProcAddr(r.vkdev, "vkSignalSemaphoreKHR");
        }

        if ((c->sync_mode == tp5_sync_mode::SYNCFD || c->sync_mode == tp5_sync_mode::TIMELINE || c->sync_mode == tp5_sync_mode::DRM || c->sync_mode == tp5_sync_mode::STAR || c->sync_mode == tp5_sync_mode::RELAY) && (!r.pfn_get_sem_fd || !r.pfn_import_sem_fd)) {
            fprintf(stderr, "ggml-vulkan-collective: rank %zu missing vkGetSemaphoreFdKHR or vkImportSemaphoreFdKHR proc addr\n", i);
            for (auto & rr : c->ranks) tp5_destroy_rank(rr);
            delete c;
            return nullptr;
        }
        if ((c->sync_mode == tp5_sync_mode::TIMELINE || c->sync_mode == tp5_sync_mode::DRM || c->sync_mode == tp5_sync_mode::STAR || c->sync_mode == tp5_sync_mode::RELAY) && (!r.pfn_wait_semaphores || !r.pfn_get_sem_counter)) {
            fprintf(stderr, "ggml-vulkan-collective: rank %zu missing vkWaitSemaphores or vkGetSemaphoreCounterValue proc addr\n", i);
            for (auto & rr : c->ranks) tp5_destroy_rank(rr);
            delete c;
            return nullptr;
        }

        if (c->sync_mode == tp5_sync_mode::TIMELINE || c->sync_mode == tp5_sync_mode::DRM || c->sync_mode == tp5_sync_mode::STAR || c->sync_mode == tp5_sync_mode::RELAY) {
            VkSemaphoreTypeCreateInfo tci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            tci.initialValue = 0;

            VkExportSemaphoreCreateInfo esci{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
            esci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
            tci.pNext = &esci;

            VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            sci.pNext = &tci;
            if (vkCreateSemaphore(r.vkdev, &sci, nullptr, &r.timeline_sem) != VK_SUCCESS) {
                fprintf(stderr, "ggml-vulkan-collective: rank %zu failed to create timeline semaphore\n", i);
                for (auto & rr : c->ranks) tp5_destroy_rank(rr);
                delete c;
                return nullptr;
            }

            VkSemaphoreGetFdInfoKHR gfi{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
            gfi.semaphore = r.timeline_sem;
            gfi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
            int fd = -1;
            if (r.pfn_get_sem_fd(r.vkdev, &gfi, &fd) != VK_SUCCESS || fd < 0) {
                fprintf(stderr, "ggml-vulkan-collective: rank %zu failed to export timeline semaphore OPAQUE_FD\n", i);
                for (auto & rr : c->ranks) tp5_destroy_rank(rr);
                delete c;
                return nullptr;
            }
            r.timeline_export_fd = fd;
        }

        if (c->sync_mode == tp5_sync_mode::SYNCFD) {
        VkExportSemaphoreCreateInfo esci{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
        esci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;

        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        sci.pNext = &esci;
        vkCreateSemaphore(r.vkdev, &sci, nullptr, &r.sem_p1_done);

        r.wait_sems.resize(n - 1, VK_NULL_HANDLE);
        for (size_t p = 0; p < n - 1; ++p) {
            VkSemaphoreCreateInfo wsci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            vkCreateSemaphore(r.vkdev, &wsci, nullptr, &r.wait_sems[p]);
        }
        }
    }

    // Permanent cross-rank timeline semaphore import (for timeline mode)
    if (c->sync_mode == tp5_sync_mode::TIMELINE || c->sync_mode == tp5_sync_mode::DRM || c->sync_mode == tp5_sync_mode::STAR) {
        for (size_t i = 0; i < n; ++i) {
            tp5_rank & r_dst = c->ranks[i];
            r_dst.peer_timeline_sems.assign(n, VK_NULL_HANDLE);
            for (size_t j = 0; j < n; ++j) {
                if (i == j) {
                    r_dst.peer_timeline_sems[i] = r_dst.timeline_sem;
                    continue;
                }
                int dup_fd = ::dup(c->ranks[j].timeline_export_fd);
                if (dup_fd < 0) {
                    fprintf(stderr, "ggml-vulkan-collective: dup timeline fd failed (dst %zu <- src %zu)\n", i, j);
                    for (auto & rr : c->ranks) tp5_destroy_rank(rr);
                    delete c;
                    return nullptr;
                }
                VkSemaphoreTypeCreateInfo tci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
                tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
                tci.initialValue = 0;
                VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
                sci.pNext = &tci;
                VkSemaphore peer_sem = VK_NULL_HANDLE;
                if (vkCreateSemaphore(r_dst.vkdev, &sci, nullptr, &peer_sem) != VK_SUCCESS) {
                    ::close(dup_fd);
                    fprintf(stderr, "ggml-vulkan-collective: create peer timeline semaphore failed (dst %zu <- src %zu)\n", i, j);
                    for (auto & rr : c->ranks) tp5_destroy_rank(rr);
                    delete c;
                    return nullptr;
                }
                VkImportSemaphoreFdInfoKHR ifi{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR};
                ifi.semaphore = peer_sem;
                ifi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
                ifi.flags = 0; // Permanent import
                ifi.fd = dup_fd;
                if (r_dst.pfn_import_sem_fd(r_dst.vkdev, &ifi) != VK_SUCCESS) {
                    ::close(dup_fd);
                    vkDestroySemaphore(r_dst.vkdev, peer_sem, nullptr);
                    fprintf(stderr, "ggml-vulkan-collective: import peer timeline semaphore failed (dst %zu <- src %zu)\n", i, j);
                    for (auto & rr : c->ranks) tp5_destroy_rank(rr);
                    delete c;
                    return nullptr;
                }
                r_dst.peer_timeline_sems[j] = peer_sem;
            }
        }
        if (c->sync_mode == tp5_sync_mode::DRM) {
            for (size_t i = 0; i < n; ++i) {
                tp5_rank & r = c->ranks[i];
                std::string dri_path = tp5_discover_dri_render_node(c->backends[i], r.device);
                if (dri_path.empty()) {
                    char fallback[32];
                    snprintf(fallback, sizeof(fallback), "/dev/dri/renderD%d", 128 + (int) i);
                    dri_path = fallback;
                }
                r.dri_fd = open(dri_path.c_str(), O_RDWR);
                if (r.dri_fd < 0) {
                    fprintf(stderr, "ggml-vulkan-collective: failed to open DRM render node '%s' for rank %zu (errno=%d: %s)\n",
                            dri_path.c_str(), i, errno, strerror(errno));
                    for (auto & rr : c->ranks) tp5_destroy_rank(rr);
                    delete c;
                    return nullptr;
                }
                fprintf(stderr, "ggml-vulkan-collective: rank %zu bound to DRM node '%s' (dri_fd=%d)\n",
                        i, dri_path.c_str(), r.dri_fd);
                int own_fd = -1;
                VkSemaphoreGetFdInfoKHR gfi_own{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR, nullptr, r.timeline_sem, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT};
                int get_res = r.pfn_get_sem_fd(r.vkdev, &gfi_own, &own_fd);
                if (get_res != VK_SUCCESS || own_fd < 0) {
                    fprintf(stderr, "ggml-vulkan-collective: vkGetSemaphoreFdKHR failed for rank %zu\n", i);
                    for (auto & rr : c->ranks) tp5_destroy_rank(rr);
                    delete c;
                    return nullptr;
                }
                int ret = drmSyncobjFDToHandle(r.dri_fd, own_fd, &r.own_syncobj);
                ::close(own_fd);
                if (ret != 0 || r.own_syncobj == 0) {
                    fprintf(stderr, "ggml-vulkan-collective: drmSyncobjFDToHandle failed for rank %zu (ret=%d, handle=%u, errno=%d: %s)\n",
                            i, ret, r.own_syncobj, errno, strerror(errno));
                    for (auto & rr : c->ranks) tp5_destroy_rank(rr);
                    delete c;
                    return nullptr;
                }
                r.peer_syncobjs.resize(n, 0);
                r.peer_syncobjs[i] = r.own_syncobj;
            }
        }
        // All imports completed: close exporter FDs
        for (size_t j = 0; j < n; ++j) {
            if (c->ranks[j].timeline_export_fd >= 0) {
                ::close(c->ranks[j].timeline_export_fd);
                c->ranks[j].timeline_export_fd = -1;
            }
        }
    }

    if (c->sync_mode == tp5_sync_mode::STAR) {
        for (size_t i = 0; i < n; ++i) {
            tp5_rank & r = c->ranks[i];
            std::string dri_path = tp5_discover_dri_render_node(c->backends[i], r.device);
            if (dri_path.empty()) {
                char fallback[32];
                snprintf(fallback, sizeof(fallback), "/dev/dri/renderD%d", 128 + (int) i);
                dri_path = fallback;
            }
            r.dri_fd = open(dri_path.c_str(), O_RDWR);
            if (r.dri_fd < 0) {
                fprintf(stderr, "ggml-vulkan-collective: failed to open DRM render node '%s' for rank %zu (errno=%d: %s)\n",
                        dri_path.c_str(), i, errno, strerror(errno));
                for (auto & rr : c->ranks) tp5_destroy_rank(rr);
                delete c;
                return nullptr;
            }
            fprintf(stderr, "ggml-vulkan-collective: rank %zu bound to DRM node '%s' (dri_fd=%d)\n",
                    i, dri_path.c_str(), r.dri_fd);
            int own_sync_fd = r.timeline_export_fd;
            bool need_close_own = false;
            if (own_sync_fd < 0) {
                VkSemaphoreGetFdInfoKHR gfi_own{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
                gfi_own.semaphore = r.timeline_sem;
                gfi_own.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
                if (r.pfn_get_sem_fd(r.vkdev, &gfi_own, &own_sync_fd) == VK_SUCCESS && own_sync_fd >= 0) {
                    need_close_own = true;
                }
            }
            int ret = (own_sync_fd >= 0) ? drmSyncobjFDToHandle(r.dri_fd, own_sync_fd, &r.own_syncobj) : -1;
            if (need_close_own && own_sync_fd >= 0) {
                ::close(own_sync_fd);
            }
            if (ret != 0 || r.own_syncobj == 0) {
                fprintf(stderr, "ggml-vulkan-collective: drmSyncobjFDToHandle failed for rank %zu (ret=%d, handle=%u, errno=%d: %s)\n",
                        i, ret, r.own_syncobj, errno, strerror(errno));
                for (auto & rr : c->ranks) tp5_destroy_rank(rr);
                delete c;
                return nullptr;
            }
            // Create dedicated host_ready_event doorbells for zero-syscall CPU -> GPU Phase 2 handoff
            for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
                VkEventCreateInfo eci{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
                vkCreateEvent(r.vkdev, &eci, nullptr, &r.host_ready_event[b]);
            }
            // Create dedicated host_ready_sem for lockless CPU -> GPU Phase 2 handoff
            VkSemaphoreTypeCreateInfo tci_ready{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            tci_ready.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            tci_ready.initialValue = 0;
            VkExportSemaphoreCreateInfo esci_ready{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
            esci_ready.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
            tci_ready.pNext = &esci_ready;
            VkSemaphoreCreateInfo sci_ready{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            sci_ready.pNext = &tci_ready;
            if (vkCreateSemaphore(r.vkdev, &sci_ready, nullptr, &r.host_ready_sem) == VK_SUCCESS) {
                VkSemaphoreGetFdInfoKHR gfi_ready{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
                gfi_ready.semaphore = r.host_ready_sem;
                gfi_ready.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
                int ready_fd = -1;
                if (r.pfn_get_sem_fd(r.vkdev, &gfi_ready, &ready_fd) == VK_SUCCESS && ready_fd >= 0) {
                    int s_ret = drmSyncobjFDToHandle(r.dri_fd, ready_fd, &r.host_ready_syncobj);
                    fprintf(stderr, "HOST_READY_SYNCOBJ rank %zu: handle=%u s_ret=%d dri_fd=%d\n",
                            i, r.host_ready_syncobj, s_ret, r.dri_fd);
                    ::close(ready_fd);
                }
            }
            if (r.host_ready_sem == VK_NULL_HANDLE || !r.pfn_signal_semaphore) {
                fprintf(stderr, "ggml-vulkan-collective: STAR host timeline signal unavailable on rank %zu\n", i);
                for (auto & rr : c->ranks) tp5_destroy_rank(rr);
                delete c;
                return nullptr;
            }
            r.peer_syncobjs.resize(n, 0);
            r.peer_syncobjs[i] = r.own_syncobj;
            if (r.timeline_export_fd >= 0) {
                ::close(r.timeline_export_fd);
                r.timeline_export_fd = -1;
            }
        }
    }

    for (size_t i = 0; i < n; ++i) {
        if (!tp5_build_rank_pipelines(*c, c->ranks[i])) {
            fprintf(stderr, "ggml-vulkan-collective: pipeline build failed on rank %zu\n", i);
            for (auto & rr : c->ranks) tp5_destroy_rank(rr);
            delete c;
            return nullptr;
        }
    }

    if (c->sync_mode == tp5_sync_mode::STAR) {
        const char * workers = getenv("GGML_TP5_STAR_WORKERS");
        if (!workers || atoi(workers) != 1) {
            c->avx2_pool = std::make_unique<tp5_avx2_pool>();
        }
    }

    if (!tp5_setup_workspace(*c, 2560)) {
        fprintf(stderr, "ggml-vulkan-collective: direct mesh workspace setup failed: %s\n", c->fail_reason.c_str());
        for (auto & rr : c->ranks) tp5_destroy_rank(rr);
        delete c;
        return nullptr;
    }
    {
        const char * sync_name = (c->sync_mode == tp5_sync_mode::TIMELINE) ? "timeline" :
                                 (c->sync_mode == tp5_sync_mode::STAR) ? "star" :
                                 (c->sync_mode == tp5_sync_mode::RELAY) ? "relay" :
                                 (c->sync_mode == tp5_sync_mode::SYNCFD) ? "syncfd" :
                                 (c->sync_mode == tp5_sync_mode::GPUFLAG) ? "gpuflag" : "host";
        fprintf(stderr, "ggml-vulkan-collective: init %zu ranks, wire=%s sync=%s relay=%s, "
                        "replay=%s, relay_handoff_timeout_ms=%u\n",
                n, c->wire == tp5_wire_type::F16 ? "f16" : "f32",
                sync_name,
                c->sync_mode == tp5_sync_mode::RELAY ? "vram-doorbell" : "off",
                c->cmd_replay_enabled ? "on" : "off", c->relay_handoff_timeout_ms);
    }
    return c;
}

bool ggml_backend_vk_tp5_comm_free_safe(void * comm) {
    if (!comm) return true;
    auto * c = (tp5_comm *) comm;
    std::lock_guard<std::mutex> lock(c->mutex);
    const bool timeline_mode = c->sync_mode == tp5_sync_mode::TIMELINE || c->sync_mode == tp5_sync_mode::DRM ||
                               c->sync_mode == tp5_sync_mode::RELAY;
    if (c->failed) {
        if (!timeline_mode || !tp5_drain_submitted(*c)) {
            fprintf(stderr, "ggml-vulkan-collective: failed collective could not drain actual submissions; retaining resources\n");
            return false;
        }
    } else if (timeline_mode || c->sync_mode == tp5_sync_mode::STAR || c->sync_mode == tp5_sync_mode::RELAY) {
        if (!tp5_drain_epoch(*c, c->allreduce_calls)) {
            fprintf(stderr, "ggml-vulkan-collective: native teardown drain failed for epoch %llu; retaining resources\n",
                    (unsigned long long) c->allreduce_calls);
            return false;
        }
    }
    c->clear_cached_plans(timeline_mode);
    if (!timeline_mode && c->failed) {
        fprintf(stderr, "ggml-vulkan-collective: plan cleanup failed; retaining resources\n");
        return false;
    }
    if (c->sync_mode == tp5_sync_mode::GPUFLAG) {
        tp5_gpuflag_drain_all(*c);
    } else if (!timeline_mode && c->sync_mode != tp5_sync_mode::STAR &&
               c->sync_mode != tp5_sync_mode::RELAY) {
    for (auto & r : c->ranks) {
        if (r.vkdev != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(r.vkdev);
        }
    }
    }
    tp5_poll_gpu_timing(*c);

    if (c->sync_mode == tp5_sync_mode::RELAY) {
        for (auto backend : c->backends) {
            ggml_vk_tp5_clear_relay_payload_binding(backend);
        }
    }

    // Release imported buffers before their backing host allocations.
    c->avx2_pool.reset();
    c->drm_signaler.reset();
    for (auto & r : c->ranks) tp5_destroy_rank(r);
    for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
        if (c->star_host_raw[b]) {
            free(c->star_host_raw[b]);
            c->star_host_raw[b] = nullptr;
            c->star_host_aligned[b] = nullptr;
        }
    }
    delete c;
    return true;
}

void ggml_backend_vk_tp5_comm_free(void * comm) {
    if (!ggml_backend_vk_tp5_comm_free_safe(comm)) {
        fprintf(stderr, "ggml-vulkan-collective: unsafe communicator teardown retained for process lifetime\n");
    }
}

bool ggml_backend_vk_tp5_allreduce_tensor(void * comm, ggml_tensor ** tensors) {
    auto * c = (tp5_comm *) comm;
    if (!c || c->failed) return false;

    const size_t n = c->n_ranks;
    for (size_t j = 0; j < n; ++j) {
        if (!tensors[j] || !ggml_is_contiguous(tensors[j])) return false;
        if (tensors[j]->type != GGML_TYPE_F32) return false;
    }
    const int64_t ne = ggml_nelements(tensors[0]);
    for (size_t j = 1; j < n; ++j) {
        if (ggml_nelements(tensors[j]) != ne) return false;
    }
    if (ne == 0) return true;

    std::lock_guard<std::mutex> lock(c->mutex);
    if (c->failed) return false;

    const bool debug_ar = (getenv("GGML_TP5_DEBUG_AR") != nullptr);
    if (debug_ar) {
        bool res_ok = true;
        for (size_t j = 0; j < n; ++j) {
            VkBuffer b = VK_NULL_HANDLE; VkDeviceSize o = 0, s = 0;
            if (!ggml_vk_tp5_tensor_dev_ref(tensors[j], &b, &o, &s)) res_ok = false;
        }
        fprintf(stderr, "[tp5-diag] allreduce: ne=%lld name[0]=%s op[0]=%s devref=%s\n",
                (long long) ne, tensors[0]->name, ggml_op_name(tensors[0]->op), res_ok ? "OK" : "FAIL");
    }

    const bool ok = (c->sync_mode == tp5_sync_mode::STAR) ?
                    tp5_allreduce_star(*c, tensors, (size_t) ne) :
                    tp5_allreduce_mesh(*c, tensors, (size_t) ne);

    if (debug_ar) {
        static uint64_t dbg_ok = 0, dbg_fail = 0;
        if (ok) { dbg_ok++; } else { dbg_fail++; }
        fprintf(stderr, "[tp5-diag] allreduce result: %s (ok=%llu fail=%llu)\n",
                ok ? "OK" : "FAILED", (unsigned long long) dbg_ok, (unsigned long long) dbg_fail);
        if (ne > 0) {
            float out_dbg[4] = {0};
            size_t get_bytes = std::min(sizeof(out_dbg), (size_t) ne * sizeof(float));
            ggml_backend_tensor_get(tensors[0], out_dbg, 0, get_bytes);
            fprintf(stderr, "[tp5-diag-vals] AR #%llu rank0: [%f, %f, %f, %f]\n",
                    (unsigned long long)c->allreduce_calls, out_dbg[0], out_dbg[1], out_dbg[2], out_dbg[3]);
            for (size_t j = 1; j < c->n_ranks; ++j) {
                float peer[2] = {0};
                size_t p_bytes = std::min(sizeof(peer), (size_t) ne * sizeof(float));
                ggml_backend_tensor_get(tensors[j], peer, 0, p_bytes);
                fprintf(stderr, "[tp5-diag-vals] AR #%llu rank%zu: [%f, %f]\n",
                        (unsigned long long)c->allreduce_calls, j, peer[0], peer[1]);
            }
        }
    }
    return ok;
}

} // extern "C"

bool ggml_backend_vk_tp5_prepare_graph(void * comm, size_t rank, ggml_cgraph * graph, bool reduce, size_t stage) {
    auto * c = (tp5_comm *) comm;
    if (!c)
        return false;
    std::lock_guard<std::mutex> lock(c->mutex);
    if (c->failed)
        return false;
    if (rank >= c->n_ranks || !graph || graph->n_nodes < 0 || (reduce && graph->n_nodes == 0) ||
        stage > tp5_comm::MAX_OUTSTANDING_EPOCHS ||
        (reduce && stage >= tp5_comm::MAX_OUTSTANDING_EPOCHS))
        return false;

    static const bool disable_producer_wire = (getenv("GGML_VK_DISABLE_PRODUCER_WIRE") != nullptr);
    // Direct writes from a large terminal GEMV/MoE producer into imported host
    // memory are deliberately not the automatic RELAY policy.  On discrete
    // GPUs that couples the producer's critical path to scalar/coherent system
    // memory stores and HOST visibility.  The normal RELAY fast path keeps the
    // producer on VRAM and uses a tiny vector transport epilogue (P1).  Direct
    // host publication remains available as an explicit experiment.
    static const bool direct_producer_host = [] {
        const char * env = getenv("GGML_TP5_DIRECT_PRODUCER_HOST");
        return env && atoi(env) != 0;
    }();
    // Consumer recipe capture is independent of producer-wire. Every subgraph
    // after stage 0 consumes the previous reduction, including the final
    // non-reducing tail. This keeps GGML_VK_DISABLE_PRODUCER_WIRE=1 on the
    // exact legacy P1 producer path while still allowing LateBind to replace
    // the downstream HC prefix/W_down.
    uint32_t predefined_active = 0, predefined_capacity = 0;
    const bool predefined_definition =
        ggml_vk_tp5_predefined_rows(c->backends[rank], &predefined_active, &predefined_capacity);
    const bool define_program = c->sync_mode == tp5_sync_mode::RELAY && c->wire == tp5_wire_type::F32 &&
                                (tp5_latebind_hc_enabled() || predefined_definition);
    if (define_program && tp5_latebind_fused_finalize_enabled()) {
        c->fail("GGML_TP5_LATEBIND_FUSED_FINALIZE is incompatible with independent y/Q generations; unset it");
        return false;
    }
    const bool latebind_capture = tp5_latebind_hc_enabled() && define_program && stage > 0;
    ggml_vk_tp5_set_latebind_capture(c->backends[rank], graph, latebind_capture, define_program, stage);
    // Producer-local wire and direct-host transport are separate optimizations:
    //  * F16 may profit from producing a device-local F16 companion, then P1
    //    performs only a vector copy.
    //  * F32 already has the desired local representation, so producing an
    //    unused F16 companion is pure overhead.
    //  * Direct-host is orthogonal and opt-in; if disabled, LateBind capture
    //    and all consumer-side fusion remain fully available.
    const bool local_wire =
        !disable_producer_wire && reduce &&
        ((c->sync_mode == tp5_sync_mode::TIMELINE && c->wire == tp5_wire_type::F16) ||
         (c->sync_mode == tp5_sync_mode::RELAY && c->wire == tp5_wire_type::F16));
    const bool direct_host =
        !disable_producer_wire && direct_producer_host && reduce && c->sync_mode == tp5_sync_mode::RELAY;

    ggml_tensor * target = (local_wire || direct_host) ? graph->nodes[graph->n_nodes - 1] : nullptr;
    const size_t relay_stage = direct_host ? stage : SIZE_MAX;
    if (relay_stage != SIZE_MAX && target) {
        const size_t n_elems = (size_t) ggml_nelements(target);
        if (n_elems > c->max_elems && !tp5_setup_workspace(*c, n_elems)) {
            return false;
        }
    }
    const bool relay_f32 = c->sync_mode == tp5_sync_mode::RELAY && c->wire == tp5_wire_type::F32;
    ggml_vk_tp5_set_wire_output(c->backends[rank], target, c->sync_mode == tp5_sync_mode::RELAY,
                                relay_stage, relay_f32);
    if (rank == 0 && stage == 0 && getenv("GGML_TP5_RELAY_DEBUG")) {
        fprintf(stderr, "[tp5-relay-config] producer_transport=%s wire=%s legacy_disable=%d\n",
                direct_host ? "direct-host" : (local_wire ? "local-wire+p1" : "p1"),
                c->wire == tp5_wire_type::F32 ? "f32" : "f16", disable_producer_wire ? 1 : 0);
    }
    if (relay_stage != SIZE_MAX) {
        const uint64_t epoch = c->allreduce_calls + 1;
        const size_t bank = tp5_mailbox_bank(epoch);
        const auto payload = tp5_relay_payload_binding(*c, c->ranks[rank]);
        if (!ggml_vk_tp5_update_relay_route(c->backends[rank], stage, (uint32_t) bank, epoch, payload, nullptr,
                                            tp5_latebind_hc_enabled(), target ? uint64_t(ggml_nelements(target)) : 0)) {
            // Direct producer is an optimization, not a correctness fallback.
            // Reconfigure this recording for the established scratch/P1 path.
            ggml_vk_tp5_set_wire_output(c->backends[rank], target, true, SIZE_MAX, relay_f32);
        } else {
            std::atomic_thread_fence(std::memory_order_release);
        }
    }
    return true;
}

// Isolated hardware experiment: CPU arithmetic remains part of the protocol;
// only GPU queue progression is autonomous. No model graph uses this entrypoint.
int ggml_vk_tp5_relay_probe(ggml_backend_t * backends, size_t n, const ggml_vk_relay_config & cfg) {
    if (n != 5 || !backends || cfg.stages < 1 || cfg.stages > 96 || cfg.replays < 1 || cfg.replays > 32 ||
        cfg.elements < 1 || cfg.elements > 4096 || cfg.spin_max < 1 || cfg.spin_max > 1000000 ||
        cfg.delay_us > 2000 || cfg.withhold_stage > cfg.stages || cfg.partial_submit_ranks >= n ||
        (cfg.withhold_stage && cfg.partial_submit_ranks)) {
        fprintf(stderr, "relay: invalid bounded probe configuration\n");
        return 1;
    }
    struct resources {
        tp5_comm comm;
        std::array<VkPipeline, 5> pipelines{};
        std::array<VkPipelineLayout, 5> layouts{};
        std::array<VkDescriptorSetLayout, 5> dsls{};
        std::array<VkCommandBuffer, 5> commands{};
        std::array<bool, 5> pending{};
        bool drained = true;
        ~resources() {
            // Retain all allocations if an unexpected driver failure prevents
            // bounded draining. Never free USERPTR pages still in use by GPU.
            if (!drained) return;
            for (size_t i = 0; i < comm.ranks.size(); ++i) {
                auto & r = comm.ranks[i];
                if (pipelines[i]) vkDestroyPipeline(r.vkdev, pipelines[i], nullptr);
                if (layouts[i]) vkDestroyPipelineLayout(r.vkdev, layouts[i], nullptr);
                if (dsls[i]) vkDestroyDescriptorSetLayout(r.vkdev, dsls[i], nullptr);
                tp5_destroy_rank(r);
            }
            for (auto ptr : comm.star_host_raw) free(ptr);
        }
    } res;
    auto & c = res.comm;
    c.n_ranks = n;
    c.sync_mode = tp5_sync_mode::RELAY;
    c.wire = tp5_wire_type::F16;
    c.ranks.resize(n);
    c.backends.assign(backends, backends + n);
    size_t alignment = 4096;
    for (size_t i = 0; i < n; ++i) {
        auto & r = c.ranks[i];
        r.device = ggml_vk_tp5_backend_device(backends[i]);
        if (!r.device) return 1;
        r.vkdev = ggml_vk_tp5_vk_device(r.device);
        r.caps = ggml_vk_tp5_device_caps(r.device);
        // This PoC is a RADV/Navi21 experiment, not a portable Vulkan promise.
        if (r.caps.vendor_id != 0x1002 || r.caps.device_id != 0x73bf) {
            fprintf(stderr, "relay: unvalidated device %s (%04x:%04x)\n", r.caps.name, r.caps.vendor_id, r.caps.device_id);
            return 1;
        }
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT host{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &host};
        vkGetPhysicalDeviceProperties2(ggml_vk_tp5_vk_physical_device(r.device), &props);
        alignment = std::max(alignment, (size_t) host.minImportedHostPointerAlignment);
        ggml_vk_tp5_get_queue(r.device, &r.queue, &r.queue_family);
    }
    constexpr size_t status_bytes = 512; // upload + progress + 96 completed polling counts
    const size_t stride = ((cfg.elements * sizeof(float) + status_bytes + alignment - 1) / alignment) * alignment;
    c.star_rank_stride = stride;
    for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
        if (posix_memalign(&c.star_host_raw[b], alignment, n * stride) != 0) return 1;
        c.star_host_aligned[b] = c.star_host_raw[b];
        memset(c.star_host_raw[b], 0, n * stride);
    }
    struct relay_pc {
        uint32_t n_elems, seq, rank, spin_max, mode, reserved;
    };
    static_assert(sizeof(relay_pc) == 24, "shader push constant layout");
    const auto barrier = [](VkCommandBuffer cmd, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                            VkAccessFlags writes, VkAccessFlags reads) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, writes, reads};
        vkCmdPipelineBarrier(cmd, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };
    for (size_t i = 0; i < n; ++i) {
        auto & r = c.ranks[i];
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.queueFamilyIndex = r.queue_family;
        if (vkCreateCommandPool(r.vkdev, &pool, nullptr, &r.cmd_pool) != VK_SUCCESS) return 1;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(r.vkdev, &fci, nullptr, &r.fence_p2) != VK_SUCCESS) return 1;
        VkQueryPoolCreateInfo query{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        query.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query.queryCount = 2;
        if (vkCreateQueryPool(r.vkdev, &query, nullptr, &r.timing_pool) != VK_SUCCESS) return 1;
        if (!tp5_build_rank_pipelines(c, r)) return 1;
        for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
            void * ptr = (char *) c.star_host_raw[b] + i * stride;
            auto host_props_fn = (PFN_vkGetMemoryHostPointerPropertiesEXT) vkGetDeviceProcAddr(r.vkdev, "vkGetMemoryHostPointerPropertiesEXT");
            VkMemoryHostPointerPropertiesEXT host_props{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
            VkPhysicalDeviceMemoryProperties props{};
            ggml_vk_tp5_mem_props(r.device, &props);
            if (!host_props_fn || host_props_fn(r.vkdev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, ptr, &host_props) != VK_SUCCESS ||
                find_memory_type(props, host_props.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == UINT32_MAX ||
                !tp5_alloc_host_import_buffer(r, ptr, stride, false,
                                              r.host_import_buf[b], r.host_import_mem[b], r.bda_addr[b])) {
                fprintf(stderr, "relay: coherent host import unavailable on rank %zu\n", i);
                return 1;
            }
            const VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
            uint32_t memory_type = UINT32_MAX;
            if (!tp5_alloc_host_visible_buffer(r, stride, r.bcast_buf[b], r.bcast_mem[b], &r.bcast_host[b], required, &memory_type)) {
                fprintf(stderr, "relay: mapped local VRAM unavailable on rank %zu; refusing host-memory polling fallback\n", i);
                return 1;
            }
            fprintf(stderr, "relay rank=%zu bank=%zu memory_type=%u flags=0x%x local_vram=yes bytes=%zu\n",
                    i, b, memory_type, props.memoryTypes[memory_type].propertyFlags, stride);
        }
        if (!tp5_alloc_device_buffer(r, cfg.elements * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                false, r.wire_buf, r.wire_mem, nullptr)) return 1;
        VkDescriptorSetLayoutBinding bindings[4]{};
        for (uint32_t j = 0; j < 4; ++j) bindings[j] = {j, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dci.bindingCount = 4;
        dci.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(r.vkdev, &dci, nullptr, &res.dsls[i]) != VK_SUCCESS) return 1;
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(relay_pc)};
        VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        lci.setLayoutCount = 1;
        lci.pSetLayouts = &res.dsls[i];
        lci.pushConstantRangeCount = 1;
        lci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(r.vkdev, &lci, nullptr, &res.layouts[i]) != VK_SUCCESS) return 1;
        VkShaderModule module = VK_NULL_HANDLE;
        if (!tp5_create_shader_module(r.vkdev, tp5_relay_data, tp5_relay_len, &module)) return 1;
        VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
        pci.layout = res.layouts[i];
        const VkResult pipeline_result = vkCreateComputePipelines(r.vkdev, VK_NULL_HANDLE, 1, &pci, nullptr, &res.pipelines[i]);
        vkDestroyShaderModule(r.vkdev, module, nullptr);
        if (pipeline_result != VK_SUCCESS) return 1;
        VkDescriptorSet sets[TP5_MAILBOX_BANKS]{};
        VkDescriptorSet pack_sets[TP5_MAILBOX_BANKS]{};
        for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, r.desc_pool, 1, &res.dsls[i]};
            if (vkAllocateDescriptorSets(r.vkdev, &ai, &sets[b]) != VK_SUCCESS) return 1;
            VkDescriptorBufferInfo infos[4] = {{r.bcast_buf[b], 0, stride}, {r.wire_buf, 0, cfg.elements * sizeof(float)},
                                               {r.bcast_buf[0], 0, 64}, {r.host_import_buf[b], stride - status_bytes, status_bytes}};
            VkWriteDescriptorSet writes[4]{};
            for (uint32_t j = 0; j < 4; ++j) writes[j] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sets[b], j, 0, 1,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &infos[j], nullptr};
            vkUpdateDescriptorSets(r.vkdev, 4, writes, 0, nullptr);
            VkDescriptorSetAllocateInfo pack_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, r.desc_pool, 1, &r.dsl};
            if (vkAllocateDescriptorSets(r.vkdev, &pack_ai, &pack_sets[b]) != VK_SUCCESS) return 1;
            tp5_update_pack_descriptor(r, pack_sets[b], r.wire_buf, 0, cfg.elements * sizeof(float),
                                       (uint32_t) n, r.host_import_buf[b], 0, cfg.elements * sizeof(ggml_fp16_t));
        }
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, r.cmd_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        if (vkAllocateCommandBuffers(r.vkdev, &ai, &res.commands[i]) != VK_SUCCESS) return 1;
        const VkCommandBuffer cmd = res.commands[i];
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) return 1;
        vkCmdResetQueryPool(cmd, r.timing_pool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, r.timing_pool, 0);
        relay_pc pc{cfg.elements, 0, (uint32_t) i, cfg.spin_max, 0, 0};
        const auto consume = [&](uint32_t seq, uint32_t mode) {
            const size_t bank = seq ? tp5_mailbox_bank(seq) : 0;
            pc.seq = seq;
            pc.mode = mode;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, res.pipelines[i]);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, res.layouts[i], 0, 1, &sets[bank], 0, nullptr);
            vkCmdPushConstants(cmd, res.layouts[i], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, 1, 1, 1);
            barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
        };
        consume(0, 0);
        // RELAY's fused ordering: COMPUTE -> descriptor pack -> dependent COMPUTE.
        // All 96 stages are recorded once. No peer/host semaphore waits exist.
        for (uint32_t seq = 1; seq <= cfg.stages; ++seq) {
            const size_t b = tp5_mailbox_bank(seq);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.pack_pipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipe_layout, 0, 1,
                                    &pack_sets[b], 0, nullptr);
            const uint32_t n_elems = cfg.elements;
            vkCmdPushConstants(cmd, r.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(n_elems), &n_elems);
            vkCmdDispatch(cmd, (cfg.elements + 255) / 256, 1, 1);
            barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
            vkCmdFillBuffer(cmd, r.host_import_buf[b], stride - status_bytes, sizeof(uint32_t), seq);
            barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT);
            consume(seq, 1);
        }
        VkBufferCopy copy{0, 0, cfg.elements * sizeof(float)};
        vkCmdCopyBuffer(cmd, r.wire_buf, r.host_import_buf[0], 1, &copy);
        barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, r.timing_pool, 1);
        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return 1;
    }
    const auto status = [&](size_t rank) {
        return (volatile uint32_t *) ((char *) c.star_host_raw[0] + (rank + 1) * stride - status_bytes);
    };
    const auto abort_all = [&] {
        for (auto & r : c.ranks) ((volatile uint32_t *) r.bcast_host[0])[1] = 1;
#if defined(__x86_64__) || defined(_M_X64)
        _mm_sfence();
#endif
    };
    const auto drain = [&] {
        bool ok = true;
        for (size_t i = 0; i < n; ++i) if (res.pending[i]) {
            const VkResult result = vkWaitForFences(c.ranks[i].vkdev, 1, &c.ranks[i].fence_p2, VK_TRUE, 2000000000ULL);
            if (result == VK_SUCCESS) res.pending[i] = false;
            else { fprintf(stderr, "relay: bounded drain failed rank=%zu result=%d\n", i, result); ok = false; }
        }
        res.drained = ok;
        return ok;
    };
    const size_t stage_size = n * cfg.elements;
    std::vector<float> reference((cfg.stages + 1) * stage_size);
    std::vector<float> sum(cfg.elements);
    size_t checked = 0, prepublication_checks = 0, waits_observed = 0, submits = 0;
    const bool inject = cfg.withhold_stage || cfg.partial_submit_ranks;
    const uint32_t replay_count = inject ? 1 : cfg.replays;
    double total_us = 0;
    for (uint32_t run = 0; run < replay_count; ++run) {
        for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) memset(c.star_host_raw[b], 0, n * stride);
        for (size_t i = 0; i < n; ++i) {
            auto & r = c.ranks[i];
            for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
                memset(r.bcast_host[b], 0, stride);
                auto * words = (volatile uint32_t *) r.bcast_host[b];
                words[0] = cfg.stale_doorbell ? UINT32_MAX : 0;
                words[4] = run;
            }
            if (vkResetFences(r.vkdev, 1, &r.fence_p2) != VK_SUCCESS) return 1;
            for (size_t e = 0; e < cfg.elements; ++e) reference[i * cfg.elements + e] =
                (float) (i + 1) * 0.25f + (float) (e % 7) * 0.03125f + (float) (run % 8) * 0.0625f;
        }
        for (uint32_t s = 0; s < cfg.stages; ++s) for (size_t e = 0; e < cfg.elements; ++e) {
            float value = 0;
            for (size_t i = 0; i < n; ++i) value += ggml_fp16_to_fp32(ggml_fp32_to_fp16(reference[s * stage_size + i * cfg.elements + e]));
            for (size_t i = 0; i < n; ++i) reference[(s + 1) * stage_size + i * cfg.elements + e] =
                value * 0.125f + (float) (i + 1) * 0.25f + (float) (e % 7) * 0.03125f;
        }
#if defined(__x86_64__) || defined(_M_X64)
        _mm_sfence();
#endif
        const auto started = std::chrono::steady_clock::now();
        bool ok = true;
        const size_t submit_count = cfg.partial_submit_ranks ? cfg.partial_submit_ranks : n;
        for (size_t i = 0; i < submit_count; ++i) {
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &res.commands[i];
            if (vkQueueSubmit(c.ranks[i].queue, 1, &si, c.ranks[i].fence_p2) != VK_SUCCESS) {
                fprintf(stderr, "relay: rank %zu submit failed\n", i);
                ok = false;
                break;
            }
            res.pending[i] = true;
            res.drained = false;
            ++submits;
        }
        if (cfg.partial_submit_ranks || !ok) {
            abort_all();
            if (!drain()) return 2;
            if (!ok) return 1;
            fprintf(stderr, "relay partial-submit: %zu/%zu submitted, abort drained every submitted rank; no peer waits\n", submit_count, n);
            break;
        }
        uint32_t published = 0;
        const auto submitted = std::chrono::steady_clock::now();
        for (uint32_t seq = 1; seq <= cfg.stages && ok; ++seq) {
            const size_t b = tp5_mailbox_bank(seq);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
            bool ready = false;
            for (uint64_t poll = 0; !ready; ++poll) {
                ready = true;
                for (size_t i = 0; i < n; ++i) {
                    const auto * st = status(i);
                    const auto * uploaded = (volatile uint32_t *) ((char *) c.star_host_raw[b] + (i + 1) * stride - status_bytes);
                    if (st[2] != 0) { ok = false; break; }
                    ready &= *uploaded == seq;
                }
                if (!ok) break;
                if ((poll & 255) == 0 && std::chrono::steady_clock::now() >= deadline) { ok = false; break; }
#if defined(__x86_64__) || defined(_M_X64)
                _mm_pause();
#endif
            }
            if (!ok) break;
            std::atomic_thread_fence(std::memory_order_acquire);
            if (cfg.delay_us) std::this_thread::sleep_for(std::chrono::microseconds(cfg.delay_us));
            const void * ptrs[5];
            for (size_t i = 0; i < n; ++i) {
                if (status(i)[2] || status(i)[3] >= seq) { ok = false; break; }
                ++prepublication_checks;
                ptrs[i] = (char *) c.star_host_raw[b] + i * stride;
                const auto * actual = (const uint16_t *) ptrs[i];
                for (size_t e = 0; e < cfg.elements; ++e) {
                    const uint16_t expected = ggml_fp32_to_fp16(reference[(seq - 1) * stage_size + i * cfg.elements + e]);
                    if (actual[e] != expected) {
                        fprintf(stderr, "relay upload mismatch replay=%u stage=%u rank=%zu elem=%zu got=%04x expected=%04x\n", run, seq, i, e, actual[e], expected);
                        ok = false; break;
                    }
                    ++checked;
                }
                if (!ok) break;
            }
            if (!ok) break;
            tp5_avx2_accumulate_star(ptrs, sum.data(), cfg.elements, 0, 1, true);
            for (size_t i = 0; i < n; ++i) {
                if (seq == cfg.withhold_stage && i == n - 1) continue;
                std::memcpy((char *) c.ranks[i].bcast_host[b] + 64, sum.data(), cfg.elements * sizeof(float));
            }
#if defined(__x86_64__) || defined(_M_X64)
            _mm_sfence();
#endif
            for (size_t i = 0; i < n; ++i) {
                if (seq == cfg.withhold_stage && i == n - 1) continue;
                ((volatile uint32_t *) c.ranks[i].bcast_host[b])[0] = seq;
            }
#if defined(__x86_64__) || defined(_M_X64)
            _mm_sfence();
#endif
            for (size_t i = 0; i < n; ++i) {
                if (seq == cfg.withhold_stage && i == n - 1) continue;
                VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
                range.memory = c.ranks[i].bcast_mem[b];
                range.offset = 0;
                range.size = VK_WHOLE_SIZE;
                if (vkFlushMappedMemoryRanges(c.ranks[i].vkdev, 1, &range) != VK_SUCCESS) {
                    fprintf(stderr, "relay: host-to-VRAM flush failed rank=%zu stage=%u\n", i, seq);
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
            published = seq;
            if (seq == cfg.withhold_stage) break;
        }
        if (!ok) abort_all();
        // The withheld consumer exits by its shader bound, not a forged ready.
        if (!drain()) { abort_all(); if (!drain()) return 2; ok = false; }
        total_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
        if (!ok || inject) {
            fprintf(stderr, "relay diagnostic: submit_us=%.3f published=%u\n",
                    std::chrono::duration<double, std::micro>(submitted - started).count(), published);
            for (size_t i = 0; i < n; ++i) {
                uint64_t timestamps[2]{};
                const auto & r = c.ranks[i];
                const VkResult qr = vkGetQueryPoolResults(r.vkdev, r.timing_pool, 0, 2, sizeof(timestamps), timestamps,
                                                          sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
                const auto * control_words = (volatile uint32_t *) c.ranks[i].bcast_host[0];
                const size_t diag_bank = tp5_mailbox_bank(std::max<uint32_t>(published, 1));
                const auto * inbox_words = (volatile uint32_t *) c.ranks[i].bcast_host[diag_bank];
                fprintf(stderr, "relay rank=%zu entered=%u error=%u consumed=%u gpu_spin=%u control=%u/%u/%u inbox0=%u gpu_us=%.3f query=%d\n", i,
                        status(i)[1], status(i)[2], status(i)[3], status(i)[4],
                        control_words[0], control_words[1], control_words[2], inbox_words[0],
                        (double) (timestamps[1] - timestamps[0]) * r.caps.timestamp_period / 1000.0, qr);
            }
        }
        if (cfg.withhold_stage) {
            const auto * st = status(n - 1);
            if (!ok || st[2] != cfg.withhold_stage || st[3] >= cfg.withhold_stage) {
                fprintf(stderr, "relay: expected timeout not observed (error=%u consumed=%u published=%u)\n", st[2], st[3], published);
                return 1;
            }
            const auto * actual = (const float *) ((char *) c.star_host_raw[0] + (n - 1) * stride);
            const auto * expected = reference.data() + (cfg.withhold_stage - 1) * stage_size + (n - 1) * cfg.elements;
            if (memcmp(actual, expected, cfg.elements * sizeof(float)) != 0) {
                fprintf(stderr, "relay: timed-out consumer modified its state\n");
                return 1;
            }
            fprintf(stderr, "relay timeout: rank=%zu stage=%u state untouched, sticky error, all ranks drained\n", n - 1, cfg.withhold_stage);
            break;
        }
        if (!ok) { fprintf(stderr, "relay: handoff failed replay=%u published=%u\n", run, published); return 1; }
        for (size_t i = 0; i < n; ++i) {
            if (status(i)[2] || status(i)[3] != cfg.stages) {
                fprintf(stderr, "relay: consumer failed rank=%zu error=%u consumed=%u\n", i, status(i)[2], status(i)[3]); return 1;
            }
            for (uint32_t seq = 1; seq <= cfg.stages; ++seq) {
                if (status(i)[3 + seq] > 0) ++waits_observed;
                else if (cfg.delay_us) {
                    fprintf(stderr, "relay: delayed trial did not exercise an active wait rank=%zu stage=%u\n", i, seq);
                    return 1;
                }
            }
            const auto * actual = (const float *) ((char *) c.star_host_raw[0] + i * stride);
            const auto * expected = reference.data() + cfg.stages * stage_size + i * cfg.elements;
            if (memcmp(actual, expected, cfg.elements * sizeof(float)) != 0) {
                fprintf(stderr, "relay: final state mismatch rank=%zu replay=%u\n", i, run); return 1;
            }
            checked += cfg.elements;
        }
    }
    fprintf(stderr, "relay: PASS stages=%u replays=%u elements=%u checked=%zu prepublication_checks=%zu waits_observed=%zu queue_submits=%zu host_signals=0\n",
            cfg.stages, replay_count, cfg.elements, checked, prepublication_checks, waits_observed, submits);
    if (!inject) fprintf(stderr, "relay chain wall: %.3f us/replay (includes CPU reference checks/delay; NOT model tok/s)\n", total_us / replay_count);
    return 0;
}
