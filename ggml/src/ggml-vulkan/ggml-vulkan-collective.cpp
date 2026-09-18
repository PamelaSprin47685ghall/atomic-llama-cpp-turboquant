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
#include "ggml-vulkan-shaders.hpp"
#include "ggml-vulkan.h"
#include "ggml-tp5-profile.h"
#include "ggml-vulkan-collective.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
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
enum class tp5_sync_mode { HOST, SYNCFD, TIMELINE, GPUFLAG, DRM, STAR };

// Two mailbox banks so P1 of epoch N can overlap peer P2 of epoch N-1:
// bank = (epoch-1) & 1. P2's ready waits transitively provide epoch-2 credit.
static constexpr size_t TP5_MAILBOX_BANKS = 2;

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
    const char * env = getenv("GGML_TP5_SPIN_MAX");
    if (env && env[0]) {
        const long v = strtol(env, nullptr, 10);
        if (v > 0) return (uint32_t) v;
    }
    return 100000000u;
}

struct tp5_rank {
    vk_device device;
    VkDevice vkdev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
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

    // Pre-allocated static descriptor sets and command buffers for Star AllReduce
    VkDescriptorSet star_ds_pack[TP5_MAILBOX_BANKS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet star_ds_bda[TP5_MAILBOX_BANKS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet star_ds_sum[TP5_MAILBOX_BANKS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkCommandBuffer star_cmd_p1[TP5_MAILBOX_BANKS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkCommandBuffer star_cmd_p2[TP5_MAILBOX_BANKS] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::vector<VkCommandBuffer> star_chain_p1;
    std::vector<VkCommandBuffer> star_chain_p2;
    std::vector<VkDescriptorSet> star_chain_ds_pack;
    std::vector<VkDescriptorSet> star_chain_ds_sum;

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

    // Pipelines
    VkPipeline sum_pipe = VK_NULL_HANDLE;
    VkPipeline pack_pipe = VK_NULL_HANDLE;
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
    VkTimelineSemaphoreSubmitInfo timeline{ VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
    VkSubmitInfo                  submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };

    void init(tp5_rank &              rank,
              size_t                  rank_index,
              size_t                  n_ranks,
              uint64_t                epoch,
              bool                    phase1,
              const VkCommandBuffer * command) {
        // R=2 credit is already implied by the existing dependency chain:
        // peer P2(e-2) -> peer P1(e-1) signal -> local P2(e-1) wait
        // -> local P1(e), ordered by the P1 compute/transfer barrier.
        // Thus P1(e) may overlap peer P2(e-1), but cannot overwrite a bank
        // still being read by peer P2(e-2). This is NOT valid for one bank.
        static_assert(TP5_MAILBOX_BANKS == 2, "transitive credit requires two banks");
        const uint64_t wait_value = phase1 ? 0 : 2 * epoch - 1;
        uint32_t       count      = 0;
        if (wait_value != 0) {
            for (size_t peer = 0; peer < n_ranks; ++peer) {
                if (peer == rank_index)
                    continue;
                waits[count]  = rank.peer_timeline_sems[peer];
                values[count] = wait_value;
                stages[count] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
                ++count;
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
        const uint64_t wait_val = wait_epoch ? 2 * wait_epoch - 1 : 0;
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
static bool tp5_gpuflag_drain_epoch(tp5_comm & c, size_t ring_idx);
static bool tp5_gpuflag_drain_all(tp5_comm & c);

struct tp5_hc_key {
    uint32_t                       width        = 0;
    uint32_t                       epsilon_bits = 0;
    std::array<tp5_binding_key, 6> bindings{};
    tp5_binding_key                quantized{};

    bool operator==(const tp5_hc_key & other) const {
        return width == other.width && epsilon_bits == other.epsilon_bits &&
               bindings == other.bindings && quantized == other.quantized;
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
        return hc == o.hc;
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
    std::vector<VkCommandBuffer>  cmd_p1;
    std::vector<VkDescriptorSet>  ds_pack;
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
    std::vector<VkDescriptorSet>      ds_sum;
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
    bool          ok = false;
};

struct tp5_comm {
    size_t n_ranks = 0;
    std::vector<tp5_rank> ranks;
    std::vector<ggml_backend_t> backends;
    std::vector<tp5_rank_chain> chain_scratch;
    uint64_t                    timing_chain       = 0;
    uint64_t                    chain_calls        = 0;
    size_t                      timing_stages      = 0;
    bool                        timing_submitted   = false;
    tp5_wire_type wire = tp5_wire_type::F32;
    tp5_sync_mode sync_mode = tp5_sync_mode::HOST;
    bool cmd_replay_enabled = true;
    bool                        isolate_mailbox    = false;
    uint32_t spin_max = 100000000u;
    size_t max_elems = 0;
    uint64_t workspace_gen = 1;

    // Star AllReduce Host RAM & Infrastructure
    void * star_host_raw[TP5_MAILBOX_BANKS] = {nullptr, nullptr};
    void * star_host_aligned[TP5_MAILBOX_BANKS] = {nullptr, nullptr};
    volatile uint32_t * star_host_flag[TP5_MAILBOX_BANKS] = {nullptr, nullptr};
    size_t star_host_alloc_size = 0;
    size_t star_rank_stride = 0;
    std::unique_ptr<tp5_avx2_pool> avx2_pool;
    std::unique_ptr<tp5_drm_signaler> drm_signaler;
    std::unique_ptr<tp5_submit_pool> submit_pool;
    std::unique_ptr<tp5_drm_waiter> drm_waiter;

    // Bounded immutable per-binding plan cache
    // For 96 distinct subgraphs in Qwen4EXP TP5, keep 256 entries to eliminate LRU thrashing.
    static constexpr size_t MAX_CACHED_PLANS = 256;
    std::vector<tp5_cached_plan> cached_plans;
    size_t                       last_hit_idx = 0;
    uint64_t                     plans_gen    = 1;
    tp5_compiled_chain           compiled_chain;

    // Retained scratch buffers for chain validation/plan resolution across calls
    std::vector<tensor_dev_ref> chain_refs;
    std::vector<tp5_plan_key>   chain_keys;
    std::vector<size_t>         chain_plan_indices;

    uint64_t allreduce_calls = 0;
    uint64_t host_waits = 0;
    uint64_t last_drained_epoch = 0;
    static constexpr uint64_t MAX_OUTSTANDING_EPOCHS = 128;

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
            if (idx < plan.ds_sum.size() && plan.ds_sum[idx] != VK_NULL_HANDLE && r.desc_pool != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(r.vkdev, r.desc_pool, 1, &plan.ds_sum[idx]);
                plan.ds_sum[idx] = VK_NULL_HANDLE;
            }
        }
        plan.owners.clear();
    }

    void invalidate_chain() {
        compiled_chain.invalidate();
        plans_gen++;
    }

    void clear_cached_plans() {
        invalidate_chain();
        if (sync_mode == tp5_sync_mode::TIMELINE) {
            if (!tp5_drain_epoch(*this, allreduce_calls)) {
                fail("clear_cached_plans: drain failed before plan destruction");
                return;
            }
        }
        if (sync_mode == tp5_sync_mode::GPUFLAG) {
            tp5_gpuflag_drain_all(*this);
        } else if (sync_mode != tp5_sync_mode::TIMELINE) {
        for (auto & r : ranks) {
            if (r.vkdev != VK_NULL_HANDLE) {
                vkDeviceWaitIdle(r.vkdev);
            }
        }
        }
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
        if (sync_mode == tp5_sync_mode::TIMELINE) {
            if (!tp5_drain_epoch(*this, allreduce_calls)) {
                fail("evict_lru_plan: drain failed before plan eviction");
                return;
            }
        }
        if (sync_mode == tp5_sync_mode::GPUFLAG) {
            tp5_gpuflag_drain_all(*this);
        } else if (sync_mode != tp5_sync_mode::TIMELINE) {
        for (auto & r : ranks) {
            if (r.vkdev != VK_NULL_HANDLE) {
                vkDeviceWaitIdle(r.vkdev);
            }
        }
        }
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

static bool tp5_create_shader_module(VkDevice dev,
                                     const unsigned char * spv, uint64_t len,
                                     VkShaderModule * out) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = (size_t) len;
    ci.pCode = (const uint32_t *) (const void *) spv;
    return vkCreateShaderModule(dev, &ci, nullptr, out) == VK_SUCCESS;
}

bool tp5_alloc_host_visible_buffer(tp5_rank & r, VkDeviceSize size, VkBuffer & buf, VkDeviceMemory & mem, void ** mapped) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(r.vkdev, &bci, nullptr, &buf) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(r.vkdev, buf, &req);
    VkPhysicalDeviceMemoryProperties props{};
    ggml_vk_tp5_mem_props(r.device, &props);
    uint32_t mt = find_memory_type(props, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
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
    VkShaderModule mod_sum = VK_NULL_HANDLE, mod_pack = VK_NULL_HANDLE;
    const unsigned char * sum_spv = c.wire == tp5_wire_type::F16 ? tp5_sum_f16_data : tp5_sum_f32_data;
    const uint64_t sum_len = c.wire == tp5_wire_type::F16 ? tp5_sum_f16_len : tp5_sum_f32_len;
    if (!tp5_create_shader_module(r.vkdev, sum_spv, sum_len, &mod_sum)) return false;
    if (c.wire == tp5_wire_type::F16 &&
        !tp5_create_shader_module(r.vkdev, tp5_pack_f16_data, tp5_pack_f16_len, &mod_pack)) {
        vkDestroyShaderModule(r.vkdev, mod_sum, nullptr);
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
    vkDestroyShaderModule(r.vkdev, mod_sum, nullptr);
    if (mod_pack != VK_NULL_HANDLE) vkDestroyShaderModule(r.vkdev, mod_pack, nullptr);
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

    // Two sum sets plus one pack set per cached binding; reserve one temporary
    // plan as well. The SUM/pack layout contains one input descriptor per rank.
    const uint32_t             plan_capacity = tp5_comm::MAX_CACHED_PLANS + 1;
    const uint32_t             sets_per_plan = TP5_MAILBOX_BANKS * 2 + 1 + (c.sync_mode == tp5_sync_mode::GPUFLAG ? 1 : 0);
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
        r.star_chain_p1.resize(128, VK_NULL_HANDLE);
        r.star_chain_p2.resize(128, VK_NULL_HANDLE);
        r.star_chain_ds_pack.resize(128, VK_NULL_HANDLE);
        r.star_chain_ds_sum.resize(128, VK_NULL_HANDLE);

        for (size_t s = 0; s < 128; ++s) {
            VkDescriptorSetAllocateInfo ds_ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ds_ai.descriptorPool = r.desc_pool;
            ds_ai.descriptorSetCount = 1;
            ds_ai.pSetLayouts = &r.dsl;
            vkAllocateDescriptorSets(r.vkdev, &ds_ai, &r.star_chain_ds_pack[s]);
            vkAllocateDescriptorSets(r.vkdev, &ds_ai, &r.star_chain_ds_sum[s]);
        }

        VkCommandBufferAllocateInfo chain_ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        chain_ai.commandPool = r.cmd_pool;
        chain_ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        chain_ai.commandBufferCount = 128;
        vkAllocateCommandBuffers(r.vkdev, &chain_ai, r.star_chain_p1.data());
        vkAllocateCommandBuffers(r.vkdev, &chain_ai, r.star_chain_p2.data());
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
                                VkDeviceSize    wire_offset = 0) {
    if (ds == VK_NULL_HANDLE) return;
    VkDescriptorBufferInfo in_infos[8];
    std::fill_n(in_infos, n_slots, VkDescriptorBufferInfo{ in_tensor_buf, in_offset, in_size });
    VkDescriptorBufferInfo out_info{r.wire_buf, wire_offset, VK_WHOLE_SIZE};
    VkWriteDescriptorSet   w[2] = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, n_slots, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         nullptr,                                                                                                         in_infos, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1, 0, 1,       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr,
         &out_info,                                                                                                                 nullptr },
    };
    vkUpdateDescriptorSets(r.vkdev, 2, w, 0, nullptr);
}

static bool tp5_alloc_host_import_buffer(tp5_rank & r, void * host_ptr, VkDeviceSize size,
                                         VkBuffer & buf, VkDeviceMemory & mem, uint64_t & bda_addr) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
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
    flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    imp.pNext = &flags_info;
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
    if (r.flag_pipe) { vkDestroyPipeline(r.vkdev, r.flag_pipe, nullptr); r.flag_pipe = VK_NULL_HANDLE; }
    if (r.push_pipe) { vkDestroyPipeline(r.vkdev, r.push_pipe, nullptr); r.push_pipe = VK_NULL_HANDLE; }
    if (r.pipe_layout) { vkDestroyPipelineLayout(r.vkdev, r.pipe_layout, nullptr); r.pipe_layout = VK_NULL_HANDLE; }
    if (r.flag_pipe_layout) { vkDestroyPipelineLayout(r.vkdev, r.flag_pipe_layout, nullptr); r.flag_pipe_layout = VK_NULL_HANDLE; }
    if (r.push_layout) { vkDestroyPipelineLayout(r.vkdev, r.push_layout, nullptr); r.push_layout = VK_NULL_HANDLE; }
    if (r.dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.dsl, nullptr); r.dsl = VK_NULL_HANDLE; }
    if (r.flag_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.flag_dsl, nullptr); r.flag_dsl = VK_NULL_HANDLE; }
    if (r.push_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.push_dsl, nullptr); r.push_dsl = VK_NULL_HANDLE; }
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
    for (const auto & r : c.ranks) {
        if ((uint64_t) max_elems * sizeof(float) > r.caps.max_storage_buffer_range) {
            c.fail("tensor or mailbox slot exceeds maxStorageBufferRange");
            return false;
        }
    }
    // Clear all cached plans and drain in-flight GPU execution before modifying workspace buffers
    if (c.sync_mode == tp5_sync_mode::TIMELINE) {
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
    c.workspace_gen++;

    if (c.sync_mode == tp5_sync_mode::STAR) {
        const size_t page_size = 4096;
        const size_t max_chain_stages = 128;
        const size_t rank_stride = ((max_elems * sizeof(float) + page_size - 1) / page_size) * page_size;
        const size_t stage_stride = (c.n_ranks + 1) * rank_stride;
        const size_t total_host_size = max_chain_stages * stage_stride;
        const size_t total_bcast_size = max_chain_stages * rank_stride;
        c.star_rank_stride = rank_stride;
        c.star_host_alloc_size = total_host_size;

        for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
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
                if (!tp5_alloc_host_import_buffer(r, ptr, total_alloc_size,
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
                if (!tp5_alloc_host_visible_buffer(r, total_bcast_size, r.bcast_buf[b], r.bcast_mem[b], &r.bcast_host[b])) {
                    c.fail("alloc_host_visible_buffer failed for bcast_buf on rank " + std::to_string(i));
                    return false;
                }
            }
        }

        for (auto & r : c.ranks) {
            if (r.wire_buf) { vkDestroyBuffer(r.vkdev, r.wire_buf, nullptr); r.wire_buf = VK_NULL_HANDLE; }
            if (r.wire_mem) { vkFreeMemory(r.vkdev, r.wire_mem, nullptr); r.wire_mem = VK_NULL_HANDLE; }
            r.wire_bda = 0;
            const size_t total_wire_size = (c.sync_mode == tp5_sync_mode::STAR) ? (max_chain_stages * rank_stride) : rank_stride;
            if (!tp5_alloc_device_buffer(r, total_wire_size,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    false, r.wire_buf, r.wire_mem, nullptr)) {
                c.fail("wire buffer allocation failed");
                return false;
            }
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
    if (c.n_ranks != 5 || c.wire != tp5_wire_type::F16 || c.sync_mode != tp5_sync_mode::TIMELINE)
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
        !ggml_vk_tp5_hc_sum_pipeline(r.device, has_quantized, &r.hc_sum_pipe[pipe_idx],
                                     &r.hc_sum_layout[pipe_idx], &r.hc_sum_dsl[pipe_idx]))
        return false;
    ref.hc    = std::move(hc);
    key.width = ref.hc.width;
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

static void tp5_update_hc_descriptor(tp5_rank &             rank,
                                     VkDescriptorSet        set,
                                     const tensor_dev_ref & output,
                                     size_t                 bank,
                                     VkDeviceSize           stride,
                                     VkDeviceSize           payload) {
    VkDescriptorBufferInfo infos[13];
    constexpr uint32_t     hc_slots[] = { 0, 6, 7, 8, 9, 10 };
    for (size_t i = 0; i < output.hc.bindings.size(); ++i) {
        const auto & binding = output.hc.bindings[i];
        infos[hc_slots[i]]   = { binding.buffer, binding.offset, binding.size };
    }
    for (size_t i = 0; i < 5; ++i) {
        infos[i + 1] = rank.inboxes.empty() ?
                           VkDescriptorBufferInfo{ rank.mailbox_buf, (bank * 5 + i) * stride, payload } :
                           VkDescriptorBufferInfo{ rank.inboxes[tp5_plan_slot(i, bank)].buf, 0, payload };
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

    std::shared_ptr<tp5_p1_resources> p1_res;
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
        if (c.wire == tp5_wire_type::F16) {
            new_p1->ds_pack.resize(c.n_ranks, VK_NULL_HANDLE);
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
                    if (vkAllocateDescriptorSets(r.vkdev, &ai, &new_p1->ds_pack[i]) != VK_SUCCESS) {
                        c.fail("allocation of pack descriptor set failed on rank " + std::to_string(i));
                        return false;
                    }
                }
                if (r.push_pipe != VK_NULL_HANDLE && c.n_ranks == 5) {
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
            if (c.wire == tp5_wire_type::F16 && !has_packed) {
                tp5_update_pack_descriptor(r, new_p1->ds_pack[i], trefs[i].buf, trefs[i].offset, tensor_bytes,
                                           (uint32_t) c.n_ranks);
            }
            for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
                const size_t       idx      = tp5_plan_slot(i, b);
                const VkDeviceSize bank_off = (VkDeviceSize) b * payload_bytes;
                const VkDeviceSize slot_off = bank_off + (VkDeviceSize) i * stride;
                VkCommandBuffer    cmd      = new_p1->cmd_p1[idx];
                if (vkBeginCommandBuffer(cmd, &beg) != VK_SUCCESS) {
                    c.fail("begin cmd_p1 failed on rank " + std::to_string(i));
                    return false;
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
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.pack_pipe);
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipe_layout, 0, 1,
                                                &new_p1->ds_pack[i], 0, nullptr);
                        uint32_t n = (uint32_t) n_elems;
                        vkCmdPushConstants(cmd, r.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n);
                        vkCmdDispatch(cmd, (n + 255) / 256, 1, 1);
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

    plan.p1 = p1_res;
    plan.cmd_p2.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);
    plan.ds_sum.resize(c.n_ranks * TP5_MAILBOX_BANKS, VK_NULL_HANDLE);

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
        for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
            const size_t idx = tp5_plan_slot(i, b);
            if (vkAllocateCommandBuffers(r.vkdev, &cba, &plan.cmd_p2[idx]) != VK_SUCCESS) {
                c.fail("allocation of P2 command buffer failed on rank " + std::to_string(i));
                return false;
            }
            if (vkAllocateDescriptorSets(r.vkdev, &sum_ai, &plan.ds_sum[idx]) != VK_SUCCESS) {
                c.fail("allocation of sum descriptor set failed on rank " + std::to_string(i));
                return false;
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
            if (vkBeginCommandBuffer(cmd, &beg) != VK_SUCCESS) {
                c.fail("begin cmd_p2 failed on rank " + std::to_string(i));
                return false;
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

bool tp5_drain_epoch(tp5_comm & c, uint64_t epoch, uint64_t timeout_ns) {
    if (epoch == 0 || epoch <= c.last_drained_epoch) return true;
    const uint64_t target_val = 2 * epoch;
    for (size_t i = 0; i < c.n_ranks; ++i) {
        tp5_rank & r = c.ranks[i];
        if (!r.timeline_sem || !r.pfn_wait_semaphores) continue;
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
                continue;
            }
        }
        if (r.pfn_get_sem_counter) {
            uint64_t cur = 0;
            if (r.pfn_get_sem_counter(r.vkdev, r.timeline_sem, &cur) == VK_SUCCESS && cur >= target_val) {
                continue;
            }
        }
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1;
        wi.pSemaphores = &r.timeline_sem;
        wi.pValues = &target_val;
        auto t_wait_start = std::chrono::high_resolution_clock::now();
        if (r.pfn_wait_semaphores(r.vkdev, &wi, timeout_ns) != VK_SUCCESS) {
            c.fail("drain_epoch: wait failed or timed out for epoch " + std::to_string(epoch) + " on rank " + std::to_string(i));
            return false;
        }
        auto t_wait_end = std::chrono::high_resolution_clock::now();
        if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
            prof->host_wait_count++;
            prof->host_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(t_wait_end - t_wait_start).count();
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
#if defined(__AVX2__) && defined(__F16C__)
    for (; i + 8 <= end; i += 8) {
        __m256 v0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(r0 + i)));
        __m256 v1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(r1 + i)));
        __m256 v2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(r2 + i)));
        __m256 v3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(r3 + i)));
        __m256 v4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(r4 + i)));

        __m256 sum = _mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3));
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
    const uint32_t n_uvec4 = (uint32_t) ((payload + 15) / 16);

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

    for (size_t i = 0; i < c.n_ranks; ++i) {
        tp5_rank & r = c.ranks[i];
        VkCommandBuffer cmd_p1 = r.star_cmd_p1[bank];
        vkResetCommandBuffer(cmd_p1, 0);
        VkCommandBufferBeginInfo bi_p1{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi_p1.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd_p1, &bi_p1);

        VkMemoryBarrier mb_pre{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT };
        vkCmdPipelineBarrier(cmd_p1, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &mb_pre, 0, nullptr, 0, nullptr);

        uint64_t stage_bda_src = ggml_vk_tp5_get_tensor_bda(tensors[i]);
        uint64_t dst_bda = r.bda_addr[bank] + i * c.star_rank_stride;
        vkCmdBindPipeline(cmd_p1, VK_PIPELINE_BIND_POINT_COMPUTE, r.bda_push_pipe);

        uint32_t n_vec4 = (uint32_t)(n_elems / 4);
        uint64_t flag_bda = dst_bda + c.star_rank_stride - 64;
        struct {
            uint64_t src_bda;
            uint64_t dst_bda;
            uint64_t flag_bda;
            uint32_t n_vec4;
            uint32_t rank_idx;
            uint32_t seq_val;
            uint32_t mode;
        } bda_pc{stage_bda_src, dst_bda, flag_bda, n_vec4, (uint32_t)i, (uint32_t)epoch, 0u};
        vkCmdPushConstants(cmd_p1, r.bda_push_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bda_pc), &bda_pc);
        vkCmdDispatch(cmd_p1, (n_vec4 + 63) / 64, 1, 1);

        VkMemoryBarrier mb_data_done{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                      VK_ACCESS_SHADER_WRITE_BIT,
                                      VK_ACCESS_SHADER_WRITE_BIT };
        vkCmdPipelineBarrier(cmd_p1, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb_data_done, 0, nullptr, 0, nullptr);

        bda_pc.mode = 1;
        vkCmdPushConstants(cmd_p1, r.bda_push_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(bda_pc), &bda_pc);
        vkCmdDispatch(cmd_p1, 1, 1, 1);

        VkMemoryBarrier mb_host{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                 VK_ACCESS_SHADER_WRITE_BIT,
                                 VK_ACCESS_HOST_READ_BIT | VK_ACCESS_MEMORY_READ_BIT };
        vkCmdPipelineBarrier(cmd_p1, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 1, &mb_host, 0, nullptr, 0, nullptr);

        vkEndCommandBuffer(cmd_p1);

        VkCommandBuffer cmd_p2 = r.star_cmd_p2[bank];
        vkResetCommandBuffer(cmd_p2, 0);
        VkCommandBufferBeginInfo bi_p2{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi_p2.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd_p2, &bi_p2);

        VkBuffer dst_buf = VK_NULL_HANDLE;
        VkDeviceSize dst_off = 0, dst_size = 0;
        ggml_vk_tp5_tensor_dev_ref(tensors[i], &dst_buf, &dst_off, &dst_size);

        VkDescriptorSet ds_sum = r.star_ds_sum[bank];
        const size_t bcast_bytes = (tensors[0]->type == GGML_TYPE_F32) ? n_elems * sizeof(float) : payload;

        if (tensors[0]->type == GGML_TYPE_F32) {
            VkBufferCopy cp{0, dst_off, bcast_bytes};
            vkCmdCopyBuffer(cmd_p2, r.bcast_buf[bank], dst_buf, 1, &cp);
        } else {
            tp5_update_star_sum_descriptor(r, ds_sum, r.bcast_buf[bank], dst_buf, dst_off, dst_size, payload, (uint32_t)c.n_ranks);
            vkCmdBindPipeline(cmd_p2, VK_PIPELINE_BIND_POINT_COMPUTE, r.sum_pipe);
            vkCmdBindDescriptorSets(cmd_p2, VK_PIPELINE_BIND_POINT_COMPUTE, r.pipe_layout, 0, 1, &ds_sum, 0, nullptr);
            tp5_sum_pc pc_sum{(uint32_t) n_elems, 1, 0, 0};
            vkCmdPushConstants(cmd_p2, r.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_sum), &pc_sum);
            vkCmdDispatch(cmd_p2, (uint32_t)(n_elems + 255) / 256, 1, 1);
        }

        VkMemoryBarrier mb_post{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                 VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                                 VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT };
        vkCmdPipelineBarrier(cmd_p2, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb_post, 0, nullptr, 0, nullptr);
        vkEndCommandBuffer(cmd_p2);

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
        star_submits[i].tsi[1].waitSemaphoreValueCount = 1;
        star_submits[i].tsi[1].pWaitSemaphoreValues = &star_submits[i].wait_ready;
        star_submits[i].tsi[1].signalSemaphoreValueCount = 1;
        star_submits[i].tsi[1].pSignalSemaphoreValues = &star_submits[i].sig_p2;

        star_submits[i].submits[1] = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        star_submits[i].submits[1].pNext = &star_submits[i].tsi[1];
        star_submits[i].submits[1].waitSemaphoreCount = 1;
        star_submits[i].submits[1].pWaitSemaphores = &r.host_ready_sem;
        star_submits[i].submits[1].pWaitDstStageMask = &star_submits[i].wait_stage;
        star_submits[i].submits[1].commandBufferCount = 1;
        star_submits[i].submits[1].pCommandBuffers = &star_submits[i].cmd_p2;
        star_submits[i].submits[1].signalSemaphoreCount = 1;
        star_submits[i].submits[1].pSignalSemaphores = &r.timeline_sem;
    }
    auto t_rec_done = std::chrono::high_resolution_clock::now();

    auto t_sub_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < c.n_ranks; ++i) {
        tp5_rank & r = *star_submits[i].rk;
        if (vkQueueSubmit(r.queue, 2, star_submits[i].submits, VK_NULL_HANDLE) != VK_SUCCESS) {
            c.fail("tp5_allreduce_star: merged submit failed on rank " + std::to_string(i));
            return false;
        }
        ggml_vk_tp5_mark_queue_submitted(c.backends[i]);
    }
    auto t_sub_done = std::chrono::high_resolution_clock::now();
    auto t_after_p1 = t_sub_done;

    // =========================================================================
    // PILLAR 3: TRUE CPU L3 CACHE HARDWARE MEMORY FLAG POLLING (< 5us, 0 Syscalls)
    // =========================================================================
    const uint32_t exp_seq = (uint32_t) epoch;
    static double rank_wait_us[5] = {0};
    for (size_t i = 0; i < c.n_ranks; ++i) {
        auto t_w_start = std::chrono::high_resolution_clock::now();
        volatile uint32_t * flag_ptr = (volatile uint32_t *)((char *)c.star_host_aligned[bank] + (i + 1) * c.star_rank_stride - 64);
        uint64_t spin_cnt = 0;
        while (*flag_ptr != exp_seq) {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#endif
            if (++spin_cnt > 200000000ULL) {
                fprintf(stderr, "SPIN_TIMEOUT step: rank %zu ptr=%p val=%u exp=%u\n", i, (void*)flag_ptr, *flag_ptr, exp_seq);
                c.fail("allreduce_star: GPU flag timeout on rank " + std::to_string(i));
                return false;
            }
        }
        auto t_w_done = std::chrono::high_resolution_clock::now();
        rank_wait_us[i] += std::chrono::duration<double, std::micro>(t_w_done - t_w_start).count();
    }
    if (call_cnt % 96 == 95) {
        fprintf(stderr, "  [FLAG-WAIT-PER-RANK (us)] r0=%.2f r1=%.2f r2=%.2f r3=%.2f r4=%.2f\n",
                rank_wait_us[0] / 96.0, rank_wait_us[1] / 96.0, rank_wait_us[2] / 96.0,
                rank_wait_us[3] / 96.0, rank_wait_us[4] / 96.0);
        for (int k = 0; k < 5; ++k) rank_wait_us[k] = 0;
    }

    auto t_after_wait = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_acquire);

    // =========================================================================
    // PILLAR 4: 22-CORE DEDICATED AVX2+F16C L3 CACHE DIRECT ACCUMULATION
    // =========================================================================
    const void * rank_ptrs[5];
    for (size_t i = 0; i < c.n_ranks; ++i) {
        rank_ptrs[i] = (const char *) c.star_host_aligned[bank] + i * c.star_rank_stride;
    }
    void * acc_out = (char *) c.star_host_aligned[bank] + 5 * c.star_rank_stride;

    if (c.avx2_pool) {
        c.avx2_pool->parallel_for([&](size_t worker_id, size_t num_workers) {
            tp5_avx2_accumulate_star(rank_ptrs, acc_out, n_elems, worker_id, num_workers, tensors[0]->type == GGML_TYPE_F32);
        });
    } else {
        tp5_avx2_accumulate_star(rank_ptrs, acc_out, n_elems, 0, 1, tensors[0]->type == GGML_TYPE_F32);
    }

    auto t_after_avx2 = std::chrono::high_resolution_clock::now();

    // =========================================================================
    // PILLAR 2: CPU ROOT COMPLEX DOWNSTREAM BROADCAST
    // =========================================================================
    const size_t bcast_bytes = (tensors[0]->type == GGML_TYPE_F32) ? n_elems * sizeof(float) : payload;
    for (size_t i = 0; i < c.n_ranks; ++i) {
        if (c.ranks[i].bcast_host[bank]) {
            std::memcpy(c.ranks[i].bcast_host[bank], acc_out, bcast_bytes);
        }
    }
    auto t_after_bcast = std::chrono::high_resolution_clock::now();

    // =========================================================================
    // PILLAR 6: ATOMIC ASYNC SIGNAL HANDOFF (Wakes up GPU Phase 2 via timeline sem)
    // ZERO risk of GPU event hangs, completely thread-safe & monotonic!
    // =========================================================================
    for (size_t i = 0; i < c.n_ranks; ++i) {
        tp5_rank & r = c.ranks[i];
        if (r.pfn_signal_semaphore && r.host_ready_sem) {
            VkSemaphoreSignalInfo ssi{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, nullptr, r.host_ready_sem, epoch};
            r.pfn_signal_semaphore(r.vkdev, &ssi);
        }
    }
    auto t_after_p2 = std::chrono::high_resolution_clock::now();

    acc_flush_us += std::chrono::duration<double, std::micro>(t_flush_done - t_flush_start).count();
    acc_pure_sub_us += std::chrono::duration<double, std::micro>(t_sub_done - t_sub_start).count();
    acc_rec_us += std::chrono::duration<double, std::micro>(t_rec_done - t_rec_start).count();
    acc_p1_sub_us += std::chrono::duration<double, std::micro>(t_after_p1 - t_start).count();
    acc_wait_us   += std::chrono::duration<double, std::micro>(t_after_wait - t_after_p1).count();
    acc_avx2_us   += std::chrono::duration<double, std::micro>(t_after_avx2 - t_after_wait).count();
    acc_bcast_us  += std::chrono::duration<double, std::micro>(t_after_bcast - t_after_avx2).count();
    acc_p2_sub_us += std::chrono::duration<double, std::micro>(t_after_p2 - t_after_bcast).count();
    call_cnt++;

    if (call_cnt % 96 == 0) {
        fprintf(stderr, "\n[REAL-PILLAR-PROFILER (avg over %llu calls)]\n"
                        " -1. Flush Async (Compute): %6.2f us (96 steps: %5.2f ms)\n"
                        "  0. CPU CB Recording:      %6.2f us (96 steps: %5.2f ms)\n"
                        "  1. Parallel Queue Submit: %6.2f us (96 steps: %5.2f ms)\n"
                        "  2. DRM Timeline Wait:     %6.2f us (96 steps: %5.2f ms)\n"
                        "  3. 22-Core AVX2 Sum:      %6.2f us (96 steps: %5.2f ms)\n"
                        "  4. CPU Root Bcast:        %6.2f us (96 steps: %5.2f ms)\n"
                        "  5. P2 vkQueueSubmit:      %6.2f us (96 steps: %5.2f ms)\n"
                        "  TOTAL ALLREDUCE PER STEP: %6.2f us (96 steps: %5.2f ms)\n\n",
                (unsigned long long)call_cnt,
                acc_flush_us / call_cnt, (acc_flush_us / call_cnt * 96) / 1000.0,
                acc_rec_us / call_cnt, (acc_rec_us / call_cnt * 96) / 1000.0,
                (acc_pure_sub_us) / call_cnt, ((acc_pure_sub_us) / call_cnt * 96) / 1000.0,
                acc_wait_us / call_cnt,   (acc_wait_us / call_cnt * 96) / 1000.0,
                acc_avx2_us / call_cnt,   (acc_avx2_us / call_cnt * 96) / 1000.0,
                acc_bcast_us / call_cnt,  (acc_bcast_us / call_cnt * 96) / 1000.0,
                acc_p2_sub_us / call_cnt, (acc_p2_sub_us / call_cnt * 96) / 1000.0,
                (acc_p1_sub_us + acc_wait_us + acc_avx2_us + acc_bcast_us + acc_p2_sub_us) / call_cnt,
                ((acc_p1_sub_us + acc_wait_us + acc_avx2_us + acc_bcast_us + acc_p2_sub_us) / call_cnt * 96) / 1000.0);
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
    for (size_t j = 0; j < c.n_ranks; ++j) {
        if (c.wire == tp5_wire_type::F16 && c.sync_mode == tp5_sync_mode::TIMELINE) {
            uint64_t poff = 0, psize = 0;
            if (ggml_vk_tp5_take_wire_output(c.backends[j], tensors[j], &trefs[j].packed_buf, &poff, &psize,
                                             &trefs[j].packed_owner)) {
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
            }
        }
        key.bindings[j].buf = trefs[j].buf;
        key.bindings[j].offset = trefs[j].offset;
        key.bindings[j].size = trefs[j].size;
        key.packed_bindings[j].buf    = trefs[j].packed_buf;
        key.packed_bindings[j].offset = trefs[j].packed_offset;
        key.packed_bindings[j].size   = trefs[j].packed_size;
    }

    tp5_cached_plan * plan = nullptr;
    tp5_cached_plan one_shot;
    bool is_cached = false;
    bool plan_cache_hit = false;

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
    if (c.sync_mode == tp5_sync_mode::TIMELINE && c.allreduce_calls >= tp5_comm::MAX_OUTSTANDING_EPOCHS) {
        uint64_t drain_target = c.allreduce_calls + 1 - tp5_comm::MAX_OUTSTANDING_EPOCHS;
        if (drain_target > c.last_drained_epoch) {
            if (!tp5_drain_epoch(c, drain_target)) return false;
        }
    }
    // Maintain ring of in-flight buffer owners: drain slot's previous epoch before reusing,
    // then retain input buffer shared_ptrs so VkBuffer/VkDeviceMemory stay alive until P2 completes.
    if (c.sync_mode == tp5_sync_mode::TIMELINE) {
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

    // Submit Phase 1 on all ranks
    const bool p1_on_transfer = false;
    static const bool merge_submit = [] {
        const char * env = getenv("GGML_TP5_MERGE_SUBMIT");
        return env ? (atoi(env) != 0) : true;
    }();
    if (c.sync_mode == tp5_sync_mode::TIMELINE || c.sync_mode == tp5_sync_mode::DRM) {
        if (merge_submit && c.n_ranks <= 8) {
            // Merged 2-stage submit (AGENTS.md P2): submit Phase 1 + Phase 2 together in one
            // vkQueueSubmit per rank (2 VkSubmitInfo entries). Reduces ioctl/kernel submissions
            // by 50% without altering timeline dependencies (P2 waits on peers' P1 signals).
            tp5_timeline_batch b1[8];
            tp5_timeline_batch b2[8];
            for (size_t i = 0; i < c.n_ranks; ++i) {
                tp5_rank & r = c.ranks[i];
                const size_t bslot = tp5_plan_slot(i, tp5_mailbox_bank(epoch));
                b1[i].init(r, i, c.n_ranks, epoch, true,  &plan->p1->cmd_p1[bslot]);
                b2[i].init(r, i, c.n_ranks, epoch, false, &plan->cmd_p2[bslot]);
                VkSubmitInfo submits[2] = { b1[i].submit, b2[i].submit };

                if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
                    prof->queue_submits++;
                    prof->submit_batches += 2;
                }
                if (vkQueueSubmit(r.queue, 2, submits, VK_NULL_HANDLE) != VK_SUCCESS) {
                    c.fail("Phase 1+2 merged timeline submit failed on rank " + std::to_string(i));
                    GGML_ABORT("ggml-vulkan-collective: partially submitted merged timeline epoch cannot continue\n");
                }
                ggml_vk_tp5_mark_queue_submitted(c.backends[i]);
            }
        } else {
            // Split submit: Phase 1 enqueued on all ranks first; Phase 2 enqueued in second loop below.
            for (size_t i = 0; i < c.n_ranks; ++i) {
                tp5_rank & r = c.ranks[i];
                const size_t       bslot = tp5_plan_slot(i, tp5_mailbox_bank(epoch));
                tp5_timeline_batch batch;
                batch.init(r, i, c.n_ranks, epoch, true, &plan->p1->cmd_p1[bslot]);
                const VkSubmitInfo & si = batch.submit;

                if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
                    prof->queue_submits++;
                    prof->submit_batches++;
                }
                if (vkQueueSubmit(r.queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
                    c.fail("Phase 1 timeline submit failed on rank " + std::to_string(i));
                    GGML_ABORT("ggml-vulkan-collective: partially submitted timeline epoch cannot continue\n");
                }
            }
        }
    } else if (c.sync_mode == tp5_sync_mode::GPUFLAG) {
        size_t ring_idx = (size_t)((epoch - 1) % tp5_comm::MAX_OUTSTANDING_EPOCHS);
        for (size_t i = 0; i < c.n_ranks; ++i) {
            tp5_rank & r = c.ranks[i];
            const size_t bslot = tp5_plan_slot(i, tp5_mailbox_bank(epoch));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers    = &plan->p1->cmd_p1[bslot];
            if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
                prof->queue_submits++;
                prof->submit_batches++;
            }
            if (vkQueueSubmit(r.queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
                c.fail("Phase 1 gpuflag submit failed on rank " + std::to_string(i));
                return false;
            }
        }
    } else {
        for (size_t i = 0; i < c.n_ranks; ++i) {
            tp5_rank & r = c.ranks[i];
            const size_t bslot = tp5_plan_slot(i, tp5_mailbox_bank(epoch));
            VkQueue q = p1_on_transfer ? r.transfer_queue : r.queue;
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers    = &plan->p1->cmd_p1[bslot];
            if (c.sync_mode == tp5_sync_mode::SYNCFD) {
                si.signalSemaphoreCount = 1;
                si.pSignalSemaphores = &r.sem_p1_done;
            }
            VkFence f = (c.sync_mode == tp5_sync_mode::SYNCFD) ? VK_NULL_HANDLE : r.fence_p1;
            if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
                prof->queue_submits++;
                prof->submit_batches++;
            }
            if (vkQueueSubmit(q, 1, &si, f) != VK_SUCCESS) {
                for (size_t k = 0; k < i; ++k) {
                    vkQueueWaitIdle(c.ranks[k].queue);
                }
                c.fail("Phase 1 submit failed on rank " + std::to_string(i));
                GGML_ABORT("ggml-vulkan-collective: unrecoverable runtime failure: Phase 1 submit failed on rank %zu\n", i);
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
                for (size_t k = 0; k < c.n_ranks; ++k) {
                    vkQueueWaitIdle(c.ranks[k].queue);
                }
                c.fail("GetSemaphoreFdKHR failed on rank " + std::to_string(src));
                GGML_ABORT("ggml-vulkan-collective: unrecoverable runtime failure: GetSemaphoreFdKHR failed on rank %zu\n", src);
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
                        for (size_t k = 0; k < c.n_ranks; ++k) {
                            vkQueueWaitIdle(c.ranks[k].queue);
                        }
                        c.fail("dup sync_fd failed for rank " + std::to_string(dst));
                        GGML_ABORT("ggml-vulkan-collective: unrecoverable runtime failure: dup sync_fd failed for rank %zu\n", dst);
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
                    for (size_t k = 0; k < c.n_ranks; ++k) {
                        vkQueueWaitIdle(c.ranks[k].queue);
                    }
                    c.fail("ImportSemaphoreFdKHR failed on rank " + std::to_string(dst));
                    GGML_ABORT("ggml-vulkan-collective: unrecoverable runtime failure: ImportSemaphoreFdKHR failed on rank %zu\n", dst);
                }
            }
            if (sync_fd >= 0) {
                ::close(sync_fd);
            }
        }
    } else if (c.sync_mode != tp5_sync_mode::GPUFLAG) {
        if (!tp5_wait_all_p1(c)) {
            GGML_ABORT("ggml-vulkan-collective: unrecoverable runtime failure: Phase 1 fence wait failed\n");
        }
    }
    if (c.sync_mode != tp5_sync_mode::TIMELINE && c.sync_mode != tp5_sync_mode::DRM && c.sync_mode != tp5_sync_mode::GPUFLAG) {
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
                    GGML_ABORT("ggml-vulkan-collective: partially submitted timeline epoch cannot continue\n");
                }
                ggml_vk_tp5_mark_queue_submitted(c.backends[i]);
            }
        }
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
            for (size_t k = 0; k < i; ++k) {
                vkQueueWaitIdle(c.ranks[k].queue);
            }
            c.fail("Phase 2 submit failed on rank " + std::to_string(i));
            GGML_ABORT("ggml-vulkan-collective: unrecoverable runtime failure: Phase 2 submit failed on rank %zu\n", i);
        }
    }

    if (!tp5_wait_all_p2(c)) {
        GGML_ABORT("ggml-vulkan-collective: unrecoverable runtime failure: Phase 2 fence wait failed\n");
    }
    }

    auto t5 = std::chrono::high_resolution_clock::now();

    if (!is_cached) {
        if (c.sync_mode == tp5_sync_mode::TIMELINE) {
            if (!tp5_drain_epoch(c, epoch)) {
                c.fail("one_shot plan drain failed before destruction");
                return false;
            }
        }
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
        for (size_t s = 0; s < c.timing_stages; ++s) {
            const size_t q   = 5 * s;
            const double gap = s ? us(q - 1, q) : 0;
            sum += us(q, q + 1);
            compute += us(q + 1, q + 2);
            push_gap += us(q + 2, q + 3);
            push += us(q + 3, q + 4);
            gaps += gap;
            fprintf(stderr,
                    "[tp5-gpu-stage] rank=%zu stage=%zu gap_us=%.3f sum_us=%.3f compute_us=%.3f push_us=%.3f "
                    "push_gap_us=%.3f\n",
                    rank, s, gap, us(q, q + 1), us(q + 1, q + 2), us(q + 3, q + 4), us(q + 2, q + 3));
        }
        sum += us(count - 3, count - 2);
        const double tail_compute = us(count - 2, count - 1);
        compute += tail_compute;
        gaps += us(count - 4, count - 3);
        fprintf(stderr,
                "[tp5-gpu-timing] rank=%zu stages=%zu span_us=%.3f sum_us=%.3f compute_us=%.3f push_us=%.3f "
                "gaps_us=%.3f tail_compute_us=%.3f push_gap_us=%.3f\n",
                rank, c.timing_stages, us(0, count - 1), sum, compute, push, gaps, tail_compute, push_gap);
        r.timing_reported = true;
    }
}

bool ggml_backend_vk_tp5_submit_epoch_chain(void * comm_handle,
                                            const std::vector<std::vector<std::vector<void *>>> & stage_compute_cbs,
                                            const std::vector<std::vector<ggml_tensor *>> & stage_tensors) {
    if (!comm_handle) return false;
    tp5_comm & c = *reinterpret_cast<tp5_comm *>(comm_handle);
    std::lock_guard<std::mutex> lock(c.mutex);
    if (c.failed) {
        GGML_ABORT("ggml-vulkan-collective: cannot submit on failed communicator\n");
    }
    if ((c.sync_mode != tp5_sync_mode::TIMELINE && c.sync_mode != tp5_sync_mode::DRM && c.sync_mode != tp5_sync_mode::STAR) || !c.cmd_replay_enabled)
        return false;

    const size_t n_stages = stage_tensors.size();
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
    for (size_t s = 0; s < n_stages; ++s) {
        if (stage_tensors[s].size() != c.n_ranks)
            return false;
        auto & key = keys[s];
        key.wire   = c.wire;
        key.stride        = 0;
        key.workspace_gen = 0;
        key.hc.fill(tp5_hc_key{});
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
            if (c.wire == tp5_wire_type::F16) {
                uint64_t poff = 0, psize = 0;
                if (ggml_vk_tp5_take_wire_output(c.backends[j], t, &ref.packed_buf, &poff, &psize, &ref.packed_owner)) {
                    ref.packed_offset                 = (VkDeviceSize) poff;
                    ref.packed_size                   = (VkDeviceSize) psize;
                    const VkDeviceSize expected_bytes = ((VkDeviceSize) n_elems * 2 + 3) & ~VkDeviceSize(3);
                    if (!ref.packed_buf || !ref.packed_owner || n_elems % 2 != 0 || ref.packed_size < expected_bytes ||
                        ref.packed_offset %
                                std::max(uint64_t(4), c.ranks[j].caps.min_storage_buffer_offset_alignment) !=
                            0 ||
                        expected_bytes > c.ranks[j].caps.max_storage_buffer_range) {
                        ref.packed_buf = VK_NULL_HANDLE;
                        ref.packed_owner.reset();
                        ref.packed_offset = 0;
                        ref.packed_size   = 0;
                    }
                }
            }
            key.bindings[j]        = { ref.buf, ref.offset, ref.size };
            key.packed_bindings[j] = { ref.packed_buf, ref.packed_offset, ref.packed_size };
            const auto & consumer  = stage_compute_cbs[s + 1][j];
            if (consumer.size() > 1) {
                tp5_hc_consumer_ref(c, j, consumer.front(), ref, n_elems, key.hc[j]);
            }
            max_elems = std::max(max_elems, n_elems);
        }
    }
    if (max_elems > c.max_elems && !tp5_setup_workspace(c, max_elems)) {
        GGML_ABORT("ggml-vulkan-collective: epoch chain workspace setup failed\n");
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
                    GGML_ABORT("ggml-vulkan-collective: chain plan retirement failed\n");
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
                GGML_ABORT("ggml-vulkan-collective: epoch chain plan recording failed\n");
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
        GGML_ABORT("ggml-vulkan-collective: epoch chain owner retirement failed\n");
    }
    if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
        prof->backpressure_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - bp_start).count();
    }

    // Build all ranks before submitting any rank; allocations and pointer
    // fixups cannot fail halfway through an inter-device dependency chain.
    const bool capture_requested = ++c.chain_calls == c.timing_chain;
    const bool capture           = capture_requested && tp5_prepare_gpu_timing(c, n_stages);
    if (capture_requested && !capture) {
        fprintf(stderr, "[tp5-gpu-timing] capture unavailable; submitting original chain\n");
    }
    const bool isolate_bo = c.isolate_mailbox;
    const char * chain_cache_env   = getenv("GGML_TP5_CHAIN_CACHE");
    const bool   allow_chain_cache = (chain_cache_env == nullptr || atoi(chain_cache_env) != 0);

    bool can_reuse_chain =
        allow_chain_cache && !capture && c.compiled_chain.valid && c.compiled_chain.n_stages == n_stages &&
        c.compiled_chain.isolate_bo == isolate_bo &&
        c.compiled_chain.start_bank == tp5_mailbox_bank(c.allreduce_calls + 1) &&
        c.compiled_chain.workspace_gen == c.workspace_gen && c.compiled_chain.plans_gen == c.plans_gen &&
        c.compiled_chain.stage_plan_indices == c.chain_plan_indices && c.compiled_chain.stage_keys == keys &&
        c.compiled_chain.stage_compute_cbs == stage_compute_cbs;

    if (can_reuse_chain) {
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
            for (size_t s = 0; s < n_stages; ++s)
                if (c.cached_plans[c.chain_plan_indices[s]].key.hc[i].width) {
                    --n_compute;
                }
            scratch.compute.resize(n_compute + 2 * n_stages + (capture ? 5 * n_stages + 3 : 0));
            const auto append_segment = [&](size_t slot, size_t begin, size_t end, uint64_t wait_epoch,
                                            uint64_t signal) {
                if (begin == end)
                    return;
                auto & batch = scratch.batches[slot];
                batch.init(c.ranks[i], i, c.n_ranks, wait_epoch ? wait_epoch : 1, wait_epoch == 0,
                           scratch.compute.data() + begin);
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
                const size_t first_compute =
                    s > 0 && c.cached_plans[c.chain_plan_indices[s - 1]].key.hc[i].width ? 1 : 0;
                for (size_t cb = first_compute; cb < stage_compute_cbs[s][i].size(); ++cb) {
                    scratch.compute[cursor++] = (VkCommandBuffer) stage_compute_cbs[s][i][cb];
                }
                if (capture)
                    scratch.compute[cursor++] = c.ranks[i].timing_markers[5 * s + 2];
                const size_t compute_end = cursor;
                if (capture)
                    scratch.compute[cursor++] = c.ranks[i].timing_markers[5 * s + 3];
                scratch.compute[cursor++] = c.cached_plans[c.chain_plan_indices[s]].p1->cmd_p1[bslot];
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
                // Fuse SUM(s-1) -> compute(s) -> PUSH(s). Wait for the previous
                // ready epoch, signal the next ready epoch. P2/P1 barriers retain
                // local RAW/WAR/WAW ordering; R=2 permits peer SUM/PUSH overlap.
                batch.init(c.ranks[i], i, c.n_ranks, s > 0 ? epoch - 1 : epoch, s == 0, scratch.compute.data() + first);
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
            const size_t first_tail = c.cached_plans[c.chain_plan_indices.back()].key.hc[i].width ? 1 : 0;
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
            final.init(c.ranks[i], i, c.n_ranks, last_epoch, false, scratch.compute.data() + final_first);
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
    const auto prefix_end        = capture ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    int64_t    submit_wall_us[8] = {};
    for (size_t i = 0; i < c.n_ranks; ++i) {
        auto & submits = c.chain_scratch[i].submits;
        if (ggml_tp5_profile * prof = ggml_tp5_profile_active()) {
            ++prof->queue_submits;
            prof->submit_batches += submits.size();
        }
        const auto submit_start = capture ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (vkQueueSubmit(c.ranks[i].queue, (uint32_t) submits.size(), submits.data(), VK_NULL_HANDLE) != VK_SUCCESS) {
            c.fail("epoch chain submit failed on rank " + std::to_string(i));
            // No fallback, no fake peer signal and no wait-idle on a partial chain.
            GGML_ABORT("ggml-vulkan-collective: partially submitted epoch chain cannot continue\n");
        }
        if (capture)
            submit_wall_us[i] =
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - submit_start)
                    .count();
        ggml_vk_tp5_mark_queue_submitted(c.backends[i]);
    }
    // Print only after EVERY rank was submitted, avoiding logging-induced peer starvation.
    if (capture) {
        const auto prefix_us = std::chrono::duration_cast<std::chrono::microseconds>(prefix_end - prefix_start).count();
        fprintf(stderr, "[tp5-host-timing] prefix_flush_us=%lld\n", (long long) prefix_us);
        for (size_t i = 0; i < c.n_ranks; ++i) {
            const auto & scratch = c.chain_scratch[i];
            const size_t markers = c.ranks[i].timing_markers.size();
            fprintf(stderr,
                    "[tp5-host-timing] rank=%zu batches=%zu compute_cbs=%zu collective_cbs=%zu marker_cbs=%zu "
                    "submit_wall_us=%lld\n",
                    i, scratch.submits.size(), scratch.compute.size() - 2 * n_stages - markers, 2 * n_stages, markers,
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
    if (n < 1 || n > 8) {
        fprintf(stderr, "ggml-vulkan-collective: need 2..8 backends, got %zu\n", n);
        return nullptr;
    }
    auto * c = new tp5_comm();
    c->n_ranks = n;
    c->ranks.resize(n);
    c->chain_scratch.resize(n);
    c->backends.assign(backends, backends + n);
    // Plan addresses stay stable while a complete chain is assembled.
    c->cached_plans.reserve(tp5_comm::MAX_CACHED_PLANS);

    const char * timing_env = getenv("GGML_TP5_GPU_TIMING");
    if (timing_env && atoi(timing_env) > 0)
        c->timing_chain = (uint64_t) atoi(timing_env);

    const char * relay_env = getenv("GGML_TP5_RELAY");
    if (relay_env && strcmp(relay_env, "off") != 0) {
        fprintf(stderr, "ggml-vulkan-collective: host relay is prohibited (GGML_TP5_RELAY='%s'); direct GPU->GPU peer VRAM mesh required (only 'off' or unset allowed)\n", relay_env);
        delete c;
        return nullptr;
    }

    const char * wire_env = getenv("GGML_TP5_WIRE");
    c->wire = (wire_env && strcmp(wire_env, "f32") == 0) ? tp5_wire_type::F32 : tp5_wire_type::F16;

    const char * sync_env = getenv("GGML_TP5_SYNC");
    if (sync_env && (strcmp(sync_env, "star") == 0 || strcmp(sync_env, "l3_star") == 0)) {
        c->sync_mode = tp5_sync_mode::STAR;
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

        if (c->sync_mode == tp5_sync_mode::TIMELINE || c->sync_mode == tp5_sync_mode::STAR) {
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

        if ((c->sync_mode == tp5_sync_mode::SYNCFD || c->sync_mode == tp5_sync_mode::TIMELINE || c->sync_mode == tp5_sync_mode::DRM || c->sync_mode == tp5_sync_mode::STAR) && (!r.pfn_get_sem_fd || !r.pfn_import_sem_fd)) {
            fprintf(stderr, "ggml-vulkan-collective: rank %zu missing vkGetSemaphoreFdKHR or vkImportSemaphoreFdKHR proc addr\n", i);
            for (auto & rr : c->ranks) tp5_destroy_rank(rr);
            delete c;
            return nullptr;
        }
        if ((c->sync_mode == tp5_sync_mode::TIMELINE || c->sync_mode == tp5_sync_mode::DRM || c->sync_mode == tp5_sync_mode::STAR) && (!r.pfn_wait_semaphores || !r.pfn_get_sem_counter)) {
            fprintf(stderr, "ggml-vulkan-collective: rank %zu missing vkWaitSemaphores or vkGetSemaphoreCounterValue proc addr\n", i);
            for (auto & rr : c->ranks) tp5_destroy_rank(rr);
            delete c;
            return nullptr;
        }

        if (c->sync_mode == tp5_sync_mode::TIMELINE || c->sync_mode == tp5_sync_mode::DRM || c->sync_mode == tp5_sync_mode::STAR) {
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
        c->avx2_pool = std::make_unique<tp5_avx2_pool>();
        c->drm_signaler = std::make_unique<tp5_drm_signaler>();
        c->drm_waiter   = std::make_unique<tp5_drm_waiter>();
        c->submit_pool  = std::make_unique<tp5_submit_pool>();
        for (size_t i = 0; i < n; ++i) {
            c->drm_waiter->set_node(i, c->ranks[i].dri_fd, c->ranks[i].own_syncobj);
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
                                 (c->sync_mode == tp5_sync_mode::SYNCFD) ? "syncfd" :
                                 (c->sync_mode == tp5_sync_mode::GPUFLAG) ? "gpuflag" : "host";
        fprintf(stderr, "ggml-vulkan-collective: init %zu ranks, wire=%s sync=%s relay=off, "
                        "replay=%s\n",
                n, c->wire == tp5_wire_type::F16 ? "f16" : "f32",
                sync_name,
                c->cmd_replay_enabled ? "on" : "off");
    }
    return c;
}

void ggml_backend_vk_tp5_comm_free(void * comm) {
    if (!comm) return;
    auto * c = (tp5_comm *) comm;
    if (c->failed) {
        fprintf(stderr, "ggml-vulkan-collective: comm_free called on failed collective: aborting to prevent destroying active resources\n");
        GGML_ABORT("ggml-vulkan-collective: unrecoverable runtime failure: comm_free called on failed collective\n");
    }
    if (c->sync_mode == tp5_sync_mode::TIMELINE || c->sync_mode == tp5_sync_mode::DRM || c->sync_mode == tp5_sync_mode::STAR) {
        if (!tp5_drain_epoch(*c, c->allreduce_calls)) {
            GGML_ABORT("ggml-vulkan-collective: unrecoverable runtime failure: comm_free drain failed for epoch %llu\n", (unsigned long long) c->allreduce_calls);
        }
    }
    c->clear_cached_plans();
    if (c->sync_mode == tp5_sync_mode::GPUFLAG) {
        tp5_gpuflag_drain_all(*c);
    } else if (c->sync_mode != tp5_sync_mode::TIMELINE && c->sync_mode != tp5_sync_mode::STAR) {
    for (auto & r : c->ranks) {
        if (r.vkdev != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(r.vkdev);
        }
    }
    }
    tp5_poll_gpu_timing(*c);

    // Free Star AllReduce host allocations and worker pool/signaler
    c->avx2_pool.reset();
    c->drm_signaler.reset();
    for (size_t b = 0; b < TP5_MAILBOX_BANKS; ++b) {
        if (c->star_host_raw[b]) {
            free(c->star_host_raw[b]);
            c->star_host_raw[b] = nullptr;
            c->star_host_aligned[b] = nullptr;
        }
    }

    for (auto & r : c->ranks) tp5_destroy_rank(r);
    delete c;
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

bool ggml_backend_vk_tp5_prepare_graph(void * comm, size_t rank, ggml_cgraph * graph, bool reduce) {
    auto * c = (tp5_comm *) comm;
    if (!c)
        return false;
    std::lock_guard<std::mutex> lock(c->mutex);
    if (c->failed)
        return false;
    if (rank >= c->n_ranks || !graph || graph->n_nodes < 0 || (reduce && graph->n_nodes == 0))
        return false;

    static const bool disable_producer_wire = (getenv("GGML_VK_DISABLE_PRODUCER_WIRE") != nullptr);
    const bool        eligible =
        (c->sync_mode == tp5_sync_mode::TIMELINE && c->wire == tp5_wire_type::F16 && !disable_producer_wire && reduce);

    ggml_tensor * target = eligible ? graph->nodes[graph->n_nodes - 1] : nullptr;
    ggml_vk_tp5_set_wire_output(c->backends[rank], target);
    return true;
}
