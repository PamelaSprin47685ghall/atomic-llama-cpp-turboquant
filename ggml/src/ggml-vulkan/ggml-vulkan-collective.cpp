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

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vulkan-internal.h"
#include "ggml-vulkan-shaders.hpp"
#include "ggml-vulkan.h"
#include "ggml-tp5-profile.h"

#include <vulkan/vulkan.h>

#include <algorithm>
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
enum class tp5_sync_mode { HOST, SYNCFD, TIMELINE, GPUFLAG };

static VkDeviceSize tp5_flags_byte_offset(size_t n_ranks, VkDeviceSize stride) {
    return (VkDeviceSize) n_ranks * stride;
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

    // Fences: separate Phase 1 and Phase 2
    VkFence fence_p1 = VK_NULL_HANDLE;
    VkFence fence_p2 = VK_NULL_HANDLE;

    // Mailbox: device-local buffer with slots (slot r = data pushed by rank r)
    VkBuffer mailbox_buf = VK_NULL_HANDLE;
    VkDeviceMemory mailbox_mem = VK_NULL_HANDLE;
    VkDeviceSize mailbox_bytes = 0;
    VkDeviceSize mailbox_alloc_bytes = 0; // exporter vkAllocateMemory size (import must match)
    int mailbox_export_fd = -1;

    struct imported {
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        int peer = -1;
    };
    std::vector<imported> imports;

    // Wire staging buffer (canonical wire dtype)
    VkBuffer wire_buf = VK_NULL_HANDLE;
    VkDeviceMemory wire_mem = VK_NULL_HANDLE;

    // Semaphores (for syncfd mode)
    VkSemaphore sem_p1_done = VK_NULL_HANDLE;
    std::vector<VkSemaphore> wait_sems;
    PFN_vkGetSemaphoreFdKHR pfn_get_sem_fd = nullptr;
    PFN_vkImportSemaphoreFdKHR pfn_import_sem_fd = nullptr;

    // Timeline semaphores (for timeline mode, persistent across workspace resize)
    VkSemaphore timeline_sem = VK_NULL_HANDLE; // own timeline semaphore (rank signals this)
    int timeline_export_fd = -1;
    std::vector<VkSemaphore> peer_timeline_sems; // imported peer timeline semaphores (size n_ranks)

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

    // Per-epoch params for GPUFLAG replay-safe dynamic seq (host-coherent)
    VkBuffer epoch_buf = VK_NULL_HANDLE;
    VkDeviceMemory epoch_mem = VK_NULL_HANDLE;
    uint32_t * epoch_host = nullptr;
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
static bool tp5_gpuflag_drain_ring_slot(tp5_comm & c);

struct tp5_plan_key {
    size_t n_elems = 0;
    tp5_wire_type wire = tp5_wire_type::F32;
    VkDeviceSize stride = 0;
    uint64_t workspace_gen = 0;
    std::vector<tp5_binding_key> bindings;

    bool operator==(const tp5_plan_key & o) const {
        if (n_elems != o.n_elems || wire != o.wire || stride != o.stride || workspace_gen != o.workspace_gen) {
            return false;
        }
        if (bindings.size() != o.bindings.size()) return false;
        for (size_t i = 0; i < bindings.size(); ++i) {
            if (!(bindings[i] == o.bindings[i])) return false;
        }
        return true;
    }
};

struct tp5_cached_plan {
    tp5_plan_key key;
    std::vector<VkCommandBuffer> cmd_p1;
    std::vector<VkCommandBuffer> cmd_p2;
    std::vector<VkDescriptorSet> ds_sum;
    std::vector<VkDescriptorSet> ds_pack;
    std::vector<VkDescriptorSet> ds_flag;
    uint64_t last_used_call = 0;
};

struct tp5_comm {
    size_t n_ranks = 0;
    std::vector<tp5_rank> ranks;
    std::vector<ggml_backend_t> backends;
    tp5_wire_type wire = tp5_wire_type::F32;
    tp5_sync_mode sync_mode = tp5_sync_mode::HOST;
    bool cmd_replay_enabled = true;
    uint32_t spin_max = 100000000u;
    size_t max_elems = 0;
    uint64_t workspace_gen = 1;

    // Bounded immutable per-binding plan cache
    static constexpr size_t MAX_CACHED_PLANS = 64;
    std::vector<tp5_cached_plan> cached_plans;

    uint64_t allreduce_calls = 0;
    uint64_t host_waits = 0;
    uint64_t last_drained_epoch = 0;
    static constexpr uint64_t MAX_OUTSTANDING_EPOCHS = 4;

    struct tp5_in_flight_slot {
        uint64_t epoch = 0;
        std::vector<vk_buffer> owners;
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
        for (size_t i = 0; i < n_ranks; ++i) {
            tp5_rank & r = ranks[i];
            if (r.vkdev == VK_NULL_HANDLE) continue;
            if (i < plan.cmd_p1.size() && plan.cmd_p1[i] != VK_NULL_HANDLE && r.cmd_pool != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(r.vkdev, r.cmd_pool, 1, &plan.cmd_p1[i]);
                plan.cmd_p1[i] = VK_NULL_HANDLE;
            }
            if (i < plan.cmd_p2.size() && plan.cmd_p2[i] != VK_NULL_HANDLE && r.cmd_pool != VK_NULL_HANDLE) {
                vkFreeCommandBuffers(r.vkdev, r.cmd_pool, 1, &plan.cmd_p2[i]);
                plan.cmd_p2[i] = VK_NULL_HANDLE;
            }
            if (i < plan.ds_sum.size() && plan.ds_sum[i] != VK_NULL_HANDLE && r.desc_pool != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(r.vkdev, r.desc_pool, 1, &plan.ds_sum[i]);
                plan.ds_sum[i] = VK_NULL_HANDLE;
            }
            if (i < plan.ds_pack.size() && plan.ds_pack[i] != VK_NULL_HANDLE && r.desc_pool != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(r.vkdev, r.desc_pool, 1, &plan.ds_pack[i]);
                plan.ds_pack[i] = VK_NULL_HANDLE;
            }
            if (i < plan.ds_flag.size() && plan.ds_flag[i] != VK_NULL_HANDLE && r.desc_pool != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(r.vkdev, r.desc_pool, 1, &plan.ds_flag[i]);
                plan.ds_flag[i] = VK_NULL_HANDLE;
            }
        }
    }

    void clear_cached_plans() {
        if (sync_mode == tp5_sync_mode::TIMELINE) {
            if (!tp5_drain_epoch(*this, allreduce_calls)) {
                fail("clear_cached_plans: drain failed before plan destruction");
                return;
            }
        }
        if (sync_mode == tp5_sync_mode::GPUFLAG) {
            tp5_gpuflag_drain_ring_slot(*this);
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
            tp5_gpuflag_drain_ring_slot(*this);
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
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
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
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = 4;
        ci.pBindings = b;
        if (vkCreateDescriptorSetLayout(r.vkdev, &ci, nullptr, &r.dsl) != VK_SUCCESS) return false;
    }
    {
        VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, 20};
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
    auto mk = [&](VkShaderModule m, VkPipeline * p) -> bool {
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = m;
        ci.stage.pName = "main";
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

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024};
    VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ci.maxSets = 256;
    ci.poolSizeCount = 1;
    ci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(r.vkdev, &ci, nullptr, &r.desc_pool) != VK_SUCCESS) return false;

    return true;
}

// Bind mailbox payload (slots only, excludes flag tail) -> out_tensor in the sum descriptor set.
void tp5_update_sum_descriptor(tp5_rank & r, VkDescriptorSet ds,
                               VkBuffer out_tensor_buf, VkDeviceSize out_offset, VkDeviceSize out_size,
                               VkDeviceSize payload_bytes, VkDeviceSize flags_byte_offset, VkDeviceSize flags_bytes,
                               bool bind_epoch) {
    VkDescriptorBufferInfo in_info{r.mailbox_buf, 0, payload_bytes};
    VkDescriptorBufferInfo out_info{out_tensor_buf, out_offset, out_size};
    VkDescriptorBufferInfo flag_info{r.mailbox_buf, flags_byte_offset, flags_bytes};
    VkWriteDescriptorSet w[4] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &in_info, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &out_info, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 2, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &flag_info, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 3, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, nullptr, nullptr},
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

void tp5_record_flag_dispatch(VkCommandBuffer cmd, tp5_rank & r, VkDescriptorSet ds_flag, const tp5_flag_pc & pc) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.flag_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, r.flag_pipe_layout, 0, 1, &ds_flag, 0, nullptr);
    vkCmdPushConstants(cmd, r.flag_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, 1, 1, 1);
}

void tp5_update_pack_descriptor(tp5_rank & r, VkDescriptorSet ds,
                                VkBuffer in_tensor_buf, VkDeviceSize in_offset, VkDeviceSize in_size) {
    VkDescriptorBufferInfo in_info{in_tensor_buf, in_offset, in_size};
    VkDescriptorBufferInfo out_info{r.wire_buf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[2] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &in_info, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1, 0, 1,
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &out_info, nullptr},
    };
    vkUpdateDescriptorSets(r.vkdev, 2, w, 0, nullptr);
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

void tp5_destroy_rank(tp5_rank & r) {
    if (r.vkdev == VK_NULL_HANDLE) return;

    if (r.fence_p1) { vkDestroyFence(r.vkdev, r.fence_p1, nullptr); r.fence_p1 = VK_NULL_HANDLE; }
    if (r.fence_p2) { vkDestroyFence(r.vkdev, r.fence_p2, nullptr); r.fence_p2 = VK_NULL_HANDLE; }
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

    for (auto & im : r.imports) {
        if (im.buf) { vkDestroyBuffer(r.vkdev, im.buf, nullptr); im.buf = VK_NULL_HANDLE; }
        if (im.mem) { vkFreeMemory(r.vkdev, im.mem, nullptr); im.mem = VK_NULL_HANDLE; }
    }
    r.imports.clear();

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
    if (r.pipe_layout) { vkDestroyPipelineLayout(r.vkdev, r.pipe_layout, nullptr); r.pipe_layout = VK_NULL_HANDLE; }
    if (r.flag_pipe_layout) { vkDestroyPipelineLayout(r.vkdev, r.flag_pipe_layout, nullptr); r.flag_pipe_layout = VK_NULL_HANDLE; }
    if (r.dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.dsl, nullptr); r.dsl = VK_NULL_HANDLE; }
    if (r.flag_dsl) { vkDestroyDescriptorSetLayout(r.vkdev, r.flag_dsl, nullptr); r.flag_dsl = VK_NULL_HANDLE; }
    if (r.desc_pool) { vkDestroyDescriptorPool(r.vkdev, r.desc_pool, nullptr); r.desc_pool = VK_NULL_HANDLE; }
    if (r.cmd_pool) { vkDestroyCommandPool(r.vkdev, r.cmd_pool, nullptr); r.cmd_pool = VK_NULL_HANDLE; }
}

bool tp5_setup_workspace(tp5_comm & c, size_t max_elems) {
    // Clear all cached plans and drain in-flight GPU execution before modifying workspace buffers
    if (c.sync_mode == tp5_sync_mode::TIMELINE) {
        if (!tp5_drain_epoch(c, c.allreduce_calls)) {
            c.fail("setup_workspace: drain failed before workspace reallocation");
            return false;
        }
    } else if (c.sync_mode == tp5_sync_mode::GPUFLAG) {
        for (auto & r : c.ranks) {
            if (r.vkdev != VK_NULL_HANDLE && r.fence_p2 != VK_NULL_HANDLE) {
                vkWaitForFences(r.vkdev, 1, &r.fence_p2, VK_TRUE, UINT64_MAX);
                vkResetFences(r.vkdev, 1, &r.fence_p2);
            }
        }
    }
    c.clear_cached_plans();
    if (c.failed) return false;
    c.workspace_gen++;

    const size_t wire_b = c.wire == tp5_wire_type::F16 ? 2 : 4;
    const VkDeviceSize wire_bytes = (VkDeviceSize) max_elems * wire_b;
    const VkDeviceSize stride = wire_bytes;
    const VkDeviceSize flags_bytes = tp5_flags_region_bytes(c.n_ranks, c.sync_mode);
    const VkDeviceSize mailbox_bytes = (VkDeviceSize) c.n_ranks * stride + flags_bytes;

    for (auto & r : c.ranks) {
        for (auto & im : r.imports) {
            if (im.buf) { vkDestroyBuffer(r.vkdev, im.buf, nullptr); im.buf = VK_NULL_HANDLE; }
            if (im.mem) { vkFreeMemory(r.vkdev, im.mem, nullptr); im.mem = VK_NULL_HANDLE; }
        }
        r.imports.clear();
        if (r.wire_buf) { vkDestroyBuffer(r.vkdev, r.wire_buf, nullptr); r.wire_buf = VK_NULL_HANDLE; }
        if (r.wire_mem) { vkFreeMemory(r.vkdev, r.wire_mem, nullptr); r.wire_mem = VK_NULL_HANDLE; }
        if (r.mailbox_export_fd >= 0) { ::close(r.mailbox_export_fd); r.mailbox_export_fd = -1; }
        if (r.mailbox_buf) { vkDestroyBuffer(r.vkdev, r.mailbox_buf, nullptr); r.mailbox_buf = VK_NULL_HANDLE; }
        if (r.mailbox_mem) { vkFreeMemory(r.vkdev, r.mailbox_mem, nullptr); r.mailbox_mem = VK_NULL_HANDLE; }

        // Mailbox: exported DMA-BUF with full TRANSFER + STORAGE usage (matches POC)
        if (!tp5_alloc_device_buffer(r, mailbox_bytes,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                true, r.mailbox_buf, r.mailbox_mem, &r.mailbox_export_fd)) {
            c.fail("mailbox allocation failed");
            return false;
        }
        {
            VkMemoryRequirements mreq{};
            vkGetBufferMemoryRequirements(r.vkdev, r.mailbox_buf, &mreq);
            r.mailbox_alloc_bytes = mreq.size;
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
            tp5_fd fd(::dup(c.ranks[j].mailbox_export_fd));
            if (fd.fd < 0) {
                c.fail("dup of peer mailbox fd failed");
                return false;
            }
            tp5_rank::imported im;
            im.peer = (int) j;
            if (!tp5_import_peer_buffer(c.ranks[i], fd.release(), mailbox_bytes,
                    c.ranks[j].mailbox_alloc_bytes, im.buf, im.mem)) {
                c.fail("import of peer mailbox failed (rank " + std::to_string(i) +
                       " <- rank " + std::to_string(j) + ")");
                return false;
            }
            c.ranks[i].imports.push_back(im);
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

struct tensor_dev_ref {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    vk_buffer owner;
    bool ok = false;
};

tensor_dev_ref tp5_tensor_dev_ref(ggml_tensor * t) {
    tensor_dev_ref r{};
    r.ok = ggml_vk_tp5_tensor_dev_ref(t, &r.buf, &r.offset, &r.size, &r.owner);
    return r;
}

struct tp5_sum_pc {
    uint32_t n_elems;
    uint32_t n_slots;
    uint32_t stride_elems;
    uint32_t use_flags;
    uint32_t ready_off_u32;
};

bool tp5_record_plan(tp5_comm & c, tp5_cached_plan & plan, const std::vector<tensor_dev_ref> & trefs,
                     size_t n_elems, VkDeviceSize flags_base) {
    const size_t wire_b = c.wire == tp5_wire_type::F16 ? 2 : 4;
    const VkDeviceSize payload = (VkDeviceSize) n_elems * wire_b;
    const VkDeviceSize stride = (VkDeviceSize) c.max_elems * wire_b;
    const VkDeviceSize tensor_bytes = (VkDeviceSize) n_elems * sizeof(float);
    const VkDeviceSize payload_bytes = (VkDeviceSize) c.n_ranks * stride;
    const VkDeviceSize flags_bytes = tp5_flags_region_bytes(c.n_ranks, c.sync_mode);
    const VkDeviceSize mailbox_bytes = payload_bytes + flags_bytes;
    const uint32_t flags_base_u32 = (uint32_t) (flags_base / sizeof(uint32_t));
    const bool gpuflag = (c.sync_mode == tp5_sync_mode::GPUFLAG);
    const tp5_flag_pc flag_pc_base{0u, flags_base_u32, 0u};

    if (gpuflag) {
        plan.ds_flag.resize(c.n_ranks, VK_NULL_HANDLE);
    }

    // Allocate per-plan command buffers and descriptor sets
    for (size_t i = 0; i < c.n_ranks; ++i) {
        tp5_rank & r = c.ranks[i];
        VkCommandBufferAllocateInfo cba{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr,
                                        r.cmd_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        if (vkAllocateCommandBuffers(r.vkdev, &cba, &plan.cmd_p1[i]) != VK_SUCCESS ||
            vkAllocateCommandBuffers(r.vkdev, &cba, &plan.cmd_p2[i]) != VK_SUCCESS) {
            c.fail("allocation of command buffer failed on rank " + std::to_string(i));
            return false;
        }

        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                       r.desc_pool, 1, &r.dsl};
        if (vkAllocateDescriptorSets(r.vkdev, &ai, &plan.ds_sum[i]) != VK_SUCCESS) {
            c.fail("allocation of sum descriptor set failed on rank " + std::to_string(i));
            return false;
        }
        if (c.wire == tp5_wire_type::F16) {
            if (vkAllocateDescriptorSets(r.vkdev, &ai, &plan.ds_pack[i]) != VK_SUCCESS) {
                c.fail("allocation of pack descriptor set failed on rank " + std::to_string(i));
                return false;
            }
        }
        if (gpuflag) {
            VkDescriptorSetAllocateInfo fai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                            r.desc_pool, 1, &r.flag_dsl};
            if (vkAllocateDescriptorSets(r.vkdev, &fai, &plan.ds_flag[i]) != VK_SUCCESS) {
                c.fail("allocation of flag descriptor set failed on rank " + std::to_string(i));
                return false;
            }
            tp5_update_flag_descriptor(r, i, plan.ds_flag[i], mailbox_bytes);
        }
    }

    // Record Phase 1 command buffers (SIMULTANEOUS_USE allows asynchronous resubmission across pipelined calls)
    VkCommandBufferUsageFlags cb_flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, cb_flags, nullptr};
    for (size_t i = 0; i < c.n_ranks; ++i) {
        tp5_rank & r = c.ranks[i];
        if (vkBeginCommandBuffer(plan.cmd_p1[i], &beg) != VK_SUCCESS) {
            c.fail("begin cmd_p1 failed on rank " + std::to_string(i));
            return false;
        }

        // Global GPU hardware pipeline barrier: ensure preceding COMPUTE shaders have fully flushed to VRAM
        VkMemoryBarrier mb_pre{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(plan.cmd_p1[i],
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0, 1, &mb_pre, 0, nullptr, 0, nullptr);

        if (gpuflag) {
            tp5_flag_pc wait_pc = flag_pc_base;
            wait_pc.mode = 0u;
            tp5_record_flag_dispatch(plan.cmd_p1[i], r, plan.ds_flag[i], wait_pc);
            VkMemoryBarrier mb_flag_wait{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(plan.cmd_p1[i], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb_flag_wait, 0, nullptr, 0, nullptr);
        }

        // 1. tensor -> canonical wire copy
        if (c.wire == tp5_wire_type::F32) {
            VkBufferCopy cp{trefs[i].offset, 0, tensor_bytes};
            vkCmdCopyBuffer(plan.cmd_p1[i], trefs[i].buf, r.wire_buf, 1, &cp);
        } else {
            tp5_update_pack_descriptor(r, plan.ds_pack[i], trefs[i].buf, trefs[i].offset, tensor_bytes);
            vkCmdBindPipeline(plan.cmd_p1[i], VK_PIPELINE_BIND_POINT_COMPUTE, r.pack_pipe);
            vkCmdBindDescriptorSets(plan.cmd_p1[i], VK_PIPELINE_BIND_POINT_COMPUTE, r.pipe_layout, 0, 1, &plan.ds_pack[i], 0, nullptr);
            uint32_t n = (uint32_t) n_elems;
            vkCmdPushConstants(plan.cmd_p1[i], r.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n);
            vkCmdDispatch(plan.cmd_p1[i], (n + 255) / 256, 1, 1);
        }

        VkMemoryBarrier mb_wire{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
            (c.wire == tp5_wire_type::F32 ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_SHADER_WRITE_BIT),
            VK_ACCESS_TRANSFER_READ_BIT};
        vkCmdPipelineBarrier(plan.cmd_p1[i],
            (c.wire == tp5_wire_type::F32 ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb_wire, 0, nullptr, 0, nullptr);

        // 2. Local PUSH: wire_buf -> own mailbox slot i (offset i * stride)
        VkBufferCopy local_cp{0, (VkDeviceSize) i * stride, payload};
        vkCmdCopyBuffer(plan.cmd_p1[i], r.wire_buf, r.mailbox_buf, 1, &local_cp);

        // 3. Peer PUSH: wire_buf -> peer's mailbox slot i (offset i * stride)
        for (size_t p = 0; p < r.imports.size(); ++p) {
            VkBufferCopy peer_cp{0, (VkDeviceSize) i * stride, payload};
            vkCmdCopyBuffer(plan.cmd_p1[i], r.wire_buf, r.imports[p].buf, 1, &peer_cp);
        }

        VkMemoryBarrier mb_p1{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
        vkCmdPipelineBarrier(plan.cmd_p1[i], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 1, &mb_p1, 0, nullptr, 0, nullptr);

        if (gpuflag) {
            tp5_flag_pc pub_pc = flag_pc_base;
            pub_pc.mode = 1u;
            tp5_record_flag_dispatch(plan.cmd_p1[i], r, plan.ds_flag[i], pub_pc);
            VkMemoryBarrier mb_flag_pub{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT};
            vkCmdPipelineBarrier(plan.cmd_p1[i], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb_flag_pub, 0, nullptr, 0, nullptr);
        }

        if (vkEndCommandBuffer(plan.cmd_p1[i]) != VK_SUCCESS) {
            c.fail("end cmd_p1 failed on rank " + std::to_string(i));
            return false;
        }
    }

    // Record Phase 2 command buffers (timeline/host: flags unused; gpuflag: spin + consumed publish)
    for (size_t i = 0; i < c.n_ranks; ++i) {
        tp5_rank & r = c.ranks[i];
        if (vkBeginCommandBuffer(plan.cmd_p2[i], &beg) != VK_SUCCESS) {
            c.fail("begin cmd_p2 failed on rank " + std::to_string(i));
            return false;
        }

        if (!gpuflag) {
            VkMemoryBarrier mb_p2_in{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
            vkCmdPipelineBarrier(plan.cmd_p2[i], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &mb_p2_in, 0, nullptr, 0, nullptr);
        }

        tp5_update_sum_descriptor(r, plan.ds_sum[i], trefs[i].buf, trefs[i].offset, tensor_bytes,
                                  payload_bytes, flags_base, flags_bytes, gpuflag);
        vkCmdBindPipeline(plan.cmd_p2[i], VK_PIPELINE_BIND_POINT_COMPUTE, r.sum_pipe);
        vkCmdBindDescriptorSets(plan.cmd_p2[i], VK_PIPELINE_BIND_POINT_COMPUTE, r.pipe_layout, 0, 1, &plan.ds_sum[i], 0, nullptr);

        tp5_sum_pc pc{
            (uint32_t) n_elems,
            (uint32_t) c.n_ranks,
            (uint32_t) c.max_elems,
            gpuflag ? 1u : 0u,
            0u,
        };
        vkCmdPushConstants(plan.cmd_p2[i], r.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(plan.cmd_p2[i], (uint32_t)((n_elems + 255) / 256), 1, 1);

        if (gpuflag) {
            VkMemoryBarrier mb_sum{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                VK_ACCESS_SHADER_WRITE_BIT};
            vkCmdPipelineBarrier(plan.cmd_p2[i], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb_sum, 0, nullptr, 0, nullptr);
            tp5_flag_pc done_pc = flag_pc_base;
            done_pc.mode = 2u;
            tp5_record_flag_dispatch(plan.cmd_p2[i], r, plan.ds_flag[i], done_pc);
        }

        VkMemoryBarrier mb_post{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT};
        vkCmdPipelineBarrier(plan.cmd_p2[i],
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0, 1, &mb_post, 0, nullptr, 0, nullptr);

        if (vkEndCommandBuffer(plan.cmd_p2[i]) != VK_SUCCESS) {
            c.fail("end cmd_p2 failed on rank " + std::to_string(i));
            return false;
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
        if (r.pfn_wait_semaphores(r.vkdev, &wi, timeout_ns) != VK_SUCCESS) {
            c.fail("drain_epoch: wait failed or timed out for epoch " + std::to_string(epoch) + " on rank " + std::to_string(i));
            return false;
        }
    }
    c.last_drained_epoch = std::max(c.last_drained_epoch, epoch);
    // Retire completed buffer owners in the ring up to the drained epoch
    for (size_t s = 0; s < tp5_comm::MAX_OUTSTANDING_EPOCHS; ++s) {
        if (c.in_flight_ring[s].epoch > 0 && c.in_flight_ring[s].epoch <= c.last_drained_epoch) {
            c.in_flight_ring[s].owners.clear();
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

static bool tp5_gpuflag_drain_ring_slot(tp5_comm & c) {
    for (size_t i = 0; i < c.n_ranks; ++i) {
        tp5_rank & r = c.ranks[i];
        if (r.fence_p2 == VK_NULL_HANDLE) continue;
        if (vkWaitForFences(r.vkdev, 1, &r.fence_p2, VK_TRUE, 5000000000ULL) != VK_SUCCESS) {
            c.fail("gpuflag ring drain failed on rank " + std::to_string(i));
            return false;
        }
        vkResetFences(r.vkdev, 1, &r.fence_p2);
    }
    return true;
}

// ===========================================================================
// Two-Stage Mesh AllReduce: Proven Hardware Synchronization
// ===========================================================================

bool tp5_allreduce_mesh(tp5_comm & c, ggml_tensor ** tensors, size_t n_elems) {
    if (c.failed) return false;
    if (n_elems == 0) return true;
    if (c.allreduce_calls >= (UINT64_MAX / 2 - 1)) {
        c.fail("epoch overflow guard triggered");
        return false;
    }

    for (size_t j = 0; j < c.n_ranks; ++j) {
        if (c.backends[j]) {
            ggml_vk_tp5_flush_async(c.backends[j]);
        }
    }

    if (n_elems > c.max_elems) {
        if (!tp5_setup_workspace(c, n_elems)) return false;
    }

    const size_t wire_b = c.wire == tp5_wire_type::F16 ? 2 : 4;
    const VkDeviceSize stride = (VkDeviceSize) c.max_elems * wire_b;
    const VkDeviceSize tensor_bytes = (VkDeviceSize) n_elems * sizeof(float);
    const VkDeviceSize flags_base = tp5_flags_byte_offset(c.n_ranks, stride);
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
    }

    tp5_plan_key key;
    key.n_elems = n_elems;
    key.wire = c.wire;
    key.stride = stride;
    key.workspace_gen = c.workspace_gen;
    key.bindings.resize(c.n_ranks);
    for (size_t j = 0; j < c.n_ranks; ++j) {
        key.bindings[j].buf = trefs[j].buf;
        key.bindings[j].offset = trefs[j].offset;
        key.bindings[j].size = trefs[j].size;
    }

    tp5_cached_plan * plan = nullptr;
    tp5_cached_plan one_shot;
    bool is_cached = false;
    bool plan_cache_hit = false;

    if (c.cmd_replay_enabled) {
        for (auto & p : c.cached_plans) {
            if (p.key == key) {
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
        new_plan.key = key;
        new_plan.cmd_p1.resize(c.n_ranks, VK_NULL_HANDLE);
        new_plan.cmd_p2.resize(c.n_ranks, VK_NULL_HANDLE);
        new_plan.ds_sum.resize(c.n_ranks, VK_NULL_HANDLE);
        if (c.wire == tp5_wire_type::F16) {
            new_plan.ds_pack.resize(c.n_ranks, VK_NULL_HANDLE);
        }
        if (c.sync_mode == tp5_sync_mode::GPUFLAG) {
            new_plan.ds_flag.resize(c.n_ranks, VK_NULL_HANDLE);
        }
        new_plan.last_used_call = c.allreduce_calls;

        if (!tp5_record_plan(c, new_plan, trefs, n_elems, flags_base)) {
            c.destroy_plan(new_plan);
            return false;
        }

        if (c.cmd_replay_enabled) {
            if (c.cached_plans.size() >= tp5_comm::MAX_CACHED_PLANS) {
                c.evict_lru_plan();
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

    // CPU backpressure: under timeline mode, bound outstanding pipelined epochs to MAX_OUTSTANDING_EPOCHS (4)
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
        slot.owners.resize(c.n_ranks);
        for (size_t j = 0; j < c.n_ranks; ++j) {
            slot.owners[j] = trefs[j].owner;
        }
    } else if (c.sync_mode == tp5_sync_mode::GPUFLAG) {
        size_t ring_idx = (size_t)((epoch - 1) % tp5_comm::MAX_OUTSTANDING_EPOCHS);
        auto & slot = c.in_flight_ring[ring_idx];
        if (slot.epoch > 0) {
            if (!tp5_gpuflag_drain_ring_slot(c)) return false;
            slot.owners.clear();
            slot.epoch = 0;
        }
        slot.epoch = epoch;
        slot.owners.resize(c.n_ranks);
        for (size_t j = 0; j < c.n_ranks; ++j) {
            slot.owners[j] = trefs[j].owner;
        }
        tp5_update_epoch_bufs(c, seq);
    }

    const auto t_bp1 = std::chrono::high_resolution_clock::now();

    // Submit Phase 1 on all ranks
    const bool p1_on_transfer = false;
    if (c.sync_mode == tp5_sync_mode::TIMELINE) {
        // Split submit (proven): Phase 1 enqueued on all ranks first; Phase 2 enqueued
        // in a second loop below. Zero host wait, zero fences between phases.
        for (size_t i = 0; i < c.n_ranks; ++i) {
            tp5_rank & r = c.ranks[i];
            VkSemaphore wait_sems[8];
            uint64_t wait_vals[8];
            VkPipelineStageFlags wait_stages[8];
            uint32_t wait_cnt = 0;
            if (epoch > 1) {
                for (size_t j = 0; j < c.n_ranks; ++j) {
                    if (j == i) continue;
                    wait_sems[wait_cnt] = r.peer_timeline_sems[j];
                    wait_vals[wait_cnt] = 2 * (epoch - 1);
                    wait_stages[wait_cnt] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                    wait_cnt++;
                }
            }
            uint64_t sig_val = 2 * epoch - 1;
            VkTimelineSemaphoreSubmitInfo tl_si{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
            tl_si.waitSemaphoreValueCount = wait_cnt;
            tl_si.pWaitSemaphoreValues = wait_vals;
            tl_si.signalSemaphoreValueCount = 1;
            tl_si.pSignalSemaphoreValues = &sig_val;

            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.pNext = &tl_si;
            si.waitSemaphoreCount = wait_cnt;
            si.pWaitSemaphores = wait_sems;
            si.pWaitDstStageMask = wait_stages;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &plan->cmd_p1[i];
            si.signalSemaphoreCount = 1;
            si.pSignalSemaphores = &r.timeline_sem;

            if (vkQueueSubmit(r.queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
                // Do NOT call vkQueueWaitIdle on prior ranks: their submitted commands may be blocked
                // waiting on missing peer timeline signals that will never arrive!
                c.fail("Phase 1 timeline submit failed on rank " + std::to_string(i));
                return false;
            }
        }
    } else if (c.sync_mode == tp5_sync_mode::GPUFLAG) {
        for (size_t i = 0; i < c.n_ranks; ++i) {
            tp5_rank & r = c.ranks[i];
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &plan->cmd_p1[i];
            if (vkQueueSubmit(r.queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
                c.fail("Phase 1 gpuflag submit failed on rank " + std::to_string(i));
                return false;
            }
        }
    } else {
        for (size_t i = 0; i < c.n_ranks; ++i) {
            tp5_rank & r = c.ranks[i];
            VkQueue q = p1_on_transfer ? r.transfer_queue : r.queue;
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &plan->cmd_p1[i];
            if (c.sync_mode == tp5_sync_mode::SYNCFD) {
                si.signalSemaphoreCount = 1;
                si.pSignalSemaphores = &r.sem_p1_done;
            }
            VkFence f = (c.sync_mode == tp5_sync_mode::SYNCFD) ? VK_NULL_HANDLE : r.fence_p1;
            if (vkQueueSubmit(q, 1, &si, f) != VK_SUCCESS) {
                for (size_t k = 0; k < i; ++k) {
                    vkQueueWaitIdle(c.ranks[k].queue);
                }
                if (!is_cached) c.destroy_plan(one_shot);
                c.fail("Phase 1 submit failed on rank " + std::to_string(i));
                GGML_ABORT("ggml-vulkan-collective: unrecoverable runtime failure: Phase 1 submit failed on rank %zu\n", i);
            }
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    if (c.sync_mode == tp5_sync_mode::TIMELINE) {
        // Zero host wait, zero per-AR export/import, zero vkDeviceWaitIdle:
        // Phase 2 compute stages on each rank wait on peer timeline semaphores over PCIe.
    } else if (c.sync_mode == tp5_sync_mode::SYNCFD) {
        for (size_t src = 0; src < c.n_ranks; ++src) {
            tp5_rank & r_src = c.ranks[src];
            VkSemaphoreGetFdInfoKHR gfi{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
            gfi.semaphore = r_src.sem_p1_done;
            gfi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
            int sync_fd = -1;
            VkResult res = r_src.pfn_get_sem_fd(r_src.vkdev, &gfi, &sync_fd);
            if (res != VK_SUCCESS) {
                for (size_t k = 0; k < c.n_ranks; ++k) {
                    vkQueueWaitIdle(c.ranks[k].queue);
                }
                if (!is_cached) c.destroy_plan(one_shot);
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
                        if (!is_cached) c.destroy_plan(one_shot);
                        c.fail("dup sync_fd failed for rank " + std::to_string(dst));
                        GGML_ABORT("ggml-vulkan-collective: unrecoverable runtime failure: dup sync_fd failed for rank %zu\n", dst);
                    }
                }

                VkImportSemaphoreFdInfoKHR ifi{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR};
                ifi.semaphore = r_dst.wait_sems[slot_idx];
                ifi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
                ifi.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
                ifi.fd = fd_for_import;
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
                    if (!is_cached) c.destroy_plan(one_shot);
                    c.fail("ImportSemaphoreFdKHR failed on rank " + std::to_string(dst));
                    GGML_ABORT("ggml-vulkan-collective: unrecoverable runtime failure: ImportSemaphoreFdKHR failed on rank %zu\n", dst);
                }
            }
            if (sync_fd >= 0) {
                ::close(sync_fd);
            }
        }
    } else {
        if (!tp5_wait_all_p1(c)) {
            if (!is_cached) c.destroy_plan(one_shot);
            GGML_ABORT("ggml-vulkan-collective: unrecoverable runtime failure: Phase 1 fence wait failed\n");
        }
    }
    if (c.sync_mode != tp5_sync_mode::TIMELINE && c.sync_mode != tp5_sync_mode::GPUFLAG) {
    tp5_p2p_visibility_barrier(c);
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    auto t4 = std::chrono::high_resolution_clock::now();

    // Submit Phase 2 on all ranks
    if (c.sync_mode == tp5_sync_mode::TIMELINE) {
        for (size_t i = 0; i < c.n_ranks; ++i) {
            tp5_rank & r = c.ranks[i];
            VkSemaphore wait_sems[8];
            uint64_t wait_vals[8];
            VkPipelineStageFlags wait_stages[8];
            uint32_t wait_cnt = 0;
            for (size_t j = 0; j < c.n_ranks; ++j) {
                if (j == i) continue;
                wait_sems[wait_cnt] = r.peer_timeline_sems[j];
                wait_vals[wait_cnt] = 2 * epoch - 1;
                wait_stages[wait_cnt] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                wait_cnt++;
            }
            uint64_t sig_val = 2 * epoch;
            VkTimelineSemaphoreSubmitInfo tl_si{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
            tl_si.waitSemaphoreValueCount = wait_cnt;
            tl_si.pWaitSemaphoreValues = wait_vals;
            tl_si.signalSemaphoreValueCount = 1;
            tl_si.pSignalSemaphoreValues = &sig_val;

            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.pNext = &tl_si;
            si.waitSemaphoreCount = wait_cnt;
            si.pWaitSemaphores = wait_sems;
            si.pWaitDstStageMask = wait_stages;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &plan->cmd_p2[i];
            si.signalSemaphoreCount = 1;
            si.pSignalSemaphores = &r.timeline_sem;

            if (vkQueueSubmit(r.queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
                // Do NOT call vkQueueWaitIdle on prior ranks: avoid deadlock on missing signals
                c.fail("Phase 2 timeline submit failed on rank " + std::to_string(i));
                return false;
            }
        }
    } else if (c.sync_mode == tp5_sync_mode::GPUFLAG) {
        for (size_t i = 0; i < c.n_ranks; ++i) {
            tp5_rank & r = c.ranks[i];
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &plan->cmd_p2[i];
            vkResetFences(r.vkdev, 1, &r.fence_p2);
            if (vkQueueSubmit(r.queue, 1, &si, r.fence_p2) != VK_SUCCESS) {
                c.fail("Phase 2 gpuflag submit failed on rank " + std::to_string(i));
                return false;
            }
        }
        if (!tp5_wait_all_p2(c)) {
            if (!is_cached) c.destroy_plan(one_shot);
            return false;
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
        si.pCommandBuffers = &plan->cmd_p2[i];
        vkResetFences(r.vkdev, 1, &r.fence_p2);
        if (vkQueueSubmit(r.queue, 1, &si, r.fence_p2) != VK_SUCCESS) {
            for (size_t k = 0; k < i; ++k) {
                vkQueueWaitIdle(c.ranks[k].queue);
            }
            if (!is_cached) c.destroy_plan(one_shot);
            c.fail("Phase 2 submit failed on rank " + std::to_string(i));
            GGML_ABORT("ggml-vulkan-collective: unrecoverable runtime failure: Phase 2 submit failed on rank %zu\n", i);
        }
    }

    if (!tp5_wait_all_p2(c)) {
        if (!is_cached) c.destroy_plan(one_shot);
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
    c->backends.assign(backends, backends + n);

    const char * relay_env = getenv("GGML_TP5_RELAY");
    if (relay_env && strcmp(relay_env, "off") != 0) {
        fprintf(stderr, "ggml-vulkan-collective: host relay is prohibited (GGML_TP5_RELAY='%s'); direct GPU->GPU peer VRAM mesh required (only 'off' or unset allowed)\n", relay_env);
        delete c;
        return nullptr;
    }

    const char * wire_env = getenv("GGML_TP5_WIRE");
    c->wire = (wire_env && strcmp(wire_env, "f32") == 0) ? tp5_wire_type::F32 : tp5_wire_type::F16;

    const char * sync_env = getenv("GGML_TP5_SYNC");
    if (sync_env && strcmp(sync_env, "timeline") == 0) {
        c->sync_mode = tp5_sync_mode::TIMELINE;
    } else if (sync_env && strcmp(sync_env, "syncfd") == 0) {
        c->sync_mode = tp5_sync_mode::SYNCFD;
    } else if (sync_env && (strcmp(sync_env, "gpu") == 0 || strcmp(sync_env, "gpuflag") == 0)) {
        c->sync_mode = tp5_sync_mode::GPUFLAG;
        c->spin_max = tp5_default_spin_max();
        fprintf(stderr, "ggml-vulkan-collective: WARNING: sync mode '%s' is experimental GPU-flag spin; "
                        "not validated for production (timeline remains default fast path)\n", sync_env);
    } else {
        c->sync_mode = tp5_sync_mode::HOST;
    }

    const char * replay_env = getenv("GGML_TP5_CMD_REPLAY");
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

        if (c->sync_mode == tp5_sync_mode::TIMELINE) {
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

        if ((c->sync_mode == tp5_sync_mode::SYNCFD || c->sync_mode == tp5_sync_mode::TIMELINE) && (!r.pfn_get_sem_fd || !r.pfn_import_sem_fd)) {
            fprintf(stderr, "ggml-vulkan-collective: rank %zu missing vkGetSemaphoreFdKHR or vkImportSemaphoreFdKHR proc addr\n", i);
            for (auto & rr : c->ranks) tp5_destroy_rank(rr);
            delete c;
            return nullptr;
        }
        if (c->sync_mode == tp5_sync_mode::TIMELINE && (!r.pfn_wait_semaphores || !r.pfn_get_sem_counter)) {
            fprintf(stderr, "ggml-vulkan-collective: rank %zu missing vkWaitSemaphores or vkGetSemaphoreCounterValue proc addr\n", i);
            for (auto & rr : c->ranks) tp5_destroy_rank(rr);
            delete c;
            return nullptr;
        }

        if (c->sync_mode == tp5_sync_mode::TIMELINE) {
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
    if (c->sync_mode == tp5_sync_mode::TIMELINE) {
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
        // All imports completed: close exporter FDs
        for (size_t j = 0; j < n; ++j) {
            if (c->ranks[j].timeline_export_fd >= 0) {
                ::close(c->ranks[j].timeline_export_fd);
                c->ranks[j].timeline_export_fd = -1;
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

    if (!tp5_setup_workspace(*c, 2560)) {
        fprintf(stderr, "ggml-vulkan-collective: direct mesh workspace setup failed: %s\n", c->fail_reason.c_str());
        for (auto & rr : c->ranks) tp5_destroy_rank(rr);
        delete c;
        return nullptr;
    }

    {
        const char * sync_name = (c->sync_mode == tp5_sync_mode::TIMELINE) ? "timeline" :
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
    if (c->sync_mode == tp5_sync_mode::TIMELINE) {
        if (!tp5_drain_epoch(*c, c->allreduce_calls)) {
            GGML_ABORT("ggml-vulkan-collective: unrecoverable runtime failure: comm_free drain failed for epoch %llu\n", (unsigned long long) c->allreduce_calls);
        }
    }
    c->clear_cached_plans();
    if (c->sync_mode == tp5_sync_mode::GPUFLAG) {
        tp5_gpuflag_drain_ring_slot(*c);
    } else if (c->sync_mode != tp5_sync_mode::TIMELINE) {
    for (auto & r : c->ranks) {
        if (r.vkdev != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(r.vkdev);
        }
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

    const bool ok = tp5_allreduce_mesh(*c, tensors, (size_t) ne);

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
