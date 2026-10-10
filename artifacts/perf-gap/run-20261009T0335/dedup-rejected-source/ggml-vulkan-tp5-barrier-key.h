#pragma once
#include <vulkan/vulkan_core.h>
#include <vector>

// Only literally identical buffer/global memory dependencies are candidates.
// Image transitions, extension chains and ownership transfers are excluded.
struct vk_tp5_barrier_key {
    VkPipelineStageFlags src = 0, dst = 0;
    VkDependencyFlags flags = 0;
    std::vector<VkMemoryBarrier> memory;
    std::vector<VkBufferMemoryBarrier> buffers;
    bool eligible() const {
        for (const auto & x : memory) if (x.pNext) return false;
        for (const auto & x : buffers)
            if (x.pNext || x.srcQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED ||
                x.dstQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED) return false;
        return true;
    }
    bool operator==(const vk_tp5_barrier_key & b) const {
        if (!eligible() || !b.eligible() || src != b.src || dst != b.dst || flags != b.flags ||
                memory.size() != b.memory.size() || buffers.size() != b.buffers.size()) return false;
        for (size_t i = 0; i < memory.size(); ++i)
            if (memory[i].srcAccessMask != b.memory[i].srcAccessMask || memory[i].dstAccessMask != b.memory[i].dstAccessMask) return false;
        for (size_t i = 0; i < buffers.size(); ++i) {
            const auto & x = buffers[i]; const auto & y = b.buffers[i];
            if (x.srcAccessMask != y.srcAccessMask || x.dstAccessMask != y.dstAccessMask ||
                x.buffer != y.buffer || x.offset != y.offset || x.size != y.size) return false;
        }
        return true;
    }
};
