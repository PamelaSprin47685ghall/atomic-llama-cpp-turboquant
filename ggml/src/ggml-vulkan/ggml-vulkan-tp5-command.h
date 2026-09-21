// Definition-time Vulkan command IR for TP5.  This is NOT a secondary-CB
// executor: emit() writes ordinary commands into the caller's primary CB.
// No std::function, descriptor update, or IR walk runs on a warm token.
#pragma once

#include "ggml-vulkan-internal.h"
#include "ggml-predefined.h"
#include <vulkan/vulkan_core.h>
#include <algorithm>
#include <cstring>
#include <exception>
#include <functional>
#include <string>
#include <vector>

struct vk_tp5_command_tape {
    enum class kind { state, dispatch, barrier, copy, query };
    struct instruction {
        kind type;
        std::function<void(VkCommandBuffer)> record;
    };
    std::vector<instruction> code;
    std::vector<VkCommandBuffer> sources;
    std::vector<vk_buffer> buffer_owners;
    bool valid = true;
    std::string rejection;

    void reject(const char * why) {
        valid = false;
        if (rejection.empty()) rejection = why;
    }
    bool emit(VkCommandBuffer dst, size_t first = 0, size_t last = SIZE_MAX) const {
        if (last == SIZE_MAX) last = code.size();
        if (!valid || first > last || last > code.size()) return false;
        for (size_t i = first; i < last; ++i) code[i].record(dst);
        return true;
    }
    size_t count(kind type) const {
        return (size_t) std::count_if(code.begin(), code.end(),
            [type](const instruction & op) { return op.type == type; });
    }
};

// Pools are immutable after definition and remain pinned by the emitted
// primary program even if its source graph/CB is evicted or re-recorded.
struct vk_tp5_graph_program {
    vk_tp5_command_tape commands;
    vk_device device_owner;
    VkDevice device = VK_NULL_HANDLE;
    std::vector<VkDescriptorPool> descriptor_pools;
    std::vector<vk_buffer> buffers;
    std::vector<std::shared_ptr<void>> pipeline_owners;
    vk_tp5_hc_binding normalized;
    const ggml_tensor * normalized_tensor = nullptr;
    size_t hc_norm_end = 0;
    size_t hc_down_end = 0;
    size_t hc_up_end = 0;
    vk_tp5_hc_sum hc;
    // One immutable dispatch recipe per capacity-aware dispatch in this graph.
    // Indirect arguments themselves live in the stage's stable host-coherent
    // route arena and are rewritten before submission, never in the CB.
    size_t predefined_stage = SIZE_MAX;
    uint32_t predefined_capacity_rows = 0;
    uint32_t predefined_capacity_outputs = 0;
    std::vector<ggml_predefined_dispatch> predefined_dispatches;
    size_t predefined_classified_dispatches = 0;
    bool predefined_complete = false;
    ~vk_tp5_graph_program() {
        for (VkDescriptorPool pool : descriptor_pools)
            if (pool) vkDestroyDescriptorPool(device, pool, nullptr);
    }
    vk_tp5_graph_program() = default;
    vk_tp5_graph_program(const vk_tp5_graph_program &) = delete;
    vk_tp5_graph_program & operator=(const vk_tp5_graph_program &) = delete;
};

inline vk_tp5_command_tape *& vk_tp5_active_tape() {
    static thread_local vk_tp5_command_tape * current = nullptr;
    return current;
}
inline void vk_tp5_pin_buffer(const vk_buffer & buffer) {
    if (auto * tape = vk_tp5_active_tape()) {
        if (buffer && std::find(tape->buffer_owners.begin(), tape->buffer_owners.end(), buffer) ==
                          tape->buffer_owners.end())
            tape->buffer_owners.push_back(buffer);
    }
}
struct vk_tp5_capture_scope {
    vk_tp5_command_tape * previous = vk_tp5_active_tape();
    explicit vk_tp5_capture_scope(vk_tp5_command_tape * tape = nullptr) { vk_tp5_active_tape() = tape; }
    void activate(vk_tp5_command_tape * tape) { vk_tp5_active_tape() = tape; }
    ~vk_tp5_capture_scope() { vk_tp5_active_tape() = previous; }
    vk_tp5_capture_scope(const vk_tp5_capture_scope &) = delete;
    vk_tp5_capture_scope & operator=(const vk_tp5_capture_scope &) = delete;
};
inline void vk_tp5_register_source(VkCommandBuffer cmd) {
    if (auto * tape = vk_tp5_active_tape()) {
        if (std::find(tape->sources.begin(), tape->sources.end(), cmd) == tape->sources.end())
            tape->sources.push_back(cmd);
    }
}
inline vk_tp5_command_tape * vk_tp5_capture_for(VkCommandBuffer cmd) {
    auto * tape = vk_tp5_active_tape();
    if (tape && std::find(tape->sources.begin(), tape->sources.end(), cmd) == tape->sources.end()) {
        tape->reject("command on an unregistered/transfer CB during graph definition");
        return nullptr;
    }
    return tape;
}
template<class T> inline std::vector<T> vk_tp5_copy_array(const T * data, uint32_t count) {
    return count ? std::vector<T>(data, data + count) : std::vector<T>{};
}

inline void tp5_cmd_bind_pipeline(VkCommandBuffer cmd, VkPipelineBindPoint point, VkPipeline pipeline) {
    vkCmdBindPipeline(cmd, point, pipeline);
    if (auto * t = vk_tp5_capture_for(cmd)) t->code.push_back({vk_tp5_command_tape::kind::state,
        [=](VkCommandBuffer out) { vkCmdBindPipeline(out, point, pipeline); }});
}
inline void tp5_cmd_bind_descriptors(VkCommandBuffer cmd, VkPipelineBindPoint point, VkPipelineLayout layout,
                                    uint32_t first, uint32_t count, const VkDescriptorSet * sets,
                                    uint32_t dynamic_count, const uint32_t * dynamic_offsets) {
    vkCmdBindDescriptorSets(cmd, point, layout, first, count, sets, dynamic_count, dynamic_offsets);
    if (auto * t = vk_tp5_capture_for(cmd)) {
        auto s = vk_tp5_copy_array(sets, count);
        auto d = vk_tp5_copy_array(dynamic_offsets, dynamic_count);
        t->code.push_back({vk_tp5_command_tape::kind::state, [=](VkCommandBuffer out) {
            vkCmdBindDescriptorSets(out, point, layout, first, (uint32_t) s.size(), s.data(),
                                    (uint32_t) d.size(), d.data());
        }});
    }
}
inline void tp5_cmd_push(VkCommandBuffer cmd, VkPipelineLayout layout, VkShaderStageFlags stages,
                         uint32_t offset, uint32_t size, const void * bytes) {
    vkCmdPushConstants(cmd, layout, stages, offset, size, bytes);
    if (auto * t = vk_tp5_capture_for(cmd)) {
        auto data = vk_tp5_copy_array((const uint8_t *) bytes, size);
        t->code.push_back({vk_tp5_command_tape::kind::state, [=](VkCommandBuffer out) {
            vkCmdPushConstants(out, layout, stages, offset, size, data.data());
        }});
    }
}
inline void tp5_cmd_dispatch(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z) {
    vkCmdDispatch(cmd, x, y, z);
    if (auto * t = vk_tp5_capture_for(cmd)) t->code.push_back({vk_tp5_command_tape::kind::dispatch,
        [=](VkCommandBuffer out) { vkCmdDispatch(out, x, y, z); }});
}
inline void tp5_cmd_dispatch_indirect(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset) {
    vkCmdDispatchIndirect(cmd, buffer, offset);
    if (auto * t = vk_tp5_capture_for(cmd)) t->code.push_back({vk_tp5_command_tape::kind::dispatch,
        [=](VkCommandBuffer out) { vkCmdDispatchIndirect(out, buffer, offset); }});
}
inline void tp5_cmd_barrier(VkCommandBuffer cmd, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                            VkDependencyFlags flags, uint32_t nm, const VkMemoryBarrier * memory,
                            uint32_t nb, const VkBufferMemoryBarrier * buffers,
                            uint32_t ni, const VkImageMemoryBarrier * images) {
    vkCmdPipelineBarrier(cmd, src, dst, flags, nm, memory, nb, buffers, ni, images);
    if (auto * t = vk_tp5_capture_for(cmd)) {
        auto m = vk_tp5_copy_array(memory, nm);
        auto b = vk_tp5_copy_array(buffers, nb);
        auto i = vk_tp5_copy_array(images, ni);
        for (const auto & v : m) if (v.pNext) t->reject("extended memory barrier is not copyable");
        for (const auto & v : b) if (v.pNext) t->reject("extended buffer barrier is not copyable");
        if (ni) t->reject("image ownership is outside TP5 buffer-only graph definitions");
        t->code.push_back({vk_tp5_command_tape::kind::barrier, [=](VkCommandBuffer out) {
            vkCmdPipelineBarrier(out, src, dst, flags, (uint32_t) m.size(), m.data(),
                                 (uint32_t) b.size(), b.data(), (uint32_t) i.size(), i.data());
        }});
    }
}
inline void tp5_cmd_copy(VkCommandBuffer cmd, VkBuffer src, VkBuffer dst,
                         uint32_t count, const VkBufferCopy * regions) {
    vkCmdCopyBuffer(cmd, src, dst, count, regions);
    if (auto * t = vk_tp5_capture_for(cmd)) {
        auto r = vk_tp5_copy_array(regions, count);
        t->code.push_back({vk_tp5_command_tape::kind::copy, [=](VkCommandBuffer out) {
            vkCmdCopyBuffer(out, src, dst, (uint32_t) r.size(), r.data());
        }});
    }
}
inline void tp5_cmd_fill(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, uint32_t value) {
    vkCmdFillBuffer(cmd, buffer, offset, size, value);
    if (auto * t = vk_tp5_capture_for(cmd)) t->code.push_back({vk_tp5_command_tape::kind::copy,
        [=](VkCommandBuffer out) { vkCmdFillBuffer(out, buffer, offset, size, value); }});
}
inline void tp5_cmd_update(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, const void * bytes) {
    vkCmdUpdateBuffer(cmd, buffer, offset, size, bytes);
    if (auto * t = vk_tp5_capture_for(cmd)) {
        if (size > UINT32_MAX) { t->reject("oversized inline buffer update"); return; }
        auto data = vk_tp5_copy_array((const uint8_t *) bytes, (uint32_t) size);
        t->code.push_back({vk_tp5_command_tape::kind::copy, [=](VkCommandBuffer out) {
            vkCmdUpdateBuffer(out, buffer, offset, size, data.data());
        }});
    }
}
inline void tp5_cmd_timestamp(VkCommandBuffer cmd, VkPipelineStageFlagBits stage, VkQueryPool pool, uint32_t query) {
    vkCmdWriteTimestamp(cmd, stage, pool, query);
    if (auto * t = vk_tp5_capture_for(cmd)) t->code.push_back({vk_tp5_command_tape::kind::query,
        [=](VkCommandBuffer out) { vkCmdWriteTimestamp(out, stage, pool, query); }});
}

// Hpp adapter used by the existing backend recorders. The actual commands and
// arguments are preserved; the optional definition copy owns all array data.
#ifdef VULKAN_HPP
struct vk_tp5_hpp_commands {
    VkCommandBuffer cmd;
    explicit vk_tp5_hpp_commands(vk::CommandBuffer cb) : cmd((VkCommandBuffer) cb) {}
    void pushConstants(vk::PipelineLayout l, vk::ShaderStageFlags s, uint32_t o, uint32_t n, const void * p) const {
        tp5_cmd_push(cmd, (VkPipelineLayout) l, (VkShaderStageFlags) s, o, n, p);
    }
    void bindPipeline(vk::PipelineBindPoint p, vk::Pipeline l) const {
        tp5_cmd_bind_pipeline(cmd, (VkPipelineBindPoint) p, (VkPipeline) l);
    }
    void bindDescriptorSets(vk::PipelineBindPoint p, vk::PipelineLayout l, uint32_t first,
                             vk::ArrayProxy<const vk::DescriptorSet> sets,
                             vk::ArrayProxy<const uint32_t> offsets) const {
        tp5_cmd_bind_descriptors(cmd, (VkPipelineBindPoint) p, (VkPipelineLayout) l, first,
            sets.size(), reinterpret_cast<const VkDescriptorSet *>(sets.data()), offsets.size(), offsets.data());
    }
    void dispatch(uint32_t x, uint32_t y, uint32_t z) const { tp5_cmd_dispatch(cmd, x, y, z); }
    void dispatchIndirect(vk::Buffer b, vk::DeviceSize o) const { tp5_cmd_dispatch_indirect(cmd, (VkBuffer) b, o); }
    void pipelineBarrier(vk::PipelineStageFlags s, vk::PipelineStageFlags d, vk::DependencyFlags flags,
                          vk::ArrayProxy<const vk::MemoryBarrier> m,
                          vk::ArrayProxy<const vk::BufferMemoryBarrier> b,
                          vk::ArrayProxy<const vk::ImageMemoryBarrier> i) const {
        tp5_cmd_barrier(cmd, (VkPipelineStageFlags) s, (VkPipelineStageFlags) d, (VkDependencyFlags) flags,
            m.size(), reinterpret_cast<const VkMemoryBarrier *>(m.data()),
            b.size(), reinterpret_cast<const VkBufferMemoryBarrier *>(b.data()),
            i.size(), reinterpret_cast<const VkImageMemoryBarrier *>(i.data()));
    }
    void copyBuffer(vk::Buffer s, vk::Buffer d, vk::ArrayProxy<const vk::BufferCopy> r) const {
        tp5_cmd_copy(cmd, (VkBuffer) s, (VkBuffer) d, r.size(), reinterpret_cast<const VkBufferCopy *>(r.data()));
    }
    void fillBuffer(vk::Buffer b, vk::DeviceSize o, vk::DeviceSize n, uint32_t v) const {
        tp5_cmd_fill(cmd, (VkBuffer) b, o, n, v);
    }
    void updateBuffer(vk::Buffer b, vk::DeviceSize o, vk::DeviceSize n, const void * p) const {
        tp5_cmd_update(cmd, (VkBuffer) b, o, n, p);
    }
    void writeTimestamp(vk::PipelineStageFlagBits s, vk::QueryPool p, uint32_t q) const {
        tp5_cmd_timestamp(cmd, (VkPipelineStageFlagBits) s, (VkQueryPool) p, q);
    }
    void setEvent(vk::Event event, vk::PipelineStageFlags stages) const {
        vk::CommandBuffer(cmd).setEvent(event, stages);
        if (auto * t = vk_tp5_capture_for(cmd))
            t->reject("recycled backend events cannot be retained by a TP5 definition");
    }
    void resetEvent(vk::Event event, vk::PipelineStageFlags stages) const {
        vk::CommandBuffer(cmd).resetEvent(event, stages);
        if (auto * t = vk_tp5_capture_for(cmd))
            t->reject("recycled backend events cannot be retained by a TP5 definition");
    }
    void waitEvents(vk::ArrayProxy<const vk::Event> events, vk::PipelineStageFlags s, vk::PipelineStageFlags d,
                    vk::ArrayProxy<const vk::MemoryBarrier> m,
                    vk::ArrayProxy<const vk::BufferMemoryBarrier> b,
                    vk::ArrayProxy<const vk::ImageMemoryBarrier> i) const {
        vk::CommandBuffer(cmd).waitEvents(events, s, d, m, b, i);
        if (auto * t = vk_tp5_capture_for(cmd))
            t->reject("recycled backend events cannot be retained by a TP5 definition");
    }
};
#endif
