// Execute the production TP5_LATE_PACK_WEIGHTS SPIR-V on a real AMD GPU.
// Inputs/outputs are device-local; staging is used only for initialization/readback.
// This is a byte-correctness probe, not a kernel performance benchmark.
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(VkResult result, const char * operation) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(operation) + ": VkResult=" + std::to_string(result));
}

struct Buffer {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void * mapped = nullptr;
    Buffer(VkDevice d, const VkPhysicalDeviceMemoryProperties & props, VkDeviceSize bytes,
           VkBufferUsageFlags usage, VkMemoryPropertyFlags required) : device(d) {
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        ci.size = bytes; ci.usage = usage; ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device, &ci, nullptr, &buffer), "create buffer");
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, buffer, &req);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
            if ((req.memoryTypeBits & (1u << i)) && (props.memoryTypes[i].propertyFlags & required) == required) { type = i; break; }
        if (type == UINT32_MAX) throw std::runtime_error("No suitable memory type");
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size; ai.memoryTypeIndex = type;
        check(vkAllocateMemory(device, &ai, nullptr, &memory), "allocate memory");
        check(vkBindBufferMemory(device, buffer, memory, 0), "bind memory");
        if (required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            check(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped), "map memory");
    }
    Buffer(const Buffer &) = delete;
    ~Buffer() {
        if (mapped) vkUnmapMemory(device, memory);
        if (buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
    }
};

static uint32_t pattern(uint32_t block, uint32_t word) {
    return (block * 0x9e3779b1u) ^ (word * 0x85ebca77u) ^ 0xa5c3719bu;
}

int main(int argc, char ** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: %s shader.spv blocks amd-device-index\n", argv[0]); return 2; }
    const uint32_t blocks = static_cast<uint32_t>(std::stoul(argv[2]));
    const unsigned dev_index = static_cast<unsigned>(std::stoul(argv[3]));
    const bool single = std::getenv("TP5_TEST_PACK_SINGLE_WG") && std::string(std::getenv("TP5_TEST_PACK_SINGLE_WG"))=="1";
    const uint32_t workgroups=single?1u:(blocks+255)/256;
    if (!blocks || blocks > 1048576 || dev_index > 4) return 2;
    try {
        std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error("Missing shader");
        const auto bytes = input.tellg();
        if (bytes <= 0 || bytes % 4) throw std::runtime_error("Invalid SPIR-V length");
        std::vector<uint32_t> spirv(static_cast<size_t>(bytes) / 4);
        input.seekg(0); input.read(reinterpret_cast<char *>(spirv.data()), bytes);
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "tp5-production-pack-correctness"; app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
        VkInstance instance = VK_NULL_HANDLE;
        check(vkCreateInstance(&ici, nullptr, &instance), "create instance");
        uint32_t nd = 0; check(vkEnumeratePhysicalDevices(instance, &nd, nullptr), "count devices");
        std::vector<VkPhysicalDevice> devices(nd);
        check(vkEnumeratePhysicalDevices(instance, &nd, devices.data()), "enumerate devices");
        std::vector<VkPhysicalDevice> amd;
        for (auto device : devices) {
            VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(device, &p);
            if (p.vendorID == 0x1002 && p.deviceID == 0x73bf) amd.push_back(device);
        }
        if (amd.size() != 5 || dev_index >= amd.size()) throw std::runtime_error("Need five physical Navi21 devices");
        const auto physical = amd[dev_index];
        VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(physical, &props);
        std::fprintf(stderr, "device=%u name=%s blocks=%u workgroups=%u\n", dev_index, props.deviceName, blocks, workgroups);
        uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(physical, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &nq, qprops.data());
        uint32_t qf = UINT32_MAX;
        for (uint32_t i = 0; i < nq; ++i) if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
        if (qf == UINT32_MAX) throw std::runtime_error("No compute queue");
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES}; f11.pNext = &f12;
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f2.pNext = &f11;
        vkGetPhysicalDeviceFeatures2(physical, &f2);
        if (!f2.features.shaderInt16 || !f11.storageBuffer16BitAccess || !f12.shaderFloat16) throw std::runtime_error("Required shader features missing");
        VkPhysicalDeviceVulkan12Features enable12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        enable12.shaderFloat16 = VK_TRUE; enable12.shaderInt8 = f12.shaderInt8;
        VkPhysicalDeviceVulkan11Features enable11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        enable11.storageBuffer16BitAccess = VK_TRUE; enable11.pNext = &enable12;
        VkPhysicalDeviceFeatures enable{}; enable.shaderInt16 = VK_TRUE;
        const float priority = 1;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = qf; qci.queueCount = 1; qci.pQueuePriorities = &priority;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &enable11; dci.pEnabledFeatures = &enable; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
        VkDevice device = VK_NULL_HANDLE; check(vkCreateDevice(physical, &dci, nullptr, &device), "create device");
        VkQueue queue; vkGetDeviceQueue(device, qf, 0, &queue);
        VkPhysicalDeviceMemoryProperties mp{}; vkGetPhysicalDeviceMemoryProperties(physical, &mp);
        bool failed = false;
        {
            const VkDeviceSize source_bytes = (VkDeviceSize(blocks) * 34 + 3u) & ~VkDeviceSize(3u);
            const VkDeviceSize output_bytes = VkDeviceSize(blocks + 1) * 36;
            Buffer upload(device, mp, source_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            Buffer src(device, mp, source_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            Buffer dst(device, mp, output_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            Buffer readback(device, mp, output_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            auto * raw = static_cast<uint8_t *>(upload.mapped);
            std::memset(raw, 0, static_cast<size_t>(source_bytes));
            for (uint32_t b = 0; b < blocks; ++b) {
                const uint16_t scale = 0x3c00u; // Exactly 1.0, represented as binary16.
                std::memcpy(raw + 34ull * b, &scale, sizeof(scale));
                for (uint32_t w = 0; w < 8; ++w) {
                    const uint32_t value = pattern(b, w);
                    std::memcpy(raw + 34ull * b + 2 + 4 * w, &value, sizeof(value));
                }
            }
            VkDescriptorSetLayoutBinding binding[2]{};
            for (uint32_t b = 0; b < 2; ++b) { binding[b].binding = b; binding[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; binding[b].descriptorCount = 1; binding[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
            VkDescriptorSetLayoutCreateInfo slci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; slci.bindingCount = 2; slci.pBindings = binding;
            VkDescriptorSetLayout sl; check(vkCreateDescriptorSetLayout(device, &slci, nullptr, &sl), "descriptor layout");
            VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
            VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            plci.setLayoutCount = 1; plci.pSetLayouts = &sl; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &range;
            VkPipelineLayout pl; check(vkCreatePipelineLayout(device, &plci, nullptr, &pl), "pipeline layout");
            VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize = spirv.size() * 4; smci.pCode = spirv.data();
            VkShaderModule module; check(vkCreateShaderModule(device, &smci, nullptr, &module), "shader module");
            VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            pci.stage.module = module; pci.stage.pName = "main"; pci.layout = pl;
            VkPipeline pipeline; check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline), "pipeline");
            VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
            VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &pool_size;
            VkDescriptorPool dp; check(vkCreateDescriptorPool(device, &dpci, nullptr, &dp), "descriptor pool");
            VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dai.descriptorPool = dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &sl;
            VkDescriptorSet ds; check(vkAllocateDescriptorSets(device, &dai, &ds), "descriptor allocation");
            VkDescriptorBufferInfo db[2]{{src.buffer, 0, source_bytes}, {dst.buffer, 0, output_bytes}};
            VkWriteDescriptorSet writes[2]{};
            for (uint32_t b = 0; b < 2; ++b) { writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[b].dstSet = ds; writes[b].dstBinding = b; writes[b].descriptorCount = 1; writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[b].pBufferInfo = &db[b]; }
            vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
            VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci.queueFamilyIndex = qf; cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            VkCommandPool cp; check(vkCreateCommandPool(device, &cpci, nullptr, &cp), "command pool");
            VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cai.commandPool = cp; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
            VkCommandBuffer cmd; check(vkAllocateCommandBuffers(device, &cai, &cmd), "command buffer");
            VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; check(vkCreateFence(device, &fci, nullptr, &fence), "fence");
            for (int trial = 0; trial < 6; ++trial) {
                check(vkResetCommandBuffer(cmd, 0), "reset CB");
                VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; check(vkBeginCommandBuffer(cmd, &bi), "begin CB");
                const bool reset_payload = trial % 2 == 0;
                if (reset_payload) {
                    VkBufferCopy copy{0, 0, source_bytes}; vkCmdCopyBuffer(cmd, upload.buffer, src.buffer, 1, &copy);
                    vkCmdFillBuffer(cmd, dst.buffer, 0, output_bytes, 0);
                }
                VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                before.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
                vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &blocks);
                vkCmdDispatch(cmd, workgroups, 1, 1);
                VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; after.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
                VkBufferCopy copy{0, 0, output_bytes}; vkCmdCopyBuffer(cmd, dst.buffer, readback.buffer, 1, &copy);
                after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
                check(vkEndCommandBuffer(cmd), "end CB");
                check(vkResetFences(device, 1, &fence), "reset fence");
                VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
                check(vkQueueSubmit(queue, 1, &si, fence), "submit");
                check(vkWaitForFences(device, 1, &fence, VK_TRUE, 10000000000ull), "wait bounded fence");
                const auto * out = static_cast<const uint32_t *>(readback.mapped);
                uint32_t bad = 0, first_bad = UINT32_MAX;
                for (uint32_t b = 0; b < blocks; ++b) {
                    bool same = out[9ull * (b + 1)] == 0x3f800000u;
                    for (uint32_t w = 0; w < 8; ++w) same &= out[9ull * (b + 1) + 1 + w] == pattern(b, w);
                    if (!same) { ++bad; first_bad = std::min(first_bad, b); }
                }
                std::printf("{\"device\":%u,\"blocks\":%u,\"trial\":%d,\"reset_payload\":%s,\"done\":%u,\"bad_blocks\":%u,\"first_bad\":%u}\n", dev_index, blocks, trial, reset_payload ? "true" : "false", out[0], bad, first_bad);
                std::fflush(stdout);
                failed |= bad != 0 || out[0] != 1;
            }
            vkDestroyFence(device, fence, nullptr); vkDestroyCommandPool(device, cp, nullptr);
            vkDestroyDescriptorPool(device, dp, nullptr); vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyShaderModule(device, module, nullptr); vkDestroyPipelineLayout(device, pl, nullptr); vkDestroyDescriptorSetLayout(device, sl, nullptr);
        }
        vkDestroyDevice(device, nullptr); vkDestroyInstance(instance, nullptr);
        return failed ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "PACK PROBE FAILED: %s\n", e.what());
        return 2;
    }
}
