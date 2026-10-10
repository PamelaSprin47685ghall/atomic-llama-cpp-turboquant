#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(x) do { VkResult rc = (x); if (rc != VK_SUCCESS) throw std::runtime_error(std::string(#x) + " failed: " + std::to_string(rc)); } while (0)

static constexpr uint32_t groups = 4096;
static constexpr uint32_t lanes = 256;
static constexpr uint32_t rounds = 4096;
static constexpr double flops_per_dispatch = double(groups) * lanes * rounds * 32 * 2;
static const std::array<const char *, 5> bdfs = {
    "0000:0f:00.0", "0000:09:00.0", "0000:0c:00.0", "0000:06:00.0", "0000:03:00.0",
};

static void measure(VkPhysicalDevice physical, const std::string & bdf, const std::vector<uint32_t> & spirv) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical, &properties);
    uint32_t n_families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &n_families, nullptr);
    std::vector<VkQueueFamilyProperties> families(n_families);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &n_families, families.data());
    uint32_t family = n_families;
    for (uint32_t i = 0; i < n_families; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && families[i].timestampValidBits > 0) {
            family = i;
            break;
        }
    }
    if (family == n_families) throw std::runtime_error("no timestamp-capable compute queue on " + bdf);

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    VkDevice device;
    CHECK(vkCreateDevice(physical, &device_info, nullptr, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, family, 0, &queue);

    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = VkDeviceSize(groups) * lanes * sizeof(float);
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer;
    CHECK(vkCreateBuffer(device, &buffer_info, nullptr, &buffer));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, buffer, &req);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    uint32_t type = memory.memoryTypeCount;
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            type = i;
            break;
        }
    }
    if (type == memory.memoryTypeCount) throw std::runtime_error("no device-local memory on " + bdf);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = type;
    VkDeviceMemory device_memory;
    CHECK(vkAllocateMemory(device, &alloc, nullptr, &device_memory));
    CHECK(vkBindBufferMemory(device, buffer, device_memory, 0));

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = 1;
    layout_info.pBindings = &binding;
    VkDescriptorSetLayout layout;
    CHECK(vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &layout));
    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = 8;
    VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push;
    VkPipelineLayout pipeline_layout;
    CHECK(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout));
    VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize = spirv.size() * sizeof(uint32_t);
    shader_info.pCode = spirv.data();
    VkShaderModule shader;
    CHECK(vkCreateShaderModule(device, &shader_info, nullptr, &shader));
    VkComputePipelineCreateInfo pipe_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipe_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipe_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipe_info.stage.module = shader;
    pipe_info.stage.pName = "main";
    pipe_info.layout = pipeline_layout;
    VkPipeline pipeline;
    CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipe_info, nullptr, &pipeline));

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VkDescriptorPool pool;
    CHECK(vkCreateDescriptorPool(device, &pool_info, nullptr, &pool));
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &layout;
    VkDescriptorSet set;
    CHECK(vkAllocateDescriptorSets(device, &set_info, &set));
    VkDescriptorBufferInfo buffer_binding{buffer, 0, buffer_info.size};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buffer_binding;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    VkQueryPoolCreateInfo query_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query_info.queryCount = 2;
    VkQueryPool query;
    CHECK(vkCreateQueryPool(device, &query_info, nullptr, &query));
    VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool_info.queueFamilyIndex = family;
    VkCommandPool command_pool;
    CHECK(vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool));
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    VkCommandBuffer command;
    CHECK(vkAllocateCommandBuffers(device, &command_info, &command));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK(vkBeginCommandBuffer(command, &begin));
    vkCmdResetQueryPool(command, query, 0, 2);
    vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query, 0);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, nullptr);
    struct { uint32_t n; float seed; } control{rounds, 0.25f};
    vkCmdPushConstants(command, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(control), &control);
    vkCmdDispatch(command, groups, 1, 1);
    vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query, 1);
    CHECK(vkEndCommandBuffer(command));
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CHECK(vkCreateFence(device, &fence_info, nullptr, &fence));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;

    double measured_ns = 0.0;
    int runs = 0;
    const auto started = std::chrono::steady_clock::now();
    do {
        CHECK(vkResetFences(device, 1, &fence));
        CHECK(vkQueueSubmit(queue, 1, &submit, fence));
        CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, 5000000000ull));
        uint64_t stamps[2]{};
        CHECK(vkGetQueryPoolResults(device, query, 0, 2, sizeof(stamps), stamps,
                                    sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        if (stamps[1] <= stamps[0]) throw std::runtime_error("GPU timestamp did not advance");
        measured_ns += (stamps[1] - stamps[0]) * properties.limits.timestampPeriod;
        ++runs;
    } while (measured_ns < 1.0e9 && runs < 512);
    const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    const double tflops = runs * flops_per_dispatch / measured_ns / 1000.0;
    std::printf("{\"bdf\":\"%s\",\"runs\":%d,\"gpu_ms\":%.3f,\"wall_ms\":%.3f,\"fp32_tflops\":%.3f}\n",
                bdf.c_str(), runs, measured_ns / 1.0e6, wall_ms, tflops);
    std::fflush(stdout);

    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyQueryPool(device, query, nullptr);
    vkDestroyDescriptorPool(device, pool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyShaderModule(device, shader, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(device, layout, nullptr);
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, device_memory, nullptr);
    vkDestroyDevice(device, nullptr);
}

int main(int argc, char ** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: tp5-alu1s shader.spv");
        std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("cannot open SPIR-V shader");
        const auto bytes = file.tellg();
        if (bytes <= 0 || bytes % 4 != 0) throw std::runtime_error("invalid SPIR-V size");
        std::vector<uint32_t> spirv(size_t(bytes) / 4);
        file.seekg(0);
        file.read(reinterpret_cast<char *>(spirv.data()), bytes);
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        info.pApplicationInfo = &app;
        VkInstance instance;
        CHECK(vkCreateInstance(&info, nullptr, &instance));
        uint32_t count = 0;
        CHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
        std::vector<VkPhysicalDevice> devices(count);
        CHECK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));
        std::array<VkPhysicalDevice, 5> mapped{};
        for (VkPhysicalDevice physical : devices) {
            uint32_t n_extensions = 0;
            CHECK(vkEnumerateDeviceExtensionProperties(physical, nullptr, &n_extensions, nullptr));
            std::vector<VkExtensionProperties> extensions(n_extensions);
            CHECK(vkEnumerateDeviceExtensionProperties(physical, nullptr, &n_extensions, extensions.data()));
            if (std::none_of(extensions.begin(), extensions.end(), [](const auto & ext) {
                return std::string(ext.extensionName) == VK_EXT_PCI_BUS_INFO_EXTENSION_NAME;
            })) continue;
            VkPhysicalDevicePCIBusInfoPropertiesEXT pci{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
            VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            properties.pNext = &pci;
            vkGetPhysicalDeviceProperties2(physical, &properties);
            char bdf[20];
            std::snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.%u", pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
            for (size_t i = 0; i < bdfs.size(); ++i) if (bdfs[i] == std::string(bdf)) mapped[i] = physical;
        }
        for (size_t i = 0; i < bdfs.size(); ++i) {
            if (mapped[i] == VK_NULL_HANDLE) throw std::runtime_error(std::string("missing PCI-mapped Vulkan GPU ") + bdfs[i]);
        }
        for (size_t i = 0; i < bdfs.size(); ++i) measure(mapped[i], bdfs[i], spirv);
        vkDestroyInstance(instance, nullptr);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "ALU benchmark failed: %s\n", e.what());
        return 1;
    }
}
