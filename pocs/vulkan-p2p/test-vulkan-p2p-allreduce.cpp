// test-vulkan-p2p-allreduce.cpp
//
// Standalone hardware verification benchmark for Vulkan 5-GPU P2P collective operations:
//   1. Physical full-mesh DMA-BUF import/export between 5 discrete AMD GPUs (Navi 21 / RX 6800).
//   2. Two-stage AllReduce (Phase 1: ReduceScatter PUSH -> Barrier -> Local compute sum -> Barrier -> Phase 2: AllGather PUSH).
//   3. Mathematical correctness test: Rank i inputs float(i + 1) across all elements.
//      Verification: Every single element across all 5 GPUs must strictly evaluate to 15.0f (0 mismatch).
//   4. Multi-round latency benchmarking with full timing (µs/AllReduce and total ms per token pass).
//   5. Accurate accounting:
//      Total vector size S = 40 KiB (10,240 FP32 elements)
//      Slice per rank S/5  = 8 KiB (2,048 FP32 elements)
//      Phase 1 sent: 4 * 8 KiB = 32 KiB per rank
//      Phase 2 sent: 4 * 8 KiB = 32 KiB per rank
//      Total sent per rank = 64 KiB (= 1.6 * S)

#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <fstream>
#include <unistd.h>
#include <iomanip>
#include <cmath>

#define VK_CHECK(res) do { \
    VkResult _r = (res); \
    if (_r != VK_SUCCESS) { \
        std::cerr << "VK Error: " << _r << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

const uint32_t NUM_GPUS = 5;
const uint32_t NUM_ELEMENTS = 10240; // 40 KiB FP32
const VkDeviceSize TOTAL_BYTES = NUM_ELEMENTS * sizeof(float); // 40,960 B (40 KiB)
const uint32_t CHUNK_ELEMENTS = NUM_ELEMENTS / NUM_GPUS; // 2,048 floats per rank
const VkDeviceSize CHUNK_BYTES = CHUNK_ELEMENTS * sizeof(float); // 8,192 B (8 KiB)

struct GPUContext {
    VkPhysicalDevice physDev;
    VkDevice dev;
    VkQueue queue;
    VkCommandPool cmdPool;
    VkFence fence;
    
    // Send buffer: owns local 40 KiB input
    VkBuffer sendBuf;
    VkDeviceMemory sendMem;
    int sendFd = -1;

    // Recv buffer: holds 5 slots of 8 KiB (total 40 KiB)
    VkBuffer recvBuf;
    VkDeviceMemory recvMem;
    int recvFd = -1;

    // Final buffer: holds 5 chunks of 8 KiB (total 40 KiB)
    VkBuffer finalBuf;
    VkDeviceMemory finalMem;
    int finalFd = -1;

    // Imported buffers on GPU i:
    std::vector<VkBuffer> peerRecvBufs;
    std::vector<VkDeviceMemory> peerRecvMems;

    std::vector<VkBuffer> peerFinalBufs;
    std::vector<VkDeviceMemory> peerFinalMems;

    VkCommandBuffer cmdPhase1; // ReduceScatter PUSH
    VkCommandBuffer cmdPhase2; // Compute Sum + AllGather PUSH

    // Compute reduction pipeline
    VkDescriptorPool descPool;
    VkDescriptorSetLayout dsLayout;
    VkDescriptorSet descSet;
    VkPipelineLayout plLayout;
    VkPipeline pipeline;
};

static std::vector<uint32_t> compileOrLoadSPV(const std::string& glslSource, const std::string& spvPath) {
    std::string compFile = spvPath + ".comp";
    std::ofstream src(compFile);
    src << glslSource;
    src.close();
    std::string cmd = "glslc -O " + compFile + " -o " + spvPath;
    if (system(cmd.c_str()) != 0) {
        std::cerr << "Failed to compile shader with glslc!" << std::endl;
        exit(1);
    }

    std::ifstream file(spvPath, std::ios::ate | std::ios::binary);
    size_t size = file.tellg();
    std::vector<uint32_t> spv(size / 4);
    file.seekg(0);
    file.read((char*)spv.data(), size);
    file.close();
    return spv;
}

int main() {
    VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "VulkanP2PAllReduce", 1, "Engine", 1, VK_API_VERSION_1_2 };
    std::vector<const char*> instanceExtensions = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME
    };
    VkInstanceCreateInfo instanceInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    instanceInfo.pApplicationInfo = &appInfo;
    instanceInfo.enabledExtensionCount = instanceExtensions.size();
    instanceInfo.ppEnabledExtensionNames = instanceExtensions.data();
    VkInstance instance;
    VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &instance));

    uint32_t gpuCount = 0;
    vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr);
    std::vector<VkPhysicalDevice> gpus(gpuCount);
    vkEnumeratePhysicalDevices(instance, &gpuCount, gpus.data());

    if (gpuCount < NUM_GPUS) {
        std::cerr << "Error: need at least " << NUM_GPUS << " GPUs, found " << gpuCount << std::endl;
        return 1;
    }

    std::vector<const char*> deviceExtensions = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME
    };

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, 0, 1, &queuePriority };
    VkDeviceCreateInfo devInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &queueInfo, 0, nullptr, (uint32_t)deviceExtensions.size(), deviceExtensions.data(), nullptr };

    std::vector<GPUContext> ctx(NUM_GPUS);
    for (uint32_t i = 0; i < NUM_GPUS; ++i) {
        ctx[i].physDev = gpus[i];
        VK_CHECK(vkCreateDevice(ctx[i].physDev, &devInfo, nullptr, &ctx[i].dev));
        vkGetDeviceQueue(ctx[i].dev, 0, 0, &ctx[i].queue);
        
        VkCommandPoolCreateInfo cpInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, 0 };
        VK_CHECK(vkCreateCommandPool(ctx[i].dev, &cpInfo, nullptr, &ctx[i].cmdPool));
        
        VkCommandBufferAllocateInfo cbAlloc = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, ctx[i].cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
        VK_CHECK(vkAllocateCommandBuffers(ctx[i].dev, &cbAlloc, &ctx[i].cmdPhase1));
        VK_CHECK(vkAllocateCommandBuffers(ctx[i].dev, &cbAlloc, &ctx[i].cmdPhase2));
        
        VkFenceCreateInfo fInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VK_CHECK(vkCreateFence(ctx[i].dev, &fInfo, nullptr, &ctx[i].fence));
    }

    auto allocBuf = [](VkDevice dev, VkPhysicalDevice physDev, VkDeviceSize size, bool exportable, VkBuffer& buf, VkDeviceMemory& mem, int* exportFd, bool hostVis) {
        VkExternalMemoryBufferCreateInfo extBufInfo = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, nullptr, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
        VkBufferCreateInfo bufInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, exportable ? &extBufInfo : nullptr, 0, size, 
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE };
        VK_CHECK(vkCreateBuffer(dev, &bufInfo, nullptr, &buf));

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(dev, buf, &memReq);
        VkExportMemoryAllocateInfo expAlloc = { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, nullptr, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
        VkMemoryAllocateInfo alloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, exportable ? &expAlloc : nullptr, memReq.size, 0 };
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);
        
        VkMemoryPropertyFlags desiredFlags = hostVis ? 
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) : 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

        for (uint32_t m = 0; m < memProps.memoryTypeCount; ++m) {
            if ((memReq.memoryTypeBits & (1 << m)) && (memProps.memoryTypes[m].propertyFlags & desiredFlags) == desiredFlags) {
                alloc.memoryTypeIndex = m; break;
            }
        }
        VK_CHECK(vkAllocateMemory(dev, &alloc, nullptr, &mem));
        VK_CHECK(vkBindBufferMemory(dev, buf, mem, 0));

        if (exportFd) {
            auto vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryFdKHR");
            VkMemoryGetFdInfoKHR getFd = { VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, nullptr, mem, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
            VK_CHECK(vkGetMemoryFdKHR(dev, &getFd, exportFd));
        }
    };

    auto importPeerBuf = [](VkDevice dev, VkPhysicalDevice physDev, int peerFd, VkDeviceSize size, VkBuffer& buf, VkDeviceMemory& mem) {
        VkExternalMemoryBufferCreateInfo extBufInfo = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, nullptr, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
        VkBufferCreateInfo bufInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &extBufInfo, 0, size, 
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE };
        VK_CHECK(vkCreateBuffer(dev, &bufInfo, nullptr, &buf));

        auto vkGetMemoryFdPropertiesKHR = (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryFdPropertiesKHR");
        VkMemoryFdPropertiesKHR fdProps = { VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR };
        VK_CHECK(vkGetMemoryFdPropertiesKHR(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, peerFd, &fdProps));

        VkImportMemoryFdInfoKHR impAlloc = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, nullptr, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, peerFd };
        VkMemoryAllocateInfo alloc = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &impAlloc, size, 0 };
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);
        for (uint32_t m = 0; m < memProps.memoryTypeCount; ++m) {
            if (fdProps.memoryTypeBits & (1 << m)) {
                alloc.memoryTypeIndex = m; break;
            }
        }
        VK_CHECK(vkAllocateMemory(dev, &alloc, nullptr, &mem));
        VK_CHECK(vkBindBufferMemory(dev, buf, mem, 0));
    };

    std::vector<VkBuffer> hostIn(NUM_GPUS), hostOut(NUM_GPUS);
    std::vector<VkDeviceMemory> hostInMem(NUM_GPUS), hostOutMem(NUM_GPUS);

    for (uint32_t i = 0; i < NUM_GPUS; ++i) {
        allocBuf(ctx[i].dev, ctx[i].physDev, TOTAL_BYTES, true, ctx[i].sendBuf, ctx[i].sendMem, &ctx[i].sendFd, false);
        allocBuf(ctx[i].dev, ctx[i].physDev, TOTAL_BYTES, true, ctx[i].recvBuf, ctx[i].recvMem, &ctx[i].recvFd, false);
        allocBuf(ctx[i].dev, ctx[i].physDev, TOTAL_BYTES, true, ctx[i].finalBuf, ctx[i].finalMem, &ctx[i].finalFd, false);
        allocBuf(ctx[i].dev, ctx[i].physDev, TOTAL_BYTES, false, hostIn[i], hostInMem[i], nullptr, true);
        allocBuf(ctx[i].dev, ctx[i].physDev, TOTAL_BYTES, false, hostOut[i], hostOutMem[i], nullptr, true);

        // Fill initial inputs: Rank i gets float(i + 1)
        float* pIn = nullptr;
        VK_CHECK(vkMapMemory(ctx[i].dev, hostInMem[i], 0, TOTAL_BYTES, 0, (void**)&pIn));
        for (uint32_t k = 0; k < NUM_ELEMENTS; ++k) pIn[k] = float(i + 1);
        vkUnmapMemory(ctx[i].dev, hostInMem[i]);
    }

    // Connect peers via full-mesh DMA-BUF P2P
    for (uint32_t i = 0; i < NUM_GPUS; ++i) {
        for (uint32_t j = 0; j < NUM_GPUS; ++j) {
            if (i == j) continue;
            VkBuffer bR, bF;
            VkDeviceMemory mR, mF;
            importPeerBuf(ctx[i].dev, ctx[i].physDev, dup(ctx[j].recvFd), TOTAL_BYTES, bR, mR);
            importPeerBuf(ctx[i].dev, ctx[i].physDev, dup(ctx[j].finalFd), TOTAL_BYTES, bF, mF);
            ctx[i].peerRecvBufs.push_back(bR);
            ctx[i].peerRecvMems.push_back(mR);
            ctx[i].peerFinalBufs.push_back(bF);
            ctx[i].peerFinalMems.push_back(mF);
        }
    }

    // Populate initial inputs
    for (uint32_t i = 0; i < NUM_GPUS; ++i) {
        VkCommandBuffer cb;
        VkCommandBufferAllocateInfo cba = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, ctx[i].cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
        vkAllocateCommandBuffers(ctx[i].dev, &cba, &cb);
        VkCommandBufferBeginInfo beg = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cb, &beg);
        VkBufferCopy cpy = { 0, 0, TOTAL_BYTES };
        vkCmdCopyBuffer(cb, hostIn[i], ctx[i].sendBuf, 1, &cpy);
        vkEndCommandBuffer(cb);
        VkSubmitInfo sub = { VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb, 0, nullptr };
        vkQueueSubmit(ctx[i].queue, 1, &sub, ctx[i].fence);
        vkWaitForFences(ctx[i].dev, 1, &ctx[i].fence, VK_TRUE, UINT64_MAX);
        vkResetFences(ctx[i].dev, 1, &ctx[i].fence);
    }

    // GLSL 5-way reduction shader
    std::string glslSum = R"(
        #version 450
        layout(std430, set = 0, binding = 0) readonly buffer InBuf { float in_data[]; };
        layout(std430, set = 0, binding = 1) buffer OutBuf { float out_data[]; };
        layout(local_size_x = 256) in;
        layout(push_constant) uniform PC { uint chunk_len; } pc;
        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx < pc.chunk_len) {
                float s = in_data[0 * pc.chunk_len + idx]
                        + in_data[1 * pc.chunk_len + idx]
                        + in_data[2 * pc.chunk_len + idx]
                        + in_data[3 * pc.chunk_len + idx]
                        + in_data[4 * pc.chunk_len + idx];
                out_data[idx] = s;
            }
        }
    )";
    auto spvSum = compileOrLoadSPV(glslSum, "/tmp/p2p_sum5_repo.spv");

    for (uint32_t i = 0; i < NUM_GPUS; ++i) {
        VkDescriptorPoolSize psize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
        VkDescriptorPoolCreateInfo dpInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &psize };
        VK_CHECK(vkCreateDescriptorPool(ctx[i].dev, &dpInfo, nullptr, &ctx[i].descPool));

        VkDescriptorSetLayoutBinding bds[2] = {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
        };
        VkDescriptorSetLayoutCreateInfo dslInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, bds };
        VK_CHECK(vkCreateDescriptorSetLayout(ctx[i].dev, &dslInfo, nullptr, &ctx[i].dsLayout));

        VkDescriptorSetAllocateInfo dsAlloc = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, ctx[i].descPool, 1, &ctx[i].dsLayout };
        VK_CHECK(vkAllocateDescriptorSets(ctx[i].dev, &dsAlloc, &ctx[i].descSet));

        VkDescriptorBufferInfo dbiRecv = { ctx[i].recvBuf, 0, TOTAL_BYTES };
        VkDescriptorBufferInfo dbiOut  = { ctx[i].finalBuf, (VkDeviceSize)i * CHUNK_BYTES, CHUNK_BYTES };
        VkWriteDescriptorSet writes[2] = {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ctx[i].descSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dbiRecv, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ctx[i].descSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dbiOut, nullptr }
        };
        vkUpdateDescriptorSets(ctx[i].dev, 2, writes, 0, nullptr);

        VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) };
        VkPipelineLayoutCreateInfo plInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &ctx[i].dsLayout, 1, &pcr };
        VK_CHECK(vkCreatePipelineLayout(ctx[i].dev, &plInfo, nullptr, &ctx[i].plLayout));

        VkShaderModuleCreateInfo smInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, spvSum.size() * 4, spvSum.data() };
        VkShaderModule sm;
        VK_CHECK(vkCreateShaderModule(ctx[i].dev, &smInfo, nullptr, &sm));

        VkComputePipelineCreateInfo cpInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0,
            { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr },
            ctx[i].plLayout, VK_NULL_HANDLE, 0 };
        VK_CHECK(vkCreateComputePipelines(ctx[i].dev, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &ctx[i].pipeline));
        vkDestroyShaderModule(ctx[i].dev, sm, nullptr);
    }

    // Record Two-Stage AllReduce:
    // Phase 1: ReduceScatter (Each rank sends 4 chunks of 8 KiB = 32 KiB sent per rank)
    VkCommandBufferBeginInfo beg = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    for (uint32_t i = 0; i < NUM_GPUS; ++i) {
        VK_CHECK(vkBeginCommandBuffer(ctx[i].cmdPhase1, &beg));

        // 1. Local copy: sendBuf chunk i -> recvBuf slot i
        VkBufferCopy localCpy = { (VkDeviceSize)i * CHUNK_BYTES, (VkDeviceSize)i * CHUNK_BYTES, CHUNK_BYTES };
        vkCmdCopyBuffer(ctx[i].cmdPhase1, ctx[i].sendBuf, ctx[i].recvBuf, 1, &localCpy);

        // 2. Peer PUSH: sendBuf chunk j -> peer j's recvBuf slot i (offset i*CHUNK_BYTES)
        for (uint32_t j = 0; j < NUM_GPUS; ++j) {
            if (i == j) continue;
            int peerSlotIdx = (j < i) ? j : (j - 1);
            VkBufferCopy peerCpy = { (VkDeviceSize)j * CHUNK_BYTES, (VkDeviceSize)i * CHUNK_BYTES, CHUNK_BYTES };
            vkCmdCopyBuffer(ctx[i].cmdPhase1, ctx[i].sendBuf, ctx[i].peerRecvBufs[peerSlotIdx], 1, &peerCpy);
        }

        VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT };
        vkCmdPipelineBarrier(ctx[i].cmdPhase1, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        VK_CHECK(vkEndCommandBuffer(ctx[i].cmdPhase1));
    }

    // Phase 2: Compute Sum + AllGather PUSH (Each rank pushes reduced chunk of 8 KiB to 4 peers = 32 KiB sent per rank)
    for (uint32_t i = 0; i < NUM_GPUS; ++i) {
        VK_CHECK(vkBeginCommandBuffer(ctx[i].cmdPhase2, &beg));

        // 1. Compute reduction sum in local VRAM
        VkMemoryBarrier mb1 = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
        vkCmdPipelineBarrier(ctx[i].cmdPhase2, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb1, 0, nullptr, 0, nullptr);

        vkCmdBindPipeline(ctx[i].cmdPhase2, VK_PIPELINE_BIND_POINT_COMPUTE, ctx[i].pipeline);
        vkCmdBindDescriptorSets(ctx[i].cmdPhase2, VK_PIPELINE_BIND_POINT_COMPUTE, ctx[i].plLayout, 0, 1, &ctx[i].descSet, 0, nullptr);
        uint32_t count = CHUNK_ELEMENTS;
        vkCmdPushConstants(ctx[i].cmdPhase2, ctx[i].plLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &count);
        vkCmdDispatch(ctx[i].cmdPhase2, (CHUNK_ELEMENTS + 255) / 256, 1, 1);

        VkMemoryBarrier mb2 = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT };
        vkCmdPipelineBarrier(ctx[i].cmdPhase2, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb2, 0, nullptr, 0, nullptr);

        // 2. AllGather PUSH: chunk i -> all 4 peers' finalBuf at chunk i
        VkBufferCopy agCpy = { (VkDeviceSize)i * CHUNK_BYTES, (VkDeviceSize)i * CHUNK_BYTES, CHUNK_BYTES };
        for (uint32_t j = 0; j < NUM_GPUS; ++j) {
            if (i == j) continue;
            int peerSlotIdx = (j < i) ? j : (j - 1);
            vkCmdCopyBuffer(ctx[i].cmdPhase2, ctx[i].finalBuf, ctx[i].peerFinalBufs[peerSlotIdx], 1, &agCpy);
        }

        VkMemoryBarrier mb3 = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT };
        vkCmdPipelineBarrier(ctx[i].cmdPhase2, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb3, 0, nullptr, 0, nullptr);
        VK_CHECK(vkEndCommandBuffer(ctx[i].cmdPhase2));
    }

    std::cout << "=== 1. Executing Two-Stage AllReduce with Stage Synchronization ===" << std::endl;
    // Step 1: Phase 1 (ReduceScatter PUSH)
    for (uint32_t i = 0; i < NUM_GPUS; ++i) {
        VkSubmitInfo sub = { VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &ctx[i].cmdPhase1, 0, nullptr };
        VK_CHECK(vkQueueSubmit(ctx[i].queue, 1, &sub, ctx[i].fence));
    }
    for (uint32_t i = 0; i < NUM_GPUS; ++i) {
        VK_CHECK(vkWaitForFences(ctx[i].dev, 1, &ctx[i].fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(ctx[i].dev, 1, &ctx[i].fence));
    }

    // Step 2: Phase 2 (Compute Sum + AllGather PUSH)
    for (uint32_t i = 0; i < NUM_GPUS; ++i) {
        VkSubmitInfo sub = { VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &ctx[i].cmdPhase2, 0, nullptr };
        VK_CHECK(vkQueueSubmit(ctx[i].queue, 1, &sub, ctx[i].fence));
    }
    for (uint32_t i = 0; i < NUM_GPUS; ++i) {
        VK_CHECK(vkWaitForFences(ctx[i].dev, 1, &ctx[i].fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(ctx[i].dev, 1, &ctx[i].fence));
    }

    // Step 3: Strict mathematical verification across all 51,200 elements
    std::cout << "\n=== 2. Mathematical Correctness Verification (All 51,200 elements) ===" << std::endl;
    size_t totalErrors = 0;
    for (uint32_t i = 0; i < NUM_GPUS; ++i) {
        VkCommandBuffer cb;
        VkCommandBufferAllocateInfo cba = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, ctx[i].cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
        vkAllocateCommandBuffers(ctx[i].dev, &cba, &cb);
        VkCommandBufferBeginInfo binfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cb, &binfo);
        VkBufferCopy cpy = { 0, 0, TOTAL_BYTES };
        vkCmdCopyBuffer(cb, ctx[i].finalBuf, hostOut[i], 1, &cpy);
        VkMemoryBarrier mbHost = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mbHost, 0, nullptr, 0, nullptr);
        vkEndCommandBuffer(cb);

        VkSubmitInfo sub = { VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb, 0, nullptr };
        vkQueueSubmit(ctx[i].queue, 1, &sub, ctx[i].fence);
        vkWaitForFences(ctx[i].dev, 1, &ctx[i].fence, VK_TRUE, UINT64_MAX);
        vkResetFences(ctx[i].dev, 1, &ctx[i].fence);

        float* pOut = nullptr;
        VK_CHECK(vkMapMemory(ctx[i].dev, hostOutMem[i], 0, TOTAL_BYTES, 0, (void**)&pOut));
        
        size_t mismatches = 0;
        for (uint32_t k = 0; k < NUM_ELEMENTS; ++k) {
            float val = pOut[k];
            if (val != 15.0f) {
                if (mismatches < 3) {
                    std::cout << "GPU " << i << " mismatch at [" << k << "]: expected 15.0, got " << val << std::endl;
                }
                mismatches++;
            }
        }
        totalErrors += mismatches;
        std::cout << "GPU " << i << " verification: " << (mismatches == 0 ? "PASSED (All 10,240 elements are 15.0f)" : "FAILED") 
                  << " (mismatches=" << mismatches << ")" << std::endl;
        vkUnmapMemory(ctx[i].dev, hostOutMem[i]);
    }

    if (totalErrors == 0) {
        std::cout << ">>> 100% PERFECT ALLREDUCE VERIFICATION! ALL 51,200 ELEMENTS ARE 15.0 ACROSS ALL 5 GPUS! <<<" << std::endl;
    } else {
        std::cout << ">>> VERIFICATION FAILED: total errors = " << totalErrors << " <<<" << std::endl;
        return 1;
    }

    // Step 4: Multi-round Latency Benchmarking (100 runs)
    std::cout << "\n=== 3. Latency Benchmarking (100 Iterations of True Two-Stage AllReduce) ===" << std::endl;
    const int RUNS = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < RUNS; ++r) {
        // Phase 1 (ReduceScatter)
        for (uint32_t i = 0; i < NUM_GPUS; ++i) {
            VkSubmitInfo sub = { VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &ctx[i].cmdPhase1, 0, nullptr };
            vkQueueSubmit(ctx[i].queue, 1, &sub, ctx[i].fence);
        }
        for (uint32_t i = 0; i < NUM_GPUS; ++i) {
            vkWaitForFences(ctx[i].dev, 1, &ctx[i].fence, VK_TRUE, UINT64_MAX);
            vkResetFences(ctx[i].dev, 1, &ctx[i].fence);
        }

        // Phase 2 (Compute Sum + AllGather)
        for (uint32_t i = 0; i < NUM_GPUS; ++i) {
            VkSubmitInfo sub = { VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &ctx[i].cmdPhase2, 0, nullptr };
            vkQueueSubmit(ctx[i].queue, 1, &sub, ctx[i].fence);
        }
        for (uint32_t i = 0; i < NUM_GPUS; ++i) {
            vkWaitForFences(ctx[i].dev, 1, &ctx[i].fence, VK_TRUE, UINT64_MAX);
            vkResetFences(ctx[i].dev, 1, &ctx[i].fence);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_ar_us = total_us / RUNS;

    std::cout << "Data Accounting per AllReduce:" << std::endl;
    std::cout << "  Total reconstructed vector (S)      : " << TOTAL_BYTES << " bytes (40 KiB)" << std::endl;
    std::cout << "  Slice per rank (S/5)                : " << CHUNK_BYTES << " bytes (8 KiB)" << std::endl;
    std::cout << "  Phase 1 sent (4 * S/5)              : " << 4 * CHUNK_BYTES << " bytes (32 KiB)" << std::endl;
    std::cout << "  Phase 2 sent (4 * S/5)              : " << 4 * CHUNK_BYTES << " bytes (32 KiB)" << std::endl;
    std::cout << "  Total data sent per rank (1.6 * S)  : " << 8 * CHUNK_BYTES << " bytes (64 KiB)" << std::endl;
    std::cout << "Timing Results:" << std::endl;
    std::cout << "  Latency per full Two-Stage AllReduce: " << std::fixed << std::setprecision(2) << per_ar_us << " µs" << std::endl;
    std::cout << "  Cumulative 96 AllReduces per Token  : " << std::setprecision(2) << (per_ar_us * 96 / 1000.0) << " ms" << std::endl;

    return 0;
}