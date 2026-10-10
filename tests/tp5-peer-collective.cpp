// Standalone experimental protocol, not a production llama.cpp code path.
#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>
#include <exception>

static void check(VkResult r,const char * what){if(r!=VK_SUCCESS)throw std::runtime_error(std::string(what)+": "+std::to_string(r));}
static void require(bool v,const char *what){if(!v)throw std::runtime_error(what);}
struct Device {
    VkPhysicalDevice physical{};VkDevice device{};VkQueue queue{};uint32_t family{};
    VkBuffer buffer{};VkDeviceMemory memory{};VkDescriptorSetLayout dsl{};
    VkPipelineLayout layout{};VkPipeline pipeline{};VkShaderModule shader{};
    VkDescriptorPool descriptors{};VkDescriptorSet set{};VkCommandPool pool{};
    VkCommandBuffer command{};VkFence fence{};VkQueryPool queries{};
    float period{};uint32_t timestamp_bits{};
    std::array<VkSemaphore,5> peers{};
    std::vector<VkCommandBuffer> commands;
};
static bool extension(VkPhysicalDevice p,const char *name){
    uint32_t n=0;check(vkEnumerateDeviceExtensionProperties(p,nullptr,&n,nullptr),"extensions");
    std::vector<VkExtensionProperties>x(n);check(vkEnumerateDeviceExtensionProperties(p,nullptr,&n,x.data()),"extensions");
    return std::any_of(x.begin(),x.end(),[&](const auto &e){return std::strcmp(e.extensionName,name)==0;});
}

int main(int argc,char **argv)try{
    const bool timeline=std::getenv("TP5_PEER_TIMELINE")&&std::strcmp(std::getenv("TP5_PEER_TIMELINE"),"1")==0;
    const bool dma_peer=std::getenv("TP5_PEER_DMABUF")&&std::strcmp(std::getenv("TP5_PEER_DMABUF"),"1")==0;
    require(!dma_peer||!timeline,"DMA peer and external timeline are separate protocol cases");
    require(argc==5||argc==7,"usage: peer-probe SHADER elements stages replays [withhold_rank withhold_stage]");
    const uint32_t elements=std::stoul(argv[2]),stages=std::stoul(argv[3]),replays=std::stoul(argv[4]);
    const uint32_t withhold_rank=argc==7?std::stoul(argv[5]):UINT32_MAX;
    const uint32_t withhold_stage=argc==7?std::stoul(argv[6]):UINT32_MAX;
    require(elements>=4&&elements<=4096&&elements%4==0&&stages>=1&&stages<=1024&&replays>=1&&replays<=64,"bounded shape/replay constraints");
    require(argc==5||(withhold_rank<5&&withhold_stage<stages&&replays==1),"bounded fault injection only");
    require(!timeline||argc==5,"timeline probe does not inject unsignalled waits");
    std::ifstream file(argv[1],std::ios::binary|std::ios::ate);require(bool(file),"shader file");
    const auto bytes=file.tellg();require(bytes>0&&size_t(bytes)%4==0,"SPIR-V extent");
    std::vector<uint32_t>spirv(size_t(bytes)/4);file.seekg(0);file.read((char*)spirv.data(),bytes);
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="tp5-peer-proof";app.apiVersion=VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ici.pApplicationInfo=&app;
    VkInstance instance{};check(vkCreateInstance(&ici,nullptr,&instance),"instance");
    uint32_t np=0;check(vkEnumeratePhysicalDevices(instance,&np,nullptr),"devices");
    std::vector<VkPhysicalDevice>all(np);check(vkEnumeratePhysicalDevices(instance,&np,all.data()),"devices");
    std::vector<Device>ds;
    VkDeviceSize alignment=4096;
    for(auto physical:all){
        VkPhysicalDeviceProperties props{};vkGetPhysicalDeviceProperties(physical,&props);
        if(props.vendorID!=0x1002||props.deviceID!=0x73bf)continue;
        const char * synchronization_extension=timeline?VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME:VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME;
        require(extension(physical,VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)&&extension(physical,synchronization_extension),"required import/synchronization extension unavailable");
        if(dma_peer)require(extension(physical,VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME)&&extension(physical,VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME),"DMA-BUF export/import extensions unavailable");
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT host{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};p2.pNext=&host;vkGetPhysicalDeviceProperties2(physical,&p2);
        alignment=std::max(alignment,host.minImportedHostPointerAlignment);
        uint32_t nq=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,nullptr);std::vector<VkQueueFamilyProperties>q(nq);vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,q.data());
        Device d;d.physical=physical;d.family=UINT32_MAX;
        for(uint32_t i=0;i<nq;++i)if(q[i].queueFlags&VK_QUEUE_COMPUTE_BIT){d.family=i;break;}
        require(d.family!=UINT32_MAX&&q[d.family].timestampValidBits,"compute/timestamp support");
        d.period=props.limits.timestampPeriod;d.timestamp_bits=q[d.family].timestampValidBits;
        VkPhysicalDeviceCoherentMemoryFeaturesAMD cf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD};
        VkPhysicalDeviceVulkan12Features available12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};cf.pNext=&available12;
        VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};f.pNext=&cf;vkGetPhysicalDeviceFeatures2(physical,&f);
        require(timeline?bool(available12.timelineSemaphore):bool(cf.deviceCoherentMemory),"required synchronization feature unavailable");
        cf.pNext=nullptr;VkPhysicalDeviceVulkan12Features enabled12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};enabled12.timelineSemaphore=timeline;
        VkPhysicalDeviceFeatures2 enable{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};enable.pNext=timeline?(void*)&enabled12:(void*)&cf;
        float priority=1;VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=d.family;qi.queueCount=1;qi.pQueuePriorities=&priority;
        const char *exts[]={VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME,synchronization_extension,VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME};
        VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};ci.pNext=&enable;ci.queueCreateInfoCount=1;ci.pQueueCreateInfos=&qi;ci.enabledExtensionCount=dma_peer?4:2;ci.ppEnabledExtensionNames=exts;
        check(vkCreateDevice(physical,&ci,nullptr,&d.device),"device");vkGetDeviceQueue(d.device,d.family,0,&d.queue);ds.push_back(d);
    }
    require(ds.size()==5,"exactly five distinct physical Navi21 devices required");
    require((alignment&(alignment-1))==0,"power-of-two host alignment");
    const uint32_t stride_words=((16+elements+63)/64)*64;
    const uint32_t output_words=2*5*stride_words;
    const VkDeviceSize used=(output_words+5*elements)*sizeof(uint32_t);
    const VkDeviceSize size=(used+alignment-1)&~(alignment-1);
    void *storage=nullptr;require(posix_memalign(&storage,alignment,size)==0,"aligned host allocation");std::memset(storage,0,size);
    auto *words=(uint32_t*)storage;
    VkDeviceSize exported_allocation_size=0;
    void *mapped_export=nullptr;
    // This allocation remains alive until every VkDevice has drained and all
    // imported VkDeviceMemory objects have been released.
    for(size_t rank=0;rank<5;++rank){auto &d=ds[rank];
        VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};external.handleTypes=dma_peer?VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT:VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.pNext=&external;bi.size=size;bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        check(vkCreateBuffer(d.device,&bi,nullptr,&d.buffer),"external buffer");VkMemoryRequirements req{};vkGetBufferMemoryRequirements(d.device,d.buffer,&req);require(req.size<=size,"aligned external size covers requirements");
        VkMemoryHostPointerPropertiesEXT supported{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};supported.memoryTypeBits=UINT32_MAX;
        int imported_fd=-1;
        if(!dma_peer){
            auto hp=(PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(d.device,"vkGetMemoryHostPointerPropertiesEXT");require(hp,"host pointer query");
            check(hp(d.device,VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,storage,&supported),"host import properties");
        }else if(rank!=0){
            auto get_fd=(PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(ds[0].device,"vkGetMemoryFdKHR");
            auto properties=(PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(d.device,"vkGetMemoryFdPropertiesKHR");require(get_fd&&properties,"memoryFD functions");
            VkMemoryGetFdInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};gi.memory=ds[0].memory;gi.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;check(get_fd(ds[0].device,&gi,&imported_fd),"export owner allocation fd");
            VkMemoryFdPropertiesKHR fp{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};check(properties(d.device,VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,imported_fd,&fp),"DMA-BUF properties");supported.memoryTypeBits=fp.memoryTypeBits;
        }
        VkPhysicalDeviceMemoryProperties mp{};vkGetPhysicalDeviceMemoryProperties(d.physical,&mp);
        const bool gpu_only_import=dma_peer&&rank!=0;
        const uint32_t required=(gpu_only_import?0u:VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)|
            (timeline?0u:VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD)|(dma_peer?VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT:0u);
        uint32_t mt=UINT32_MAX;
        std::fprintf(stderr,"peer rank=%zu buffer_memory_bits=0x%x import_memory_bits=0x%x required_flags=0x%x\n",rank,req.memoryTypeBits,supported.memoryTypeBits,required);
        for(uint32_t i=0;i<mp.memoryTypeCount;++i)if((req.memoryTypeBits&supported.memoryTypeBits)&(1u<<i))
            std::fprintf(stderr,"peer eligible-type rank=%zu index=%u flags=0x%x\n",rank,i,mp.memoryTypes[i].propertyFlags);
        for(uint32_t i=0;i<mp.memoryTypeCount;++i)if(((req.memoryTypeBits&supported.memoryTypeBits)&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&required)==required){mt=i;break;}
        require(mt!=UINT32_MAX,timeline?"No eligible coherent host import for external-semaphore protocol":"import must provide bothHOST_COHERENT andDEVICE_COHERENT; no weaker fallback");
        std::fprintf(stderr,"peer rank=%zu memory_type=%u flags=0x%x bytes=%llu family=%u\n",rank,mt,mp.memoryTypes[mt].propertyFlags,(unsigned long long)size,d.family);
        VkImportMemoryHostPointerInfoEXT im{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};im.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;im.pHostPointer=storage;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.pNext=&im;ai.allocationSize=size;ai.memoryTypeIndex=mt;
        VkExportMemoryAllocateInfo export_info{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};export_info.handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        VkImportMemoryFdInfoKHR import_info{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};import_info.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;import_info.fd=imported_fd;
        VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};dedicated.buffer=d.buffer;
        if(dma_peer){
            dedicated.pNext=rank==0?(void*)&export_info:(void*)&import_info;ai.pNext=&dedicated;
            ai.allocationSize=rank==0?req.size:exported_allocation_size;
            require(ai.allocationSize>=req.size,"peer allocation must cover exact exported extent");
        }
        check(vkAllocateMemory(d.device,&ai,nullptr,&d.memory),dma_peer?"DMA memory":"import memory");check(vkBindBufferMemory(d.device,d.buffer,d.memory,0),"bind import");
        if(dma_peer&&rank==0){
            exported_allocation_size=ai.allocationSize;check(vkMapMemory(d.device,d.memory,0,size,0,&mapped_export),"map owner coherent allocation");
            std::memset(mapped_export,0,size);words=(uint32_t*)mapped_export;
        }
        VkDescriptorSetLayoutBinding binding{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};sli.bindingCount=1;sli.pBindings=&binding;check(vkCreateDescriptorSetLayout(d.device,&sli,nullptr,&d.dsl),"descriptor layout");
        VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT,0,40};VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pli.setLayoutCount=1;pli.pSetLayouts=&d.dsl;pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&pc;check(vkCreatePipelineLayout(d.device,&pli,nullptr,&d.layout),"pipeline layout");
        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};sm.codeSize=spirv.size()*4;sm.pCode=spirv.data();check(vkCreateShaderModule(d.device,&sm,nullptr,&d.shader),"shader");
        VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cp.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,d.shader,"main",nullptr};cp.layout=d.layout;check(vkCreateComputePipelines(d.device,VK_NULL_HANDLE,1,&cp,nullptr,&d.pipeline),"pipeline");
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1};VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dpi.maxSets=1;dpi.poolSizeCount=1;dpi.pPoolSizes=&ps;check(vkCreateDescriptorPool(d.device,&dpi,nullptr,&d.descriptors),"descriptor pool");
        VkDescriptorSetAllocateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};di.descriptorPool=d.descriptors;di.descriptorSetCount=1;di.pSetLayouts=&d.dsl;check(vkAllocateDescriptorSets(d.device,&di,&d.set),"descriptor set");
        VkDescriptorBufferInfo db{d.buffer,0,size};VkWriteDescriptorSet dw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,d.set,0,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,nullptr,&db,nullptr};vkUpdateDescriptorSets(d.device,1,&dw,0,nullptr);
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pool.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;pool.queueFamilyIndex=d.family;check(vkCreateCommandPool(d.device,&pool,nullptr,&d.pool),"command pool");
        d.commands.resize(timeline?stages+1:1);
        VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ca.commandPool=d.pool;ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ca.commandBufferCount=d.commands.size();check(vkAllocateCommandBuffers(d.device,&ca,d.commands.data()),"command buffer");d.command=d.commands[0];
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};check(vkCreateFence(d.device,&fi,nullptr,&d.fence),"fence");
        VkQueryPoolCreateInfo qp{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};qp.queryType=VK_QUERY_TYPE_TIMESTAMP;qp.queryCount=2;check(vkCreateQueryPool(d.device,&qp,nullptr,&d.queries),"queries");
    }
    if(timeline){
        for(size_t r=0;r<5;++r){
            VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};type.semaphoreType=VK_SEMAPHORE_TYPE_TIMELINE;
            VkPhysicalDeviceExternalSemaphoreInfo ei{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO};ei.pNext=&type;ei.handleType=VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
            VkExternalSemaphoreProperties ep{VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};vkGetPhysicalDeviceExternalSemaphoreProperties(ds[r].physical,&ei,&ep);
            require((ep.externalSemaphoreFeatures&(VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT|VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT))==(VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT|VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT),"Timeline OPAQUE_FD import/export required");
            VkExportSemaphoreCreateInfo export_info{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};export_info.handleTypes=VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;type.pNext=&export_info;
            VkSemaphoreCreateInfo ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};ci.pNext=&type;check(vkCreateSemaphore(ds[r].device,&ci,nullptr,&ds[r].peers[r]),"create export timeline");
        }
        for(size_t owner=0;owner<5;++owner)for(size_t consumer=0;consumer<5;++consumer)if(owner!=consumer){
            auto getfd=(PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(ds[owner].device,"vkGetSemaphoreFdKHR");
            auto importfd=(PFN_vkImportSemaphoreFdKHR)vkGetDeviceProcAddr(ds[consumer].device,"vkImportSemaphoreFdKHR");require(getfd&&importfd,"external semaphore entrypoints");
            VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};type.semaphoreType=VK_SEMAPHORE_TYPE_TIMELINE;
            VkSemaphoreCreateInfo ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};ci.pNext=&type;check(vkCreateSemaphore(ds[consumer].device,&ci,nullptr,&ds[consumer].peers[owner]),"create imported timeline");
            VkSemaphoreGetFdInfoKHR gi{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};gi.semaphore=ds[owner].peers[owner];gi.handleType=VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;int fd=-1;check(getfd(ds[owner].device,&gi,&fd),"export timeline fd");
            VkImportSemaphoreFdInfoKHR ii{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR};ii.semaphore=ds[consumer].peers[owner];ii.handleType=VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;ii.fd=fd;check(importfd(ds[consumer].device,&ii),"import timeline fd");
        }
    }
    std::vector<float>reference(5*elements),next(5*elements);std::vector<double>wall,gpu;
    for(uint32_t run=0;run<replays;++run){
        for(uint32_t r=0;r<5;++r)for(uint32_t e=0;e<elements;++e)reference[r*elements+e]=float(r+1)*.25f+float(e%7)*.03125f+float(run%8)*.0625f;
        for(uint32_t s=0;s<stages;++s){for(uint32_t e=0;e<elements;++e){float sum=0;for(uint32_t r=0;r<5;++r)sum+=reference[r*elements+e];for(uint32_t r=0;r<5;++r)next[r*elements+e]=sum*.125f+float(r+1)*.25f+float(e%7)*.03125f;}reference.swap(next);}
        for(uint32_t r=0;r<5;++r){auto &d=ds[r];check(vkResetFences(d.device,1,&d.fence),"reset fence");
            if(timeline){
                for(uint32_t stage=0;stage<=stages;++stage){const auto cmd=d.commands[stage];check(vkResetCommandBuffer(cmd,0),"reset stage command");VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};check(vkBeginCommandBuffer(cmd,&begin),"begin stage");
                    if(stage==0)vkCmdResetQueryPool(cmd,d.queries,0,2);
                    VkMemoryBarrier acquire{VK_STRUCTURE_TYPE_MEMORY_BARRIER};acquire.srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT;acquire.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&acquire,0,nullptr,0,nullptr);
                    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,d.layout,0,1,&d.set,0,nullptr);
                    const uint32_t push[]={r,elements,stage,stages,run*(stages+1)+1,run,stride_words,output_words,0,0};vkCmdPushConstants(cmd,d.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(push),push);
                    if(stage==0)vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,d.queries,0);
                    vkCmdDispatch(cmd,1,1,1);
                    VkMemoryBarrier release{VK_STRUCTURE_TYPE_MEMORY_BARRIER};release.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;release.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&release,0,nullptr,0,nullptr);
                    if(stage==stages)vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,d.queries,1);
                    check(vkEndCommandBuffer(cmd),"end stage");
                }
                continue;
            }
            check(vkResetCommandBuffer(d.command,0),"reset command");VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};check(vkBeginCommandBuffer(d.command,&bi),"begin");vkCmdResetQueryPool(d.command,d.queries,0,2);
            VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};host.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT;host.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;vkCmdPipelineBarrier(d.command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&host,0,nullptr,0,nullptr);
            vkCmdBindPipeline(d.command,VK_PIPELINE_BIND_POINT_COMPUTE,d.pipeline);vkCmdBindDescriptorSets(d.command,VK_PIPELINE_BIND_POINT_COMPUTE,d.layout,0,1,&d.set,0,nullptr);
            const uint32_t push[]={r,elements,stages,run*(stages+1)+1,run,stride_words,output_words,100000,withhold_rank,withhold_stage};vkCmdPushConstants(d.command,d.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(push),push);
            vkCmdWriteTimestamp(d.command,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,d.queries,0);vkCmdDispatch(d.command,1,1,1);vkCmdWriteTimestamp(d.command,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,d.queries,1);check(vkEndCommandBuffer(d.command),"end");}
        const auto start=std::chrono::steady_clock::now();
        const auto submit_rank=[&](size_t r){auto &d=ds[r];
            if(timeline){
                std::vector<VkSubmitInfo> submissions(stages+1);
                std::vector<VkTimelineSemaphoreSubmitInfo> times(stages+1);
                std::vector<std::array<uint64_t,5>> waits(stages+1);std::vector<uint64_t>signals(stages+1);
                const VkPipelineStageFlags dst_stages[5]={VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
                for(uint32_t stage=0;stage<=stages;++stage){signals[stage]=run*(stages+1)+1+stage;waits[stage].fill(signals[stage]-1);
                    auto &ti=times[stage];ti={VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};ti.waitSemaphoreValueCount=stage?5:0;ti.pWaitSemaphoreValues=waits[stage].data();ti.signalSemaphoreValueCount=1;ti.pSignalSemaphoreValues=&signals[stage];
                    auto &si=submissions[stage];si={VK_STRUCTURE_TYPE_SUBMIT_INFO};si.pNext=&ti;si.waitSemaphoreCount=stage?5:0;si.pWaitSemaphores=d.peers.data();si.pWaitDstStageMask=dst_stages;si.commandBufferCount=1;si.pCommandBuffers=&d.commands[stage];si.signalSemaphoreCount=1;si.pSignalSemaphores=&d.peers[r];
                }
                check(vkQueueSubmit(d.queue,submissions.size(),submissions.data(),d.fence),"submit external timeline chain");
            }else{VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&d.command;check(vkQueueSubmit(d.queue,1,&si,d.fence),"submit");}
        };
        if(timeline){
            // Out-of-order cross-device timeline waits may depend on another
            // queue's not-yet-submitted signal. Submit distinct VkQueues from
            // independent threads; never rely on a blocked submit to advance
            // another physical device on the same host thread.
            std::array<std::exception_ptr,5> errors{};std::vector<std::thread> submitters;
            for(size_t r=0;r<5;++r)submitters.emplace_back([&,r]{try{submit_rank(r);}catch(...){errors[r]=std::current_exception();}});
            for(auto &worker:submitters)worker.join();
            for(auto &error:errors)if(error)std::rethrow_exception(error);
        }else for(size_t r=0;r<5;++r)submit_rank(r);
        for(auto &d:ds)check(vkWaitForFences(d.device,1,&d.fence,VK_TRUE,5000000000ull),"bounded fence drain");
        const double elapsed=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();double max_gpu=0;
        for(auto &d:ds){uint64_t t[2]{};check(vkGetQueryPoolResults(d.device,d.queries,0,2,sizeof(t),t,8,VK_QUERY_RESULT_64_BIT),"query result");const uint64_t mask=d.timestamp_bits>=64?UINT64_MAX:((1ull<<d.timestamp_bits)-1);max_gpu=std::max(max_gpu,double((t[1]-t[0])&mask)*d.period/1000);}
        const float *actual=(float*)(words+output_words);
        if(argc==5){
            for(size_t i=0;i<reference.size();++i)if(std::memcmp(actual+i,reference.data()+i,4)){std::fprintf(stderr,"mismatch run=%u index=%zu got=%.9g expected=%.9g\n",run,i,actual[i],reference[i]);throw std::runtime_error("full output mismatch");}
            for(uint32_t r=0;r<5;++r)for(uint32_t bank=0;bank<2;++bank)require(words[(bank*5+r)*stride_words+1]==0,"unexpected sticky error");
        }else{
            for(size_t i=0;i<reference.size();++i)require(std::isnan(actual[i]),"withheld producer must poison all ranks' live output");
            for(uint32_t r=0;r<5;++r)require(words[((withhold_stage&1u)*5+r)*stride_words+1]!=0,"missing bounded failure");
        }
        if(run>=2||replays<3){wall.push_back(elapsed);gpu.push_back(max_gpu);}
    }
    std::sort(wall.begin(),wall.end());std::sort(gpu.begin(),gpu.end());
    std::fprintf(stderr,"peer_protocol=%s\n",timeline?"external-timeline-semaphores":dma_peer?"exported-device-coherent-DMA-BUF":"device-coherent-host-import-polling");
    std::printf("{\"kind\":\"CONSTRUCTIVE_GPU_PEER_COLLECTIVE_NOT_MODEL_TPS\",\"ranks\":5,\"elements\":%u,\"stages\":%u,\"replays\":%u,\"timed\":%zu,\"host_stage_handoffs\":0,\"median_wall_us\":%.6f,\"median_gpu_span_us\":%.6f,\"gpu_us_per_stage\":%.6f,\"full_outputs_checked\":true,\"failure_injected\":%s}\n",elements,stages,replays,gpu.size(),wall[wall.size()/2],gpu[gpu.size()/2],gpu[gpu.size()/2]/stages,argc==7?"true":"false");
    for(auto &d:ds)check(vkDeviceWaitIdle(d.device),"final all-device drain");
    if(mapped_export)vkUnmapMemory(ds[0].device,ds[0].memory);
    for(auto &d:ds){for(auto sem:d.peers)if(sem)vkDestroySemaphore(d.device,sem,nullptr);vkDestroyQueryPool(d.device,d.queries,nullptr);vkDestroyFence(d.device,d.fence,nullptr);vkDestroyCommandPool(d.device,d.pool,nullptr);vkDestroyDescriptorPool(d.device,d.descriptors,nullptr);vkDestroyPipeline(d.device,d.pipeline,nullptr);vkDestroyShaderModule(d.device,d.shader,nullptr);vkDestroyPipelineLayout(d.device,d.layout,nullptr);vkDestroyDescriptorSetLayout(d.device,d.dsl,nullptr);vkDestroyBuffer(d.device,d.buffer,nullptr);vkFreeMemory(d.device,d.memory,nullptr);vkDestroyDevice(d.device,nullptr);}
    std::free(storage);vkDestroyInstance(instance,nullptr);return 0;
}catch(const std::exception &e){std::fprintf(stderr,"peer-probe failed: %s\n",e.what());return 1;}
