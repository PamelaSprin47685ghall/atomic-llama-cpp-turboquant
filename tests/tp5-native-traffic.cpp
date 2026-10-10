// Independent RDNA2 native-F32-activation matvec traffic measurement.
// Enumerates named counters and storage types; never reuses old MMVQ indices.
#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(VkResult r,const char * s) { if(r!=VK_SUCCESS)throw std::runtime_error(std::string(s)+":"+std::to_string(r)); }
static void require(bool b,const char * s) { if(!b)throw std::runtime_error(s); }
struct buffer { VkBuffer b{};VkDeviceMemory m{};VkDeviceSize size{}; };

int main(int argc,char **argv) try {
    require(argc==6,"usage: tp5-native-traffic shader.spv q6|q8 K M empty|dispatch");
    const bool q6=std::string(argv[2])=="q6", q8=std::string(argv[2])=="q8";
    const uint32_t k=std::stoul(argv[3]),m=std::stoul(argv[4]);
    const bool empty=std::string(argv[5])=="empty";
    require((q6||q8)&&k>=32&&k<=16384&&m>=2&&m<=248320&&k%(q6?256:32)==0&&m%2==0&&
            (empty||std::string(argv[5])=="dispatch"),"bounded input/shape");
    std::ifstream f(argv[1],std::ios::binary|std::ios::ate);require(f.good(),"missing shader");
    const auto bytes=f.tellg();require(bytes>0&&bytes%4==0,"invalid shader length");
    std::vector<uint32_t> spv(size_t(bytes)/4);f.seekg(0);f.read((char*)spv.data(),bytes);require(f.good(),"shader read");
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.apiVersion=VK_API_VERSION_1_3;app.pApplicationName="tp5-native-traffic";
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ici.pApplicationInfo=&app;
    VkInstance instance{};check(vkCreateInstance(&ici,nullptr,&instance),"instance");
    uint32_t nd=0;check(vkEnumeratePhysicalDevices(instance,&nd,nullptr),"enumerate");
    std::vector<VkPhysicalDevice> ds(nd);check(vkEnumeratePhysicalDevices(instance,&nd,ds.data()),"devices");
    VkPhysicalDevice physical{};VkPhysicalDeviceProperties prop{};
    for(auto d:ds){VkPhysicalDeviceProperties p{};vkGetPhysicalDeviceProperties(d,&p);if(p.vendorID==0x1002&&p.deviceID==0x73bf){physical=d;prop=p;break;}}
    require(physical!=VK_NULL_HANDLE,"no RX6800");
    uint32_t nq=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,nullptr);std::vector<VkQueueFamilyProperties> qs(nq);vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,qs.data());
    uint32_t family=UINT32_MAX;for(uint32_t i=0;i<nq;++i)if(qs[i].queueFlags&VK_QUEUE_GRAPHICS_BIT){family=i;break;}
    require(family!=UINT32_MAX,"no production-like graphics/compute queue");
    auto enumerate=(PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR)vkGetInstanceProcAddr(instance,"vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR");
    require(enumerate!=nullptr,"counter enumeration unavailable");
    uint32_t nc=0;check(enumerate(physical,family,&nc,nullptr,nullptr),"counter count");
    std::vector<VkPerformanceCounterKHR> counters(nc,{VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_KHR});
    std::vector<VkPerformanceCounterDescriptionKHR> descriptions(nc,{VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_DESCRIPTION_KHR});
    check(enumerate(physical,family,&nc,counters.data(),descriptions.data()),"counter descriptions");
    std::vector<uint32_t> selected;
    for(uint32_t i=0;i<nc;++i){
        std::printf("counter_catalog index=%u storage=%d scope=%d unit=%d name=%s\n",i,(int)counters[i].storage,(int)counters[i].scope,(int)counters[i].unit,descriptions[i].name);
        std::string name(descriptions[i].name);
        std::transform(name.begin(),name.end(),name.begin(),[](unsigned char c){return char(std::tolower(c));});
        if(name=="vram read size"||name=="vmem load instructions"||name=="waves"||
           name=="valu instructions"||name=="salu instructions"||name=="gpu active cycles")selected.push_back(i);
    }
    require(selected.size()==6,"expected six currently enumerated named counters absent; no hard-coded fallback");
    VkPhysicalDevicePerformanceQueryFeaturesKHR perf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PERFORMANCE_QUERY_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};f13.pNext=&perf;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};f12.pNext=&f13;
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};f11.pNext=&f12;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};f2.pNext=&f11;vkGetPhysicalDeviceFeatures2(physical,&f2);
    require(perf.performanceCounterQueryPools&&f13.subgroupSizeControl&&f2.features.shaderInt16&&f12.storageBuffer8BitAccess,"required features unavailable");
    const float priority=1;VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qci.queueFamilyIndex=family;qci.queueCount=1;qci.pQueuePriorities=&priority;
    const char * ext=VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};dci.pNext=&f2;dci.queueCreateInfoCount=1;dci.pQueueCreateInfos=&qci;dci.enabledExtensionCount=1;dci.ppEnabledExtensionNames=&ext;
    VkDevice device{};check(vkCreateDevice(physical,&dci,nullptr,&device),"device");VkQueue queue{};vkGetDeviceQueue(device,family,0,&queue);
    VkPhysicalDeviceMemoryProperties memory{};vkGetPhysicalDeviceMemoryProperties(physical,&memory);
    const auto allocate=[&](VkDeviceSize size,VkBufferUsageFlags usage,VkMemoryPropertyFlags flags){
        buffer b{};b.size=size;VkBufferCreateInfo bc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bc.size=size;bc.usage=usage;bc.sharingMode=VK_SHARING_MODE_EXCLUSIVE;check(vkCreateBuffer(device,&bc,nullptr,&b.b),"buffer");
        VkMemoryRequirements mr{};vkGetBufferMemoryRequirements(device,b.b,&mr);uint32_t mt=UINT32_MAX;
        for(uint32_t i=0;i<memory.memoryTypeCount;++i){
            const auto have=memory.memoryTypes[i].propertyFlags;
            std::printf("memory_catalog index=%u flags=0x%x heap=%u\n",i,have,memory.memoryTypes[i].heapIndex);
            if(!(mr.memoryTypeBits&(1u<<i))||(have&flags)!=flags)continue;
            // AMD coherent/uncached types are not ordinary cached model VRAM,
            // and the corresponding feature was not enabled on this device.
            if(have&(VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD|VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD))continue;
            if((flags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)&&(have&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))continue;
            if(mt==UINT32_MAX)mt=i;
        }
        require(mt!=UINT32_MAX,"ordinary cached memory type");
        std::printf("memory_selected bytes=%llu type=%u flags=0x%x requested=0x%x\n",(unsigned long long)size,mt,memory.memoryTypes[mt].propertyFlags,flags);
        VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ma.allocationSize=mr.size;ma.memoryTypeIndex=mt;check(vkAllocateMemory(device,&ma,nullptr,&b.m),"memory");check(vkBindBufferMemory(device,b.b,b.m,0),"bind");return b;
    };
    const VkDeviceSize wb=VkDeviceSize(k)*m/(q6?256:32)*(q6?210:34),ab=VkDeviceSize(k)*4,ob=VkDeviceSize(m)*4;
    require(wb<600ULL*1024*1024,"memory budget");
    auto w=allocate(wb,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    auto a=allocate(ab,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    auto o=allocate(ob,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    auto host=allocate(wb+ab+ob,VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void * mapped=nullptr;check(vkMapMemory(device,host.m,0,VK_WHOLE_SIZE,0,&mapped),"map");auto * raw=(uint8_t*)mapped;
    for(VkDeviceSize off=0;off<wb;off+=(q6?210:34)){
        if(q6){std::memset(raw+off,0x11,128);std::memset(raw+off+128,0x55,64);std::memset(raw+off+192,1,16);raw[off+208]=0;raw[off+209]=0x3c;}
        else{raw[off]=0;raw[off+1]=0x3c;std::memset(raw+off+2,1,32);}
    }
    for(uint32_t i=0;i<k;++i)((float*)(raw+wb))[i]=1;
    VkDescriptorSetLayoutBinding bindings[5]{};for(uint32_t i=0;i<5;++i)bindings[i]={i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};dl.bindingCount=5;dl.pBindings=bindings;VkDescriptorSetLayout dsl{};check(vkCreateDescriptorSetLayout(device,&dl,nullptr,&dsl),"dsl");
    VkDescriptorPoolSize poolsize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,5};VkDescriptorPoolCreateInfo dp{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dp.maxSets=1;dp.poolSizeCount=1;dp.pPoolSizes=&poolsize;VkDescriptorPool pool{};check(vkCreateDescriptorPool(device,&dp,nullptr,&pool),"pool");
    VkDescriptorSetAllocateInfo da{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};da.descriptorPool=pool;da.descriptorSetCount=1;da.pSetLayouts=&dsl;VkDescriptorSet set{};check(vkAllocateDescriptorSets(device,&da,&set),"set");
    VkDescriptorBufferInfo infos[5]{{w.b,0,wb},{a.b,0,ab},{o.b,0,ob},{o.b,0,ob},{o.b,0,ob}};VkWriteDescriptorSet writes[5]{};
    for(uint32_t i=0;i<5;++i){writes[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,set,i,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,nullptr,&infos[i],nullptr};}vkUpdateDescriptorSets(device,5,writes,0,nullptr);
    struct PC{uint32_t ncols,stride_a,stride_b,stride_d,batch_stride_a,batch_stride_b,batch_stride_d,fusion_flags,base_work_group_y,ne02,ne12,broadcast2,broadcast3;};
    const PC pc{k,k,k,m,k*m,k,m,0,0,1,1,1,1};static_assert(sizeof(PC)==52);
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pc)};VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&dsl;pl.pushConstantRangeCount=1;pl.pPushConstantRanges=&range;VkPipelineLayout layout{};check(vkCreatePipelineLayout(device,&pl,nullptr,&layout),"layout");
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};sm.codeSize=spv.size()*4;sm.pCode=spv.data();VkShaderModule module{};check(vkCreateShaderModule(device,&sm,nullptr,&module),"shader");
    const std::array<uint32_t,3> constants{{32,q6?2u:1u,1}};const VkSpecializationMapEntry entries[3]{{0,0,4},{1,4,4},{2,8,4}};VkSpecializationInfo special{3,entries,sizeof(constants),constants.data()};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};subgroup.requiredSubgroupSize=32;
    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cp.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,&subgroup,0,VK_SHADER_STAGE_COMPUTE_BIT,module,"main",&special};cp.layout=layout;VkPipeline pipeline{};check(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&cp,nullptr,&pipeline),"pipeline");
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pi.queueFamilyIndex=family;VkCommandPool commands{};check(vkCreateCommandPool(device,&pi,nullptr,&commands),"commandpool");
    const auto begin=[&](){VkCommandBufferAllocateInfo ci{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ci.commandPool=commands;ci.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ci.commandBufferCount=1;VkCommandBuffer cb{};check(vkAllocateCommandBuffers(device,&ci,&cb),"cb");VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};check(vkBeginCommandBuffer(cb,&bi),"begin");return cb;};
    const auto submit=[&](VkCommandBuffer cb,const void * next=nullptr){check(vkEndCommandBuffer(cb),"end");VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.pNext=next;si.commandBufferCount=1;si.pCommandBuffers=&cb;VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};VkFence fence{};check(vkCreateFence(device,&fi,nullptr,&fence),"fence");check(vkQueueSubmit(queue,1,&si,fence),"submit");check(vkWaitForFences(device,1,&fence,VK_TRUE,3000000000ULL),"bounded wait");vkDestroyFence(device,fence,nullptr);};
    const auto dispatch=[&](VkCommandBuffer cb){vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&set,0,nullptr);vkCmdPushConstants(cb,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pc),&pc);vkCmdDispatch(cb,m/(q6?2:1),1,1);};
    auto prep=begin();VkBufferCopy wc{0,0,wb},ac{wb,0,ab};vkCmdCopyBuffer(prep,host.b,w.b,1,&wc);vkCmdCopyBuffer(prep,host.b,a.b,1,&ac);vkCmdFillBuffer(prep,o.b,0,ob,0);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER,nullptr,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};vkCmdPipelineBarrier(prep,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,nullptr,0,nullptr);submit(prep);
    VkQueryPoolPerformanceCreateInfoKHR pcq{VK_STRUCTURE_TYPE_QUERY_POOL_PERFORMANCE_CREATE_INFO_KHR};pcq.queueFamilyIndex=family;pcq.counterIndexCount=selected.size();pcq.pCounterIndices=selected.data();
    auto passfn=(PFN_vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR)vkGetInstanceProcAddr(instance,"vkGetPhysicalDeviceQueueFamilyPerformanceQueryPassesKHR");require(passfn,"pass function");uint32_t passes=0;passfn(physical,&pcq,&passes);require(passes&&passes<=8,"pass count");
    VkQueryPoolCreateInfo qc{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};qc.pNext=&pcq;qc.queryType=VK_QUERY_TYPE_PERFORMANCE_QUERY_KHR;qc.queryCount=1;VkQueryPool query{};check(vkCreateQueryPool(device,&qc,nullptr,&query),"perfquery");
    auto acquire=(PFN_vkAcquireProfilingLockKHR)vkGetDeviceProcAddr(device,"vkAcquireProfilingLockKHR");auto release=(PFN_vkReleaseProfilingLockKHR)vkGetDeviceProcAddr(device,"vkReleaseProfilingLockKHR");require(acquire&&release,"profiling lock functions");VkAcquireProfilingLockInfoKHR li{VK_STRUCTURE_TYPE_ACQUIRE_PROFILING_LOCK_INFO_KHR};li.timeout=3000000000ULL;check(acquire(device,&li),"profiling lock");
    for(unsigned trial=0;trial<3;++trial){
        auto reset=begin();vkCmdResetQueryPool(reset,query,0,1);submit(reset);
        for(uint32_t pass=0;pass<passes;++pass){auto cb=begin();vkCmdBeginQuery(cb,query,0,0);if(!empty)dispatch(cb);vkCmdEndQuery(cb,query,0);VkPerformanceQuerySubmitInfoKHR ps{VK_STRUCTURE_TYPE_PERFORMANCE_QUERY_SUBMIT_INFO_KHR};ps.counterPassIndex=pass;submit(cb,&ps);}
        std::vector<VkPerformanceCounterResultKHR> values(selected.size());check(vkGetQueryPoolResults(device,query,0,1,values.size()*sizeof(values[0]),values.data(),values.size()*sizeof(values[0]),0),"results");
        std::printf("{\"kind\":\"INDEPENDENT_NATIVE_KERNEL_COUNTERS_NOT_MODEL\",\"dtype\":\"%s\",\"K\":%u,\"M\":%u,\"activation\":\"F32\",\"rows_per_wg\":%u,\"wave\":32,\"trial\":%u,\"passes\":%u,\"empty\":%s,\"logical_weight_bytes\":%llu,\"counters\":{",q6?"Q6_K":"Q8_0",k,m,q6?2:1,trial,passes,empty?"true":"false",(unsigned long long)wb);
        for(size_t i=0;i<selected.size();++i){double v=0;switch(counters[selected[i]].storage){case VK_PERFORMANCE_COUNTER_STORAGE_FLOAT64_KHR:v=values[i].float64;break;case VK_PERFORMANCE_COUNTER_STORAGE_FLOAT32_KHR:v=values[i].float32;break;case VK_PERFORMANCE_COUNTER_STORAGE_UINT64_KHR:v=values[i].uint64;break;case VK_PERFORMANCE_COUNTER_STORAGE_UINT32_KHR:v=values[i].uint32;break;default:throw std::runtime_error("unsupported counter storage");}std::printf("%s\"%s\":%.9g",i?",":"",descriptions[selected[i]].name,v);}std::puts("}}");
    }
    release(device);
    auto read=begin();VkMemoryBarrier rb{VK_STRUCTURE_TYPE_MEMORY_BARRIER,nullptr,empty?VK_ACCESS_TRANSFER_WRITE_BIT:VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT};vkCmdPipelineBarrier(read,empty?VK_PIPELINE_STAGE_TRANSFER_BIT:VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&rb,0,nullptr,0,nullptr);VkBufferCopy oc{0,wb+ab,ob};vkCmdCopyBuffer(read,o.b,host.b,1,&oc);VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER,nullptr,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT};vkCmdPipelineBarrier(read,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&hb,0,nullptr,0,nullptr);submit(read);
    const float expected=empty?0:float(k)*(q6?-15:1);for(uint32_t i=0;i<m;++i)require(((float*)(raw+wb+ab))[i]==expected,"full output mismatch");
    std::printf("PASS all_outputs=%u expected=%g device=%s\n",m,expected,prop.deviceName);
    vkUnmapMemory(device,host.m);vkDestroyCommandPool(device,commands,nullptr);vkDestroyQueryPool(device,query,nullptr);vkDestroyPipeline(device,pipeline,nullptr);vkDestroyShaderModule(device,module,nullptr);vkDestroyPipelineLayout(device,layout,nullptr);vkDestroyDescriptorPool(device,pool,nullptr);vkDestroyDescriptorSetLayout(device,dsl,nullptr);
    for(auto b:{w,a,o,host}){vkDestroyBuffer(device,b.b,nullptr);vkFreeMemory(device,b.m,nullptr);}vkDestroyDevice(device,nullptr);vkDestroyInstance(instance,nullptr);return 0;
}catch(const std::exception & e){std::fprintf(stderr,"traffic probe stopped: %s\n",e.what());return 1;}
