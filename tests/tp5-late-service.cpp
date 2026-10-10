// Model-independent production LateBind shader calibration, timestamped on GPU.
// Fixed numerical fixtures, all output elements and Q publication metadata checked.
// Cached local versus coherent system-memory Q aperture is an explicit factor.
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#ifdef TP5_MOE_SERVICE
#include "ggml.h"
#include "ggml-impl.h"
#endif

static void ck(VkResult r,const char * s){if(r!=VK_SUCCESS)throw std::runtime_error(std::string(s)+":"+std::to_string(r));}
static void require(bool b,const char *s){if(!b)throw std::runtime_error(s);}
static uint32_t bits(float v){uint32_t x;std::memcpy(&x,&v,4);return x;}
struct Buf{
    VkDevice device{};VkBuffer buffer{};VkDeviceMemory memory{};void * mapped{};size_t bytes;
    Buf(VkDevice d,const VkPhysicalDeviceMemoryProperties &mp,size_t size,bool host,bool coherent_vram=false):device(d),bytes(size){
        VkBufferCreateInfo bc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bc.size=size;bc.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;bc.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        ck(vkCreateBuffer(d,&bc,nullptr,&buffer),"buffer");VkMemoryRequirements req{};vkGetBufferMemoryRequirements(d,buffer,&req);
        uint32_t mt=UINT32_MAX;
        const VkMemoryPropertyFlags flags=coherent_vram?
            (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT|VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD):
            host?(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT):VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        for(uint32_t i=0;i<mp.memoryTypeCount;++i){const auto f=mp.memoryTypes[i].propertyFlags;
            if(!(req.memoryTypeBits&(1u<<i))||(f&flags)!=flags||(!coherent_vram&&(f&(VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD|VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD))))continue;
            if(!host&&!coherent_vram&&(f&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))continue;
            if(host&&!coherent_vram&&(f&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))continue;
            mt=i;break;}
        require(mt!=UINT32_MAX,"ordinary VRAM/system-RAM type unavailable");
        std::fprintf(stderr,"late_micro_buffer bytes=%zu host=%d memory_type=%u flags=0x%x\n",size,host?1:0,mt,mp.memoryTypes[mt].propertyFlags);
        VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ma.allocationSize=req.size;ma.memoryTypeIndex=mt;ck(vkAllocateMemory(d,&ma,nullptr,&memory),"memory");ck(vkBindBufferMemory(d,buffer,memory,0),"bind");
        if(host||coherent_vram)ck(vkMapMemory(d,memory,0,VK_WHOLE_SIZE,0,&mapped),"map");
    }
    Buf(const Buf&)=delete;
    ~Buf(){if(mapped)vkUnmapMemory(device,memory);if(buffer)vkDestroyBuffer(device,buffer,nullptr);if(memory)vkFreeMemory(device,memory,nullptr);}
};

int main(int argc,char**argv)try{
    require(argc>=5&&argc<=7,"usage: tp5-late-service shader-directory inject|act|q|up|norm|lo local|system device-index [Q-workgroup-size [active-rows]]");
    const std::string kind=argv[2],domain=argv[3];const unsigned device_index=std::stoul(argv[4]);
    const bool moe=kind=="moe2"||kind=="moe3";
    const uint32_t weight_sets=moe&&std::getenv("TP5_TEST_STREAM_WEIGHTS")&&std::string(std::getenv("TP5_TEST_STREAM_WEIGHTS"))=="1"?64u:1u;
    const unsigned q_workgroup = argc>=6 ? std::stoul(argv[5]) : 256u;
    const unsigned active = argc==7 ? std::stoul(argv[6]) : 1u;
    const uint32_t prefetch = std::getenv("TP5_TEST_NORM_PREFETCH") ? std::stoul(std::getenv("TP5_TEST_NORM_PREFETCH")) : 0u;
    const bool withhold = std::getenv("TP5_TEST_NORM_WITHHOLD") && std::string(std::getenv("TP5_TEST_NORM_WITHHOLD"))=="1";
    const bool batch_timestamps = kind=="tick" || (std::getenv("TP5_TEST_BATCH_TIMING") && std::string(std::getenv("TP5_TEST_BATCH_TIMING"))=="1");
    const unsigned warm_ms=std::getenv("TP5_TEST_WARM_MS")?std::stoul(std::getenv("TP5_TEST_WARM_MS")):0u;
    require(warm_ms<=1000&&(!warm_ms||(kind!="tick"&&!withhold)),"bounded non-fault warmup only");
    require(prefetch<=1&&((!prefetch&&!withhold)||kind=="norm"),"norm-only bounded prefetch/withhold test");
    require(active>=1&&active<=4&&(kind=="q"||kind=="norm"||active==1),"active-row fixture bound");
    require((kind=="norm"&&(q_workgroup==256||q_workgroup==512||q_workgroup==1024))||
            (kind!="norm"&&(q_workgroup==64||q_workgroup==128||q_workgroup==256)),"invalid bounded workgroup");
    require(kind=="q"||kind=="norm"||q_workgroup==256,"workgroup selection is Q/norm-only");
    require((kind=="inject"||kind=="act"||kind=="q"||kind=="up"||kind=="norm"||kind=="lo"||kind=="tick"||kind=="packnoop"||kind=="relayconsume"||moe)&&(domain=="local"||domain=="system"||domain=="production")&&device_index<5,"invalid bounded case");
    require(domain=="local"||kind=="q"||kind=="norm"||kind=="lo"||kind=="relayconsume","system aperture applies to mailbox cases");
    constexpr uint32_t width=2560,streams=4,rank=320,blocks=80;
    const char * command_count=std::getenv("TP5_COMMANDS");
    const uint32_t iterations=kind=="tick"?(command_count?std::stoul(command_count):1510u):64u;
    require(iterations>=16&&iterations<=4096,"bounded command calibration");
    std::vector<std::vector<uint32_t>> data;std::vector<uint32_t> pc;std::vector<unsigned> outputs;
    std::vector<std::vector<float>> moe_expected;size_t moe_weight_bytes=0;
    uint32_t groups=0;std::string shader;std::vector<uint32_t> expected_q;std::vector<std::vector<float>> expected_norm;
    auto vec=[](size_t n,float value){return std::vector<uint32_t>(n,bits(value));};
    const auto packed=[](size_t n,bool weight){std::vector<uint32_t> v((n+(weight?1:0))*9,0);for(size_t b=weight?1:0;b<n+(weight?1:0);++b){v[b*9]=bits(1.0f);for(unsigned j=1;j<9;++j)v[b*9+j]=0x01010101u;}if(weight)v[0]=1;return v;};
    const auto header=[](){std::vector<uint32_t>v(16,0);v[5]=1;v[6]=17;return v;};
    if(moe){
#ifdef TP5_MOE_SERVICE
        require(domain=="local"&&active==1,"MoE fixture uses local weights, single token");
        constexpr unsigned selected=10,expert_k=128,shared_k=128;
        shader=kind=="moe2"?"qwen4_moe_projections_iq2s_iq3xxs.spv":"qwen4_moe_projections_iq3xxs_iq3s.spv";
        groups=selected*expert_k/4+shared_k/2+1;pc={width,expert_k,shared_k,selected};
        const ggml_type types[4]={kind=="moe2"?GGML_TYPE_IQ2_S:GGML_TYPE_IQ3_XXS,
                                  kind=="moe2"?GGML_TYPE_IQ3_XXS:GGML_TYPE_IQ3_S,GGML_TYPE_Q6_K,GGML_TYPE_Q6_K};
        uint32_t seed=137;const auto random=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return float(int(seed%20001)-10000)/10000.f;};
        std::vector<float> activation(width),scalar(width);
        for(auto &x:activation)x=random();for(auto &x:scalar)x=random()*.02f;
        data.resize(10);std::vector<std::vector<double>> dots(4);
        for(unsigned w=0;w<4;++w){
            const size_t rows=w<2?selected*expert_k:shared_k;
            std::vector<float> f(rows*width),imatrix(width,1.0f),decoded(width);
            for(auto &x:f)x=random()*.02f;
            const size_t bytes=ggml_row_size(types[w],width)*rows;
            data[w].resize((bytes+3)/4);
            require(ggml_quantize_chunk(types[w],f.data(),data[w].data(),0,rows,width,imatrix.data())==bytes,"MoE quantized extent mismatch");
            moe_weight_bytes+=bytes;dots[w].resize(rows);
            for(size_t row=0;row<rows;++row){
                ggml_get_type_traits(types[w])->to_float((const uint8_t*)data[w].data()+row*ggml_row_size(types[w],width),decoded.data(),width);
                double sum=0;for(unsigned j=0;j<width;++j)sum+=double(decoded[j])*activation[j];dots[w][row]=sum;
            }
        }
        data[4].resize(width);std::memcpy(data[4].data(),activation.data(),width*4);
        data[5].resize(selected);for(unsigned i=0;i<selected;++i)data[5][i]=selected-1-i;
        data[6].resize(width);std::memcpy(data[6].data(),scalar.data(),width*4);moe_weight_bytes+=width*4;
        data[7]=vec(selected*expert_k,-17);data[8]=vec(shared_k,-17);data[9]=vec(1,-17);outputs={7,8,9};
        moe_expected.resize(10);moe_expected[7].resize(selected*expert_k);moe_expected[8].resize(shared_k);moe_expected[9].resize(1);
        for(unsigned slot=0;slot<selected;++slot)for(unsigned row=0;row<expert_k;++row){const size_t src=(selected-1-slot)*expert_k+row;const double g=dots[0][src];moe_expected[7][slot*expert_k+row]=float((g/(1.0+std::exp(-g)))*dots[1][src]);}
        for(unsigned row=0;row<shared_k;++row){const double g=dots[2][row];moe_expected[8][row]=float((g/(1.0+std::exp(-g)))*dots[3][row]);}
        double scalar_sum=0;for(unsigned i=0;i<width;++i)scalar_sum+=double(scalar[i])*activation[i];moe_expected[9][0]=float(scalar_sum);
#else
        throw std::runtime_error("Compile with TP5_MOE_SERVICE and ggml-base for MoE fixtures");
#endif
    }else if(kind=="relayconsume"){
        require(domain=="production","relay consumer must use actual coherent VRAM inbox and system status");
        shader="tp5_relay.spv";groups=1;pc={width,17,0,10000,1,0};
        auto inbox=vec(16+width,0);inbox[0]=17;
        for(unsigned e=0;e<width;++e)inbox[16+e]=bits(float(e%7)*.03125f);
        data={inbox,vec(width,-17),vec(16,0),vec(128,0)};outputs={1,3};
    }else if(kind=="packnoop"){
        const bool serial=std::getenv("TP5_TEST_PACK_SINGLE_WG")&&std::string(std::getenv("TP5_TEST_PACK_SINGLE_WG"))=="1";
        shader=serial?"tp5_hc_late_pack_single.spv":"tp5_hc_late_pack.spv";const unsigned blocks=width*streams*rank/32;
        std::vector<uint32_t> raw((size_t(blocks)*34+3)/4,0);auto * bytes=(uint8_t*)raw.data();
        for(unsigned b=0;b<blocks;++b){bytes[b*34]=0;bytes[b*34+1]=0x3c;std::memset(bytes+b*34+2,1,32);}
        data={raw,vec((blocks+1)*9,-17)};data[1][0]=1;
        pc={blocks};groups=serial?1u:(blocks+255)/256;outputs={1};
    }else if(kind=="inject"){
        shader="tp5_hc_late_inject.spv";groups=4;pc={width,streams,1};
        std::vector<uint32_t> w((streams*streams*width/32*34+3)/4,0);auto *p=(uint8_t*)w.data();
        for(size_t b=0;b<streams*streams*width/32;++b){p[b*34]=0;p[b*34+1]=0x3c;std::memset(p+b*34+2,1,32);}
        data={w,vec(streams*width,1),vec(4,0),header()};outputs={2};
    }else if(kind=="act"){
        shader="tp5_hc_late_act_q8.spv";groups=40;pc={width,streams,1,1};
        data={vec(streams*width,1),vec(streams*width,0),vec(width,127),vec(streams,1),std::vector<uint32_t>(streams*blocks*9),header()};outputs={4};
    }else if(kind=="q"){
        shader="tp5_hc_late_q_q8dot.spv";groups=rank/(q_workgroup/16u);pc={width,rank,streams,q_workgroup/16u,groups,16,1};
        // Distinct token/stream/row/block values exercise all indexing, not just
        // an all-ones payload that could hide stride or capacity mistakes.
        std::vector<uint32_t> mailbox(16+4*streams*rank/2,0xdeadbeefu);std::fill(mailbox.begin(),mailbox.begin()+16,0);mailbox[2]=active;
        auto weights=packed(rank*streams*blocks,true), acts=packed(4*streams*blocks,false);
        const auto byteword=[](int x){const uint32_t b=uint32_t(x)&255u;return b*0x01010101u;};
        for(uint32_t row=0;row<rank;++row)for(uint32_t s=0;s<streams;++s)for(uint32_t b=0;b<blocks;++b){
            const size_t off=((row*streams+s)*blocks+b+1)*9;weights[off]=bits((b&1)?0.5f:1.0f);
            for(uint32_t w=0;w<8;++w)weights[off+1+w]=byteword(int((row*5+s*3+b*7+w*11)%5)-2);
        }
        for(uint32_t t=0;t<4;++t)for(uint32_t s=0;s<streams;++s)for(uint32_t b=0;b<blocks;++b){
            const size_t off=((t*streams+s)*blocks+b)*9;acts[off]=bits(1.0f);
            for(uint32_t w=0;w<8;++w)acts[off+1+w]=byteword(int((t+s+b+w)%7)-3);
        }
        expected_q.assign(4*streams*rank/2,0xdeadbeefu);
        for(uint32_t t=0;t<active;++t)for(uint32_t s=0;s<streams;++s)for(uint32_t row=0;row<rank;++row){
            float value=0;
            for(uint32_t b=0;b<blocks;++b){int isum=0;for(uint32_t w=0;w<8;++w)isum+=4*(int((row*5+s*3+b*7+w*11)%5)-2)*(int((t+s+b+w)%7)-3);value+=float(isum)*((b&1)?0.5f:1.0f);}
            const _Float16 h=(_Float16)value;uint16_t hb;std::memcpy(&hb,&h,2);const uint32_t index=(t*streams+s)*rank+row;
            if(!(index&1))expected_q[index/2]=hb;else expected_q[index/2]|=uint32_t(hb)<<16;
        }
        auto hdr=header();hdr[5]=active;
        data={weights,acts,mailbox,hdr,std::vector<uint32_t>(groups*4)};outputs={2,4};
    }else if(kind=="norm"){
        const unsigned staged=std::getenv("TP5_TEST_NORM_STAGE_Y")?std::stoul(std::getenv("TP5_TEST_NORM_STAGE_Y")):0;
        require(staged<=2,"norm staging variant0..2");
        shader=staged==2?"tp5_hc_norm_shared.spv":staged==1?"tp5_hc_resume_norm_stage_y.spv":"tp5_hc_resume_norm.spv";
        groups=staged==2?1:4;pc={width,streams,bits(1e-5f),10000,0,0,4};
        auto inbox=vec(16+4*width,0);inbox[0]=withhold?0:17;inbox[5]=active;
        std::vector<uint32_t> st(16,0);st[1]=17;
        data={inbox,vec(4*streams*width,0),vec(streams*width,0),vec(4*streams*width,-17),vec(4*streams*width,-17),vec(4*streams,0),vec(4*streams,-17),st,vec(4*width,-17)};outputs={3,4,6,7,8};
        expected_norm.resize(9);
        for(unsigned o:{3u,4u,6u,8u})expected_norm[o].assign(data[o].size(),-17);
        for(uint32_t c=0;c<streams*width;++c)data[2][c]=bits(float(int(c%31)-15)/16);
        for(uint32_t t=0;t<active;++t){
            for(uint32_t c=0;c<width;++c){const float y=float(int((c+7*t)%47)-23)/8;data[0][16+t*width+c]=bits(y);expected_norm[8][t*width+c]=y;}
            for(uint32_t s=0;s<streams;++s){
                const float inject=(s+1)/8.0f;data[5][t*streams+s]=bits(inject);double squares=0;
                for(uint32_t c=0;c<width;++c){const size_t i=(t*streams+s)*width+c;const float r=float(int((c*7+s*11+t*13)%61)-30)/16;data[1][i]=bits(r);const float x=r+inject*expected_norm[8][t*width+c];expected_norm[3][i]=x;squares+=double(x)*x;}
                const float rho=float(1/std::sqrt(squares/width+double(1e-5f)));expected_norm[6][t*streams+s]=rho;
                for(uint32_t c=0;c<width;++c){const size_t i=(t*streams+s)*width+c;float g;const uint32_t gb=data[2][s*width+c];std::memcpy(&g,&gb,4);expected_norm[4][i]=expected_norm[3][i]*rho*g;}
            }
        }
    }else if(kind=="lo"){
        shader="tp5_hc_resume_lo_q8.spv";groups=1;pc={rank,streams,16,10000,0,0,0,1};
        std::vector<uint32_t> inbox(16+streams*rank,0);inbox[2]=17;inbox[5]=1;
        std::vector<uint32_t> st(16,0);st[1]=17;
        data={inbox,vec(streams,1),vec(rank,0),st,std::vector<uint32_t>(10*9,0)};outputs={2,3,4};
    }else if(kind=="tick"){
        shader="tp5_command_tick.spv";groups=1;pc={1};data={std::vector<uint32_t>(1,0)};outputs={0};
    }else{
        shader="tp5_hc_late_up_q8dot.spv";groups=80;pc={width,rank,streams,32,1};
        data={packed(streams*width*10,true),packed(10,false),vec(streams*width,1),vec(width,0),header()};outputs={3};
    }
    std::ifstream in(std::string(argv[1])+"/"+shader,std::ios::binary|std::ios::ate);require(in.good(),"missing production shader");auto length=in.tellg();require(length>0&&length%4==0,"invalid SPIRV");std::vector<uint32_t>spv(size_t(length)/4);in.seekg(0);in.read((char*)spv.data(),length);require(in.good(),"shader read");
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.apiVersion=VK_API_VERSION_1_3;app.pApplicationName="tp5-independent-late-service";
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ici.pApplicationInfo=&app;VkInstance instance{};ck(vkCreateInstance(&ici,nullptr,&instance),"instance");uint32_t nd=0;ck(vkEnumeratePhysicalDevices(instance,&nd,nullptr),"count");std::vector<VkPhysicalDevice> all(nd);ck(vkEnumeratePhysicalDevices(instance,&nd,all.data()),"devices");
    std::vector<VkPhysicalDevice> devices;for(auto d:all){VkPhysicalDeviceProperties p{};vkGetPhysicalDeviceProperties(d,&p);if(p.vendorID==0x1002&&p.deviceID==0x73bf)devices.push_back(d);}require(devices.size()==5,"five physical Navi21 required");const auto physical=devices[device_index];
    VkPhysicalDeviceProperties props{};vkGetPhysicalDeviceProperties(physical,&props);uint32_t nq=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,nullptr);std::vector<VkQueueFamilyProperties> qp(nq);vkGetPhysicalDeviceQueueFamilyProperties(physical,&nq,qp.data());uint32_t qf=UINT32_MAX;for(uint32_t i=0;i<nq;++i)if((qp[i].queueFlags&VK_QUEUE_GRAPHICS_BIT)&&qp[i].timestampValidBits){qf=i;break;}require(qf!=UINT32_MAX,"timestamped graphics/compute queue required");
    VkPhysicalDeviceCoherentMemoryFeaturesAMD coherent{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};f13.pNext=domain=="production"?&coherent:nullptr;VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};f12.pNext=&f13;VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};f11.pNext=&f12;VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};f2.pNext=&f11;vkGetPhysicalDeviceFeatures2(physical,&f2);
    require(f13.subgroupSizeControl&&f13.shaderIntegerDotProduct&&f12.shaderFloat16&&f2.features.shaderInt16,"shader features unavailable");
    const char * coherent_ext=VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME;
    require(domain!="production"||coherent.deviceCoherentMemory,"production coherent-memory feature unavailable");
    float priority=1;VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qci.queueFamilyIndex=qf;qci.queueCount=1;qci.pQueuePriorities=&priority;VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};dci.pNext=&f2;dci.queueCreateInfoCount=1;dci.pQueueCreateInfos=&qci;
    if(domain=="production"){dci.enabledExtensionCount=1;dci.ppEnabledExtensionNames=&coherent_ext;}
    VkDevice device{};ck(vkCreateDevice(physical,&dci,nullptr,&device),"device");VkQueue queue{};vkGetDeviceQueue(device,qf,0,&queue);VkPhysicalDeviceMemoryProperties mp{};vkGetPhysicalDeviceMemoryProperties(physical,&mp);
    {
        std::vector<size_t> offsets;size_t total=0;for(auto&v:data){offsets.push_back(total);total+=(v.size()*4+63)/64*64;}
        Buf host(device,mp,total,true);auto *mapped=(uint8_t*)host.mapped;std::memset(mapped,0,total);std::vector<std::unique_ptr<Buf>> buffers;
        for(size_t i=0;i<data.size();++i){std::memcpy(mapped+offsets[i],data[i].data(),data[i].size()*4);
            const bool prod=domain=="production";
            const bool sys=(domain=="system"||prod)&&((kind=="q"&&(i==2||i==4))||(kind=="norm"&&(i==7||(!prod&&i==0)))||(kind=="lo"&&(i==3||(!prod&&i==0)))||(kind=="relayconsume"&&i==3));
            const bool cv=prod&&((kind=="norm"||kind=="lo"||kind=="relayconsume")&&i==0);
            buffers.emplace_back(new Buf(device,mp,data[i].size()*4*(moe&&i<4?weight_sets:1u),sys,cv));}
        const uint32_t nb=data.size();std::vector<VkDescriptorSetLayoutBinding> bs(nb);for(uint32_t i=0;i<nb;++i)bs[i]={i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};VkDescriptorSetLayoutCreateInfo slci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};slci.bindingCount=nb;slci.pBindings=bs.data();VkDescriptorSetLayout sl{};ck(vkCreateDescriptorSetLayout(device,&slci,nullptr,&sl),"setlayout");VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,uint32_t(pc.size()*4)};VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};plci.setLayoutCount=1;plci.pSetLayouts=&sl;plci.pushConstantRangeCount=1;plci.pPushConstantRanges=&range;VkPipelineLayout pl{};ck(vkCreatePipelineLayout(device,&plci,nullptr,&pl),"layout");
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};smci.codeSize=spv.size()*4;smci.pCode=spv.data();VkShaderModule sm{};ck(vkCreateShaderModule(device,&smci,nullptr,&sm),"module");VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sg{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};sg.requiredSubgroupSize=32;
        const VkSpecializationMapEntry qe[2]={{0u,0u,sizeof(uint32_t)},{1u,sizeof(uint32_t),sizeof(uint32_t)}};
        const uint32_t qdata[2]={q_workgroup,prefetch};
        const VkSpecializationInfo qspec{kind=="norm"?2u:1u,qe,kind=="norm"?2*sizeof(uint32_t):sizeof(uint32_t),qdata};
        const VkSpecializationMapEntry me[3]={{0,0,4},{1,4,4},{2,8,4}};const uint32_t md[3]={32,4,2};const VkSpecializationInfo mspec{3,me,12,md};
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cpci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,kind=="relayconsume"?nullptr:&sg,0,VK_SHADER_STAGE_COMPUTE_BIT,sm,"main",moe?&mspec:(kind=="q"||kind=="norm")?&qspec:nullptr};cpci.layout=pl;VkPipeline pipeline{};ck(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&cpci,nullptr,&pipeline),"pipeline");
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,nb};VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dpci.maxSets=1;dpci.poolSizeCount=1;dpci.pPoolSizes=&ps;VkDescriptorPool dp{};ck(vkCreateDescriptorPool(device,&dpci,nullptr,&dp),"descriptorpool");VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};dai.descriptorPool=dp;dai.descriptorSetCount=1;dai.pSetLayouts=&sl;VkDescriptorSet ds{};ck(vkAllocateDescriptorSets(device,&dai,&ds),"set");
        std::vector<VkDescriptorBufferInfo> bi(nb);std::vector<VkWriteDescriptorSet> ws(nb);for(uint32_t i=0;i<nb;++i){bi[i]={buffers[i]->buffer,0,data[i].size()*4};ws[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,nullptr,ds,i,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,nullptr,&bi[i],nullptr};}vkUpdateDescriptorSets(device,nb,ws.data(),0,nullptr);
        if(kind=="relayconsume"){bi[2]={buffers[0]->buffer,0,64};vkUpdateDescriptorSets(device,nb,ws.data(),0,nullptr);}
        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cpi.queueFamilyIndex=qf;VkCommandPool pool{};ck(vkCreateCommandPool(device,&cpi,nullptr,&pool),"commandpool");VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};qpi.queryType=VK_QUERY_TYPE_TIMESTAMP;qpi.queryCount=iterations*2;VkQueryPool queries{};ck(vkCreateQueryPool(device,&qpi,nullptr,&queries),"queries");
        const auto begin=[&](){VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};a.commandPool=pool;a.commandBufferCount=1;a.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;VkCommandBuffer cmd{};ck(vkAllocateCommandBuffers(device,&a,&cmd),"cb");VkCommandBufferBeginInfo b{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};ck(vkBeginCommandBuffer(cmd,&b),"begin");return cmd;};
        const auto submit=[&](VkCommandBuffer cb){ck(vkEndCommandBuffer(cb),"end");VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};VkFence f{};ck(vkCreateFence(device,&fci,nullptr,&f),"fence");VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&cb;ck(vkQueueSubmit(queue,1,&si,f),"submit");ck(vkWaitForFences(device,1,&f,VK_TRUE,3000000000ULL),"bounded wait");vkDestroyFence(device,f,nullptr);};
        const auto barrier=[](VkCommandBuffer cb,VkPipelineStageFlags src,VkPipelineStageFlags dst,VkAccessFlags sa,VkAccessFlags da){VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER,nullptr,sa,da};vkCmdPipelineBarrier(cb,src,dst,0,1,&b,0,nullptr,0,nullptr);};
        std::vector<VkDescriptorSet> extra_sets;
        VkDescriptorPool stream_pool=VK_NULL_HANDLE;
        if(weight_sets>1){
            VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,nb*weight_sets};VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};info.maxSets=weight_sets;info.poolSizeCount=1;info.pPoolSizes=&size;ck(vkCreateDescriptorPool(device,&info,nullptr,&stream_pool),"stream descriptor pool");
            std::vector<VkDescriptorSetLayout> layouts(weight_sets,sl);extra_sets.resize(weight_sets);VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};alloc.descriptorPool=stream_pool;alloc.descriptorSetCount=weight_sets;alloc.pSetLayouts=layouts.data();ck(vkAllocateDescriptorSets(device,&alloc,extra_sets.data()),"stream descriptors");
            for(uint32_t set=0;set<weight_sets;++set){for(uint32_t i=0;i<nb;++i){bi[i].offset=i<4?set*data[i].size()*4:0;ws[i].dstSet=extra_sets[set];}vkUpdateDescriptorSets(device,nb,ws.data(),0,nullptr);}
        }
        auto prep=begin();for(uint32_t i=0;i<nb;++i){for(uint32_t set=0;set<(moe&&i<4?weight_sets:1u);++set){VkBufferCopy c{offsets[i],set*data[i].size()*4,data[i].size()*4};vkCmdCopyBuffer(prep,host.buffer,buffers[i]->buffer,1,&c);}}barrier(prep,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);submit(prep);
        if(warm_ms){
            const auto started=std::chrono::steady_clock::now();unsigned warm_dispatches=0;
            do{
                auto warm=begin();vkCmdBindPipeline(warm,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);vkCmdBindDescriptorSets(warm,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,nullptr);vkCmdPushConstants(warm,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,pc.size()*4,pc.data());
                for(unsigned j=0;j<1024;++j){
                    if(weight_sets>1)vkCmdBindDescriptorSets(warm,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&extra_sets[j%weight_sets],0,nullptr);
                    if(kind=="q"){barrier(warm,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);vkCmdFillBuffer(warm,buffers[2]->buffer,0,8,0);barrier(warm,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);}
                    else barrier(warm,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
                    vkCmdDispatch(warm,groups,1,1);
                }
                submit(warm);warm_dispatches+=1024;
            }while(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()<warm_ms);
            std::fprintf(stderr,"independent_workload_warmup requested_ms=%u actual_wall_ms=%.3f dispatches=%u no_clock_writes=1\n",warm_ms,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count(),warm_dispatches);
        }
        std::vector<double> timings;
        for(unsigned trial=0;trial<4;++trial){
            auto cb=begin();vkCmdResetQueryPool(cb,queries,0,iterations*2);vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,nullptr);vkCmdPushConstants(cb,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,pc.size()*4,pc.data());
            for(uint32_t j=0;j<iterations;++j){
                if(weight_sets>1)vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&extra_sets[j%weight_sets],0,nullptr);
                if(kind=="q"){
                    barrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);vkCmdFillBuffer(cb,buffers[2]->buffer,0,8,0);barrier(cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
                }else barrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
                if(!batch_timestamps||j==0)vkCmdWriteTimestamp(cb,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,queries,2*j);
                vkCmdDispatch(cb,groups,1,1);
                if(!batch_timestamps||j+1==iterations)vkCmdWriteTimestamp(cb,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,queries,2*j+1);
            }
            barrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);for(unsigned o:outputs){VkBufferCopy c{0,offsets[o],data[o].size()*4};vkCmdCopyBuffer(cb,buffers[o]->buffer,host.buffer,1,&c);}barrier(cb,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT);submit(cb);
            std::vector<uint64_t> ticks(iterations*2);const uint64_t mask=qp[qf].timestampValidBits==64?UINT64_MAX:((uint64_t(1)<<qp[qf].timestampValidBits)-1);
            if(batch_timestamps){
                ck(vkGetQueryPoolResults(device,queries,0,1,8,&ticks.front(),8,VK_QUERY_RESULT_64_BIT),"first timestamp");
                ck(vkGetQueryPoolResults(device,queries,iterations*2-1,1,8,&ticks.back(),8,VK_QUERY_RESULT_64_BIT),"last timestamp");
                if(trial)timings.push_back(double((ticks.back()-ticks.front())&mask)*props.limits.timestampPeriod/1000.0/iterations);
                if(kind=="tick")require(*(uint32_t*)(mapped+offsets[0])==(trial+1)*iterations,"every serial tick must execute");
            }else{
                ck(vkGetQueryPoolResults(device,queries,0,ticks.size(),ticks.size()*8,ticks.data(),8,VK_QUERY_RESULT_64_BIT),"timestamp results");
                if(trial)for(unsigned j=0;j<iterations;++j)timings.push_back(double((ticks[2*j+1]-ticks[2*j])&mask)*props.limits.timestampPeriod/1000.0);
            }
            if(kind=="inject"){const auto*x=(float*)(mapped+offsets[2]);for(unsigned i=0;i<4;++i)require(x[i]==2.0f,"inject full output mismatch");}
            if(kind=="act"){const auto*x=(uint32_t*)(mapped+offsets[4]);for(unsigned b=0;b<320;++b){require(x[b*9]==bits(1.0f),"ACT scale mismatch");for(unsigned j=1;j<9;++j)require(x[b*9+j]==0x7f7f7f7fu,"ACT byte mismatch");}}
            if(kind=="q"){
                const auto*x=(uint32_t*)(mapped+offsets[2]);require(x[0]==1&&x[1]==groups&&x[2]==active,"Q ready/count/extent mismatch");
                for(size_t j=0;j<expected_q.size();++j)require(x[16+j]==expected_q[j],"Q indexed live/inactive payload mismatch");
                const auto*e=(uint32_t*)(mapped+offsets[4]);const uint32_t words_per_wg=q_workgroup/32u;
                for(unsigned g=0;g<groups;++g){uint32_t v=0;for(unsigned ts=0;ts<active*streams;++ts)for(unsigned j=0;j<words_per_wg;++j)v^=expected_q[ts*rank/2+g*words_per_wg+j];require(e[4*g]==v&&e[4*g+1]==17&&e[4*g+2]==active*640,"Q checksum/epoch/live extent mismatch");}
            }
            if(kind=="norm"){
                for(unsigned o:{3u,4u,6u,8u}){const auto*x=(float*)(mapped+offsets[o]);
                    const size_t live=active*(o==6?streams:o==8?width:streams*width);
                    for(size_t j=0;j<data[o].size();++j){
                        if(withhold&&j<live)require(std::isnan(x[j]),"withheld norm must poison every live output");
                        else require(std::abs(x[j]-expected_norm[o][j])<=1e-6f*std::max(1e-4f,std::abs(expected_norm[o][j])),"norm live/inactive full output mismatch");
                    }
                }
                const auto*st=(uint32_t*)(mapped+offsets[7]);require(st[2]==(withhold?17u:0u)&&st[3]==0,"norm failure/retirement contract mismatch");
            }
            if(kind=="lo"){
                for(unsigned o:{2u,4u}){const auto*x=(uint32_t*)(mapped+offsets[o]);for(size_t j=0;j<data[o].size();++j)require(x[j]==0,"LO zero fixture mismatch");}
                const auto*st=(uint32_t*)(mapped+offsets[3]);require(st[2]==0&&st[3]==17,"LO bank retirement mismatch");
            }
            if(kind=="up"){const auto*x=(float*)(mapped+offsets[3]);for(unsigned i=0;i<width;++i)require(x[i]==1.0f,"UP full output mismatch");}
            if(kind=="packnoop")require(std::memcmp(mapped+offsets[1],data[1].data(),data[1].size()*4)==0,"completed pack must leave every payload/header word unchanged");
            if(kind=="relayconsume"){
                const auto*x=(float*)(mapped+offsets[1]);for(unsigned e=0;e<width;++e){
                    const float expected=float(e%7)*.03125f*.125f+.25f+float(e%7)*.03125f;
                    require(x[e]==expected,"ready relay consumer full state mismatch");
                }
                const auto*st=(uint32_t*)(mapped+offsets[3]);require(st[1]==17&&st[2]==0&&st[3]==17&&st[20]==0,"relay consumer generation/error/retirement mismatch");
            }
            if(moe){for(unsigned o:outputs){const auto*x=(float*)(mapped+offsets[o]);for(size_t i=0;i<moe_expected[o].size();++i){const float e=moe_expected[o][i];require(std::isfinite(x[i])&&std::abs(x[i]-e)<=2e-4f*std::max(1.f,std::abs(e)),"MoE full output CPU reference mismatch");}}}
        }
        std::sort(timings.begin(),timings.end());
        std::fprintf(stderr,"independent_timestamp_scope scheme=%s samples=%zu repeated_dispatches=%u reset_and_between_dispatch_barriers_included=%u\n",
            batch_timestamps?"two-per-chain":"two-per-dispatch",timings.size(),iterations,batch_timestamps?1u:0u);
        if(kind=="norm")std::fprintf(stderr,"norm_prefetch_test prefetch=%u withhold=%u active_rows=%u all_outputs_and_retirement_checked=1\n",prefetch,withhold?1u:0u,active);
        if(moe)std::fprintf(stderr,"{\"kind\":\"INDEPENDENT_FUSED_MOE_SERVICE\",\"family\":\"%s\",\"width\":2560,\"expert_k\":128,\"shared_k\":128,\"selected\":10,\"groups\":385,\"threads\":32,\"routed_rows_per_group\":4,\"shared_rows_per_group\":2,\"weight_bytes\":%zu,\"weight_sets\":%u,\"working_bytes\":%zu,\"median_us\":%.6f,\"min_us\":%.6f,\"all_outputs_checked\":true,\"timestamp_scheme\":\"%s\"}\n",kind.c_str(),moe_weight_bytes,weight_sets,(moe_weight_bytes-width*4)*weight_sets+width*4,timings[timings.size()/2],timings.front(),batch_timestamps?"chain":"dispatch");
        if(kind=="tick")std::printf("{\"kind\":\"MINIMAL_SERIAL_COMMAND_CALIBRATION\",\"commands\":%u,\"measured_chains\":%zu,\"min_us_per_command\":%.6f,\"median_us_per_command\":%.6f,\"max_us_per_command\":%.6f,\"counter_exact\":true,\"scope\":\"one-lane serial dependent dispatches; two GPU timestamps per chain; empirical service, not universal hardware lower bound\"}\n",iterations,timings.size(),timings.front(),timings[timings.size()/2],timings.back());
        else std::printf("{\"kind\":\"INDEPENDENT_PRODUCTION_LATE_SHADER_SERVICE\",\"shader\":\"%s\",\"aperture\":\"%s\",\"device\":%u,\"groups\":%u,\"threads\":%u,\"width\":2560,\"rows\":%u,\"samples\":%zu,\"min_us\":%.6f,\"median_us\":%.6f,\"p95_us\":%.6f,\"all_outputs_checked\":true,\"scope\":\"isolated fixture; actual memory domains in buffer log; no cross-device producer waits; timing scheme and weight sets stated separately; not hardware lower bound\"}\n",shader.c_str(),domain.c_str(),device_index,groups,kind=="relayconsume"?64:moe?32:q_workgroup,active,timings.size(),timings.front(),timings[timings.size()/2],timings[timings.size()*95/100]);
        vkDestroyQueryPool(device,queries,nullptr);vkDestroyCommandPool(device,pool,nullptr);if(stream_pool)vkDestroyDescriptorPool(device,stream_pool,nullptr);vkDestroyDescriptorPool(device,dp,nullptr);vkDestroyPipeline(device,pipeline,nullptr);vkDestroyShaderModule(device,sm,nullptr);vkDestroyPipelineLayout(device,pl,nullptr);vkDestroyDescriptorSetLayout(device,sl,nullptr);
    }
    vkDestroyDevice(device,nullptr);vkDestroyInstance(instance,nullptr);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"Late shader calibration stopped: %s\n",e.what());return 1;}
