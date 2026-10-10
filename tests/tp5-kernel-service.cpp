// Model-independent service-time experiment for actual Vulkan matvec kernels.
// Outputs alias one address so WAW hazards serialize the contractions; cold
// weights rotate over >128 MiB, resident control repeats one weight address.
// Host graph time is reported honestly, never called a hardware time lower bound.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include "ggml-impl.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) try {
    if (argc != 7) { std::fprintf(stderr,"usage: tp5-kernel-service TYPE K M BATCH WORKING_MiB resident|stream\n"); return 1; }
    const int type_id = std::stoi(argv[1]);
    const int64_t k = std::stoll(argv[2]), m = std::stoll(argv[3]), batches = std::stoll(argv[4]);
    const size_t mib = std::stoull(argv[5]);
    const bool resident = std::string(argv[6]) == "resident";
    if (type_id < 0 || type_id >= GGML_TYPE_COUNT || k < 32 || k > 16384 || m < 1 || m > 248320 ||
            batches < 1 || batches > 16 || mib < 160 || mib > 512 ||
            (!resident && std::string(argv[6]) != "stream")) throw std::runtime_error("invalid bounded input");
    const auto type = (ggml_type)type_id;
    if (k % ggml_blck_size(type)) throw std::runtime_error("block alignment");
    ggml_time_init();
    auto * reg = ggml_backend_vk_reg();
    if (ggml_backend_reg_dev_count(reg) < 1) throw std::runtime_error("no Vulkan GPU");
    const unsigned device_index=std::getenv("TP5_SERVICE_DEVICE")?std::stoul(std::getenv("TP5_SERVICE_DEVICE")):0u;
    if(device_index>=ggml_backend_reg_dev_count(reg)||device_index>4)throw std::runtime_error("physical device index out of range");
    auto * dev = ggml_backend_reg_dev_get(reg,device_index);
    if (std::strstr(ggml_backend_dev_description(dev),"Radeon RX 6800") == nullptr)
        throw std::runtime_error("not the qualified RX6800 micro target");
    ggml_backend_t backend = ggml_backend_dev_init(dev,nullptr);
    if (!backend) throw std::runtime_error("backend init failed");
    auto * buft = ggml_backend_get_default_buffer_type(backend);
    const size_t align = ggml_backend_buft_get_alignment(buft);
    const auto aligned = [align](size_t n) { return (n+align-1)/align*align; };
    const size_t rowbytes = ggml_row_size(type,k), weightbytes = rowbytes*m*batches;
    const size_t copies = std::max<size_t>(4,(mib*1024*1024+weightbytes-1)/weightbytes);
    if (copies > 2048 || copies*weightbytes > 1024ULL*1024*1024) throw std::runtime_error("working set outside micro budget");
    const size_t ctxbytes = ggml_tensor_overhead()*(2*copies+8)+ggml_graph_overhead_custom(4*copies+16,false);
    auto * ctx = ggml_init({ctxbytes,nullptr,true});
    if (!ctx) throw std::runtime_error("context allocation");
    auto * graph = ggml_new_graph_custom(ctx,4*copies+16,false);
    // ggml MUL_MAT broadcasts weights toward activation batches, not the
    // reverse. Each selected-expert microbatch needs its own activation row.
    auto * input = ggml_new_tensor_3d(ctx,GGML_TYPE_F32,k,1,batches);
    std::vector<ggml_tensor *> weights, outputs;
    for (size_t i=0; i<copies; ++i) {
        auto * w=ggml_new_tensor_3d(ctx,type,k,m,batches);
        auto * o=ggml_mul_mat(ctx,w,input);
        ggml_set_output(o);
        ggml_build_forward_expand(graph,o);
        weights.push_back(w);outputs.push_back(o);
    }
    const size_t weightalloc=aligned(ggml_backend_buft_get_alloc_size(buft,weights[0]));
    const size_t inalloc=aligned(ggml_backend_buft_get_alloc_size(buft,input));
    const size_t outalloc=aligned(ggml_backend_buft_get_alloc_size(buft,outputs[0]));
    const size_t weightsets=resident?1:copies;
    ggml_backend_buffer_t buffer=ggml_backend_buft_alloc_buffer(buft,weightsets*weightalloc+inalloc+outalloc+align);
    if (!buffer) throw std::runtime_error("device allocation");
    char * base=(char *)ggml_backend_buffer_get_base(buffer);
    const auto allocate = [&](ggml_tensor * t,size_t offset) {
        if (ggml_backend_tensor_alloc(buffer,t,base+offset) != GGML_STATUS_SUCCESS) throw std::runtime_error("tensor allocation");
    };
    for (size_t i=0;i<copies;++i) allocate(weights[i],(resident?0:i)*weightalloc);
    allocate(input,weightsets*weightalloc);
    for (auto * o:outputs) allocate(o,weightsets*weightalloc+inalloc);
    std::vector<float> floats(k*m*batches), activation(k*batches);
    uint32_t seed=12345;
    const auto random = [&]() { seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return (float(int(seed%20001)-10000))/10000; };
    for (auto & x:floats) x=random();
    for (auto & x:activation) x=random();
    std::vector<uint8_t> quant(weightbytes);
    if (type==GGML_TYPE_F32) std::memcpy(quant.data(),floats.data(),weightbytes);
    else {
        std::vector<float> imatrix(k,1.0f);
        const size_t wrote=ggml_quantize_chunk(type,floats.data(),quant.data(),0,m*batches,k,imatrix.data());
        if (wrote!=weightbytes) throw std::runtime_error("quantizer extent mismatch");
    }
    for (size_t i=0;i<weightsets;++i) ggml_backend_tensor_set(weights[i],quant.data(),0,weightbytes);
    ggml_backend_tensor_set(input,activation.data(),0,activation.size()*sizeof(float));
    if (!ggml_backend_supports_op(backend,outputs[0])) throw std::runtime_error("kernel unsupported");
    const unsigned warm_ms=std::getenv("TP5_SERVICE_WARM_MS")?std::stoul(std::getenv("TP5_SERVICE_WARM_MS")):0u;
    if(warm_ms>1000)throw std::runtime_error("warmup must be<=1000ms");
    if(warm_ms){
        const int64_t started=ggml_time_us();size_t warmed=0;
        do{
            if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("warmup compute failed");
            ggml_backend_synchronize(backend);warmed+=copies;
        }while(ggml_time_us()-started<int64_t(warm_ms)*1000);
        std::fprintf(stderr,"independent_native_warmup device=%u requested_ms=%u actual_ms=%.3f operations=%zu clock_writes=0\n",device_index,warm_ms,double(ggml_time_us()-started)/1000,warmed);
    }
    std::vector<double> us;
    for (int repeat=0;repeat<9;++repeat) {
        ggml_backend_synchronize(backend);
        const int64_t begin=ggml_time_us();
        const auto status=ggml_backend_graph_compute(backend,graph);
        ggml_backend_synchronize(backend);
        const double elapsed=double(ggml_time_us()-begin);
        if (status!=GGML_STATUS_SUCCESS) throw std::runtime_error("graph compute failed");
        if (repeat>=2) us.push_back(elapsed/copies);
    }
    std::vector<float> out(m*batches), decoded(k);
    ggml_backend_tensor_get(outputs.back(),out.data(),0,out.size()*sizeof(float));
    double worst=0;
    for (int64_t b=0;b<batches;++b) for (int64_t row : {int64_t(0),m/2,m-1}) {
        const auto * ptr=quant.data()+(b*m+row)*rowbytes;
        if (type==GGML_TYPE_F32) std::memcpy(decoded.data(),ptr,k*sizeof(float));
        else ggml_get_type_traits(type)->to_float(ptr,decoded.data(),k);
        double expected=0;for(int64_t j=0;j<k;++j)expected+=double(decoded[j])*activation[b*k+j];
        const double error=std::abs(out[b*m+row]-expected)/std::max(1.0,std::abs(expected));
        if (!std::isfinite(error) || error>2e-4) throw std::runtime_error("selected-row CPU reference mismatch");
        worst=std::max(worst,error);
    }
    std::sort(us.begin(),us.end());
    std::printf("{\"kind\":\"INDEPENDENT_SERIALIZED_KERNEL_SERVICE_NOT_HARDWARE_BOUND\",\"type\":\"%s\",\"K\":%lld,\"M\":%lld,\"batches\":%lld,\"mode\":\"%s\",\"copies\":%zu,\"weight_bytes_per_op\":%zu,\"working_weight_bytes\":%zu,\"median_us_per_op\":%.6f,\"min_us_per_op\":%.6f,\"max_reference_rel_error\":%.9g,\"samples_us\":[",
        ggml_type_name(type),(long long)k,(long long)m,(long long)batches,resident?"resident":"stream",copies,weightbytes,weightsets*weightalloc,us[3],us[0],worst);
    for(size_t i=0;i<us.size();++i)std::printf("%s%.6f",i?",":"",us[i]);
    std::puts("]}");
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
    return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr,"micro refused/failed: %s\n",e.what());return 1;
}
