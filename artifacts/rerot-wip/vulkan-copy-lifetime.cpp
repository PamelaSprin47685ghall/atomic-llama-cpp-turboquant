#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

int main(){
    ggml_backend_load_all();auto dev=ggml_backend_dev_by_name("Vulkan0");if(!dev)return 2;
    auto backend=ggml_backend_dev_init(dev,nullptr);auto host_type=ggml_backend_dev_host_buffer_type(dev);if(!backend||!host_type)return 2;
    std::printf("backend=%s host_type=%s\n",ggml_backend_name(backend),ggml_backend_buft_name(host_type));
    constexpr size_t count=4096, width=19, rows=11, device_stride=31, host_stride=29, offset=7;
    size_t checked=0;
    for(unsigned round=0;round<48;++round){
        auto ctx=ggml_init({ggml_tensor_overhead()*2,nullptr,true});auto*t=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,count);
        auto device_buffer=ggml_backend_alloc_buffer(backend,count*sizeof(uint32_t)+256);
        if(ggml_backend_tensor_alloc(device_buffer,t,ggml_backend_buffer_get_base(device_buffer))!=GGML_STATUS_SUCCESS)return 2;
        ggml_backend_tensor_memset(t,0,0,count*sizeof(uint32_t));
        auto source=ggml_backend_buft_alloc_buffer(host_type,count*sizeof(uint32_t));auto*src=(uint32_t*)ggml_backend_buffer_get_base(source);
        for(size_t j=0;j<count;++j)src[j]=(round+1)*0x10000+j;
        std::vector<uint32_t> expected(count,0);for(size_t r=0;r<rows;++r)for(size_t j=0;j<width;++j)expected[offset+r*device_stride+j]=src[r*host_stride+j];
        ggml_backend_tensor_set_2d_async(backend,t,src,offset*sizeof(uint32_t),width*sizeof(uint32_t),rows,device_stride*sizeof(uint32_t),host_stride*sizeof(uint32_t));
        // The queued copy must retain the pinned allocation after its wrapper
        // is released, and a newly allocated buffer must not alias its bytes.
        ggml_backend_buffer_free(source);
        auto churn=ggml_backend_buft_alloc_buffer(host_type,count*sizeof(uint32_t));std::memset(ggml_backend_buffer_get_base(churn),0xff,count*sizeof(uint32_t));
        ggml_backend_synchronize(backend);
        std::vector<uint32_t> got(count);ggml_backend_tensor_get(t,got.data(),0,count*sizeof(uint32_t));
        if(got!=expected){std::fprintf(stderr,"FAIL write lifetime round=%u\n",round);return 1;}checked+=count;
        auto destination=ggml_backend_buft_alloc_buffer(host_type,count*sizeof(uint32_t));auto*dst=(uint32_t*)ggml_backend_buffer_get_base(destination);std::fill(dst,dst+count,0xa5a5a5a5u);
        ggml_backend_tensor_get_2d_async(backend,t,dst,offset*sizeof(uint32_t),width*sizeof(uint32_t),rows,device_stride*sizeof(uint32_t),host_stride*sizeof(uint32_t));
        // Exercise the allocator dropping the source before async readback.
        ggml_backend_buffer_free(device_buffer);
        ggml_backend_synchronize(backend);
        for(size_t r=0;r<rows;++r)for(size_t j=0;j<width;++j){if(dst[r*host_stride+j]!=expected[offset+r*device_stride+j]){std::fprintf(stderr,"FAIL read lifetime round=%u row=%zu col=%zu\n",round,r,j);return 1;}++checked;}
        for(size_t r=0;r<rows-1;++r)for(size_t j=width;j<host_stride;++j)if(dst[r*host_stride+j]!=0xa5a5a5a5u)return 1;
        ggml_backend_buffer_free(destination);ggml_backend_buffer_free(churn);ggml_free(ctx);
    }
    std::printf("PASS async write/read retirement: 48 rounds, %zu exact integers checked, nonzero offsets and unequal strides\n",checked);
    ggml_backend_free(backend);
}
