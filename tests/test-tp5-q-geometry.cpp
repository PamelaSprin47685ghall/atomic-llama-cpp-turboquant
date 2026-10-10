#include "ggml-vulkan-tp5-q-geometry.h"
#include "ggml-vulkan-tp5-qcheck.h"
#include <cassert>
#include <cstdio>
#include <vector>

int main() {
    assert(vk_tp5_q_threads(nullptr)==256);
    for(const char * s:{"0","65","64x",""," 64"}) assert(!vk_tp5_q_threads(s));
    for(uint32_t threads:{64u,128u,256u}) {
        const uint32_t rows=threads/16u;
        for(uint32_t rank:{16u,320u,512u}) for(uint32_t tokens:{1u,2u,4u}) {
            const auto groups=vk_tp5_q_groups(rank,threads);assert(groups==rank/rows && groups<=128);
            std::vector<uint32_t> payload(tokens*4u*rank/2u);
            for(size_t i=0;i<payload.size();++i) payload[i]=uint32_t(i*0x9e3779b9u+17u);
            // vector storage is normally >=16-byte aligned on this target, but
            // use explicit alignment for the production metadata load contract.
            alignas(16) uint32_t entries[128*4]{};
            for(uint32_t g=0;g<groups;++g) {
                uint32_t x=0;
                for(uint32_t v=0;v<tokens*4u;++v) for(uint32_t j=0;j<rows/2u;++j)
                    x^=payload[v*rank/2u+g*rows/2u+j];
                entries[g*4]=x;entries[g*4+1]=77;entries[g*4+2]=payload.size();
            }
            assert(vk_tp5_qcheck_avx2(entries,payload.data(),true,77,payload.size(),rank,nullptr,rows));
            for(uint32_t g=0;g<groups;++g) {
                uint32_t pending=999;entries[g*4+1]^=1;
                assert(!vk_tp5_qcheck_avx2(entries,payload.data(),true,77,payload.size(),rank,&pending,rows)&&pending==g);
                entries[g*4+1]^=1;payload[g*rows/2u]^=1;
                assert(!vk_tp5_qcheck_avx2(entries,payload.data(),true,77,payload.size(),rank,&pending,rows)&&pending==g);
                payload[g*rows/2u]^=1;
            }
        }
    }
    assert(!vk_tp5_q_groups(513,64));assert(!vk_tp5_q_groups(319,64));
    std::puts("Q64/Q128/Q256 group ownership,1/2/4 rows,epoch and corruption rejection passed");
}
