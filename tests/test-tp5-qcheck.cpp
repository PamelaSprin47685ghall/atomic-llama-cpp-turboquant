#include "ggml-vulkan-tp5-qcheck.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

__attribute__((target("avx2"),noinline))
static uint32_t old_xor8(const uint32_t * p) {
    __m256i v=_mm256_loadu_si256((const __m256i*)p);
    __m128i x=_mm_xor_si128(_mm256_castsi256_si128(v),_mm256_extracti128_si256(v,1));
    x=_mm_xor_si128(x,_mm_shuffle_epi32(x,_MM_SHUFFLE(1,0,3,2)));
    x=_mm_xor_si128(x,_mm_shuffle_epi32(x,_MM_SHUFFLE(2,3,0,1)));
    return (uint32_t)_mm_cvtsi128_si32(x);
}
static bool reference(const uint32_t * e,const uint32_t * w,bool f16,uint32_t epoch,size_t n,uint32_t rank,uint32_t * pending) {
    const uint32_t groups=f16?rank/16:1;
    const size_t vector=f16?rank/2:n;
    if(!groups||!vector)return false;
    for(uint32_t g=0;g<groups;++g){
        std::atomic_thread_fence(std::memory_order_acquire);
        if(e[4*g+1]!=epoch||e[4*g+2]!=n){*pending=g;return false;}
        uint32_t x=0;
        for(size_t v=0;v<n/vector;++v){size_t i=0,step=f16?8:n;const uint32_t*p=w+v*vector+(f16?g*8:0);
            for(;i+8<=step;i+=8)x^=old_xor8(p+i);for(;i<step;++i)x^=p[i];}
        if(x!=e[4*g]){*pending=g;return false;}
    }
    return true;
}
int main(){
    if(!__builtin_cpu_supports("avx2")){std::puts("SKIP no AVX2");return 77;}
    std::mt19937 rng(734);size_t checks=0;
    alignas(64) uint32_t entries[4*32]{};
    std::vector<uint32_t>w;
    const auto initialize=[&](bool f16,uint32_t rank,size_t n){
        w.resize(n);for(auto &v:w)v=rng();std::fill(std::begin(entries),std::end(entries),0);
        const uint32_t groups=f16?rank/16:1;const size_t vector=f16?rank/2:n;
        for(uint32_t g=0;g<groups;++g){entries[4*g+1]=17;entries[4*g+2]=n;
            for(size_t v=0;v<n/vector;++v)for(size_t i=0;i<(f16?8:n);++i)entries[4*g]^=w[v*vector+(f16?8*g:0)+i];}
    };
    for(bool f16:{false,true})for(uint32_t rank:{32u,320u,512u})for(size_t rows:{1u,2u,4u}){
        size_t n=rows*4*rank/(f16?2:1);initialize(f16,rank,n);
        const auto compare=[&](){uint32_t a=999,b=999;const bool x=reference(entries,w.data(),f16,17,n,rank,&a);
            const bool y=vk_tp5_qcheck_avx2(entries,w.data(),f16,17,n,rank,&b);assert(x==y&&a==b);++checks;};
        compare();
        for(size_t i=0;i<n;++i){w[i]^=1;compare();w[i]^=1;}
        for(uint32_t g=0;g<(f16?rank/16:1);++g)for(unsigned k=0;k<3;++k){entries[4*g+k]^=1;compare();entries[4*g+k]^=1;}
    }
    initialize(true,320,640);
    volatile size_t passes=0;uint32_t pending=0;
    const int count=200000;
    const auto begin=std::chrono::steady_clock::now();
    for(int i=0;i<count;++i)passes+=reference(entries,w.data(),true,17,w.size(),320,&pending);
    const auto mid=std::chrono::steady_clock::now();
    for(int i=0;i<count;++i)passes+=vk_tp5_qcheck_avx2(entries,w.data(),true,17,w.size(),320,&pending);
    const auto end=std::chrono::steady_clock::now();
    assert(passes==size_t(2*count));
    std::printf("{\"kind\":\"CPU_QCHECK_MICRO_NOT_MODEL\",\"mutation_checks\":%zu,\"iterations\":%d,\"old_ns\":%.3f,\"new_ns\":%.3f}\n",checks,count,
       std::chrono::duration<double,std::nano>(mid-begin).count()/count,std::chrono::duration<double,std::nano>(end-mid).count()/count);
}
