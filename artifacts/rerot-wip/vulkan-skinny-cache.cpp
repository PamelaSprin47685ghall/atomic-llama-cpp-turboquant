#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc,char**argv){
    if(argc>1){unsetenv("GGML_VK_DISABLE_MMVQ");setenv("GGML_VK_FORCE_MMVQ","1",1);}
    ggml_backend_load_all();auto dev=ggml_backend_dev_by_name("Vulkan0");if(!dev)return 2;auto be=ggml_backend_dev_init(dev,nullptr);
    constexpr int K=2048,M=64,N=27;
    auto*c=ggml_init({ggml_tensor_overhead()*16+ggml_graph_overhead_custom(16,false),nullptr,true});auto*g=ggml_new_graph_custom(c,16,false);
    auto*w=ggml_new_tensor_2d(c,GGML_TYPE_Q8_0,K,M);auto*x=ggml_new_tensor_2d(c,GGML_TYPE_F32,K,N);auto*y=ggml_new_tensor_2d(c,GGML_TYPE_F32,K,N);
    ggml_set_input(x);ggml_set_input(y);auto*a=ggml_mul_mat(c,w,x);auto*b=ggml_mul_mat(c,w,y);ggml_set_output(a);ggml_set_output(b);ggml_build_forward_expand(g,a);ggml_build_forward_expand(g,b);
    auto buffer=ggml_backend_alloc_ctx_tensors(c,be);if(!buffer)return 2;
    std::vector<float> wf(K*M,0.0f),xf(K*N,1.0f),yf(K*N,-2.0f);for(int m=0;m<M;++m)wf[m*K+(m%K)]=127.0f;
    std::vector<char> qw(ggml_nbytes(w));ggml_quantize_chunk(GGML_TYPE_Q8_0,wf.data(),qw.data(),0,M,K,nullptr);
    ggml_backend_tensor_set(w,qw.data(),0,qw.size());ggml_backend_tensor_set(x,xf.data(),0,xf.size()*sizeof(float));ggml_backend_tensor_set(y,yf.data(),0,yf.size()*sizeof(float));
    if(ggml_backend_graph_compute(be,g)!=GGML_STATUS_SUCCESS)return 2;
    std::vector<float> av(M*N),bv(M*N);ggml_backend_tensor_get(a,av.data(),0,av.size()*sizeof(float));ggml_backend_tensor_get(b,bv.data(),0,bv.size()*sizeof(float));
    double ea=0,eb=0;for(size_t i=0;i<av.size();++i){ea=std::max(ea,std::abs(double(av[i])-127.0));eb=std::max(eb,std::abs(double(bv[i])+254.0));}
    std::printf("skinny mode=%s first=%g expected=127 second=%g expected=-254 max_errors=%g,%g\n",argc>1?"forced":"auto",av[0],bv[0],ea,eb);
    ggml_backend_buffer_free(buffer);ggml_free(c);ggml_backend_free(be);return ea<0.1&&eb<0.1?0:1;
}
