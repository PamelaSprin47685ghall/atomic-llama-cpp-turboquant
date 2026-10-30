#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int S = 10, E = 13;
using counter_fn = int64_t (*)(ggml_backend_t, const char *);
void check(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
    x *= 0x846ca68bU; return x ^ (x >> 16);
}
struct Weight {
    ggml_tensor * storage;
    ggml_tensor * view;
    std::vector<uint8_t> bytes;
};
Weight weight(ggml_context * ctx, int k, int m, bool padded, uint32_t seed) {
    const int storage_k = padded ? (k == 128 ? 640 : k + 64) : k;
    auto * storage = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, storage_k, m + int(padded), E);
    auto * view = padded ? ggml_view_3d(ctx, storage, k, m, E, storage->nb[1], storage->nb[2], 0) : storage;
    Weight w{storage, view, std::vector<uint8_t>(ggml_nbytes(storage), 0xa5)};
    for (int e = 0; e < E; ++e) for (int row = 0; row < m; ++row) for (int b = 0; b < k/64; ++b) {
        const uint32_t h = mix(seed + uint32_t((e*m+row)*(k/64)+b));
        auto * p = w.bytes.data() + e*view->nb[2] + row*view->nb[1] + b*18;
        const float d = (h & 1 ? -1.0f : 1.0f) / (h & 2 ? 128.0f : 256.0f);
        const auto half = ggml_fp32_to_fp16(d); std::memcpy(p, &half, 2);
        for (int j = 0; j < 16; ++j) p[2+j] = uint8_t(mix(h+j*13));
    }
    return w;
}
double decoded(const Weight & w, int e, int row, int col) {
    const auto * p = w.bytes.data() + e*w.view->nb[2] + row*w.view->nb[1] + (col/64)*18;
    ggml_fp16_t half; std::memcpy(&half, p, 2);
    return double(ggml_fp16_to_fp32(half)) * (int((p[2+(col%64)/4] >> (2*(col%4))) & 3)-1);
}
size_t run(ggml_backend_t backend, counter_fn counter, int k, bool padded, int slots, bool allow_fusion) {
    // One and five input slots fit native fusion. Ten slots use separate dots.
    const bool expect_fusion = allow_fusion && slots != S;
    const int m = k == 128 ? 129 : 128;
    const ggml_init_params ip{40*ggml_tensor_overhead()+ggml_graph_overhead_custom(40,false),nullptr,true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(ip),ggml_free);
    check(bool(ctx), "metadata allocation");
    auto wg=weight(ctx.get(),k,m,padded,137), wu=weight(ctx.get(),k,m,padded,927);
    auto * xs=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,k+int(padded),slots,1);
    auto * x=padded ? ggml_view_3d(ctx.get(),xs,k,slots,1,xs->nb[1],xs->nb[2],0) : xs;
    auto * ids=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,S,1);
    std::vector<uint8_t> xb(ggml_nbytes(xs),0xff);
    std::vector<int32_t> selected(S);
    std::vector<float> xv(size_t(slots)*k);
    for (int s=0;s<slots;++s) for(int j=0;j<k;++j) {
        // Every Q8_1 block has maxabs=127/32. Its scale and every input are
        // exactly representable, so activation quantization introduces no error.
        const int q=j%32==0 ? 127 : int(mix(uint32_t(s*k+j)+713)%255)-127;
        const float v=float(q)/32.0f;
        xv[size_t(s)*k+j]=v;
        std::memcpy(xb.data()+s*x->nb[1]+j*sizeof(float),&v,sizeof(v));
    }
    for(int s=0;s<S;++s) selected[s]=(3*s+1)%E;
    auto * gate=ggml_mul_mat_id(ctx.get(),wg.view,x,ids);
    auto * up=ggml_mul_mat_id(ctx.get(),wu.view,x,ids);
    auto * out=ggml_swiglu_split(ctx.get(),gate,up);
    auto * graph=ggml_new_graph_custom(ctx.get(),40,false);
    for(auto * t:{wg.view,wu.view,x,ids}) ggml_build_forward_expand(graph,t);
    ggml_set_output(out);ggml_build_forward_expand(graph,out);
        if(!allow_fusion) {
        ggml_set_output(gate);ggml_set_output(up);
        ggml_build_forward_expand(graph,ggml_sqr(ctx.get(),up));
    }
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(),backend),ggml_backend_buffer_free);
    check(bool(buffer),"tensor allocation");
    // Original model tensors are weights, including strided shard views.
    ggml_backend_buffer_set_usage(buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(wg.storage,wg.bytes.data(),0,wg.bytes.size());
    ggml_backend_tensor_set(wu.storage,wu.bytes.data(),0,wu.bytes.size());
    ggml_backend_tensor_set(xs,xb.data(),0,xb.size());
    ggml_backend_tensor_set(ids,selected.data(),0,selected.size()*sizeof(int32_t));
    const int64_t before=counter(backend,"mul_mat_glu");
    check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"native graph compute");
    ggml_backend_synchronize(backend);
    const auto used=counter(backend,"mul_mat_glu")-before;
        check(used == int64_t(expect_fusion), "fusion/use-count admission");
    std::vector<float> y(size_t(m)*S),g(y.size()),u(y.size());
    ggml_backend_tensor_get(out,y.data(),0,y.size()*sizeof(float));
        if(!expect_fusion) {
        ggml_backend_tensor_get(gate,g.data(),0,g.size()*sizeof(float));
        ggml_backend_tensor_get(up,u.data(),0,u.size()*sizeof(float));
    }
    for(int s=0;s<S;++s) for(int row=0;row<m;++row) {
        double eg=0,eu=0;
        for(int j=0;j<k;++j) {
            const double a=xv[size_t(s % slots)*k+j];
            eg+=decoded(wg,selected[s],row,j)*a;
            eu+=decoded(wu,selected[s],row,j)*a;
        }
        const size_t pos=size_t(s)*m+row;
                if(!expect_fusion) check(g[pos]==float(eg) && u[pos]==float(eu),"exact Q2/Q8 dot oracle");
        const double expected=(eg/(1+std::exp(-eg)))*eu;
        check(std::isfinite(y[pos]) && std::abs(double(y[pos])-expected)<=2e-5*(1+std::abs(expected)),"independent SwiGLU oracle");
    }
    for(const auto & w:{wg,wu}) {
        std::vector<uint8_t> back(w.bytes.size());
        ggml_backend_tensor_get(w.storage,back.data(),0,back.size());
        check(back==w.bytes,"encoded weights or padding changed");
    }
    std::vector<uint8_t> back(xb.size());std::vector<int32_t> ib(S);
    ggml_backend_tensor_get(xs,back.data(),0,back.size());
    ggml_backend_tensor_get(ids,ib.data(),0,ib.size()*sizeof(int32_t));
    check(back==xb && ib==selected,"activation or IDs changed");
        const char * rejection = !allow_fusion ? "external_use" : slots == S ? "input_slots" : "none";
    std::printf("NATIVE_Q2_CASE device=%s k=%d m=%d slots=%d padded=%d fusion_allowed=%d fusion_used=%lld rejection=%s oracle=true sources_preserved=true\n",
        ggml_backend_name(backend),k,m,slots,int(padded),int(allow_fusion),static_cast<long long>(used),rejection);
    return y.size();
}
}
int main(int argc,char ** argv) {
    try {
        check(argc==2,"usage: native-q2-mmvq <installed-backend-dir>");
        check(ggml_blck_size(GGML_TYPE_Q2_0)==64 && ggml_type_size(GGML_TYPE_Q2_0)==18,"Q2_0 layout");
        ggml_backend_load_all_from_path(argv[1]);
        size_t devices=0,cases=0,values=0;
        for(size_t i=0;i<ggml_backend_dev_count();++i) {
            auto * dev=ggml_backend_dev_get(i);auto * reg=ggml_backend_dev_backend_reg(dev);
            if(ggml_backend_dev_type(dev)!=GGML_BACKEND_DEVICE_TYPE_GPU || std::string(ggml_backend_reg_name(reg))!="ROCm") continue;
            auto counter=reinterpret_cast<counter_fn>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_fusion_count"));
            check(counter!=nullptr,"fusion counter");
            std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(dev,nullptr),ggml_backend_free);
            check(bool(backend),"native backend initialization");
            for(int k:{128,2560}) for(bool padded:{false,true}) for(int slots:{1,5,S}) for(bool allow_fusion:{false,true}) {
                values+=run(backend.get(),counter,k,padded,slots,allow_fusion);++cases;
            }
            ++devices;
        }
        check(devices==5,"five real ROCm devices required");
        std::printf("NATIVE_Q2_PASS devices=%zu cases=%zu values=%zu\n",devices,cases,values);return 0;
    } catch(const std::exception & e) {std::fprintf(stderr,"NATIVE_Q2_FAIL %s\n",e.what());return 1;}
}
