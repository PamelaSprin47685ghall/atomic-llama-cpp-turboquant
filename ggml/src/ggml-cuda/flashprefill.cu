#include "flashprefill.cuh"
#include "ggml-flashprefill.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

struct ggml_compute_params {
    int ith, nth;
    size_t wsize;
    void * wdata;
    void * threadpool;
    bool use_ref;
};

extern "C" {
    void ggml_compute_forward_flash_prefill_pool(const struct ggml_compute_params * params, struct ggml_tensor * dst);
    void ggml_compute_forward_flash_prefill_select(const struct ggml_compute_params * params, struct ggml_tensor * dst);
    void ggml_compute_forward_flash_prefill_attn(const struct ggml_compute_params * params, struct ggml_tensor * dst);
}

static void ggml_cuda_fp_sync_to_host(const ggml_tensor * t, std::vector<char> & buf) {
    buf.resize(ggml_nbytes(t));
    CUDA_CHECK(cudaMemcpy(buf.data(), t->data, buf.size(), cudaMemcpyDeviceToHost));
}

static void ggml_cuda_fp_sync_to_device(ggml_tensor * t, const std::vector<char> & buf) {
    CUDA_CHECK(cudaMemcpy(t->data, buf.data(), buf.size(), cudaMemcpyHostToDevice));
}

void ggml_cuda_flash_prefill_pool(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const ggml_tensor * K = dst->src[0];
    const ggml_tensor * V = dst->src[1];
    const ggml_tensor * MT = dst->src[2];

    std::vector<char> k_host, v_host, mt_host, dst_host(ggml_nbytes(dst), 0);
    ggml_cuda_fp_sync_to_host(K, k_host);
    ggml_cuda_fp_sync_to_host(V, v_host);
    ggml_cuda_fp_sync_to_host(MT, mt_host);

    ggml_tensor k_tmp = *K; k_tmp.data = k_host.data();
    ggml_tensor v_tmp = *V; v_tmp.data = v_host.data();
    ggml_tensor mt_tmp = *MT; mt_tmp.data = mt_host.data();
    ggml_tensor dst_tmp = *dst; dst_tmp.data = dst_host.data();
    dst_tmp.src[0] = &k_tmp;
    dst_tmp.src[1] = &v_tmp;
    dst_tmp.src[2] = &mt_tmp;

    ggml_compute_params params = {};
    std::vector<char> wdata(4 * 1024 * 1024, 0);
    params.ith = 0;
    params.nth = 1;
    params.wdata = wdata.data();
    params.wsize = wdata.size();
    params.use_ref = true;
    ggml_compute_forward_flash_prefill_pool(&params, &dst_tmp);

    ggml_cuda_fp_sync_to_device(dst, dst_host);
}

void ggml_cuda_flash_prefill_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * pool = dst->src[1];
    const ggml_tensor * MT = dst->src[2];

    std::vector<char> q_host, pool_host, mt_host, dst_host(ggml_nbytes(dst), 0);
    ggml_cuda_fp_sync_to_host(Q, q_host);
    ggml_cuda_fp_sync_to_host(pool, pool_host);
    ggml_cuda_fp_sync_to_host(MT, mt_host);
    ggml_cuda_fp_sync_to_host(dst, dst_host);

    int32_t * meta_i32 = (int32_t *) mt_host.data();
    int64_t meta_words = ggml_nelements(MT);
    if (ggml_flashprefill_metadata_validate(meta_i32, meta_words) != GGML_FLASHPREFILL_OK) {
        int32_t * plan_i32 = (int32_t *) dst_host.data();
        int64_t plan_words = ggml_nelements(dst);
        if (plan_words >= 11) {
            plan_i32[10] = GGML_FLASHPREFILL_ERR_BAD_ARG;
            ggml_cuda_fp_sync_to_device(dst, dst_host);
        }
        return;
    }

    ggml_tensor q_tmp = *Q; q_tmp.data = q_host.data();
    ggml_tensor pool_tmp = *pool; pool_tmp.data = pool_host.data();
    ggml_tensor mt_tmp = *MT; mt_tmp.data = mt_host.data();
    ggml_tensor dst_tmp = *dst; dst_tmp.data = dst_host.data();
    dst_tmp.src[0] = &q_tmp;
    dst_tmp.src[1] = &pool_tmp;
    dst_tmp.src[2] = &mt_tmp;

    ggml_compute_params params = {};
    std::vector<char> wdata(4 * 1024 * 1024, 0);
    params.ith = 0;
    params.nth = 1;
    params.wdata = wdata.data();
    params.wsize = wdata.size();
    params.use_ref = true;
    ggml_compute_forward_flash_prefill_select(&params, &dst_tmp);

    ggml_cuda_fp_sync_to_device(dst, dst_host);
}

void ggml_cuda_flash_prefill_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * pool = dst->src[3];
    const ggml_tensor * plan = dst->src[4];
    const ggml_tensor * meta = dst->src[5];
    const ggml_tensor * sinks = dst->src[6];

    std::vector<char> q_host, k_host, v_host, pool_host, plan_host, meta_host, sinks_host;
    std::vector<char> dst_host(ggml_nbytes(dst), 0);

    ggml_cuda_fp_sync_to_host(Q, q_host);
    ggml_cuda_fp_sync_to_host(K, k_host);
    ggml_cuda_fp_sync_to_host(V, v_host);
    ggml_cuda_fp_sync_to_host(pool, pool_host);
    ggml_cuda_fp_sync_to_host(plan, plan_host);
    ggml_cuda_fp_sync_to_host(meta, meta_host);
    if (sinks) {
        ggml_cuda_fp_sync_to_host(sinks, sinks_host);
    }

    int32_t * meta_i32 = (int32_t *) meta_host.data();
    int32_t * plan_i32 = (int32_t *) plan_host.data();
    int64_t meta_words = ggml_nelements(meta);
    int64_t plan_words = ggml_nelements(plan);

    // Negative coverage validation: corrupt plans report error in plan[10] instead of hard abort
    if (ggml_flashprefill_metadata_validate(meta_i32, meta_words) != GGML_FLASHPREFILL_OK ||
        ggml_flashprefill_plan_validate(plan_i32, plan_words) != GGML_FLASHPREFILL_OK ||
        ggml_flashprefill_plan_validate_against_metadata(plan_i32, plan_words, meta_i32, meta_words) != GGML_FLASHPREFILL_OK) {
        if (plan_words >= 11) {
            plan_i32[10] = GGML_FLASHPREFILL_ERR_BAD_INPUT;
            ggml_cuda_fp_sync_to_device(const_cast<ggml_tensor*>(plan), plan_host);
        }
        return;
    }

    ggml_tensor q_tmp = *Q; q_tmp.data = q_host.data();
    ggml_tensor k_tmp = *K; k_tmp.data = k_host.data();
    ggml_tensor v_tmp = *V; v_tmp.data = v_host.data();
    ggml_tensor pool_tmp = *pool; pool_tmp.data = pool_host.data();
    ggml_tensor plan_tmp = *plan; plan_tmp.data = plan_host.data();
    ggml_tensor meta_tmp = *meta; meta_tmp.data = meta_host.data();
    ggml_tensor sinks_tmp;
    if (sinks) {
        sinks_tmp = *sinks;
        sinks_tmp.data = sinks_host.data();
    }

    ggml_tensor dst_tmp = *dst; dst_tmp.data = dst_host.data();
    dst_tmp.src[0] = &q_tmp;
    dst_tmp.src[1] = &k_tmp;
    dst_tmp.src[2] = &v_tmp;
    dst_tmp.src[3] = &pool_tmp;
    dst_tmp.src[4] = &plan_tmp;
    dst_tmp.src[5] = &meta_tmp;
    dst_tmp.src[6] = sinks ? &sinks_tmp : nullptr;

    ggml_compute_params params = {};
    std::vector<char> wdata(4 * 1024 * 1024, 0);
    params.ith = 0;
    params.nth = 1;
    params.wdata = wdata.data();
    params.wsize = wdata.size();
    params.use_ref = true;
    ggml_compute_forward_flash_prefill_attn(&params, &dst_tmp);

    ggml_cuda_fp_sync_to_device(dst, dst_host);
}
