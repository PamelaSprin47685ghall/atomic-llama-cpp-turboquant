// ggml-vulkan-xkv-landmark-build.inl — Vulkan dispatch for
// GGML_OP_XKV_LANDMARK_BUILD. Included once inside ggml-vulkan.cpp (vk_context
// scope) by the shared-switch integration owner. Three ordered device steps
// device-local barriers: s0 validates params -> status; s1 builds one chunk
// per thread from caller-owned scratch. Host syncs once at graph completion
// and reads only the scalar status (+ eb/srcfp telemetry via backend copies
// owned by the seal bridge, never whole-K readback).
//
// Scratch layout (floats, per chunk c):
//   base = c * (maxDim*P + P + 4*D + pad_D)
//   tile[0,maxDim*P) arow[..+P) recon/ phased/acc/mean[..+D each] dec[..+pad_D)

// 22 u32 = 88B push (xkv_landmark_build.comp)
struct vk_op_xkv_lmbuild_push {
    uint32_t step;
    uint32_t n_rows, n_chunks, chunk_tokens;
    uint32_t total_dim, padded_dim, rank, pad_rank, n_layers;
    uint32_t landmark_type, a_type, b_type;
    uint32_t a_stride, b_stride, dst_stride;
    uint32_t max_dim, pos_is_64, rope_nelems;
    uint32_t status_idx;
    uint32_t seed_lo, phase_lo, phase_hi;
};
static_assert(sizeof(vk_op_xkv_lmbuild_push) == 88, "xkv lmbuild push must be 88B");

static void ggml_vk_xkv_landmark_build(ggml_backend_vk_context * ctx, vk_context & subctx,
        const ggml_tensor * a_k, const ggml_tensor * b_k,
        const ggml_tensor * rows, const ggml_tensor * positions,
        const ggml_tensor * layer_meta, const ggml_tensor * rope_tables,
        const ggml_tensor * scratch, const ggml_tensor * eb,
        const ggml_tensor * srcfp, const ggml_tensor * status,
        ggml_tensor * dst) {
    ggml_xkv_landmark_build_params p;
    memcpy(&p, dst->op_params, sizeof(p));
    char err[256] = {0};
    if (!ggml_xkv_landmark_build_supports(a_k, b_k, rows, positions, layer_meta, rope_tables, scratch, eb, srcfp, status, dst, &p, err, sizeof(err))) {
        fprintf(stderr, "VULKAN INL SUPPORTS FAILED: %s\n", err);
    }
    GGML_ASSERT(ggml_xkv_landmark_build_supports(a_k, b_k, rows, positions, layer_meta,
        rope_tables, scratch, eb, srcfp, status, dst, &p, err, sizeof(err)));

    uint32_t max_dim = p.max_feature_dim ? p.max_feature_dim : p.total_dim;

    vk_op_xkv_lmbuild_push pc = {};
    pc.n_rows = p.n_rows; pc.n_chunks = p.n_chunks; pc.chunk_tokens = p.chunk_tokens;
    pc.total_dim = p.total_dim; pc.padded_dim = p.padded_dim;
    pc.rank = p.rank; pc.pad_rank = p.pad_rank; pc.n_layers = p.n_layers;
    pc.landmark_type = p.landmark_type; pc.a_type = p.a_type; pc.b_type = p.b_type;
    pc.a_stride = (uint32_t)ggml_row_size(a_k->type, a_k->ne[0]);
    pc.b_stride = (uint32_t)ggml_row_size(b_k->type, b_k->ne[0]);
    pc.dst_stride = (uint32_t)ggml_row_size(dst->type, dst->ne[0]);
    pc.max_dim = max_dim;
    pc.pos_is_64 = (positions->type == GGML_TYPE_I64) ? 1u : 0u;
    pc.rope_nelems = (uint32_t)ggml_nelements(rope_tables);
    pc.status_idx = 0u;
    pc.seed_lo = p.seed;
    pc.phase_lo = p.phase_lo;
    pc.phase_hi = p.phase_hi;

    vk_pipeline pipeline = ctx->device->pipeline_xkv_landmark_build;
    GGML_ASSERT(pipeline != nullptr);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 3);
    vk_subbuffer ba  = ggml_vk_tensor_subbuffer(ctx, a_k, false);
    vk_subbuffer bb  = ggml_vk_tensor_subbuffer(ctx, b_k, false);
    vk_subbuffer br  = ggml_vk_tensor_subbuffer(ctx, rows, false);
    vk_subbuffer bp  = ggml_vk_tensor_subbuffer(ctx, positions, false);
    vk_subbuffer blm = ggml_vk_tensor_subbuffer(ctx, layer_meta, false);
    vk_subbuffer brp = ggml_vk_tensor_subbuffer(ctx, rope_tables, false);
    vk_subbuffer bsc = ggml_vk_tensor_subbuffer(ctx, scratch, false);
    vk_subbuffer beb = ggml_vk_tensor_subbuffer(ctx, eb, false);
    vk_subbuffer bfp = ggml_vk_tensor_subbuffer(ctx, srcfp, false);
    vk_subbuffer bdst = ggml_vk_tensor_subbuffer(ctx, dst, false);
    vk_subbuffer bst = ggml_vk_tensor_subbuffer(ctx, status, true);

    pc.step = 0u;
    ggml_vk_sync_buffers(ctx, subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { ba, bb, br, bp, blm, brp, bsc, beb, bfp, bdst, bst }, pc, { 1, 1, 1 });
    ggml_vk_sync_buffers(ctx, subctx);
    pc.step = 1u;
    const uint32_t block = p.landmark_type == GGML_TYPE_Q8_0 ? 32u : 128u;
    const uint32_t n_blocks = p.padded_dim / block;
    const uint64_t n_workgroups = (uint64_t)p.n_chunks * n_blocks;
    const uint64_t step1_invocations = n_workgroups * 128u;
    const uint64_t step2_invocations = (uint64_t)p.n_chunks * 128u;
    GGML_ASSERT(n_workgroups > 0 && step1_invocations <= UINT32_MAX && step2_invocations <= UINT32_MAX);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { ba, bb, br, bp, blm, brp, bsc, beb, bfp, bdst, bst },
        pc, { (uint32_t)step1_invocations, 1, 1 });
    ggml_vk_sync_buffers(ctx, subctx);
    pc.step = 2u;
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { ba, bb, br, bp, blm, brp, bsc, beb, bfp, bdst, bst },
        pc, { (uint32_t)step2_invocations, 1, 1 });
    ggml_vk_sync_buffers(ctx, subctx);
}
