// ggml-vulkan-xkv-factor.inl — Vulkan dispatch for GGML_OP_XKV_FACTORIZE /
// GGML_OP_XKV_CANONICALIZE. Included once inside ggml-vulkan.cpp (vk_context
// scope, after ggml_vk_xkv_reconstruct) by the shared-switch integration
// owner. All intermediate steps use device-local pipeline barriers; the host
// synchronizes once at graph completion and reads only the scalar status.
//
// Scratch layout (floats, exact ggml_xkv_factorize_scratch_bytes):
//   Y[0,n*l) Z[n*l,n*l+m*l) C[..,..+l*m) G[..+l*l) V[..+l*l) S[..+l)
//   ORD[..+l) AF[..+n*r) BTF[..+m*r) RES[.., ..+640) (64 bands x 10)

// 24 u32 = 96B push (xkv_factorize.comp)
struct vk_op_xkv_factor_push {
    uint32_t step;
    uint32_t n, m, r, l;
    uint32_t seed_lo, seed_hi;
    uint32_t balance;
    uint32_t type_a, type_b;
    uint32_t pad_a, pad_b;
    uint32_t off_Y, off_Z, off_C;
    uint32_t off_G, off_V, off_S;
    uint32_t off_ORD, off_AF, off_BTF;
    uint32_t status_idx;
    uint32_t qr_rows;
    uint32_t aux;
};
static_assert(sizeof(vk_op_xkv_factor_push) == 96, "xkv factor push must be 96B");

// 13 u32 = 52B push (xkv_canonicalize.comp)
// 14 u32 = 56B push (xkv_canonicalize.comp)
struct vk_op_xkv_canon_push {
    uint32_t step;
    uint32_t n, nvec, hd, phd;
    uint32_t rotary_dim, rope_mode, input_type, is_k, hadamard_dim;
    uint32_t hot_row_bytes, hot_nphys;
    uint32_t pos_is_64;
    uint32_t status_idx;
};
static_assert(sizeof(vk_op_xkv_canon_push) == 56, "xkv canon push must be 56B");

static void ggml_vk_xkv_factor_dispatch(ggml_backend_vk_context * ctx, vk_context & subctx,
        vk_pipeline pipeline, vk_subbuffer bx, vk_subbuffer bscr,
        vk_subbuffer ba, vk_subbuffer bb, vk_subbuffer bst,
        const vk_op_xkv_factor_push & pc, uint32_t groups_x) {
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_vk_sync_buffers(ctx, subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { bx, bscr, ba, bb, bst }, pc, { groups_x, 1, 1 });
    ggml_vk_sync_buffers(ctx, subctx);
}

static void ggml_vk_xkv_factorize(ggml_backend_vk_context * ctx, vk_context & subctx,
        const ggml_tensor * x, const ggml_tensor * scratch,
        const ggml_tensor * b_out, const ggml_tensor * status,
        ggml_tensor * dst_a) {
    ggml_xkv_factorize_params p;
    memcpy(&p, dst_a->op_params, sizeof(p));
    char err[256] = {0};
    GGML_ASSERT(ggml_xkv_factorize_supports(x, scratch, b_out, status, dst_a, &p, err, sizeof(err)));

    uint32_t n = p.rows_n, m = p.cols_m;
    uint32_t r = p.requested_rank;
    uint32_t mn = n < m ? n : m;
    if (r > mn) r = mn;
    uint32_t l = r + p.oversampling;
    if (l > mn) l = mn;

    vk_op_xkv_factor_push pc = {};
    pc.n = n; pc.m = m; pc.r = r; pc.l = l;
    pc.seed_lo = p.seed_low; pc.seed_hi = p.seed_high;
    pc.balance = p.balance_mode;
    pc.type_a = (uint32_t)dst_a->type; pc.type_b = (uint32_t)b_out->type;
    pc.pad_a = (uint32_t)dst_a->ne[0]; pc.pad_b = (uint32_t)b_out->ne[0];
    pc.off_Y = 0;
    pc.off_Z = pc.off_Y + n * l;
    pc.off_C = pc.off_Z + m * l;
    pc.off_G = pc.off_C + l * m;
    pc.off_V = pc.off_G + l * l;
    pc.off_S = pc.off_V + l * l;
    pc.off_ORD = pc.off_S + l;
    pc.off_AF = pc.off_ORD + l;
    pc.off_BTF = pc.off_AF + n * r;
    pc.status_idx = 0;
    pc.qr_rows = n;
    pc.aux = pc.off_Y; // QR base: Y by default, off_Z for Z passes

    vk_pipeline pipeline = ctx->device->pipeline_xkv_factorize;
    GGML_ASSERT(pipeline != nullptr);
    vk_subbuffer bx    = ggml_vk_tensor_subbuffer(ctx, x, false);
    vk_subbuffer bscr  = ggml_vk_tensor_subbuffer(ctx, scratch, false);
    vk_subbuffer ba    = ggml_vk_tensor_subbuffer(ctx, dst_a, false);
    vk_subbuffer bb    = ggml_vk_tensor_subbuffer(ctx, b_out, false);
    vk_subbuffer bst   = ggml_vk_tensor_subbuffer(ctx, status, false);

    // NOTE: ggml_vk_dispatch_pipeline takes ELEMENT counts (it divides by
    // wg_denoms itself). Single-workgroup steps (QR/Jacobi/sign) use count 1.
    pc.step = 0; // validate -> status
    ggml_vk_xkv_factor_dispatch(ctx, subctx, pipeline, bx, bscr, ba, bb, bst, pc, 1);

    pc.step = 1; // Y = X*Omega fused
    ggml_vk_xkv_factor_dispatch(ctx, subctx, pipeline, bx, bscr, ba, bb, bst, pc, n * l);

    for (uint32_t it = 0; it <= p.power_iterations; ++it) {
        if (it > 0) {
            // Z = X^T Q ; Y = X Z (power iteration body)
            pc.step = 3; pc.qr_rows = m;
            ggml_vk_xkv_factor_dispatch(ctx, subctx, pipeline, bx, bscr, ba, bb, bst, pc, m * l);
            pc.step = 2; pc.qr_rows = m; pc.aux = pc.off_Z; // QR(Z)
            ggml_vk_xkv_factor_dispatch(ctx, subctx, pipeline, bx, bscr, ba, bb, bst, pc, 1);
            pc.step = 4;
            ggml_vk_xkv_factor_dispatch(ctx, subctx, pipeline, bx, bscr, ba, bb, bst, pc, n * l);
        }
        pc.step = 2; pc.qr_rows = n; pc.aux = pc.off_Y; // QR(Y)
        ggml_vk_xkv_factor_dispatch(ctx, subctx, pipeline, bx, bscr, ba, bb, bst, pc, 1);
    }

    pc.step = 5; // C = Q^T X
    ggml_vk_xkv_factor_dispatch(ctx, subctx, pipeline, bx, bscr, ba, bb, bst, pc, l * m);
    pc.step = 6; // G = C C^T
    ggml_vk_xkv_factor_dispatch(ctx, subctx, pipeline, bx, bscr, ba, bb, bst, pc, l * l);
    pc.step = 7; // Jacobi + sort
    ggml_vk_xkv_factor_dispatch(ctx, subctx, pipeline, bx, bscr, ba, bb, bst, pc, 1);
    pc.step = 8; // factors
    ggml_vk_xkv_factor_dispatch(ctx, subctx, pipeline, bx, bscr, ba, bb, bst, pc, (n + m) * r);
    pc.step = 9; // sign + balance
    ggml_vk_xkv_factor_dispatch(ctx, subctx, pipeline, bx, bscr, ba, bb, bst, pc, 1);
    pc.step = 10; // encode A
    ggml_vk_xkv_factor_dispatch(ctx, subctx, pipeline, bx, bscr, ba, bb, bst, pc, n);
    pc.step = 11; // encode B
    ggml_vk_xkv_factor_dispatch(ctx, subctx, pipeline, bx, bscr, ba, bb, bst, pc, m);
    // Checked multiply helper: rejects overflow (would wrap dispatch size).
    auto ck_mul_u32 = [](uint32_t a, uint32_t b) -> uint32_t {
        uint64_t prod = (uint64_t)a * b;
        GGML_ASSERT(prod <= UINT32_MAX);
        return (uint32_t)prod;
    };
    // Steps 12/13: EXACT all-row tiled final-codec residual (no sampling).
    // G bands (1..64) parallelize reduction across all n rows; step 13 folds
    // bands deterministically -> status[1..9]. Obsolete trailing step 12 deleted.
    {
        uint32_t G = (n + 255u) / 256u;
        if (G > 64u) G = 64u;
        if (G == 0u) G = 1u;
        pc.step = 12; pc.aux = G; // band partials -> RES region
        ggml_vk_xkv_factor_dispatch(ctx, subctx, pipeline, bx, bscr, ba, bb, bst, pc, ck_mul_u32(G, 256u));
        pc.step = 13; pc.aux = G; // deterministic fold -> status[1..9]
        ggml_vk_xkv_factor_dispatch(ctx, subctx, pipeline, bx, bscr, ba, bb, bst, pc, 1u);
    }
}

static void ggml_vk_xkv_canonicalize(ggml_backend_vk_context * ctx, vk_context & subctx,
        const ggml_tensor * hot, const ggml_tensor * rows,
        const ggml_tensor * positions, const ggml_tensor * rope_tables,
        const ggml_tensor * hadamard, const ggml_tensor * status,
        ggml_tensor * dst) {
    ggml_xkv_canonicalize_params p;
    memcpy(&p, dst->op_params, sizeof(p));
    char err[256] = {0};
    GGML_ASSERT(ggml_xkv_canonicalize_supports(hot, rows, positions, rope_tables,
                                               hadamard, status, dst, &p, err, sizeof(err)));

    vk_op_xkv_canon_push pc = {};
    pc.step = 0;
    pc.n = p.n_rows;
    pc.nvec = p.n_layers * p.n_heads;
    pc.hd = p.head_dim;
    pc.phd = p.padded_head_dim;
    pc.rotary_dim = p.rotary_dim;
    pc.rope_mode = p.rope_mode;
    pc.input_type = p.input_type;
    pc.is_k = p.is_k;
    pc.hadamard_dim = p.hadamard_dim;
    pc.hot_row_bytes = (uint32_t)ggml_row_size(hot->type, hot->ne[0]);
    pc.hot_nphys = (uint32_t)hot->ne[1];
    pc.pos_is_64 = (positions->type == GGML_TYPE_I64) ? 1u : 0u;
    pc.status_idx = 0;

    vk_pipeline pipeline = ctx->device->pipeline_xkv_canonicalize;
    GGML_ASSERT(pipeline != nullptr);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    vk_subbuffer bhot = ggml_vk_tensor_subbuffer(ctx, hot, false);
    vk_subbuffer brows = ggml_vk_tensor_subbuffer(ctx, rows, false);
    vk_subbuffer bpos = ggml_vk_tensor_subbuffer(ctx, positions, false);
    vk_subbuffer brope = ggml_vk_tensor_subbuffer(ctx, rope_tables, false);
    vk_subbuffer bhad = ggml_vk_tensor_subbuffer(ctx, hadamard, false);
    vk_subbuffer bdst = ggml_vk_tensor_subbuffer(ctx, dst, false);
    vk_subbuffer bst = ggml_vk_tensor_subbuffer(ctx, status, false);

    ggml_vk_sync_buffers(ctx, subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { bhot, brows, bpos, brope, bhad, bdst, bst }, pc, { 1, 1, 1 });
    ggml_vk_sync_buffers(ctx, subctx);

    pc.step = 1;
    uint32_t total = pc.n * pc.nvec; // element count; helper divides by wg denom
    uint32_t groups = total;
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    ggml_vk_sync_buffers(ctx, subctx);
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { bhot, brows, bpos, brope, bhad, bdst, bst }, pc, { groups, 1, 1 });
    ggml_vk_sync_buffers(ctx, subctx);
}
