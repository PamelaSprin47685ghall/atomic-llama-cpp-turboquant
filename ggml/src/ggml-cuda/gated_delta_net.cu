#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// Device solver caps for the RBB block path. The packed Cholesky factor is
// N(N+1)/2 doubles of per-thread state; the tested and production frontier
// sizes (N <= 64 writers, S_v <= 128) fit comfortably. Larger shapes are
// honestly declined in supports_op so they fall back to the CPU oracle.
#define GGML_CUDA_RBB_MAXN 64
#define GGML_CUDA_RBB_MAXS 128

// Exact order-free Parallel Delta block update, one thread per (head, state
// column). Solves the same symmetrically sqrt(beta)-scaled system as the CPU
// oracle (A = C K K^T C + diag(density*(1-beta+eps*beta))) with a packed
// double-precision Cholesky factor, then emits the same dst layout:
// output rows, merged-brain rows, per-writer hand rows (scratch untouched).
__global__ void gated_delta_net_rbb_cuda(
        const float * q, const float * k, const float * v,
        const float * g, const float * beta,
        const float * brain, const float * native, float * dst,
        int64_t S, int64_t H, int64_t N,
        int64_t neq1, int64_t neq3, size_t nbq1, size_t nbq3,
        int64_t nek1, int64_t nek3, size_t nbk1, size_t nbk3,
        size_t nbv1, size_t nbv3, size_t nbg1, size_t nbg3, size_t nbb1, size_t nbb3,
        float scale, int density_norm) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * S) {
        return;
    }
    const int64_t h   = idx / S;
    const int64_t col = idx % S;

    const char * qc = (const char *) q;
    const char * kc = (const char *) k;
    const char * vc = (const char *) v;
    const char * gc = (const char *) g;
    const char * bc = (const char *) beta;

    const int64_t hq = h % neq1;
    const int64_t hk = h % nek1;
    const int64_t qdiv = N / neq3;
    const int64_t kdiv = N / nek3;

    double beta_a[GGML_CUDA_RBB_MAXN];
    double root_b[GGML_CUDA_RBB_MAXN];
    double alpha[GGML_CUDA_RBB_MAXN];
    double vcol[GGML_CUDA_RBB_MAXN];
    double log_decay = 0.0;
    for (int64_t n = 0; n < N; ++n) {
        const float b = *(const float *) (bc + n * nbb3 + h * nbb1);
        const float gn = *(const float *) (gc + n * nbg3 + h * nbg1);
        beta_a[n] = (double) b;
        root_b[n] = sqrt((double) b);
        alpha[n] = exp((double) gn);
        log_decay += (double) gn;
        vcol[n] = (double) *(const float *) (vc + n * nbv3 + h * nbv1 + col * sizeof(float));
    }
    const double decay = exp(log_decay / (double) N);

    double base[GGML_CUDA_RBB_MAXS];
    const int64_t brain_col = (h * S + col) * S;
    for (int64_t d = 0; d < S; ++d) {
        base[d] = (double) brain[brain_col + d];
    }

    double weights[GGML_CUDA_RBB_MAXN];
    for (int64_t n = 0; n < N; ++n) {
        const int64_t ik3 = n / kdiv;
        const char * krow = kc + ik3 * nbk3 + hk * nbk1;
        double residual = vcol[n];
        for (int64_t d = 0; d < S; ++d) {
            residual -= (double) *(const float *) (krow + d * sizeof(float)) * decay * base[d];
        }
        weights[n] = (N == 1 ? beta_a[n] : root_b[n]) * residual;
    }

    if (N > 1) {
        double density[GGML_CUDA_RBB_MAXN];
        for (int64_t n = 0; n < N; ++n) {
            density[n] = 1.0;
        }
        if (density_norm && N > 1) {
            for (int64_t i = 0; i < N; ++i) {
                const int64_t ik3_i = i / kdiv;
                const char * kirow = kc + ik3_i * nbk3 + hk * nbk1;
                for (int64_t j = 0; j < N; ++j) {
                    if (i == j || beta_a[j] == 0.0) {
                        continue;
                    }
                    const int64_t ik3_j = j / kdiv;
                    const char * kjrow = kc + ik3_j * nbk3 + hk * nbk1;
                    double dot = 0.0, ni = 0.0, nj = 0.0;
                    for (int64_t d = 0; d < S; ++d) {
                        const double ki = (double) *(const float *) (kirow + d * sizeof(float));
                        const double kj = (double) *(const float *) (kjrow + d * sizeof(float));
                        dot += ki * kj;
                        ni += ki * ki;
                        nj += kj * kj;
                    }
                    if (ni * nj > 1e-20) {
                        density[i] += dot * dot / (ni * nj);
                    }
                }
            }
        }
        // Packed lower-triangular Cholesky factor L[i*(i+1)/2+j].
        double lower[GGML_CUDA_RBB_MAXN * (GGML_CUDA_RBB_MAXN + 1) / 2];
        for (int64_t i = 0; i < N; ++i) {
            const int64_t ik3_i = i / kdiv;
            const char * kirow = kc + ik3_i * nbk3 + hk * nbk1;
            for (int64_t j = 0; j <= i; ++j) {
                const int64_t ik3_j = j / kdiv;
                const char * kjrow = kc + ik3_j * nbk3 + hk * nbk1;
                double value = 0.0;
                for (int64_t d = 0; d < S; ++d) {
                    value += (double) *(const float *) (kirow + d * sizeof(float)) *
                             (double) *(const float *) (kjrow + d * sizeof(float));
                }
                value *= root_b[i] * root_b[j];
                if (i == j) {
                    value += density[i] * (1.0 - beta_a[i] + 1.0e-4 * beta_a[i]);
                }
                for (int64_t j0 = 0; j0 < j; ++j0) {
                    value -= lower[i * (i + 1) / 2 + j0] * lower[j * (j + 1) / 2 + j0];
                }
                if (i == j) {
                    lower[i * (i + 1) / 2 + j] = sqrt(value);
                } else {
                    lower[i * (i + 1) / 2 + j] = value / lower[j * (j + 1) / 2 + j];
                }
            }
        }
        for (int64_t i = 0; i < N; ++i) {
            for (int64_t j = 0; j < i; ++j) {
                weights[i] -= lower[i * (i + 1) / 2 + j] * weights[j];
            }
            weights[i] /= lower[i * (i + 1) / 2 + i];
        }
        for (int64_t i = N; i-- > 0;) {
            for (int64_t j = i + 1; j < N; ++j) {
                weights[i] -= lower[j * (j + 1) / 2 + i] * weights[j];
            }
            weights[i] /= lower[i * (i + 1) / 2 + i];
        }
        for (int64_t i = 0; i < N; ++i) {
            weights[i] *= root_b[i];
        }
    }

    const int64_t state_size = S * S * H;
    const int64_t brain_off = S * H * N;
    const int64_t hand_off = brain_off + state_size;

    double merged[GGML_CUDA_RBB_MAXS];
    for (int64_t d = 0; d < S; ++d) {
        double acc = decay * base[d];
        for (int64_t n = 0; n < N; ++n) {
            const int64_t ik3 = n / kdiv;
            acc += (double) *(const float *) (kc + ik3 * nbk3 + hk * nbk1 + d * sizeof(float)) * weights[n];
        }
        merged[d] = acc;
        dst[brain_off + brain_col + d] = (float) acc;
    }

    for (int64_t n = 0; n < N; ++n) {
        const int64_t iq3 = n / qdiv;
        const int64_t ik3 = n / kdiv;
        const char * qrow = qc + iq3 * nbq3 + hq * nbq1;
        const char * krow = kc + ik3 * nbk3 + hk * nbk1;
        const int64_t nat_base = n * state_size + brain_col;
        double proj_hand = 0.0, proj_native = 0.0;
        for (int64_t d = 0; d < S; ++d) {
            const double kval = (double) *(const float *) (krow + d * sizeof(float));
            const double nat = (double) native[nat_base + d];
            proj_hand += kval * (nat - base[d]);
            proj_native += kval * alpha[n] * nat;
        }
        const double native_delta = beta_a[n] * (vcol[n] - proj_native);
        double readout = 0.0;
        for (int64_t d = 0; d < S; ++d) {
            const double kval = (double) *(const float *) (krow + d * sizeof(float));
            const double nat = (double) native[nat_base + d];
            const double local_hand = alpha[n] * (nat - base[d] - beta_a[n] * kval * proj_hand);
            const double shared_write = merged[d] - decay * base[d];
            const double self_echo = N == 1 ? 0.0 : kval * weights[n] - shared_write / (double) N;
            dst[hand_off + nat_base + d] = (float) (local_hand + self_echo);
            const double native_candidate = alpha[n] * nat + kval * native_delta;
            readout += (double) *(const float *) (qrow + d * sizeof(float)) * native_candidate;
        }
        dst[(n * H + h) * S + col] = (float) (readout * (double) scale);
    }
}

static void ggml_cuda_op_gated_delta_net_rbb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];
    ggml_tensor * src_nat   = dst->src[6];

    const int64_t S_v      = src_v->ne[0];
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_v->ne[3];

    GGML_ASSERT(src_g->ne[0] == 1);
    GGML_ASSERT(n_tokens == 1);
    GGML_ASSERT(ggml_get_op_params_i32(dst, 0) == 1);
    GGML_ASSERT(S_v <= GGML_CUDA_RBB_MAXS);
    GGML_ASSERT(n_seqs <= GGML_CUDA_RBB_MAXN);
    GGML_ASSERT(ggml_is_contiguous(src_state));
    GGML_ASSERT(ggml_is_contiguous(src_nat));

    const int density_norm = ggml_get_op_params_i32(dst, 2);
    const float scale = 1.0f / sqrtf((float) S_v);

    const int64_t total = H * S_v;
    const int block = 256;
    const int grid = (int) ((total + block - 1) / block);
    ggml_cuda_kernel_launch(gated_delta_net_rbb_cuda,
        ggml_cuda_kernel_launch_params(grid, block, 0, ctx.stream()),
        (const float *) src_q->data, (const float *) src_k->data, (const float *) src_v->data,
        (const float *) src_g->data, (const float *) src_beta->data,
        (const float *) src_state->data, (const float *) src_nat->data, (float *) dst->data,
        S_v, H, n_seqs,
        src_q->ne[1], src_q->ne[3], src_q->nb[1], src_q->nb[3],
        src_k->ne[1], src_k->ne[3], src_k->nb[1], src_k->nb[3],
        src_v->nb[1], src_v->nb[3], src_g->nb[1], src_g->nb[3], src_beta->nb[1], src_beta->nb[3],
        scale, density_norm);
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    // RBB (parallel-delta) block path: op_params[1] != 0 selects the exact
    // order-free Parallel Delta solve, mirroring
    // ggml_compute_forward_gated_delta_net_rbb_f32 in double precision.
    // op_params[2] selects evidence-density normalization (1 = default
    // production contract, 0 = raw-redundant research ablation).
    if (ggml_get_op_params_i32(dst, 1) != 0) {
        GGML_ASSERT(cache == nullptr); // cache fusion only applies to the native path
        ggml_cuda_op_gated_delta_net_rbb(ctx, dst);
        return;
    }
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
