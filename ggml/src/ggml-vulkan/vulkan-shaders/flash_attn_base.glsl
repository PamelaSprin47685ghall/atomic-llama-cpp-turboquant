
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout (constant_id =  0) const uint32_t WorkGroupSize = 128;
layout (constant_id =  1) const uint32_t Br = 1;
layout (constant_id =  2) const uint32_t Bc = 32;
layout (constant_id =  3) const uint32_t HSK = 32;
layout (constant_id =  4) const uint32_t HSV = 32;
layout (constant_id =  5) const uint32_t Clamp = 0;
layout (constant_id =  6) const uint32_t D_split = 16;
layout (constant_id =  7) const uint32_t row_split = 1;
layout (constant_id =  8) const uint32_t SubGroupSize = 32;
layout (constant_id =  9) const uint32_t SHMEM_STAGING = 0;
layout (constant_id = 10) const uint32_t Flags = 0;
layout (constant_id = 11) const uint32_t LIMIT_OCCUPANCY_SHMEM = 0;
// ggml_type enumerant for K/V
layout (constant_id = 12) const uint32_t FaTypeK = 0;
layout (constant_id = 13) const uint32_t FaTypeV = 0;
// sizeof(decode buffer): quants -> ggml block size; F32 -> 16 (decodeBufF32 vec4).
layout (constant_id = 14) const uint32_t FaBlockBytesK = 2;
layout (constant_id = 15) const uint32_t FaBlockBytesV = 2;

const bool USE_MASK_OPT    = (Flags & 1) != 0;
const bool MASK_ENABLE     = (Flags & 2) != 0;
const bool LOGIT_SOFTCAP   = (Flags & 4) != 0;
const bool OLD_AMD_WINDOWS = (Flags & 8) != 0;

// Round up head sizes to a multiple of 16, for coopmat1/coopmat2 paths
const uint32_t HSK_pad = (HSK + 15) & ~15;
const uint32_t HSV_pad = (HSV + 15) & ~15;

const bool KV_bounds_check = Clamp != 0;

layout (push_constant) uniform parameter {
    uint32_t N;
    uint32_t KV;

    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;

    uint32_t neq2;
    uint32_t neq3;
    uint32_t nek2;
    uint32_t nek3;
    uint32_t nev2;
    uint32_t nev3;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nem3;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t nb21;
    uint32_t nb22;
    uint32_t nb23;

    float scale;
    float max_bias;
    float logit_softcap;

    uint32_t mask_n_head_log2;
    float m0;
    float m1;

    uint32_t gqa_ratio;
    uint32_t split_kv;
    uint32_t k_num;
} p;

#define SINK_ENABLE_BIT (1<<24)
#define N_LOG2_MASK 0xFFFF

layout (binding = 4) readonly buffer S {float data_s[];};

layout (binding = 5) writeonly buffer O {D_TYPE data_o[];};
layout (binding = 5) writeonly buffer OV4 {D_TYPEV4 data_ov4[];};

layout (binding = 6) readonly buffer MO {uint32_t data_mask_opt[];};

#define MASK_OPT_ALL_NEG_INF 1
#define MASK_OPT_ALL_ZERO 2

#define BINDING_IDX_K 0
#define BINDING_IDX_V 1

// FaTypeK / FaTypeV spec constant values. These mirror enum ggml_type so the
// host can pass the type directly. Keep in sync with ggml.h.
#define FA_TYPE_F32   0u
#define FA_TYPE_F16   1u
#define FA_TYPE_Q4_0  2u
#define FA_TYPE_Q4_1  3u
#define FA_TYPE_Q5_0  6u
#define FA_TYPE_Q5_1  7u
#define FA_TYPE_Q8_0  8u
#define FA_TYPE_IQ4_NL 20u
#define FA_TYPE_BF16 30u
#define FA_TYPE_TURBO2_0 42u
#define FA_TYPE_TURBO3_0 43u
#define FA_TYPE_TURBO4_0 44u

#if defined(BFLOAT16)
#define O_TYPE float
#define O_TYPEV4 vec4
#else
#define O_TYPE FLOAT_TYPE
#define O_TYPEV4 FLOAT_TYPEV4
#endif

// Number of matrix elements per buffer block, derived from the K/V type spec
// constant. F32 is treated as a vec4 "block" of 4 floats. F16 uses block size 1
// and bypasses the dequant path entirely. Quants follow their ggml block sizes.
uint fa_block_elems(uint ty) {
    switch (ty) {
        case FA_TYPE_F32:  return 4u;
        case FA_TYPE_F16:  return 1u;
        case FA_TYPE_Q4_0: return uint(QUANT_K_Q4_0);
        case FA_TYPE_Q4_1: return uint(QUANT_K_Q4_1);
        case FA_TYPE_Q5_0: return uint(QUANT_K_Q5_0);
        case FA_TYPE_Q5_1: return uint(QUANT_K_Q5_1);
        case FA_TYPE_Q8_0: return uint(QUANT_K_Q8_0);
        case FA_TYPE_IQ4_NL: return uint(QUANT_K_IQ4_NL);
        case FA_TYPE_BF16: return 1u;
        case FA_TYPE_TURBO2_0: return uint(QUANT_K_TURBO2_0);
        case FA_TYPE_TURBO3_0: return uint(QUANT_K_TURBO3_0);
        case FA_TYPE_TURBO4_0: return uint(QUANT_K_TURBO4_0);
        default:           return 1u;
    }
}

// QUANT_R_MMQ for FA-eligible K types. Q4_*/Q5_* store two nibbles per byte
// (R==2); Q8_0 stores one byte per element (R==1). Used to derive the number
// of int32s per 32-element block on the MMQ K path: ints_per_block == 8 / R.
uint fa_quant_r_mmq(uint ty) {
    switch (ty) {
        case FA_TYPE_Q4_0: return uint(QUANT_R_Q4_0);
        case FA_TYPE_Q4_1: return uint(QUANT_R_Q4_1);
        case FA_TYPE_Q5_0: return uint(QUANT_R_Q5_0);
        case FA_TYPE_Q5_1: return uint(QUANT_R_Q5_1);
        case FA_TYPE_Q8_0: return uint(QUANT_R_Q8_0);
        default:           return 1u;
    }
}

bool fa_type_needs_shmem(uint ty) {
    switch (ty) {
        case FA_TYPE_IQ4_NL: return true;
        default:             return false;
    }
}

// These can't be `const` globals because GLSL forbids function calls in global
// const initializers, even when the spec constants would let the driver fold
// them. Macros expand at the use site and fold after specialization.
#define BLOCK_SIZE_K fa_block_elems(FaTypeK)
#define BLOCK_SIZE_V fa_block_elems(FaTypeV)
// F16 reads f16 elements directly from the binding; everything else routes
// through dequantize4 / the MMQ helpers to unpack from the packed block layout.
#define USE_DECODE_K (FaTypeK != FA_TYPE_F16)
#define USE_DECODE_V (FaTypeV != FA_TYPE_F16)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))


// Store column zero. This is used to save per-row m and L values for split_k.
ACC_TYPE perElemOpStoreCol0(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    if (r < N && c == 0) {
        uint32_t offset = iq2 + r;
        data_o[o_offset + offset] = D_TYPE(elem);
    }
    return elem;
}

// Load the slope matrix, indexed by Q's dimension 2.
ACC_TYPE perElemOpComputeSlope(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    uint32_t n_head_log2 = p.mask_n_head_log2 & N_LOG2_MASK;

    const ACC_TYPE base = ACC_TYPE(h < n_head_log2 ? p.m0 : p.m1);
    const int      exph = int(h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1);

    return ACC_TYPE(pow(base, ACC_TYPE(exph)));
}

// Load the sink value, indexed by Q's dimension 2.
ACC_TYPE perElemOpGetSink(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    return ACC_TYPE(data_s[h]);
}

uint32_t i, N, KV, split_k_index, Tr, start_j, end_j,
         gqa_iq1, iq2, iq3, rk2, rk3, rv2, rv3, ik2, ik3, iv2, iv3,
         q_stride, k_stride, v_stride, m_stride;

void init_indices()
{
    N = p.N;
    KV = p.KV;

    if (p.k_num > 1) {
        if (p.gqa_ratio > 1) {
            i = 0;
            // batch and split_k share gl_WorkGroupID.x
            gqa_iq1 = gl_WorkGroupID.x / p.k_num;
            split_k_index = gl_WorkGroupID.x % p.k_num;
        } else {
            gqa_iq1 = 0;
            split_k_index = gl_WorkGroupID.x % p.k_num;
            i = gl_WorkGroupID.x / p.k_num;
        }
    } else if (p.gqa_ratio > 1) {
        i = 0;
        gqa_iq1 = gl_WorkGroupID.x;
        split_k_index = 0;
    } else {
        i = gl_WorkGroupID.x;
        gqa_iq1 = 0;
        split_k_index = 0;
    }

    Tr = CEIL_DIV(N, Br);

    start_j = split_k_index * p.split_kv / Bc;
    end_j = CEIL_DIV(min(KV, (split_k_index + 1) * p.split_kv), Bc);

    // When not using grouped query attention, all rows share the same iq2, equal to gl_WorkGroupID.y.
    // When using grouped query attention, each workgroup does gqa_ratio consecutive values of iq2.
    iq2 = gl_WorkGroupID.y * p.gqa_ratio;
    iq3 = gl_WorkGroupID.z;

    // broadcast factors
    rk2 = p.neq2/p.nek2;
    rk3 = p.neq3/p.nek3;

    rv2 = p.neq2/p.nev2;
    rv3 = p.neq3/p.nev3;

    // k indices
    ik3 = iq3 / rk3;
    ik2 = iq2 / rk2;

    // v indices
    iv3 = iq3 / rv3;
    iv2 = iq2 / rv2;

    // nb?1 are already divided by the type size and are in units of elements.
    // When using grouped query attention, Q is indexed by iq2, so the stride
    // should be nb02 (which is in bytes).
    q_stride = p.gqa_ratio > 1 ? (p.nb02 / 4) : p.nb01;
    k_stride = p.nb11;
    v_stride = p.nb21;
    // When using grouped query attention, all rows use the same mask (stride 0).
    // "p.gqa_ratio >> 16" is just a roundabout way of writing zero
    // that prevents the compiler from folding the "&" through the select
    // and breaking the alignment detection.
    m_stride = (p.gqa_ratio > 1) ? (p.gqa_ratio >> 16) : KV;
}

// Bias applied to softmax to stay in fp16 range.
// Based on ggml-cuda issue https://github.com/ggml-org/llama.cpp/issues/18606
const float FATTN_KQ_MAX_OFFSET = 3.0f*0.6931f;

// Store the output when doing grouped query attention.
// Rows index by Q's dimension 2, and the first N rows are valid.
void gqaStore(const in uint32_t r, const in uint32_t c, const in O_TYPEV4 elems, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    uint32_t offset = (iq2 + r) * HSV / 4 + c;
    data_ov4[o_offset + offset] = D_TYPEV4(elems);
}

// ============================================================================
// RERoT-DDVR indexed FlashAttention (fused variant, entry point rerot_main).
//
// Production contract (frozen with the CPU reference
// ggml_compute_forward_flash_attn_ext_rerot in ggml/src/ggml-cpu/ops.cpp):
//   * src0 q_groups F32 [D, n_groups, n_head] holds queries ALREADY rotated to
//     their reader-relative effective phase by the graph
//     (effective = query_virtual + storage_pos - virtual_pos, IMRoPE text
//     position (p, p, p, 0); a TurboQuant build then applies forward WHT, so
//     this kernel rotates nothing itself: re-deriving the delta in-shader
//     would double-rotate. The buffers below ARE the DDVR spans in indexed
//     form: entry = (physical key index, Q-group index)).
//   * src3 entries I32 [2, n_entries], src4 offsets I32 [n_queries+1]: every
//     entry of one query range shares ONE global online softmax (m, L, O
//     accumulation across spans). The layout builder pre-filters entries to
//     the strong-frontier visible set, so this kernel applies no mask.
//   * Turbo2/3/4 + Q4/Q5/Q8/IQ4_NL K/V flow through dequantize4(), exactly
//     like the ordinary scalar FA path (Q is pre-WHT when K is Turbo, per the
//     frozen Q-side order RoPE -> WHT -> dot; K stays at writer storage
//     phase; V is consumed as stored).
//   * GQA via head-index division; head_dim padding flows through HSK/HSV
//     (never assumed 128); sinks / logit-softcap mirror the ordinary path.
//   * Single-row workgroups (host forces Br = 1, D_split = 1,
//     SubGroupSize = 0) and shared-memory-only reductions (no subgroup
//     ops), so this compiles into every scalar FA module variant
//     (fp16 / fp32 / dot2 / int8).
//
// Build note: this block is scalar-module-only. Coopmat1 modules are excluded
// via COOPMAT; coopmat2 modules enable GL_NV_cooperative_matrix2 before
// including this file, which glslc predefines as a macro (the same idiom
// flash_attn_cm2.comp itself uses for GL_NV_cooperative_matrix_decode_vector),
// so they are excluded via that. Ordinary pipelines are unaffected: they use
// entry point main(), never call rerot_main(), and leave RerotMode at 0.
// ============================================================================
#if !defined(COOPMAT) && !defined(GL_NV_cooperative_matrix2)

// constant_id 16 appends the FA specialization list (see
// get_fa_spec_constants): 0 = ordinary FA (default, so existing pipelines keep
// working unchanged), 1 = RERoT-DDVR path (entry point rerot_main).
layout (constant_id = 16) const uint32_t RerotMode = 0;

// Q-group views (F32). Aliased at bindings 0/1/2 exactly like the quant views
// in flash_attn_dequant.glsl; unique block names keep them distinct.
layout (binding = 0) readonly buffer RQ_F32   { float     rerot_q[]; };
layout (binding = 0) readonly buffer RQV4     { vec4      rerot_qv4[]; };
// Direct F16 views for FaTypeK/V == F16 (USE_DECODE_* false), mirroring the
// data_kv4 / data_vv4 declarations of flash_attn.comp.
layout (binding = 1) readonly buffer RK_F16   { float16_t rerot_k[]; };
layout (binding = 1) readonly buffer RK_F16V4 { f16vec4   rerot_kv4[]; };
layout (binding = 2) readonly buffer RV_F16   { float16_t rerot_v[]; };
layout (binding = 2) readonly buffer RV_F16V4 { f16vec4   rerot_vv4[]; };
// DDVR span descriptors: entries[2*e+0] = physical key index,
// entries[2*e+1] = Q-group index; offsets[q]..offsets[q+1] is query q's entry
// range (one global softmax per range).
layout (binding = 7) readonly buffer RE_ENTRIES { int rerot_entries[]; };
layout (binding = 8) readonly buffer RO_OFFSETS { int rerot_offsets[]; };

// Defined by flash_attn_dequant.glsl (included after this file in scalar
// modules); declared here so rerot_main() below can reference it. Modules
// without the definition never call rerot_main().
FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx);

// Push-constant overlay for rerot_main(): the SAME vk_flash_attn_push_constants
// block (128 B, layout unchanged), reinterpreted as built by
// ggml_vk_flash_attn_rerot(): N = n_queries, KV = n_kv_physical, ne1 = Dv,
// ne2 = n_head_q, ne3 = n_queries, neq2 = n_head_q, nek2 = n_head_kv,
// nev2 = n_head_v, nem1 = n_entries, nem2 = n_groups, nb01/nb02 = Q
// group/head strides, nb11/nb12 and nb21/nb22 = K/V key/head strides in
// ordinary units, k_num = entry-split count. max_bias/m0/m1 are unused
// (no ALiBi/mask in the indexed op).

// Horizontal sum in f32. RERoT forces f32 accumulation (the graph pins the op
// to GGML_PREC_F32 and all quantized K/V already require it), so unlike the
// ordinary path there is no f16acc variant to preserve.
float rerot_hsum(vec4 a) {
    return a.x + a.y + a.z + a.w;
}

vec4 rerot_load_k(const in uint key, const in uint dvec, const in uint k_stride, const in uint k_offset) {
    if (USE_DECODE_K == false) {
        return vec4(rerot_kv4[k_offset / 4u + key * k_stride / 4u + dvec]);
    }
    uint coord = key * k_stride * BLOCK_SIZE_K + 4u * dvec;
    return vec4(dequantize4(coord / BLOCK_SIZE_K, coord % BLOCK_SIZE_K, k_offset, BINDING_IDX_K));
}

vec4 rerot_load_v(const in uint key, const in uint dvec, const in uint v_stride, const in uint v_offset) {
    if (USE_DECODE_V == false) {
        return vec4(rerot_vv4[v_offset / 4u + key * v_stride / 4u + dvec]);
    }
    uint coord = key * v_stride * BLOCK_SIZE_V + 4u * dvec;
    return vec4(dequantize4(coord / BLOCK_SIZE_V, coord % BLOCK_SIZE_V, v_offset, BINDING_IDX_V));
}

// Shmem sized for the host-enforced config (Br = 1, D_split = 1): one staged
// Q row, one staged K/V block reused for K then V, plus reduction scratch.
const uint32_t REROT_Q_WORDS = HSK / 4u + 1u;
const uint32_t REROT_D = HSK > HSV ? HSK : HSV;
const uint32_t REROT_KV_STRIDE = REROT_D / 4u + 1u;
shared vec4 rerot_qf[REROT_Q_WORDS];
shared vec4 rerot_kvsh[SHMEM_STAGING != 0 ? Bc * REROT_KV_STRIDE : 1];
shared float rerot_tmpsh[WorkGroupSize];
shared vec4 rerot_tmpv4[WorkGroupSize];
shared vec4 rerot_occlim[LIMIT_OCCUPANCY_SHMEM > 0 ? LIMIT_OCCUPANCY_SHMEM : 1];

// Max columns owned by one thread when Bc entry-columns are dealt strided
// across the workgroup. Spec-constant arithmetic like the ordinary tmpsh_size.
const uint32_t REROT_MAXC = (Bc + WorkGroupSize - 1u) / WorkGroupSize;

// Use -FLT_MAX/2 rather than -inf to reduce the possibility of NaNs, matching
// the ordinary path's convention.
const float REROT_NEG = uintBitsToFloat(0xFEFFFFFFu);

void rerot_main() {
    if (RerotMode != 1u) {
        return;
    }
#ifdef NEEDS_INIT_IQ_SHMEM
    if (fa_type_needs_shmem(FaTypeK) || fa_type_needs_shmem(FaTypeV)) {
        init_iq_shmem(gl_WorkGroupSize);
    }
#endif

    const uint tid = gl_LocalInvocationIndex;
    const uint WGS = gl_WorkGroupSize.x;

    if (LIMIT_OCCUPANCY_SHMEM > 0) {
        // Same occupancy-throttle idiom as the ordinary path.
        rerot_occlim[tid] = vec4(float(tid));

        barrier();

        if (rerot_occlim[tid] == vec4(99999.0)) {
            data_ov4[0] = D_TYPEV4(rerot_occlim[tid]);
        }
    }

    // Grid: x = n_queries * n_splits, y = n_head_q, z = 1 (Br = 1: one query
    // row per workgroup; heads are unfolded, GQA maps h -> kv head below).
    const uint S = p.k_num > 0u ? p.k_num : 1u;
    const uint q = gl_WorkGroupID.x / S;
    const uint s = gl_WorkGroupID.x % S;
    const uint h = gl_WorkGroupID.y;
    if (q >= p.N || h >= p.neq2) {
        return;
    }

    const uint gqa_div = (p.nek2 > 0u) ? (p.neq2 / p.nek2) : 1u;
    const uint vdiv    = (p.nev2 > 0u) ? (p.neq2 / p.nev2) : 1u;
    const uint kh = (gqa_div > 0u) ? (h / gqa_div) : h;
    const uint vh = (vdiv    > 0u) ? (h / vdiv)    : h;

    // This query's global entry range, split across entry-splits. Splits of an
    // empty range stay empty; their partials (L = 0) are identity elements for
    // the split-K reduce.
    const int off_b = rerot_offsets[q];
    const int off_e = rerot_offsets[q + 1u];
    const uint len  = (off_e > off_b && off_b >= 0) ? uint(off_e - off_b) : 0u;
    const uint base = (off_b > 0) ? uint(off_b) : 0u;
    const uint chunk = (len + S - 1u) / S;
    const uint my_b = base + s * chunk;
    const uint my_e = min(base + len, base + (s + 1u) * chunk);

    const uint HSK4 = HSK / 4u;
    const uint HSV4 = HSV / 4u;
    const uint q_grp_stride = p.nb01 / 4u;
    const uint k_stride = p.nb11;
    const uint v_stride = p.nb21;
    const uint k_offset = (kh * p.nb12) / FaBlockBytesK;
    const uint v_offset = (vh * p.nb22) / FaBlockBytesV;

    // Single-row online accumulators: one global softmax over the whole entry
    // range (never per-span/per-block softmax).
    float Lf = 0.0f;
    float Mf = REROT_NEG;
    vec4 Of[HSV4];
    for (uint d = 0; d < HSV4; ++d) {
        Of[d] = vec4(0.0);
    }

    float scol[REROT_MAXC];
    bool cvalid[REROT_MAXC];

    // Group runs: entries of one Q-group share the staged Q row, so each run
    // stages Q once. Run detection is uniform across the workgroup.
    uint run_b = my_b;
    while (run_b < my_e) {
        const uint g = uint(rerot_entries[2u * run_b + 1u]);
        uint run_e = run_b + 1u;
        while (run_e < my_e && uint(rerot_entries[2u * run_e + 1u]) == g) {
            ++run_e;
        }
        // Out-of-range groups are excluded (treated as -inf); the layout
        // builder never emits them, this only guards against garbage.
        const bool run_ok = (g < p.nem2);

        if (run_ok) {
            const uint q_base = g * q_grp_stride + (h * p.nb02) / 4u;
            for (uint d = tid; d < HSK4; d += WGS) {
                rerot_qf[d] = rerot_qv4[q_base + d] * p.scale;
            }
        }
        barrier();

        for (uint b = run_b; b < run_e; b += Bc) {
            const uint b_end = min(b + Bc, run_e);

            if (SHMEM_STAGING != 0) {
                barrier();
                for (uint idx = tid; idx < Bc * HSK4; idx += WGS) {
                    const uint c = idx / HSK4;
                    const uint d = idx % HSK4;
                    const uint e = b + c;
                    vec4 Kv = vec4(0.0);
                    if (e < b_end && run_ok) {
                        const uint key = uint(rerot_entries[2u * e]);
                        if (KV_bounds_check == false || key < p.KV) {
                            Kv = rerot_load_k(key, d, k_stride, k_offset);
                        }
                    }
                    rerot_kvsh[c * REROT_KV_STRIDE + d] = Kv;
                }
                barrier();
            }

            // Scores for strided-owned columns.
            uint ncol = 0u;
            for (uint c = tid; c < Bc; c += WGS) {
                const uint e = b + c;
                float sc = REROT_NEG;
                bool ok = false;
                if (e < b_end && run_ok) {
                    const uint key = uint(rerot_entries[2u * e]);
                    if (KV_bounds_check == false || key < p.KV) {
                        vec4 acc = vec4(0.0);
                        if (SHMEM_STAGING != 0) {
                            for (uint d = 0u; d < HSK4; ++d) {
                                acc = fma(rerot_qf[d], rerot_kvsh[c * REROT_KV_STRIDE + d], acc);
                            }
                        } else {
                            for (uint d = 0u; d < HSK4; ++d) {
                                acc = fma(rerot_qf[d], rerot_load_k(key, d, k_stride, k_offset), acc);
                            }
                        }
                        sc = rerot_hsum(acc);
                        if (LOGIT_SOFTCAP) {
                            sc = p.logit_softcap * tanh(sc);
                        }
                        ok = true;
                    }
                }
                scol[ncol] = sc;
                cvalid[ncol] = ok;
                ++ncol;
            }

            // Block max across threads (shared-memory tree; no subgroup ops).
            float local_max = REROT_NEG;
            for (uint t = 0u; t < ncol; ++t) {
                local_max = max(local_max, scol[t]);
            }
            rerot_tmpsh[tid] = local_max;
            barrier();
            for (uint st = WGS / 2u; st > 0u; st >>= 1u) {
                if (tid < st) {
                    rerot_tmpsh[tid] = max(rerot_tmpsh[tid], rerot_tmpsh[tid + st]);
                }
                barrier();
            }
            const float bmax = rerot_tmpsh[0];
            const float Mold = Mf;
            Mf = max(bmax, Mold);
            const float eM = exp(Mold - Mf);
            Lf *= eM;
            for (uint d = 0u; d < HSV4; ++d) {
                Of[d] *= eM;
            }

            if (SHMEM_STAGING != 0) {
                barrier();
                for (uint idx = tid; idx < Bc * HSV4; idx += WGS) {
                    const uint c = idx / HSV4;
                    const uint d = idx % HSV4;
                    const uint e = b + c;
                    vec4 Vv = vec4(0.0);
                    if (e < b_end && run_ok) {
                        const uint key = uint(rerot_entries[2u * e]);
                        if (KV_bounds_check == false || key < p.KV) {
                            Vv = rerot_load_v(key, d, v_stride, v_offset);
                        }
                    }
                    rerot_kvsh[c * REROT_KV_STRIDE + d] = Vv;
                }
                barrier();
            }

            uint t = 0u;
            for (uint c = tid; c < Bc; c += WGS) {
                const uint e = b + c;
                const float Pw = cvalid[t] ? exp(scol[t] - Mf) : 0.0;
                ++t;
                Lf += Pw;
                const vec4 Pwv = vec4(Pw);
                for (uint d = 0u; d < HSV4; ++d) {
                    vec4 Vv;
                    if (SHMEM_STAGING != 0) {
                        Vv = rerot_kvsh[c * REROT_KV_STRIDE + d];
                    } else {
                        Vv = vec4(0.0);
                        if (e < b_end && run_ok) {
                            const uint key = uint(rerot_entries[2u * e]);
                            if (KV_bounds_check == false || key < p.KV) {
                                Vv = rerot_load_v(key, d, v_stride, v_offset);
                            }
                        }
                    }
                    Of[d] = fma(Pwv, Vv, Of[d]);
                }
            }
        }

        barrier();
        run_b = run_e;
    }

    // Reduce per-thread partials across the workgroup (same rescaling algebra
    // as the ordinary cross-thread reduce, over disjoint column sets).
    rerot_tmpsh[tid] = Mf;
    barrier();
    for (uint st = WGS / 2u; st > 0u; st >>= 1u) {
        if (tid < st) {
            rerot_tmpsh[tid] = max(rerot_tmpsh[tid], rerot_tmpsh[tid + st]);
        }
        barrier();
    }
    {
        const float Mnew = rerot_tmpsh[0];
        const float eM = exp(Mf - Mnew);
        Mf = Mnew;
        Lf *= eM;
        for (uint d = 0u; d < HSV4; ++d) {
            Of[d] *= eM;
        }
    }
    barrier();
    rerot_tmpsh[tid] = Lf;
    barrier();
    for (uint st = WGS / 2u; st > 0u; st >>= 1u) {
        if (tid < st) {
            rerot_tmpsh[tid] += rerot_tmpsh[tid + st];
        }
        barrier();
    }
    Lf = rerot_tmpsh[0];
    barrier();
    for (uint d = 0u; d < HSV4; ++d) {
        rerot_tmpv4[tid] = Of[d];
        barrier();
        for (uint st = WGS / 2u; st > 0u; st >>= 1u) {
            if (tid < st) {
                rerot_tmpv4[tid] += rerot_tmpv4[tid + st];
            }
            barrier();
        }
        Of[d] = rerot_tmpv4[0];
        barrier();
    }

    if (S == 1u && (p.mask_n_head_log2 & SINK_ENABLE_BIT) != 0u) {
        // Sink handling mirrors the ordinary path (split-K + sinks forces the
        // single-split path host-side, so the reduce never sees sinks).
        const float sink = float(data_s[h]);
        float ms = 1.0f;
        float vs = 1.0f;
        if (sink > Mf) {
            ms = exp(Mf - sink);
            for (uint d = 0u; d < HSV4; ++d) {
                Of[d] *= ms;
            }
        } else {
            vs = exp(sink - Mf);
        }
        Lf = Lf * ms + vs;
    }

    // Empty ranges (Lf == 0) yield zeros, matching the CPU reference.
    const float rcp = (Lf == 0.0f) ? 0.0f : (1.0f / Lf);

    if (S == 1u) {
        // Contiguous dst [Dv, n_head_q, n_queries] (asserted host-side).
        const uint o_base = ((q * p.ne2) + h) * HSV4;
        for (uint d = tid; d < HSV4; d += WGS) {
            data_ov4[o_base + d] = D_TYPEV4(Of[d] * rcp);
        }
        return;
    }

    // Entry-split partials in flash_attn_split_k_reduce layout with flattened
    // rows n = h + n_head_q * q (ne1 = n_queries * n_head_q, ne2 = ne3 = 1).
    const uint Nflat = p.ne2 * p.N;
    const uint n = h + p.ne2 * q;
    const uint o_base = HSV * Nflat * s + HSV * n;
    for (uint d = tid; d < HSV4; d += WGS) {
        data_ov4[o_base / 4u + d] = D_TYPEV4(Of[d]);
    }
    if (tid == 0u) {
        const uint lm_base = HSV * Nflat * S + 2u * Nflat * s + n;
        data_o[lm_base] = D_TYPE(Lf);
        data_o[lm_base + Nflat] = D_TYPE(Mf);
    }
}

#endif // !defined(COOPMAT) && !defined(GL_NV_cooperative_matrix2)
