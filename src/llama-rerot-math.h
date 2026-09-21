// llama-rerot-math.h — pure mathematical reference layer for the RERoT
// compute-organization research line (2026-09-21 round).
//
// Scope: exact algebraic identities only. Every helper here is a reference
// formulation of work the production path currently performs in a more
// expensive representation. They are deliberately backend-neutral, allocation
// explicit, and free of llama_kv_cells / server / graph state, so they can be
// used (a) as FP64 oracles in tests, (b) as the mathematical contract for any
// future GPU kernel, and (c) as the equivalence target a faster representation
// must match before it is allowed into the decode path.
//
// The four families (numbers refer to the 2026-09-21 research questions):
//
//   [Q3] Multi-reader shared-KV block attention. One physical K/V block is
//        consumed once while producing an independent (m, z, u) attention
//        state per reader; states merge per reader into one global softmax.
//        Readers share the *data supply*, never the softmax statistics.
//
//   [Q5] GDN lane-local state as common base + per-lane low-rank increments.
//        S_i = a_i * B + U_i * V_i^T with the exact gated-delta transition
//        preserved. No lane's update ever enters another lane; the base is
//        read-only shared data. This is NOT shared-RBB and NOT state merging.
//
//   [Q6] Known-token chunk recurrence for GDN. The affine recurrence
//        S_t = A_t S_{t-1} + C_t composes over a chunk of *already known*
//        (teacher-forced) tokens into one compact (M, Y) WY-style transition.
//        It is a replay identity for fixed entries / known candidate blocks;
//        it never predicts unknown future tokens and never relaxes per-row
//        attention causality (attention still runs its own per-row views).
//
//   [Q7] PQ2_0 two-bit weight inner product as bit-plane subset sums. The
//        code {0,1,2,3} -> {-1,0,+1,+2} makes the per-block product a two
//        bit-plane subset sum minus the plain activation sum, with the block
//        scale factored out. Bit-exact with ggml's integer accumulation.
//
// Nothing in this header changes execution semantics: STRONG frontier
// visibility, DDVR phases, and lane-local recurrence are inputs, not
// redefinitions. Approximate schemes (skip-block bounds, joint multi-step
// speculation) are explicitly out of scope here.

#pragma once

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// [Q3] Shared-KV multi-reader block attention
// ---------------------------------------------------------------------------

// Per-reader online-softmax state. One instance per (reader, block); merging
// happens per reader only, so each reader keeps an independent global softmax.
struct llama_rerot_attn_state {
    double m = 0.0;            // running max score (valid once used)
    double z = 0.0;            // running exp-mass
    std::vector<double> u;      // running weighted value sum [value_dim]; empty = unused

    bool empty() const { return u.empty(); }
};

// Merge block state `src` into `dst`:
//   m = max(m_dst, m_src)
//   z = exp(m_dst - m) * z_dst + exp(m_src - m) * z_src
//   u = exp(m_dst - m) * u_dst + exp(m_src - m) * u_src
// Throws std::invalid_argument on value-dim mismatch or empty/non-empty mismatch.
void llama_rerot_attn_state_merge(llama_rerot_attn_state & dst, const llama_rerot_attn_state & src);

// Consume one physical K/V block for a set of readers, producing one
// independent state per reader. `keys`/`values` hold the block's rows
// (row-major: n_block keys of head_dim, then n_block values of value_dim).
// `queries` holds n_readers queries of head_dim each, ALREADY RoPE'd at each
// reader's effective position for THIS block — the caller owns DDVR phases
// (see phase_bias in llama_rerot_table_fragment); this function never
// re-phases. `reader_visible` marks which readers may see the block
// (visibility boundary); invisible readers get an empty state.
//
// This is the mathematical contract for "one shared K/V load serves many
// readers": K/V bytes are read once, dot products are per (reader, key), and
// each reader keeps its own (m, z, u). Returns n_readers states (empty for
// readers with no visible block).
// Throws std::invalid_argument on size mismatches.
std::vector<llama_rerot_attn_state> llama_rerot_shared_block_attention(
    const std::vector<double> & queries,         // [n_readers][head_dim]
    const std::vector<double> & keys,            // [n_block][head_dim]
    const std::vector<double> & values,          // [n_block][value_dim]
    uint32_t n_readers,
    uint32_t head_dim,
    uint32_t value_dim,
    const std::vector<uint8_t> & reader_visible, // [n_readers] 0/1
    double scale);

// Finalize one reader's merged state to o = u / z. Throws if the state is
// unused, empty, or z == 0.
std::vector<double> llama_rerot_attn_state_output(const llama_rerot_attn_state & state);

// ---------------------------------------------------------------------------
// [Q5] GDN common base + per-lane low-rank increments
// ---------------------------------------------------------------------------

// S = a * B + U V^T with U in R^{d_k x r} (row-major [d_k * r]) and V in
// R^{d_v x r} (row-major [d_v * r]).
struct llama_rerot_gdn_lowrank_state {
    double a = 0.0;
    std::vector<double> u;   // [d_k * r]
    std::vector<double> v;   // [d_v * r]
    uint32_t r = 0;
};

// Shared-base projection B^T x for many vectors at once: one pass over the
// shared base serves every consumer (all lanes' k_i / q_i). `x_vectors` is
// [n_vectors][d_k]; `base` is [d_k * d_v] row-major. Returns
// [n_vectors][d_v] = B^T x. Throws on size mismatch.
std::vector<double> llama_rerot_gdn_base_project(
    const std::vector<double> & base,
    const std::vector<double> & x_vectors,
    uint32_t d_k,
    uint32_t d_v);

// One gated-delta step on the factored state (RERoT.md §2.3):
//   Sbar  = alpha * S
//   delta = v - Sbar^T k
//   S'    = Sbar + beta * k * delta^T
// Factored update: a' = alpha * a, every old U column scales by alpha, and
// one new rank-1 term (beta * k) delta^T appends. `base_proj_k` is [d_v] =
// B^T k (shared-base projection, computed once per step per lane and reused
// across every consumer of the base). Throws on size mismatch.
llama_rerot_gdn_lowrank_state llama_rerot_gdn_lowrank_step(
    const llama_rerot_gdn_lowrank_state & state,
    const std::vector<double> & k,          // [d_k]
    const std::vector<double> & v,          // [d_v]
    double alpha,
    double beta,
    const std::vector<double> & base_proj_k);

// Output o = S^T q without materializing S:
//   o = a * (B^T q) + V (U^T q)
// `base_proj_q` is [d_v] = B^T q. Throws on size mismatch.
std::vector<double> llama_rerot_gdn_lowrank_output(
    const llama_rerot_gdn_lowrank_state & state,
    const std::vector<double> & q,           // [d_k]
    const std::vector<double> & base_proj_q); // [d_v]

// Materialize S = a * B + U V^T (oracle / rank-threshold checks).
// [d_k * d_v] row-major. Throws on size mismatch.
std::vector<double> llama_rerot_gdn_lowrank_dense(
    const llama_rerot_gdn_lowrank_state & state,
    const std::vector<double> & base,
    uint32_t d_k,
    uint32_t d_v);

// ---------------------------------------------------------------------------
// [Q6] Known-token chunk recurrence (WY-style compact transition)
// ---------------------------------------------------------------------------

// Compact chunk transition: S_T = M * S_0 + Y for a chunk of known tokens,
// with M in R^{d_k x d_k} and Y in R^{d_k x d_v} (both row-major). For the
// gated delta rule, M factors as (prod alphas) * (I - K B K^T)-shaped low-rank
// correction; the fold below builds the exact M and Y.
struct llama_rerot_gdn_chunk {
    std::vector<double> m;   // [d_k * d_k]
    std::vector<double> y;   // [d_k * d_v]
};

// Fold a chunk of KNOWN (teacher-forced) tokens into (M, Y) such that for ANY
// S_0: llama_rerot_gdn_chunk_apply == running the scalar §2.3 recurrence
// step-by-step. Inputs: ks [T][d_k], vs [T][d_v], alphas [T], betas [T].
// Throws on size mismatch or T == 0.
//
// Derivation (DeltaNet WY form, adapted to per-token alpha/beta):
//   S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
//   is affine: S_T = Phi S_0 + Y with Phi = G * M, G = prod alphas, and M
//   the product of the rank-1 corrections (I - beta_t k_t k_t^T).
// WY factorization of M (row-indexed beta — verified by induction on T):
//   M = I - K W K^T,   W = (I + L)^{-1} diag(beta),
//   L[j, c] = beta_j (k_j . k_c)  for j > c (strictly lower).
// Y is the zero-state arm, accumulated exactly by the same recurrence.
// M is returned materialized for the reference; a production kernel applies
// it as the rank-T operator x -> G*(x - K (W (K^T x))) — and, the actual
// RERoT payoff, applies the SAME folded chunk to many lanes' states as one
// matrix product over shared fixed-entry replay work.
llama_rerot_gdn_chunk llama_rerot_gdn_chunk_fold(
    uint32_t d_k,
    uint32_t d_v,
    const std::vector<double> & ks,
    const std::vector<double> & vs,
    const std::vector<double> & alphas,
    const std::vector<double> & betas);

// Apply a folded chunk to a dense state: S_out = M * S_in + Y.
// [d_k * d_v] in/out, row-major. Throws on size mismatch.
std::vector<double> llama_rerot_gdn_chunk_apply(
    const llama_rerot_gdn_chunk & chunk,
    const std::vector<double> & s_in,
    uint32_t d_k,
    uint32_t d_v);

// Step-by-step scalar reference (oracle) for the same recurrence.
std::vector<double> llama_rerot_gdn_steps_reference(
    const std::vector<double> & s0,          // [d_k * d_v]
    uint32_t d_k,
    uint32_t d_v,
    const std::vector<double> & ks,
    const std::vector<double> & vs,
    const std::vector<double> & alphas,
    const std::vector<double> & betas);

// ---------------------------------------------------------------------------
// [Q7] PQ2_0 bit-plane subset-sum inner product
// ---------------------------------------------------------------------------

// Bit-plane inner product for one PQ2_0 block (128 weights) against an
// activation vector x:
//
//   code q_j in {0,1,2,3} decodes to (q_j - 1) in {-1, 0, +1, +2}
//   q_j = b0_j + 2 * b1_j   (b0 = low bit of the 2-bit code, b1 = high bit)
//   sum_j (q_j - 1) x_j = sum_{b0_j=1} x_j + 2 sum_{b1_j=1} x_j - sum_j x_j
//
// `qs` points at the 32 bytes of 2-bit codes (4 weights per byte, low 2 bits
// first). `x` holds n == 128 activations in the same order as the codes.
// Returns the unscaled block sum; the caller multiplies by the block scale d.
// Double accumulation keeps the identity checkable against float paths.
double llama_rerot_pq2_bitplane_dot(const uint8_t * qs, const double * x, uint32_t n);

// Same identity as a 16-entry LUT per 4 activations (T-MAC-style): for each
// group of 4 activations build T[m] = sum_{j : m_j = 1} x_j over the 16
// submasks, then the group contributes T[m0] + 2 T[m1] - sum_j x_j where m0/m1
// are the group's low/high bit-plane masks. Returns the block's unscaled sum.
double llama_rerot_pq2_lut_dot(const uint8_t * qs, const double * x, uint32_t n);
