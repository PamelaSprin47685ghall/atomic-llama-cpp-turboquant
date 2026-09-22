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
//   [Q2] Span-view reference. A reader view is address ranges + one phase
//        per span + a visibility boundary — not per-key entries. Within a
//        span of unit storage/virtual step the DDVR phase is constant, so ONE
//        effective query serves the whole span, and causal truncation is one
//        cut. The span count — not the token count — is the view's real
//        description complexity.
//
//   [Q4] Structure/numeric separation for reader run orders. The run ORDER
//        is structure (changes on stage/FRAME/run-set events); lengths,
//        watermarks and virtual starts are numeric (prefix sums over the
//        known order, incrementally updatable without re-sorting).
//
//   [Q8] Skip-block bounds — APPROXIMATE research route, explicitly gated:
//        no exact finite sufficient statistic exists for frozen softmax
//        contexts (Z(q) depends on the full key set); only sound upper
//        bounds on skipped mass and output deviation are provided.
//
//   [Q9] Joint sampler contract. Per-pen RNG streams seeded by (base, pen)
//        make each pen's draws independent of cohort size and row order;
//        rows apply temperature -> top-k -> top-p with lowest-index
//        tie-break. Greedy path reduces per-block maxima without
//        materializing full logits rows on the host.
//
//   [Q10] Frontier-grid joint verification. Draft grids are verified against
//        FRONTIER dependencies (peers' committed tokens), never per row;
//        a rejected pen publishes a replacement and dependents of a rejected
//        pen consume its REPLACEMENT, not its draft. The naive per-row engine
//        is kept as the executable counterexample of what goes wrong.
//
// Nothing in this header changes execution semantics: STRONG frontier
// visibility, DDVR phases, and lane-local recurrence are inputs, not
// redefinitions. Approximate schemes (skip-block bounds, joint multi-step
// speculation) are explicitly out of scope here.

#pragma once

#include <cstdint>
#include <vector>
#include <utility>

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

// ---------------------------------------------------------------------------
// [Q2] Span-view reference: ranges + phase + visibility boundary
// ---------------------------------------------------------------------------

// One span of a reader view: a half-open virtual interval with CONSTANT DDVR
// phase (storage_pos - virtual_pos) over unit steps, plus a visibility
// boundary. This is the description a reader view needs per span — not one
// entry per key.
struct llama_rerot_span_view {
    uint32_t begin = 0;        // first virtual position covered
    uint32_t len = 0;           // span length (>= 1)
    int64_t phase = 0;          // DDVR phase: storage_pos - virtual_pos, constant
};

// Effective RoPE position for a reader whose query sits at virtual position
// `q_v`: within a span, q_v + s_j - v_j = q_v + phase for every j, so the
// whole span shares ONE effective query (RERoT.md §2.4). Returns q_v + span
// phase. The span's j-dependence cancels — this is the identity that kills
// per-key entry construction.
int64_t llama_rerot_span_effective_pos(int64_t query_virtual_pos, const llama_rerot_span_view & span);

// Causal cut for a unit-step span: the largest virtual position a query at
// q_v may read, i.e. min(q_v, span_end - 1). The span's visible prefix is
// [begin, begin + visible_len). Returns visible length (0 if fully future).
// One comparison replaces a per-key causal mask over the span.
uint32_t llama_rerot_span_causal_len(const llama_rerot_span_view & span, int64_t query_virtual_pos);

// Fragmentation metric: view description cost should scale with the number
// of spans (runs/pages), not with history token count. Returns the fraction of
// the covered length carried by spans of length >= min_long (0.0 when empty).
// A well-maintained write layout keeps this close to 1.
double llama_rerot_span_long_fraction(const std::vector<llama_rerot_span_view> & spans, uint32_t min_long);

// ---------------------------------------------------------------------------
// [Q4] Structure/numeric separation over a frozen run order
// ---------------------------------------------------------------------------

// Reader run-order structure: WHICH runs precede which (topology), frozen
// until a structural event (stage start, FRAME change, run-set change, KV
// migration, recovery, capacity crossing) redefines it.
struct llama_rerot_run_order {
    std::vector<uint32_t> run_ids; // topological run order (structure)
};

// Numeric layer over a frozen order: lengths + watermarks. Virtual start of
// run g is the prefix sum of the CURRENT lengths of all runs preceding it in
// the order — a numeric update when a run grows, never a re-sort.
struct llama_rerot_run_lengths {
    std::vector<uint32_t> len;    // current visible length per run (by index into run_order.run_ids)
};

// Compute virtual starts for every run from the frozen order + current
// lengths (prefix sums). Returns [n_runs] virtual starts. Throws on size
// mismatch. This is the whole numeric path: no topological work.
std::vector<int64_t> llama_rerot_virtual_starts(
    const llama_rerot_run_order & order,
    const llama_rerot_run_lengths & lengths);

// Incremental numeric update: run `grown_run` gains `delta` tokens. Returns
// the new virtual starts. Only runs AFTER grown_run in the order shift; the
// update is O(n_runs) arithmetic with zero structural work. Throws on bad
// index or overflow.
std::vector<int64_t> llama_rerot_virtual_starts_after_growth(
    const llama_rerot_run_order & order,
    const llama_rerot_run_lengths & lengths,
    size_t grown_run,
    uint32_t delta);

// Structure signature: changes ONLY on structural events. Two views with the
// same signature share the same run order (so numeric updates suffice);
// different signatures require re-definition. Content hash over run ids.
uint64_t llama_rerot_run_order_signature(const llama_rerot_run_order & order);

// ---------------------------------------------------------------------------
// [Q8] Skip-block bounds (approximate research route — NOT exact)
// ---------------------------------------------------------------------------

// Sound upper bound on the unnormalized attention mass a skipped block can
// contribute, from a per-block key center c and radius r:
//   q . k_j <= q . c + |q| * r        (Cauchy-Schwarz on the residual)
//   Z_skip <= n * exp(scale * (q . c + |q| * r))
// All phases, scales, softcaps, quantization domains and visibility must be
// accounted for by the caller when choosing c/r — a loose bound saves nothing.
// Throws on empty block or negative radius.
double llama_rerot_skip_mass_bound(
    const std::vector<double> & query,
    const std::vector<double> & center,   // [head_dim] block key center
    double radius,                        // bound on |k_j - c| for all j
    uint32_t n_block,                     // number of keys in the skipped block
    double scale);

// Output deviation bound: o = convex combination of keep/skip parts gives
//   ||o - o_keep|| <= 2 * V_max * delta,  delta = Z_skip / (Z_keep + Z_skip)
// Returns {delta, bound} where bound = 2 * v_max * delta. Throws on
// non-finite or non-positive masses.
std::pair<double, double> llama_rerot_skip_output_bound(
    double z_keep,
    double z_skip_bound,
    double v_max);   // bound on |v_j| (any norm; caller must supply a valid one)

// ---------------------------------------------------------------------------
// [Q9] Joint sampler contract (per-pen RNG streams, row-order independence)
// ---------------------------------------------------------------------------

// One row of the joint sampler: temperature -> top-k -> top-p, then a draw
// from the row's OWN RNG stream (never a shared cohort stream). Ties break to
// the LOWEST token index. `rng_state` is advanced in place; identical inputs
// and identical (base_seed, pen) reproduce identical draws.
uint32_t llama_rerot_joint_sample_row(
    const std::vector<double> & logits,  // [vocab] one pen's row
    double temperature,
    uint32_t top_k,
    double top_p,
    uint64_t & rng_state);               // per-pen stream state (xorshift64*)

// Derive a per-pen stream seed from (base_seed, pen): different pens get
// independent streams; the same pen keeps its stream across cohort-size and
// row-order changes (the contract that makes joint sampling trajectory-safe).
uint64_t llama_rerot_joint_sample_seed(uint64_t base_seed, uint32_t pen);

// Greedy joint head: per-block partial maxima reduce to per-row argmax with
// lowest-index tie-break, without materializing full rows on the host.
// `block_maxima` is [n_blocks][vocab] per-block maxima; returns [n_rows]
// argmax tokens. Throws on empty input or mismatched block sizes.
std::vector<uint32_t> llama_rerot_joint_argmax_rows(
    const std::vector<std::vector<double>> & block_maxima);

// ---------------------------------------------------------------------------
// [Q10] Frontier-grid joint verification
// ---------------------------------------------------------------------------

// A draft grid: pens x horizon. draft[pen][h] is pen's proposed token at
// frontier step h. The verification question: which cells survive, given
// that column h+1 of pen i may depend on column h of OTHER pens (STRONG
// frontier semantics: peers' COMMITTED tokens, not drafts).
struct llama_rerot_draft_grid {
    uint32_t n_pens = 0;
    uint32_t horizon = 0;
    std::vector<uint32_t> draft;    // [n_pens * horizon]
    std::vector<uint32_t> truth;    // [n_pens * horizon] oracle committed tokens
};

// Verification outcome per pen: accepted prefix length within the grid, and
// (if < horizon) the replacement token the pen must publish instead of the
// draft (the first rejected draft cell's truth).
struct llama_rerot_grid_verdict {
    std::vector<uint32_t> accepted;     // [n_pens] accepted prefix length per pen
    std::vector<uint32_t> replacement;   // [n_pens] truth token at the first rejection (valid iff accepted < horizon)
};

// Dependency-tracked joint verification: cell (pen, h) survives only if
// every cell it READS survived. Reads are declared per column: reads[h] is
// the list of (reader_pen -> source_pen) edges for column h (the source is
// read at column h-1's commit boundary, STRONG barrier-after). A pen's cell
// at column h is accepted iff the pen's own prefix through h-1 is accepted
// AND every source pen's cell at column h-1 was accepted (or the source is
// the pen itself). Rejected pens publish replacements; dependents then
// consume replacements, NOT drafts — the naive per-row engine that ignores
// this is the executable counterexample (see test).
llama_rerot_grid_verdict llama_rerot_verify_grid(
    const llama_rerot_draft_grid & grid,
    const std::vector<std::vector<std::pair<uint32_t, uint32_t>>> & reads); // [horizon] edges

// Naive per-row verification (WRONG under cross-pen dependencies): each pen
// accepts its own draft prefix against truth independently. Reference only —
// the counterexample engine. Same signature as the correct one.
llama_rerot_grid_verdict llama_rerot_verify_grid_naive(const llama_rerot_draft_grid & grid);

// ---------------------------------------------------------------------------
// [R02] Live Q-prep contract (C07): gather + effective-position RoPE over
// the LIVE group prefix only. This is the input contract a fused GPU kernel
// must implement; the old chain (GET_ROWS over the padded bucket + ROPE over
// the full bucket) computes the same active rows and wastes the padding.
// ---------------------------------------------------------------------------

struct llama_rerot_q_prep_contract {
    // Inputs
    int64_t head_dim = 0;    // rotary-aware head dim (raw head dim)
    int64_t heads    = 0;    // query heads
    int64_t n_tokens = 0;    // source token count (raw Q rows)
    int64_t capacity = 0;    // group capacity bucket (32-aligned); groups below
    int64_t active   = 0;    // live groups; MUST be 0 <= active <= capacity
    int64_t n_pos    = 1;    // RoPE coordinates: 1 (NORMAL/NEOX) or 4 (MROPE/IMROPE)
    int64_t n_rot    = 0;    // rotary dimensions (<= head_dim)
    int     rope_mode = 0;   // GGML_ROPE_TYPE_*
    float   freq_base  = 10000.0f;
    float   freq_scale = 1.0f;
    float   ext_factor = 0.0f;   // YaRN: 0 = disabled
    float   attn_factor = 1.0f;
    float   beta_fast   = 32.0f;
    float   beta_slow   = 1.0f;
    int64_t n_ctx_orig = 0;      // context length for YaRN correction
    const float * freq_factors = nullptr;  // [n_rot/2] optional per-dim factors
    const int   * mrope_sections = nullptr; // [4] for MROPE/IMROPE

    // Layout invariants (the protocol):
    //  - q_raw: row-major [head_dim, heads, n_tokens], F32.
    //  - q_indices[i] for i in [0, active) picks the token each live group
    //    gathers from q_raw; values in [0, n_tokens). Duplicates are legal
    //    (a token may serve several groups).
    //  - q_pos: per group per coordinate effective position, strided by
    //    capacity: q_pos[coord * capacity + i] (capacity stride, matching
    //    fill_spans; the padded suffix is never read).
    //  - Output: row-major [head_dim, heads, capacity], F32, capacity
    //    stride. ONLY [0, active) groups are written with rotated queries;
    //    [active, capacity) MUST be poison (NaN) so any consumer reading
    //    inactive rows fails loudly.
    //  - Replay contract: for a fixed (capacity, active) pair the kernel
    //    must be replayable (same inputs -> same outputs) across frontiers;
    //    active may change between frontiers without redefinition.
};

// CPU reference for the R02 live Q-prep candidate: gather + RoPE over only
// the active prefix, poison the padding. This is the equivalence target the
// fused kernel must match (same numerical mode, atol/rtol vs the old chain).
// Throws std::invalid_argument on contract violations (active > capacity,
// q_indices out of range, unsupported rope mode for this reference, n_pos
// mismatch).
//
// NOTE: the rotation math is implemented with the production ggml rope
// helpers so the reference shares the math but independently expresses the
// active/stride/gather structure (independent of GET_ROWS/ROPE graph ops).
// YaRN ext_factor != 0 and MROPE/IMROPE section modes are exercised by the
// caller's tests; unsupported combinations must route the fused candidate
// back to the old chain.
std::vector<float> llama_rerot_q_prep_reference(
    const float * q_raw,
    const int32_t * q_indices,
    const int32_t * q_pos,
    const llama_rerot_q_prep_contract & contract);

// RoPE-mode support table for the R02 candidate (declarative, single
// source for the future Vulkan dispatch gate): returns the supported rope
// modes (GGML_ROPE_TYPE_* values) and whether the reference covers them.
std::vector<int> llama_rerot_q_prep_supported_modes();
