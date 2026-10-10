#!/usr/bin/env python3
"""Necessity certificate v4 for the fixed TP5 single-user generation task.

v4 supersedes v3 and fixes three soundness defects found by audit:

  DEFECT-1 (factual): DENSE_BYTES_PER_LAYER was 90.8 MB; the audited GGUF
  inventory (necessity-restart-20261009/audit.json, 1222 tensors) gives
  63.82 MB average dense bytes per layer (36 qkv layers 63.79 MB, 12 q/k
  layers 61.65 MB). v3 overstated the dense term by 42%.

  DEFECT-2 (soundness): v3 computed the L2 retention credit as
  L2/pool where pool = 11.9 GB is the FULL per-GPU weight pool, then
  applied that 1.07% hit fraction to the ACTIVE per-token set. For a
  valid lower bound the implementation must receive MAXIMUM caching
  credit: the L2 can hold up to 128 MB of whatever per-token active
  bytes it chooses (dense or expert). The correct generous credit is
  l2_hit = min(1, L2 / active_per_gpu_per_token). v3's smaller credit
  made the bound tighter than provable -> invalid as a lower bound.

  DEFECT-3 (soundness, the largest): v3 stated "bound is per target
  row, so per-public-token bound equals it" under the perfect-acceptance
  MTP relaxation. That is false for R3 (and any weight-traffic term):
  with MTP a cycle verifies h+1 rows in ONE batched pass, and the active
  weight bytes are read ONCE per cycle, not once per row. Under the
  allowed set (any draft policy, perfect acceptance, unlimited h) the
  per-public-token DRAM weight traffic amortizes toward zero. Only terms
  that scale with rows (ALU FLOPs) or with public tokens (output
  emission, causal cut per row) survive.

v4 therefore reports a SPLIT certificate instead of one number:

  L_global  = max(R1, R2, R4)  -- valid for the FULL allowed set A
              (R3 excluded because it does not survive MTP amortization).
              Binding term: R4 (FP32 ALU), ~6.7e3 tok/s ceiling. Loose
              but sound.

  L_nomtp   = max(R1, R2, R3, R4) -- valid only for the declared
              conditional class A_nomtp (implementations that run one
              target row per public token, i.e. no speculative
              batching). This is the honest bound for ranking no-MTP
              arms. ~7.3e2 tok/s ceiling with the corrected dense
              bytes and L2 credit.

Both bounds use spec capacity ceilings (512 GB/s DRAM, 80.85 TFLOP/s
aggregate FP32, 32 GB/s PCIe, 25 GB/s host readback), maximal overlap
across distinct resources, and only information-forced byte/FLOP counts.
No measured latency of any implementation enters the bound.

The certificate deliberately does NOT claim U/L convergence: the
demonstrated U (see necessity-status.json) is two orders of magnitude
above L_nomtp. Closing that gap requires either tighter provable
latency coupling (not available under the fully relaxed allowed set)
or a faster demonstrated implementation. v4 records both bounds and
the explicit non-convergence status instead of a single flattering
number.

Usage:
  python3 tp5-necessity-certificate.py --json out.json
"""
import argparse
import json
import math

# Fixed model contract (Qwen4EXP-AF, verified from GGUF metadata).
N_LAYERS = 48
N_EXPERTS = 512
HIDDEN = 2560
N_RANKS = 5
VOCAB = 248320

# Verified weight geometry (audited against the six-shard GGUF metadata
# scan in artifacts/perf-gap/necessity-restart-20261009/audit.json):
# expert weights (gate/up/down exps) per layer = 1153.4336 MB on disk
# (blk.N.ffn_gate_exps 321.13 + ffn_up_exps 360.45 + ffn_down_exps 471.86).
EXPERT_BYTES_PER_LAYER = 1153.4336e6
# Dense (always-active) weights per layer, 48-layer average from the same
# audit: attn_qkv 18.02 (36 layers) / attn_q+attn_k+attn_v 21.63 (12
# layers), attn_gate 10.81, ssm_out 10.81, ple_key 21.50 (global, counted
# once per layer share), ple_value 5.40, router ffn_gate_inp 5.24, HC
# down/up x4 + inject + norm 14.10, shared experts 4.44, ssm/ple misc.
DENSE_BYTES_PER_LAYER = 63.820906666666666e6
L2_BYTES_PER_GPU = 128e6
FP32_TFLOPS_PER_GPU = 16.17

# Hardware interface ceilings (RX 6800, conservative spec upper bounds).
PCIE_GEN4_X16_GB_S = 32.0
HOST_READBACK_GB_S = 25.0
DRAM_GB_S = 512.0

# R1: sampler-visible data per committed public token.
R1_BYTES_STRONG = N_EXPERTS * 4          # full scores host-visible
R1_BYTES_WEAK = 4                        # token id only

# R2: per-boundary cross-rank payload (minimum shard of the router input
# that the consumer rank cannot produce itself).
R2_BYTES_PER_BOUNDARY = (HIDDEN // N_RANKS) * 4   # 2048 bytes
R2_BOUNDARIES = N_LAYERS                            # 48 sequential layers


def interface_time(bytes_, ceiling_gb_s):
    if bytes_ < 0 or ceiling_gb_s <= 0:
        raise ValueError("Invalid resource input")
    return bytes_ / (ceiling_gb_s * 1e9)


def active_expert_bytes_per_gpu():
    """10/512 experts active per layer, sharded across ranks."""
    return EXPERT_BYTES_PER_LAYER * (10.0 / N_EXPERTS) * N_LAYERS / N_RANKS


def active_dense_bytes_per_gpu():
    """Dense weights active every token, sharded across ranks.

    Generous to implementations: assumes the dense set can be split
    1/5 per GPU (allowed by 'arbitrary partitioning'), so each GPU
    reads only its share. A mirrored layout would read more, not less.
    """
    return DENSE_BYTES_PER_LAYER * N_LAYERS / N_RANKS


def derive():
    # Per target row = per public token for no-MTP implementations.
    r1_strong = interface_time(R1_BYTES_STRONG, HOST_READBACK_GB_S)
    r1_weak = interface_time(R1_BYTES_WEAK, HOST_READBACK_GB_S)
    r2_total = R2_BOUNDARIES * interface_time(R2_BYTES_PER_BOUNDARY, PCIE_GEN4_X16_GB_S)

    # R3: active-weight DRAM traffic, per token, for the no-MTP class.
    # Max L2 credit: L2 may retain up to L2_BYTES of the per-token active
    # set (dense or expert, any split) -> dram >= active - L2.
    active_per_gpu = active_expert_bytes_per_gpu() + active_dense_bytes_per_gpu()
    l2_credit = min(L2_BYTES_PER_GPU, active_per_gpu)
    dram_bytes_per_gpu = active_per_gpu - l2_credit
    r3_time = dram_bytes_per_gpu / (DRAM_GB_S * 1e9)

    weight_pool_per_gpu = (EXPERT_BYTES_PER_LAYER + DENSE_BYTES_PER_LAYER) * N_LAYERS / N_RANKS

    # R4: FP32 vector ALU work. FLOPs scale linearly with verified rows,
    # so the per-public-token FLOP count is invariant under MTP batching
    # (h+1 rows per cycle each do the full per-row FLOP work). This is the
    # only weight-side term that survives the perfect-acceptance MTP
    # relaxation, which is why it binds L_global.
    qkv_rows = 6144 + 512 + 512          # q 24*256, k 256, v 256 (36 layers)
    q_rows = 4224                        # 12 full-attention layers (audited)
    ffn = 640
    hc_dim = 4 * HIDDEN
    hc_rank = 320
    ple_key_elems = 21.5e6 / 2           # BF16 bytes -> elements
    ple_value_elems = 5.4e6 / 2
    attn_gate_elems = 10.8e6 / 2
    ssm_out_elems = 10.8e6 / 2
    per_layer_flops = (
        10 * 3 * 2 * ffn * HIDDEN +          # 10 active routed experts, 3 GEMVs
        3 * 2 * ffn * HIDDEN +               # shared expert, 3 GEMVs
        36 * (2 * qkv_rows * HIDDEN) / 48 +  # linear-attention QKV (36 layers)
        12 * (2 * (q_rows + 2 * 256 + 2 * 256) * HIDDEN) / 48 +  # full-attn qkv (12 layers, avg)
        2 * 4224 * HIDDEN * 12 / 48 +        # attn_output (12 layers, avg)
        2 * 6144 * HIDDEN * 36 / 48 +        # linear-attn output proj (avg)
        2 * N_EXPERTS * HIDDEN +             # router GEMV
        2 * 2 * 2 * hc_rank * hc_dim +       # HC down+up GEMVs (4 sites of 320x10240)
        2 * 2 * (ple_key_elems + ple_value_elems + attn_gate_elems + ssm_out_elems)
        # PLE key/value + attn_gate + ssm_out GEMVs: 2 FLOPs per element
    )
    r4_flops = per_layer_flops * N_LAYERS
    r4_time = r4_flops / (N_RANKS * FP32_TFLOPS_PER_GPU * 1e12)

    # L_nomtp: valid for the declared class A_nomtp (one target row per
    # public token; no speculative batching). R3 applies.
    per_token_lo_nomtp = max(r1_weak, r2_total, r3_time, r4_time)
    per_token_lo_nomtp_strong = max(r1_strong, r2_total, r3_time, r4_time)

    # L_global: valid for the FULL allowed set A. Under MTP with any
    # draft policy and perfect acceptance, weight-DRAM traffic (R3) is
    # read once per cycle and amortizes across the h+1 committed rows,
    # so R3 cannot appear in the global bound. R1/R2/R4 survive:
    # R1 is per public token; R2 is per row (rows cross boundaries); R4
    # FLOPs scale with rows. The bound is per public token with
    # rows-per-public-token >= 1.
    per_token_lo_global = max(r1_weak, r2_total, r4_time)
    per_token_lo_global_strong = max(r1_strong, r2_total, r4_time)

    return {
        "task": "TP5 ordinary single-user generation, Qwen4EXP-AF, 5x RX 6800",
        "version": "v4",
        "supersedes": "v3 (DEFECT-1 dense bytes 90.8->63.82 MB/layer; DEFECT-2 L2 credit "
                      "pool->active denominator; DEFECT-3 MTP amortization invalidates R3 "
                      "in the global bound)",
        "allowed_set_A": [
            "arbitrary lossless weight representation (incl. exact BF16)",
            "unrestricted caching in VRAM/L2/LDS",
            "arbitrary owner/coordinate partitioning across 5 GPUs",
            "arbitrary async prefetch across causal stages",
            "arbitrary kernel/operator fusion",
            "MTP/speculative with any draft policy incl. perfect acceptance",
        ],
        "declared_conditional_class_A_nomtp": (
            "implementations that process exactly one target row per public "
            "token (no speculative row batching); R3 is valid only here"),
        "resources": {
            "R1_output_emission": {
                "bytes_strong": R1_BYTES_STRONG,
                "bytes_weak": R1_BYTES_WEAK,
                "ceiling_gb_s": HOST_READBACK_GB_S,
                "time_strong_s": r1_strong,
                "time_weak_s": r1_weak,
                "justification": "each committed token requires the host to observe either the token id (4B, weak) or the full 512-expert routing scores (2048B, strong, current sampler contract); survives MTP (per public token)",
                "survives_mtp_relaxation": True,
            },
            "R2_causal_cut_set": {
                "bytes_per_boundary": R2_BYTES_PER_BOUNDARY,
                "boundaries_per_row": R2_BOUNDARIES,
                "total_bytes_per_row": R2_BYTES_PER_BOUNDARY * R2_BOUNDARIES,
                "ceiling_gb_s": PCIE_GEN4_X16_GB_S,
                "time_per_row_s": r2_total,
                "justification": "48 sequential layer boundaries; router input (2560 F32) must be complete before layer l runs; with 5 disjoint coordinate shards at least 2048B crosses a rank cut per boundary per ROW; survives MTP (scales with rows)",
                "survives_mtp_relaxation": True,
            },
            "R3_active_weight_dram": {
                "weight_pool_per_gpu_bytes": weight_pool_per_gpu,
                "dense_bytes_per_layer": DENSE_BYTES_PER_LAYER,
                "expert_bytes_per_layer": EXPERT_BYTES_PER_LAYER,
                "active_expert_bytes_per_gpu": active_expert_bytes_per_gpu() + active_dense_bytes_per_gpu(),
                "active_per_gpu_per_token_bytes": active_per_gpu,
                "l2_credit_bytes": l2_credit,
                "dram_bytes_per_gpu_per_token": dram_bytes_per_gpu,
                "ceiling_gb_s": DRAM_GB_S,
                "time_per_row_s": r3_time,
                "justification": "10/512 experts + all dense weights active per token; max L2 credit = 128MB of the active set (any split); valid ONLY for A_nomtp: under MTP row batching the active weights are read once per cycle and amortize across committed rows",
                "survives_mtp_relaxation": False,
            },
            "R4_fp32_alu": {
                "flops_per_token": r4_flops,
                "per_layer_flops": per_layer_flops,
                "peak_flops_s": N_RANKS * FP32_TFLOPS_PER_GPU * 1e12,
                "time_per_row_s": r4_time,
                "justification": "attention projections + router + 11 experts + PLE + HC per layer, 48 layers; FLOPs scale with verified rows so the per-public-token cost is MTP-invariant; RX 6800 has no FP32 matrix cores so vector ALU peak applies; lossless representation changes bytes, not FLOPs",
                "survives_mtp_relaxation": True,
            },
        },
        "combination_rule": "per-token lower bound = max over resources: distinct physical resources may overlap in a legal implementation; same-resource work adds",
        "mtp_relaxation_note": (
            "perfect-acceptance relaxation (any draft policy, unlimited h). "
            "R3 weight traffic is per CYCLE, not per row, and amortizes across "
            "the h+1 committed rows; it is therefore excluded from L_global. "
            "R4 FLOPs and R2 cut bytes scale with rows and survive. R1 is per "
            "public token and survives."),
        "bounds": {
            "L_global": {
                "per_token_lower_bound_s_weak": per_token_lo_global,
                "per_token_lower_bound_s_strong": per_token_lo_global_strong,
                "tps_upper_bound_weak": 1.0 / per_token_lo_global,
                "tps_upper_bound_strong": 1.0 / per_token_lo_global_strong,
                "valid_for": "full allowed set A",
            },
            "L_nomtp": {
                "per_token_lower_bound_s_weak": per_token_lo_nomtp,
                "per_token_lower_bound_s_strong": per_token_lo_nomtp_strong,
                "tps_upper_bound_weak": 1.0 / per_token_lo_nomtp,
                "tps_upper_bound_strong": 1.0 / per_token_lo_nomtp_strong,
                "valid_for": "declared conditional class A_nomtp (no speculative row batching)",
            },
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    args = ap.parse_args()
    cert = derive()
    with open(args.json, 'w') as f:
        json.dump(cert, f, indent=2)
    b = cert["bounds"]
    r = cert["resources"]
    print(f"R1 weak/strong per row: {r['R1_output_emission']['time_weak_s']:.3e} / {r['R1_output_emission']['time_strong_s']:.3e} s")
    print(f"R2 per row: {r['R2_causal_cut_set']['time_per_row_s']:.3e} s")
    print(f"R3 per row: {r['R3_active_weight_dram']['time_per_row_s']:.3e} s (DRAM {r['R3_active_weight_dram']['dram_bytes_per_gpu_per_token']/1e6:.1f} MB/GPU, A_nomtp only)")
    print(f"R4 per row: {r['R4_fp32_alu']['time_per_row_s']:.3e} s ({r['R4_fp32_alu']['flops_per_token']/1e9:.1f} GFLOP)")
    print(f"L_global  (full A):      {b['L_global']['per_token_lower_bound_s_weak']:.6e} s -> ceiling {b['L_global']['tps_upper_bound_weak']:.0f} tok/s")
    print(f"L_nomtp   (A_nomtp):     {b['L_nomtp']['per_token_lower_bound_s_weak']:.6e} s -> ceiling {b['L_nomtp']['tps_upper_bound_weak']:.0f} tok/s")
    print("NOTE: v4 splits the certificate. L_global excludes R3 because MTP row-batching "
          "amortizes weight-DRAM traffic (v3's single-number claim was unsound). L_nomtp "
          "keeps R3 for the no-MTP conditional class with corrected dense bytes (63.82 "
          "MB/layer) and max L2 credit (128MB of the active set). Both bounds remain "
          "loose: spec ceilings, maximal overlap, no KV-state or latency-chain terms.")


if __name__ == '__main__':
    main()
