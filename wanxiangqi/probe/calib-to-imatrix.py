#!/usr/bin/env python3
"""calib-full2.bin (WXQCAL03) -> imatrix GGUF for llama-quantize.

The IQ family refuses to quantise without an importance matrix, and the shapes
line up exactly with what the trie calibration already collected:

    blk.N.ffn_gate_exps.weight   in_sum2[e][j] = sum_t x_j^2   (per-expert input)
    blk.N.ffn_up_exps.weight     same input, same statistic
    blk.N.ffn_down_exps.weight   in_sum2[e][j] = sum_t h_j^2   (per-expert hidden)

llama.cpp stores raw sums plus per-expert counts and divides on load, so the
sums go in untouched.

Dead experts (blk.38 has five) would otherwise get an all-zero importance row,
which makes the quantiser's scale search degenerate rather than merely
uninformed. Those rows are backfilled with the layer mean over live experts.
"""

import argparse
import sys

import numpy as np

import wxqcal

sys.path.insert(0, "gguf-py")
from gguf import GGUFWriter  # noqa: E402


def load(path):
    meta, layers = wxqcal.load(path)
    if not meta.flags & wxqcal.FLAG_EXPERT_IN:
        raise SystemExit("dump lacks per-expert input stats (need --calib-cov/--calib-hidden run)")
    if not meta.flags & wxqcal.FLAG_HIDDEN:
        raise SystemExit("dump lacks per-neuron hidden energy (down projection would have no imatrix)")
    print(f"dump: {meta.describe()}")
    return layers, meta.n_decoded


def backfill(sums, counts, what, layer):
    """dead experts -> layer mean over live ones, count 1"""
    live = counts > 0
    n_dead = int((~live).sum())
    if n_dead == 0:
        return sums, counts.astype(np.float32), 0
    if not live.any():
        raise SystemExit(f"blk.{layer} {what}: no live experts at all")
    # per-token mean profile of the live experts, scaled to one token
    mean_profile = (sums[live].sum(0) / counts[live].sum()).astype(np.float64)
    sums = sums.copy()
    sums[~live] = mean_profile
    counts = counts.astype(np.float32).copy()
    counts[~live] = 1.0
    return sums, counts, n_dead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="/tmp/opencode/calib-full2.bin")
    ap.add_argument("--out", default="/tmp/opencode/ornith-imatrix.gguf")
    ap.add_argument("--dataset", default="wanxiangqi/calibration")
    args = ap.parse_args()

    layers, n_decoded = load(args.calib)

    w = GGUFWriter(args.out, "imatrix")
    w.add_string("general.type", "imatrix")
    w.add_array("imatrix.datasets", [args.dataset])
    w.add_uint32("imatrix.chunk_count", int(n_decoded))
    w.add_uint32("imatrix.chunk_size", 1)

    total_dead = 0
    for layer in sorted(layers):
        L = layers[layer]
        # gate and up share the expert input, so they share the statistic
        gs, gc, d1 = backfill(L["exp_diag"], L["exp_n"], "expert-input", layer)
        for proj in ("gate", "up"):
            w.add_tensor(f"blk.{layer}.ffn_{proj}_exps.weight.in_sum2", gs.astype(np.float32))
            w.add_tensor(f"blk.{layer}.ffn_{proj}_exps.weight.counts", gc.reshape(-1, 1))
        ds, dc, d2 = backfill(L["hid_sum"], L["hid_n"], "hidden", layer)
        w.add_tensor(f"blk.{layer}.ffn_down_exps.weight.in_sum2", ds.astype(np.float32))
        w.add_tensor(f"blk.{layer}.ffn_down_exps.weight.counts", dc.reshape(-1, 1))
        total_dead += max(d1, d2)
        if d1 or d2:
            print(f"  blk.{layer}: backfilled {d1} dead expert-input, {d2} dead hidden")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}: {len(layers)} layers, {len(layers)*6} tensors, {total_dead} dead experts backfilled")


if __name__ == "__main__":
    main()
