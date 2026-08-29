#!/usr/bin/env python3
"""Build per-layer input transforms M (x_hat = M x) for the tail ablation.

The question being tested: the 256 experts of a layer all read the same x, so an
input basis can be *shared* across them. Per-expert gate/up then cost 512*k
instead of 512*2048, while the shared basis costs 2048*k once per layer and
amortises to nothing. Total per-expert parameters are 1024k + 1048576, so

    2x parameter reduction  <=>  k <= 512

and k = 512 is exactly the break-even point worth testing.

Three transforms:
  ctrl      identity            isolates requantisation error alone
  proj-K    P_K                 drop the tail: what the shared-basis design does
  rot-K     P_K + R_tail        scramble the tail, preserving its covariance

proj and rot differ in what they destroy. Dropping the tail removes both its
content and its energy; rotating it removes only the content. If proj breaks and
rot does not, the expert needs the tail's statistics but not its information,
and the tail can be synthesised. If both break, the linear head is insufficient
(which does not by itself rule out a nonlinear encoder: intrinsic dimension is
~30 while 90% of the *linear* energy needs 190-559 directions, and that gap is
curvature a linear probe cannot see across).

R_tail = U_T Lam_T^(1/2) Q Lam_T^(-1/2) U_T^T with Q random orthogonal, which
preserves the tail covariance exactly: whitening gives identity, Q keeps it,
recolouring returns U_T Lam_T U_T^T. The tail is capped at the 99.9% energy
index so Lam^(-1/2) never divides by a near-zero eigenvalue; directions past
that carry <0.1% of the energy and are left alone.
"""

import argparse
import struct
import sys

import numpy as np

import wxqcal

MAGIC = b"WXQXFM01"


def load_cov(path):
    meta, layers = wxqcal.load(path, want_cov=True)
    if not meta.flags & wxqcal.FLAG_COV:
        raise SystemExit("dump has no covariance")
    return {layer: wxqcal.symmetrised_cov(e) for layer, e in layers.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="/tmp/opencode/calib-full2.bin")
    ap.add_argument("--out-prefix", default="/tmp/opencode/xfm")
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cov = load_cov(args.calib)
    layers = sorted(cov)
    d = cov[layers[0]].shape[0]
    K = args.k
    print(f"{len(layers)} layers, n_embd {d}, k {K}")

    modes = [f"proj{K}", f"rot{K}"]
    files = {}
    for m in modes:
        fh = open(f"{args.out_prefix}-{m}.bin", "wb")
        fh.write(MAGIC)
        fh.write(struct.pack("<2I", len(layers), d))
        files[m] = fh

    I = np.eye(d, dtype=np.float64)
    for layer in layers:
        w, U = np.linalg.eigh(cov[layer])
        w = w[::-1].clip(0.0, None)
        U = U[:, ::-1]
        cs = np.cumsum(w) / w.sum()
        r = int(np.searchsorted(cs, 0.999) + 1)          # tail cap
        r = max(r, K + 2)

        Uk = U[:, :K]
        P = Uk @ Uk.T                                     # head projector

        UT = U[:, K:r]
        lam = w[K:r]
        nt = r - K
        # Haar-random orthogonal on the tail
        Q, _ = np.linalg.qr(rng.standard_normal((nt, nt)))
        Q *= np.sign(np.diag(Q))                          # fix reflection sign
        Rt = (UT * np.sqrt(lam)) @ Q @ (UT / np.sqrt(lam)).T
        Ptail = UT @ UT.T

        M = {
            f"proj{K}": P,
            f"rot{K}": I - Ptail + Rt,
        }
        for m, fh in files.items():
            fh.write(struct.pack("<2I", layer, 0))
            fh.write(np.ascontiguousarray(M[m], dtype=np.float32).tobytes())

        # sanity: energy kept by the head, and covariance error of the rotation
        keep = cs[K - 1]
        cov_err = np.linalg.norm(M[f"rot{K}"] @ cov[layer] @ M[f"rot{K}"].T - cov[layer]) / np.linalg.norm(cov[layer])
        print(f"  blk.{layer:<2d} tail cap r={r:4d}  head keeps {100*keep:5.1f}% of input energy"
              f"   rot cov error {cov_err:.2e}")

    for fh in files.values():
        fh.close()
    print("wrote:", ", ".join(f"{args.out_prefix}-{m}.bin" for m in modes))


if __name__ == "__main__":
    main()
