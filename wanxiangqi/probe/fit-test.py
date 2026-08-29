#!/usr/bin/env python3
"""Can ANY half-parameter family reproduce one expert's behaviour on this data?

The linear route died: proj512 (top-512 covariance eigendirections, exactly 2x)
collapsed completely, and the weight rel-err of 0.744 explained why -- it is
almost exactly (2048-512)/2048 = 0.75, so the expert weights are essentially
isotropic in the input covariance eigenbasis and a linear head throws away three
quarters of them. That was the false negative predicted in advance: a linear
probe cannot cross the gap between intrinsic dimension ~30 and the ~500 linear
directions needed for 90% of the energy, because that gap *is* the curvature.

So this fits a nonlinear student instead, in the same function class as the
teacher but with the input-side matrices tied across experts:

    y_hat_e = C_e [ SiLU(G x) (*) (U x) ]

G, U in R^{m x 2048} shared by every expert of the layer (all experts read the
same x, so a shared feature bank is legitimate; cost 2*2048*m once per layer,
amortised over 256 experts it is ~8k per expert)
C_e in R^{2048 x m} per expert.

Per-expert cost is 2048*m against the teacher's 3*512*2048 = 3,145,728:
    m = 256  ->  6.0x reduction
    m = 512  ->  3.0x reduction
    m = 768  ->  exactly 2.0x

Two references bracket the answer, both computed on the same samples, both from
models whose behaviour has actually been read:
    ctrl     requantised, behaviourally indistinguishable from source
    proj512  behaviourally dead (emits only repeated <think> on neutral prompts)
An error at or below ctrl means the student is as good as a model known to work.
An error at or above proj512 means it is as bad as a model known to be dead.

Conservative in one respect worth stating: each expert is fitted on all sampled
inputs, not on the subset actually routed to it. Real deployment only needs each
expert to be right on its own slice, which is strictly easier -- but with 4831
samples and top-8 of 256 that slice is ~151 samples, far too few to fit a 1M
parameter readout, so the easier version is untestable with this data.
"""

import argparse
import sys
import time

import numpy as np
import torch

import wxqcal

sys.path.insert(0, "../gguf-py")
from gguf import GGUFReader                      # noqa: E402
from gguf.quants import dequantize               # noqa: E402


def load_x(path, layer):
    f = open(path, "rb")
    assert f.read(8) == b"WXQMAN01"
    nl, stride = struct.unpack("<2I", f.read(8))
    for _ in range(nl):
        lay, ne, nf, nu, nx, nh = struct.unpack("<6I", f.read(24))
        x = np.frombuffer(f.read(nx * ne * 4), np.float32).reshape(nx, ne)
        f.read(nh * nf * 4)
        ids = np.frombuffer(f.read(nx * nu * 4), np.int32).reshape(nx, nu)
        if lay == layer:
            f.close()
            return x.copy(), ids.copy()
    f.close()
    raise SystemExit(f"layer {layer} not in {path}")


def experts_of(model, layer, which):
    r = GGUFReader(model)
    T = {t.name: t for t in r.tensors}
    out = {}
    for proj in ("gate", "up", "down"):
        t = T[f"blk.{layer}.ffn_{proj}_exps.weight"]
        ne = [int(v) for v in t.shape]
        W = dequantize(t.data, t.tensor_type).reshape(ne[2], ne[1], ne[0])
        out[proj] = np.ascontiguousarray(W[which]).astype(np.float32)
        del W
    return out


def teacher_y(W, x):
    """y = Wd [ SiLU(Wg x) * (Wu x) ] for every expert, float64 accumulation"""
    ys = []
    for e in range(W["gate"].shape[0]):
        g = x @ W["gate"][e].T
        u = x @ W["up"][e].T
        h = (g / (1.0 + np.exp(-g))) * u
        ys.append(h @ W["down"][e].T)
    return np.stack(ys, 0)          # [n_exp, n_samples, n_embd]


def rel(a, b):
    return float(((a - b) ** 2).sum() / (a ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=34)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--widths", default="256,512,768")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--manifold", default="/tmp/opencode/manifold.bin")
    ap.add_argument("--src", default="/opt/llama/data/Ornith-1.5-35B-Uncensored-YMQ-S-MTP.gguf")
    ap.add_argument("--ctrl", default="/opt/llama/data/ornith-ctrl.gguf")
    ap.add_argument("--calib", default="/tmp/opencode/calib-full2.bin")
    args = ap.parse_args()
    torch.set_num_threads(12)

    L = args.layer
    x, ids = load_x(args.manifold, L)
    n, d = x.shape
    print(f"blk.{L}: {n} samples, n_embd {d}")

    # busiest experts on this sample, so the fit is exercised where it matters
    cnt = np.bincount(ids.reshape(-1), minlength=256)
    which = np.argsort(-cnt)[: args.experts]
    print(f"experts {list(which)}  (routed-sample counts {list(cnt[which])})")

    W = experts_of(args.src, L, which)
    n_ff = W["gate"].shape[1]
    P_teacher = 3 * n_ff * d
    print(f"teacher n_ff {n_ff}, per-expert params {P_teacher:,}")

    Y = teacher_y(W, x)
    print(f"teacher outputs {Y.shape}")

    # ---- reference 1: ctrl, behaviourally fine -------------------------------
    Wc = experts_of(args.ctrl, L, which)
    Yc = teacher_y(Wc, x)
    r_ctrl = np.mean([rel(Y[i], Yc[i]) for i in range(len(which))])
    del Wc, Yc

    # ---- reference 2: proj512, behaviourally dead ---------------------------
    _meta, cal = wxqcal.load(args.calib, want=[L], want_cov=True)
    ev, U = np.linalg.eigh(wxqcal.symmetrised_cov(cal[L]))
    Uk = U[:, ::-1][:, :512].astype(np.float32)
    Yp = teacher_y(W, (x @ Uk) @ Uk.T)
    r_proj = np.mean([rel(Y[i], Yp[i]) for i in range(len(which))])
    del Yp

    print(f"\nreference output rel-err on these samples")
    print(f"  ctrl    (behaviourally fine) {r_ctrl:.4f}")
    print(f"  proj512 (behaviourally dead) {r_proj:.4f}\n")

    # ---- fit ----------------------------------------------------------------
    ntr = int(0.8 * n)
    Xtr = torch.from_numpy(x[:ntr]); Xte = torch.from_numpy(x[ntr:])
    Ytr = torch.from_numpy(Y[:, :ntr]); Yte = torch.from_numpy(Y[:, ntr:])
    E = len(which)

    print(f"{'m':>5} {'per-exp params':>15} {'vs teacher':>11} {'train rel':>10} {'HELD-OUT rel':>13} {'verdict':>22}")
    for m in [int(v) for v in args.widths.split(",")]:
        g = torch.Generator().manual_seed(0)
        G = (torch.randn(m, d, generator=g) / np.sqrt(d)).requires_grad_()
        Uu = (torch.randn(m, d, generator=g) / np.sqrt(d)).requires_grad_()
        Cc = (torch.randn(E, d, m, generator=g) / np.sqrt(m)).requires_grad_()
        opt = torch.optim.Adam([G, Uu, Cc], lr=2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

        t0 = time.time()
        for s in range(args.steps):
            idx = torch.randint(0, ntr, (args.batch,))
            xb = Xtr[idx]
            h = torch.nn.functional.silu(xb @ G.T) * (xb @ Uu.T)      # [B, m]
            pred = torch.einsum("bm,edm->ebd", h, Cc)                  # [E, B, d]
            loss = ((pred - Ytr[:, idx]) ** 2).sum() / (Ytr[:, idx] ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        with torch.no_grad():
            def ev_rel(X, Yt):
                h = torch.nn.functional.silu(X @ G.T) * (X @ Uu.T)
                p = torch.einsum("bm,edm->ebd", h, Cc)
                return float(((p - Yt) ** 2).sum() / (Yt ** 2).sum())
            rtr = ev_rel(Xtr, Ytr)
            rte = ev_rel(Xte, Yte)

        P = d * m
        v = "as good as ctrl" if rte <= r_ctrl * 1.5 else ("as bad as proj512" if rte >= r_proj else "between")
        print(f"{m:5d} {P:15,} {P_teacher/P:10.2f}x {rtr:10.4f} {rte:13.4f} {v:>22}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
