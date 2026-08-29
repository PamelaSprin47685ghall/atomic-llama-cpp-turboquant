"""Reader for the WXQCAL03 MoE calibration dump.

The authoritative description of the format is the C++ header that writes it,
wanxiangqi/lib/calib-dump.h. This module is the Python half of the same contract:
three separate scripts used to walk those byte offsets by hand, which meant a
format change had three places to break and two of them silently -- skipping the
wrong number of bytes yields plausible-looking garbage, not an error.

Field names match the C++ `calib_layer` exactly.
"""

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"WXQCAL03"

FLAG_COV = 1 << 0
FLAG_HIDDEN = 1 << 1
FLAG_EXPERT_IN = 1 << 2


@dataclass
class Meta:
    n_entries: int
    max_ctx: int
    act_stride: int
    act_tokens: int
    route_stride: int
    flags: int
    n_decoded: int
    n_total: int
    n_ctx_resets: int
    complete: bool

    def describe(self):
        return (f"{self.n_entries} layers, flags 0x{self.flags:x}, "
                f"{self.n_decoded}/{self.n_total} nodes, "
                f"{'complete' if self.complete else 'PARTIAL'}")


def _u64(f, n):
    return np.frombuffer(f.read(n * 8), np.uint64).copy()


def _f64(f, n):
    return np.frombuffer(f.read(n * 8), np.float64).copy()


def load(path, want=None, want_cov=False):
    """Returns (Meta, {layer_index: dict}).

    `want` is an iterable of layer indices, or None for all. `want_cov` gates the
    n_embd^2 covariance block, which is 32 MiB per layer at n_embd 2048 and is
    seeked past unless asked for.
    """
    want = None if want is None else set(want)

    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise SystemExit(f"bad magic: {path} is not a WXQCAL03 dump")
        head = struct.unpack("<6I3QB7x", f.read(24 + 24 + 8))
        meta = Meta(*head[:9], complete=bool(head[9]))

        layers = {}
        for _ in range(meta.n_entries):
            layer, n_expert, n_embd, n_ff = struct.unpack("<4I", f.read(16))
            n_pos, n_route, n_cov = struct.unpack("<3Q", f.read(24))
            keep = want is None or layer in want

            e = dict(n_expert=n_expert, n_embd=n_embd, n_ff=n_ff,
                     n_pos=n_pos, n_route=n_route, n_cov=n_cov)

            e["counts"] = _u64(f, n_expert)
            e["act_sum"] = _f64(f, n_embd)
            if meta.flags & FLAG_EXPERT_IN:
                e["exp_n"] = _u64(f, n_expert)
                e["exp_mean"] = _f64(f, n_expert * n_embd).reshape(n_expert, n_embd)
                e["exp_diag"] = _f64(f, n_expert * n_embd).reshape(n_expert, n_embd)
            if meta.flags & FLAG_HIDDEN:
                e["hid_n"] = _u64(f, n_expert)
                e["hid_sum"] = _f64(f, n_expert * n_ff).reshape(n_expert, n_ff)
            if meta.flags & FLAG_COV:
                if keep and want_cov:
                    e["cov"] = _f64(f, n_embd * n_embd).reshape(n_embd, n_embd)
                else:
                    f.seek(n_embd * n_embd * 8, 1)

            if keep:
                layers[layer] = e

    return meta, layers


def require(meta, flags, why):
    """Fail loudly when a dump lacks a statistic the caller needs."""
    missing = [name for bit, name in ((FLAG_COV, "--calib-cov"),
                                      (FLAG_HIDDEN, "--calib-hidden"),
                                      (FLAG_EXPERT_IN, "--calib-cov/--calib-hidden"))
               if (flags & bit) and not (meta.flags & bit)]
    if missing:
        raise SystemExit(f"dump lacks {why} (collect with {', '.join(sorted(set(missing)))})")


def symmetrised_cov(layer):
    """cov is written as a full square, but round-tripping through f64 mirroring
    leaves it only near-symmetric; eigh wants exact symmetry."""
    C = layer["cov"]
    return 0.5 * (C + C.T)
