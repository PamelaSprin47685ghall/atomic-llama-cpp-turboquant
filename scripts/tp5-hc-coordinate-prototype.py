#!/usr/bin/env python3
"""CPU algebra witness for coordinate-owned HC, not a Vulkan implementation.

Corresponds to qwen4exp.cpp build_hc_combine -> build_hc_mix. Each rank owns
the same coordinate subset in every residual stream. Y is reduce-scattered
BEFORE squaring. Norm, down and inject partials are reduced; up and stream
fold remain coordinate-local. Gathering returned arrays is for the oracle,
not a claim that subsequent model consumers accept this layout for free.

Python float/finite inputs only. This proves neither production floating-point
parity, resource ownership/epoch safety, model quality, nor a timing bound.
"""
import math
from typing import NamedTuple


class Weights(NamedTuple):
    gamma: list   # [streams * width]
    down: list    # [latent][streams * width]
    up: list      # [streams * width][latent]
    inject: list  # [streams][streams * width]


class Result(NamedTuple):
    state: list       # [rows][streams][width], after incoming block combine
    normalized: list  # same shape, including gamma
    low: list         # [rows][latent], after SiLU
    mixed: list       # [rows][width]
    inject: list      # [rows][streams], logits for the next combine


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _shape(value, dimensions):
    if not dimensions:
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("Expected finite numeric inputs")
        return
    if not isinstance(value, (list, tuple)) or len(value) != dimensions[0]:
        raise ValueError("Shape mismatch")
    for item in value:
        _shape(item, dimensions[1:])


def _validate(residual, partial_y, scatter, weights, eps):
    try:
        rows, streams, width = len(residual), len(residual[0]), len(residual[0][0])
        ranks, latent = len(partial_y), len(weights.down)
    except (IndexError, TypeError) as e:
        raise ValueError("Empty or malformed inputs") from e
    if min(rows, streams, width, ranks, latent) < 1:
        raise ValueError("All logical dimensions must be positive")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    _shape(residual, (rows, streams, width))
    _shape(partial_y, (ranks, rows, width))
    _shape(scatter, (rows, streams))
    _shape(weights.gamma, (streams * width,))
    _shape(weights.down, (latent, streams * width))
    _shape(weights.up, (streams * width, latent))
    _shape(weights.inject, (streams, streams * width))
    return rows, streams, width, ranks, latent


def reference(residual, partial_y, scatter, weights, eps=1e-6):
    """Materialized grouped RMSNorm -> down -> SiLU -> up/gate/fold + inject."""
    rows, streams, width, ranks, latent = _validate(residual, partial_y, scatter, weights, eps)
    state, normalized, low, mixed, inject = [], [], [], [], []
    for t in range(rows):
        y = [math.fsum(partial_y[r][t][j] for r in range(ranks)) for j in range(width)]
        x = [[residual[t][b][j] + scatter[t][b] * y[j] for j in range(width)] for b in range(streams)]
        xn = []
        for b in range(streams):
            rho = 1.0 / math.sqrt(math.fsum(v * v for v in x[b]) / width + eps)
            xn.append([x[b][j] * rho * weights.gamma[b * width + j] for j in range(width)])
        flat = [v for stream in xn for v in stream]
        u = [math.fsum(a * v for a, v in zip(w, flat)) / streams for w in weights.down]
        h = [v * sigmoid(v) for v in u]
        gate = [sigmoid(math.fsum(w[k] * h[k] for k in range(latent))) for w in weights.up]
        mix = [math.fsum(xn[b][j] * gate[b * width + j] for b in range(streams)) / streams for j in range(width)]
        inj = [math.fsum(a * v for a, v in zip(w, flat)) for w in weights.inject]
        state.append(x); normalized.append(xn); low.append(h); mixed.append(mix); inject.append(inj)
    return Result(state, normalized, low, mixed, inject)


def coordinate_owned(residual, partial_y, scatter, weights, partitions=None, eps=1e-6):
    """Norm/down/inject algebra over disjoint coordinate owners, with ragged tails."""
    rows, streams, width, ranks, latent = _validate(residual, partial_y, scatter, weights, eps)
    if partitions is None:
        partitions = [list(range(width * r // ranks, width * (r + 1) // ranks)) for r in range(ranks)]
    if len(partitions) != ranks:
        raise ValueError("One partition per rank is required")
    coords = [j for part in partitions for j in part]
    if any(type(j) is not int for j in coords) or sorted(coords) != list(range(width)):
        raise ValueError("Partitions must cover every coordinate exactly once")
    state, normalized, low, mixed, inject = [], [], [], [], []
    for t in range(rows):
        x = [[0.0] * width for _ in range(streams)]
        gx = [[0.0] * width for _ in range(streams)]
        v_parts, d_parts, i_parts = [], [], []
        for part in partitions:
            # C1. Reduce-scatter complete Y on the coordinates this rank owns.
            y = {j: math.fsum(partial_y[r][t][j] for r in range(ranks)) for j in part}
            for b in range(streams):
                for j in part:
                    x[b][j] = residual[t][b][j] + scatter[t][b] * y[j]
                    gx[b][j] = x[b][j] * weights.gamma[b * width + j]
            # C2. Square complete Y-derived X, never unreduced producer partials.
            v_parts.append([math.fsum(x[b][j] ** 2 for j in part) for b in range(streams)])
            d_parts.append([[math.fsum(weights.down[k][b * width + j] * gx[b][j] for j in part)
                             for k in range(latent)] for b in range(streams)])
            i_parts.append([[math.fsum(weights.inject[o][b * width + j] * gx[b][j] for j in part)
                             for o in range(streams)] for b in range(streams)])
        # C3. All-reduce small sufficient statistics, then apply stream-specific rho.
        rho = [1.0 / math.sqrt(math.fsum(v[b] for v in v_parts) / width + eps) for b in range(streams)]
        u = [math.fsum(rho[b] * math.fsum(d[b][k] for d in d_parts) for b in range(streams)) / streams
             for k in range(latent)]
        h = [value * sigmoid(value) for value in u]
        inj = [math.fsum(rho[b] * math.fsum(p[b][o] for p in i_parts) for b in range(streams))
               for o in range(streams)]
        # C4. Up output rows, normalization, sigmoid gate and stream fold stay local.
        xn = [[0.0] * width for _ in range(streams)]
        mix = [0.0] * width
        for part in partitions:
            for j in part:
                pieces = []
                for b in range(streams):
                    idx = b * width + j
                    xn[b][j] = gx[b][j] * rho[b]
                    gate = sigmoid(math.fsum(weights.up[idx][k] * h[k] for k in range(latent)))
                    pieces.append(xn[b][j] * gate)
                mix[j] = math.fsum(pieces) / streams
        # These gathered arrays are oracle output, NOT free production communication.
        state.append(x); normalized.append(xn); low.append(h); mixed.append(mix); inject.append(inj)
    return Result(state, normalized, low, mixed, inject)


def combine_next(result, next_block):
    """Exercise the next residual update, including the 2*sigmoid(inject/HC) rule."""
    rows, streams, width = len(result.state), len(result.state[0]), len(result.state[0][0])
    _shape(next_block, (rows, width))
    return [[[result.state[t][b][j] + 2 * sigmoid(result.inject[t][b] / streams) * next_block[t][j]
              for j in range(width)] for b in range(streams)] for t in range(rows)]
