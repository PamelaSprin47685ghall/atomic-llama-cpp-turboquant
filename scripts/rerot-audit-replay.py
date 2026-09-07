#!/usr/bin/env python3
"""Independent float64 replay of captured RBB inputs, without model weights."""
import argparse
import json
from pathlib import Path
import numpy as np


def replay(directory: Path) -> dict:
    mode_file = directory / 'density-mode.txt'
    density_mode = int(mode_file.read_text()) if mode_file.exists() else 0
    if density_mode not in (0, 1):
        raise ValueError('unsupported density mode')
    tensors = {}
    for line in (directory / 'tensors.tsv').read_text().splitlines():
        name, dtype, *dims = line.split('\t')
        if dtype != 'f32':
            raise ValueError(f'{name}: expected f32, got {dtype}')
        ne, nb = tuple(map(int, dims[:4])), tuple(map(int, dims[4:8]))
        raw = (directory / (name + '.bin')).read_bytes()
        if len(raw) != int(dims[8]):
            raise ValueError(f'{name}: incomplete snapshot')
        tensors[name] = np.ndarray(ne[::-1], dtype='<f4', buffer=raw, strides=nb[::-1]).astype(np.float64)
    q, k, v, g, beta, brain, native = (tensors[f'src{i}'] for i in range(7))
    n, tokens, heads, dim = v.shape
    assert tokens == 1 and brain.shape == (1, heads, dim, dim)
    expected_b = np.zeros((heads, dim, dim))
    expected_h = np.zeros((n, heads, dim, dim))
    expected_o = np.zeros((n, heads, dim))
    conditions = []
    native_gap_num = native_norm_sq = 0.0
    for h in range(heads):
        key = k[np.arange(n) // (n // k.shape[0]), 0, h % k.shape[2], :]
        query = q[np.arange(n) // (n // q.shape[0]), 0, h % q.shape[2], :]
        gate, log_decay, base = beta[:, 0, h, 0], g[:, 0, h, 0], brain[0, h]
        decayed = np.exp(log_decay.mean()) * base
        residual = v[:, 0, h] - key @ decayed.T
        root = np.sqrt(gate)
        scaled_k = root[:, None] * key
        density = np.ones(n)
        if density_mode:
            gram_keys = key @ key.T
            norm_products = np.diag(gram_keys)[:, None] * np.diag(gram_keys)[None, :]
            squared_cos = np.divide(gram_keys ** 2, norm_products, out=np.zeros_like(gram_keys), where=norm_products > 1e-20)
            np.fill_diagonal(squared_cos, 0.0)
            density += (squared_cos * (gate > 0)[None, :]).sum(axis=1)
        gram = scaled_k @ scaled_k.T + np.diag(density * (1.0 - gate + 1e-4 * gate))
        conditions.append(float(np.linalg.cond(gram)))
        weights = gate[:, None] * residual if n == 1 else root[:, None] * np.linalg.solve(gram, root[:, None] * residual)
        merged = decayed + (key.T @ weights).T
        hand = native[:, h] - base
        projected = np.einsum('ncr,nr->nc', hand, key)
        evolved = np.exp(log_decay)[:, None, None] * (hand - gate[:, None, None] * projected[:, :, None] * key[:, None, :])
        out = np.einsum('ncr,nr->nc', merged[None] + evolved, query) / np.sqrt(dim)
        expected_b[h], expected_h[:, h], expected_o[:, h] = merged, evolved, out
        decay_native = np.exp(log_decay)[:, None, None] * native[:, h]
        native_residual = v[:, 0, h] - np.einsum('ncr,nr->nc', decay_native, key)
        candidate = decay_native + gate[:, None, None] * native_residual[:, :, None] * key[:, None, :]
        native_o = np.einsum('ncr,nr->nc', candidate, query) / np.sqrt(dim)
        native_gap_num += float(np.square(out - native_o).sum())
        native_norm_sq += float(np.square(native_o).sum())
    flat = tensors['result'].ravel()
    a, state_size = n * heads * dim, heads * dim * dim
    actual = (flat[:a].reshape(expected_o.shape), flat[a:a + state_size].reshape(expected_b.shape),
              flat[a + state_size:a + state_size * (n + 1)].reshape(expected_h.shape))
    report = {'path': str(directory), 'writers': n, 'heads': heads, 'dim': dim, 'density_mode': density_mode,
              'beta_min': float(beta.min()), 'beta_max': float(beta.max()),
              'condition_max': max(conditions),
              'block_vs_native_read_relative_l2': float(np.sqrt(native_gap_num / max(native_norm_sq, 1e-30)))}
    passed = True
    for label, got, expected in zip(('output', 'brain', 'hand'), actual, (expected_o, expected_b, expected_h)):
        absolute = float(np.max(np.abs(got - expected)))
        reference_max = float(np.max(np.abs(expected)))
        finite = bool(np.isfinite(got).all())
        report[label] = {'max_abs_error': absolute, 'reference_max_abs': reference_max,
                         'relative_l2': float(np.linalg.norm((got - expected).ravel()) / max(np.linalg.norm(expected.ravel()), 1e-30)),
                         'finite': finite}
        passed &= finite and absolute <= 2e-4 * max(1.0, reference_max)
    report['numerical_check'] = bool(passed)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    blocks = sorted(args.directory.glob('block-*'))
    if not blocks:
        raise ValueError('no captured block inputs')
    reports = [replay(block) for block in blocks]
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0 if all(r['numerical_check'] for r in reports) else 1


if __name__ == '__main__':
    raise SystemExit(main())
