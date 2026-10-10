#!/usr/bin/env python3
"""Small CPU algebra fixtures, not five-GPU or real-model qualification."""
import importlib.util
import math
from pathlib import Path
import random
import unittest

spec = importlib.util.spec_from_file_location('hc_coordinate', Path(__file__).resolve().parents[1] / 'scripts/tp5-hc-coordinate-prototype.py')
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)


def fixture(seed=1, rows=2, width=7, streams=4, latent=3, ranks=5):
    rng = random.Random(seed)
    def values(*shape):
        return [values(*shape[1:]) for _ in range(shape[0])] if shape else rng.uniform(-1, 1)
    weights = hc.Weights(values(streams * width), values(latent, streams * width),
                         values(streams * width, latent), values(streams, streams * width))
    return values(rows, streams, width), values(ranks, rows, width), values(rows, streams), weights


class CoordinateTests(unittest.TestCase):
    def equal_tree(self, a, b):
        if isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.equal_tree(x, y)
        else:
            self.assertTrue(math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11), (a, b))

    def test_full_mix_and_next_state_across_shapes(self):
        for rows in (1, 2, 3):
            for width in (3, 7, 13):
                for seed in range(4):
                    with self.subTest(rows=rows, width=width, seed=seed):
                        args = fixture(seed=seed, rows=rows, width=width)
                        a, b = hc.reference(*args), hc.coordinate_owned(*args)
                        self.equal_tree(a, b)
                        block = [[(t + j) / 7 for j in range(width)] for t in range(rows)]
                        self.equal_tree(hc.combine_next(a, block), hc.combine_next(b, block))

    def test_noncontiguous_coordinate_ownership(self):
        args = fixture(width=13)
        partitions = [list(range(r, 13, 5)) for r in range(5)]
        self.equal_tree(hc.reference(*args), hc.coordinate_owned(*args, partitions=partitions))

    def test_missing_duplicate_and_invalid_coordinates_rejected(self):
        args = fixture(width=3)
        for parts in ([[0], [1], [], [], []], [[0], [1], [2], [2], []],
                      [[0], [1], [3], [], []], [[0], [1], [True], [], []]):
            with self.assertRaises(ValueError):
                hc.coordinate_owned(*args, partitions=parts)

    def test_cross_rank_square_terms_are_required(self):
        residual, partials, scatter, weights = fixture(rows=1, width=3)
        residual = [[[0.0] * 3 for _ in range(4)]]
        partials = [[[1.0] * 3], [[-1.0] * 3], [[0.0] * 3], [[0.0] * 3], [[0.0] * 3]]
        scatter = [[1.0] * 4]
        self.assertEqual(sum(sum(v * v for v in p[0]) for p in partials), 6.0)
        result = hc.coordinate_owned(residual, partials, scatter, weights)
        self.equal_tree(result.state, residual)
        self.equal_tree(result.mixed, [[0.0] * 3])
        self.equal_tree(result, hc.reference(residual, partials, scatter, weights))

    def test_zero_and_large_finite_residuals(self):
        for magnitude in (0.0, 1e4):
            residual, partials, scatter, weights = fixture()
            residual = [[[magnitude for _ in s] for s in row] for row in residual]
            self.equal_tree(hc.reference(residual, partials, scatter, weights),
                            hc.coordinate_owned(residual, partials, scatter, weights))

    def test_bad_shapes_and_nonfinite_values_rejected(self):
        args = fixture()
        with self.assertRaises(ValueError):
            hc.coordinate_owned(args[0], args[1], [], args[3])
        for eps in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                hc.coordinate_owned(*args, eps=eps)
        args[0][0][0][0] = float('inf')
        with self.assertRaises(ValueError):
            hc.coordinate_owned(*args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
