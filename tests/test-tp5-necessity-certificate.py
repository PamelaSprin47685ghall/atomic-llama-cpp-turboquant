#!/usr/bin/env python3
"""Tests for the v4 necessity certificate (scripts/tp5-necessity-certificate.py).

Each test pins one soundness property that v3 violated or that v4 newly
guarantees. The certificate is pure arithmetic on declared constants; no
GPU, no measured latency, no model load.
"""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'necessity_certificate', ROOT / 'scripts/tp5-necessity-certificate.py')
m = importlib.util.module_from_spec(spec)
sys.modules['necessity_certificate'] = m
spec.loader.exec_module(m)


class CertificateV4Tests(unittest.TestCase):
    def setUp(self):
        self.cert = m.derive()
        self.r = self.cert['resources']
        self.b = self.cert['bounds']

    def test_dense_bytes_match_audited_inventory(self):
        # DEFECT-1: v3 used 90.8 MB/layer; audited average is 63.82 MB.
        self.assertAlmostEqual(m.DENSE_BYTES_PER_LAYER, 63.820906666666666e6, places=3)
        self.assertLess(m.DENSE_BYTES_PER_LAYER, 90.8e6)

    def test_l2_credit_uses_active_set_not_weight_pool(self):
        # DEFECT-2: v3 divided L2 by the 11.9 GB pool and applied the 1.07%
        # hit fraction to the active set. v4 must credit the implementation
        # with up to 128 MB retained from the ACTIVE set.
        active = self.r['R3_active_weight_dram']['active_per_gpu_per_token_bytes']
        credit = self.r['R3_active_weight_dram']['l2_credit_bytes']
        self.assertEqual(credit, min(128e6, active))
        dram = self.r['R3_active_weight_dram']['dram_bytes_per_gpu_per_token']
        self.assertEqual(dram, active - credit)
        # The v3 pool-denominator credit would have been much smaller.
        pool = self.r['R3_active_weight_dram']['weight_pool_per_gpu_bytes']
        self.assertGreater(credit, m.L2_BYTES_PER_GPU * active / pool)

    def test_r3_excluded_from_global_bound(self):
        # DEFECT-3: under MTP row batching the active weights are read once
        # per CYCLE, so R3 amortizes across committed rows and cannot bound
        # the full allowed set. L_global must not include it.
        self.assertFalse(self.r['R3_active_weight_dram']['survives_mtp_relaxation'])
        r3 = self.r['R3_active_weight_dram']['time_per_row_s']
        g = self.b['L_global']['per_token_lower_bound_s_weak']
        self.assertLess(g, r3)
        # R4 binds the global bound.
        r4 = self.r['R4_fp32_alu']['time_per_row_s']
        self.assertAlmostEqual(g, r4, places=12)

    def test_r3_kept_for_nomtp_conditional_class(self):
        # The no-MTP class keeps R3; the bound must be the max including it.
        r3 = self.r['R3_active_weight_dram']['time_per_row_s']
        n = self.b['L_nomtp']['per_token_lower_bound_s_weak']
        self.assertAlmostEqual(n, r3, places=12)
        self.assertIn('no speculative row batching',
                      self.cert['declared_conditional_class_A_nomtp'])

    def test_global_bound_weaker_than_nomtp_bound(self):
        # More allowed implementations can only lower the optimum time.
        self.assertLess(self.b['L_global']['per_token_lower_bound_s_weak'],
                        self.b['L_nomtp']['per_token_lower_bound_s_weak'])

    def test_r4_flops_scale_with_rows_and_survive_mtp(self):
        # FLOPs are per row; MTP batching does not reduce per-public-token
        # FLOPs (h+1 rows each pay the full per-row work).
        self.assertTrue(self.r['R4_fp32_alu']['survives_mtp_relaxation'])
        self.assertGreater(self.r['R4_fp32_alu']['flops_per_token'], 1e10)
        self.assertLess(self.r['R4_fp32_alu']['flops_per_token'], 1e11)

    def test_r1_r2_survive_mtp(self):
        self.assertTrue(self.r['R1_output_emission']['survives_mtp_relaxation'])
        self.assertTrue(self.r['R2_causal_cut_set']['survives_mtp_relaxation'])

    def test_bounds_are_positive_and_finite(self):
        for name, bound in self.b.items():
            for key in ('per_token_lower_bound_s_weak', 'per_token_lower_bound_s_strong'):
                v = bound[key]
                self.assertTrue(math.isfinite(v) and v > 0, f'{name}.{key}')
            for key in ('tps_upper_bound_weak', 'tps_upper_bound_strong'):
                v = bound[key]
                self.assertTrue(math.isfinite(v) and v > 0, f'{name}.{key}')

    def test_strong_bound_at_least_weak(self):
        for name, bound in self.b.items():
            self.assertGreaterEqual(bound['per_token_lower_bound_s_strong'],
                                    bound['per_token_lower_bound_s_weak'])

    def test_active_expert_bytes_use_10_of_512(self):
        expert = m.active_expert_bytes_per_gpu()
        self.assertAlmostEqual(expert, m.EXPERT_BYTES_PER_LAYER * 10 / 512 * 48 / 5)
        dense = m.active_dense_bytes_per_gpu()
        self.assertAlmostEqual(dense, m.DENSE_BYTES_PER_LAYER * 48 / 5)
        self.assertGreater(expert, 0)
        self.assertGreater(dense, 0)

    def test_dram_time_consistent_with_bytes(self):
        dram = self.r['R3_active_weight_dram']['dram_bytes_per_gpu_per_token']
        t = self.r['R3_active_weight_dram']['time_per_row_s']
        self.assertAlmostEqual(t, dram / (m.DRAM_GB_S * 1e9), places=15)

    def test_v3_supersession_recorded(self):
        self.assertEqual(self.cert['version'], 'v4')
        self.assertIn('DEFECT-3', self.cert['supersedes'])
        self.assertIn('DEFECT-1', self.cert['supersedes'])
        self.assertIn('DEFECT-2', self.cert['supersedes'])

    def test_main_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'cert.json'
            import contextlib
            import io
            argv = sys.argv
            sys.argv = ['prog', '--json', str(out)]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    m.main()
            finally:
                sys.argv = argv
            data = json.loads(out.read_text())
            self.assertEqual(data['version'], 'v4')
            self.assertIn('L_global', data['bounds'])
            self.assertIn('L_nomtp', data['bounds'])

    def test_no_measured_latency_enters_bound(self):
        # All resource times derive from declared byte/FLOP counts over spec
        # capacity ceilings; none of the numeric bound inputs may cite any
        # measured time value (prose may mention the audit history).
        src = (ROOT / 'scripts/tp5-necessity-certificate.py').read_text()
        import re
        constants = re.findall(r'^[A-Z][A-Z0-9_]* = ([^#\n]+)', src, re.M)
        joined = ' '.join(constants)
        for banned in ('67.5', '50.8', '273.86', '28.6', '73.46'):
            self.assertNotIn(banned, joined, f'certificate constants must not cite {banned}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
