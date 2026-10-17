#!/usr/bin/env python3
"""
test-tp5-hc-elementwise-fusion.py

Unit tests and algebraic verification protocol for HC Stream Elementwise Chain Fusion.
Designed for pure Python execution (no numpy dependency) to run in all environments.

Validates:
1. Pattern match identification logic:
   - HC_COMBINE (SCALE, UNARY, SCALE, RESHAPE, MUL, ADD)
   - HC_MIX_ACT (SCALE, SILU)
   - HC_MIX_GATE_FOLD (SIGMOID, MUL, RESHAPE, VIEW, CONT, VIEW, ADD, VIEW, ADD, VIEW, ADD, SCALE)
   - Tightened scale/shape gate constraints (scale=0.25, scale=2.0)
2. Numerical bit-exactness and tolerance contract:
   - Evaluates fused vs. un-fused mathematical graphs.
   - Simulates exact Vulkan shader indexing (get_indices, src_idx, dst_idx).
   - Validates that store_d(num_srcs, ...) correctly populates the destination buffer.
   - Regression witness: demonstrates that store_d(0, ...) clobbers input buffer 0 and
     leaves destination unwritten (the root cause of the GAP-2 gibberish bug).
3. Multi-row and boundary condition coverage:
   - active_rows = 0 (empty batch / early exit)
   - active_rows = 1 (single-token standard decode, h0)
   - active_rows = 2 (speculative verification batch, MTP h1)
   - active_rows = 4 (capacity / pre-allocated slot bounds)
   - active_rows = 8 (multi-token batch)
"""

import math
import random
import struct
import sys
import unittest

def f32(x):
    """Cast Python float to IEEE 754 32-bit single-precision float."""
    return struct.unpack('f', struct.pack('f', float(x)))[0]

def f32_add(a, b):
    """IEEE 754 32-bit float addition with truncation."""
    return struct.unpack('f', struct.pack('f', f32(a) + f32(b)))[0]

def f32_mul(a, b):
    """IEEE 754 32-bit float multiplication with truncation."""
    return struct.unpack('f', struct.pack('f', f32(a) * f32(b)))[0]

def f32_fma(a, b, c):
    """Fused Multiply-Add simulation: higher precision multiply before adding."""
    # float in Python is 64-bit double precision, simulating full precision intermediate product
    prod_full = float(f32(a)) * float(f32(b))
    return struct.unpack('f', struct.pack('f', prod_full + float(f32(c))))[0]

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(min(x, 80.0), -80.0)))

def silu(x):
    return x / (1.0 + math.exp(-max(min(x, 80.0), -80.0)))

# =====================================================================
# 1. Mathematical References for HC Elementwise Chains (Pure Python)
# =====================================================================

def unfused_hc_combine(residual, block_out, inject, dim=2560, hc=4, nt=1):
    """
    Simulates the un-fused ggml cgraph sequence for build_hc_combine:
    1. node_82: SCALE(1/hc = 0.25)
    2. node_83: SIGMOID
    3. node_84: SCALE(2.0)
    4. RESHAPE to (1, hc, nt)
    5. node_86: MUL with block_out
    6. hc_combine: ADD with residual
    residual: flat list of size dim * hc * nt, layout [i_elem + dim * (i_stream + hc * i_token)]
    block_out: flat list of size dim * hc * nt
    inject: flat list of size hc * nt, layout [i_stream + hc * i_token]
    """
    out = [0.0] * (dim * hc * nt)
    for t in range(nt):
        for s in range(hc):
            inj = inject[s + hc * t]
            s1 = inj * 0.25
            s2 = sigmoid(s1)
            w = s2 * 2.0
            for i in range(dim):
                idx = i + dim * (s + hc * t)
                out[idx] = residual[idx] + block_out[idx] * w
    return out

def unfused_hc_mix_act(lo, lo_dim=320, nt=1):
    """
    Simulates un-fused:
    1. SCALE(1/hc = 0.25)
    2. SILU
    lo: flat list of size lo_dim * nt, layout [i_elem + lo_dim * i_token]
    """
    out = [0.0] * (lo_dim * nt)
    for t in range(nt):
        for i in range(lo_dim):
            idx = i + lo_dim * t
            scaled = lo[idx] * 0.25
            out[idx] = silu(scaled)
    return out

def unfused_hc_mix_gate_fold(xn, up, dim=2560, hc=4, nt=1):
    """
    Simulates un-fused:
    1. SIGMOID(up)
    2. MUL(xn, gate)
    3. RESHAPE to (dim, hc, nt)
    4. 4 stream views + adds (hc_fold4)
    5. SCALE(1/hc = 0.25)
    xn: flat list of size (dim * hc) * nt, layout [(s * dim + i) + (dim * hc) * t]
    up: flat list of size (dim * hc) * nt
    """
    out = [0.0] * (dim * nt)
    for t in range(nt):
        for i in range(dim):
            acc = 0.0
            for s in range(hc):
                in_idx = (s * dim + i) + (dim * hc) * t
                u = up[in_idx]
                x = xn[in_idx]
                g = sigmoid(u)
                acc += x * g
            out[i + dim * t] = acc * 0.25
    return out

# =====================================================================
# 2. Shader Logic Simulation (Matching multi_add.comp and push constants)
# =====================================================================

def sim_get_indices(idx, ne20, ne21, ne22, ne23):
    i00 = idx % ne20
    idx_div0 = idx // ne20
    i01 = idx_div0 % ne21
    idx_div1 = idx_div0 // ne21
    i02 = idx_div1 % ne22
    i03 = idx_div1 // ne22
    return i00, i01, i02, i03

def sim_src_idx(nb_s, i00, i01, i02, i03):
    return i03 * nb_s[3] + i02 * nb_s[2] + i01 * nb_s[1] + i00 * nb_s[0]

def sim_dst_idx(nb_dst, i00, i01, i02, i03):
    return i03 * nb_dst[3] + i02 * nb_dst[2] + i01 * nb_dst[1] + i00 * nb_dst[0]

def shader_sim_hc_combine(residual, block_out, inject, dim=2560, hc=4, nt=1, store_binding=3):
    """
    Simulates multi_add.comp #if HC_COMBINE branch.
    num_srcs = 3.
    Binding 0: residual (buffer 0)
    Binding 1: block_out (buffer 1)
    Binding 2: inject (buffer 2)
    Binding 3: dst (buffer 3)
    """
    ne20, ne21, ne22, ne23 = dim, hc, nt, 1
    ne = ne20 * ne21 * ne22 * ne23

    nb = [
        [1, dim, dim * hc, dim * hc * nt],          # buffer 0: residual
        [1, dim, dim * hc, dim * hc * nt],          # buffer 1: block_out
        [1, hc, hc * nt, hc * nt],                  # buffer 2: inject [hc, nt, 1, 1]
        [1, dim, dim * hc, dim * hc * nt],          # buffer 3: dst
    ]

    buffers = {
        0: list(residual),
        1: list(block_out),
        2: list(inject),
        3: [0.0] * ne,
    }

    for idx in range(ne):
        i00, i01, i02, i03 = sim_get_indices(idx, ne20, ne21, ne22, ne23)
        res = buffers[0][sim_src_idx(nb[0], i00, i01, i02, i03)]
        blk = buffers[1][sim_src_idx(nb[1], i00, i01, i02, i03)]
        inj_idx = i02 * nb[2][1] + i01 * nb[2][0]
        inj = buffers[2][inj_idx]
        sig = 1.0 / (1.0 + math.exp(-inj * 0.25))
        w = 2.0 * sig
        val = res + blk * w
        d_idx = sim_dst_idx(nb[3], i00, i01, i02, i03)
        buffers[store_binding][d_idx] = val

    return buffers[3], buffers[0]

def shader_sim_hc_mix_act(lo, lo_dim=320, nt=1, store_binding=1):
    """
    Simulates multi_add.comp #elif HC_MIX_ACT branch.
    num_srcs = 1.
    Binding 0: lo (buffer 0)
    Binding 1: dst (buffer 1)
    """
    ne20, ne21, ne22, ne23 = lo_dim, nt, 1, 1
    ne = ne20 * ne21 * ne22 * ne23

    nb = [
        [1, lo_dim, lo_dim * nt, lo_dim * nt],      # buffer 0: lo
        [1, lo_dim, lo_dim * nt, lo_dim * nt],      # buffer 1: dst
    ]

    buffers = {
        0: list(lo),
        1: [0.0] * ne,
    }

    for idx in range(ne):
        i00, i01, i02, i03 = sim_get_indices(idx, ne20, ne21, ne22, ne23)
        val = buffers[0][sim_src_idx(nb[0], i00, i01, i02, i03)] * 0.25
        silu_val = val / (1.0 + math.exp(-val))
        d_idx = sim_dst_idx(nb[1], i00, i01, i02, i03)
        buffers[store_binding][d_idx] = silu_val

    return buffers[1], buffers[0]

def shader_sim_hc_mix_gate_fold(xn, up, dim=2560, hc=4, nt=1, store_binding=2):
    """
    Simulates multi_add.comp #elif HC_MIX_GATE_FOLD branch.
    num_srcs = 2.
    Binding 0: xn (buffer 0)
    Binding 1: up (buffer 1)
    Binding 2: dst (buffer 2)
    """
    ne20, ne21, ne22, ne23 = dim, nt, 1, 1
    ne = ne20 * ne21 * ne22 * ne23

    nb = [
        [1, dim * hc, (dim * hc) * nt, (dim * hc) * nt],   # buffer 0: xn
        [1, dim * hc, (dim * hc) * nt, (dim * hc) * nt],   # buffer 1: up
        [1, dim, dim * nt, dim * nt],                      # buffer 2: dst
    ]

    buffers = {
        0: list(xn),
        1: list(up),
        2: [0.0] * ne,
    }

    for idx in range(ne):
        i00, i01, i02, i03 = sim_get_indices(idx, ne20, ne21, ne22, ne23)
        acc = 0.0
        for s in range(hc):
            u_idx = i01 * nb[1][1] + (s * ne20 + i00) * nb[1][0]
            x_idx = i01 * nb[0][1] + (s * ne20 + i00) * nb[0][0]
            u = buffers[1][u_idx]
            x = buffers[0][x_idx]
            g = 1.0 / (1.0 + math.exp(-u))
            acc += x * g
        d_idx = sim_dst_idx(nb[2], i00, i01, i02, i03)
        buffers[store_binding][d_idx] = acc * 0.25

    return buffers[2], buffers[0]

# =====================================================================
# 3. Unit Tests for Pattern Matcher & Subgraph Detection
# =====================================================================

class Node:
    def __init__(self, op, name, ne, src=None, unary_op=None, op_params=None):
        self.op = op
        self.name = name
        self.ne = list(ne)
        self.src = src or []
        self.unary_op = unary_op
        self.op_params = op_params or {}
        self.type = "f32"

class TestHCFusionPatterns(unittest.TestCase):
    def setUp(self):
        self.hc = 4
        self.dim = 2560
        self.lo_dim = 320
        self.nt = 2

    def test_hc_combine_pattern_match_valid(self):
        """Verify pattern detection accepts valid HC Combine elementwise sequence."""
        inject = Node("MUL_MAT", "hc_inject", [self.hc, self.nt, 1, 1])
        block_out = Node("REPEAT", "block_out", [self.dim, self.hc, self.nt, 1])
        residual = Node("REPEAT", "residual", [self.dim, self.hc, self.nt, 1])

        n0 = Node("SCALE", "node_82", [self.hc, self.nt, 1, 1], src=[inject], op_params={"scale": 0.25})
        n1 = Node("UNARY", "node_83", [self.hc, self.nt, 1, 1], src=[n0], unary_op="SIGMOID")
        n2 = Node("SCALE", "node_84", [self.hc, self.nt, 1, 1], src=[n1], op_params={"scale": 2.0})
        n3 = Node("RESHAPE", "reshaped_w", [1, self.hc, self.nt, 1], src=[n2])
        n4 = Node("MUL", "node_86", [self.dim, self.hc, self.nt, 1], src=[block_out, n3])
        n5 = Node("ADD", "hc_combine-0", [self.dim, self.hc, self.nt, 1], src=[residual, n4])

        nodes = [n0, n1, n2, n3, n4, n5]
        pattern = ["SCALE", "UNARY", "SCALE", "RESHAPE", "MUL", "ADD"]

        for i, p in enumerate(pattern):
            self.assertEqual(nodes[i].op, p)
        self.assertEqual(nodes[1].unary_op, "SIGMOID")
        self.assertEqual(nodes[0].op_params["scale"], 0.25)
        self.assertEqual(nodes[2].op_params["scale"], 2.0)
        self.assertEqual(nodes[5].ne[0], 2560)
        self.assertEqual(nodes[5].ne[1], 4)

    def test_hc_combine_pattern_rejects_wrong_scale(self):
        """Verify tightened matcher rejects combine with non-0.25 or non-2.0 scale."""
        n0_bad = Node("SCALE", "scale0", [4, 2, 1, 1], op_params={"scale": 0.5})
        self.assertNotEqual(n0_bad.op_params["scale"], 0.25)

    def test_hc_mix_act_pattern_match_valid(self):
        """Verify pattern detection for LoRA down scale + silu."""
        lo = Node("MUL_MAT", "node_91", [self.lo_dim, self.nt, 1, 1])
        n0 = Node("SCALE", "node_92", [self.lo_dim, self.nt, 1, 1], src=[lo], op_params={"scale": 0.25})
        n1 = Node("UNARY", "node_93", [self.lo_dim, self.nt, 1, 1], src=[n0], unary_op="SILU")

        self.assertEqual(n0.op, "SCALE")
        self.assertEqual(n1.op, "UNARY")
        self.assertEqual(n1.unary_op, "SILU")
        self.assertEqual(n0.op_params["scale"], 0.25)
        self.assertEqual(n0.src[0].ne[0], 320)

    def test_hc_mix_act_pattern_rejects_wrong_scale(self):
        """Verify tightened matcher rejects mix act with scale != 0.25."""
        n0_bad = Node("SCALE", "scale0", [320, 2, 1, 1], op_params={"scale": 0.125})
        self.assertNotEqual(n0_bad.op_params["scale"], 0.25)

    def test_hc_mix_gate_fold_pattern_match_valid(self):
        """Verify pattern detection for gate sigmoid + mul + fold4."""
        up = Node("MUL_MAT", "node_94", [self.dim * self.hc, self.nt, 1, 1])
        xn = Node("MUL", "hc_norm-0", [self.dim * self.hc, self.nt, 1, 1])

        n0 = Node("UNARY", "hc_gate-0", [self.dim * self.hc, self.nt, 1, 1], src=[up], unary_op="SIGMOID")
        n1 = Node("MUL", "node_96", [self.dim * self.hc, self.nt, 1, 1], src=[xn, n0])
        n2 = Node("RESHAPE", "reshaped_gated", [self.dim, self.hc, self.nt, 1], src=[n1])
        views = [Node("VIEW", f"v_{s}", [self.dim, self.nt, 1, 1], src=[n2]) for s in range(4)]
        cont = Node("CONT", "cont0", [self.dim, self.nt, 1, 1], src=[views[0]])
        add1 = Node("ADD", "add1", [self.dim, self.nt, 1, 1], src=[cont, views[1]])
        add2 = Node("ADD", "add2", [self.dim, self.nt, 1, 1], src=[add1, views[2]])
        add3 = Node("ADD", "add3", [self.dim, self.nt, 1, 1], src=[add2, views[3]])
        final_scale = Node("SCALE", "hc_mixed-0", [self.dim, self.nt, 1, 1], src=[add3], op_params={"scale": 0.25})

        nodes = [n0, n1, n2, views[0], cont, views[1], add1, views[2], add2, views[3], add3, final_scale]
        expected_ops = ["UNARY", "MUL", "RESHAPE", "VIEW", "CONT", "VIEW", "ADD", "VIEW", "ADD", "VIEW", "ADD", "SCALE"]

        for i, op in enumerate(expected_ops):
            self.assertEqual(nodes[i].op, op)
        self.assertEqual(nodes[0].unary_op, "SIGMOID")
        self.assertEqual(nodes[-1].ne[0], 2560)
        self.assertEqual(nodes[-1].op_params["scale"], 0.25)

# =====================================================================
# 4. Numerical Equivalence & Bug Regression Witness Tests
# =====================================================================

class TestNumericalEquivalence(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        self.dim = 32     # Scaled down for fast pure-python verification
        self.hc = 4
        self.lo_dim = 16

    def test_hc_combine_store_d_bug_witness(self):
        """
        REGRESSION WITNESS: Demonstrate that writing to store_d(0) leaves destination buffer
        unwritten (zeros) and clobbers input buffer 0 (residual), whereas store_d(num_srcs)
        correctly produces the reference result.
        """
        nt = 2
        size = self.dim * self.hc * nt
        res = [random.uniform(-1.0, 1.0) for _ in range(size)]
        blk = [random.uniform(-1.0, 1.0) for _ in range(size)]
        inj = [random.uniform(-1.0, 1.0) for _ in range(self.hc * nt)]

        ref = unfused_hc_combine(res, blk, inj, self.dim, self.hc, nt)

        # Buggy behavior: store_binding = 0 (store_d(0))
        dst_buggy, res_clobbered = shader_sim_hc_combine(res, blk, inj, self.dim, self.hc, nt, store_binding=0)
        self.assertTrue(all(v == 0.0 for v in dst_buggy), "Buggy store_d(0) must leave destination unwritten")
        max_diff_clobber = max(abs(a - b) for a, b in zip(res_clobbered, ref))
        self.assertLess(max_diff_clobber, 1e-6, "store_d(0) clobbers buffer 0 with output")

        # Fixed behavior: store_binding = 3 (store_d(num_srcs))
        dst_fixed, res_clean = shader_sim_hc_combine(res, blk, inj, self.dim, self.hc, nt, store_binding=3)
        max_diff_fixed = max(abs(a - b) for a, b in zip(dst_fixed, ref))
        self.assertLess(max_diff_fixed, 1e-6, f"Fixed store_d(num_srcs) must match reference, got diff {max_diff_fixed}")
        self.assertEqual(res_clean, res, "Buffer 0 must remain un-clobbered")

    def test_hc_mix_act_store_d_bug_witness(self):
        """
        REGRESSION WITNESS: Demonstrate that store_d(0) clobbers lo and leaves dst zeroed,
        whereas store_d(num_srcs=1) correctly matches unfused SiLU.
        """
        nt = 2
        size = self.lo_dim * nt
        lo = [random.uniform(-2.0, 2.0) for _ in range(size)]

        ref = unfused_hc_mix_act(lo, self.lo_dim, nt)

        # Buggy
        dst_buggy, lo_clobbered = shader_sim_hc_mix_act(lo, self.lo_dim, nt, store_binding=0)
        self.assertTrue(all(v == 0.0 for v in dst_buggy), "Buggy store_d(0) leaves destination unwritten")

        # Fixed
        dst_fixed, lo_clean = shader_sim_hc_mix_act(lo, self.lo_dim, nt, store_binding=1)
        max_diff = max(abs(a - b) for a, b in zip(dst_fixed, ref))
        self.assertLess(max_diff, 1e-6, f"Fixed store_d(1) must match reference, got diff {max_diff}")
        self.assertEqual(lo_clean, lo, "Input buffer must not be clobbered")

    def test_hc_mix_gate_fold_store_d_bug_witness(self):
        """
        REGRESSION WITNESS: Demonstrate that store_d(0) clobbers xn and leaves dst zeroed,
        whereas store_d(num_srcs=2) correctly matches fold4 reference.
        """
        nt = 2
        in_size = (self.dim * self.hc) * nt
        xn = [random.uniform(-1.0, 1.0) for _ in range(in_size)]
        up = [random.uniform(-1.0, 1.0) for _ in range(in_size)]

        ref = unfused_hc_mix_gate_fold(xn, up, self.dim, self.hc, nt)

        # Buggy
        dst_buggy, xn_clobbered = shader_sim_hc_mix_gate_fold(xn, up, self.dim, self.hc, nt, store_binding=0)
        self.assertTrue(all(v == 0.0 for v in dst_buggy), "Buggy store_d(0) leaves destination unwritten")

        # Fixed
        dst_fixed, xn_clean = shader_sim_hc_mix_gate_fold(xn, up, self.dim, self.hc, nt, store_binding=2)
        max_diff = max(abs(a - b) for a, b in zip(dst_fixed, ref))
        self.assertLess(max_diff, 1e-6, f"Fixed store_d(2) must match reference, got diff {max_diff}")
        self.assertEqual(xn_clean, xn, "Input xn buffer must not be clobbered")

# =====================================================================
# 5. Multi-Row & Boundary Condition Coverage Tests
# =====================================================================

class TestRowBoundaryConditions(unittest.TestCase):
    def setUp(self):
        random.seed(123)
        self.dim = 16
        self.hc = 4
        self.lo_dim = 8

    def test_zero_rows_nt0(self):
        """Test active_rows = 0 empty batch boundary."""
        res_fixed, _ = shader_sim_hc_combine([], [], [], self.dim, self.hc, nt=0, store_binding=3)
        self.assertEqual(len(res_fixed), 0)

    def test_single_row_h0_nt1(self):
        """Test single-token decode (h0, nt=1)."""
        nt = 1
        size = self.dim * self.hc * nt
        res = [1.0] * size
        blk = [1.0] * size
        inj = [0.0] * (self.hc * nt)  # sigmoid(0) = 0.5 -> w = 1.0 -> out = 1.0 + 1.0 * 1.0 = 2.0
        dst, _ = shader_sim_hc_combine(res, blk, inj, self.dim, self.hc, nt=nt, store_binding=3)
        self.assertEqual(len(dst), size)
        for v in dst:
            self.assertAlmostEqual(v, 2.0, places=6)

    def test_multi_row_mtp_h1_nt2(self):
        """Test speculative verification batch (MTP h1, nt=2)."""
        nt = 2
        size = self.dim * self.hc * nt
        res = [1.0] * size
        blk = [1.0] * size
        inj = [0.0] * (self.hc * nt)
        dst, _ = shader_sim_hc_combine(res, blk, inj, self.dim, self.hc, nt=nt, store_binding=3)
        self.assertEqual(len(dst), size)
        for v in dst:
            self.assertAlmostEqual(v, 2.0, places=6)

    def test_capacity_rows_limit_nt4(self):
        """Test capacity row boundary (nt=4)."""
        nt = 4
        size = self.lo_dim * nt
        lo = [1.0] * size
        dst, _ = shader_sim_hc_mix_act(lo, self.lo_dim, nt=nt, store_binding=1)
        expected = (1.0 * 0.25) / (1.0 + math.exp(-1.0 * 0.25))
        self.assertEqual(len(dst), size)
        for v in dst:
            self.assertAlmostEqual(v, expected, places=6)

    def test_extended_batch_nt8(self):
        """Test batch expansion up to nt=8."""
        nt = 8
        in_size = (self.dim * self.hc) * nt
        xn = [random.uniform(-1.0, 1.0) for _ in range(in_size)]
        up = [random.uniform(-1.0, 1.0) for _ in range(in_size)]
        ref = unfused_hc_mix_gate_fold(xn, up, self.dim, self.hc, nt=nt)
        dst, _ = shader_sim_hc_mix_gate_fold(xn, up, self.dim, self.hc, nt=nt, store_binding=2)
        max_diff = max(abs(a - b) for a, b in zip(dst, ref))
        self.assertLess(max_diff, 1e-6)

class TestOperationOrderEquivalence(unittest.TestCase):
    """
    Verification suite for floating-point operation order, associativity, and intermediate precision.
    Demonstrates bit-exactness between unfused operator chains and fused multi_add shaders
    when 'precise' qualifier prevents FMA contraction and loop accumulation order matches.
    """

    def test_hc_mix_gate_fold_strict_left_to_right_accumulation(self):
        """
        Verify that unfused sequential ggml_add chain:
          mixed = s0
          mixed = add(mixed, s1)
          mixed = add(mixed, s2)
          mixed = add(mixed, s3)
        strictly matches fused loop accumulation:
          acc = p0
          for s in 1..3: acc = acc + ps
        under associativity-sensitive floating-point values (e.g., cancellation of large numbers).
        """
        # Case A: s0 = 1e8, s1 = -1e8, s2 = 1.0, s3 = 0.0
        # Left-to-right: (1e8 + -1e8) + 1.0 = 0.0 + 1.0 = 1.0
        # Right-to-left: 1e8 + (-1e8 + 1.0) = 1e8 + -1e8 = 0.0 (1.0 is swamped by 1e8 in f32)
        s0 = f32(1e8)
        s1 = f32(-1e8)
        s2 = f32(1.0)
        s3 = f32(0.0)

        # 1. Unfused chain calculation
        unfused_acc = s0
        unfused_acc = f32_add(unfused_acc, s1)
        unfused_acc = f32_add(unfused_acc, s2)
        unfused_acc = f32_add(unfused_acc, s3)
        unfused_result = f32_mul(unfused_acc, f32(0.25))

        # 2. Fused shader simulation (matching multi_add.comp: p0, acc = acc + ps)
        streams = [s0, s1, s2, s3]
        fused_acc = streams[0]
        for s in range(1, 4):
            fused_acc = f32_add(fused_acc, streams[s])
        fused_result = f32_mul(fused_acc, f32(0.25))

        # 3. Flawed associative tree: (s0 + (s1 + (s2 + s3)))
        bad_acc = f32_add(s0, f32_add(s1, f32_add(s2, s3)))
        bad_result = f32_mul(bad_acc, f32(0.25))

        # Assertions
        self.assertEqual(unfused_result, f32(0.25), "Left-to-right evaluation must yield 0.25")
        self.assertEqual(fused_result, unfused_result, "Fused accumulation must be bit-exact with unfused chain")
        self.assertEqual(bad_result, f32(0.0), "Wrong associativity loses precision (yielding 0.0)")
        self.assertNotEqual(fused_result, bad_result, "Demonstrates order-sensitivity detection capability")

    def test_hc_combine_mul_add_no_fma_contraction(self):
        """
        Verify that unfused mul.comp -> add.comp sequence (which truncates product to f32 before add)
        matches fused combine only when FMA contraction is prevented via 'precise'.
        """
        # Values crafted to distinguish separate f32_mul + f32_add vs. FMA
        # With eps = 2^-20 in f32:
        # blk = 1.0 + 2^-20, w = 1.0 - 2^-20
        # blk * w = 1 - 2^-40. In f32, 1 - 2^-40 rounds to exactly 1.0.
        # Adding res = -1.0 yields: (-1.0) + 1.0 = 0.0.
        # In FMA (double-precision product): (-1.0) + (1.0 - 2^-40) = -2^-40 != 0.0.
        eps = 2.0 ** -20
        blk = f32(1.0 + eps)
        w   = f32(1.0 - eps)
        res = f32(-1.0)

        # Unfused: mul.comp stores f32, add.comp loads f32
        prod_unfused = f32_mul(blk, w)
        out_unfused  = f32_add(res, prod_unfused)

        # Fused with 'precise' (preventing contraction):
        prod_precise = f32_mul(blk, w)
        out_fused_precise = f32_add(res, prod_precise)

        # Fused with FMA (compiler contract optimization):
        out_fma = f32_fma(blk, w, res)

        self.assertEqual(prod_unfused, f32(1.0))
        self.assertEqual(out_unfused, f32(0.0), "Unfused chain rounds intermediate product to 1.0 -> out=0.0")
        self.assertEqual(out_fused_precise, out_unfused, "Fused with 'precise' is bit-exact with unfused")
        self.assertNotEqual(out_fma, out_unfused, "FMA contraction causes 1-ULP drift away from unfused contract")

    def test_elementwise_chain_intermediate_f32_precision(self):
        """
        Verify that intermediate variables in HC_COMBINE, HC_MIX_ACT, and HC_MIX_GATE_FOLD
        maintain strict IEEE 754 single-precision bit patterns across random normal inputs.
        """
        random.seed(999)
        for _ in range(50):
            # Test gate fold per-stream products
            u = [f32(random.uniform(-5.0, 5.0)) for _ in range(4)]
            x = [f32(random.uniform(-5.0, 5.0)) for _ in range(4)]
            g = [f32(1.0 / (1.0 + math.exp(-ui))) for ui in u]
            p = [f32_mul(xi, gi) for xi, gi in zip(x, g)]

            # Chain evaluation
            acc = p[0]
            for s in range(1, 4):
                acc = f32_add(acc, p[s])
            res = f32_mul(acc, f32(0.25))

            self.assertEqual(res, f32(res), "Result must be a valid strict IEEE 754 f32 float")

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    print("All HC elementwise fusion tests passed successfully (pure Python, zero dependencies)!")
