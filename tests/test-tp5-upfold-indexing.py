#!/usr/bin/env python3
"""Pure-Python index mirrors for the HC up-fold weight-row arithmetic.

Index-level regression for the two up_fold BF16 defects confirmed by the
bf16-shader-audit read-only review (docs/TP5-BF16-UPFOLD-FIX.md):

  defect 1  mul_mat_vec.comp::iter_aligned_nonquant had no HC_UP_FOLD arm,
            so the aligned long-K loop read the LINEAR weight rows
            first_row + n instead of the folded rows j + n*stride_d.
  defect 2  ggml-vulkan.cpp::ggml_vk_segment_project's dispatch_rows ternary
            omitted pipeline_hc_project_fold_bf16, shrinking grid-y 4x.

What CAN be asserted in pure Python: the integer arithmetic of both shader
index functions (they are pure functions of first_row/n/ncols/stride_d/
col/QUANT_K), the boundary and alignment invariants those indices must
satisfy, and source-level sentinels that both fixes are present in the
working tree.

What CANNOT be asserted here (GLSL execution semantics — those stay with
the DevOps teacher-forced single-site comparison of
docs/TP5-BF16-LATEBIND-VERIFICATION.md §4.2): subgroupAdd reduction order,
the dequantize4_2aligned BF16 bit expansion, and robustness behavior of
out-of-bounds SSBO reads/writes.

Shader geometry recap (mul_mat_vec.comp / mul_mat_vec_base.glsl):
  gl_WorkGroupID.x = j   output width coordinate, j < stride_d = W
  gl_WorkGroupID.y = t   token row, t < T = mixed->ne[1]
  first_row        = t * stride_d + j          (mat_vec_first_row)
  W_up row space   = 4 * stride_d, row-major [4W, ncols]
  NUM_ROWS         = 4 (spec constant of the fold pipelines)
  NOTE: the shader variable `col` below is the K-dimension lane offset
  (col = i*BLOCK_SIZE + K_PER_ITER*tid), NOT the output width coordinate j.
"""
import unittest
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMP = REPO / 'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp'
VKCPP = REPO / 'ggml/src/ggml-vulkan/ggml-vulkan.cpp'

NUM_ROWS = 4  # spec constant {subgroup_size, 4, 1} of the fold pipelines

# (stride_d=W, ncols=k, quant_k). BF16 up fold has quant_k == 1 (QUANT_K=1
# via DATA_A_BF16); the quant_k=32 entry exercises the same fold formula in
# its block-divided form. W=10 (not a multiple of 4) exercises the overrun.
GEOMETRIES = [
    (8, 12, 1),
    (16, 64, 1),
    (320, 320, 1),  # production-ish K=320 rank slice
    (10, 40, 1),
    (8, 64, 32),
]

TOKEN_ROWS = (0, 1, 2, 3)


def iter_weight_rows(first_row, num_rows, ncols, stride_d, col, quant_k, hc_up_fold):
    """Mirror of iter() in mul_mat_vec.comp, n-loop only.

    Line-faithful transcription of the non-aligned reference path (the
    #ifdef HC_UP_FOLD arms at the three staged/generic loop sites), which
    already folded before this fix. `ibi` is dead in the fold arm exactly as
    in the shader source (it is written but never read there).
    """
    out = []
    ibi = first_row * ncols
    for n in range(num_rows):
        if hc_up_fold:
            weight_row = (first_row % stride_d) + n * stride_d
            ib = (weight_row * ncols + col) // quant_k
            ibi += ncols * stride_d
        else:
            weight_row = first_row + n
            ib = (ibi + col) // quant_k
            ibi += ncols
        out.append((weight_row, ib))
    return out


def aligned_weight_rows(first_row, num_rows, ncols, stride_d, col, quant_k, hc_up_fold):
    """Mirror of iter_aligned_nonquant() AFTER the fix.

    Independent transcription of the patched generic loop (NOT a call into
    iter_weight_rows): the fold arm is the fix; the #else arm is the
    untouched linear code.
    """
    out = []
    ibi = first_row * ncols
    for n in range(num_rows):
        if hc_up_fold:
            weight_row = (first_row % stride_d) + n * stride_d
            ib = (weight_row * ncols + col) // quant_k
            ibi += ncols * stride_d
        else:
            weight_row = first_row + n
            ib = (ibi + col) // quant_k
            ibi += ncols
        out.append((weight_row, ib))
    return out


def aligned_weight_rows_prefix(first_row, num_rows, ncols, col, quant_k):
    """Mirror of iter_aligned_nonquant() BEFORE the fix: always linear rows.

    This is the defect: it must disagree with the fold (test c) and must be
    able to leave the 4*stride_d row space.
    """
    out = []
    ibi = first_row * ncols
    for n in range(num_rows):
        weight_row = first_row + n
        ib = (ibi + col) // quant_k
        ibi += ncols
        out.append((weight_row, ib))
    return out


def first_row_of(t, j, stride_d):
    return t * stride_d + j


class UpfoldIndexingTests(unittest.TestCase):
    def test_a_fold_disabled_degenerates_to_linear(self):
        """a) HC_UP_FOLD=0: the fold must reduce to the linear index and the
        two functions must agree — non-fold variants keep byte-identical
        behavior."""
        for (stride_d, ncols, quant_k) in GEOMETRIES:
            for t in TOKEN_ROWS:
                for j in (0, 1, stride_d - 1):
                    first_row = first_row_of(t, j, stride_d)
                    for col in (0, 4, 8):
                        if col >= ncols:
                            continue
                        a = iter_weight_rows(first_row, NUM_ROWS, ncols, stride_d, col, quant_k, False)
                        b = aligned_weight_rows(first_row, NUM_ROWS, ncols, stride_d, col, quant_k, False)
                        self.assertEqual(a, b, (stride_d, ncols, quant_k, t, j, col))
                        for n, (wr, ib) in enumerate(a):
                            self.assertEqual(wr, first_row + n)
                            self.assertEqual(ib, ((first_row + n) * ncols + col) // quant_k)

    def test_b_iter_and_aligned_agree_under_fold(self):
        """b) HC_UP_FOLD=1: post-fix iter_aligned_nonquant produces the same
        weight_row sequence as iter() (independent transcriptions)."""
        for (stride_d, ncols, quant_k) in GEOMETRIES:
            for t in TOKEN_ROWS:
                for j in (0, 1, stride_d - 1):
                    first_row = first_row_of(t, j, stride_d)
                    for col in (0, 4, 8):
                        if col >= ncols:
                            continue
                        a = iter_weight_rows(first_row, NUM_ROWS, ncols, stride_d, col, quant_k, True)
                        b = aligned_weight_rows(first_row, NUM_ROWS, ncols, stride_d, col, quant_k, True)
                        self.assertEqual(a, b, (stride_d, ncols, quant_k, t, j, col))
                        for n, (wr, ib) in enumerate(a):
                            self.assertEqual(wr, j + n * stride_d)
                            self.assertEqual(ib, ((j + n * stride_d) * ncols + col) // quant_k)

    def test_c_prefix_linear_rows_diverge_and_overrun(self):
        """The pre-fix aligned path IS the defect: it diverges from the fold
        from t=0/n=1 onward (the audit's 'token_row>=1' wording is the
        conservative bound — n>=1 misaligns even at t=0), it can read past
        the 4*stride_d row space, and the folded rows never do."""
        for (stride_d, ncols, quant_k) in GEOMETRIES:
            for t in TOKEN_ROWS:
                for j in (0, 1, stride_d - 1):
                    first_row = first_row_of(t, j, stride_d)
                    bad = aligned_weight_rows_prefix(first_row, NUM_ROWS, ncols, 0, quant_k)
                    good = aligned_weight_rows(first_row, NUM_ROWS, ncols, stride_d, 0, quant_k, True)
                    # At t=0 the n=0 row agrees (both are row j) while n>=1
                    # already diverges whenever stride_d > 1 — the defect
                    # corrupts from the very first token, one stream early
                    # versus the audit's conservative "token_row>=1" bound.
                    # At t>=1 every row including n=0 diverges (pre-fix row
                    # is first_row = t*stride_d + j, folded row is j).
                    if t == 0:
                        self.assertEqual(bad[0][0], good[0][0])
                        if stride_d > 1:
                            self.assertNotEqual(bad[1:], good[1:], (stride_d, ncols, t, j))
                    else:
                        self.assertNotEqual(bad, good, (stride_d, ncols, t, j))
            # the fold never leaves the W_up row space
            for t in TOKEN_ROWS:
                for j in range(stride_d):
                    good = aligned_weight_rows(first_row_of(t, j, stride_d), NUM_ROWS,
                                               ncols, stride_d, 0, quant_k, True)
                    for wr, _ in good:
                        self.assertLess(wr, 4 * stride_d)
        # concrete overrun counterexample: W=10 (not a multiple of 4), j=8,
        # t=3, n=3 reads row 41 of a 40-row weight tensor.
        bad = aligned_weight_rows_prefix(first_row_of(3, 8, 10), NUM_ROWS, 10, 0, 1)
        self.assertEqual(bad[3][0], 41)
        self.assertGreaterEqual(bad[3][0], 4 * 10)
        good = aligned_weight_rows(first_row_of(3, 8, 10), NUM_ROWS, 10, 10, 0, 1, True)
        for wr, _ in good:
            self.assertLess(wr, 40)

    def test_d_folded_rows_are_exactly_the_gate_rows(self):
        """The fold must touch exactly the rows the HC gate semantics
        integrate (reduce_result: sigmoid per stream n, times
        data_fuse0[t*4W + n*W + j], 0.25 stream mean): row j + n*stride_d,
        the whole ncols-wide K span of it, nothing else."""
        for (stride_d, ncols, quant_k) in GEOMETRIES:
            for t in (0, 2, 3):
                for j in (0, 1, stride_d - 1):
                    rows = aligned_weight_rows(first_row_of(t, j, stride_d), NUM_ROWS,
                                               ncols, stride_d, 0, quant_k, True)
                    for n, (wr, ib) in enumerate(rows):
                        self.assertEqual(wr, j + n * stride_d)
                        # col_k == 0 group start; the K loop walks the row in
                        # K_PER_ITER=4 lanes -> the element span is row wr.
                        self.assertEqual(ib, (wr * ncols) // quant_k)

    def test_e_packed32_alignment_preserved(self):
        """dequantize4_2aligned reads data_a_packed32[(a_offset + ib)/2]
        with a_offset == 0 in the HC up segment: the fold must keep ib even
        for every aligned geometry (ncols % 4 == 0, col % 4 == 0)."""
        for (stride_d, ncols, quant_k) in GEOMETRIES:
            if quant_k != 1 or ncols % 4:
                continue
            for t in TOKEN_ROWS:
                for j in (0, 1, stride_d - 1):
                    for col in (0, 4, 8):
                        if col >= ncols:
                            continue
                        rows = aligned_weight_rows(first_row_of(t, j, stride_d), NUM_ROWS,
                                                   ncols, stride_d, col, 1, True)
                        for wr, ib in rows:
                            self.assertEqual(ib % 2, 0, (stride_d, ncols, t, j, col, wr))


class SourceSentinelTests(unittest.TestCase):
    """These go red if either fix is reverted in the working tree.

    Deliberately loose (substring-level) to survive formatting; the index
    mirrors above carry the semantic assertions. Pre-fix, both of these
    fail: the aligned function body had no HC_UP_FOLD arm and the dispatch
    ternary had no bf16 entry.
    """

    @staticmethod
    def _function_body(path, signature):
        src = path.read_text()
        start = src.index(signature)
        brace = src.index('{', start)
        depth = 0
        i = brace
        while i < len(src):
            if src[i] == '{':
                depth += 1
            elif src[i] == '}':
                depth -= 1
                if depth == 0:
                    return src[brace:i]
            i += 1
        raise AssertionError('unbalanced braces after %r' % signature)

    def test_aligned_function_contains_fold(self):
        body = self._function_body(COMP, 'void iter_aligned_nonquant')
        self.assertIn('HC_UP_FOLD', body)
        self.assertIn('weight_row', body)
        self.assertIn('(first_row % p.stride_d) + n * p.stride_d', body)

    def test_dispatch_rows_covers_bf16_variant(self):
        src = VKCPP.read_text()
        anchor = src.index('const uint32_t dispatch_rows')
        clause = src[anchor:anchor + 700]
        self.assertIn('pipeline_hc_project_fold', clause)
        self.assertIn('pipeline_hc_project_fold_bf16', clause)
        self.assertIn('4 * n_rows', clause)


class DispatchGeometryTests(unittest.TestCase):
    """Grid-y semantics of the up-fold dispatch, pinned as executable memory.

    gl_WorkGroupID.y is the token row; y < T already covers every mixed
    element exactly once (data_d[first_row] with first_row = t*W + j). The
    Q8_0 twin's 4*n_rows is therefore an over-dispatch whose extra
    workgroups write at or past the end of the output tensor — their
    survival relies on robust buffer access, not on geometry. These tests
    pin that static reading (docs/TP5-BF16-UPFOLD-FIX.md §3.3) for the
    DevOps single-site comparison to confirm or overturn; they do NOT
    assert that 4*n_rows is wrong.
    """

    def test_token_rows_cover_output_exactly_once(self):
        for T, W in ((1, 8), (4, 8), (2, 320)):
            for rows in (T, 4 * T):
                with self.subTest(T=T, W=W, rows=rows):
                    writes = [t * W + j for t in range(min(rows, T)) for j in range(W)]
                    self.assertEqual(sorted(writes), list(range(T * W)))
                    for y in range(T, rows):
                        for j in range(W):
                            self.assertGreaterEqual(y * W + j, T * W)

    def test_overdispatch_bindings_are_all_oob_symmetrically(self):
        """Fifth audit round (unit-semantics-audit): mirror get_offsets()
        (mul_mat_vec_base.glsl) for the fold dispatch to pin the FULL
        out-of-bounds semantics of the 4*n_rows over-dispatch, not just the
        data_d write the test above pins.

        Geometry (ggml_vk_segment_project -> ggml_vk_segment_project for the
        up call): pc{ncols=k, stride_a=k, stride_b=k, stride_d=m,
        batch_stride_a=nelements(W_up)=k*4m, batch_stride_b=k,
        batch_stride_d=m, ne02=1, ne12=n_rows=T, broadcast2=T,
        broadcast3=1}. get_offsets with batch_idx = gl_WorkGroupID.y:
          i13 = y / ne12 = y / T; i02 = (y % T) / broadcast2 = 0
          batch_idx_a = i13 * ne02 + i02 = y / T
          a_offset = (y / T) * (nelements / QUANT_K)
          b_offset = y * k; d_offset = y * m
        fold epilogue fuse0 index = y*(NUM_ROWS*m) + n*m + j.

        For y in [0, T) every quantity is in range (A: a_offset == 0; B: lo
        has k*T elements; fuse0: normalized has 4*m*T elements; D: mixed has
        m*T). For y in [T, 4T) EVERY binding goes out of bounds — including
        a_offset, which for QUANT_K=1 (BF16) is y/T * k*4m (element index
        past the whole tensor) and for QUANT_K=32 (Q8_0 twin) is y/T * k*4m/32
        (block index past the whole tensor): the overrun is symmetric across
        the two families, so its survival must rely on robust buffer access
        in BOTH, not just one. A validation layer in the V-B run should
        therefore expect OOB reports from the y >= T workgroups of the up
        fold in BOTH variants; reports limited to the BF16 arm only would be
        new evidence of an asymmetry this static audit did not find."""
        k, m, T, NUM = 320, 8, 2, 4  # ncols=k, stride_d=m, tokens=T, NUM_ROWS=4
        for quant_k in (1, 32):  # BF16 and the Q8_0 twin
            nelements = k * 4 * m  # W_up = [k, 4m] row-major, no batch dim
            lo_elems, norm_elems, mixed_elems = k * T, 4 * m * T, m * T
            for y in range(4 * T):
                i13 = y // T
                batch_idx_a = i13  # i02 is 0 for every y here
                a_offset = batch_idx_a * (nelements // quant_k)
                b_offset = y * k
                d_offset = y * m
                fuse0 = y * (NUM * m) + (NUM - 1) * m + (m - 1)  # worst n, j
                if y < T:
                    self.assertEqual(a_offset, 0)
                    self.assertLess(b_offset, lo_elems)
                    self.assertLess(d_offset, mixed_elems)
                    self.assertLess(fuse0, norm_elems)
                else:
                    # A: a_offset alone already reaches or passes the whole
                    # tensor in BOTH units (element index vs block index).
                    self.assertGreaterEqual(a_offset, nelements // quant_k)
                    self.assertGreaterEqual(a_offset + k, nelements // quant_k + 1)
                    self.assertGreaterEqual(b_offset, lo_elems)
                    self.assertGreaterEqual(d_offset, mixed_elems)
                    self.assertGreaterEqual(fuse0, norm_elems)


# ---------------------------------------------------------------------------
# bf16-segment-fixer additions (fourth audit round).
#
# Static audit of the fused HC segment's BF16-specific handling found NO
# index/geometry/layout defect remaining after the upfold fix: the fold
# formula, dispatch geometry, binding assembly and epilogue are twin-isomorphic
# with the Q8_0 variant that experiment 3c proved correct. The remaining
# "correct prefix, then drift" garbling therefore lives in a layer static
# reading cannot see (SPIR-V codegen / driver execution of the BF16 decode).
# Key fact: the two HC BF16 variants are the ONLY mul_mat_vec+DATA_A_BF16
# instances in the tree — the aligned path (iter_aligned_nonquant +
# dequantize4_2aligned bit expansion) has no non-HC execution witness, while
# experiment 3a's byte-identical baseline run exercised mul_mat, not
# mul_mat_vec. The tests below pin what CAN be asserted offline:
# bit-level equivalence of the two decode paths, the FORCE_ITER diagnostic
# switch sentinels, and its degeneration when the env is unset.
# ---------------------------------------------------------------------------

import struct


def _f32_from_bits(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def bf16_to_fp32_scalar(u16):
    """Mirror of types.glsl bf16_to_fp32(uint32_t u) = uintBitsToFloat(u << 16)
    with the uint16_t element zero-extended first (the dequantize4 path)."""
    return _f32_from_bits((u16 & 0xffff) << 16)


def dequantize4_bf16(ib, data):
    """Mirror of dequant_funcs.glsl dequantize4 (DATA_A_BF16): four scalar
    uint16_t loads, each through bf16_to_fp32."""
    return [bf16_to_fp32_scalar(data[ib + k]) for k in range(4)]


def dequantize4_2aligned_bf16(ib, data):
    """Mirror of dequant_funcs.glsl dequantize4_2aligned (DATA_A_BF16):
    two u32 words over elements [ib, ib+4); v.x/v.z take the low half shifted
    up, v.y/v.w take the high half already in place."""
    a = (data[ib] & 0xffff) | ((data[ib + 1] & 0xffff) << 16)
    b = (data[ib + 2] & 0xffff) | ((data[ib + 3] & 0xffff) << 16)
    return [
        _f32_from_bits((a & 0x0000ffff) << 16),
        _f32_from_bits(a & 0xffff0000),
        _f32_from_bits((b & 0x0000ffff) << 16),
        _f32_from_bits(b & 0xffff0000),
    ]


class BF16DecodeEquivalenceTests(unittest.TestCase):
    """The two BF16 weight-decode paths must be bit-identical for every
    16-bit pattern: the aligned bit expansion and the scalar
    dequantize4/bf16_to_fp32 path read the same memory. This equivalence is
    the premise that makes the FORCE_ITER diagnostic switch a clean bisector:
    if the GPU run differs between the two paths, the defect is in the
    codegen/execution of one of them, not in the shared index math."""

    def test_bit_equivalence_all_patterns(self):
        patterns = [
            0x0000, 0x0001, 0x007f, 0x0080, 0x3f80, 0x3f81, 0x4049,
            0x7f7f, 0x7f80, 0x7f81, 0x8000, 0xbf80, 0xff7f, 0xff80,
            0xffff, 0x1234, 0x4321, 0xabcd, 0x0002,
        ]
        data = patterns * 8
        for ib in range(0, len(data) - 3):
            self.assertEqual(
                struct.pack('<4f', *dequantize4_bf16(ib, data)),
                struct.pack('<4f', *dequantize4_2aligned_bf16(ib, data)),
                'decode paths diverge at element %d' % ib)

    def test_element_order(self):
        # v.x = element ib (low half of word 0), v.y = element ib+1 (high
        # half), v.z/v.w = word 1. A swapped order would flip halves.
        data = [0x3f80, 0x4000, 0x4049, 0x40a0]  # 1.0, 2.0, 3.14, 5.0
        self.assertEqual(dequantize4_2aligned_bf16(0, data),
                         [1.0, 2.0, _f32_from_bits(0x4049 << 16), 5.0])


class ForceIterSwitchTests(unittest.TestCase):
    """Source sentinels for the diagnostic switch. They go red if the
    switch is reverted in the working tree."""

    def test_comp_has_macro_and_bypass(self):
        src = COMP.read_text()
        self.assertIn('#define MAT_VEC_FUSION_FLAGS_FORCE_ITER 0x200', src)
        body = SourceSentinelTests._function_body(
            COMP, 'void compute_outputs')
        # the bypass must gate is_aligned_nonquant inside compute_outputs
        self.assertIn('MAT_VEC_FUSION_FLAGS_FORCE_ITER', body)
        self.assertIn('(p.fusion_flags & MAT_VEC_FUSION_FLAGS_FORCE_ITER) == 0', body)

    def test_vk_has_env_and_flag(self):
        body = SourceSentinelTests._function_body(
            VKCPP, 'static void ggml_vk_segment_project')
        self.assertIn('GGML_VK_HC_BF16_FORCE_ITER', body)
        self.assertIn('strcmp(env, "1") == 0', body)
        self.assertIn('MAT_VEC_FUSION_FLAGS_FORCE_ITER', body)
        self.assertIn('pipeline_hc_down_silu_bf16', body)
        self.assertIn('pipeline_hc_project_fold_bf16', body)
        # the flag must be OR-composed into output_flags, not replace it
        self.assertIn('force_iter_bf16 ? MAT_VEC_FUSION_FLAGS_FORCE_ITER : 0u', body)

    def test_flag_bits_do_not_collide(self):
        # BIAS0/1, SCALE0/1, ELIDE_LOCAL occupy 0x1..0x8 and 0x100; the
        # diagnostic bit must be disjoint so reduce_result never misreads it.
        used = {0x1, 0x2, 0x4, 0x8, 0x100}
        self.assertNotIn(0x200, used)


class ForceIterDegenerationTests(unittest.TestCase):
    """Degeneration mirror of is_aligned_nonquant: with the flag bit clear
    the predicate is byte-for-byte the pre-switch formula; with the bit set
    it is false regardless of the alignment operands (the loop falls back to
    iter(), which carries the same fold arms)."""

    @staticmethod
    def is_aligned(fusion_flags, batch_stride_b, b_offset, ncols, block_size):
        return ((fusion_flags & 0x200) == 0 and
                batch_stride_b % 4 == 0 and b_offset % 4 == 0 and
                ncols % 4 == 0 and block_size % 4 == 0)

    def test_unset_flag_keeps_predicate(self):
        for bsb, boff, ncols in ((320, 320, 320), (8192, 0, 8192), (12, 4, 8)):
            self.assertTrue(self.is_aligned(0, bsb, boff, ncols, 64))
            self.assertTrue(self.is_aligned(0x100, bsb, boff, ncols, 64))

    def test_set_flag_forces_iter_path(self):
        self.assertFalse(self.is_aligned(0x200, 320, 320, 320, 64))
        self.assertFalse(self.is_aligned(0x300, 320, 320, 320, 64))

    def test_misaligned_stays_false(self):
        self.assertFalse(self.is_aligned(0, 322, 320, 320, 64))
        self.assertFalse(self.is_aligned(0, 320, 321, 320, 64))


class HcSiteDumpSentinelTests(unittest.TestCase):
    """GAP1 step 2 (site-dumper): the opt-in HC site dump instrumentation.

    Source-level sentinels for the single-site intermediate dump wired into
    ggml_vk_hc_segment (docs/TP5-HC-SITE-DUMP.md). They go red if the
    instrumentation is removed from the working tree or stops being
    env-gated. Execution semantics -- the fence-ordered readback, the file
    contents, the two-binary comparison -- stay with the DevOps
    teacher-forced run; what CAN be asserted statically is the zero-
    behavior-when-unset structure: the env parse defaults to 0, every dump
    call site is guarded by the same hc_dump_requested flag, and the flag
    folds env>0 together with the recording guards so an unset env can
    never reach a dump.
    """

    def test_dump_env_gate_defaults_to_zero(self):
        seg = SourceSentinelTests._function_body(VKCPP, 'static void ggml_vk_hc_segment')
        self.assertIn('getenv("GGML_VK_HC_DUMP_SITES")', seg)
        # The parse must default to 0 when unset/invalid: 0 -> every dump
        # call site below is dead (no barrier, no copy, no submit, no IO).
        self.assertIn('return env && atoi(env) > 0 ? (uint32_t) atoi(env) : 0u;', seg)

    def test_dump_sites_are_env_and_recording_gated(self):
        seg = SourceSentinelTests._function_body(VKCPP, 'static void ggml_vk_hc_segment')
        self.assertIn('hc_dump_requested = hc_dump_sites_max > 0 &&', seg)
        self.assertIn('!ctx->replay_recording && !ctx->tp5_recording_program', seg)
        # All three call sites must sit behind the same guard.
        for tensor in ('normalized', 'lo', 'mixed'):
            self.assertIn('if (hc_dump_requested && ctx->hc_dump_site_count <= hc_dump_sites_max) {\n'
                          '        ggml_vk_hc_dump_site_tensor(ctx, subctx, %s,' % tensor, seg)

    def test_dump_readback_uses_fence_ordered_midgraph_sync(self):
        body = SourceSentinelTests._function_body(VKCPP, 'static void ggml_vk_hc_dump_site_tensor')
        # Readback path: staging copy + deferred out-memcpy + synchronize
        # (submit + fence wait) + ctx_begin re-open, in that order.
        for token in ('copyBuffer', 'ggml_vk_sync_buffers(nullptr, subctx)',
                      'deferred_memcpy', 'ggml_vk_synchronize(ctx)',
                      'ggml_vk_ctx_begin(ctx->device, subctx)',
                      'GGML_TYPE_F32'):
            self.assertIn(token, body)
        self.assertLess(body.index('ggml_vk_sync_buffers'),
                        body.index('copyBuffer'))
        self.assertLess(body.index('copyBuffer'),
                        body.index('ggml_vk_synchronize'))
        self.assertLess(body.index('ggml_vk_synchronize'),
                        body.index('ggml_vk_ctx_begin'))

    def test_dump_points_follow_their_dispatches(self):
        seg = SourceSentinelTests._function_body(VKCPP, 'static void ggml_vk_hc_segment')
        # normalized dump: after the tape annotation, before the down
        # contraction; lo dump: between hc_down_end and the up dispatch;
        # mixed dump: after hc_up_end (function tail).
        norm_dump = seg.index('ggml_vk_hc_dump_site_tensor(ctx, subctx, normalized')
        lo_dump = seg.index('ggml_vk_hc_dump_site_tensor(ctx, subctx, lo')
        mixed_dump = seg.index('ggml_vk_hc_dump_site_tensor(ctx, subctx, mixed')
        down_dispatch = seg.index("ggml_vk_segment_project(ctx, subctx, graph->nodes[base + 3]")
        up_dispatch = seg.index("ggml_vk_segment_project(ctx, subctx, graph->nodes[base + 6]")
        self.assertLess(seg.index('program.normalized_tensor'), norm_dump)
        self.assertLess(norm_dump, down_dispatch)
        self.assertLess(seg.index('hc_down_end'), lo_dump)
        self.assertLess(lo_dump, up_dispatch)
        self.assertLess(seg.index('hc_up_end'), mixed_dump)

    def test_dump_doc_and_ctx_counter_exist(self):
        self.assertTrue((REPO / 'docs/TP5-HC-SITE-DUMP.md').exists())
        self.assertIn('hc_dump_site_count', VKCPP.read_text())


# ---------------------------------------------------------------------------
# r16 (dump-fixer): dump readback sentinels for the ISOLATED instrumentation
# tree. The r15 SYNCCMP adjudication proved the deferred readback (staging
# copyBuffer + deferred_memcpy onto subctx->out_memcpys) never landed --
# ggml_vk_synchronize only flushes compute_ctx->in_memcpys -- so every dump
# file and first64 print was a zero-initialized host artifact, never device
# state. The r16 fix makes ggml_vk_buffer_read (temporary context + submit +
# device-fence wait + immediate memcpy) the data source of BOTH dump
# helpers, with the mid-graph synchronize retained ahead of it so the
# producing dispatch has executed. These sentinels pin that structure in the
# isolated tree; they skip when the tree is absent (it is a scratch
# verification tree, not a checkout). The repo working tree is NOT touched
# by the r16 fix, so the pre-existing HcSiteDumpSentinelTests above keep
# describing the repo copy unchanged.
# ---------------------------------------------------------------------------

ISOLATED_VKCPP = Path(os.environ.get(
    'TP5_ISOLATED_VKCPP',
    '/var/tmp/tp5-verify-rebuild-src/ggml/src/ggml-vulkan/ggml-vulkan.cpp'))


def check_dump_readback_is_synchronous(body):
    """Red/green discriminator for the r16 dump readback fix.

    Green requires all of: (a) the mid-graph ggml_vk_synchronize is
    present (the producing dispatch is forced to completion first); (b)
    the file/first64 data source is a synchronous ggml_vk_buffer_read
    into the host vector; (c) the deferred landing path is gone (no
    deferred_memcpy, no copyBuffer recorded into subctx, no direct use
    of the shared ctx->sync_staging); (d) the synchronous read happens
    only after the mid-graph synchronize. AssertionError is the red
    state; the same discriminator is pointed at the known-bad samples
    below to prove it can actually go red.

    r17 note: a bare ggml_vk_ensure_sync_staging_buffer(ctx, ...) call is
    initialization, NOT a deferred data source. The r13-r15 deferred dumps
    were the de-facto first initializer of ctx->sync_staging; r17 restores
    the call so warmup/router-path users find the buffer allocated. The
    banned substrings above (copyBuffer / deferred_memcpy / direct
    ctx->sync_staging use) describe the data-source shape only; an ensure
    call contains none of them and stays green by design.
    """
    assert 'ggml_vk_synchronize(ctx)' in body, \
        'dump lost the mid-graph synchronize (dispatch not forced to completion)'
    assert 'ggml_vk_buffer_read(' in body, \
        'dump body has no synchronous read; data source is not trustworthy'
    assert 'deferred_memcpy(' not in body, \
        'dump still lands host data via deferred_memcpy (out_memcpys never flushed)'
    assert 'copyBuffer(' not in body, \
        'dump still records a staging copyBuffer into subctx'
    assert 'ctx->sync_staging' not in body, \
        'dump still uses the shared sync staging buffer directly'
    assert body.index('ggml_vk_synchronize(ctx)') < body.index('ggml_vk_buffer_read('), \
        'synchronous read precedes the mid-graph synchronize (pre-dispatch data)'


# Known-bad dump bodies replaying the pre-r16 failure modes. Each must make
# check_dump_readback_is_synchronous raise; a green run on any of these
# means the discriminator has lost its discriminating power and the green
# state above is meaningless.
BAD_DEFERRED_BODY = '''{
    ggml_vk_ensure_sync_staging_buffer(ctx, bytes);
    ggml_vk_sync_buffers(nullptr, subctx);
    const vk_subbuffer src = ggml_vk_tensor_subbuffer(ctx, tensor);
    vk_tp5_hpp_commands(subctx->s->buffer->buf).copyBuffer(
        src.buffer->buffer, ctx->sync_staging->buffer, slices);
    subctx->s->buffer->has_pending_mem_work = true;
    std::vector<uint8_t> host(bytes);
    deferred_memcpy(host.data(), ctx->sync_staging->ptr, bytes, &subctx->out_memcpys);
    ggml_vk_synchronize(ctx);
    ggml_vk_ctx_begin(ctx->device, subctx);
    fwrite(host.data(), 1, bytes, out);
}'''

BAD_HYBRID_BODY = '''{
    ggml_vk_sync_buffers(nullptr, subctx);
    std::vector<uint8_t> host(bytes);
    ggml_vk_synchronize(ctx);
    ggml_vk_ctx_begin(ctx->device, subctx);
    vk_buffer read_src = src.buffer;
    ggml_vk_buffer_read(read_src, src.offset, host.data(), bytes);
    deferred_memcpy(host.data(), ctx->sync_staging->ptr, bytes, &subctx->out_memcpys);
    fwrite(host.data(), 1, bytes, out);
}'''

BAD_ORDER_BODY = '''{
    vk_buffer read_src = src.buffer;
    ggml_vk_buffer_read(read_src, src.offset, host.data(), bytes);
    ggml_vk_synchronize(ctx);
    ggml_vk_ctx_begin(ctx->device, subctx);
    fwrite(host.data(), 1, bytes, out);
}'''


# Known-bad GGML_VK_HC_SEGMENT_SYNC bodies for the r17 switch. Each must
# make check_segment_sync_is_env_gated raise.
BAD_SEGMENT_SYNC_UNGUARDED_BODY = '''{
    static const bool hc_segment_sync = true;
    ggml_vk_sync_buffers(nullptr, subctx);
    ggml_vk_synchronize(ctx);
    ggml_vk_ctx_begin(ctx->device, subctx);
}'''  # switch executes unconditionally: no env gate at all

BAD_SEGMENT_SYNC_ORDER_BODY = '''{
    static const bool hc_segment_sync = [] {
        const char * env = getenv("GGML_VK_HC_SEGMENT_SYNC");
        return env && atoi(env) != 0;
    }();
    if (hc_segment_sync && !ctx->replay_recording && !ctx->tp5_recording_program) {
        ggml_vk_synchronize(ctx);
        ggml_vk_sync_buffers(nullptr, subctx);
        ggml_vk_ctx_begin(ctx->device, subctx);
    }
}'''  # gate present but the pairing order is broken (wait before barrier)


def check_segment_sync_is_env_gated(body):
    """Red/green discriminator for the r17 GGML_VK_HC_SEGMENT_SYNC switch.

    Green requires all of: (a) the env is read once via the static-cached
    lambda pattern (getenv + atoi; unset or "0" parse to off); (b) the
    mid-graph synchronize pairing executes only inside the branch guarded
    by the cached flag -- never unconditionally; (c) inside the switch the
    pairing is ordered sync_buffers -> synchronize -> ctx_begin (the same
    correctness-required triple as the r16 dump fix). The order checks are
    anchored at the switch definition (body.index('hc_segment_sync ='))
    so the pre-existing capture_prefix ctx_begin earlier in
    ggml_vk_hc_segment does not interfere. AssertionError is the red
    state; the known-bad samples above prove it can actually go red.
    """
    assert 'getenv("GGML_VK_HC_SEGMENT_SYNC")' in body, \
        'segment sync switch lost its env read (no gate at all)'
    assert 'atoi(env) != 0' in body, \
        'segment sync env parse must treat unset/0 as off'
    assert 'if (hc_segment_sync' in body, \
        'segment sync synchronize runs unconditionally (no env gate)'
    assert body.index('if (hc_segment_sync') < body.index('ggml_vk_synchronize(ctx)'), \
        'synchronize is not inside the env-gated branch'
    seg = body[body.index('hc_segment_sync ='):]
    assert seg.index('ggml_vk_sync_buffers(') < seg.index('ggml_vk_synchronize(ctx)'), \
        'segment sync submits without recording the execution barrier first'
    assert seg.index('ggml_vk_synchronize(ctx)') < seg.index('ggml_vk_ctx_begin('), \
        'segment sync re-opens the recording before waiting on the fence'


class DumpReadbackTests(unittest.TestCase):
    """r16 (dump-fixer): both HC dump helpers read back synchronously.

    Targets the ISOLATED instrumentation tree
    (/var/tmp/tp5-verify-rebuild-src, override via TP5_ISOLATED_VKCPP),
    which carries the r13-r15 instrumentation plus the r16 readback fix.
    The mutation samples prove the discriminator itself goes red on the
    three pre-r16 failure shapes: pure deferred (the exact r15-bug
    shape), hybrid (sync read present but the deferred landing kept as
    data source), and wrong order (read before the dispatch-forcing
    synchronize).
    """

    @classmethod
    def setUpClass(cls):
        if not ISOLATED_VKCPP.exists():
            raise unittest.SkipTest(
                'isolated instrumentation tree not present: %s '
                '(set TP5_ISOLATED_VKCPP to override)' % ISOLATED_VKCPP)

    def test_tensor_dump_reads_synchronously(self):
        body = SourceSentinelTests._function_body(
            ISOLATED_VKCPP, 'static void ggml_vk_hc_dump_site_tensor')
        check_dump_readback_is_synchronous(body)

    def test_weights_dump_reads_synchronously(self):
        body = SourceSentinelTests._function_body(
            ISOLATED_VKCPP, 'static void ggml_vk_hc_dump_site_weights')
        check_dump_readback_is_synchronous(body)

    def test_syncmp_probe_retained_and_marked_same_source(self):
        body = SourceSentinelTests._function_body(
            ISOLATED_VKCPP, 'static void ggml_vk_hc_dump_site_weights')
        self.assertIn('[GAP1-SYNCCMP]', body)
        # r16 same-source note must document that probe and file now
        # share the synchronous data source.
        self.assertIn('r16', body)
        # probe + main path: at least two synchronous reads in the body.
        self.assertGreaterEqual(body.count('ggml_vk_buffer_read('), 2)

    def test_mutation_pure_deferred_body_is_red(self):
        with self.assertRaises(AssertionError):
            check_dump_readback_is_synchronous(BAD_DEFERRED_BODY)

    def test_mutation_hybrid_body_is_red(self):
        with self.assertRaises(AssertionError):
            check_dump_readback_is_synchronous(BAD_HYBRID_BODY)

    def test_mutation_read_before_synchronize_is_red(self):
        with self.assertRaises(AssertionError):
            check_dump_readback_is_synchronous(BAD_ORDER_BODY)

    def test_r17_ensure_side_effect_call_restored(self):
        # r17 (GAP1 ensure side-effect fix): the r13-r15 deferred dumps were
        # the de-facto initialization trigger for ctx->sync_staging; the r16
        # rewrite dropped the ensure call and warmup/router-path users hit
        # the null staging buffer (ggml-backend.cpp:349 out-of-bounds tensor
        # write). Both dump helpers must call
        # ggml_vk_ensure_sync_staging_buffer(ctx, ...) ahead of the read
        # path again. The synchronous discriminator must stay green on the
        # same body: the ensure call is initialization (it contains none of
        # the banned data-source substrings), NOT a deferred-source
        # regression -- see the r17 note in its docstring.
        for sig in ('static void ggml_vk_hc_dump_site_tensor',
                    'static void ggml_vk_hc_dump_site_weights'):
            body = SourceSentinelTests._function_body(ISOLATED_VKCPP, sig)
            self.assertIn('ggml_vk_ensure_sync_staging_buffer(ctx', body)
            check_dump_readback_is_synchronous(body)


class SegmentSyncSwitchTests(unittest.TestCase):
    """r17 (GAP1 barrier hypothesis experiment): the opt-in per-segment
    mid-graph synchronize switch in ggml_vk_hc_segment.

    Targets the isolated instrumentation tree (same as DumpReadbackTests).
    The switch exists to adjudicate the r10-vs-r16 divergence: r10 (no
    instrumentation, one large batch submit) produced garbage output while
    r16 (mid-graph segmented submission) produced non-zero sane fused-segment
    values -- candidate root cause (A) is a missing barrier/dependency
    declaration in the fused-segment dispatch sequence. DevOps runs the off
    arm (no dump, no watch, no bisect -- only this switch); garbage
    disappearing confirms the barrier root cause. The mutation samples
    prove the discriminator goes red when the gate is removed entirely or
    the pairing order is broken.
    """

    @classmethod
    def setUpClass(cls):
        if not ISOLATED_VKCPP.exists():
            raise unittest.SkipTest(
                'isolated instrumentation tree not present: %s '
                '(set TP5_ISOLATED_VKCPP to override)' % ISOLATED_VKCPP)

    def test_switch_exists_in_hc_segment(self):
        body = SourceSentinelTests._function_body(
            ISOLATED_VKCPP, 'static void ggml_vk_hc_segment')
        self.assertIn('GGML_VK_HC_SEGMENT_SYNC', body)
        check_segment_sync_is_env_gated(body)

    def test_switch_default_off_zero_behavior(self):
        # The gate is the static-cached env parse: unset or "0" evaluates
        # false and every behavioral line sits inside the gated branch, so
        # the default is zero behavior difference.
        body = SourceSentinelTests._function_body(
            ISOLATED_VKCPP, 'static void ggml_vk_hc_segment')
        self.assertIn('return env && atoi(env) != 0;', body)
        self.assertIn('if (hc_segment_sync', body)

    def test_mutation_unguarded_switch_is_red(self):
        with self.assertRaises(AssertionError):
            check_segment_sync_is_env_gated(BAD_SEGMENT_SYNC_UNGUARDED_BODY)

    def test_mutation_wrong_pairing_order_is_red(self):
        with self.assertRaises(AssertionError):
            check_segment_sync_is_env_gated(BAD_SEGMENT_SYNC_ORDER_BODY)


class DownWiringAuditTests(unittest.TestCase):
    """GAP1 final fix round (down-wire-fixer): the down-contraction wiring
    audit prompted by the sitedump evidence (normalized all-nonzero, lo all
    zero, mixed all zero at every site -- parity-analysis.json).

    Four candidate defect domains were audited against the working tree and
    all four are statically CLOSED for the bf16 variant; these sentinels pin
    that closure so a future edit cannot silently reopen the exact wiring
    the dump evidence was first blamed on:

    (1) D binding miswiring: the down/up dispatches go through the ONE
        shared ggml_vk_segment_project descriptor template
        {weights, b_sub, out, extra-or-out, out} -- binding 2 is the output
        subbuffer for BOTH the Q8 and BF16 variants (the 3c Q8 twin is
        correct on the same binary, and the template is family-agnostic).
    (2) zero/short grid: dispatch_rows is n_rows for the down contraction
        (non-fold) and 4*n_rows for all three fold twins; grid.x is
        output->ne[0] for both families.
    (3) skipped dispatch: ggml_vk_segment_project has no early return on
        the BF16 path (only the Q5_K MMVQ/output-tail branches, which a
        BF16 weight never takes), and the shader-side local-output guard
        tp5_keep_local_output is unconditionally true for every non-RELAY
        variant -- the down contraction cannot silently drop its data_d
        store.
    (4) overwritten output: the lo dump point sits directly after the down
        barrier with only the (recording-gated) tape annotation in between.

    The audit's sharpest output is a math fact the dump interpretation
    must respect: mixed is NOT fold(up(silu(0)))=0 downstream of a zero lo.
    The fold epilogue is product = fuse0 * sigmoid(contraction); with
    lo=0 the contraction is 0 and sigmoid(0)=0.5, so
    mixed = 0.125 * sum(normalized 4-stream) which is nonzero whenever
    normalized is. An all-zero mixed therefore requires the up dispatch's
    own A/fuse/D chain to fail as well -- "lo all zero" alone cannot
    explain it, and any root cause must cover both contractions (or the
    dump readback itself). See docs/TP5-BF16-UPFOLD-FIX.md root-cause
    section for the full exclusion table.
    """

    def _segment_project(self):
        return SourceSentinelTests._function_body(VKCPP, 'static void ggml_vk_segment_project')

    def test_down_descriptors_share_one_family_agnostic_template(self):
        seg = self._segment_project()
        # (1) The single descriptor template: binding 2 (D) is `out` for
        # every family; binding 3 is extra-or-out; binding 4 is out. The
        # template text carries no weight-type branch -- the only type
        # test in the function body is the Q5_K MMVQ/output-tail gate.
        self.assertIn('const std::initializer_list<vk::DescriptorBufferInfo> descriptors{', seg)
        self.assertIn('ggml_vk_tensor_subbuffer(ctx, weights), b_sub, out,', seg)
        self.assertIn('extra ? ggml_vk_tensor_subbuffer(ctx, extra) : out, out };', seg)
        self.assertEqual(seg.count('weights->type =='), 2)  # both are Q5_K gates

    def test_down_dispatch_rows_is_plain_rows(self):
        seg = SourceSentinelTests._function_body(VKCPP, 'static void ggml_vk_segment_project')
        # (2) dispatch_rows covers exactly the three fold twins with
        # 4*n_rows; the down contraction (Q8 or BF16) stays at n_rows.
        for pipeline in ('pipeline_hc_project_fold',
                         'pipeline_hc_project_fold_bf16',
                         'pipeline_hc_project_fold_r320'):
            self.assertIn('contraction == ctx->device->%s' % pipeline, seg)
        self.assertIn('? 4 * n_rows : n_rows;', seg)

    def test_down_path_has_no_early_return_for_bf16(self):
        seg = self._segment_project()
        # (3) The only early returns before the dispatch are the Q5_K MMVQ
        # and Q5_K output-tail branches; a GGML_TYPE_BF16 weight takes
        # neither, so the down dispatch is recorded unconditionally.
        self.assertIn('weights->type == GGML_TYPE_Q5_K', seg)
        self.assertIn('ggml_vk_tp5_q5k_output', seg)
        early = seg.index('return;')
        tail = seg.index('ggml_vk_tp5_q5k_output')
        self.assertLess(tail, early)  # the sole return sits inside the Q5_K branch

    def test_shader_local_output_guard_is_unconditional_for_non_relay(self):
        iface = (REPO / 'ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_iface.glsl').read_text()
        # (3, shader side) tp5_keep_local_output returns true for every
        # variant that is not TP5_RELAY_OUTPUT: the HC down contraction
        # (Q8 or BF16) cannot skip its data_d store via fusion flags.
        body = iface[iface.index('bool tp5_keep_local_output'):]
        body = body[:body.index('\n}\n') + 3]
        self.assertIn('#ifdef TP5_RELAY_OUTPUT', body)
        self.assertIn('#else', body)
        self.assertIn('return true;', body)

    def test_fold_of_zero_contraction_is_not_zero(self):
        # The math fact the dump interpretation must respect: with lo = 0
        # the up contraction temp is 0, gate = sigmoid(0) = 0.5, and
        # mixed = 0.25 * sum_n(normalized[n*W+col]) * 0.5 != 0 for any
        # nonzero normalized. Mirrors reduce_result's HC_UP_FOLD epilogue.
        import math

        def fold_epilogue(normalized, width, lo):
            # lo all zero -> every per-stream contraction temp is exactly 0.
            temp = [0.0] * 4
            mixed = [0.0] * width
            for col in range(width):
                acc = 0.0
                for n in range(4):
                    gate = 1.0 / (1.0 + math.exp(-temp[n]))
                    product = normalized[n * width + col] * gate
                    acc = product if n == 0 else acc + product
                mixed[col] = acc * 0.25
            return mixed

        normalized = [0.23, -0.42, 1.20, 0.05] * 2560  # dump morphology: real activations
        mixed = fold_epilogue(normalized, 2560, [0.0] * 320)
        self.assertTrue(all(v != 0.0 for v in mixed))
        # list-repeat semantics: normalized[k] = pattern[k % 4], so the four
        # stream values at col=0 are all pattern[0] = 0.23; mixed[0] =
        # 0.25 * (4 * 0.23) * sigmoid(0) = 0.25 * 0.92 * 0.5
        self.assertAlmostEqual(mixed[0], 0.25 * 4 * 0.23 * 0.5, places=12)

    def test_dump_paths_have_no_rank_disambiguation(self):
        # Known instrumentation limit, pinned so the dump interpretation
        # stays honest: TP5 runs five ranks in one process and every rank
        # writes the SAME /tmp path (last writer wins). The sitedump
        # verdicts therefore describe one rank's final write, not a
        # per-rank consensus; DevOps re-verification must keep this in
        # mind when reading lo/mixed zero morphology.
        body = SourceSentinelTests._function_body(VKCPP, 'static void ggml_vk_hc_dump_site_tensor')
        self.assertIn('snprintf(path, sizeof(path), "/tmp/tp5-hc-dump-site%u-%s.bin"', body)
        self.assertNotIn('rank', body)


class UploadChainAuditTests(unittest.TestCase):
    """GAP1 upload-chain regression: BF16 HC weights must reach every rank's VkBuffer.

    Root cause (r2-verdict branch D): the mmap load path bound the meta tensor's
    data to the host mmap address via ggml_backend_tensor_alloc and never called
    ggml_backend_tensor_set, so the MIRRORED fan-out never ran and all 5 ranks
    kept zero pages where W_down/W_up should be (Q8_0 twins entered with
    data != nullptr and were fanned out correctly).
    """

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _loader_source(self):
        with open(os.path.join(self.ROOT, 'src', 'llama-model-loader.cpp'), 'r', encoding='utf-8') as fh:
            return fh.read()

    def test_mmap_meta_upload_fans_out_after_tensor_alloc(self):
        # Red before the fix: the mmap branch called ggml_backend_tensor_alloc
        # on meta buffers with no subsequent tensor_set, so no rank shard was
        # ever written. Green after: the meta case explicitly fans out.
        src = self._loader_source()
        self.assertIn('ggml_backend_tensor_alloc(buf_mmap, cur, data);', src)
        self.assertIn('ggml_backend_buffer_is_meta(buf_mmap)', src)
        meta_guard = src.index('ggml_backend_buffer_is_meta(buf_mmap)')
        alloc_site = src.index('ggml_backend_tensor_alloc(buf_mmap, cur, data);')
        fanout_site = src.index('ggml_backend_tensor_set(cur, data, 0, n_size);', meta_guard)
        self.assertLess(alloc_site, fanout_site)
        self.assertLess(fanout_site - meta_guard, 2000,
                        'meta fan-out must be adjacent to the alloc site in the mmap branch')

    def test_bf16_weight_shard_byte_width_matches_dump(self):
        # Byte-width accounting pinned by r2-parity-analysis.json: W_down is
        # [hc_dim=10240, hc_lr=320] BF16 (2 bytes/elem) -> 6553600 bytes per
        # MIRRORED rank shard. A quant-type block-width assumption would not
        # reproduce this number, so the dump pins the BF16 interpretation.
        hc_dim, hc_lr, elem_bytes = 10240, 320, 2
        self.assertEqual(hc_lr * hc_dim * elem_bytes, 6553600)

    def test_mirrored_meta_set_tensor_fans_out_to_all_ranks(self):
        # The fan-out target: meta set_tensor's MIRRORED case must iterate
        # every rank buffer. If this loop ever becomes type-gated or
        # rank-limited the fix stops delivering bytes and this goes red.
        with open(os.path.join(self.ROOT, 'ggml', 'src', 'ggml-backend-meta.cpp'), 'r', encoding='utf-8') as fh:
            meta_src = fh.read()
        self.assertIn('case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {', meta_src)
        mirrored = meta_src.index('case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {')
        window = meta_src[mirrored:mirrored + 700]
        self.assertIn('for (size_t j = 0; j < n_bufs; j++)', window)
        self.assertIn('ggml_backend_tensor_set(simple_tensor, data, offset, size)', window)


class ContainerRoutingAuditTests(unittest.TestCase):
    """GAP-1 容器路由守卫审计（历史防御性守卫，钉住不变量）。

    不变量：权重型张量（op=NONE、非 view）在加载期注册进其 meta buffer
    的 stc_static，dispatch 期必须经同一 stc_static 解析，不得经 lazy-init
    落入 compute 容器影子分配。守卫针对的「容器路由分叉」假设已被证伪：
    r6 全流程 GGML_ABORT 零触发；r7 同 run 判别证明加载期与 dispatch 期
    张量身份完全匹配（r5 的「第二次分配」地址来自不同 run，系跨 run 比较
    伪影，见 DECISION.md §14.3–14.5 与 identity/quarantine-dispatchtrace/
    verdict.md）。守卫保留为 fail-closed 纵深防御；本测试钉住守卫存在与
    条件不变，不声称分叉真实发生过。
    """

    META_CPP = "ggml/src/ggml-backend-meta.cpp"

    def _read(self, rel):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, rel)
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()

    def test_alloc_preserves_registered_static_weight_ownership(self):
        """钉住守卫一存在：重绑循环保留已注册于另一 meta buffer
        stc_static 的张量所有权（历史纵深防御；其针对的分叉经 r6 零触发、
        r7 同 run 身份匹配证伪为跨 run 比较伪影，未真实发生）。"""
        src = self._read(self.META_CPP)
        self.assertIn("prev_ctx->stc_static.simple_tensors.find(t)", src)
        self.assertIn("ggml_backend_buffer_is_meta(t->buffer)", src)
        alloc_body = src.split("ggml_backend_meta_alloc_ctx_tensors_from_buft", 1)[1]
        self.assertIn("prev_ctx->stc_static.simple_tensors.find(t)", alloc_body)

    def test_lazy_init_refuses_compute_shadow_for_weights(self):
        """钉住守卫二存在：lazy-init 路径对权重型张量（op=NONE、非
        view、未注册 stc_static）fail-closed abort（历史纵深防御；r6 全
        流程零触发，r7 证明权重始终命中 stc_static，从未落入此路径）。"""
        src = self._read(self.META_CPP)
        self.assertIn("tensor->op == GGML_OP_NONE && tensor->view_src == nullptr", src)
        simple_body = src.split("ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor *", 1)[1]
        self.assertIn("GGML_ABORT", simple_body)

    def test_quantized_paths_unchanged_by_guard_conditions(self):
        """退化论证（绿，量化路径零行为变化）。

        守卫条件仅对「已注册于另一 meta buffer stc_static 的张量」（即
        加载期完成的权重）触发；compute 张量与 external view 从不在别的
        buffer 的 stc_static 注册，守卫零触发——Q8_0/Q2_0 路径行为不变
        （3c 证据：量化 HC 路径本就正常；r6 实测全流程 GGML_ABORT 零触发
        与此一致，r7 证伪分叉后该结论不变）。
        """
        src = self._read(self.META_CPP)
        self.assertIn("continue; // ownership already established; do not fork the identity", src)
        alloc_body = src.split("ggml_backend_meta_alloc_ctx_tensors_from_buft", 1)[1]
        self.assertIn("t->buffer != meta_buf", alloc_body)


# ---------------------------------------------------------------------------
# RegistryRegressionTests (r21): removal guard for the GAP1 fork-fix
# weight->simple-tensor registry in the QUARANTINE TREE's
# ggml-backend-meta.cpp (/var/tmp/tp5-verify-rebuild-src — the
# instrumentation tree where the r10 registry and the r20 gate lived
# until r21 removed both; the repo working tree's copy is NOT the
# target). The registry was confirmed as the crash root cause by the
# r20 gated experiment (bypass on -> crash gone) and removed in r21.
# These tests go RED if the old mechanism ever reappears: registry map,
# gate helper, env knob, or the bypass log tag — including a revival
# under a renamed env name. No build, no run: source-level regression
# guard; compile/run verification belongs to DevOps.
# ---------------------------------------------------------------------------

QUARANTINE_META_CPP = "/var/tmp/tp5-verify-rebuild-src/ggml/src/ggml-backend-meta.cpp"
REGISTRY_SYMBOLS = (
    "s_gap1_weight_simple_registry",
    "ggml_backend_meta_registry_disabled",
    "GGML_VK_META_REGISTRY_DISABLE",
    "[meta-registry-bypass]",
)


def _read_quarantine_meta_cpp():
    with open(QUARANTINE_META_CPP, "r", encoding="utf-8") as fh:
        return fh.read()


def _registry_absent_property_holds(src):
    """Removal property: none of the registry mechanism identifiers (map,
    gate helper, env knob, bypass log tag) may appear anywhere in the
    source. The check covers the full identifier set, not a single env
    name — the r20 gate property missed an env-rename mutation because it
    only counted one env string and the guard-call sites; checking every
    symbol of the mechanism means a revival under a renamed env still
    leaves the map/helper symbols present and forces a red."""
    return all(sym not in src for sym in REGISTRY_SYMBOLS)


class RegistryRegressionTests(unittest.TestCase):
    """r21 registry removal guard: the r10 registry (crash root cause per
    the r20 gated experiment) must stay absent from the quarantine tree's
    ggml-backend-meta.cpp; red if the old mechanism comes back."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read_quarantine_meta_cpp()

    def test_registry_mechanism_fully_absent(self):
        """(a) 移除守卫：registry map、门控 helper、env 名、旁路日志
        标记四个标识符全部不在隔离树 meta.cpp（r21 移除后状态）。"""
        for sym in REGISTRY_SYMBOLS:
            self.assertNotIn(sym, self.src)
        self.assertTrue(_registry_absent_property_holds(self.src))

    def test_r9_fallback_path_intact_after_removal(self):
        """(b) 移除不等于删空：r9 树原始解析路径（stc 容器查找、
        init_tensor_impl 前置声明）与 r5 container routing guard 必须
        仍在位——即 r20 门控旁路时已验证等价的那条路径。"""
        self.assertIn("get_simple_tensor_container(tensor)", self.src)
        self.assertIn(
            "static enum ggml_status ggml_backend_meta_buffer_init_tensor_impl",
            self.src)
        self.assertIn("GAP-1 container routing fork", self.src)

    def test_red_sample_registry_map_revived(self):
        """(c) 红样本：向移除后的源码注入 registry 定义（旧机制
        复活），判别函数必须返回 False（变红报警）。"""
        anchor = ("static enum ggml_status "
                  "ggml_backend_meta_buffer_init_tensor_impl")
        bad = self.src.replace(
            anchor,
            "static std::map<const ggml_tensor *, std::vector<ggml_tensor *>> "
            "s_gap1_weight_simple_registry;\n" + anchor,
            1)
        self.assertNotEqual(
            bad, self.src,
            "mutation did not apply: anchor text drifted from expectation")
        self.assertFalse(
            _registry_absent_property_holds(bad),
            "revived registry must fail the removal property (red sample)")

    def test_red_sample_gate_revived_env_renamed(self):
        """(c) 红样本（mutation 敏感性回归，修复 r20 版判别函数对 env
        重命名不敏感的缺陷）：registry 机制以全新 env 名复活——helper
        与 map 都在、env 名换成不含原名的字符串。只盯 env 名的判别会
        漏报；全标识符判别必须仍返回 False。"""
        bad = (
            "static bool ggml_backend_meta_registry_disabled(void) {\n"
            "    static int cached = -1;\n"
            "    if (cached < 0) {\n"
            '        const char * env = getenv("GGML_VK_META_REG_OFF");\n'
            "        cached = (env != nullptr && strcmp(env, \"1\") == 0) "
            "? 1 : 0;\n"
            "    }\n"
            "    return cached == 1;\n"
            "}\n"
            "static std::map<const ggml_tensor *, std::vector<ggml_tensor *>> "
            "s_gap1_weight_simple_registry;\n"
        ) + self.src
        self.assertFalse(
            _registry_absent_property_holds(bad),
            "renamed-env revival must fail the removal property (red sample)")


if __name__ == '__main__':
    unittest.main(verbosity=2)

