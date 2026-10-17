#!/usr/bin/env python3
"""Necessity certificate v5 for the fixed TP5 single-user generation task.

v4 supersedes v3 and fixes three soundness defects found by audit:

  DEFECT-1 (factual): DENSE_BYTES_PER_LAYER was 90.8 MB; the audited GGUF
  inventory (necessity-restart-20261009/audit.json, 1222 tensors) gives
  63.82 MB average dense bytes per layer (36 qkv layers 63.79 MB, 12 q/k
  layers 61.65 MB). v3 overstated the dense term by 42%.

  DEFECT-2 (soundness): v3 computed the L2 retention credit as
  L2/pool where pool = 11.9 GB is the FULL per-GPU weight pool, then
  applied that 1.07% hit fraction to the ACTIVE per-token set. For a
  valid lower bound the implementation must receive MAXIMUM caching
  credit: the L2 can hold up to 128 MB of whatever per-token active
  bytes it chooses (dense or expert). The correct generous credit is
  l2_hit = min(1, L2 / active_per_gpu_per_token). v3's smaller credit
  made the bound tighter than provable -> invalid as a lower bound.

  DEFECT-3 (soundness, the largest): v3 stated "bound is per target
  row, so per-public-token bound equals it" under the perfect-acceptance
  MTP relaxation. That is false for R3 (and any weight-traffic term):
  with MTP a cycle verifies h+1 rows in ONE batched pass, and the active
  weight bytes are read ONCE per cycle, not once per row. Under the
  allowed set (any draft policy, perfect acceptance, unlimited h) the
  per-public-token DRAM weight traffic amortizes toward zero. Only terms
  that scale with rows (ALU FLOPs) or with public tokens (output
  emission, causal cut per row) survive.

v4 therefore reports a SPLIT certificate instead of one number:

  L_global  = max(R1, R2, R4)  -- valid for the FULL allowed set A
              (R3 excluded because it does not survive MTP amortization).
              Binding term: R4 (FP32 ALU), ~6.7e3 tok/s ceiling. Loose
              but sound.

  L_nomtp   = max(R1, R2, R3, R4) -- valid only for the declared
              conditional class A_nomtp (implementations that run one
              target row per public token, i.e. no speculative
              batching). This is the honest bound for ranking no-MTP
              arms. ~7.3e2 tok/s ceiling with the corrected dense
              bytes and L2 credit.

Both bounds use spec capacity ceilings (512 GB/s DRAM, 80.85 TFLOP/s
aggregate FP32, 32 GB/s PCIe, 25 GB/s host readback), maximal overlap
across distinct resources, and only information-forced byte/FLOP counts.
No measured latency of any implementation enters the bound.

The certificate deliberately does NOT claim U/L convergence: the
demonstrated U (see necessity-status.json) is two orders of magnitude
above L_nomtp. Closing that gap requires either tighter provable
latency coupling (not available under the fully relaxed allowed set)
or a faster demonstrated implementation. v4 records both bounds and
the explicit non-convergence status instead of a single flattering
number.

Usage:
  python3 tp5-necessity-certificate.py --json out.json
  python3 tp5-necessity-certificate.py --json out.json --u-seconds 0.0148 --u-source "run-YYYYMMDD arm" --u-qualification "conservative statistic of a qualified arm"

v5 adds the AGENTS.md §15 algorithm-P branch-family ledger on top of v4:

  P1  structured enumeration of legal branches: layout family
      {mirrored, owner, coordinate, layer, hybrid} x communication family
      {host-relay, peer-copy, device-side} x representation family
      {original, qualified-quant} x MTP depth {0,1,2,3} -> 120 records.
  P2  each branch must explicitly declare four premise slots (mandatory
      weight/state information, physical cut-set, fast-storage capacity
      credit, overlap declaration); any undeclared slot makes the branch
      UNRESOLVED automatically.
  P3/P4  Lj is emitted only when every premise slot is declared AND every
      bound claim carries verifiable certificate material: a resource
      counting lemma (proved-work basis / capacity-upper basis) with its
      counting-argument justification, or an LP dual-feasible solution with
      the relaxation-mapping argument. Solver incumbents are refused.
  P5  global L = min_j Lj over ALL branches; any UNRESOLVED or REJECTED
      branch keeps global_L UNRESOLVED (never the min of a proven subset).
  P6/P7  Gap = U/L - 1 is computed only when global_L is PROVEN and a
      qualified U (value + source + qualification statement) is supplied;
      otherwise the gap is reported UNPROVEN and no ratio is emitted.

Current honest state: no branch has its four premise slots declared, so all
120 branches are UNRESOLVED, global_L is UNRESOLVED and the gap is UNPROVEN.
The v4 L_global/L_nomtp bounds below are retained as declared conditional
arguments over the whole allowed set; they are NOT per-branch premise
declarations under algorithm P and are deliberately not promoted into the
ledger (see branch_ledger.relation_to_v4_bounds).
"""
import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

# Fixed model contract (Qwen4EXP-AF, verified from GGUF metadata).
N_LAYERS = 48
N_EXPERTS = 512
HIDDEN = 2560
N_RANKS = 5
VOCAB = 248320

# Verified weight geometry (audited against the six-shard GGUF metadata
# scan in artifacts/perf-gap/necessity-restart-20261009/audit.json):
# expert weights (gate/up/down exps) per layer = 1153.4336 MB on disk
# (blk.N.ffn_gate_exps 321.13 + ffn_up_exps 360.45 + ffn_down_exps 471.86).
EXPERT_BYTES_PER_LAYER = 1153.4336e6
# Dense (always-active) weights per layer, 48-layer average from the same
# audit: attn_qkv 18.02 (36 layers) / attn_q+attn_k+attn_v 21.63 (12
# layers), attn_gate 10.81, ssm_out 10.81, ple_key 21.50 (global, counted
# once per layer share), ple_value 5.40, router ffn_gate_inp 5.24, HC
# down/up x4 + inject + norm 14.10, shared experts 4.44, ssm/ple misc.
DENSE_BYTES_PER_LAYER = 63.820906666666666e6
L2_BYTES_PER_GPU = 128e6
FP32_TFLOPS_PER_GPU = 16.17

# Hardware interface ceilings (RX 6800, conservative spec upper bounds).
PCIE_GEN4_X16_GB_S = 32.0
HOST_READBACK_GB_S = 25.0
DRAM_GB_S = 512.0

# R1: sampler-visible data per committed public token.
R1_BYTES_STRONG = N_EXPERTS * 4          # full scores host-visible
R1_BYTES_WEAK = 4                        # token id only

# R2: per-boundary cross-rank payload (minimum shard of the router input
# that the consumer rank cannot produce itself).
R2_BYTES_PER_BOUNDARY = (HIDDEN // N_RANKS) * 4   # 2048 bytes
R2_BOUNDARIES = N_LAYERS                            # 48 sequential layers


# ============================================================================
# AGENTS.md §15 algorithm P: branch-family ledger (P1-P7)
# ============================================================================

# Shared, already-audited proof primitives. The audited label checks are NOT
# automatic theorem proving: they refuse known-bad bases (work measured on a
# current implementation, capacity taken from an observed throughput, empty
# premises) but never verify that a declared premise is true.
_AUDIT_PATH = Path(__file__).resolve().parent / 'tp5-necessity-audit.py'


def _load_audit_module():
    spec = importlib.util.spec_from_file_location('tp5_necessity_audit_shared', _AUDIT_PATH)
    if not spec or not spec.loader:
        raise ValueError(f'Unable to load audited necessity module: {_AUDIT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_audit = _load_audit_module()
NecessaryResource = _audit.NecessaryResource
joint_lower = _audit.joint_lower

# P1: structured enumeration of the legal implementation-branch families.
LAYOUT_FAMILIES = ('mirrored', 'owner', 'coordinate', 'layer', 'hybrid')
COMMUNICATION_FAMILIES = ('host-relay', 'peer-copy', 'device-side')
REPRESENTATION_FAMILIES = ('original', 'qualified-quant')
MTP_DEPTHS = (0, 1, 2, 3)

# P2: the four premise slots every branch must declare explicitly.
PREMISE_SLOTS = (
    'required_weight_state_information',
    'physical_cut_set',
    'storage_capacity_assumption',
    'overlap_declaration',
)

PREMISE_SLOT_DESCRIPTIONS = {
    'required_weight_state_information': (
        'Which weight/state bytes are provably mandatory for EVERY legal '
        'implementation of this branch (the W_min basis), and how the count '
        'excludes current-layout amplification.'),
    'physical_cut_set': (
        'Which information provably crosses which physical boundary in this '
        'branch (the cut-set basis), including why no smaller sufficient '
        'statistic or local recomputation avoids the crossing.'),
    'storage_capacity_assumption': (
        'The fast-storage capacity credit C for this branch: which levels '
        '(registers/LDS/L2/Infinity Cache/prefetch buffers) count and the '
        'assumed bound; stated conservatively for the implementer.'),
    'overlap_declaration': (
        'Which mandatory resources are declared overlapping (max rule) and '
        'which non-overlap couplings are actually proven; summing bounds is '
        'forbidden without this.'),
}

_WORK_BASIS_PROVED = 'proved_for_declared_class'
_CAPACITY_BASIS_UPPER = 'upper_bound_for_declared_class'


@dataclass(frozen=True)
class ResourceLemmaClaim:
    """One resource-lemma bound claim: T >= required_work / capacity_upper.

    The bases go through the audited label checks: work measured on a
    current implementation and capacity taken from an observed throughput
    are both refused.
    """
    required_work: float
    capacity_upper: float
    work_basis: str = _WORK_BASIS_PROVED
    capacity_basis: str = _CAPACITY_BASIS_UPPER
    justification: str = ''


@dataclass(frozen=True)
class LpRelaxationClaim:
    """One relaxation bound claim for  min c.x  s.t.  A x >= b, x >= 0.

    kind='dual': 'vector' is a dual-feasible y >= 0 with A^T y <= c. Weak
    duality makes the verified dual objective b.y a checkable lower-bound
    certificate. kind='incumbent': 'vector' is only a primal-feasible x; an
    incumbent proves nothing about the minimum and is always refused.
    'relaxation_description' must argue why every legal implementation of
    the branch maps to a feasible point of the relaxation (algorithm P3);
    without that mapping the number is not a bound for this branch.
    """
    kind: str
    A: tuple
    b: tuple
    c: tuple
    vector: tuple
    relaxation_description: str = ''


@dataclass(frozen=True)
class BranchRecord:
    """P1: one legal branch of the allowed implementation set."""
    branch_id: str
    layout: str
    communication: str
    representation: str
    mtp_depth: int
    premises: dict
    bound_claims: tuple


def make_branch_id(layout, communication, representation, mtp_depth):
    return f'{layout}+{communication}+{representation}+h{mtp_depth}'


def enumerate_branches():
    """P1: one record per legal branch; premises start undeclared."""
    records = []
    for layout in LAYOUT_FAMILIES:
        for communication in COMMUNICATION_FAMILIES:
            for representation in REPRESENTATION_FAMILIES:
                for depth in MTP_DEPTHS:
                    records.append(BranchRecord(
                        branch_id=make_branch_id(layout, communication, representation, depth),
                        layout=layout,
                        communication=communication,
                        representation=representation,
                        mtp_depth=depth,
                        premises={slot: None for slot in PREMISE_SLOTS},
                        bound_claims=()))
    return records


def branch_premise_class(record):
    """The single implementation class all of a branch's claims must address."""
    return f'branch:{record.branch_id}'


def _lp_dimensions_ok(A, b, c_vec, v):
    if not A or not b or not c_vec or not v:
        return False
    n = len(A[0])
    if any(len(row) != n for row in A):
        return False
    return len(b) == len(A) and len(c_vec) == n and len(v) == len(A)


def verify_lp_dual(A, b, c_vec, y, tol=1e-9):
    """Verify a dual-feasible certificate; return its dual objective value.

    LP form:  min c.x  s.t.  A x >= b, x >= 0.
    Dual:     max b.y  s.t.  A^T y <= c, y >= 0.
    Weak duality makes b.y a lower bound on the primal optimum, so a
    dual-feasible y is a checkable lower-bound certificate. Incumbent
    primal points are never accepted (see LpRelaxationClaim.kind).
    """
    if not _lp_dimensions_ok(A, b, c_vec, y):
        raise ValueError('LP certificate dimensions inconsistent')
    m = len(A)
    n = len(c_vec)
    for row in A:
        for a_ij in row:
            _audit.finite(a_ij)
    for vec in (b, c_vec, y):
        for value in vec:
            _audit.finite(value)
    for y_i in y:
        if y_i < 0:
            raise ValueError('Dual certificate must satisfy y >= 0')
    for j in range(n):
        col_sum = sum(A[i][j] * y[i] for i in range(m))
        if col_sum > c_vec[j] + tol:
            raise ValueError(f'Dual certificate violates A^T y <= c at column {j}')
    return float(sum(b[i] * y[i] for i in range(m)))


def _positive_finite(x):
    return (not isinstance(x, bool) and isinstance(x, (int, float))
            and math.isfinite(x) and x > 0)


def _verify_claim(claim, premise_class):
    """Verify one bound claim; return its lower-bound value or raise."""
    if isinstance(claim, ResourceLemmaClaim):
        if not claim.justification.strip():
            raise ValueError('Resource-lemma claim carries no counting-argument '
                             'justification; that text is the certificate material')
        resource = NecessaryResource(claim.required_work, claim.work_basis,
                                     claim.capacity_upper, claim.capacity_basis,
                                     premise_class)
        return resource.latency_lower()
    if isinstance(claim, LpRelaxationClaim):
        if not claim.relaxation_description.strip():
            raise ValueError('LP relaxation claim carries no mapping argument '
                             '(why every legal implementation of the branch maps '
                             'to a feasible point of the relaxation)')
        if claim.kind == 'incumbent':
            raise ValueError('Incumbent/primal feasible solutions are not '
                             'lower-bound certificates; only dual-feasible '
                             'solutions or equivalent proofs are accepted')
        if claim.kind != 'dual':
            raise ValueError(f'Unknown LP claim kind: {claim.kind!r}')
        return verify_lp_dual(claim.A, claim.b, claim.c, claim.vector)
    raise ValueError(f'Unknown bound claim type: {type(claim).__name__}')


def _claim_summary(claim, value):
    if isinstance(claim, ResourceLemmaClaim):
        return {
            'claim_kind': 'resource_lemma',
            'verified_lower_bound_seconds': value,
            'work_basis': claim.work_basis,
            'capacity_basis': claim.capacity_basis,
            'justification': claim.justification,
        }
    return {
        'claim_kind': 'lp_dual',
        'verified_lower_bound_seconds': value,
        'relaxation_description': claim.relaxation_description,
    }


def _branch_base(record):
    return {
        'branch_id': record.branch_id,
        'layout': record.layout,
        'communication': record.communication,
        'representation': record.representation,
        'mtp_depth': record.mtp_depth,
    }


def evaluate_branch(record):
    """P3/P4: emit Lj only with full premises and verified certificates.

    Undeclared premise slots -> UNRESOLVED (automatic, per P2). Fabricated
    or unverifiable claim material -> REJECTED. Only fully declared and
    fully verified branches get an Lj.
    """
    result = _branch_base(record)
    missing = [slot for slot in PREMISE_SLOTS
               if not str(record.premises.get(slot) or '').strip()]
    if missing:
        result.update({
            'status': 'UNRESOLVED',
            'Lj_seconds': None,
            'missing_premise_slots': missing,
            'rejection_reasons': [],
            'certificates': None,
        })
        return result
    if not record.bound_claims:
        result.update({
            'status': 'UNRESOLVED',
            'Lj_seconds': None,
            'missing_premise_slots': [],
            'rejection_reasons': ['premises declared but no bound claims or '
                                  'proof material supplied'],
            'certificates': None,
        })
        return result
    premise_class = branch_premise_class(record)
    certificates = []
    rejections = []
    resource_claims = []
    lp_values = []
    for claim in record.bound_claims:
        try:
            value = _verify_claim(claim, premise_class)
        except ValueError as exc:
            rejections.append(str(exc))
            continue
        certificates.append(_claim_summary(claim, value))
        if isinstance(claim, ResourceLemmaClaim):
            resource_claims.append(claim)
        else:
            lp_values.append(value)
    if rejections:
        # Fabricated or unverifiable material invalidates the whole branch
        # claim: the branch is rejected, never partially credited.
        result.update({
            'status': 'REJECTED',
            'Lj_seconds': None,
            'missing_premise_slots': [],
            'rejection_reasons': rejections,
            'certificates': certificates,
        })
        return result
    values = []
    if resource_claims:
        # joint_lower re-checks the same-premise-class requirement and
        # applies the max rule (docs/performance-necessity.md §4.2).
        values.append(joint_lower([
            NecessaryResource(cl.required_work, cl.work_basis,
                              cl.capacity_upper, cl.capacity_basis, premise_class)
            for cl in resource_claims]))
    values.extend(lp_values)
    result.update({
        'status': 'PROVEN',
        'Lj_seconds': max(values),
        'missing_premise_slots': [],
        'rejection_reasons': [],
        'certificates': certificates,
    })
    return result


def merge_lower_bounds(results):
    """P5: global L = min_j Lj over ALL branches, or UNRESOLVED.

    Any UNRESOLVED or REJECTED branch keeps the global lower bound
    UNRESOLVED. The minimum over a proven subset is never promoted as the
    global L: T* is the infimum over the UNION of all branches.
    """
    if not results:
        raise ValueError('No branch results to merge')
    allowed = {'PROVEN', 'UNRESOLVED', 'REJECTED'}
    for r in results:
        if r.get('status') not in allowed:
            raise ValueError(f"Unknown branch status: {r.get('status')!r}")
        if r['status'] == 'PROVEN':
            if not r.get('certificates'):
                raise ValueError(f"PROVEN branch {r['branch_id']} carries no "
                                 'certificate material')
            if not _positive_finite(r.get('Lj_seconds')):
                raise ValueError(f"PROVEN branch {r['branch_id']} has no valid Lj")
    unresolved = [r['branch_id'] for r in results if r['status'] == 'UNRESOLVED']
    rejected = [r['branch_id'] for r in results if r['status'] == 'REJECTED']
    proven = [r for r in results if r['status'] == 'PROVEN']
    if unresolved or rejected or not proven:
        return {
            'status': 'UNRESOLVED',
            'global_L_seconds': None,
            'rule': ('L = min_j Lj over ALL branches of the allowed set; any '
                     'UNRESOLVED or REJECTED branch forces global_L UNRESOLVED; '
                     'the minimum over a proven subset is never promoted as the '
                     'global L'),
            'proven_branch_count': len(proven),
            'unresolved_branches': unresolved,
            'rejected_branches': rejected,
        }
    return {
        'status': 'PROVEN',
        'global_L_seconds': min(r['Lj_seconds'] for r in proven),
        'rule': ('L = min_j Lj: T* is the infimum of T(a) over the union of '
                 'all branch classes'),
        'proven_branch_count': len(proven),
        'unresolved_branches': [],
        'rejected_branches': [],
    }


def gap_assessment(global_lower, u_seconds=None, u_source=None,
                   u_qualification=None):
    """P6/P7: U/L is computed only when L and a qualified U are both valid.

    A fabricated ratio is worse than none: with an invalid global L or an
    unqualified/missing U the gap stays UNPROVEN and no numeric ratio is
    emitted.
    """
    out = {
        'status': 'UNPROVEN',
        'U_seconds': None,
        'U_source': None,
        'U_qualification': None,
        'U_over_L': None,
        'gap_minus_1': None,
        'reason': None,
    }
    if (not isinstance(global_lower, dict)
            or global_lower.get('status') != 'PROVEN'
            or not _positive_finite(global_lower.get('global_L_seconds'))):
        out['reason'] = 'global L is not PROVEN; no U/L ratio may be computed'
        return out
    if (u_seconds is None or isinstance(u_seconds, bool)
            or not isinstance(u_seconds, (int, float))
            or not math.isfinite(u_seconds) or u_seconds <= 0
            or not str(u_source or '').strip()
            or not str(u_qualification or '').strip()):
        out['reason'] = ('U is absent, non-finite or lacks a source/qualification '
                         'statement; no U/L ratio may be computed')
        return out
    L = global_lower['global_L_seconds']
    ratio = float(u_seconds) / L
    out.update({
        'status': 'PROVEN_RATIO',
        'U_seconds': float(u_seconds),
        'U_source': str(u_source),
        'U_qualification': str(u_qualification),
        'U_over_L': ratio,
        'gap_minus_1': ratio - 1.0,
    })
    return out


def build_branch_ledger(u_seconds=None, u_source=None, u_qualification=None):
    """Assemble the full algorithm-P branch ledger for the allowed set."""
    records = enumerate_branches()
    results = [evaluate_branch(record) for record in records]
    merged = merge_lower_bounds(results)
    gap = gap_assessment(merged, u_seconds, u_source, u_qualification)
    return {
        'kind': 'BRANCH_FAMILY_LEDGER_PER_AGENTS_ALGORITHM_P',
        'families': {
            'layout': list(LAYOUT_FAMILIES),
            'communication': list(COMMUNICATION_FAMILIES),
            'representation': list(REPRESENTATION_FAMILIES),
            'mtp_depths': list(MTP_DEPTHS),
        },
        'premise_slots': dict(PREMISE_SLOT_DESCRIPTIONS),
        'combination_rules': {
            'within_branch': ('Lj = max over verified claims; distinct resources '
                              'may overlap in one elapsed interval, and a sum '
                              'needs a proven non-overlap coupling'),
            'across_branches': ('L = min_j Lj (T* is the infimum over the union '
                                'of all branches)'),
            'global_guard': ('any UNRESOLVED or REJECTED branch keeps global_L '
                             'UNRESOLVED; never min over a proven subset'),
        },
        'total_branches': len(records),
        'branch_results': results,
        'global_L': merged,
        'gap': gap,
        'relation_to_v4_bounds': (
            'The v4 L_global/L_nomtp numbers are declared conditional '
            'arguments over the whole allowed set; they are NOT per-branch '
            'premise declarations under algorithm P and are deliberately not '
            'promoted into this ledger. Under the P2 per-branch standard no '
            'branch currently declares all four premise slots, so global_L '
            'stays UNRESOLVED and the gap stays UNPROVEN.'),
    }


def interface_time(bytes_, ceiling_gb_s):
    if bytes_ < 0 or ceiling_gb_s <= 0:
        raise ValueError("Invalid resource input")
    return bytes_ / (ceiling_gb_s * 1e9)


def active_expert_bytes_per_gpu():
    """10/512 experts active per layer, sharded across ranks."""
    return EXPERT_BYTES_PER_LAYER * (10.0 / N_EXPERTS) * N_LAYERS / N_RANKS


def active_dense_bytes_per_gpu():
    """Dense weights active every token, sharded across ranks.

    Generous to implementations: assumes the dense set can be split
    1/5 per GPU (allowed by 'arbitrary partitioning'), so each GPU
    reads only its share. A mirrored layout would read more, not less.
    """
    return DENSE_BYTES_PER_LAYER * N_LAYERS / N_RANKS


def derive(u_seconds=None, u_source=None, u_qualification=None):
    # Per target row = per public token for no-MTP implementations.
    r1_strong = interface_time(R1_BYTES_STRONG, HOST_READBACK_GB_S)
    r1_weak = interface_time(R1_BYTES_WEAK, HOST_READBACK_GB_S)
    r2_total = R2_BOUNDARIES * interface_time(R2_BYTES_PER_BOUNDARY, PCIE_GEN4_X16_GB_S)

    # R3: active-weight DRAM traffic, per token, for the no-MTP class.
    # Max L2 credit: L2 may retain up to L2_BYTES of the per-token active
    # set (dense or expert, any split) -> dram >= active - L2.
    active_per_gpu = active_expert_bytes_per_gpu() + active_dense_bytes_per_gpu()
    l2_credit = min(L2_BYTES_PER_GPU, active_per_gpu)
    dram_bytes_per_gpu = active_per_gpu - l2_credit
    r3_time = dram_bytes_per_gpu / (DRAM_GB_S * 1e9)

    weight_pool_per_gpu = (EXPERT_BYTES_PER_LAYER + DENSE_BYTES_PER_LAYER) * N_LAYERS / N_RANKS

    # R4: FP32 vector ALU work. FLOPs scale linearly with verified rows,
    # so the per-public-token FLOP count is invariant under MTP batching
    # (h+1 rows per cycle each do the full per-row FLOP work). This is the
    # only weight-side term that survives the perfect-acceptance MTP
    # relaxation, which is why it binds L_global.
    qkv_rows = 6144 + 512 + 512          # q 24*256, k 256, v 256 (36 layers)
    q_rows = 4224                        # 12 full-attention layers (audited)
    ffn = 640
    hc_dim = 4 * HIDDEN
    hc_rank = 320
    ple_key_elems = 21.5e6 / 2           # BF16 bytes -> elements
    ple_value_elems = 5.4e6 / 2
    attn_gate_elems = 10.8e6 / 2
    ssm_out_elems = 10.8e6 / 2
    per_layer_flops = (
        10 * 3 * 2 * ffn * HIDDEN +          # 10 active routed experts, 3 GEMVs
        3 * 2 * ffn * HIDDEN +               # shared expert, 3 GEMVs
        36 * (2 * qkv_rows * HIDDEN) / 48 +  # linear-attention QKV (36 layers)
        12 * (2 * (q_rows + 2 * 256 + 2 * 256) * HIDDEN) / 48 +  # full-attn qkv (12 layers, avg)
        2 * 4224 * HIDDEN * 12 / 48 +        # attn_output (12 layers, avg)
        2 * 6144 * HIDDEN * 36 / 48 +        # linear-attn output proj (avg)
        2 * N_EXPERTS * HIDDEN +             # router GEMV
        2 * 2 * 2 * hc_rank * hc_dim +       # HC down+up GEMVs (4 sites of 320x10240)
        2 * 2 * (ple_key_elems + ple_value_elems + attn_gate_elems + ssm_out_elems)
        # PLE key/value + attn_gate + ssm_out GEMVs: 2 FLOPs per element
    )
    r4_flops = per_layer_flops * N_LAYERS
    r4_time = r4_flops / (N_RANKS * FP32_TFLOPS_PER_GPU * 1e12)

    # L_nomtp: valid for the declared class A_nomtp (one target row per
    # public token; no speculative batching). R3 applies.
    per_token_lo_nomtp = max(r1_weak, r2_total, r3_time, r4_time)
    per_token_lo_nomtp_strong = max(r1_strong, r2_total, r3_time, r4_time)

    # L_global: valid for the FULL allowed set A. Under MTP with any
    # draft policy and perfect acceptance, weight-DRAM traffic (R3) is
    # read once per cycle and amortizes across the h+1 committed rows,
    # so R3 cannot appear in the global bound. R1/R2/R4 survive:
    # R1 is per public token; R2 is per row (rows cross boundaries); R4
    # FLOPs scale with rows. The bound is per public token with
    # rows-per-public-token >= 1.
    per_token_lo_global = max(r1_weak, r2_total, r4_time)
    per_token_lo_global_strong = max(r1_strong, r2_total, r4_time)

    ledger = build_branch_ledger(u_seconds, u_source, u_qualification)

    return {
        "task": "TP5 ordinary single-user generation, Qwen4EXP-AF, 5x RX 6800",
        "version": "v5",
        "supersedes": "v3 (DEFECT-1 dense bytes 90.8->63.82 MB/layer; DEFECT-2 L2 credit "
                      "pool->active denominator; DEFECT-3 MTP amortization invalidates R3 "
                      "in the global bound)",
        "allowed_set_A": [
            "arbitrary lossless weight representation (incl. exact BF16)",
            "unrestricted caching in VRAM/L2/LDS",
            "arbitrary owner/coordinate partitioning across 5 GPUs",
            "arbitrary async prefetch across causal stages",
            "arbitrary kernel/operator fusion",
            "MTP/speculative with any draft policy incl. perfect acceptance",
        ],
        "declared_conditional_class_A_nomtp": (
            "implementations that process exactly one target row per public "
            "token (no speculative row batching); R3 is valid only here"),
        "resources": {
            "R1_output_emission": {
                "bytes_strong": R1_BYTES_STRONG,
                "bytes_weak": R1_BYTES_WEAK,
                "ceiling_gb_s": HOST_READBACK_GB_S,
                "time_strong_s": r1_strong,
                "time_weak_s": r1_weak,
                "justification": "each committed token requires the host to observe either the token id (4B, weak) or the full 512-expert routing scores (2048B, strong, current sampler contract); survives MTP (per public token)",
                "survives_mtp_relaxation": True,
            },
            "R2_causal_cut_set": {
                "bytes_per_boundary": R2_BYTES_PER_BOUNDARY,
                "boundaries_per_row": R2_BOUNDARIES,
                "total_bytes_per_row": R2_BYTES_PER_BOUNDARY * R2_BOUNDARIES,
                "ceiling_gb_s": PCIE_GEN4_X16_GB_S,
                "time_per_row_s": r2_total,
                "justification": "48 sequential layer boundaries; router input (2560 F32) must be complete before layer l runs; with 5 disjoint coordinate shards at least 2048B crosses a rank cut per boundary per ROW; survives MTP (scales with rows)",
                "survives_mtp_relaxation": True,
            },
            "R3_active_weight_dram": {
                "weight_pool_per_gpu_bytes": weight_pool_per_gpu,
                "dense_bytes_per_layer": DENSE_BYTES_PER_LAYER,
                "expert_bytes_per_layer": EXPERT_BYTES_PER_LAYER,
                "active_expert_bytes_per_gpu": active_expert_bytes_per_gpu() + active_dense_bytes_per_gpu(),
                "active_per_gpu_per_token_bytes": active_per_gpu,
                "l2_credit_bytes": l2_credit,
                "dram_bytes_per_gpu_per_token": dram_bytes_per_gpu,
                "ceiling_gb_s": DRAM_GB_S,
                "time_per_row_s": r3_time,
                "justification": "10/512 experts + all dense weights active per token; max L2 credit = 128MB of the active set (any split); valid ONLY for A_nomtp: under MTP row batching the active weights are read once per cycle and amortize across committed rows",
                "survives_mtp_relaxation": False,
            },
            "R4_fp32_alu": {
                "flops_per_token": r4_flops,
                "per_layer_flops": per_layer_flops,
                "peak_flops_s": N_RANKS * FP32_TFLOPS_PER_GPU * 1e12,
                "time_per_row_s": r4_time,
                "justification": "attention projections + router + 11 experts + PLE + HC per layer, 48 layers; FLOPs scale with verified rows so the per-public-token cost is MTP-invariant; RX 6800 has no FP32 matrix cores so vector ALU peak applies; lossless representation changes bytes, not FLOPs",
                "survives_mtp_relaxation": True,
            },
        },
        "combination_rule": "per-token lower bound = max over resources: distinct physical resources may overlap in a legal implementation; same-resource work adds",
        "mtp_relaxation_note": (
            "perfect-acceptance relaxation (any draft policy, unlimited h). "
            "R3 weight traffic is per CYCLE, not per row, and amortizes across "
            "the h+1 committed rows; it is therefore excluded from L_global. "
            "R4 FLOPs and R2 cut bytes scale with rows and survive. R1 is per "
            "public token and survives."),
        "bounds": {
            "L_global": {
                "per_token_lower_bound_s_weak": per_token_lo_global,
                "per_token_lower_bound_s_strong": per_token_lo_global_strong,
                "tps_upper_bound_weak": 1.0 / per_token_lo_global,
                "tps_upper_bound_strong": 1.0 / per_token_lo_global_strong,
                "valid_for": "full allowed set A",
            },
            "L_nomtp": {
                "per_token_lower_bound_s_weak": per_token_lo_nomtp,
                "per_token_lower_bound_s_strong": per_token_lo_nomtp_strong,
                "tps_upper_bound_weak": 1.0 / per_token_lo_nomtp,
                "tps_upper_bound_strong": 1.0 / per_token_lo_nomtp_strong,
                "valid_for": "declared conditional class A_nomtp (no speculative row batching)",
            },
        },
        "branch_ledger": ledger,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--u-seconds', type=float, default=None,
                    help='optional achievable U per public token in seconds')
    ap.add_argument('--u-source', default=None,
                    help='provenance statement required for a valid U')
    ap.add_argument('--u-qualification', default=None,
                    help='qualification statement (conservative statistic) for a valid U')
    args = ap.parse_args()
    cert = derive(args.u_seconds, args.u_source, args.u_qualification)
    with open(args.json, 'w') as f:
        json.dump(cert, f, indent=2)
    b = cert["bounds"]
    r = cert["resources"]
    print(f"R1 weak/strong per row: {r['R1_output_emission']['time_weak_s']:.3e} / {r['R1_output_emission']['time_strong_s']:.3e} s")
    print(f"R2 per row: {r['R2_causal_cut_set']['time_per_row_s']:.3e} s")
    print(f"R3 per row: {r['R3_active_weight_dram']['time_per_row_s']:.3e} s (DRAM {r['R3_active_weight_dram']['dram_bytes_per_gpu_per_token']/1e6:.1f} MB/GPU, A_nomtp only)")
    print(f"R4 per row: {r['R4_fp32_alu']['time_per_row_s']:.3e} s ({r['R4_fp32_alu']['flops_per_token']/1e9:.1f} GFLOP)")
    print(f"L_global  (full A):      {b['L_global']['per_token_lower_bound_s_weak']:.6e} s -> ceiling {b['L_global']['tps_upper_bound_weak']:.0f} tok/s")
    print(f"L_nomtp   (A_nomtp):     {b['L_nomtp']['per_token_lower_bound_s_weak']:.6e} s -> ceiling {b['L_nomtp']['tps_upper_bound_weak']:.0f} tok/s")
    print("NOTE: v4 splits the certificate. L_global excludes R3 because MTP row-batching "
          "amortizes weight-DRAM traffic (v3's single-number claim was unsound). L_nomtp "
          "keeps R3 for the no-MTP conditional class with corrected dense bytes (63.82 "
          "MB/layer) and max L2 credit (128MB of the active set). Both bounds remain "
          "loose: spec ceilings, maximal overlap, no KV-state or latency-chain terms.")
    ledger = cert["branch_ledger"]
    print(f"Branch ledger (algorithm P): {ledger['total_branches']} branches, "
          f"proven={ledger['global_L']['proven_branch_count']}")
    print(f"global_L: {ledger['global_L']['status']}")
    reason = ledger['gap'].get('reason')
    print(f"Gap:      {ledger['gap']['status']}" + (f" ({reason})" if reason else ""))


if __name__ == '__main__':
    main()
