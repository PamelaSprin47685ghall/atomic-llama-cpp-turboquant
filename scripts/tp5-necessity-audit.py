#!/usr/bin/env python3
"""Audit necessary-cost claims without fitting any measured time.

The source inventory is NOT a general algorithmic I/O lower bound. Conditional
resource lemmas and numerical counterexamples are kept separate from current
placement, shader service times, and model-request measurements.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def finite(x: float, *, positive: bool = False) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError('Finite numeric input required')
    if x < 0 or (positive and x == 0):
        raise ValueError('Invalid negative/zero resource input')
    return float(x)


@dataclass(frozen=True)
class NecessaryResource:
    """A conditional counting lemma, not automatic verification of its premises."""
    required_work: float
    work_basis: str
    capacity_upper: float
    capacity_basis: str
    premise: str

    def latency_lower(self) -> float:
        finite(self.required_work)
        finite(self.capacity_upper, positive=True)
        if self.work_basis != 'proved_for_declared_class':
            raise ValueError('Current implementation work is not necessary work')
        if self.capacity_basis != 'upper_bound_for_declared_class':
            raise ValueError('Observed throughput is not a resource capacity upper bound')
        if not self.premise.strip():
            raise ValueError('An explicit implementation class/premise is required')
        return self.required_work / self.capacity_upper


def joint_lower(terms: list[NecessaryResource]) -> float:
    if not terms:
        raise ValueError('No proven/conditional resource terms')
    if len({t.premise for t in terms}) != 1:
        raise ValueError('Cannot mix bounds for different implementation classes')
    # The same elapsed interval can pay both resource costs. Addition requires
    # a separate proof that their mandatory intervals cannot overlap.
    return max(t.latency_lower() for t in terms)


def mandatory_io(distinct_required_bytes: float, initial_fast_capacity: float) -> float:
    """Pigeonhole lemma, conditional on incompressible mandatory original bytes.

    All producers of an equivalent representation belong inside the boundary;
    code constants, prefetched weights, registers and retained intermediates
    must be included in initial_fast_capacity. This function does not establish
    these premises for a fixed neural network.
    """
    return max(0.0, finite(distinct_required_bytes) - finite(initial_fast_capacity))


def mtp_cost(cycle_ms: float, public_tokens: float) -> float:
    return finite(cycle_ms, positive=True) / finite(public_tokens, positive=True)


def exact_counterexamples() -> dict[str, Any]:
    # Two serial mathematical layers, two independent memory channels. Rank1
    # needs90 then10 bytes; rank2 needs10 then90. All immutable addresses known.
    # r1 reads[0,90],[90,100]; r2 reads[0,10],[10,100]. Layer1 computes at90,
    # layer2 at100. Capacity90 bytes/rank suffices; arithmetic is zero-time in
    # this deliberately simplified memory model. No second-layer activation
    # is computed before the first-layer result.
    return {
        'quantifier_direction': {
            'two_correct_implementations_ms': [15, 10],
            'true_minimum_in_this_finite_class_ms': 10,
            'observing_15_does_not_prove_minimum_at_least_15': True},
        'future_weight_prefetch': {
            'weight_bytes_by_layer_and_rank': [[90, 10], [10, 90]],
            'bytes_per_time_unit_each_rank': 1,
            'old_sum_stage_max': 180,
            'constructive_prefetched_finish_time': 100,
            'rank1_reads': [[0, 90], [90, 100]],
            'rank2_reads': [[0, 10], [10, 100]],
            'layer_compute_completion_times': [90, 100],
            'fast_capacity_bytes_per_rank': 90,
            'scope': 'An exact abstract scheduling counterexample, not a measured TP5 implementation'},
        'overlapped_resources': {
            'necessary_memory_ms': 4, 'necessary_compute_ms': 5,
            'joint_unconditional_lower_bound_ms': 5,
            'sum_9_requires_extra_nonoverlap_proof': True},
        'replicated_function': {
            'expression': 'y=f(x;W), identical x andW on five ranks',
            'one_valid_alternative': 'Evaluate f once and distribute y without changing its arithmetic.',
            'what_it_refutes': 'Five evaluations/five weight reads are not required by the mathematical function.',
            'what_it_does_not_prove': 'This alternative is faster, fits memory, or is an optimal TP5 schedule.'},
        'local_speedup_under_fixed_wait': {
            'wait_us': 73.46, 'old_preparation_us': 20.47, 'new_preparation_us': 9.32,
            'old_join_us': max(73.46, 20.47), 'new_join_us': max(73.46, 9.32),
            'meaning': 'Explains a zero local gain conditional on the current wait; does NOT prove the wait necessary.'},
        'speculative_ratio': {
            'ordinary_ms_per_token': 15,
            'cycle_ms': 20, 'committed_tokens': 2,
            'speculative_ms_per_token': mtp_cost(20, 2),
            'meaning': 'Serial token semantics do not imply one completed target call per public token. Not a measured MTP gain.'}}


def import_source(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise ValueError(f'Unable to read source module {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inspect_inventory(metadata: Path, layout: Path, shards: int) -> dict[str, Any]:
    records = json.loads(metadata.read_text())
    if not 0 < shards <= len(records):
        raise ValueError('Invalid shard count')
    tensors = [t for r in records[:shards] for t in r['tensors']]
    if len({t['name'] for t in tensors}) != len(tensors):
        raise ValueError('Duplicate tensor names require alias/deduplication review')
    const = import_source('necessity_gguf_constants', ROOT/'gguf-py/gguf/constants.py')
    old = import_source('necessity_old_roof', ROOT/'scripts/tp5-roofline.py')
    kinds = {t['name']: t['kind'] for t in json.loads(layout.read_text())['tensor_layouts']}
    unique = Fraction(0)
    rank_bytes = [Fraction(0)]*5
    stage_bytes: dict[Any, list[Fraction]] = defaultdict(lambda: [Fraction(0)]*5)
    old_stage_bytes: dict[Any, list[float]] = defaultdict(lambda: [0.0]*5)
    families: dict[str, Fraction] = defaultdict(Fraction)
    excluded = []
    tensor_records = []
    for t in tensors:
        name = t['name']; kind = kinds[name]
        shape = t['shape']
        if not shape or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in shape):
            raise ValueError(f'Invalid tensor shape: {name}')
        block, size = const.GGML_QUANT_SIZES[const.GGMLQuantizationType(t['raw_type'])]
        elements = math.prod(shape)
        if elements % block:
            raise ValueError(f'Non-integral quantization extent: {name}')
        payload = elements // block * size
        if payload > t['storage_extent_bytes']:
            raise ValueError(f'Payload larger than recorded storage extent: {name}')
        # Recompute the original274 arithmetic exactly as originally defined,
        # but never promote it to a necessary lower bound.
        m = re.match(r'blk\.(\d+)\.(.*)', name)
        stage = ('tail', 0) if not m else (int(m[1]), int('ffn' in m[2]))
        for r in range(5):
            old_stage_bytes[stage][r] += t['storage_extent_bytes']*old.selected_fraction(t)*old.rank_fraction(t,kind,r,False)
        if kind == 'cpu' or name in ('token_embd.weight','per_layer_token_embd.weight'):
            excluded.append(name)
            continue
        fraction = Fraction(10, shape[-1]) if re.search(r'_exps?\.weight$',name) else Fraction(1)
        if fraction > 1:
            raise ValueError('Selected experts exceed available experts')
        active = payload*fraction
        unique += active
        families['routed_expert' if fraction < 1 else 'mirrored' if kind=='mirror' else 'other'] += active
        copies = []
        for r in range(5):
            f = Fraction(old.rank_fraction(t,kind,r,False)).limit_denominator(1000000)
            v = active*f
            rank_bytes[r] += v; stage_bytes[stage][r] += v; copies.append(v)
        tensor_records.append({'name':name, 'kind':kind, 'encoded_payload_bytes':payload,
                               'selected_fraction':str(fraction), 'active_unique_bytes':float(active),
                               'current_plan_total_copy_bytes':float(sum(copies))})
    old_b = sum(max(v) for v in old_stage_bytes.values())
    return {
        'kind': 'SOURCE_INVENTORY_NOT_MANDATORY_IO_PROOF',
        'metadata_sha256': hashlib.sha256(metadata.read_bytes()).hexdigest(),
        'layout_sha256': hashlib.sha256(layout.read_bytes()).hexdigest(),
        'quantization_constants_sha256': hashlib.sha256((ROOT/'gguf-py/gguf/constants.py').read_bytes()).hexdigest(),
        'tensors_seen':len(tensors), 'tensors_counted':len(tensor_records),
        'excluded_indexed_or_current_cpu_tensors':excluded,
        'active_unique_original_encoded_bytes':float(unique),
        'active_current_plan_copy_bytes_by_rank':[float(x) for x in rank_bytes],
        'active_current_plan_total_copy_bytes':float(sum(rank_bytes)),
        'copy_amplification_ratio':float(sum(rank_bytes)/unique),
        'current_no_prefetch_sum_stage_max_encoded_bytes':float(sum(max(v) for v in stage_bytes.values())),
        'current_prefetch_aggregate_rank_resource_bytes':float(max(rank_bytes)),
        'families_unique_bytes':{k:float(v) for k,v in families.items()},
        'original_extent_based_stage_max_bytes':old_b,
        'original_512GBs_arithmetic_tps':512e9/old_b,
        'caveats':[
            'The original encoded payload is not a proved minimum representation for these fixed model weights.',
            'Ten experts per token is the stated graph family; actual routing, prefixes, cache residency and multi-token reuse remain variable.',
            'Current CPU/indexed tensors are excluded to reproduce a GPU-weight inventory, not to prove every allowed implementation omits them.',
            'Cross-tensor physical/content aliases were not proven absent; unique here means tensor-name deduplication, not information-theoretic incompressibility.',
            'Current rank fractions are observed design decisions. Only the inventory, not their necessity, is established.'],
        'tensors':tensor_records}


def run(metadata: Path, layout: Path, shards: int) -> dict[str, Any]:
    inv = inspect_inventory(metadata,layout,shards)
    u = inv['active_unique_original_encoded_bytes']
    sensitivity = []
    # Deliberately labelled illustrative premises, NOT discovered hardware
    # cache capacity or a bound for unrestricted exact inference.
    for fast_mib_per_rank in (0,128,256):
        io = mandatory_io(u, 5*fast_mib_per_rank*2**20)
        sensitivity.append({'assumed_aggregate_fast_capacity_MiB':5*fast_mib_per_rank,
                            'conditional_required_original_bytes':io,
                            'assumed_aggregate_capacity_GB_s':5*512,
                            'conditional_resource_ms':io/(5*512*1e6),
                            'conditional_rate_ceiling_tps':None if io==0 else 5*512e9/io,
                            'eligible_as_unrestricted_model_bound':False})
    return {
        'kind':'NECESSITY_RESTART_NO_FITTED_OR_MEASURED_LATENCY',
        'retracted':'The 274→95–104→64.74–67.68 chain has not established a necessary hardware upper bound.',
        'global_necessary_tps_upper_bound':None,
        'global_optimality_gap_certified':False,
        'inventory':inv,
        'conditional_io_sensitivity':sensitivity,
        'conditional_io_premise':
            'Every counted original byte must be consumed within the interval; no smaller equivalent representation, code constants, reconstruction or retained sufficient state; all initially available fast information counted inC; no other input path; aggregate bandwidth bounded by5×512GB/s. These premises are NOT proved for the unrestricted user task.',
        'counterexamples':exact_counterexamples(),
        'proof_obligations':[
            'Freeze the semantic workload/quality and latency scope, not the current Vulkan implementation.',
            'For every alleged bound prove required work for ALL allowed implementations and resource CAPACITY upper bounds.',
            'Keep mathematical dependencies distinct from avoidable barriers, dispatches, mailbox layouts and host round trips.',
            'Minimize the relaxation over allowed placement, caching, prefetch and MTP choices before claiming a uniform bound.',
            'Only compare proved lower latencyL and a correct measured achievable latencyU on the same workload; low prediction error is not a small optimality gap.']}


def inspect_observations(root: Path) -> dict[str, Any]:
    """Read-only recheck of already collected observations, outside the theory.

    This is mechanical provenance/content consistency, NOT comprehensive
    semantic qualification and NOT a new GPU benchmark.
    """
    groups: dict[str, list[dict[str, Any]]] = {'A': [], 'B': []}
    identities = set()
    raw_checks = 0
    for block in range(1,5):
        bodies = {}
        for arm in ('A','B'):
            folder = root/f'pack-b{block}-{arm}'
            result = json.loads((folder/'result.json').read_text())
            manifest = json.loads((folder/'manifest.json').read_text())
            if result['profile'] != 'none' or result['shutdown']['returncode'] or result['shutdown']['forced']:
                raise ValueError('Observation is instrumented or had abnormal teardown')
            identities.add(json.dumps({'binary':manifest['bin_sha256'],
                'libraries':json.loads((folder/'mapped-libraries.json').read_text()),
                'model':manifest['model_stat_identity']},sort_keys=True))
            tokens = 0; elapsed = 0.0; body_records=[]; original_hashes=[]
            for request in result['requests']:
                if request['phase']!='measured':continue
                path=folder/f"request-{request['idx']}.json"
                raw=json.loads(path.read_text());response=raw['response'];choice=response['choices'][0]
                body=choice['message']['content'];count=response['usage']['completion_tokens']
                milliseconds=response['timings']['predicted_ms']
                sha=hashlib.sha256(body.encode()).hexdigest()
                if (raw['phase']!='measured' or count!=request['n'] or count!=response['timings']['predicted_n'] or
                    milliseconds!=request['predicted_ms'] or sha!=request['content_sha256'] or choice['finish_reason']!='stop' or
                    request['issues'] or request['fallback_lines']):
                    raise ValueError(f'Observation raw consistency failed: {path}')
                finite(milliseconds,positive=True)
                if not math.isclose(count*1000/milliseconds,request['tps'],rel_tol=1e-12):
                    raise ValueError('Wrong derived measured rate')
                tokens+=count;elapsed+=milliseconds;raw_checks+=1
                body_records.append((count,body));original_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            if len(body_records)!=3:raise ValueError('Expected three measured requests per process')
            bodies[arm]=body_records
            groups[arm].append({'block':block,'public_tokens':tokens,'generation_ms':elapsed,
                                'tps':tokens*1000/elapsed,'raw_response_sha256':original_hashes})
        if bodies['A']!=bodies['B']:raise ValueError('Matched measured bodies differ')
    if len(identities)!=1:raise ValueError('Mixed executable/model/mapped library identities')
    return {'kind':'HISTORICAL_RAW_OBSERVATIONS_RECHECKED_NOT_NEW_BENCHMARK',
            'source':str(root), 'normal_process_exits':8,'raw_measured_responses_rechecked':raw_checks,
            'matched_measured_bodies':True,'identity_sets':1,
            'arms':{arm:{'pooled_tps':sum(x['public_tokens'] for x in rows)*1000/sum(x['generation_ms'] for x in rows),
                         'processes':rows} for arm,rows in groups.items()},
            'eligible_as_latency_lower_bound':False,
            'quality_scope':'Mechanical raw identity/content/timing consistency; not a new numerical or semantic qualification.'}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--layout',type=Path,default=Path('tp5-manifest.json'))
    p.add_argument('--shards',type=int,default=6)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--observations',type=Path,help='Separately recheck existing four pack A/B pairs; never used by bound calculations')
    a=p.parse_args()
    if a.out.exists():p.error('Refuse to overwrite an audit record')
    try:
        d=run(a.metadata,a.layout,a.shards)
        if a.observations:d['observations']=inspect_observations(a.observations)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        brief={k:v for k,v in d.items() if k not in ('inventory','counterexamples')}
        brief['inventory']={k:v for k,v in d['inventory'].items() if k!='tensors'}
        print(json.dumps(brief,ensure_ascii=False,indent=2))
    except (OSError,ValueError,KeyError,TypeError) as e:
        p.error(str(e))


if __name__=='__main__':main()
