#!/usr/bin/env python3
"""Freeze a host-timing-calibrated implementation model, then test held-out runs.

This is not a silicon speed limit. It predicts measured generation duration
conditional on the public output length, using disjoint calibrated intervals.
Unexplained server time is reported but NEVER fitted into a fudge factor.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

COMPONENTS = ('setup', 'graph', 'alloc', 'input', 'submit', 'finish',
              'decode_other', 'sample_sync', 'sample_select')

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def fields(line):
    return dict(re.findall(r'(\w+)=(\S+)', line))

def identity(folder):
    m = load(folder / 'manifest.json')
    env = {k: v for k, v in m['env'].items()
           if k not in ('GGML_DECODE_COST', 'GGML_TP5_RELAY_DEBUG', 'GGML_TP5_MTP_PROFILE')}
    return {'bin_sha256': m['bin_sha256'], 'libraries': load(folder / 'mapped-libraries.json'),
            'model': m['model_stat_identity'], 'env': env,
            'args': {k: m['args'][k] for k in ('horizon', 'latebind', 'mmvq', 'threads', 'ctx')}}

def request_cost(path):
    raw = load(path)
    n = raw['response']['usage']['completion_tokens']
    timing = raw['response']['timings']
    if n != timing['predicted_n'] or n < 3:
        raise ValueError('Expected at least three consistent public tokens')
    ubatches, decodes, samples = [], [], []
    for line in path.with_suffix('.log').read_text().splitlines():
        f = fields(line)
        if '[decode-cost]' in line and f['rows'] == '1':
            row = {k: int(v) for k, v in f.items() if k.endswith('_us') or k == 'reuse'}
            (ubatches if f['scope'] == 'ubatch' else decodes).append(row)
        if '[sample-cost]' in line:
            samples.append({k: int(v) for k, v in f.items() if k.endswith('_us')})
    # First public token is sampled from prefill before the generation timer.
    if not (len(ubatches) == len(decodes) == n - 1 and len(samples) == n):
        raise ValueError(f'Unclosed ordinary single-stream population: {path}: '
                         f'{n=} ubatches={len(ubatches)} decodes={len(decodes)} samples={len(samples)}')
    if [i for i, r in enumerate(ubatches) if r['reuse'] == 0] != [0]:
        raise ValueError('This model requires exactly one initial scalar graph rebuild')
    steps = []
    for u, d, s in zip(ubatches, decodes, samples[1:]):
        parts = {k: u[k + '_us'] / 1000 for k in COMPONENTS[:6]}
        parts.update(decode_other=(d['total_us'] - u['total_us']) / 1000,
                     sample_sync=s['sync_us'] / 1000,
                     sample_select=s['select_us'] / 1000)
        if any(v < 0 or not math.isfinite(v) for v in parts.values()):
            raise ValueError('Negative/nonfinite interval or mismatched nested scopes')
        steps.append(parts)
    explained = sum(sum(s.values()) for s in steps)
    return {'path': str(path), 'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'trace_sha256': hashlib.sha256(path.with_suffix('.log').read_bytes()).hexdigest(),
            'tokens': n, 'generation_ms': timing['predicted_ms'],
            'first_ms': steps[0], 'steady_ms': {k: statistics.mean(s[k] for s in steps[1:]) for k in COMPONENTS},
            'unexplained_server_ms': timing['predicted_ms'] - explained,
            'unexplained_fraction': (timing['predicted_ms'] - explained) / timing['predicted_ms']}

def calibrate(folders):
    expected = None
    rows = []
    for folder in folders:
        result = load(folder / 'result.json')
        if result['status'] == 'FAILED' or result['shutdown']['returncode'] or result['shutdown']['forced']:
            raise ValueError('Failed calibration process')
        ident = identity(folder)
        if ident['args']['horizon'] != 0:
            raise ValueError('MTP requires a different cycle model')
        if expected is not None and expected != ident:
            raise ValueError('Mixed calibration identities')
        expected = ident
        for p in sorted(folder.glob('request-*.json')):
            r = load(p)
            if r.get('phase') == 'measured':
                rows.append(request_cost(p))
    if len(rows) < 2:
        raise ValueError('At least two calibration requests required')
    model = {'kind': 'CALIBRATED_IMPLEMENTATION_PREDICTION_NOT_HARDWARE_BOUND',
             'necessity_status': 'UNPROVEN_CURRENT_IMPLEMENTATION_COSTS',
             'certifies_optimality_match': False,
             'frozen_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
             'identity': expected, 'calibration': rows,
             'first_ms': {k: statistics.mean(r['first_ms'][k] for r in rows) for k in COMPONENTS},
             'steady_ms': {k: statistics.mean(r['steady_ms'][k] for r in rows) for k in COMPONENTS},
             'formula': 'T(N)=sum(first_ms)+(N-2)*sum(steady_ms), N>=3; first prefill token excluded from timer',
             'match_relative_error_limit': 0.05,
             'excluded_from_prediction': 'Unexplained server remainder; no post-hoc scaling or fitted overhead',
             'limitations': 'One initial scalar rebuild, fixed configuration, warm request, conditional on output length. Not cold start, not hardware optimum, not quality qualification.'}
    return model

def predict(model, n):
    if isinstance(n, bool) or not isinstance(n, int) or n < 3:
        raise ValueError('N must be an integer >=3')
    values = list(model['first_ms'].values()) + list(model['steady_ms'].values())
    if any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Invalid model components')
    ms = sum(model['first_ms'].values()) + (n - 2) * sum(model['steady_ms'].values())
    if not math.isfinite(ms) or ms <= 0:
        raise ValueError('Invalid predicted duration')
    return {'tokens': n, 'predicted_ms': ms, 'predicted_tps': n * 1000 / ms}

def sampler_counterfactual(baseline, micro):
    """Predict B using ONLY baseline A timings and the independent CPU microtest."""
    if baseline['identity']['env'].get('GGML_TP5_GREEDY_HOST') != '0':
        raise ValueError('Counterfactual requires the untouched sampler baseline')
    if micro.get('scope') != 'CPU_SAMPLING_MICRO_NOT_MODEL' or micro.get('vocabulary') != 248320:
        raise ValueError('Wrong independent CPU calibration')
    old, new = micro['ordinary_us'], micro['unique_us']
    if any(isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in (old, new)) or new >= old:
        raise ValueError('Invalid or non-improving microkernel costs')
    delta = (old - new) / 1000
    result = copy.deepcopy(baseline)
    result['kind'] = 'BASELINE_PLUS_INDEPENDENT_CPU_COUNTERFACTUAL'
    result['frozen_utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    result['baseline_identity'] = copy.deepcopy(baseline['identity'])
    result['identity']['env']['GGML_TP5_GREEDY_HOST'] = '1'
    for part in ('first_ms', 'steady_ms'):
        result[part]['sample_select'] -= delta
        if result[part]['sample_select'] < 0:
            raise ValueError('Independent reduction exceeds calibrated sampling cost; no clamping')
    result['independent_cpu_micro'] = micro
    result['predicted_saving_ms_per_scalar_sample'] = delta
    result['candidate_measurements_used'] = False
    result['limitations'] += ' Counterfactual assumes all non-sampler intervals unchanged; CPU micro saving transfers unchanged. No B timing is fitted.'
    return result

def validate(model, folders):
    results = []
    for folder in folders:
        r = load(folder / 'result.json')
        m = load(folder / 'manifest.json')
        if r['status'] == 'FAILED' or r['profile'] != 'none' or r['shutdown']['returncode'] or r['shutdown']['forced']:
            raise ValueError('Validation must be successful and uninstrumented')
        if m['started_utc'] <= model['frozen_utc']:
            raise ValueError('Validation started before model was frozen')
        if identity(folder) != model['identity']:
            raise ValueError('Validation identity/configuration differs from calibration')
        measured = [q for q in r['requests'] if q['phase'] == 'measured']
        if not measured or any(q['issues'] or q['fallback_lines'] for q in measured):
            raise ValueError('Validation quality/fallback gate failed')
        rows = []
        for q in measured:
            raw = load(folder / f"request-{q['idx']}.json")
            response = raw['response']
            body = response['choices'][0]['message']['content']
            if (response['usage']['completion_tokens'] != q['n'] or
                    response['timings']['predicted_n'] != q['n'] or
                    response['timings']['predicted_ms'] != q['predicted_ms'] or
                    hashlib.sha256(body.encode()).hexdigest() != q['content_sha256'] or
                    response['choices'][0]['finish_reason'] != 'stop'):
                raise ValueError('Raw validation response differs from summary')
            if not math.isfinite(q['predicted_ms']) or q['predicted_ms'] <= 0:
                raise ValueError('Invalid measured duration')
            pred = predict(model, q['n'])
            rows.append({**pred, 'actual_ms': q['predicted_ms'], 'actual_tps': q['tps'],
                         'content_sha256': q['content_sha256'],
                         'relative_time_error': pred['predicted_ms'] / q['predicted_ms'] - 1})
        prediction = sum(q['predicted_ms'] for q in rows)
        actual = sum(q['actual_ms'] for q in rows)
        n = sum(q['tokens'] for q in rows)
        error = prediction / actual - 1
        results.append({'run': str(folder), 'requests': rows, 'predicted_tps': n*1000/prediction,
                        'actual_tps': n*1000/actual, 'relative_time_error': error,
                        'within_limit': abs(error) <= model['match_relative_error_limit']})
    if not results:
        raise ValueError('No validation processes')
    return {'kind': 'HELD_OUT_MODEL_VALIDATION', 'frozen_utc': model['frozen_utc'],
            'certifies_optimality_match': False,
            'match_kind': 'PREDICTION_ERROR_ONLY_NOT_OPTIMALITY_GAP',
            'runs': results, 'all_within_limit': all(r['within_limit'] for r in results),
            'not_a_claim': 'Matching a calibrated model does not prove hardware optimality or resolve prior protocol failures.'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('calibrate'); c.add_argument('--runs', nargs='+', type=Path, required=True); c.add_argument('--out', type=Path, required=True)
    v = sub.add_parser('validate'); v.add_argument('--model', type=Path, required=True); v.add_argument('--runs', nargs='+', type=Path, required=True); v.add_argument('--out', type=Path, required=True)
    cf = sub.add_parser('counterfactual'); cf.add_argument('--baseline', type=Path, required=True); cf.add_argument('--micro', type=Path, required=True); cf.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('Output exists; do not overwrite frozen models or validation results')
    try:
        if a.command == 'calibrate':
            data = calibrate(a.runs)
        elif a.command == 'counterfactual':
            data = sampler_counterfactual(load(a.baseline), json.loads(a.micro.read_text().splitlines()[-1]))
        else:
            data = validate(load(a.model), a.runs)
        a.out.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in data.items() if k not in ('identity','calibration')},ensure_ascii=False,indent=2))
    except (OSError, ValueError, KeyError) as e:
        p.error(str(e))

if __name__ == '__main__':
    main()
