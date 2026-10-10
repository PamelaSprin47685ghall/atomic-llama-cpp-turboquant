#!/usr/bin/env python3
"""Summarize retained model directory and TP5 measurements; no GPU execution.

Weight bytes are logical source bytes, not physical bandwidth counters. The
roofline scenarios below are explicitly conditional streaming models.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
import math


def inventory(records, manifest):
    layouts = {t['name']: t for t in manifest['tensor_layouts']}
    groups = defaultdict(lambda: {'stored_bytes': 0, 'active_source_bytes': 0, 'active_elements': 0, 'tensors': 0})
    by_layout = defaultdict(float)
    details = []
    for record in records:
        for t in record['tensors']:
            name, shape = t['name'], t['shape']
            category, fraction = 'dense_other', 1
            if '_exps.' in name:
                category, fraction = 'routed_experts', 10 / shape[-1]
            elif name == 'per_layer_token_embd.weight':
                category, fraction = 'ple_lookup', 16 / shape[-1]
            elif 'token_embd' in name:
                category, fraction = 'token_lookup', 1 / shape[-1]
            elif 'ple' in name and len(shape) > 1 and shape[-1] > 1_000_000:
                category, fraction = 'ple_lookup', 16 / shape[-1]
            elif 'hc_' in name or '_hc_' in name:
                category = 'hc'
            elif 'gate_inp' in name:
                category = 'router'
            elif '_shexp.' in name:
                category = 'shared_expert'
            elif name == 'output.weight':
                category = 'vocabulary'
            kind = layouts.get(name, {}).get('kind', 'unmapped')
            row = {**t, 'category': category, 'active_fraction': fraction, 'historical_layout_kind': kind,
                   'active_source_bytes': t['storage_extent_bytes'] * fraction}
            details.append(row)
            by_layout[kind] += row['active_source_bytes']
            g = groups[category]
            g['stored_bytes'] += t['storage_extent_bytes']
            g['active_source_bytes'] += row['active_source_bytes']
            g['active_elements'] += t['elements'] * fraction
            g['tensors'] += 1
    return {'groups': dict(groups), 'active_source_bytes_by_historical_layout': dict(by_layout),
            'raw_types': dict(Counter(t['raw_type'] for t in details)),
            'layout_kinds': dict(Counter(t['historical_layout_kind'] for t in details)),
            'large_tables': [t for t in details if t['storage_extent_bytes'] > 5_000_000_000],
            'details': details,
            'scope': 'Logical once-per-token weights with 10/512 routed experts and indexed lookups; not measured VRAM traffic.'}


def effective_profile(result, manifest):
    # Earlier host-timer runs used --env instead of --profile. Preserve the raw
    # result, but never let that legacy label qualify an instrumented score.
    env = manifest.get('env', {})
    if env.get('GGML_DECODE_COST', '0') not in ('0', 'off'):
        return 'host-cost'
    return result.get('profile', manifest.get('args', {}).get('profile', 'none'))


def summarize(root):
    rows = []
    for path in sorted(root.glob('*/result.json')):
        r = json.loads(path.read_text())
        manifest_path = path.parent / 'manifest.json'
        m = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        a = m.get('args', {})
        prof = effective_profile(r, m)
        status = 'DIAGNOSTIC_NOT_RANKED' if prof != 'none' and r['status'] == 'MEASURED_NOT_QUALIFIED' else r['status']
        rows.append({'run': path.parent.name, 'status': status, 'recorded_status': r['status'], 'pooled_tps': r.get('pooled_tps'),
                     'horizon': a.get('horizon'), 'latebind': a.get('latebind'), 'mmvq': a.get('mmvq'),
                     'threads': a.get('threads'), 'profile': prof, 'error': r.get('error'),
                     'shutdown': r.get('shutdown'),
                     'measured_tokens': [q['n'] for q in r.get('requests', []) if q['phase'] == 'measured']})
    return rows


def fields(text):
    return {k: float(v) for k, v in re.findall(r'([A-Za-z_][A-Za-z0-9_]*)=(-?[0-9]+(?:\.[0-9]+)?)(?![0-9.])', text)}


def profile(root):
    reports = []
    for run in sorted(root.iterdir()):
        if not run.is_dir():
            continue
        for path in sorted(run.glob('request-*.json')):
            obj = json.loads(path.read_text())
            if obj.get('phase') != 'measured' or not path.with_suffix('.log').exists():
                continue
            text = path.with_suffix('.log').read_text(errors='replace')
            groups = defaultdict(list)
            for tag, rest in re.findall(r'\[(tp5-[\w-]+)\]([^\n]*)', text):
                groups[tag].append(fields(rest))
            cycles = groups['tp5-mtp-cycle']
            if not cycles and not groups['tp5-relay-gpu']:
                continue
            r = obj['response']; timing = r['timings']; n = r['usage']['completion_tokens']
            sums = {k: sum(c.get(k, 0) for c in cycles) / 1000 for k in ('draft_us','target_us','catchup_us','handoff_us','total_us')}
            row = {'run': run.name, 'request': path.name, 'public_tokens': n, 'cycles': len(cycles),
                   'public_tokens_per_observed_cycle': n / len(cycles) if cycles else None,
                   'cycle_component_ms': sums if cycles else None,
                   'generation_ms': timing['predicted_ms'], 'prompt_ms': timing['prompt_ms'], 'wall_ms': obj['wall_ms'],
                   'generation_minus_cycle_components_ms': timing['predicted_ms'] - sums['total_us'] if cycles else None,
                   'max_target_us': max((c.get('target_us', 0) for c in cycles), default=None),
                   'accepted_histogram': dict(Counter(int(c.get('accepted_tokens', 0)) for c in cycles)),
                   'logged_final_tokens': sum(c.get('final_tokens', 0) for c in cycles) if cycles else None,
                   'gpu_capture_scope': 'Captured chains only, not whole-request totals; ranks overlap; tail_compute is nested.'}
            row['target_cycle_median_us'] = statistics.median(c.get('target_us', 0) for c in cycles) if cycles else None
            row['target_first_cycle_us'] = cycles[0].get('target_us', 0) if cycles else None
            row['definition_lines_rank0'] = len(re.findall(r'\[tp5-linear-definition\] rank=0 ', text))
            gpu = defaultdict(list)
            for g in groups['tp5-relay-gpu']:
                gpu[(int(g['rank']), int(g.get('rows', 0)))].append(g)
            row['gpu_by_rank_rows'] = {
                f'rank{rank}_rows{rows}': {'samples': len(gs), **{k: statistics.median(x.get(k, 0) for x in gs)
                    for k in ('span_us', 'p2_us', 'compute_us', 'p1_us', 'late_pre_us', 'gaps_us', 'tail_compute_us')}}
                for (rank, rows), gs in gpu.items()}
            for tag in ('tp5-relay-profile','tp5-latebind-profile'):
                gs = groups[tag]
                row[tag] = {'count': len(gs), 'totals': {k: sum(g.get(k, 0) for g in gs) for k in
                    ('stages','ready_us','skew_us','arm_us','cpu_data_us','generation_us','handoff_us','submit_us','sidecar_wait_us','sidecar_data_us')}}
            reports.append(row)
    (root / 'profiles.json').write_text(json.dumps(reports, indent=2) + '\n')
    return reports


def paired_report(root, prefix):
    blocks = []
    identities = set()
    for b in range(1, 5):
        arms = {}
        for arm in ('A', 'B'):
            folder = root / f'{prefix}-b{b}-{arm}'
            r = json.loads((folder / 'result.json').read_text())
            m = json.loads((folder / 'manifest.json').read_text())
            libs = json.loads((folder / 'mapped-libraries.json').read_text())
            if r['status'] != 'MEASURED_NOT_QUALIFIED' or effective_profile(r, m) != 'none' or r['shutdown']['forced'] or r['shutdown']['returncode']:
                raise ValueError(f'Unqualified execution: {folder}')
            identities.add(json.dumps({'bin': m['bin_sha256'], 'libraries': libs}, sort_keys=True))
            measured = [q for q in r['requests'] if q['phase'] == 'measured']
            if not measured or any(q['issues'] or q['finish'] != 'stop' for q in measured):
                raise ValueError(f'Invalid measured requests: {folder}')
            nt = sum(q['n'] for q in measured)
            arms[arm] = {'ms_per_token': sum(q['predicted_ms'] for q in measured) / nt,
                         'wall_ms_per_token': sum(q['wall_ms'] for q in measured) / nt,
                         'generation_ms': statistics.mean(q['predicted_ms'] for q in measured),
                         'client_wall_ms': statistics.mean(q['wall_ms'] for q in measured),
                         'prompt_ms': statistics.mean(q['timings']['prompt_ms'] for q in measured),
                         'tokens': nt, 'hashes': sorted(set(q['content_sha256'] for q in measured)),
                         'draft_proposed': sum(q['timings'].get('draft_n', 0) for q in measured),
                         'draft_accepted': sum(q['timings'].get('draft_n_accepted', 0) for q in measured),
                         'fallback_lines': sum(len(q['fallback_lines']) for q in measured),
                         'definition_lines_rank0': [len(re.findall(r'\[tp5-linear-definition\] rank=0 ',
                             (folder / f'request-{q["idx"]}.log').read_text())) for q in measured]}
        blocks.append(arms)
    if len(identities) != 1:
        raise ValueError('Paired process binary/DSO identity changed')
    summary = {}
    for metric in ('ms_per_token', 'wall_ms_per_token', 'prompt_ms', 'generation_ms', 'client_wall_ms'):
        vals = [math.log(b['B'][metric] / b['A'][metric]) for b in blocks]
        center = statistics.mean(vals)
        radius = 3.182446305284263 * statistics.stdev(vals) / 2
        summary[metric] = {'B_over_A_geometric': math.exp(center),
                           'latency_ratio_95pct_CI': [math.exp(center-radius), math.exp(center+radius)],
                           'speedup_geometric': math.exp(-center),
                           'speedup_95pct_CI': [math.exp(-center-radius), math.exp(-center+radius)],
                           'paired_log_ratios': vals}
    summary['arm_process_tps'] = {arm: [1000 / b[arm]['ms_per_token'] for b in blocks] for arm in ('A','B')}
    summary['arms_pooled'] = {arm: {
        'public_tokens': sum(b[arm]['tokens'] for b in blocks),
        'tps': 1000 * sum(b[arm]['tokens'] for b in blocks) / sum(b[arm]['tokens'] * b[arm]['ms_per_token'] for b in blocks),
        'mean_generation_ms_per_request': statistics.mean(b[arm]['generation_ms'] for b in blocks),
        'mean_client_wall_ms_per_request': statistics.mean(b[arm]['client_wall_ms'] for b in blocks),
        'mean_prompt_ms_per_request': statistics.mean(b[arm]['prompt_ms'] for b in blocks),
        'fallback_line_count': sum(b[arm]['fallback_lines'] for b in blocks)} for arm in ('A','B')}
    for arm in ('A', 'B'):
        proposed = sum(b[arm]['draft_proposed'] for b in blocks)
        accepted = sum(b[arm]['draft_accepted'] for b in blocks)
        nt = summary['arms_pooled'][arm]['public_tokens']
        if proposed:
            summary['arms_pooled'][arm].update({
                'draft_proposed': proposed, 'draft_accepted': accepted,
                'draft_acceptance': accepted / proposed,
                'public_tokens_per_proposed_h1_cycle': nt / proposed,
                'h1_count_identity_public_equals_proposed_plus_accepted': nt == proposed + accepted,
                'generation_ms_per_proposed_h1_cycle': sum(b[arm]['tokens'] * b[arm]['ms_per_token'] for b in blocks) / proposed})
    summary['blocks'] = blocks
    summary['method'] = 'Four independent paired process blocks; per-process pooled request latency; two-sided Student-t on log ratio, df=3. Semantic quality is reviewed separately.'
    return summary


def unique_texts(root):
    texts = {}
    for p in sorted(root.glob('*/request-*.json')):
        obj = json.loads(p.read_text())
        if 'response' not in obj:
            continue
        r = obj['response']
        text = r['choices'][0]['message'].get('content') or ''
        import hashlib
        key = hashlib.sha256(text.encode()).hexdigest()
        row = texts.setdefault(key, {'text': text, 'occurrences': [], 'finish': set(), 'tokens': set()})
        row['occurrences'].append(str(p.relative_to(root)))
        row['finish'].add(r['choices'][0]['finish_reason'])
        row['tokens'].add(r['usage']['completion_tokens'])
    for row in texts.values():
        row['finish'] = sorted(row['finish'])
        row['tokens'] = sorted(row['tokens'])
    (root / 'unique-outputs.json').write_text(json.dumps(texts, ensure_ascii=False, indent=2)+'\n')
    (root / 'unique-outputs.txt').write_text('\n\n'.join(f'{key} tokens={row["tokens"]} count={len(row["occurrences"])} first={row["occurrences"][0]}\n{row["text"]}' for key,row in texts.items()))
    return len(texts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--metadata', type=Path)
    p.add_argument('--layout', type=Path, default=Path('tp5-manifest.json'))
    p.add_argument('--profiles', action='store_true')
    p.add_argument('--brief', action='store_true')
    p.add_argument('--paired-prefix', action='append', default=[])
    p.add_argument('--unique-texts', action='store_true')
    a = p.parse_args()
    report = {'runs': summarize(a.root)}
    conversion_path = a.root / 'hcq8-conversion.json'
    if conversion_path.exists():
        conv = json.loads(conversion_path.read_text())
        changed = conv.get('changed_tensors', [])
        report['hc_conversion'] = {'status': conv['status'], 'changed_tensors': len(changed),
            'unchanged_tensors_byte_verified': len(conv.get('unchanged_tensors', [])),
            'bytes_saved': sum(f['source_bytes'] - f['output_bytes'] for f in conv.get('files', [])),
            'max_weight_relative_l2': max((t['relative_l2'] for t in changed), default=None),
            'max_weight_absolute_error': max((t['max_abs'] for t in changed), default=None),
            'scope': 'Weight conversion statistics only, not logits error or semantic equivalence.'}
    if a.profiles:
        report['profiles'] = profile(a.root)
    report['paired'] = {prefix: paired_report(a.root, prefix) for prefix in a.paired_prefix}
    if a.unique_texts:
        report['unique_output_count'] = unique_texts(a.root)
    if a.metadata:
        records = json.loads(a.metadata.read_text())
        target = inventory(records[:2], json.loads(a.layout.read_text()))
        draft = inventory(records[2:], json.loads(a.layout.read_text()))
        (a.root / 'weight-details.json').write_text(json.dumps({'target': target, 'draft': draft}, ensure_ascii=False, indent=2) + '\n')
        report['weights'] = {role: {k: v for k, v in obj.items() if k != 'details'} for role, obj in (('target', target), ('draft', draft))}
    (a.root / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if a.brief:
        for r in report['runs']:
            print(r['run'], r['status'], r.get('pooled_tps'))
        for r in report.get('profiles', []):
            print(r['run'], r['request'], 'cycles', r['cycles'], 'sum_ms', r['cycle_component_ms'],
                  'first_us', r['target_first_cycle_us'], 'median_us', r['target_cycle_median_us'],
                  'definitions', r['definition_lines_rank0'], 'gpu', r['gpu_by_rank_rows'])
        for role, w in report.get('weights', {}).items():
            print(role, 'active_layout_bytes', w['active_source_bytes_by_historical_layout'])
        for prefix, paired in report['paired'].items():
            print('PAIRED', prefix, {k:v for k,v in paired.items() if k != 'blocks'})
        if 'hc_conversion' in report:
            print('HC_CONVERSION', report['hc_conversion'])
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
