#!/usr/bin/env python3
"""Compare one real prompt alone and in a batch using a supplied C binary."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--gpu-layers', type=int, default=0)
    parser.add_argument('--group-start', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.group_start < 0:
        parser.error('--group-start must be non-negative')
    lines = args.prompts.read_text(encoding='utf-8').splitlines()[args.group_start:args.group_start + 3]
    if len(lines) != 3:
        parser.error('--prompts must contain three lines from the selected group start')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='run-', dir=args.output_dir))
    common = ['--model', str(args.model), '--prompts', '-', '--predict', '16', '--ctx-size', '256',
              '--batch-size', '8', '--ubatch-size', '8', '--threads', '2',
              '--gpu-layers', str(args.gpu_layers), '--ignore-eog', '--no-phase-memory', '--json']
    records = {}
    for name, prompts, capacity, index in [('three', lines, 2048, 1),
                                           ('alone-wide', [lines[1]], 2048, 0),
                                           ('alone-tight', [lines[1]], 256, 0)]:
        command = [str(args.binary), *common, '--total-kv', str(capacity)]
        run = subprocess.run(command, input=('\n'.join(prompts) + '\n').encode('utf-8'),
                             capture_output=True, timeout=600)
        (output / f'{name}.jsonl').write_bytes(run.stdout)
        (output / f'{name}.log').write_bytes(run.stderr)
        if run.returncode:
            raise RuntimeError(f'{name} failed: {run.stderr[-2000:].decode("utf-8", "replace")}')
        events = [json.loads(line) for line in run.stdout.decode('utf-8').splitlines() if line.strip()]
        results = [event for event in events if event['event'] == 'result']
        if len(results) != len(prompts) or any(event['event'] == 'error' for event in events):
            raise RuntimeError(f'{name} did not complete every request')
        records[name] = {'config': next(event for event in events if event['event'] == 'config'),
                         'result': results[index], 'command': command}
    summary = {'binary': str(args.binary.resolve()), 'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
               'output_dir': str(output), 'runs': records,
               'alone_wide_vs_tight_equal': records['alone-wide']['result']['tokens'] == records['alone-tight']['result']['tokens'],
               'three_vs_alone_equal': records['three']['result']['tokens'] == records['alone-wide']['result']['tokens'],
               'limits': ['A shape comparison can expose pre-existing differences; this does not explain their numerical cause.']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
