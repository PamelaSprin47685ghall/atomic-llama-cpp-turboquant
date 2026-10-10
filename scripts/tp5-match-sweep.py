#!/usr/bin/env python3
"""Bounded preregistered configuration sweep. Delegates GPU ownership to the runner."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--bin', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--draft', type=Path, required=True)
    a = p.parse_args()
    cases = [('baseline', 'aggressive', 'disabled', 22, []),
             ('q8first', 'q8first', 'disabled', 22, []),
             ('pack-once', 'pack-once', 'disabled', 22, []),
             ('ap', 'ap', 'disabled', 22, []),
             ('l2', 'l2', 'disabled', 22, []),
             ('mmvq-auto', 'aggressive', 'auto', 22, []),
             ('mmvq-forced', 'aggressive', 'forced', 22, []),
             ('threads8', 'aggressive', 'disabled', 8, []),
             ('threads1', 'aggressive', 'disabled', 1, []),
             ('definition-cache4', 'aggressive', 'disabled', 22,
              ['--env', 'GGML_TP5_DEFINITION_CACHE=4'])]
    a.root.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, str(Path(__file__).with_name('tp5-gap-bench.py')),
            '--bin', str(a.bin), '--model', str(a.model), '--draft', str(a.draft),
            '--horizon', '0', '--warmup', '2', '--repeats', '2',
            '--env', 'GGML_TP5_REPLICATE_ATTN=0', '--env', 'GGML_TP5_QSA_HEADMAP=1',
            '--env', 'GGML_TP5_GDN_HEADMAP=1', '--env', 'GGML_VK_SMALL_Q8_COLUMNS=1']
    done = []
    for name, late, mmvq, threads, extra in cases:
        dest = a.root / ('screen-' + name)
        cmd = base + ['--latebind', late, '--mmvq', mmvq, '--threads', str(threads), '--out', str(dest)] + extra
        print('BEGIN', name, flush=True)
        rc = subprocess.run(cmd, check=False).returncode
        done.append({'name': name, 'returncode': rc, 'argv': cmd})
        (a.root / 'sweep-execution.json').write_text(json.dumps(done, indent=2) + '\n')
        if rc:
            print('STOP_ON_FAILURE', name, rc, flush=True)
            return rc
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
