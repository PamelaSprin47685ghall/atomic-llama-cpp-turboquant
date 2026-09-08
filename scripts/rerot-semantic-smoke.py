#!/usr/bin/env python3
"""Bounded, isolated RERoT reproduction; records evidence, NOT an answer-quality score.

The child server belongs to this process and is stopped in finally. No production
unit, existing process, sampler setting or answer stop string is changed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def resolve_context(context: int | None, total_kv: str) -> int:
    """Keep an explicit manual capacity legal without silently changing it."""
    if context is not None and context < 1:
        raise ValueError('context must be >= 1')
    if total_kv == 'auto':
        return 131072 if context is None else context
    try:
        capacity = int(total_kv)
    except ValueError as exc:
        raise ValueError('total-kv must be auto or a positive integer') from exc
    if capacity < 1:
        raise ValueError('total-kv must be >= 1')
    if context is None:
        return min(131072, capacity)
    if context > capacity:
        raise ValueError('without TriAttention, manual total-kv must be >= context; '
                         'choose an explicit smaller --context or use --total-kv auto')
    return context


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--build-dir', default='build-vulkan-localhost')
    parser.add_argument('--context', type=int, help='default: 131072 for auto KV; min(131072,total-kv) for manual KV')
    parser.add_argument('--total-kv', default='auto', help='auto for production-style joint B/P/K fitting, or an explicit token capacity')
    parser.add_argument('--parallel', type=int, help='explicit server slot count; only valid with non-auto total-kv for RERoT')
    parser.add_argument('--cache', choices=('turbo', 'f16'), default='turbo', help='f16 is an explicit numerical control, not the production gate')
    parser.add_argument('--prompt', default='世界上每个大洲有哪些国家')
    parser.add_argument('--seed', type=int, default=424242)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--port', type=int, default=18081)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout', type=float, default=240)
    parser.add_argument('--audit', action='store_true', help='read-only raw-logit diagnostics; not a performance run')
    parser.add_argument('--serial-baseline', action='store_true', help='explicit RERoT-OFF diagnostic with the same question/seed/budget')
    parser.add_argument('--rbb-ablation', choices=('none', 'local-state', 'shared-rbb', 'raw-redundant'), default='none', help='default child recurrence is local; shared-rbb/raw-redundant are explicit research modes')
    parser.add_argument('--frontier', choices=('strong', 'lag1'), default='strong', help='lag1 is a research control, not the strong-frontier release gate')
    parser.add_argument('--child-contract', action='store_true', help='research control: explicit PRIVATE topic-independent task/exit instruction')
    parser.add_argument('--ancestors-only', action='store_true', help='research control: no peer lexical KV; never a release run')
    parser.add_argument('--concurrency', type=int, default=1, help='independent RERoT-OFF requests for batched-backend control')
    args = parser.parse_args()
    requested_context = args.context
    try:
        args.context = resolve_context(args.context, args.total_kv)
    except ValueError as exc:
        parser.error(str(exc))
    if args.concurrency < 1 or (args.concurrency != 1 and not args.serial_baseline):
        parser.error('concurrency >1 is restricted to the explicit RERoT-OFF baseline')
    if args.parallel is not None and args.parallel < 1:
        parser.error('parallel must be >= 1')
    if not args.serial_baseline and args.parallel is not None and args.total_kv == 'auto':
        parser.error('RERoT full-auto owns B/P/K; use explicit --total-kv with --parallel')
    run = args.output or Path(tempfile.mkdtemp(prefix='rerot-semantic-'))
    if args.output:
        run.mkdir(parents=True, exist_ok=False)
    run = run.resolve()
    bindir = Path(args.build_dir).resolve() / 'bin'
    key = secrets.token_urlsafe(24)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', args.port))
    command = [str(bindir / 'llama-server'), '-m', args.model, '-a', 'ornith-1.5',
               '-c', str(args.context), '--total-kv', str(args.total_kv), '-ngl', '40', '-kvo',
               '-b', '4096', '-ub', '2048', '-ctk', 'turbo4', '-ctv', 'turbo2',
               '--rerot', '--rerot-frontier', args.frontier, '--run-dump', str(run / 'trie'),
               '--metrics', '--fit', 'off', '--load-mode', 'mmap', '--host', '127.0.0.1',
               '--port', str(args.port), '--api-key', key, '--jinja', '--reasoning-preserve']
    if args.parallel is not None:
        command.extend(['-np', str(args.parallel)])
    if args.cache == 'f16':
        command[command.index('-ctk') + 1] = 'f16'
        command[command.index('-ctv') + 1] = 'f16'
    request = {'model': 'ornith-1.5', 'messages': [{'role': 'user', 'content': args.prompt}],
               'temperature': args.temperature, 'seed': args.seed, 'max_tokens': args.max_tokens,
               'stream': False, 'rerot': True, 'rerot_trace': True}
    if args.serial_baseline:
        command.remove('--rerot')
        frontier_index = command.index('--rerot-frontier')
        del command[frontier_index:frontier_index + 2]
        request['rerot'] = False
        request['rerot_trace'] = False
        if args.concurrency > 1 and args.parallel is None:
            command.extend(['-np', str(args.concurrency)])

    def save_json(name: str, obj: object) -> None:
        (run / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(bindir) + ':' + env.get('LD_LIBRARY_PATH', '')
    env.pop('LLAMA_REROT_AUDIT', None)
    env.pop('LLAMA_REROT_AUDIT_DIR', None)
    env.pop('LLAMA_REROT_RBB_ABLATION', None)
    env.pop('LLAMA_REROT_CHILD_CONTRACT', None)
    env.pop('LLAMA_REROT_ANCESTORS_ONLY', None)
    if args.ancestors_only:
        env['LLAMA_REROT_ANCESTORS_ONLY'] = '1'
    if args.child_contract:
        env['LLAMA_REROT_CHILD_CONTRACT'] = '1'
    if args.rbb_ablation != 'none':
        env['LLAMA_REROT_RBB_ABLATION'] = args.rbb_ablation
    if args.audit:
        env['LLAMA_REROT_AUDIT'] = '1'
        env['LLAMA_REROT_AUDIT_DIR'] = str(run / 'tensor-audit')
    save_json('request.json', request)
    save_json('args.json', ['<redacted>' if value == key else value for value in command])
    libraries = ['llama-server', 'libllama-server-impl.so', 'libllama.so', 'libllama-common.so',
                 'libmtmd.so', 'libggml.so', 'libggml-base.so', 'libggml-vulkan.so', 'libggml-cpu.so']
    save_json('hashes.json', {name: sha256_file(bindir / name) for name in libraries})
    save_json('library-paths.json', {name: str((bindir / name).resolve()) for name in libraries})
    save_json('environment.json', {name: value for name, value in env.items()
                                 if name.startswith(('LLAMA_', 'GGML_', 'VK_')) or name == 'LD_LIBRARY_PATH'})
    repo = Path(__file__).resolve().parents[1]
    def capture(argv: list[str]) -> str:
        return subprocess.run(argv, cwd=repo, env=env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=False).stdout.strip()
    save_json('source.json', {
        'head': capture(['git', 'rev-parse', 'HEAD']),
        'status_porcelain': capture(['git', 'status', '--porcelain=v1']),
        'server_version': capture([str(bindir / 'llama-server'), '--version']),
    })
    (run / 'source.patch').write_text(capture(['git', 'diff', '--binary']) + '\n', encoding='utf-8')
    save_json('diagnostics.json', {'audit': args.audit, 'serial_baseline': args.serial_baseline,
                                'rbb_ablation': args.rbb_ablation,
                                'frontier': args.frontier,
                                'child_contract': args.child_contract,
                                'ancestors_only': args.ancestors_only,
                                'concurrency': args.concurrency,
                                'cache': args.cache,
                                'response_timeout': args.timeout, 'context': args.context,
                                'context_requested': requested_context,
                                'context_resolution': 'explicit' if requested_context is not None else
                                    ('auto-default' if args.total_kv == 'auto' else 'manual-capacity-default'),
                                'total_kv': args.total_kv, 'parallel': args.parallel,
                                'seed': args.seed, 'temperature': args.temperature,
                                'max_tokens': args.max_tokens})

    def http(path: str, data: bytes | None = None, timeout: float = 3) -> tuple[int, bytes]:
        req = urllib.request.Request(f'http://127.0.0.1:{args.port}' + path, data=data,
                                     headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    proc = None
    report = {}
    print('EVIDENCE_DIR=' + str(run), flush=True)
    try:
        with (run / 'server.log').open('wb') as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f'server exited before ready: {proc.returncode}')
                try:
                    if http('/health')[0] == 200:
                        break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError('server readiness timeout')
            # Record what the live process actually mapped, not just sonames
            # next to the wrapper binary. This also catches library-path drift.
            maps = Path(f'/proc/{proc.pid}/maps')
            if maps.is_file():
                mapped = maps.read_text()
                (run / 'process-maps.txt').write_text(mapped, encoding='utf-8')
                selected = set()
                for line in mapped.splitlines():
                    fields = line.split(None, 5)
                    if len(fields) == 6 and fields[5].startswith('/'):
                        path = Path(fields[5])
                        if path.name.startswith(('libllama', 'libggml', 'libmtmd')) and path.is_file():
                            selected.add(path)
                save_json('mapped-library-hashes.json', {str(path): sha256_file(path) for path in sorted(selected)})
            (run / 'metrics-before.txt').write_bytes(http('/metrics')[1])
            def inference(index: int) -> dict:
                start = time.monotonic()
                try:
                    status, raw = http('/v1/chat/completions', json.dumps(request, ensure_ascii=False).encode(), args.timeout)
                    wall = time.monotonic() - start
                    filename = 'response.json' if args.concurrency == 1 else f'response-{index}.json'
                    (run / filename).write_bytes(raw)
                    obj = json.loads(raw)
                    choice = obj.get('choices', [{}])[0]
                    message = choice.get('message', {})
                    return {'http_status': status, 'wall_seconds': wall, 'finish_reason': choice.get('finish_reason'),
                            'content_chars': len(message.get('content') or ''),
                            'reasoning_chars': len(message.get('reasoning_content') or message.get('reasoning') or ''),
                            'usage': obj.get('usage'), 'error': obj.get('error')}
                except Exception as exc:
                    return {'harness_error': repr(exc), 'wall_seconds': time.monotonic() - start}
            if args.concurrency == 1:
                report = inference(0)
            else:
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    report = {'requests': list(pool.map(inference, range(args.concurrency)))}
            (run / 'metrics-after.txt').write_bytes(http('/metrics')[1])
    except Exception as exc:
        report['harness_error'] = repr(exc)
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        report['dev_server_exit'] = proc.returncode if proc else None
        save_json('report.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    # These are transport/lifecycle checks only. A country list can be wrong
    # even when all three pass; inspect response.json and the public document.
    rows = report.get('requests', [report])
    return 0 if (not report.get('harness_error') and all(
        row.get('http_status') == 200 and row.get('finish_reason') == 'stop' and row.get('content_chars', 0) > 0
        for row in rows)) else 1


if __name__ == '__main__':
    raise SystemExit(main())
