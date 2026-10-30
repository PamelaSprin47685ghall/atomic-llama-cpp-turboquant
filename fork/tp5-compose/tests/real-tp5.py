#!/usr/bin/env python3
"""Real isolated TP5 CLI acceptance; no substitute model, synthetic logits, or retries."""

import argparse
import array
import json
import math
import os
import pathlib
import re
import shlex
import signal
import subprocess
import sys
import time


PROMPTS = ('A', 'B', 'C', 'D')
BDF = re.compile(r'^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$')
PROFILES = {
    'qwen3-q8': {
        'numeric_policy': 'q8_0-decoded-weights-f32-activations-ordered-f32',
        'reference_mode': 'single-gpu-reference',
        'context_tokens': 512,
    },
    'swift': {
        'numeric_policy': 'mixed-original-weights-f32-activations-ordered-f32',
        'reference_mode': 'cpu-staged-five-gpu-reference',
        'context_tokens': 262144,
    },
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def save_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f'invalid {name}: {value!r}')
    return value


def load_json(path):
    def reject_constant(value):
        raise ValueError(f'non-JSON numeric constant: {value}')

    def reject_duplicate(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'duplicate JSON key: {key}')
            result[key] = value
        return result

    with path.open(encoding='utf-8') as stream:
        return json.load(stream, parse_constant=reject_constant, object_pairs_hook=reject_duplicate)


class Runner:
    def __init__(self, args):
        self.args = args
        self.records = []
        self.common = [
            '--model', str(args.model), '--predict', str(args.predict),
            '--ctx-size', str(args.ctx_size), '--threads', str(args.threads),
            '--deadline', str(args.deadline),
            '--flash-attn', args.flash_attn,
            '--ignore-eos',
        ]
        for prompt in args.prompts:
            self.common += ['--prompt', prompt]

    def run(self, case, extra, expected_success=True, timeout=None):
        directory = self.args.output_dir / case / 'attempt-1'
        directory.mkdir(parents=True, exist_ok=False)
        report = directory / 'report.json'
        command = [str(self.args.binary), *self.common, *extra, '--output', str(report)]
        record = {
            'case': case, 'attempt': 1, 'command': command,
            'shell_command': shlex.join(command), 'expected_success': expected_success,
            'timeout_seconds': timeout if timeout is not None else self.args.timeout,
            'report': str(report), 'stdout': str(directory / 'stdout.log'),
            'stderr': str(directory / 'stderr.log'),
        }
        save_json(directory / 'command.json', record)
        self.records.append(record)
        started = time.monotonic_ns()
        timed_out = False
        returncode = None
        try:
            with (directory / 'stdout.log').open('xb') as stdout, (directory / 'stderr.log').open('xb') as stderr:
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
                try:
                    returncode = process.wait(timeout=record['timeout_seconds'])
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        returncode = process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        record['kill_wait_timed_out'] = True
        finally:
            record.update(returncode=returncode, timed_out=timed_out,
                          elapsed_ns=time.monotonic_ns() - started)
            save_json(directory / 'exit.json', record)
        require(not timed_out, f'{case}: CLI did not exit within its bound (killed; not an accepted failure)')
        if expected_success:
            require(returncode == 0, f'{case}: CLI exited {returncode}; see {directory / "stderr.log"}')
            result = load_json(report)
            require(result['schema_version'] == 1 and result['status'] == 'success', f'{case}: incomplete report')
            return report, result
        require(returncode == 1, f'{case}: injection did not produce a caught exception (exit {returncode}; fail-stop/crash is not a clean abort)')
        marker = (f'TP5_INJECTED_ABORT window={self.args.abort_window} channel={self.args.abort_channel}'
                  f' epoch={self.args.abort_epoch}').encode('ascii')
        require(re.search(re.escape(marker) + rb'(?![0-9])', (directory / 'stderr.log').read_bytes()),
                f'{case}: intended injection was not reached')
        require(not report.exists(), f'{case}: injected abort published a success report')
        return report, None


def check_report(report, args, mode, cancelled=None, capture=False):
    run = report['run']
    profile = PROFILES[args.profile]
    expected_mode = 'tp5-relay' if mode == 'relay' else profile['reference_mode']
    require(run['execution_mode'] == expected_mode, 'wrong execution mode / fallback')
    require(run.get('numeric_policy') == profile['numeric_policy'],
            'numeric policy differs from the explicitly selected profile')
    require(isinstance(run.get('model_description'), str) and run['model_description'],
            'missing actual model description')
    require(report['options']['model'] == str(args.model), 'requested model changed')
    require(report['options']['context_per_request'] == args.ctx_size, 'requested context changed')
    require(report['options']['mode'] == mode, 'option mode mismatch')
    require(report['options']['capture_logits'] is capture, 'full logits capture setting changed')
    require(report['options']['cancel_request'] == (-1 if cancelled is None else cancelled),
            'cancellation target changed')
    require(report['options']['abort_window'] == -1, 'unexpected abort injection in a successful case')
    expected_flash_attention = args.flash_attn == 'on'
    require(report['options'].get('flash_attention') is expected_flash_attention,
            'requested Flash Attention setting changed')
    require(run.get('flash_attention') is expected_flash_attention,
            'executed Flash Attention setting differs from preregistration / fallback')
    require(report['options']['prompts'] == list(args.prompts), 'the fixed requests changed')
    require(report['options']['predict'] == args.predict, 'prediction budget changed')
    require(report['options']['ignore_eos'] is True, 'fixed trajectory requires ignore-eos')
    integer(run['vocabulary'], 'vocabulary', 1)
    start = integer(run['started_ns'], 'started_ns', 1)
    finish = integer(run['finished_ns'], 'finished_ns', start)
    require(run['reserved_tokens_after_close'] == 0, 'token reservation leaked')
    require(run['active_requests_after_close'] == 0, 'active request budget leaked')
    expected_windows = args.windows if mode == 'relay' else 1
    require(report['options']['windows'] == expected_windows, 'wrong requested window count')
    windows = report['windows']
    require([w['id'] for w in windows] == list(range(expected_windows)), 'missing/reordered windows')
    for window in windows:
        require(window['native_drained'] is True, 'unproved native drain')
        assigned = sum(request['window'] == window['id'] for request in report['requests'])
        integer(window['context_tokens'], 'context_tokens', args.ctx_size * assigned)
        integer(window['sequence_tokens'], 'sequence_tokens', args.ctx_size)
        integer(window['kv_capacity_tokens'], 'kv_capacity_tokens', args.ctx_size)
        first = integer(window['first_submit_ns'], 'first_submit_ns', start)
        last = integer(window['last_finish_ns'], 'last_finish_ns', first)
        require(last <= finish, 'window timestamp outside run')
        integer(window['decode_calls'], 'decode_calls', 1)
        integer(window['input_rows'], 'input_rows', 1)
        if mode == 'relay':
            for key in ('epochs', 'channels', 'cpu_reductions', 'input_retirements', 'output_retirements',
                        'hip_runtime_version'):
                integer(window[key], key, 1)
            for key in ('rank_graphs', 'rank_compute_nodes', 'rank_relay_nodes', 'rank_published_bytes', 'rank_consumed_bytes'):
                require(len(window[key]) == 5, f'{key}: not five ranks')
                for value in window[key]:
                    integer(value, key, 1)
            require(window['input_retirements'] == window['cpu_reductions'] == window['output_retirements'],
                    'CPU input completion and actual last-reader retirement differ')
            require(window['rank_published_bytes'] == window['rank_consumed_bytes'],
                    'a rank did not consume all of its published result bytes')
            bdfs = window['pci_bdf']
            require(len(bdfs) == 5 and all(BDF.fullmatch(value) for value in bdfs), 'invalid five-rank BDF evidence')
            require(len({value.lower() for value in bdfs}) == 5, 'ranks do not use five distinct BDFs')
            if args.expected_bdf:
                require([b.lower() for b in bdfs] == [b.lower() for b in args.expected_bdf], 'rank BDF mapping differs from preregistration')
    if mode == 'relay' and expected_windows > 1:
        require(windows[0]['pci_bdf'] == windows[1]['pci_bdf'], 'windows changed rank-to-device mapping')
        require(run['max_simultaneous_windows'] >= 2, 'no simultaneous native windows')
        require(max(w['first_submit_ns'] for w in windows) < min(w['last_finish_ns'] for w in windows),
                'window execution intervals did not overlap')
    elif mode == 'relay':
        require(run['max_simultaneous_windows'] == 1, 'single-window scope changed')
    requests = report['requests']
    require([r['id'] for r in requests] == list(range(len(args.prompts))), 'missing or reordered requests')
    used_windows = set()
    for request in requests:
        require(request['window'] in range(expected_windows), 'invalid request window')
        require(request['window'] == request['id'] % expected_windows, 'request-to-window assignment changed')
        used_windows.add(request['window'])
        require(request['prompt_tokens'], 'empty tokenized prompt')
        require(len(request['prompt_tokens']) + args.predict <= args.ctx_size,
                'prompt and generation exceed the requested per-request context')
        for token in request['prompt_tokens']:
            require(type(token) is int and 0 <= token < run['vocabulary'], 'prompt token outside vocabulary')
        admitted = integer(request['admitted_ns'], 'admitted_ns', start)
        ended = integer(request['finished_ns'], 'request finished_ns', admitted)
        require(ended <= finish, 'request timestamp outside run')
        expected_count = args.cancel_after if request['id'] == cancelled else args.predict
        expected_reason = 'cancelled' if request['id'] == cancelled else 'budget'
        require(len(request['tokens']) == expected_count, f'request {request["id"]}: incorrect accepted token count')
        require(request['stop_reason'] == expected_reason, f'request {request["id"]}: unexpected stop reason')
        previous = admitted
        for index, token in enumerate(request['tokens']):
            for key in ('input_token', 'sampled_token', 'emitted_token'):
                require(type(token[key]) is int and 0 <= token[key] < run['vocabulary'], f'invalid {key}')
            require(token['input_position'] == len(request['prompt_tokens']) - 1 + index, 'input position was lost or replayed')
            expected_input = request['prompt_tokens'][-1] if index == 0 else request['tokens'][index - 1]['emitted_token']
            require(token['input_token'] == expected_input, 'emitted token was not the next consumed input')
            previous = integer(token['committed_ns'], 'committed_ns', previous)
            require(previous <= ended, 'token committed after request finished')
    require(used_windows == set(range(expected_windows)), 'a context window had no actual requests')
    return {'mode': run['execution_mode'], 'numeric_policy': run['numeric_policy'],
            'model_description': run['model_description'],
            'flash_attention': run['flash_attention'], 'requests': len(requests),
            'tokens': sum(len(r['tokens']) for r in requests),
            'window_capacity': [{key: w[key] for key in ('id', 'context_tokens', 'sequence_tokens', 'kv_capacity_tokens')}
                                for w in windows],
            'max_simultaneous_windows': run['max_simultaneous_windows'],
            'rank_bdfs': [w['pci_bdf'] for w in windows],
            'reserved_tokens_after_close': run['reserved_tokens_after_close'],
            'active_requests_after_close': run['active_requests_after_close']}


def check_teacher(reference, relay):
    require(reference['run']['vocabulary'] == relay['run']['vocabulary'], 'vocabulary changed')
    for key in ('numeric_policy', 'model_description'):
        require(reference['run'][key] == relay['run'][key], f'reference and relay differ in {key}')
    for key in ('model', 'context_per_request'):
        require(reference['options'][key] == relay['options'][key], f'reference and relay differ in {key}')
    require(reference['run']['flash_attention'] is relay['run']['flash_attention'],
            'reference and relay used different Flash Attention settings')
    mismatches = 0
    for expected, actual in zip(reference['requests'], relay['requests']):
        require(expected['prompt_tokens'] == actual['prompt_tokens'], 'tokenized prompt changed')
        require(len(expected['tokens']) == len(actual['tokens']), 'trajectory length changed')
        for before, after in zip(expected['tokens'], actual['tokens']):
            require(before['sampled_token'] == before['emitted_token'], 'reference was teacher-forced')
            for key in ('input_token', 'input_position', 'emitted_token'):
                require(before[key] == after[key], f'teacher trajectory differs in {key}')
            mismatches += after['sampled_token'] != after['emitted_token']
    return mismatches


def companion(report_path, filename):
    require(isinstance(filename, str) and pathlib.Path(filename).name == filename and filename not in ('.', '..'),
            'logits companion is not a local basename')
    path = report_path.parent / filename
    require(path.is_file() and not path.is_symlink(), 'missing or symlinked logits companion')
    return path


def logit_rows(report_path, report):
    """Yield validated row identities and full binary rows; retain at most one row."""
    metadata = report['logits']
    require(metadata and metadata['dtype'] == 'f32le', 'missing full binary32 capture')
    vocabulary = report['run']['vocabulary']
    count = sum(len(request['tokens']) for request in report['requests'])
    require(metadata['vocabulary'] == vocabulary and metadata['rows'] == count, 'wrong capture row/vocabulary counts')
    require(metadata['bytes'] == count * vocabulary * 4, 'wrong capture byte count')
    binary_path = companion(report_path, metadata['binary'])
    rows_path = companion(report_path, metadata['rows_file'])
    require(binary_path.stat().st_size == metadata['bytes'], 'truncated or extra binary logits')
    row_number = 0
    with binary_path.open('rb') as binary, rows_path.open(encoding='utf-8') as identities:
        for request in report['requests']:
            for index, token in enumerate(request['tokens']):
                line = identities.readline()
                require(line != '', 'missing logits identity row')
                identity = json.loads(line)
                expected = {
                    'schema_version': 1, 'row': row_number, 'request_id': request['id'],
                    'window_id': request['window'], 'token_index': index,
                    'vocabulary': vocabulary, 'offset_bytes': row_number * vocabulary * 4,
                    'count': vocabulary, 'dtype': 'f32le',
                    **{key: token[key] for key in ('input_token', 'input_position', 'sampled_token', 'emitted_token', 'committed_ns')},
                }
                require(identity == expected, f'logits row {row_number}: identity differs from report')
                require(token['logit_row'] == row_number, 'token points to wrong logits row')
                values = array.array('f')
                raw = binary.read(vocabulary * 4)
                require(len(raw) == vocabulary * 4, 'short logits row')
                values.frombytes(raw)
                if sys.byteorder != 'little':
                    values.byteswap()
                require(len(values) == vocabulary, 'host float32 representation mismatch')
                yield identity, values
                row_number += 1
        require(identities.readline() == '', 'extra logits identity rows')
        require(binary.read(1) == b'', 'extra logits binary values')
    require(row_number == count, 'row count differs from report')


def compare_logits(reference_path, reference, relay_path, relay, args):
    metrics = {
        'atol': args.atol, 'rtol': args.rtol,
        'criterion': 'abs(relay-reference) <= atol + rtol * abs(reference), every full-vocabulary value',
        'rows': 0, 'values': 0, 'error_count': 0, 'nonfinite_count': 0,
        'max_absolute_error': 0.0, 'max_relative_error_nonzero_reference': 0.0,
        'zero_reference_nonzero_error_count': 0,
        'worst_absolute_identity': None,
        'argmax_mismatch_rows': 0, 'max_relative_l2': 0.0,
        'max_kl_reference_to_relay': 0.0, 'max_total_variation': 0.0,
        'max_target_nll_absolute_difference': 0.0,
    }
    expected = iter(logit_rows(reference_path, reference))
    actual = iter(logit_rows(relay_path, relay))
    while True:
        before = next(expected, None)
        after = next(actual, None)
        if before is None or after is None:
            require(before is None and after is None, 'full capture row counts differ')
            break
        identity, left = before
        other_identity, right = after
        for key in ('row', 'request_id', 'token_index', 'input_token', 'input_position', 'emitted_token', 'vocabulary', 'count'):
            require(identity[key] == other_identity[key], f'comparison row identity differs: {key}')
        require(len(left) == len(right), 'comparison vocabulary sizes differ')
        metrics['rows'] += 1
        row_finite = True
        for vocabulary_index, (a, b) in enumerate(zip(left, right)):
            metrics['values'] += 1
            if not (math.isfinite(a) and math.isfinite(b)):
                metrics['nonfinite_count'] += 1
                metrics['error_count'] += 1
                row_finite = False
                continue
            error = abs(b - a)
            if error > metrics['max_absolute_error']:
                metrics['max_absolute_error'] = error
                metrics['worst_absolute_identity'] = {
                    'request_id': identity['request_id'], 'token_index': identity['token_index'],
                    'vocabulary_index': vocabulary_index, 'reference': a, 'relay': b,
                }
            if a:
                metrics['max_relative_error_nonzero_reference'] = max(metrics['max_relative_error_nonzero_reference'], error / abs(a))
            elif error:
                metrics['zero_reference_nonzero_error_count'] += 1
            if error > args.atol + args.rtol * abs(a):
                metrics['error_count'] += 1
        if row_finite:
            squared_error = math.fsum((b - a) ** 2 for a, b in zip(left, right))
            squared_reference = math.fsum(a * a for a in left)
            relative_l2 = math.sqrt(squared_error / max(squared_reference, 1e-30))
            metrics['max_relative_l2'] = max(metrics['max_relative_l2'], relative_l2)
            best_left = max(range(len(left)), key=left.__getitem__)
            best_right = max(range(len(right)), key=right.__getitem__)
            metrics['argmax_mismatch_rows'] += best_left != best_right
            maximum_left, maximum_right = left[best_left], right[best_right]
            probabilities_left = [math.exp(value - maximum_left) for value in left]
            probabilities_right = [math.exp(value - maximum_right) for value in right]
            sum_left, sum_right = math.fsum(probabilities_left), math.fsum(probabilities_right)
            log_z_left = maximum_left + math.log(sum_left)
            log_z_right = maximum_right + math.log(sum_right)
            kl = math.fsum((p / sum_left) * (a - log_z_left - b + log_z_right)
                          for a, b, p in zip(left, right, probabilities_left))
            variation = 0.5 * math.fsum(abs(p / sum_left - q / sum_right)
                                     for p, q in zip(probabilities_left, probabilities_right))
            target = identity['emitted_token']
            nll_difference = abs((log_z_left - left[target]) - (log_z_right - right[target]))
            metrics['max_kl_reference_to_relay'] = max(metrics['max_kl_reference_to_relay'], kl)
            metrics['max_total_variation'] = max(metrics['max_total_variation'], variation)
            metrics['max_target_nll_absolute_difference'] = max(metrics['max_target_nll_absolute_difference'], nll_difference)
    metrics['within_preregistered_tolerance'] = metrics['error_count'] == 0
    return metrics


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def tolerance(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError('must be finite and nonnegative')
    return result


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=pathlib.Path, required=True)
    parser.add_argument('--model', type=pathlib.Path, required=True)
    parser.add_argument('--output-dir', type=pathlib.Path, required=True,
                        help='must not exist; all evidence is exclusively created here')
    parser.add_argument('--profile', choices=tuple(PROFILES), default='qwen3-q8',
                        help='explicit expected public numeric/reference profile; Swift uses only fixed prompt A and one window')
    parser.add_argument('--predict', type=positive_int, default=8)
    parser.add_argument('--ctx-size', type=positive_int,
                        help='per-request capacity; default 512 for qwen3-q8, required 262144 for swift')
    parser.add_argument('--threads', type=positive_int, default=4)
    parser.add_argument('--flash-attn', choices=('on', 'off'), default='off',
                        help='same attention setting for every case; off is the Qwen3 performance default, not a qualification claim for another profile')
    parser.add_argument('--deadline', type=positive_int, default=300)
    parser.add_argument('--timeout', type=positive_int, help='external process bound; default deadline + 60 seconds')
    parser.add_argument('--atol', type=tolerance,
                        help='default .02 for legacy Qwen3 candidate runs, .002 for Swift; Swift cannot widen this gate')
    parser.add_argument('--rtol', type=tolerance,
                        help='default .01 for legacy Qwen3 candidate runs, .001 for Swift; Swift cannot widen this gate')
    parser.add_argument('--expected-bdf', action='append', default=[], help='repeat exactly five times in rank order')
    parser.add_argument('--delay-ms', type=positive_int, default=2000)
    parser.add_argument('--cancel-after', type=positive_int, default=2)
    parser.add_argument('--abort-timeout', type=positive_int,
                        help='external bound including model load; default 90 seconds for Qwen3, deadline + 60 for Swift')
    parser.add_argument('--abort-window', type=int, choices=(0, 1), default=0)
    parser.add_argument('--abort-channel', type=int, default=0)
    parser.add_argument('--abort-epoch', type=positive_int, default=1,
                        help='match a later epoch to exercise captured/replayed work')
    parser.add_argument('--cases', default='main,slow,cancel,abort',
                        help='comma-separated subset of main,slow,cancel,abort; omitted cases are never claimed')
    args = parser.parse_args()
    swift = args.profile == 'swift'
    args.prompts = PROMPTS[:1] if swift else PROMPTS
    args.windows = 1 if swift else 2
    args.ctx_size = args.ctx_size if args.ctx_size is not None else PROFILES[args.profile]['context_tokens']
    args.atol = args.atol if args.atol is not None else (0.002 if swift else 0.02)
    args.rtol = args.rtol if args.rtol is not None else (0.001 if swift else 0.01)
    if swift and args.ctx_size != 262144:
        parser.error('--profile swift requires --ctx-size 262144; context shrinking is not supported')
    if swift and (args.atol > 0.002 or args.rtol > 0.001):
        parser.error('Swift full-vector gates cannot exceed --atol .002 --rtol .001')
    if swift and args.abort_window != 0:
        parser.error('Swift single-request scope requires --abort-window 0')
    if args.abort_channel < 0:
        parser.error('abort channel must be nonnegative')
    if args.expected_bdf and (len(args.expected_bdf) != 5 or not all(BDF.fullmatch(bdf) for bdf in args.expected_bdf)):
        parser.error('--expected-bdf must specify exactly five canonical BDFs')
    args.cases = args.cases.split(',')
    if not args.cases or len(set(args.cases)) != len(args.cases) or not set(args.cases) <= {'main', 'slow', 'cancel', 'abort'}:
        parser.error('--cases must be a nonempty unique subset of main,slow,cancel,abort')
    if 'cancel' in args.cases and args.predict <= args.cancel_after:
        parser.error('--predict must exceed --cancel-after for the cancellation case')
    args.timeout = args.timeout if args.timeout is not None else args.deadline + 60
    args.abort_timeout = args.abort_timeout if args.abort_timeout is not None else (args.deadline + 60 if swift else 90)
    args.binary = args.binary.absolute()
    args.model = args.model.absolute()
    args.output_dir = args.output_dir.absolute()
    return args


def main():
    args = parse()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    swift = args.profile == 'swift'
    summary = {
        'schema_version': 1, 'status': 'failed', 'selected_cases': args.cases,
        'requested_flash_attention': args.flash_attn == 'on',
        'full_acceptance_selected': not swift and set(args.cases) == {'main', 'slow', 'cancel', 'abort'},
        'preregistered_atol': args.atol, 'preregistered_rtol': args.rtol,
        'default_thresholds_candidate_only': args.atol == 0.02 and args.rtol == 0.01,
        'numerical_qualification': 'not-executed',
        'profile': args.profile,
        'expected_numeric_policy': PROFILES[args.profile]['numeric_policy'],
        'expected_reference_mode': PROFILES[args.profile]['reference_mode'],
        'request_scope': 'single-request-single-window' if swift else 'four-requests-two-relay-windows',
        'requested_context_per_request': args.ctx_size,
        'multiwindow_qualification_exercised': False,
        'package_qualification_confirmed': False,
        'performance_confirmed': False,
        'counter_semantics': {
            'rank_graphs': 'prepared GGML stages accepted directly or in a native replay, not host launch API calls',
            'rank_compute_nodes': 'GGML graph non-view compute nodes, not actual HIP kernel launches',
            'rank_relay_nodes': 'relay kernel nodes accepted directly or in a native replay, not completion evidence',
        },
        'limits': [
            'Full-vector threshold results apply only to the executed fixed teacher trajectory, not model-wide qualification.',
            'These traced acceptance runs are not throughput, latency, memory-peak, or performance confirmation.',
            'Device model, encoded-weight/activation/KV policy, and absence of private fallback rely on the strict public runtime contract; '
            'the report exposes BDFs and real rank work, not independent hardware inventory.',
            'Slow-consumer evidence checks elapsed delay and, only for Qwen3, other-window token commits; '
            'it is not a publication-event trace.',
            'Abort acceptance requires a real nonzero exit before the external bound; a watchdog kill never passes.',
            'Abort-only selections expose no success report metadata and do not independently bind the executed model/profile.',
            'Swift uses only fixed prompt A and one window: slow proves delay, cancellation proves request stop/cleanup; '
            'neither proves other-window progress or survivor isolation, and no full lifecycle/package qualification is claimed.',
        ],
        'cases': {},
    }
    runner = Runner(args)
    save_json(args.output_dir / 'preregistration.json', {
        'atol': args.atol, 'rtol': args.rtol, 'selected_cases': args.cases,
        'profile': args.profile, 'expected_numeric_policy': PROFILES[args.profile]['numeric_policy'],
        'expected_reference_mode': PROFILES[args.profile]['reference_mode'],
        'model': str(args.model), 'context_per_request': args.ctx_size, 'relay_windows': args.windows,
        'predict': args.predict, 'prompts': list(args.prompts), 'expected_bdf': args.expected_bdf,
        'requested_flash_attention': args.flash_attn == 'on',
        'deadline_seconds': args.deadline, 'external_timeout_seconds': args.timeout,
        'abort_timeout_seconds': args.abort_timeout, 'attempts_per_case': 1,
    })
    failure = None
    try:
        if 'main' in args.cases:
            ref_path, reference = runner.run('reference', ['--mode', 'reference', '--windows', '1', '--capture-logits'])
            summary['cases']['reference'] = check_report(reference, args, 'reference', capture=True)
            teacher_path = args.output_dir / 'teacher.tokens'
            with teacher_path.open('x', encoding='ascii') as teacher:
                for request in reference['requests']:
                    teacher.write(' '.join(str(token['emitted_token']) for token in request['tokens']) + '\n')
            relay_path, relay = runner.run('relay-teacher', [
                '--mode', 'relay', '--windows', str(args.windows), '--capture-logits', '--teacher-file', str(teacher_path)])
            summary['cases']['relay-teacher'] = check_report(relay, args, 'relay', capture=True)
            summary['cases']['relay-teacher']['sampled_vs_emitted_differences'] = check_teacher(reference, relay)
            comparison = compare_logits(ref_path, reference, relay_path, relay, args)
            save_json(args.output_dir / 'full-logits-comparison.json', comparison)
            summary['full_logits_comparison'] = comparison
            require(comparison['nonfinite_count'] == 0, 'nonfinite full-vocabulary logits')
            require(comparison['error_count'] == 0, 'full-vocabulary logits exceed preregistered tolerance')
            summary['numerical_qualification'] = ('candidate-only' if args.atol == 0.02 and args.rtol == 0.01
                                                   else 'passed-preregistered-full-vector-thresholds-only')
        if 'slow' in args.cases:
            _, slow = runner.run('slow-consumer', [
                '--mode', 'relay', '--windows', str(args.windows), '--delay-window', '0', '--delay-publish-ms', str(args.delay_ms)])
            evidence = check_report(slow, args, 'relay')
            require(slow['options']['delay_window'] == 0 and slow['options']['delay_publish_ms'] == args.delay_ms,
                    'controlled consumer delay settings changed')
            prompt_lengths = [len(request['prompt_tokens']) for request in slow['requests']]
            require(all(length in (1, 2) for length in prompt_lengths),
                    'slow-consumer prompts were not single text tokens with at most one tokenizer-added token')
            delayed = slow['windows'][0]
            slow_tokens = [token['committed_ns'] for request in slow['requests'] if request['window'] == 0 for token in request['tokens']]
            other_tokens = [token['committed_ns'] for request in slow['requests'] if request['window'] == 1 for token in request['tokens']]
            progress = [stamp for stamp in other_tokens if delayed['first_submit_ns'] < stamp < min(slow_tokens)]
            duration = delayed['last_finish_ns'] - delayed['first_submit_ns']
            first_token_latency = min(slow_tokens) - delayed['first_submit_ns']
            evidence.update(delay_ms=args.delay_ms, delayed_window_duration_ns=duration,
                            first_delayed_token_latency_ns=first_token_latency,
                            tokenized_prompt_lengths=prompt_lengths,
                            other_window_progress_checked=not swift,
                            other_window_commits_before_first_delayed_token=len(progress))
            summary['cases']['slow-consumer'] = evidence
            require(duration >= args.delay_ms * 1_000_000, 'controlled consumer delay not observed')
            if swift:
                require(first_token_latency >= args.delay_ms * 1_000_000,
                        'single-window first token did not observe the controlled consumer delay')
            else:
                require(progress, 'other context made no token progress while delayed context awaited its first token')
        if 'cancel' in args.cases:
            _, cancelled = runner.run('cancel-request', [
                '--mode', 'relay', '--windows', str(args.windows), '--cancel-request', '0', '--cancel-after', str(args.cancel_after)])
            evidence = check_report(cancelled, args, 'relay', cancelled=0)
            require(cancelled['options']['cancel_after'] == args.cancel_after, 'cancellation point changed')
            target = cancelled['requests'][0]
            survivors = cancelled['requests'][1:]
            progressed = [r['id'] for r in survivors if any(t['committed_ns'] > target['finished_ns'] for t in r['tokens'])]
            evidence['survivors_with_commits_after_cancel'] = progressed
            evidence['survivor_isolation_checked'] = not swift
            summary['cases']['cancel-request'] = evidence
            if not swift:
                require(progressed, 'no surviving request progressed after cancellation')
                require(any(r['id'] in progressed and r['window'] == target['window'] for r in survivors),
                        'cancelling one request stopped its entire context window')
        if 'abort' in args.cases:
            runner.run('abort', ['--mode', 'relay', '--windows', str(args.windows),
                                '--abort-window', str(args.abort_window), '--abort-channel', str(args.abort_channel),
                                '--abort-epoch', str(args.abort_epoch)],
                       expected_success=False, timeout=args.abort_timeout)
            diagnostic = pathlib.Path(runner.records[-1]['stderr']).read_text(encoding='utf-8', errors='replace')
            pattern = re.compile(r'^TP5_NATIVE_DRAIN window=(\d+) epoch=(\d+) attempted=(\d+) accepted=(\d+) '
                                 r'unknown=(\d+) completed=(\d+) sealed=(\d+) finished=(\d+) error=(-?\d+)'
                                 r'(?: detail=[^\r\n]*)?$', re.M)
            fields = ('window', 'epoch', 'attempted', 'accepted', 'unknown', 'completed', 'sealed', 'finished', 'error')
            drain_receipts = [dict(zip(fields, map(int, match))) for match in pattern.findall(diagnostic)]
            require(len(drain_receipts) == args.windows and
                    {r['window'] for r in drain_receipts} == set(range(args.windows)),
                    'abort lacks a real native receipt for each window')
            for receipt in drain_receipts:
                mask = receipt['attempted'] | receipt['accepted'] | receipt['unknown']
                require(0 <= mask <= 31 and 0 <= receipt['completed'] <= 31 and receipt['finished'] == 1,
                        'abort native receipt is incomplete')
                require(mask & ~receipt['completed'] == 0, 'abort left accepted or unknown native work in flight')
            require('TP5_FAILURE_CLEANUP reserved_tokens=0 active_requests=0' in diagnostic,
                    'abort did not prove request state cleanup before reservation release')
            summary['cases']['abort'] = {
                'nonzero_bounded_exit': True, 'success_report_absent': True,
                'injection_marker_observed': True, 'caught_exception_exit': True,
                'abort_window': args.abort_window, 'abort_channel': args.abort_channel, 'abort_epoch': args.abort_epoch,
                'elapsed_ns': runner.records[-1]['elapsed_ns'], 'returncode': runner.records[-1]['returncode'],
                'native_drain_receipts': drain_receipts, 'request_reservations_released': True,
            }
        summary['status'] = 'passed-selected-contracts'
        summary['all_contracts_passed'] = summary['full_acceptance_selected']
        summary['multiwindow_qualification_exercised'] = summary['all_contracts_passed']
    except Exception as error:
        failure = error
        summary['error'] = f'{type(error).__name__}: {error}'
        summary['all_contracts_passed'] = False
    finally:
        summary['attempts'] = runner.records
        save_json(args.output_dir / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 1 if failure is not None else 0


if __name__ == '__main__':
    sys.exit(main())
