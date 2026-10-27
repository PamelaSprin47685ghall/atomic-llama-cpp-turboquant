#!/usr/bin/env python3
"""Verify rendered deltas and stop accounting with the actual installed model runner."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import threading


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def reject_constant(value):
    raise ValueError(f'non-finite JSON value: {value}')


def execute(binary, arguments, output, name, success=True):
    command = [str(binary), *arguments]
    events = []
    first_text_while_running = None
    with (output / f'{name}.jsonl').open('wb') as stdout, (output / f'{name}.log').open('wb') as stderr:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr) as process:
            timer = threading.Timer(600, process.kill)
            timer.daemon = True
            timer.start()
            try:
                for line in process.stdout:
                    stdout.write(line)
                    if not line.strip():
                        continue
                    event = json.loads(line.decode('utf-8'), parse_constant=reject_constant)
                    events.append(event)
                    if event['event'] == 'text' and first_text_while_running is None:
                        first_text_while_running = process.poll() is None
                returned = process.wait()
            finally:
                timer.cancel()
                if process.poll() is None:
                    process.kill()
                    process.wait()
    log = (output / f'{name}.log').read_bytes()
    errors = [event for event in events if event['event'] == 'error']
    if success:
        require(returned == 0, f'{name}: {log[-2000:].decode("utf-8", "replace")}')
        require(not errors, f'{name}: successful execution reported an error')
    else:
        require(returned != 0 and errors, f'{name}: invalid options succeeded or lacked a JSON error')
        require(not any(event['event'] in ('config', 'batch', 'result') for event in events),
                f'{name}: invalid options started model execution')
    offload = re.search(rb'offloaded (\d+)/(\d+) layers to GPU', log)
    record = {'command': command, 'returncode': returned,
              'gpu_layers_observed': int(offload.group(1)) if offload else 0,
              'first_text_observed_before_process_exit': first_text_while_running}
    return events, record


def inspect(events, streaming, request_count, ignore_eog=True):
    configs = [event for event in events if event['event'] == 'config']
    require(len(configs) == 1 and configs[0]['stream'] is streaming, 'stream option was not consumed')
    results = [event for event in events if event['event'] == 'result']
    require([value['request_id'] for value in results] == list(range(request_count)), 'request results were lost or reordered')
    active = set()
    finished = set()
    released = set()
    samples = {request: [] for request in range(request_count)}
    rows = {request: [] for request in range(request_count)}
    feedback = {request: [] for request in range(request_count)}
    text = {request: '' for request in range(request_count)}
    terminal = {}
    sample_counts = {request: 0 for request in range(request_count)}
    for event in events:
        kind = event['event']
        if kind == 'admit':
            request = event['request_id']
            require(request not in active and request not in released, 'request was admitted twice')
            active.add(request)
        elif kind == 'batch':
            require(event['decode_return'] == 0 and event['rows'], 'batch did not execute successfully')
            for row in event['rows']:
                request = row['request_id']
                require(request in active and request not in finished, 'request was decoded after stopping')
                rows[request].append(row['position'])
                if event['phase'] == 'generation':
                    feedback[request].append(row['token'])
            for sample in event['samples']:
                request = sample['request_id']
                require(request not in finished, 'finished request sampled again')
                samples[request].append(sample)
                if ignore_eog or not sample['eog']:
                    sample_counts[request] += 1
                if sample['finished']:
                    finished.add(request)
        elif kind == 'text':
            request = event['request_id']
            require(streaming and request in active and request not in terminal, 'text escaped its request lifetime')
            require(event['offset_bytes'] == len(text[request].encode('utf-8')), 'text byte offsets have a gap or overlap')
            require(event['generated_tokens'] == sample_counts[request], 'text token count lacks actual samples')
            require(event['finished'] is (request in finished), 'text termination disagrees with consumed A samples')
            text[request] += event['text']
            if event['finished']:
                terminal[request] = event
            else:
                require(event['text'] and event['finish_reason'] is None and event['stop_index'] is None,
                        'nonterminal text has no progress or fabricated completion')
        elif kind == 'release':
            request = event['request_id']
            require(request in active and request in finished, 'request was released before A output consumption')
            require(not streaming or request in terminal, 'sequence was released before its terminal text event')
            active.remove(request)
            released.add(request)
    require(not active and len(released) == request_count, 'completion retained a resident request')
    for result in results:
        request = result['request_id']
        actual = samples[request]
        generated = [sample['token'] for sample in actual if ignore_eog or not sample['eog']]
        require(generated == result['tokens'], 'result omitted a non-EOG sampled token, including a stopping token')
        require(actual and actual[-1]['finished'], 'result lacks an actual terminal sample')
        require(rows[request] == list(range(result['decoded_rows'])), 'positions were lost or repeated')
        require(result['decoded_rows'] == result['prompt_tokens'] + len(actual) - 1, 'decode accounting includes an unsubmitted token')
        require(feedback[request] == [sample['token'] for sample in actual[:-1]], 'generation did not feed back its samples')
        if streaming:
            require(text[request] == result['text'], 'concatenated UTF-8 deltas differ from the final result')
            require(request in terminal, 'result has no terminal delta')
            require(terminal[request]['finish_reason'] == result['finish_reason'] and
                    terminal[request]['stop_index'] == result['stop_index'], 'terminal delta changed its reason or stop')
        if result['finish_reason'] == 'stop':
            require(result['stop_index'] is not None, 'stop result has no matched pattern identity')
        else:
            require(result['stop_index'] is None, 'non-stop result has a pattern identity')
    return results


def equal_execution(first, second, label):
    def relevant(events):
        return [event for event in events if event['event'] in ('batch', 'result', 'admit', 'release')]
    require(relevant(first) == relevant(second), f'{label}: rendering changed A rows, samples, results, or residency')


def choose_overlap(events):
    deltas = [event['text'] for event in events if event['event'] == 'text' and event['request_id'] == 0 and event['text']]
    text = ''.join(deltas)
    boundary = 0
    for chunk in deltas[:-1]:
        boundary += len(chunk)
        for suffix_size in range(2, 6):
            end = boundary + suffix_size
            if end >= len(text):
                continue
            begin = max(1, boundary - 3)
            short = text[boundary:end]
            long = text[begin:end]
            if begin < boundary and text.find(short) == boundary and text.find(long) == begin:
                return [short, long], {'request_id': 0, 'crossed_character_boundary': boundary,
                                       'long_stop_begin': begin, 'stop_end': end}
    raise AssertionError('actual generation did not yield a unique stop spanning two text deltas')


def expected_stop(text, stops):
    encoded = text.encode('utf-8')
    matches = []
    for index, stop in enumerate(stops):
        pattern = stop.encode('utf-8')
        begin = encoded.find(pattern)
        if begin >= 0:
            matches.append((begin + len(pattern), -len(pattern), index, begin))
    if not matches:
        return None
    end, negative_size, index, begin = min(matches)
    return {'text': encoded[:begin].decode('utf-8'), 'index': index, 'end': end}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--gpu-layers', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require(args.binary.is_file() and args.model.is_file(), 'a real binary and model are required')
    output_parent = args.output_dir.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    args.output_dir = Path(tempfile.mkdtemp(prefix='run-', dir=output_parent))
    prompts = args.output_dir / 'prompts.txt'
    prompts.write_text(
        'Continue a list of facts about the ocean. Water covers much of Earth. Waves travel across the surface. 1.\n'
        'Continue a list of facts about forests. Trees have leaves. Roots grow in soil. Birds build nests. 1.\n'
        'Continue a list of facts about mountains. High peaks can hold snow. Rivers flow through valleys. 1.\n', encoding='utf-8')
    common = ['--model', str(args.model.resolve()), '--prompts', str(prompts.resolve()), '--predict', '24',
              '--ctx-size', '256', '--total-kv', '1024', '--batch-size', '16', '--ubatch-size', '8',
              '--threads', '2', '--gpu-layers', str(args.gpu_layers), '--ignore-eog', '--json', '--no-phase-memory']
    records = {}
    runs = {}
    results = {}

    def run_case(name, parallel, streaming, stops=()):
        arguments = [*common, '--parallel', str(parallel)]
        if streaming:
            arguments.append('--stream')
        for stop in stops:
            arguments.extend(['--stop', stop])
        events, record = execute(args.binary, arguments, args.output_dir, name)
        if args.gpu_layers > 0:
            require(record['gpu_layers_observed'] > 0, f'{name}: GPU request executed on CPU only')
        records[name] = record
        runs[name] = events
        results[name] = inspect(events, streaming, 3)
        return events

    run_case('serial-buffered', 1, False)
    stream = run_case('serial-stream', 1, True)
    equal_execution(runs['serial-buffered'], stream, 'serial stream switch')
    require(records['serial-stream']['first_text_observed_before_process_exit'], 'text was not observed during model execution')
    stops, selection = choose_overlap(stream)
    run_case('stop-buffered', 1, False, stops)
    run_case('stop-stream', 1, True, stops)
    equal_execution(runs['stop-buffered'], runs['stop-stream'], 'stop stream switch')
    matched = []
    for baseline, stopped in zip(results['serial-stream'], results['stop-stream']):
        expected = expected_stop(baseline['text'], stops)
        if expected is None:
            require(stopped == baseline, 'nonmatching stop changed a request')
            continue
        require(stopped['finish_reason'] == 'stop' and stopped['stop_index'] == expected['index'],
                'first-completed or longest-at-same-end stop policy changed')
        require(stopped['text'] == expected['text'], 'a stop or its held prefix leaked into the result')
        count = len(stopped['tokens'])
        require(0 < count < len(baseline['tokens']), 'stop did not shorten actual model execution')
        require(stopped['tokens'] == baseline['tokens'][:count], 'stopping changed preceding model samples')
        baseline_deltas = [event for event in stream if event['event'] == 'text' and event['request_id'] == baseline['request_id']]
        completing = next(event for event in baseline_deltas
                          if event['offset_bytes'] + len(event['text'].encode('utf-8')) >= expected['end'])
        require(count == completing['generated_tokens'], 'stopping token was omitted or a later token was generated')
        matched.append({'request_id': baseline['request_id'], 'stop_index': expected['index'], 'generated_tokens': count})
    require(matched and matched[0]['request_id'] == 0 and matched[0]['stop_index'] == 1,
            'fixture did not exercise a cross-delta stop winning over a shorter same-end pattern')
    run_case('parallel-buffered', 3, False)
    run_case('parallel-stream', 3, True)
    equal_execution(runs['parallel-buffered'], runs['parallel-stream'], 'parallel stream switch')

    invalid = [('empty-stop', ['--stop', '']), ('oversize-stop', ['--stop', 'a' * 4097]),
               ('too-many-stops', [part for _ in range(33) for part in ('--stop', 'x')])]
    for name, extra in invalid:
        events, record = execute(args.binary, ['--model', str(args.output_dir / 'absent-model.gguf'), '--json', *extra],
                                 args.output_dir, name, success=False)
        message = next(event['message'] for event in events if event['event'] == 'error')
        require('model' not in message.lower(), f'{name}: failure came from model loading instead of stop validation')
        records[name] = record
    missing_json = subprocess.run([str(args.binary), '--model', str(args.output_dir / 'absent-model.gguf'), '--stream'],
                                  capture_output=True, timeout=30)
    (args.output_dir / 'missing-json.log').write_bytes(missing_json.stderr)
    require(missing_json.returncode != 0 and b'--stream requires --json' in missing_json.stderr,
            'stream without a machine-readable output contract was accepted')
    events, record = execute(args.binary,
                             ['--model', str(args.output_dir / 'absent-model.gguf'), '--json', '--stop', '--help', '--parallel', '0'],
                             args.output_dir, 'help-as-stop-value', success=False)
    require(any('--parallel' in event.get('message', '') for event in events), 'stop value was parsed as a help flag')
    records['help-as-stop-value'] = record
    json_value = subprocess.run([str(args.binary), '--model', str(args.output_dir / 'absent-model.gguf'), '--stop', '--json', '--stream'],
                                capture_output=True, timeout=30)
    (args.output_dir / 'json-as-stop-value.log').write_bytes(json_value.stderr)
    require(json_value.returncode != 0 and not json_value.stdout and b'--stream requires --json' in json_value.stderr,
            'stop value changed the JSON output mode')

    summary = {'status': 'passed', 'runs': records, 'model': str(args.model.resolve()),
               'output_dir': str(args.output_dir),
               'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
               'stop_patterns': stops, 'selection': selection, 'matched': matched,
               'same_geometry_stream_switch_equal': ['serial', 'stop', 'parallel'],
               'text_deltas_equal_results': True, 'stop_token_accounted_without_further_decode': True,
               'limits': ['Stop controls use one resident request so early completion does not change concurrent batch geometry.',
                          'Greedy tokens are not compared between serial and parallel batch shapes.',
                          'This suite uses ignore-eog; B tests cover finish-time UTF-8 replacement and stop matching.',
                          'The CLI output contract does not qualify HTTP cancellation, transport backpressure, or model quality.']}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': summary['status'], 'output_dir': summary['output_dir'],
                      'runs': len(records), 'matched': matched}, ensure_ascii=True, allow_nan=False))


if __name__ == '__main__':
    main()
