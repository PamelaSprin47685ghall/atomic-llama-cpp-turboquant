#!/usr/bin/env python3
"""CPU-only tests for the performance evidence tools; never starts a server."""
import importlib.util
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), ROOT / 'scripts' / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gguf = load('tp5-gguf-metadata')
roof = load('tp5-roofline')
report = load('tp5-gap-report')
bench = load('tp5-gap-bench')


class EvidenceTests(unittest.TestCase):
    def test_busy_gpu_lock_preserves_failure_record(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            fake = root / 'input.bin'
            fake.write_bytes(b'fixture')
            out = root / 'result'
            argv = ['tp5-gap-bench', '--bin', str(fake), '--model', str(fake),
                    '--draft', str(fake), '--out', str(out)]
            with patch('sys.argv', argv), \
                 patch.object(bench, 'open', return_value=io.StringIO(), create=True), \
                 patch.object(bench.fcntl, 'flock', side_effect=BlockingIOError('busy fixture')), \
                 patch.object(bench, 'idle_gate') as idle, \
                 patch.object(bench, 'gpu_state', return_value=[]), \
                 patch('builtins.print'):
                self.assertEqual(bench.main(), 1)
                idle.assert_not_called()
            result = json.loads((out / 'result.json').read_text())
            self.assertEqual(result['status'], 'FAILED')
            self.assertIn('busy fixture', result['error'])
            self.assertNotIn('process_id', result)

    def test_build_gate_accepts_idle_environment(self):
        with patch.object(bench.subprocess, 'run', return_value=SimpleNamespace(returncode=1)):
            bench.build_gate()

    def test_build_gate_rejects_observed_compiler(self):
        def fake_run(cmd, **kwargs):
            return SimpleNamespace(returncode=0 if cmd[-1] == 'cc1plus' else 1)
        with patch.object(bench.subprocess, 'run', side_effect=fake_run):
            with self.assertRaisesRegex(RuntimeError, 'cc1plus'):
                bench.build_gate()

    def test_request_defaults_preserve_historical_contract(self):
        self.assertEqual(bench.request_payload(bench.CANONICAL), {
            'messages': [{'role': 'user', 'content': bench.CANONICAL}],
            'temperature': 0, 'seed': 42, 'max_tokens': 400,
            'cache_prompt': False, 'chat_template_kwargs': {'enable_thinking': False},
            'stream': False})

    def test_system_instruction_keeps_user_question_separate(self):
        payload = bench.request_payload('如何减肥', '请用中文简洁回答，约200个token。', 320)
        self.assertEqual(payload['messages'], [
            {'role': 'system', 'content': '请用中文简洁回答，约200个token。'},
            {'role': 'user', 'content': '如何减肥'}])
        self.assertEqual(payload['max_tokens'], 320)
        self.assertEqual(bench.request_payload(bench.GUARD)['max_tokens'], 400)
        self.assertEqual(len(bench.request_payload(bench.GUARD)['messages']), 1)

    def test_invalid_request_contract_is_rejected(self):
        for prompt, system, cap in [('', None, 400), ('  ', None, 400),
                                    ('x', '', 400), ('x', '  ', 400),
                                    ('x', None, 0), ('x', None, -1),
                                    ('x', None, True), ('x', None, 1.5)]:
            with self.subTest(prompt=prompt, system=system, cap=cap):
                with self.assertRaises(ValueError):
                    bench.request_payload(prompt, system, cap)

    def paired_fixture(self, root):
        for block in range(1, 5):
            for arm, ms in (('A', 100.0), ('B', 80.0)):
                p = root / f'pair-b{block}-{arm}'
                p.mkdir()
                r = {'status': 'MEASURED_NOT_QUALIFIED', 'profile': 'none',
                     'shutdown': {'forced': False, 'returncode': 0},
                     'requests': [{'idx': 0, 'phase': 'measured', 'n': 10, 'issues': [], 'finish': 'stop',
                                   'predicted_ms': ms, 'wall_ms': ms + 10, 'content_sha256': 'fixture-only',
                                   'timings': {'prompt_ms': 10, 'draft_n': 6, 'draft_n_accepted': 4},
                                   'fallback_lines': []}]}
                (p / 'result.json').write_text(json.dumps(r))
                (p / 'manifest.json').write_text(json.dumps({'bin_sha256': 'fixture-only'}))
                (p / 'mapped-libraries.json').write_text('{}')
                (p / 'request-0.log').write_text('')

    def test_paired_ratio_and_public_cycle_accounting(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            self.paired_fixture(root)
            p = report.paired_report(root, 'pair')
            self.assertAlmostEqual(p['ms_per_token']['speedup_geometric'], 1.25)
            self.assertEqual(p['arms_pooled']['B']['tps'], 125)
            self.assertTrue(p['arms_pooled']['B']['h1_count_identity_public_equals_proposed_plus_accepted'])
            self.assertAlmostEqual(p['ms_per_token']['speedup_95pct_CI'][0], 1.25)

    def test_paired_identity_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            self.paired_fixture(root)
            (root / 'pair-b4-B' / 'mapped-libraries.json').write_text('{"changed.so":"different"}')
            with self.assertRaises(ValueError):
                report.paired_report(root, 'pair')

    def tensor(self, name='blk.0.hc_attn_down.weight', shape=None, size=640):
        return {'name': name, 'shape': shape or [32, 10], 'storage_extent_bytes': size, 'raw_type': 30}

    def test_hc_replicated(self):
        self.assertEqual(roof.rank_fraction(self.tensor(), 'mirror', 3, False), 1)

    def test_gdn_qk_duplication(self):
        t = self.tensor('blk.0.attn_qkv.weight', [2560, 10240])
        self.assertEqual(roof.rank_fraction(t, 'split_axis1_vhead_rows', 0, False), .375)

    def test_kv_replication_not_fifth(self):
        t = self.tensor('blk.3.attn_k.weight', [2560, 512])
        self.assertEqual(roof.rank_fraction(t, 'split_axis1_head_rows_replicated', 2, False), 1)
        self.assertEqual(roof.rank_fraction(t, 'split_axis1_head_rows_replicated', 0, False), .5)

    def test_moe_ten_experts(self):
        t = self.tensor('blk.0.ffn_down_exps.weight', [640, 2560, 512])
        self.assertEqual(roof.selected_fraction(t), 10 / 512)

    def test_ple_indexed_not_dense(self):
        t = self.tensor('per_layer_token_embd.weight', [160, 320001536], 28800138240)
        self.assertAlmostEqual(roof.selected_fraction(t) * t['storage_extent_bytes'], 1440)

    def test_unknown_layout_fails(self):
        with self.assertRaises(ValueError):
            roof.rank_fraction(self.tensor(), 'unknown', 0, False)

    def test_bad_rates_fail(self):
        for bandwidth, k in ((0, 1.7), (float('nan'), 1.7), (512, 0), (512, 3)):
            with self.assertRaises(ValueError):
                roof.estimate([], [], {}, bandwidth, k)

    def test_fields_do_not_confuse_names(self):
        x = report.fields('target_us=2000 draft_us=12.5 accepted_tokens=1')
        self.assertEqual(x['target_us'], 2000)
        self.assertEqual(x['draft_us'], 12.5)

    def test_host_timer_overrides_legacy_none_label(self):
        self.assertEqual(report.effective_profile({'profile':'none'}, {'env':{'GGML_DECODE_COST':'1'}}), 'host-cost')
        self.assertEqual(report.effective_profile({'profile':'none'}, {'env':{'GGML_DECODE_COST':'0'}}), 'none')

    def test_summary_does_not_promote_legacy_host_profile(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root=Path(tmp);run=root/'diagnostic';run.mkdir()
            (run/'manifest.json').write_text(json.dumps({'env':{'GGML_DECODE_COST':'1'}}))
            (run/'result.json').write_text(json.dumps({'status':'MEASURED_NOT_QUALIFIED','profile':'none'}))
            row=report.summarize(root)[0]
            self.assertEqual(row['status'],'DIAGNOSTIC_NOT_RANKED')
            self.assertEqual(row['recorded_status'],'MEASURED_NOT_QUALIFIED')

    def test_no_mtp_gpu_capture_is_not_dropped_or_called_request_total(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            run = root / 'no-mtp'
            run.mkdir()
            response = {'usage': {'completion_tokens': 10},
                        'timings': {'predicted_ms': 200, 'prompt_ms': 30}}
            (run / 'request-0.json').write_text(json.dumps(
                {'phase': 'measured', 'response': response, 'wall_ms': 240}))
            (run / 'request-0.log').write_text(
                '[tp5-relay-gpu] rank=0 rows=1 span_us=15000 compute_us=9000 p2_us=4000 tail_compute_us=1000\n')
            rows = report.profile(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['cycles'], 0)
            self.assertIsNone(rows[0]['cycle_component_ms'])
            self.assertIsNone(rows[0]['generation_minus_cycle_components_ms'])
            self.assertIsNone(rows[0]['logged_final_tokens'])
            self.assertEqual(rows[0]['gpu_by_rank_rows']['rank0_rows1']['span_us'], 15000)

    def test_mtp_cycle_accounting_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            run = root / 'mtp'
            run.mkdir()
            response = {'usage': {'completion_tokens': 2},
                        'timings': {'predicted_ms': 30, 'prompt_ms': 10}}
            (run / 'request-0.json').write_text(json.dumps(
                {'phase': 'measured', 'response': response, 'wall_ms': 42}))
            (run / 'request-0.log').write_text(
                '[tp5-mtp-cycle] draft_us=3000 target_us=25000 catchup_us=1000 handoff_us=0 total_us=29000 accepted_tokens=1 final_tokens=2\n')
            row = report.profile(root)[0]
            self.assertEqual(row['generation_minus_cycle_components_ms'], 1)
            self.assertEqual(row['public_tokens_per_observed_cycle'], 2)
            self.assertEqual(row['cycle_component_ms']['target_us'], 25)

    def test_gguf_header_and_offsets(self):
        def s(text):
            raw = text.encode()
            return struct.pack('<Q', len(raw)) + raw
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / 'fixture.gguf'
            data = b'GGUF' + struct.pack('<IQQ', 3, 2, 1)
            data += s('general.alignment') + struct.pack('<II', 4, 32)
            data += s('a.weight') + struct.pack('<IQQIQ', 2, 32, 2, 30, 0)
            data += s('b.weight') + struct.pack('<IQQIQ', 2, 32, 1, 0, 128)
            data += b'\0' * (-len(data) % 32) + b'\0' * 256
            path.write_bytes(data)
            r = gguf.directory(path)
            self.assertEqual([t['storage_extent_bytes'] for t in r['tensors']], [128, 128])
            self.assertEqual(struct.unpack_from('<I', data, r['tensors'][0]['header_type_position'])[0], 30)
            self.assertEqual(struct.unpack_from('<Q', data, r['tensors'][1]['header_offset_position'])[0], 128)
            self.assertEqual(bench.model_stat_identity(path)[str(path)]['bytes'], len(data))

    def test_gguf_bad_magic(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p = Path(tmp) / 'bad.gguf'
            p.write_bytes(b'NOPE')
            with self.assertRaises(ValueError):
                gguf.directory(p)


if __name__ == '__main__':
    unittest.main(verbosity=2)
