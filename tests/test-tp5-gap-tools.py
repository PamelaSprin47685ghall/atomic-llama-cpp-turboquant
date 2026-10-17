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

    def test_length_gate_rejects_short_measured_completion(self):
        requests = [{'idx': 0, 'phase': 'measured', 'n': 140}]
        gate = bench.evaluate_length_gate(requests, 160, 240)
        self.assertFalse(gate['passed'])
        self.assertEqual(gate['length_min'], 160)
        self.assertEqual(gate['length_max'], 240)
        self.assertEqual(gate['requests'], [{'idx': 0, 'n': 140, 'in_range': False}])

    def test_length_gate_accepts_in_range_completion(self):
        requests = [{'idx': 0, 'phase': 'measured', 'n': 160},
                    {'idx': 1, 'phase': 'measured', 'n': 240}]
        self.assertTrue(bench.evaluate_length_gate(requests, 160, 240)['passed'])

    def test_length_gate_bounds_are_inclusive(self):
        for n, ok in ((159, False), (160, True), (240, True), (241, False)):
            with self.subTest(n=n):
                requests = [{'idx': 0, 'phase': 'measured', 'n': n}]
                gate = bench.evaluate_length_gate(requests, 160, 240)
                self.assertEqual(gate['passed'], ok)

    def test_length_gate_judges_measured_phase_only(self):
        requests = [{'idx': 0, 'phase': 'warmup', 'n': 140},
                    {'idx': 2, 'phase': 'measured', 'n': 200},
                    {'idx': 3, 'phase': 'guard', 'n': 140}]
        gate = bench.evaluate_length_gate(requests, 160, 240)
        self.assertTrue(gate['passed'])
        self.assertEqual([r['n'] for r in gate['requests']], [200])
        requests[1]['n'] = 140
        self.assertFalse(bench.evaluate_length_gate(requests, 160, 240)['passed'])

    def test_cycle_ledger_reports_not_collected_without_log_evidence(self):
        requests = [{'idx': 0, 'phase': 'measured', 'n': 200,
                     'mtp_cycle_lines': [], 'timings': {}}]
        ledger = bench.extract_cycle_ledger(requests, 1)
        self.assertIn('accepted_prefix_distribution', ledger)
        self.assertIn('cycle_D_V_S_H_K', ledger)
        self.assertEqual(ledger['accepted_prefix_distribution'], 'NOT_COLLECTED')
        self.assertEqual(ledger['cycle_D_V_S_H_K'], 'NOT_COLLECTED')

    def test_cycle_ledger_is_not_applicable_without_mtp(self):
        ledger = bench.extract_cycle_ledger([], 0)
        self.assertEqual(ledger['accepted_prefix_distribution'], 'NOT_APPLICABLE')
        self.assertEqual(ledger['cycle_D_V_S_H_K'], 'NOT_APPLICABLE')

    def test_cycle_ledger_extracts_logged_cycle_lines(self):
        requests = [{'idx': 0, 'phase': 'measured', 'n': 2,
                     'mtp_cycle_lines': [
                         '[tp5-mtp-cycle] draft_us=3000 target_us=25000 catchup_us=1000 handoff_us=0 total_us=29000 accepted_tokens=1 final_tokens=2',
                         '[tp5-mtp-cycle] draft_us=6000 target_us=25000 catchup_us=1000 handoff_us=0 total_us=32000 accepted_tokens=2 final_tokens=4'],
                     'timings': {}}]
        ledger = bench.extract_cycle_ledger(requests, 1)
        cycles = ledger['cycle_D_V_S_H_K']
        self.assertEqual(cycles['cycles_observed'], 2)
        self.assertEqual(cycles['per_cycle'][0]['K'], 1)
        self.assertAlmostEqual(cycles['per_cycle'][1]['D_ms'], 6.0)
        self.assertEqual(ledger['accepted_prefix_distribution']['K_per_cycle'], [1, 2])
        self.assertEqual(ledger['accepted_prefix_distribution']['histogram'], {'1': 1, '2': 1})

    def test_cycle_ledger_falls_back_to_timings_counters(self):
        requests = [{'idx': 0, 'phase': 'measured', 'n': 4,
                     'mtp_cycle_lines': [],
                     'timings': {'draft_n': 6, 'draft_n_accepted': 4}}]
        ledger = bench.extract_cycle_ledger(requests, 1)
        self.assertEqual(ledger['accepted_prefix_distribution']['source'],
                         'timings.draft_n_accepted_per_request')
        self.assertEqual(ledger['accepted_prefix_distribution']['values'], [4])
        self.assertEqual(ledger['cycle_D_V_S_H_K'], 'NOT_COLLECTED')

    def test_audit_minimum_fields_marks_unmechanizable_fields_not_collected(self):
        audited = bench.audit_minimum_fields({'requests': []}, None, None)
        self.assertEqual(audited['fields']['workload_hash'], 'NOT_COLLECTED')
        self.assertEqual(audited['fields']['quality_verdict'], 'NOT_COLLECTED')
        self.assertEqual(audited['fields']['binary_SHA256'], 'NOT_COLLECTED')
        self.assertEqual(audited['fields']['active_rows'], 'NOT_COLLECTED')
        self.assertEqual(audited['fields']['bank_epoch'], 'NOT_COLLECTED')
        self.assertTrue(audited['checklist_covered'])
        self.assertIn('mtp_horizon', audited['fields'])
        self.assertIn('teardown', audited['fields'])

    def test_failed_run_result_carries_ledger_placeholders(self):
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
                 patch.object(bench, 'idle_gate'), \
                 patch.object(bench, 'gpu_state', return_value=[]), \
                 patch('builtins.print'):
                self.assertEqual(bench.main(), 1)
            result = json.loads((out / 'result.json').read_text())
            self.assertEqual(result['rank_stage_ledger']['status'], 'NOT_COLLECTED')
            self.assertIn('reason', result['rank_stage_ledger'])
            self.assertIn('length_gate', result)
            self.assertEqual(result['length_gate']['requests'], 'NOT_COLLECTED')
            self.assertEqual(result['minimum_fields']['fields']['binary_SHA256'], 'NOT_COLLECTED')
            self.assertTrue(result['minimum_fields']['checklist_covered'])

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


class RankStageLedgerTests(unittest.TestCase):
    """GGML_TP5_PROFILE-family line parsing: real formats, malformed inputs, ledger aggregation."""

    RELAY_LINE = ("[tp5-relay-stage] exec=141 stage=10 epoch=55 direct=1 wire=f32 elems=1024 "
                  "ready_first_us=12.5 ready_all_us=56.25 skew_us=3.5 arm_us=1.2 cpu_data_us=0.8 "
                  "generation_us=40.0 total_us=61.0 polls=12 gpu_spin_avg=3.5 gpu_spin_max=9 "
                  "gpu_spin_n=5 ready_rank_us=10.0,20.0,30.0,40.0,56.25")

    def test_relay_stage_line_parses_all_printed_fields(self):
        category, rec = bench.parse_rank_stage_line(self.RELAY_LINE)
        self.assertEqual(category, 'relay_stage')
        self.assertEqual(rec['exec'], 141)
        self.assertEqual(rec['stage'], 10)
        self.assertEqual(rec['epoch'], 55)
        self.assertEqual(rec['direct'], 1)
        self.assertEqual(rec['wire'], 'f32')
        self.assertEqual(rec['total_us'], 61.0)
        self.assertEqual(rec['ready_rank_us'], [10.0, 20.0, 30.0, 40.0, 56.25])

    def test_field_order_does_not_confuse_keys(self):
        category, rec = bench.parse_rank_stage_line(
            '[tp5-relay-stage] stage=10 exec=141 epoch=55 ready_rank_us=1.0,2.0,3.0,4.0,5.0')
        self.assertEqual(category, 'relay_stage')
        self.assertEqual(rec['exec'], 141)
        self.assertEqual(rec['stage'], 10)
        self.assertEqual(rec['ready_rank_us'], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_latebind_stage_line_parses_negative_averages(self):
        category, rec = bench.parse_rank_stage_line(
            '[tp5-latebind-stage] exec=141 stage=10 sidecar_wait_us=85.0 sidecar_data_us=1.1 '
            'y_publish_us=2.2 q_publish_us=0.5 q_spin_avg=-1.0 q_spin_max=0 q_spin_n=0')
        self.assertEqual(category, 'latebind_stage')
        self.assertEqual(rec['sidecar_wait_us'], 85.0)
        self.assertEqual(rec['q_spin_avg'], -1.0)

    def test_exec_summary_keeps_slash_counters_verbatim(self):
        category, rec = bench.parse_rank_stage_line(
            '[tp5-profile] exec=141 decode=1 submits=96 wait_us=1000 wait_n=5 collective=48 '
            'drm_wait=48 creplay=12/3 plan=45/3')
        self.assertEqual(category, 'exec_summary')
        self.assertEqual(rec['creplay'], '12/3')
        self.assertEqual(rec['plan'], '45/3')
        self.assertEqual(rec['decode'], 1)

    def test_definition_and_dependency_lines_parse(self):
        _, d = bench.parse_rank_stage_line(
            '[tp5-linear-definition] rank=2 stages=48 primary_cbs=1 late=94 bank=1 dispatches=50 timing=1')
        self.assertEqual(d['rank'], 2)
        self.assertEqual(d['stages'], 48)
        _, dep = bench.parse_rank_stage_line(
            '[tp5-dependency-ledger] rank=2 stages=48 dispatches=50 compute_epochs=48 '
            'native_dispatches=48 native_epochs=48 ignored_manual_boundaries=1')
        self.assertEqual(dep['rank'], 2)
        self.assertEqual(dep['native_epochs'], 48)

    def test_graph_line_parses_behind_log_prefix(self):
        category, rec = bench.parse_rank_stage_line(
            'info: [tp5-mtp-graph] type=3 reuse=1 definition_uid=0x1a2b3c4d n_reused=5 '
            'ubatch_tokens=2 execution_phase=2 frame_valid=1 frame_role=1 target_capacity=6')
        self.assertEqual(category, 'graph')
        self.assertEqual(rec['type'], 3)
        self.assertEqual(rec['reuse'], 1)
        self.assertEqual(rec['ubatch_tokens'], 2)
        self.assertEqual(rec['target_capacity'], 6)
        self.assertIsInstance(rec['definition_uid'], str)

    def test_missing_or_non_numeric_required_keys_are_malformed(self):
        self.assertIsNone(bench.parse_rank_stage_line('[tp5-relay-stage] epoch=55 direct=1'))
        self.assertIsNone(bench.parse_rank_stage_line('[tp5-relay-stage] exec=abc stage=10'))
        self.assertIsNone(bench.parse_rank_stage_line('[tp5-mtp-graph] type=3'))
        self.assertIsNone(bench.parse_rank_stage_line('[tp5-linear-definition] stages=48'))
        self.assertIsNone(bench.parse_rank_stage_line('unrelated line without tag'))

    def test_bad_rank_list_is_malformed(self):
        self.assertIsNone(bench.parse_rank_stage_line(
            '[tp5-relay-stage] exec=1 stage=2 epoch=3 ready_rank_us=1.0,abc,3.0'))
        self.assertIsNone(bench.parse_rank_stage_line(
            '[tp5-relay-stage] exec=1 stage=2 epoch=3 ready_rank_us=1.0,'))

    def test_truncated_line_keeps_partial_record_without_fabrication(self):
        category, rec = bench.parse_rank_stage_line(
            '[tp5-relay-stage] exec=141 stage=10 epoch=55 direct=1 wire=f32 elems=1024')
        self.assertEqual(category, 'relay_stage')
        self.assertNotIn('ready_rank_us', rec)
        self.assertNotIn('total_us', rec)

    def test_extract_not_collected_reasons_are_mode_specific(self):
        for mode, needle in (('none', '--profile none'), ('cycle', 'GGML_TP5_MTP_PROFILE'),
                             ('kernels', 'GGML_VK_PERF_LOGGER'), ('full', '--profile full')):
            with self.subTest(mode=mode):
                ledger = bench.extract_rank_stage_ledger([], mode)
                self.assertEqual(ledger['status'], 'NOT_COLLECTED')
                self.assertIn(needle, ledger['reason'])
                self.assertIn('active_rows', ledger['field_boundaries'])

    def test_extract_collects_aggregates_and_localizes_slowest_rank(self):
        segments = [
            (0, 'measured', [self.RELAY_LINE,
                             '[tp5-relay-stage] exec=142 stage=11 epoch=56 direct=0 wire=f32 '
                             'total_us=80.0 ready_rank_us=5.0,90.0,6.0,7.0,8.0',
                             '[tp5-relay-stage] epoch=57 direct=1']),
            (1, 'guard', ['[tp5-profile] exec=143 decode=1 creplay=2/1 plan=9/0']),
        ]
        ledger = bench.extract_rank_stage_ledger(segments, 'full')
        self.assertEqual(ledger['status'], 'COLLECTED')
        self.assertEqual(ledger['lines_total'], 4)
        self.assertEqual(ledger['lines_parsed'], 3)
        self.assertEqual(ledger['lines_malformed'], 1)
        self.assertEqual(ledger['counts']['relay_stage'], 2)
        self.assertEqual(ledger['counts']['exec_summary'], 1)
        self.assertEqual(ledger['per_request']['0']['[tp5-relay-stage]'], 2)
        self.assertEqual(ledger['per_request']['0']['phase'], 'measured')
        self.assertEqual(ledger['relay_stage_stats']['total_us']['max'], 80.0)
        self.assertEqual(ledger['slowest_rank']['slowest_rank'], 1)
        self.assertEqual(ledger['slowest_rank']['max_ready_rank_us'][1], 90.0)
        self.assertEqual(ledger['derived']['actual_pipeline_hits']['compute_replay'], ['2/1'])
        self.assertEqual(ledger['malformed_samples'][0]['tag'], '[tp5-relay-stage]')
        self.assertIn('epoch=57', ledger['malformed_samples'][0]['line'])

    def test_full_detail_file_written_only_when_rows_exist(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            out = Path(tmp)
            ledger = bench.extract_rank_stage_ledger([], 'full', out)
            self.assertEqual(ledger['status'], 'NOT_COLLECTED')
            self.assertFalse((out / 'rank-stage.json').exists())
            ledger = bench.extract_rank_stage_ledger(
                [(0, 'measured', [self.RELAY_LINE])], 'full', out)
            self.assertEqual(ledger['full_detail_file'], 'rank-stage.json')
            detail = json.loads((out / 'rank-stage.json').read_text())
            self.assertEqual(len(detail['relay_stage']), 1)
            self.assertEqual(detail['relay_stage'][0]['request_idx'], 0)
            self.assertEqual(detail['relay_stage'][0]['phase'], 'measured')
            self.assertEqual(detail['relay_stage'][0]['ready_rank_us'][4], 56.25)

    def test_audit_derives_minimum_fields_from_collected_ledger(self):
        ledger = bench.extract_rank_stage_ledger(
            [(0, 'measured', [self.RELAY_LINE,
                              'info: [tp5-mtp-graph] type=3 reuse=1 ubatch_tokens=2 target_capacity=6',
                              '[tp5-profile] exec=141 decode=1 creplay=12/3 plan=45/3'])], 'full')
        audited = bench.audit_minimum_fields({'requests': [], 'rank_stage_ledger': ledger}, None, None)
        self.assertEqual(audited['fields']['actual_pipeline_hits']['compute_replay'], ['12/3'])
        self.assertEqual(audited['fields']['active_rows']['values'], [2])
        self.assertIn('ubatch_tokens', audited['fields']['active_rows']['source'])
        self.assertIn('no print site', audited['fields']['active_rows']['source'])
        self.assertEqual(audited['fields']['capacity_rows']['values'], [6])
        self.assertEqual(audited['fields']['bank_epoch'], 'NOT_COLLECTED')
        self.assertIn('bank_epoch', audited['not_collected'])
        self.assertNotIn('active_rows', audited['not_collected'])

    def test_audit_stays_not_collected_without_collected_ledger(self):
        audited = bench.audit_minimum_fields({'requests': []}, None, None)
        for field in ('actual_pipeline_hits', 'active_rows', 'capacity_rows', 'bank_epoch'):
            self.assertEqual(audited['fields'][field], 'NOT_COLLECTED')
        ledger = bench.audit_minimum_fields(
            {'requests': [], 'rank_stage_ledger': {'status': 'NOT_COLLECTED', 'reason': 'x'}}, None, None)
        self.assertEqual(ledger['fields']['active_rows'], 'NOT_COLLECTED')


if __name__ == '__main__':
    unittest.main(verbosity=2)
