#!/usr/bin/env python3
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('match_model', ROOT/'scripts/tp5-match-model.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class MatchTests(unittest.TestCase):
    def test_finite_length_excludes_prefill_token(self):
        model = {'first_ms': {'cost': 220}, 'steady_ms': {'cost': 14}}
        self.assertEqual(m.predict(model, 176)['predicted_ms'], 220 + 174*14)

    def test_smallest_supported_population(self):
        self.assertEqual(m.predict({'first_ms': {'c': 10}, 'steady_ms': {'c': 2}}, 3)['predicted_ms'], 12)

    def test_bad_lengths(self):
        for n in (True, 1, 2, -1, 3.2):
            with self.assertRaises(ValueError): m.predict({}, n)

    def test_nonfinite_negative_model(self):
        for x in (math.nan, math.inf, -1, True):
            with self.assertRaises(ValueError):
                m.predict({'first_ms': {'c': x}, 'steady_ms': {'c': 2}}, 4)

    def fixture(self, root, extra_rebuild=False, omit_sample=False):
        p = root/'request-0.json'
        p.write_text(json.dumps({'phase':'measured','response':{'usage':{'completion_tokens':4},
                     'timings':{'predicted_n':4,'predicted_ms':0.081}}}))
        lines=['[sample-cost] ctx=x idx=30 sync_us=99 select_us=1 total_us=100']
        for i in range(3):
            reuse = 0 if i == 0 or extra_rebuild else 1
            lines += [f'[decode-cost] scope=ubatch ctx=x type=0 rows=1 reuse={reuse} setup_us=1 graph_us=2 alloc_us=3 input_us=1 submit_us=10 finish_us=1 total_us=18',
                      '[decode-cost] scope=decode ctx=x type=0 rows=1 reuse=-1 total_us=20',
                      '[sample-cost] ctx=x idx=0 sync_us=2 select_us=3 total_us=5']
        if omit_sample: lines.pop()
        p.with_suffix('.log').write_text('\n'.join(lines))
        return p

    def test_nested_scopes_not_double_counted(self):
        with tempfile.TemporaryDirectory() as d:
            r=m.request_cost(self.fixture(Path(d)))
            self.assertAlmostEqual(sum(r['first_ms'].values()), .025)
            self.assertAlmostEqual(r['unexplained_server_ms'], .006)
            self.assertAlmostEqual(m.predict(r, 4)['predicted_ms'], .075)

    def test_extra_rebuild_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):m.request_cost(self.fixture(Path(d),extra_rebuild=True))

    def test_missing_samples_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):m.request_cost(self.fixture(Path(d),omit_sample=True))

    def test_zero_total_rejected(self):
        with self.assertRaises(ValueError):m.predict({'first_ms':{'c':0},'steady_ms':{'c':0}},4)

    def test_empty_validation_rejected(self):
        with self.assertRaises(ValueError):m.validate({}, [])

    def test_counterfactual_uses_A_plus_micro_only(self):
        a={'identity':{'env':{'GGML_TP5_GREEDY_HOST':'0'}}, 'first_ms':{'sample_select':.6,'other':200},
           'steady_ms':{'sample_select':.5,'other':14},'limitations':'Test.'}
        micro={'scope':'CPU_SAMPLING_MICRO_NOT_MODEL','vocabulary':248320,'ordinary_us':480,'unique_us':80}
        b=m.sampler_counterfactual(a,micro)
        self.assertAlmostEqual(m.predict(a,176)['predicted_ms']-m.predict(b,176)['predicted_ms'],175*.4)
        self.assertFalse(b['candidate_measurements_used'])
        self.assertEqual(a['identity']['env']['GGML_TP5_GREEDY_HOST'],'0')
        self.assertEqual(b['identity']['env']['GGML_TP5_GREEDY_HOST'],'1')

    def test_counterfactual_refuses_negative_fudge(self):
        a={'identity':{'env':{'GGML_TP5_GREEDY_HOST':'0'}}, 'first_ms':{'sample_select':.1},
           'steady_ms':{'sample_select':.2},'limitations':'Test.'}
        micro={'scope':'CPU_SAMPLING_MICRO_NOT_MODEL','vocabulary':248320,'ordinary_us':480,'unique_us':80}
        with self.assertRaises(ValueError):m.sampler_counterfactual(a,micro)

if __name__=='__main__': unittest.main(verbosity=2)
