#!/usr/bin/env python3
import importlib.util,math,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('envelope',Path(__file__).resolve().parents[1]/'scripts/tp5-independent-envelope.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
    def test_parallel_q_norm_max_not_sum(self):
        x=dict(inject=1,act=2,q=3,norm=5,lo=7,up=11,pack=13)
        self.assertEqual(m.hc_budget(x,1,0),(1+2+5+7+11+26)/1000)
    def test_hoisted_injects_not_hidden_other_work(self):
        x=dict(inject=10,act=1,q=1,norm=1,lo=1,up=1,pack=1)
        self.assertAlmostEqual(m.hc_budget(x,2,1),.022)
    def test_bytes_replaced_not_added(self):self.assertEqual(m.replace_bytes(1024e6,512e6,512),1)
    def test_overlapping_replacement_rejected(self):
        with self.assertRaises(ValueError):m.replace_bytes(100,101,512)
    def test_nonfinite_inputs(self):
        for x in (math.nan,math.inf,0,-1,True):
            with self.assertRaises(ValueError):m.positive(x)
    def test_invalid_hoist(self):
        with self.assertRaises(ValueError):m.hc_budget({},94,95)
    def test_service_reduction_is_opportunity_not_claim(self):
        x=dict(inject=3,act=1,q=6,norm=30,lo=3,up=4,pack=9);y=dict(x,pack=3)
        self.assertAlmostEqual(m.hc_budget(x)-m.hc_budget(y),94*2*6/1000)
if __name__=='__main__':unittest.main(verbosity=2)
