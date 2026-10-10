#!/usr/bin/env python3
import importlib.util
import math
from pathlib import Path
import unittest
s=importlib.util.spec_from_file_location('envelope',Path(__file__).resolve().parents[1]/'scripts/tp5-hc-envelope.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class EnvelopeTests(unittest.TestCase):
    def test_act_is_serial_before_both_consumers(self):
        self.assertEqual(m.hc_ms(dict(inject=1,act=3,q=2,norm=4,lo=5,up=6),1,0),.019)
    def test_q_reduction_hidden_by_norm(self):
        a=dict(inject=1,act=3,q=2,norm=4,lo=5,up=6)
        self.assertEqual(m.hc_ms(a),m.hc_ms({**a,'q':1}))
    def test_hiding_does_not_erase_nonhoisted_work(self):
        a=dict(inject=10,act=0,q=0,norm=0,lo=0,up=0)
        self.assertEqual(m.hc_ms(a,94,23),.71)
    def test_invalid_counts(self):
        with self.assertRaises(ValueError):m.hc_ms({},1,2)
    def test_nonfinite_service(self):
        for x in (-1,math.nan,math.inf):
            with self.assertRaises(ValueError):m.hc_ms({'q':x})
if __name__=='__main__':unittest.main(verbosity=2)
