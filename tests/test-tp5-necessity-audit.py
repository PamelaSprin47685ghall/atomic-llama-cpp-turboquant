#!/usr/bin/env python3
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('necessity_test',ROOT/'scripts/tp5-necessity-audit.py')
m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)


class NecessityTests(unittest.TestCase):
    def resource(self,work,cap,premise='declared mathematical class'):
        return m.NecessaryResource(work,'proved_for_declared_class',cap,'upper_bound_for_declared_class',premise)

    def test_correct_bound_direction(self):
        observed=15;optimal=min(15,10)
        self.assertLessEqual(optimal,observed)
        self.assertFalse(optimal>=observed)

    def test_measured_kernel_cannot_be_necessary_work(self):
        x=m.NecessaryResource(28.6,'measured_shader_time',1,'upper_bound_for_declared_class','same fixture')
        with self.assertRaises(ValueError):x.latency_lower()

    def test_current_dispatch_count_not_necessary(self):
        x=m.NecessaryResource(1510,'current_dispatch_count',1,'upper_bound_for_declared_class','all legal rewrites')
        with self.assertRaises(ValueError):x.latency_lower()

    def test_observed_bandwidth_not_capacity_upper(self):
        x=m.NecessaryResource(1e9,'proved_for_declared_class',390e9,'measured_throughput','read model')
        with self.assertRaises(ValueError):x.latency_lower()

    def test_premise_must_be_explicit(self):
        with self.assertRaises(ValueError):self.resource(1,2,'').latency_lower()

    def test_nonfinite(self):
        for x in (math.inf,math.nan,True,-1):
            with self.assertRaises(ValueError):m.finite(x)

    def test_zero_capacity_rejected(self):
        with self.assertRaises(ValueError):self.resource(10,0).latency_lower()

    def test_distinct_resource_bounds_use_max(self):
        self.assertEqual(m.joint_lower([self.resource(4,1),self.resource(5,1)]),5)

    def test_mixed_classes_rejected(self):
        with self.assertRaises(ValueError):m.joint_lower([self.resource(4,1,'one token'),self.resource(5,1,'MTP cycle')])

    def test_empty_lower_bound_rejected(self):
        with self.assertRaises(ValueError):m.joint_lower([])

    def test_cache_counting_lemma(self):
        self.assertEqual(m.mandatory_io(100,30),70)

    def test_all_required_bytes_can_be_resident(self):
        self.assertEqual(m.mandatory_io(100,120),0)

    def test_prefetch_witness_respects_channels_and_dependencies(self):
        e=m.exact_counterexamples()['future_weight_prefetch']
        work=e['weight_bytes_by_layer_and_rank']
        ready=[]
        for rank in range(2):
            reads=e[f'rank{rank+1}_reads']
            self.assertGreaterEqual(reads[1][0],reads[0][1])
            for layer,(start,end) in enumerate(reads):self.assertEqual(end-start,work[layer][rank])
            ready.append([r[1] for r in reads])
        layer1=max(ready[0][0],ready[1][0]);layer2=max(layer1,ready[0][1],ready[1][1])
        self.assertEqual([layer1,layer2],e['layer_compute_completion_times'])
        self.assertEqual(layer2,e['constructive_prefetched_finish_time'])
        self.assertLess(layer2,e['old_sum_stage_max'])

    def test_hiding_local_work_not_proof_of_fixed_wait(self):
        self.assertEqual(max(73.46,20.47),max(73.46,9.32))
        self.assertLess(max(10,9.32),max(73.46,9.32))

    def test_cycle_denominator_not_one_public_token(self):
        self.assertEqual(m.mtp_cost(20,2),10)

    def test_zero_production_rejected(self):
        with self.assertRaises(ValueError):m.mtp_cost(20,0)

    def fixture(self,root):
        def tensor(name,shape,type_id,payload):
            return {'name':name,'shape':shape,'raw_type':type_id,'storage_extent_bytes':payload+128}
        records=[{'tensors':[
            tensor('blk.0.hc.weight',[32,10],8,340),
            tensor('blk.0.attn.weight',[5,10],0,200),
            tensor('blk.0.ffn_gate_exps.weight',[32,2,512],8,34816),
            tensor('token_embd.weight',[32,8],0,1024),
            tensor('per_layer_token_embd.weight',[16,64],0,4096)]}]
        kinds=['mirror','split_axis1_rows','split_axis1_rows','mirror','cpu']
        layouts={'tensor_layouts':[{'name':t['name'],'kind':k} for t,k in zip(records[0]['tensors'],kinds)]}
        meta=root/'metadata.json';layout=root/'layout.json'
        meta.write_text(json.dumps(records));layout.write_text(json.dumps(layouts))
        return meta,layout

    def test_inventory_does_not_count_padding_as_required_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta,layout=self.fixture(Path(tmp));v=m.inspect_inventory(meta,layout,1)
            self.assertEqual(v['active_unique_original_encoded_bytes'],1220)
            self.assertEqual(v['tensors_counted'],3)

    def test_layout_amplification_separate_from_unique_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta,layout=self.fixture(Path(tmp));v=m.inspect_inventory(meta,layout,1)
            self.assertEqual(v['active_current_plan_total_copy_bytes'],2580)
            self.assertEqual(v['active_current_plan_copy_bytes_by_rank'],[516]*5)

    def test_audit_does_not_certify_a_global_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta,layout=self.fixture(Path(tmp));v=m.run(meta,layout,1)
            self.assertIsNone(v['global_necessary_tps_upper_bound'])
            self.assertFalse(v['global_optimality_gap_certified'])
            self.assertTrue(all(not x['eligible_as_unrestricted_model_bound'] for x in v['conditional_io_sensitivity']))

    def test_duplicate_tensor_names_require_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta,layout=self.fixture(Path(tmp));v=json.loads(meta.read_text())
            v[0]['tensors'].append(v[0]['tensors'][0]);meta.write_text(json.dumps(v))
            with self.assertRaises(ValueError):m.inspect_inventory(meta,layout,1)

    def test_more_allowed_implementations_cannot_raise_the_optimum_time(self):
        restricted=[15,14];expanded=restricted+[10]
        self.assertLessEqual(min(expanded),min(restricted))


if __name__=='__main__':unittest.main(verbosity=2)
