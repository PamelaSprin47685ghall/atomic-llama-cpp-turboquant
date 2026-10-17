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
cs=importlib.util.spec_from_file_location('necessity_certificate',ROOT/'scripts/tp5-necessity-certificate.py')
c=importlib.util.module_from_spec(cs);sys.modules[cs.name]=c;cs.loader.exec_module(c)

LP_A=((1.0,0.0),(0.0,1.0),(1.0,1.0));LP_B=(2.0,3.0,4.0);LP_C=(5.0,4.0)
LP_Y=(5.0,4.0,0.0);LP_X=(2.0,3.0)


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


class BranchLedgerTests(unittest.TestCase):
    """AGENTS.md §15 algorithm-P branch ledger (P1-P7) checks.

    Constructed small inputs only: no GPU, no real model. The known-bad
    assertions (a)-(e) from the assignment are the test_* methods whose
    names carry the corresponding letter.
    """

    def declared_premises(self,bid):
        return {slot:f'{slot} declared for {bid}' for slot in c.PREMISE_SLOTS}

    def proven_branch(self,bid,claims,**families):
        fam={'layout':'coordinate','communication':'peer-copy',
             'representation':'original','mtp_depth':0}
        fam.update(families)
        return c.BranchRecord(bid,fam['layout'],fam['communication'],
            fam['representation'],fam['mtp_depth'],
            self.declared_premises(bid),tuple(claims))

    def test_p1_enumeration_complete_and_unique(self):
        branches=c.enumerate_branches()
        self.assertEqual(len(branches),5*3*2*4)
        ids=[b.branch_id for b in branches]
        self.assertEqual(len(set(ids)),len(ids))
        for layout in c.LAYOUT_FAMILIES:
            for comm in c.COMMUNICATION_FAMILIES:
                for rep in c.REPRESENTATION_FAMILIES:
                    for h in c.MTP_DEPTHS:
                        self.assertIn(c.make_branch_id(layout,comm,rep,h),ids)

    def test_p2_undeclared_premises_force_unresolved(self):
        r=c.evaluate_branch(c.enumerate_branches()[0])
        self.assertEqual(r['status'],'UNRESOLVED')
        self.assertIsNone(r['Lj_seconds'])
        self.assertEqual(set(r['missing_premise_slots']),set(c.PREMISE_SLOTS))

    def test_p2_partially_declared_premises_still_unresolved(self):
        premises={slot:f'declared {slot}' for slot in c.PREMISE_SLOTS}
        premises['overlap_declaration']=None
        rec=c.BranchRecord('partial','owner','host-relay','original',1,premises,
            (c.ResourceLemmaClaim(4,1,justification='counting'),))
        r=c.evaluate_branch(rec)
        self.assertEqual(r['status'],'UNRESOLVED')
        self.assertIsNone(r['Lj_seconds'])
        self.assertEqual(r['missing_premise_slots'],['overlap_declaration'])

    def test_p2_premises_without_claims_stay_unresolved(self):
        r=c.evaluate_branch(self.proven_branch('noclaims',()))
        self.assertEqual(r['status'],'UNRESOLVED')
        self.assertIsNone(r['Lj_seconds'])

    def test_a_unresolved_branch_forces_global_unresolved(self):
        good=[self.proven_branch('g1',[c.ResourceLemmaClaim(4,1,justification='counting')]),
              self.proven_branch('g2',[c.ResourceLemmaClaim(5,1,justification='counting')])]
        bad=c.enumerate_branches()[11]
        merged=c.merge_lower_bounds([c.evaluate_branch(x) for x in good+[bad]])
        self.assertEqual(merged['status'],'UNRESOLVED')
        self.assertIsNone(merged['global_L_seconds'])
        self.assertEqual(merged['proven_branch_count'],2)
        self.assertIn(bad.branch_id,merged['unresolved_branches'])
        # The min over the proven subset (4.0) must never surface as global L.
        self.assertNotEqual(merged['global_L_seconds'],4.0)

    def test_b_fabricated_Lj_without_premises_rejected(self):
        empty={slot:None for slot in c.PREMISE_SLOTS}
        rec=c.BranchRecord('fab-no-premise','mirrored','host-relay','original',0,
            empty,(c.ResourceLemmaClaim(4,1,justification='counting'),))
        r=c.evaluate_branch(rec)
        self.assertEqual(r['status'],'UNRESOLVED')
        self.assertIsNone(r['Lj_seconds'])

    def test_b_measured_work_or_throughput_bases_rejected(self):
        r1=c.evaluate_branch(self.proven_branch('fab-measured',
            [c.ResourceLemmaClaim(28.6,1,work_basis='measured_shader_time',
                                  justification='counting')]))
        self.assertEqual(r1['status'],'REJECTED')
        self.assertIsNone(r1['Lj_seconds'])
        r2=c.evaluate_branch(self.proven_branch('fab-throughput',
            [c.ResourceLemmaClaim(1e9,390e9,capacity_basis='measured_throughput',
                                  justification='counting')]))
        self.assertEqual(r2['status'],'REJECTED')
        self.assertIsNone(r2['Lj_seconds'])

    def test_b_claim_without_certificate_material_rejected(self):
        r=c.evaluate_branch(self.proven_branch('fab-no-material',
            [c.ResourceLemmaClaim(4,1)]))
        self.assertEqual(r['status'],'REJECTED')
        self.assertIsNone(r['Lj_seconds'])

    def test_b_handwritten_proven_without_certificates_rejected(self):
        forged={'branch_id':'forged','status':'PROVEN','Lj_seconds':9.0,
                'missing_premise_slots':[],'rejection_reasons':[],'certificates':None}
        with self.assertRaises(ValueError):
            c.merge_lower_bounds([forged])

    def test_b_unknown_status_rejected(self):
        with self.assertRaises(ValueError):
            c.merge_lower_bounds([{'branch_id':'x','status':'TRUST_ME','Lj_seconds':1.0}])

    def test_c_incumbent_cannot_pose_as_lower_bound(self):
        claim=c.LpRelaxationClaim('incumbent',LP_A,LP_B,LP_C,LP_X,
            'relaxation covers every legal implementation')
        r=c.evaluate_branch(self.proven_branch('inc',(claim,)))
        self.assertEqual(r['status'],'REJECTED')
        self.assertIsNone(r['Lj_seconds'])
        self.assertTrue(any('Incumbent' in x for x in r['rejection_reasons']))

    def test_c_lp_claim_without_mapping_description_rejected(self):
        claim=c.LpRelaxationClaim('dual',LP_A,LP_B,LP_C,LP_Y,'')
        r=c.evaluate_branch(self.proven_branch('nomap',(claim,)))
        self.assertEqual(r['status'],'REJECTED')

    def test_c_lp_dual_infeasible_certificates_rejected(self):
        with self.assertRaises(ValueError):
            c.verify_lp_dual(LP_A,LP_B,LP_C,(6.0,0.0,0.0))
        with self.assertRaises(ValueError):
            c.verify_lp_dual(LP_A,LP_B,LP_C,(-1.0,0.0,0.0))
        with self.assertRaises(ValueError):
            c.verify_lp_dual(LP_A,LP_B,LP_C,(1.0,1.0))

    def test_d_gap_unproven_when_global_L_invalid(self):
        merged=c.merge_lower_bounds([c.evaluate_branch(c.enumerate_branches()[0])])
        g=c.gap_assessment(merged,0.015,'measured arm','conservative statistic')
        self.assertEqual(g['status'],'UNPROVEN')
        self.assertIsNone(g['U_over_L'])
        self.assertIsNone(g['gap_minus_1'])

    def test_d_gap_unproven_when_u_absent_or_unqualified(self):
        good=[c.evaluate_branch(self.proven_branch('u1',
                [c.ResourceLemmaClaim(4,1,justification='counting')])),
              c.evaluate_branch(self.proven_branch('u2',
                [c.ResourceLemmaClaim(5,1,justification='counting')]))]
        merged=c.merge_lower_bounds(good)
        for u,src,qual in ((None,None,None),(0.015,'',''),(0.015,'source','')):
            g=c.gap_assessment(merged,u,src,qual)
            self.assertEqual(g['status'],'UNPROVEN')
            self.assertIsNone(g['U_over_L'])
            self.assertIsNone(g['gap_minus_1'])

    def test_e_two_fully_declared_branches_merge_to_min(self):
        b1=self.proven_branch('e1',[c.ResourceLemmaClaim(4,1,justification='counting')])
        b2=self.proven_branch('e2',[c.ResourceLemmaClaim(5,1,justification='counting')])
        merged=c.merge_lower_bounds([c.evaluate_branch(b1),c.evaluate_branch(b2)])
        self.assertEqual(merged['status'],'PROVEN')
        self.assertEqual(merged['global_L_seconds'],4.0)
        g=c.gap_assessment(merged,8.0,'measured best qualified arm',
            'conservative statistic of a qualified implementation')
        self.assertEqual(g['status'],'PROVEN_RATIO')
        self.assertAlmostEqual(g['U_over_L'],2.0)
        self.assertAlmostEqual(g['gap_minus_1'],1.0)

    def test_within_branch_claims_combine_with_max_not_sum(self):
        claims=(c.ResourceLemmaClaim(4,1,justification='counting'),
                c.ResourceLemmaClaim(5,1,justification='counting'))
        r=c.evaluate_branch(self.proven_branch('maxrule',claims))
        self.assertEqual(r['status'],'PROVEN')
        self.assertEqual(r['Lj_seconds'],5.0)

    def test_lp_dual_claim_produces_verified_Lj(self):
        claim=c.LpRelaxationClaim('dual',LP_A,LP_B,LP_C,LP_Y,
            'every legal implementation of the branch maps to a feasible point')
        r=c.evaluate_branch(self.proven_branch('lp',(claim,)))
        self.assertEqual(r['status'],'PROVEN')
        self.assertEqual(r['Lj_seconds'],22.0)
        self.assertEqual(r['certificates'][0]['claim_kind'],'lp_dual')

    def test_real_ledger_all_branches_unresolved_and_gap_unproven(self):
        d=c.derive()
        ledger=d['branch_ledger']
        self.assertEqual(ledger['total_branches'],120)
        self.assertTrue(all(x['status']=='UNRESOLVED' for x in ledger['branch_results']))
        self.assertEqual(ledger['global_L']['status'],'UNRESOLVED')
        self.assertIsNone(ledger['global_L']['global_L_seconds'])
        self.assertEqual(ledger['gap']['status'],'UNPROVEN')
        self.assertIsNone(ledger['gap']['U_over_L'])
        self.assertIsNone(ledger['gap']['gap_minus_1'])


if __name__=='__main__':unittest.main(verbosity=2)
