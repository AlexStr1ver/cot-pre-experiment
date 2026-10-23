"""CPU tests for the frozen schedule, audit gates, accounting and immutable reuse."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

import paired_experiment as exp
from actions import run_action
from test_actions import Tokenizer, ScriptedDecoder


def fixture_candidates():
    tok=Tokenizer();base=[tok.bos_token_id]+tok.encode('Question Q\nAssistant:')
    prefix=base+tok.encode('<think>\nold reasoning')
    return [dict(case=c,cut=cut,target=t,unique_id=f'fixture/{c}',input_token_ids=prefix,
                 base_token_ids=base,prefix_token_ids_sha256=exp.ids_sha(prefix))
            for c,cuts in [(1,[1041,2104,4103]),(6,[1055,2092,4203]),(8,[1084,2055,4173])]
            for t,cut in zip([1024,2048,4096],cuts)]


class FixtureDecoder(ScriptedDecoder):
    """No inference: deliberate wrong answers verify expansion never selects winners."""
    def __init__(self,model,tok,device,**kwargs):
        self.tok=tok;super().__init__(tok,[])
    def start(self,ids,seed):
        text=self.tok.decode(ids)
        answer='saved fact</summary>' if text.endswith('\n</think><summary>\n') else '</think>\\boxed{999}<eos>'
        self.outputs.append(self.tok.encode(answer))
        super().start(ids,seed)


class DummyModel:
    def to(self,device):return self
    def eval(self):return self
    def parameters(self):return iter([torch.zeros(1)])


class ExperimentTests(unittest.TestCase):
    def test_schedule_is_exact_and_predeclared_no_label_selection(self):
        cs=fixture_candidates()
        smoke,main,rep=[exp.schedule(cs,m) for m in ['smoke','main','replicate']]
        self.assertEqual([len(smoke),len(main),len(rep)],[2,18,36])
        self.assertEqual([(x[0]['case'],x[0]['cut'],x[1],x[2]) for x in smoke],
            [(1,4103,'A_continue',2026),(1,4103,'B_summary',2026)])
        self.assertEqual(len({exp.job_key(*x) for x in main}),18)
        self.assertEqual(sum(16384-c['cut'] for c,b,s in main),251092)
        self.assertEqual({s for c,b,s in rep},{2027,2028})
        for index in range(0,18,2):
            self.assertEqual(main[index][0],main[index+1][0])
            self.assertEqual({main[index][1],main[index+1][1]},{'A_continue','B_summary'})

    def test_record_accounting_rejects_forward_counts_and_corruption(self):
        c=fixture_candidates()[0];tok=Tokenizer()
        r=run_action(c,'B_summary',2026,ScriptedDecoder(tok,['fact</summary>','</think>\\boxed{6}<eos>']),tok)
        self.assertTrue(exp.validate_record(r,c,'B_summary',2026))
        self.assertGreater(r['extra_generated_tokens'],sum(e['decoder_stats']['forward_generated_tokens'] for e in r['events']))
        for key,val in [('extra_generated_tokens',r['extra_generated_tokens']-1),('generation_budget',1),
                        ('record_finished',False),('prefill_tokens_total',0),('prefix_token_ids_sha256','changed')]:
            bad=copy.deepcopy(r);bad[key]=val
            with self.subTest(key=key),self.assertRaises(ValueError):exp.validate_record(bad,c,'B_summary',2026)

    def test_actual_audit_gate_rejects_old_signal_mismatch_and_saves_diagnostics(self):
        signals={n:{exp.H_KEY:100.} for n in ['raw_tail32','asag_style_probe_window']}
        logits={n:torch.zeros(1,5) for n in signals}
        probe={'selected_token_probabilities':[.9], 'generated_token_ids':[1],
               'C_mean_selected_token_probability':.9,'status':'closed_box_review_required'}
        saved={'signals':signals,'probe':probe}
        c=fixture_candidates()[0]
        with tempfile.TemporaryDirectory() as d, patch.object(exp,'measure_modes',return_value=(signals,logits)),patch.object(exp,'probe_asag_style',return_value=probe):
            result=exp.audit_point(None,None,c,saved,'cpu',Path(d)/'good.json')
            self.assertTrue(result['passed'])
            bad=copy.deepcopy(saved);bad['signals']['raw_tail32'][exp.H_KEY]=120.
            with self.assertRaises(RuntimeError):exp.audit_point(None,None,c,bad,'cpu',Path(d)/'bad.json')
            self.assertFalse(exp.read(Path(d)/'bad.json')['passed'])
            bad=copy.deepcopy(saved);bad['probe']['generated_token_ids']=[2]
            with self.assertRaises(RuntimeError):exp.audit_point(None,None,c,bad,'cpu',Path(d)/'badC.json')

    def test_integer_grading_is_conservative(self):
        self.assertTrue(exp.integer_grade('-1','-1'))
        self.assertFalse(exp.integer_grade('56','28'))
        self.assertIsNone(exp.integer_grade(r'28^\circ','28'))
        self.assertIsNone(exp.integer_grade(None,'6'))

    def test_streaming_weight_fingerprint_is_content_sensitive(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.safetensors';path.write_bytes(b'fixture weight A')
            first=exp.weight_fingerprint(Path(d));path.write_bytes(b'fixture weight B')
            second=exp.weight_fingerprint(Path(d))
            self.assertEqual(first[0]['bytes'],second[0]['bytes'])
            self.assertNotEqual(first[0]['sha256'],second[0]['sha256'])

    def fixture_setup(self,root):
        phase=root/'phase2';phase.mkdir();(phase/'plan.json').write_text('{}');(phase/'evidence_manifest.json').write_text('{}')
        baseline=root/'baseline';(baseline/'cases').mkdir(parents=True)
        for case,answer in [(1,'6'),(6,'28'),(8,'-1')]:
            (baseline/'cases'/f'{case:03d}.json').write_text(json.dumps({'reference_answer':answer}))
        cs=fixture_candidates()
        points={(c['case'],c['cut']):{'probe':{'answer':'999','C_mean_selected_token_probability':.8},
              'relative_changes':{'asag_style_probe_window':{'normalized_H':0.}}} for c in cs}
        plan={'cohort_cases':[1,6,8],'model_and_tokenizer_file_sha256':{}}
        return phase,baseline,cs,points,plan

    def run_fixture(self,root,mode,source=None,fail_audit=False):
        if not hasattr(self,'fixture'):
            self.fixture=self.fixture_setup(root)
        phase,baseline,cs,points,plan=self.fixture
        args=['--model',str(root/'model'),'--baseline',str(baseline),'--phase2-input',str(phase),
              '--output',str(root/mode),'--device','cpu','--mode',mode]
        if source:args+=['--resume-from' if mode=='main' else '--prior-run',str(source)]
        def audit(*a,**k):
            if fail_audit:raise RuntimeError('fixture numeric failure')
            return {'passed':True}
        with patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()), \
             patch('transformers.AutoModelForCausalLM.from_pretrained',return_value=DummyModel()), \
             patch.object(exp,'load_evidence',return_value=(plan,cs,points,{})), \
             patch.object(exp,'weight_fingerprint',return_value=[{'name':'fixture','bytes':1,'sha256':'fixture'}]), \
             patch.object(exp,'audit_point',side_effect=audit), \
             patch('decoder.CachedDecoder',FixtureDecoder),patch.object(exp.signal,'signal'),contextlib.redirect_stdout(io.StringIO()):
            exp.main(args)
        return root/mode

    def test_smoke_main_replicate_end_to_end_reuses_only_finished_exact_records(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);smoke=self.run_fixture(root,'smoke')
            self.assertEqual(exp.read(smoke/'summary.json')['completed_branches'],2)
            smoke_bytes={p.name:p.read_bytes() for p in (smoke/'branches').glob('*.json')}
            main=self.run_fixture(root,'main',smoke)
            summary=exp.read(main/'summary.json')
            self.assertTrue(summary['completed']);self.assertEqual(summary['new_branches'],16)
            self.assertEqual(summary['reused_branches'],2);self.assertFalse(summary['actual_GPU_execution_this_run'])
            for name,content in smoke_bytes.items():self.assertEqual((main/'branches'/name).read_bytes(),content)
            review=exp.read(main/'paired_review.json');self.assertEqual(len(review['pairs']),9)
            # Every fixture answer is wrong. All pairs must still execute.
            self.assertTrue(all(not p['both_strict_complete_correct'] for p in review['pairs']))
            replicate=self.run_fixture(root,'replicate',main)
            self.assertEqual(exp.read(replicate/'summary.json')['new_branches'],36)
            self.assertTrue(exp.read(replicate/'summary.json')['completed'])
            config=exp.read(main/'run_config.json')
            branch=next((smoke/'branches').glob('*.json'));record=exp.read(branch);record['pred_boxed']='6';exp.dump(branch,record)
            with self.assertRaisesRegex(ValueError,'checksum'):
                exp.validate_prior(smoke,config,self.fixture[2],'main')

    def test_failed_audit_never_starts_actions_and_preserves_failed_summary(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaisesRegex(RuntimeError,'fixture numeric failure'):
                self.run_fixture(root,'smoke',fail_audit=True)
            summary=exp.read(root/'smoke/summary.json')
            self.assertFalse(summary['completed']);self.assertEqual(summary['stage'],'failed')
            self.assertEqual(list((root/'smoke/branches').glob('*.json')),[])

    def test_prior_fingerprint_or_hardware_mismatch_blocks_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);smoke=self.run_fixture(root,'smoke')
            cfg=exp.read(smoke/'run_config.json')
            for key in ['fingerprint','hardware']:
                bad=copy.deepcopy(cfg)
                if key=='fingerprint':bad['experiment_fingerprint']='changed'
                else:bad['hardware']['name']='different GPU'
                with self.subTest(key=key),self.assertRaises(ValueError):
                    exp.validate_prior(smoke,bad,self.fixture[2],'main')


if __name__=='__main__':unittest.main(verbosity=2)
