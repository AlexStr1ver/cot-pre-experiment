"""Read-only replay and integrity audit; no model inference and no result edits."""
import argparse,collections,copy,hashlib,json,math,sys
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
for name in ['main','replicate','delivered-source','model','baseline','output']:
 parser.add_argument('--'+name,type=Path,required=True)
args=parser.parse_args()
NEW=args.replicate.resolve();OLD=args.main.resolve();DELIVERY=args.delivered_source.resolve()
OUT=args.output.resolve()
if any(OUT.is_relative_to(raw) for raw in [NEW,OLD,DELIVERY,args.model.resolve(),args.baseline.resolve()]):
 raise ValueError('Audit output must be outside all input directories')
OUT.mkdir(parents=True,exist_ok=False)
sys.dont_write_bytecode=True
sys.path.insert(0,str(NEW/'code'))
import paired_experiment as exp
import actions
from resume_compatibility import fingerprint_bridge
from hardware_compatibility import compare_hardware
from transformers import AutoTokenizer

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(p):return {str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file() and '__pycache__' not in f.parts}
originals={str(p):tree(p) for p in [NEW,OLD]}
code_files=tree(NEW/'code')
assert code_files==tree(DELIVERY)
assert (NEW/'exit_status.txt').read_text().strip()=='0'
assert 'Ran 43 tests' in (NEW/'tests.log').read_text() and (NEW/'tests.log').read_text().rstrip().endswith('OK')
for line in (NEW/'code/checksums.sha256').read_text().splitlines():
 h,name=line.split('  ',1);assert sha(NEW/'code'/name)==h
cfg=read(NEW/'results/run_config.json');oldcfg=read(OLD/'results/run_config.json');summary=read(NEW/'results/summary.json')
assert exp.object_sha(cfg['experiment_identity'])==cfg['experiment_fingerprint']
assert cfg['actual_model_dtype']=='torch.float32'
bridge=fingerprint_bridge(oldcfg,cfg);hardware=compare_hardware(cfg['hardware'],oldcfg['hardware'])
manifest=read(NEW/'results/reuse_manifest.json')
assert bridge==manifest['fingerprint_compatibility'] and hardware==manifest['hardware_compatibility']
for n in ['run_config.json','summary.json','completed_artifacts.json']:
 assert (NEW/'results/prior_provenance'/n).read_bytes()==(OLD/'results'/n).read_bytes()
assert manifest['source_run_config_sha256']==sha(OLD/'results/run_config.json')
assert manifest['branches']=={}
assert summary['mode']=='replicate' and summary['completed'] and summary['new_branches']==36 and summary['reused_branches']==0
model=args.model;baseline=args.baseline
tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
plan,candidates,points,evidence=exp.load_evidence(NEW/'code/evidence_phase2fix',baseline,model,tok)
prior,_,oldcomplete=exp.validate_prior(OLD/'results',cfg,candidates,'replicate')
assert len(oldcomplete)==18
_,_,newcomplete=exp.validate_prior(NEW/'results',cfg,candidates,'replicate')
assert len(newcomplete)==36
class ReplayDecoder:
 def __init__(self,events):self.events=events;self.index=-1;self.token_index=0
 def start(self,ids,seed):
  self.index+=1;self.token_index=0;self.event=self.events[self.index]
  assert ids==self.event['input_token_ids'];assert seed==self.event['seed']
 def next_token(self):
  v=self.event['generated_token_ids'][self.token_index];self.token_index+=1;return v
 def close(self):assert self.token_index==len(self.event['generated_token_ids'])
 def stats(self):return copy.deepcopy(self.event['decoder_stats'])

def audit_records(root,mode):
 records=[];branch_hashes=read(root/'results/completed_artifacts.json')['branches_sha256'];jobs=exp.schedule(candidates,mode)
 assert set(branch_hashes)=={exp.job_key(*j) for j in jobs}
 assert {p.stem for p in (root/'results/branches').glob('*.json')}==set(branch_hashes)
 replay_keys=['status','termination_reason','answer_completed','pred_boxed','candidate_boxed','final_text','extra_generated_tokens','total_generated_tokens_with_shared_prefix','forced_summary_succeeded','forced_summary_fallback_planned','forced_summary_fallback_used','executed_summary_restarts','inserted_control_tokens','summary_generated_tokens_total','prefill_tokens_total']
 for c,b,s in jobs:
  key=exp.job_key(c,b,s);p=root/'results/branches'/(key+'.json');r=read(p)
  assert sha(p)==branch_hashes[key];exp.validate_record(r,c,b,s)
  decoder=ReplayDecoder(r['events']);replayed=actions.run_action(c,b,s,decoder,tok)
  assert decoder.index+1==len(r['events'])
  for k in replay_keys:assert replayed[k]==r[k],(key,k)
  for e,original in zip(replayed['events'],r['events']):
   for k in set(e)|set(original):
    if k!='seconds':assert e.get(k)==original.get(k),(key,k)
  records.append((key,r))
 return records
newrecords=audit_records(NEW,'replicate');oldrecords=audit_records(OLD,'main')
fallback_equivalence=[]
combined_map=dict(oldrecords+newrecords)
for key,b in oldrecords+newrecords:
 if not b['forced_summary_fallback_used']:continue
 a=combined_map[key.replace('B_summary','A_continue')]
 assert [e['generated_token_ids'] for e in a['events']]==[e['generated_token_ids'] for e in b['events'][1:]]
 assert [e['input_token_ids'] for e in a['events']]==[e['input_token_ids'] for e in b['events'][1:]]
 assert [e['seed'] for e in a['events']]==[e['seed'] for e in b['events'][1:]]
 assert b['extra_generated_tokens']-a['extra_generated_tokens']==1024
 fallback_equivalence.append({'B_key':key,'continuation_inputs_seeds_and_outputs_identical_to_A':True,'B_minus_A_generated_tokens':1024})
audits=[]
for case in [1,6,8]:
 a=read(NEW/'results/numerical_audits'/f'case_{case:03d}.json')
 c=max((c for c in candidates if c['case']==case),key=lambda c:c['cut'])
 assert (a['case'],a['cut'],a['prefix_token_ids_sha256'])==(case,c['cut'],c['prefix_token_ids_sha256'])
 assert a['passed'] and a['stage']=='finished' and a['chunk_comparison']['passed']
 saved=points[case,c['cut']]
 for mode in ['raw_tail32','asag_style_probe_window']:
  actual=abs(a['primary_signals'][mode][exp.H_KEY]-a['alternative_signals'][mode][exp.H_KEY])/abs(a['primary_signals'][mode][exp.H_KEY])
  chunk=a['chunk_comparison']['signals'][mode]
  assert math.isclose(actual,chunk['H_relative_difference'],rel_tol=1e-8,abs_tol=1e-12)
  assert actual<=.01 and chunk['logit_mean_absolute_difference']<=.05 and chunk['greedy_next_token_agrees'] and chunk['gate_passed']
  cross=abs(a['primary_signals'][mode][exp.H_KEY]-saved['signals'][mode][exp.H_KEY])/abs(saved['signals'][mode][exp.H_KEY])
  assert cross==a['cross_run_phase2_comparison']['relative_H_differences'][mode] and cross<=.01
 assert exp.compare_probes(a['primary_probe'],a['alternative_probe'],.01)==a['probe_comparison']
 assert exp.compare_probes(a['primary_probe'],saved['probe'],.01)==a['cross_run_phase2_comparison']['probe_comparison']
 assert a['resource_passed'] and a['seconds_including_both_chunks_and_probes']<=120 and a['peak_allocated_fraction']<=.85
 audits.append({'case':case,'cut':c['cut'],'passed':True,'seconds':a['seconds_including_both_chunks_and_probes'],'peak_allocated_fraction':a['peak_allocated_fraction'],'max_chunk_H_rel':max(s['H_relative_difference'] for s in a['chunk_comparison']['signals'].values()),'max_chunk_mean_logit_abs':max(s['logit_mean_absolute_difference'] for s in a['chunk_comparison']['signals'].values()),'cross_phase2_H_rel':a['cross_run_phase2_comparison']['relative_H_differences'],'cross_phase2_C_abs':a['cross_run_phase2_comparison']['probe_comparison']['C_absolute_difference']})

def totals(records):
 rs=[r for _,r in records]
 out={'branches':len(rs),'statuses':dict(collections.Counter(r['status'] for r in rs)),'answer_completed':sum(r['answer_completed'] for r in rs),'extra_generated_tokens':sum(r['extra_generated_tokens'] for r in rs),'branch_wall_seconds':sum(r['branch_wall_seconds'] for r in rs),'events':sum(len(r['events']) for r in rs),'summary_restarts':sum(r['executed_summary_restarts'] for r in rs),'max_peak_allocated_gib':max(r['peak_allocated_gib'] for r in rs),'max_peak_reserved_gib':max(r['peak_reserved_gib'] for r in rs),'forced_summary_attempts':sum(r['branch']=='B_summary' for r in rs),'forced_summary_successes':sum(r['forced_summary_succeeded'] is True for r in rs),'forced_summary_fallbacks':[{'key':k,'forced_status':r['events'][0]['status'],'forced_tokens':r['events'][0]['generated_tokens'],'branch_tokens':r['extra_generated_tokens'],'answer_completed':r['answer_completed']} for k,r in records if r['forced_summary_fallback_used']],'incomplete_answers':[{'key':k,'status':r['status'],'tokens':r['extra_generated_tokens'],'budget':r['generation_budget'],'candidate_boxed':r['candidate_boxed'],'last_event_status':r['events'][-1]['status']} for k,r in records if not r['answer_completed']]}
 return out
newstats=totals(newrecords)
assert summary['answer_completed_branches']==newstats['answer_completed']
assert summary['all_completed_branch_generated_tokens']==newstats['extra_generated_tokens']
assert math.isclose(summary['newly_executed_branch_wall_seconds'],newstats['branch_wall_seconds'])
for p in [NEW,OLD]:assert originals[str(p)]==tree(p)
report={'date_beijing':'2026-10-29','engineering_passed':True,'code_files_byte_identical_to_delivery':len(code_files),'autodl_CPU_tests_passed':43,'exit_status':0,'all_old_and_new_record_sha_and_accounting_validated':54,'all_old_and_new_actions_replayed_exact_token_stream_without_inference':54,'new':newstats,'old':totals(oldrecords),'combined':totals(oldrecords+newrecords),'forced_fallback_exact_equivalence':fallback_equivalence,'numerical_audits':audits,'fingerprint_bridge':bridge,'hardware_compatibility':hardware,'environment_selection':read(NEW/'environment/preparation.json')['selection'],'original_results_unchanged':True,'limitations':['Replay verifies state-machine transitions and parsing/accounting of saved tokens; it does not regenerate token probabilities or prove stochastic samples independently.','Raw logits are not serialized; chunk logit gate uses returned numerical audit telemetry.','GPU model/runtime matches plus numeric canaries do not establish same physical GPU or identical performance; compare costs within paired runs and prefer token cost for merged descriptive statistics.','34 of 36 new branches finish with final EOS; two budget-truncated outcomes must remain failures in strict answer correctness.']}
(OUT/'ENGINEERING_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
