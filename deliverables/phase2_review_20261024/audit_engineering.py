"""Read-only independent arithmetic/provenance checks; never loads checkpoint weights."""
from pathlib import Path
import argparse,hashlib,json,math,sys
from transformers import AutoTokenizer
parser=argparse.ArgumentParser(description=__doc__)
for name in ['result','released-code','baseline','model','output']:
 parser.add_argument('--'+name,type=Path,required=True)
parser.add_argument('--previous-bf16',type=Path)
args=parser.parse_args()
root,release,baseline,model=args.result,args.released_code,args.baseline,args.model
if args.output.exists():
 raise FileExistsError('Refusing to overwrite audit output: '+str(args.output))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
close=lambda a,b:math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
config=read(root/'signals/run_config.json');plan=read(root/'plan.json');summary=read(root/'signals/summary.json')
code_comparison={p.name:sha(p)==sha(release/p.name) for p in sorted((root/'code').iterdir())}
assert all(code_comparison.values())
assert config['script_sha256']==sha(root/'code/collect_phase2.py')
assert config['v1_core_sha256']==sha(root/'code/v1_core.py')
assert config['plan_sha256']==sha(root/'plan.json')
assert config['plan']==plan
assert config['baseline_run_config_sha256']==sha(baseline/'run_config.json')==plan['baseline_run_config_sha256']
assert plan['planner_sha256']==sha(root/'code/prepare_plan.py')
assert all(sha(baseline/n)==h for n,h in plan['baseline_source_sha256'].items())
assert all(sha(model/n)==h for n,h in plan['model_and_tokenizer_file_sha256'].items())
assert hashlib.sha256(''.join(sha(p) for p in sorted((baseline/'cases').glob('*.json'))).encode()).hexdigest()==plan['baseline_case_collection_sha256']
tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
selected={(p['case'],p['cut']):p for p in plan['selected']}
points=[read(p) for p in sorted((root/'signals').glob('case_*.json'))]
assert len(points)==9 and {(p['case'],p['cut']) for p in points}==set(selected)
max_mass=0; times=[]; max_gib=0
for p in points:
 key=(p['case'],p['cut']);e=selected[key]
 assert all(p[k]==v for k,v in e.items())
 record_path=baseline/'cases'/f"{p['case']:03d}.json"
 assert sha(record_path)==e['case_sha256']
 record=read(record_path);r=record['rounds'][0]
 ids=r['input_token_ids']+r['generated_token_ids'][:p['cut']]
 assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==e['prefix_token_ids_sha256']
 assert p['input_tokens']==len(ids)
 # Independently find first legal paragraph edge in the specified interval.
 stops={tokenizer.convert_tokens_to_ids(t) for t in ['</think>','<summary>','</summary>']};stops.add(tokenizer.eos_token_id)
 end=next((i for i,t in enumerate(r['generated_token_ids']) if t in stops),len(r['generated_token_ids']))
 first=next(i for i in range(e['target'],min(end,e['target']+plan['boundary_window'])+1) if tokenizer.decode(r['generated_token_ids'][:i],skip_special_tokens=False).endswith('\n\n'))
 assert first==p['cut']
 assert p['status']=='measured_review_required' and 'error' not in p
 for name,q,extra in [('raw_tail32',32,0),('asag_style_probe_window',12,6)]:
  s=p['signals'][name]
  assert s['queries']==q and s['keys']==len(ids)+extra and s['heads_per_layer']==12 and s['layer_indices_zero_based']==[24,25,26,27]
  assert close(sum(s['raw_entropy_sum_by_layer_nats']),s['entropy_sum_nats'])
  assert close(s['entropy_sum_nats']/math.log(s['keys']),s['H_sum_over_layers_heads_queries_div_log_keys'])
  assert close(s['H_sum_over_layers_heads_queries_div_log_keys']/q,s['H_query_mean_sum_over_layers_heads_div_log_keys'])
  assert close(s['H_sum_over_layers_heads_queries_div_log_keys']/(q*48),s['H_mean_over_layers_heads_queries_div_log_keys'])
  assert 0<s['H_mean_over_layers_heads_queries_div_log_keys']<1
  max_mass=max(max_mass,s['max_attention_row_mass_error'])
 probe=p['probe'];probs=probe['selected_token_probabilities'];token_ids=probe['generated_token_ids']
 assert len(probs)==len(token_ids)==probe['generated_tokens']
 assert all(math.isfinite(x) and 0<=x<=1 for x in probs)
 assert close(sum(probs)/len(probs),probe['C_mean_selected_token_probability'])
 assert tokenizer.decode(token_ids,skip_special_tokens=False)==probe['generated_text']
 assert tokenizer.encode(probe['inserted_suffix'],add_special_tokens=False)==probe['inserted_suffix_token_ids']==config['probe_instruction_ids']
 assert probe['prefill_input_tokens']==len(ids)+6
 assert probe['status']=='closed_box_review_required' and probe['answer']
 d=probe['format_diagnostic'];idx=d['strict_content_token_indices']
 if idx:assert close(sum(probs[i] for i in idx)/len(idx),d['strict_content_C'])
 assert close(probs[-1],probe['closure_token_probability'])
 point_seconds=sum(s['replay_seconds_including_prefill_and_entropy_reduction'] for s in p['signals'].values())+probe['seconds_including_independent_prefill']
 assert close(point_seconds,p['measurement_seconds_excluding_canary_audit'])
 times.append(point_seconds);max_gib=max(max_gib,p['peak_allocated_gib'])
 group=[x for x in points if x['case']==p['case']];ref=min(group,key=lambda x:x['cut'])
 assert p['delta_reference_cut']==ref['cut']
 for n in ['raw_tail32','asag_style_probe_window']:
  a,b=p['signals'][n],ref['signals'][n]
  delta=p['relative_changes'][n]
  assert close(delta['normalized_H'],(a['H_sum_over_layers_heads_queries_div_log_keys']-b['H_sum_over_layers_heads_queries_div_log_keys'])/b['H_sum_over_layers_heads_queries_div_log_keys'])
  assert close(delta['raw_entropy_sum'],(a['entropy_sum_nats']-b['entropy_sum_nats'])/b['entropy_sum_nats'])
canary=next(p for p in points if p['is_canary'])
assert sum(bool(p['is_canary']) for p in points)==1
assert (canary['case'],canary['cut'])==(1,4103)
alt=canary['canary_alternative_probe'];primary=canary['probe']
assert alt['generated_token_ids']==primary['generated_token_ids']
assert close(max(abs(a-b) for a,b in zip(alt['selected_token_probabilities'],primary['selected_token_probabilities'])),canary['canary_probe_comparison']['max_aligned_token_probability_difference'])
assert close(abs(alt['C_mean_selected_token_probability']-primary['C_mean_selected_token_probability']),canary['canary_probe_comparison']['C_absolute_difference'])
assert all(canary[k]['passed'] for k in ['canary_chunk_comparison','canary_probe_comparison','canary_resource_gate'])
for n,c in canary['canary_chunk_comparison']['signals'].items():
 a=canary['signals'][n]['H_sum_over_layers_heads_queries_div_log_keys'];b=canary['canary_alternative_signals'][n]['H_sum_over_layers_heads_queries_div_log_keys']
 assert close(c['H_relative_difference'],abs(a-b)/abs(a))
 assert c['H_relative_difference']<=.01 and c['logit_mean_absolute_difference']<=.05
assert close(sum(times),summary['measurement_seconds_sum_excluding_canary_audit'])
assert close(sum(p.get('canary_audit_seconds',0) for p in points),summary['canary_audit_seconds_sum'])
assert summary['closed_nonempty_probes']==9 and summary['executed_positions']==summary['planned_positions']==len(points)
assert config['actual_model_dtype']=='torch.float32' and config['dtype']=='float32'
assert config['arithmetic_settings']['float32_matmul_precision']=='highest'
assert not config['arithmetic_settings']['cuda_matmul_allow_tf32'] and not config['arithmetic_settings']['cudnn_allow_tf32']
result={'all_assertions_passed':True,'returned_code_identical_to_release':code_comparison,'actual_AutoDL_tests':{'planner':7,'collector':11,'total':18,'runner_tests_not_reexecuted_on_AutoDL':4},'prefixes_validated_against_original_native20':9,'all_nine_point_arithmetic_consistent':True,'plan_byte_identical_to_old_failed_BF16_plan':((root/'plan.json').read_bytes()==(args.previous_bf16/'plan.json').read_bytes()) if args.previous_bf16 else None,'maximum_attention_row_mass_error':max_mass,'max_peak_allocated_gib':max_gib,'max_peak_reserved_gib':max(p['peak_reserved_gib'] for p in points),'measurement_seconds':sum(times),'audit_seconds':summary['canary_audit_seconds_sum'],'model_load_seconds':summary['model_load_seconds'],'mean_seconds_per_point':sum(times)/9,'canary_input_tokens':canary['input_tokens'],'actual_longest_input_tokens':max(p['input_tokens'] for p in points),'actual_longest_case':max(points,key=lambda p:p['input_tokens'])['case'],'canary':{'H_logit_checks':canary['canary_chunk_comparison'],'C_check':canary['canary_probe_comparison'],'resource_check':canary['canary_resource_gate']},'limitations':['Only one real-model canary has a second chunk size; largest sequence is another question','No full long-sequence real-model attention reference, no stored raw attention matrices/logit vectors','No BF16-vs-FP32 answer/probability agreement calibration across nine points','Local safetensors revision not independently verified by content hash','Measurements are offline on three previously explored long trajectories, not nine independent questions','No actual summary/reset action, predictive-policy evaluation or total-cost comparison']}
with args.output.open('x',encoding='utf-8') as f:
 f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
