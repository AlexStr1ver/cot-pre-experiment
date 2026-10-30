"""Independent descriptive analysis of 54 branches; no inference and no raw edits."""
from pathlib import Path
import argparse, csv, json, hashlib, re
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--main',type=Path,default=Path('/workspace/infty-dev/results_20261024/paired_v3_kt2eZTc2'))
parser.add_argument('--replicate',type=Path,default=Path('/workspace/infty-dev/results_20261029/paired_v3_rsiFEcc2'))
parser.add_argument('--output',type=Path,required=True,help='New directory outside both raw runs; existing directories are refused')
args=parser.parse_args()
ROOTS=[args.main/'results',args.replicate/'results'];OUT=args.output.resolve()
if any(OUT.is_relative_to(raw.resolve()) for raw in (args.main,args.replicate)):
 raise ValueError('Analysis output must not be inside a raw run')
OUT.mkdir(parents=True,exist_ok=False)
REF={1:'6',6:'28',8:'-1'}
records={}; grades=[]
for root in ROOTS:
 for p in sorted((root/'branches').glob('*.json')):
  r=json.loads(p.read_text()); raw=r['pred_boxed']; value=raw
  reason='Exact final integer comparison'
  if raw is not None and re.fullmatch(r'[+-]?\d+\^\\+circ', raw):
   value=raw.split('^')[0];reason='Manual angle-unit equivalence for case6; final-answer-only grade'
   assert r['case']==6
  correct=bool(r['answer_completed']) and value==REF[r['case']]
  if not r['answer_completed']:reason='Budget incomplete: no accepted final answer; retained as unsuccessful'
  r['review_correct']=correct
  key=(r['case'],r['cut'],r['replicate_seed'],r['branch']);assert key not in records;records[key]=r
  grades.append({'key':p.stem,'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'case':r['case'],'cut':r['cut'],'seed':r['replicate_seed'],'branch':r['branch'],'raw_box':raw,'reference':REF[r['case']],'answer_completed':r['answer_completed'],'correct':correct,'reason':reason,'note':'Final-answer correctness is not proof correctness.'})
assert len(records)==54
signals={}
for p in sorted((ROOTS[1].parent/'code/evidence_phase2fix/signals').glob('case_*.json')):
 r=json.loads(p.read_text());signals[(r['case'],r['cut'])]=r
pairs=[]
for case,cut,seed in sorted({k[:3]for k in records}):
 a,b=[records[(case,cut,seed,x)]for x in ['A_continue','B_summary']]
 s=signals[(case,cut)]
 label='both_correct'if a['review_correct']and b['review_correct']else 'A_only_correct'if a['review_correct']else 'B_only_correct'if b['review_correct']else 'both_wrong'
 pairs.append({'case':case,'cut':cut,'target':1024 if cut<1500 else 2048 if cut<3000 else 4096,'seed':seed,'A_answer':a['pred_boxed'],'B_answer':b['pred_boxed'],'A_correct':a['review_correct'],'B_correct':b['review_correct'],'outcome':label,'A_tokens':a['extra_generated_tokens'],'B_tokens':b['extra_generated_tokens'],'B_minus_A_tokens':b['extra_generated_tokens']-a['extra_generated_tokens'],'A_wall':a['branch_wall_seconds'],'B_wall':b['branch_wall_seconds'],'B_forced_summary_succeeded':b['forced_summary_succeeded'],'B_fallback_used':b['forced_summary_fallback_used'],'B_status':b['status'],'C_answer':s['probe']['answer'],'C_mean':s['probe']['C_mean_selected_token_probability'],'H_relative':s['relative_changes']['asag_style_probe_window']['normalized_H'],'H_raw_relative':s['relative_changes']['asag_style_probe_window']['raw_entropy_sum']})
def aggregate(rows):
 ac=sum(x['A_correct']for x in rows);bc=sum(x['B_correct']for x in rows)
 ta=sum(x['A_tokens']for x in rows);tb=sum(x['B_tokens']for x in rows)
 return {'pairs':len(rows),'A_correct':ac,'B_correct':bc,'A_tokens':ta,'B_tokens':tb,'token_saving_fraction':1-tb/ta,'A_branch_wall_sum':sum(x['A_wall']for x in rows),'B_branch_wall_sum':sum(x['B_wall']for x in rows),'outcomes':{k:sum(x['outcome']==k for x in rows)for k in ['both_correct','A_only_correct','B_only_correct','both_wrong']},'B_forced_summary_successes':sum(x['B_forced_summary_succeeded']is True for x in rows),'B_fallbacks':sum(x['B_fallback_used']for x in rows),'B_budget_truncations':sum(x['B_status']=='generation_budget_limit'for x in rows)}
summary={'all':aggregate(pairs),'new36':aggregate([p for p in pairs if p['seed']in[2027,2028]]),'by_seed':{str(seed):aggregate([p for p in pairs if p['seed']==seed])for seed in [2026,2027,2028]},'by_state':{f'{case}/{cut}':aggregate([p for p in pairs if (p['case'],p['cut'])==(case,cut)])for case,cut in sorted(signals)},'by_case':{str(case):aggregate([p for p in pairs if p['case']==case])for case in [1,6,8]},'by_target':{str(target):aggregate([p for p in pairs if p['target']==target])for target in [1024,2048,4096]},'inferential_scope':'Three previously explored development questions, nine fixed prefixes, three seeds, 27 correlated pairs. No population noninferiority, significance or online speedup claim. Timing aggregate crosses two execution sessions, retain run-level reporting. C values are reused deterministic probes, not independently replicated.'}
for name,data in [('summary.json',summary),('manual_grades.json',grades),('paired_records.json',pairs)]:
 (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
with (OUT/'paired_records.csv').open('w',newline='',encoding='utf-8-sig')as f:
 w=csv.DictWriter(f,fieldnames=list(pairs[0]),lineterminator="\n");w.writeheader();w.writerows(pairs)
for group in ['by_case','by_target','by_state','by_seed']:
 rows=[{'group':k,**{name:v for name,v in value.items() if not isinstance(v,dict)},**value['outcomes']}for k,value in summary[group].items()]
 with (OUT/(group+'.csv')).open('w',newline='',encoding='utf-8-sig')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
print(json.dumps(summary,ensure_ascii=False,indent=2))
