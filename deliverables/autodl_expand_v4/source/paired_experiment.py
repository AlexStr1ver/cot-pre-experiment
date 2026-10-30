#!/usr/bin/env python3
"""Frozen-prefix FP32 paired-action pilot. Not an online timing controller."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import shutil
import signal
import sys
import time

from collect_phase2 import (dump, verify_plan, ids_sha, precision_settings, hardware_info,
    measure_modes, compare_chunks, probe_asag_style, compare_probes, H_KEY, ASAG_SUFFIX)
from v1_core import synchronize
from hardware_compatibility import compare_hardware
from resume_compatibility import CURRENT_CODE_FILES, fingerprint_bridge

PROTOCOL = {
    'name': 'paired_actions_fp32_v3', 'total_generated_budget_including_shared_cut': 16384,
    'forced_summary_budget': 1024, 'temperature': .7, 'top_p': .95,
    'dtype': 'float32', 'attention_backend': 'eager', 'tf32': False,
    'prefill_chunk': 128, 'alternative_chunk': 64, 'heartbeat_tokens': 256,
    'max_relative_H_difference': .01, 'max_mean_absolute_logit_difference': .05,
    'max_token_probability_difference': .01, 'probe_tokens': 128,
    'max_canary_seconds': 120., 'max_canary_allocated_fraction': .85,
    'forced_failure_policy': 'charge_failed_summary_then_continue_original_prefix',
    'native_invalid_summary_policy': 'retain_as_model_format_outcome_no_silent_retry',
    'generation_stop': 'EOS_or_total_generated_budget; valid_native_summary_resets_without_segment_limit',
    'grading': 'Strict completed-EOS result and budget-truncated final-region boxed candidate reported separately; noninteger boxes require review',
    'seed_mapping': 'actions.request_seed based on case,cut,replicate,role,index; no heartbeat reseeding',
    'sampling_implementation': 'decoder.py torch multinomial private generator; descending nucleus with shifted removal mask; no top_k',
}
CODE_FILES = list(CURRENT_CODE_FILES)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def weight_fingerprint(model_dir):
    paths = sorted(model_dir.glob('*.safetensors'))
    if not paths:
        raise ValueError('No local safetensors weights found')
    return [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': file_sha(p)} for p in paths]


def load_evidence(phase2, baseline, model_dir, tokenizer):
    manifest = read(phase2/'evidence_manifest.json')
    for name, expected in manifest['files_sha256'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or file_sha(phase2/rel) != expected:
            raise ValueError('Phase2 evidence checksum mismatch: '+name)
    plan, cfg, summary = read(phase2/'plan.json'), read(phase2/'signals/run_config.json'), read(phase2/'signals/summary.json')
    if not summary.get('completed') or summary.get('executed_positions') != 9 or not summary.get('canary_passed'):
        raise ValueError('Phase2 must be the completed nine-point FP32 run')
    if cfg['dtype'] != 'float32' or cfg['actual_model_dtype'] != 'torch.float32' or cfg['plan'] != plan or cfg['plan_sha256'] != file_sha(phase2/'plan.json'):
        raise ValueError('Phase2 dtype/plan provenance mismatch')
    if plan['cohort_cases'] != [1,6,8] or plan['targets'] != [1024,2048,4096] or len(plan['selected']) != 9:
        raise ValueError('This pilot is frozen to the existing three-by-three cohort')
    for name, expected in plan['model_and_tokenizer_file_sha256'].items():
        if Path(name).name != name or file_sha(model_dir/name) != expected:
            raise ValueError('Model/tokenizer metadata changed: '+name)
    collection = ''.join(file_sha(p) for p in sorted((baseline/'cases').glob('*.json')))
    if hashlib.sha256(collection.encode()).hexdigest() != plan['baseline_case_collection_sha256']:
        raise ValueError('Original native20 case collection changed')
    prepared = verify_plan(plan, baseline, tokenizer)
    for entry in prepared:
        record = read(baseline/'cases'/f"{entry['case']:03d}.json")
        original = record['rounds'][0]['input_token_ids']
        think = tokenizer.convert_tokens_to_ids('<think>')
        if original.count(think) != 1:
            raise ValueError('Expected exactly one initial thinking marker')
        base = original[:original.index(think)]
        if base + tokenizer.encode('<think>\n', add_special_tokens=False) != original:
            raise ValueError('Original prompt is not base + exact thinking prefix')
        template = tokenizer.apply_chat_template([{'role':'user','content':record['problem']}], tokenize=False, add_generation_prompt=True)
        if tokenizer.encode(template, add_special_tokens=False) != base:
            raise ValueError('Original base differs from checkpoint user-only chat template')
        if base[0] != tokenizer.bos_token_id or base.count(tokenizer.bos_token_id) != 1:
            raise ValueError('Invalid or duplicated BOS in reset base')
        entry['base_token_ids'] = base
    points = [read(p) for p in sorted((phase2/'signals').glob('case_*.json'))]
    if len(points) != 9 or {(p['case'],p['cut']) for p in points} != {(p['case'],p['cut']) for p in prepared}:
        raise ValueError('Missing/duplicate phase2 measurements')
    for point in points:
        entry = next(p for p in prepared if (p['case'],p['cut'])==(point['case'],point['cut']))
        if point['status'] != 'measured_review_required' or point['prefix_token_ids_sha256'] != entry['prefix_token_ids_sha256']:
            raise ValueError('Phase2 point/prefix identity mismatch')
    return plan, prepared, {(p['case'],p['cut']): p for p in points}, manifest


def schedule(candidates, mode):
    first = next(p for p in candidates if (p['case'],p['cut']) == (1,4103))
    order = [first] + [p for p in sorted(candidates,key=lambda x:(x['case'],x['cut'])) if p is not first]
    seeds = [2027,2028] if mode == 'replicate' else [2026]
    if mode == 'smoke':
        order = order[:1]
    jobs = []
    for rep_index, seed in enumerate(seeds):
        for index, candidate in enumerate(order):
            branches = ['A_continue','B_summary'] if (index+rep_index)%2 == 0 else ['B_summary','A_continue']
            for branch in branches:
                jobs.append((candidate, branch, seed))
    return jobs


def job_key(candidate, branch, seed):
    return f"case_{candidate['case']:03d}_cut_{candidate['cut']:05d}_seed_{seed}_{branch}"


def validate_record(record, candidate, branch, seed):
    for key in ['case','cut','unique_id','prefix_token_ids_sha256']:
        if record.get(key) != candidate[key]:
            raise ValueError('Branch identity mismatch: '+key)
    if record.get('branch') != branch or record.get('replicate_seed') != seed:
        raise ValueError('Branch/replicate mismatch')
    if not record.get('record_finished') or record.get('status') == 'execution_error':
        raise ValueError('Cannot reuse incomplete or execution-failed branch')
    budget = PROTOCOL['total_generated_budget_including_shared_cut']-candidate['cut']
    events = record.get('events', [])
    count = sum(len(e['generated_token_ids']) for e in events)
    if not events or record.get('generation_budget') != budget or record.get('extra_generated_tokens') != count or not 0 <= count <= budget:
        raise ValueError('Branch generated-token budget mismatch')
    if record.get('total_generated_tokens_with_shared_prefix') != candidate['cut']+count:
        raise ValueError('Shared prefix budget mismatch')
    for event in events:
        if event['generated_tokens'] != len(event['generated_token_ids']) or not math.isfinite(event['seconds']) or event['seconds'] < 0:
            raise ValueError('Invalid event accounting')
    if branch == 'A_continue' and events[0]['input_token_ids'] != candidate['input_token_ids']:
        raise ValueError('A initial prefix changed')
    expected_prefill = sum(len(e['input_token_ids']) for e in events)
    if record['prefill_tokens_total'] != expected_prefill:
        raise ValueError('Prefill accounting mismatch')
    if not math.isclose(record['model_call_seconds'],sum(e['seconds'] for e in events),rel_tol=1e-8,abs_tol=1e-8):
        raise ValueError('Model event timing mismatch')
    return True


def validate_prior(source, config, candidates, requested_mode):
    prior, summary = read(source/'run_config.json'), read(source/'summary.json')
    artifacts = read(source/'completed_artifacts.json')
    if artifacts.get('experiment_fingerprint') != prior.get('experiment_fingerprint'):
        raise ValueError('Prior artifact manifest identity differs')
    fingerprint_bridge(prior, config)
    compatibility = compare_hardware(config['hardware'], prior['hardware'])
    if compatibility['compatible'] is not True:
        raise ValueError('Prior GPU/runtime is incompatible: ' + json.dumps(compatibility, sort_keys=True))
    if not summary.get('numerical_audit_passed'):
        raise ValueError('Prior numerical audit has not passed')
    mode = prior['mode']
    if requested_mode == 'main':
        if mode not in ['smoke','main'] or (mode == 'smoke' and not summary.get('completed')):
            raise ValueError('Main needs a completed smoke or compatible interrupted/completed main')
    elif requested_mode == 'replicate':
        if mode not in ['main','replicate'] or (mode == 'main' and not summary.get('completed')):
            raise ValueError('Replicates require completed main, or a compatible replicate resumption')
    else:
        raise ValueError('Smoke cannot reuse prior branches')
    prior_jobs = schedule(candidates, mode)
    complete = {}
    for candidate,branch,seed in prior_jobs:
        key = job_key(candidate,branch,seed); path = source/'branches'/(key+'.json')
        if not path.exists():
            continue
        record = read(path)
        if not record.get('record_finished') or record.get('status') == 'execution_error':
            continue
        if artifacts.get('branches_sha256',{}).get(key) != file_sha(path):
            raise ValueError('Prior completed branch checksum mismatch: '+key)
        validate_record(record,candidate,branch,seed)
        complete[key] = (path,record)
    if summary.get('completed') and len(complete) != len(prior_jobs):
        raise ValueError('Prior completed summary disagrees with valid completed branches')
    # Even an interrupted main must already contain the fixed smoke pair.
    if requested_mode == 'main':
        for candidate,branch,seed in schedule(candidates,'smoke'):
            if job_key(candidate,branch,seed) not in complete:
                raise ValueError('Prior fixed smoke pair is incomplete; return its error logs before expansion')
    return prior, summary, complete


def audit_point(model, tokenizer, candidate, saved, device, path):
    import torch
    ids = candidate['input_token_ids']
    result = {'case':candidate['case'],'cut':candidate['cut'],'prefix_token_ids_sha256':candidate['prefix_token_ids_sha256'],'passed':False,'stage':'started'}
    dump(path,result)
    if device == 'cuda': torch.cuda.reset_peak_memory_stats()
    synchronize(device); started = time.perf_counter()
    primary_chunk, alternate_chunk = PROTOCOL['prefill_chunk'],PROTOCOL['alternative_chunk']
    h_limit,logit_limit,p_limit = PROTOCOL['max_relative_H_difference'],PROTOCOL['max_mean_absolute_logit_difference'],PROTOCOL['max_token_probability_difference']
    primary, logits = measure_modes(model,tokenizer,ids,primary_chunk,device)
    result.update(primary_signals=primary,stage='primary_H_saved');dump(path,result)
    alternate, alternate_logits = measure_modes(model,tokenizer,ids,alternate_chunk,device)
    result['chunk_comparison'] = compare_chunks(primary,logits,alternate,alternate_logits,h_limit,logit_limit)
    result['alternative_signals'] = alternate
    del logits, alternate_logits
    result['stage']='H_audit_saved';dump(path,result)
    if not result['chunk_comparison']['passed']:
        raise RuntimeError('FP32 H/logit chunk gate failed; no action expansion')
    probe = probe_asag_style(model,tokenizer,ids,PROTOCOL['probe_tokens'],primary_chunk,device)
    result.update(primary_probe=probe,stage='primary_C_saved');dump(path,result)
    alternative_probe = probe_asag_style(model,tokenizer,ids,PROTOCOL['probe_tokens'],alternate_chunk,device)
    result['alternative_probe'] = alternative_probe
    result['probe_comparison'] = compare_probes(probe,alternative_probe,p_limit)
    cross = {name: abs(primary[name][H_KEY]-saved['signals'][name][H_KEY])/abs(saved['signals'][name][H_KEY]) for name in primary}
    result['cross_run_phase2_comparison'] = {'relative_H_differences':cross,
        'probe_comparison':compare_probes(probe,saved['probe'],p_limit),
        'note':'Same fixed prefixes and metadata; phase2 has no weight SHA, so this is numerical evidence, not cryptographic proof of old weight identity.'}
    synchronize(device)
    elapsed = time.perf_counter()-started
    fraction = torch.cuda.max_memory_allocated()/torch.cuda.get_device_properties(0).total_memory if device=='cuda' else None
    result['seconds_including_both_chunks_and_probes'] = elapsed
    result['peak_allocated_fraction'] = fraction
    result['resource_passed'] = elapsed <= PROTOCOL['max_canary_seconds'] and (fraction is None or fraction <= PROTOCOL['max_canary_allocated_fraction'])
    result['passed'] = bool(result['probe_comparison']['passed'] and all(math.isfinite(v) and v<=h_limit for v in cross.values())
        and result['cross_run_phase2_comparison']['probe_comparison']['passed'] and result['resource_passed'])
    result['stage']='finished';dump(path,result)
    if not result['passed']:
        raise RuntimeError('Supplementary FP32 C/cross-run/resource gate failed; no actions started')
    return result


def integer_grade(answer, reference):
    if answer is None:
        return None
    answer, reference = str(answer).strip(), str(reference).strip()
    if re.fullmatch(r'[+-]?\d+',answer) and re.fullmatch(r'[+-]?\d+',reference):
        return int(answer)==int(reference)
    return None


def write_tables(output, records, points, baseline, reused_keys):
    rows=[]
    for key, record in sorted(records.items()):
        ref=read(baseline/'cases'/f"{record['case']:03d}.json")['reference_answer']
        point=points[(record['case'],record['cut'])]
        box=record.get('pred_boxed') or record.get('candidate_boxed')
        boxed_grade=integer_grade(box,ref)
        strict_grade=boxed_grade if record.get('answer_completed') else None
        rows.append({'case':record['case'],'cut':record['cut'],'seed':record['replicate_seed'],'branch':record['branch'],
            'status':record['status'],'answer_completed_EOS':record.get('answer_completed',False),
            'pred_boxed_or_candidate':box,'reference_answer':ref,'strict_complete_integer_correct':strict_grade,
            'boxed_candidate_integer_correct_provisional_if_truncated':boxed_grade,
            'generated_tokens':record['extra_generated_tokens'],'generation_budget':record['generation_budget'],
            'model_call_seconds':record['model_call_seconds'],'branch_wall_seconds':record.get('branch_wall_seconds'),
            'prefill_input_tokens':record['prefill_tokens_total'],'forced_summary_succeeded':record.get('forced_summary_succeeded'),
            'summary_generated_tokens_total':record.get('summary_generated_tokens_total'),
            'forced_summary_fallback_used':record.get('forced_summary_fallback_used'),
            'executed_summary_restarts':record.get('executed_summary_restarts'),
            'C_reused_phase2_answer':point['probe']['answer'],'C_reused_phase2_mean':point['probe']['C_mean_selected_token_probability'],
            'C_reused_phase2_integer_correct':integer_grade(point['probe']['answer'],ref),
            'phase2_delta_H_asag':point['relative_changes']['asag_style_probe_window']['normalized_H'],
            'reused_completed_branch':key in reused_keys})
    if rows:
        path=output/'branch_review.csv';temporary=path.with_suffix('.tmp')
        with temporary.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
        temporary.replace(path)
    pairs=[]
    grouped={}
    for r in rows:grouped.setdefault((r['case'],r['cut'],r['seed']),{})[r['branch']]=r
    for identity,g in sorted(grouped.items()):
        if len(g)!=2:continue
        a,b=g['A_continue'],g['B_summary']
        grades=[a['strict_complete_integer_correct'],b['strict_complete_integer_correct']]
        pairs.append({'case':identity[0],'cut':identity[1],'seed':identity[2],
            'A_status':a['status'],'B_status':b['status'],'A_box':a['pred_boxed_or_candidate'],'B_box':b['pred_boxed_or_candidate'],
            'B_minus_A_strict_correct':int(grades[1])-int(grades[0]) if all(x is not None for x in grades) else None,
            'both_strict_complete_correct':all(x is True for x in grades),
            'B_minus_A_generated_tokens':b['generated_tokens']-a['generated_tokens'],
            'B_minus_A_model_call_seconds':b['model_call_seconds']-a['model_call_seconds'],
            'B_forced_summary_succeeded':b['forced_summary_succeeded'],'B_forced_fallback_used':b['forced_summary_fallback_used']})
    dump(output/'paired_review.json',{'pairs':pairs,'note':'Development action outcomes, not an online policy or speedup. Missing/manual grades and truncations retained. C is reused, not a new independent replicate. Monitoring/audit/model-load costs are separate.'})


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['model','baseline','phase2-input','output']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    ap.add_argument('--mode',choices=['smoke','main','replicate'],default='smoke')
    ap.add_argument('--resume-from',type=Path)
    ap.add_argument('--prior-run',type=Path)
    args=ap.parse_args(argv)
    if (args.mode=='smoke' and (args.resume_from or args.prior_run)) or (args.mode=='main' and (not args.resume_from or args.prior_run)) or (args.mode=='replicate' and (not args.prior_run or args.resume_from)):
        ap.error('smoke: no prior; main: --resume-from required; replicate: --prior-run required')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'branches').mkdir();(args.output/'numerical_audits').mkdir()
    summary={'completed':False,'mode':args.mode,'planned_branches':0,'completed_branches':0,'new_branches':0,'reused_branches':0,
             'numerical_audit_passed':False,'actual_GPU_execution_this_run':False,'stage':'initializing'}
    records,reused_keys={},set()
    def progress(stage,**kwargs):
        summary.update(stage=stage,**kwargs);dump(args.output/'summary.json',summary)
        print(json.dumps({'stage':stage,**kwargs},ensure_ascii=False),flush=True)
    def interrupted(signum,frame):
        raise InterruptedError('Received termination signal '+str(signum)+'; partial records retained')
    signal.signal(signal.SIGTERM,interrupted)
    progress('validating_inputs')
    try:
        import torch,transformers
        from transformers import AutoTokenizer,AutoModelForCausalLM
        from actions import run_action
        from decoder import CachedDecoder
        if transformers.__version__!='4.51.3':raise ValueError('Requires existing Transformers4.51.3; do not upgrade environment')
        if args.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable; real AutoDL run required')
        arithmetic=precision_settings(torch,'float32')
        tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        plan,candidates,points,evidence=load_evidence(args.phase2_input,args.baseline,args.model,tokenizer)
        jobs=schedule(candidates,args.mode)
        summary['planned_branches']=len(jobs)
        progress('hashing_local_weights',planned_branches=len(jobs))
        weights=weight_fingerprint(args.model)
        identity={'protocol':PROTOCOL,'code_sha256':{n:file_sha(Path(__file__).with_name(n)) for n in CODE_FILES},
            'weights':weights,'plan_sha256':file_sha(args.phase2_input/'plan.json'),
            'phase2_manifest_sha256':file_sha(args.phase2_input/'evidence_manifest.json'),
            'model_tokenizer_metadata_sha256':plan['model_and_tokenizer_file_sha256']}
        config={'protocol':PROTOCOL,'experiment_fingerprint':object_sha(identity),'experiment_identity':identity,
            'mode':args.mode,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'hardware':hardware_info(torch,args.device),'arithmetic_settings':arithmetic,'python':sys.version,
            'platform':platform.platform(),'transformers':transformers.__version__,'local_weight_identity_claim':'Content hashes fix this and future runs; historical phase2 weight revision and paper training stage are not independently proven.',
            'schedule':[{'key':job_key(c,b,s),'case':c['case'],'cut':c['cut'],'branch':b,'replicate_seed':s} for c,b,s in jobs],
            'max_additional_generated_tokens':sum(PROTOCOL['total_generated_budget_including_shared_cut']-c['cut'] for c,b,s in jobs),
            'not_online_controller':True,'no_new_training':True}
        dump(args.output/'run_config.json',config);dump(args.output/'plan.json',plan)
        source=args.resume_from or args.prior_run
        reuse_manifest={'source':str(source) if source else None,'branches':{},'historical_costs_not_reexecuted':True}
        if source:
            prior,prior_summary,complete=validate_prior(source,config,candidates,args.mode)
            (args.output/'prior_provenance').mkdir()
            for name in ['run_config.json','summary.json','completed_artifacts.json']:
                shutil.copy2(source/name,args.output/'prior_provenance'/name)
            reuse_manifest.update(source_run_config_sha256=file_sha(source/'run_config.json'),
                source_hardware=prior['hardware'],current_hardware=config['hardware'],source_mode=prior['mode'],
                fingerprint_compatibility=fingerprint_bridge(prior,config),
                hardware_compatibility=compare_hardware(config['hardware'],prior['hardware']))
            for c,b,s in jobs:
                key=job_key(c,b,s)
                if key in complete:
                    path,record=complete[key]
                    shutil.copy2(path,args.output/'branches'/path.name)
                    records[key]=record;reused_keys.add(key)
                    reuse_manifest['branches'][key]={'source_path':str(path),'sha256':file_sha(path)}
        dump(args.output/'reuse_manifest.json',reuse_manifest)
        def save_artifacts():
            dump(args.output/'completed_artifacts.json',{'experiment_fingerprint':config['experiment_fingerprint'],
                 'branches_sha256':{k:file_sha(args.output/'branches'/(k+'.json')) for k in records}})
        save_artifacts()
        summary.update(completed_branches=len(records),reused_branches=len(records))
        progress('model_loading')
        started=time.perf_counter()
        model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager').to(args.device).eval()
        synchronize(args.device);summary['model_load_seconds']=time.perf_counter()-started
        if next(model.parameters()).dtype != torch.float32:raise ValueError('Actual model is not FP32')
        summary['actual_GPU_execution_this_run']=args.device=='cuda'
        config['actual_model_dtype']=str(next(model.parameters()).dtype);dump(args.output/'run_config.json',config)
        audit_started=time.perf_counter()
        for case in plan['cohort_cases']:
            candidate=max((c for c in candidates if c['case']==case),key=lambda c:c['cut'])
            progress('numerical_audit',current_case=case,current_cut=candidate['cut'])
            audit_point(model,tokenizer,candidate,points[(case,candidate['cut'])],args.device,args.output/'numerical_audits'/f"case_{case:03d}.json")
        summary['numerical_audit_seconds']=time.perf_counter()-audit_started
        summary['numerical_audit_passed']=True
        # Independent warmup of the action backend; never enters a branch or its RNG.
        warmup=CachedDecoder(model,tokenizer,args.device,prefill_chunk=PROTOCOL['prefill_chunk'],temperature=PROTOCOL['temperature'],top_p=PROTOCOL['top_p'])
        warmup.start(candidates[0]['input_token_ids'][:128],seed=0)
        for _ in range(4):warmup.next_token()
        warmup.close();summary['warmup_stats']=warmup.stats()
        for candidate,branch,seed in jobs:
            key=job_key(candidate,branch,seed)
            if key in records:continue
            path=args.output/'branches'/(key+'.json')
            progress('running_branch',current_key=key,completed_branches=len(records))
            decoder=CachedDecoder(model,tokenizer,args.device,prefill_chunk=PROTOCOL['prefill_chunk'],temperature=PROTOCOL['temperature'],top_p=PROTOCOL['top_p'])
            def save(record):
                dump(path,record)
                print(json.dumps({'branch':key,'status':record.get('status'),'sampled_tokens':record.get('extra_generated_tokens'),'budget':record.get('generation_budget')},ensure_ascii=False),flush=True)
            if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
            synchronize(args.device);branch_started=time.perf_counter()
            record=run_action(candidate,branch,seed,decoder,tokenizer,
                total_budget=PROTOCOL['total_generated_budget_including_shared_cut'],summary_budget=PROTOCOL['forced_summary_budget'],heartbeat_tokens=PROTOCOL['heartbeat_tokens'],on_update=save)
            synchronize(args.device)
            record['branch_wall_seconds']=time.perf_counter()-branch_started
            record['branch_wall_scope']='Includes initialization/prefills/decode/resets/checkpoints; excludes model load, numerical audits, warmup and old phase2 signal replay.'
            if args.device=='cuda':
                record['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
                record['peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
            validate_record(record,candidate,branch,seed);dump(path,record)
            records[key]=record
            save_artifacts()
            summary.update(completed_branches=len(records),new_branches=len(records)-len(reused_keys))
            write_tables(args.output,records,points,args.baseline,reused_keys)
            progress('branch_completed',completed_branches=len(records),new_branches=summary['new_branches'])
        write_tables(args.output,records,points,args.baseline,reused_keys)
        summary.update(completed=len(records)==len(jobs),answer_completed_branches=sum(bool(r.get('answer_completed')) for r in records.values()),
            actual_GPU_execution_this_run=args.device=='cuda',
            all_completed_branch_generated_tokens=sum(r['extra_generated_tokens'] for r in records.values()),
            newly_executed_branch_wall_seconds=sum(r.get('branch_wall_seconds',0) for k,r in records.items() if k not in reused_keys),
            reused_branch_wall_seconds=sum(r.get('branch_wall_seconds',0) for k,r in records.items() if k in reused_keys),
            note='Completed means execution/accounting finished, not answer correctness. Each model outcome retained; no threshold fitting or online speedup claim.')
        new_tokens=sum(r['extra_generated_tokens'] for k,r in records.items() if k not in reused_keys)
        summary['new_generated_tokens']=new_tokens
        summary['new_generated_tokens_per_branch_wall_second']=(new_tokens/summary['newly_executed_branch_wall_seconds']) if summary['newly_executed_branch_wall_seconds']>0 else None
        progress('completed')
    except (Exception,KeyboardInterrupt) as exc:
        summary['error']=repr(exc);summary['failed_stage']=summary.get('stage');summary['stage']='failed'
        if args.device=='cuda' and 'torch' in locals() and torch.cuda.is_initialized():
            summary['peak_allocated_gib_at_failure']=torch.cuda.max_memory_allocated()/2**30
            summary['peak_reserved_gib_at_failure']=torch.cuda.max_memory_reserved()/2**30
        raise
    finally:
        dump(args.output/'summary.json',summary)


if __name__=='__main__':
    main()
