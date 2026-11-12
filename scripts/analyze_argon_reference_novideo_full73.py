"""Full73-video comparison; never fabricate paired historical outcomes from aggregates."""
import argparse,csv,json,hashlib,gzip,statistics
from pathlib import Path
from collections import Counter
from run_argon_reference_novideo_full73 import validate
from run_argon_reference_novideo import current_records
from analyze_argon_reference_novideo import bootstrap,export

def main(root,history_csv=None):
    plan,rows,gt,retained=validate(root);out=root/'execution_20261112';records=current_records(out,rows)
    if len(records)!=4672:raise ValueError(f'Incomplete: {len(records)}/4672')
    by={(r['request']['report_id'],r['request']['field_id'],r['request']['condition']):r for r in records}
    single={}
    for r in csv.DictReader((root/'historical_single_predictions.csv').open()):
        key=(r['report_id'],r['field_id'])
        if key not in gt:continue
        if key in single or r['target']!=gt[key] or (r['correct']=='True')!=(r['prediction']==gt[key]):raise ValueError('Historical single duplicate/GT/score mismatch')
        single[key]=r
    if set(single)!=set(gt):raise ValueError('Historical single incomplete')
    context={};history_csv=history_csv or root/'historical_full_paired_comparison.csv'
    if history_csv.exists():
        for r in csv.DictReader(history_csv.open()):
            if r.get('condition')!='context_7':continue
            key=(r['report_id'],r['field_id'])
            if key not in gt:continue
            if key in context or r['target']!=gt[key] or (r['correct']=='True')!=(r['prediction']==gt[key]):raise ValueError('Historical context duplicate/GT/score mismatch')
            context[key]=r
        if set(context)!=set(gt):raise ValueError('Historical context CSV is not full2336 cohort; refuse partial substitution')
    paired=[]
    for rid,fid in sorted(gt):
        target=gt[rid,fid];sn=by[rid,fid,'single'];cn=by[rid,fid,'context_7'];sv=single[rid,fid]['prediction']==target
        if any(r['target']!=target or r['correct']!=(r['prediction']==target) for r in [sn,cn]):raise ValueError('New score mismatch')
        row={'report_id':rid,'field_id':fid,'target':target,'video_single_prediction':single[rid,fid]['prediction'],'video_context7_prediction':context[rid,fid]['prediction'] if context else '', 'novideo_single_prediction':sn['prediction'],'novideo_context7_prediction':cn['prediction'],'video_single_correct':int(sv),'video_context7_correct':int(context[rid,fid]['correct']=='True') if context else '', 'novideo_single_correct':int(sn['correct']),'novideo_context7_correct':int(cn['correct']),'novideo_context_gain':int(cn['correct'])-int(sn['correct']),'video_gain_single':int(sv)-int(sn['correct'])}
        if context:row.update(video_context_gain=row['video_context7_correct']-int(sv),interaction_D=row['video_context7_correct']-int(sv)-int(cn['correct'])+int(sn['correct']),video_gain_context7=row['video_context7_correct']-int(cn['correct']))
        paired.append(row)
    keys=['novideo_context_gain','video_gain_single']+(['video_context_gain','interaction_D','video_gain_context7'] if context else [])
    comparisons=bootstrap(paired,keys)
    hist=json.loads((root/'historical_full_summary.json').read_text());oldgain=hist['comparisons']['context7_minus_single']
    if sum(r['video_single_correct'] for r in paired)!=1402:raise ValueError('Historical single aggregate differs')
    if context and sum(r['video_context7_correct'] for r in paired)!=1572:raise ValueError('Historical context aggregate differs')
    oldgainpoint=100*(1572-1402)/2336
    if not context:
        comparisons['video_context_gain']={'estimate_pp':oldgainpoint,'paired_video_bootstrap_95_pp':oldgain['paired_video_bootstrap_95_interval_pp'],'source':'Historical completed full73-video analysis; raw per-target file not local','bootstrap_seed':20261028,'draws':10000}
        comparisons['interaction_D']={'estimate_pp':oldgainpoint-comparisons['novideo_context_gain']['estimate_pp'],'paired_video_bootstrap_95_pp':None,'reason':'Full historical context per-target outcomes unavailable locally; paired interval pending retrieval'}
        comparisons['video_gain_context7']={'estimate_pp':100*(1572-sum(r['novideo_context7_correct'] for r in paired))/2336,'paired_video_bootstrap_95_pp':None,'reason':'Full historical context per-target outcomes unavailable locally'}
    conditions={}
    for key in ['video_single','video_context7','novideo_single','novideo_context7']:
        ncorrect=1572 if key=='video_context7' and not context else sum(r[key+'_correct'] for r in paired)
        conditions[key]={'targets':2336,'correct':ncorrect,'accuracy_pct':100*ncorrect/2336,'prediction_letters':dict(Counter(r[key+'_prediction'] for r in paired)) if key!='video_context7' or context else None}
    costs={'estimated_success_usd_including_retained':sum(r['usage']['estimated_usd'] for r in records),'estimated_retained_success_usd':sum(r['estimated_usd'] for r in retained),'prompt_tokens':sum(r['usage']['prompt_tokens'] for r in records),'answer_tokens':sum(r['usage']['answer_tokens'] for r in records),'reasoning_tokens':sum(r['usage']['reasoning_tokens'] for r in records),'completion_including_reasoning':sum(r['usage']['completion_including_reasoning'] for r in records),'rate_source':records[0]['usage']['cost_source']}
    costs['estimated_new_success_usd']=costs['estimated_success_usd_including_retained']-costs['estimated_retained_success_usd']
    archived=[]
    for path in (out/'attempt_history').rglob('*.json'):
        r=json.loads(path.read_text())
        if r.get('status')=='failed' and 'request' in r:archived.append({'call_id':r['request']['call_id'],'error_type':r.get('error_type'),'billing':'unknown','path':str(path.relative_to(out))})
    lat=sorted(r['api_latency_s'] for r in records)
    result={'status':'full_comparison_audited' if context else 'new_conditions_complete_historical_pairing_pending','targets':2336,'videos':73,'accepted_calls':4672,'retained_successes':640,'new_successes':4032,'invalid':0,'unknown_billing_attempts':len(archived),'http_attempts_including_retained':4672+len(archived),'imported_failed_attempts':sum('imported_subset' in a['path'] for a in archived),'new_failed_attempts':sum('imported_subset' not in a['path'] for a in archived),'archived_failed_attempts':archived,'historical_context_paired_predictions_available':bool(context),'conditions':conditions,'comparisons':comparisons,'usage_cost':costs,'latency':{'median_request_s':statistics.median(lat),'p95_request_s':lat[int(.95*(len(lat)-1))],'meaning':'Successful API roundtrip only; not experiment elapsed; excludes unsuccessful attempts and queue/downtime.'},'bootstrap':{'unit':'recording','draws':10000,'seed':20261112,'multiple_comparison_adjustment':False},'limitations':['Historical video outcomes and current no-video outcomes are not contemporaneous.','Same returned model alias does not prove no provider update.','Context protocol changes position/instructions as well as reference content.','Missing historical paired context file means no paired D confidence interval until retrieval.' if not context else 'Sampling uncertainty reflects recordings, not model/provider variation.'],'provenance':{'manifest_sha256':hashlib.sha256((root/'full_requests.jsonl').read_bytes()).hexdigest(),'historical_summary_sha256':hashlib.sha256((root/'historical_full_summary.json').read_bytes()).hexdigest(),'historical_context_csv_sha256':hashlib.sha256(history_csv.read_bytes()).hexdigest() if context else None,'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    d=out/'full_analysis';d.mkdir(exist_ok=True);export(d/'paired_predictions.csv',paired)
    calls=[]
    retained_ids={r['call_id'] for r in retained}
    for r in records:
        q=r['request'];calls.append({'call_id':q['call_id'],'report_id':q['report_id'],'field_id':q['field_id'],'condition':q['condition'],'retained':q['call_id'] in retained_ids,'target':r['target'],'prediction':r['prediction'],'correct':r['correct'],'valid':r['valid'],'api_latency_s':r['api_latency_s'],**r['usage']})
    export(d/'per_call.csv',calls);(d/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with gzip.open(d/'experiment_log.jsonl.gz','wt') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--history-csv',type=Path);a=p.parse_args();main(a.root,a.history_csv)
