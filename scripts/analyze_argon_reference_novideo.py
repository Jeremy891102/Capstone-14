"""Audit saved calls and compare all four conditions on the same fixed targets. Offline."""
import argparse,csv,json,random,statistics,hashlib,gzip
from pathlib import Path
from collections import defaultdict,Counter
from run_argon_reference_novideo import validate_rows,current_records,digest
HISTORY=Path('/Users/jeremy/Documents/Codex/2026-10-06/git/outputs/argon_context10_20261028/verified_results/paired_comparison.csv')

def export(path,rows):
    if not rows:raise ValueError('No rows')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def bootstrap(rows,keys):
    by=defaultdict(list)
    for r in rows:by[r['report_id']].append(r)
    videos=sorted(by);sizes={len(v) for v in by.values()}
    if len(sizes)!=1:raise ValueError('Primary cohort must be balanced per recording')
    stats={key:[sum(r[key] for r in by[v])/len(by[v])*100 for v in videos] for key in keys}
    rng=random.Random(20261112);samples={key:[] for key in keys}
    for _ in range(10000):
        indices=rng.choices(range(len(videos)),k=len(videos))
        for key in keys:samples[key].append(sum(stats[key][i] for i in indices)/len(videos))
    result={}
    for key in keys:
        a=sorted(samples[key]);result[key]={'estimate_pp':sum(stats[key])/len(videos),'paired_video_bootstrap_95_pp':[a[249],a[9749]]}
    return result

def analyze(root,scope):
    planned,gt=validate_rows(root);out=root/'execution_20261112'
    rows=[r for r in planned if r['pilot']] if scope=='pilot' else planned
    records=current_records(out,rows)
    if len(records)!=len(rows):raise ValueError(f'Incomplete {scope}: {len(records)}/{len(rows)}')
    hist={}
    for r in csv.DictReader(HISTORY.open()):
        if r['condition']=='context_7':
            key=(r['report_id'],r['field_id'])
            if key in hist:raise ValueError('Duplicate historical target')
            if any((r[col]=='True')!=(r[pred]==r['target']) for col,pred in [('correct','prediction'),('k1_correct','k1_prediction')]):raise ValueError('Historical prediction/score mismatch')
            hist[key]=r
    if len(hist)!=320 or set(hist)!=set(gt):raise ValueError('Historical target coverage differs')
    by={(r['request']['report_id'],r['request']['field_id'],r['request']['condition']):r for r in records}
    keys=sorted({(r['report_id'],r['field_id']) for r in rows});paired=[]
    for rid,fid in keys:
        h=hist[rid,fid];single=by[rid,fid,'single'];context=by[rid,fid,'context_7'];target=gt[rid,fid]
        if not all(r['target']==target and r['correct']==(r['prediction']==target) for r in [single,context]):raise ValueError('GT/score identity differs')
        if h['target']!=target:raise ValueError('Historical target differs')
        sv=h['k1_correct']=='True';cv=h['correct']=='True';sn=single['correct'];cn=context['correct']
        paired.append({'report_id':rid,'field_id':fid,'target':target,'video_single_prediction':h['k1_prediction'],'video_context7_prediction':h['prediction'],'novideo_single_prediction':single['prediction'],'novideo_context7_prediction':context['prediction'],'video_single_correct':int(sv),'video_context7_correct':int(cv),'novideo_single_correct':int(sn),'novideo_context7_correct':int(cn),'video_context_gain':int(cv)-int(sv),'novideo_context_gain':int(cn)-int(sn),'interaction_D':int(cv)-int(sv)-int(cn)+int(sn),'video_gain_single':int(sv)-int(sn),'video_gain_context7':int(cv)-int(cn)})
    metrics=bootstrap(paired,['video_context_gain','novideo_context_gain','interaction_D','video_gain_single','video_gain_context7'])
    conditions={}
    for key in ['video_single','video_context7','novideo_single','novideo_context7']:
        conditions[key]={'targets':len(paired),'correct':sum(r[key+'_correct'] for r in paired),'accuracy_pct':sum(r[key+'_correct'] for r in paired)/len(paired)*100,'prediction_letters':dict(Counter(r[key+'_prediction'] for r in paired))}
    transitions={}
    for v in ['video','novideo']:
        transitions[v]={'wrong_to_right':sum(not r[v+'_single_correct'] and r[v+'_context7_correct'] for r in paired),'right_to_wrong':sum(r[v+'_single_correct'] and not r[v+'_context7_correct'] for r in paired)}
    costs={'estimated_usd':sum(r['usage']['estimated_usd'] for r in records),'prompt_tokens':sum(r['usage']['prompt_tokens'] for r in records),'answer_tokens':sum(r['usage']['answer_tokens'] for r in records),'reasoning_tokens':sum(r['usage']['reasoning_tokens'] for r in records),'completion_including_reasoning':sum(r['usage']['completion_including_reasoning'] for r in records),'cached_prompt_tokens':sum(r['usage']['cached_prompt_tokens'] for r in records),'cost_source':records[0]['usage']['cost_source']}
    latencies=sorted(r['api_latency_s'] for r in records)
    result={'scope':scope,'status':'audited_complete','model':'gemini-4-argon','targets':len(keys),'videos':len({r[0] for r in keys}),'new_calls':len(records),'historical_video_calls_reused':2*len(keys),'invalid':sum(not r['valid'] for r in records),'provider_failures_in_accepted_calls':0,'conditions':conditions,'comparisons':metrics,'transitions':transitions,'usage_cost':costs,'latency':{'median_request_s':statistics.median(latencies),'mean_request_s':statistics.mean(latencies),'p95_request_s':latencies[int(.95*(len(latencies)-1))],'sum_request_s':sum(latencies),'sum_request_meaning':'Sum of successful request roundtrip latencies across concurrent requests; NOT experiment elapsed time; excludes queue waits/downtime.'},'bootstrap':{'unit':'recording','draws':10000,'seed':20261112,'adjusted_for_multiple_comparisons':False},'limitations':['Historical video outcomes were not contemporaneous; same returned model alias does not prove unchanged provider internals.','Target-only/context protocol also changes prompt framing and target position. D is a protocol interaction, not isolated visual reasoning.','No-video means media removed; video-referencing text retained.','Exploratory selected 10-video cohort; pilot has 8 video clusters, not a powered negative test.','Cost is frozen-rate usage estimate, not invoice; historical video costs excluded.'],'provenance':{'historical_paired_csv':str(HISTORY),'historical_paired_csv_sha256':hashlib.sha256(HISTORY.read_bytes()).hexdigest(),'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'full_manifest_sha256':hashlib.sha256((root/'full_requests.jsonl').read_bytes()).hexdigest()},'raw_call_hashes':{r['request']['call_id']:hashlib.sha256((out/'calls'/(r['request']['call_id'].replace('::','__').replace('=','-')+'.json')).read_bytes()).hexdigest() for r in records}}
    destination=out/(scope+'_analysis');destination.mkdir(exist_ok=True)
    export(destination/'paired_predictions.csv',paired)
    calls=[]
    for r in records:
        q=r['request'];calls.append({'call_id':q['call_id'],'report_id':q['report_id'],'field_id':q['field_id'],'condition':q['condition'],'model':r['response_model'],'target':r['target'],'prediction':r['prediction'],'valid':r['valid'],'correct':r['correct'],'api_latency_s':r['api_latency_s'],**r['usage']})
    export(destination/'per_call.csv',calls);(destination/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with gzip.open(destination/'experiment_log.jsonl.gz','wt') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--scope',choices=['pilot','full'],default='full');args=p.parse_args();s=analyze(args.root,args.scope);print(json.dumps({k:v for k,v in s.items() if k!='raw_call_hashes'},indent=2))
