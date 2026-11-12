"""Expand the frozen Argon no-video study to all73reports. Explicit --execute only."""
import argparse,hashlib,json,math,time,fcntl
from pathlib import Path
from collections import Counter
import run_argon_reference_novideo as core

def validate(root):
    plan=json.loads((root/'plan.json').read_text())
    for name,sha in plan['files_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen expansion input changed: '+name)
    rows=list(map(json.loads,(root/'full_requests.jsonl').read_text().splitlines()))
    gt={(r['report_id'],r['field_id']):r['target'] for r in map(json.loads,(root/'evaluation_only.jsonl').read_text().splitlines())}
    if len(rows)!=4672 or len(gt)!=2336 or len({r['call_id'] for r in rows})!=4672:raise ValueError('Wrong full scope')
    pairs={(r['report_id'],r['field_id'],r['condition']) for r in rows}
    if len(pairs)!=4672 or pairs!={(rid,fid,c) for rid,fid in gt for c in ['single','context_7']}:raise ValueError('Unpaired/duplicate target conditions')
    if Counter(r['condition'] for r in rows)!=Counter({'single':2336,'context_7':2336}):raise ValueError('Wrong conditions')
    if len({r['report_id'] for r in rows})!=73 or set(Counter(r['report_id'] for r in rows).values())!={64}:raise ValueError('Wrong report count')
    for r in rows:
        if r['model']!=core.MODEL or r['body_sha256']!=core.digest(r['body']) or r['body']['generationConfig']!=core.EXPECTED_GENERATION:raise ValueError('Changed model/body/settings')
        if gt[r['report_id'],r['field_id']] not in ['A','B','C','D','E']:raise ValueError('Wrong target')
        if len(r['body']['contents'])!=1 or any(set(part)!={'text'} for c in r['body']['contents'] for part in c['parts']):raise ValueError('Not text-only')
    retained=json.loads((root/'retained_successes.json').read_text());out=root/'execution_20261112';lookup={r['call_id']:r for r in rows}
    if len(retained)!=640 or len({r['call_id'] for r in retained})!=640:raise ValueError('Wrong retention scope')
    for r in retained:
        p=core.path_for(out,lookup[r['call_id']])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Retained success changed')
        core.check_reuse(json.loads(p.read_text()),lookup[r['call_id']])
    return plan,rows,gt,retained

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--workers',type=int,default=8);p.add_argument('--max-estimated-usd',type=float,default=40);p.add_argument('--env-file',type=Path,default=Path('/Users/jeremy/Desktop/NYU/Capstone/Capstone-14/.env'));a=p.parse_args()
    plan,rows,gt,retained=validate(a.root);out=a.root/'execution_20261112'
    if not 1<=a.workers<=8 or not math.isfinite(a.max_estimated_usd) or a.max_estimated_usd<=0:raise ValueError('Wrong limits')
    with (out/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not a.execute:
            print(json.dumps({'mode':'offline','targets':2336,'reports':73,'calls':4672,'retained':640,'new':4032,'checks':'passed','workers':a.workers}));return
        imported_ids={r['call_id'] for r in retained};retained_cost=sum(r['estimated_usd'] for r in retained)
        old_summarize=core.summarize
        def summary(out,allrows,phase,status,records=None):
            s=old_summarize(out,allrows,phase,status,records)
            archived=0
            for path in (out/'attempt_history').rglob('*.json'):
                old=json.loads(path.read_text())
                archived+=old.get('status')=='failed' and 'request' in old
            s.update(retained_successes=640,new_planned=4032,new_succeeded=s['succeeded']-640,new_estimated_success_usd=s['estimated_success_usd']-retained_cost,archived_failed_attempts=archived,unknown_billing_attempts=s['unknown_billing_attempts']+archived)
            core.write(out/'status.json',s);return s
        core.summarize=summary
        # Historical failed billing is unknown: reserve its whole per-call estimate in the guard.
        initial=summary(out,rows,'preflight','running');effective_budget=a.max_estimated_usd-initial['unknown_billing_attempts']*core.RESERVE_PER_CALL
        core.write(out/'authorization.json',{'instruction':'Expand to same73videos2336targets as prior fullwithvideo study','accepted_calls_max':4672,'retained_successes':640,'new_calls_max':4032,'workers':a.workers,'max_estimated_usd_total':a.max_estimated_usd,'currency_status':'Frozen historical-rate estimate only; failed billing unknown','controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'started_unix':time.time()})
        key=core.load_key(a.env_file)
        try:
            gate_rows=[r for r in rows if r['call_id'] not in imported_ids][:128]
            core.run_phase(out,gate_rows,rows,gt,key,'expansion_gate',a.workers,effective_budget)
            gate=core.current_records(out,gate_rows)
            if len(gate)!=128 or any(not r['valid'] for r in gate):raise ValueError('Expansion gate incomplete/invalid')
            mean=sum(r['usage']['estimated_usd'] for r in gate)/128
            projected=retained_cost+2*mean*4032+a.workers*core.RESERVE_PER_CALL
            if projected>effective_budget:raise ValueError('Expansion cost projection exceeds guard')
            core.write(out/'expansion_gate.json',{'status':'passed','new_calls_checked':128,'model':core.MODEL,'estimated_gate_usd':mean*128,'conservative_total_projection_usd':projected,'estimated_guard_usd':effective_budget,'all_valid':True,'continue_authorized':True})
            core.run_phase(out,rows,rows,gt,key,'full73',a.workers,effective_budget)
            summary(out,rows,'full73','completed')
        except Exception as e:
            summary(out,rows,'stopped','review_needed');core.write(out/'stop_reason.json',{'type':type(e).__name__,'reason':str(e),'time':time.time()});raise
if __name__=='__main__':main()
