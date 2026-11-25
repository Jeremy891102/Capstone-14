"""Remaining 931 oracle calls, reusing pilot20; gross total budget USD30.
No API calls without --execute. Failed/unknown outcomes require inspection before resume.
"""
import argparse
import hashlib
import json
import os
import threading
from pathlib import Path

import yaml
from video_report.io_utils import atomic_write_json, strict_json_loads
from video_report.pipeline import create_run, execute_run, check_resume_compatibility
from video_report.providers.vertex_gemini import VertexGeminiProvider
from video_report.run_store import RunDir
from video_report.snapshot import build_snapshot
from run_argon_oracle_pilot import load_credentials

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'runs/argon_oracle_pilot20'
RUN=ROOT/'runs/argon_oracle_remaining931'
CONFIG=ROOT/'experiments/cc4d_oracle_argon/remaining931/config.yaml'


class BudgetStop(KeyboardInterrupt):
    pass


def gross_cost(u):
    if not all(isinstance(u.get(k),int) and u[k]>=0 for k in ['promptTokenCount','candidatesTokenCount']):
        return None
    cache=u.get('cachedContentTokenCount',0)
    return ((u['promptTokenCount']-cache)*4+cache*.2+(u['candidatesTokenCount']+u.get('thoughtsTokenCount',0))*20)/1e6


class Budget:
    def __init__(self,path,limit,spent):
        self.path,self.limit,self.spent=path,limit,spent
        self.reservations={};self.halted=False;self.lock=threading.Lock()

    def save(self):
        atomic_write_json(self.path,{'gross_usd':self.spent,'limit_gross_usd':self.limit,'reserved_gross_usd':self.reservations,'halted':self.halted})

    def reserve(self,key,amount):
        with self.lock:
            if self.halted or self.spent+sum(self.reservations.values())+amount>self.limit:
                raise BudgetStop('budget / unknown outcome stop')
            assert key not in self.reservations
            self.reservations[key]=amount;self.save()

    def settle(self,key,cost):
        with self.lock:
            if cost is None:
                self.halted=True
            else:
                reserved=self.reservations.pop(key)
                self.spent+=cost
                if cost>reserved:self.halted=True
            self.save()


def read_rows(path):
    return [strict_json_loads(s) for s in path.read_text().splitlines() if s.strip()]


def prepare():
    all_reports=read_rows(ROOT/'data/cc4d_oracle_single_v1/reports.jsonl')
    pilot_reports=read_rows(PILOT/'snapshot/reports.jsonl')
    ids={r['id'] for r in pilot_reports}
    remaining=[r for r in all_reports if r['id'] not in ids]
    assert len(ids)==20 and len(remaining)==931 and len(all_reports)==951
    byid={r['id']:r for r in all_reports}
    # Frozen reports are schema-normalized; compare all inference-relevant input/fields.
    from video_report.benchmarks.schema import Report
    for r in pilot_reports:
        assert Report.model_validate(byid[r['id']]).model_dump(mode='json')==r
    out=ROOT/'data/cc4d_oracle_single_remaining931';out.mkdir(exist_ok=True)
    truths=read_rows(ROOT/'data/cc4d_oracle_single_v1/ground_truth.jsonl')
    for name,rows in [('reports',remaining),('ground_truth',[g for g in truths if g['report_id'] not in ids])]:
        path=out/(name+'.jsonl');text=''.join(json.dumps(r)+'\n' for r in rows)
        if path.exists():assert path.read_text()==text
        else:path.write_text(text)
    (out/'manifest.json').write_bytes((ROOT/'data/cc4d_oracle_single_v1/manifest.json').read_bytes())
    cfg=yaml.safe_load((ROOT/'experiments/cc4d_oracle_argon/pilot20/config.yaml').read_text())
    cfg['name']='cc4d_oracle_argon_remaining931'
    cfg['data']['benchmark']='../../../data/cc4d_oracle_single_remaining931'
    cfg['execution']['concurrency']=4
    CONFIG.parent.mkdir(exist_ok=True)
    text=yaml.safe_dump(cfg,sort_keys=False)
    if CONFIG.exists():assert CONFIG.read_text()==text
    else:CONFIG.write_text(text)
    snap=build_snapshot(CONFIG)
    old=list(RunDir(PILOT).load_requests().values())
    assert len(snap.requests)==931
    assert not {r.report_id for r in old}&{r.report_id for r in snap.requests}
    assert all(r.generation==old[0].generation and r.model==old[0].model and r.fps==old[0].fps and r.system==old[0].system for r in snap.requests)
    assert all('["A", "C"]' not in r.user for r in snap.requests)
    return snap


def known_spend():
    total=0
    for run in [PILOT,RUN]:
        for p in (run/'api_raw').glob('*.json'):
            if p.name.endswith('_request.json'):continue
            cost=gross_cost(json.loads(p.read_text()).get('usageMetadata',{}))
            if cost is None:raise RuntimeError(f'unknown usage: {p}')
            total+=cost
    return total


def make_factory(budget):
    def factory(cfg,run):
        provider=VertexGeminiProvider(cfg.provider.model,cfg.provider.gemini)
        post,generate=provider.http.post,provider.generate
        local=threading.local()
        rawdir=run.path/'api_raw';rawdir.mkdir(exist_ok=True)

        def capture(url,headers,body,timeout_s,max_bytes):
            data=post(url,headers,body,timeout_s,max_bytes)
            key=hashlib.sha256(local.call_id.encode()).hexdigest()[:16]
            assert not (rawdir/(key+'.json')).exists(), 'refuse duplicate successful API call'
            atomic_write_json(rawdir/(key+'.json'),data)
            clean=json.loads(json.dumps(body))
            for part in clean['contents'][0]['parts']:
                if 'inlineData' in part:
                    media=part['inlineData']['data']
                    part['inlineData']['data']={'base64_sha256':hashlib.sha256(media.encode()).hexdigest(),'base64_length':len(media)}
            atomic_write_json(rawdir/(key+'_request.json'),{'call_id':local.call_id,'url':url,'body':clean})
            budget.settle(local.call_id,gross_cost(data.get('usageMetadata',{})))
            local.settled=True
            return data

        def guarded(request,*,timeout_s):
            # Pilot measured ~73 input tokens/sec. Reserve 100/sec + 5k text and
            # the full 8192 output ceiling; halt if actual spend exceeds reservation.
            seconds=sum(v.end_s-v.start_s for v in request.videos)
            reserve=((100*seconds+5000)*4+8192*20)/1e6
            budget.reserve(request.call_id,reserve)
            local.call_id=request.call_id;local.settled=False
            try:
                return generate(request,timeout_s=timeout_s)
            except BaseException:
                if not local.settled:budget.settle(request.call_id,None)
                raise
        provider.http.post=capture;provider.generate=guarded
        return provider
    return factory


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--execute',action='store_true');ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();os.chdir(ROOT)
    prepare()
    print('Verified 931 remaining requests; pilot20 disjoint and unchanged.',flush=True)
    if not args.execute:return
    load_credentials()
    if RUN.exists():
        if not args.resume:raise SystemExit('run exists; inspect before explicit --resume')
        run=RunDir(RUN)
        for s in run.read_all_calls():
            if s['status']=='in_progress' or any(a.get('error',{}).get('outcome_unknown') for a in s['attempts']):
                raise SystemExit('unknown outcome requires manual reconciliation, not automatic retry')
        check_resume_compatibility(run,config_path=CONFIG,video_root=None,allow_code_change=False)
        mode='resume'
    else:
        run=create_run(CONFIG,ROOT/'runs',run_id=RUN.name);mode='create'
    budget=Budget(RUN/'total_spend.json',limit=30,spent=known_spend())
    def on_done(state):
        print(state['call_id'],state['status'],f'gross_total={budget.spent:.6f}',flush=True)
        if state['status']!='succeeded':raise BudgetStop('failed call; inspect before continuing')
    result=execute_run(run,mode=mode,provider_factory=make_factory(budget),retry_failed=args.resume,on_call_done=on_done)
    print(result,flush=True)

if __name__=='__main__':main()
