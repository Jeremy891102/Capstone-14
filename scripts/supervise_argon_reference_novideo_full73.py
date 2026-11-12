"""Bounded, transport-only recovery. Never retries a successful/invalid/model-mismatched response."""
import argparse,collections,fcntl,hashlib,json,subprocess,sys,time,math
from pathlib import Path
from run_argon_reference_novideo_full73 import validate
from run_argon_reference_novideo import digest,write,RESERVE_PER_CALL

def eligible(record,row,prior_count):
    return (record.get('status')=='failed' and record.get('error_type')=='TimeoutError'
            and 'raw_response' not in record and not record.get('raw_response_text')
            and record.get('request')==row and record.get('input_sha256')==row['body_sha256']
            and prior_count==0)

def recover(root,max_timeouts,budget):
    _,rows,_,retained=validate(root);out=root/'execution_20261112';lookup={r['call_id']:r for r in rows};history=collections.Counter();newfailed=0
    for path in (out/'attempt_history').rglob('*.json'):
        r=json.loads(path.read_text())
        if r.get('status')=='failed' and 'request' in r:
            history[r['request']['call_id']]+=1
            if 'imported_subset' not in path.parts:newfailed+=1
    allrecords=[(p,json.loads(p.read_text())) for p in (out/'calls').glob('*.json')]
    failed=[(p,r) for p,r in allrecords if r['status']=='failed']
    if any(r['status']=='in_flight' for _,r in allrecords):raise ValueError('Uncertain in-flight call; manual review needed')
    if not failed:raise ValueError('No eligible transport failure; manual review needed')
    if newfailed+len(failed)>max_timeouts:raise ValueError('Timeout retry ceiling reached')
    goodcost=sum(r['usage']['estimated_usd'] for _,r in allrecords if r['status']=='succeeded')
    if goodcost+(sum(history.values())+len(failed)+8)*RESERVE_PER_CALL>budget:raise ValueError('Unknown-billing reserve plus retries exceeds budget')
    for path,r in failed:
        row=lookup.get(r['request']['call_id'])
        if row is None or not eligible(r,row,history[r['request']['call_id']]):raise ValueError('Failure not eligible for bounded transport-only retry')
    for path,r in failed:
        n=1
        while (out/'attempt_history'/f'expansion_timeout_{n:03d}').exists():n+=1
        folder=out/'attempt_history'/f'expansion_timeout_{n:03d}';folder.mkdir()
        sha=hashlib.sha256(path.read_bytes()).hexdigest();path.rename(folder/path.name)
        write(folder/'policy_review.json',{'policy':'One retry per call for TimeoutError with no response only; max16 new timeout failures; exact input required; no successful call repeated','call_id':r['request']['call_id'],'failed_attempt_sha256':sha,'billing':'unknown, retained separately','input_sha256':r['input_sha256'],'checked_unix':time.time()})
    stop=out/'stop_reason.json'
    if stop.exists():stop.rename(folder/'stop_reason.json')
    return len(failed)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--workers',type=int,default=8);p.add_argument('--max-estimated-usd',type=float,default=40);p.add_argument('--max-timeouts',type=int,default=16);a=p.parse_args()
    if not 1<=a.max_timeouts<=16:raise ValueError('Timeout ceiling must be between1and16')
    if not math.isfinite(a.max_estimated_usd) or not 0<a.max_estimated_usd<=40 or not 1<=a.workers<=8:raise ValueError('Invalid authorized resource limits')
    validate(a.root)
    if not a.execute:print('Offline policy preflight passed; no request sent');return
    out=a.root/'execution_20261112'
    with (out/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (out/'runner.lock').open('a') as runnerlock:
            fcntl.flock(runnerlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            statuspath=out/'status.json'
            if statuspath.exists() and json.loads(statuspath.read_text()).get('status')=='review_needed':
                count=recover(a.root,a.max_timeouts,a.max_estimated_usd);print(f'Archived {count} eligible timeout(s); same frozen input will be retried once',flush=True)
        controller=Path(__file__).with_name('run_argon_reference_novideo_full73.py')
        for launch in range(1,a.max_timeouts+3):
            print(json.dumps({'event':'controller_start','launch':launch,'time':time.time()}),flush=True)
            with (out/f'controller_{int(time.time())}.log').open('w') as log:
                result=subprocess.run([sys.executable,str(controller),'--root',str(a.root),'--execute','--workers',str(a.workers),'--max-estimated-usd',str(a.max_estimated_usd)],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode==0:
                status=json.loads((out/'status.json').read_text())
                if status['status']!='completed' or status['succeeded']!=4672:raise ValueError('Controller exited without full completion')
                print('FULL73 COMPLETED',flush=True);return
            with (out/'runner.lock').open('a') as runnerlock:
                fcntl.flock(runnerlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                n=recover(a.root,a.max_timeouts,a.max_estimated_usd)
            print(json.dumps({'event':'bounded_timeout_retry','count':n,'cooldown_s':10,'time':time.time()}),flush=True);time.sleep(10)
        raise ValueError('Supervisor restart ceiling reached')
if __name__=='__main__':main()
