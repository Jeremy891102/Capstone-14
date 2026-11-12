"""Frozen HD-EPIC Argon reference/no-video study. Offline default; explicit execution only."""
import argparse,concurrent.futures,hashlib,json,math,os,shlex,statistics,time,urllib.request,urllib.error
from pathlib import Path

MODEL='gemini-4-argon'
URL='https://aiplatform.googleapis.com/v1/projects/663601952811/locations/global/publishers/google/models/gemini-4-argon:generateContent'
EXPECTED_GENERATION={'maxOutputTokens':32768,'thinkingConfig':{'thinkingLevel':'MEDIUM'},'responseMimeType':'application/json'}
RESERVE_PER_CALL=(32768*10+20000*2)/1e6

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)
def load_key(path):
    value=os.environ.get('GEMINI_API_KEY')
    if value:return value
    for line in path.read_text().splitlines():
        line=line.strip().removeprefix('export ').strip()
        if '=' in line and line.split('=',1)[0].strip()=='GEMINI_API_KEY':
            parts=shlex.split(line.split('=',1)[1],comments=True)
            if len(parts)==1 and parts[0]:return parts[0]
    raise ValueError('GEMINI_API_KEY missing; no request sent')
def validate_rows(root):
    plan=json.loads((root/'plan.json').read_text())
    for name in ['full_requests.jsonl','pilot_requests.jsonl','evaluation_only.jsonl']:
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=plan['files_sha256'][name]:raise ValueError('Frozen file changed: '+name)
    rows=[json.loads(s) for s in (root/'full_requests.jsonl').read_text().splitlines()]
    gt={(r['report_id'],r['field_id']):r['target'] for r in map(json.loads,(root/'evaluation_only.jsonl').read_text().splitlines())}
    if len(rows)!=640 or len(gt)!=320 or len({r['call_id'] for r in rows})!=640:raise ValueError('Unexpected manifest scope')
    if sum(r['pilot'] for r in rows)!=128:raise ValueError('Unexpected pilot scope')
    pilots=[json.loads(s) for s in (root/'pilot_requests.jsonl').read_text().splitlines()]
    if pilots!=[r for r in rows if r['pilot']]:raise ValueError('Pilot differs from full manifest')
    for r in rows:
        if r['model']!=MODEL or digest(r['body'])!=r['body_sha256']:raise ValueError('Wrong model/input hash')
        if r['body']['generationConfig']!=EXPECTED_GENERATION:raise ValueError('Changed generation policy')
        if len(r['body']['contents'])!=1:raise ValueError('Unexpected contents')
        for content in r['body']['contents']:
            if any(set(p)!={'text'} for p in content['parts']):raise ValueError('Payload must be text-only')
        if gt[r['report_id'],r['field_id']] not in ['A','B','C','D','E']:raise ValueError('Invalid target')
    return rows,gt

def usage(raw):
    if not isinstance(raw,dict):raise ValueError('Missing usageMetadata')
    def token(k,required=False):
        v=raw.get(k,None if required else 0)
        if type(v) is not int or v<0:raise ValueError('Invalid usage '+k)
        return v
    prompt=token('promptTokenCount',True);answer=token('candidatesTokenCount',True);reasoning=token('thoughtsTokenCount');cached=token('cachedContentTokenCount')
    if cached>prompt:raise ValueError('Cached tokens exceed input')
    completion=answer+reasoning
    if prompt<=0 or answer<=0:raise ValueError('Usage must contain positive prompt/answer tokens')
    return {'prompt_tokens':prompt,'answer_tokens':answer,'reasoning_tokens':reasoning,'completion_including_reasoning':completion,'cached_prompt_tokens':cached,'estimated_usd':((prompt-cached)*2+cached*.1+completion*10)/1e6,'cost_source':'Frozen historical estimate $2/M uncached input,$0.10/M cached input,$10/M output incl reasoning; NOT provider invoice/current-price verification'}

def unique_object(pairs):
    obj={}
    for key,value in pairs:
        if key in obj:raise ValueError('Duplicate JSON key')
        obj[key]=value
    return obj

def parse(raw,fid):
    if not isinstance(raw,dict):raise ValueError('API outer body is not an object')
    if raw.get('modelVersion')!=MODEL:raise ValueError('Unexpected modelVersion: '+str(raw.get('modelVersion')))
    candidates=raw.get('candidates')
    if not isinstance(candidates,list) or len(candidates)!=1:raise ValueError('Missing/ambiguous candidate')
    c=candidates[0]
    if c.get('finishReason')!='STOP':raise ValueError('Unexpected finish reason: '+str(c.get('finishReason')))
    parts=c.get('content',{}).get('parts',[])
    answer=''.join(p.get('text','') for p in parts if not p.get('thought',False))
    valid=False;prediction=None
    try:
        obj=json.loads(answer,object_pairs_hook=unique_object)
        valid=isinstance(obj,dict) and set(obj)=={fid} and type(obj[fid]) is str and obj[fid] in ['A','B','C','D','E']
        if valid:prediction=obj[fid]
    except (ValueError,TypeError):pass
    return answer,prediction,valid,usage(raw.get('usageMetadata'))

def path_for(out,row):return out/'calls'/(row['call_id'].replace('::','__').replace('=','-')+'.json')
def check_reuse(record,row):
    if record.get('status')!='succeeded' or record.get('valid') is not True or record.get('input_sha256')!=row['body_sha256'] or record.get('request')!=row or record.get('response_model')!=MODEL:raise ValueError('Uncertain/failed/different result; manual review needed: '+row['call_id'])
    answer,prediction,valid,u=parse(record['raw_response'],row['field_id'])
    if any([answer!=record['answer'],prediction!=record['prediction'],valid!=record['valid'],u!=record['usage']]):raise ValueError('Saved response evidence differs')

def attempt(out,row,target,key):
    path=path_for(out,row)
    if path.exists():
        existing=json.loads(path.read_text());check_reuse(existing,row);return existing
    record={'status':'in_flight','request':row,'input_sha256':row['body_sha256'],'wire_url':URL,'started_unix':time.time(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(path,record);started=time.perf_counter()
    try:
        req=urllib.request.Request(URL,data=json.dumps(row['body']).encode(),headers={'x-goog-api-key':key,'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(req,timeout=180) as response:status=response.status;payload=response.read().decode()
        except urllib.error.HTTPError as error:status=error.code;payload=error.read().decode(errors='replace')
        record.update(http_status=status,api_latency_s=time.perf_counter()-started,raw_response_text=payload)
        if status!=200:raise ValueError('HTTP '+str(status))
        raw=json.loads(payload);record['raw_response']=raw
        answer,prediction,valid,u=parse(raw,row['field_id'])
        record.update(status='succeeded',response_model=raw['modelVersion'],answer=answer,prediction=prediction,valid=valid,usage=u,correct=valid and prediction==target,target=target)
    except Exception as error:
        record.update(status='failed',api_latency_s=time.perf_counter()-started,error_type=type(error).__name__,error=str(error),billing='unknown unless usage exists; no automatic retry')
    record['finished_unix']=time.time();write(path,record);return record

def current_records(out,rows):
    records=[]
    for row in rows:
        p=path_for(out,row)
        if p.exists():
            r=json.loads(p.read_text());check_reuse(r,row);records.append(r)
    return records

def summarize(out,rows,phase,status,records=None):
    if records is None:
        records=[]
        for row in rows:
            p=path_for(out,row)
            if p.exists():records.append(json.loads(p.read_text()))
    good=[r for r in records if r['status']=='succeeded'];cost=sum(r['usage']['estimated_usd'] for r in good)
    s={'status':status,'phase':phase,'planned':len(rows),'succeeded':len(good),'invalid':sum(not r['valid'] for r in good),'failed':sum(r['status']=='failed' for r in records),'in_flight':sum(r['status']=='in_flight' for r in records),'estimated_success_usd':cost,'unknown_billing_attempts':sum(r['status']!='succeeded' for r in records),'timestamp_unix':time.time(),'model_versions':sorted({r['response_model'] for r in good}),'conditions':{}}
    for c in ['single','context_7']:
        r=[z for z in good if z['request']['condition']==c]
        s['conditions'][c]={'calls':len(r),'valid':sum(z['valid'] for z in r),'correct':sum(z['correct'] for z in r),'estimated_usd':sum(z['usage']['estimated_usd'] for z in r),'prompt_tokens':sum(z['usage']['prompt_tokens'] for z in r),'completion_tokens_including_reasoning':sum(z['usage']['completion_including_reasoning'] for z in r),'reasoning_tokens':sum(z['usage']['reasoning_tokens'] for z in r),'api_seconds':sum(z['api_latency_s'] for z in r)}
    if good:s['median_api_latency_s']=statistics.median(r['api_latency_s'] for r in good)
    write(out/'status.json',s);return s

def run_phase(out,rows,allrows,gt,key,phase,workers,budget):
    finished=current_records(out,allrows) # Refuse uncertain attempts anywhere in this run.
    spent=sum(r['usage']['estimated_usd'] for r in finished)
    pending=[r for r in rows if not path_for(out,r).exists()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        active={};halt=False
        while pending or active:
            while pending and len(active)<workers and not halt:
                if spent+(len(active)+1)*RESERVE_PER_CALL>budget:halt=True;break
                row=pending.pop(0);active[pool.submit(attempt,out,row,gt[row['report_id'],row['field_id']],key)]=row
            if not active:
                if pending:raise ValueError('Budget reserve reached; stopped before sending')
                break
            done,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                row=active.pop(future);r=future.result();finished.append(r)
                if r['status']=='succeeded':spent+=r['usage']['estimated_usd']
                if r['status']!='succeeded' or not r.get('valid'):halt=True
            s=summarize(out,allrows,phase,'review_needed' if halt else 'running',finished+[{'status':'in_flight'} for _ in active]);print(json.dumps(s),flush=True)
            if halt and not active:raise ValueError('Stopped for result/budget review; successful calls retained')
    return summarize(out,allrows,phase,'phase_completed')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--canary',action='store_true');p.add_argument('--workers',type=int,default=4);p.add_argument('--max-estimated-usd',type=float,default=10);p.add_argument('--env-file',type=Path,default=Path('/Users/jeremy/Desktop/NYU/Capstone/Capstone-14/.env'));args=p.parse_args()
    rows,gt=validate_rows(args.root);out=args.root/'execution_20261112';out.mkdir(exist_ok=True)
    if not 1<=args.workers<=4 or not math.isfinite(args.max_estimated_usd) or args.max_estimated_usd<=0:raise ValueError('Invalid workers/budget')
    # Process lock prevents two controllers from billing the same missing call.
    import fcntl
    with (out/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not args.execute:
            print(json.dumps({'mode':'offline','pilot_calls':128,'full_calls':640,'model':MODEL,'manifest_checks':'passed','estimated_cost_guard_usd':args.max_estimated_usd}));return
        key=load_key(args.env_file)
        write(out/'authorization.json',{'user_instruction':'Run pilot; if no problems, complete remaining conditions','new_calls_max':640,'pilot_calls':128,'workers':args.workers,'estimated_cost_guard_usd':args.max_estimated_usd,'cost_rate_status':'historical frozen estimate, not invoice','new_video_calls':0,'utc_start_unix':time.time()})
        try:
            pilot=[r for r in rows if r['pilot']]
            if args.canary:
                run_phase(out,pilot[:1],rows,gt,key,'canary',1,args.max_estimated_usd)
                summarize(out,rows,'canary','canary_completed');return
            run_phase(out,pilot,rows,gt,key,'pilot',args.workers,args.max_estimated_usd)
            pilot_records=current_records(out,pilot)
            if len(pilot_records)!=128 or any(not r['valid'] for r in pilot_records):raise ValueError('Pilot coverage/schema gate failed')
            projected=2*sum(r['usage']['estimated_usd'] for r in pilot_records)*5+args.workers*RESERVE_PER_CALL
            if projected>args.max_estimated_usd:raise ValueError('Pilot full-run conservative cost projection exceeds guard')
            write(out/'pilot_gate.json',{'status':'passed','calls':128,'valid':128,'response_model':MODEL,'pilot_estimated_usd':sum(r['usage']['estimated_usd'] for r in pilot_records),'conservative_full_projection_usd':projected,'checks':['manifest exact','no media','model version','STOP finish','strict single target JSON','usage nonnegative','successful calls not repeated'],'continue_authorized':True})
            print('PILOT GATE PASSED',flush=True)
            run_phase(out,rows,rows,gt,key,'full',args.workers,args.max_estimated_usd)
            summarize(out,rows,'full','completed')
        except Exception as e:
            summarize(out,rows,'stopped','review_needed');write(out/'stop_reason.json',{'type':type(e).__name__,'reason':str(e),'time':time.time()});raise
if __name__=='__main__':main()
