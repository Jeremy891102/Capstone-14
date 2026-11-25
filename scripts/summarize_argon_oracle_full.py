"""Verify and summarize both completed runs; no API calls."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from video_report.evaluation.parser import parse_response
from run_argon_oracle_full import gross_cost

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cc4d_oracle_argon_20261125'


def main():
    cohort=json.loads((ROOT/'evidence/cohort.json').read_text());rows=[];seen=set();all_fields=set();versions=[];responses=set();raw_hashes={};sessions=[]
    for name,n in [('argon_oracle_pilot20',20),('argon_oracle_remaining931',931)]:
        run=ROOT/'runs'/name
        meta=json.loads((run/'run.json').read_text());assert meta['state']=='completed'
        calls=[json.loads(p.read_text()) for p in (run/'calls').glob('*.json')];assert len(calls)==n
        sessions.extend(meta['sessions'])
        for c in calls:
            assert c['status']=='succeeded' and len(c['attempts'])==1
            assert c['report_id'] not in seen;seen.add(c['report_id'])
            for f in c['field_ids']:
                assert (c['report_id'],f) not in all_fields;all_fields.add((c['report_id'],f))
            key=hashlib.sha256(c['call_id'].encode()).hexdigest()[:16]
            rawpath=run/'api_raw'/(key+'.json');raw=json.loads(rawpath.read_text())
            raw_hashes[str(rawpath.relative_to(ROOT))]=hashlib.sha256(rawpath.read_bytes()).hexdigest()
            payload=json.loads((run/'api_raw'/(key+'_request.json')).read_text())
            assert payload['call_id']==c['call_id']
            assert payload['body']['generationConfig']=={'temperature':0.0,'maxOutputTokens':8192,'responseMimeType':'application/json'}
            cost=gross_cost(raw.get('usageMetadata',{}));assert cost is not None
            resp=json.loads((run/c['attempts'][0]['response_file']).read_text())['response']
            p=parse_response(resp['text'],c['field_ids']);valid=p.status=='ok' and not p.missing and not p.extra_keys
            for f,v in p.answers.items():
                if v=='not_visible':continue
                valid=valid and (type(v) is bool if f.endswith('_correct') else isinstance(v,list) and all(isinstance(x,str) and x in list('ABCDE') for x in v) and len(set(v))==len(v))
            u=raw['usageMetadata'];versions.append(raw.get('modelVersion'))
            if raw.get('responseId'):
                assert raw['responseId'] not in responses;responses.add(raw['responseId'])
            rows.append({'run':name,'call_id':c['call_id'],'report_id':c['report_id'],**cohort[c['report_id']], 'fields':len(c['field_ids']),'strict_format_valid':bool(valid),'input_tokens':u['promptTokenCount'],'output_tokens':u['candidatesTokenCount'],'thinking_tokens':u.get('thoughtsTokenCount',0),'cached_tokens':u.get('cachedContentTokenCount',0),'gross_usd':cost,'latency_s':c['attempts'][0]['latency_s'],'model_version':raw.get('modelVersion')})
    assert len(seen)==951 and len(all_fields)==1901 and set(cohort)==seen
    summary={'calls':951,'fields':1901,'unique_recordings':len({r['source_report'] for r in rows}),'attempts':951,'failed_calls':0,'strict_format_valid_calls':sum(r['strict_format_valid'] for r in rows),'tokens':{k:sum(r[k] for r in rows) for k in ['input_tokens','output_tokens','thinking_tokens','cached_tokens']},'gross_usd':sum(r['gross_usd'] for r in rows),'model_versions':dict(Counter(versions)),'sessions':sessions,'budget_gross_usd':30,'pricing_usd_per_million':{'input':4,'cache':.2,'output_and_thinking':20},'pricing_source':'https://cloud.google.com/model-garden/pricing checked 2026-11-25; account credits/bill not verified'}
    summary['conditional_50pct_credit_net_usd']=summary['gross_usd']/2
    assert summary['gross_usd']<=30
    OUT.mkdir(exist_ok=True)
    (OUT/'execution.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'raw_response_sha256.json').write_text(json.dumps(raw_hashes,indent=2)+'\n')
    with (OUT/'per_call.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
