"""Offline operational summary of frozen pilot responses; never invokes a model."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    run = ROOT/'runs/argon_oracle_pilot20'
    audit = json.loads((ROOT/'evidence/readiness_audit.json').read_text())
    requests = {r['call_id']:r for r in map(json.loads,(run/'snapshot/requests.jsonl').read_text().splitlines())}
    pilot = audit['pilot']; rows = []
    for state_path in sorted((run/'calls').glob('*.json')):
        s = json.loads(state_path.read_text()); req = requests[s['call_id']]
        row = {'report_id':s['report_id'],'call_id':s['call_id'],'status':s['status'],**pilot[s['report_id']]}
        row['attempts'] = len(s['attempts'])
        row['format_valid'] = False
        if s['status'] == 'succeeded':
            a = s['attempts'][-1]
            resp = json.loads((run/a['response_file']).read_text())['response']
            u = resp.get('usage') or {}
            row.update({k:u.get(k) for k in ['input_tokens','output_tokens','reasoning_tokens','cached_input_tokens']})
            row['latency_s'] = a['latency_s']
            row['model_version'] = resp.get('model_version')
            try:
                ans=json.loads(resp['text'])
                assert set(ans)==set(req['field_ids'])
                for k,v in ans.items():
                    assert v == 'not_visible' or (type(v) is bool if k.endswith('_correct') else isinstance(v,list) and len(v)==len(set(v)) and all(x in list('ABCDE') for x in v))
                row['format_valid']=True
                row['answers']=json.dumps(ans)
            except (ValueError,TypeError,AssertionError,KeyError):
                row['answers']=resp.get('text')
            inp=u.get('input_tokens') or 0; cache=u.get('cached_input_tokens') or 0
            row['gross_usd']=((inp-cache)*4+cache*.2+((u.get('output_tokens') or 0)+(u.get('reasoning_tokens') or 0))*20)/1e6
        rows.append(row)
    completed=[r for r in rows if r['status']=='succeeded']
    summary={'calls':len(rows),'statuses':dict(Counter(r['status'] for r in rows)), 'format_valid':sum(r['format_valid'] for r in rows), 'tokens':{k:sum(r.get(k) or 0 for r in completed) for k in ['input_tokens','output_tokens','reasoning_tokens','cached_input_tokens']}, 'gross_usd':sum(r.get('gross_usd',0) for r in rows), 'model_versions':dict(Counter(r.get('model_version') for r in completed))}
    summary['conditional_credit_net_usd']=summary['gross_usd']/2
    if len(completed)==20:
        counts=audit['counts']
        groupmeans={g:sum(r['gross_usd'] for r in completed if r['group']==g)/5 for g in counts}
        summary['full951_gross_group_weighted_usd']=sum(counts[g]*groupmeans[g] for g in counts)
        summary['full951_conditional_credit_net_usd']=summary['full951_gross_group_weighted_usd']/2
        summary['group_mean_gross_usd']=groupmeans
        # Adjust input to the cohort mean length using a pooled linear fit; output group means retained.
        import numpy as np
        cohort=json.loads((ROOT/'evidence/cohort.json').read_text())
        x=np.array([r['seconds'] for r in completed]); y=np.array([r['input_tokens'] for r in completed])
        intercept,slope=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
        n=951; total_seconds=sum(r['seconds'] for r in cohort.values())
        predinput=float(n*intercept+slope*total_seconds)
        output=sum(counts[g]*sum((r['output_tokens'] or 0)+(r['reasoning_tokens'] or 0) for r in completed if r['group']==g)/5 for g in counts)
        summary['input_tokens_per_second_fit']={'intercept':float(intercept),'slope':float(slope)}
        summary['full951_gross_length_adjusted_usd']=(predinput*4+output*20)/1e6
        summary['pilot_mean_seconds']=float(x.mean());summary['cohort_mean_seconds']=total_seconds/n
    out=ROOT/'results/cc4d_oracle_argon_pilot_20261125';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'per_call.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
