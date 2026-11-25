"""Offline Argon oracle analysis, paired recording bootstrap (seed20261124, B2000).
Pilot and remaining-run scores are unioned without dropping failed/missing answers.
"""
import csv
import gzip
import json
import re
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'results'))
from cc4d_oracle_order import step_group, visible_target, parsed_bool
OUT=ROOT/'results/cc4d_oracle_argon_20261125'


def load_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def labels(value):
    if not isinstance(value,list) or any(not isinstance(v,str) or v not in 'ABCDEFG' or len(v)!=1 for v in value):
        return None
    return set(value)-{'F','G'}


class Cluster:
    def __init__(self,recordings,draws=2000,seed=20261124):
        ids=sorted(set(recordings));self.index=np.searchsorted(ids,recordings);self.n=len(ids)
        samples=np.random.default_rng(seed).integers(0,self.n,size=(draws,self.n))
        self.weights=np.array([np.bincount(s,minlength=self.n) for s in samples])

    def rate(self,value,mask):
        mask=np.asarray(mask,bool);value=np.asarray(value,float)
        num=np.bincount(self.index,weights=mask*value,minlength=self.n)
        den=np.bincount(self.index,weights=mask.astype(float),minlength=self.n)
        bn=self.weights@num;bd=self.weights@den
        estimate=num.sum()/den.sum() if den.sum() else np.nan
        boot=np.divide(bn,bd,out=np.full(len(bd),np.nan),where=bd>0)
        return estimate,boot


def newest(run):
    return sorted((ROOT/'runs'/run/'evaluations').glob('*/field_scores.jsonl'))[-1]


def main():
    truths=load_rows(ROOT/'data/cc4d_oracle_single_v1/ground_truth.jsonl')
    truth={(r['report_id'].split('__')[0],r['field_id']):r for r in truths}
    boolkeys=sorted(k for k in truth if k[1].endswith('_correct'))
    etkeys=sorted(k for k in truth if k[1].endswith('_error_types'))
    assert len(boolkeys)==951 and len(etkeys)==950
    oracle={};sources=[];statuses={}
    for run in ['argon_oracle_pilot20','argon_oracle_remaining931']:
        p=newest(run);sources.append(str(p))
        for r in load_rows(p):
            key=(r['report_id'].split('__')[0],r['field_id'])
            assert key not in oracle
            oracle[key]=r['predicted'];statuses[key]=r.get('outcome')
    assert set(oracle)==set(truth), 'all 1901 fields, including failures, must remain in analysis'
    baseline={};baseline_sources={}
    oldrows=json.load(gzip.open(ROOT/'results/cc4d_k32_v1_gemini4_argon_20261109/rows.json.gz','rt'))
    for name,k in [('k8','8'),('kall','all')]:
        raw=ROOT/'evidence'/('baseline_fwd_'+name+'_scores.jsonl')
        if raw.exists():
            baseline[name]={(r['report_id'],r['field_id']):r['predicted'] for r in load_rows(raw) if (r['report_id'],r['field_id']) in truth}
            baseline_sources[name]=str(raw)
        else:
            vals={}
            for o,kk,rep,f,typ,ok,ab,nv,*_ in oldrows:
                key=(rep,f)
                if o=='fwd' and kk==k and key in truth and f.endswith('_correct'):
                    t=truth[key]['target'];vals[key]='not_visible' if nv else t if ok else not t
            baseline[name]=vals
            baseline_sources[name]='rows reconstructed bool only; assumes non-abstaining incorrect bool is opposite GT'
        assert all(key in baseline[name] for key in boolkeys)
    data={'oracle':oracle,**baseline}
    cluster=Cluster([r for r,f in boolkeys]);allmask=np.ones(951,bool)
    t=np.array([truth[k]['target'] for k in boolkeys],bool)
    groups=np.array([step_group(truth[k]['target'],[e['tag'] for e in truth[k]['evidence']['errors']]) for k in boolkeys])
    out=[];estimates={};boots={}
    def save(run,metric,estimate,boot,n):
        valid=boot[np.isfinite(boot)]
        lo,hi=np.percentile(valid,[2.5,97.5]) if len(valid) else (np.nan,np.nan)
        out.append({'run':run,'metric':metric,'n':int(n),'value':float(estimate*100),'ci_low':float(lo*100),'ci_high':float(hi*100),'valid_bootstraps':len(valid)})
        estimates[run,metric]=estimate;boots[run,metric]=boot
    for name,answers in data.items():
        pred=np.array([parsed_bool(answers[k]) for k in boolkeys],object)
        for label,target,mask in [
            ('standard',t,allmask),
            ('order_only_relabel',t|(groups=='Order only'),allmask),
            ('exclude_order',t,~np.isin(groups,['Order only','Order + other']))]:
            right=np.array([p is bool(g) for p,g in zip(pred,target)])
            pos,pb=cluster.rate(right,mask&target);neg,nb=cluster.rate(right,mask&~target)
            save(name,'balanced_accuracy_'+label,(pos+neg)/2,(pb+nb)/2,mask.sum())
            acc,ab=cluster.rate(right,mask);save(name,'accuracy_'+label,acc,ab,mask.sum())
            save(name,'correct_recall_'+label,pos,pb,(mask&target).sum())
            save(name,'wrong_recall_'+label,neg,nb,(mask&~target).sum())
        for metric,values in [('answers_true',pred==True),('answers_not_visible',np.array([answers[k]=='not_visible' for k in boolkeys])),('invalid_or_missing_bool',np.array([p is None and answers[k]!='not_visible' for p,k in zip(pred,boolkeys)]))]:
            e,b=cluster.rate(values,allmask);save(name,metric,e,b,951)
        right=np.array([p is bool(g) for p,g in zip(pred,t)])
        for group in ['done correctly','Order only','Order + other','no Order']:
            mask=groups==group;e,b=cluster.rate(right,mask);save(name,'accuracy_group_'+group,e,b,mask.sum())
        for tag in ['Order','Technique','Preparation','Measurement','Timing','Temperature']:
            mask=np.array([any(tag in e['tag'] for e in truth[k]['evidence']['errors']) for k in boolkeys])&~t
            e,b=cluster.rate(pred==False,mask);save(name,'flags_wrong_given_'+tag,e,b,mask.sum())
    for name in ['k8','kall']:
        for metric in ['balanced_accuracy_standard','balanced_accuracy_order_only_relabel','balanced_accuracy_exclude_order','accuracy_standard']:
            save('oracle - '+name,metric,estimates['oracle',metric]-estimates[name,metric],boots['oracle',metric]-boots[name,metric],951 if 'exclude_order' not in metric else int((~np.isin(groups,['Order only','Order + other'])).sum()))
    for name in data:
        metric='balanced_accuracy_order_only_relabel'
        save(name,'balanced_accuracy_relabel_change',estimates[name,metric]-estimates[name,'balanced_accuracy_standard'],boots[name,metric]-boots[name,'balanced_accuracy_standard'],951)
    ec=Cluster([r for r,f in etkeys]);emask=np.ones(len(etkeys),bool)
    targets=[set(truth[k]['target'])-{'F','G'} for k in etkeys]
    for name,answers in data.items():
        if not all(k in answers for k in etkeys):continue
        ps=[labels(answers[k]) for k in etkeys]
        for metric,values in [('A_E_exact_match',np.array([p is not None and p==g for p,g in zip(ps,targets)])),('A_E_answers_empty',np.array([p==set() for p in ps])),('A_E_abstention',np.array([answers[k]=='not_visible' for k in etkeys]))]:
            e,b=ec.rate(values,emask);save(name,metric,e,b,len(etkeys))
        for letter,tag in zip('ABCDE',['Preparation','Measurement','Technique','Timing','Temperature']):
            present=np.array([letter in g for g in targets]);chosen=np.array([p is not None and letter in p for p in ps])
            for metric,mask in [('recall',present),('false_positive_rate',~present)]:
                e,b=ec.rate(chosen,mask);save(name,tag+'_'+metric,e,b,mask.sum())
    e,b=ec.rate(np.array([not g for g in targets]),emask);save('always_empty','A_E_exact_match',e,b,len(etkeys))
    for name in ['k8','kall','always_empty']:
        metric='A_E_exact_match'
        if (name,metric) in estimates:
            save('oracle - '+name,metric,estimates['oracle',metric]-estimates[name,metric],boots['oracle',metric]-boots[name,metric],len(etkeys))
    OUT.mkdir(exist_ok=True)
    with (OUT/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    perstep=[]
    for i,k in enumerate(boolkeys):
        perstep.append({'recording':k[0],'field':k[1],'target':bool(t[i]),'group':str(groups[i]),**{name:answers[k] for name,answers in data.items()}})
    (OUT/'paired_bool.json').write_text(json.dumps(perstep,indent=2)+'\n')
    (OUT/'analysis.json').write_text(json.dumps({'bootstrap':{'unit':'recording','draws':2000,'seed':20261124,'recordings':64},'n_bool':951,'n_error_types':950,'sources':sources,'baseline_sources':baseline_sources,'metrics':out},indent=2,allow_nan=False)+'\n')
    for r in out:
        if r['metric'].startswith(('balanced_accuracy','A_E_exact')):print(json.dumps(r))

if __name__=='__main__':main()
