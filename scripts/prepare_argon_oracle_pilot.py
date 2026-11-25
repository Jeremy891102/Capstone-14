"""Offline audit and frozen pilot; no model calls. Run from repository root."""
import collections
import gzip
import hashlib
import json
from pathlib import Path


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def main():
    reports = rows('data/cc4d_oracle_single_v1/reports.jsonl')
    truths = rows('data/cc4d_oracle_single_v1/ground_truth.jsonl')
    old = {r['id']: r for r in rows('evidence/old_oracle_reports.jsonl')}
    gt = {(r['report_id'], r['field_id']): r for r in truths}
    assert len(reports) == 951 and len(truths) == 1901
    assert set(old) == {r['id'] for r in reports}
    groups = collections.defaultdict(list)
    audit = {}
    for r in reports:
        assert r['input']['videos'] == old[r['id']]['input']['videos']
        assert set(r['input']['context']) == {'task_name', 'step', 'error_definitions'}
        f = r['fields'][0]['id']
        g = gt[r['id'], f]
        tags = {e['tag'].removesuffix(' Error') for e in g['evidence']['errors']}
        group = ('done correctly' if g['target'] else 'Order only' if tags == {'Order'}
                 else 'Order + other' if 'Order' in tags else 'no Order')
        v = r['input']['videos'][0]['time_range']
        audit[r['id']] = {'group': group, 'seconds': v['end'] - v['start'], 'source_report': r['metadata']['source_report']}
        groups[group].append(r)
    counts = {g: len(rs) for g, rs in groups.items()}
    assert counts == {'done correctly': 411, 'Order only': 104, 'Order + other': 94, 'no Order': 342}
    selected, used = [], set()
    for group in ['done correctly', 'Order only', 'Order + other', 'no Order']:
        candidates = sorted(groups[group], key=lambda r: (audit[r['id']]['seconds'], r['id']))
        for i in range(5):
            lo, hi = len(candidates)*i//5, len(candidates)*(i+1)//5
            pool = candidates[lo:hi]
            center = (lo+hi-1)/2
            ordered = sorted(enumerate(pool, lo), key=lambda x: (abs(x[0]-center), x[1]['id']))
            pick = next(r for _, r in ordered if r['metadata']['source_report'] not in used)
            selected.append(pick)
            used.add(pick['metadata']['source_report'])
    assert len(selected) == 20 and len(used) == 20
    out = Path('data/cc4d_oracle_single_pilot20'); out.mkdir(exist_ok=False)
    ids = {r['id'] for r in selected}
    for name, values in [('reports', selected), ('ground_truth', [g for g in truths if g['report_id'] in ids])]:
        (out/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in values))
    (out/'manifest.json').write_bytes(Path('data/cc4d_oracle_single_v1/manifest.json').read_bytes())
    # Boolean predictions can be reconstructed from GT + correctness + abstention; multi-choice cannot.
    orig = {(r['report_id'].split('__')[0], r['field_id']): r['target'] for r in truths if r['field_id'].endswith('_correct')}
    baseline = {}
    raw = json.load(gzip.open('results/cc4d_k32_v1_gemini4_argon_20261109/rows.json.gz','rt'))
    for order, k in [('fwd','8'), ('fwd','all'), ('rev','8'), ('rev','all')]:
        items = []
        for o, kk, rep, field, typ, ok, ab, nv, *_ in raw:
            if o == order and kk == k and (rep,field) in orig:
                t = orig[rep,field]
                p = None if nv else t if ok else not t
                items.append((t,p,ok))
        assert len(items) == 951 and len({(r[2], r[3]) for r in raw if r[0]==order and r[1]==k and (r[2],r[3]) in orig}) == 951
        pos = [ok for t,p,ok in items if t]; neg = [ok for t,p,ok in items if not t]
        baseline[order+'_'+k] = {'n':len(items), 'accuracy':sum(ok for t,p,ok in items)/len(items), 'true_rate':sum(p is True for t,p,ok in items)/len(items), 'correct_recall':sum(pos)/len(pos), 'wrong_recall':sum(neg)/len(neg), 'balanced_accuracy':(sum(pos)/len(pos)+sum(neg)/len(neg))/2}
    evidence = Path('evidence')
    (evidence/'cohort.json').write_text(json.dumps(audit,indent=2)+'\n')
    (evidence/'readiness_audit.json').write_text(json.dumps({'counts':counts,'reports':951,'fields':1901,'recordings':len({r['metadata']['source_report'] for r in reports}), 'identical_old_clips':True,'baseline':baseline,'pilot':{r['id']:audit[r['id']] for r in selected}, 'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('data/cc4d_oracle_single_v1/reports.jsonl'),out/'reports.jsonl',Path('experiments/cc4d_oracle_argon/prompts/user.txt')]}},indent=2)+'\n')
    print(json.dumps({'counts':counts,'baseline':baseline,'pilot_calls':len(selected)},indent=2))

if __name__ == '__main__':
    main()
