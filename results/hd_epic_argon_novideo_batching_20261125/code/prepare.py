"""Offline preparation; derives payloads from each frozen baseline runner."""
import ast
import copy
import hashlib
import json
from pathlib import Path

SOURCE = Path('/Users/jeremy/Documents/Codex/2026-10-06/git/outputs')
REFERENCE = SOURCE / 'research_next_step_20261111'
OUT = Path(__file__).resolve().parent / 'frozen'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def remove_media(body):
    result = copy.deepcopy(body)
    removed = 0
    for content in result['contents']:
        parts = content['parts']
        removed += sum('inlineData' in p or 'fileData' in p for p in parts)
        content['parts'] = [p for p in parts if 'inlineData' not in p and 'fileData' not in p]
        if any(set(p) != {'text'} for p in content['parts']):
            raise ValueError('Unexpected non-text part')
    if removed != 1:
        raise ValueError('Expected exactly one media part')
    return result


def dump_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def main():
    if OUT.exists():
        raise ValueError('Frozen directory exists; refusing overwrite')
    runs = {}
    sources = {}
    for k, date in [(1, '20261026'), (4, '20261026'), (8, '20261028'),
                    (16, '20261028'), (32, '20261028')]:
        run = SOURCE / f'vertex_argon_full_size{k}_{date}'
        runs[k] = read_rows(run / 'inputs/requests.jsonl')
        sources[k] = run
    cohort = {(r['request']['report_id'], f) for r in runs[32] for f in r['request']['field_ids']}
    assert len(cohort) == 2336 and len({r for r, _ in cohort}) == 73
    reference = [r for r in read_rows(REFERENCE / 'full_requests.jsonl') if r['condition'] == 'single']
    historical = {(r['report_id'], r['field_id']): r for r in reference}
    assert len(historical) == 320
    payloads = {}; targets = {}; provenance = {}; audit = {}
    for k, source_rows in runs.items():
        run = sources[k]
        # Extract only the original pure snapshot function; never execute runner main/imports.
        tree = ast.parse((run / 'full.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'snapshot_for')
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(run / 'full.py'), 'exec'), namespace)
        gt = {(g['report_id'], g['field_id']): g['target'] for g in read_rows(run / 'inputs/ground_truth.jsonl')}
        rows = []
        for source in source_rows:
            r = source['request']
            keys = [(r['report_id'], f) for f in r['field_ids']]
            if not all(key in cohort for key in keys):
                continue
            assert len(keys) == k
            for key in keys:
                assert key not in targets or targets[key] == gt[key]
                targets[key] = gt[key]
            video_body = namespace['snapshot_for'](source)
            body = remove_media(video_body)
            assert body['systemInstruction']['parts'] == [{'text': r['system']}]
            assert body['contents'][0]['parts'][-1] == {'text': r['user']}
            rows.append({'call_id': r['call_id'], 'report_id': r['report_id'],
                         'field_ids': r['field_ids'], 'batch_size': k, 'model': 'gemini-4-argon',
                         'body': body, 'body_sha256': digest(body),
                         'source_row_sha256': digest(source), 'video_body_sha256': digest(video_body)})
        pairs = [(r['report_id'], f) for r in rows for f in r['field_ids']]
        assert len(pairs) == len(set(pairs)) == 2336 and set(pairs) == cohort
        payloads[k] = rows
        audit[k] = {'calls': len(rows), 'targets': len(pairs)}
        for path in [run / 'inputs/requests.jsonl', run / 'inputs/ground_truth.jsonl', run / 'full.py']:
            provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    sequence = lambda rs: {rid: [f for r in rs if r['report_id'] == rid for f in r['field_ids']]
                           for rid, _ in cohort}
    assert all(sequence(payloads[k]) == sequence(payloads[32]) for k in [4, 8, 16])
    trial = [r for r in payloads[1] if (r['report_id'], r['field_ids'][0]) in historical]
    assert len(trial) == 320
    assert all(r['body'] == historical[r['report_id'], r['field_ids'][0]]['body'] for r in trial)
    OUT.mkdir()
    for k, rows in payloads.items():
        dump_rows(OUT / f'k{k}.jsonl', rows)
    dump_rows(OUT / 'k1_subset10.jsonl', trial)
    # Five distinct videos, evenly spaced through the frozen k32 report order.
    videos = [r['report_id'] for r in payloads[32]]
    pilot_videos = [videos[i] for i in [0, 18, 36, 54, 72]]
    pilot = [next(r for r in payloads[k] if r['report_id'] == rid)
             for k in [4, 8, 16, 32] for rid in pilot_videos]
    dump_rows(OUT / 'cost_pilot.jsonl', pilot)
    dump_rows(OUT / 'evaluation_only.jsonl', [{'report_id': r, 'field_id': f, 'target': targets[r, f]}
                                            for r, f in sorted(cohort)])
    dump_rows(OUT / 'historical_subset10.jsonl', reference)
    manifest = {'status': 'offline_verified', 'scope': audit, 'k1_subset10_exact_historical_bodies': 320,
                'cost_pilot_calls': 20, 'cost_pilot_videos': pilot_videos,
                'source_files_sha256': provenance,
                'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.jsonl')}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ['source_files_sha256', 'files_sha256']}))


if __name__ == '__main__':
    main()
