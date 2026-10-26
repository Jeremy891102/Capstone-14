"""Build data/per_field.csv and data/per_call.csv from the E1 baseline CSVs and the E1q2 run.
Usage (from repo root, after copying the VM run dir locally):
python results/hd_epic_q2_full_v1_gemini35_flash_lite_20261022/collect.py <local copy of runs/E1q2>"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
E1 = REPO / 'results/hd_epic_q40_v1_gemini35_flash_lite_20261020/data'
DATA = REPO / 'data/hd_epic_q2_full_v1'
VIDEO_ROOT = '/home/azureuser/datasets/hd-epic/Videos/'
PRICE_IN, PRICE_OUT = 0.3e-6, 2.5e-6  # same pricing block as the E1 config
FIELD_COLS = ['report_id', 'field_id', 'call_id', 'outcome', 'correct', 'predicted', 'predicted_text',
              'target', 'target_text', 'detail', 'experiment', 'category', 'source_video', 'window']
CALL_COLS = ['experiment', 'call_id', 'report_id', 'input_tokens', 'output_tokens', 'cost_usd',
             'call_latency_s', 'source_video', 'window_start_s', 'window_end_s']


def jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def main(run):
    run = Path(run)
    reports = {r['id']: r for r in jl(DATA / 'reports.jsonl')}
    category = {(t['report_id'], t['field_id']): t['evidence']['category'] for t in jl(DATA / 'ground_truth.jsonl')}
    keep = set(category)  # how/why/weight from q40_v1 windows + all new windows
    with (E1 / 'per_field.csv').open(newline='') as f:
        fields = [dict(r, window='q40_v1') for r in csv.DictReader(f) if r['experiment'] == 'E1']
    assert len(fields) == 3212 and sum((r['report_id'], r['field_id']) in keep for r in fields) == 351
    scores = jl(next(run.glob('evaluations/*/field_scores.jsonl')))
    assert len(scores) == 514 and all(s['report_id'].startswith('hdepic-q2-') for s in scores)
    for s in scores:
        video = reports[s['report_id']]['input']['videos'][0]
        fields.append({**{k: s.get(k) for k in FIELD_COLS[:10]}, 'experiment': 'E1q2',
                       'category': category[(s['report_id'], s['field_id'])],
                       'source_video': VIDEO_ROOT + video['uri'], 'window': 'new'})
    calls = []
    for path in sorted((run / 'calls').glob('*.json')):
        c = json.loads(path.read_text())
        a = c['attempts'][c['final_attempt'] - 1]
        u = a['usage']
        video = reports[c['report_id']]['input']['videos'][0]
        calls.append({'experiment': 'E1q2', 'call_id': c['call_id'], 'report_id': c['report_id'],
                      'input_tokens': u['input_tokens'], 'output_tokens': u['output_tokens'],
                      'cost_usd': u['input_tokens'] * PRICE_IN + u['output_tokens'] * PRICE_OUT,
                      'call_latency_s': a.get('latency_s'), 'source_video': VIDEO_ROOT + video['uri'],
                      'window_start_s': video['time_range']['start'], 'window_end_s': video['time_range']['end']})
    assert len(calls) == 514
    (ROOT / 'data').mkdir(exist_ok=True)
    for name, cols, rows in [('per_field.csv', FIELD_COLS, fields), ('per_call.csv', CALL_COLS, calls)]:
        with (ROOT / 'data' / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
    print(len(fields), 'fields,', len(calls), 'new calls')


if __name__ == '__main__':
    main(sys.argv[1])
