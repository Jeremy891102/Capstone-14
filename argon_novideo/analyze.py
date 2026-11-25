"""Offline paired analysis. Requires all 1,095 frozen batch responses."""
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

import batch_pilot as batch

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'results/hd_epic_argon_novideo_batching_20261125'
BASELINE = Path('/Users/jeremy/Documents/Codex/2026-10-06/git/outputs/baseline_complete_20261109/per_field.csv')
OLD = Path('/Users/jeremy/Documents/Codex/2026-10-06/git/outputs/argon_reference_novideo_full73_20261112')
KS = [1, 4, 8, 16, 32]
SEED = 20261027
DRAWS = 10000


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def estimate(videos):
    videos = np.asarray(videos, dtype=float)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(videos), (DRAWS, len(videos)))
    samples = videos[indices].mean(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return tuple(float(x * 100) for x in [videos.mean(), low, high])


def load_baseline():
    batch.validate()  # Validates every frozen input SHA, including all five k files.
    gt = {(r['report_id'], r['field_id']): r['target']
          for r in read_rows(batch.FROZEN / 'evaluation_only.jsonl')}
    with BASELINE.open() as stream:
        rows = [r for r in csv.DictReader(stream)
                if r['model'] == 'argon' and (r['report_id'], r['field_id']) in gt
                and int(r['batch_size']) in KS]
    keys = {(int(r['batch_size']), r['report_id'], r['field_id']) for r in rows}
    assert len(rows) == len(keys) == 2336 * 5
    for r in rows:
        assert r['target'] == gt[r['report_id'], r['field_id']]
        assert (r['prediction'] == r['target']) == (r['correct'].lower() == 'true')
    assert Counter(r['report_id'] for r in rows) == Counter({r: 160 for r, _ in gt})
    assert sum(r['correct'].lower() == 'true' for r in rows if r['batch_size'] == '1') == 1402
    return rows, gt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-baseline', action='store_true')
    args = p.parse_args()
    video_rows, gt = load_baseline()
    if args.check_baseline:
        print(json.dumps({'baseline_targets_per_k': 2336, 'videos': 73,
                          'correct_counts': {k: sum(r['correct'].lower() == 'true' for r in video_rows
                                                  if r['batch_size'] == str(k)) for k in KS},
                          'bootstrap_draws': DRAWS, 'seed': SEED}))
        return
    state = json.loads((OUT / 'status.json').read_text())
    assert state['status'] == 'completed' and state['succeeded'] == 1095 and state['failed_or_uncertain'] == 0
    batch.OUT = OUT
    no_rows = []; calls = []; source_hashes = {}; raw_hashes = {}
    historical_rows = {(r['report_id'], r['field_id']): r
                       for r in read_rows(OLD / 'full_requests.jsonl') if r['condition'] == 'single'}
    assert set(historical_rows) == set(gt)
    inputs = OUT / 'inputs'; inputs.mkdir(exist_ok=True)
    for path in batch.FROZEN.glob('*'):
        if path.is_file():
            shutil.copyfile(path, inputs / path.name)
    for k in KS:
        rows = read_rows(batch.FROZEN / f'k{k}.jsonl')
        for row in rows:
            if k == 1:
                fid = row['field_ids'][0]
                oldrow = historical_rows[row['report_id'], fid]
                assert oldrow['body'] == row['body']
                source = batch.runner.path_for(OLD / 'execution_20261112', oldrow)
                record = json.loads(source.read_text())
                batch.runner.check_reuse(record, oldrow)
                assert record['target'] == gt[row['report_id'], fid]
                predictions = {fid: record['prediction']}
                dest = OUT / 'historical_k1_calls' / source.name
                dest.parent.mkdir(exist_ok=True)
                shutil.copyfile(source, dest)
            else:
                source = batch.path_for(row)
                record = json.loads(source.read_text())
                batch.check_reuse(record, row)
                predictions = record['predictions']
                dest = source
            raw_hashes[str(dest.relative_to(OUT))] = hashlib.sha256(dest.read_bytes()).hexdigest()
            assert set(predictions) == set(row['field_ids'])
            for fid in row['field_ids']:
                target = gt[row['report_id'], fid]
                no_rows.append(dict(model='argon', batch_size=k, report_id=row['report_id'], field_id=fid,
                                    call_id=row['call_id'], target=target, prediction=predictions[fid],
                                    correct=predictions[fid] == target, backend='direct_vertex_global'))
            calls.append(dict(model='argon', batch_size=k, report_id=row['report_id'], call_id=row['call_id'],
                              historical_reuse=k == 1, source_record=str(dest.relative_to(OUT)),
                              api_latency_s=record['api_latency_s'], **record['usage']))
    assert len(calls) == 3431 and len(no_rows) == 11680
    assert len({(r['batch_size'], r['report_id'], r['field_id']) for r in no_rows}) == 11680
    assert len(list((OUT / 'calls').glob('*.json'))) == 1095
    videos = sorted({r for r, _ in gt})
    video_index = {(int(r['batch_size']), r['report_id'], r['field_id']): r for r in video_rows}
    paired = []
    for r in no_rows:
        other = video_index[r['batch_size'], r['report_id'], r['field_id']]
        paired.append({**r, 'video_prediction': other['prediction'],
                       'video_correct': other['correct'].lower() == 'true'})
    arrays = {}
    per_video = []
    for k in KS:
        selected = [r for r in paired if r['batch_size'] == k]
        assert Counter(r['report_id'] for r in selected) == Counter({v: 32 for v in videos})
        va = []; na = []
        for v in videos:
            rs = [r for r in selected if r['report_id'] == v]
            a = sum(r['video_correct'] for r in rs) / 32
            b = sum(r['correct'] for r in rs) / 32
            va.append(a); na.append(b)
            per_video.append(dict(batch_size=k, report_id=v, questions=32, video_accuracy=a,
                                  novideo_accuracy=b, video_gain=a - b))
        arrays[k] = (np.array(va), np.array(na))
    accuracy = []
    for k, (va, na) in arrays.items():
        row = dict(batch_size=k, videos=73, questions=2336,
                   video_correct=int(round(va.sum() * 32)), novideo_correct=int(round(na.sum() * 32)))
        for name, values in [('video', va), ('novideo', na), ('gain', va - na)]:
            point, low, high = estimate(values)
            row.update({name + '_pct_or_pp': point, name + '_ci_low': low, name + '_ci_high': high})
        accuracy.append(row)
    contrasts = []
    for later, earlier in [(8, 1), (32, 8)]:
        vdelta = arrays[later][0] - arrays[earlier][0]
        ndelta = arrays[later][1] - arrays[earlier][1]
        for name, values in [('video', vdelta), ('novideo', ndelta), ('video_gain', vdelta - ndelta)]:
            point, low, high = estimate(values)
            contrasts.append(dict(contrast=f'{name}_k{later}_minus_k{earlier}',
                                  estimate_pp=point, ci_low_pp=low, ci_high_pp=high))
    write_csv(OUT / 'per_field.csv', no_rows)
    write_csv(OUT / 'baseline_per_field.csv', video_rows)
    write_csv(OUT / 'paired_per_field.csv', paired)
    write_csv(OUT / 'per_call.csv', calls)
    write_csv(OUT / 'per_video.csv', per_video)
    write_csv(OUT / 'accuracy_summary.csv', accuracy)
    write_csv(OUT / 'contrasts.csv', contrasts)
    for path in [BASELINE, OLD / 'full_requests.jsonl', OLD / 'evaluation_only.jsonl', Path(__file__),
                 ROOT / 'batch_pilot.py', ROOT / 'full_run.py', ROOT / 'code/historical_runner.py']:
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    batch_cost = sum(r['estimated_usd'] for r in calls if r['batch_size'] != 1)
    k1_cost = sum(r['estimated_usd'] for r in calls if r['batch_size'] == 1)
    trial_cost = json.loads((ROOT / 'trial_results/status.json').read_text())['estimated_success_usd']
    failures = []
    for path in (OUT / 'attempt_history').glob('**/*.json'):
        record = json.loads(path.read_text())
        failures.append({'path': str(path.relative_to(OUT)), 'http_status': record.get('http_status'),
                         'billing': record.get('billing'), 'input_sha256': record.get('input_sha256')})
        raw_hashes[str(path.relative_to(OUT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {'status': 'verified_completed', 'accuracy': accuracy, 'contrasts': contrasts,
               'bootstrap': {'unit': 'video', 'videos': 73, 'questions_per_video': 32,
                             'draws': DRAWS, 'seed': SEED, 'ci': 'percentile 95%, paired across conditions and k',
                             'weights': 'equal video weights; identical to question weights at 32 questions/video',
                             'multiple_comparison_adjusted': False},
               'calls': {'new_batch_including_pilot': 1095, 'reused_pilot': 20,
                         'historical_k1': 2336, 'separate_drift_trial': 320, 'failed_in_accepted_results': 0,
                         'archived_failed_attempts': failures, 'total_batch_http_attempts': 1095 + len(failures)},
               'cost_estimates_usd': {'batch_including_pilot': batch_cost,
                                      'separate_drift_trial': trial_cost,
                                      'new_work_total': batch_cost + trial_cost,
                                      'historical_k1_excluded_from_new_cost': k1_cost,
                                      'precredit_new_work_equivalent': 2 * (batch_cost + trial_cost),
                                      'unknown_failed_billing_excluded': len(failures),
                                      'invoice_verified': False},
               'source_sha256': source_hashes,
               'limitations': ['Video baselines and no-video controls were not collected contemporaneously.',
                               'k1 no-video primary curve reuses historical full73 data; current subset10 is a separate check.',
                               'Subset10 answer agreement 55.94%; similar accuracy does not exclude prediction drift.',
                               'Model alias and wire settings do not prove unchanged provider internals.',
                               'Bootstrap intervals measure video sampling variation, not systematic time confounding.',
                               'Video-gain contrasts are protocol interactions, not proof of improved visual perception.']}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (OUT / 'raw_response_hashes.json').write_text(json.dumps(raw_hashes, indent=2) + '\n')
    plot(accuracy)
    print(json.dumps(summary, indent=2))


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.2), sharex=True,
                             gridspec_kw={'height_ratios': [1.4, 1]})
    x = np.arange(5)
    for name, label, color in [('video', 'With video (historical)', '#137f79'),
                               ('novideo', 'Without video', '#555b65')]:
        y = np.array([r[name + '_pct_or_pp'] for r in rows])
        low = np.array([r[name + '_ci_low'] for r in rows])
        high = np.array([r[name + '_ci_high'] for r in rows])
        axes[0].errorbar(x, y, yerr=[y - low, high - y], marker='o', capsize=4, color=color, label=label)
    axes[0].axhline(20, color='#999999', linestyle=':', linewidth=1, label='Chance (20%)')
    axes[0].set(ylabel='Accuracy (%)', title='HD-EPIC Argon: with and without video')
    axes[0].legend(frameon=False, fontsize=9)
    y = np.array([r['gain_pct_or_pp'] for r in rows])
    low = np.array([r['gain_ci_low'] for r in rows])
    high = np.array([r['gain_ci_high'] for r in rows])
    axes[1].errorbar(x, y, yerr=[y - low, high - y], marker='o', capsize=4, color='#137f79', label='With video minus without video')
    axes[1].set(ylabel='Video gain (percentage points)', xlabel='Questions per API call (k)',
                title='Paired video gain')
    axes[1].set_xticks(x, KS)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(.12, .018, '2,336 shared targets / 73 videos; 95% paired video-bootstrap CIs (10,000 draws).\n'
             'Sources: frozen Vertex runs and raw no-video responses; non-contemporaneous controls.', fontsize=8)
    fig.tight_layout(rect=[0, .065, 1, 1])
    fig.savefig(OUT / 'accuracy_and_video_gain.png', dpi=180)
    fig.savefig(OUT / 'accuracy_and_video_gain.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
