"""Recompute Q2 (accuracy per question subtype) from data/per_field.csv; no API calls or packages.
python results/hd_epic_q2_full_v1_gemini35_flash_lite_20261022/analyze.py"""
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED, DRAWS = 20261020, 10000  # same estimator settings as the E1 report
NAMES = {'fine_grained_action_localization': 'Localization', 'fine_grained_how_recognition': 'How',
         'fine_grained_why_recognition': 'Why', 'ingredient_ingredient_weight': 'Weight'}


def q2(rows):
    """Video-weighted accuracy per subtype with a 95% percentile video-bootstrap interval."""
    out, rng = [], random.Random(SEED)
    for cat, name in NAMES.items():
        by_video = defaultdict(list)
        for r in rows:
            if r['category'] == cat:
                by_video[r['source_video']].append(r['correct'] == 'True')
        if not by_video:
            continue
        rates = [statistics.mean(v) for _, v in sorted(by_video.items())]
        draws = sorted(statistics.mean(rng.choices(rates, k=len(rates))) for _ in range(DRAWS))
        correct = sum(sum(v) for v in by_video.values())
        total = sum(len(v) for v in by_video.values())
        out.append({'subtype': name, 'video_weighted_accuracy': statistics.mean(rates),
                    'ci95': [draws[int(0.025 * DRAWS)], draws[int(0.975 * DRAWS) - 1]],
                    'pooled_correct': correct, 'questions': total, 'videos': len(by_video)})
    return out


def main():
    with (ROOT / 'data/per_field.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3212 + 514
    result = {
        'full': q2(rows),  # primary: E1 (all 3212) + E1q2 (514 new)
        'e1_only': q2([r for r in rows if r['experiment'] == 'E1']),  # previous Q2, for reference
        # same subtypes split by window origin: a large gap would point to a window effect
        'q40_windows_only': q2([r for r in rows if r['window'] == 'q40_v1' and not r['field_id'].startswith('loc_')]),
        'new_windows_only': q2([r for r in rows if r['window'] == 'new']),
    }
    with (ROOT / 'data/per_call.csv').open(newline='') as f:
        calls = list(csv.DictReader(f))
    result['new_calls'] = {'calls': len(calls), 'cost_usd': sum(float(c['cost_usd']) for c in calls),
                           'median_input_tokens': statistics.median(int(c['input_tokens']) for c in calls)}
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for key in ('full', 'e1_only', 'q40_windows_only', 'new_windows_only'):
        print(f'\n{key}')
        for r in result[key]:
            print(f"  {r['subtype']:<12} {r['video_weighted_accuracy']:.2%}  "
                  f"[{r['ci95'][0]:.2%}, {r['ci95'][1]:.2%}]  {r['pooled_correct']}/{r['questions']}  "
                  f"videos={r['videos']}")
    print('\nnew calls', result['new_calls'])


if __name__ == '__main__':
    main()
