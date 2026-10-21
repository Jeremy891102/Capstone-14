"""Analyze saved baseline evidence without API calls or third-party packages."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENT = {1: 'E1', 4: 'E2', 8: 'E3', 16: 'E4', 32: 'E5'}


def read_csv(name):
    with (ROOT / 'data' / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def yes(value):
    return value == 'True'


def matched_scores(fields, size):
    return {
        (row['report_id'], row['field_id']): row
        for row in fields
        if row['experiment'] == EXPERIMENT[size] and yes(row['matched_localization'])
    }


def summarize(fields, calls):
    q1 = []
    for size in EXPERIMENT:
        scores = matched_scores(fields, size)
        selected = [row for row in calls if row['experiment'] == EXPERIMENT[size]
                    and int(row['matched_localization_fields']) > 0]
        targets = len(scores)
        assert targets == 2336
        assert sum(int(row['matched_localization_fields']) for row in selected) == targets
        q1.append({'questions_per_call': size, 'correct': sum(yes(r['correct']) for r in scores.values()),
                   'fields': targets, 'accuracy': sum(yes(r['correct']) for r in scores.values()) / targets,
                   'cost_per_32_usd': sum(float(r['cost_usd']) for r in selected) / 73,
                   'api_seconds_per_32': sum(float(r['api_round_trip_s']) for r in selected) / 73})
    subtype = defaultdict(lambda: defaultdict(list))
    for row in fields:
        if row['experiment'] == 'E1':
            subtype[row['category']][row['source_video']].append(yes(row['correct']))
    q2 = []
    for category, groups in sorted(subtype.items()):
        correct = sum(sum(v) for v in groups.values())
        total = sum(len(v) for v in groups.values())
        q2.append({'subtype': category, 'videos': len(groups), 'fields': total,
                   'video_weighted_accuracy': statistics.mean(statistics.mean(v) for v in groups.values()),
                   'pooled_question_accuracy': correct / total, 'pooled_correct': correct})
    common = {}
    for experiment in ('E6_homogeneous', 'E6_mixed'):
        common[experiment] = {(r['report_id'], r['field_id']): yes(r['correct'])
                              for r in fields if r['experiment'] == experiment and yes(r['e6_common_target'])}
    pure, mixed = common.values()
    assert set(pure) == set(mixed) and len(pure) == 96
    transition = Counter(('correct' if pure[k] else 'wrong') + '_to_' +
                         ('correct' if mixed[k] else 'wrong') for k in pure)
    q3 = {'common_targets': 96, 'pure_correct': sum(pure.values()), 'mixed_correct': sum(mixed.values()),
          'mixed_minus_pure_pp': (sum(mixed.values()) - sum(pure.values())) / 96 * 100,
          'transitions': dict(transition)}
    return {'Q1': q1, 'Q2': q2, 'Q3': q3,
            'effective_calls': len(calls), 'effective_fields': len(fields),
            'effective_cost_usd': sum(float(r['cost_usd']) for r in calls)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('summary', help='Recompute Q1/Q2/Q3 and cost; print JSON.')
    flips = sub.add_parser('flips', help='Find the exact same targets that change correctness.')
    flips.add_argument('--from-size', type=int, choices=EXPERIMENT, default=8)
    flips.add_argument('--to-size', type=int, choices=EXPERIMENT, default=32)
    flips.add_argument('--output', type=Path, help='Write all paired targets as CSV.')
    trace = sub.add_parser('trace', help='Read a saved call with its exact prompt and response.')
    trace.add_argument('--experiment', choices=list(EXPERIMENT.values()) + ['E6_homogeneous', 'E6_mixed'], required=True)
    trace.add_argument('--call-id')
    trace.add_argument('--report-id')
    trace.add_argument('--field-id')
    args = parser.parse_args()
    if args.command == 'trace':
        if not any((args.call_id, args.report_id, args.field_id)):
            parser.error('trace requires --call-id, --report-id, or --field-id')
        with gzip.open(ROOT / 'data/all_model_inputs_outputs.jsonl.gz', 'rt') as stream:
            for line in stream:
                item = json.loads(line)
                request = item['request']
                if (item['experiment'] == args.experiment
                        and (args.call_id is None or request['call_id'] == args.call_id)
                        and (args.report_id is None or request['report_id'] == args.report_id)
                        and (args.field_id is None or args.field_id in request['field_ids'])):
                    print(json.dumps(item, ensure_ascii=False, indent=2))
                    return
        parser.error('No matching saved call. Check the experiment and field/report IDs.')
    fields, calls = read_csv('per_field.csv'), read_csv('per_call.csv')
    if args.command == 'summary':
        result = summarize(fields, calls)
        assert result['effective_calls'] == 4355 and result['effective_fields'] == 12940
        assert [r['correct'] for r in result['Q1']] == [771, 816, 818, 718, 623]
        assert abs(result['effective_cost_usd'] - 43.4853869) < 1e-7
        print(json.dumps(result, indent=2))
        return
    before, after = matched_scores(fields, args.from_size), matched_scores(fields, args.to_size)
    assert set(before) == set(after) and len(before) == 2336
    pairs = []
    for key in sorted(before):
        a, b = before[key], after[key]
        pairs.append({'report_id': key[0], 'field_id': key[1], 'from_size': args.from_size,
                      'to_size': args.to_size, 'from_prediction': a['predicted'], 'to_prediction': b['predicted'],
                      'target': a['target'], 'from_correct': yes(a['correct']), 'to_correct': yes(b['correct'])})
        assert a['target'] == b['target']
    counts = Counter(('correct' if r['from_correct'] else 'wrong') + '_to_' +
                     ('correct' if r['to_correct'] else 'wrong') for r in pairs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
    print(json.dumps({'from_size': args.from_size, 'to_size': args.to_size,
                      'shared_targets': len(pairs), 'transitions': dict(counts),
                      'paired_csv': str(args.output) if args.output else None}, indent=2))


if __name__ == '__main__':
    main()
