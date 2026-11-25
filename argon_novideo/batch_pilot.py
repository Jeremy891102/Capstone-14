"""Frozen 20-call pilot only. Offline by default; never starts a full run."""
import argparse
import csv
import fcntl
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from code import historical_runner as runner

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / 'frozen'
OUT = ROOT / 'cost_pilot_results'
COUNTS = {4: 584, 8: 292, 16: 146, 32: 73}


def parse_batch(raw, field_ids):
    if raw.get('modelVersion') != runner.MODEL:
        raise ValueError('Unexpected modelVersion')
    candidates = raw.get('candidates', [])
    if len(candidates) != 1 or candidates[0].get('finishReason') != 'STOP':
        raise ValueError('Missing candidate or non-STOP finish')
    answer = ''.join(p.get('text', '') for p in candidates[0].get('content', {}).get('parts', [])
                     if not p.get('thought', False))
    predictions = json.loads(answer, object_pairs_hook=runner.unique_object)
    if not isinstance(predictions, dict) or set(predictions) != set(field_ids):
        raise ValueError('Answer keys differ from requested batch')
    if any(type(v) is not str or v not in 'ABCDE' or len(v) != 1 for v in predictions.values()):
        raise ValueError('Answer must be one A-E letter per field')
    return answer, predictions, runner.usage(raw.get('usageMetadata'))


def path_for(row):
    name = row['call_id'].replace('::', '__').replace('=', '-')
    return OUT / 'calls' / f"k{row['batch_size']}__{name}.json"


def check_reuse(record, row):
    if record.get('status') != 'succeeded' or record.get('request') != row:
        raise ValueError('Uncertain/failed/changed call; review required')
    answer, predictions, usage = parse_batch(record['raw_response'], row['field_ids'])
    if (answer, predictions, usage) != (record['answer'], record['predictions'], record['usage']):
        raise ValueError('Saved response evidence differs')


def validate():
    manifest = json.loads((FROZEN / 'manifest.json').read_text())
    audit = json.loads((ROOT / 'readiness_audit.json').read_text())
    if hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest() != audit['preserved_harness_sha256']:
        raise ValueError('Historical helpers changed')
    for name, sha in manifest['files_sha256'].items():
        if hashlib.sha256((FROZEN / name).read_bytes()).hexdigest() != sha:
            raise ValueError('Frozen file changed: ' + name)
    rows = [json.loads(s) for s in (FROZEN / 'cost_pilot.jsonl').read_text().splitlines()]
    if len(rows) != 20 or len({(r['batch_size'], r['call_id']) for r in rows}) != 20:
        raise ValueError('Wrong pilot scope')
    for k in COUNTS:
        full = [json.loads(s) for s in (FROZEN / f'k{k}.jsonl').read_text().splitlines()]
        selected = [r for r in rows if r['batch_size'] == k]
        if len(selected) != 5 or any(r not in full for r in selected):
            raise ValueError('Pilot differs from full frozen inputs')
    for row in rows:
        if runner.digest(row['body']) != row['body_sha256'] or row['body']['generationConfig'] != runner.EXPECTED_GENERATION:
            raise ValueError('Changed payload or generation')
        if len(row['field_ids']) != row['batch_size'] or len(set(row['field_ids'])) != row['batch_size']:
            raise ValueError('Wrong batch size')
        if any(set(p) != {'text'} for c in row['body']['contents'] for p in c['parts']):
            raise ValueError('Media present')
    return rows


def attempt(row, key):
    record = {'status': 'in_flight', 'request': row, 'wire_url': runner.URL,
              'input_sha256': row['body_sha256'], 'started_unix': time.time(),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    runner.write(path_for(row), record)
    start = time.perf_counter()
    try:
        req = urllib.request.Request(runner.URL, data=json.dumps(row['body']).encode(),
                                     headers={'x-goog-api-key': key, 'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                status, payload = response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            status, payload = error.code, error.read().decode(errors='replace')
        record.update(http_status=status, raw_response_text=payload)
        if status != 200:
            raise ValueError('HTTP ' + str(status))
        record['raw_response'] = json.loads(payload)
        answer, predictions, usage = parse_batch(record['raw_response'], row['field_ids'])
        record.update(status='succeeded', answer=answer, predictions=predictions, usage=usage)
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error),
                      billing='unknown unless usage exists; no automatic retry')
    record.update(api_latency_s=time.perf_counter() - start, finished_unix=time.time())
    runner.write(path_for(row), record)
    return record


def summarize(records, completed=False):
    good = [r for r in records if r['status'] == 'succeeded']
    summary = {'status': 'completed' if completed else 'running', 'planned': 20, 'succeeded': len(good),
               'failed': len(records) - len(good), 'estimated_usd': sum(r['usage']['estimated_usd'] for r in good),
               'next_phase_started': False, 'by_k': {}}
    for k, calls in COUNTS.items():
        selected = [r for r in good if r['request']['batch_size'] == k]
        if not selected:
            continue
        sums = {key: sum(r['usage'][key] for r in selected)
                for key in ['prompt_tokens', 'answer_tokens', 'reasoning_tokens', 'cached_prompt_tokens', 'estimated_usd']}
        n = len(selected)
        summary['by_k'][k] = {'calls': n, **sums, 'mean_usd_per_call': sums['estimated_usd'] / n,
                              'projected_total_usd': sums['estimated_usd'] / n * calls,
                              'projected_remaining_usd': sums['estimated_usd'] / n * (calls - n),
                              'observed_max_usd_per_call': max(r['usage']['estimated_usd'] for r in selected)}
    runner.write(OUT / 'status.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--env-file', type=Path, default=Path('/Users/jeremy/Desktop/NYU/Capstone/Capstone-14/.env'))
    args = parser.parse_args()
    rows = validate()
    print(json.dumps({'mode': 'execute' if args.execute else 'offline', 'calls': len(rows), 'estimated_usd_guard': 2}), flush=True)
    if not args.execute:
        return
    OUT.mkdir(exist_ok=True)
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = []
        for row in rows:
            if path_for(row).exists():
                record = json.loads(path_for(row).read_text())
                check_reuse(record, row)
                records.append(record)
        key = runner.load_key(args.env_file)
        runner.write(OUT / 'authorization.json', {'instruction': '繼續', 'scope': '20-call cost pilot only',
                     'full_run_authorized': False, 'estimated_usd_guard': 2})
        for row in rows:
            if path_for(row).exists():
                continue
            reserve = ((len(json.dumps(row['body']).encode()) + 2048) * 2 + 32768 * 10) / 1e6
            if sum(r['usage']['estimated_usd'] for r in records) + reserve > 2:
                raise ValueError('Budget reserve reached before sending')
            record = attempt(row, key)
            records.append(record)
            summary = summarize(records)
            print(json.dumps({k: v for k, v in summary.items() if k != 'by_k'}), flush=True)
            if record['status'] != 'succeeded':
                raise ValueError('Stopped for failed/uncertain call; no automatic retry')
        summary = summarize(records, completed=True)
        gt = {(r['report_id'], r['field_id']): r['target'] for r in map(json.loads, (FROZEN / 'evaluation_only.jsonl').read_text().splitlines())}
        fields = []
        for record in records:
            row = record['request']
            for fid, prediction in record['predictions'].items():
                target = gt[row['report_id'], fid]
                fields.append(dict(model='argon', batch_size=row['batch_size'], call_id=row['call_id'],
                                   report_id=row['report_id'], field_id=fid, target=target,
                                   prediction=prediction, correct=int(prediction == target)))
        with (OUT / 'per_field.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fields[0]))
            writer.writeheader(); writer.writerows(fields)
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
