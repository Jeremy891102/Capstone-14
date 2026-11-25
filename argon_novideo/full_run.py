"""Frozen full run, with pilot reuse; offline default, separate execution required."""
import argparse
import concurrent.futures
import fcntl
import json
import math
import shutil
from pathlib import Path

import batch_pilot as batch

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'results/hd_epic_argon_novideo_batching_20261125'


def reserve(row):
    return ((len(json.dumps(row['body']).encode()) + 2048) * 2 + 32768 * 10) / 1e6


def status(records, state):
    good = [r for r in records if r['status'] == 'succeeded']
    summary = {'status': state, 'planned': 1095, 'succeeded': len(good),
               'failed_or_uncertain': len(records) - len(good), 'pilot_calls_reused': 20,
               'estimated_success_usd': sum(r['usage']['estimated_usd'] for r in good),
               'model': batch.runner.MODEL, 'next_experiment_started': False}
    batch.runner.write(OUT / 'status.json', summary)
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--max-estimated-usd', type=float, default=10)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--env-file', type=Path, default=Path('/Users/jeremy/Desktop/NYU/Capstone/Capstone-14/.env'))
    args = p.parse_args()
    if not 1 <= args.workers <= 4 or not math.isfinite(args.max_estimated_usd) or not 0 < args.max_estimated_usd <= 10:
        raise ValueError('Expected 1-4 workers and positive estimated budget at most $10')
    pilots = batch.validate()
    rows = [r for k in batch.COUNTS for r in map(json.loads, (batch.FROZEN / f'k{k}.jsonl').read_text().splitlines())]
    if len(rows) != 1095 or len({(r['batch_size'], r['call_id']) for r in rows}) != 1095:
        raise ValueError('Wrong full-run scope')
    for row in rows:
        if batch.runner.digest(row['body']) != row['body_sha256'] or row['body']['generationConfig'] != batch.runner.EXPECTED_GENERATION:
            raise ValueError('Changed body or generation')
        if any(set(p) != {'text'} for c in row['body']['contents'] for p in c['parts']):
            raise ValueError('Unexpected non-text input')
    source_paths = []
    for row in pilots:
        path = batch.path_for(row)
        record = json.loads(path.read_text())
        batch.check_reuse(record, row)
        source_paths.append((path, row))
    print(json.dumps({'mode': 'execute' if args.execute else 'offline', 'total_calls': 1095,
                      'validated_pilot_calls': 20, 'max_new_calls': 1075,
                      'estimated_total_usd_guard_including_pilot': args.max_estimated_usd}), flush=True)
    if not args.execute:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    # One lock covers pilot reuse, resume checks and all scheduled requests.
    with (OUT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch.OUT = OUT
        for source, row in source_paths:
            dest = batch.path_for(row)
            dest.parent.mkdir(exist_ok=True)
            if not dest.exists():
                shutil.copyfile(source, dest)
        records = []
        pending = []
        for row in rows:
            path = batch.path_for(row)
            if path.exists():
                record = json.loads(path.read_text())
                batch.check_reuse(record, row)
                records.append(record)
            else:
                pending.append(row)
        key = batch.runner.load_key(args.env_file)
        spent = sum(r['usage']['estimated_usd'] for r in records)
        halt = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            active = {}
            while pending or active:
                while pending and len(active) < args.workers and halt is None:
                    row = pending[0]
                    reserved = sum(reserve(r) for r in active.values())
                    if spent + reserved + reserve(row) > args.max_estimated_usd:
                        halt = 'Estimated budget reserve reached'
                        break
                    pending.pop(0)
                    active[pool.submit(batch.attempt, row, key)] = row
                if not active:
                    break
                done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    active.pop(future)
                    record = future.result()
                    records.append(record)
                    if record['status'] == 'succeeded':
                        spent += record['usage']['estimated_usd']
                    else:
                        halt = 'Failed/uncertain call; no automatic retry'
                status(records, 'stopping' if halt else 'running')
        status(records, 'review_needed' if halt else 'completed')
        if halt:
            raise ValueError(halt)


if __name__ == '__main__':
    main()
