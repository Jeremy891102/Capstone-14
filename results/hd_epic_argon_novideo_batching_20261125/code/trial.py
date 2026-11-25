"""Run only the k1 subset10 comparison with the preserved historical harness."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

from code import historical_runner as runner

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / 'frozen'
OLD = Path('/Users/jeremy/Documents/Codex/2026-10-06/git/outputs/research_next_step_20261111/execution_20261112')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--max-estimated-usd', type=float, default=2.0)
    p.add_argument('--env-file', type=Path, default=Path('/Users/jeremy/Desktop/NYU/Capstone/Capstone-14/.env'))
    args = p.parse_args()
    if not 0 < args.max_estimated_usd <= 2:
        raise ValueError('Trial-only budget must be positive and at most $2 at frozen rates')
    manifest = json.loads((FROZEN / 'manifest.json').read_text())
    audit = json.loads((ROOT / 'readiness_audit.json').read_text())
    if hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest() != audit['preserved_harness_sha256']:
        raise ValueError('Preserved historical harness changed')
    for name, sha in manifest['files_sha256'].items():
        if hashlib.sha256((FROZEN / name).read_bytes()).hexdigest() != sha:
            raise ValueError('Frozen file changed: ' + name)
    rows = [json.loads(s) for s in (FROZEN / 'k1_subset10.jsonl').read_text().splitlines()]
    historical = {(r['report_id'], r['field_id']): r for r in map(json.loads, (FROZEN / 'historical_subset10.jsonl').read_text().splitlines())}
    gt = {(r['report_id'], r['field_id']): r['target'] for r in map(json.loads, (FROZEN / 'evaluation_only.jsonl').read_text().splitlines())}
    for r in rows:
        r.update(field_id=r['field_ids'][0], condition='single')
        if r['body'] != historical[r['report_id'], r['field_id']]['body'] or runner.digest(r['body']) != r['body_sha256']:
            raise ValueError('Historical payload mismatch')
        oldrow = historical[r['report_id'], r['field_id']]
        runner.check_reuse(json.loads(runner.path_for(OLD, oldrow).read_text()), oldrow)
    if len(rows) != 320 or len({r['call_id'] for r in rows}) != 320:
        raise ValueError('Wrong trial scope')
    print(json.dumps({'mode': 'execute' if args.execute else 'offline', 'calls': 320,
                      'exact_historical_payloads': 320, 'estimated_usd_guard': args.max_estimated_usd}), flush=True)
    if not args.execute:
        return
    out = ROOT / 'trial_results'
    out.mkdir(exist_ok=True)
    with (out / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key = runner.load_key(args.env_file)
        runner.run_phase(out, rows, rows, gt, key, 'k1_subset10', 1, args.max_estimated_usd)
        results = runner.current_records(out, rows)
        agreements = 0
        historical_correct = 0
        for r, result in zip(rows, results):
            oldrow = historical[r['report_id'], r['field_id']]
            old = json.loads(runner.path_for(OLD, oldrow).read_text())
            runner.check_reuse(old, oldrow)
            agreements += old['prediction'] == result['prediction']
            historical_correct += old['prediction'] == gt[r['report_id'], r['field_id']]
        accuracy = sum(r['correct'] for r in results) / 320 * 100
        delta = accuracy - historical_correct / 320 * 100
        runner.write(out / 'comparison.json', {'calls': 320, 'accuracy_pct': accuracy,
                     'historical_accuracy_pct': historical_correct / 320 * 100,
                     'delta_pp': delta, 'answer_agreement_pct': agreements / 320 * 100,
                     'gate': 'investigate' if abs(delta) > 5 else 'review',
                     'next_phase_started': False})


if __name__ == '__main__':
    main()
