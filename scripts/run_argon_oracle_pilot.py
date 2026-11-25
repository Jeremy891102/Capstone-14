"""Run only the frozen 20-call pilot. Serial, no retries; preserve raw API responses.
Usage: PYTHONPATH=src python scripts/run_argon_oracle_pilot.py [--execute]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from video_report.pipeline import create_run, execute_run
from video_report.providers.vertex_gemini import VertexGeminiProvider
from video_report.run_store import RunDir

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/argon_oracle_pilot20'


def load_credentials():
    # Only import the expected key, without printing or copying the env file.
    for line in Path('/home/nyu-data/cc4d_run/.env').read_text().splitlines():
        if line.startswith('GEMINI_API_KEY='):
            os.environ.setdefault('GEMINI_API_KEY', line.split('=', 1)[1].strip().strip('\"\''))
    assert os.environ.get('GEMINI_API_KEY'), 'missing GEMINI_API_KEY'


def factory(cfg, run):
    provider = VertexGeminiProvider(cfg.provider.model, cfg.provider.gemini)
    post = provider.http.post
    generate = provider.generate
    ledger = {'gross_usd': 0.0, 'call_id': None, 'unknown_usage': False}
    rawdir = run.path/'api_raw'; rawdir.mkdir(exist_ok=False)

    def capture(url, headers, body, timeout_s, max_bytes):
        data = post(url, headers, body, timeout_s, max_bytes)
        key = hashlib.sha256(ledger['call_id'].encode()).hexdigest()[:16]
        (rawdir/(key+'.json')).write_text(json.dumps(data, indent=2)+'\n')
        # Media bytes excluded from the readable payload; hashes bind exact sent bytes.
        clean = json.loads(json.dumps(body))
        for part in clean['contents'][0]['parts']:
            if 'inlineData' in part:
                media = part['inlineData']['data']
                part['inlineData']['data'] = {'base64_sha256': hashlib.sha256(media.encode()).hexdigest(), 'base64_length': len(media)}
        (rawdir/(key+'_request.json')).write_text(json.dumps({'call_id':ledger['call_id'],'url':url,'body':clean},indent=2)+'\n')
        u = data.get('usageMetadata', {})
        if not all(k in u for k in ['promptTokenCount', 'candidatesTokenCount']):
            ledger['unknown_usage'] = True
        else:
            inp, cache = u['promptTokenCount'], u.get('cachedContentTokenCount',0)
            ledger['gross_usd'] += ((inp-cache)*4+cache*.2+(u['candidatesTokenCount']+u.get('thoughtsTokenCount',0))*20)/1e6
        (run.path/'pilot_spend.json').write_text(json.dumps(ledger,indent=2)+'\n')
        return data

    def guarded(request, *, timeout_s):
        # Stop before another call once measured gross spend reaches $2 or usage is unknown.
        if ledger['gross_usd'] >= 2 or ledger['unknown_usage']:
            raise KeyboardInterrupt('pilot spend / unknown usage stop')
        ledger['call_id'] = request.call_id
        return generate(request, timeout_s=timeout_s)

    provider.http.post = capture
    provider.generate = guarded
    return provider


def on_done(state):
    print(state["call_id"], state["status"], flush=True)
    if state["status"] != "succeeded":
        raise KeyboardInterrupt("pilot call failed; inspect before continuing")


def main():
    ap = argparse.ArgumentParser(__doc__); ap.add_argument('--execute',action='store_true'); args=ap.parse_args()
    os.chdir(ROOT)
    if not args.execute:
        from video_report.snapshot import build_snapshot
        snap = build_snapshot(ROOT/'experiments/cc4d_oracle_argon/pilot20/config.yaml')
        assert len(snap.requests) == 20
        assert all('["A", "C"]' not in r.user for r in snap.requests)
        print('20 valid requests; no credentials loaded, no calls, no run created.')
        return
    load_credentials()
    if RUN.exists():
        raise SystemExit('Pilot run already exists; inspect it before any resume. No automatic retries.')
    run = create_run(ROOT/'experiments/cc4d_oracle_argon/pilot20/config.yaml', ROOT/'runs',run_id=RUN.name)
    assert len(run.load_requests()) == 20
    reqs = list(run.load_requests().values())
    assert all(len(r.field_ids) in (1,2) and '["A", "C"]' not in r.user for r in reqs)
    (run.path/'full_prompt_example.txt').write_text((reqs[0].system or '')+'\n\n'+reqs[0].user)
    result = execute_run(run,mode='create',provider_factory=factory,on_call_done=on_done)
    print(result)

if __name__ == '__main__':
    main()
