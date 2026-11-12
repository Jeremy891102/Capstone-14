"""Reconstruct frozen local inputs and640 exact retained calls. Offline only; never sends API requests."""
import argparse,gzip,hashlib,json,shutil
from pathlib import Path

def main(destination,scope):
 repo=Path(__file__).resolve().parents[1];name='hd_epic_argon_reference_video_control_20261112';source=repo/'experiments'/name/scope/'frozen';results=repo/'results'/name/'subset10';destination.mkdir(parents=True,exist_ok=True)
 def exact_write(path,data):
  path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists() and path.read_bytes()!=data:raise ValueError('Refuse to overwrite differing evidence: '+str(path))
  path.write_bytes(data)
 exact_write(destination/'plan.json',(source/'plan.json').read_bytes())
 for p in source.glob('*.gz'):
  target=p.name[:-3]
  path=destination/target
  if target=='historical_video_comparison.csv':path=destination/'frozen_inputs'/target
  exact_write(path,gzip.decompress(p.read_bytes()))
 records=[json.loads(s) for s in gzip.decompress((results/'experiment_log.jsonl.gz').read_bytes()).decode().splitlines()]
 if len(records)!=640:raise ValueError('Wrong retained response count')
 out=destination/'execution_20261112';(out/'calls').mkdir(parents=True,exist_ok=True)
 for r in records:
  p=out/'calls'/(r['request']['call_id'].replace('::','__').replace('=','-')+'.json');exact_write(p,(json.dumps(r,indent=2)+'\n').encode())
 if scope=='full73':
  retained=json.loads((destination/'retained_successes.json').read_text())
  for r in retained:
   p=out/'calls'/(r['call_id'].replace('::','__').replace('=','-')+'.json')
   if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Retained response bytes differ')
 history=results/'attempt_history'
 if history.exists():
  # Imported unknown billing is preserved for budget and attempt accounting.
  for p in history.rglob('*'):
   if p.is_file():exact_write(out/'attempt_history/imported_subset'/p.relative_to(history),p.read_bytes())
 print(json.dumps({'status':'materialized offline','scope':scope,'destination':str(destination),'retained_calls':640,'api_requests':0}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True);p.add_argument('--scope',choices=['full73','subset10'],default='full73');a=p.parse_args();main(a.destination,a.scope)
