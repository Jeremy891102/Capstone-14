"""Freeze 73-video expansion and import exact existing successful calls. Offline only."""
import json,hashlib,shutil
from pathlib import Path
from collections import Counter
from run_argon_reference_novideo import digest,check_reuse,path_for,MODEL,EXPECTED_GENERATION
O=Path('/Users/jeremy/Documents/Codex/2026-10-06/git/outputs')
OLD=O/'research_next_step_20261111';R=O/'argon_reference_novideo_full73_20261112';E=R/'execution_20261112'

def main():
 R.mkdir(exist_ok=True);(E/'calls').mkdir(parents=True,exist_ok=True)
 contexts={(r['request']['report_id'],r['request']['field_ids'][0]):r for r in map(json.loads,(O/'argon_context_full_20261029/inputs/full_requests.jsonl').read_text().splitlines()) if r['condition']=='context_7'}
 singles={(r['request']['report_id'],r['request']['field_ids'][0]):r for r in map(json.loads,(O/'vertex_argon_full_size1_20261026/inputs/requests.jsonl').read_text().splitlines())}
 old={(r['report_id'],r['field_id'],r['condition']):r for r in map(json.loads,(OLD/'full_requests.jsonl').read_text().splitlines())}
 keys=sorted(contexts);assert len(keys)==2336 and len({k[0] for k in keys})==73
 counts=Counter(k[0] for k in keys);assert set(counts.values())=={32}
 rows=[];imports=[]
 for key in keys:
  for condition,source in [('single',singles[key]),('context_7',contexts[key])]:
   q=source['request'];assert q['field_ids']==[key[1]]
   body={'systemInstruction':{'parts':[{'text':q['system']}]},'contents':[{'role':'user','parts':[{'text':'Video head_cam; this clip starts at 0 seconds.'},{'text':q['user']}]}],'generationConfig':EXPECTED_GENERATION}
   row={'call_id':f'{key[0]}::novideo-{condition}={key[1]}','report_id':key[0],'field_id':key[1],'condition':condition,'pilot':False,'model':MODEL,'source_call_id':q['call_id'],'source_row_sha256':digest(source),'source_media_sha256':source['canonical_clip']['clip_sha256'],'body':body,'body_sha256':digest(body)}
   prior=old.get((*key,condition))
   if prior:
    row['pilot']=prior['pilot'];assert row==prior
    p=path_for(OLD/'execution_20261112',prior);record=json.loads(p.read_text());check_reuse(record,row)
    dest=path_for(E,row)
    if dest.exists():assert dest.read_bytes()==p.read_bytes()
    else:shutil.copyfile(p,dest)
    imports.append({'call_id':row['call_id'],'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'estimated_usd':record['usage']['estimated_usd']})
   rows.append(row)
 assert len(rows)==4672 and len(imports)==640
 gt={(r['report_id'],r['field_id']):r['target'] for r in map(json.loads,(O/'argon_context_full_20261029/inputs/ground_truth.jsonl').read_text().splitlines())}
 assert set(keys)==set(gt)
 (R/'full_requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 (R/'evaluation_only.jsonl').write_text(''.join(json.dumps({'report_id':rid,'field_id':fid,'target':gt[rid,fid]})+'\n' for rid,fid in keys))
 (R/'retained_successes.json').write_text(json.dumps(imports,indent=2))
 source=OLD/'execution_20261112/attempt_history';dest=E/'attempt_history/imported_subset'
 if source.exists():shutil.copytree(source,dest,dirs_exist_ok=True)
 historical=json.loads((O/'argon_context_full_20261029/analysis.json').read_text())
 (R/'historical_full_summary.json').write_text(json.dumps(historical,indent=2))
 (R/'historical_single_predictions.csv').write_bytes((O/'vertex_argon_full_size1_20261026/per_field.csv').read_bytes())
 plan={'status':'prepared','authorization':'User explicitly requested all73baselinevideos to match the prior full context study','reports':sorted(counts),'targets':2336,'planned_calls':4672,'retained_successful_calls':640,'new_calls':4032,'model':MODEL,'generation':EXPECTED_GENERATION,'primary_metric':'(video context7 - video single) - (no-video context7 - no-video single)','historical_video_totals':{'single_correct':1402,'context7_correct':1572,'targets':2336},'remaining_cost_projection_frozen_rates_usd':sum(r['estimated_usd'] for r in imports)/640*4032,'estimated_cost_guard_usd_including_retained':40,'currency_caveat':'Historical frozen rate estimate only; unknown failed billing separate; no invoice/balance verification','historical_paired_context_status':'Full aggregate and all single predictions local; full context per-target predictions require retrieval for paired D CI','files_sha256':{name:hashlib.sha256((R/name).read_bytes()).hexdigest() for name in ['full_requests.jsonl','evaluation_only.jsonl','retained_successes.json','historical_full_summary.json','historical_single_predictions.csv']}}
 (R/'plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps({k:plan[k] for k in ['targets','planned_calls','retained_successful_calls','new_calls','remaining_cost_projection_frozen_rates_usd']}))
if __name__=='__main__':main()
