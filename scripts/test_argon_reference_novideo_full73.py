import json,tempfile,hashlib,unittest,shutil
from pathlib import Path
from run_argon_reference_novideo_full73 import validate
ROOT=Path('/Users/jeremy/Documents/Codex/2026-10-06/git/outputs/argon_reference_novideo_full73_20261112')
class FullManifestSafety(unittest.TestCase):
 def test_full_reuse_identity(self):
  plan,rows,gt,retained=validate(ROOT);self.assertEqual((len(rows),len(gt),len(retained)),(4672,2336,640))
 def changed(self,mutate):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);plan=json.loads((ROOT/'plan.json').read_text())
   for name in plan['files_sha256']:shutil.copyfile(ROOT/name,root/name)
   rows=list(map(json.loads,(root/'full_requests.jsonl').read_text().splitlines()));mutate(rows)
   (root/'full_requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));plan['files_sha256']['full_requests.jsonl']=hashlib.sha256((root/'full_requests.jsonl').read_bytes()).hexdigest();(root/'plan.json').write_text(json.dumps(plan))
   with self.assertRaises(ValueError):validate(root)
 def test_duplicate_rejected_even_if_hash_resealed(self):self.changed(lambda rows:rows.__setitem__(0,rows[1]))
 def test_missing_call_rejected_even_if_hash_resealed(self):self.changed(lambda rows:rows.pop())
 def test_media_rejected_even_if_body_hash_resealed(self):
  def mutate(rows):
   from run_argon_reference_novideo import digest
   rows[0]['body']['contents'][0]['parts'].append({'inlineData':{'data':'not-video','mimeType':'video/mp4'}});rows[0]['body_sha256']=digest(rows[0]['body'])
  self.changed(mutate)
if __name__=='__main__':unittest.main()
