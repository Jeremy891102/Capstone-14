import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import run_argon_reference_novideo as r
ROOT=Path('/Users/jeremy/Documents/Codex/2026-10-06/git/outputs/research_next_step_20261111')
class Response:
    status=200
    def __init__(self,obj):self.obj=obj
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self):return json.dumps(self.obj).encode()
def response(fid):
 return {'modelVersion':r.MODEL,'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps({fid:'A'})}]}}],'usageMetadata':{'promptTokenCount':100,'candidatesTokenCount':10,'thoughtsTokenCount':20,'cachedContentTokenCount':40}}
class Safety(unittest.TestCase):
 def setUp(self):self.rows,self.gt=r.validate_rows(ROOT);self.row=self.rows[0]
 def test_scope(self):
  self.assertEqual(len(self.rows),640);self.assertEqual(sum(x['pilot'] for x in self.rows),128)
 def test_reasoning_counted_once(self):
  u=r.usage(response('x')['usageMetadata']);self.assertEqual(u['completion_including_reasoning'],30);self.assertAlmostEqual(u['estimated_usd'],(60*2+40*.1+30*10)/1e6)
 def test_zero_usage_rejected(self):
  for key in ['promptTokenCount','candidatesTokenCount']:
   u=response('x')['usageMetadata'];u[key]=0
   with self.assertRaises(ValueError):r.usage(u)
 def test_outer_json_and_version_and_finish(self):
  for obj in [[],{},dict(response('x'),modelVersion='other')]:
   with self.assertRaises(ValueError):r.parse(obj,'x')
  obj=response('x');obj['candidates'][0]['finishReason']='MAX_TOKENS'
  with self.assertRaises(ValueError):r.parse(obj,'x')
 def test_missing_extra_and_nonletter_are_invalid_not_retryable(self):
  for answer in ['I think A','{"x":"A","x":"B"}',json.dumps({'x':'A','extra':'B'}),json.dumps({'x':'AA'}),json.dumps({'wrong':'A'})]:
   obj=response('x');obj['candidates'][0]['content']['parts'][0]['text']=answer
   self.assertFalse(r.parse(obj,'x')[2])
 def test_exact_resume_does_not_send(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)
   with patch.object(r.urllib.request,'urlopen',return_value=Response(response(self.row['field_id']))) as http:
    a=r.attempt(out,self.row,'A','dummy');b=r.attempt(out,self.row,'A','dummy');self.assertEqual(http.call_count,1);self.assertEqual(a,b)
   changed=copy.deepcopy(self.row);changed['body']['contents'][0]['parts'][1]['text']+=' changed';changed['body_sha256']=r.digest(changed['body'])
   with self.assertRaises(ValueError):r.check_reuse(a,changed)
 def test_invalid_result_blocks_resume(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp);obj=response(self.row['field_id']);obj['candidates'][0]['content']['parts'][0]['text']='not JSON'
   with patch.object(r.urllib.request,'urlopen',return_value=Response(obj)) as http:
    a=r.attempt(out,self.row,'A','dummy');self.assertFalse(a['valid'])
    with self.assertRaises(ValueError):r.attempt(out,self.row,'A','dummy')
    self.assertEqual(http.call_count,1)
 def test_uncertain_attempt_blocks_reuse(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp);r.write(r.path_for(out,self.row),{'status':'in_flight'})
   with patch.object(r.urllib.request,'urlopen') as http:
    with self.assertRaises(ValueError):r.attempt(out,self.row,'A','dummy')
    http.assert_not_called()
 def test_budget_stops_before_request(self):
  with tempfile.TemporaryDirectory() as tmp:
   with patch.object(r.urllib.request,'urlopen') as http:
    with self.assertRaises(ValueError):r.run_phase(Path(tmp),[self.row],[self.row],self.gt,'dummy','pilot',1,.0001)
    http.assert_not_called()
 def test_provider_failure_is_retained_without_automatic_retry(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)
   with patch.object(r.urllib.request,'urlopen',side_effect=TimeoutError('test')) as http:
    a=r.attempt(out,self.row,'A','dummy');self.assertEqual(a['status'],'failed')
    with self.assertRaises(ValueError):r.attempt(out,self.row,'A','dummy')
    self.assertEqual(http.call_count,1)
if __name__=='__main__':unittest.main()
