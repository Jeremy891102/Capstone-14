import copy,unittest
from supervise_argon_reference_novideo_full73 import eligible
class TimeoutPolicy(unittest.TestCase):
 def setUp(self):
  self.row={'body_sha256':'hash','call_id':'call'};self.record={'status':'failed','error_type':'TimeoutError','request':copy.deepcopy(self.row),'input_sha256':'hash'}
 def test_only_first_response_free_timeout(self):self.assertTrue(eligible(self.record,self.row,0))
 def test_second_failure_never_retried(self):self.assertFalse(eligible(self.record,self.row,1))
 def test_success_never_retried(self):
  self.record['status']='succeeded';self.assertFalse(eligible(self.record,self.row,0))
 def test_http_failure_never_retried(self):
  self.record['error_type']='HTTPError';self.assertFalse(eligible(self.record,self.row,0))
 def test_received_response_never_retried(self):
  self.record['raw_response']={};self.assertFalse(eligible(self.record,self.row,0))
 def test_changed_input_never_retried(self):
  self.record['input_sha256']='other';self.assertFalse(eligible(self.record,self.row,0))
if __name__=='__main__':unittest.main()
