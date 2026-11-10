"""Check matched cohorts, backend disclosure, valid embedded data and the retained context section."""
import base64,csv,json
from pathlib import Path
from bs4 import BeautifulSoup
R=Path(__file__).resolve().parent;s=json.loads((R/'analysis.json').read_text())
assert len(s['q1'])==20,'Four models at five sizes required'
assert [x['correct'] for x in s['q1'] if x['model']=='qwen32b']==[769,783,761,701,691]
assert all(x['questions']==2336 for x in s['q1'])
assert next(x for x in s['e1_full_pool'] if x['model']=='qwen32b')['correct']==1130
assert sum(x['calls'] for x in s['qwen32b_backend_work'])==4307
assert sum(x['calls'] for x in s['qwen32b_backend_work'] if x['backend']=='hf')==1962
assert sum(x['calls'] for x in s['qwen32b_backend_work'] if x['backend']=='vllm_tp4_apc')==2345
assert all(x['cost_per_32_usd'] is None for x in s['q1'] if x['model'].startswith('qwen'))
for m in ['lite','argon','qwen8b']:
 old=json.loads((R/'previous_analysis.json').read_text());assert [x for x in s['q1'] if x['model']==m]==[x for x in old['q1'] if x['model']==m]
assert s['context_full']==json.loads((R/'previous_analysis.json').read_text())['context_full']
text=(R/'baseline_report.html').read_text();soup=BeautifulSoup(text,'html.parser')
assert soup.find(id='backend-control') is None
assert not soup.find_all('script')
assert soup.find(id='context-control')
assert 'capacity tests only' not in soup.get_text()
assert 'mixed backend' not in soup.get_text().lower()
assert '63.91%' in soup.find(id='context-control').get_text() and '67.29%' in soup.find(id='context-control').get_text()
ids=[x['id'] for x in soup.find_all(id=True)];assert len(ids)==len(set(ids))
for a in soup.find_all('a',href=True):
 h=a['href']
 if h.startswith('#'):assert h[1:] in ids
 elif h.startswith('data:'):assert base64.b64decode(h.split(',',1)[1])==(R/a['download']).read_bytes()
 else:raise AssertionError('External or broken report link '+h)
with (R/'per_field.csv').open() as f:fs=list(csv.DictReader(f))
assert len(fs)==50224 and len({(x['model'],x['batch_size'],x['report_id'],x['field_id']) for x in fs})==50224
with (R/'per_call.csv').open() as f:cs=list(csv.DictReader(f))
assert len(cs)==17228
print('PASS: four models, exact matched scores, backend accounting, original results preserved, context retained, portable downloads')
