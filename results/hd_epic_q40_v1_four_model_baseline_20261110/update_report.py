"""Add the audited 32B run to the existing standalone report. Offline only."""
import base64,csv,gzip,hashlib,html,io,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
R=Path(__file__).resolve().parent;K=[1,4,8,16,32]
LABEL={'lite':'Gemini 3.5 Flash-Lite','argon':'Gemini 4 Argon','qwen8b':'Qwen3-VL-8B-Instruct','qwen32b':'Qwen3-VL-32B-Instruct'}
COLOR={'lite':'#526fa0','argon':'#16877f','qwen8b':'#985e39','qwen32b':'#72549a'}
def read(n):
 with (R/n).open() as f:return list(csv.DictReader(f))
def export(n,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (R/n).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def yes(x):return str(x).lower()=='true'
def ci(vals):
 a=np.asarray(vals,dtype=float);g=np.random.default_rng(20261027);return np.quantile(a[g.integers(0,len(a),(10000,len(a)))].mean(axis=1),[.025,.975]).tolist()
def groups(rows):
 d=defaultdict(list)
 for f in rows:d[f['report_id']].append(f['correct'])
 return {r:float(np.mean(v)) for r,v in sorted(d.items())}
def pct(x):return f'{x*100:.2f}%'
def interval(x):return f'{pct(x[0])}–{pct(x[1])}'
S=json.loads((R/'previous_analysis.json').read_text());fields=read('previous_per_field.csv');calls=read('previous_per_call.csv')
matched={(x['report_id'],x['field_id']) for x in fields if x['model']=='lite' and int(x['batch_size'])==32};assert len(matched)==2336
ref={(int(x['batch_size']),x['report_id'],x['field_id']):x['target'] for x in fields if x['model']=='lite'}
audit=json.loads((R/'qwen32b_completion_audit.json').read_text());ea=json.loads((R/'qwen32b_export_audit.json').read_text());assert audit['status']=='verified_completed' and audit['calls']==4307
assert hashlib.sha256((R/'qwen32b_calls.jsonl.gz').read_bytes()).hexdigest()==ea['sha256']
raw=[json.loads(l) for l in gzip.open(R/'qwen32b_calls.jsonl.gz','rt')];assert len(raw)==4307
newfields=[];newcalls=[]
for x in raw:
 k=x['batch_size'];q=x['source_request'];assert x['status']=='succeeded'
 for f in x['scoring']['fields']:
  assert f['target']==ref[k,q['report_id'],f['field_id']]
  newfields.append({'model':'qwen32b','batch_size':k,'report_id':q['report_id'],'call_id':q['call_id'],'backend':x['backend'],**f})
 newcalls.append({'model':'qwen32b','batch_size':k,'report_id':q['report_id'],'call_id':q['call_id'],'input_tokens':x['prompt_tokens'],'completion_tokens_including_reasoning':x['completion_tokens'],'cached_input_tokens':x.get('num_cached_tokens') or 0,'normalized_cost_usd':None,'observed_api_s':None,'model_generation_s':x['inference_s'],'preprocess_s':x['preprocess_s'],'total_call_s':x['total_call_s'],'placement':x['config'].get('placement','tensor_parallel_4gpu'),'valid_json':x['scoring']['valid'],'source_record':x['raw_source_path'],'backend':x['backend'],'timing_boundary':x.get('timing_boundary','HF model.generate; CPU preprocessing excluded'),'report_decode_s':x.get('report_decode_s')})
assert len(newfields)==12556 and sum(not x['valid_json'] for x in newcalls)==0
# The exported field CSV must exactly reproduce every raw score, independent of ordering.
actual={(str(x['batch_size']),x['report_id'],x['field_id']):(x['prediction'],x['target'],x['correct']) for x in newfields}
for x in read('qwen32b_per_field.csv'):assert actual[x['batch_size'],x['report_id'],x['field_id']]==(x['prediction'],x['target'],yes(x['correct']))
fields+=newfields;calls+=newcalls
assert len(fields)==50224 and len(calls)==17228
for k in K:
 fs=[f for f in newfields if f['batch_size']==k];sub=[f for f in fs if (f['report_id'],f['field_id']) in matched];assert len(fs)==(3212 if k==1 else 2336) and len(sub)==2336
 g=groups(sub);assert len(g)==73 and all(sum(f['report_id']==r for f in sub)==32 for r in g)
 ids={f['call_id'] for f in sub};cs=[c for c in newcalls if c['batch_size']==k and c['call_id'] in ids];assert len(cs)==2336//k
 S['q1'].append({'model':'qwen32b','batch_size':k,'correct':sum(f['correct'] for f in sub),'questions':2336,'accuracy':sum(f['correct'] for f in sub)/2336,'ci95':ci(list(g.values())),'calls':len(cs),'cost_per_32_usd':None,'tokens_per_32':sum(c['input_tokens']+c['completion_tokens_including_reasoning'] for c in cs)/73,'input_tokens_per_32':sum(c['input_tokens'] for c in cs)/73,'completion_tokens_per_32':sum(c['completion_tokens_including_reasoning'] for c in cs)/73,'seconds_per_32':sum(c['model_generation_s'] for c in cs)/73,'preprocess_seconds_per_32':sum(c['preprocess_s'] for c in cs)/73,'local_total_seconds_per_32':sum(c['total_call_s'] for c in cs)/73,'backend_calls':dict((b,sum(c['backend']==b for c in cs)) for b in ['hf','vllm_tp4_apc']),'timing_boundary':'Mixed accepted-call HF generate and vLLM generate wall; not a homogeneous speed metric'})
e1=[f for f in newfields if f['batch_size']==1];S['e1_full_pool'].append({'model':'qwen32b','correct':sum(f['correct'] for f in e1),'questions':3212,'accuracy':sum(f['correct'] for f in e1)/3212})
for t in ['loc','how','why','weight']:
 fs=[f for f in e1 if f['field_id'].split('_')[0]==t];g=groups(fs)
 S['q2_matched_e1'].append({'model':'qwen32b','type':t,'correct':sum(f['correct'] for f in fs),'questions':len(fs),'videos':len(g),'pooled_accuracy':sum(f['correct'] for f in fs)/len(fs),'video_mean_accuracy':float(np.mean(list(g.values()))),'ci95':ci(list(g.values()))})
for a,b in [(4,1),(8,1),(16,8),(32,8)]:
 A=groups([f for f in newfields if f['batch_size']==a and (f['report_id'],f['field_id']) in matched]);B=groups([f for f in newfields if f['batch_size']==b and (f['report_id'],f['field_id']) in matched]);v=[A[r]-B[r] for r in sorted(A)]
 S['paired_batch_contrasts'].append({'model':'qwen32b','comparison':f'k{a} minus k{b}','difference_pp':float(np.mean(v)*100),'ci95_pp':[x*100 for x in ci(v)],'limitation':'Mixed-backend descriptive contrast; not an isolated batch-size effect'})
S['qwen32b_validity']=[{'batch_size':k,'calls':sum(x['batch_size']==k for x in newcalls),'invalid':0} for k in K]
work=[]
for b in ['hf','vllm_tp4_apc']:
 for k in K:
  cs=[x for x in newcalls if x['backend']==b and x['batch_size']==k];wall=[x['model_generation_s'] for x in cs]
  work.append({'backend':b,'batch_size':k,'calls':len(cs),'accepted_inference_s':sum(wall),'median_inference_s':float(np.median(wall)),'mean_inference_s':float(np.mean(wall)),'cached_tokens':sum(x['cached_input_tokens'] for x in cs),'cold_calls':sum(x['cached_input_tokens']==0 for x in cs) if b!='hf' else None,'warm_calls':sum(x['cached_input_tokens']>0 for x in cs) if b!='hf' else None})
S.update(qwen32b_backend_work=work,qwen32b_audit=audit,qwen32b_status='Completed; 1,962 retained HF + 2,345 vLLM TP4/APC calls; mixed backend',time_comparison='Gemini API round trip; Qwen8B HF generate; Qwen32B HF/vLLM shown separately. Different timing boundaries/cohorts/cache states; no speed ranking.')
(R/'analysis.json').write_text(json.dumps(S,indent=2)+'\n')
for n,rows in [('per_field.csv',fields),('per_call.csv',calls),('batch_summary.csv',S['q1']),('question_type_summary.csv',S['q2_matched_e1']),('paired_batch_contrasts.csv',S['paired_batch_contrasts']),('qwen32b_backend_work.csv',work)]:export(n,rows)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'text.color':'#263647'})
Q={(x['model'],x['batch_size']):x for x in S['q1']};FIG={}
def save(fig,n):
 fig.tight_layout(pad=1.7);b=io.StringIO();fig.savefig(b,format='svg');fig.savefig(R/(n+'.png'),dpi=160);plt.close(fig);svg=b.getvalue();FIG[n]=svg[svg.index('<svg'):];(R/(n+'.svg')).write_text(FIG[n])
def accuracy(models,name,limits):
 fig,ax=plt.subplots(figsize=(12,5.3))
 for m in models:
  ys=np.array([Q[m,k]['accuracy']*100 for k in K]);cis=np.array([Q[m,k]['ci95'] for k in K])*100
  ax.errorbar(range(5),ys,yerr=[ys-cis[:,0],cis[:,1]-ys],fmt='o--' if m=='qwen32b' else 'o-',color=COLOR[m],ecolor='#92999f',capsize=3,label=LABEL[m],lw=2)
  if len(models)==1 or m in ['argon','qwen8b']:
   for i,y in enumerate(ys):ax.annotate(f'{y:.2f}%',(i,y),xytext=(0,10 if m!='qwen8b' else -18),textcoords='offset points',ha='center',fontsize=9,color=COLOR[m])
 ax.set(xticks=range(5),xticklabels=K,xlabel='Questions answered per call (k)',ylabel='Localization accuracy (%) · zoomed axis',ylim=limits,title='Same 2,336 localization targets · '+('four models' if len(models)>1 else LABEL[models[0]]));ax.grid(axis='y',alpha=.16);ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.0),fontsize=9,ncol=2 if len(models)>1 else 1);save(fig,name)
accuracy(list(LABEL),'accuracy_compare',(18,78))
for m,lim in [('lite',(20,42)),('argon',(52,77)),('qwen8b',(17,34)),('qwen32b',(23,41))]:accuracy([m],'accuracy_'+m,lim)
fig,ax=plt.subplots(figsize=(13,5.5));ts=['loc','how','why','weight'];xx=np.arange(4)*2.7
for i,m in enumerate(LABEL):
 rr=[next(x for x in S['q2_matched_e1'] if x['model']==m and x['type']==t) for t in ts];ys=np.array([x['video_mean_accuracy']*100 for x in rr]);cis=np.array([x['ci95'] for x in rr])*100;pos=xx+(i-1.5)*.46
 ax.bar(pos,ys,.39,color=COLOR[m],label=LABEL[m]);ax.errorbar(pos,ys,yerr=[ys-cis[:,0],cis[:,1]-ys],fmt='none',ecolor='#8b9299',capsize=3)
 for a,b in zip(pos,ys):ax.annotate(f'{b:.1f}',(a,b),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','pad':.3})
counts=[next(x for x in S['q2_matched_e1'] if x['model']=='lite' and x['type']==t) for t in ts];ax.set(xticks=xx,xticklabels=[f'{t}\n{x["questions"]} Q / {x["videos"]} videos' for t,x in zip(['Localization','How','Why','Weight'],counts)],ylim=(0,118),yticks=range(0,101,20),ylabel='Mean per-video accuracy (%)',title='Original E1 · same question IDs for all four models');ax.legend(frameon=False,ncol=2,loc='upper left',fontsize=9);ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True);save(fig,'types_matched')
fig,ax=plt.subplots(figsize=(12,5))
for m in LABEL:
 ax.plot(range(5),[Q[m,k]['tokens_per_32'] for k in K],'o--' if m=='qwen32b' else 'o-',label=LABEL[m],color=COLOR[m])
ax.set(xticks=range(5),xticklabels=K,xlabel='Questions answered per call (k)',ylabel='Recorded input + completion tokens / 32 · log scale',yscale='log',title='Recorded tokens · cached tokens remain included');ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.15);save(fig,'tokens')
fig,axes=plt.subplots(1,3,figsize=(15,4.8))
for m in ['lite','argon']:axes[0].plot(range(5),[Q[m,k]['seconds_per_32'] for k in K],'o-',label=LABEL[m],color=COLOR[m])
axes[0].set(title='Gemini API round trip',ylabel='Successful API seconds / 32 questions')
axes[1].plot(range(5),[Q['qwen8b',k]['seconds_per_32'] for k in K],'o-',color=COLOR['qwen8b'],label='8B · HF');axes[1].set(title='Qwen 8B · HF',ylabel='Generation seconds / 32 questions')
for b,label,style in [('hf','32B · retained HF','o-'),('vllm_tp4_apc','32B · vLLM, cold/warm pooled','s--')]:axes[2].plot(range(5),[next(x for x in work if x['backend']==b and x['batch_size']==k)['median_inference_s'] for k in K],style,label=label)
axes[2].set(title='Qwen 32B · separate backends',ylabel='Median inference seconds / call')
for ax in axes:ax.set(xticks=range(5),xticklabels=K,xlabel='Questions answered per call (k)');ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15);ax.set_ylim(bottom=0)
save(fig,'time')
# Edit the preserved report so existing mixing/context/expanded Q2 stay intact.
soup=BeautifulSoup((R/'previous_baseline_report.html').read_text(),'html.parser')
def frag(s):return BeautifulSoup(s,'html.parser')
def replace(node,text):node.replace_with(frag(text))
def table(headers,rows):return '<div class="wrap"><table><thead><tr>'+''.join('<th scope="col">'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def figure(name,caption):return '<div class="figure">'+FIG[name]+'<p class="caption">'+caption+'</p></div>'
def download(name,label):
 p=R/name;mime='text/csv' if p.suffix=='.csv' else 'application/json';return '<a class="download" download="'+p.name+'" href="data:'+mime+';base64,'+base64.b64encode(p.read_bytes()).decode()+'">'+label+'</a>'
soup.h1.string='Video QA baseline: four models';soup.select_one('.eyebrow').string='Capstone 14 · HD-EPIC q40_v1 · Updated 10 November 2026'
soup.header.find('p').decompose()
for tag in soup.header.select('.tag'):
 if tag.get_text().startswith('Qwen'):tag.decompose()
soup.find(id='findings').decompose()
q1=soup.find(id='q1');replace(q1.find(class_='figure'),figure('accuracy_compare','2,336 shared questions across 73 videos. Whiskers: 95% video-bootstrap intervals.'))
replace(q1.find('table').find_parent('div',class_='wrap'),table(['Model','k','Correct / same targets','Accuracy','95% video interval'],[[LABEL[x['model']],x['batch_size'],f"{x['correct']}/2,336",pct(x['accuracy']),interval(x['ci95'])] for x in S['q1']]))
for d in list(q1.find_all('details')):
 if 'enlarged accuracy scale' in d.summary.get_text():d.decompose()
for m in LABEL:q1.append(frag('<details><summary>'+LABEL[m]+' · enlarged accuracy scale</summary>'+figure('accuracy_'+m,'Same questions and intervals, on a zoomed scale.')+'</details>'))
d=next(d for d in q1.find_all('details') if 'Paired batch-size' in d.summary.get_text());replace(d.find('table').find_parent('div',class_='wrap'),table(['Model','Comparison','Accuracy change','95% paired video interval'],[[LABEL[x['model']],x['comparison'],f"{x['difference_pp']:+.2f} pp",f"{x['ci95_pp'][0]:+.2f} to {x['ci95_pp'][1]:+.2f} pp"] for x in S['paired_batch_contrasts']]))
for txt in list(q1.find_all(string=True)):
 if 'Qwen k=1/4/8' in txt:txt.replace_with(str(txt).replace('Qwen k=','Qwen8B k=').replace('for Qwen.','for Qwen8B.'))
eff=soup.find(id='efficiency');figs=eff.find_all('div',class_='figure',recursive=False)
# The first four are cost, time, tokens and API tradeoff; leave API cost plots unchanged.
assert len(figs)==4
replace(figs[1],figure('time','Units differ by panel. Times exclude downtime and model loading.'))
replace(figs[2],figure('tokens','Counts include cached input tokens.'))
replace(eff.find('table').find_parent('div',class_='wrap'),table(['Model','k','Estimated USD / 32','Accepted inference seconds / 32','Recorded tokens / 32'],[[LABEL[x['model']],x['batch_size'],'Unknown compute' if x['model'].startswith('qwen') else f"${x['cost_per_32_usd']:.4f}",f"{x['seconds_per_32']:,.1f}",f"{x['tokens_per_32']:,.0f}"] for x in S['q1']]))
# Replace old totals with four-model totals, keeping API and local inference boundaries explicit.
for d in eff.find_all('details'):
 if 'Accepted E1' in d.summary.get_text():
  rr=[]
  for m in LABEL:
   cs=[x for x in calls if x['model']==m];inp=sum(int(x['input_tokens']) for x in cs);comp=sum(int(x['completion_tokens_including_reasoning']) for x in cs);seconds=sum(float(x['model_generation_s'] if m.startswith('qwen') else x['observed_api_s']) for x in cs)
   rr.append([LABEL[m],len(cs),'12,556',f'{inp:,}',f'{comp:,}','Unknown VM compute' if m.startswith('qwen') else f"${sum(float(x['normalized_cost_usd']) for x in cs):.2f}",f'{seconds/3600:.2f} h'])
  replace(d.find('table').find_parent('div',class_='wrap'),table(['Model','Calls','Answers','Input tokens','Completion tokens','Estimated USD','Summed accepted inference time'],rr))
eff.append(frag('<details><summary>Qwen32B timing and cache · separate backend cohorts</summary>'+table(['Backend','k','Calls','Median inference s / call','Summed inference s','Cold / warm calls'],[[x['backend'],x['batch_size'],x['calls'],f"{x['median_inference_s']:.2f}",f"{x['accepted_inference_s']:.1f}",'Not recorded' if x['backend']=='hf' else f"{x['cold_calls']} / {x['warm_calls']}"] for x in work])+'<p class="caption">These are different video/call cohorts, not paired acceleration benchmarks. All 32B k=32 vLLM calls are cold first calls of a report; many smaller calls are warm. This affects both timing and potentially numerical outputs. Excludes model loads, failures and interruption gaps.</p></details>'))
q2=soup.find(id='q2');q2.find(class_='answer').decompose();replace(q2.find(class_='figure'),figure('types_matched','One question per call. Bars average videos equally; whiskers show 95% intervals.'))
replace(q2.find('table').find_parent('div',class_='wrap'),table(['Model','Subtype','Correct / Q','Videos','Video mean','95% interval','Pooled accuracy'],[[LABEL[x['model']],x['type'],f"{x['correct']}/{x['questions']}",x['videos'],pct(x['video_mean_accuracy']),interval(x['ci95']),pct(x['pooled_accuracy'])] for x in S['q2_matched_e1']]))
d=next(d for d in q2.find_all('details') if 'Entire original E1' in d.summary.get_text());replace(d.find('table').find_parent('div',class_='wrap'),table(['Model','Correct / Q','Pooled accuracy'],[[LABEL[x['model']],f"{x['correct']}/3,212",pct(x['accuracy'])] for x in S['e1_full_pool']]))
# Clarify older Qwen8B-only claims without changing the context or mixing numbers.
for txt in list(q2.find_all(string=True)):
 st=str(txt)
 if 'All 55 Qwen contract failures' in st:txt.replace_with(st.replace('All 55 Qwen contract failures','All 55 Qwen8B contract failures'))
settings=soup.find(id='settings');st=settings.find('table');st.thead.tr.append(frag('<th scope="col">Qwen 32B</th>'))
extra=['Qwen3-VL-32B-Instruct','HF 1,962 calls; vLLM TP4/APC 2,345 calls','BF16; greedy; cap 2,048; seed 20261103; HF SDPA / vLLM FlashAttention2','Same complete canonical frames; 480px / 2 FPS; no resize or sampling','Same text; official visual-first Qwen chat template','3,212 original E1 questions','Same 2,336 localization targets','Not run','Unknown VM compute cost; no API calls']
assert len(st.tbody.find_all('tr',recursive=False))==len(extra)
for tr,value in zip(st.tbody.find_all('tr',recursive=False),extra):tr.append(frag('<td>'+html.escape(value)+'</td>'))
for d in list(settings.find_all('details')):
 if 'Qwen 32B' in d.summary.get_text():d.decompose()
settings.append(frag('<details><summary>Qwen32B · completed run, provenance and failures</summary><p>All 4,307 logical calls completed exactly once: 1,962 retained HF calls plus 2,345 vLLM calls. All 4,307 scores were independently recomputed against frozen GT, with zero discrepancies. All 2,345 new snapshots passed source/SHA/frame-index/input-ID checks. Zero invalid outputs. The earlier two-GPU capacity OOM did not prevent completion on four GPUs.</p>'+table(['k','All completed calls','Retained HF','New vLLM','Invalid'],[[k,sum(x['batch_size']==k for x in newcalls),sum(x['batch_size']==k and x['backend']=='hf' for x in newcalls),sum(x['batch_size']==k and x['backend']!='hf' for x in newcalls),0] for k in K])+'<p>Long reports 074 and 075 triggered archived CUDA OOM attempts after switching videos. Report074 passed in a fresh engine; reports075/076 then completed with a full engine restart between reports. An expandable-allocator attempt was interrupted before producing an answer and retained as attempt evidence. No successful call was retried. These failures consumed compute and are excluded from accepted-call timing, not assumed free.</p><p>Revision: <code>0cfaf48183f594c314753d30a4c4974bc75f3ccb</code>. vLLM0.33.0, tensor parallel4, BF16, eager, max context230000, max batched tokens212992, max sequences1, prefix caching enabled, GPU memory utilization0.80. Shared video-prefix caching does not add previous responses to subsequent prompts. Exact input IDs do not guarantee equivalent floating-point arithmetic or answers across backends.</p></details>'))
files=soup.find(id='files');downloads=files.find(class_='downloads');downloads.clear()
for n,label in [('batch_summary.csv','Batch summary CSV'),('per_field.csv','All four-model predictions CSV'),('per_call.csv','All four-model calls CSV'),('question_type_summary.csv','Matched E1 subtype summary'),('paired_batch_contrasts.csv','Paired batch contrasts'),('context_summary.csv','Full context summary'),('qwen32b_backend_work.csv','Qwen32B backend timing CSV'),('analysis.json','Complete analysis JSON'),('audit.json','Qwen8B evidence audit'),('qwen32b_completion_audit.json','Qwen32B completion audit'),('qwen_input_output_example.json','Qwen8B input/output example')]:downloads.append(frag(download(n,label)))
# Make the distinction visible rather than silently presenting vLLM as HF generation.
files.append(frag('<p class="caption">For Qwen32B, filter <code>model=qwen32b</code> and use <code>backend</code>, <code>cached_input_tokens</code> and <code>timing_boundary</code> in per_call.csv. HF and vLLM entries are separate cohorts. Raw accepted32B calls are in the companion results package as qwen32b_calls.jsonl.gz; full input token snapshots remain on the VM. This report does not include Qwen reverse-order, mixing, context or extended Q2 runs.</p>'))
soup.footer.string='HD-EPIC q40_v1 · Updated 10 November 2026'
# Remove stale capacity-only claims and disambiguate old8B notes.
for txt in list(soup.find_all(string=True)):
 st=str(txt)
 if 'Qwen 8B mixing/E6 was not run' in st:txt.replace_with(st.replace('Qwen 8B mixing/E6 was not run; no Qwen mixing conclusion is available.','Qwen8B and Qwen32B mixing/E6 were not run; no Qwen mixing conclusion is available.'))
# Older timing, validity and audit descriptions refer specifically to the 8B run.
for txt in list(soup.find_all(string=True)):
 st=str(txt)
 fixes={'Qwen timing breakdown':'Qwen8B timing breakdown','Qwen k=1 has 55':'Qwen8B k=1 has 55','Other Qwen sizes have zero':'Other Qwen8B sizes have zero','Raw evidence audit and exclusions':'Qwen8B raw evidence audit and exclusions','Actual Qwen input / output':'Actual Qwen8B input / output','Qwen invalid k=1 outputs by subtype':'Qwen8B invalid k=1 outputs by subtype'}
 for a,b in fixes.items():st=st.replace(a,b)
 if st=='Qwen k':st='Qwen8B k'
 if st!=str(txt):txt.replace_with(st)
example=next(x for x in raw if x['batch_size']==1 and x['source_request']['field_ids']==['loc_445'])
(R/'qwen32b_input_output_example.json').write_text(json.dumps(example,indent=2))
settings.append(frag('<details><summary>Actual Qwen32B input / output · loc_445</summary><p>Accepted backend: '+html.escape(example['backend'])+'. Historical source_request model/generation fields are Gemini provenance; actual Qwen model settings are the top-level config.</p><h3>System</h3><pre>'+html.escape(example['source_request']['system'])+'</pre><h3>User</h3><pre>'+html.escape(example['source_request']['user'])+'</pre><h3>Raw output</h3><pre>'+html.escape(example['answer'])+'</pre></details>'))
downloads.append(frag(download('qwen32b_input_output_example.json','Qwen32B input/output example')))
q1.h2.string='Q1 · Accuracy by batch size'
q2.h2.string='Q2 · Accuracy by question type'
# Keep charts in view; move the numeric detail behind disclosures.
def fold(nodes,title):
 if not nodes:return
 panel=soup.new_tag('details');summary=soup.new_tag('summary');summary.string=title;panel.append(summary)
 nodes[0].insert_before(panel)
 for node in nodes:panel.append(node.extract())

for node in soup.select('.note'):node.decompose()
for sec,title in [(q1,'Accuracy by model and batch size'),(eff,'Cost, time and token values'),(q2,'Scores by question type')]:
 fold([sec.find('div',class_='wrap',recursive=False)],title)
eff.select('.figure > .caption')[0].string='Estimated API cost at uncached rates. Qwen compute cost is unavailable.'
eff.select('.figure > .caption')[3].string='API models only.'
for node in q2.find_all('p',recursive=False):node.decompose()
q2.select('.figure > .caption')[-1].string='Gemini only · 514 additional non-localization questions.'
q3=soup.find(id='q3').find_parent('section')
q3.find(class_='answer').string='Mixing changed Flash-Lite little. Argon scored lower, but the paired interval reaches zero.'
q3.select_one('.figure > .caption').string='Same 96 target questions across 24 videos. Whiskers: 95% intervals.'
fold(q3.find_all('p',recursive=False),'Paired results and run details')
ctx=soup.find(id='context-control')
ctx.find(class_='answer').string='Reference questions improved Argon’s target accuracy. The comparison with answering the full batch is inconclusive.'
ctx.select_one('.figure > .caption').string='Same 2,336 targets across 73 videos. References include questions and options, without answers.'
fold(ctx.find_all('div',class_='wrap',recursive=False),'Exact scores and paired comparisons')
fold(settings.find_all(['div','p'],recursive=False),'Run settings and answer validity')
for node in files.find_all('p',recursive=False):node.decompose()
fold(files.find_all('ol',recursive=False),'How to use the data')

# Matplotlib and retained historical SVGs must all have unique IDs in the same HTML.
index=0
def namespace(m):
 global index
 index+=1;s=m.group(0)
 for name in re.findall(r'\bid="([^"]+)"',s):
  new=f'nov9chart{index}_{name}';s=s.replace('id="'+name+'"','id="'+new+'"').replace('url(#'+name+')','url(#'+new+')').replace('href="#'+name+'"','href="#'+new+'"')
 return s
text=re.sub(r'<svg\b.*?</svg>',namespace,str(soup),flags=re.S);(R/'baseline_report.html').write_text(text)
print(json.dumps({'models':4,'calls':len(calls),'answers':len(fields),'charts':index,'qwen32b_curve':[(k,Q['qwen32b',k]['correct']) for k in K],'qwen32b_types':[x for x in S['q2_matched_e1'] if x['model']=='qwen32b']},indent=2))
