from pathlib import Path
import json,html,csv,runpy,shutil,gzip
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.utils import ImageReader
P=Path(__file__).resolve().parent;S=json.loads((P/'summary.json').read_text());D=json.loads(gzip.decompress((P/'data/analysis_inputs.json.gz').read_bytes()));B=S['matched_localization'];blue='#245781';orange='#b56a29'
# Recompute the proposal estimators from saved predictions; no inference.
from collections import defaultdict, Counter
rng=np.random.default_rng(20261020)
for row in S['question_types']:
 grouped=defaultdict(list)
 for x in D['scores']:
  if x['experiment']=='E1' and x['category']==row['category']:grouped[x['source_video']].append(x['correct'])
 rates=np.array([np.mean(grouped[k]) for k in sorted(grouped)])
 draws=rates[rng.integers(0,len(rates),(10000,len(rates)))].mean(axis=1)
 row.update(pooled_accuracy=row['accuracy'],video_weighted_accuracy=float(rates.mean()),video_weighted_ci95=list(map(float,np.quantile(draws,[.025,.975]))))
a={(x['report_id'],x['field_id']):x['correct'] for x in D['scores'] if x['experiment']=='E6_homogeneous' and x['e6_common_target']}
b={(x['report_id'],x['field_id']):x['correct'] for x in D['scores'] if x['experiment']=='E6_mixed' and x['e6_common_target']}
S['e6_transition_counts']=dict(Counter(('correct' if a[k] else 'wrong')+'_to_'+('correct' if b[k] else 'wrong') for k in a))
for row in B:
 calls=[x for x in D['calls'] if x['experiment']==row['experiment'] and x['matched_localization_fields']>0]
 row.update(input_tokens_per_32=sum(x['input_tokens'] for x in calls)/73,output_tokens_per_32=sum(x['output_tokens'] for x in calls)/73,delta_vs_size1_pp=(row['accuracy']-B[0]['accuracy'])*100)
videos=sorted(set(x['source_video'] for x in D['scores'] if x['experiment']=='E5'));m={}
for exp in ['E1','E4']:
 m[exp]=np.array([np.mean([x['correct'] for x in D['scores'] if x['experiment']==exp and x['matched_localization'] and x['source_video']==v]) for v in videos])
draw=np.random.default_rng(20261020).integers(0,len(videos),(10000,len(videos)));delta=(m['E4']-m['E1'])[draw].mean(axis=1)
S['paired_differences']['1_to_16']={'delta':float((m['E4']-m['E1']).mean()),'ci95':list(map(float,np.quantile(delta,[.025,.975])))}
for row in B:
 row['api_seconds_per_32']=row['api_round_trip_sum_s']/73
 row['api_speedup_vs_single']=B[0]['api_round_trip_sum_s']/row['api_round_trip_sum_s']
S['report_revision']='2026-10-21 cost and API time per32; explicit CI meaning; simplified E6 setup and effect'
(P/'summary.json').write_text(json.dumps(S,indent=2)+'\n')
labels={'fine_grained_action_localization':'Localization','fine_grained_how_recognition':'How','fine_grained_why_recognition':'Why','ingredient_ingredient_weight':'Weight'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(f,name):
 f.savefig(P/'figures'/f'{name}.png',dpi=240,bbox_inches='tight');f.savefig(P/'figures'/f'{name}.svg',bbox_inches='tight');plt.close(f)
x=np.arange(4);t=S['question_types'];y=np.array([r['video_weighted_accuracy'] for r in t])*100;lo=np.array([r['video_weighted_ci95'][0] for r in t])*100;hi=np.array([r['video_weighted_ci95'][1] for r in t])*100
f,a=plt.subplots(figsize=(7.2,3.3));a.bar(x,y,color=blue);a.errorbar(x,y,yerr=[y-lo,hi-y],fmt='none',color='gray',capsize=4);a.set(xticks=x,xticklabels=[labels[r['category']]+f"\n{r['fields']} Q / {r['videos']} videos" for r in t],ylabel='Mean per-video accuracy (%)',xlabel='Question subtype',ylim=(0,100),title='Q2  Individual-question performance with videos weighted equally');a.grid(axis='y',alpha=.16);a.set_axisbelow(True)
for xx,yy in zip(x,y):a.text(xx,yy+3,f'{yy:.2f}%',ha='center',bbox={'facecolor':'white','edgecolor':'none','pad':1})
f.subplots_adjust(bottom=.23);save(f,'03_question_types')
f,(a,b)=plt.subplots(1,2,figsize=(7.2,2.9));xx=np.arange(5);a.bar(xx,[r['cost_per_32_usd'] for r in B],color=blue);inp=np.array([r['input_tokens_per_32'] for r in B])/1000;out=np.array([r['output_tokens_per_32'] for r in B])/1000;b.bar(xx,inp,color=blue,label='Input');b.bar(xx,out,bottom=inp,color=orange,label='Output')
for ax in [a,b]:ax.set(xticks=xx,xticklabels=[1,4,8,16,32],xlabel='Questions per call');ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
a.set(ylabel='USD / 32 questions',title='Completion cost');b.set(ylabel='Tokens / 32 questions (thousands)',title='Input and output tokens');b.legend(frameon=False,fontsize=8);f.tight_layout();save(f,'02_cost_and_tokens')
# Q1: wide accuracy plot with a clearly labeled zoomed axis; cost/time below.
f=plt.figure(figsize=(8.2,4.4));grid=f.add_gridspec(2,2,height_ratios=[1,1.05]);ax=[f.add_subplot(grid[0,:]),f.add_subplot(grid[1,0]),f.add_subplot(grid[1,1])];xx=np.arange(5)*1.6
vals=[[r['accuracy']*100 for r in B],[r['cost_per_32_usd'] for r in B],[r['api_seconds_per_32'] for r in B]]
for j,(a,v) in enumerate(zip(ax,vals)):
 a.bar(xx,v,color=blue,width=.65);a.set(xticks=xx,xticklabels=[1,4,8,16,32],xlabel='Questions per call');a.grid(axis='y',alpha=.16);a.set_axisbelow(True);a.tick_params(labelsize=9)
 for x,y in zip(xx,v):a.annotate(f'{y:.2f}%' if j==0 else (f'${y:.3f}' if j==1 else f'{y:.1f}s'),(x,y),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8.5)
 a.set_ylim((20,40) if j==0 else (0,max(v)*1.28))
ax[0].set(title='Accuracy (zoomed axis: 20-40%)',ylabel='Accuracy (%)');ax[1].set(title='Cost per 32 questions',ylabel='USD / 32 questions');ax[2].set(title='API time per 32 questions',ylabel='Seconds / 32 questions')
f.tight_layout(h_pad=1.5,w_pad=2);save(f,'06_batch_tradeoffs')
# Appendix accuracy curve uses the same explicitly zoomed scale.
f,a=plt.subplots(figsize=(7.2,3.4));v=np.array(vals[0]);lo=np.array([r['ci95'][0] for r in B])*100;hi=np.array([r['ci95'][1] for r in B])*100
a.errorbar(xx,v,yerr=[v-lo,hi-v],fmt='o-',color=blue,capsize=4);a.set(xticks=xx,xticklabels=[1,4,8,16,32],xlabel='Questions per call',ylabel='Accuracy (%)',ylim=(20,40),title='Localization accuracy (zoomed axis: 20-40%)');a.grid(axis='y',alpha=.16)
for x,y in zip(xx,v):a.annotate(f'{y:.2f}%',(x,y),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
f.tight_layout();save(f,'01_accuracy_vs_size')
# Q3: only the two comparable target-accuracy bars remain.
f,a=plt.subplots(figsize=(6.8,2.8));mean=[41/96*100,40/96*100];cis=[S['e6_common'][e]['ci95'] for e in ['E6_homogeneous','E6_mixed']]
a.bar([0,1.8],mean,color=[blue,orange],width=.6);a.errorbar([0,1.8],mean,yerr=[[mean[i]-cis[i][0]*100 for i in range(2)],[cis[i][1]*100-mean[i] for i in range(2)]],fmt='none',color='#64717d',capsize=5)
a.set(xticks=[0,1.8],xticklabels=['Pure localization','Mixed subtypes'],ylabel='Common-target accuracy (%)',ylim=(0,100),xlim=(-.9,2.7),title='Accuracy on the same 96 target questions');a.grid(axis='y',alpha=.16);a.set_axisbelow(True)
for x,y,n in [(0,mean[0],41),(1.8,mean[1],40)]:a.text(x,63,f'{y:.2f}%\n{n}/96 correct',ha='center',fontsize=10)
f.tight_layout();save(f,'04_e6_paired')
# Setup shows which four questions are scored and which four only change context.
f,a=plt.subplots(figsize=(7.4,1.8));a.axis('off')
for y,label,other in [(.72,'PURE: 8 questions','4 localization companions'),(.30,'MIXED: 8 questions','2 How + 2 Why companions')]:
 a.text(.01,y,label,va='center',fontsize=10,fontweight='bold')
 a.text(.46,y,'Same 4 localization targets',ha='center',va='center',fontsize=10,color=blue,bbox={'boxstyle':'round,pad=.5','facecolor':'#e9f0f6','edgecolor':'none'})
 a.text(.71,y,'+',ha='center',va='center',fontsize=13)
 a.text(.88,y,other,ha='center',va='center',fontsize=9,bbox={'boxstyle':'round,pad=.5','facecolor':'#f0f1f2','edgecolor':'none'})
a.text(.5,-.1,'Same video, same target positions. Score only the shared targets: 24 videos x 4 = 96.',ha='center',fontsize=9);save(f,'08_e6_setup')
# Keep detailed methods and output examples in a separate appendix.
p=P/'build_report.py';src=p.read_text();src=src.replace('E1 accuracy over the whole prepared pool is 1,142/3,212 (35.55%).','E1 pooled accuracy over the whole prepared pool is 1,142/3,212 (35.55%). Q2 primary estimates average each video\'s subtype accuracy equally, as specified in the proposal. The older report used pooled question accuracy; this revision corrects that estimator.')
src=src.replace("pct(r['accuracy']),ci(r['ci95'])+'%'] for r in S['question_types']", "pct(r['video_weighted_accuracy']),ci(r['video_weighted_ci95'])+'%'] for r in S['question_types']")
src=src.replace("['Question family','Correct / fields','Accuracy','95% interval']", "['Question family','Pooled correct / fields','Video mean accuracy','95% interval']")
src=src.replace('Per-field accuracy', 'Per-video subtype accuracy').replace('Figure 4. Single-question E1 accuracy by original question family','Figure 4. Equal-video single-question E1 accuracy by original question family')
src=src.replace('Figure 5. Each line links the same four target questions in one report. Circles are pure localization; crosses are mixed companions.','Figure 5. Each thin line links one video across conditions. Black diamonds show the mean across 24 videos.')
p.write_text(src);runpy.run_path(str(p))
for ext in ['pdf','md','html']:shutil.move(P/f'report.{ext}',P/f'technical_appendix.{ext}')
def pct(x):return f'{x*100:.2f}%'
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyM',fontSize=9.5,leading=13,spaceAfter=7));styles.add(ParagraphStyle(name='SmallM',fontSize=8,leading=10,spaceAfter=5,textColor=colors.HexColor('#52616c')));styles.add(ParagraphStyle(name='AnswerM',fontSize=12,leading=16,spaceAfter=11,textColor=colors.HexColor(blue)));styles['Heading1'].fontSize=17;styles['Heading1'].leading=21;styles['Heading1'].spaceAfter=9;styles['Heading2'].fontSize=11
story=[];md=['# Baseline Results\n','21 October 2026 | Gemini 3.5 Flash Lite | Question dataset hd_epic_q40_v1\n'];htmlparts=[]
def paragraph(t,sty='BodyM'):
 story.append(Paragraph(html.escape(t),styles[sty]));md.append(t+'\n');htmlparts.append('<p>'+html.escape(t)+'</p>')
def title(t):
 story.append(Paragraph(html.escape(t),styles['Heading1']));md.extend(['## '+t,'']);htmlparts.append('<h2>'+html.escape(t)+'</h2>')
def table(rows,widths=None):
 cells=[[Paragraph(html.escape(str(v)),styles['SmallM']) for v in row] for row in rows];tb=Table(cells,colWidths=widths or [510/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT');tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eef3')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f8f9')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),1)]));story.extend([tb,Spacer(1,8)]);md.extend([' | '.join(rows[0]),' | '.join(['---']*len(rows[0]))]+[' | '.join(map(str,row)) for row in rows[1:]]+['']);htmlparts.append('<table>'+''.join('<tr>'+''.join(('<th>' if i==0 else '<td>')+html.escape(str(x))+('</th>' if i==0 else '</td>') for x in row)+'</tr>' for i,row in enumerate(rows))+'</table>')
def figure(name,height):
 p=P/'figures'/name;iw,ih=ImageReader(str(p)).getSize();story.extend([Image(str(p),width=height*iw/ih,height=height),Spacer(1,5)]);md.extend([f'![{name}](figures/{name})','']);htmlparts.append(f'<img src="figures/{name}" alt="{name}">')
def page():story.append(PageBreak())
title('Q1  Does asking more questions change accuracy and efficiency?')
paragraph('Answer: accuracy is similar at 1-8 questions, then falls at 16-32. Batching sharply reduces completion cost.','AnswerM')
paragraph('Gemini 3.5 Flash Lite. Same 73 videos and 2,336 localization targets at every size. Same fixed video bytes, model, prompts and question ordering. One run per condition. Dataset: hd_epic_q40_v1.','SmallM')
figure('06_batch_tradeoffs.png',240)
table([['Q / call','Accuracy','USD / 32 Q','API seconds / 32 Q','Speedup vs 1']]+[[str(r['size']),pct(r['accuracy']),f"{r['cost_per_32_usd']:.5f}",f"{r['api_seconds_per_32']:.2f}",f"{r['api_speedup_vs_single']:.2f}x"] for r in B],widths=[65,90,105,145,105])
paragraph('Size 8 -> 32: -8.35 percentage points; paired 95% video-bootstrap interval -11.04 to -5.65. Sizes 4 and 8 differ by only two correct answers; this does not establish an optimum at eight.')
paragraph('Cheapest and fastest observed: size 32 ($0.01341 and 16.08 s per 32 targets), with lower accuracy. Highest observed accuracy: size 8 (35.02%; size 4 is almost identical). Size 8 is the candidate balance: $0.04322 and 25.99 s, saving 86.1% cost and 81.3% API time versus size 1. No single setting is best on all three metrics.')
paragraph('API time is the sum of serial request round trips to complete 32 targets, averaged over 73 videos. It includes upload/network/provider time; it excludes local preprocessing and is not backend-only inference time. This compares 32 size-1 calls with one size-32 call fairly. Input/output tokens and latency distributions remain in the appendix and CSVs.','SmallM')
paragraph('Execution: 4,355 effective calls; 0 provider failures; 0 missing/invalid outputs. Effective API cost $43.49; 148.53M tokens. Prior v1 E5 reruns, earlier pilots and VM costs are excluded. Latency and all raw outputs are in the appendix and CSVs.','SmallM')
page();title('Q2  How does the model perform on each question subtype?')
paragraph('Answer: Why and How score higher than Localization in this run. Weight is too sparse for a stable estimate.','AnswerM')
paragraph('E1 ran ALL 3,212 prepared questions across 76 videos, one question per call. This is the entire hd_epic_q40_v1 input pool, not the full official HD-EPIC benchmark. Primary accuracy weights videos equally within each subtype.')
figure('03_question_types.png',220)
table([['Subtype','Video mean','95% interval','Pooled correct / Q','Videos']]+[[labels[r['category']],pct(r['video_weighted_accuracy']),f"{r['video_weighted_ci95'][0]*100:.2f}-{r['video_weighted_ci95'][1]*100:.2f}%",f"{r['correct']}/{r['fields']}",str(r['videos'])] for r in t],widths=[100,91,104,125,90])
paragraph('Gray whiskers = 95% confidence intervals from 10,000 video-bootstrap resamples. They show uncertainty across sampled videos, not the range of individual answers or a 95% guarantee about model correctness. Weight has the widest uncertainty because it has only 10 videos.')
paragraph('Why are How/Why higher? Not established by this experiment. Saved How/Why prompts identify the action and give a narrow time segment; localization must choose a precise temporal alternative. Some Why answers may also benefit from everyday action knowledge. At 2 FPS, fine time distinctions may be harder. These are plausible explanations, not measured causes; no text-only or higher-FPS control was run.')
paragraph('Weight has only 16 questions from 10 videos, with a 0-40% interval. Do not use its rank as a general conclusion. This study covers four subtypes, not all HD-EPIC question types.')
paragraph('Actual prompt example: how_279 asks how the person rinses a chopping board and supplies a 1.09-second segment. This gives a temporal pointer that localization is being asked to recover. Raw input and predictions are unchanged; pooled rates remain in summary.json.','SmallM')
page();title('Q3  Does subtype mixing affect the same localization questions?')
paragraph('Answer: no clear mixing effect was observed. Mixed companions changed the net score by one question.','AnswerM')
paragraph('Question: when the SAME four localization questions are accompanied by different question types, do their answers change? Both conditions contain eight questions; only the four companion questions change.')
figure('08_e6_setup.png',100)
table([['Condition','Correct / targets','Target accuracy'],['Pure localization','41 / 96','42.71%'],['Mixed subtypes','40 / 96','41.67%']])
figure('04_e6_paired.png',150)
paragraph('Result: pure answers 41/96 correctly; mixed answers 40/96. Five targets change correct -> wrong and four wrong -> correct. Net change: -1.04 pp. The 95% interval (-7.29 to +5.21) crosses zero: this run does not establish benefit or harm, and does not prove equivalence.')
paragraph('Read the figure: bars compare the SAME 96 targets; gray whiskers show 95% video-bootstrap intervals. The all-192-field mixed/pure totals are excluded because companion questions differ.','SmallM')
paragraph('Outputs: per_field.csv = every prediction/target/score; per_call.csv = tokens/cost/timing; all_model_inputs_outputs.jsonl.gz = exact prompts and responses; logs/ = full logs. Open report.html for direct links. Full settings, the four planned figures plus accuracy-vs-cost, and actual output examples are in technical_appendix.pdf.','SmallM')
paragraph('Validity is not hallucination: zero invalid option outputs does not establish zero semantic hallucinations. All inference evidence is unchanged.','SmallM')
def foot(c,d):c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#65727b'));c.drawString(45,25,'Capstone 14 | Final v1 question dataset baseline | Revised 21 October 2026');c.drawRightString(567,25,str(d.page))
SimpleDocTemplate(str(P/'report.pdf'),pagesize=(612,792),leftMargin=45,rightMargin=45,topMargin=36,bottomMargin=38,title='Baseline Results Q1 Q2 Q3').build(story,onFirstPage=foot,onLaterPages=foot)
(P/'report.md').write_text('\n'.join(md))
links='<nav><a href="report.pdf">Three page report</a> · <a href="technical_appendix.pdf">Technical appendix</a> · <a href="data/per_call.csv">Calls and cost</a> · <a href="data/per_field.csv">Predictions</a> · <a href="data/all_model_inputs_outputs.jsonl.gz">Actual input/output</a> · <a href="logs/">Logs</a></nav>'
(P/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Baseline Results Q1 Q2 Q3</title><style>body{font:16px/1.6 system-ui;max-width:950px;margin:40px auto;padding:0 24px;color:#26323b}h2{color:#245781;margin-top:44px}table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #dce2e7}th,nav{background:#eef2f5}nav{padding:14px}img{max-width:100%}a{color:#245781}</style><h1>Baseline Results</h1>'+links+''.join(htmlparts)+'</html>')
with (P/'data/question_types.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(t[0]));w.writeheader();w.writerows(t)
with (P/'data/matched_localization.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(B[0]));w.writeheader();w.writerows(B)
print('Report rebuilt')
